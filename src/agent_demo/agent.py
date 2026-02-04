"""Agent 核心模組。

實作與 Claude API 互動的對話代理。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast  # 用於 client 類型

import anthropic
from anthropic import APIConnectionError, APIStatusError, AuthenticationError
from anthropic.types import MessageParam, ToolResultBlockParam

from agent_demo.tools.registry import ToolRegistry
from agent_demo.usage_monitor import UsageMonitor

logger = logging.getLogger(__name__)

# 工具摘要的最大長度
_SUMMARY_MAX_LEN: int = 120

# 工具名稱與摘要格式對應
_TOOL_SUMMARY_MAP: dict[str, tuple[str, str]] = {
    'read_file': ('讀取檔案', 'path'),
    'edit_file': ('編輯檔案', 'path'),
    'list_files': ('列出檔案', 'path'),
    'bash': ('執行命令', 'command'),
    'grep_search': ('搜尋程式碼', 'pattern'),
}


def get_tool_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    """根據工具名稱與參數產生人類可讀的摘要。

    Args:
        tool_name: 工具名稱
        tool_input: 工具參數

    Returns:
        人類可讀的摘要字串
    """
    if tool_name in _TOOL_SUMMARY_MAP:
        label, param_key = _TOOL_SUMMARY_MAP[tool_name]
        value = str(tool_input.get(param_key, ''))
        if len(value) > _SUMMARY_MAX_LEN:
            value = value[:_SUMMARY_MAX_LEN] + '...'
        return f'{label} {value}'

    # 未知工具：顯示工具名稱
    return f'使用工具 {tool_name}'


# 預設配置
DEFAULT_MODEL = 'claude-sonnet-4-20250514'
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SYSTEM_PROMPT = """你是一位專業的程式開發助手。

工作原則：
- 遇到複雜任務時，先理解需求，再逐步執行
- 執行操作前，思考是否需要先讀取相關檔案了解現況
- 解釋你的思考過程和選擇的理由
- 遇到不確定的情況，主動詢問使用者

請使用繁體中文回答。"""


@dataclass
class AgentConfig:
    """Agent 配置。"""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    timeout: float = 30.0


@dataclass
class Agent:
    """對話代理。

    負責管理與 Claude API 的對話互動，維護對話歷史，
    並支援串流回應。支援工具調用迴圈。

    Attributes:
        config: Agent 配置
        client: Anthropic API 客戶端
        conversation: 對話歷史紀錄
        tool_registry: 工具註冊表（可選）
        usage_monitor: 使用量監控器（可選）
    """

    config: AgentConfig = field(default_factory=AgentConfig)
    client: Any = None  # anthropic.AsyncAnthropic | MagicMock
    conversation: list[MessageParam] = field(default_factory=lambda: [])
    tool_registry: ToolRegistry | None = None
    usage_monitor: UsageMonitor | None = field(default_factory=lambda: UsageMonitor())

    def __post_init__(self) -> None:
        """初始化 API 客戶端與快取固定參數。"""
        if self.client is None:
            self.client = anthropic.AsyncAnthropic()

        # 快取固定的 API 參數
        self._base_kwargs: dict[str, Any] = {
            'model': self.config.model,
            'max_tokens': self.config.max_tokens,
            'system': [
                {
                    'type': 'text',
                    'text': self.config.system_prompt,
                    'cache_control': {'type': 'ephemeral'},
                }
            ],
            'timeout': self.config.timeout,
        }

        # 快取工具定義（如果有）
        if self.tool_registry:
            tool_defs = self.tool_registry.get_tool_definitions()
            if tool_defs:
                self._base_kwargs['tools'] = tool_defs

        logger.info('Agent 已初始化', extra={'model': self.config.model})

    def reset_conversation(self) -> None:
        """重設對話歷史。"""
        self.conversation = []
        logger.debug('對話歷史已重設')

    def _prepare_messages_with_cache(self) -> list[dict[str, Any]]:
        """準備帶有緩存控制的訊息列表。

        在最後一個 message 的最後一個 content block 添加 cache_control，
        以啟用 prompt caching 優化。不修改原始 self.conversation。

        Returns:
            帶有 cache_control 的訊息列表副本
        """
        if not self.conversation:
            return []

        # 深拷貝整個 conversation（確保不修改原始資料）
        import copy

        # deepcopy 回傳型別仍為 list[MessageParam]（TypedDict），
        # 但實際上已是純 dict，後續需要動態添加 cache_control 欄位，
        # 透過 Any 中轉為 list[dict[str, Any]] 以繞過 TypedDict 的欄位限制
        messages_raw: Any = copy.deepcopy(self.conversation)
        messages: list[dict[str, Any]] = messages_raw

        # 取得最後一個 message
        last_msg = messages[-1]
        content = last_msg.get('content')

        if isinstance(content, str):
            # 字串格式 → 轉換為 text block 格式並添加 cache_control
            last_msg['content'] = [
                {
                    'type': 'text',
                    'text': content,
                    'cache_control': {'type': 'ephemeral'},
                }
            ]
        elif isinstance(content, list) and content:
            # 列表格式 → 在最後一個 block 添加 cache_control
            last_block = cast(dict[str, Any], content[-1])
            last_block['cache_control'] = {'type': 'ephemeral'}

        return messages

    def build_stream_kwargs(self) -> dict[str, Any]:
        """建立 messages.stream() 的參數。

        Returns:
            API 呼叫參數字典
        """
        # 準備帶有 cache_control 的 messages
        messages = self._prepare_messages_with_cache()

        # 複製固定參數 + 加入動態 messages
        return {
            **self._base_kwargs,
            'messages': messages,
        }

    async def _execute_tool_calls(
        self,
        content_blocks: list[Any],
    ) -> list[ToolResultBlockParam]:
        """執行回應中的工具調用。

        Args:
            content_blocks: Claude 回應的 content blocks

        Returns:
            tool_result blocks 列表
        """
        tool_results: list[ToolResultBlockParam] = []

        for block in content_blocks:
            if block.type != 'tool_use':
                continue

            logger.info('執行工具', extra={'tool_name': block.name, 'tool_id': block.id})

            try:
                assert self.tool_registry is not None
                result = await self.tool_registry.execute(block.name, block.input)
                result_content = (
                    json.dumps(result, ensure_ascii=False)
                    if isinstance(result, dict)
                    else str(result)
                )
                tool_results.append(
                    ToolResultBlockParam(
                        type='tool_result',
                        tool_use_id=block.id,
                        content=result_content,
                    )
                )
            except Exception as e:
                logger.warning(
                    '工具執行失敗',
                    extra={'tool_name': block.name, 'error': str(e)},
                )
                tool_results.append(
                    ToolResultBlockParam(
                        type='tool_result',
                        tool_use_id=block.id,
                        content=str(e),
                        is_error=True,
                    )
                )

        return tool_results

    def _handle_stream_error(
        self,
        error: Exception,
        response_parts: list[str],
    ) -> None:
        """處理串流錯誤並管理對話歷史。

        Args:
            error: API 錯誤
            response_parts: 已收到的回應片段

        Raises:
            PermissionError: API 認證失敗
            TimeoutError: API 回應超時
            ConnectionError: API 連線失敗
            RuntimeError: 其他 API 錯誤
        """
        # 處理認證錯誤
        if isinstance(error, AuthenticationError):
            self.conversation.pop()
            logger.error('API 認證失敗', extra={'error': str(error)})
            raise PermissionError(
                'API 金鑰無效或已過期。請檢查 ANTHROPIC_API_KEY 環境變數是否正確設定。'
            ) from error

        # 處理超時錯誤（必須在 APIConnectionError 之前檢查）
        if isinstance(error, anthropic.APITimeoutError):
            if response_parts:
                partial = ''.join(response_parts)
                self.conversation.append({'role': 'assistant', 'content': partial})
                logger.warning('串流超時，已保留部分回應', extra={'partial_length': len(partial)})
            else:
                self.conversation.pop()
            logger.error('API 回應超時', extra={'error': str(error)})
            raise TimeoutError('串流回應超時。') from error

        # 處理連線錯誤
        if isinstance(error, APIConnectionError):
            if response_parts:
                partial = ''.join(response_parts)
                self.conversation.append({'role': 'assistant', 'content': partial})
                logger.warning('串流中斷，已保留部分回應', extra={'partial_length': len(partial)})
            else:
                self.conversation.pop()
            logger.error('API 連線失敗', extra={'error': str(error)})
            raise ConnectionError('串流連線中斷，請檢查網路連線並稍後重試。') from error

        # 處理其他 API 狀態錯誤
        if isinstance(error, APIStatusError):
            self.conversation.pop()
            logger.error('API 錯誤', extra={'status_code': error.status_code, 'error': str(error)})
            raise RuntimeError(f'API 錯誤 ({error.status_code}): {error.message}') from error

    async def _stream_with_tool_loop(
        self,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """執行串流迴圈，支援工具調用。

        Yields:
            str: 回應的每個 token
            dict: 事件通知（tool_call、preamble_end）

        Raises:
            各種 API 錯誤（由 _handle_stream_error 處理）
        """
        response_parts: list[str] = []

        try:
            while True:
                kwargs = self.build_stream_kwargs()

                async with self.client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        response_parts.append(text)
                        yield text

                    final_message = await stream.get_final_message()

                # 記錄 API 使用量
                if self.usage_monitor and hasattr(final_message, 'usage'):
                    self.usage_monitor.record(final_message.usage)

                # 將 assistant 回應加入對話歷史（轉換為可序列化格式）
                self.conversation.append(
                    {
                        'role': 'assistant',
                        'content': [block.model_dump() for block in final_message.content],
                    }
                )

                # 若無工具調用，結束迴圈
                if final_message.stop_reason != 'tool_use' or not self.tool_registry:
                    logger.debug(
                        '串流回應完成',
                        extra={'response_length': len(''.join(response_parts))},
                    )
                    break

                # 標記 preamble 結束（僅在有文字時）
                if response_parts:
                    yield {'type': 'preamble_end', 'data': {}}

                # 收集所有工具調用區塊
                assert self.tool_registry is not None
                tool_use_blocks = [b for b in final_message.content if b.type == 'tool_use']

                # 先通知前端所有工具開始執行（讓前端同時顯示 running 狀態）
                summaries: dict[str, str] = {}
                for block in tool_use_blocks:
                    summary = get_tool_summary(block.name, block.input)
                    summaries[block.id] = summary
                    logger.info('執行工具', extra={'tool_name': block.name, 'tool_id': block.id})
                    yield {
                        'type': 'tool_call',
                        'data': {
                            'name': block.name,
                            'status': 'started',
                            'summary': summary,
                        },
                    }

                # 並行執行所有工具
                async def _run_tool(
                    tool_block: Any,
                ) -> tuple[Any, Exception | None]:
                    """執行單一工具，回傳 (結果, 錯誤)。"""
                    try:
                        res = await self.tool_registry.execute(  # type: ignore[union-attr]
                            tool_block.name,
                            tool_block.input,
                        )
                        return (res, None)
                    except Exception as exc:
                        return (None, exc)

                exec_results = await asyncio.gather(*[_run_tool(b) for b in tool_use_blocks])

                # 收集結果並通知前端完成狀態
                tool_results: list[ToolResultBlockParam] = []
                for block, (result, error) in zip(tool_use_blocks, exec_results):
                    summary = summaries[block.id]
                    if error is None:
                        result_content = (
                            json.dumps(result, ensure_ascii=False)
                            if isinstance(result, dict)
                            else str(result)
                        )
                        tool_results.append(
                            ToolResultBlockParam(
                                type='tool_result',
                                tool_use_id=block.id,
                                content=result_content,
                            )
                        )
                        yield {
                            'type': 'tool_call',
                            'data': {
                                'name': block.name,
                                'status': 'completed',
                                'summary': summary,
                            },
                        }
                    else:
                        logger.warning(
                            '工具執行失敗',
                            extra={'tool_name': block.name, 'error': str(error)},
                        )
                        tool_results.append(
                            ToolResultBlockParam(
                                type='tool_result',
                                tool_use_id=block.id,
                                content=str(error),
                                is_error=True,
                            )
                        )
                        yield {
                            'type': 'tool_call',
                            'data': {
                                'name': block.name,
                                'status': 'failed',
                                'summary': summary,
                                'error': str(error),
                            },
                        }

                self.conversation.append(
                    {
                        'role': 'user',
                        'content': tool_results,
                    }
                )
                logger.debug('工具結果已回傳，繼續對話', extra={'tool_count': len(tool_results)})

                # 重置 response_parts 以收集下一輪串流
                response_parts = []

        except (
            AuthenticationError,
            anthropic.APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ) as e:
            self._handle_stream_error(e, response_parts)
            raise  # 重新拋出已轉換的例外

    async def stream_message(
        self,
        content: str,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """以串流方式發送訊息並逐步取得回應。

        支援工具調用迴圈：當 Claude 回傳 tool_use 時，
        自動執行工具並將結果回傳，直到取得最終文字回應。

        Args:
            content: 使用者訊息內容

        Yields:
            str: 回應的每個 token
            dict: 事件通知（tool_call、preamble_end）

        Raises:
            ValueError: 訊息為空白
            ConnectionError: API 連線失敗
            PermissionError: API 認證失敗
            TimeoutError: API 回應超時
        """
        # 驗證輸入
        content = content.strip()
        if not content:
            raise ValueError('訊息不可為空白，請輸入有效內容')

        # 加入使用者訊息到對話歷史
        self.conversation.append({'role': 'user', 'content': content})
        logger.debug('收到使用者訊息 (串流模式)', extra={'content_length': len(content)})

        # 執行串流迴圈（包含工具調用處理）
        async for token in self._stream_with_tool_loop():
            yield token
