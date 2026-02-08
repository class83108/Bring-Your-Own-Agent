"""Agent 核心模組。

實作與 LLM Provider 互動的對話代理，支援工具調用迴圈。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agent_core.compact import COMPACT_THRESHOLD, compact_conversation
from agent_core.config import AgentCoreConfig
from agent_core.multimodal import Attachment, build_content_blocks
from agent_core.providers.base import LLMProvider
from agent_core.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderTimeoutError,
)
from agent_core.skills.registry import SkillRegistry
from agent_core.token_counter import TokenCounter
from agent_core.tools.registry import ToolRegistry
from agent_core.usage_monitor import UsageMonitor

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """對話代理。

    負責管理與 LLM Provider 的對話互動，維護對話歷史，
    並支援串流回應與工具調用迴圈。

    Attributes:
        config: Agent 配置
        provider: LLM Provider 實例
        conversation: 對話歷史紀錄
        tool_registry: 工具註冊表（可選）
        usage_monitor: 使用量監控器（可選）
        token_counter: Token 計數器（可選）
    """

    config: AgentCoreConfig
    provider: LLMProvider
    conversation: list[dict[str, Any]] = field(default_factory=lambda: [])
    tool_registry: ToolRegistry | None = None
    skill_registry: SkillRegistry | None = None
    usage_monitor: UsageMonitor | None = field(default_factory=UsageMonitor)
    token_counter: TokenCounter | None = field(default_factory=TokenCounter)

    def __post_init__(self) -> None:
        """初始化日誌。"""
        logger.info(
            'Agent 已初始化',
            extra={'model': self.config.provider.model},
        )

    def reset_conversation(self) -> None:
        """重設對話歷史。"""
        self.conversation = []
        logger.debug('對話歷史已重設')

    def _get_system_prompt(self) -> str:
        """取得合併後的 system prompt。"""
        if self.skill_registry:
            return self.skill_registry.get_combined_system_prompt(self.config.system_prompt)
        return self.config.system_prompt

    async def _execute_single_tool(
        self,
        tool_block: dict[str, Any],
    ) -> tuple[Any, Exception | None]:
        """執行單一工具，回傳 (結果, 錯誤)。"""
        try:
            res = await self.tool_registry.execute(  # type: ignore[union-attr]
                tool_block['name'],
                tool_block['input'],
            )
            return (res, None)
        except Exception as exc:
            return (None, exc)

    def _build_tool_result_entry(
        self,
        block: dict[str, Any],
        result_val: Any,
        error: Exception | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """建立單一工具的結果與事件通知。

        Returns:
            (tool_result, event) 的 tuple
        """
        if error is None:
            result_content = (
                json.dumps(result_val, ensure_ascii=False)
                if isinstance(result_val, dict)
                else str(result_val)
            )
            tool_result = {
                'type': 'tool_result',
                'tool_use_id': block['id'],
                'content': result_content,
            }
            event = {
                'type': 'tool_call',
                'data': {'name': block['name'], 'status': 'completed'},
            }
        else:
            logger.warning(
                '工具執行失敗',
                extra={'tool_name': block['name'], 'error': str(error)},
            )
            tool_result = {
                'type': 'tool_result',
                'tool_use_id': block['id'],
                'content': str(error),
                'is_error': True,
            }
            event = {
                'type': 'tool_call',
                'data': {
                    'name': block['name'],
                    'status': 'failed',
                    'error': str(error),
                },
            }
        return tool_result, event

    async def _maybe_compact(self) -> dict[str, Any] | None:
        """檢查並在需要時執行上下文壓縮。

        Returns:
            壓縮結果字典，若未觸發則回傳 None
        """
        if not self.token_counter:
            return None

        if self.token_counter.usage_percent < COMPACT_THRESHOLD:
            return None

        logger.info(
            '偵測到 context window 使用率超過閾值，開始 compact',
            extra={'usage_percent': round(self.token_counter.usage_percent, 2)},
        )

        result = await compact_conversation(
            conversation=self.conversation,
            provider=self.provider,
            system_prompt=self._get_system_prompt(),
            token_counter=self.token_counter,
        )

        return result

    def _record_usage(self, final_message: Any) -> None:
        """記錄 API 使用量並更新 token 計數。"""
        if final_message.usage:
            if self.usage_monitor:
                self.usage_monitor.record(final_message.usage)
            if self.token_counter:
                self.token_counter.update_from_usage(final_message.usage)

    def _has_tool_calls(self, final_message: Any) -> bool:
        """判斷回應是否包含工具調用。"""
        return final_message.stop_reason == 'tool_use' and self.tool_registry is not None

    def _handle_stream_interruption(self, response_parts: list[str]) -> None:
        """處理串流中斷時的回應保留邏輯。"""
        if response_parts:
            partial = ''.join(response_parts)
            self.conversation.append({'role': 'assistant', 'content': partial})
            logger.warning(
                '串流中斷，已保留部分回應',
                extra={'partial_length': len(partial)},
            )
        else:
            self.conversation.pop()

    async def _execute_tool_calls(
        self,
        final_message: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """執行工具調用並 yield 事件通知。

        Args:
            final_message: 包含 tool_use block 的 API 回應

        Yields:
            工具調用事件通知（started / completed / failed）
        """
        tool_use_blocks = [b for b in final_message.content if b.get('type') == 'tool_use']

        for block in tool_use_blocks:
            logger.info(
                '執行工具',
                extra={'tool_name': block['name'], 'tool_id': block['id']},
            )
            yield {
                'type': 'tool_call',
                'data': {'name': block['name'], 'status': 'started'},
            }

        exec_results = await asyncio.gather(
            *[self._execute_single_tool(b) for b in tool_use_blocks]
        )

        # 收集結果並通知前端完成狀態
        tool_results: list[dict[str, Any]] = []
        for block, (result_val, error) in zip(tool_use_blocks, exec_results):
            tool_result, event = self._build_tool_result_entry(block, result_val, error)
            tool_results.append(tool_result)
            yield event

        self.conversation.append({'role': 'user', 'content': tool_results})
        logger.debug(
            '工具結果已回傳，繼續對話',
            extra={'tool_count': len(tool_results)},
        )

    async def _stream_with_tool_loop(
        self,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """執行串流迴圈，支援工具調用。

        Yields:
            str: 回應的每個 token
            dict: 事件通知（tool_call、preamble_end）

        Raises:
            ProviderAuthError: Provider 認證失敗
            ProviderConnectionError: Provider 連線失敗
            ProviderTimeoutError: Provider 回應超時
        """
        response_parts: list[str] = []

        try:
            while True:
                # 檢查是否需要 compact
                compact_result = await self._maybe_compact()
                if compact_result is not None:
                    yield {
                        'type': 'compact',
                        'data': compact_result,
                    }

                system = self._get_system_prompt()
                tools = self.tool_registry.get_tool_definitions() if self.tool_registry else None

                async with self.provider.stream(
                    messages=self.conversation,
                    system=system,
                    tools=tools,
                    max_tokens=self.config.provider.max_tokens,
                ) as result:
                    async for text in result.text_stream:
                        response_parts.append(text)
                        yield text

                    final_message = await result.get_final_result()

                self._record_usage(final_message)
                self.conversation.append({'role': 'assistant', 'content': final_message.content})

                # 若無工具調用，結束迴圈
                if not self._has_tool_calls(final_message):
                    logger.debug(
                        '串流回應完成',
                        extra={'response_length': len(''.join(response_parts))},
                    )
                    break

                if response_parts:
                    yield {'type': 'preamble_end', 'data': {}}

                async for event in self._execute_tool_calls(final_message):
                    yield event

                response_parts = []

        except ProviderAuthError:
            self.conversation.pop()
            raise
        except (ProviderConnectionError, ProviderTimeoutError):
            self._handle_stream_interruption(response_parts)
            raise

    async def stream_message(
        self,
        content: str,
        attachments: list[Attachment] | None = None,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """以串流方式發送訊息並逐步取得回應。

        支援工具調用迴圈：當 LLM 回傳 tool_use 時，
        自動執行工具並將結果回傳，直到取得最終文字回應。
        支援多模態輸入（圖片、PDF）。

        Args:
            content: 使用者文字訊息內容
            attachments: 附件列表（圖片或 PDF，可選）

        Yields:
            str: 回應的每個 token
            dict: 事件通知（tool_call、preamble_end）

        Raises:
            ValueError: 訊息為空白、附件格式不支援或過大
            ProviderConnectionError: Provider 連線失敗
            ProviderAuthError: Provider 認證失敗
            ProviderTimeoutError: Provider 回應超時
        """
        # 驗證輸入
        content = content.strip()
        if not content:
            raise ValueError('訊息不可為空白，請輸入有效內容')

        # 組合文字與附件為 content blocks（無附件時維持字串格式）
        message_content = build_content_blocks(content, attachments)

        # 加入使用者訊息到對話歷史
        self.conversation.append({'role': 'user', 'content': message_content})
        logger.debug('收到使用者訊息 (串流模式)', extra={'content_length': len(content)})

        # 執行串流迴圈（包含工具調用處理）
        async for token in self._stream_with_tool_loop():
            yield token
