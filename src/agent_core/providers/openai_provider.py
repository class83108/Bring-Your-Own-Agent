"""OpenAI Provider 實作。

封裝 OpenAI SDK 的串流呼叫，實作 LLMProvider 介面。
負責內部 MessageParam 格式與 OpenAI API 格式之間的雙向轉換。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import openai
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncStream,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from agent_core.config import ProviderConfig
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from agent_core.types import (
    ContentBlock,
    MessageParam,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


# OpenAI finish_reason 到標準 StopReason 的映射
_OPENAI_STOP_REASON_MAP: dict[str, StopReason] = {
    'stop': 'end_turn',
    'tool_calls': 'tool_use',
    'length': 'max_tokens',
}

# 預設的 StopReason（當 vendor 回傳 None 或未知值時使用）
_DEFAULT_STOP_REASON: StopReason = 'end_turn'


class OpenAIProvider:
    """OpenAI LLM Provider。

    負責與 OpenAI API 互動，將 SDK 特定邏輯封裝在此層。
    包含訊息格式轉換、例外轉換、重試等功能。
    """

    def __init__(
        self,
        config: ProviderConfig,
        client: Any = None,
    ) -> None:
        """初始化 OpenAI Provider。

        Args:
            config: Provider 配置
            client: 自訂 OpenAI client（主要用於測試注入 mock）
        """
        self._config = config
        self._client = client or openai.AsyncOpenAI(
            api_key=config.get_api_key(),
            timeout=config.timeout,
        )

    # --- 訊息格式轉換 ---

    def _convert_messages(
        self,
        messages: list[MessageParam],
        system: str,
    ) -> list[dict[str, Any]]:
        """將內部 MessageParam 格式轉換為 OpenAI API 訊息格式。

        主要轉換：
        1. system prompt → {"role": "system"} 訊息（放在最前面）
        2. assistant 訊息中的 ToolUseBlock → tool_calls 格式
        3. user 訊息中的 ToolResultBlock → 獨立的 {"role": "tool"} 訊息

        Args:
            messages: 內部格式的對話訊息列表
            system: 系統提示詞

        Returns:
            OpenAI API 格式的訊息列表
        """
        result: list[dict[str, Any]] = []

        # System prompt 作為第一則訊息
        if system:
            result.append({'role': 'system', 'content': system})

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'assistant':
                result.append(self._convert_assistant_message(content))
            elif role == 'user':
                result.extend(self._convert_user_message(content))

        return result

    def _convert_assistant_message(
        self,
        content: str | list[ContentBlock],
    ) -> dict[str, Any]:
        """轉換 assistant 訊息。

        若 content 包含 ToolUseBlock，需拆分為 content 文字和 tool_calls。
        """
        if isinstance(content, str):
            return {'role': 'assistant', 'content': content}

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if block['type'] == 'text':
                text_parts.append(block['text'])
            elif block['type'] == 'tool_use':
                tool_calls.append(
                    {
                        'id': block['id'],
                        'type': 'function',
                        'function': {
                            'name': block['name'],
                            'arguments': json.dumps(block['input'], ensure_ascii=False),
                        },
                    }
                )

        msg: dict[str, Any] = {
            'role': 'assistant',
            'content': ''.join(text_parts) or None,
        }
        if tool_calls:
            msg['tool_calls'] = tool_calls

        return msg

    def _convert_user_message(
        self,
        content: str | list[ContentBlock],
    ) -> list[dict[str, Any]]:
        """轉換 user 訊息。

        若 content 包含 ToolResultBlock，每個 tool_result 轉為獨立的
        {"role": "tool"} 訊息。其餘內容保留為 user 訊息。
        """
        if isinstance(content, str):
            return [{'role': 'user', 'content': content}]

        result: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for block in content:
            if block['type'] == 'tool_result':
                result.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'],
                        'content': block['content'],
                    }
                )
            elif block['type'] == 'text':
                text_parts.append(block['text'])

        # 若有非 tool_result 的文字內容，保留為 user 訊息（放在前面）
        if text_parts:
            result.insert(0, {'role': 'user', 'content': ''.join(text_parts)})

        return result

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """將標準 ToolDefinition 轉換為 OpenAI function calling 格式。

        Args:
            tools: 標準工具定義列表

        Returns:
            OpenAI 格式的工具定義列表
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': t['name'],
                    'description': t['description'],
                    'parameters': t['input_schema'],
                },
            }
            for t in tools
        ]

    # --- 回應解析 ---

    @staticmethod
    def _extract_cached_tokens(usage: CompletionUsage | None) -> int:
        """從 OpenAI usage 物件提取 cached_tokens。"""
        if usage is None:
            return 0
        details = usage.prompt_tokens_details
        if details is None:
            return 0
        return details.cached_tokens or 0

    def _parse_response(self, response: ChatCompletion) -> FinalMessage:
        """將 OpenAI ChatCompletion 回應轉為 FinalMessage。

        Args:
            response: OpenAI SDK 的 ChatCompletion 物件

        Returns:
            轉換後的 FinalMessage
        """
        choice = response.choices[0]
        message = choice.message

        content: list[ContentBlock] = []

        # 文字內容
        if message.content:
            content.append(cast(ContentBlock, TextBlock(type='text', text=message.content)))

        # 工具調用（只處理 function 類型的 tool_calls）
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.type != 'function':
                    continue
                content.append(
                    cast(
                        ContentBlock,
                        ToolUseBlock(
                            type='tool_use',
                            id=tc.id,
                            name=tc.function.name,
                            input=json.loads(tc.function.arguments),
                        ),
                    )
                )

        # Usage（含 prompt caching）
        usage = UsageInfo(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            cache_read_input_tokens=self._extract_cached_tokens(response.usage),
        )

        # Stop reason 映射
        stop_reason = _OPENAI_STOP_REASON_MAP.get(choice.finish_reason or '', _DEFAULT_STOP_REASON)

        return FinalMessage(content=content, stop_reason=stop_reason, usage=usage)

    # --- 錯誤轉換與重試 ---

    # 可重試的 HTTP 狀態碼：429 (Rate Limit)、5xx (伺服器錯誤)
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def _convert_error(self, error: openai.APIError) -> ProviderError:
        """將 OpenAI SDK 例外轉換為通用 Provider 例外。

        Args:
            error: OpenAI SDK 例外

        Returns:
            對應的 Provider 例外
        """
        if isinstance(error, AuthenticationError):
            return ProviderAuthError(
                'API 金鑰無效或已過期。請檢查 OPENAI_API_KEY 環境變數是否正確設定。'
            )
        if isinstance(error, APITimeoutError):
            return ProviderTimeoutError('API 請求超時。')
        if isinstance(error, APIConnectionError):
            return ProviderConnectionError('API 連線失敗，請檢查網路連線並稍後重試。')
        if isinstance(error, RateLimitError):
            return ProviderRateLimitError(f'API 速率限制: {error.message}')
        if isinstance(error, APIStatusError):
            return ProviderError(f'API 錯誤 ({error.status_code}): {error.message}')
        return ProviderError(str(error))

    def _is_retryable(self, error: Exception) -> bool:
        """判斷錯誤是否可重試。"""
        if isinstance(error, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in self._RETRYABLE_STATUS_CODES
        return False

    def _check_retryable_or_raise(self, error: openai.APIError, attempt: int) -> None:
        """檢查錯誤是否可重試，不可重試則直接拋出。"""
        if not self._is_retryable(error) or attempt >= self._config.max_retries:
            raise self._convert_error(error) from error

    async def _wait_for_retry(
        self,
        attempt: int,
        error: Exception,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> None:
        """等待指數退避延遲並通知回調。"""
        delay = self._config.retry_initial_delay * (2**attempt)
        logger.warning(
            '可重試錯誤，準備重試',
            extra={
                'attempt': attempt + 1,
                'max_retries': self._config.max_retries,
                'delay': delay,
                'error': str(error),
            },
        )
        if on_retry:
            on_retry(attempt + 1, error, delay)
        await asyncio.sleep(delay)

    async def _retry(
        self,
        fn: Callable[[], Any],
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> Any:
        """以指數退避重試執行 async 函數。"""
        for attempt in range(1 + self._config.max_retries):
            try:
                return await fn()
            except (
                AuthenticationError,
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            ) as e:
                self._check_retryable_or_raise(e, attempt)
                await self._wait_for_retry(attempt, e, on_retry)

    # --- 公開 API ---

    @asynccontextmanager
    async def stream(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> AsyncIterator[StreamResult]:
        """建立串流回應，支援自動重試。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數
            on_retry: 重試回調函數（attempt, error, delay）

        Yields:
            StreamResult 包含 text_stream 和 get_final_result

        Raises:
            ProviderAuthError: API 認證失敗
            ProviderConnectionError: API 連線失敗
            ProviderTimeoutError: API 請求超時
            ProviderRateLimitError: API 速率限制，重試耗盡
            ProviderError: 其他 API 錯誤
        """
        openai_messages = self._convert_messages(messages, system)
        kwargs: dict[str, Any] = {
            'model': self._config.model,
            'max_tokens': max_tokens,
            'messages': openai_messages,
            'stream': True,
            'stream_options': {'include_usage': True},
            'timeout': self._config.timeout,
        }
        if tools:
            kwargs['tools'] = self._convert_tools(tools)

        for attempt in range(1 + self._config.max_retries):
            try:
                # **kwargs 讓 SDK 無法推導 overload，在反序列化邊界 cast 為正確型別
                response_stream = cast(
                    AsyncStream[ChatCompletionChunk],
                    await self._client.chat.completions.create(**kwargs),
                )

                # 累積狀態（在 closure 中共享）
                _final_message: FinalMessage | None = None
                _text_content: str = ''
                _tool_calls_acc: dict[int, dict[str, Any]] = {}
                _finish_reason: str | None = None
                _usage: CompletionUsage | None = None

                async def _text_stream_iter() -> AsyncIterator[str]:
                    nonlocal _text_content, _finish_reason, _usage
                    async for chunk in response_stream:
                        if not chunk.choices:
                            if chunk.usage:
                                _usage = chunk.usage
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        if choice.finish_reason:
                            _finish_reason = choice.finish_reason

                        # 文字 delta
                        if delta.content:
                            _text_content += delta.content
                            yield delta.content

                        # 工具調用 delta（需要跨 chunk 累積）
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in _tool_calls_acc:
                                    _tool_calls_acc[idx] = {
                                        'id': tc_delta.id or '',
                                        'name': '',
                                        'arguments': '',
                                    }
                                if tc_delta.id:
                                    _tool_calls_acc[idx]['id'] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        _tool_calls_acc[idx]['name'] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        _tool_calls_acc[idx]['arguments'] += (
                                            tc_delta.function.arguments
                                        )

                        # usage 在最後的 chunk
                        if chunk.usage:
                            _usage = chunk.usage

                async def _get_final_result() -> FinalMessage:
                    nonlocal _final_message
                    if _final_message is not None:
                        return _final_message

                    content: list[ContentBlock] = []
                    if _text_content:
                        content.append(
                            cast(ContentBlock, TextBlock(type='text', text=_text_content))
                        )

                    for idx in sorted(_tool_calls_acc.keys()):
                        tc = _tool_calls_acc[idx]
                        content.append(
                            cast(
                                ContentBlock,
                                ToolUseBlock(
                                    type='tool_use',
                                    id=tc['id'],
                                    name=tc['name'],
                                    input=json.loads(tc['arguments']) if tc['arguments'] else {},
                                ),
                            )
                        )

                    usage = UsageInfo(
                        input_tokens=_usage.prompt_tokens if _usage else 0,
                        output_tokens=_usage.completion_tokens if _usage else 0,
                        cache_read_input_tokens=self._extract_cached_tokens(_usage),
                    )

                    stop_reason = _OPENAI_STOP_REASON_MAP.get(
                        _finish_reason or '', _DEFAULT_STOP_REASON
                    )

                    _final_message = FinalMessage(
                        content=content,
                        stop_reason=stop_reason,
                        usage=usage,
                    )
                    return _final_message

                yield StreamResult(
                    text_stream=_text_stream_iter(),
                    get_final_result=_get_final_result,
                )
                return

            except (
                AuthenticationError,
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            ) as e:
                self._check_retryable_or_raise(e, attempt)
                await self._wait_for_retry(attempt, e, on_retry)

    async def count_tokens(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        """使用 tiktoken 本地計算 token 數。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數（未使用，保持介面一致）

        Returns:
            估算的 input token 數量
        """
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self._config.model)
        except KeyError:
            enc = tiktoken.get_encoding('cl100k_base')

        openai_messages = self._convert_messages(messages, system)

        # OpenAI token 計算公式（參考官方 cookbook）
        tokens_per_message = 3  # 每則訊息的 overhead
        token_count = 0

        for msg in openai_messages:
            token_count += tokens_per_message
            for value in msg.values():
                if isinstance(value, str):
                    token_count += len(enc.encode(value))
                elif isinstance(value, list):
                    # tool_calls 等複雜結構
                    token_count += len(enc.encode(json.dumps(value, ensure_ascii=False)))

        token_count += 3  # reply priming tokens

        # 工具定義的 token
        if tools:
            tool_str = json.dumps(self._convert_tools(tools), ensure_ascii=False)
            token_count += len(enc.encode(tool_str))

        return token_count

    async def create(
        self,
        messages: list[MessageParam],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        """非串流呼叫，支援自動重試。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            max_tokens: 最大回應 token 數

        Returns:
            完整的回應訊息

        Raises:
            ProviderAuthError: API 認證失敗
            ProviderConnectionError: API 連線失敗
            ProviderTimeoutError: API 請求超時
            ProviderError: 其他 API 錯誤
        """
        openai_messages = self._convert_messages(messages, system)

        async def _call() -> FinalMessage:
            # 內部轉換後的 messages 格式符合 OpenAI API，在反序列化邊界 cast
            raw_response = await self._client.chat.completions.create(
                model=self._config.model,
                max_tokens=max_tokens,
                messages=openai_messages,  # type: ignore[arg-type]
                timeout=self._config.timeout,
            )
            return self._parse_response(raw_response)

        return await self._retry(_call)
