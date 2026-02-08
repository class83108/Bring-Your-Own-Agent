"""Anthropic Provider 實作。

封裝 Anthropic SDK 的串流呼叫，實作 LLMProvider 介面。
Prompt Caching 邏輯在此層處理。
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anthropic
from anthropic import APIConnectionError, APIStatusError, AuthenticationError

from agent_core.config import ProviderConfig
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic LLM Provider。

    負責與 Anthropic API 互動，將 SDK 特定邏輯封裝在此層。
    包含 Prompt Caching、例外轉換等功能。
    """

    def __init__(
        self,
        config: ProviderConfig,
        client: Any = None,
    ) -> None:
        """初始化 Anthropic Provider。

        Args:
            config: Provider 配置
            client: 自訂 Anthropic client（主要用於測試注入 mock）
        """
        self._config = config
        self._client = client or anthropic.AsyncAnthropic(
            api_key=config.get_api_key(),
            timeout=config.timeout,
        )

    def _prepare_messages_with_cache(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """為訊息列表的最後一則添加 cache_control。

        不修改原始 messages。

        Args:
            messages: 對話訊息列表

        Returns:
            帶有 cache_control 的訊息列表副本
        """
        if not messages:
            return []

        msgs = copy.deepcopy(messages)
        last_msg = msgs[-1]
        content = last_msg.get('content')

        if isinstance(content, str):
            last_msg['content'] = [
                {
                    'type': 'text',
                    'text': content,
                    'cache_control': {'type': 'ephemeral'},
                }
            ]
        elif isinstance(content, list) and content:
            content[-1]['cache_control'] = {'type': 'ephemeral'}

        return msgs

    def build_stream_kwargs(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """建立 Anthropic messages.stream() 的參數。

        處理 Prompt Caching：在 system prompt、最後一個工具定義、
        最後一則訊息加上 cache_control。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數

        Returns:
            API 呼叫參數字典
        """
        # 處理 messages 快取
        if self._config.enable_prompt_caching:
            prepared_messages = self._prepare_messages_with_cache(messages)
        else:
            prepared_messages = messages

        kwargs: dict[str, Any] = {
            'model': self._config.model,
            'max_tokens': max_tokens or self._config.max_tokens,
            'messages': prepared_messages,
            'timeout': self._config.timeout,
        }

        # 處理 system prompt（含 cache_control）
        if self._config.enable_prompt_caching:
            kwargs['system'] = [
                {
                    'type': 'text',
                    'text': system,
                    'cache_control': {'type': 'ephemeral'},
                }
            ]
        else:
            kwargs['system'] = system

        # 處理工具定義（含 cache_control）
        if tools:
            tool_defs = copy.deepcopy(tools)
            if self._config.enable_prompt_caching and tool_defs:
                tool_defs[-1]['cache_control'] = {'type': 'ephemeral'}
            kwargs['tools'] = tool_defs

        return kwargs

    # 可重試的 HTTP 狀態碼：429 (Rate Limit)、5xx (伺服器錯誤)
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}

    def _convert_error(self, error: anthropic.APIError) -> ProviderError:
        """將 Anthropic SDK 例外轉換為通用 Provider 例外。

        Args:
            error: Anthropic SDK 例外

        Returns:
            對應的 Provider 例外
        """
        if isinstance(error, AuthenticationError):
            return ProviderAuthError(
                'API 金鑰無效或已過期。請檢查 ANTHROPIC_API_KEY 環境變數是否正確設定。'
            )
        if isinstance(error, anthropic.APITimeoutError):
            return ProviderTimeoutError('API 請求超時。')
        if isinstance(error, APIConnectionError):
            return ProviderConnectionError('API 連線失敗，請檢查網路連線並稍後重試。')
        if isinstance(error, APIStatusError):
            if error.status_code == 429:
                return ProviderRateLimitError(
                    f'API 速率限制 ({error.status_code}): {error.message}'
                )
            return ProviderError(f'API 錯誤 ({error.status_code}): {error.message}')
        return ProviderError(str(error))

    def _is_retryable(self, error: Exception) -> bool:
        """判斷錯誤是否可重試。

        可重試的錯誤包含：429、5xx 狀態碼、超時、連線失敗。
        不可重試的錯誤包含：401（認證）、400（請求無效）等。

        Args:
            error: 原始 SDK 例外

        Returns:
            是否可重試
        """
        if isinstance(error, (anthropic.APITimeoutError, APIConnectionError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in self._RETRYABLE_STATUS_CODES
        return False

    def _parse_final_message(self, raw_msg: Any) -> FinalMessage:
        """將 SDK 原始回應轉換為 provider-agnostic 格式。

        Args:
            raw_msg: Anthropic SDK 的原始 Message 物件

        Returns:
            轉換後的 FinalMessage
        """
        content = [block.model_dump() for block in raw_msg.content]
        usage = UsageInfo(
            input_tokens=raw_msg.usage.input_tokens,
            output_tokens=raw_msg.usage.output_tokens,
            cache_creation_input_tokens=getattr(raw_msg.usage, 'cache_creation_input_tokens', 0)
            or 0,
            cache_read_input_tokens=getattr(raw_msg.usage, 'cache_read_input_tokens', 0) or 0,
        )
        return FinalMessage(
            content=content,
            stop_reason=raw_msg.stop_reason or 'end_turn',
            usage=usage,
        )

    def _check_retryable_or_raise(self, error: anthropic.APIError, attempt: int) -> None:
        """檢查錯誤是否可重試，不可重試則直接拋出。

        Args:
            error: 原始 SDK 例外
            attempt: 目前重試次數（0-based）

        Raises:
            ProviderError: 不可重試或已超過最大重試次數
        """
        if not self._is_retryable(error) or attempt >= self._config.max_retries:
            raise self._convert_error(error) from error

    async def _wait_for_retry(
        self,
        attempt: int,
        error: Exception,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> None:
        """等待指數退避延遲並通知回調。

        Args:
            attempt: 目前重試次數（0-based）
            error: 觸發重試的原始錯誤
            on_retry: 重試回調函數（attempt, error, delay）
        """
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
        """以指數退避重試執行 async 函數。

        Args:
            fn: 要執行的 async 函數
            on_retry: 重試回調函數（attempt, error, delay）

        Returns:
            fn 的回傳值

        Raises:
            ProviderError: 不可重試錯誤或重試耗盡
        """
        for attempt in range(1 + self._config.max_retries):
            try:
                return await fn()
            except (
                AuthenticationError,
                anthropic.APITimeoutError,
                APIConnectionError,
                APIStatusError,
            ) as e:
                self._check_retryable_or_raise(e, attempt)
                await self._wait_for_retry(attempt, e, on_retry)

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
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
        kwargs = self.build_stream_kwargs(messages, system, tools, max_tokens)

        for attempt in range(1 + self._config.max_retries):
            try:
                async with self._client.messages.stream(**kwargs) as sdk_stream:
                    _final_message: FinalMessage | None = None

                    async def _get_final_result() -> FinalMessage:
                        nonlocal _final_message
                        if _final_message is not None:
                            return _final_message

                        raw_msg = await sdk_stream.get_final_message()
                        _final_message = self._parse_final_message(raw_msg)
                        return _final_message

                    yield StreamResult(
                        text_stream=sdk_stream.text_stream,
                        get_final_result=_get_final_result,
                    )
                    return

            except (
                AuthenticationError,
                anthropic.APITimeoutError,
                APIConnectionError,
                APIStatusError,
            ) as e:
                self._check_retryable_or_raise(e, attempt)
                await self._wait_for_retry(attempt, e, on_retry)

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        """計算給定訊息的 token 數量，支援自動重試。

        使用 Anthropic token counting API 精確計算。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數

        Returns:
            input token 數量

        Raises:
            ProviderAuthError: API 認證失敗
            ProviderConnectionError: API 連線失敗
            ProviderError: 其他 API 錯誤
        """
        kwargs: dict[str, Any] = {
            'model': self._config.model,
            'messages': messages,
            'system': system,
            'max_tokens': max_tokens,
        }
        if tools:
            kwargs['tools'] = tools

        async def _call() -> int:
            result = await self._client.messages.count_tokens(**kwargs)
            return result.input_tokens

        return await self._retry(_call)

    async def create(
        self,
        messages: list[dict[str, Any]],
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

        async def _call() -> FinalMessage:
            raw_msg = await self._client.messages.create(
                model=self._config.model,
                max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
                system=system,
                timeout=self._config.timeout,
            )
            return self._parse_final_message(raw_msg)

        return await self._retry(_call)
