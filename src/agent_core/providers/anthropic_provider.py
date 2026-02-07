"""Anthropic Provider 實作。

封裝 Anthropic SDK 的串流呼叫，實作 LLMProvider 介面。
Prompt Caching 邏輯在此層處理。
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator
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
            return ProviderError(f'API 錯誤 ({error.status_code}): {error.message}')
        return ProviderError(str(error))

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        """建立串流回應。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數

        Yields:
            StreamResult 包含 text_stream 和 get_final_result

        Raises:
            ProviderAuthError: API 認證失敗
            ProviderConnectionError: API 連線失敗
            ProviderTimeoutError: API 請求超時
            ProviderError: 其他 API 錯誤
        """
        kwargs = self.build_stream_kwargs(messages, system, tools, max_tokens)

        try:
            async with self._client.messages.stream(**kwargs) as sdk_stream:
                # 暫存 final message 供 get_final_result 使用
                _final_message: FinalMessage | None = None

                async def _get_final_result() -> FinalMessage:
                    nonlocal _final_message
                    if _final_message is not None:
                        return _final_message

                    raw_msg = await sdk_stream.get_final_message()

                    # 轉換為 provider-agnostic 格式
                    content = [block.model_dump() for block in raw_msg.content]
                    usage = UsageInfo(
                        input_tokens=raw_msg.usage.input_tokens,
                        output_tokens=raw_msg.usage.output_tokens,
                        cache_creation_input_tokens=getattr(
                            raw_msg.usage, 'cache_creation_input_tokens', 0
                        )
                        or 0,
                        cache_read_input_tokens=getattr(raw_msg.usage, 'cache_read_input_tokens', 0)
                        or 0,
                    )

                    _final_message = FinalMessage(
                        content=content,
                        stop_reason=raw_msg.stop_reason or 'end_turn',
                        usage=usage,
                    )
                    return _final_message

                yield StreamResult(
                    text_stream=sdk_stream.text_stream,
                    get_final_result=_get_final_result,
                )

        except (
            AuthenticationError,
            anthropic.APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ) as e:
            raise self._convert_error(e) from e
