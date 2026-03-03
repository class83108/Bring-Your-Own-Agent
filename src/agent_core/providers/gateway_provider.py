"""Gateway Provider 實作。

根據模型名稱自動路由至對應的底層 Provider（Anthropic、OpenAI、Gemini），
支援 fallback 備援與 middleware 鏈。實作 LLMProvider 介面。

架構定位：
- agent_core（本模組）：路由、fallback、middleware 執行框架
- agent_app（應用層）：middleware 具體實作（cost tracking、OTel tracing 等）
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from agent_core.config import ProviderConfig
from agent_core.providers.base import FinalMessage, LLMProvider, StreamResult
from agent_core.providers.exceptions import ProviderError
from agent_core.providers.gateway_middleware import (
    GatewayMiddleware,
    RequestContext,
)
from agent_core.types import MessageParam, ToolDefinition

logger = logging.getLogger(__name__)

# 模型名稱前綴 → provider_type 映射
_MODEL_PREFIX_MAP: list[tuple[str, str]] = [
    ('claude-', 'anthropic'),
    ('gpt-', 'openai'),
    ('o1', 'openai'),
    ('o3', 'openai'),
    ('o4', 'openai'),
    ('gemini-', 'gemini'),
]

# provider_type → Provider 類別完整路徑（lazy import 避免迴圈依賴）
_PROVIDER_CLASS_MAP: dict[str, str] = {
    'anthropic': 'agent_core.providers.anthropic_provider.AnthropicProvider',
    'openai': 'agent_core.providers.openai_provider.OpenAIProvider',
    'gemini': 'agent_core.providers.gemini_provider.GeminiProvider',
}


def infer_provider_type(model: str) -> str:
    """根據模型名稱推斷 provider_type。

    Args:
        model: 模型名稱

    Returns:
        provider_type 字串（例如 'anthropic', 'openai', 'gemini'）

    Raises:
        ValueError: 無法辨識的模型名稱
    """
    for prefix, provider_type in _MODEL_PREFIX_MAP:
        if model.startswith(prefix):
            return provider_type
    known_prefixes = [p for p, _ in _MODEL_PREFIX_MAP]
    msg = (
        f'無法從模型名稱 "{model}" 推斷 provider。'
        f'已知前綴: {known_prefixes}。'
        '請明確指定 provider_type 或注入自訂 provider。'
    )
    raise ValueError(msg)


def _create_provider(config: ProviderConfig) -> LLMProvider:
    """根據 provider_type 建立對應的 Provider 實例。

    使用 importlib lazy import 避免迴圈依賴。

    Args:
        config: 已設定正確 provider_type 的配置

    Returns:
        LLMProvider 實例
    """
    class_path = _PROVIDER_CLASS_MAP[config.provider_type]
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    return provider_class(config)


class GatewayProvider:
    """Gateway LLM Provider。

    根據模型名稱自動路由至對應的底層 Provider，
    支援 fallback 備援與 middleware 鏈。

    使用方式：
        # 自動推斷（gpt-4o → OpenAI）
        gateway = GatewayProvider(ProviderConfig(model='gpt-4o'))

        # 自訂 provider
        gateway = GatewayProvider(config, provider=my_provider)

        # 帶 fallback
        gateway = GatewayProvider(
            ProviderConfig(model='claude-sonnet-4-20250514'),
            fallback_models=['gpt-4o', 'gemini-2.0-flash'],
        )

        # 帶 middleware
        gateway = GatewayProvider(
            config,
            middlewares=[my_cost_tracker, my_rate_limiter],
        )
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider: LLMProvider | None = None,
        fallback_models: list[str] | None = None,
        middlewares: list[GatewayMiddleware] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """初始化 Gateway Provider。

        Args:
            config: Provider 配置（model 欄位用於推斷路由）
            provider: 自訂 Provider 實例（注入時跳過自動推斷）
            fallback_models: 備援模型名稱列表，依序嘗試
            middlewares: Middleware 鏈
            trace_id: 外部傳入的 trace ID（OTel 相容）
            parent_span_id: 外部傳入的父 span ID
            user_id: 使用者 ID（傳遞給 RequestContext）
        """
        self._config = config
        self._middlewares = middlewares or []
        self._trace_id = trace_id
        self._parent_span_id = parent_span_id
        self._user_id = user_id

        # 建立 provider 列表（主 + fallback）
        self._providers: list[tuple[str, LLMProvider]] = []

        if provider is not None:
            # 使用者注入自訂 Provider
            self._providers.append((config.model, provider))
            logger.info(
                'GatewayProvider 使用注入的自訂 Provider',
                extra={'model': config.model},
            )
        else:
            # 自動推斷主 provider
            resolved_type = infer_provider_type(config.model)
            resolved_config = replace(config, provider_type=resolved_type)
            self._providers.append((config.model, _create_provider(resolved_config)))
            logger.info(
                'GatewayProvider 已自動路由',
                extra={'model': config.model, 'provider_type': resolved_type},
            )

        # 建立 fallback providers
        for fallback_model in fallback_models or []:
            fb_type = infer_provider_type(fallback_model)
            fb_config = replace(config, model=fallback_model, provider_type=fb_type)
            self._providers.append((fallback_model, _create_provider(fb_config)))
            logger.info(
                'GatewayProvider 已註冊 fallback',
                extra={'model': fallback_model, 'provider_type': fb_type},
            )

    def _create_context(self) -> RequestContext:
        """建立新的 RequestContext。"""
        model = self._config.model
        # 推斷 provider_type（注入自訂 provider 時可能沒有）
        try:
            provider_type = infer_provider_type(model)
        except ValueError:
            provider_type = self._config.provider_type

        kwargs: dict[str, Any] = {
            'model': model,
            'provider_type': provider_type,
            'user_id': self._user_id,
        }
        if self._trace_id is not None:
            kwargs['trace_id'] = self._trace_id
        if self._parent_span_id is not None:
            kwargs['parent_span_id'] = self._parent_span_id

        return RequestContext(**kwargs)

    async def _run_before_middlewares(self, context: RequestContext) -> None:
        """依正序執行所有 middleware 的 before_request。"""
        for mw in self._middlewares:
            await mw.before_request(context)

    async def _run_after_middlewares(self, context: RequestContext, result: FinalMessage) -> None:
        """依逆序執行所有 middleware 的 after_request。"""
        for mw in reversed(self._middlewares):
            await mw.after_request(context, result)

    @asynccontextmanager
    async def stream(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
        **kwargs: Any,
    ) -> AsyncIterator[StreamResult]:
        """串流回應，支援 fallback 與 middleware。"""
        context = self._create_context()
        await self._run_before_middlewares(context)

        last_error: ProviderError | None = None

        for model_name, provider in self._providers:
            try:
                async with provider.stream(
                    messages=messages,
                    system=system,
                    tools=tools,
                    max_tokens=max_tokens,
                    **kwargs,
                ) as result:
                    # 包裝 get_final_result 以觸發 after middleware
                    original_get_final = result.get_final_result

                    async def _get_final_with_middleware() -> FinalMessage:
                        final = await original_get_final()
                        await self._run_after_middlewares(context, final)
                        return final

                    yield StreamResult(
                        text_stream=result.text_stream,
                        get_final_result=_get_final_with_middleware,
                    )
                    return
            except ProviderError as e:
                last_error = e
                logger.warning(
                    'Provider 失敗，嘗試 fallback',
                    extra={'model': model_name, 'error': str(e)},
                )
                continue

        # 所有 provider 都失敗
        assert last_error is not None
        raise last_error

    async def create(
        self,
        messages: list[MessageParam],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        """非串流呼叫，支援 fallback 與 middleware。"""
        context = self._create_context()
        await self._run_before_middlewares(context)

        last_error: ProviderError | None = None

        for model_name, provider in self._providers:
            try:
                result = await provider.create(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                )
                await self._run_after_middlewares(context, result)
                return result
            except ProviderError as e:
                last_error = e
                logger.warning(
                    'Provider 失敗，嘗試 fallback',
                    extra={'model': model_name, 'error': str(e)},
                )
                continue

        assert last_error is not None
        raise last_error

    async def count_tokens(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        """Token 計數，委派給主 Provider（不觸發 middleware）。"""
        # count_tokens 使用主 provider，不做 fallback
        _, primary = self._providers[0]
        return await primary.count_tokens(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
        )
