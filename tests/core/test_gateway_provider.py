"""Gateway Provider 測試模組。

根據 docs/features/core/gateway_provider.feature 規格撰寫測試案例。
涵蓋：
- Rule: GatewayProvider 應根據模型名稱自動路由至正確的 Provider
- Rule: GatewayProvider 應支援注入自訂 Provider
- Rule: GatewayProvider 應支援 Fallback 備援
- Rule: GatewayProvider 應實作 LLMProvider 介面（委派）
- Rule: GatewayProvider 應支援 Middleware 鏈
- Rule: RequestContext 應支援 OpenTelemetry 相容的 trace 結構
- Rule: GatewayProvider 應正確傳遞 ProviderConfig
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import allure
import pytest

from agent_core.config import ProviderConfig
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.providers.exceptions import ProviderConnectionError
from agent_core.providers.gateway_middleware import RequestContext
from agent_core.types import MessageParam, TextBlock

# --- 輔助工具 ---


def _make_final_message(text: str = 'Hello') -> FinalMessage:
    """建立測試用 FinalMessage。"""
    content: list[TextBlock] = [TextBlock(type='text', text=text)]
    return FinalMessage(
        content=content,  # type: ignore[arg-type]
        stop_reason='end_turn',
        usage=UsageInfo(input_tokens=10, output_tokens=20),
    )


def _make_mock_provider(final_message: FinalMessage | None = None) -> MagicMock:
    """建立 mock LLMProvider。"""
    provider = MagicMock()
    msg = final_message or _make_final_message()

    # create() 回傳 FinalMessage
    provider.create = AsyncMock(return_value=msg)

    # count_tokens() 回傳 token 數
    provider.count_tokens = AsyncMock(return_value=42)

    # stream() 回傳 async context manager
    @asynccontextmanager
    async def mock_stream(**kwargs: Any) -> AsyncIterator[StreamResult]:
        async def text_gen() -> AsyncIterator[str]:
            yield 'Hello'

        async def get_final() -> FinalMessage:
            return msg

        yield StreamResult(text_stream=text_gen(), get_final_result=get_final)

    provider.stream = mock_stream

    return provider


class _RecordingMiddleware:
    """記錄 before/after 呼叫順序的 Middleware。"""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    async def before_request(self, context: RequestContext) -> None:
        self.log.append(f'{self.name}:before')

    async def after_request(self, context: RequestContext, result: FinalMessage) -> None:
        self.log.append(f'{self.name}:after')


# --- 測試類別 ---


@allure.feature('Gateway Provider')
@allure.story('根據模型名稱自動路由至正確的 Provider')
class TestGatewayRouting:
    """Rule: GatewayProvider 應根據模型名稱自動路由至正確的 Provider。"""

    @allure.title('claude 模型路由至 Anthropic Provider')
    def test_route_claude_to_anthropic(self) -> None:
        """Scenario: claude 模型路由至 Anthropic Provider。"""
        from agent_core.providers.gateway_provider import infer_provider_type

        assert infer_provider_type('claude-sonnet-4-20250514') == 'anthropic'
        assert infer_provider_type('claude-haiku-4-20250514') == 'anthropic'
        assert infer_provider_type('claude-opus-4-20250514') == 'anthropic'

    @allure.title('gpt 模型路由至 OpenAI Provider')
    def test_route_gpt_to_openai(self) -> None:
        """Scenario: gpt 模型路由至 OpenAI Provider。"""
        from agent_core.providers.gateway_provider import infer_provider_type

        assert infer_provider_type('gpt-4o') == 'openai'
        assert infer_provider_type('gpt-4o-mini') == 'openai'
        assert infer_provider_type('gpt-4.1') == 'openai'

    @allure.title('o 系列模型路由至 OpenAI Provider')
    def test_route_o_series_to_openai(self) -> None:
        """Scenario: o 系列模型路由至 OpenAI Provider。"""
        from agent_core.providers.gateway_provider import infer_provider_type

        assert infer_provider_type('o1') == 'openai'
        assert infer_provider_type('o1-mini') == 'openai'
        assert infer_provider_type('o3') == 'openai'
        assert infer_provider_type('o3-mini') == 'openai'
        assert infer_provider_type('o4-mini') == 'openai'

    @allure.title('gemini 模型路由至 Gemini Provider')
    def test_route_gemini_to_gemini(self) -> None:
        """Scenario: gemini 模型路由至 Gemini Provider。"""
        from agent_core.providers.gateway_provider import infer_provider_type

        assert infer_provider_type('gemini-2.0-flash') == 'gemini'
        assert infer_provider_type('gemini-2.5-pro') == 'gemini'

    @allure.title('無法識別的模型名稱應拋出 ValueError')
    def test_unknown_model_raises_error(self) -> None:
        """Scenario: 無法識別的模型名稱應拋出錯誤。"""
        from agent_core.providers.gateway_provider import infer_provider_type

        with pytest.raises(ValueError, match='無法從模型名稱'):
            infer_provider_type('unknown-model-xyz')


@allure.feature('Gateway Provider')
@allure.story('支援注入自訂 Provider')
class TestGatewayCustomProvider:
    """Rule: GatewayProvider 應支援注入自訂 Provider。"""

    @allure.title('注入自訂 Provider 覆蓋自動推斷')
    async def test_custom_provider_overrides_inference(self) -> None:
        """Scenario: 注入自訂 Provider 覆蓋自動推斷。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        custom_provider = _make_mock_provider()
        config = ProviderConfig(model='any-model', api_key='test')

        gateway = GatewayProvider(config, provider=custom_provider)

        # 應使用注入的 provider
        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        result = await gateway.create(messages=messages, system='test')
        assert result.stop_reason == 'end_turn'
        custom_provider.create.assert_awaited_once()


@allure.feature('Gateway Provider')
@allure.story('Fallback 備援')
class TestGatewayFallback:
    """Rule: GatewayProvider 應支援 Fallback 備援。"""

    @allure.title('主 Provider 失敗時自動切換至 Fallback')
    async def test_fallback_on_primary_failure(self) -> None:
        """Scenario: 主 Provider 失敗時自動切換至 Fallback。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        # 主 provider 失敗
        primary = _make_mock_provider()
        primary.create = AsyncMock(side_effect=ProviderConnectionError('主 provider 失敗'))

        # fallback provider 成功
        fallback_msg = _make_final_message('Fallback response')
        fallback = _make_mock_provider(fallback_msg)

        config = ProviderConfig(model='gpt-4o', api_key='test')

        # 使用 patch 避免真正建立 provider
        with patch(
            'agent_core.providers.gateway_provider._create_provider',
            side_effect=[primary, fallback],
        ):
            gateway = GatewayProvider(config, fallback_models=['gemini-2.0-flash'])

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        result = await gateway.create(messages=messages, system='test')

        # 應使用 fallback 的回應
        assert result.content[0]['text'] == 'Fallback response'  # type: ignore[typeddict-item]

    @allure.title('所有 Provider 皆失敗時拋出最後的錯誤')
    async def test_all_providers_fail_raises_last_error(self) -> None:
        """Scenario: 所有 Provider 皆失敗時拋出最後的錯誤。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        primary = _make_mock_provider()
        primary.create = AsyncMock(side_effect=ProviderConnectionError('primary failed'))

        fallback = _make_mock_provider()
        fallback.create = AsyncMock(side_effect=ProviderConnectionError('fallback failed'))

        config = ProviderConfig(model='gpt-4o', api_key='test')

        with patch(
            'agent_core.providers.gateway_provider._create_provider',
            side_effect=[primary, fallback],
        ):
            gateway = GatewayProvider(config, fallback_models=['gemini-2.0-flash'])

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        with pytest.raises(ProviderConnectionError, match='fallback failed'):
            await gateway.create(messages=messages, system='test')

    @allure.title('stream() 支援 Fallback')
    async def test_stream_fallback(self) -> None:
        """Scenario: stream() 也支援 Fallback。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        # 主 provider stream 失敗
        primary = _make_mock_provider()

        @asynccontextmanager
        async def failing_stream(**kwargs: Any) -> AsyncIterator[StreamResult]:
            raise ProviderConnectionError('stream failed')
            yield  # pragma: no cover

        primary.stream = failing_stream

        # fallback provider stream 成功
        fallback = _make_mock_provider(_make_final_message('Fallback'))

        config = ProviderConfig(model='gpt-4o', api_key='test')

        with patch(
            'agent_core.providers.gateway_provider._create_provider',
            side_effect=[primary, fallback],
        ):
            gateway = GatewayProvider(config, fallback_models=['gemini-2.0-flash'])

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        async with gateway.stream(messages=messages, system='test') as result:
            chunks: list[str] = []
            async for text in result.text_stream:
                chunks.append(text)
            assert len(chunks) > 0


@allure.feature('Gateway Provider')
@allure.story('LLMProvider 介面委派')
class TestGatewayDelegation:
    """Rule: GatewayProvider 應實作 LLMProvider 介面。"""

    @allure.title('create() 委派給底層 Provider')
    async def test_create_delegates(self) -> None:
        """Scenario: create() 委派給底層 Provider。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(config, provider=mock_provider)

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        result = await gateway.create(messages=messages, system='test')

        assert result.stop_reason == 'end_turn'
        mock_provider.create.assert_awaited_once()

    @allure.title('count_tokens() 委派給底層 Provider')
    async def test_count_tokens_delegates(self) -> None:
        """Scenario: count_tokens() 委派給底層 Provider。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(config, provider=mock_provider)

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        count = await gateway.count_tokens(messages=messages, system='test')

        assert count == 42
        mock_provider.count_tokens.assert_awaited_once()

    @allure.title('stream() 委派給底層 Provider')
    async def test_stream_delegates(self) -> None:
        """Scenario: stream() 委派給底層 Provider。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(config, provider=mock_provider)

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        async with gateway.stream(messages=messages, system='test') as result:
            chunks: list[str] = []
            async for text in result.text_stream:
                chunks.append(text)
            final = await result.get_final_result()

        assert chunks == ['Hello']
        assert final.stop_reason == 'end_turn'


@allure.feature('Gateway Provider')
@allure.story('Middleware 鏈')
class TestGatewayMiddleware:
    """Rule: GatewayProvider 應支援 Middleware 鏈。"""

    @allure.title('Middleware 在 create() 前後依序執行')
    async def test_middleware_execution_order(self) -> None:
        """Scenario: before_request 正序、after_request 逆序。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        log: list[str] = []
        mw1 = _RecordingMiddleware('mw1', log)
        mw2 = _RecordingMiddleware('mw2', log)

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(config, provider=mock_provider, middlewares=[mw1, mw2])

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        await gateway.create(messages=messages, system='test')

        assert log == ['mw1:before', 'mw2:before', 'mw2:after', 'mw1:after']

    @allure.title('Middleware 收到包含 trace 資訊的 RequestContext')
    async def test_middleware_receives_context(self) -> None:
        """Scenario: Middleware 透過 RequestContext 共享資訊。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        captured_contexts: list[RequestContext] = []

        class CapturingMiddleware:
            async def before_request(self, context: RequestContext) -> None:
                captured_contexts.append(context)

            async def after_request(self, context: RequestContext, result: FinalMessage) -> None:
                captured_contexts.append(context)

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(
            config,
            provider=mock_provider,
            middlewares=[CapturingMiddleware()],
            user_id='user-123',
        )

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        await gateway.create(messages=messages, system='test')

        assert len(captured_contexts) == 2
        ctx = captured_contexts[0]
        assert ctx.model == 'gpt-4o'
        assert ctx.user_id == 'user-123'
        assert len(ctx.trace_id) > 0
        assert len(ctx.span_id) > 0
        # before 和 after 收到同一個 context
        assert captured_contexts[0] is captured_contexts[1]

    @allure.title('Middleware 在 stream() 前後也執行')
    async def test_middleware_with_stream(self) -> None:
        """Scenario: stream() 也觸發 middleware。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        log: list[str] = []
        mw = _RecordingMiddleware('mw', log)

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(config, provider=mock_provider, middlewares=[mw])

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        async with gateway.stream(messages=messages, system='test') as result:
            async for _ in result.text_stream:
                pass
            await result.get_final_result()

        assert 'mw:before' in log
        assert 'mw:after' in log

    @allure.title('count_tokens() 不觸發 middleware')
    async def test_count_tokens_skips_middleware(self) -> None:
        """Scenario: count_tokens() 不是 LLM 呼叫，不觸發 middleware。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        log: list[str] = []
        mw = _RecordingMiddleware('mw', log)

        mock_provider = _make_mock_provider()
        config = ProviderConfig(model='gpt-4o', api_key='test')
        gateway = GatewayProvider(config, provider=mock_provider, middlewares=[mw])

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        await gateway.count_tokens(messages=messages, system='test')

        assert log == []


@allure.feature('Gateway Provider')
@allure.story('RequestContext trace 結構')
class TestRequestContext:
    """Rule: RequestContext 應支援 OpenTelemetry 相容的 trace 結構。"""

    @allure.title('自動產生 trace_id 和 span_id')
    def test_auto_generate_ids(self) -> None:
        """Scenario: 自動產生 trace_id 和 span_id。"""
        ctx = RequestContext(model='gpt-4o', provider_type='openai')

        assert len(ctx.trace_id) == 32  # uuid4().hex 長度
        assert len(ctx.span_id) == 32
        assert ctx.parent_span_id is None

    @allure.title('接受外部傳入的 trace context')
    def test_external_trace_context(self) -> None:
        """Scenario: 接受外部傳入的 trace context。"""
        ctx = RequestContext(
            trace_id='external-trace-id',
            parent_span_id='parent-span-id',
            model='gpt-4o',
            provider_type='openai',
        )

        assert ctx.trace_id == 'external-trace-id'
        assert ctx.parent_span_id == 'parent-span-id'
        # span_id 仍然是自動產生的
        assert len(ctx.span_id) == 32
        assert ctx.span_id != 'external-trace-id'

    @allure.title('不同 RequestContext 有不同的 span_id')
    def test_unique_span_ids(self) -> None:
        """每次建立 RequestContext 應有不同的 span_id。"""
        ctx1 = RequestContext(model='gpt-4o', provider_type='openai')
        ctx2 = RequestContext(model='gpt-4o', provider_type='openai')

        assert ctx1.span_id != ctx2.span_id


@allure.feature('Gateway Provider')
@allure.story('ProviderConfig 傳遞')
class TestGatewayConfig:
    """Rule: GatewayProvider 應正確傳遞 ProviderConfig。"""

    @allure.title('provider_type 根據模型名稱自動設定')
    def test_provider_type_auto_set(self) -> None:
        """Scenario: provider_type 應根據模型名稱自動設定。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        config = ProviderConfig(model='gpt-4o', api_key='test')

        with patch('agent_core.providers.gateway_provider._create_provider') as mock_create:
            mock_create.return_value = _make_mock_provider()
            GatewayProvider(config)

            # 傳給 _create_provider 的 config 應有 provider_type='openai'
            called_config = mock_create.call_args[0][0]
            assert called_config.provider_type == 'openai'

    @allure.title('原始 config 不被修改')
    def test_original_config_not_mutated(self) -> None:
        """Scenario: 使用 dataclasses.replace() 避免修改原始 config。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        config = ProviderConfig(provider_type='gateway', model='gpt-4o', api_key='test')

        with patch('agent_core.providers.gateway_provider._create_provider') as mock_create:
            mock_create.return_value = _make_mock_provider()
            GatewayProvider(config)

        # 原始 config 應保持不變
        assert config.provider_type == 'gateway'

    @allure.title('使用者指定的 api_key 應傳遞給底層 Provider')
    def test_api_key_passed_through(self) -> None:
        """Scenario: 使用者指定的 api_key 應直接傳遞。"""
        from agent_core.providers.gateway_provider import GatewayProvider

        config = ProviderConfig(model='gpt-4o', api_key='sk-my-key')

        with patch('agent_core.providers.gateway_provider._create_provider') as mock_create:
            mock_create.return_value = _make_mock_provider()
            GatewayProvider(config)

            called_config = mock_create.call_args[0][0]
            assert called_config.api_key == 'sk-my-key'
