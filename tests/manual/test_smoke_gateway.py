"""Gateway Provider Smoke Test。

驗證 GatewayProvider 能正確路由至各個真實 API，確認：
- 模型名稱推斷與 provider 建立能正常運作
- 各 provider 的串流回應通過 gateway 後仍正確

執行方式：
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    export GEMINI_API_KEY=your_key
    uv run pytest tests/manual/test_smoke_gateway.py --run-smoke -v
"""

from __future__ import annotations

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.gateway_provider import GatewayProvider

pytestmark = pytest.mark.smoke


def _make_gateway_agent(model: str) -> Agent:
    """建立使用 GatewayProvider 的 Agent。"""
    provider_config = ProviderConfig(model=model, max_tokens=256)
    config = AgentCoreConfig(provider=provider_config)
    provider = GatewayProvider(provider_config)
    return Agent(config=config, provider=provider)


@allure.feature('Gateway Provider')
@allure.story('驗證 GatewayProvider 路由至各 Provider (Smoke)')
class TestSmokeGateway:
    """Smoke test - 驗證 GatewayProvider 能正確路由至真實 API。"""

    @allure.title('Gateway 路由至 OpenAI (gpt-4o-mini)')
    async def test_gateway_routes_to_openai(self) -> None:
        """驗證 GatewayProvider 使用 gpt-4o-mini 能成功回應。"""
        agent = _make_gateway_agent('gpt-4o-mini')

        chunks: list[str] = []
        async for chunk in agent.stream_message('請回答 "OK"'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0
        assert len(agent.conversation) == 2

    @allure.title('Gateway 路由至 Anthropic (claude-sonnet-4-20250514)')
    async def test_gateway_routes_to_anthropic(self) -> None:
        """驗證 GatewayProvider 使用 claude-sonnet 能成功回應。"""
        agent = _make_gateway_agent('claude-sonnet-4-20250514')

        chunks: list[str] = []
        async for chunk in agent.stream_message('請回答 "OK"'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0
        assert len(agent.conversation) == 2

    @allure.title('Gateway 路由至 Gemini (gemini-2.0-flash)')
    async def test_gateway_routes_to_gemini(self) -> None:
        """驗證 GatewayProvider 使用 gemini-2.0-flash 能成功回應。"""
        agent = _make_gateway_agent('gemini-2.0-flash')

        chunks: list[str] = []
        async for chunk in agent.stream_message('請回答 "OK"'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0
        assert len(agent.conversation) == 2
