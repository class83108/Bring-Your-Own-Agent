"""OpenAI Provider 最低限度 Smoke Test。

驗證 OpenAI Provider 能正確與真實 API 串接，確認：
- SDK 回應結構與我們的 mock 假設一致
- 訊息格式轉換在真實 API 下能正常運作

執行方式：
    export OPENAI_API_KEY=your_api_key_here
    uv run pytest tests/manual/test_smoke_openai.py --run-smoke -v
"""

from __future__ import annotations

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.openai_provider import OpenAIProvider

pytestmark = pytest.mark.smoke


def _make_openai_agent() -> Agent:
    """建立使用真實 OpenAI API 的 Agent。"""
    provider_config = ProviderConfig(
        provider_type='openai',
        model='gpt-4o-mini',
        max_tokens=256,
    )
    config = AgentCoreConfig(provider=provider_config)
    provider = OpenAIProvider(provider_config)
    return Agent(config=config, provider=provider)


@allure.feature('OpenAI Provider')
@allure.story('驗證 OpenAI Provider 能正常運作 (Smoke)')
class TestSmokeOpenAI:
    """Smoke test - 驗證 OpenAI Provider 能正確與真實 API 串接。"""

    @allure.title('驗證 OpenAI Provider 能成功回應')
    async def test_openai_can_respond(self) -> None:
        """驗證透過 OpenAI Provider 發送訊息能收到回應。"""
        agent = _make_openai_agent()

        chunks: list[str] = []
        async for chunk in agent.stream_message('請回答 "OK"'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0
        assert len(agent.conversation) == 2

    @allure.title('驗證 OpenAI 串流確實分多次回傳')
    async def test_openai_stream_receives_multiple_chunks(self) -> None:
        """驗證 OpenAI 串流確實產生多個 chunk。"""
        agent = _make_openai_agent()

        chunks: list[str] = []
        async for chunk in agent.stream_message('請用 50 個字介紹 Python'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        assert len(chunks) > 1
        response = ''.join(chunks)
        assert len(response) > 20
