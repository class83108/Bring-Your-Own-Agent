"""Gemini Provider 最低限度 Smoke Test。

驗證 Gemini Provider 能正確與真實 API 串接，確認：
- SDK 回應結構與我們的 mock 假設一致
- 訊息格式轉換在真實 API 下能正常運作

執行方式：
    export GEMINI_API_KEY=your_api_key_here
    uv run pytest tests/manual/test_smoke_gemini.py --run-smoke -v
"""

from __future__ import annotations

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.gemini_provider import GeminiProvider

pytestmark = pytest.mark.smoke


def _make_gemini_agent() -> Agent:
    """建立使用真實 Gemini API 的 Agent。"""
    provider_config = ProviderConfig(
        provider_type='gemini',
        model='gemini-2.0-flash',
        max_tokens=256,
    )
    config = AgentCoreConfig(provider=provider_config)
    provider = GeminiProvider(provider_config)
    return Agent(config=config, provider=provider)


@allure.feature('Gemini Provider')
@allure.story('驗證 Gemini Provider 能正常運作 (Smoke)')
class TestSmokeGemini:
    """Smoke test - 驗證 Gemini Provider 能正確與真實 API 串接。"""

    @allure.title('驗證 Gemini Provider 能成功回應')
    async def test_gemini_can_respond(self) -> None:
        """驗證透過 Gemini Provider 發送訊息能收到回應。"""
        agent = _make_gemini_agent()

        chunks: list[str] = []
        async for chunk in agent.stream_message('請回答 "OK"'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0
        assert len(agent.conversation) == 2

    @allure.title('驗證 Gemini 串流確實分多次回傳')
    async def test_gemini_stream_receives_multiple_chunks(self) -> None:
        """驗證 Gemini 串流確實產生多個 chunk。"""
        agent = _make_gemini_agent()

        chunks: list[str] = []
        async for chunk in agent.stream_message('請用 50 個字介紹 Python'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        assert len(chunks) > 1
        response = ''.join(chunks)
        assert len(response) > 20
