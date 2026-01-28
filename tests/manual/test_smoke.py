"""手動執行的 Smoke Test。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual -v
"""

from __future__ import annotations

import pytest

from agent_demo.agent import Agent

pytestmark = pytest.mark.skip(reason='手動執行：uv run pytest tests/manual -v --no-header -rN')


class TestSmoke:
    """Smoke test - 驗證 Agent 能正常運作。"""

    async def test_agent_can_respond(self) -> None:
        """驗證 Agent 能成功發送訊息並收到回應。"""
        agent = Agent()

        chunks: list[str] = []
        async for chunk in agent.stream_message('請回答 "OK"'):
            chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0
        assert len(agent.conversation) == 2

    async def test_conversation_history_works(self) -> None:
        """驗證多輪對話歷史正確維護。"""
        agent = Agent()

        # 第一輪
        async for _ in agent.stream_message('我的名字是小明'):
            pass

        # 第二輪
        chunks: list[str] = []
        async for chunk in agent.stream_message('我剛才說我的名字是什麼？'):
            chunks.append(chunk)

        response = ''.join(chunks)
        # 驗證結構
        assert len(agent.conversation) == 4
        # 回應中應該提到「小明」（雖然這依賴 LLM，但這是 smoke test）
        assert '小明' in response

    async def test_stream_receives_multiple_chunks(self) -> None:
        """驗證串流確實分多次回傳。"""
        agent = Agent()

        chunks: list[str] = []
        async for chunk in agent.stream_message('請用 50 個字介紹 Python'):
            chunks.append(chunk)

        # 應該收到多個 chunk（不只是一個大塊）
        assert len(chunks) > 1
        # 完整回應應該有合理長度
        response = ''.join(chunks)
        assert len(response) > 20
