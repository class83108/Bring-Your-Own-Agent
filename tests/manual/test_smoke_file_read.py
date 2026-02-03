"""File Read 工具 Smoke Test。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_file_read.py -v --run-smoke
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_demo.agent import Agent, AgentConfig
from agent_demo.tools.setup import create_default_registry

pytestmark = pytest.mark.smoke


@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """建立臨時 sandbox 目錄含測試檔案。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    (sandbox / 'src').mkdir()
    (sandbox / 'src' / 'main.py').write_text(
        "def hello():\n    print('world')\n",
        encoding='utf-8',
    )

    return sandbox


class TestSmokeFileRead:
    """Smoke test - 驗證 Agent 能透過工具讀取檔案。"""

    async def test_agent_reads_file_with_tool(self, sandbox_dir: Path) -> None:
        """驗證 Agent 能調用 read_file 工具並回傳檔案內容。"""
        registry = create_default_registry(sandbox_dir)
        prompt = '你是程式開發助手。當被要求讀取檔案時，請使用 read_file 工具。'
        agent = Agent(
            config=AgentConfig(system_prompt=prompt),
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message('請讀取 src/main.py 的內容'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)

        # 回應應包含檔案內容的關鍵片段
        assert len(response) > 0
        assert 'hello' in response or 'world' in response or 'print' in response

        # 對話歷史應包含工具調用記錄（至少 4 條：user, assistant+tool_use, tool_result, assistant）
        assert len(agent.conversation) >= 4
