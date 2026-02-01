"""File Edit 工具 Smoke Test。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_file_edit.py -v --run-smoke
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
        'def old_function():\n    return "old"\n',
        encoding='utf-8',
    )

    return sandbox


class TestSmokeFileEdit:
    """Smoke test - 驗證 Agent 能透過工具編輯檔案。"""

    async def test_agent_edits_file_with_tool(self, sandbox_dir: Path) -> None:
        """驗證 Agent 能調用 edit_file 工具並修改檔案內容。"""
        registry = create_default_registry(sandbox_dir)
        prompt = '你是程式開發助手。當被要求編輯檔案時，請使用 edit_file 工具。'
        agent = Agent(
            config=AgentConfig(system_prompt=prompt),
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message(
            '請將 src/main.py 中的 old_function 重新命名為 new_function'
        ):
            chunks.append(chunk)

        response = ''.join(chunks)

        # 回應應該有內容
        assert len(response) > 0

        # 對話歷史應包含工具調用記錄
        assert len(agent.conversation) >= 4

        # 驗證檔案確實被修改
        content = (sandbox_dir / 'src' / 'main.py').read_text(encoding='utf-8')
        assert 'new_function' in content
        assert 'old_function' not in content

    async def test_agent_creates_new_file(self, sandbox_dir: Path) -> None:
        """驗證 Agent 能建立新檔案。"""
        registry = create_default_registry(sandbox_dir)
        prompt = '你是程式開發助手。當被要求建立檔案時，請使用 edit_file 工具。'
        agent = Agent(
            config=AgentConfig(system_prompt=prompt),
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message(
            '請建立一個新檔案 src/utils.py，內容為一個簡單的 helper 函數'
        ):
            chunks.append(chunk)

        response = ''.join(chunks)

        # 回應應該有內容
        assert len(response) > 0

        # 驗證檔案確實被建立
        file_path = sandbox_dir / 'src' / 'utils.py'
        assert file_path.exists()

        # 檔案應該包含一些 Python 程式碼
        content = file_path.read_text(encoding='utf-8')
        assert len(content) > 0
        assert 'def' in content or 'function' in content.lower()
