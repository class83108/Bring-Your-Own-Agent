"""File List 工具 Smoke Test。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_file_list.py -v --run-smoke
"""

from __future__ import annotations

from pathlib import Path

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.sandbox import LocalSandbox
from agent_core.tools.setup import create_default_registry

pytestmark = pytest.mark.smoke


@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """建立臨時 sandbox 目錄含測試檔案結構。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    # 建立多層目錄結構
    (sandbox / 'src').mkdir()
    (sandbox / 'src' / 'utils').mkdir()
    (sandbox / 'tests').mkdir()

    # 建立檔案
    (sandbox / 'README.md').write_text('# Test Project\n', encoding='utf-8')
    (sandbox / 'src' / 'main.py').write_text(
        "def main():\n    print('hello')\n",
        encoding='utf-8',
    )
    (sandbox / 'src' / 'utils' / 'helper.py').write_text(
        'def helper():\n    pass\n',
        encoding='utf-8',
    )
    (sandbox / 'tests' / 'test_main.py').write_text(
        'def test_main():\n    assert True\n',
        encoding='utf-8',
    )

    return sandbox


@allure.feature('檔案列表工具')
@allure.story('驗證 Agent 能透過工具列出檔案 (Smoke)')
class TestSmokeFileList:
    """Smoke test - 驗證 Agent 能透過工具列出檔案。"""

    @allure.title('驗證 Agent 能調用 list_files 工具並列出目錄內容')
    async def test_agent_lists_directory(self, sandbox_dir: Path) -> None:
        """驗證 Agent 能調用 list_files 工具並列出目錄內容。"""
        registry = create_default_registry(LocalSandbox(root=sandbox_dir))
        prompt = '你是程式開發助手。當被要求列出檔案時，請使用 list_files 工具。'
        config = AgentCoreConfig(system_prompt=prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message('請列出專案根目錄的檔案'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)

        # 回應應包含檔案或目錄名稱
        assert len(response) > 0
        assert 'README' in response or 'src' in response or 'tests' in response

        # 對話歷史應包含工具調用記錄
        assert len(agent.conversation) >= 4

    @allure.title('驗證 Agent 能遞迴列出所有 Python 檔案')
    async def test_agent_lists_recursively(self, sandbox_dir: Path) -> None:
        """驗證 Agent 能遞迴列出所有 Python 檔案。"""
        registry = create_default_registry(LocalSandbox(root=sandbox_dir))
        prompt = '你是程式開發助手。當被要求列出檔案時，請使用 list_files 工具。'
        config = AgentCoreConfig(system_prompt=prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message('請列出所有的 Python 檔案'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)

        # 回應應包含 Python 檔案名稱
        assert len(response) > 0
        # 應該提到某些 .py 檔案
        assert 'main.py' in response or 'helper.py' in response or 'test_main.py' in response

        # 對話歷史應包含工具調用記錄
        assert len(agent.conversation) >= 4
