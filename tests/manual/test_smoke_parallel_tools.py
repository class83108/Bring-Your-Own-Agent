"""並行工具執行 Smoke Test。

驗證 Agent 在 Claude 回傳多個工具調用時，能正確並行執行並回傳結果。

執行方式：
    uv run pytest tests/manual/test_smoke_parallel_tools.py -v --run-smoke
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
from agent_core.types import AgentEvent

pytestmark = pytest.mark.smoke


@pytest.fixture
def sandbox_with_files(tmp_path: Path) -> Path:
    """建立含多個檔案的 sandbox，方便觸發多工具調用。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    (sandbox / 'hello.py').write_text(
        "def hello():\n    return 'HELLO_MARKER'\n",
        encoding='utf-8',
    )
    (sandbox / 'world.py').write_text(
        "def world():\n    return 'WORLD_MARKER'\n",
        encoding='utf-8',
    )

    return sandbox


@allure.feature('Agent 核心架構')
@allure.story('Agent 應支援並行工具執行 (Smoke)')
class TestSmokeParallelTools:
    """Smoke test - 驗證多工具並行執行。"""

    @allure.title('驗證 Agent 能處理多個工具調用並回傳所有結果')
    async def test_agent_handles_multiple_tool_calls(self, sandbox_with_files: Path) -> None:
        """驗證 Agent 能處理多個工具調用並回傳所有結果。"""
        registry = create_default_registry(LocalSandbox(root=sandbox_with_files))
        prompt = '你是程式開發助手。當被要求讀取多個檔案時，請一次調用多個 read_file 工具同時讀取。'
        config = AgentCoreConfig(system_prompt=prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            tool_registry=registry,
        )

        chunks: list[str] = []
        events: list[AgentEvent] = []
        async for item in agent.stream_message(
            '請同時讀取 hello.py 和 world.py 這兩個檔案的內容，並告訴我各自的回傳值'
        ):
            if isinstance(item, str):
                chunks.append(item)
            elif item.get('type') == 'tool_call':
                events.append(item)

        response = ''.join(chunks)

        # 回應應包含兩個檔案的內容
        assert 'HELLO_MARKER' in response or 'hello' in response.lower()
        assert 'WORLD_MARKER' in response or 'world' in response.lower()

        # 應有工具調用事件（至少 2 個 started + 2 個 completed）
        started = [e for e in events if e['data']['status'] == 'started']
        completed = [e for e in events if e['data']['status'] == 'completed']
        assert len(started) >= 2, f'預期至少 2 個 started 事件，實際 {len(started)}'
        assert len(completed) >= 2, f'預期至少 2 個 completed 事件，實際 {len(completed)}'

        # 對話歷史應包含工具調用（user, assistant+tool_use, tool_results, assistant）
        assert len(agent.conversation) >= 4
