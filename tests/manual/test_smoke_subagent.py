"""Subagent Smoke Test。

驗證 create_subagent 工具在真實 API 下的行為：
- 父 Agent 能建立子 Agent 並取得結果
- 子 Agent 能使用工具完成任務
- 結果正確回傳給父 Agent

執行方式：
    uv run pytest tests/manual/test_smoke_subagent.py --run-smoke -v
"""

from __future__ import annotations

from pathlib import Path

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.sandbox import LocalSandbox
from agent_core.tools.setup import create_default_registry
from agent_core.types import AgentEvent

pytestmark = pytest.mark.smoke


# =============================================================================
# 輔助函數
# =============================================================================


async def _collect_response_with_events(
    agent: Agent,
    message: str,
) -> tuple[str, list[AgentEvent]]:
    """收集串流回應的完整文字與事件。"""
    chunks: list[str] = []
    events: list[AgentEvent] = []
    async for chunk in agent.stream_message(message):
        if isinstance(chunk, str):
            chunks.append(chunk)
        else:
            events.append(chunk)
    return ''.join(chunks), events


def _create_agent_with_subagent(
    sandbox_path: Path,
    system_prompt: str = '你是助手。請用繁體中文簡短回答。',
) -> Agent:
    """建立啟用 create_subagent 工具的 Agent。"""
    config = AgentCoreConfig(
        provider=ProviderConfig(model='claude-sonnet-4-20250514'),
        system_prompt=system_prompt,
    )
    provider = AnthropicProvider(config.provider)
    registry = create_default_registry(
        LocalSandbox(root=sandbox_path),
        subagent_provider=provider,
        subagent_config=config,
    )
    return Agent(
        config=config,
        provider=provider,
        tool_registry=registry,
    )


# =============================================================================
# Smoke Tests
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('Subagent 子代理機制 (Smoke)')
class TestSubagentSmoke:
    """Smoke test — 驗證 Subagent 在真實 API 下的行為。"""

    @allure.title('父 Agent 使用 create_subagent 完成任務')
    async def test_subagent_completes_task(self, tmp_path: Path) -> None:
        """父 Agent 使用 create_subagent 完成任務。

        Scenario: 父 Agent 使用 create_subagent 完成任務
          Given Agent 已啟動且工具包含 create_subagent
          And sandbox 中有數個 Python 檔案
          When 使用者明確要求用子代理來列出檔案
          Then 子代理應被呼叫（有 create_subagent tool_call 事件）
          And 父 Agent 的回覆應包含檔案相關資訊
        """
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()

        # 建立測試檔案
        (sandbox / 'app.py').write_text('def main():\n    print("hello")\n', encoding='utf-8')
        (sandbox / 'utils.py').write_text('def helper():\n    return 42\n', encoding='utf-8')
        (sandbox / 'README.md').write_text('# Project\n', encoding='utf-8')

        agent = _create_agent_with_subagent(sandbox)

        response, events = await _collect_response_with_events(
            agent,
            '請使用 create_subagent 工具，'
            '指派一個子代理列出 sandbox 中所有的檔案，並告訴我有哪些。',
        )

        # 驗證 create_subagent 工具被呼叫
        tool_events = [
            e
            for e in events
            if e['type'] == 'tool_call' and e['data'].get('name') == 'create_subagent'
        ]
        started = [e for e in tool_events if e['data']['status'] == 'started']
        completed = [e for e in tool_events if e['data']['status'] == 'completed']

        assert len(started) >= 1, f'create_subagent 應至少被呼叫一次，實際事件：{tool_events}'
        assert len(completed) >= 1, f'create_subagent 應至少完成一次，實際事件：{tool_events}'

        # 驗證回覆中包含檔案資訊
        assert len(response) > 0, 'Agent 應有回覆'
        # 至少提到一個我們建立的檔案
        mentioned_files = ['app.py', 'utils.py', 'README.md']
        mentioned_any = any(f in response for f in mentioned_files)
        assert mentioned_any, f'回覆應提到至少一個檔案，實際回覆：{response[:300]}'

    @allure.title('子 Agent 使用工具後回傳結果')
    async def test_subagent_uses_tools_and_returns(self, tmp_path: Path) -> None:
        """子 Agent 使用工具後回傳結果。

        Scenario: 子 Agent 使用工具後回傳結果
          Given Agent 已啟動且工具包含 create_subagent
          And sandbox 中有一個 Python 檔案含有特定函數
          When 使用者要求子代理讀取該檔案並告知函數名稱
          Then 子代理應能讀取檔案並回傳函數名稱
          And 父 Agent 的回覆應包含該函數名稱
        """
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()

        (sandbox / 'calculator.py').write_text(
            'def calculate_fibonacci(n: int) -> int:\n'
            '    """計算第 n 個費氏數列。"""\n'
            '    if n <= 1:\n'
            '        return n\n'
            '    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)\n',
            encoding='utf-8',
        )

        agent = _create_agent_with_subagent(sandbox)

        response, events = await _collect_response_with_events(
            agent,
            '請使用 create_subagent 建立一個子代理，'
            '讓它讀取 calculator.py 檔案，然後告訴我裡面定義了什麼函數。',
        )

        # 驗證 create_subagent 被呼叫
        subagent_events = [
            e
            for e in events
            if e['type'] == 'tool_call' and e['data'].get('name') == 'create_subagent'
        ]
        assert len(subagent_events) >= 1, 'create_subagent 應被呼叫'

        # 驗證回覆提到函數名稱
        assert 'calculate_fibonacci' in response or 'fibonacci' in response.lower(), (
            f'回覆應包含函數名稱 "calculate_fibonacci"，實際回覆：{response[:300]}'
        )
