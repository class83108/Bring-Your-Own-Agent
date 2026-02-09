"""工具使用狀態顯示測試模組。

根據 docs/features/tool_status.feature 規格撰寫測試案例。
涵蓋：
- Rule: 工具狀態應顯示名稱與參數摘要
- Rule: Agent 呼叫工具時應通知使用者（事件產生）
- Rule: 工具呼叫前的文字應與最終回覆區隔（preamble_end 事件）
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import allure

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.tools.registry import ToolRegistry

# =============================================================================
# Mock Helpers
# =============================================================================


def _make_final_message(
    text: str = '',
    stop_reason: str = 'end_turn',
    content: list[dict[str, Any]] | None = None,
) -> FinalMessage:
    """建立 FinalMessage。"""
    if content is None:
        content = [{'type': 'text', 'text': text}]
    return FinalMessage(
        content=content,
        stop_reason=stop_reason,
        usage=UsageInfo(input_tokens=10, output_tokens=20),
    )


class MockProvider:
    """模擬的 LLM Provider。"""

    def __init__(self, responses: list[tuple[list[str], FinalMessage]]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        text_chunks, final_msg = self._responses[self._call_count]
        self._call_count += 1

        async def _text_stream() -> AsyncIterator[str]:
            for chunk in text_chunks:
                yield chunk

        async def _get_final() -> FinalMessage:
            return final_msg

        yield StreamResult(text_stream=_text_stream(), get_final_result=_get_final)


def _make_agent(
    provider: Any,
    tool_registry: ToolRegistry | None = None,
) -> Agent:
    """建立測試用 Agent。"""
    config = AgentCoreConfig(
        provider=ProviderConfig(api_key='sk-test'),
        system_prompt='測試',
    )
    return Agent(
        config=config,
        provider=provider,
        tool_registry=tool_registry,
    )


async def collect_stream_with_events(
    agent: Agent, message: str
) -> tuple[str, list[dict[str, Any]]]:
    """收集串流回應，分離文字 token 與事件。"""
    text_parts: list[str] = []
    events: list[dict[str, Any]] = []

    async for item in agent.stream_message(message):
        if isinstance(item, str):
            text_parts.append(item)
        else:
            events.append(item)

    return ''.join(text_parts), events


# =============================================================================
# Rule: 工具狀態應顯示名稱與參數摘要
# =============================================================================


@allure.feature('工具使用狀態顯示')
@allure.story('工具狀態應顯示名稱與參數摘要')
class TestToolSummary:
    """測試工具摘要產生。"""

    @allure.title('讀取檔案時顯示檔案路徑')
    def test_read_file_summary(self) -> None:
        """Scenario: 讀取檔案時顯示檔案路徑。"""
        from agent_core.tool_summary import get_tool_summary

        summary = get_tool_summary('read_file', {'path': 'src/main.py'})
        assert summary == '讀取檔案 src/main.py'

    @allure.title('搜尋程式碼時顯示搜尋模式')
    def test_grep_search_summary(self) -> None:
        """Scenario: 搜尋程式碼時顯示搜尋模式。"""
        from agent_core.tool_summary import get_tool_summary

        summary = get_tool_summary('grep_search', {'pattern': 'logger'})
        assert summary == '搜尋程式碼 logger'

    @allure.title('編輯檔案時顯示檔案路徑')
    def test_edit_file_summary(self) -> None:
        """Scenario: 編輯檔案時顯示檔案路徑。"""
        from agent_core.tool_summary import get_tool_summary

        summary = get_tool_summary('edit_file', {'path': 'src/main.py'})
        assert summary == '編輯檔案 src/main.py'

    @allure.title('執行命令時顯示命令摘要')
    def test_bash_summary(self) -> None:
        """Scenario: 執行命令時顯示命令摘要。"""
        from agent_core.tool_summary import get_tool_summary

        summary = get_tool_summary('bash', {'command': 'uv run pytest'})
        assert summary == '執行命令 uv run pytest'

    @allure.title('列出檔案時顯示路徑')
    def test_list_files_summary(self) -> None:
        """Scenario: 列出檔案時顯示路徑。"""
        from agent_core.tool_summary import get_tool_summary

        summary = get_tool_summary('list_files', {'path': 'src/'})
        assert summary == '列出檔案 src/'

    @allure.title('未知工具應顯示工具名稱')
    def test_unknown_tool_summary(self) -> None:
        """未知工具應顯示工具名稱。"""
        from agent_core.tool_summary import get_tool_summary

        summary = get_tool_summary('custom_tool', {'key': 'value'})
        assert 'custom_tool' in summary

    @allure.title('過長的命令參數應截斷顯示')
    def test_long_command_truncated(self) -> None:
        """過長的命令參數應截斷顯示。"""
        from agent_core.tool_summary import get_tool_summary

        long_cmd = 'a' * 200
        summary = get_tool_summary('bash', {'command': long_cmd})
        assert len(summary) < 150


# =============================================================================
# Rule: Agent 呼叫工具時應通知使用者
# =============================================================================


@allure.feature('工具使用狀態顯示')
@allure.story('Agent 呼叫工具時應通知使用者')
class TestToolCallEvents:
    """測試工具呼叫事件產生。"""

    @allure.title('工具成功執行時產生 started 和 completed 事件')
    async def test_tool_call_emits_started_and_completed(self) -> None:
        """Scenario: 工具成功執行時產生 started 和 completed 事件。"""
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': 'file content', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )

        tool_content = [
            {
                'type': 'tool_use',
                'id': 't1',
                'name': 'read_file',
                'input': {'path': 'main.py'},
            }
        ]
        first_msg = _make_final_message(content=tool_content, stop_reason='tool_use')
        second_msg = _make_final_message('結果')

        provider = MockProvider(
            [
                ([], first_msg),
                (['結果'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        text, events = await collect_stream_with_events(agent, '讀取 main.py')

        assert text == '結果'

        tool_events = [e for e in events if e['type'] == 'tool_call']
        assert len(tool_events) == 2

        started = tool_events[0]
        assert started['data']['status'] == 'started'
        assert started['data']['name'] == 'read_file'

        completed = tool_events[1]
        assert completed['data']['status'] == 'completed'
        assert completed['data']['name'] == 'read_file'

    @allure.title('工具執行失敗時產生 failed 事件')
    async def test_tool_call_failure_emits_failed_event(self) -> None:
        """Scenario: 工具執行失敗時產生 failed 事件。"""

        def failing_handler(path: str) -> str:
            raise FileNotFoundError('檔案不存在')

        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=failing_handler,
        )

        tool_content = [
            {
                'type': 'tool_use',
                'id': 't1',
                'name': 'read_file',
                'input': {'path': 'x.py'},
            }
        ]
        first_msg = _make_final_message(content=tool_content, stop_reason='tool_use')
        second_msg = _make_final_message('找不到檔案')

        provider = MockProvider(
            [
                ([], first_msg),
                (['找不到檔案'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        _, events = await collect_stream_with_events(agent, '讀取 x.py')

        tool_events = [e for e in events if e['type'] == 'tool_call']
        assert len(tool_events) == 2

        started = tool_events[0]
        assert started['data']['status'] == 'started'

        failed = tool_events[1]
        assert failed['data']['status'] == 'failed'
        assert 'error' in failed['data']


# =============================================================================
# Rule: 工具呼叫前的文字應與最終回覆區隔
# =============================================================================


@allure.feature('工具使用狀態顯示')
@allure.story('工具呼叫前的文字應與最終回覆區隔')
class TestPreambleEvents:
    """測試 preamble_end 事件。"""

    @allure.title('Preamble 文字應觸發 preamble_end 事件')
    async def test_preamble_end_emitted_when_text_before_tool(self) -> None:
        """Scenario: Preamble 文字應觸發 preamble_end 事件。"""
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': 'ok', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )

        first_content = [
            {'type': 'text', 'text': '讓我查看程式碼...'},
            {
                'type': 'tool_use',
                'id': 't1',
                'name': 'read_file',
                'input': {'path': 'main.py'},
            },
        ]
        first_msg = _make_final_message(content=first_content, stop_reason='tool_use')
        second_msg = _make_final_message('程式碼內容如下')

        provider = MockProvider(
            [
                (['讓我查看', '程式碼...'], first_msg),
                (['程式碼內容如下'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        _, events = await collect_stream_with_events(agent, '看程式碼')

        preamble_events = [e for e in events if e['type'] == 'preamble_end']
        assert len(preamble_events) == 1

        event_types = [e['type'] for e in events]
        preamble_idx = event_types.index('preamble_end')
        tool_idx = event_types.index('tool_call')
        assert preamble_idx < tool_idx

    @allure.title('無 preamble 文字時不應產生 preamble_end 事件')
    async def test_no_preamble_when_no_text_before_tool(self) -> None:
        """Scenario: 無 preamble 文字時不應產生 preamble_end 事件。"""
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': 'ok', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )

        tool_content = [
            {
                'type': 'tool_use',
                'id': 't1',
                'name': 'read_file',
                'input': {'path': 'main.py'},
            }
        ]
        first_msg = _make_final_message(content=tool_content, stop_reason='tool_use')
        second_msg = _make_final_message('結果')

        provider = MockProvider(
            [
                ([], first_msg),
                (['結果'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        _, events = await collect_stream_with_events(agent, '讀取')

        preamble_events = [e for e in events if e['type'] == 'preamble_end']
        assert len(preamble_events) == 0

    @allure.title('多次工具呼叫產生多個 preamble')
    async def test_multiple_tool_calls_produce_multiple_preambles(self) -> None:
        """Scenario: 多次工具呼叫產生多個 preamble。"""
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': 'ok', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )

        first_content = [
            {'type': 'text', 'text': '先讀取第一個檔案...'},
            {
                'type': 'tool_use',
                'id': 't1',
                'name': 'read_file',
                'input': {'path': 'a.py'},
            },
        ]
        first_msg = _make_final_message(content=first_content, stop_reason='tool_use')

        second_content = [
            {'type': 'text', 'text': '接下來讀取第二個...'},
            {
                'type': 'tool_use',
                'id': 't2',
                'name': 'read_file',
                'input': {'path': 'b.py'},
            },
        ]
        second_msg = _make_final_message(content=second_content, stop_reason='tool_use')

        third_msg = _make_final_message('完成')

        provider = MockProvider(
            [
                (['先讀取第一個檔案...'], first_msg),
                (['接下來讀取第二個...'], second_msg),
                (['完成'], third_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        _, events = await collect_stream_with_events(agent, '讀取兩個檔案')

        preamble_events = [e for e in events if e['type'] == 'preamble_end']
        assert len(preamble_events) == 2
