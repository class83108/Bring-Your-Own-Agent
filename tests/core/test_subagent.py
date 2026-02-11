"""Subagent 子代理機制測試模組。

根據 docs/features/subagent.feature 規格撰寫測試案例。
測試 create_subagent 工具的核心行為：
- 建立子 Agent 執行獨立任務
- 子 Agent 使用與父 Agent 相同的 Sandbox（工具閉包共享）
- 子 Agent 排除 create_subagent 工具（防遞迴）
- 子 Agent 有獨立 context，完成後回傳摘要
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
from agent_core.tools.subagent import create_subagent_handler
from agent_core.types import ContentBlock, MessageParam

# =============================================================================
# Mock Helpers
# =============================================================================


def _make_final_message(
    text: str = '回應內容',
    stop_reason: str = 'end_turn',
    content: list[ContentBlock] | None = None,
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
        self.call_args_list: list[dict[str, Any]] = []

    @asynccontextmanager
    async def stream(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        """模擬串流回應。"""
        self.call_args_list.append(
            {
                'messages': messages,
                'system': system,
                'tools': tools,
                'max_tokens': max_tokens,
            }
        )
        text_chunks, final_msg = self._responses[self._call_count]
        self._call_count += 1

        async def _text_stream() -> AsyncIterator[str]:
            for chunk in text_chunks:
                yield chunk

        async def _get_final() -> FinalMessage:
            return final_msg

        yield StreamResult(
            text_stream=_text_stream(),
            get_final_result=_get_final,
        )

    async def count_tokens(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        """模擬 token 計數。"""
        return 0

    async def create(
        self,
        messages: list[MessageParam],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        """模擬非串流回應。"""
        return FinalMessage(
            content=[{'type': 'text', 'text': ''}],
            stop_reason='end_turn',
            usage=UsageInfo(),
        )


# =============================================================================
# Fixtures
# =============================================================================


def _make_agent(
    provider: Any,
    tool_registry: ToolRegistry | None = None,
    system_prompt: str = '你是一位專業的程式開發助手。',
) -> Agent:
    """建立測試用 Agent。"""
    config = AgentCoreConfig(
        provider=ProviderConfig(api_key='sk-test'),
        system_prompt=system_prompt,
    )
    return Agent(
        config=config,
        provider=provider,
        tool_registry=tool_registry,
    )


async def collect_stream(agent: Agent, message: str) -> str:
    """收集串流回應並返回完整文字（忽略事件）。"""
    chunks: list[str] = []
    async for chunk in agent.stream_message(message):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return ''.join(chunks)


def _stub_read_file(path: str) -> dict[str, str]:
    return {'content': f'內容: {path}'}


def _stub_edit_file(path: str) -> dict[str, str]:
    return {'status': 'ok'}


def _stub_bash(command: str) -> dict[str, str]:
    return {'stdout': command}


def _stub_subagent(task: str) -> None:
    return None


def _make_registry_with_tools() -> ToolRegistry:
    """建立含有多個工具的測試用 ToolRegistry。"""
    registry = ToolRegistry()
    registry.register(
        name='read_file',
        description='讀取檔案',
        parameters={
            'type': 'object',
            'properties': {'path': {'type': 'string'}},
            'required': ['path'],
        },
        handler=_stub_read_file,
    )
    registry.register(
        name='edit_file',
        description='編輯檔案',
        parameters={
            'type': 'object',
            'properties': {'path': {'type': 'string'}},
            'required': ['path'],
        },
        handler=_stub_edit_file,
    )
    registry.register(
        name='bash',
        description='執行指令',
        parameters={
            'type': 'object',
            'properties': {'command': {'type': 'string'}},
            'required': ['command'],
        },
        handler=_stub_bash,
    )
    return registry


# =============================================================================
# Rule: ToolRegistry 應支援 clone（前置條件）
# =============================================================================


@allure.feature('Subagent 子代理機制')
@allure.story('ToolRegistry 應支援 clone')
class TestToolRegistryClone:
    """測試 ToolRegistry.clone() 方法。"""

    @allure.title('clone 不排除任何工具時應複製全部')
    def test_clone_copies_all_tools(self) -> None:
        """clone 不排除時應複製全部工具。"""
        registry = _make_registry_with_tools()

        cloned = registry.clone()

        assert set(cloned.list_tools()) == {'read_file', 'edit_file', 'bash'}

    @allure.title('clone 排除指定工具')
    def test_clone_excludes_specified_tools(self) -> None:
        """clone 應排除指定的工具。"""
        registry = _make_registry_with_tools()
        registry.register(
            name='create_subagent',
            description='建立子代理',
            parameters={'type': 'object', 'properties': {}},
            handler=_stub_subagent,
        )

        cloned = registry.clone(exclude=['create_subagent'])

        assert 'create_subagent' not in cloned.list_tools()
        assert set(cloned.list_tools()) == {'read_file', 'edit_file', 'bash'}

    @allure.title('clone 的工具 handler 應與原始相同')
    async def test_clone_preserves_handlers(self) -> None:
        """clone 後的工具 handler 應與原始相同（共享閉包，即共享 sandbox）。"""
        registry = ToolRegistry()
        call_log: list[str] = []

        def _handler(path: str) -> str:
            call_log.append(path)
            return f'result: {path}'

        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=_handler,
        )

        cloned = registry.clone()
        await cloned.execute('read_file', {'path': 'test.py'})

        # handler 是同一個閉包，代表共享同一個 sandbox
        assert call_log == ['test.py']

    @allure.title('clone 不影響原始 registry')
    def test_clone_does_not_affect_original(self) -> None:
        """對 clone 的修改不應影響原始 registry。"""
        registry = _make_registry_with_tools()

        cloned = registry.clone()
        cloned.register(
            name='new_tool',
            description='新工具',
            parameters={'type': 'object', 'properties': {}},
            handler=lambda: 'ok',
        )

        assert 'new_tool' not in registry.list_tools()
        assert 'new_tool' in cloned.list_tools()


# =============================================================================
# Rule: Agent 應能建立子 Agent
# =============================================================================


@allure.feature('Subagent 子代理機制')
@allure.story('Agent 應能建立子 Agent')
class TestCreateSubagent:
    """測試 create_subagent_handler 核心功能。"""

    @allure.title('建立子 Agent 執行獨立任務')
    async def test_subagent_executes_task_and_returns_result(self) -> None:
        """Scenario: 建立子 Agent 執行獨立任務。

        子 Agent 收到任務後執行，並回傳結果給父 Agent。
        """
        child_provider = MockProvider(
            [
                (['找到 3 個 Python 檔案'], _make_final_message('找到 3 個 Python 檔案')),
            ]
        )
        registry = _make_registry_with_tools()
        config = AgentCoreConfig(provider=ProviderConfig(api_key='sk-test'))

        result = await create_subagent_handler(
            task='列出所有 Python 檔案',
            provider=child_provider,
            config=config,
            tool_registry=registry,
        )

        assert 'result' in result
        assert '找到 3 個 Python 檔案' in result['result']

    @allure.title('子 Agent 使用與父 Agent 相同的 Sandbox')
    async def test_subagent_shares_sandbox_via_tool_handlers(self) -> None:
        """Scenario: 子 Agent 使用與父 Agent 相同的 Sandbox。

        因為 clone 共享 handler 閉包，子 Agent 的工具操作同一個 sandbox。
        """
        sandbox_operations: list[str] = []

        def _read_handler(path: str) -> dict[str, str]:
            sandbox_operations.append(f'read:{path}')
            return {'content': '檔案內容'}

        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=_read_handler,
        )

        # 子 Agent 呼叫 read_file 工具
        tool_content: list[ContentBlock] = [
            {
                'type': 'tool_use',
                'id': 'tool_1',
                'name': 'read_file',
                'input': {'path': 'main.py'},
            },
        ]
        child_provider = MockProvider(
            [
                ([], _make_final_message(content=tool_content, stop_reason='tool_use')),
                (['讀取完成'], _make_final_message('讀取完成')),
            ]
        )
        config = AgentCoreConfig(provider=ProviderConfig(api_key='sk-test'))

        await create_subagent_handler(
            task='讀取 main.py',
            provider=child_provider,
            config=config,
            tool_registry=registry,
        )

        # 驗證子 Agent 使用了與父 Agent 相同的 handler（同一個 sandbox）
        assert sandbox_operations == ['read:main.py']


# =============================================================================
# Rule: 子 Agent 的工具應受到限制
# =============================================================================


@allure.feature('Subagent 子代理機制')
@allure.story('子 Agent 的工具應受到限制')
class TestSubagentToolRestrictions:
    """測試子 Agent 的工具限制。"""

    @allure.title('子 Agent 預設不能建立子 Agent')
    async def test_subagent_excludes_create_subagent_tool(self) -> None:
        """Scenario: 子 Agent 預設不能建立子 Agent。

        子 Agent 的工具清單不應包含 create_subagent。
        """
        child_provider = MockProvider(
            [
                (['完成'], _make_final_message('完成')),
            ]
        )
        registry = _make_registry_with_tools()
        # 父 Agent 有 create_subagent 工具
        registry.register(
            name='create_subagent',
            description='建立子代理',
            parameters={
                'type': 'object',
                'properties': {'task': {'type': 'string'}},
                'required': ['task'],
            },
            handler=_stub_subagent,
        )
        config = AgentCoreConfig(provider=ProviderConfig(api_key='sk-test'))

        await create_subagent_handler(
            task='測試',
            provider=child_provider,
            config=config,
            tool_registry=registry,
        )

        # 驗證子 Agent 的 Provider 收到的工具清單不包含 create_subagent
        tools_sent = child_provider.call_args_list[0]['tools']
        tool_names = [t['name'] for t in tools_sent]
        assert 'create_subagent' not in tool_names

    @allure.title('子 Agent 擁有其餘所有工具')
    async def test_subagent_has_remaining_tools(self) -> None:
        """Scenario: 子 Agent 擁有其餘所有工具。

        父 Agent 的工具清單為 read_file, edit_file, bash, create_subagent。
        子 Agent 的工具清單應為 read_file, edit_file, bash。
        """
        child_provider = MockProvider(
            [
                (['完成'], _make_final_message('完成')),
            ]
        )
        registry = _make_registry_with_tools()  # read_file, edit_file, bash
        registry.register(
            name='create_subagent',
            description='建立子代理',
            parameters={'type': 'object', 'properties': {}},
            handler=_stub_subagent,
        )
        config = AgentCoreConfig(provider=ProviderConfig(api_key='sk-test'))

        await create_subagent_handler(
            task='測試',
            provider=child_provider,
            config=config,
            tool_registry=registry,
        )

        # 驗證子 Agent 的 Provider 收到 read_file, edit_file, bash
        tools_sent = child_provider.call_args_list[0]['tools']
        tool_names = {t['name'] for t in tools_sent}
        assert tool_names == {'read_file', 'edit_file', 'bash'}


# =============================================================================
# Rule: 子 Agent 應有獨立的 context
# =============================================================================


@allure.feature('Subagent 子代理機制')
@allure.story('子 Agent 應有獨立的 context')
class TestSubagentContextIsolation:
    """測試子 Agent 的 context 隔離。"""

    @allure.title('子 Agent 的對話歷史與父 Agent 獨立')
    async def test_subagent_has_independent_conversation(self) -> None:
        """Scenario: 子 Agent 的對話歷史與父 Agent 獨立。

        建立子 Agent 後，父 Agent 的對話歷史不應受影響。
        """
        # 子 Agent 的 Provider
        child_provider = MockProvider(
            [
                (['子代理結果'], _make_final_message('子代理結果')),
            ]
        )

        # 父 Agent 的 Provider：先 tool_use (create_subagent)，再回覆
        parent_tool_content: list[ContentBlock] = [
            {
                'type': 'tool_use',
                'id': 'sub_1',
                'name': 'create_subagent',
                'input': {'task': '列出 Python 檔案'},
            },
        ]
        parent_provider = MockProvider(
            [
                ([], _make_final_message(content=parent_tool_content, stop_reason='tool_use')),
                (['收到子代理回報'], _make_final_message('收到子代理回報')),
            ]
        )

        # 建立含 create_subagent 的 registry
        registry = _make_registry_with_tools()

        async def _subagent_handler(task: str) -> dict[str, Any]:
            return await create_subagent_handler(
                task=task,
                provider=child_provider,
                config=AgentCoreConfig(provider=ProviderConfig(api_key='sk-test')),
                tool_registry=registry,
            )

        registry.register(
            name='create_subagent',
            description='建立子代理',
            parameters={
                'type': 'object',
                'properties': {'task': {'type': 'string'}},
                'required': ['task'],
            },
            handler=_subagent_handler,
        )

        parent_agent = _make_agent(parent_provider, tool_registry=registry)

        # 父 Agent 先有一輪對話歷史
        parent_agent.conversation = [
            {'role': 'user', 'content': '先前的問題'},
            {'role': 'assistant', 'content': '先前的回答'},
        ]

        await collect_stream(parent_agent, '請用子代理列出 Python 檔案')

        # 父 Agent 的先前歷史不應被汙染
        assert parent_agent.conversation[0]['content'] == '先前的問題'
        assert parent_agent.conversation[1]['content'] == '先前的回答'

        # 子 Agent 的內部對話（如 system prompt）不應出現在父 Agent 歷史中
        parent_roles = [m['role'] for m in parent_agent.conversation]
        # 父 Agent 歷史結構：先前2筆 + user + assistant(tool_use) + user(tool_result) + assistant
        assert parent_roles == ['user', 'assistant', 'user', 'assistant', 'user', 'assistant']

    @allure.title('子 Agent 完成後回傳摘要')
    async def test_subagent_returns_summary_not_full_history(self) -> None:
        """Scenario: 子 Agent 完成後回傳摘要。

        父 Agent 應收到子 Agent 的執行結果摘要，
        而非子 Agent 的完整對話歷史。
        """
        # 子 Agent 有多輪工具調用
        child_tool_content: list[ContentBlock] = [
            {
                'type': 'tool_use',
                'id': 'child_tool_1',
                'name': 'read_file',
                'input': {'path': 'a.py'},
            },
        ]
        child_provider = MockProvider(
            [
                # 第一輪：子 Agent 呼叫工具
                ([], _make_final_message(content=child_tool_content, stop_reason='tool_use')),
                # 第二輪：子 Agent 回覆最終結果
                (['找到 a.py，內容為 hello'], _make_final_message('找到 a.py，內容為 hello')),
            ]
        )

        registry = _make_registry_with_tools()
        config = AgentCoreConfig(provider=ProviderConfig(api_key='sk-test'))

        result = await create_subagent_handler(
            task='讀取 a.py',
            provider=child_provider,
            config=config,
            tool_registry=registry,
        )

        # 回傳的是摘要文字，不是完整的對話歷史
        assert isinstance(result, dict)
        assert 'result' in result
        assert isinstance(result['result'], str)
        assert '找到 a.py' in result['result']
        # 不應包含子 Agent 的工具調用細節
        assert 'tool_use_id' not in str(result)
        assert 'conversation' not in result
