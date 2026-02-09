"""Agent 測試模組。

根據 docs/features/chat.feature 與 agent_core.feature 規格撰寫測試案例。
Agent 透過 Provider 抽象層與 LLM 互動，測試中使用 mock Provider。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderTimeoutError,
)
from agent_core.skills.base import Skill
from agent_core.skills.registry import SkillRegistry
from agent_core.tools.registry import ToolRegistry

# =============================================================================
# Mock Helpers
# =============================================================================


def _make_final_message(
    text: str = '回應內容',
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
    """模擬的 LLM Provider。

    每次呼叫 stream() 從 responses 佇列取出下一個回應。
    """

    def __init__(self, responses: list[tuple[list[str], FinalMessage]]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.call_args_list: list[dict[str, Any]] = []

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        return 0

    async def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        return FinalMessage(
            content=[{'type': 'text', 'text': ''}],
            stop_reason='end_turn',
            usage=UsageInfo(),
        )


class ErrorProvider:
    """在 stream 進入時拋出指定錯誤的 Provider。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        raise self._error
        yield  # type: ignore[misc]  # 讓 Python 識別為 generator


class PartialStreamProvider:
    """模擬串流中斷的 Provider — 在 text_stream 中途拋出錯誤。"""

    def __init__(
        self,
        partial_chunks: list[str],
        error: Exception,
    ) -> None:
        self._partial_chunks = partial_chunks
        self._error = error

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        async def _text_stream() -> AsyncIterator[str]:
            for chunk in self._partial_chunks:
                yield chunk
            raise self._error

        async def _get_final() -> FinalMessage:
            raise self._error

        yield StreamResult(
            text_stream=_text_stream(),
            get_final_result=_get_final,
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


def _get_assistant_text(conversation_entry: dict[str, Any]) -> str:
    """從對話歷史中的 assistant 條目取得文字內容。"""
    content: Any = conversation_entry['content']
    if isinstance(content, str):
        return content
    texts: list[str] = []
    content_list: list[dict[str, Any]] = content
    for block in content_list:
        if block.get('type') == 'text':
            texts.append(block['text'])
    return ''.join(texts)


# =============================================================================
# Rule: Agent 應驗證使用者輸入
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應驗證使用者輸入')
class TestInputValidation:
    """測試使用者輸入驗證。"""

    @allure.title('使用者發送空白訊息')
    async def test_empty_message_raises_value_error(self) -> None:
        """Scenario: 使用者發送空白訊息。"""
        provider = MockProvider([([], _make_final_message())])
        agent = _make_agent(provider)

        with pytest.raises(ValueError, match='空白|有效'):
            async for _ in agent.stream_message(''):
                pass

    @allure.title('測試只有空白字元的訊息也應拋出 ValueError')
    async def test_whitespace_only_raises_value_error(self) -> None:
        """測試只有空白字元的訊息也應拋出 ValueError。"""
        provider = MockProvider([([], _make_final_message())])
        agent = _make_agent(provider)

        with pytest.raises(ValueError, match='空白|有效'):
            async for _ in agent.stream_message('   '):
                pass

        with pytest.raises(ValueError, match='空白|有效'):
            async for _ in agent.stream_message('\n\t  '):
                pass


# =============================================================================
# Rule: Agent 應維護對話歷史
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應維護對話歷史')
class TestConversationHistory:
    """測試對話歷史維護。"""

    @allure.title('單輪對話後歷史正確記錄')
    async def test_single_turn_conversation_history(self) -> None:
        """Scenario: 單輪對話後歷史正確記錄。"""
        provider = MockProvider([(['回應內容'], _make_final_message('回應內容'))])
        agent = _make_agent(provider)

        await collect_stream(agent, '測試訊息')

        assert len(agent.conversation) == 2
        assert agent.conversation[0]['role'] == 'user'
        assert agent.conversation[0]['content'] == '測試訊息'
        assert agent.conversation[1]['role'] == 'assistant'
        assert _get_assistant_text(agent.conversation[1]) == '回應內容'

    @allure.title('多輪對話後歷史正確累積')
    async def test_multi_turn_conversation_history(self) -> None:
        """Scenario: 多輪對話後歷史正確累積。"""
        provider = MockProvider(
            [
                (['第一次回應'], _make_final_message('第一次回應')),
                (['第二次回應'], _make_final_message('第二次回應')),
            ]
        )
        agent = _make_agent(provider)

        await collect_stream(agent, '第一則訊息')
        await collect_stream(agent, '第二則訊息')

        assert len(agent.conversation) == 4
        assert agent.conversation[0]['content'] == '第一則訊息'
        assert _get_assistant_text(agent.conversation[1]) == '第一次回應'
        assert agent.conversation[2]['content'] == '第二則訊息'
        assert _get_assistant_text(agent.conversation[3]) == '第二次回應'

    @allure.title('重設對話歷史')
    def test_reset_conversation(self) -> None:
        """Scenario: 重設對話歷史。"""
        provider = MockProvider([])
        agent = _make_agent(provider)
        agent.conversation = [
            {'role': 'user', 'content': 'test'},
            {'role': 'assistant', 'content': 'response'},
        ]

        agent.reset_conversation()

        assert len(agent.conversation) == 0


# =============================================================================
# Rule: Agent 應正確處理錯誤情況（使用 Provider 例外）
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應正確處理錯誤情況')
class TestErrorHandling:
    """測試錯誤處理（Provider 例外）。"""

    @allure.title('Provider 連線失敗')
    async def test_provider_connection_error(self) -> None:
        """Scenario: Provider 連線失敗。"""
        provider = ErrorProvider(ProviderConnectionError('連線失敗'))
        agent = _make_agent(provider)
        initial_length = len(agent.conversation)

        with pytest.raises(ProviderConnectionError):
            async for _ in agent.stream_message('測試訊息'):
                pass

        assert len(agent.conversation) == initial_length

    @allure.title('Provider 認證失敗')
    async def test_provider_auth_error(self) -> None:
        """Scenario: Provider 認證失敗。"""
        provider = ErrorProvider(ProviderAuthError('API 金鑰無效'))
        agent = _make_agent(provider)
        initial_length = len(agent.conversation)

        with pytest.raises(ProviderAuthError):
            async for _ in agent.stream_message('測試訊息'):
                pass

        assert len(agent.conversation) == initial_length

    @allure.title('Provider 回應超時')
    async def test_provider_timeout_error(self) -> None:
        """Scenario: Provider 回應超時。"""
        provider = ErrorProvider(ProviderTimeoutError('請求超時'))
        agent = _make_agent(provider)
        initial_length = len(agent.conversation)

        with pytest.raises(ProviderTimeoutError):
            async for _ in agent.stream_message('測試訊息'):
                pass

        assert len(agent.conversation) == initial_length


# =============================================================================
# Rule: Agent 應支援串流回應
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應支援串流回應')
class TestStreamingResponse:
    """測試串流回應功能。"""

    @allure.title('串流方式逐步回傳 token')
    async def test_stream_collects_chunks_into_conversation(self) -> None:
        """Scenario: 串流方式逐步回傳 token。"""
        chunks = ['這是', '一個', '串流', '回應']
        provider = MockProvider([(chunks, _make_final_message('這是一個串流回應'))])
        agent = _make_agent(provider)

        async for _ in agent.stream_message('測試'):
            pass

        assert _get_assistant_text(agent.conversation[1]) == '這是一個串流回應'

    @allure.title('串流中斷時保留部分回應')
    async def test_stream_interruption_preserves_partial_response(self) -> None:
        """Scenario: 串流中斷時保留部分回應。"""
        provider = PartialStreamProvider(
            partial_chunks=['這是部分', '回應'],
            error=ProviderConnectionError('串流連線中斷，請檢查網路連線並稍後重試。'),
        )
        agent = _make_agent(provider)

        received: list[str] = []
        with pytest.raises(ProviderConnectionError):
            async for chunk in agent.stream_message('測試'):
                if isinstance(chunk, str):
                    received.append(chunk)

        assert received == ['這是部分', '回應']
        # 部分回應應被保留在對話歷史中
        assert len(agent.conversation) == 2
        assert agent.conversation[1]['content'] == '這是部分回應'


# =============================================================================
# Rule: Agent Loop 應持續運作直到任務完成（工具調用迴圈）
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent Loop 應持續運作直到任務完成')
class TestToolUseLoop:
    """測試工具調用迴圈功能。"""

    @allure.title('無 tool_registry 時行為不變')
    async def test_no_tool_registry_unchanged(self) -> None:
        """Scenario: 無 tool_registry 時行為不變。"""
        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        agent = _make_agent(provider)

        await collect_stream(agent, '測試')

        # 不應傳遞 tools 參數
        call_kwargs = provider.call_args_list[0]
        assert call_kwargs['tools'] is None

    @allure.title('單輪對話有工具調用')
    async def test_single_turn_with_tool_call(self) -> None:
        """Scenario: 單輪對話有工具調用。"""
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': f'檔案內容: {path}', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )

        # 第一次：tool_use
        tool_content = [
            {
                'type': 'tool_use',
                'id': 'tool_1',
                'name': 'read_file',
                'input': {'path': 'main.py'},
            }
        ]
        first_msg = _make_final_message(content=tool_content, stop_reason='tool_use')

        # 第二次：最終文字
        second_msg = _make_final_message('檔案內容如下...')

        provider = MockProvider(
            [
                ([], first_msg),
                (['檔案內容如下...'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        result = await collect_stream(agent, '請讀取 main.py')

        assert result == '檔案內容如下...'
        assert len(agent.conversation) == 4
        assert agent.conversation[0]['role'] == 'user'
        assert agent.conversation[1]['role'] == 'assistant'
        assert agent.conversation[2]['role'] == 'user'
        assert agent.conversation[3]['role'] == 'assistant'

        # 驗證 tool_result
        tool_results: Any = agent.conversation[2]['content']
        assert len(tool_results) == 1
        assert tool_results[0]['tool_use_id'] == 'tool_1'
        assert '檔案內容: main.py' in tool_results[0]['content']

    @allure.title('同時執行多個獨立工具')
    async def test_parallel_tool_calls(self) -> None:
        """Scenario: 同時執行多個獨立工具。"""
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': f'內容: {path}', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )

        # 第一次：兩個 tool_use
        first_content = [
            {'type': 'text', 'text': '讓我讀取這兩個檔案'},
            {'type': 'tool_use', 'id': 'tool_a', 'name': 'read_file', 'input': {'path': 'a.py'}},
            {'type': 'tool_use', 'id': 'tool_b', 'name': 'read_file', 'input': {'path': 'b.py'}},
        ]
        first_msg = _make_final_message(content=first_content, stop_reason='tool_use')

        # 第二次：最終文字
        second_msg = _make_final_message('兩個檔案內容如上。')

        provider = MockProvider(
            [
                (['讓我讀取這兩個檔案'], first_msg),
                (['兩個檔案內容如上。'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        chunks: list[str] = []
        events: list[dict[str, Any]] = []
        async for item in agent.stream_message('請讀取 a.py 和 b.py'):
            if isinstance(item, str):
                chunks.append(item)
            else:
                events.append(item)

        result = ''.join(chunks)
        assert '兩個檔案內容如上' in result

        # 應有 started + completed 事件
        tool_events = [e for e in events if e.get('type') == 'tool_call']
        started = [e for e in tool_events if e['data']['status'] == 'started']
        completed = [e for e in tool_events if e['data']['status'] == 'completed']
        assert len(started) == 2
        assert len(completed) == 2

        # 所有 started 事件應在所有 completed 之前
        started_indices = [events.index(e) for e in started]
        completed_indices = [events.index(e) for e in completed]
        assert max(started_indices) < min(completed_indices)

        # 對話歷史應包含兩個 tool_result
        tool_results: Any = agent.conversation[2]['content']
        assert len(tool_results) == 2
        result_ids = {r['tool_use_id'] for r in tool_results}
        assert result_ids == {'tool_a', 'tool_b'}

    @allure.title('工具執行失敗時回傳 is_error')
    async def test_tool_execution_error(self) -> None:
        """Scenario: 工具執行失敗時回傳 is_error。"""

        def failing_handler(path: str) -> str:
            raise FileNotFoundError('檔案不存在: missing.py')

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
                'id': 'tool_err',
                'name': 'read_file',
                'input': {'path': 'missing.py'},
            }
        ]
        first_msg = _make_final_message(content=tool_content, stop_reason='tool_use')
        second_msg = _make_final_message('找不到該檔案。')

        provider = MockProvider(
            [
                ([], first_msg),
                (['找不到該檔案。'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        result = await collect_stream(agent, '讀取 missing.py')

        # tool_result 應包含 is_error
        tool_results: Any = agent.conversation[2]['content']
        assert tool_results[0]['is_error'] is True
        assert '檔案不存在' in tool_results[0]['content']

        assert result == '找不到該檔案。'


# =============================================================================
# Rule: Agent 應透過 Provider 抽象層呼叫 LLM
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應透過 Provider 抽象層呼叫 LLM')
class TestProviderIntegration:
    """測試 Agent 與 Provider 的整合。"""

    @allure.title('Agent 應將 system prompt 傳給 Provider')
    async def test_agent_passes_system_prompt_to_provider(self) -> None:
        """Agent 應將 system prompt 傳給 Provider。"""
        provider = MockProvider([(['Hi'], _make_final_message('Hi'))])
        agent = _make_agent(provider, system_prompt='你是健身教練')

        await collect_stream(agent, '你好')

        assert provider.call_args_list[0]['system'] == '你是健身教練'

    @allure.title('Agent 應將工具定義傳給 Provider')
    async def test_agent_passes_tools_to_provider(self) -> None:
        """Agent 應將工具定義傳給 Provider。"""
        registry = ToolRegistry()
        registry.register(
            name='test_tool',
            description='測試工具',
            parameters={'type': 'object', 'properties': {}},
            handler=lambda: 'ok',
        )

        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        agent = _make_agent(provider, tool_registry=registry)

        await collect_stream(agent, '測試')

        tools = provider.call_args_list[0]['tools']
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]['name'] == 'test_tool'


# =============================================================================
# Rule: Agent 應支援 SkillRegistry 動態組合 system prompt
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應支援 SkillRegistry 動態組合 system prompt')
class TestSkillRegistryIntegration:
    """測試 Agent 與 SkillRegistry 的整合。"""

    @allure.title('無 SkillRegistry 時使用原始 system prompt')
    async def test_agent_without_skill_registry_uses_base_prompt(self) -> None:
        """無 SkillRegistry 時使用原始 system prompt。"""
        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        agent = _make_agent(provider, system_prompt='基礎 prompt')

        await collect_stream(agent, '測試')

        assert provider.call_args_list[0]['system'] == '基礎 prompt'

    @allure.title('SkillRegistry 為空時使用原始 system prompt')
    async def test_agent_with_empty_skill_registry_uses_base_prompt(self) -> None:
        """SkillRegistry 為空時使用原始 system prompt。"""
        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        config = AgentCoreConfig(
            provider=ProviderConfig(api_key='sk-test'),
            system_prompt='基礎 prompt',
        )
        skill_registry = SkillRegistry()
        agent = Agent(
            config=config,
            provider=provider,
            skill_registry=skill_registry,
        )

        await collect_stream(agent, '測試')

        assert provider.call_args_list[0]['system'] == '基礎 prompt'

    @allure.title('已註冊的 Skill 描述應出現在 system prompt（Phase 1）')
    async def test_agent_with_registered_skill_includes_description(self) -> None:
        """已註冊的 Skill 描述應出現在 system prompt（Phase 1）。"""
        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        config = AgentCoreConfig(
            provider=ProviderConfig(api_key='sk-test'),
            system_prompt='基礎 prompt',
        )
        skill_registry = SkillRegistry()
        skill_registry.register(
            Skill(
                name='code_review',
                description='程式碼審查',
                instructions='詳細審查指令...',
            )
        )
        agent = Agent(
            config=config,
            provider=provider,
            skill_registry=skill_registry,
        )

        await collect_stream(agent, '測試')

        system = provider.call_args_list[0]['system']
        assert '基礎 prompt' in system
        assert 'code_review' in system
        assert '程式碼審查' in system
        # Phase 2 未啟用，不應包含完整 instructions
        assert '詳細審查指令' not in system

    @allure.title('啟用的 Skill 應注入完整 instructions（Phase 2）')
    async def test_agent_with_active_skill_includes_instructions(self) -> None:
        """啟用的 Skill 應注入完整 instructions（Phase 2）。"""
        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        config = AgentCoreConfig(
            provider=ProviderConfig(api_key='sk-test'),
            system_prompt='基礎 prompt',
        )
        skill_registry = SkillRegistry()
        skill_registry.register(
            Skill(
                name='code_review',
                description='程式碼審查',
                instructions='詳細審查指令...',
            )
        )
        skill_registry.activate('code_review')
        agent = Agent(
            config=config,
            provider=provider,
            skill_registry=skill_registry,
        )

        await collect_stream(agent, '測試')

        system = provider.call_args_list[0]['system']
        assert '基礎 prompt' in system
        assert '程式碼審查' in system
        assert '詳細審查指令' in system


# =============================================================================
# Rule: Agent 應在每次 API 回應後更新 TokenCounter
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應在每次 API 回應後更新 TokenCounter')
class TestTokenCounterIntegration:
    """測試 Agent 與 TokenCounter 的整合。"""

    @allure.title('API 回應後 token_counter 應被更新')
    async def test_token_counter_updated_after_response(self) -> None:
        """API 回應後 token_counter 應被更新。"""
        usage = UsageInfo(input_tokens=1000, output_tokens=500)
        final_msg = FinalMessage(
            content=[{'type': 'text', 'text': '回應'}],
            stop_reason='end_turn',
            usage=usage,
        )
        provider = MockProvider([(['回應'], final_msg)])
        agent = _make_agent(provider)

        await collect_stream(agent, '測試')

        assert agent.token_counter is not None
        assert agent.token_counter.current_context_tokens == 1500

    @allure.title('含快取 token 的回應也應正確更新 token_counter')
    async def test_token_counter_updated_with_cache_tokens(self) -> None:
        """含快取 token 的回應也應正確更新 token_counter。"""
        usage = UsageInfo(
            input_tokens=500,
            output_tokens=400,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=200,
        )
        final_msg = FinalMessage(
            content=[{'type': 'text', 'text': '回應'}],
            stop_reason='end_turn',
            usage=usage,
        )
        provider = MockProvider([(['回應'], final_msg)])
        agent = _make_agent(provider)

        await collect_stream(agent, '測試')

        # input: 500 + 300 + 200 = 1000, output: 400
        assert agent.token_counter is not None
        assert agent.token_counter.current_context_tokens == 1400

    @allure.title('token_counter 為 None 時不應報錯')
    async def test_token_counter_disabled(self) -> None:
        """token_counter 為 None 時不應報錯。"""
        provider = MockProvider([(['回應'], _make_final_message('回應'))])
        config = AgentCoreConfig(
            provider=ProviderConfig(api_key='sk-test'),
        )
        agent = Agent(
            config=config,
            provider=provider,
            token_counter=None,
        )

        # 應正常執行，不拋出異常
        await collect_stream(agent, '測試')

    @allure.title('工具迴圈中 token_counter 應反映最後一次 API 呼叫的 token 數')
    async def test_token_counter_reflects_latest_call_in_tool_loop(self) -> None:
        """工具迴圈中 token_counter 應反映最後一次 API 呼叫的 token 數。"""
        registry = ToolRegistry()
        registry.register(
            name='test_tool',
            description='測試工具',
            parameters={
                'type': 'object',
                'properties': {'input': {'type': 'string'}},
                'required': ['input'],
            },
            handler=lambda input: 'result',  # type: ignore[reportUnknownLambdaType]
        )

        # 第一次呼叫：tool_use，input_tokens=1000
        first_usage = UsageInfo(input_tokens=1000, output_tokens=100)
        first_msg = FinalMessage(
            content=[
                {'type': 'tool_use', 'id': 'tool_1', 'name': 'test_tool', 'input': {'input': 'x'}},
            ],
            stop_reason='tool_use',
            usage=first_usage,
        )

        # 第二次呼叫：end_turn，input_tokens=2000（含工具結果）
        second_usage = UsageInfo(input_tokens=2000, output_tokens=500)
        second_msg = FinalMessage(
            content=[{'type': 'text', 'text': '完成'}],
            stop_reason='end_turn',
            usage=second_usage,
        )

        provider = MockProvider(
            [
                ([], first_msg),
                (['完成'], second_msg),
            ]
        )
        agent = _make_agent(provider, tool_registry=registry)

        await collect_stream(agent, '測試')

        # 應反映最後一次 API 呼叫的 token 數
        assert agent.token_counter is not None
        assert agent.token_counter.current_context_tokens == 2500


# =============================================================================
# Rule: Agent 應在適當時機自動觸發 compact
# =============================================================================


@allure.feature('基礎聊天功能')
@allure.story('Agent 應在適當時機自動觸發 compact')
class TestCompactIntegration:
    """測試 Agent 與 Compact 的整合。"""

    @allure.title('超過閾值時自動觸發 compact')
    async def test_compact_triggered_when_threshold_exceeded(self) -> None:
        """Scenario: 超過閾值時自動觸發 compact。"""
        # 設定兩輪回應：第一輪正常，第二輪觸發 compact
        first_msg = FinalMessage(
            content=[{'type': 'text', 'text': '第一輪回應'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=170_000, output_tokens=5000),
        )
        second_msg = FinalMessage(
            content=[{'type': 'text', 'text': '第二輪回應'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=50_000, output_tokens=2000),
        )

        provider = MockProvider(
            [
                (['第一輪回應'], first_msg),
                (['第二輪回應'], second_msg),
            ]
        )
        # 同時為 provider 加上 create 方法供 compact 摘要使用
        provider.create = AsyncMock(  # type: ignore[attr-defined]
            return_value=FinalMessage(
                content=[{'type': 'text', 'text': '對話摘要'}],
                stop_reason='end_turn',
                usage=UsageInfo(input_tokens=100, output_tokens=50),
            )
        )

        agent = _make_agent(provider)

        # 第一輪：使用率 87.5% → 超過 80%
        await collect_stream(agent, '第一則訊息')

        # token_counter 應反映高使用率
        assert agent.token_counter is not None
        assert agent.token_counter.usage_percent > 80.0

        # 第二輪：應觸發 compact
        events: list[dict[str, Any]] = []
        async for item in agent.stream_message('第二則訊息'):
            if isinstance(item, dict):
                events.append(item)

        # 應有 compact 事件
        compact_events = [e for e in events if e.get('type') == 'compact']
        assert len(compact_events) > 0

    @allure.title('未超過閾值時不觸發 compact')
    async def test_compact_not_triggered_below_threshold(self) -> None:
        """Scenario: 未超過閾值時不觸發 compact。"""
        final_msg = FinalMessage(
            content=[{'type': 'text', 'text': '回應'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=10_000, output_tokens=1000),
        )

        provider = MockProvider([(['回應'], final_msg)])
        agent = _make_agent(provider)

        events: list[dict[str, Any]] = []
        async for item in agent.stream_message('測試'):
            if isinstance(item, dict):
                events.append(item)

        # 不應有 compact 事件
        compact_events = [e for e in events if e.get('type') == 'compact']
        assert len(compact_events) == 0

    @allure.title('compact 時應 yield SSE 事件通知前端')
    async def test_compact_yields_events(self) -> None:
        """compact 時應 yield SSE 事件通知前端。"""
        # 手動設定高 usage，讓下一輪觸發 compact
        first_msg = FinalMessage(
            content=[{'type': 'text', 'text': '回應'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=170_000, output_tokens=5000),
        )
        second_msg = FinalMessage(
            content=[{'type': 'text', 'text': '第二輪回應'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=50_000, output_tokens=2000),
        )

        provider = MockProvider(
            [
                (['回應'], first_msg),
                (['第二輪回應'], second_msg),
            ]
        )
        provider.create = AsyncMock(  # type: ignore[attr-defined]
            return_value=FinalMessage(
                content=[{'type': 'text', 'text': '對話摘要'}],
                stop_reason='end_turn',
                usage=UsageInfo(input_tokens=100, output_tokens=50),
            )
        )

        agent = _make_agent(provider)

        # 第一輪建立高 usage
        await collect_stream(agent, '第一則訊息')

        # 第二輪應觸發 compact 並 yield 事件
        events: list[dict[str, Any]] = []
        async for item in agent.stream_message('第二則訊息'):
            if isinstance(item, dict):
                events.append(item)

        compact_events = [e for e in events if e.get('type') == 'compact']
        assert len(compact_events) >= 1
        # compact 事件應包含壓縮結果資訊
        assert 'data' in compact_events[0]
