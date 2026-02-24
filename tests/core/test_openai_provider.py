"""OpenAI Provider 測試模組。

根據 docs/features/core/openai_provider.feature 規格撰寫測試案例。
涵蓋：
- Rule: OpenAI Provider 應實作 LLMProvider 介面
- Rule: OpenAI Provider 應正確轉換訊息格式
- Rule: OpenAI Provider 應映射 StopReason
- Rule: OpenAI Provider 應轉換特定例外為通用例外
- Rule: OpenAI Provider 應支援 Prompt Caching 使用量回報
- Rule: OpenAI Provider 應支援 token 計數
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import allure
import pytest

from agent_core.types import MessageParam, ToolDefinition

# --- 輔助工具 ---


def _make_openai_response(
    *,
    content: str | None = 'Hello',
    finish_reason: str = 'stop',
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
) -> MagicMock:
    """建立模擬的 OpenAI ChatCompletion 回應。"""
    response = MagicMock()

    # Choice
    choice = MagicMock()
    choice.finish_reason = finish_reason
    message = MagicMock()
    message.content = content

    if tool_calls:
        tc_mocks: list[MagicMock] = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.id = tc['id']
            tc_mock.type = 'function'
            tc_mock.function = MagicMock()
            tc_mock.function.name = tc['name']
            tc_mock.function.arguments = json.dumps(tc['arguments'])
            tc_mocks.append(tc_mock)
        message.tool_calls = tc_mocks
    else:
        message.tool_calls = None

    choice.message = message
    response.choices = [choice]

    # Usage
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    # prompt_tokens_details
    if cached_tokens > 0:
        details = MagicMock()
        details.cached_tokens = cached_tokens
        usage.prompt_tokens_details = details
    else:
        usage.prompt_tokens_details = None
    response.usage = usage

    return response


def _make_openai_stream_chunks(
    *,
    text_deltas: list[str] | None = None,
    tool_call_deltas: list[dict[str, Any]] | None = None,
    finish_reason: str = 'stop',
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
) -> list[MagicMock]:
    """建立模擬的 OpenAI 串流 chunk 列表。

    最後一個有 choices 的 chunk 包含 finish_reason，
    最後一個 chunk 只有 usage（choices 為空）。
    """
    chunks: list[MagicMock] = []

    # 文字 delta chunks
    if text_deltas:
        for text in text_deltas:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = text
            delta.tool_calls = None
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = None
            chunk.choices = [choice]
            chunk.usage = None
            chunks.append(chunk)

    # 工具調用 delta chunks
    if tool_call_deltas:
        for tc_delta in tool_call_deltas:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = None

            tc_mock = MagicMock()
            tc_mock.index = tc_delta['index']
            tc_mock.id = tc_delta.get('id')
            if 'name' in tc_delta or 'arguments' in tc_delta:
                tc_mock.function = MagicMock()
                tc_mock.function.name = tc_delta.get('name')
                tc_mock.function.arguments = tc_delta.get('arguments')
            else:
                tc_mock.function = None

            delta.tool_calls = [tc_mock]
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = None
            chunk.choices = [choice]
            chunk.usage = None
            chunks.append(chunk)

    # finish_reason chunk
    finish_chunk = MagicMock()
    finish_delta = MagicMock()
    finish_delta.content = None
    finish_delta.tool_calls = None
    finish_choice = MagicMock()
    finish_choice.delta = finish_delta
    finish_choice.finish_reason = finish_reason
    finish_chunk.choices = [finish_choice]
    finish_chunk.usage = None
    chunks.append(finish_chunk)

    # 最後一個 chunk：只有 usage，沒有 choices
    usage_chunk = MagicMock()
    usage_chunk.choices = []
    usage_mock = MagicMock()
    usage_mock.prompt_tokens = prompt_tokens
    usage_mock.completion_tokens = completion_tokens
    if cached_tokens > 0:
        details = MagicMock()
        details.cached_tokens = cached_tokens
        usage_mock.prompt_tokens_details = details
    else:
        usage_mock.prompt_tokens_details = None
    usage_chunk.usage = usage_mock
    chunks.append(usage_chunk)

    return chunks


class _MockAsyncStream:
    """模擬的 async iterator（OpenAI stream response）。"""

    def __init__(self, chunks: list[MagicMock]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> _MockAsyncStream:
        return self

    async def __anext__(self) -> MagicMock:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


# --- 測試類別 ---


@allure.feature('OpenAI Provider')
@allure.story('Provider 應支援串流')
class TestOpenAIProviderStream:
    """Rule: OpenAI Provider 應實作 LLMProvider 介面。"""

    @allure.title('OpenAI Provider 串流回應')
    async def test_stream_text_response(self) -> None:
        """Scenario: OpenAI Provider 串流回應。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        chunks = _make_openai_stream_chunks(text_deltas=['Hello', ' World'])
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_MockAsyncStream(chunks))

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        text_parts: list[str] = []

        async with provider.stream(messages=messages, system='test') as result:
            async for token in result.text_stream:
                text_parts.append(token)
            final = await result.get_final_result()

        assert ''.join(text_parts) == 'Hello World'
        assert final.stop_reason == 'end_turn'
        assert len(final.content) == 1
        assert final.content[0]['type'] == 'text'

    @allure.title('OpenAI Provider 處理工具調用串流')
    async def test_stream_tool_use_response(self) -> None:
        """Scenario: OpenAI Provider 處理工具調用。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        chunks = _make_openai_stream_chunks(
            text_deltas=['讓我讀取檔案'],
            tool_call_deltas=[
                {'index': 0, 'id': 'call_abc', 'name': 'read_file', 'arguments': '{"path":'},
                {'index': 0, 'arguments': ' "test.py"}'},
            ],
            finish_reason='tool_calls',
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_MockAsyncStream(chunks))

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        messages: list[MessageParam] = [{'role': 'user', 'content': '讀取 test.py'}]

        async with provider.stream(messages=messages, system='test') as result:
            async for _ in result.text_stream:
                pass
            final = await result.get_final_result()

        assert final.stop_reason == 'tool_use'
        assert len(final.content) == 2
        assert final.content[0]['type'] == 'text'
        assert final.content[1]['type'] == 'tool_use'
        assert final.content[1]['name'] == 'read_file'
        assert final.content[1]['input'] == {'path': 'test.py'}

    @allure.title('串流完成後應回傳 usage 資訊（含 cached_tokens）')
    async def test_stream_returns_usage_with_cache(self) -> None:
        """串流完成後應回傳 usage 資訊，包含 prompt caching。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        chunks = _make_openai_stream_chunks(
            text_deltas=['Hi'],
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=30,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_MockAsyncStream(chunks))

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            async for _ in result.text_stream:
                pass
            final = await result.get_final_result()

        assert final.usage is not None
        assert final.usage.input_tokens == 100
        assert final.usage.output_tokens == 50
        assert final.usage.cache_read_input_tokens == 30
        assert final.usage.cache_creation_input_tokens == 0


@allure.feature('OpenAI Provider')
@allure.story('Provider 應支援非串流呼叫')
class TestOpenAIProviderCreate:
    """Rule: OpenAI Provider 應實作 LLMProvider 介面（create）。"""

    @allure.title('OpenAI Provider 非串流呼叫')
    async def test_create_basic(self) -> None:
        """Scenario: OpenAI Provider 非串流呼叫。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        response = _make_openai_response(content='Hello World', finish_reason='stop')
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        final = await provider.create(messages=messages, system='test')

        assert final.stop_reason == 'end_turn'
        assert len(final.content) == 1
        assert final.content[0]['type'] == 'text'
        assert final.content[0]['text'] == 'Hello World'

    @allure.title('非串流呼叫應回傳 cached_tokens')
    async def test_create_with_cache(self) -> None:
        """非串流呼叫應正確解析 cached_tokens。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        response = _make_openai_response(prompt_tokens=200, completion_tokens=50, cached_tokens=80)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        final = await provider.create(messages=[{'role': 'user', 'content': 'Hi'}], system='test')

        assert final.usage is not None
        assert final.usage.input_tokens == 200
        assert final.usage.cache_read_input_tokens == 80


@allure.feature('OpenAI Provider')
@allure.story('Provider 應將錯誤轉換為特定例外')
class TestOpenAIProviderErrors:
    """Rule: OpenAI Provider 應轉換特定例外為通用例外。"""

    @allure.title('API 金鑰無效 → ProviderAuthError')
    async def test_auth_error(self) -> None:
        """Scenario: API 金鑰無效 → ProviderAuthError。"""
        from openai import AuthenticationError

        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderAuthError
        from agent_core.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        auth_error = AuthenticationError(
            message='Invalid API Key',
            response=MagicMock(status_code=401),
            body={'error': {'message': 'Invalid API Key'}},
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=auth_error)

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-bad')
        provider = OpenAIProvider(config, client=mock_client)

        with pytest.raises(ProviderAuthError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API 連線失敗 → ProviderConnectionError')
    async def test_connection_error(self) -> None:
        """Scenario: API 連線失敗 → ProviderConnectionError。"""
        from openai import APIConnectionError

        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderConnectionError
        from agent_core.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        conn_error = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create = AsyncMock(side_effect=conn_error)

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        with pytest.raises(ProviderConnectionError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API 回應超時 → ProviderTimeoutError')
    async def test_timeout_error(self) -> None:
        """Scenario: API 回應超時 → ProviderTimeoutError。"""
        from openai import APITimeoutError

        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderTimeoutError
        from agent_core.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        timeout_error = APITimeoutError(request=MagicMock())
        mock_client.chat.completions.create = AsyncMock(side_effect=timeout_error)

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=mock_client)

        with pytest.raises(ProviderTimeoutError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API 速率限制 → ProviderRateLimitError')
    async def test_rate_limit_error(self) -> None:
        """Scenario: API 速率限制 → ProviderRateLimitError。"""
        from openai import RateLimitError

        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderRateLimitError
        from agent_core.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        rate_error = RateLimitError(
            message='Rate limit exceeded',
            response=MagicMock(status_code=429),
            body={'error': {'message': 'Rate limit exceeded'}},
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=rate_error)

        config = ProviderConfig(
            provider_type='openai', model='gpt-4o', api_key='sk-test', max_retries=0
        )
        provider = OpenAIProvider(config, client=mock_client)

        with pytest.raises(ProviderRateLimitError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass


@allure.feature('OpenAI Provider')
@allure.story('Provider 應正確轉換訊息格式')
class TestOpenAIMessageConversion:
    """Rule: OpenAI Provider 應正確轉換訊息格式。"""

    def _get_provider(self) -> Any:
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        return OpenAIProvider(config, client=MagicMock())

    @allure.title('system prompt 轉為 system role 訊息')
    def test_system_prompt_as_system_message(self) -> None:
        """Scenario: system prompt 轉為 system role 訊息。"""
        provider = self._get_provider()
        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]

        # _convert_messages 是 protected method，直接測試轉換邏輯
        result = provider._convert_messages(messages, '你是助手')  # pyright: ignore[reportPrivateUsage]

        assert result[0] == {'role': 'system', 'content': '你是助手'}
        assert result[1] == {'role': 'user', 'content': 'Hi'}

    @allure.title('tool_result 區塊轉為 tool role 訊息')
    def test_tool_result_to_tool_role(self) -> None:
        """Scenario: tool_result 區塊轉為獨立的 tool role 訊息。"""
        provider = self._get_provider()
        messages: list[MessageParam] = [
            {'role': 'user', 'content': 'Hi'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '讓我查查'},
                    {
                        'type': 'tool_use',
                        'id': 'call_123',
                        'name': 'search',
                        'input': {'q': 'test'},
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'call_123',
                        'content': '搜尋結果',
                    },
                ],
            },
        ]

        result = provider._convert_messages(messages, 'test')  # pyright: ignore[reportPrivateUsage]

        # system + user + assistant + tool = 4
        assert len(result) == 4
        assert result[3]['role'] == 'tool'
        assert result[3]['tool_call_id'] == 'call_123'
        assert result[3]['content'] == '搜尋結果'

    @allure.title('assistant 訊息中的 ToolUseBlock 轉為 tool_calls')
    def test_assistant_tool_use_to_tool_calls(self) -> None:
        """Scenario: assistant 訊息中的 ToolUseBlock 轉為 tool_calls 格式。"""
        provider = self._get_provider()
        messages: list[MessageParam] = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '讓我查查'},
                    {
                        'type': 'tool_use',
                        'id': 'call_456',
                        'name': 'read_file',
                        'input': {'path': 'main.py'},
                    },
                ],
            },
        ]

        result = provider._convert_messages(messages, 'test')  # pyright: ignore[reportPrivateUsage]

        # system + assistant = 2
        assert len(result) == 2
        assistant_msg = result[1]
        assert assistant_msg['role'] == 'assistant'
        assert assistant_msg['content'] == '讓我查查'
        assert len(assistant_msg['tool_calls']) == 1
        tc = assistant_msg['tool_calls'][0]
        assert tc['id'] == 'call_456'
        assert tc['function']['name'] == 'read_file'
        assert json.loads(tc['function']['arguments']) == {'path': 'main.py'}

    @allure.title('純文字 assistant 訊息保持不變')
    def test_plain_text_assistant(self) -> None:
        """純文字 assistant 訊息應保持原始格式。"""
        provider = self._get_provider()
        messages: list[MessageParam] = [
            {'role': 'assistant', 'content': '你好'},
        ]

        result = provider._convert_messages(messages, 'test')  # pyright: ignore[reportPrivateUsage]
        assert result[1] == {'role': 'assistant', 'content': '你好'}


@allure.feature('OpenAI Provider')
@allure.story('工具定義應轉換為 OpenAI function 格式')
class TestOpenAIToolConversion:
    """Rule: OpenAI Provider 應正確轉換 ToolDefinition。"""

    @allure.title('ToolDefinition 轉為 OpenAI function 格式')
    def test_convert_tools(self) -> None:
        """Scenario: ToolDefinition 轉為 OpenAI function 格式。"""
        from agent_core.providers.openai_provider import OpenAIProvider

        tools: list[ToolDefinition] = [
            ToolDefinition(
                name='read_file',
                description='讀取檔案內容',
                input_schema={
                    'type': 'object',
                    'properties': {'path': {'type': 'string'}},
                    'required': ['path'],
                },
            ),
        ]

        result = OpenAIProvider._convert_tools(tools)  # pyright: ignore[reportPrivateUsage]

        assert len(result) == 1
        assert result[0]['type'] == 'function'
        assert result[0]['function']['name'] == 'read_file'
        assert result[0]['function']['description'] == '讀取檔案內容'
        assert result[0]['function']['parameters'] == tools[0]['input_schema']


@allure.feature('OpenAI Provider')
@allure.story('Provider 應使用標準化的 StopReason')
class TestOpenAIStopReasonMapping:
    """Rule: OpenAI Provider 應映射 StopReason。"""

    def _parse(self, finish_reason: str | None) -> Any:
        """輔助方法：建立 mock 並呼叫 _parse_response。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test'),
            client=MagicMock(),
        )
        response = _make_openai_response(finish_reason=finish_reason or '')
        if finish_reason is None:
            response.choices[0].finish_reason = None
        return provider._parse_response(response)  # pyright: ignore[reportPrivateUsage]

    @allure.title('"stop" 映射為 "end_turn"')
    def test_stop_to_end_turn(self) -> None:
        result = self._parse('stop')
        assert result.stop_reason == 'end_turn'

    @allure.title('"tool_calls" 映射為 "tool_use"')
    def test_tool_calls_to_tool_use(self) -> None:
        result = self._parse('tool_calls')
        assert result.stop_reason == 'tool_use'

    @allure.title('"length" 映射為 "max_tokens"')
    def test_length_to_max_tokens(self) -> None:
        result = self._parse('length')
        assert result.stop_reason == 'max_tokens'

    @allure.title('None finish_reason 應回退為 end_turn')
    def test_none_fallback(self) -> None:
        result = self._parse(None)
        assert result.stop_reason == 'end_turn'

    @allure.title('未知 finish_reason 應回退為 end_turn')
    def test_unknown_fallback(self) -> None:
        result = self._parse('content_filter')
        assert result.stop_reason == 'end_turn'


@allure.feature('OpenAI Provider')
@allure.story('Provider 應支援 token 計數')
class TestOpenAITokenCounting:
    """Rule: OpenAI Provider 應支援 token 計數。"""

    @allure.title('使用 tiktoken 計算 token 數')
    async def test_count_tokens_returns_positive_int(self) -> None:
        """Scenario: 使用 tiktoken 計算 token 數。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=MagicMock())

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hello world'}]
        count = await provider.count_tokens(messages=messages, system='你是助手')

        assert isinstance(count, int)
        assert count > 0

    @allure.title('未知模型應使用 fallback encoding')
    async def test_count_tokens_unknown_model_fallback(self) -> None:
        """未知模型應使用 cl100k_base fallback。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        config = ProviderConfig(
            provider_type='openai', model='unknown-model-xyz', api_key='sk-test'
        )
        provider = OpenAIProvider(config, client=MagicMock())

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hello'}]
        count = await provider.count_tokens(messages=messages, system='test')

        assert isinstance(count, int)
        assert count > 0

    @allure.title('含工具定義的 token 計數')
    async def test_count_tokens_with_tools(self) -> None:
        """含工具定義時 token 數應更多。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.openai_provider import OpenAIProvider

        config = ProviderConfig(provider_type='openai', model='gpt-4o', api_key='sk-test')
        provider = OpenAIProvider(config, client=MagicMock())

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hello'}]
        tools: list[ToolDefinition] = [
            ToolDefinition(
                name='search',
                description='搜尋內容',
                input_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}},
            ),
        ]

        count_without = await provider.count_tokens(messages=messages, system='test')
        count_with = await provider.count_tokens(messages=messages, system='test', tools=tools)

        assert count_with > count_without
