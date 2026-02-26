"""Gemini Provider 測試模組。

根據 docs/features/core/gemini_provider.feature 規格撰寫測試案例。
涵蓋：
- Rule: Gemini Provider 應實作 LLMProvider 介面
- Rule: Gemini Provider 應正確轉換訊息格式
- Rule: Gemini Provider 應映射 StopReason
- Rule: Gemini Provider 應轉換特定例外為通用例外
- Rule: Gemini Provider 應支援 token 計數
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import allure
import pytest
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from agent_core.types import MessageParam, ToolDefinition

# --- 輔助工具 ---


def _make_gemini_response(
    *,
    text: str | None = 'Hello',
    function_calls: list[dict[str, Any]] | None = None,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
    prompt_tokens: int = 10,
    candidates_tokens: int = 20,
    cached_tokens: int = 0,
) -> types.GenerateContentResponse:
    """建立模擬的 Gemini GenerateContentResponse。"""
    parts: list[types.Part] = []
    if text is not None:
        parts.append(types.Part(text=text))
    if function_calls:
        for fc in function_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=fc.get('id'),
                        name=fc['name'],
                        args=fc.get('args', {}),
                    )
                )
            )

    candidate = types.Candidate(
        content=types.Content(role='model', parts=parts),
        finish_reason=finish_reason,
    )

    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidates_tokens,
        cached_content_token_count=cached_tokens if cached_tokens > 0 else None,
        total_token_count=prompt_tokens + candidates_tokens,
    )

    return types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=usage,
    )


def _make_gemini_stream_chunks(
    *,
    text_deltas: list[str] | None = None,
    function_calls: list[dict[str, Any]] | None = None,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
    prompt_tokens: int = 10,
    candidates_tokens: int = 20,
) -> list[types.GenerateContentResponse]:
    """建立模擬的 Gemini 串流 chunk 列表。"""
    chunks: list[types.GenerateContentResponse] = []

    # 文字 delta chunks
    if text_deltas:
        for delta in text_deltas:
            chunk = types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role='model', parts=[types.Part(text=delta)]),
                        finish_reason=None,
                    )
                ],
            )
            chunks.append(chunk)

    # function_call chunk
    if function_calls:
        fc_parts: list[types.Part] = []
        for fc in function_calls:
            fc_parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=fc.get('id'),
                        name=fc['name'],
                        args=fc.get('args', {}),
                    )
                )
            )
        chunk = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role='model', parts=fc_parts),
                    finish_reason=None,
                )
            ],
        )
        chunks.append(chunk)

    # 最後一個 chunk：包含 finish_reason 和 usage
    final_chunk = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role='model', parts=[]),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidates_tokens,
            total_token_count=prompt_tokens + candidates_tokens,
        ),
    )
    chunks.append(final_chunk)

    return chunks


class _MockAsyncIterator:
    """模擬的 async iterator（Gemini stream response）。"""

    def __init__(self, chunks: list[types.GenerateContentResponse]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> _MockAsyncIterator:
        return self

    async def __anext__(self) -> types.GenerateContentResponse:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


# --- 測試類別 ---


@allure.feature('Gemini Provider')
@allure.story('Provider 應支援串流')
class TestGeminiProviderStream:
    """Rule: Gemini Provider 應實作 LLMProvider 介面。"""

    @allure.title('Gemini Provider 串流回應')
    async def test_stream_text_response(self) -> None:
        """Scenario: Gemini Provider 串流回應。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        chunks = _make_gemini_stream_chunks(text_deltas=['Hello', ' World'])
        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=_MockAsyncIterator(chunks)
        )

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

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

    @allure.title('Gemini Provider 處理工具調用串流')
    async def test_stream_tool_use_response(self) -> None:
        """Scenario: Gemini Provider 處理工具調用。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        chunks = _make_gemini_stream_chunks(
            text_deltas=['讓我讀取檔案'],
            function_calls=[
                {'id': 'call_abc', 'name': 'read_file', 'args': {'path': 'test.py'}},
            ],
            finish_reason=types.FinishReason.STOP,
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=_MockAsyncIterator(chunks)
        )

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

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

    @allure.title('串流完成後應回傳 usage 資訊')
    async def test_stream_returns_usage(self) -> None:
        """串流完成後應回傳 usage 資訊。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        chunks = _make_gemini_stream_chunks(
            text_deltas=['Hi'],
            prompt_tokens=100,
            candidates_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=_MockAsyncIterator(chunks)
        )

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            async for _ in result.text_stream:
                pass
            final = await result.get_final_result()

        assert final.usage is not None
        assert final.usage.input_tokens == 100
        assert final.usage.output_tokens == 50


@allure.feature('Gemini Provider')
@allure.story('Provider 應支援非串流呼叫')
class TestGeminiProviderCreate:
    """Rule: Gemini Provider 應實作 LLMProvider 介面（create）。"""

    @allure.title('Gemini Provider 非串流呼叫')
    async def test_create_basic(self) -> None:
        """Scenario: Gemini Provider 非串流呼叫。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        response = _make_gemini_response(text='Hello World')
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hi'}]
        final = await provider.create(messages=messages, system='test')

        assert final.stop_reason == 'end_turn'
        assert len(final.content) == 1
        assert final.content[0]['type'] == 'text'
        assert final.content[0]['text'] == 'Hello World'


@allure.feature('Gemini Provider')
@allure.story('Provider 應將錯誤轉換為特定例外')
class TestGeminiProviderErrors:
    """Rule: Gemini Provider 應轉換特定例外為通用例外。"""

    @allure.title('API 金鑰無效 → ProviderAuthError')
    async def test_auth_error(self) -> None:
        """Scenario: API 金鑰無效 → ProviderAuthError。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderAuthError
        from agent_core.providers.gemini_provider import GeminiProvider

        mock_client = MagicMock()
        auth_error = ClientError(401, {'error': {'message': 'Invalid API Key'}})
        mock_client.aio.models.generate_content_stream = AsyncMock(side_effect=auth_error)

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='bad-key')
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

        with pytest.raises(ProviderAuthError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API 速率限制 → ProviderRateLimitError')
    async def test_rate_limit_error(self) -> None:
        """Scenario: API 速率限制 → ProviderRateLimitError。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderRateLimitError
        from agent_core.providers.gemini_provider import GeminiProvider

        mock_client = MagicMock()
        rate_error = ClientError(429, {'error': {'message': 'Rate limit exceeded'}})
        mock_client.aio.models.generate_content_stream = AsyncMock(side_effect=rate_error)

        config = ProviderConfig(
            provider_type='gemini', model='gemini-2.0-flash', api_key='test', max_retries=0
        )
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

        with pytest.raises(ProviderRateLimitError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API Server 錯誤 → ProviderError（可重試）')
    async def test_server_error(self) -> None:
        """Scenario: API Server 錯誤應可重試。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.exceptions import ProviderError
        from agent_core.providers.gemini_provider import GeminiProvider

        mock_client = MagicMock()
        server_error = ServerError(500, {'error': {'message': 'Internal error'}})
        mock_client.aio.models.generate_content_stream = AsyncMock(side_effect=server_error)

        config = ProviderConfig(
            provider_type='gemini', model='gemini-2.0-flash', api_key='test', max_retries=0
        )
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

        with pytest.raises(ProviderError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass


@allure.feature('Gemini Provider')
@allure.story('Provider 應正確轉換訊息格式')
class TestGeminiMessageConversion:
    """Rule: Gemini Provider 應正確轉換訊息格式。"""

    def _get_provider(self) -> Any:
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        return GeminiProvider(config, client=cast(genai.Client, MagicMock()))

    @allure.title('assistant 角色轉為 model 角色')
    def test_assistant_to_model_role(self) -> None:
        """Scenario: assistant 角色轉為 model 角色。"""
        provider = self._get_provider()
        messages: list[MessageParam] = [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello'},
        ]

        result = provider._convert_messages(messages)  # pyright: ignore[reportPrivateUsage]

        assert len(result) == 2
        assert result[0].role == 'user'
        assert result[1].role == 'model'

    @allure.title('tool_result 區塊轉為 function_response Part')
    def test_tool_result_to_function_response(self) -> None:
        """Scenario: tool_result 區塊轉為 function_response Part。"""
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

        result = provider._convert_messages(messages)  # pyright: ignore[reportPrivateUsage]

        # user + model + user(function_response) = 3
        assert len(result) == 3
        # 最後一則應包含 function_response part
        last_msg = result[2]
        assert last_msg.role == 'user'
        assert last_msg.parts is not None
        assert len(last_msg.parts) >= 1
        # 確認有 function_response part
        fr_part = last_msg.parts[0]
        assert fr_part.function_response is not None
        assert fr_part.function_response.name == 'search'

    @allure.title('assistant 訊息中的 ToolUseBlock 轉為 function_call Part')
    def test_assistant_tool_use_to_function_call(self) -> None:
        """Scenario: assistant 訊息中的 ToolUseBlock 轉為 function_call Part。"""
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

        result = provider._convert_messages(messages)  # pyright: ignore[reportPrivateUsage]

        assert len(result) == 1
        model_msg = result[0]
        assert model_msg.role == 'model'
        assert model_msg.parts is not None
        assert len(model_msg.parts) == 2
        # 第一個 part 是文字
        assert model_msg.parts[0].text == '讓我查查'
        # 第二個 part 是 function_call
        assert model_msg.parts[1].function_call is not None
        assert model_msg.parts[1].function_call.name == 'read_file'
        assert model_msg.parts[1].function_call.args == {'path': 'main.py'}


@allure.feature('Gemini Provider')
@allure.story('工具定義應轉換為 Gemini FunctionDeclaration 格式')
class TestGeminiToolConversion:
    """Rule: Gemini Provider 應正確轉換 ToolDefinition。"""

    @allure.title('ToolDefinition 轉為 FunctionDeclaration')
    def test_convert_tools(self) -> None:
        """Scenario: ToolDefinition 轉為 FunctionDeclaration。"""
        from agent_core.providers.gemini_provider import GeminiProvider

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

        result = GeminiProvider._convert_tools(tools)  # pyright: ignore[reportPrivateUsage]

        assert len(result) == 1
        assert result[0].function_declarations is not None
        fd = result[0].function_declarations[0]
        assert fd.name == 'read_file'
        assert fd.description == '讀取檔案內容'


@allure.feature('Gemini Provider')
@allure.story('Provider 應使用標準化的 StopReason')
class TestGeminiStopReasonMapping:
    """Rule: Gemini Provider 應映射 StopReason。"""

    def _get_provider(self) -> Any:
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        return GeminiProvider(config, client=cast(genai.Client, MagicMock()))

    @allure.title('"STOP" 映射為 "end_turn"')
    def test_stop_to_end_turn(self) -> None:
        provider = self._get_provider()
        response = _make_gemini_response(finish_reason=types.FinishReason.STOP)
        result = provider._parse_response(response)  # pyright: ignore[reportPrivateUsage]
        assert result.stop_reason == 'end_turn'

    @allure.title('包含 function_call 時應回傳 "tool_use"')
    def test_function_call_to_tool_use(self) -> None:
        provider = self._get_provider()
        response = _make_gemini_response(
            text='讓我查查',
            function_calls=[{'name': 'search', 'args': {'q': 'test'}}],
            finish_reason=types.FinishReason.STOP,
        )
        result = provider._parse_response(response)  # pyright: ignore[reportPrivateUsage]
        assert result.stop_reason == 'tool_use'

    @allure.title('"MAX_TOKENS" 映射為 "max_tokens"')
    def test_max_tokens(self) -> None:
        provider = self._get_provider()
        response = _make_gemini_response(finish_reason=types.FinishReason.MAX_TOKENS)
        result = provider._parse_response(response)  # pyright: ignore[reportPrivateUsage]
        assert result.stop_reason == 'max_tokens'

    @allure.title('"SAFETY" 映射為 "end_turn"')
    def test_safety_to_end_turn(self) -> None:
        provider = self._get_provider()
        response = _make_gemini_response(
            text=None,
            finish_reason=types.FinishReason.SAFETY,
        )
        result = provider._parse_response(response)  # pyright: ignore[reportPrivateUsage]
        assert result.stop_reason == 'end_turn'


@allure.feature('Gemini Provider')
@allure.story('Provider 應支援 token 計數')
class TestGeminiTokenCounting:
    """Rule: Gemini Provider 應支援 token 計數。"""

    @allure.title('使用 Gemini API 計算 token 數')
    async def test_count_tokens_returns_positive_int(self) -> None:
        """Scenario: 使用 Gemini API 計算 token 數。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.gemini_provider import GeminiProvider

        mock_client = MagicMock()
        mock_count_response = MagicMock()
        mock_count_response.total_tokens = 42
        mock_client.aio.models.count_tokens = AsyncMock(return_value=mock_count_response)

        config = ProviderConfig(provider_type='gemini', model='gemini-2.0-flash', api_key='test')
        provider = GeminiProvider(config, client=cast(genai.Client, mock_client))

        messages: list[MessageParam] = [{'role': 'user', 'content': 'Hello world'}]
        count = await provider.count_tokens(messages=messages, system='你是助手')

        assert isinstance(count, int)
        assert count == 42


@allure.feature('Gemini Provider')
@allure.story('ProviderConfig 環境變數映射')
class TestGeminiConfigApiKey:
    """Rule: ProviderConfig 應根據 provider_type 讀取正確的環境變數。"""

    @allure.title('Gemini provider 讀取 GEMINI_API_KEY')
    def test_gemini_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scenario: Gemini provider 讀取 GEMINI_API_KEY。"""
        from agent_core.config import ProviderConfig

        monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
        config = ProviderConfig(provider_type='gemini')

        assert config.get_api_key() == 'test-gemini-key'
