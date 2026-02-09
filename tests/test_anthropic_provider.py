"""Anthropic Provider 測試模組。

根據 docs/features/provider.feature 規格撰寫測試案例。
涵蓋：
- Rule: Provider 應封裝 LLM 特定邏輯
- Rule: Provider 應轉換特定例外為通用例外
- Rule: Anthropic Provider 應支援 Prompt Caching
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import allure
import pytest

# --- 輔助工具 ---


def _make_final_message(
    *,
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = 'end_turn',
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> MagicMock:
    """建立模擬的 final message。"""
    msg = MagicMock()

    if content is None:
        content = [{'type': 'text', 'text': 'Hello'}]

    # 將 content dict 轉換為有 .type 屬性的物件
    blocks: list[MagicMock] = []
    for block_dict in content:
        block = MagicMock()
        block.type = block_dict['type']
        block.model_dump = MagicMock(return_value=block_dict)
        if block_dict['type'] == 'tool_use':
            block.id = block_dict.get('id', 'tool_1')
            block.name = block_dict.get('name', 'test_tool')
            block.input = block_dict.get('input', {})
        elif block_dict['type'] == 'text':
            block.text = block_dict.get('text', '')
        blocks.append(block)

    msg.content = blocks
    msg.stop_reason = stop_reason

    # Usage 資訊
    msg.usage = MagicMock()
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_creation_input_tokens = cache_creation_input_tokens
    msg.usage.cache_read_input_tokens = cache_read_input_tokens

    return msg


def _make_mock_stream(
    text_chunks: list[str],
    final_message: MagicMock,
) -> AsyncMock:
    """建立模擬的 Anthropic stream context manager。"""
    stream = AsyncMock()

    # text_stream 是一個 async iterator
    async def _text_stream() -> Any:
        for chunk in text_chunks:
            yield chunk

    stream.text_stream = _text_stream()
    stream.get_final_message = AsyncMock(return_value=final_message)

    # 作為 context manager
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


@allure.feature('LLM Provider 抽象層')
@allure.story('Provider 應支援串流')
class TestAnthropicProviderStream:
    """Rule: Provider 應封裝 LLM 特定邏輯。"""

    @allure.title('Anthropic Provider 串流回應')
    async def test_stream_text_response(self) -> None:
        """Scenario: Anthropic Provider 串流回應。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider

        final_msg = _make_final_message(
            content=[{'type': 'text', 'text': 'Hello World'}],
            stop_reason='end_turn',
        )
        mock_stream = _make_mock_stream(['Hello', ' World'], final_msg)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        config = ProviderConfig(api_key='sk-test')
        provider = AnthropicProvider(config, client=mock_client)

        messages = [{'role': 'user', 'content': 'Hi'}]
        text_parts: list[str] = []

        async with provider.stream(messages=messages, system='test') as result:
            async for token in result.text_stream:
                text_parts.append(token)
            final = await result.get_final_result()

        assert ''.join(text_parts) == 'Hello World'
        assert final.stop_reason == 'end_turn'
        assert len(final.content) == 1
        assert final.content[0]['type'] == 'text'

    @allure.title('Anthropic Provider 處理工具調用')
    async def test_stream_tool_use_response(self) -> None:
        """Scenario: Anthropic Provider 處理工具調用。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider

        tool_content = [
            {'type': 'text', 'text': '讓我讀取檔案'},
            {
                'type': 'tool_use',
                'id': 'tool_abc',
                'name': 'read_file',
                'input': {'path': 'test.py'},
            },
        ]
        final_msg = _make_final_message(content=tool_content, stop_reason='tool_use')
        mock_stream = _make_mock_stream(['讓我讀取檔案'], final_msg)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        config = ProviderConfig(api_key='sk-test')
        provider = AnthropicProvider(config, client=mock_client)

        messages = [{'role': 'user', 'content': '讀取 test.py'}]

        async with provider.stream(messages=messages, system='test') as result:
            async for _ in result.text_stream:
                pass
            final = await result.get_final_result()

        assert final.stop_reason == 'tool_use'
        assert len(final.content) == 2
        assert final.content[1]['type'] == 'tool_use'
        assert final.content[1]['name'] == 'read_file'

    @allure.title('串流完成後應回傳 usage 資訊')
    async def test_stream_returns_usage_info(self) -> None:
        """串流完成後應回傳 usage 資訊。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider

        final_msg = _make_final_message(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=80,
            cache_read_input_tokens=20,
        )
        mock_stream = _make_mock_stream(['Hi'], final_msg)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        config = ProviderConfig(api_key='sk-test')
        provider = AnthropicProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            async for _ in result.text_stream:
                pass
            final = await result.get_final_result()

        assert final.usage is not None
        assert final.usage.input_tokens == 100
        assert final.usage.output_tokens == 50
        assert final.usage.cache_creation_input_tokens == 80
        assert final.usage.cache_read_input_tokens == 20


@allure.feature('LLM Provider 抽象層')
@allure.story('Provider 應將錯誤轉換為特定例外')
class TestAnthropicProviderErrors:
    """Rule: Provider 應轉換特定例外為通用例外。"""

    @allure.title('API 金鑰無效 → ProviderAuthError')
    async def test_auth_error(self) -> None:
        """Scenario: API 金鑰無效 → ProviderAuthError。"""
        from anthropic import AuthenticationError

        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider
        from agent_core.providers.exceptions import ProviderAuthError

        mock_client = MagicMock()
        # AuthenticationError 需要 response 和 body 參數
        auth_error = AuthenticationError(
            message='Invalid API Key',
            response=MagicMock(status_code=401),
            body={'error': {'message': 'Invalid API Key'}},
        )

        # 讓 stream 在進入 context manager 時拋出錯誤
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=auth_error)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=mock_ctx)

        config = ProviderConfig(api_key='sk-bad')
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderAuthError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API 連線失敗 → ProviderConnectionError')
    async def test_connection_error(self) -> None:
        """Scenario: API 連線失敗 → ProviderConnectionError。"""
        from anthropic import APIConnectionError

        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider
        from agent_core.providers.exceptions import ProviderConnectionError

        mock_client = MagicMock()
        conn_error = APIConnectionError(request=MagicMock())

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=conn_error)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=mock_ctx)

        config = ProviderConfig(api_key='sk-test')
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderConnectionError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

    @allure.title('API 回應超時 → ProviderTimeoutError')
    async def test_timeout_error(self) -> None:
        """Scenario: API 回應超時 → ProviderTimeoutError。"""
        from anthropic import APITimeoutError

        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider
        from agent_core.providers.exceptions import ProviderTimeoutError

        mock_client = MagicMock()
        timeout_error = APITimeoutError(request=MagicMock())

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=timeout_error)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=mock_ctx)

        config = ProviderConfig(api_key='sk-test')
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderTimeoutError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass


@allure.feature('LLM Provider 抽象層')
@allure.story('Provider 應支援 Prompt Caching')
class TestAnthropicProviderCaching:
    """Rule: Anthropic Provider 應支援 Prompt Caching。"""

    @allure.title('在 system prompt 加上 cache_control')
    def test_system_prompt_cache_control(self) -> None:
        """Scenario: 在 system prompt 加上 cache_control。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider

        config = ProviderConfig(api_key='sk-test', enable_prompt_caching=True)
        provider = AnthropicProvider(config, client=MagicMock())

        kwargs = provider.build_stream_kwargs(
            messages=[{'role': 'user', 'content': 'Hi'}],
            system='你是助手',
        )

        # system 應為包含 cache_control 的 list
        assert isinstance(kwargs['system'], list)
        assert kwargs['system'][0]['cache_control'] == {'type': 'ephemeral'}

    @allure.title('在工具定義最後加上 cache_control')
    def test_tools_cache_control(self) -> None:
        """Scenario: 在工具定義最後加上 cache_control。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider

        config = ProviderConfig(api_key='sk-test', enable_prompt_caching=True)
        provider = AnthropicProvider(config, client=MagicMock())

        tools: list[dict[str, Any]] = [
            {'name': 'tool_a', 'description': 'A', 'input_schema': {}},
            {'name': 'tool_b', 'description': 'B', 'input_schema': {}},
        ]

        kwargs = provider.build_stream_kwargs(
            messages=[{'role': 'user', 'content': 'Hi'}],
            system='test',
            tools=tools,
        )

        # 最後一個工具應有 cache_control
        assert 'cache_control' not in kwargs['tools'][0]
        assert kwargs['tools'][1]['cache_control'] == {'type': 'ephemeral'}

    @allure.title('停用 Prompt Caching')
    def test_caching_disabled(self) -> None:
        """Scenario: 停用 Prompt Caching。"""
        from agent_core.config import ProviderConfig
        from agent_core.providers.anthropic_provider import AnthropicProvider

        config = ProviderConfig(api_key='sk-test', enable_prompt_caching=False)
        provider = AnthropicProvider(config, client=MagicMock())

        tools: list[dict[str, Any]] = [
            {'name': 'tool_a', 'description': 'A', 'input_schema': {}},
        ]

        kwargs = provider.build_stream_kwargs(
            messages=[{'role': 'user', 'content': 'Hi'}],
            system='test',
            tools=tools,
        )

        # system 應為普通字串
        assert kwargs['system'] == 'test'
        # 工具不應有 cache_control
        assert 'cache_control' not in kwargs['tools'][0]
