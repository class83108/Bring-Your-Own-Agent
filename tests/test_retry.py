"""Retry with Backoff 測試模組。

根據 docs/features/retry.feature 規格撰寫測試案例。
涵蓋：
- Rule: 可重試錯誤應自動重試並使用指數退避
- Rule: 不可重試錯誤應立即拋出
- Rule: 超過最大重試次數應拋出最後的錯誤
- Rule: 重試應適用於所有 Provider 方法
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import allure
import pytest

from agent_core.config import ProviderConfig
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.providers.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)

# --- 輔助工具 ---


def _make_final_message(
    *,
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = 'end_turn',
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    """建立模擬的 final message。"""
    msg = MagicMock()

    if content is None:
        content = [{'type': 'text', 'text': 'Hello'}]

    blocks: list[MagicMock] = []
    for block_dict in content:
        block = MagicMock()
        block.type = block_dict['type']
        block.model_dump = MagicMock(return_value=block_dict)
        if block_dict['type'] == 'text':
            block.text = block_dict.get('text', '')
        blocks.append(block)

    msg.content = blocks
    msg.stop_reason = stop_reason

    msg.usage = MagicMock()
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0

    return msg


def _make_mock_stream(
    text_chunks: list[str],
    final_message: MagicMock,
) -> AsyncMock:
    """建立模擬的 Anthropic stream context manager。"""
    stream = AsyncMock()

    async def _text_stream() -> Any:
        for chunk in text_chunks:
            yield chunk

    stream.text_stream = _text_stream()
    stream.get_final_message = AsyncMock(return_value=final_message)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=stream)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_api_status_error(status_code: int, message: str = 'Error') -> Any:
    """建立模擬的 APIStatusError。"""
    from anthropic import APIStatusError

    response = MagicMock()
    response.status_code = status_code
    response.headers = {}

    return APIStatusError(
        message=message,
        response=response,
        body={'error': {'message': message}},
    )


def _make_auth_error() -> Any:
    """建立模擬的 AuthenticationError。"""
    from anthropic import AuthenticationError

    return AuthenticationError(
        message='Invalid API Key',
        response=MagicMock(status_code=401),
        body={'error': {'message': 'Invalid API Key'}},
    )


def _make_timeout_error() -> Any:
    """建立模擬的 APITimeoutError。"""
    from anthropic import APITimeoutError

    return APITimeoutError(request=MagicMock())


def _make_connection_error() -> Any:
    """建立模擬的 APIConnectionError。"""
    from anthropic import APIConnectionError

    return APIConnectionError(request=MagicMock())


@allure.feature('API 錯誤自動重試')
@allure.story('可重試錯誤應自動重試並使用指數退避')
class TestRetryableErrors:
    """Rule: 可重試錯誤應自動重試並使用指數退避。"""

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('429 Rate Limit 錯誤觸發重試')
    async def test_429_rate_limit_triggers_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: 429 Rate Limit 錯誤觸發重試。"""
        rate_limit_error = _make_api_status_error(429, 'Rate limit exceeded')

        final_msg = _make_final_message()
        success_stream = _make_mock_stream(['Hello'], final_msg)

        mock_client = MagicMock()
        # 前 2 次失敗，第 3 次成功
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=rate_limit_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx, fail_ctx, success_stream])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            chunks = [t async for t in result.text_stream]
            final = await result.get_final_result()

        assert ''.join(chunks) == 'Hello'
        assert final.stop_reason == 'end_turn'
        # 應該呼叫了 3 次 stream
        assert mock_client.messages.stream.call_count == 3
        # 指數退避：第一次等 1 秒，第二次等 2 秒
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('5xx 伺服器錯誤觸發重試')
    async def test_5xx_server_error_triggers_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: 5xx 伺服器錯誤觸發重試。"""
        server_error = _make_api_status_error(500, 'Internal Server Error')

        final_msg = _make_final_message()
        success_stream = _make_mock_stream(['OK'], final_msg)

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=server_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx, success_stream])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            chunks = [t async for t in result.text_stream]

        assert ''.join(chunks) == 'OK'
        assert mock_client.messages.stream.call_count == 2

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('網路超時觸發重試')
    async def test_timeout_triggers_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: 網路超時觸發重試。"""
        timeout_error = _make_timeout_error()

        final_msg = _make_final_message()
        success_stream = _make_mock_stream(['OK'], final_msg)

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=timeout_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx, success_stream])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            chunks = [t async for t in result.text_stream]

        assert ''.join(chunks) == 'OK'
        assert mock_client.messages.stream.call_count == 2

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('連線失敗觸發重試')
    async def test_connection_error_triggers_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: 連線失敗觸發重試。"""
        conn_error = _make_connection_error()

        final_msg = _make_final_message()
        success_stream = _make_mock_stream(['OK'], final_msg)

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=conn_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx, success_stream])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            chunks = [t async for t in result.text_stream]

        assert ''.join(chunks) == 'OK'
        assert mock_client.messages.stream.call_count == 2


@allure.feature('API 錯誤自動重試')
@allure.story('不可重試錯誤應立即拋出')
class TestNonRetryableErrors:
    """Rule: 不可重試錯誤應立即拋出。"""

    @allure.title('401 認證錯誤不重試')
    async def test_401_auth_error_no_retry(self) -> None:
        """Scenario: 401 認證錯誤不重試。"""
        auth_error = _make_auth_error()

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=auth_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=fail_ctx)

        config = ProviderConfig(api_key='sk-bad', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderAuthError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

        # 應只呼叫 1 次，不重試
        assert mock_client.messages.stream.call_count == 1

    @allure.title('400 Bad Request 不重試')
    async def test_400_bad_request_no_retry(self) -> None:
        """Scenario: 400 Bad Request 不重試。"""
        bad_request_error = _make_api_status_error(400, 'Bad Request')

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=bad_request_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=fail_ctx)

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

        # 應只呼叫 1 次，不重試
        assert mock_client.messages.stream.call_count == 1


@allure.feature('API 錯誤自動重試')
@allure.story('超過最大重試次數應拋出最後的錯誤')
class TestRetryExhaustion:
    """Rule: 超過最大重試次數應拋出最後的錯誤。"""

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('重試耗盡後拋出 ProviderRateLimitError')
    async def test_exhausted_retries_raises_last_error(self, mock_sleep: AsyncMock) -> None:
        """Scenario: 重試耗盡後拋出 ProviderRateLimitError。"""
        rate_limit_error = _make_api_status_error(429, 'Rate limit exceeded')

        mock_client = MagicMock()
        fail_ctx_factory = lambda: MagicMock(  # noqa: E731
            __aenter__=AsyncMock(side_effect=rate_limit_error),
            __aexit__=AsyncMock(return_value=False),
        )

        # 所有嘗試都失敗（初始 + 3 次重試 = 4 次）
        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx_factory() for _ in range(4)])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderRateLimitError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

        # 初始嘗試 + 3 次重試 = 4 次
        assert mock_client.messages.stream.call_count == 4


@allure.feature('API 錯誤自動重試')
@allure.story('重試應適用於所有 Provider 方法')
class TestRetryAllMethods:
    """Rule: 重試應適用於所有 Provider 方法。"""

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('stream() 方法支援重試')
    async def test_stream_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: stream() 方法支援重試。"""
        timeout_error = _make_timeout_error()

        final_msg = _make_final_message()
        success_stream = _make_mock_stream(['Hello'], final_msg)

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=timeout_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx, success_stream])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        ) as result:
            chunks = [t async for t in result.text_stream]
            final = await result.get_final_result()

        assert ''.join(chunks) == 'Hello'
        assert final.stop_reason == 'end_turn'

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('create() 方法支援重試')
    async def test_create_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: create() 方法支援重試。"""
        timeout_error = _make_timeout_error()
        final_msg = _make_final_message()

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[timeout_error, final_msg])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        result = await provider.create(messages=[{'role': 'user', 'content': 'Hi'}], system='test')

        assert result.stop_reason == 'end_turn'
        assert mock_client.messages.create.call_count == 2

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('count_tokens() 方法支援重試')
    async def test_count_tokens_retry(self, mock_sleep: AsyncMock) -> None:
        """Scenario: count_tokens() 方法支援重試。"""
        timeout_error = _make_timeout_error()

        token_result = MagicMock()
        token_result.input_tokens = 42

        mock_client = MagicMock()
        mock_client.messages.count_tokens = AsyncMock(side_effect=[timeout_error, token_result])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        result = await provider.count_tokens(
            messages=[{'role': 'user', 'content': 'Hi'}], system='test'
        )

        assert result == 42
        assert mock_client.messages.count_tokens.call_count == 2


@allure.feature('API 錯誤自動重試')
@allure.story('可重試錯誤應自動重試並使用指數退避')
class TestRetryBackoffTiming:
    """驗證指數退避的時間間隔。"""

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('退避間隔應為 initial_delay * 2^attempt')
    async def test_exponential_backoff_delays(self, mock_sleep: AsyncMock) -> None:
        """退避間隔應為 initial_delay * 2^attempt。"""
        rate_limit_error = _make_api_status_error(429, 'Rate limit')

        mock_client = MagicMock()
        fail_ctx_factory = lambda: MagicMock(  # noqa: E731
            __aenter__=AsyncMock(side_effect=rate_limit_error),
            __aexit__=AsyncMock(return_value=False),
        )

        # 全部失敗（4 次嘗試，initial_delay=0.5）
        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx_factory() for _ in range(4)])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=0.5)
        provider = AnthropicProvider(config, client=mock_client)

        with pytest.raises(ProviderRateLimitError):
            async with provider.stream(
                messages=[{'role': 'user', 'content': 'Hi'}], system='test'
            ) as result:
                async for _ in result.text_stream:
                    pass

        # 3 次重��� → 3 次 sleep
        assert mock_sleep.call_count == 3
        # 退避間隔：0.5, 1.0, 2.0
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


@allure.feature('API 錯誤自動重試')
@allure.story('重試過程可透過 SSE 通知前端')
class TestRetrySSENotification:
    """Rule: 重試過程可透過 SSE 通知前端。"""

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @allure.title('重試時觸發回調通知')
    async def test_retry_callback_invoked(self, mock_sleep: AsyncMock) -> None:
        """Scenario: 重試時觸發回調通知。"""
        rate_limit_error = _make_api_status_error(429, 'Rate limit')

        final_msg = _make_final_message()
        success_stream = _make_mock_stream(['OK'], final_msg)

        mock_client = MagicMock()
        fail_ctx = AsyncMock()
        fail_ctx.__aenter__ = AsyncMock(side_effect=rate_limit_error)
        fail_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client.messages.stream = MagicMock(side_effect=[fail_ctx, success_stream])

        config = ProviderConfig(api_key='sk-test', max_retries=3, retry_initial_delay=1.0)
        provider = AnthropicProvider(config, client=mock_client)

        # 驗證 on_retry 回調被觸發
        retry_events: list[dict[str, Any]] = []

        def on_retry(attempt: int, error: Exception, delay: float) -> None:
            retry_events.append(
                {
                    'attempt': attempt,
                    'error_type': type(error).__name__,
                    'delay': delay,
                }
            )

        async with provider.stream(
            messages=[{'role': 'user', 'content': 'Hi'}],
            system='test',
            on_retry=on_retry,
        ) as result:
            async for _ in result.text_stream:
                pass

        assert len(retry_events) == 1
        assert retry_events[0]['attempt'] == 1
        assert retry_events[0]['delay'] == 1.0
