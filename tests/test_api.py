"""聊天 API 單元測試。

對應 docs/features/chat_api.feature 中定義的驗收規格。
使用 httpx AsyncClient + Mock 隔離 Redis 與 Anthropic API。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from agent_demo.main import app

# --- 測試用常數 ---
STREAM_URL = '/api/chat/stream'
RESET_URL = '/api/chat/reset'
SESSION_COOKIE = {'session_id': 'test-session-abc'}


# --- 辅助函數 ---
def parse_sse(text: str) -> list[dict[str, str]]:
    """解析 SSE 回應文本為事件列表。

    Args:
        text: 原始 SSE 文本

    Returns:
        事件字典列表，每個包含 type 和 data
    """
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in text.split('\n'):
        if line.startswith('event:'):
            # SSE 慣例：冒號後的第一個空格為分隔符，僅裁掉該空格
            value = line.split(':', 1)[1]
            current['type'] = value[1:] if value.startswith(' ') else value
        elif line.startswith('data:'):
            value = line.split(':', 1)[1]
            current['data'] = value[1:] if value.startswith(' ') else value
        elif line == '' and current:
            events.append(current)
            current = {}

    if current:
        events.append(current)

    return events


async def _mock_stream(tokens: list[str]) -> AsyncIterator[str]:
    """模擬 Agent 串流回應。

    Args:
        tokens: 要逐個 yield 的 token 列表
    """
    for token in tokens:
        yield token


def _make_mock_session_manager() -> MagicMock:
    """建立模擬的 SessionManager。"""
    manager = MagicMock()
    manager.load = AsyncMock(return_value=[])
    manager.save = AsyncMock()
    manager.reset = AsyncMock()
    return manager


# --- 測試類別 ---
class TestSSEStreaming:
    """測試 SSE 串流回應 — 對應 Rule: SSE 端點應正確串流回傳 Agent 回應"""

    @pytest.mark.asyncio
    async def test_normal_stream_returns_token_and_done_events(self) -> None:
        """正常訊息透過 SSE 逐步傳回。"""
        mock_session = _make_mock_session_manager()
        tokens = ['Hello', ' ', 'World']

        with (
            patch('agent_demo.main.session_manager', mock_session),
            patch('agent_demo.main.Agent') as MockAgent,
        ):
            instance = MagicMock()
            instance.stream_message = MagicMock(return_value=_mock_stream(tokens))
            instance.conversation = [
                {'role': 'user', 'content': '測試'},
                {'role': 'assistant', 'content': 'Hello World'},
            ]
            MockAgent.return_value = instance

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '測試'},
                    cookies=SESSION_COOKIE,
                )

            assert response.status_code == 200
            assert 'text/event-stream' in response.headers['content-type']

            events = parse_sse(response.text)
            token_events = [e for e in events if e['type'] == 'token']
            done_events = [e for e in events if e['type'] == 'done']

            # 驗證 token 事件數量與內容
            assert len(token_events) == 3
            assert ''.join(e['data'] for e in token_events) == 'Hello World'
            assert len(done_events) == 1

        # 驗證歷史已儲存到 Redis
        mock_session.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_message_returns_error_event(self) -> None:
        """空白訊息傳回 SSE error 事件。"""
        mock_session = _make_mock_session_manager()

        with patch('agent_demo.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '   '},
                    cookies=SESSION_COOKIE,
                )

        events = parse_sse(response.text)
        error_events = [e for e in events if e['type'] == 'error']

        assert len(error_events) == 1
        error = json.loads(error_events[0]['data'])
        assert error['type'] == 'ValueError'

    @pytest.mark.asyncio
    async def test_connection_error_returns_error_event(self) -> None:
        """Agent 拋出 ConnectionError 時傳回 SSE error 事件。"""
        mock_session = _make_mock_session_manager()

        async def _failing_stream(content: str) -> AsyncIterator[str]:
            raise ConnectionError('連線中斷')
            yield  # noqa: RET503 — 使 Python 將此視為 async generator

        with (
            patch('agent_demo.main.session_manager', mock_session),
            patch('agent_demo.main.Agent') as MockAgent,
        ):
            instance = MagicMock()
            instance.stream_message = MagicMock(return_value=_failing_stream('測試'))
            instance.conversation = []
            MockAgent.return_value = instance

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '測試'},
                    cookies=SESSION_COOKIE,
                )

        events = parse_sse(response.text)
        error_events = [e for e in events if e['type'] == 'error']

        assert len(error_events) == 1
        error = json.loads(error_events[0]['data'])
        assert error['type'] == 'ConnectionError'
        assert '連線' in error['message']


class TestSessionCookie:
    """測試會話 Cookie 管理 — 對應 Rule: 會話 Cookie 應自動管理"""

    @pytest.mark.asyncio
    async def test_new_session_sets_cookie(self) -> None:
        """首次請求生成新會話 Cookie。"""
        mock_session = _make_mock_session_manager()

        with (
            patch('agent_demo.main.session_manager', mock_session),
            patch('agent_demo.main.Agent') as MockAgent,
        ):
            instance = MagicMock()
            instance.stream_message = MagicMock(return_value=_mock_stream(['回應']))
            instance.conversation = [
                {'role': 'user', 'content': '訊息'},
                {'role': 'assistant', 'content': '回應'},
            ]
            MockAgent.return_value = instance

            # 不帶 Cookie 請求
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '訊息'},
                )

        # 應設定 session_id Cookie
        assert 'session_id' in response.cookies

    @pytest.mark.asyncio
    async def test_existing_session_does_not_reset_cookie(self) -> None:
        """既有會話不重複設定 Cookie。"""
        mock_session = _make_mock_session_manager()

        with (
            patch('agent_demo.main.session_manager', mock_session),
            patch('agent_demo.main.Agent') as MockAgent,
        ):
            instance = MagicMock()
            instance.stream_message = MagicMock(return_value=_mock_stream(['回應']))
            instance.conversation = [
                {'role': 'user', 'content': '訊息'},
                {'role': 'assistant', 'content': '回應'},
            ]
            MockAgent.return_value = instance

            # 帶有既有 Cookie
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '訊息'},
                    cookies=SESSION_COOKIE,
                )

        # 不應重新設定 Cookie
        assert 'session_id' not in response.cookies


class TestSessionHistory:
    """測試會話歷史持久化 — 對應 Rule: 會話歷史應透過 Redis 持久化"""

    @pytest.mark.asyncio
    async def test_conversation_history_accumulates(self) -> None:
        """連續對話歷史累積正確。"""
        # 模擬 Redis 已有一組歷史
        existing_history = [
            {'role': 'user', 'content': '第一問'},
            {'role': 'assistant', 'content': '第一答'},
        ]
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=existing_history)

        with (
            patch('agent_demo.main.session_manager', mock_session),
            patch('agent_demo.main.Agent') as MockAgent,
        ):
            instance = MagicMock()

            # 模擬 stream_message 對 conversation 的副作用（同 Agent 實際行為）
            async def _stream_and_update(content: str) -> AsyncIterator[str]:
                instance.conversation.append({'role': 'user', 'content': content})
                yield '第二答'
                instance.conversation.append({'role': 'assistant', 'content': '第二答'})

            instance.stream_message = _stream_and_update
            # conversation 由 main.py 在調用前賦值為 loaded history，此處不設定
            MockAgent.return_value = instance

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                await client.post(
                    STREAM_URL,
                    json={'message': '第二問'},
                    cookies=SESSION_COOKIE,
                )

        # 驗證 save 被呼叫且歷史包含兩組
        mock_session.save.assert_called_once()
        saved_conversation = mock_session.save.call_args[0][1]
        assert len(saved_conversation) == 4
        assert saved_conversation[2]['role'] == 'user'
        assert saved_conversation[3]['role'] == 'assistant'

    @pytest.mark.asyncio
    async def test_reset_clears_history(self) -> None:
        """清除會話歷史。"""
        mock_session = _make_mock_session_manager()

        with patch('agent_demo.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    RESET_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        mock_session.reset.assert_called_once_with('test-session-abc')
