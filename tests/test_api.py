"""聊天 API 單元測試。

對應 docs/features/chat_api.feature 中定義的驗收規格。
使用 httpx AsyncClient + Mock 隔離 Redis 與 Anthropic API。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import allure
import pytest
from httpx import ASGITransport, AsyncClient

from agent_app.main import app

# --- 測試用常數 ---
STREAM_URL = '/api/chat/stream'
HISTORY_URL = '/api/chat/history'
SESSION_COOKIE = {'session_id': 'test-session-abc'}


# --- 辅助函數 ---
def parse_sse(text: str) -> list[dict[str, Any]]:
    """解析 SSE 回應文本為事件列表。

    Args:
        text: 原始 SSE 文本

    Returns:
        事件字典列表，每個包含 type (str) 和 data (Any, 已 JSON 解析)
    """
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for line in text.split('\n'):
        if line.startswith('event:'):
            # SSE 慣例：冒號後的第一個空格為分隔符，僅裁掉該空格
            value = line.split(':', 1)[1]
            current['type'] = value[1:] if value.startswith(' ') else value
        elif line.startswith('data:'):
            value = line.split(':', 1)[1]
            raw_data = value[1:] if value.startswith(' ') else value
            # 對 data 進行 JSON 解析
            try:
                current['data'] = json.loads(raw_data)
            except json.JSONDecodeError:
                current['data'] = raw_data
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
    manager.load_usage = AsyncMock(return_value=[])
    manager.save_usage = AsyncMock()
    manager.reset_usage = AsyncMock()
    return manager


# --- 測試類別 ---
@allure.feature('聊天 API 端點')
@allure.story('SSE 端點應正確串流回傳 Agent 回應')
class TestSSEStreaming:
    """測試 SSE 串流回應 — 對應 Rule: SSE 端點應正確串流回傳 Agent 回應"""

    @pytest.mark.asyncio
    @allure.title('正常訊息透過 SSE 逐步傳回')
    async def test_normal_stream_returns_token_and_done_events(self) -> None:
        """正常訊息透過 SSE 逐步傳回。"""
        mock_session = _make_mock_session_manager()
        tokens = ['Hello', ' ', 'World']

        with (
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
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
    @allure.title('空白訊息傳回 SSE error 事件')
    async def test_empty_message_returns_error_event(self) -> None:
        """空白訊息傳回 SSE error 事件。"""
        mock_session = _make_mock_session_manager()

        with patch('agent_app.main.session_manager', mock_session):
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
        error = error_events[0]['data']
        assert error['type'] == 'ValueError'

    @pytest.mark.asyncio
    @allure.title('Agent 拋出 ConnectionError 時傳回 SSE error 事件')
    async def test_connection_error_returns_error_event(self) -> None:
        """Agent 拋出 ConnectionError 時傳回 SSE error 事件。"""
        mock_session = _make_mock_session_manager()

        async def _failing_stream(content: str) -> AsyncIterator[str]:
            raise ConnectionError('連線中斷')
            yield  # noqa: RET503 — 使 Python 將此視為 async generator

        with (
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
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
        error = error_events[0]['data']
        assert error['type'] == 'ConnectionError'
        assert '連線' in error['message']


@allure.feature('聊天 API 端點')
@allure.story('會話 Cookie 應自動管理')
class TestSessionCookie:
    """測試會話 Cookie 管理 — 對應 Rule: 會話 Cookie 應自動管理"""

    @pytest.mark.asyncio
    @allure.title('首次請求生成新會話 Cookie')
    async def test_new_session_sets_cookie(self) -> None:
        """首次請求生成新會話 Cookie。"""
        mock_session = _make_mock_session_manager()

        with (
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
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
    @allure.title('既有會話應更新 Cookie 過期時間')
    async def test_existing_session_updates_cookie_expiry(self) -> None:
        """既有會話應更新 Cookie 過期時間。"""
        mock_session = _make_mock_session_manager()

        with (
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
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

        # 應更新 Cookie（包含相同的 session_id）
        assert 'session_id' in response.cookies
        assert response.cookies['session_id'] == 'test-session-abc'


@allure.feature('聊天 API 端點')
@allure.story('會話歷史應透過 Redis 持久化')
class TestSessionHistory:
    """測試會話歷史持久化 — 對應 Rule: 會話歷史應透過 Redis 持久化"""

    @pytest.mark.asyncio
    @allure.title('連續對話歷史累積正確')
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
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
        ):
            instance = MagicMock()

            # 模擬 stream_message 對 conversation 的副作用（同 Agent 實際行為）
            async def _stream_and_update(content: str, **kwargs: Any) -> AsyncIterator[str]:
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
    @allure.title('透過 DELETE /api/sessions/{id} 清除會話歷史')
    async def test_delete_session_clears_history(self) -> None:
        """透過 DELETE /api/sessions/{id} 清除會話歷史。"""
        mock_session = _make_mock_session_manager()
        mock_session.delete_session = AsyncMock()

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.delete('/api/sessions/test-session-abc')

        assert response.status_code == 200
        mock_session.delete_session.assert_called_once_with('test-session-abc')


@allure.feature('聊天 API 端點')
@allure.story('會話歷史應可透過 API 讀取')
class TestChatHistory:
    """測試會話歷史讀取 — 對應 Rule: 會話歷史應可透過 API 讀取"""

    @pytest.mark.asyncio
    @allure.title('取得現有會話的歷史記錄')
    async def test_get_existing_session_history(self) -> None:
        """取得現有會話的歷史記錄。"""
        # 模擬 Redis 已有兩組對話
        existing_history = [
            {'role': 'user', 'content': '第一問'},
            {'role': 'assistant', 'content': '第一答'},
            {'role': 'user', 'content': '第二問'},
            {'role': 'assistant', 'content': '第二答'},
        ]
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=existing_history)

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    HISTORY_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        assert 'messages' in data
        messages = data['messages']
        assert len(messages) == 4
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == '第一問'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == '第一答'

    @pytest.mark.asyncio
    @allure.title('取得空會話的歷史記錄')
    async def test_get_empty_session_history(self) -> None:
        """取得空會話的歷史記錄。"""
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=[])

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    HISTORY_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        assert 'messages' in data
        assert data['messages'] == []

    @pytest.mark.asyncio
    @allure.title('無會話時取得歷史記錄')
    async def test_get_history_without_session(self) -> None:
        """無會話時取得歷史記錄。"""
        mock_session = _make_mock_session_manager()

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                # 不帶 Cookie
                response = await client.get(HISTORY_URL)

        assert response.status_code == 200
        data = response.json()
        assert 'messages' in data
        assert data['messages'] == []
        # 確認未嘗試載入會話（因為沒有 session_id）
        mock_session.load.assert_not_called()

    @pytest.mark.asyncio
    @allure.title('取得包含 text blocks 的歷史記錄')
    async def test_get_history_with_text_blocks(self) -> None:
        """取得包含 text blocks 的歷史記錄。"""
        # 模擬 content 是 list，包含 text blocks（來自 tool_use 迴圈）
        existing_history = [
            {'role': 'user', 'content': '請讀取檔案'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '好的，讓我讀取檔案內容'},
                    {
                        'type': 'tool_use',
                        'id': 'tool_1',
                        'name': 'read_file',
                        'input': {'path': 'main.py'},
                    },
                ],
            },
        ]
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=existing_history)

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    HISTORY_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        messages = data['messages']

        # 應該只提取 text 部分，過濾掉 tool_use
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == '請讀取檔案'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == '好的，讓我讀取檔案內容'

    @pytest.mark.asyncio
    @allure.title('取得包含多個 text blocks 的歷史記錄（應合併）')
    async def test_get_history_with_multiple_text_blocks(self) -> None:
        """取得包含多個 text blocks 的歷史記錄（應合併）。"""
        existing_history = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '第一段文字'},
                    {'type': 'text', 'text': '第二段文字'},
                    {'type': 'text', 'text': '第三段文字'},
                ],
            },
        ]
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=existing_history)

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    HISTORY_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        messages = data['messages']

        # 多個 text blocks 應該合併為一個 message
        assert len(messages) == 1
        assert messages[0]['content'] == '第一段文字第二段文字第三段文字'

    @pytest.mark.asyncio
    @allure.title('取得只包含 tool_use blocks 的歷史記錄（應被過濾）')
    async def test_get_history_with_only_tool_use_blocks(self) -> None:
        """取得只包含 tool_use blocks 的歷史記錄（應被過濾）。"""
        existing_history = [
            {'role': 'user', 'content': '請執行工具'},
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'tool_use',
                        'id': 'tool_1',
                        'name': 'read_file',
                        'input': {'path': 'main.py'},
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tool_1', 'content': 'file content'},
                ],
            },
        ]
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=existing_history)

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    HISTORY_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        messages = data['messages']

        # 只有包含 text 的 message 應該被保留
        # assistant 的 tool_use 和 user 的 tool_result 都應該被過濾
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == '請執行工具'

    @pytest.mark.asyncio
    @allure.title('取得包含混合 text 和 tool_use blocks 的歷史記錄')
    async def test_get_history_with_mixed_blocks(self) -> None:
        """取得包含混合 text 和 tool_use blocks 的歷史記錄。"""
        existing_history = [
            {'role': 'user', 'content': '分析這個檔案'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '讓我先讀取檔案'},
                    {
                        'type': 'tool_use',
                        'id': 'tool_1',
                        'name': 'read_file',
                        'input': {'path': 'test.py'},
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tool_1', 'content': 'def test(): pass'},
                ],
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '這個檔案包含一個'},
                    {'type': 'text', 'text': 'test 函數'},
                ],
            },
        ]
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=existing_history)

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    HISTORY_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        messages = data['messages']

        # 應該只保留有 text 的 messages，並合併多個 text blocks
        assert len(messages) == 3
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == '分析這個檔案'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == '讓我先讀取檔案'
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == '這個檔案包含一個test 函數'


STATUS_URL = '/api/agent/status'


@allure.feature('聊天 API 端點')
@allure.story('Agent 配置狀態端點')
class TestAgentStatus:
    """測試 Agent 配置狀態端點 — GET /api/agent/status"""

    @pytest.mark.asyncio
    @allure.title('應回傳目前的 model 名稱與 max_tokens')
    async def test_status_returns_model_and_max_tokens(self) -> None:
        """應回傳目前的 model 名稱與 max_tokens。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get(STATUS_URL)

        assert response.status_code == 200
        data = response.json()
        assert 'model' in data
        assert 'max_tokens' in data
        assert isinstance(data['model'], str)
        assert isinstance(data['max_tokens'], int)

    @pytest.mark.asyncio
    @allure.title('應回傳 context_window 欄位')
    async def test_status_returns_context_window(self) -> None:
        """應回傳 context_window 欄位。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get(STATUS_URL)

        assert response.status_code == 200
        data = response.json()
        assert 'context_window' in data
        assert isinstance(data['context_window'], int)
        assert data['context_window'] > 0

    @pytest.mark.asyncio
    @allure.title('應回傳工具清單（含 source）')
    async def test_status_returns_tools_list(self) -> None:
        """應回傳工具清單（含 source）。"""
        from agent_core.tools.registry import ToolRegistry

        mock_registry = ToolRegistry()
        mock_registry.register(
            name='test_tool',
            description='測試工具',
            parameters={'type': 'object', 'properties': {}},
            handler=lambda: 'ok',
        )

        with patch('agent_app.main.tool_registry', mock_registry):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(STATUS_URL)

        data = response.json()
        assert 'tools' in data
        assert len(data['tools']) == 1
        assert data['tools'][0]['name'] == 'test_tool'
        assert data['tools'][0]['source'] == 'native'

    @pytest.mark.asyncio
    @allure.title('應回傳技能註冊與啟用狀態')
    async def test_status_returns_skills_info(self) -> None:
        """應回傳技能註冊與啟用狀態。"""
        from agent_core.skills.base import Skill
        from agent_core.skills.registry import SkillRegistry

        mock_skills = SkillRegistry()
        mock_skills.register(
            Skill(
                name='code_review',
                description='程式碼審查',
                instructions='...',
            )
        )
        mock_skills.register(
            Skill(
                name='tdd',
                description='測試驅動開發',
                instructions='...',
            )
        )
        mock_skills.activate('code_review')

        with patch('agent_app.main.skill_registry', mock_skills):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(STATUS_URL)

        data = response.json()
        assert 'skills' in data
        assert set(data['skills']['registered']) == {'code_review', 'tdd'}
        assert data['skills']['active'] == ['code_review']

    @pytest.mark.asyncio
    @allure.title('registry 為 None 時應回傳空列表')
    async def test_status_with_no_registries(self) -> None:
        """registry 為 None 時應回傳空列表。"""
        with (
            patch('agent_app.main.tool_registry', None),
            patch('agent_app.main.skill_registry', None),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(STATUS_URL)

        data = response.json()
        assert data['tools'] == []
        assert data['skills'] == {'registered': [], 'active': []}


@allure.feature('聊天 API 端點')
@allure.story('Skill 啟用/停用端點')
class TestSkillManagement:
    """測試 Skill 啟用/停用端點 — POST /api/skills/{name}/activate & deactivate"""

    @pytest.mark.asyncio
    @allure.title('啟用已註冊的 Skill 應成功')
    async def test_activate_registered_skill(self) -> None:
        """啟用已註冊的 Skill 應成功。"""
        from agent_core.skills.base import Skill
        from agent_core.skills.registry import SkillRegistry

        mock_skills = SkillRegistry()
        mock_skills.register(
            Skill(name='code_review', description='程式碼審查', instructions='...')
        )

        with patch('agent_app.main.skill_registry', mock_skills):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post('/api/skills/code_review/activate')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'code_review' in mock_skills.list_active_skills()

    @pytest.mark.asyncio
    @allure.title('啟用不存在的 Skill 應回傳 404')
    async def test_activate_unknown_skill_returns_404(self) -> None:
        """啟用不存在的 Skill 應回傳 404。"""
        from agent_core.skills.registry import SkillRegistry

        mock_skills = SkillRegistry()

        with patch('agent_app.main.skill_registry', mock_skills):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post('/api/skills/nonexistent/activate')

        assert response.status_code == 404

    @pytest.mark.asyncio
    @allure.title('停用已啟用的 Skill 應成功')
    async def test_deactivate_active_skill(self) -> None:
        """停用已啟用的 Skill 應成功。"""
        from agent_core.skills.base import Skill
        from agent_core.skills.registry import SkillRegistry

        mock_skills = SkillRegistry()
        mock_skills.register(
            Skill(name='code_review', description='程式碼審查', instructions='...')
        )
        mock_skills.activate('code_review')

        with patch('agent_app.main.skill_registry', mock_skills):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post('/api/skills/code_review/deactivate')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'code_review' not in mock_skills.list_active_skills()

    @pytest.mark.asyncio
    @allure.title('停用不存在的 Skill 應回傳 404')
    async def test_deactivate_unknown_skill_returns_404(self) -> None:
        """停用不存在的 Skill 應回傳 404。"""
        from agent_core.skills.registry import SkillRegistry

        mock_skills = SkillRegistry()

        with patch('agent_app.main.skill_registry', mock_skills):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post('/api/skills/nonexistent/deactivate')

        assert response.status_code == 404

    @pytest.mark.asyncio
    @allure.title('skill_registry 為 None 時應回傳 404')
    async def test_activate_when_no_skill_registry(self) -> None:
        """skill_registry 為 None 時應回傳 404。"""
        with patch('agent_app.main.skill_registry', None):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post('/api/skills/any/activate')

        assert response.status_code == 404

    @pytest.mark.asyncio
    @allure.title('啟用 Skill 後，/api/agent/status 應反映變更')
    async def test_status_reflects_activation(self) -> None:
        """啟用 Skill 後，/api/agent/status 應反映變更。"""
        from agent_core.skills.base import Skill
        from agent_core.skills.registry import SkillRegistry

        mock_skills = SkillRegistry()
        mock_skills.register(Skill(name='tdd', description='測試驅動開發', instructions='...'))

        with patch('agent_app.main.skill_registry', mock_skills):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                # 啟用前
                before = await client.get(STATUS_URL)
                assert before.json()['skills']['active'] == []

                # 啟用
                await client.post('/api/skills/tdd/activate')

                # 啟用後
                after = await client.get(STATUS_URL)
                assert 'tdd' in after.json()['skills']['active']


@allure.feature('聊天 API 端點')
@allure.story('Session API 應支援 RESTful 操作')
class TestSessionAPI:
    """測試 RESTful Session API — 對應 Rule: Session API 應支援 RESTful 操作"""

    @pytest.mark.asyncio
    @allure.title('POST /api/sessions 應建立新 session')
    async def test_create_session(self) -> None:
        """POST /api/sessions 應建立新 session。"""
        mock_session = _make_mock_session_manager()

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post('/api/sessions')

        assert response.status_code == 201
        data = response.json()
        assert 'session_id' in data
        assert len(data['session_id']) > 0

    @pytest.mark.asyncio
    @allure.title('GET /api/sessions 應回傳 session 摘要列表')
    async def test_list_sessions(self) -> None:
        """GET /api/sessions 應回傳 session 摘要列表。"""
        mock_session = _make_mock_session_manager()
        mock_session.list_sessions = AsyncMock(
            return_value=[
                {
                    'session_id': 's1',
                    'created_at': '2025-01-01 00:00:00',
                    'updated_at': '2025-01-01 00:01:00',
                    'message_count': 2,
                },
                {
                    'session_id': 's2',
                    'created_at': '2025-01-01 00:02:00',
                    'updated_at': '2025-01-01 00:03:00',
                    'message_count': 4,
                },
            ]
        )

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get('/api/sessions')

        assert response.status_code == 200
        data = response.json()
        assert 'sessions' in data
        assert len(data['sessions']) == 2

    @pytest.mark.asyncio
    @allure.title('GET /api/sessions/{id} 應回傳特定 session 的對話歷史')
    async def test_get_session_history(self) -> None:
        """GET /api/sessions/{id} 應回傳特定 session 的對話歷史。"""
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(
            return_value=[
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'},
            ]
        )

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get('/api/sessions/abc')

        assert response.status_code == 200
        data = response.json()
        assert 'messages' in data
        assert len(data['messages']) == 2
        mock_session.load.assert_called_once_with('abc')

    @pytest.mark.asyncio
    @allure.title('GET /api/sessions/{id} 不存在的 session 應回傳 404')
    async def test_get_nonexistent_session_returns_404(self) -> None:
        """GET /api/sessions/{id} 不存在的 session 應回傳 404。"""
        mock_session = _make_mock_session_manager()
        mock_session.load = AsyncMock(return_value=[])

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get('/api/sessions/not-exist')

        assert response.status_code == 404

    @pytest.mark.asyncio
    @allure.title('DELETE /api/sessions/{id} 應刪除特定 session')
    async def test_delete_session(self) -> None:
        """DELETE /api/sessions/{id} 應刪除特定 session。"""
        mock_session = _make_mock_session_manager()
        mock_session.delete_session = AsyncMock()

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.delete('/api/sessions/abc')

        assert response.status_code == 200
        mock_session.delete_session.assert_called_once_with('abc')


@allure.feature('聊天 API 端點')
@allure.story('工具執行時應推送 SSE 事件')
class TestFileEventStreaming:
    """測試檔案事件 SSE 推送 — 對應 Rule: 工具執行時應推送 SSE 事件"""

    @pytest.mark.asyncio
    @allure.title('_stream_chat 應推送工具回傳的 file_open 事件。')
    async def test_stream_chat_emits_file_open_event(self) -> None:
        """_stream_chat 應推送工具回傳的 file_open 事件。

        Given Agent 執行 read_file 工具
        When 工具回傳包含 sse_events 的結果
        Then SSE 串流應包含 file_open 事件
        """
        mock_session = _make_mock_session_manager()
        tokens = ['檔案內容是', '...']

        # 模擬 Agent 執行工具後的 conversation
        conversation_after_stream = [
            {'role': 'user', 'content': '讀取 main.py'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '好的'},
                    {
                        'type': 'tool_use',
                        'id': 'tool_1',
                        'name': 'read_file',
                        'input': {'path': 'main.py'},
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'tool_1',
                        'content': json.dumps(
                            {
                                'path': 'main.py',
                                'content': 'print("hello")',
                                'language': 'python',
                                'sse_events': [
                                    {
                                        'type': 'file_open',
                                        'data': {
                                            'path': 'main.py',
                                            'content': 'print("hello")',
                                            'language': 'python',
                                        },
                                    }
                                ],
                            }
                        ),
                    }
                ],
            },
            {'role': 'assistant', 'content': '檔案內容是...'},
        ]

        with (
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
        ):
            instance = MagicMock()

            # 模擬 stream_message 執行後更新 conversation
            async def _mock_stream_with_side_effect(msg: str, **kwargs: Any) -> AsyncIterator[str]:
                for token in tokens:
                    yield token
                # 執行後更新 conversation
                instance.conversation = conversation_after_stream

            instance.stream_message = _mock_stream_with_side_effect
            instance.conversation = []
            MockAgent.return_value = instance

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '讀取 main.py'},
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        events = parse_sse(response.text)

        # 驗證包含 file_open 事件
        file_open_events = [e for e in events if e['type'] == 'file_open']
        assert len(file_open_events) == 1
        assert file_open_events[0]['data']['path'] == 'main.py'
        assert file_open_events[0]['data']['content'] == 'print("hello")'

    @pytest.mark.asyncio
    @allure.title('_stream_chat 應推送工具回傳的 file_change 事件。')
    async def test_stream_chat_emits_file_change_event(self) -> None:
        """_stream_chat 應推送工具回傳的 file_change 事件。

        Given Agent 執行 edit_file 工具
        When 工具回傳包含 sse_events 的結果
        Then SSE 串流應包含 file_change 事件與 diff
        """
        mock_session = _make_mock_session_manager()
        tokens = ['已修改']

        diff_text = '--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old\n+new'

        conversation_after_stream = [
            {'role': 'user', 'content': '修改檔案'},
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'tool_use',
                        'id': 'tool_1',
                        'name': 'edit_file',
                        'input': {
                            'path': 'main.py',
                            'old_content': 'old',
                            'new_content': 'new',
                        },
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'tool_1',
                        'content': json.dumps(
                            {
                                'path': 'main.py',
                                'modified': True,
                                'sse_events': [
                                    {
                                        'type': 'file_change',
                                        'data': {
                                            'path': 'main.py',
                                            'diff': diff_text,
                                        },
                                    }
                                ],
                            }
                        ),
                    }
                ],
            },
            {'role': 'assistant', 'content': '已修改'},
        ]

        with (
            patch('agent_app.main.session_manager', mock_session),
            patch('agent_app.main.Agent') as MockAgent,
        ):
            instance = MagicMock()

            # 模擬 stream_message 執行後更新 conversation
            async def _mock_stream_with_side_effect(_msg: str, **kwargs: Any) -> AsyncIterator[str]:
                for token in tokens:
                    yield token
                instance.conversation = conversation_after_stream

            instance.stream_message = _mock_stream_with_side_effect
            instance.conversation = []
            MockAgent.return_value = instance

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.post(
                    STREAM_URL,
                    json={'message': '修改檔案'},
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        events = parse_sse(response.text)

        # 驗證包含 file_change 事件
        file_change_events = [e for e in events if e['type'] == 'file_change']
        assert len(file_change_events) == 1
        assert file_change_events[0]['data']['path'] == 'main.py'
        assert file_change_events[0]['data']['diff'] == diff_text


USAGE_URL = '/api/chat/usage'


@allure.feature('聊天 API 端點')
@allure.story('Usage 端點應回傳 context 區塊')
class TestChatUsageContext:
    """測試 /api/chat/usage 端點的 context 區塊。"""

    @pytest.mark.asyncio
    @allure.title('有使用記錄時應回傳 context 區塊')
    async def test_usage_returns_context_block_with_records(self) -> None:
        """有使用記錄時應回傳 context 區塊。"""
        mock_session = _make_mock_session_manager()
        # 模擬有一筆使用記錄
        mock_session.load_usage = AsyncMock(
            return_value=[
                {
                    'timestamp': '2025-01-01T00:00:00',
                    'input_tokens': 5000,
                    'output_tokens': 2000,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            ]
        )

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    USAGE_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        assert 'context' in data
        ctx = data['context']
        assert ctx['current_tokens'] == 7000
        assert ctx['context_window'] > 0
        assert ctx['usage_percent'] > 0

    @pytest.mark.asyncio
    @allure.title('無使用記錄時 context 區塊的 current_tokens 應為 0')
    async def test_usage_returns_context_block_empty(self) -> None:
        """無使用記錄時 context 區塊的 current_tokens 應為 0。"""
        mock_session = _make_mock_session_manager()
        mock_session.load_usage = AsyncMock(return_value=[])

        with patch('agent_app.main.session_manager', mock_session):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get(
                    USAGE_URL,
                    cookies=SESSION_COOKIE,
                )

        assert response.status_code == 200
        data = response.json()
        assert 'context' in data
        assert data['context']['current_tokens'] == 0
        assert data['context']['usage_percent'] == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]
