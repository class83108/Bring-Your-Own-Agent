"""SQLite Session Backend 測試模組。

根據 docs/features/session.feature 規格撰寫測試案例。
涵蓋：
- Rule: SQLite 後端應支援基本 CRUD 操作
- Rule: SQLite 後端應跨程序存活
- Rule: SQLite 後端應支援多 session 隔離
- Rule: SQLite 後端應正確序列化複雜訊息結構
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_core.session.sqlite_backend import SQLiteSessionBackend

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """回傳暫存目錄中的資料庫檔案路徑。"""
    return tmp_path / 'test_session.db'


@pytest.fixture
async def backend(db_path: Path) -> SQLiteSessionBackend:
    """建立 SQLiteSessionBackend 實例。"""
    return SQLiteSessionBackend(db_path=str(db_path))


def _sample_conversation() -> list[dict[str, Any]]:
    """建立範例對話歷史。"""
    return [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there'},
    ]


# =============================================================================
# Rule: SQLite 後端應支援基本 CRUD 操作
# =============================================================================


class TestSQLiteBasicCRUD:
    """測試 SQLite 後端基本 CRUD 操作。"""

    async def test_save_and_load(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 儲存並讀取對話歷史。"""
        conversation = _sample_conversation()

        await backend.save('abc', conversation)
        result = await backend.load('abc')

        assert result == conversation

    async def test_load_nonexistent_session(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 讀取不存在的 session。"""
        result = await backend.load('not-exist')

        assert result == []

    async def test_reset_session(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 重設 session。"""
        await backend.save('abc', _sample_conversation())

        await backend.reset('abc')

        result = await backend.load('abc')
        assert result == []

    async def test_save_overwrites_existing(self, backend: SQLiteSessionBackend) -> None:
        """覆寫已存在的 session 資料。"""
        await backend.save('abc', [{'role': 'user', 'content': 'first'}])
        await backend.save('abc', [{'role': 'user', 'content': 'second'}])

        result = await backend.load('abc')

        assert len(result) == 1
        assert result[0]['content'] == 'second'


# =============================================================================
# Rule: SQLite 後端應跨程序存活
# =============================================================================


class TestSQLitePersistence:
    """測試 SQLite 後端跨程序持久化。"""

    async def test_data_survives_reopen(self, db_path: Path) -> None:
        """Scenario: 關閉後重新開啟仍保留資料。"""
        conversation = _sample_conversation()

        # 第一次：寫入
        backend1 = SQLiteSessionBackend(db_path=str(db_path))
        await backend1.save('abc', conversation)

        # 第二次：用同一路徑建立新實例，模擬重啟
        backend2 = SQLiteSessionBackend(db_path=str(db_path))
        result = await backend2.load('abc')

        assert result == conversation


# =============================================================================
# Rule: SQLite 後端應支援多 session 隔離
# =============================================================================


class TestSQLiteSessionIsolation:
    """測試多 session 隔離。"""

    async def test_different_sessions_isolated(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 不同 session 的資料互不影響。"""
        await backend.save('s1', [{'role': 'user', 'content': 'A'}])
        await backend.save('s2', [{'role': 'user', 'content': 'B'}])

        result_1 = await backend.load('s1')
        result_2 = await backend.load('s2')

        assert result_1[0]['content'] == 'A'
        assert result_2[0]['content'] == 'B'

    async def test_reset_one_session_preserves_others(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 重設單一 session 不影響其他。"""
        await backend.save('s1', [{'role': 'user', 'content': 'A'}])
        await backend.save('s2', [{'role': 'user', 'content': 'B'}])

        await backend.reset('s1')

        assert await backend.load('s1') == []
        assert len(await backend.load('s2')) == 1


# =============================================================================
# Rule: SQLite 後端應正確序列化複雜訊息結構
# =============================================================================


class TestSQLiteComplexMessages:
    """測試複雜訊息結構的序列化。"""

    async def test_tool_use_and_tool_result_preserved(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 儲存包含 tool_use 與 tool_result 的對話。"""
        conversation: list[dict[str, Any]] = [
            {'role': 'user', 'content': '請讀取 main.py'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': '讓我讀取檔案'},
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
                        'content': 'print("hello")',
                        'is_error': False,
                    },
                ],
            },
            {'role': 'assistant', 'content': '檔案內容如上。'},
        ]

        await backend.save('abc', conversation)
        result = await backend.load('abc')

        assert result == conversation

        # 驗證 tool_use 區塊完整
        tool_use_block = result[1]['content'][1]
        assert tool_use_block['type'] == 'tool_use'
        assert tool_use_block['id'] == 'tool_1'
        assert tool_use_block['name'] == 'read_file'
        assert tool_use_block['input'] == {'path': 'main.py'}

        # 驗證 tool_result 區塊完整
        tool_result_block = result[2]['content'][0]
        assert tool_result_block['type'] == 'tool_result'
        assert tool_result_block['tool_use_id'] == 'tool_1'
        assert tool_result_block['is_error'] is False


# =============================================================================
# Rule: SQLite 後端應支援列出與刪除 session
# =============================================================================


class TestSQLiteSessionManagement:
    """測試 session 列出與刪除。"""

    async def test_list_sessions_returns_summaries(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 列出所有 session。"""
        await backend.save('s1', [{'role': 'user', 'content': 'A'}])
        await backend.save(
            's2',
            [
                {'role': 'user', 'content': 'B'},
                {'role': 'assistant', 'content': 'C'},
            ],
        )

        sessions = await backend.list_sessions()

        assert len(sessions) == 2
        ids = {s['session_id'] for s in sessions}
        assert ids == {'s1', 's2'}

        # 驗證每筆摘要包含必要欄位
        for s in sessions:
            assert 'session_id' in s
            assert 'created_at' in s
            assert 'updated_at' in s
            assert 'message_count' in s

        # 驗證 message_count 正確
        s1_summary = next(s for s in sessions if s['session_id'] == 's1')
        s2_summary = next(s for s in sessions if s['session_id'] == 's2')
        assert s1_summary['message_count'] == 1
        assert s2_summary['message_count'] == 2

    async def test_list_sessions_empty(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 無 session 時列出回傳空列表。"""
        sessions = await backend.list_sessions()

        assert sessions == []

    async def test_delete_session_clears_conversation_and_usage(
        self, backend: SQLiteSessionBackend
    ) -> None:
        """Scenario: 刪除 session 同時清除對話與使用量。"""
        from datetime import datetime

        from agent_core.usage_monitor import UsageRecord

        await backend.save('abc', [{'role': 'user', 'content': 'Hello'}])
        await backend.save_usage(
            'abc',
            [
                UsageRecord(timestamp=datetime.now(), input_tokens=100, output_tokens=50),
            ],
        )

        await backend.delete_session('abc')

        assert await backend.load('abc') == []
        assert await backend.load_usage('abc') == []

        # 列出時不應包含已刪除的 session
        sessions = await backend.list_sessions()
        assert all(s['session_id'] != 'abc' for s in sessions)

    async def test_delete_nonexistent_session_no_error(self, backend: SQLiteSessionBackend) -> None:
        """Scenario: 刪除不存在的 session 不應報錯。"""
        # 不應拋出例外
        await backend.delete_session('not-exist')


class TestSQLiteProtocolCompliance:
    """測試 SQLiteSessionBackend 符合 SessionBackend Protocol。"""

    async def test_implements_session_backend_protocol(self, backend: SQLiteSessionBackend) -> None:
        """SQLiteSessionBackend 應符合 SessionBackend Protocol。"""
        from agent_core.session.base import SessionBackend

        assert isinstance(backend, SessionBackend)
