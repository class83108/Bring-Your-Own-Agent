"""SQLite Session 後端。

使用 Python 標準庫 sqlite3 持久化對話歷史與使用量統計，零外部依賴。
Server 重啟後對話自動恢復。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from agent_core.usage_monitor import UsageRecord

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = 'sessions.db'


class SQLiteSessionBackend:
    """SQLite Session 後端。

    將對話歷史與使用量統計以 JSON 序列化後存入 SQLite 資料庫。
    支援跨程序持久化，適合單機部署場景。
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """初始化 SQLite 後端。

        Args:
            db_path: 資料庫檔案路徑，預設為 sessions.db
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._create_table()
        logger.info('SQLite Session 後端已初始化', extra={'db_path': db_path})

    def _create_table(self) -> None:
        """建立資料表（若不存在）。"""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                conversation TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                session_id TEXT PRIMARY KEY,
                usage_data TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """讀取對話歷史。

        Args:
            session_id: 會話識別符

        Returns:
            對話歷史列表，若無記錄則回傳空列表
        """
        cursor = self._conn.execute(
            'SELECT conversation FROM sessions WHERE session_id = ?',
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return []

        data: list[dict[str, Any]] = json.loads(row[0])
        logger.debug(
            '讀取會話歷史（SQLite）',
            extra={'session_id': session_id, 'messages': len(data)},
        )
        return data

    async def save(self, session_id: str, conversation: list[dict[str, Any]]) -> None:
        """儲存對話歷史。

        使用 UPSERT 語法：存在則更新，不存在則新增。

        Args:
            session_id: 會話識別符
            conversation: 對話歷史列表
        """
        serialized = json.dumps(conversation, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO sessions (session_id, conversation)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                conversation = excluded.conversation,
                updated_at = datetime('now')
            """,
            (session_id, serialized),
        )
        self._conn.commit()
        logger.debug(
            '儲存會話歷史（SQLite）',
            extra={'session_id': session_id, 'messages': len(conversation)},
        )

    async def reset(self, session_id: str) -> None:
        """清除對話歷史。

        Args:
            session_id: 會話識別符
        """
        self._conn.execute(
            'DELETE FROM sessions WHERE session_id = ?',
            (session_id,),
        )
        self._conn.commit()
        logger.debug('會話歷史已清除（SQLite）', extra={'session_id': session_id})

    # =========================================================================
    # 使用量統計
    # =========================================================================

    async def load_usage(self, session_id: str) -> list[dict[str, Any]]:
        """讀取使用量統計記錄。

        Args:
            session_id: 會話識別符

        Returns:
            使用量記錄列表，若無記錄則回傳空列表
        """
        cursor = self._conn.execute(
            'SELECT usage_data FROM usage WHERE session_id = ?',
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return []

        data: list[dict[str, Any]] = json.loads(row[0])
        logger.debug(
            '讀取使用量記錄（SQLite）',
            extra={'session_id': session_id, 'records': len(data)},
        )
        return data

    async def save_usage(self, session_id: str, records: list[UsageRecord]) -> None:
        """儲存使用量統計記錄。

        Args:
            session_id: 會話識別符
            records: 使用量記錄列表
        """
        data = [r.to_dict() for r in records]
        serialized = json.dumps(data, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO usage (session_id, usage_data)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                usage_data = excluded.usage_data,
                updated_at = datetime('now')
            """,
            (session_id, serialized),
        )
        self._conn.commit()
        logger.debug(
            '儲存使用量記錄（SQLite）',
            extra={'session_id': session_id, 'records': len(records)},
        )

    async def reset_usage(self, session_id: str) -> None:
        """清除使用量統計。

        Args:
            session_id: 會話識別符
        """
        self._conn.execute(
            'DELETE FROM usage WHERE session_id = ?',
            (session_id,),
        )
        self._conn.commit()
        logger.debug('使用量統計已清除（SQLite）', extra={'session_id': session_id})

    # =========================================================================
    # Session 管理
    # =========================================================================

    async def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有 session 摘要。

        Returns:
            session 摘要列表，每筆包含 session_id、created_at、updated_at、message_count
        """
        cursor = self._conn.execute(
            'SELECT session_id, conversation, created_at, updated_at FROM sessions'
        )
        sessions: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            conversation: list[Any] = json.loads(row[1])
            sessions.append(
                {
                    'session_id': row[0],
                    'created_at': row[2],
                    'updated_at': row[3],
                    'message_count': len(conversation),
                }
            )
        return sessions

    async def delete_session(self, session_id: str) -> None:
        """刪除 session（同時清除對話與使用量）。

        Args:
            session_id: 會話識別符
        """
        self._conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        self._conn.execute('DELETE FROM usage WHERE session_id = ?', (session_id,))
        self._conn.commit()
        logger.debug('Session 已刪除（SQLite）', extra={'session_id': session_id})

    async def close(self) -> None:
        """關閉 SQLite 連線。"""
        self._conn.close()
        logger.info('SQLite Session 後端已關閉')
