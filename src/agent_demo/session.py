"""會話管理模組。

使用 Redis 持久化對話歷史，支援會話的讀取、寫入與清除。
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

import redis.asyncio as redis
from anthropic.types import MessageParam

logger = logging.getLogger(__name__)

# 會話存活時間：24 小時
SESSION_TTL = 86400

# Redis key 模板
_KEY_TEMPLATE = 'session:{session_id}:conversation'


class SessionManager:
    """Redis 會話管理器。

    負責將對話歷史序列化存入 Redis，並在請求時讀取。

    Attributes:
        _redis: Redis 異步連接
    """

    def __init__(self, redis_url: str = 'redis://localhost:6381') -> None:
        """初始化會話管理器。

        Args:
            redis_url: Redis 連接 URL
        """
        # 解析 URL 並直接建構 Redis，避免 from_url 的 stub 類型未知問題
        parsed = urlparse(redis_url)
        self._redis = redis.Redis(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 6381,
            db=int(parsed.path.lstrip('/') or 0),
            password=parsed.password,
            decode_responses=True,
        )

    def _key(self, session_id: str) -> str:
        """生成對話歷史的 Redis key。"""
        return _KEY_TEMPLATE.format(session_id=session_id)

    async def load(self, session_id: str) -> list[MessageParam]:
        """從 Redis 讀取對話歷史。

        Args:
            session_id: 會話識別符

        Returns:
            對話歷史列表，若無記錄則傳回空列表
        """
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            logger.debug('會話無歷史記錄', extra={'session_id': session_id})
            return []

        data: list[MessageParam] = json.loads(raw)
        logger.debug('讀取會話歷史', extra={'session_id': session_id, 'messages': len(data)})
        return data

    async def save(self, session_id: str, conversation: list[MessageParam]) -> None:
        """將對話歷史寫入 Redis。

        Args:
            session_id: 會話識別符
            conversation: 對話歷史列表
        """
        key = self._key(session_id)
        await self._redis.setex(key, SESSION_TTL, json.dumps(conversation))
        logger.debug(
            '儲存會話歷史',
            extra={'session_id': session_id, 'messages': len(conversation)},
        )

    async def reset(self, session_id: str) -> None:
        """清除會話歷史。

        Args:
            session_id: 會話識別符
        """
        await self._redis.delete(self._key(session_id))
        logger.debug('會話歷史已清除', extra={'session_id': session_id})

    async def close(self) -> None:
        """關閉 Redis 連接。"""
        await self._redis.close()
