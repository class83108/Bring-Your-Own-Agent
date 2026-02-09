"""Session Memory Backend 測試模組。

根據 docs/features/session.feature 規格撰寫測試案例。
涵蓋：
- Rule: 記憶體後端應支援基本操作
"""

from __future__ import annotations

from typing import Any

import allure

from agent_core.session.memory_backend import MemorySessionBackend

# =============================================================================
# Rule: 記憶體後端應支援基本操作
# =============================================================================


@allure.feature('Session 後端抽象')
@allure.story('記憶體後端應支援基本操作')
class TestMemorySessionBackend:
    """MemorySessionBackend 測試。"""

    @allure.title('儲存並讀取對話歷史')
    async def test_save_and_load(self) -> None:
        """Scenario: 儲存並讀取對話歷史。"""
        backend = MemorySessionBackend()
        conversation: list[dict[str, Any]] = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'},
        ]

        await backend.save('abc', conversation)
        result = await backend.load('abc')

        assert result == conversation

    @allure.title('讀取不存在的 session')
    async def test_load_nonexistent_session(self) -> None:
        """Scenario: 讀取不存在的 session。"""
        backend = MemorySessionBackend()

        result = await backend.load('not-exist')

        assert result == []

    @allure.title('重設 session')
    async def test_reset_session(self) -> None:
        """Scenario: 重設 session。"""
        backend = MemorySessionBackend()
        await backend.save('abc', [{'role': 'user', 'content': 'Hello'}])

        await backend.reset('abc')

        result = await backend.load('abc')
        assert result == []

    @allure.title('儲存後修改原始列表不應影響已儲存的資料')
    async def test_save_does_not_share_reference(self) -> None:
        """儲存後修改原始列表不應影響已儲存的資料。"""
        backend = MemorySessionBackend()
        conversation: list[dict[str, Any]] = [
            {'role': 'user', 'content': 'Hello'},
        ]

        await backend.save('abc', conversation)
        conversation.append({'role': 'assistant', 'content': 'Hi'})

        result = await backend.load('abc')
        assert len(result) == 1

    @allure.title('不同 session 的對話應互相隔離')
    async def test_multiple_sessions_isolated(self) -> None:
        """不同 session 的對話應互相隔離。"""
        backend = MemorySessionBackend()
        await backend.save('session-1', [{'role': 'user', 'content': 'A'}])
        await backend.save('session-2', [{'role': 'user', 'content': 'B'}])

        result_1 = await backend.load('session-1')
        result_2 = await backend.load('session-2')

        assert result_1[0]['content'] == 'A'
        assert result_2[0]['content'] == 'B'

    @allure.title('MemorySessionBackend 應符合 SessionBackend Protocol')
    async def test_implements_session_backend_protocol(self) -> None:
        """MemorySessionBackend 應符合 SessionBackend Protocol。"""
        from agent_core.session.base import SessionBackend

        backend = MemorySessionBackend()
        assert isinstance(backend, SessionBackend)
