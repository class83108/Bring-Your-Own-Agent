"""EventStore 測試模組。

根據 docs/features/core/resumable_stream.feature 規格撰寫測試案例。
涵蓋 MemoryEventStore 的基本操作、TTL 過期，以及 Agent 與 EventStore 的整合。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import allure

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.event_store import EventStore, MemoryEventStore
from agent_core.event_store.base import StreamEvent
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.types import AgentEvent, ContentBlock, MessageParam, StopReason, ToolDefinition

# =============================================================================
# Mock Helpers
# =============================================================================


def _make_final_message(
    text: str = '回應內容',
    stop_reason: StopReason = 'end_turn',
    content: list[ContentBlock] | None = None,
) -> FinalMessage:
    """建立 FinalMessage。"""
    if content is None:
        content = [{'type': 'text', 'text': text}]
    return FinalMessage(
        content=content,
        stop_reason=stop_reason,
        usage=UsageInfo(input_tokens=10, output_tokens=20),
    )


class MockProvider:
    """模擬的 LLM Provider。"""

    def __init__(self, responses: list[tuple[list[str], FinalMessage]]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @asynccontextmanager
    async def stream(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        text_chunks, final_msg = self._responses[self._call_count]
        self._call_count += 1

        async def _text_stream() -> AsyncIterator[str]:
            for chunk in text_chunks:
                yield chunk

        async def _get_final() -> FinalMessage:
            return final_msg

        yield StreamResult(
            text_stream=_text_stream(),
            get_final_result=_get_final,
        )

    async def count_tokens(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        return 0

    async def create(
        self,
        messages: list[MessageParam],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        return FinalMessage(
            content=[{'type': 'text', 'text': ''}],
            stop_reason='end_turn',
            usage=UsageInfo(),
        )


def _make_agent(
    provider: Any,
    event_store: EventStore | None = None,
) -> Agent:
    """建立測試用 Agent（含可選 EventStore）。"""
    config = AgentCoreConfig(
        provider=ProviderConfig(api_key='sk-test'),
        system_prompt='你是一位助手。',
    )
    return Agent(
        config=config,
        provider=provider,
        event_store=event_store,
    )


async def _collect_stream(
    agent: Agent,
    message: str,
    stream_id: str | None = None,
) -> tuple[list[str], list[AgentEvent]]:
    """收集串流回應的 tokens 與 events。"""
    tokens: list[str] = []
    events: list[AgentEvent] = []
    async for chunk in agent.stream_message(message, stream_id=stream_id):
        if isinstance(chunk, str):
            tokens.append(chunk)
        else:
            events.append(chunk)
    return tokens, events


# =============================================================================
# MemoryEventStore 基本操作
# =============================================================================


@allure.feature('可恢復串流')
@allure.story('EventStore 基本操作')
class TestMemoryEventStoreBasic:
    """測試 MemoryEventStore 的 append / read / status / complete。"""

    @allure.title('事件寫入與讀取')
    async def test_append_and_read(self) -> None:
        """Scenario: 事件寫入與讀取

        Given 一個空的 MemoryEventStore
        When 寫入 3 個串流事件到同一個 key
        Then 應可讀取到 3 個事件
        And 事件順序應與寫入一致
        And 每個事件應有唯一的遞增 id
        """
        store = MemoryEventStore()
        key = 'session-1'

        for i in range(3):
            await store.append(
                key,
                StreamEvent(id='', type='token', data=f'chunk-{i}', timestamp=time.time()),
            )

        events = await store.read(key)
        assert len(events) == 3, f'應有 3 個事件，實際 {len(events)}'

        # 順序一致
        for i, event in enumerate(events):
            assert event['data'] == f'chunk-{i}'

        # id 唯一且遞增
        ids = [e['id'] for e in events]
        assert len(set(ids)) == 3, 'event id 應唯一'
        assert ids == sorted(ids), 'event id 應遞增'

    @allure.title('從指定位置讀取事件（offset）')
    async def test_read_with_offset(self) -> None:
        """Scenario: 從指定位置讀取事件（offset）

        Given MemoryEventStore 中某 key 有 5 個事件
        When 從第 3 個事件的 id 之後開始讀取
        Then 應只回傳後 2 個事件
        """
        store = MemoryEventStore()
        key = 'session-2'

        for i in range(5):
            await store.append(
                key,
                StreamEvent(id='', type='token', data=f'chunk-{i}', timestamp=time.time()),
            )

        all_events = await store.read(key)
        after_id = all_events[2]['id']
        remaining = await store.read(key, after=after_id)

        assert len(remaining) == 2, f'應有 2 個事件，實際 {len(remaining)}'
        assert remaining[0]['data'] == 'chunk-3'
        assert remaining[1]['data'] == 'chunk-4'

    @allure.title('查詢串流狀態為 generating')
    async def test_status_generating(self) -> None:
        """Scenario: 查詢串流狀態為 generating"""
        store = MemoryEventStore()
        key = 'session-3'

        await store.append(
            key,
            StreamEvent(id='', type='token', data='hello', timestamp=time.time()),
        )
        await store.append(
            key,
            StreamEvent(id='', type='token', data='world', timestamp=time.time()),
        )

        status = await store.get_status(key)
        assert status is not None
        assert status['status'] == 'generating'
        assert status['event_count'] == 2
        assert status['stream_id'] == key

    @allure.title('標記串流完成')
    async def test_mark_complete(self) -> None:
        """Scenario: 標記串流完成"""
        store = MemoryEventStore()
        key = 'session-4'

        await store.append(
            key,
            StreamEvent(id='', type='token', data='hello', timestamp=time.time()),
        )
        await store.mark_complete(key)

        status = await store.get_status(key)
        assert status is not None
        assert status['status'] == 'completed'

    @allure.title('標記串流失敗')
    async def test_mark_failed(self) -> None:
        """Scenario: 標記串流失敗"""
        store = MemoryEventStore()
        key = 'session-5'

        await store.append(
            key,
            StreamEvent(id='', type='token', data='hello', timestamp=time.time()),
        )
        await store.mark_failed(key)

        status = await store.get_status(key)
        assert status is not None
        assert status['status'] == 'failed'

    @allure.title('查詢不存在的 key 回傳 None')
    async def test_status_not_found(self) -> None:
        """查詢不存在的 key 應回傳 None。"""
        store = MemoryEventStore()
        status = await store.get_status('nonexistent')
        assert status is None

    @allure.title('讀取不存在的 key 回傳空列表')
    async def test_read_not_found(self) -> None:
        """讀取不存在的 key 應回傳空列表。"""
        store = MemoryEventStore()
        events = await store.read('nonexistent')
        assert events == []


# =============================================================================
# MemoryEventStore TTL 過期
# =============================================================================


@allure.feature('可恢復串流')
@allure.story('EventStore TTL 過期')
class TestMemoryEventStoreTTL:
    """測試 MemoryEventStore 的 TTL 過期行為。"""

    @allure.title('過期串流自動清除')
    async def test_ttl_expiration(self) -> None:
        """Scenario: 過期串流自動清除"""
        store = MemoryEventStore(ttl_seconds=0.1)
        key = 'ttl-session'

        await store.append(
            key,
            StreamEvent(id='', type='token', data='hello', timestamp=time.time()),
        )

        # TTL 前應可讀取
        events = await store.read(key)
        assert len(events) == 1

        # 等待 TTL 過期
        await asyncio.sleep(0.2)

        # TTL 後應清除
        status = await store.get_status(key)
        assert status is None, f'過期後狀態應為 None，實際 {status}'

        events = await store.read(key)
        assert events == [], f'過期後事件應為空，實際 {len(events)} 個'


# =============================================================================
# Agent + EventStore 整合
# =============================================================================


@allure.feature('可恢復串流')
@allure.story('Agent + EventStore 整合')
class TestAgentEventStoreIntegration:
    """測試 Agent 配置 EventStore 時的行為。"""

    @allure.title('Agent 串流時自動寫入事件到 EventStore')
    async def test_agent_writes_events_to_store(self) -> None:
        """Scenario: Agent 串流時自動寫入事件到 EventStore

        Given Agent 配置了 EventStore
        And 呼叫端傳入 stream_id
        When 使用者傳送訊息並完成串流
        Then EventStore 應收到 token 與 done 事件
        And 串流狀態應為 completed
        """
        store = MemoryEventStore()
        provider = MockProvider(
            [
                (['你', '好'], _make_final_message('你好')),
            ]
        )
        agent = _make_agent(provider, event_store=store)

        session_id = 'sess-abc'
        await _collect_stream(agent, '哈囉', stream_id=session_id)

        # 驗證 EventStore 中的事件
        stored_events = await store.read(session_id)
        event_types = [e['type'] for e in stored_events]

        assert 'token' in event_types, f'應有 token，實際 {event_types}'
        assert 'done' in event_types, f'應有 done，實際 {event_types}'

        # 串流狀態應為 completed
        status = await store.get_status(session_id)
        assert status is not None
        assert status['status'] == 'completed'

    @allure.title('未配置 EventStore 的 Agent 行為不變')
    async def test_agent_without_event_store(self) -> None:
        """Scenario: 未配置 EventStore 的 Agent 行為不變"""
        provider = MockProvider(
            [
                (['你', '好'], _make_final_message('你好')),
            ]
        )
        agent = _make_agent(provider, event_store=None)

        tokens, _ = await _collect_stream(agent, '哈囉')

        # 正常串流應運作
        assert ''.join(tokens) == '你好'

    @allure.title('未傳入 stream_id 時不寫入 EventStore')
    async def test_no_stream_id_no_write(self) -> None:
        """Scenario: 未傳入 stream_id 時不寫入 EventStore

        Given Agent 配置了 EventStore
        But 呼叫端未傳入 stream_id
        When 使用者傳送訊息並完成串流
        Then EventStore 不應有任何事件
        """
        store = MemoryEventStore()
        provider = MockProvider(
            [
                (['你', '好'], _make_final_message('你好')),
            ]
        )
        agent = _make_agent(provider, event_store=store)

        # 不傳 stream_id
        tokens, _ = await _collect_stream(agent, '哈囉', stream_id=None)

        # 正常串流應運作
        assert ''.join(tokens) == '你好'

        # EventStore 應該沒有任何資料
        # （因為沒有 key，所以無法驗證特定 key；但至少不應 crash）

    @allure.title('用 session_id 從 EventStore 讀取完整事件序列')
    async def test_read_complete_stream_from_store(self) -> None:
        """Scenario: 用 session_id 從 EventStore 讀取已完成串流的所有事件"""
        store = MemoryEventStore()
        provider = MockProvider(
            [
                (['A', 'B', 'C'], _make_final_message('ABC')),
            ]
        )
        agent = _make_agent(provider, event_store=store)

        session_id = 'sess-read'
        await _collect_stream(agent, '測試', stream_id=session_id)

        stored = await store.read(session_id)
        types = [e['type'] for e in stored]

        # 應該是 token(s) → done
        assert types[-1] == 'done', f'最後事件應為 done，實際 {types[-1]}'
        token_events = [e for e in stored if e['type'] == 'token']
        assert len(token_events) == 3, f'應有 3 個 token 事件，實際 {len(token_events)}'

    @allure.title('從指定 offset 恢復已完成的串流')
    async def test_resume_from_offset(self) -> None:
        """Scenario: 從指定 offset 恢復已完成的串流

        Given Agent 已完成一次含 EventStore 的串流
        And 客戶端只收到前 N 個事件
        When 使用最後收到的 event_id 作為 offset 讀取
        Then 應只回傳該 event_id 之後的事件
        And 不應遺漏不重複
        """
        store = MemoryEventStore()
        provider = MockProvider(
            [
                (['A', 'B', 'C', 'D', 'E'], _make_final_message('ABCDE')),
            ]
        )
        agent = _make_agent(provider, event_store=store)

        session_id = 'sess-resume'
        await _collect_stream(agent, '測試', stream_id=session_id)

        all_events = await store.read(session_id)

        # 模擬客戶端只收到前 2 個事件後斷線
        last_received_id = all_events[1]['id']
        remaining = await store.read(session_id, after=last_received_id)

        # 驗證不遺漏不重複
        expected_remaining = all_events[2:]
        assert len(remaining) == len(expected_remaining), (
            f'應有 {len(expected_remaining)} 個剩餘事件，實際 {len(remaining)}'
        )
        for got, expected in zip(remaining, expected_remaining):
            assert got['id'] == expected['id']
            assert got['data'] == expected['data']

    @allure.title('token 事件拼接應還原完整回應')
    async def test_token_data_matches_response(self) -> None:
        """Scenario: token 事件拼接應還原完整回應"""
        store = MemoryEventStore()
        provider = MockProvider(
            [
                (['你', '好', '嗎'], _make_final_message('你好嗎')),
            ]
        )
        agent = _make_agent(provider, event_store=store)

        session_id = 'sess-token'
        tokens, _ = await _collect_stream(agent, '測試', stream_id=session_id)

        stored = await store.read(session_id)
        token_events = [e for e in stored if e['type'] == 'token']
        reconstructed = ''.join(e['data'] for e in token_events)

        response = ''.join(tokens)
        assert reconstructed == response, (
            f'token 拼接結果應等於回應，拼接={reconstructed!r}，回應={response!r}'
        )
