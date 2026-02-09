"""Compact（上下文壓縮）測試模組。

對應 docs/features/compact.feature 中定義的驗收規格。
TDD 紅燈先行：這些測試在 compact.py 實作前應全部 FAIL。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import allure

from agent_core.compact import (
    COMPACT_THRESHOLD,
    TRUNCATED_MARKER,
    compact_conversation,
    summarize_conversation,
    truncate_tool_results,
)
from agent_core.providers.base import FinalMessage, UsageInfo
from agent_core.token_counter import TokenCounter

# =============================================================================
# Helper: 建立測試用對話歷史
# =============================================================================


def _make_tool_use_assistant(tool_id: str, tool_name: str = 'read_file') -> dict[str, Any]:
    """建立含 tool_use 的 assistant 訊息。"""
    return {
        'role': 'assistant',
        'content': [
            {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {'path': 'test.py'}},
        ],
    }


def _make_tool_result_user(
    tool_id: str, result_content: str = '工具執行結果內容'
) -> dict[str, Any]:
    """建立含 tool_result 的 user 訊息。"""
    return {
        'role': 'user',
        'content': [
            {'type': 'tool_result', 'tool_use_id': tool_id, 'content': result_content},
        ],
    }


def _make_conversation_with_tools() -> list[dict[str, Any]]:
    """建立包含多輪工具調用的對話歷史。

    結構：
    [0] user: 使用者訊息
    [1] assistant: tool_use (tool_1)
    [2] user: tool_result (tool_1)  ← 舊的，應被截斷
    [3] assistant: 中間回應
    [4] user: 使用者訊息
    [5] assistant: tool_use (tool_2)
    [6] user: tool_result (tool_2)  ← 舊的，應被截斷
    [7] assistant: 中間回應
    [8] user: 使用者訊息
    [9] assistant: tool_use (tool_3)
    [10] user: tool_result (tool_3) ← 最近一輪，不應被截斷
    [11] assistant: 最終回應
    """
    return [
        {'role': 'user', 'content': '請讀取檔案'},
        _make_tool_use_assistant('tool_1'),
        _make_tool_result_user('tool_1', '第一個工具的結果，很長的內容' * 100),
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '已讀取第一個檔案'}]},
        {'role': 'user', 'content': '再讀取另一個'},
        _make_tool_use_assistant('tool_2'),
        _make_tool_result_user('tool_2', '第二個工具的結果，同樣很長' * 100),
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '已讀取第二個檔案'}]},
        {'role': 'user', 'content': '最後一個'},
        _make_tool_use_assistant('tool_3'),
        _make_tool_result_user('tool_3', '最新的工具結果'),
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '完成'}]},
    ]


# =============================================================================
# Rule: 應截斷舊的工具結果以釋放空間（Phase 1）
# =============================================================================


@allure.feature('上下文壓縮（Compact）')
@allure.story('應截斷舊的工具結果以釋放空間（Phase 1）')
class TestTruncateToolResults:
    """測試 Phase 1: tool_result 截斷。"""

    @allure.title('截斷舊的 tool_result 內容')
    def test_truncate_old_tool_results(self) -> None:
        """Scenario: 截斷舊的 tool_result 內容。"""
        conversation = _make_conversation_with_tools()

        truncated_count = truncate_tool_results(conversation, preserve_last_n_rounds=1)

        # 應截斷 tool_1 和 tool_2 的結果（共 2 個）
        assert truncated_count == 2

        # tool_1 的結果應被替換
        tool_result_1 = conversation[2]['content'][0]
        assert tool_result_1['content'] == TRUNCATED_MARKER

        # tool_2 的結果應被替換
        tool_result_2 = conversation[6]['content'][0]
        assert tool_result_2['content'] == TRUNCATED_MARKER

    @allure.title('保留最近一輪的 tool_result')
    def test_truncate_preserves_recent_tool_results(self) -> None:
        """Scenario: 保留最近一輪的 tool_result。"""
        conversation = _make_conversation_with_tools()

        truncate_tool_results(conversation, preserve_last_n_rounds=1)

        # tool_3（最近一輪）不應被截斷
        tool_result_3 = conversation[10]['content'][0]
        assert tool_result_3['content'] == '最新的工具結果'

    @allure.title('對應的 tool_use block 應保留不變')
    def test_truncate_preserves_tool_use_blocks(self) -> None:
        """對應的 tool_use block 應保留不變。"""
        conversation = _make_conversation_with_tools()

        truncate_tool_results(conversation, preserve_last_n_rounds=1)

        # tool_use block 不應被修改
        tool_use_1 = conversation[1]['content'][0]
        assert tool_use_1['type'] == 'tool_use'
        assert tool_use_1['id'] == 'tool_1'

    @allure.title('無 tool_result 時不變')
    def test_truncate_no_tool_results_unchanged(self) -> None:
        """Scenario: 無 tool_result 時不變。"""
        conversation = [
            {'role': 'user', 'content': '你好'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '你好！'}]},
        ]

        truncated_count = truncate_tool_results(conversation, preserve_last_n_rounds=1)

        assert truncated_count == 0
        assert conversation[0]['content'] == '你好'

    @allure.title('已經被截斷過的 tool_result 不應重複計算')
    def test_truncate_already_truncated_not_counted(self) -> None:
        """已經被截斷過的 tool_result 不應重複計算。"""
        conversation = _make_conversation_with_tools()

        # 第一次截斷
        first_count = truncate_tool_results(conversation, preserve_last_n_rounds=1)
        assert first_count == 2

        # 第二次截斷：已經截斷的不應再計算
        second_count = truncate_tool_results(conversation, preserve_last_n_rounds=1)
        assert second_count == 0


# =============================================================================
# Rule: 應用 LLM 摘要早期對話以進一步壓縮（Phase 2）
# =============================================================================


@allure.feature('上下文壓縮（Compact）')
@allure.story('應用 LLM 摘要早期對話以進一步壓縮（Phase 2）')
class TestSummarizeConversation:
    """測試 Phase 2: LLM 摘要。"""

    @allure.title('摘要早期對話')
    async def test_summarize_early_conversation(self) -> None:
        """Scenario: 摘要早期對話。"""
        conversation = [
            {'role': 'user', 'content': '第一輪問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '第一輪回答'}]},
            {'role': 'user', 'content': '第二輪問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '第二輪回答'}]},
            {'role': 'user', 'content': '第三輪問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '第三輪回答'}]},
            {'role': 'user', 'content': '最新問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '最新回答'}]},
        ]

        # Mock provider 的 create 方法
        mock_provider = AsyncMock()
        mock_provider.create.return_value = FinalMessage(
            content=[{'type': 'text', 'text': '使用者討論了三個主題...'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=100, output_tokens=50),
        )

        summary = await summarize_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            keep_last_n=4,
        )

        assert summary == '使用者討論了三個主題...'

        # 早期對話應被替換為摘要
        assert len(conversation) == 6  # 2（摘要）+ 4（保留）
        assert conversation[0]['role'] == 'user'
        assert '摘要' in conversation[0]['content']
        assert conversation[1]['role'] == 'assistant'

        # 最近的 4 則訊息應保��
        assert conversation[2]['content'] == '第三輪問題'
        expected = {'role': 'assistant', 'content': [{'type': 'text', 'text': '最新回答'}]}
        assert conversation[5] == expected

    @allure.title('保留最近的訊息不被摘要')
    async def test_summarize_preserves_recent_messages(self) -> None:
        """Scenario: 保留最近的訊息不被摘要。"""
        conversation = [
            {'role': 'user', 'content': '舊問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '舊回答'}]},
            {'role': 'user', 'content': '新問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '新回答'}]},
        ]

        mock_provider = AsyncMock()
        mock_provider.create.return_value = FinalMessage(
            content=[{'type': 'text', 'text': '先前討論摘要'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=50, output_tokens=30),
        )

        await summarize_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            keep_last_n=2,
        )

        # 最近 2 則訊息應保留原內容
        assert conversation[-2]['content'] == '新問題'
        assert conversation[-1]['content'] == [{'type': 'text', 'text': '新回答'}]

    @allure.title('訊息數不足時不應進行摘要')
    async def test_summarize_not_enough_messages_to_summarize(self) -> None:
        """訊息數不足時不應進行摘要。"""
        conversation = [
            {'role': 'user', 'content': '問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '回答'}]},
        ]

        mock_provider = AsyncMock()

        summary = await summarize_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            keep_last_n=4,
        )

        # 訊息不足，不應呼叫 provider
        assert summary is None
        mock_provider.create.assert_not_called()

    @allure.title('摘要切點應在完整 conversation round 邊界，不拆散 tool_use/tool_result 配對')
    async def test_summarize_respects_tool_use_boundaries(self) -> None:
        """摘要切點應在完整 conversation round 邊界，不拆散 tool_use/tool_result 配對。"""
        conversation = [
            {'role': 'user', 'content': '舊問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '舊回答'}]},
            {'role': 'user', 'content': '請使用工具'},
            _make_tool_use_assistant('tool_1'),
            _make_tool_result_user('tool_1', '工具結果'),
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '工具回應'}]},
            {'role': 'user', 'content': '最新問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '最新回答'}]},
        ]

        mock_provider = AsyncMock()
        mock_provider.create.return_value = FinalMessage(
            content=[{'type': 'text', 'text': '先前對話摘要'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=100, output_tokens=50),
        )

        await summarize_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            keep_last_n=2,
        )

        # 確認 tool_use/tool_result 配對沒有被拆散
        # 最近 2 則是 [6] 和 [7]，剩下的 [0]-[5] 被摘要
        assert conversation[-2]['content'] == '最新問題'
        assert conversation[-1]['content'] == [{'type': 'text', 'text': '最新回答'}]


# =============================================================================
# Rule: Compact 流程應按階段執行
# =============================================================================


@allure.feature('上下文壓縮（Compact）')
@allure.story('Compact 流程應按階段執行')
class TestCompactConversation:
    """測試完整 compact 流程。"""

    @allure.title('Phase 1 足夠時不觸發 Phase 2')
    async def test_compact_phase1_sufficient(self) -> None:
        """Scenario: Phase 1 足夠時不觸發 Phase 2。"""
        conversation = _make_conversation_with_tools()

        # 模擬 token_counter：截斷後低於閾值
        token_counter = TokenCounter(context_window=200_000)
        token_counter.update_from_usage(UsageInfo(input_tokens=170_000, output_tokens=0))

        mock_provider = AsyncMock()

        result = await compact_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            token_counter=token_counter,
        )

        assert result['truncated'] > 0
        assert result['summarized'] is False
        # Phase 2 不應被呼叫
        mock_provider.create.assert_not_called()

    @allure.title('Phase 1 不足時觸發 Phase 2')
    async def test_compact_full_pipeline(self) -> None:
        """Scenario: Phase 1 不足時觸發 Phase 2。"""
        # 建立較長的對話，Phase 1 截斷後仍然超過閾值
        conversation: list[dict[str, Any]] = [
            {'role': 'user', 'content': '早期問題 1'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '早期回答 1'}]},
            {'role': 'user', 'content': '早期問題 2'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '早期回答 2'}]},
            {'role': 'user', 'content': '最新問題'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '最新回答'}]},
        ]

        # 模擬 token_counter：始終超過閾值（Phase 1 無 tool_result 可截斷）
        token_counter = TokenCounter(context_window=200_000)
        token_counter.update_from_usage(UsageInfo(input_tokens=170_000, output_tokens=0))

        mock_provider = AsyncMock()
        mock_provider.create.return_value = FinalMessage(
            content=[{'type': 'text', 'text': '對話摘要內容'}],
            stop_reason='end_turn',
            usage=UsageInfo(input_tokens=100, output_tokens=50),
        )

        result = await compact_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            token_counter=token_counter,
        )

        assert result['truncated'] == 0  # 無 tool_result 可截斷
        assert result['summarized'] is True
        assert result['summary'] == '對話摘要內容'

    @allure.title('低於閾值時不應執行任何壓縮')
    async def test_compact_below_threshold_does_nothing(self) -> None:
        """低於閾值時不應執行任何壓縮。"""
        conversation = _make_conversation_with_tools()

        # 使用率低於閾值
        token_counter = TokenCounter(context_window=200_000)
        token_counter.update_from_usage(UsageInfo(input_tokens=50_000, output_tokens=0))

        mock_provider = AsyncMock()

        result = await compact_conversation(
            conversation=conversation,
            provider=mock_provider,
            system_prompt='你是助手',
            token_counter=token_counter,
        )

        assert result['truncated'] == 0
        assert result['summarized'] is False

    @allure.title('COMPACT_THRESHOLD 應為 80.0')
    def test_compact_threshold_value(self) -> None:
        """COMPACT_THRESHOLD 應為 80.0。"""
        assert COMPACT_THRESHOLD == 80.0

    @allure.title('TRUNCATED_MARKER 應為預期的壓縮標記')
    def test_truncated_marker_value(self) -> None:
        """TRUNCATED_MARKER 應為預期的壓縮標記。"""
        assert TRUNCATED_MARKER == '[已壓縮的工具結果]'
