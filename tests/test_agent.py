"""Agent 測試模組。

根據 docs/features/chat.feature 規格撰寫測試案例。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_demo.agent import Agent, AgentConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """建立模擬的 Anthropic 客戶端。"""
    return MagicMock()


@pytest.fixture
def agent(mock_client: MagicMock) -> Agent:
    """建立測試用 Agent，使用模擬客戶端。"""
    config = AgentConfig(system_prompt='你是一位專業的程式開發助手。')
    return Agent(config=config, client=mock_client)


def create_mock_stream(text_chunks: list[str]) -> MagicMock:
    """建立模擬的串流回應。

    Args:
        text_chunks: 要逐步回傳的文字片段列表
    """

    async def mock_text_stream() -> AsyncIterator[str]:
        for chunk in text_chunks:
            yield chunk

    stream_context = MagicMock()
    stream_context.__aenter__ = AsyncMock(return_value=stream_context)
    stream_context.__aexit__ = AsyncMock(return_value=None)
    stream_context.text_stream = mock_text_stream()

    return stream_context


async def collect_stream(agent: Agent, message: str) -> str:
    """收集串流回應並返回完整文字。"""
    chunks: list[str] = []
    async for chunk in agent.stream_message(message):
        chunks.append(chunk)
    return ''.join(chunks)


# =============================================================================
# Rule: Agent 應驗證使用者輸入
# =============================================================================


class TestInputValidation:
    """測試使用者輸入驗證。"""

    async def test_empty_message_raises_value_error(self, agent: Agent) -> None:
        """Scenario: 使用者發送空白訊息。

        When 使用者輸入空白訊息
        Then Agent 應拋出 ValueError
        And 錯誤訊息應提示使用者輸入有效內容
        """
        with pytest.raises(ValueError, match='空白|有效'):
            async for _ in agent.stream_message(''):
                pass

    async def test_whitespace_only_raises_value_error(self, agent: Agent) -> None:
        """測試只有空白字元的訊息也應拋出 ValueError。"""
        with pytest.raises(ValueError, match='空白|有效'):
            async for _ in agent.stream_message('   '):
                pass

        with pytest.raises(ValueError, match='空白|有效'):
            async for _ in agent.stream_message('\n\t  '):
                pass


# =============================================================================
# Rule: Agent 應維護對話歷史
# =============================================================================


class TestConversationHistory:
    """測試對話歷史維護。"""

    async def test_single_turn_conversation_history(
        self, agent: Agent, mock_client: MagicMock
    ) -> None:
        """Scenario: 單輪對話後歷史正確記錄。

        When 使用者發送一則訊息
        And Agent 回應完成
        Then 對話歷史應包含一組 user 和 assistant 訊息
        """
        # Arrange
        mock_client.messages.stream.return_value = create_mock_stream(['回應內容'])

        # Act
        await collect_stream(agent, '測試訊息')

        # Assert
        assert len(agent.conversation) == 2
        assert agent.conversation[0]['role'] == 'user'
        assert agent.conversation[0]['content'] == '測試訊息'
        assert agent.conversation[1]['role'] == 'assistant'
        assert agent.conversation[1]['content'] == '回應內容'

    async def test_multi_turn_conversation_history(
        self, agent: Agent, mock_client: MagicMock
    ) -> None:
        """Scenario: 多輪對話後歷史正確累積。

        Given 使用者已完成第一輪對話
        When 使用者發送第二則訊息
        And Agent 回應完成
        Then 對話歷史應包含兩組 user 和 assistant 訊息
        """
        # Arrange & Act - 第一輪對話
        mock_client.messages.stream.return_value = create_mock_stream(['第一次回應'])
        await collect_stream(agent, '第一則訊息')

        # Arrange & Act - 第二輪對話
        mock_client.messages.stream.return_value = create_mock_stream(['第二次回應'])
        await collect_stream(agent, '第二則訊息')

        # Assert
        assert len(agent.conversation) == 4
        assert agent.conversation[0]['content'] == '第一則訊息'
        assert agent.conversation[1]['content'] == '第一次回應'
        assert agent.conversation[2]['content'] == '第二則訊息'
        assert agent.conversation[3]['content'] == '第二次回應'

    def test_reset_conversation(self, agent: Agent) -> None:
        """Scenario: 重設對話歷史。

        Given 使用者已進行過對話
        When 呼叫重設對話功能
        Then 對話歷史應為空
        """
        # Arrange
        agent.conversation = [
            {'role': 'user', 'content': 'test'},
            {'role': 'assistant', 'content': 'response'},
        ]

        # Act
        agent.reset_conversation()

        # Assert
        assert len(agent.conversation) == 0


# =============================================================================
# Rule: Agent 應正確處理錯誤情況
# =============================================================================


class TestErrorHandling:
    """測試錯誤處理。"""

    async def test_api_connection_error(self, agent: Agent, mock_client: MagicMock) -> None:
        """Scenario: API 連線失敗。

        Given API 服務無法連線
        When 使用者發送訊息
        Then Agent 應拋出 ConnectionError
        And 錯誤訊息應建議使用者稍後重試
        And 對話歷史不應被修改
        """
        from anthropic import APIConnectionError

        # Arrange
        initial_length = len(agent.conversation)
        mock_client.messages.stream.side_effect = APIConnectionError(request=MagicMock())

        # Act & Assert
        with pytest.raises(ConnectionError) as exc_info:
            async for _ in agent.stream_message('測試訊息'):
                pass

        assert '連線' in str(exc_info.value) or '重試' in str(exc_info.value)
        assert len(agent.conversation) == initial_length

    async def test_api_auth_error(self, agent: Agent, mock_client: MagicMock) -> None:
        """Scenario: API 金鑰無效。

        Given API 金鑰設定錯誤
        When 使用者發送訊息
        Then Agent 應拋出 PermissionError
        And 錯誤訊息應說明如何設定 API 金鑰
        And 對話歷史不應被修改
        """
        from anthropic import AuthenticationError

        # Arrange
        initial_length = len(agent.conversation)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.messages.stream.side_effect = AuthenticationError(
            message='Invalid API key',
            response=mock_response,
            body={'error': {'message': 'Invalid API key'}},
        )

        # Act & Assert
        with pytest.raises(PermissionError) as exc_info:
            async for _ in agent.stream_message('測試訊息'):
                pass

        error_msg = str(exc_info.value)
        assert 'API' in error_msg or '金鑰' in error_msg
        assert len(agent.conversation) == initial_length

    async def test_api_timeout_error(self, agent: Agent, mock_client: MagicMock) -> None:
        """Scenario: API 回應超時。

        Given API 回應超過超時閾值
        When 使用者發送訊息
        Then Agent 應拋出 TimeoutError
        And 對話歷史不應被修改
        """
        import anthropic

        # Arrange
        initial_length = len(agent.conversation)
        mock_client.messages.stream.side_effect = anthropic.APITimeoutError(request=MagicMock())

        # Act & Assert
        with pytest.raises(TimeoutError):
            async for _ in agent.stream_message('測試訊息'):
                pass

        assert len(agent.conversation) == initial_length


# =============================================================================
# Rule: Agent 應支援串流回應
# =============================================================================


class TestStreamingResponse:
    """測試串流回應功能。"""

    async def test_stream_collects_chunks_into_conversation(
        self, agent: Agent, mock_client: MagicMock
    ) -> None:
        """Scenario: 串流方式逐步回傳 token。

        When 使用者發送訊息
        Then Agent 應以 AsyncIterator 逐步 yield 回應 token
        And 所有 token 組合後應為完整回應
        """
        # Arrange
        chunks = ['這是', '一個', '串流', '回應']
        mock_client.messages.stream.return_value = create_mock_stream(chunks)

        # Act
        async for _ in agent.stream_message('測試'):
            pass

        # Assert - 串流完成後，歷史中儲存的是組合後的完整回應
        assert agent.conversation[1]['content'] == '這是一個串流回應'

    async def test_stream_interruption_preserves_partial_response(
        self, agent: Agent, mock_client: MagicMock
    ) -> None:
        """Scenario: 串流中斷時保留部分回應。

        Given Agent 正在串流回應
        And 已收到部分 token
        When 串流連線意外中斷
        Then Agent 應將已收到的部分回應存入對話歷史
        And Agent 應拋出 ConnectionError 提示中斷
        """
        from anthropic import APIConnectionError

        # Arrange - 模擬部分串流後中斷
        async def partial_stream() -> AsyncIterator[str]:
            yield '這是部分'
            yield '回應'
            raise APIConnectionError(request=MagicMock())

        stream_context = MagicMock()
        stream_context.__aenter__ = AsyncMock(return_value=stream_context)
        stream_context.__aexit__ = AsyncMock(return_value=None)
        stream_context.text_stream = partial_stream()
        mock_client.messages.stream.return_value = stream_context

        # Act
        received: list[str] = []
        with pytest.raises(ConnectionError) as exc_info:
            async for chunk in agent.stream_message('測試'):
                received.append(chunk)

        # Assert - 應收到部分回應且被保留
        assert received == ['這是部分', '回應']
        assert '中斷' in str(exc_info.value)
        # 部分回應應被保留在對話歷史中
        assert len(agent.conversation) == 2
        assert agent.conversation[1]['content'] == '這是部分回應'
