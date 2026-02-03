"""Agent 測試模組。

根據 docs/features/chat.feature 與 agent_core.feature 規格撰寫測試案例。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_demo.agent import Agent, AgentConfig
from agent_demo.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# =============================================================================
# Mock Helpers
# =============================================================================


def _make_text_block(text: str) -> MagicMock:
    """建立模擬的 TextBlock。"""
    block = MagicMock()
    block.type = 'text'
    block.text = text
    # 配置 model_dump() 返回字典格式（用於序列化）
    block.model_dump.return_value = {'type': 'text', 'text': text}
    return block


def _make_tool_use_block(tool_id: str, name: str, input_data: dict[str, Any]) -> MagicMock:
    """建立模擬的 ToolUseBlock。"""
    block = MagicMock()
    block.type = 'tool_use'
    block.id = tool_id
    block.name = name
    block.input = input_data
    # 配置 model_dump() 返回字典格式（用於序列化）
    block.model_dump.return_value = {
        'type': 'tool_use',
        'id': tool_id,
        'name': name,
        'input': input_data,
    }
    return block


def _make_final_message(
    content_blocks: list[MagicMock],
    stop_reason: str = 'end_turn',
) -> MagicMock:
    """建立模擬的 Final Message。"""
    message = MagicMock()
    message.content = content_blocks
    message.stop_reason = stop_reason
    return message


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


def create_mock_stream(
    text_chunks: list[str],
    stop_reason: str = 'end_turn',
    content_blocks: list[MagicMock] | None = None,
) -> MagicMock:
    """建立模擬的串流回應。

    Args:
        text_chunks: 要逐步回傳的文字片段列表
        stop_reason: 停止原因（預設 'end_turn'）
        content_blocks: 自訂 content blocks（預設根據 text_chunks 建立 TextBlock）
    """

    async def mock_text_stream() -> AsyncIterator[str]:
        for chunk in text_chunks:
            yield chunk

    # 建立 final message
    if content_blocks is None:
        full_text = ''.join(text_chunks)
        content_blocks = [_make_text_block(full_text)]

    final_message = _make_final_message(content_blocks, stop_reason)

    stream_context = MagicMock()
    stream_context.__aenter__ = AsyncMock(return_value=stream_context)
    stream_context.__aexit__ = AsyncMock(return_value=None)
    stream_context.text_stream = mock_text_stream()
    stream_context.get_final_message = AsyncMock(return_value=final_message)

    return stream_context


def _get_assistant_text(conversation_entry: Any) -> str:
    """從對話歷史中的 assistant 條目取得文字內容。

    支援 content 為字串或 content blocks 列表（字典或對象格式）。
    """
    content = conversation_entry['content']
    if isinstance(content, str):
        return content
    # content blocks 列表（來自 final_message.content）
    texts: list[str] = []
    for block in content:
        # 支援字典格式（序列化後）和對象格式（mock）
        if isinstance(block, dict):
            if block.get('type') == 'text':  # type: ignore[reportUnknownMemberType]
                texts.append(block['text'])  # type: ignore[reportUnknownArgumentType]
        elif hasattr(block, 'type') and block.type == 'text':
            texts.append(block.text)
    return ''.join(texts)


async def collect_stream(agent: Agent, message: str) -> str:
    """收集串流回應並返回完整文字（忽略事件）。"""
    chunks: list[str] = []
    async for chunk in agent.stream_message(message):
        if isinstance(chunk, str):
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
        assert _get_assistant_text(agent.conversation[1]) == '回應內容'

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
        assert _get_assistant_text(agent.conversation[1]) == '第一次回應'
        assert agent.conversation[2]['content'] == '第二則訊息'
        assert _get_assistant_text(agent.conversation[3]) == '第二次回應'

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
        assert _get_assistant_text(agent.conversation[1]) == '這是一個串流回應'

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
                if isinstance(chunk, str):
                    received.append(chunk)

        # Assert - 應收到部分回應且被保留
        assert received == ['這是部分', '回應']
        assert '中斷' in str(exc_info.value)
        # 部分回應應被保留在對話歷史中
        assert len(agent.conversation) == 2
        assert agent.conversation[1]['content'] == '這是部分回應'


# =============================================================================
# Rule: Agent Loop 應持續運作直到任務完成（工具調用迴圈）
# =============================================================================


class TestToolUseLoop:
    """測試工具調用迴圈功能。"""

    async def test_no_tool_registry_unchanged(self, agent: Agent, mock_client: MagicMock) -> None:
        """Scenario: 無 tool_registry 時行為不變。

        Given Agent 沒有設定 tool_registry
        When 使用者發送訊息
        Then Agent 應正常回應，不傳遞 tools 參數
        """
        # Arrange
        mock_client.messages.stream.return_value = create_mock_stream(['回應'])

        # Act
        await collect_stream(agent, '測試')

        # Assert - 不應傳遞 tools 參數
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert 'tools' not in call_kwargs

    async def test_single_turn_with_tool_call(self, mock_client: MagicMock) -> None:
        """Scenario: 單輪對話有工具調用。

        Given 使用者輸入需要工具的問題
        When Agent 處理該輸入
        Then Agent 應執行所需工具
        And Agent 應將工具結果傳回 Claude
        And Agent 應回傳最終回應
        """
        # Arrange - 建立帶工具的 Agent
        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=lambda path: {'content': f'檔案內容: {path}', 'path': path},  # type: ignore[reportUnknownLambdaType]
        )
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client, tool_registry=registry)

        # 第一次 API 呼叫：Claude 回傳 tool_use
        tool_use_block = _make_tool_use_block('tool_1', 'read_file', {'path': 'main.py'})
        first_stream = create_mock_stream(
            text_chunks=[],
            stop_reason='tool_use',
            content_blocks=[tool_use_block],
        )

        # 第二次 API 呼叫：Claude 回傳最終文字
        second_stream = create_mock_stream(['檔案內容如下...'])

        mock_client.messages.stream.side_effect = [first_stream, second_stream]

        # Act
        result = await collect_stream(agent, '請讀取 main.py')

        # Assert - 最終文字回應
        assert result == '檔案內容如下...'

        # Assert - 對話歷史應包含完整流程
        # [0] user: '請讀取 main.py'
        # [1] assistant: tool_use block
        # [2] user: tool_result
        # [3] assistant: 最終回應
        assert len(agent.conversation) == 4
        assert agent.conversation[0]['role'] == 'user'
        assert agent.conversation[1]['role'] == 'assistant'
        assert agent.conversation[2]['role'] == 'user'
        assert agent.conversation[3]['role'] == 'assistant'

        # 驗證 tool_result 內容
        tool_results: Any = agent.conversation[2]['content']
        assert len(tool_results) == 1
        assert tool_results[0]['tool_use_id'] == 'tool_1'
        assert '檔案內容: main.py' in tool_results[0]['content']

        # Assert - API 應被呼叫兩次，且第一次包含 tools 參數
        assert mock_client.messages.stream.call_count == 2
        first_call_kwargs = mock_client.messages.stream.call_args_list[0][1]
        assert 'tools' in first_call_kwargs

    async def test_tool_execution_error(self, mock_client: MagicMock) -> None:
        """Scenario: 工具執行失敗時回傳 is_error。

        Given 工具執行時拋出例外
        When Agent 處理工具結果
        Then tool_result 應包含 is_error 標記
        And Agent 應繼續將錯誤回傳給 Claude
        """

        # Arrange - 建立會失敗的工具
        def failing_handler(path: str) -> str:
            raise FileNotFoundError('檔案不存在: missing.py')

        registry = ToolRegistry()
        registry.register(
            name='read_file',
            description='讀取檔案',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=failing_handler,
        )
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client, tool_registry=registry)

        # 第一次：Claude 請求工具
        tool_use_block = _make_tool_use_block('tool_err', 'read_file', {'path': 'missing.py'})
        first_stream = create_mock_stream(
            text_chunks=[],
            stop_reason='tool_use',
            content_blocks=[tool_use_block],
        )

        # 第二次：Claude 回傳錯誤說明
        second_stream = create_mock_stream(['找不到該檔案。'])

        mock_client.messages.stream.side_effect = [first_stream, second_stream]

        # Act
        result = await collect_stream(agent, '讀取 missing.py')

        # Assert - tool_result 應包含 is_error
        tool_results: Any = agent.conversation[2]['content']
        assert tool_results[0]['is_error'] is True
        assert '檔案不存在' in tool_results[0]['content']

        # Assert - 最終仍有回應
        assert result == '找不到該檔案。'


# =============================================================================
# Prompt Caching Tests (根據 agent_core.feature Rule: Prompt Caching)
# =============================================================================


class TestPromptCaching:
    """測試 Prompt Caching 功能。

    對應規格: docs/features/agent_core.feature
    Rule: Agent 應使用 Prompt Caching 優化 API 呼叫
    """

    def test_add_cache_control_to_last_message(self, mock_client: MagicMock) -> None:
        """Scenario: 在對話歷史最後添加緩存斷點。

        Given Agent 已有對話歷史
        When Agent 建立 API 請求參數
        Then 對話歷史的最後一個 message 應包含 cache_control
        And cache_control 類型應為 ephemeral
        """
        # Arrange
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client)
        agent.conversation = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi'}]},
            {'role': 'user', 'content': 'How are you?'},
        ]

        # Act
        kwargs = agent.build_stream_kwargs()

        # Assert - 最後一個 message 應包含 cache_control
        last_message = kwargs['messages'][-1]
        assert 'content' in last_message

        # 檢查最後一個 content block
        if isinstance(last_message['content'], list):
            last_block = last_message['content'][-1]
            assert 'cache_control' in last_block
            assert last_block['cache_control'] == {'type': 'ephemeral'}
        else:
            # 字串格式應被轉換
            pytest.fail('Content should be converted to list format')

    def test_preserve_original_conversation(self, mock_client: MagicMock) -> None:
        """Scenario: 不修改原始對話歷史。

        Given Agent 的對話歷史包含 3 則訊息
        When Agent 建立帶有 cache_control 的 API 請求參數
        Then 原始對話歷史應保持不變
        And 原始對話歷史不應包含任何 cache_control
        """
        # Arrange
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client)
        original_conversation = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi'}]},
            {'role': 'user', 'content': 'Bye'},
        ]
        agent.conversation = original_conversation.copy()  # type: ignore[assignment]

        # Act
        kwargs = agent.build_stream_kwargs()

        # Assert - 原始 conversation 不應被修改
        assert len(agent.conversation) == 3
        for msg in agent.conversation:
            content = msg['content']
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert 'cache_control' not in block
            # 字串格式本來就沒有 cache_control

        # Assert - kwargs 中的 messages 應該有 cache_control
        last_message = kwargs['messages'][-1]
        if isinstance(last_message['content'], list):
            last_block = last_message['content'][-1]
            assert 'cache_control' in last_block

    def test_handle_string_content(self, mock_client: MagicMock) -> None:
        """Scenario: 處理字串類型的 content。

        Given 對話歷史最後一則訊息的 content 是字串
        When Agent 建立 API 請求參數
        Then 該 content 應轉換為 text block 格式
        And text block 應包含 cache_control
        """
        # Arrange
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client)
        agent.conversation = [
            {'role': 'user', 'content': 'Hello, world!'},  # ← 字串格式
        ]

        # Act
        kwargs = agent.build_stream_kwargs()

        # Assert - 應轉換為 list 格式
        last_message = kwargs['messages'][-1]
        assert isinstance(last_message['content'], list)
        assert len(last_message['content']) == 1

        # Assert - text block 應包含 cache_control
        text_block = last_message['content'][0]
        assert text_block['type'] == 'text'
        assert text_block['text'] == 'Hello, world!'
        assert text_block['cache_control'] == {'type': 'ephemeral'}

    def test_handle_list_content(self, mock_client: MagicMock) -> None:
        """Scenario: 處理列表類型的 content。

        Given 對話歷史最後一則訊息的 content 是列表
        When Agent 建立 API 請求參數
        Then 列表的最後一個 block 應包含 cache_control
        And 其他 blocks 不應包含 cache_control
        """
        # Arrange
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client)
        agent.conversation = [
            {
                'role': 'user',
                'content': [  # ← 列表格式
                    {'type': 'tool_result', 'tool_use_id': 'tool_1', 'content': 'result 1'},
                    {'type': 'tool_result', 'tool_use_id': 'tool_2', 'content': 'result 2'},
                ],
            }
        ]

        # Act
        kwargs = agent.build_stream_kwargs()

        # Assert - 只有最後一個 block 應包含 cache_control
        last_message = kwargs['messages'][-1]
        content_list = cast(list[dict[str, Any]], last_message['content'])
        assert len(content_list) == 2

        # 第一個 block 不應有 cache_control
        assert 'cache_control' not in content_list[0]

        # 最後一個 block 應有 cache_control
        assert 'cache_control' in content_list[1]
        assert content_list[1]['cache_control'] == {'type': 'ephemeral'}

    def test_empty_conversation(self, mock_client: MagicMock) -> None:
        """測試空對話歷史的邊界情況。

        Given Agent 的對話歷史為空
        When Agent 建立 API 請求參數
        Then 不應發生錯誤
        And messages 應為空列表
        """
        # Arrange
        config = AgentConfig(system_prompt='測試')
        agent = Agent(config=config, client=mock_client)
        agent.conversation = []

        # Act
        kwargs = agent.build_stream_kwargs()

        # Assert
        assert kwargs['messages'] == []
