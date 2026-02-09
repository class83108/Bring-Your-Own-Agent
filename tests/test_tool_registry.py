"""Tool Registry 測試模組。

根據 docs/features/agent_core.feature 規格撰寫測試案例。
涵蓋 Rule: Tool Registry 應能動態管理工具
以及 Rule: Agent 應支援並行工具執行
"""

from __future__ import annotations

import asyncio
from typing import Any

import allure
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> Any:
    """建立測試用 Tool Registry。"""
    from agent_core.tools.registry import ToolRegistry

    return ToolRegistry()


def sample_read_file(path: str) -> str:
    """範例工具：讀取檔案。"""
    return f'檔案內容: {path}'


async def slow_tool_a(param: str) -> str:
    """模擬耗時工具 A。"""
    await asyncio.sleep(0.1)
    return f'工具A結果: {param}'


async def slow_tool_b(param: str) -> str:
    """模擬耗時工具 B。"""
    await asyncio.sleep(0.1)
    return f'工具B結果: {param}'


async def failing_tool(param: str) -> str:
    """會失敗的工具。"""
    raise ValueError('工具執行失敗')


def sample_search_tool(query: str) -> str:
    """範例工具：搜尋程式碼。"""
    return f'搜尋: {query}'


# =============================================================================
# Rule: Tool Registry 應能動態管理工具
# =============================================================================


@allure.feature('Agent 核心架構')
@allure.story('Tool Registry 應能動態管理工具')
class TestToolRegistryManagement:
    """測試 Tool Registry 動態管理工具功能。"""

    @allure.title('註冊新工具。')
    def test_register_new_tool(self, registry: Any) -> None:
        """Scenario: 註冊新工具。

        Given Tool Registry 已初始化
        When 註冊一個新工具
        Then 該工具應出現在可用工具列表中
        And Claude 請求應包含該工具的定義
        """
        # Act - When 註冊一個新工具
        registry.register(
            name='read_file',
            description='讀取檔案內容',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': '檔案路徑'},
                },
                'required': ['path'],
            },
            handler=sample_read_file,
        )

        # Assert - Then 該工具應出現在可用工具列表中
        assert 'read_file' in registry.list_tools()

        # Assert - And Claude 請求應包含該工具的定義
        definitions = registry.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]['name'] == 'read_file'
        assert definitions[0]['description'] == '讀取檔案內容'
        assert 'input_schema' in definitions[0]

    async def test_execute_registered_tool(self, registry: Any) -> None:
        """Scenario: 執行已註冊的工具。

        Given Tool Registry 包含 "read_file" 工具
        When Claude 請求執行 "read_file" 工具
        Then Agent 應找到並執行該工具
        And 應回傳工具執行結果
        """
        # Arrange - Given Tool Registry 包含 "read_file" 工具
        registry.register(
            name='read_file',
            description='讀取檔案內容',
            parameters={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
            handler=sample_read_file,
        )

        # Act - When Claude 請求執行 "read_file" 工具
        result = await registry.execute('read_file', {'path': 'test.py'})

        # Assert - Then 應回傳工具執行結果
        assert result == '檔案內容: test.py'

    async def test_execute_unknown_tool(self, registry: Any) -> None:
        """Scenario: 處理未知工具調用。

        Given Tool Registry 不包含 "unknown_tool" 工具
        When Claude 請求執行 "unknown_tool" 工具
        Then Agent 應回傳錯誤訊息
        And 錯誤訊息應說明工具不存在
        """
        # Act & Assert
        with pytest.raises(KeyError) as exc_info:
            await registry.execute('unknown_tool', {})

        # 錯誤訊息應說明工具不存在
        assert 'unknown_tool' in str(exc_info.value)


# =============================================================================
# Rule: 並行執行應避免檔案競爭
# =============================================================================


class MockLockProvider:
    """Mock Lock Provider 用於測試，模擬真正的鎖行為。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []  # [('acquire', 'file.py'), ...]
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, key: str) -> None:
        """取得鎖，如果已被持有則等待。"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        await self._locks[key].acquire()
        self.events.append(('acquire', key))

    async def release(self, key: str) -> None:
        """釋放鎖。"""
        self.events.append(('release', key))
        self._locks[key].release()


async def sample_file_tool(path: str) -> str:
    """範例檔案操作工具。"""
    await asyncio.sleep(0.05)  # 模擬 I/O
    return f'操作完成: {path}'


@allure.feature('Agent 核心架構')
@allure.story('Tool Registry 應能動態管理工具')
class TestToolSummaries:
    """測試工具摘要查詢功能。"""

    @allure.title('空的 registry 應回傳空列表')
    def test_get_tool_summaries_empty(self, registry: Any) -> None:
        """空的 registry 應回傳空列表。"""
        assert registry.get_tool_summaries() == []

    @allure.title('摘要應包含 name、description、source')
    def test_get_tool_summaries_returns_name_description_source(self, registry: Any) -> None:
        """摘要應包含 name、description、source。"""
        registry.register(
            name='read_file',
            description='讀取檔案內容',
            parameters={'type': 'object', 'properties': {}},
            handler=sample_read_file,
        )

        summaries = registry.get_tool_summaries()

        assert len(summaries) == 1
        assert summaries[0] == {
            'name': 'read_file',
            'description': '讀取檔案內容',
            'source': 'native',
        }

    @allure.title('修改 source 後摘要應反映變更')
    def test_get_tool_summaries_reflects_source_change(self, registry: Any) -> None:
        """修改 source 後摘要應反映變更。"""
        registry.register(
            name='mcp_tool',
            description='MCP 工具',
            parameters={'type': 'object', 'properties': {}},
            handler=sample_read_file,
        )
        registry.set_tool_source('mcp_tool', 'mcp')

        summaries = registry.get_tool_summaries()

        assert summaries[0]['source'] == 'mcp'


@allure.feature('Agent 核心架構')
@allure.story('並行執行應避免檔案競爭')
class TestFileLocking:
    """測試檔案鎖定機制，避免競爭條件。"""

    @allure.title('測試檔案工具執行時應正確取得並釋放鎖。')
    async def test_file_tool_acquires_and_releases_lock(self) -> None:
        """測試檔案工具執行時應正確取得並釋放鎖。

        Given 工具註冊時指定了 file_param
        When 執行該工具
        Then 應先取得鎖，執行完成後釋放鎖
        """
        from agent_core.tools.registry import ToolRegistry

        # Arrange
        mock_lock = MockLockProvider()
        registry = ToolRegistry(lock_provider=mock_lock)
        registry.register(
            name='edit_file',
            description='編輯檔案',
            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            handler=sample_file_tool,
            file_param='path',
        )

        # Act
        await registry.execute('edit_file', {'path': 'src/main.py'})

        # Assert - 應有 acquire 和 release 事件
        assert ('acquire', 'src/main.py') in mock_lock.events
        assert ('release', 'src/main.py') in mock_lock.events
        # 順序應為 acquire → release
        acquire_idx = mock_lock.events.index(('acquire', 'src/main.py'))
        release_idx = mock_lock.events.index(('release', 'src/main.py'))
        assert acquire_idx < release_idx

    @allure.title('測試沒有 file_param 的工具不應取得鎖。')
    async def test_tool_without_file_param_does_not_acquire_lock(self) -> None:
        """測試沒有 file_param 的工具不應取得鎖。

        Given 工具註冊時沒有指定 file_param
        When 執行該工具
        Then 不應嘗試取得任何鎖
        """
        from agent_core.tools.registry import ToolRegistry

        # Arrange
        mock_lock = MockLockProvider()
        registry = ToolRegistry(lock_provider=mock_lock)
        registry.register(
            name='search_code',
            description='搜尋程式碼',
            parameters={'type': 'object', 'properties': {'query': {'type': 'string'}}},
            handler=sample_search_tool,
        )

        # Act
        await registry.execute('search_code', {'query': 'def main'})

        # Assert - 不應有任何鎖事件
        assert len(mock_lock.events) == 0

    @allure.title('測試操作同一檔案的工具應串行執行。')
    async def test_same_file_operations_are_serialized(self) -> None:
        """測試操作同一檔案的工具應串行執行。

        Given 兩個工具都要操作同一個檔案
        When 並行執行這兩個工具
        Then 事件順序應為 acquire → release → acquire → release
        """
        from agent_core.tools.registry import ToolRegistry

        # Arrange
        mock_lock = MockLockProvider()
        registry = ToolRegistry(lock_provider=mock_lock)
        registry.register(
            name='tool_a',
            description='工具 A',
            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            handler=sample_file_tool,
            file_param='path',
        )
        registry.register(
            name='tool_b',
            description='工具 B',
            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            handler=sample_file_tool,
            file_param='path',
        )

        # Act - 使用 asyncio.gather 模擬 agent 的並行執行方式
        await asyncio.gather(
            registry.execute('tool_a', {'path': 'same_file.py'}),
            registry.execute('tool_b', {'path': 'same_file.py'}),
        )

        # Assert - 同一檔案的操作應串行：acquire → release → acquire → release
        file_events = [e for e in mock_lock.events if e[1] == 'same_file.py']
        assert len(file_events) == 4
        assert file_events[0] == ('acquire', 'same_file.py')
        assert file_events[1] == ('release', 'same_file.py')
        assert file_events[2] == ('acquire', 'same_file.py')
        assert file_events[3] == ('release', 'same_file.py')

    @allure.title('測試操作不同檔案的工具可以交錯執行（並行）。')
    async def test_different_file_operations_can_interleave(self) -> None:
        """測試操作不同檔案的工具可以交錯執行（並行）。

        Given 兩個工具操作不同的檔案
        When 並行執行這兩個工具
        Then 兩個檔案的 acquire 可以在任一 release 之前發生
        """
        from agent_core.tools.registry import ToolRegistry

        # Arrange
        mock_lock = MockLockProvider()
        registry = ToolRegistry(lock_provider=mock_lock)
        registry.register(
            name='tool_a',
            description='工具 A',
            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            handler=sample_file_tool,
            file_param='path',
        )
        registry.register(
            name='tool_b',
            description='工具 B',
            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            handler=sample_file_tool,
            file_param='path',
        )

        # Act - 使用 asyncio.gather 模擬 agent 的並行執行方式
        await asyncio.gather(
            registry.execute('tool_a', {'path': 'file_a.py'}),
            registry.execute('tool_b', {'path': 'file_b.py'}),
        )

        # Assert - 不同檔案可並行，兩個 acquire 應該都在某個 release 之前
        acquire_events = [e for e in mock_lock.events if e[0] == 'acquire']
        release_events = [e for e in mock_lock.events if e[0] == 'release']

        assert len(acquire_events) == 2
        assert len(release_events) == 2

        # 找出第一個 release 的位置
        first_release_idx = mock_lock.events.index(release_events[0])
        # 兩個 acquire 都應該在第一個 release 之前（表示並行）
        acquire_indices = [mock_lock.events.index(e) for e in acquire_events]
        assert all(idx < first_release_idx for idx in acquire_indices)
