"""Tool Result 分頁測試模組。

根據 docs/features/tool_result_pagination.feature 規格撰寫測試案例。
涵蓋 ToolRegistry 結果分頁、read_more 工具、暫存區管理。
"""

from __future__ import annotations

import allure
import pytest

from agent_core.tools.registry import ToolRegistry

# =============================================================================
# Constants
# =============================================================================

# 測試用的較小上限，方便驗證分頁行為
TEST_MAX_RESULT_CHARS = 100


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> ToolRegistry:
    """建立啟用分頁功能的 ToolRegistry。"""
    return ToolRegistry(max_result_chars=TEST_MAX_RESULT_CHARS)


def _make_large_handler(size: int):
    """建立回傳指定大小字串的工具 handler。"""

    def handler() -> str:
        return 'x' * size

    return handler


def _register_tool(registry: ToolRegistry, name: str = 'big_tool', size: int = 300) -> None:
    """註冊一個回傳指定大小結果的工具。"""
    registry.register(
        name=name,
        description='回傳大量內容的工具',
        parameters={'type': 'object', 'properties': {}},
        handler=_make_large_handler(size),
    )


# =============================================================================
# Rule: 小結果應直接回傳，不受影響
# =============================================================================


@allure.feature('Tool Result 分頁')
@allure.story('小結果應直接回傳，不受影響')
class TestSmallResultPassthrough:
    """測試小結果不受分頁影響。"""

    @allure.title('工具結果未超過上限。')
    async def test_small_result_returned_as_is(self, registry: ToolRegistry) -> None:
        """Scenario: 工具結果未超過上限。

        Given 已註冊一個工具，回傳 50 字元的結果
        When 執行該工具
        Then 應回傳完整的原始結果
        And 不應產生任何分頁暫存
        """
        _register_tool(registry, size=50)

        result = await registry.execute('big_tool', {})

        # 結果應為原始字串，不含分頁提示
        assert result == 'x' * 50
        # 暫存區應為空
        assert registry.get_paginated_result_count() == 0


# =============================================================================
# Rule: 超大結果應自動分頁並提供 read_more 機制
# =============================================================================


@allure.feature('Tool Result 分頁')
@allure.story('超大結果應自動分頁並提供 read_more 機制')
class TestLargeResultPagination:
    """測試超大結果的自動分頁。"""

    @allure.title('工具結果超過上限時回傳第一頁。')
    async def test_large_result_returns_first_page(self, registry: ToolRegistry) -> None:
        """Scenario: 工具結果超過上限時回傳第一頁。

        Given 已註冊一個工具，回傳 300 字元的結果
        When 執行該工具
        Then 回傳內容應只包含前 MAX_RESULT_CHARS 字元
        And 回傳內容應包含分頁提示
        And 完整結果應被儲存至暫存區
        """
        _register_tool(registry, size=300)

        result = await registry.execute('big_tool', {})

        assert isinstance(result, str)
        # 應包含第一頁的資料內容
        assert 'x' * TEST_MAX_RESULT_CHARS in result
        # 應包含分頁提示
        assert 'read_more' in result
        assert '1' in result  # 第 1 頁
        assert '3' in result  # 共 3 頁
        # 暫存區應有一筆
        assert registry.get_paginated_result_count() == 1

    @allure.title('透過 read_more 取得後續頁面。')
    async def test_read_more_returns_second_page(self, registry: ToolRegistry) -> None:
        """Scenario: 透過 read_more 取得後續頁面。

        Given 已有一筆分頁暫存結果（共 3 頁）
        When 呼叫 read_more 並指定 page=2
        Then 應回傳第 2 頁的內容
        And 回傳內容應包含分頁提示
        """
        _register_tool(registry, size=300)
        await registry.execute('big_tool', {})

        # 從第一頁結果中取得 result_id
        result_id = registry.get_last_result_id()

        second_page = registry.read_more(result_id=result_id, page=2)

        assert isinstance(second_page, str)
        assert 'x' * TEST_MAX_RESULT_CHARS in second_page
        assert '2' in second_page  # 第 2 頁

    @allure.title('透過 read_more 取得最後一頁。')
    async def test_read_more_returns_last_page(self, registry: ToolRegistry) -> None:
        """Scenario: 透過 read_more 取得最後一頁。

        Given 已有一筆分頁暫存結果（共 3 頁）
        When 呼叫 read_more 並指定 page=3
        Then 應回傳第 3 頁的內容
        And 分頁提示應標示為最後一頁
        """
        _register_tool(registry, size=300)
        await registry.execute('big_tool', {})
        result_id = registry.get_last_result_id()

        last_page = registry.read_more(result_id=result_id, page=3)

        assert isinstance(last_page, str)
        # 最後一頁的資料長度等於總長度減去前兩頁
        expected_remaining = 300 - TEST_MAX_RESULT_CHARS * 2
        assert 'x' * expected_remaining in last_page


# =============================================================================
# Rule: read_more 應處理無效請求
# =============================================================================


@allure.feature('Tool Result 分頁')
@allure.story('read_more 應處理無效請求')
class TestReadMoreErrorHandling:
    """測試 read_more 的錯誤處理。"""

    @allure.title('查詢不存在的 result_id。')
    def test_read_more_invalid_result_id(self, registry: ToolRegistry) -> None:
        """Scenario: 查詢不存在的 result_id。

        When 呼叫 read_more 並指定不存在的 result_id
        Then 應回傳錯誤訊息說明結果不存在或已過期
        """
        result = registry.read_more(result_id='nonexistent', page=1)

        assert '不存在' in result or '過期' in result

    @allure.title('查詢超出範圍的頁數。')
    async def test_read_more_page_out_of_range(self, registry: ToolRegistry) -> None:
        """Scenario: 查詢超出範圍的頁數。

        Given 已有一筆分頁暫存結果（共 3 頁）
        When 呼叫 read_more 並指定 page=5
        Then 應回傳錯誤訊息說明頁數超出範圍
        """
        _register_tool(registry, size=300)
        await registry.execute('big_tool', {})
        result_id = registry.get_last_result_id()

        result = registry.read_more(result_id=result_id, page=5)

        assert '超出範圍' in result or '不存在' in result


# =============================================================================
# Rule: 暫存區應有生命週期管理
# =============================================================================


@allure.feature('Tool Result 分頁')
@allure.story('暫存區應有生命週期管理')
class TestPaginationLifecycle:
    """測試暫存區生命週期。"""

    async def test_paginated_result_can_be_accessed_multiple_times(
        self, registry: ToolRegistry
    ) -> None:
        """Scenario: 暫存結果在對話中可重複存取。

        Given 已有一筆分頁暫存結果
        When 連續呼叫 read_more 取得第 1 頁兩次
        Then 兩次應回傳相同的內容
        """
        _register_tool(registry, size=300)
        await registry.execute('big_tool', {})
        result_id = registry.get_last_result_id()

        page1_first = registry.read_more(result_id=result_id, page=1)
        page1_second = registry.read_more(result_id=result_id, page=1)

        assert page1_first == page1_second

    @allure.title('清除暫存區。')
    async def test_clear_paginated_results(self, registry: ToolRegistry) -> None:
        """Scenario: 清除暫存區。

        Given 暫存區包含多筆分頁結果
        When 呼叫清除暫存區
        Then 暫存區應為空
        And 之前�� result_id 應不再可用
        """
        _register_tool(registry, name='tool_a', size=300)
        _register_tool(registry, name='tool_b', size=300)
        await registry.execute('tool_a', {})
        result_id_a = registry.get_last_result_id()
        await registry.execute('tool_b', {})

        registry.clear_paginated_results()

        assert registry.get_paginated_result_count() == 0
        result = registry.read_more(result_id=result_id_a, page=1)
        assert '不存在' in result or '過期' in result
