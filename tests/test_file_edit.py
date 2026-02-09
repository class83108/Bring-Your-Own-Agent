"""File Edit 工具測試模組。

根據 docs/features/file_edit.feature 規格撰寫測試案例。
涵蓋：
- Rule: Agent 應能建立新檔案
- Rule: Agent 應能編輯現有檔案
- Rule: Agent 應使用精確的編輯方式
- Rule: Agent 應處理編輯安全性
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import allure
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """建立測試用 sandbox 目錄。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    # 建立測試用子目錄
    (sandbox / 'src').mkdir()

    # 建立現有的 Python 檔案
    (sandbox / 'src' / 'main.py').write_text(
        'def old_function():\n    return "old"\n', encoding='utf-8'
    )

    # 建立包含類別的檔案
    (sandbox / 'src' / 'model.py').write_text(
        'class User:\n    def __init__(self):\n        pass\n', encoding='utf-8'
    )

    # 建立包含已棄用函數的檔案
    (sandbox / 'src' / 'utils.py').write_text(
        'def active_function():\n    pass\n\ndef deprecated_function():\n    pass\n',
        encoding='utf-8',
    )

    # 建立包含重複內容的檔案
    (sandbox / 'src' / 'duplicate.py').write_text(
        'value = 1\nvalue = 2\nvalue = 3\n', encoding='utf-8'
    )

    return sandbox


@pytest.fixture
def edit_file(sandbox_dir: Path) -> Any:
    """建立 edit_file 函數，已綁定 sandbox_root。"""
    from agent_core.tools.file_edit import edit_file_handler

    def _edit(
        path: str,
        old_content: str | None = None,
        new_content: str | None = None,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        return edit_file_handler(
            path=path,
            old_content=old_content,
            new_content=new_content,
            create_if_missing=create_if_missing,
            sandbox_root=sandbox_dir,
        )

    return _edit


# =============================================================================
# Rule: Agent 應能建立新檔案
# =============================================================================


@allure.feature('編輯檔案功能')
@allure.story('Agent 應能建立新檔案')
class TestCreateFile:
    """測試建立新檔案功能。"""

    @allure.title('建立新的程式檔案')
    def test_create_new_file(self, edit_file: Any, sandbox_dir: Path) -> None:
        """Scenario: 建立新的程式檔案。

        Given 檔案 "src/new_module.py" 不存在
        When 使用者要求建立該檔案
        Then 應建立新檔案
        And 檔案應包含使用者要求的內容
        """
        result = edit_file(
            path='src/new_module.py',
            new_content='def new_function():\n    pass\n',
            create_if_missing=True,
        )

        # 檢查回傳結果
        assert result['path'] == 'src/new_module.py'
        assert result['created'] is True

        # 檢查檔案確實被建立
        file_path = sandbox_dir / 'src' / 'new_module.py'
        assert file_path.exists()
        assert file_path.read_text(encoding='utf-8') == 'def new_function():\n    pass\n'

    @allure.title('建立檔案在不存在的目錄')
    def test_create_file_with_new_directory(self, edit_file: Any, sandbox_dir: Path) -> None:
        """Scenario: 建立檔案在不存在的目錄。

        Given 目錄 "src/utils/" 不存在
        When 使用者要求建立 "src/utils/helper.py"
        Then 應先建立必要的目錄
        And 應建立新檔案
        """
        result = edit_file(
            path='src/utils/helper.py',
            new_content='def helper():\n    pass\n',
            create_if_missing=True,
        )

        assert result['created'] is True

        # 檢查目錄和檔案都被建立
        dir_path = sandbox_dir / 'src' / 'utils'
        file_path = dir_path / 'helper.py'
        assert dir_path.exists()
        assert file_path.exists()

    @allure.title('拒絕覆蓋已存在的檔案')
    def test_refuse_to_overwrite_existing_file(self, edit_file: Any) -> None:
        """Scenario: 拒絕覆蓋已存在的檔案。

        Given 檔案 "src/main.py" 已存在
        When 使用者要求建立同名檔案但不提供 old_content
        Then 應回傳錯誤
        """
        with pytest.raises(FileExistsError, match='已存在'):
            edit_file(
                path='src/main.py',
                new_content='new content',
                create_if_missing=True,
            )


# =============================================================================
# Rule: Agent 應能編輯現有檔案
# =============================================================================


@allure.feature('編輯檔案功能')
@allure.story('Agent 應能編輯現有檔案')
class TestEditExistingFile:
    """測試編輯現有檔案功能。"""

    @allure.title('替換檔案中的特定內容')
    def test_replace_content(self, edit_file: Any, sandbox_dir: Path) -> None:
        """Scenario: 替換檔案中的特定內容。

        Given 檔案 "src/main.py" 包含函數 "old_function"
        When 使用者要求將 old_function 重新命名為 new_function
        Then 所有 "old_function" 應被替換為 "new_function"
        """
        result = edit_file(
            path='src/main.py',
            old_content='def old_function():',
            new_content='def new_function():',
        )

        assert result['path'] == 'src/main.py'
        assert result['modified'] is True

        # 檢查內容確實被替換
        content = (sandbox_dir / 'src' / 'main.py').read_text(encoding='utf-8')
        assert 'def new_function():' in content
        assert 'def old_function():' not in content

    @allure.title('在特定位置插入內容')
    def test_insert_content(self, edit_file: Any, sandbox_dir: Path) -> None:
        """Scenario: 在特定位置插入內容。

        Given 檔案 "src/model.py" 包含類別定義
        When 使用者要求在類別中新增一個方法
        Then 應在正確位置插入新方法
        And 應保持正確的縮排
        """
        result = edit_file(
            path='src/model.py',
            old_content='class User:\n    def __init__(self):\n        pass',
            new_content=(
                'class User:\n'
                '    def __init__(self):\n'
                '        pass\n'
                '\n'
                '    def get_name(self):\n'
                '        return self.name'
            ),
        )

        assert result['modified'] is True

        # 檢查內容確實被插入
        content = (sandbox_dir / 'src' / 'model.py').read_text(encoding='utf-8')
        assert 'def get_name(self):' in content

    @allure.title('刪除特定內容')
    def test_delete_content(self, edit_file: Any, sandbox_dir: Path) -> None:
        """Scenario: 刪除特定內容。

        Given 檔案包含已棄用的函數
        When 使用者要求刪除 deprecated_function
        Then 應移除該函數
        And 應保持檔案其餘部分不變
        """
        result = edit_file(
            path='src/utils.py',
            old_content='def deprecated_function():\n    pass\n',
            new_content='',
        )

        assert result['modified'] is True

        # 檢查函數確實被移除
        content = (sandbox_dir / 'src' / 'utils.py').read_text(encoding='utf-8')
        assert 'deprecated_function' not in content
        assert 'active_function' in content  # 其他函數應保留


# =============================================================================
# Rule: Agent 應使用精確的編輯方式
# =============================================================================


@allure.feature('編輯檔案功能')
@allure.story('Agent 應使用精確的編輯方式')
class TestPreciseEditing:
    """測試精確編輯功能。"""

    @allure.title('處理搜尋內容不存在')
    def test_search_content_not_found(self, edit_file: Any) -> None:
        """Scenario: 處理搜尋內容不存在。

        Given 檔案不包含要搜尋的內容
        When Agent 嘗試執行替換
        Then 應回傳錯誤
        """
        with pytest.raises(ValueError, match='找不到'):
            edit_file(
                path='src/main.py',
                old_content='nonexistent content',
                new_content='new content',
            )

    @allure.title('處理多個匹配')
    def test_multiple_matches_error(self, edit_file: Any) -> None:
        """Scenario: 處理多個匹配。

        Given 檔案中有多處匹配搜尋內容
        When Agent 執行替換且 old_content 不夠精確
        Then 應回傳錯誤要求更精確的搜尋內容
        """
        with pytest.raises(ValueError, match='多處匹配'):
            edit_file(
                path='src/duplicate.py',
                old_content='value',
                new_content='new_value',
            )


# =============================================================================
# Rule: Agent 應處理編輯安全性
# =============================================================================


@allure.feature('編輯檔案功能')
@allure.story('Agent 應處理編輯安全性')
class TestEditSecurity:
    """測試編輯安全性。"""

    @allure.title('阻擋編輯工作目錄外的檔案')
    def test_block_edit_outside_sandbox(self, edit_file: Any) -> None:
        """Scenario: 阻擋編輯工作目錄外的檔案。

        Given 工作目錄為 sandbox
        When 使用者要求編輯 sandbox 外的檔案
        Then 應拒絕編輯
        """
        with pytest.raises(PermissionError):
            edit_file(
                path='../../../etc/hosts',
                old_content='old',
                new_content='new',
            )

    @allure.title('阻擋路徑穿越攻擊')
    def test_block_path_traversal(self, edit_file: Any) -> None:
        """阻擋路徑穿越攻擊。"""
        with pytest.raises(PermissionError):
            edit_file(
                path='src/../../outside.txt',
                new_content='malicious content',
                create_if_missing=True,
            )


# =============================================================================
# Rule: 編輯檔案時應回傳 SSE 事件與 diff
# =============================================================================


@allure.feature('編輯檔案功能')
@allure.story('Agent 應驗證編輯結果')
class TestFileEditSSEEvents:
    """測試編輯檔案時回傳 SSE 事件資料。"""

    @allure.title('編輯檔案時回傳 SSE 事件與 diff')
    def test_edit_file_returns_sse_events_with_diff(self, edit_file: Any) -> None:
        """Scenario: 編輯檔案時回傳 SSE 事件與 diff。

        Given 檔案 "src/main.py" 包含舊內容
        When 使用者要求編輯檔案
        Then 回傳結果應包含 sse_events
        And sse_events 應包含 file_change 事件
        And 事件資料應包含路徑與 unified diff
        """
        result = edit_file(
            path='src/main.py',
            old_content='def old_function():',
            new_content='def new_function():',
        )

        # 驗證包含 sse_events
        assert 'sse_events' in result
        assert isinstance(result['sse_events'], list)
        assert len(result['sse_events']) == 1

        # 驗證事件類型
        event = result['sse_events'][0]
        assert event['type'] == 'file_change'

        # 驗證事件資料
        event_data = event['data']
        assert event_data['path'] == 'src/main.py'
        assert 'diff' in event_data

        # 驗證 diff 格式（unified diff）
        diff = event_data['diff']
        assert '--- ' in diff
        assert '+++ ' in diff
        assert '-def old_function():' in diff
        assert '+def new_function():' in diff

    @allure.title('建立新檔案時回傳 SSE 事件與 diff')
    def test_create_file_returns_sse_events_with_diff(self, edit_file: Any) -> None:
        """Scenario: 建立新檔案時回傳 SSE 事件與 diff。

        Given 檔案 "src/new.py" 不存在
        When 使用者要求建立新檔案
        Then 回傳結果應包含 sse_events
        And diff 應顯示所有內容為新增
        """
        result = edit_file(
            path='src/new.py',
            new_content='def hello():\n    print("world")\n',
            create_if_missing=True,
        )

        # 驗證包含 sse_events
        assert 'sse_events' in result
        event = result['sse_events'][0]
        assert event['type'] == 'file_change'

        # 驗證 diff 顯示為新增
        diff = event['data']['diff']
        assert '--- ' in diff or '--- /dev/null' in diff
        assert '+def hello():' in diff
        assert '+    print("world")' in diff
