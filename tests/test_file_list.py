"""File List 工具測試模組。

根據 docs/features/file_list.feature 規格撰寫測試案例。
涵蓋：
- Rule: Agent 應能列出目錄內容
- Rule: Agent 應支援遞迴列出檔案
- Rule: Agent 應支援檔案過濾
- Rule: Agent 應正確顯示檔案資訊
- Rule: Agent 應處理特殊情況
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import allure
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """建立測試用 sandbox 目錄結構。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    # 建立多層目錄結構
    (sandbox / 'src').mkdir()
    (sandbox / 'src' / 'utils').mkdir()
    (sandbox / 'tests').mkdir()
    (sandbox / 'empty_dir').mkdir()
    (sandbox / 'node_modules').mkdir()

    # 建立各種檔案
    (sandbox / 'src' / 'main.py').write_text('def main(): pass\n', encoding='utf-8')
    (sandbox / 'src' / 'utils' / 'helper.py').write_text('def helper(): pass\n', encoding='utf-8')
    (sandbox / 'src' / 'utils' / 'config.py').write_text('CONFIG = {}\n', encoding='utf-8')
    (sandbox / 'tests' / 'test_main.py').write_text('def test(): pass\n', encoding='utf-8')

    # 建立不同副檔名的檔案
    (sandbox / 'README.md').write_text('# Project\n', encoding='utf-8')
    (sandbox / 'package.json').write_text('{"name": "test"}\n', encoding='utf-8')
    (sandbox / 'script.js').write_text("console.log('hi');\n", encoding='utf-8')

    # 建立隱藏檔案
    (sandbox / '.gitignore').write_text('*.pyc\n', encoding='utf-8')
    (sandbox / '.env').write_text('SECRET=abc\n', encoding='utf-8')

    # node_modules 中的檔案
    (sandbox / 'node_modules' / 'package.json').write_text('{}\n', encoding='utf-8')

    return sandbox


@pytest.fixture
def list_files(sandbox_dir: Path) -> Any:
    """建立 list_files 函數，已綁定 sandbox_root。"""
    from agent_core.tools.file_list import list_files_handler

    def _list(
        path: str = '.',
        recursive: bool = False,
        max_depth: int | None = None,
        pattern: str | None = None,
        exclude_dirs: list[str] | None = None,
        show_hidden: bool = False,
        show_details: bool = False,
    ) -> dict[str, Any]:
        return list_files_handler(
            path=path,
            sandbox_root=sandbox_dir,
            recursive=recursive,
            max_depth=max_depth,
            pattern=pattern,
            exclude_dirs=exclude_dirs,
            show_hidden=show_hidden,
            show_details=show_details,
        )

    return _list


# =============================================================================
# Rule: Agent 應能列出目錄內容
# =============================================================================


@allure.feature('列出檔案功能')
@allure.story('Agent 應能列出目錄內容')
class TestListDirectory:
    """測試基本目錄列出功能。"""

    @allure.title('列出目錄中的檔案')
    def test_list_directory_with_files(self, list_files: Any) -> None:
        """Scenario: 列出目錄中的檔案。

        Given 存在目錄 "src/" 包含多個檔案
        When 使用者要求列出 src/ 目錄的檔案
        Then Agent 應回傳目錄中的檔案列表
        """
        result = list_files('src')

        assert result['path'] == 'src'
        assert 'main.py' in result['files']
        assert 'utils' in result['directories']

    @allure.title('列出空目錄')
    def test_list_empty_directory(self, list_files: Any) -> None:
        """Scenario: 列出空目錄。

        Given 存在空目錄 "empty_dir/"
        When 使用者要求列出該目錄
        Then Agent 應告知目錄為空
        """
        result = list_files('empty_dir')

        assert result['path'] == 'empty_dir'
        assert len(result['files']) == 0
        assert len(result['directories']) == 0

    @allure.title('列出不存在的目錄')
    def test_list_nonexistent_directory(self, list_files: Any) -> None:
        """Scenario: 列出不存在的目錄。

        Given 目錄 "nonexistent/" 不存在
        When 使用者要求列出該目錄
        Then Agent 應告知目錄不存在
        """
        with pytest.raises(FileNotFoundError):
            list_files('nonexistent')

    @allure.title('列出當前工作目錄')
    def test_list_current_directory(self, list_files: Any) -> None:
        """Scenario: 列出當前工作目錄。

        Given 工作目錄包含多個檔案和子目錄
        When 使用者要求列出當前目錄的檔案
        Then Agent 應列出工作目錄的內容
        """
        result = list_files('.')

        # 應包含根目錄的檔案和目錄
        assert 'README.md' in result['files']
        assert 'src' in result['directories']
        assert 'tests' in result['directories']


# =============================================================================
# Rule: Agent 應支援遞迴列出檔案
# =============================================================================


@allure.feature('列出檔案功能')
@allure.story('Agent 應支援遞迴列出檔案')
class TestRecursiveListing:
    """測試遞迴列出功能。"""

    @allure.title('遞迴列出所有檔案')
    def test_recursive_list_all_files(self, list_files: Any) -> None:
        """Scenario: 遞迴列出所有檔案。

        Given 存在目錄結構包含多層子目錄
        When 使用者要求遞迴列出所有檔案
        Then Agent 應回傳所有子目錄中的檔案
        """
        result = list_files('.', recursive=True)

        all_files = result['all_files']
        # 應包含所有 Python 檔案
        assert 'src/main.py' in all_files
        assert 'src/utils/helper.py' in all_files
        assert 'src/utils/config.py' in all_files
        assert 'tests/test_main.py' in all_files

    @allure.title('限制遞迴深度')
    def test_limit_recursion_depth(self, list_files: Any) -> None:
        """Scenario: 限制遞迴深度。

        Given 存在深層巢狀的目錄結構
        When 使用者要求列出檔案並限制深度為 1
        Then Agent 應只列出 1 層以內的檔案
        """
        result = list_files('.', recursive=True, max_depth=1)

        all_files = result['all_files']
        # 深度 1 應包含 src/main.py
        assert 'src/main.py' in all_files
        # 但不應包含 src/utils/helper.py（深度 2）
        assert 'src/utils/helper.py' not in all_files


# =============================================================================
# Rule: Agent 應支援檔案過濾
# =============================================================================


@allure.feature('列出檔案功能')
@allure.story('Agent 應支援檔案過濾')
class TestFileFiltering:
    """測試檔案過濾功能。"""

    @allure.title('按副檔名過濾')
    def test_filter_by_extension(self, list_files: Any) -> None:
        """Scenario: 按副檔名過濾。

        Given 目錄包含 .py, .js, .md 等多種檔案
        When 使用者要求只列出 Python 檔案
        Then Agent 應只回傳 .py 檔案
        """
        result = list_files('.', recursive=True, pattern='*.py')

        all_files = result['all_files']
        # 應包含 .py 檔案
        assert 'src/main.py' in all_files
        assert 'tests/test_main.py' in all_files
        # 不應包含其他副檔名
        assert 'README.md' not in all_files
        assert 'script.js' not in all_files

    @allure.title('按檔名模式過濾')
    def test_filter_by_pattern(self, list_files: Any) -> None:
        """Scenario: 按檔名模式過濾。

        Given 目錄包含 test_*.py 和其他檔案
        When 使用者要求列出所有測試檔案
        Then Agent 應回傳符合 test_*.py 模式的檔案
        """
        result = list_files('.', recursive=True, pattern='test_*.py')

        all_files = result['all_files']
        # 應包含測試檔案
        assert 'tests/test_main.py' in all_files
        # 不應包含其他檔案
        assert 'src/main.py' not in all_files

    @allure.title('排除特定目錄')
    def test_exclude_directories(self, list_files: Any) -> None:
        """Scenario: 排除特定目錄。

        Given 目錄包含 node_modules/, src/
        When 使用者要求列出檔案並排除 node_modules
        Then Agent 應不列出 node_modules/ 中的檔案
        """
        result = list_files('.', recursive=True, exclude_dirs=['node_modules'])

        all_files = result['all_files']
        # 應包含 src 中的檔案
        assert 'src/main.py' in all_files
        # 不應包含 node_modules 中的檔案
        assert not any('node_modules' in f for f in all_files)


# =============================================================================
# Rule: Agent 應正確顯示檔案資訊
# =============================================================================


@allure.feature('列出檔案功能')
@allure.story('Agent 應正確顯示檔案資訊')
class TestFileInformation:
    """測試檔案資訊顯示。"""

    @allure.title('顯示檔案基本資訊')
    def test_show_basic_info(self, list_files: Any) -> None:
        """Scenario: 顯示檔案基本資訊。

        When 使用者要求列出目錄檔案
        Then 列表應包含檔案名稱
        And 列表應區分檔案與目錄
        """
        result = list_files('src')

        # 應區分檔案與目錄
        assert 'main.py' in result['files']
        assert 'utils' in result['directories']
        # 目錄不應出現在檔案列表中
        assert 'utils' not in result['files']

    @allure.title('顯示詳細檔案資訊')
    def test_show_detailed_info(self, list_files: Any, sandbox_dir: Path) -> None:
        """Scenario: 顯示詳細檔案資訊。

        When 使用者要求詳細列出目錄檔案
        Then 列表應包含檔案大小
        And 列表應包含最後修改時間
        """
        result = list_files('src', show_details=True)

        # 應包含檔案詳細資訊
        assert 'file_details' in result
        details = result['file_details']

        # 檢查 main.py 的詳細資訊
        main_py_info = next(d for d in details if d['name'] == 'main.py')
        assert 'size' in main_py_info
        assert 'modified' in main_py_info
        assert main_py_info['type'] == 'file'


# =============================================================================
# Rule: Agent 應處理特殊情況
# =============================================================================


@allure.feature('列出檔案功能')
@allure.story('Agent 應處理特殊情況')
class TestSpecialCases:
    """測試特殊情況處理。"""

    @allure.title('處理隱藏檔案')
    def test_hide_hidden_files_by_default(self, list_files: Any) -> None:
        """Scenario: 處理隱藏檔案。

        Given 目錄包含 .gitignore, .env 等隱藏檔案
        When 使用者要求列出檔案
        Then Agent 預設應不顯示隱藏檔案
        """
        result = list_files('.')

        # 預設不應顯示隱藏檔案
        assert '.gitignore' not in result['files']
        assert '.env' not in result['files']

    @allure.title('顯示隱藏檔案')
    def test_show_hidden_files_when_requested(self, list_files: Any) -> None:
        """Scenario: 顯示隱藏檔案。

        When 使用者要求列出所有檔案包含隱藏檔
        Then Agent 應包含隱藏檔案在列表中
        """
        result = list_files('.', show_hidden=True)

        # 應顯示隱藏檔案
        assert '.gitignore' in result['files']
        assert '.env' in result['files']

    @allure.title('測試路徑穿越攻擊防護')
    def test_block_path_traversal(self, list_files: Any) -> None:
        """測試路徑穿越攻擊防護。

        When 使用者嘗試列出 sandbox 外的目錄
        Then Agent 應拒絕並回傳安全性錯誤
        """
        with pytest.raises(PermissionError):
            list_files('../../../etc')

    @allure.title('處理無權限的目錄')
    def test_handle_permission_denied(self, list_files: Any, sandbox_dir: Path) -> None:
        """Scenario: 處理無權限的目錄。

        Given 存在無讀取權限的目錄
        When 使用者要求列出該目錄
        Then Agent 應告知權限不足
        """
        # 建立無權限目錄
        no_perm_dir = sandbox_dir / 'no_permission'
        no_perm_dir.mkdir()
        os.chmod(no_perm_dir, 0o000)

        try:
            with pytest.raises(PermissionError):
                list_files('no_permission')
        finally:
            # 恢復權限以便清理
            os.chmod(no_perm_dir, 0o755)
