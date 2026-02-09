"""File Read 工具測試模組。

根據 docs/features/file_read.feature 規格撰寫測試案例。
涵蓋：
- Rule: Agent 應能讀取指定檔案
- Rule: Agent 應正確處理各種檔案類型
- Rule: Agent 應處理檔案路徑安全性
- Rule: Agent 應處理大型檔案
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

    # 建立 Python 檔案
    (sandbox / 'src' / 'main.py').write_text("def hello():\n    print('world')\n", encoding='utf-8')

    # 建立空檔案
    (sandbox / 'empty.txt').write_text('', encoding='utf-8')

    # 建立 JSON 檔案
    (sandbox / 'config.json').write_text('{"key": "value"}\n', encoding='utf-8')

    # 建立 Markdown 檔案
    (sandbox / 'README.md').write_text('# Hello\n\nWorld\n', encoding='utf-8')

    # 建立二進位檔案（PNG 標頭）
    (sandbox / 'image.png').write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')

    # 建立敏感檔案
    (sandbox / '.env').write_text('SECRET_KEY=abc123\n', encoding='utf-8')

    # 建立大型檔案（超過 1MB）
    (sandbox / 'large_file.log').write_text('x' * (1024 * 1024 + 1), encoding='utf-8')

    # 建立 1000 行檔案
    lines = [f'line {i}' for i in range(1, 1001)]
    (sandbox / 'long_file.py').write_text('\n'.join(lines), encoding='utf-8')

    return sandbox


@pytest.fixture
def read_file(sandbox_dir: Path) -> Any:
    """建立 read_file 函數，已綁定 sandbox_root。"""
    from agent_core.tools.file_read import read_file_handler

    def _read(
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        return read_file_handler(
            path=path,
            sandbox_root=sandbox_dir,
            start_line=start_line,
            end_line=end_line,
        )

    return _read


# =============================================================================
# Rule: Agent 應能讀取指定檔案
# =============================================================================


@allure.feature('讀取檔案功能')
@allure.story('Agent 應能讀取指定檔案')
class TestReadFile:
    """測試基本檔案讀取功能。"""

    @allure.title('讀取存在的文字檔案')
    def test_read_existing_file(self, read_file: Any) -> None:
        """Scenario: 讀取存在的文字檔案。

        Given 存在檔案 "src/main.py" 包含 Python 程式碼
        When 使用者要求讀取該檔案
        Then Agent 應回傳檔案內容
        """
        result = read_file('src/main.py')

        assert result['content'] == "def hello():\n    print('world')\n"
        assert result['path'] == 'src/main.py'
        assert result['language'] == 'python'

    @allure.title('讀取不存在的檔案')
    def test_read_nonexistent_file(self, read_file: Any) -> None:
        """Scenario: 讀取不存在的檔案。

        Given 檔案 "nonexistent.py" 不存在
        When 使用者要求讀取該檔案
        Then 工具應回傳檔案不存在的錯誤
        """
        with pytest.raises(FileNotFoundError):
            read_file('nonexistent.py')

    @allure.title('讀取空檔案')
    def test_read_empty_file(self, read_file: Any) -> None:
        """Scenario: 讀取空檔案。

        Given 存在空檔案 "empty.txt"
        When 使用者要求讀取該檔案
        Then Agent 應回傳空內容
        """
        result = read_file('empty.txt')

        assert result['content'] == ''
        assert result['path'] == 'empty.txt'

    @allure.title('讀取檔案時回傳 SSE 事件資料')
    def test_read_file_returns_sse_events(self, read_file: Any) -> None:
        """Scenario: 讀取檔案時回傳 SSE 事件資料。

        Given 存在檔案 "src/main.py"
        When 使用者要求讀取該檔案
        Then 回傳結果應包含 sse_events
        And sse_events 應包含 file_open 事件
        And 事件資料應包含路徑、內容、語言
        """
        result = read_file('src/main.py')

        # 驗證包含 sse_events
        assert 'sse_events' in result
        assert isinstance(result['sse_events'], list)
        assert len(result['sse_events']) == 1

        # 驗證事件類型
        event = result['sse_events'][0]
        assert event['type'] == 'file_open'

        # 驗證事件資料
        event_data = event['data']
        assert event_data['path'] == 'src/main.py'
        assert event_data['content'] == "def hello():\n    print('world')\n"
        assert event_data['language'] == 'python'


# =============================================================================
# Rule: Agent 應正確處理各種檔案類型
# =============================================================================


@allure.feature('讀取檔案功能')
@allure.story('Agent 應正確處理各種檔案類型')
class TestFileTypeDetection:
    """測試檔案類型識別功能。"""

    @allure.title('讀取 Python 檔案')
    def test_detect_python_file(self, read_file: Any) -> None:
        """Scenario: 讀取 Python 檔案。

        Given 存在檔案 "src/main.py"
        When 使用者要求讀取該檔案
        Then Agent 應正確識別為 Python 程式碼
        """
        result = read_file('src/main.py')
        assert result['language'] == 'python'

    @allure.title('讀取 JSON 檔案')
    def test_detect_json_file(self, read_file: Any) -> None:
        """Scenario: 讀取 JSON 檔案。

        Given 存在檔案 "config.json"
        When 使用者要求讀取該檔案
        Then Agent 應正確識別為 JSON 格式
        """
        result = read_file('config.json')
        assert result['language'] == 'json'

    @allure.title('讀取 Markdown 檔案')
    def test_detect_markdown_file(self, read_file: Any) -> None:
        """Scenario: 讀取 Markdown 檔案。

        Given 存在檔案 "README.md"
        When 使用者要求讀取該檔案
        Then Agent 應正確識別為 Markdown 格式
        """
        result = read_file('README.md')
        assert result['language'] == 'markdown'

    @allure.title('嘗試讀取二進位檔案')
    def test_reject_binary_file(self, read_file: Any) -> None:
        """Scenario: 嘗試讀取二進位檔案。

        Given 存在二進位檔案 "image.png"
        When 使用者要求讀取該檔案
        Then Agent 應識別為二進位檔案並拒絕讀取
        """
        with pytest.raises(ValueError, match='二進位'):
            read_file('image.png')


# =============================================================================
# Rule: Agent 應處理檔案路徑安全性
# =============================================================================


@allure.feature('讀取檔案功能')
@allure.story('Agent 應處理檔案路徑安全性')
class TestPathSecurity:
    """測試路徑安全性。"""

    @allure.title('使用相對路徑讀取檔案')
    def test_resolve_relative_path(self, read_file: Any) -> None:
        """Scenario: 使用相對路徑讀取檔案。

        Given 存在檔案 "src/main.py"
        When 使用者以相對路徑讀取
        Then Agent 應正確解析路徑並回傳內容
        """
        result = read_file('src/main.py')
        assert result['content'] == "def hello():\n    print('world')\n"

    @allure.title('阻擋路徑穿越攻擊')
    def test_block_path_traversal(self, read_file: Any) -> None:
        """Scenario: 阻擋路徑穿越攻擊。

        When 使用者嘗試讀取 sandbox 外的檔案
        Then Agent 應拒絕並回傳安全性錯誤
        """
        with pytest.raises(PermissionError):
            read_file('../../../etc/passwd')

    @allure.title('阻擋隱藏的路徑穿越攻擊')
    def test_block_path_traversal_with_nested_path(self, read_file: Any) -> None:
        """阻擋隱藏的路徑穿越攻擊。"""
        with pytest.raises(PermissionError):
            read_file('src/../../outside.txt')

    @allure.title('阻擋讀取敏感檔案')
    def test_warn_sensitive_file(self, read_file: Any) -> None:
        """Scenario: 阻擋讀取敏感檔案。

        When 使用者要求讀取 ".env" 檔案
        Then Agent 應警告該檔案可能包含敏感資訊
        """
        with pytest.raises(PermissionError, match='敏感'):
            read_file('.env')


# =============================================================================
# Rule: Agent 應處理大型檔案
# =============================================================================


@allure.feature('讀取檔案功能')
@allure.story('Agent 應處理大型檔案')
class TestLargeFileHandling:
    """測試大型檔案處理功能。"""

    @allure.title('讀取超過大小限制的檔案')
    def test_reject_large_file(self, read_file: Any) -> None:
        """Scenario: 讀取超過大小限制的檔案。

        Given 存在超過 1MB 的大型檔案 "large_file.log"
        When 使用者要求讀取該檔案
        Then Agent 應告知檔案過大
        """
        with pytest.raises(ValueError, match='過大'):
            read_file('large_file.log')

    @allure.title('讀取檔案的指定行數範圍')
    def test_read_file_range(self, read_file: Any) -> None:
        """Scenario: 讀取檔案的指定行數範圍。

        Given 存在 1000 行的檔案 "long_file.py"
        When 使用者要求讀取第 50 到 100 行
        Then Agent 應只回傳指定範圍的內容
        And 回應應標示行號
        """
        result = read_file('long_file.py', start_line=50, end_line=100)

        content = result['content']
        # 應包含第 50 行
        assert '  50 | line 50' in content
        # 應包含第 100 行
        assert ' 100 | line 100' in content
        # 不應包含第 49 行
        assert 'line 49' not in content
        # 不應包含第 101 行
        assert 'line 101' not in content
