"""Grep Search 工具測試模組。

根據 docs/features/code_search.feature 規格撰寫測試案例。
涵蓋：
- Rule: Agent 應能搜尋程式碼內容
- Rule: Agent 應支援正則表達式搜尋
- Rule: Agent 應支援搜尋範圍限制
- Rule: Agent 應格式化搜尋結果
- Rule: Agent 應支援進階搜尋功能
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
    """建立測試用 sandbox 目錄，包含多種程式碼檔案。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    # 建立 src 目錄結構
    (sandbox / 'src').mkdir()
    (sandbox / 'src' / 'utils').mkdir()
    (sandbox / 'tests').mkdir()

    # 建立 Python 檔案 - 主程式
    (sandbox / 'src' / 'main.py').write_text(
        '''"""主程式模組。"""

import logging

logger = logging.getLogger(__name__)


def process_data(data: list) -> dict:
    """處理資料。"""
    logger.info("Processing data")
    return {"result": data}


class UserService:
    """使用者服務。"""

    def get_user(self, user_id: int):
        # TODO: 實作取得使用者
        pass

    def create_user(self, name: str):
        # FIXME: 需要加入驗證
        logger.debug("Creating user")
        pass
''',
        encoding='utf-8',
    )

    # 建立 Python 檔案 - 工具模組
    (sandbox / 'src' / 'utils' / 'helpers.py').write_text(
        '''"""工具函數模組。"""

import logging

logger = logging.getLogger(__name__)


def format_string(text: str) -> str:
    """格式化字串。"""
    return text.strip().lower()


def validate_config(config: dict) -> bool:
    """驗證設定。"""
    # TODO: 完整驗證邏輯
    return "key" in config
''',
        encoding='utf-8',
    )

    # 建立測試檔案
    (sandbox / 'tests' / 'test_main.py').write_text(
        '''"""測試主程式。"""

import pytest


def test_process_data():
    """測試資料處理。"""
    pass


def test_user_service():
    """測試使用者服務。"""
    pass


class TestHelpers:
    def test_format_string(self):
        pass

    def test_validate_config(self):
        pass
''',
        encoding='utf-8',
    )

    # 建立 JavaScript 檔案
    (sandbox / 'src' / 'app.js').write_text(
        """// 應用程式入口
const logger = require("./logger");

function fetchData(url) {
    logger.info("Fetching: " + url);
    return fetch(url);
}

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        return fetchData(this.baseUrl + endpoint);
    }
}

module.exports = { fetchData, ApiClient };
""",
        encoding='utf-8',
    )

    # 建立 JSON 設定檔
    (sandbox / 'config.json').write_text(
        '{"key": "value", "debug": true}\n',
        encoding='utf-8',
    )

    # 建立 .git 目錄（應被排除）
    (sandbox / '.git').mkdir()
    (sandbox / '.git' / 'config').write_text('gitconfig', encoding='utf-8')

    # 建立 node_modules（應被排除）
    (sandbox / 'node_modules').mkdir()
    (sandbox / 'node_modules' / 'package.js').write_text(
        'const logger = "fake";',
        encoding='utf-8',
    )

    # 建立 __pycache__（應被排除）
    (sandbox / '__pycache__').mkdir()
    (sandbox / '__pycache__' / 'main.cpython-311.pyc').write_bytes(b'\x00\x00')

    return sandbox


@pytest.fixture
def grep_search(sandbox_dir: Path) -> Any:
    """建立 grep_search 函數，已綁定 sandbox_root。"""
    from agent_core.tools.grep_search import grep_search_handler

    def _search(
        pattern: str,
        path: str = '.',
        include: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        case_sensitive: bool = True,
        whole_word: bool = False,
        context_lines: int = 0,
        max_results: int = 100,
    ) -> dict[str, Any]:
        return grep_search_handler(
            pattern=pattern,
            path=path,
            sandbox_root=sandbox_dir,
            include=include,
            exclude_dirs=exclude_dirs,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            context_lines=context_lines,
            max_results=max_results,
        )

    return _search


# =============================================================================
# Rule: Agent 應能搜尋程式碼內容
# =============================================================================


@allure.feature('程式碼搜尋功能')
@allure.story('Agent 應能搜尋程式碼內容')
class TestBasicSearch:
    """測試基本搜尋功能。"""

    @allure.title('搜尋特定字串')
    def test_search_simple_string(self, grep_search: Any) -> None:
        """Scenario: 搜尋特定字串。

        Given 專案中有多個 Python 檔案
        When 使用者要求搜尋所有使用 logger 的地方
        Then Agent 應回傳所有包含 "logger" 的程式碼位置
        """
        result = grep_search('logger')

        assert result['total_matches'] > 0
        assert len(result['matches']) > 0

        # 應在多個檔案中找到 logger
        files = {m['file'] for m in result['matches']}
        assert 'src/main.py' in files
        assert 'src/utils/helpers.py' in files

    @allure.title('搜尋函數定義')
    def test_search_function_definition(self, grep_search: Any) -> None:
        """Scenario: 搜尋函數定義。

        When 使用者要求找出 process_data 函數在哪裡定義
        Then Agent 應搜尋 "def process_data"
        And Agent 應回傳函數定義的檔案與行號
        """
        result = grep_search('def process_data')

        assert result['total_matches'] == 1
        match = result['matches'][0]
        assert match['file'] == 'src/main.py'
        assert match['line_number'] > 0
        assert 'def process_data' in match['line_content']

    @allure.title('搜尋類別定義')
    def test_search_class_definition(self, grep_search: Any) -> None:
        """Scenario: 搜尋類別定義。

        When 使用者要求找出 UserService 類別
        Then Agent 應搜尋 "class UserService"
        And Agent 應回傳類別定義的位置
        """
        result = grep_search('class UserService')

        assert result['total_matches'] == 1
        match = result['matches'][0]
        assert match['file'] == 'src/main.py'
        assert 'class UserService' in match['line_content']

    @allure.title('搜尋無結果')
    def test_search_no_results(self, grep_search: Any) -> None:
        """Scenario: 搜尋無結果。

        When 使用者搜尋不存在的內容
        Then Agent 應告知找不到匹配結果
        """
        result = grep_search('nonexistent_xyz_123')

        assert result['total_matches'] == 0
        assert len(result['matches']) == 0


# =============================================================================
# Rule: Agent 應支援正則表達式搜尋
# =============================================================================


@allure.feature('程式碼搜尋功能')
@allure.story('Agent 應支援正則表達式搜尋')
class TestRegexSearch:
    """測試正則表達式搜尋。"""

    @allure.title('使用簡單正則表達式')
    def test_regex_or_pattern(self, grep_search: Any) -> None:
        """Scenario: 使用簡單正則表達式。

        When 使用者要求搜尋所有 TODO 或 FIXME 註解
        Then Agent 應使用正則表達式 "TODO|FIXME"
        And Agent 應回傳所有匹配結果
        """
        result = grep_search('TODO|FIXME')

        assert result['total_matches'] >= 2
        contents = [m['line_content'] for m in result['matches']]
        assert any('TODO' in c for c in contents)
        assert any('FIXME' in c for c in contents)

    @allure.title('搜尋特定模式的函數')
    def test_regex_function_pattern(self, grep_search: Any) -> None:
        """Scenario: 搜尋特定模式的函數。

        When 使用者要求找出所有 test_ 開頭的函數
        Then Agent 應使用正則表達式 "def test_\\w+"
        And Agent 應回傳所有測試函數
        """
        result = grep_search(r'def test_\w+')

        assert result['total_matches'] >= 2
        for match in result['matches']:
            assert 'def test_' in match['line_content']
            assert match['file'].startswith('tests/')


# =============================================================================
# Rule: Agent 應支援搜尋範圍限制
# =============================================================================


@allure.feature('程式碼搜尋功能')
@allure.story('Agent 應支援搜尋範圍限制')
class TestSearchScope:
    """測試搜尋範圍限制。"""

    @allure.title('限制搜尋特定目錄')
    def test_limit_to_directory(self, grep_search: Any) -> None:
        """Scenario: 限制搜尋特定目錄。

        When 使用者要求只在 src/ 目錄搜尋
        Then Agent 應只搜尋 src/ 目錄下的檔案
        """
        result = grep_search('logger', path='src')

        assert result['total_matches'] > 0
        for match in result['matches']:
            assert match['file'].startswith('src/')

    @allure.title('限制搜尋特定檔案類型')
    def test_limit_to_file_type(self, grep_search: Any) -> None:
        """Scenario: 限制搜尋特定檔案類型。

        When 使用者要求只搜尋 Python 檔案
        Then Agent 應只搜尋 .py 檔案
        """
        result = grep_search('logger', include=['*.py'])

        assert result['total_matches'] > 0
        for match in result['matches']:
            assert match['file'].endswith('.py')

    @allure.title('排除特定目錄')
    def test_exclude_directory(self, grep_search: Any) -> None:
        """Scenario: 排除特定目錄。

        When 使用者要求搜尋但排除 tests/ 目錄
        Then Agent 應跳過 tests/ 目錄
        """
        result = grep_search('def test_', exclude_dirs=['tests'])

        # 不應找到測試函數
        for match in result['matches']:
            assert not match['file'].startswith('tests/')

    @allure.title('排除常見非程式碼目錄')
    def test_exclude_common_directories(self, grep_search: Any) -> None:
        """Scenario: 排除常見非程式碼目錄。

        When Agent 執行搜尋
        Then 預設應排除 node_modules/, .git/, __pycache__/ 等目錄
        """
        # node_modules 中有 logger，但不應被搜尋到
        result = grep_search('logger')

        files = {m['file'] for m in result['matches']}
        assert not any('node_modules' in f for f in files)
        assert not any('.git' in f for f in files)
        assert not any('__pycache__' in f for f in files)


# =============================================================================
# Rule: Agent 應格式化搜尋結果
# =============================================================================


@allure.feature('程式碼搜尋功能')
@allure.story('Agent 應格式化搜尋結果')
class TestResultFormatting:
    """測試搜尋結果格式化。"""

    @allure.title('顯示匹配行與檔案路徑')
    def test_result_contains_file_path(self, grep_search: Any) -> None:
        """Scenario: 顯示匹配行與檔案路徑。

        When 搜尋找到匹配結果
        Then 應顯示匹配的檔案路徑
        """
        result = grep_search('class UserService')

        match = result['matches'][0]
        assert 'file' in match
        assert match['file'] == 'src/main.py'

    @allure.title('顯示行號')
    def test_result_contains_line_number(self, grep_search: Any) -> None:
        """Scenario: 顯示行號。

        When 搜尋找到匹配結果
        Then 應顯示行號
        """
        result = grep_search('class UserService')

        match = result['matches'][0]
        assert 'line_number' in match
        assert isinstance(match['line_number'], int)
        assert match['line_number'] > 0

    @allure.title('顯示匹配行的內容')
    def test_result_contains_line_content(self, grep_search: Any) -> None:
        """Scenario: 顯示匹配行的內容。

        When 搜尋找到匹配結果
        Then 應顯示匹配行的內容
        """
        result = grep_search('class UserService')

        match = result['matches'][0]
        assert 'line_content' in match
        assert 'class UserService' in match['line_content']

    @allure.title('顯示周圍上下文')
    def test_result_with_context_lines(self, grep_search: Any) -> None:
        """Scenario: 顯示周圍上下文。

        When 使用者要求顯示上下文
        Then 應顯示匹配行的前後各 N 行
        """
        result = grep_search('class UserService', context_lines=2)

        match = result['matches'][0]
        assert 'context_before' in match
        assert 'context_after' in match
        assert len(match['context_before']) <= 2
        assert len(match['context_after']) <= 2

    @allure.title('分組顯示結果')
    def test_results_grouped_by_file(self, grep_search: Any) -> None:
        """Scenario: 分組顯示結果。

        When 搜尋結果來自多個檔案
        Then 結果應包含檔案分組資訊
        """
        result = grep_search('logger')

        # 結果應包含 files_summary
        assert 'files_summary' in result
        assert len(result['files_summary']) > 1

    @allure.title('限制結果數量')
    def test_limit_results_count(self, grep_search: Any) -> None:
        """Scenario: 限制結果數量。

        When 搜尋結果過多
        Then Agent 應限制顯示數量
        And Agent 應告知總共找到多少結果
        """
        result = grep_search('def', max_results=3)

        assert len(result['matches']) <= 3
        assert 'total_matches' in result


# =============================================================================
# Rule: Agent 應支援進階搜尋功能
# =============================================================================


class TestAdvancedSearch:
    """測試進階搜尋功能。"""

    def test_case_insensitive_search(self, grep_search: Any) -> None:
        """Scenario: 大小寫不敏感搜尋。

        When 使用者要求搜尋 config 忽略大小寫
        Then Agent 應匹配 "config", "Config", "CONFIG" 等
        """
        # 先確認大小寫敏感時的結果
        sensitive_result = grep_search('config', case_sensitive=True)

        # 大小寫不敏感時應找到相同或更多
        insensitive_result = grep_search('config', case_sensitive=False)

        # 應至少找到 config 相關內容
        assert insensitive_result['total_matches'] >= sensitive_result['total_matches']

    def test_whole_word_match(self, grep_search: Any) -> None:
        """Scenario: 全詞匹配。

        When 使用者要求搜尋完整單詞 "test"
        Then 不應匹配 "testing" 或 "contest"
        And 應只匹配獨立的 "test" 單詞
        """
        # 非全詞匹配應找到 test 和其他包含 test 的字
        partial_result = grep_search('test', whole_word=False)

        # 全詞匹配應找到更少
        whole_result = grep_search('test', whole_word=True)

        # 非全詞匹配應找到更多結果（包含 test_ 開頭的函數名）
        assert partial_result['total_matches'] >= whole_result['total_matches']

        # 驗證全詞匹配結果確實包含 test
        for match in whole_result['matches']:
            assert 'test' in match['line_content'].lower()


# =============================================================================
# Rule: 路徑安全性
# =============================================================================


class TestPathSecurity:
    """測試路徑安全性。"""

    def test_block_path_traversal(self, grep_search: Any) -> None:
        """阻擋路徑穿越攻擊。"""
        with pytest.raises(PermissionError):
            grep_search('password', path='../../../etc')

    def test_search_within_sandbox(self, grep_search: Any) -> None:
        """確保只在 sandbox 內搜尋。"""
        result = grep_search('logger', path='.')

        for match in result['matches']:
            # 檔案路徑不應包含 ..
            assert '..' not in match['file']
