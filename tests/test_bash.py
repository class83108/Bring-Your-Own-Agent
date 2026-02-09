"""Bash 工具測試模組。

根據 docs/features/bash.feature 規格撰寫測試案例。
涵蓋：
- Rule: Agent 應能執行基本命令
- Rule: Agent 應正確處理命令輸出
- Rule: Agent 應處理命令執行狀態
- Rule: Agent 應確保命令執行安全性
- Rule: Agent 應支援常見開發命令
- Rule: Agent 應正確處理環境
"""

from __future__ import annotations

import subprocess
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

    # 建立測試檔案
    (sandbox / 'README.md').write_text('# Test Project\n', encoding='utf-8')
    (sandbox / 'test.txt').write_text('line1\nline2\nline3\n', encoding='utf-8')

    # 建立 Git 儲存庫（用於測試 git 命令）
    subprocess.run(['git', 'init'], cwd=str(sandbox), check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'],
        cwd=str(sandbox),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'],
        cwd=str(sandbox),
        check=True,
        capture_output=True,
    )

    return sandbox


@pytest.fixture
def bash_handler_fn(sandbox_dir: Path) -> Any:
    """建立 bash_handler 函數，已綁定 sandbox_root。"""
    from agent_core.tools.bash import bash_handler

    def _bash(
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> dict[str, Any]:
        return bash_handler(
            command=command,
            sandbox_root=sandbox_dir,
            timeout=timeout,
            working_dir=working_dir,
        )

    return _bash


# =============================================================================
# Rule: Agent 應能執行基本命令
# =============================================================================


@allure.feature('Bash 命令執行功能')
@allure.story('Agent 應能執行基本命令')
class TestBasicExecution:
    """測試基本命令執行功能。"""

    @allure.title('執行簡單命令')
    def test_execute_simple_command(self, bash_handler_fn: Any) -> None:
        """Scenario: 執行簡單命令。

        Given Agent 已啟動
        And "bash" 工具已註冊
        When 使用者要求 "執行 echo hello"
        Then Agent 應調用 "bash" 工具
        And Agent 應回傳命令執行結果
        """
        result = bash_handler_fn('echo "hello"')

        assert result['exit_code'] == 0
        assert 'hello' in result['stdout']
        assert result['stderr'] == ''
        assert result['command'] == 'echo "hello"'

    @allure.title('執行帶參數的命令')
    def test_execute_command_with_arguments(self, bash_handler_fn: Any) -> None:
        """Scenario: 執行帶參數的命令。

        When 使用者要求 "執行帶參數的命令"
        Then Agent 應正確傳遞所有參數
        """
        result = bash_handler_fn('printf "%s" "test"')

        assert result['exit_code'] == 0
        assert result['stdout'] == 'test'

    @allure.title('執行管道命令')
    def test_execute_pipe_command(self, bash_handler_fn: Any) -> None:
        """Scenario: 執行管道命令。

        When 使用者要求 "執行 cat test.txt | head -2"
        Then Agent 應正確處理管道
        And Agent 應回傳前 2 行內容
        """
        result = bash_handler_fn('cat test.txt | head -2')

        assert result['exit_code'] == 0
        assert 'line1' in result['stdout']
        assert 'line2' in result['stdout']
        assert 'line3' not in result['stdout']


# =============================================================================
# Rule: Agent 應確保命令執行安全性
# =============================================================================


@allure.feature('Bash 命令執行功能')
@allure.story('Agent 應確保命令執行安全性')
class TestSecurity:
    """測試安全性檢查功能。"""

    @allure.title('阻擋危險命令 - rm -rf /')
    def test_block_dangerous_command_rm_rf(self, bash_handler_fn: Any) -> None:
        """Scenario: 阻擋危險命令 - rm -rf /。

        Given Agent 已啟動
        When 使用者要求執行 "rm -rf /"
        Then Agent 應拒絕執行
        And Agent 應說明該命令具有危險性
        """
        with pytest.raises(PermissionError, match='危險'):
            bash_handler_fn('rm -rf /')

    @allure.title('阻擋危險命令 - dd')
    def test_block_dangerous_command_dd(self, bash_handler_fn: Any) -> None:
        """Scenario: 阻擋危險命令 - dd。

        When 使用者要求執行 dd 命令
        Then Agent 應拒絕執行
        """
        with pytest.raises(PermissionError, match='危險'):
            bash_handler_fn('dd if=/dev/zero of=/dev/sda')

    @allure.title('阻擋系統修改命令')
    def test_block_system_modify_command(self, bash_handler_fn: Any) -> None:
        """Scenario: 阻擋系統修改命令。

        When 使用者要求執行修改系統設定的命令
        Then Agent 應拒絕執行
        And Agent 應說明安全性考量
        """
        with pytest.raises(PermissionError, match='系統修改'):
            bash_handler_fn('sudo apt-get install curl')

    @allure.title('限制命令執行目錄')
    def test_restrict_working_directory(self, bash_handler_fn: Any) -> None:
        """Scenario: 限制命令執行目錄。

        Given 工作目錄為 sandbox
        When 使用者要求在 sandbox 外執行命令
        Then Agent 應拒絕執行
        And Agent 應說明只能在工作目錄內執行
        """
        with pytest.raises(PermissionError, match='sandbox'):
            bash_handler_fn('pwd', working_dir='../../etc')


# =============================================================================
# Rule: Agent 應正確處理命令輸出
# =============================================================================


@allure.feature('Bash 命令執行功能')
@allure.story('Agent 應正確處理命令輸出')
class TestOutputHandling:
    """測試命令輸出處理功能。"""

    @allure.title('處理標準輸出')
    def test_handle_stdout(self, bash_handler_fn: Any) -> None:
        """Scenario: 處理標準輸出。

        When 執行成功的命令
        Then Agent 應回傳 stdout 內容
        """
        result = bash_handler_fn('echo "stdout"')

        assert 'stdout' in result['stdout']
        assert result['stderr'] == ''

    @allure.title('處理標準錯誤')
    def test_handle_stderr(self, bash_handler_fn: Any) -> None:
        """Scenario: 處理標準錯誤。

        When 執行產生錯誤的命令
        Then Agent 應回傳 stderr 內容
        """
        result = bash_handler_fn('ls /nonexistent_directory_xyz 2>&1')

        assert result['exit_code'] != 0

    @allure.title('處理混合輸出')
    def test_handle_mixed_output(self, bash_handler_fn: Any) -> None:
        """Scenario: 處理混合輸出。

        When 執行同時產生 stdout 和 stderr 的命令
        Then Agent 應分別標示 stdout 和 stderr
        """
        result = bash_handler_fn('echo "out" && echo "err" >&2')

        assert 'out' in result['stdout']
        assert 'err' in result['stderr']

    @allure.title('處理無輸出的命令')
    def test_handle_no_output(self, bash_handler_fn: Any, sandbox_dir: Path) -> None:
        """Scenario: 處理無輸出的命令。

        When 執行不產生輸出的命令
        Then Agent 應告知命令已成功執行
        """
        result = bash_handler_fn('touch empty.txt')

        assert result['exit_code'] == 0
        assert result['stdout'] == ''
        assert (sandbox_dir / 'empty.txt').exists()

    @allure.title('處理大量輸出')
    def test_truncate_large_output(self, bash_handler_fn: Any) -> None:
        """Scenario: 處理大量輸出。

        When 執行產生大量輸出的命令
        Then Agent 應截斷過長的輸出
        And Agent 應告知輸出已被截斷
        """
        # 生成大量輸出（超過 100KB）
        result = bash_handler_fn('seq 1 100000')

        assert result['truncated'] is True
        assert '[輸出已截斷' in result['stdout']


# =============================================================================
# Rule: Agent 應處理命令執行狀態
# =============================================================================


@allure.feature('Bash 命令執行功能')
@allure.story('Agent 應處理命令執行狀態')
class TestExecutionStatus:
    """測試命令執行狀態處理功能。"""

    @allure.title('命令執行成功')
    def test_command_success(self, bash_handler_fn: Any) -> None:
        """Scenario: 命令執行成功。

        When 執行成功的命令
        Then Agent 應回傳 exit code 0
        """
        result = bash_handler_fn('true')

        assert result['exit_code'] == 0

    @allure.title('命令執行失敗')
    def test_command_failure(self, bash_handler_fn: Any) -> None:
        """Scenario: 命令執行失敗。

        When 執行失敗的命令
        Then Agent 應回傳非零 exit code
        """
        result = bash_handler_fn('false')

        assert result['exit_code'] != 0

    @allure.title('命令執行超時')
    def test_command_timeout(self, bash_handler_fn: Any) -> None:
        """Scenario: 命令執行超時。

        Given 設定命令超時為 1 秒
        When 執行耗時超過 1 秒的命令
        Then Agent 應終止該命令
        And Agent 應告知命令執行超時
        """
        with pytest.raises(TimeoutError, match='超時'):
            bash_handler_fn('sleep 10', timeout=1)


# =============================================================================
# Rule: Agent 應支援常見開發命令
# =============================================================================


@allure.feature('Bash 命令執行功能')
@allure.story('Agent 應支援常見開發命令')
class TestCommonCommands:
    """測試常見開發命令支援。"""

    @allure.title('執行 Git 命令')
    def test_git_command(self, bash_handler_fn: Any) -> None:
        """Scenario: 執行 Git 命令。

        When 使用者要求 "查看 git 狀態"
        Then Agent 應執行 "git status"
        """
        result = bash_handler_fn('git status')

        assert result['exit_code'] == 0
        # Git 初始狀態應顯示相關訊息
        assert len(result['stdout']) > 0

    @allure.title('執行測試命令')
    def test_pytest_command(self, bash_handler_fn: Any) -> None:
        """Scenario: 執行測試命令。

        When 使用者要求執行測試工具版本查詢
        Then 命令應能執行（可能成功或失敗）
        """
        result = bash_handler_fn('python --version')

        # Python 應該存在
        assert result['exit_code'] == 0
        assert 'Python' in result['stdout'] or 'Python' in result['stderr']

    @allure.title('執行 linting 命令')
    def test_linting_command(self, bash_handler_fn: Any, sandbox_dir: Path) -> None:
        """Scenario: 執行 linting 命令。

        When 使用者要求檢查程式碼
        Then Agent 應執行檢查工具
        """
        # 建立 Python 檔案
        (sandbox_dir / 'test.py').write_text('print("hello")\n', encoding='utf-8')

        result = bash_handler_fn('python -m py_compile test.py')

        assert result['exit_code'] == 0


# =============================================================================
# Rule: Agent 應正確處理環境
# =============================================================================


@allure.feature('Bash 命令執行功能')
@allure.story('Agent 應正確處理環境')
class TestEnvironment:
    """測試環境處理功能。"""

    @allure.title('使用正確的工作目錄')
    def test_use_correct_working_directory(
        self,
        bash_handler_fn: Any,
        sandbox_dir: Path,
    ) -> None:
        """Scenario: 使用正確的工作目錄。

        Given 工作目錄為 sandbox
        When 執行 "pwd"
        Then 結果應為 sandbox 路徑
        """
        result = bash_handler_fn('pwd')

        # 輸出應為 sandbox 目錄
        assert str(sandbox_dir) in result['stdout']

    @allure.title('存取環境變數')
    def test_access_environment_variables(self, bash_handler_fn: Any) -> None:
        """Scenario: 存取環境變數。

        When 執行訪問環境變數的命令
        Then 應能正確存取
        """
        # PATH 應該存在
        result = bash_handler_fn('echo $PATH')

        assert result['exit_code'] == 0
        assert len(result['stdout']) > 0

    @allure.title('隔離敏感環境變數')
    def test_mask_sensitive_information(self, bash_handler_fn: Any) -> None:
        """Scenario: 隔離敏感環境變數。

        When 執行可能洩漏 API 金鑰的命令
        Then Agent 應遮蔽敏感資訊
        """
        # 模擬輸出包含 API key
        result = bash_handler_fn('echo "sk-ant-api03-abc123def456"')

        # 應該被遮蔽
        assert 'sk-ant-api03' not in result['stdout']
        assert '[ANTHROPIC_API_KEY]' in result['stdout']

    @allure.title('Git 命令被限制在 sandbox 內')
    def test_git_isolated_to_sandbox(
        self,
        bash_handler_fn: Any,
        sandbox_dir: Path,
    ) -> None:
        """Scenario: Git 命令被限制在 sandbox 內。

        Given sandbox 是包含 .git 的目錄
        And sandbox 的上層目錄也是 git 儲存庫
        When 執行 "git status"
        Then 結果應為 sandbox 內的 git 狀態
        And 不應顯示上層目錄的 git 狀態
        """
        # sandbox_dir 已在 fixture 中初始化為 git repo
        # 在 sandbox 中新增檔案
        test_file = sandbox_dir / 'new_file.txt'
        test_file.write_text('test content\n', encoding='utf-8')

        result = bash_handler_fn('git status')

        # 應成功執行（sandbox 內有 .git）
        assert result['exit_code'] == 0
        # 應顯示 sandbox 內的未追蹤檔案
        assert 'new_file.txt' in result['stdout']
        # 不應提及 sandbox 外的檔案（例如專案根目錄的檔案）
        # 驗證方式：git status 輸出應只包含 sandbox 相對路徑
        assert 'agent_core' not in result['stdout']  # 專案目錄名稱不應出現
