"""LocalSandbox 測試模組。

根據 docs/features/sandbox.feature 規格撰寫測試案例。
涵蓋：
- Rule: Sandbox 應提供路徑驗證與指令執行介面
- Rule: Sandbox 應阻擋超出沙箱範圍的路徑存取
- Rule: LocalSandbox 應在本地檔案系統的指定根目錄內操作
"""

from __future__ import annotations

from pathlib import Path

import allure
import pytest

from agent_core.sandbox import LocalSandbox

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """建立測試用 sandbox 目錄並準備測試檔案。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    # 建立測試檔案
    (sandbox / 'hello.py').write_text("print('hello')", encoding='utf-8')
    (sandbox / 'data.txt').write_text('some data', encoding='utf-8')
    (sandbox / 'a.py').write_text('a = 1', encoding='utf-8')
    (sandbox / 'b.py').write_text('b = 2', encoding='utf-8')

    # 建立子目錄與檔案
    src = sandbox / 'src'
    src.mkdir()
    (src / 'main.py').write_text('def main(): pass', encoding='utf-8')

    return sandbox


@pytest.fixture
def sandbox(sandbox_dir: Path) -> LocalSandbox:
    """建立 LocalSandbox 實例。"""
    return LocalSandbox(root=sandbox_dir)


# =============================================================================
# Rule: Sandbox 應提供路徑驗證與指令執行介面
# =============================================================================


@allure.feature('Sandbox 沙箱環境')
@allure.story('Sandbox 應提供路徑驗證與指令執行介面')
class TestSandboxInterface:
    """測試 Sandbox 的核心介面。"""

    @allure.title('驗證合法路徑')
    def test_validate_path(self, sandbox: LocalSandbox) -> None:
        """validate_path 應回傳正規化後的絕對路徑。"""
        result = sandbox.validate_path('src/main.py')
        assert 'src/main.py' in result or 'main.py' in result

    @allure.title('validate_path 回傳根目錄')
    def test_validate_path_dot(self, sandbox: LocalSandbox, sandbox_dir: Path) -> None:
        """validate_path('.') 應回傳沙箱根目錄。"""
        result = sandbox.validate_path('.')
        assert result == str(sandbox_dir)

    @allure.title('執行指令')
    async def test_exec(self, sandbox: LocalSandbox) -> None:
        """Scenario: 執行指令。

        Given 沙箱已初始化
        When 透過沙箱執行指令 "echo hello"
        Then 應回傳 exit_code 為 0
        And stdout 應包含 "hello"
        """
        result = await sandbox.exec('echo hello')
        assert result['exit_code'] == 0
        assert 'hello' in result['stdout']

    @allure.title('exec 指定 working_dir')
    async def test_exec_with_working_dir(self, sandbox: LocalSandbox) -> None:
        """exec 應支援指定工作目錄。"""
        result = await sandbox.exec('ls', working_dir='src')
        assert 'main.py' in result['stdout']

    @allure.title('exec 超時')
    async def test_exec_timeout(self, sandbox: LocalSandbox) -> None:
        """Scenario: 指令執行超時。

        Given 沙箱已初始化
        When 透過沙箱執行耗時超過超時設定的指令
        Then 應拋出 TimeoutError
        """
        with pytest.raises(TimeoutError):
            await sandbox.exec('sleep 10', timeout=1)


# =============================================================================
# Rule: Sandbox 應阻擋超出沙箱範圍的路徑存取
# =============================================================================


@allure.feature('Sandbox 沙箱環境')
@allure.story('Sandbox 應阻擋超出沙箱範圍的路徑存取')
class TestPathSecurity:
    """測試路徑安全性。"""

    @allure.title('阻擋路徑穿越攻擊')
    def test_block_path_traversal(self, sandbox: LocalSandbox) -> None:
        """Scenario: 阻擋路徑穿越攻擊。

        Given 沙箱已初始化
        When 透過沙箱驗證 "../../../etc/passwd"
        Then 應拋出 PermissionError
        """
        with pytest.raises(PermissionError):
            sandbox.validate_path('../../../etc/passwd')

    @allure.title('阻擋絕對路徑存取')
    def test_block_absolute_path(self, sandbox: LocalSandbox) -> None:
        """Scenario: 阻擋絕對路徑存取。

        Given 沙箱已初始化
        When 透過沙箱驗證 "/etc/passwd"
        Then 應拋出 PermissionError
        """
        with pytest.raises(PermissionError):
            sandbox.validate_path('/etc/passwd')

    @allure.title('路徑穿越 - exec working_dir')
    async def test_exec_path_traversal(self, sandbox: LocalSandbox) -> None:
        """exec 的 working_dir 也應阻擋路徑穿越。"""
        with pytest.raises(PermissionError):
            await sandbox.exec('pwd', working_dir='../../')


# =============================================================================
# Rule: LocalSandbox 應在本地檔案系統的指定根目錄內操作
# =============================================================================


@allure.feature('Sandbox 沙箱環境')
@allure.story('LocalSandbox 應在本地檔案系統的指定根目錄內操作')
class TestLocalSandbox:
    """測試 LocalSandbox 的本地操作。"""

    @allure.title('root 屬性回傳根目錄')
    def test_root_property(self, sandbox: LocalSandbox, sandbox_dir: Path) -> None:
        """root 屬性應回傳沙箱根目錄的絕對路徑。"""
        assert sandbox.root == sandbox_dir

    @allure.title('在根目錄內執行指令')
    async def test_exec_in_root(
        self,
        sandbox: LocalSandbox,
        sandbox_dir: Path,
    ) -> None:
        """Scenario: 在根目錄內執行指令。

        Given LocalSandbox 根目錄為暫存目錄
        When 透過沙箱執行指令 "pwd"
        Then stdout 應包含該暫存目錄的路徑
        """
        result = await sandbox.exec('pwd')
        assert str(sandbox_dir) in result['stdout']

    @allure.title('exec 失敗指令回傳非零 exit_code')
    async def test_exec_failure(self, sandbox: LocalSandbox) -> None:
        """執行失敗的指令應回傳非零 exit_code。"""
        result = await sandbox.exec('false')
        assert result['exit_code'] != 0

    @allure.title('exec 不存在的工作目錄應拋出 FileNotFoundError')
    async def test_exec_nonexistent_working_dir(self, sandbox: LocalSandbox) -> None:
        """指定不存在的工作目錄應拋出錯誤。"""
        with pytest.raises(FileNotFoundError):
            await sandbox.exec('pwd', working_dir='nonexistent_dir')
