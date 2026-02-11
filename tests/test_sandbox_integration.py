"""Sandbox 與 Tool Handler 整合測試模組。

根據 docs/features/sandbox.feature 規格撰寫測試案例。
涵蓋：
- Rule: Tool handler 應透過 Sandbox 介面操作
- 驗證 create_default_registry 接受 Sandbox 參數
- 驗證所有工具透過 Sandbox 正常運作
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import allure
import pytest

from agent_core.sandbox import LocalSandbox
from agent_core.tools.setup import create_default_registry

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sandbox_dir(tmp_path: Path) -> Path:
    """建立測試用 sandbox 目錄並準備測試檔案。"""
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    (sandbox / 'hello.py').write_text("print('hello')\n", encoding='utf-8')
    (sandbox / 'data.txt').write_text('line1\nline2\nline3\n', encoding='utf-8')

    src = sandbox / 'src'
    src.mkdir()
    (src / 'main.py').write_text('def main(): pass\n', encoding='utf-8')

    # 初始化 git repo（bash 工具需要）
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
def sandbox(sandbox_dir: Path) -> LocalSandbox:
    """建立 LocalSandbox 實例。"""
    return LocalSandbox(root=sandbox_dir)


@pytest.fixture
def registry(sandbox: LocalSandbox) -> Any:
    """使用 Sandbox 建立預設工具註冊表。"""
    return create_default_registry(sandbox=sandbox)


# =============================================================================
# Rule: Tool handler 應透過 Sandbox 介面操作
# =============================================================================


@allure.feature('Sandbox 沙箱環境')
@allure.story('Tool handler 應透過 Sandbox 介面操作')
class TestRegistryWithSandbox:
    """測試 create_default_registry 接受 Sandbox 參數。"""

    @allure.title('create_default_registry 接受 Sandbox 參數')
    def test_create_registry_with_sandbox(self, registry: Any) -> None:
        """Scenario: create_default_registry 接受 Sandbox 參數。

        Given 一個 LocalSandbox 實例
        When 使用該 Sandbox 建立預設工具註冊表
        Then 所有檔案工具應已註冊
        """
        tools = registry.list_tools()
        assert 'read_file' in tools
        assert 'edit_file' in tools
        assert 'list_files' in tools
        assert 'bash' in tools
        assert 'grep_search' in tools
        assert 'think' in tools


@allure.feature('Sandbox 沙箱環境')
@allure.story('Tool handler 應透過 Sandbox 介面操作')
class TestToolsViaSandbox:
    """測試透過 Sandbox 的工具實際運作。"""

    @allure.title('read_file 透過 Sandbox 讀取檔案')
    async def test_read_file_via_sandbox(self, registry: Any) -> None:
        """read_file 工具應透過 Sandbox 讀取檔案內容。"""
        result = await registry.execute('read_file', {'path': 'hello.py'})
        assert result['content'] == "print('hello')\n"
        assert result['language'] == 'python'

    @allure.title('edit_file 透過 Sandbox 編輯檔案')
    async def test_edit_file_via_sandbox(self, registry: Any) -> None:
        """edit_file 工具應透過 Sandbox 編輯檔案。"""
        result = await registry.execute(
            'edit_file',
            {
                'path': 'hello.py',
                'old_content': "print('hello')",
                'new_content': "print('world')",
            },
        )
        assert result['modified'] is True

        # 驗證修改結果
        read_result = await registry.execute('read_file', {'path': 'hello.py'})
        assert "print('world')" in read_result['content']

    @allure.title('list_files 透過 Sandbox 列出目錄')
    async def test_list_files_via_sandbox(self, registry: Any) -> None:
        """list_files 工具應透過 Sandbox 列出目錄內容。"""
        result = await registry.execute('list_files', {'path': '.'})
        assert 'hello.py' in result['files']
        assert 'src' in result['directories']

    @allure.title('bash 透過 Sandbox 執行指令')
    async def test_bash_via_sandbox(self, registry: Any) -> None:
        """bash 工具應透過 Sandbox 執行指令。"""
        result = await registry.execute('bash', {'command': 'echo sandbox_test'})
        assert result['exit_code'] == 0
        assert 'sandbox_test' in result['stdout']

    @allure.title('grep_search 透過 Sandbox 搜尋')
    async def test_grep_search_via_sandbox(self, registry: Any) -> None:
        """grep_search 工具應透過 Sandbox 搜尋程式碼。"""
        result = await registry.execute(
            'grep_search',
            {
                'pattern': 'def main',
                'include': ['*.py'],
            },
        )
        assert result['total_matches'] >= 1
        assert any('main.py' in m['file'] for m in result['matches'])

    @allure.title('edit_file 建立新檔案透過 Sandbox')
    async def test_create_file_via_sandbox(self, registry: Any) -> None:
        """edit_file 工具建立新檔案應透過 Sandbox。"""
        result = await registry.execute(
            'edit_file',
            {
                'path': 'new_module.py',
                'new_content': 'x = 1\n',
                'create_if_missing': True,
            },
        )
        assert result['created'] is True

        read_result = await registry.execute('read_file', {'path': 'new_module.py'})
        assert read_result['content'] == 'x = 1\n'

    @allure.title('bash 路徑穿越應被阻擋')
    async def test_bash_path_traversal_blocked(self, registry: Any) -> None:
        """bash 工具的 working_dir 路徑穿越應被阻擋。"""
        with pytest.raises(PermissionError):
            await registry.execute(
                'bash',
                {
                    'command': 'pwd',
                    'working_dir': '../../etc',
                },
            )

    @allure.title('read_file 路徑穿越應被阻擋')
    async def test_read_file_path_traversal_blocked(self, registry: Any) -> None:
        """read_file 工具的路徑穿越應被阻擋。"""
        with pytest.raises(PermissionError):
            await registry.execute(
                'read_file',
                {
                    'path': '../../../etc/passwd',
                },
            )
