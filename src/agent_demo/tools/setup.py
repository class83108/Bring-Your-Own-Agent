"""工具註冊工廠模組。

提供建立預設工具註冊表的工廠函數。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_demo.tools.bash import bash_handler
from agent_demo.tools.file_list import list_files_handler
from agent_demo.tools.file_read import read_file_handler
from agent_demo.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def create_default_registry(
    sandbox_root: Path,
    lock_provider: Any | None = None,
) -> ToolRegistry:
    """建立預設的工具註冊表，包含所有內建工具。

    Args:
        sandbox_root: sandbox 根目錄，用於限制檔案操作範圍
        lock_provider: 鎖提供者（可選，用於避免檔案競爭）

    Returns:
        已註冊所有內建工具的 ToolRegistry
    """
    registry = ToolRegistry(lock_provider=lock_provider)

    # 註冊 read_file 工具
    _register_read_file(registry, sandbox_root)

    # 註冊 list_files 工具
    _register_list_files(registry, sandbox_root)

    # 註冊 bash 工具
    _register_bash(registry, sandbox_root)

    logger.info('預設工具註冊表已建立', extra={'tools': registry.list_tools()})
    return registry


def _register_read_file(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 read_file 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """read_file handler 閉包，綁定 sandbox_root。"""
        return read_file_handler(
            path=path,
            sandbox_root=sandbox_root,
            start_line=start_line,
            end_line=end_line,
        )

    registry.register(
        name='read_file',
        description="""讀取專案中的檔案內容。

        使用時機：
        - 使用者要求查看程式碼
        - 需要理解現有實作再進行修改
        - 分析錯誤訊息中提到的檔案
        - 檢查配置檔案內容

        回傳：檔案的文字內容、路徑與程式語言識別。""",
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': '檔案路徑（相對於 sandbox 根目錄）',
                },
                'start_line': {
                    'type': 'integer',
                    'description': '起始行號（可選，預設為 1）',
                },
                'end_line': {
                    'type': 'integer',
                    'description': '結束行號（可選，預設讀到檔尾）',
                },
            },
            'required': ['path'],
        },
        handler=_handler,
        file_param='path',
    )


def _register_list_files(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 list_files 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        path: str = '.',
        recursive: bool = False,
        max_depth: int | None = None,
        pattern: str | None = None,
        exclude_dirs: list[str] | None = None,
        show_hidden: bool = False,
        show_details: bool = False,
    ) -> dict[str, Any]:
        """list_files handler 閉包，綁定 sandbox_root。"""
        return list_files_handler(
            path=path,
            sandbox_root=sandbox_root,
            recursive=recursive,
            max_depth=max_depth,
            pattern=pattern,
            exclude_dirs=exclude_dirs,
            show_hidden=show_hidden,
            show_details=show_details,
        )

    registry.register(
        name='list_files',
        description="""列出目錄中的檔案和子目錄。

        使用時機：
        - 使用者要求查看專案結構
        - 需要了解目錄內容再進行操作
        - 尋找特定類型或模式的檔案
        - 探索不熟悉的專案

        回傳：檔案列表、目錄列表，以及可選的詳細資訊（大小、修改時間）。""",
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': '目錄路徑（相對於 sandbox 根目錄，預設為當前目錄 "."）',
                },
                'recursive': {
                    'type': 'boolean',
                    'description': '是否遞迴列出所有子目錄中的檔案（預設 false）',
                },
                'max_depth': {
                    'type': 'integer',
                    'description': '最大遞迴深度（僅在 recursive=true 時有效，預設無限制）',
                },
                'pattern': {
                    'type': 'string',
                    'description': '檔案名稱模式，如 "*.py", "test_*.py"（預設顯示所有檔案）',
                },
                'exclude_dirs': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': '要排除的目錄名稱列表，如 ["node_modules", ".git"]',
                },
                'show_hidden': {
                    'type': 'boolean',
                    'description': '是否顯示隱藏檔案（以 . 開頭的檔案，預設 false）',
                },
                'show_details': {
                    'type': 'boolean',
                    'description': '是否顯示詳細資訊（檔案大小、修改時間，預設 false）',
                },
            },
            'required': [],
        },
        handler=_handler,
        file_param=None,  # list_files 只讀取目錄結構，不需要檔案鎖定
    )


def _register_bash(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 bash 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> dict[str, Any]:
        """bash handler 閉包，綁定 sandbox_root。"""
        return bash_handler(
            command=command,
            sandbox_root=sandbox_root,
            timeout=timeout,
            working_dir=working_dir,
        )

    registry.register(
        name='bash',
        description="""執行 bash 命令來操作專案環境。

        常見用途：
        - 執行測試：pytest, npm test
        - 檢查程式碼品質：ruff check, ruff format, pyright
        - 套件管理：uv add, uv remove, npm install
        - Git 操作：git status, git diff, git log
        - 瀏覽檔案：ls, find, tree

        限制：僅可在 sandbox 目錄內執行，禁止危險與系統修改命令（如 rm -rf, sudo 等）。

        回傳：命令執行結果（exit code、stdout、stderr）。""",
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': '要執行的 bash 命令',
                },
                'timeout': {
                    'type': 'integer',
                    'description': '超時時間（秒），預設 120 秒',
                },
                'working_dir': {
                    'type': 'string',
                    'description': '工作目錄（相對於 sandbox 根目錄，可選）',
                },
            },
            'required': ['command'],
        },
        handler=_handler,
        file_param=None,  # bash 不操作特定檔案，不需要鎖定
    )
