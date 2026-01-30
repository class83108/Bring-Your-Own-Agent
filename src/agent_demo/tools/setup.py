"""工具註冊工廠模組。

提供建立預設工具註冊表的工廠函數。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_demo.tools.bash import bash_handler
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
        description='讀取檔案內容。回傳檔案的文字內容、路徑與程式語言識別。',
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
        description=(
            '執行 bash 命令。支援管道、重定向等 shell 功能。'
            '命令在 sandbox 目錄內執行，禁止執行危險或系統修改命令。'
            '常見用途：git 操作、執行測試、檢查程式碼風格、套件管理等。'
        ),
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
