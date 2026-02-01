"""File List 工具模組。

提供目錄列出功能，支援遞迴、過濾、詳細資訊等選項。
"""

from __future__ import annotations

import fnmatch
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_demo.tools.path_utils import validate_path

logger = logging.getLogger(__name__)


def _should_include_file(
    file_name: str,
    show_hidden: bool,
    pattern: str | None,
) -> bool:
    """判斷檔案是否應該包含在結果中。

    Args:
        file_name: 檔案名稱
        show_hidden: 是否顯示隱藏檔案
        pattern: 檔案名稱模式（如 "*.py"）

    Returns:
        是否包含該檔案
    """
    # 檢查隱藏檔案
    if not show_hidden and file_name.startswith('.'):
        return False

    # 檢查模式匹配
    if pattern and not fnmatch.fnmatch(file_name, pattern):
        return False

    return True


def _get_file_details(file_path: Path) -> dict[str, Any]:
    """取得檔案詳細資訊。

    Args:
        file_path: 檔案路徑

    Returns:
        包含檔案名稱、類型、大小、修改時間的字典
    """
    stat = file_path.stat()
    is_dir = file_path.is_dir()

    return {
        'name': file_path.name,
        'type': 'directory' if is_dir else 'file',
        'size': stat.st_size if not is_dir else None,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _list_directory(
    dir_path: Path,
    show_hidden: bool,
    pattern: str | None,
) -> tuple[list[str], list[str]]:
    """列出目錄內容。

    Args:
        dir_path: 目錄路徑
        show_hidden: 是否顯示隱藏檔案
        pattern: 檔案名稱模式

    Returns:
        (檔案列表, 目錄列表)
    """
    files: list[str] = []
    directories: list[str] = []

    for item in dir_path.iterdir():
        if item.is_file():
            if _should_include_file(item.name, show_hidden, pattern):
                files.append(item.name)
        elif item.is_dir():
            # 目錄也要檢查隱藏檔案
            if show_hidden or not item.name.startswith('.'):
                directories.append(item.name)

    return sorted(files), sorted(directories)


def _list_recursive(
    dir_path: Path,
    sandbox_root: Path,
    show_hidden: bool,
    pattern: str | None,
    exclude_dirs: list[str] | None,
    max_depth: int | None,
    current_depth: int = 0,
) -> list[str]:
    """遞迴列出所有檔案。

    Args:
        dir_path: 目錄路徑
        sandbox_root: sandbox 根目錄
        show_hidden: 是否顯示隱藏檔案
        pattern: 檔案名稱模式
        exclude_dirs: 要排除的目錄名稱列表
        max_depth: 最大遞迴深度（可進入的子目錄層數）
        current_depth: 當前深度

    Returns:
        相對於 sandbox_root 的檔案路徑列表
    """
    all_files: list[str] = []
    exclude_set = set(exclude_dirs or [])

    try:
        for item in dir_path.iterdir():
            # 計算相對路徑
            try:
                rel_path = item.relative_to(sandbox_root)
            except ValueError:
                continue

            if item.is_file():
                if _should_include_file(item.name, show_hidden, pattern):
                    all_files.append(str(rel_path))
            elif item.is_dir():
                # 檢查是否應排除
                if item.name in exclude_set:
                    continue
                # 檢查隱藏目錄
                if not show_hidden and item.name.startswith('.'):
                    continue

                # 檢查深度限制（是否可以繼續遞迴）
                if max_depth is not None and current_depth >= max_depth:
                    continue

                # 遞迴處理子目錄
                sub_files = _list_recursive(
                    item,
                    sandbox_root,
                    show_hidden,
                    pattern,
                    exclude_dirs,
                    max_depth,
                    current_depth + 1,
                )
                all_files.extend(sub_files)
    except PermissionError:
        logger.warning('無法讀取目錄', extra={'path': str(dir_path)})
        raise

    return sorted(all_files)


def list_files_handler(
    path: str,
    sandbox_root: Path,
    recursive: bool = False,
    max_depth: int | None = None,
    pattern: str | None = None,
    exclude_dirs: list[str] | None = None,
    show_hidden: bool = False,
    show_details: bool = False,
) -> dict[str, Any]:
    """列出目錄中的檔案。

    Args:
        path: 目錄路徑（相對於 sandbox 根目錄）
        sandbox_root: sandbox 根目錄
        recursive: 是否遞迴列出所有檔案
        max_depth: 最大遞迴深度（None 表示無限制）
        pattern: 檔案名稱模式（如 "*.py", "test_*.py"）
        exclude_dirs: 要排除的目錄名稱列表
        show_hidden: 是否顯示隱藏檔案
        show_details: 是否顯示詳細資訊（大小、修改時間）

    Returns:
        包含 path、files、directories 的字典

    Raises:
        FileNotFoundError: 目錄不存在
        PermissionError: 路徑安全性問題或權限不足
    """
    # 驗證路徑安全性
    resolved_path = validate_path(path, sandbox_root)

    # 檢查目錄是否存在
    if not resolved_path.exists():
        raise FileNotFoundError(f'目錄不存在: {path}')

    if not resolved_path.is_dir():
        raise FileNotFoundError(f'路徑不是目錄: {path}')

    # 列出目錄內容
    try:
        files, directories = _list_directory(resolved_path, show_hidden, pattern)
    except PermissionError:
        raise PermissionError(f'無權限讀取目錄: {path}')

    result: dict[str, Any] = {
        'path': path,
        'files': files,
        'directories': directories,
    }

    # 遞迴列出所有檔案
    if recursive:
        all_files = _list_recursive(
            resolved_path,
            sandbox_root,
            show_hidden,
            pattern,
            exclude_dirs,
            max_depth,
            current_depth=0,
        )
        result['all_files'] = all_files

    # 顯示詳細資訊
    if show_details:
        file_details: list[dict[str, Any]] = []
        for file_name in files:
            file_path = resolved_path / file_name
            file_details.append(_get_file_details(file_path))
        for dir_name in directories:
            dir_path = resolved_path / dir_name
            file_details.append(_get_file_details(dir_path))
        result['file_details'] = file_details

    logger.info('目錄列出完成', extra={'path': path, 'file_count': len(files)})

    return result
