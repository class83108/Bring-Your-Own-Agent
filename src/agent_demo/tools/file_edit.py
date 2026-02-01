"""File Edit 工具模組。

提供檔案編輯功能，包含建立、修改、刪除內容等操作。
使用精確的字串匹配來確保編輯的準確性。
採用臨時檔案 + 原子重新命名的方式確保寫入安全性。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from agent_demo.tools.path_utils import validate_path

logger = logging.getLogger(__name__)


def edit_file_handler(
    path: str,
    sandbox_root: Path,
    old_content: str | None = None,
    new_content: str | None = None,
    create_if_missing: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    """編輯或建立檔案。

    Args:
        path: 檔案路徑（相對於 sandbox 根目錄）
        sandbox_root: sandbox 根目錄
        old_content: 要搜尋的舊內容（None 表示建立新檔案）
        new_content: 要替換的新內容（None 表示刪除）
        create_if_missing: 是否在檔案不存在時建立
        backup: 是否在編輯前備份原始檔案

    Returns:
        包含 path、created、modified、backup_path 的字典

    Raises:
        FileNotFoundError: 檔案不存在且 create_if_missing=False
        FileExistsError: 檔案已存在但要求建立新檔案
        PermissionError: 路徑安全性問題
        ValueError: 搜尋內容不存在或有多處匹配
    """
    # 驗證路徑安全性
    resolved_path = validate_path(path, sandbox_root)

    # 情況 1: 建立新檔案
    if old_content is None and create_if_missing:
        return _create_file(resolved_path, path, new_content or '')

    # 情況 2: 編輯現有檔案
    if old_content is not None:
        return _edit_file(resolved_path, path, old_content, new_content or '', backup)

    # 情況 3: 無效的參數組合
    raise ValueError('必須提供 old_content（編輯）或設定 create_if_missing=True（建立）')


def _create_file(resolved_path: Path, path: str, content: str) -> dict[str, Any]:
    """建立新檔案。

    使用臨時檔案 + 原子重新命名確保安全性。

    Args:
        resolved_path: 解析後的絕對路徑
        path: 原始路徑（用於回傳）
        content: 檔案內容

    Returns:
        包含 path、created 的字典

    Raises:
        FileExistsError: 檔案已存在
    """
    # 檢查檔案是否已存在
    if resolved_path.exists():
        logger.warning('嘗試覆蓋已存在的檔案', extra={'path': path})
        raise FileExistsError(f'檔案已存在: {path}。若要編輯，請提供 old_content 參數')

    # 建立必要的目錄
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用臨時檔案寫入
    _write_file_safely(resolved_path, content)

    logger.info('檔案建立完成', extra={'path': path})

    return {
        'path': path,
        'created': True,
    }


def _edit_file(
    resolved_path: Path,
    path: str,
    old_content: str,
    new_content: str,
    backup: bool,
) -> dict[str, Any]:
    """編輯現有檔案。

    使用臨時檔案 + 原子重新命名確保安全性。
    可選擇性備份原始檔案。

    Args:
        resolved_path: 解析後的絕對路徑
        path: 原始路徑（用於回傳）
        old_content: 要搜尋的舊內容
        new_content: 要替換的新內容
        backup: 是否備份原始檔案

    Returns:
        包含 path、modified、backup_path 的字典

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 搜尋內容不存在或有多處匹配
    """
    # 檢查檔案是否存在
    if not resolved_path.exists():
        raise FileNotFoundError(f'檔案不存在: {path}')

    if not resolved_path.is_file():
        raise FileNotFoundError(f'路徑不是檔案: {path}')

    # 讀取原始內容
    try:
        original_content = resolved_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise ValueError(f'無法讀取二進位檔案: {path}')

    # 檢查要搜尋的內容是否存在
    if old_content not in original_content:
        logger.warning('找不到要替換的內容', extra={'path': path, 'old_content': old_content})
        raise ValueError(f'找不到要替換的內容: {old_content}')

    # 檢查是否有多處匹配
    count = original_content.count(old_content)
    if count > 1:
        logger.warning(
            '搜尋內容有多處匹配',
            extra={'path': path, 'old_content': old_content, 'count': count},
        )
        raise ValueError(f'搜尋內容有多處匹配（共 {count} 處），請提供更精確的搜尋內容')

    # 備份原始檔案（如果需要）
    backup_path: str | None = None
    if backup:
        backup_path = _backup_file(resolved_path, path)

    # 執行替換
    new_file_content = original_content.replace(old_content, new_content, 1)

    # 使用臨時檔案安全寫入
    _write_file_safely(resolved_path, new_file_content)

    logger.info('檔案編輯完成', extra={'path': path, 'backup': backup_path})

    result: dict[str, Any] = {
        'path': path,
        'modified': True,
    }

    if backup_path:
        result['backup_path'] = backup_path

    return result


def _backup_file(resolved_path: Path, original_path: str) -> str:
    """備份檔案。

    備份檔案命名格式：original_name.bak

    Args:
        resolved_path: 檔案的絕對路徑
        original_path: 原始路徑（用於 log）

    Returns:
        備份檔案的路徑（相對路徑）
    """
    backup_path = resolved_path.with_suffix(resolved_path.suffix + '.bak')

    # 使用 shutil.copy2 保留檔案元數據（權限、時間戳等）
    shutil.copy2(resolved_path, backup_path)

    logger.info('檔案備份完成', extra={'original': original_path, 'backup': str(backup_path)})

    # 回傳相對路徑
    return f'{original_path}.bak'


def _write_file_safely(target_path: Path, content: str) -> None:
    """使用臨時檔案安全寫入內容。

    寫入流程：
    1. 寫入到臨時檔案
    2. 使用 replace() 原子性地重新命名覆蓋目標檔案

    這確保了即使寫入過程中斷，原始檔案也不會損壞。

    Args:
        target_path: 目標檔案路徑
        content: 要寫入的內容
    """
    # 建立臨時檔案路徑（同目錄下，避免跨檔案系統問題）
    tmp_path = target_path.with_suffix(target_path.suffix + '.tmp')

    try:
        # 寫入臨時檔案
        tmp_path.write_text(content, encoding='utf-8')

        # 原子性地重新命名（在同一檔案系統上，replace 是原子操作）
        tmp_path.replace(target_path)

    except Exception:
        # 如果出錯，清理臨時檔案
        if tmp_path.exists():
            tmp_path.unlink()
        raise
