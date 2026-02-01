"""File Read 工具模組。

提供檔案讀取功能，包含路徑安全驗證、檔案類型識別、大小限制等。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_demo.tools.path_utils import validate_path as validate_path_base

logger = logging.getLogger(__name__)

# 檔案大小上限（1MB）
MAX_FILE_SIZE: int = 1 * 1024 * 1024

# 副檔名對應的程式語言
LANGUAGE_MAP: dict[str, str] = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.json': 'json',
    '.md': 'markdown',
    '.html': 'html',
    '.css': 'css',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.txt': 'plaintext',
    '.sh': 'bash',
    '.sql': 'sql',
    '.xml': 'xml',
    '.rs': 'rust',
    '.go': 'go',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
}

# 敏感檔案名稱模式
SENSITIVE_PATTERNS: list[str] = [
    '.env',
    '.env.local',
    '.env.production',
    '.env.development',
    'id_rsa',
    'id_ed25519',
    '.git/config',
    'credentials.json',
    '.aws/credentials',
]


def detect_language(file_path: Path) -> str:
    """根據副檔名識別程式語言。

    Args:
        file_path: 檔案路徑

    Returns:
        程式語言名稱
    """
    return LANGUAGE_MAP.get(file_path.suffix, 'plaintext')


def validate_path(file_path: str, sandbox_root: Path) -> Path:
    """驗證檔案路徑是否在 sandbox 內。

    Args:
        file_path: 使用者提供的檔案路徑
        sandbox_root: sandbox 根目錄

    Returns:
        解析後的絕對路徑

    Raises:
        PermissionError: 路徑在 sandbox 外或為敏感檔案
    """
    # 使用共用的基礎路徑驗證
    resolved = validate_path_base(file_path, sandbox_root)

    # 檢查敏感檔案
    if _is_sensitive_file(Path(file_path)):
        logger.warning('嘗試讀取敏感檔案', extra={'path': file_path})
        raise PermissionError(f'檔案可能包含敏感資訊，拒絕讀取: {file_path}')

    return resolved


def _is_sensitive_file(file_path: Path) -> bool:
    """檢查是否為敏感檔案。

    Args:
        file_path: 檔案路徑

    Returns:
        是否為敏感檔案
    """
    path_str = str(file_path)
    return any(pattern == file_path.name or pattern in path_str for pattern in SENSITIVE_PATTERNS)


def _check_file_size(file_path: Path) -> None:
    """檢查檔案大小是否超過限制。

    Args:
        file_path: 檔案路徑

    Raises:
        ValueError: 檔案過大
    """
    size = file_path.stat().st_size
    if size > MAX_FILE_SIZE:
        raise ValueError(f'檔案過大 ({size / 1024 / 1024:.2f}MB)，超過 1MB 限制')


def _read_content(
    file_path: Path,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """讀取檔案內容，支援指定行數範圍。

    Args:
        file_path: 檔案路徑
        start_line: 起始行號（從 1 開始）
        end_line: 結束行號（None 表示讀到檔尾）

    Returns:
        檔案內容字串

    Raises:
        ValueError: 無法讀取二進位檔案
    """
    try:
        text = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise ValueError(f'無法讀取二進位檔案: {file_path.name}')

    # 如果指定了行數範圍，回傳帶行號的格式
    if start_line != 1 or end_line is not None:
        lines = text.splitlines()

        if end_line is None:
            end_line = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        result: list[str] = []
        for i, line in enumerate(lines[start_idx:end_idx], start=start_line):
            result.append(f'{i:4d} | {line}')

        return '\n'.join(result)

    return text


def read_file_handler(
    path: str,
    sandbox_root: Path,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """讀取檔案並回傳結構化結果。

    Args:
        path: 檔案路徑（相對於 sandbox 根目錄）
        sandbox_root: sandbox 根目錄
        start_line: 起始行號（從 1 開始）
        end_line: 結束行號（None 表示讀到檔尾）

    Returns:
        包含 path、content、language 的字典

    Raises:
        FileNotFoundError: 檔案不存在
        PermissionError: 路徑安全性問題
        ValueError: 檔案過大或為二進位檔案
    """
    # 驗證路徑安全性
    resolved_path = validate_path(path, sandbox_root)

    # 檢查檔案是否存在
    if not resolved_path.exists():
        raise FileNotFoundError(f'檔案不存在: {path}')

    if not resolved_path.is_file():
        raise FileNotFoundError(f'路徑不是檔案: {path}')

    # 檢查檔案大小
    _check_file_size(resolved_path)

    # 讀取內容
    content = _read_content(resolved_path, start_line, end_line)

    # 識別語言
    language = detect_language(resolved_path)

    logger.info('檔案讀取完成', extra={'path': path, 'language': language})

    return {
        'path': path,
        'content': content,
        'language': language,
    }
