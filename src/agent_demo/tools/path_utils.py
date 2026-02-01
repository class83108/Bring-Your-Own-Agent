"""路徑工具模組。

提供路徑驗證與安全檢查的共用函數。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_path(path: str, sandbox_root: Path) -> Path:
    """驗證路徑是否在 sandbox 內。

    Args:
        path: 使用者提供的路徑
        sandbox_root: sandbox 根目錄

    Returns:
        解析後的絕對路徑

    Raises:
        PermissionError: 路徑在 sandbox 外
    """
    resolved = (sandbox_root / path).resolve()

    # 檢查路徑穿越
    if not resolved.is_relative_to(sandbox_root.resolve()):
        logger.warning('路徑穿越攻擊', extra={'path': path})
        raise PermissionError(f'無法存取 sandbox 外的路徑: {path}')

    return resolved
