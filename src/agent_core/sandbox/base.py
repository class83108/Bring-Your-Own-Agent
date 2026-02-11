"""Sandbox 基底類別與共用型別。

定義沙箱環境的抽象介面，提供路徑驗證與指令執行的統一操作。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

# =============================================================================
# 回傳型別
# =============================================================================


class ExecResult(TypedDict):
    """指令執行結果。"""

    exit_code: int
    stdout: str
    stderr: str


# =============================================================================
# Sandbox ABC
# =============================================================================


class Sandbox(ABC):
    """沙箱環境抽象介面。

    提供路徑驗證與指令執行的統一介面。
    檔案 I/O 由 tool handler 透過 Path 直接操作（LocalSandbox），
    或由容器內的 agent 自行處理（ContainerRunner）。

    職責範圍：
    - 路徑驗證（確保存取不超出沙箱邊界）
    - 指令執行

    不負責：
    - 檔案讀寫（由 tool handler 層直接操作）
    - 敏感檔案過濾（.env 等）→ 由 tool handler 層決定
    - 危險指令阻擋（rm -rf / 等）→ 由 tool handler 層決定
    """

    # --- 路徑驗證 ---

    @abstractmethod
    def validate_path(self, path: str) -> str:
        """驗證路徑是否在沙箱範圍內。

        Args:
            path: 使用者提供的相對路徑

        Returns:
            正規化後的路徑字串

        Raises:
            PermissionError: 路徑超出沙箱範圍
        """
        ...

    # --- 指令執行 ---

    @abstractmethod
    async def exec(
        self,
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> ExecResult:
        """在沙箱內執行指令。

        Args:
            command: 要執行的 shell 指令
            timeout: 超時時間（秒）
            working_dir: 工作目錄（相對於沙箱根目錄，None 表示沙箱根目錄）

        Returns:
            指令執行結果（exit_code、stdout、stderr）

        Raises:
            TimeoutError: 指令執行超時
            PermissionError: 工作目錄超出沙箱範圍
        """
        ...
