"""LocalSandbox — 本地檔案系統沙箱。

在指定的根目錄內操作，透過路徑驗證確保不超出範圍。
指令透過 subprocess 在本地執行。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from agent_core.sandbox.base import ExecResult, Sandbox

logger = logging.getLogger(__name__)


class LocalSandbox(Sandbox):
    """本地檔案系統沙箱。

    所有路徑操作限制在指定的根目錄內，
    指令透過 subprocess 在本地執行。
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        """沙箱根目錄的絕對路徑。"""
        return self._root

    # --- 路徑驗證 ---

    def validate_path(self, path: str) -> str:
        """驗證路徑在沙箱根目錄內。

        Args:
            path: 使用者提供的相對路徑

        Returns:
            解析後的絕對路徑字串

        Raises:
            PermissionError: 路徑超出沙箱範圍
        """
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            logger.warning('路徑穿越攻擊', extra={'path': path})
            raise PermissionError(f'無法存取 sandbox 外的路徑: {path}')
        return str(resolved)

    def _resolve(self, path: str) -> Path:
        """驗證路徑並回傳 Path 物件（內部使用）。"""
        return Path(self.validate_path(path))

    # --- 指令執行 ---

    async def exec(
        self,
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> ExecResult:
        """在沙箱根目錄內執行指令。"""
        if working_dir is not None:
            cwd = self._resolve(working_dir)
            if not cwd.exists():
                raise FileNotFoundError(f'工作目錄不存在: {working_dir}')
            if not cwd.is_dir():
                raise ValueError(f'路徑不是目錄: {working_dir}')
        else:
            cwd = self._root

        try:
            result = await asyncio.wait_for(
                self._run_subprocess(command, cwd),
                timeout=timeout,
            )
        except TimeoutError:
            raise TimeoutError(f'命令執行超時（{timeout} 秒）: {command}')

        return result

    async def _run_subprocess(self, command: str, cwd: Path) -> ExecResult:
        """實際執行 subprocess。"""
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        return ExecResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_bytes.decode('utf-8', errors='replace'),
            stderr=stderr_bytes.decode('utf-8', errors='replace'),
        )
