"""Sandbox 沙箱環境模組。

提供可抽換的沙箱環境介面，支援本地檔案系統與容器隔離。
"""

from agent_core.sandbox.base import ExecResult, Sandbox
from agent_core.sandbox.local import LocalSandbox

__all__ = [
    'ExecResult',
    'LocalSandbox',
    'Sandbox',
]
