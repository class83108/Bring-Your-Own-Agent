"""Bash 工具模組。

提供 bash 命令執行功能，包含安全性檢查、輸出處理、超時控制等。
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 輸出大小限制（100KB）
MAX_OUTPUT_SIZE: int = 100 * 1024

# 敏感資訊模式
SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r'\bsk-[a-zA-Z0-9]{20,}', '[OPENAI_API_KEY]'),  # OpenAI API key (至少 20 字元)
    (r'\bsk-ant-[a-zA-Z0-9-]{15,}', '[ANTHROPIC_API_KEY]'),  # Anthropic API key (至少 15 字元)
    (r'\bghp_[a-zA-Z0-9]{20,}', '[GITHUB_TOKEN]'),  # GitHub token
    (r'\bgho_[a-zA-Z0-9]{20,}', '[GITHUB_OAUTH]'),  # GitHub OAuth
    (r'\b(AWS|AKIA)[A-Z0-9]{16,}', '[AWS_ACCESS_KEY]'),  # AWS access key
    (r'password[=:]\s*["\']?([^"\'\s]+)', 'password=[REDACTED]'),
    (r'token[=:]\s*["\']?([^"\'\s]+)', 'token=[REDACTED]'),
]

# 危險命令模式（使用正則表達式）
DANGEROUS_PATTERNS: list[str] = [
    r'\brm\s+.*-rf\s+/',  # rm -rf /
    r'\brm\s+.*-rf\s+\*',  # rm -rf *（根目錄）
    r'\bdd\s+if=',  # dd if=...
    r'\bmkfs\.',  # mkfs.ext4, mkfs.xfs 等
    r'\b:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}',  # fork bomb
    r'\bshutdown\b',  # shutdown
    r'\breboot\b',  # reboot
    r'\binit\s+[0-6]',  # init 0-6
    r'\bchmod\s+777',  # chmod 777（過於寬鬆）
    r'\bchown\s+root',  # chown root
    r'\bsu\s+',  # su（切換使用者）
    r'\bkill\s+-9\s+1\b',  # kill -9 1（init）
    r'\b>/dev/sd[a-z]',  # 直接寫入磁碟設備
]

# 系統修改命令（需要警告）
SYSTEM_MODIFY_PATTERNS: list[str] = [
    r'\bsudo\b',  # sudo
    r'\bsystemctl\b',
    r'\bservice\b',
    r'\bapt\b',
    r'\bapt-get\b',
    r'\byum\b',
    r'\bpacman\b',
    r'\bnpm\s+install\s+-g',  # 全域安裝
    r'\bpip\s+install\s+.*--system',
]


def check_command_safety(command: str) -> None:
    """檢查命令安全性。

    Args:
        command: 要檢查的命令

    Raises:
        PermissionError: 命令具有危險性
    """
    # 檢查危險命令
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise PermissionError(f'拒絕執行危險命令: {command}\n該命令可能對系統造成嚴重破壞。')

    # 檢查系統修改命令
    for pattern in SYSTEM_MODIFY_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise PermissionError(
                f'拒絕執行系統修改命令: {command}\n該命令可能修改系統設定，存在安全風險。'
            )


def validate_working_dir(
    working_dir: str | None,
    sandbox_root: Path,
) -> Path:
    """驗證並解析工作目錄。

    Args:
        working_dir: 使用者指定的工作目錄（相對路徑或 None）
        sandbox_root: sandbox 根目錄

    Returns:
        解析後的絕對路徑

    Raises:
        PermissionError: 工作目錄在 sandbox 外
        FileNotFoundError: 目錄不存在
    """
    if working_dir is None:
        return sandbox_root.resolve()

    # 解析路徑
    resolved = (sandbox_root / working_dir).resolve()

    # 檢查路徑穿越
    if not resolved.is_relative_to(sandbox_root.resolve()):
        raise PermissionError(
            f'工作目錄必須在 sandbox 內: {working_dir}\nSandbox 根目錄: {sandbox_root}'
        )

    # 檢查目錄是否存在
    if not resolved.exists():
        raise FileNotFoundError(f'工作目錄不存在: {working_dir}')

    if not resolved.is_dir():
        raise ValueError(f'路徑不是目錄: {working_dir}')

    return resolved


def mask_sensitive_info(text: str) -> str:
    """遮蔽輸出中的敏感資訊。

    Args:
        text: 原始文字

    Returns:
        遮蔽後的文字
    """
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def execute_command(
    command: str,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    """執行命令並回傳結果。

    Args:
        command: 命令字串
        cwd: 工作目錄
        timeout: 超時秒數

    Returns:
        包含 exit_code, stdout, stderr 的字典

    Raises:
        TimeoutError: 命令執行超時
    """
    try:
        # 使用 shell=True 支援管道、重定向等
        # text=True 自動處理編碼
        # capture_output=True 捕獲 stdout 和 stderr
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            timeout=timeout,
            capture_output=True,
            text=True,
            # 環境變數繼承當前環境（包含 PATH 等）
            env=None,
        )

        stdout = result.stdout
        stderr = result.stderr
        truncated = False

        # 檢查輸出大小並截斷
        if len(stdout) > MAX_OUTPUT_SIZE:
            stdout = stdout[:MAX_OUTPUT_SIZE] + '\n\n[輸出已截斷...]'
            truncated = True

        if len(stderr) > MAX_OUTPUT_SIZE:
            stderr = stderr[:MAX_OUTPUT_SIZE] + '\n\n[錯誤輸出已截斷...]'
            truncated = True

        # 遮蔽敏感資訊
        stdout = mask_sensitive_info(stdout)
        stderr = mask_sensitive_info(stderr)

        return {
            'exit_code': result.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'truncated': truncated,
        }

    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f'命令執行超時（{timeout} 秒）: {command}\n請檢查命令是否卡住或需要更長執行時間。'
        ) from e
    except Exception as e:
        # 其他異常（權限、命令不存在等）
        raise RuntimeError(f'命令執行失敗: {e}') from e


def bash_handler(
    command: str,
    sandbox_root: Path,
    timeout: int = 120,
    working_dir: str | None = None,
) -> dict[str, Any]:
    """執行 bash 命令並回傳結構化結果。

    Args:
        command: 要執行的 bash 命令
        sandbox_root: sandbox 根目錄（安全限制）
        timeout: 執行超時時間（秒），預設 120 秒
        working_dir: 工作目錄（相對於 sandbox_root，可選）

    Returns:
        {
            'command': str,           # 原始命令
            'exit_code': int,         # 退出碼
            'stdout': str,            # 標準輸出
            'stderr': str,            # 標準錯誤
            'truncated': bool,        # 輸出是否被截斷
            'working_dir': str,       # 實際執行目錄
        }

    Raises:
        PermissionError: 命令被安全檢查阻擋
        TimeoutError: 命令執行超時
        ValueError: 參數無效
    """
    # 1. 參數驗證
    if not command or not command.strip():
        raise ValueError('命令不能為空')

    # 2. 安全性檢查
    check_command_safety(command)

    # 3. 驗證工作目錄
    cwd = validate_working_dir(working_dir, sandbox_root)

    # 4. 記錄日誌
    logger.info(
        '執行 bash 命令',
        extra={
            'command': command,
            'working_dir': str(cwd),
            'timeout': timeout,
        },
    )

    # 5. 執行命令
    exec_result = execute_command(command, cwd, timeout)

    # 6. 構造回應
    # 計算相對於 sandbox 的路徑（如果 cwd 在 sandbox 內）
    try:
        relative_wd = str(cwd.relative_to(sandbox_root.resolve()))
        if relative_wd == '.':
            relative_wd = ''
    except ValueError:
        relative_wd = str(cwd)

    result = {
        'command': command,
        'working_dir': relative_wd or '.',
        **exec_result,
    }

    # 7. 記錄結果
    logger.info(
        '命令執行完成',
        extra={
            'command': command,
            'exit_code': result['exit_code'],
            'stdout_len': len(result['stdout']),
            'stderr_len': len(result['stderr']),
        },
    )

    return result
