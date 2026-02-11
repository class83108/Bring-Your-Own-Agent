"""T13 — Subagent Parallel Fix (Special)。

兩個獨立模組各有一個 bug：
- auth/hasher.py：密碼雜湊時忘記加 salt（結果不含 salt prefix）
- billing/tax.py：稅率計算用了 0.5 而非 0.05（50% 而非 5%）

兩個測試檔分別測試這兩個模組，皆失敗。
Agent 應使用 create_subagent 將兩個修復任務分工，
或至少能自行修復兩個 bug 讓測試通過。

使用 subagent 可獲得額外加分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.types import AgentEvent, MessageParam
from tests.eval.framework import EvalResult, run_pytest_in_sandbox

TASK_NAME: str = 'T13 - Subagent Parallel Fix'
TASK_LEVEL: str = 'special'
TASK_PROMPT: str = (
    '這個專案有兩個獨立的 bug，分別在 auth/ 和 billing/ 模組中。\n'
    '執行 pytest 可以看到兩個測試檔都失敗。\n\n'
    '請修復這兩個 bug，讓所有測試通過。\n'
    '你可以使用 create_subagent 工具將兩個修復任務分別交給子代理處理。'
)

# 啟用 subagent 工具
TOOLS_CONFIG: dict[str, Any] = {'enable_subagent': True}


def setup(sandbox: Path) -> None:
    """建立含兩個獨立 bug 的專案。"""
    # --- auth 模組 ---
    auth_dir = sandbox / 'auth'
    auth_dir.mkdir()
    (auth_dir / '__init__.py').write_text('', encoding='utf-8')

    # auth/hasher.py — BUG：沒有把 salt 加入雜湊結果
    (auth_dir / 'hasher.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'import hashlib\n'
        'import secrets\n'
        '\n'
        '\n'
        'def hash_password(password: str) -> str:\n'
        '    """雜湊密碼，回傳 salt:hash 格式。\n'
        '\n'
        '    Args:\n'
        '        password: 明文密碼\n'
        '\n'
        '    Returns:\n'
        '        格式為 "salt:hash" 的字串\n'
        '    """\n'
        '    salt = secrets.token_hex(16)\n'
        '    hashed = hashlib.sha256((salt + password).encode()).hexdigest()\n'
        '    # BUG: 回傳時漏了 salt prefix\n'
        '    return hashed\n'
        '\n'
        '\n'
        'def verify_password(password: str, stored: str) -> bool:\n'
        '    """驗證密碼是否正確。\n'
        '\n'
        '    Args:\n'
        '        password: 明文密碼\n'
        '        stored: 儲存的 salt:hash 字串\n'
        '\n'
        '    Returns:\n'
        '        密碼是否正確\n'
        '    """\n'
        '    salt, expected_hash = stored.split(":")\n'
        '    actual_hash = hashlib.sha256((salt + password).encode()).hexdigest()\n'
        '    return actual_hash == expected_hash\n',
        encoding='utf-8',
    )

    # --- billing 模組 ---
    billing_dir = sandbox / 'billing'
    billing_dir.mkdir()
    (billing_dir / '__init__.py').write_text('', encoding='utf-8')

    # billing/tax.py — BUG：稅率 0.5 應為 0.05
    (billing_dir / 'tax.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        '# 台灣營業稅率 5%\n'
        'TAX_RATE = 0.5  # BUG: 應為 0.05\n'
        '\n'
        '\n'
        'def calculate_tax(amount: float) -> float:\n'
        '    """計算稅額。\n'
        '\n'
        '    Args:\n'
        '        amount: 未稅金額\n'
        '\n'
        '    Returns:\n'
        '        稅額\n'
        '    """\n'
        '    return round(amount * TAX_RATE, 2)\n'
        '\n'
        '\n'
        'def calculate_total_with_tax(amount: float) -> float:\n'
        '    """計算含稅總額。\n'
        '\n'
        '    Args:\n'
        '        amount: 未稅金額\n'
        '\n'
        '    Returns:\n'
        '        含稅總額\n'
        '    """\n'
        '    return round(amount + calculate_tax(amount), 2)\n',
        encoding='utf-8',
    )

    # --- 測試檔 ---

    (sandbox / 'test_auth.py').write_text(
        'from auth.hasher import hash_password, verify_password\n'
        '\n'
        '\n'
        'def test_hash_password_format() -> None:\n'
        '    """雜湊結果應為 salt:hash 格式。"""\n'
        '    result = hash_password("mypassword")\n'
        '    assert ":" in result, f"應包含 : 分隔符，實際: {result}"\n'
        '    parts = result.split(":")\n'
        '    assert len(parts) == 2, f"應有 salt 和 hash 兩部分，實際: {parts}"\n'
        '    assert len(parts[0]) == 32, f"salt 應為 32 字元 hex，實際長度: {len(parts[0])}"\n'
        '    assert len(parts[1]) == 64, f"hash 應為 64 字元 hex，實際長度: {len(parts[1])}"\n'
        '\n'
        '\n'
        'def test_hash_password_unique() -> None:\n'
        '    """每次雜湊同一密碼應產生不同結果（因為 salt 不同）。"""\n'
        '    h1 = hash_password("password")\n'
        '    h2 = hash_password("password")\n'
        '    assert h1 != h2, "不同次雜湊應產生不同結果"\n'
        '\n'
        '\n'
        'def test_verify_password() -> None:\n'
        '    """正確密碼應驗證成功。"""\n'
        '    stored = hash_password("secret123")\n'
        '    assert verify_password("secret123", stored) is True\n'
        '    assert verify_password("wrong", stored) is False\n',
        encoding='utf-8',
    )

    (sandbox / 'test_billing.py').write_text(
        'from billing.tax import calculate_tax, calculate_total_with_tax\n'
        '\n'
        '\n'
        'def test_calculate_tax() -> None:\n'
        '    """1000 元的 5% 稅額應為 50 元。"""\n'
        '    assert calculate_tax(1000) == 50.0\n'
        '    assert calculate_tax(500) == 25.0\n'
        '    assert calculate_tax(0) == 0.0\n'
        '\n'
        '\n'
        'def test_calculate_total_with_tax() -> None:\n'
        '    """1000 元含 5% 稅應為 1050 元。"""\n'
        '    assert calculate_total_with_tax(1000) == 1050.0\n'
        '    assert calculate_total_with_tax(200) == 210.0\n',
        encoding='utf-8',
    )


def evaluate(
    sandbox: Path,
    events: list[AgentEvent],
    conversation: list[MessageParam],
) -> EvalResult:
    """評估修復結果。"""
    details: dict[str, Any] = {}

    # --- 檢查 auth/hasher.py 修復 ---
    hasher_file = sandbox / 'auth' / 'hasher.py'
    if hasher_file.exists():
        content = hasher_file.read_text(encoding='utf-8')
        # 修復後應包含 f"{salt}:{hashed}" 或類似格式
        details['auth_has_salt_prefix'] = (
            'salt' in content
            and ':' in content
            and 'return' in content
            and ('f"' in content or "f'" in content or 'format' in content or '+' in content)
        )
    else:
        details['auth_has_salt_prefix'] = False

    # --- 檢查 billing/tax.py 修復 ---
    tax_file = sandbox / 'billing' / 'tax.py'
    if tax_file.exists():
        content = tax_file.read_text(encoding='utf-8')
        details['billing_correct_rate'] = '0.05' in content and '0.5' not in content.replace(
            '0.05', ''
        )
    else:
        details['billing_correct_rate'] = False

    # --- 檢查是否使用了 subagent ---
    tool_calls = [
        e['data']['name']
        for e in events
        if e['type'] == 'tool_call' and e['data'].get('status') == 'completed'
    ]
    details['used_subagent'] = 'create_subagent' in tool_calls
    details['subagent_count'] = tool_calls.count('create_subagent')
    details['tool_count'] = len(tool_calls)
    details['tool_sequence'] = tool_calls

    # --- 執行 pytest ---
    passed, output = run_pytest_in_sandbox(sandbox)
    details['pytest_passed'] = passed
    details['pytest_output'] = output[:1500]

    # --- 評分 ---
    score = 0.0
    # 修復 auth bug (0.25)
    if details['auth_has_salt_prefix']:
        score += 0.25
    # 修復 billing bug (0.25)
    if details['billing_correct_rate']:
        score += 0.25
    # 測試全通過 (0.3)
    if passed:
        score += 0.3
    # 使用了 subagent (0.2 bonus)
    if details['used_subagent']:
        score += 0.2

    return EvalResult(
        task_name=TASK_NAME,
        task_level=TASK_LEVEL,
        passed=passed,
        score=min(score, 1.0),
        details=details,
    )
