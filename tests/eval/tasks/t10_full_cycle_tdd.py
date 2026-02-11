"""T10 — Full Cycle TDD (Special)。

給 Agent 一個模糊需求：「簡單的圖書借閱功能」。
專案已有一個 User model 作為風格參考。
Agent 需要自行：分析需求 → 寫測試 → 實作 → 驗證。

評估用隱藏的驗收測試檢查關鍵行為。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from agent_core.types import AgentEvent, MessageParam
from tests.eval.framework import EvalResult, run_pytest_in_sandbox

TASK_NAME: str = 'T10 - Full Cycle TDD'
TASK_LEVEL: str = 'special'
TASK_PROMPT: str = (
    '我們需要一個簡單的圖書借閱功能：\n'
    '- 可以新增書籍（書名、作者）\n'
    '- 可以借書（指定書名）\n'
    '- 可以還書\n'
    '- 可以查詢某本書是否可借\n\n'
    '請先參考專案中現有的程式碼風格，然後：\n'
    '1. 寫測試\n'
    '2. 實作功能\n'
    '3. 確認所有測試通過'
)


def setup(sandbox: Path) -> None:
    """建立基礎專案結構，含風格參考。"""
    (sandbox / 'models').mkdir()
    (sandbox / 'models' / '__init__.py').write_text('', encoding='utf-8')

    # models/user.py — 風格參考
    (sandbox / 'models' / 'user.py').write_text(
        '"""使用者模型。"""\n'
        '\n'
        'from __future__ import annotations\n'
        '\n'
        'from dataclasses import dataclass\n'
        '\n'
        '\n'
        '@dataclass\n'
        'class User:\n'
        '    """使用者資料。"""\n'
        '\n'
        '    name: str\n'
        '    email: str\n'
        '\n'
        '    def display_name(self) -> str:\n'
        '        """格式化顯示名稱。"""\n'
        '        return f"{self.name} <{self.email}>"\n',
        encoding='utf-8',
    )

    # test_user.py — 測試風格參考
    (sandbox / 'test_user.py').write_text(
        'from models.user import User\n'
        '\n'
        '\n'
        'def test_user_display_name() -> None:\n'
        '    user = User(name="Alice", email="alice@test.com")\n'
        '    assert user.display_name() == "Alice <alice@test.com>"\n',
        encoding='utf-8',
    )


def _run_acceptance_tests(sandbox: Path) -> tuple[bool, str]:
    """執行隱藏的驗收測試。

    動態探測 Agent 寫的模組並測試關鍵行為。
    """
    errors: list[str] = []

    # 將 sandbox 加入 sys.path 以便 import
    sandbox_str = str(sandbox)
    if sandbox_str not in sys.path:
        sys.path.insert(0, sandbox_str)

    try:
        # 嘗試找到圖書相關的模組
        library_mod = None
        for py_file in sandbox.rglob('*.py'):
            if py_file.name.startswith('test_'):
                continue
            if py_file.name == '__init__.py':
                continue
            # 嘗試 import 並找到有 "add" 或 "borrow" 方法的 class
            try:
                rel = py_file.relative_to(sandbox)
                mod_name = str(rel).replace('/', '.').replace('.py', '')
                # 清除已載入的模組以確保重新載入
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
                mod = importlib.import_module(mod_name)
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    if isinstance(obj, type) and (
                        hasattr(obj, 'add_book') or hasattr(obj, 'borrow')
                    ):
                        library_mod = obj
                        break
                if library_mod:
                    break
            except Exception:
                continue

        if library_mod is None:
            errors.append('找不到圖書管理 class（需有 add_book 或 borrow 方法）')
            return False, '\n'.join(errors)

        # 測試 1: 可以建立實例
        try:
            lib = library_mod()
        except Exception as e:
            errors.append(f'無法建立實例: {e}')
            return False, '\n'.join(errors)

        # 測試 2: 可以新增書籍（後續測試依賴此步驟，失敗則 early return）
        try:
            if hasattr(lib, 'add_book'):
                lib.add_book('Python 入門', '作者A')
            elif hasattr(lib, 'add'):
                lib.add('Python 入門', '作者A')
            else:
                errors.append('找不到新增書籍的方法（add_book 或 add）')
                return False, '\n'.join(errors)
        except Exception as e:
            errors.append(f'新增書籍失敗: {e}')
            return False, '\n'.join(errors)

        # 測試 3: 可以借書
        borrow_method = getattr(lib, 'borrow', None) or getattr(lib, 'borrow_book', None)
        try:
            if borrow_method:
                borrow_method('Python 入門')
            else:
                errors.append('找不到借書方法（borrow 或 borrow_book）')
        except Exception as e:
            errors.append(f'借書失敗: {e}')

        # 測試 4: 借出後不可再借
        try:
            # 接受多種常見命名
            is_available = (
                getattr(lib, 'is_available', None)
                or getattr(lib, 'check_available', None)
                or getattr(lib, 'is_book_available', None)
            )
            if is_available:
                if is_available('Python 入門'):
                    errors.append('書已被借出但仍顯示為可借')
            # 或者嘗試再次借書應該失敗（拋例外或回傳 False 皆可）
            elif borrow_method:
                try:
                    result = borrow_method('Python 入門')
                    # 回傳 False 表示借閱失敗，視為合法行為
                    if result is not False:
                        errors.append('已借出的書不應該能再被借出')
                except (ValueError, RuntimeError):
                    pass  # 預期行為
        except Exception as e:
            errors.append(f'檢查借閱狀態失敗: {e}')

        # 測試 5: 可以還書
        try:
            return_method = getattr(lib, 'return_book', None) or getattr(lib, 'return_', None)
            if return_method:
                return_method('Python 入門')
            else:
                errors.append('找不到還書方法（return_book 或 return_）')
        except Exception as e:
            errors.append(f'還書失敗: {e}')

    finally:
        # 清理 sys.path
        if sandbox_str in sys.path:
            sys.path.remove(sandbox_str)

    return len(errors) == 0, '\n'.join(errors) if errors else 'All acceptance tests passed'


def evaluate(
    sandbox: Path,
    events: list[AgentEvent],
    conversation: list[MessageParam],
) -> EvalResult:
    """評估完整 TDD 週期。"""
    details: dict[str, Any] = {}

    # 1. 檢查 Agent 是否寫了測試
    test_files = list(sandbox.rglob('test_*.py'))
    # 排除原本就有的 test_user.py
    agent_test_files = [f for f in test_files if f.name != 'test_user.py']
    details['agent_wrote_tests'] = len(agent_test_files) > 0
    details['test_files'] = [str(f.relative_to(sandbox)) for f in agent_test_files]

    # 2. 檢查 Agent 是否寫了實作
    impl_files = [
        f
        for f in sandbox.rglob('*.py')
        if not f.name.startswith('test_') and f.name != '__init__.py' and f.name != 'user.py'
    ]
    details['agent_wrote_impl'] = len(impl_files) > 0
    details['impl_files'] = [str(f.relative_to(sandbox)) for f in impl_files]

    # 3. 檢查 TDD 順序：是否先寫測試再寫實作
    tool_calls = [
        e['data']
        for e in events
        if e['type'] == 'tool_call' and e['data'].get('status') == 'completed'
    ]
    # 簡化判斷：至少有 edit_file 呼叫
    details['total_tool_calls'] = len(tool_calls)

    # 4. Agent 自己的測試是否通過
    agent_tests_passed, agent_output = run_pytest_in_sandbox(sandbox)
    details['agent_tests_passed'] = agent_tests_passed
    details['agent_pytest_output'] = agent_output[:1000]

    # 5. 隱藏驗收測試
    acceptance_passed, acceptance_output = _run_acceptance_tests(sandbox)
    details['acceptance_passed'] = acceptance_passed
    details['acceptance_output'] = acceptance_output

    # 評分
    score = 0.0
    if details['agent_wrote_tests']:
        score += 0.15
    if details['agent_wrote_impl']:
        score += 0.15
    if agent_tests_passed:
        score += 0.3
    if acceptance_passed:
        score += 0.4

    overall_passed = agent_tests_passed and acceptance_passed

    return EvalResult(
        task_name=TASK_NAME,
        task_level=TASK_LEVEL,
        passed=overall_passed,
        score=score,
        details=details,
    )
