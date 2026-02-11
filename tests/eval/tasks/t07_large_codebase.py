"""T7 — Large Codebase Navigation (Special)。

15+ 個檔案的專案，bug 藏在 4 層深的 helper 中。
測試失敗只顯示「計算結果不對」，需要搜索才能定位根因。
grep_search 可快速找到關鍵函數；逐檔 import tracing 則需讀 5+ 個檔案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.types import AgentEvent, MessageParam
from tests.eval.framework import EvalResult, run_pytest_in_sandbox

TASK_NAME: str = 'T7 - Large Codebase Navigation'
TASK_LEVEL: str = 'special'
TASK_PROMPT: str = (
    '這個專案的測試有失敗。專案有多個模組，bug 可能在任何地方。\n'
    '請系統化地搜索程式碼，定位並修復問題，然後執行測試確認。'
)


def setup(sandbox: Path) -> None:
    """建立 15+ 檔案的專案，bug 藏在 4 層深的 helper。"""
    # 目錄結構
    for d in [
        'models',
        'core',
        'core/utils',
        'services',
        'services/handlers',
    ]:
        (sandbox / d).mkdir(parents=True, exist_ok=True)
        (sandbox / d / '__init__.py').write_text('', encoding='utf-8')

    # ── models ──

    (sandbox / 'models' / 'product.py').write_text(
        'from __future__ import annotations\n'
        'from dataclasses import dataclass\n'
        '\n\n'
        '@dataclass\n'
        'class Product:\n'
        '    name: str\n'
        '    price: float\n'
        '    category: str = "general"\n',
        encoding='utf-8',
    )

    (sandbox / 'models' / 'order.py').write_text(
        'from __future__ import annotations\n'
        'from dataclasses import dataclass, field\n'
        '\n\n'
        '@dataclass\n'
        'class OrderItem:\n'
        '    product_name: str\n'
        '    price: float\n'
        '    quantity: int\n'
        '\n\n'
        '@dataclass\n'
        'class Order:\n'
        '    items: list[OrderItem] = field(default_factory=list)\n'
        '    discount_percent: float = 0.0\n'
        '    status: str = "pending"\n',
        encoding='utf-8',
    )

    (sandbox / 'models' / 'customer.py').write_text(
        'from __future__ import annotations\n'
        'from dataclasses import dataclass\n'
        '\n\n'
        '@dataclass\n'
        'class Customer:\n'
        '    name: str\n'
        '    email: str\n'
        '    membership: str = "standard"\n',
        encoding='utf-8',
    )

    # ── core/utils ──（BUG 藏在這裡）

    # core/utils/calculator.py — BUG：除以 1000 而非 100
    (sandbox / 'core' / 'utils' / 'calculator.py').write_text(
        'from __future__ import annotations\n'
        '\n\n'
        'def calculate_percentage(amount: float, percentage: float) -> float:\n'
        '    """計算金額的百分比值。\n'
        '\n'
        '    Args:\n'
        '        amount: 原始金額\n'
        '        percentage: 百分比數值（如 10 表示 10%）\n'
        '\n'
        '    Returns:\n'
        '        百分比對應的金額\n'
        '    """\n'
        '    return round(amount * percentage / 1000, 2)\n',
        encoding='utf-8',
    )

    (sandbox / 'core' / 'utils' / 'formatters.py').write_text(
        'def format_currency(amount: float) -> str:\n'
        '    """格式化金額。"""\n'
        '    return f"${amount:,.2f}"\n'
        '\n\n'
        'def format_order_id(order_id: str) -> str:\n'
        '    """格式化訂單編號。"""\n'
        '    return f"ORD-{order_id.upper()}"\n',
        encoding='utf-8',
    )

    (sandbox / 'core' / 'utils' / 'validators.py').write_text(
        'import re\n'
        '\n'
        '_EMAIL_PATTERN = re.compile(r".+@.+\\..+")\n'
        '\n\n'
        'def validate_email(email: str) -> bool:\n'
        '    """驗證 email 格式。"""\n'
        '    return bool(_EMAIL_PATTERN.fullmatch(email))\n'
        '\n\n'
        'def validate_quantity(qty: int) -> bool:\n'
        '    """驗證數量。"""\n'
        '    return qty > 0\n',
        encoding='utf-8',
    )

    # ── core ──

    (sandbox / 'core' / 'config.py').write_text(
        'MAX_ORDER_AMOUNT = 100000\n'
        'MAX_DISCOUNT_PERCENT = 50\n'
        'TAX_RATE = 0.05\n'
        'SUPPORTED_STATUSES = ["pending", "confirmed", "shipped", "delivered"]\n',
        encoding='utf-8',
    )

    # core/promotions.py — 第 3 層：呼叫 calculator
    (sandbox / 'core' / 'promotions.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'from core.config import MAX_DISCOUNT_PERCENT\n'
        'from core.utils.calculator import calculate_percentage\n'
        '\n\n'
        'def apply_promotion(\n'
        '    subtotal: float,\n'
        '    discount_percent: float,\n'
        ') -> float:\n'
        '    """套用折扣到小計金額。\n'
        '\n'
        '    Args:\n'
        '        subtotal: 折扣前小計\n'
        '        discount_percent: 折扣百分比（0-50）\n'
        '\n'
        '    Returns:\n'
        '        折扣後金額\n'
        '    """\n'
        '    if discount_percent < 0 or discount_percent > MAX_DISCOUNT_PERCENT:\n'
        '        raise ValueError(f"無效的折扣: {discount_percent}%")\n'
        '    if discount_percent == 0:\n'
        '        return subtotal\n'
        '    discount_amount = calculate_percentage(subtotal, discount_percent)\n'
        '    return round(subtotal - discount_amount, 2)\n',
        encoding='utf-8',
    )

    # ── services ──

    # services/pricing.py — 第 2 層：呼叫 promotions
    (sandbox / 'services' / 'pricing.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'from core.promotions import apply_promotion\n'
        '\n\n'
        'def calculate_order_total(\n'
        '    items: list[dict[str, object]],\n'
        '    discount_percent: float = 0,\n'
        ') -> float:\n'
        '    """計算訂單總金額（含折扣）。\n'
        '\n'
        '    Args:\n'
        '        items: 商品列表，每項需有 price 和 quantity\n'
        '        discount_percent: 折扣百分比\n'
        '\n'
        '    Returns:\n'
        '        折扣後總金額\n'
        '    """\n'
        '    subtotal = sum(\n'
        '        float(item["price"]) * int(item["quantity"])\n'
        '        for item in items\n'
        '    )\n'
        '    if discount_percent > 0:\n'
        '        return apply_promotion(subtotal, discount_percent)\n'
        '    return subtotal\n',
        encoding='utf-8',
    )

    # services/checkout.py — 第 1 層：呼叫 pricing
    (sandbox / 'services' / 'checkout.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'import uuid\n'
        '\n'
        'from services.pricing import calculate_order_total\n'
        '\n\n'
        'def checkout_order(\n'
        '    items: list[dict[str, object]],\n'
        '    discount_percent: float = 0,\n'
        ') -> dict[str, object]:\n'
        '    """結帳處理。\n'
        '\n'
        '    Args:\n'
        '        items: 商品列表\n'
        '        discount_percent: 折扣百分比\n'
        '\n'
        '    Returns:\n'
        '        結帳結果，含 order_id、total、status\n'
        '    """\n'
        '    total = calculate_order_total(items, discount_percent)\n'
        '    return {\n'
        '        "order_id": str(uuid.uuid4())[:8],\n'
        '        "total": total,\n'
        '        "status": "confirmed",\n'
        '    }\n',
        encoding='utf-8',
    )

    # services/inventory.py（干擾）
    (sandbox / 'services' / 'inventory.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        '_stock: dict[str, int] = {}\n'
        '\n\n'
        'def add_stock(product_name: str, quantity: int) -> None:\n'
        '    """新增庫存。"""\n'
        '    _stock[product_name] = _stock.get(product_name, 0) + quantity\n'
        '\n\n'
        'def check_stock(product_name: str) -> int:\n'
        '    """查詢庫存。"""\n'
        '    return _stock.get(product_name, 0)\n'
        '\n\n'
        'def reduce_stock(product_name: str, quantity: int) -> bool:\n'
        '    """扣減庫存，回傳是否成功。"""\n'
        '    current = _stock.get(product_name, 0)\n'
        '    if current < quantity:\n'
        '        return False\n'
        '    _stock[product_name] = current - quantity\n'
        '    return True\n',
        encoding='utf-8',
    )

    # services/customer_service.py（干擾）
    (sandbox / 'services' / 'customer_service.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'from core.utils.validators import validate_email\n'
        'from models.customer import Customer\n'
        '\n\n'
        'def register_customer(name: str, email: str) -> Customer:\n'
        '    """註冊新客戶。"""\n'
        '    if not validate_email(email):\n'
        '        raise ValueError(f"無效的 email: {email}")\n'
        '    return Customer(name=name, email=email)\n',
        encoding='utf-8',
    )

    # services/notification.py（干擾）
    (sandbox / 'services' / 'notification.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'from core.utils.formatters import format_currency, format_order_id\n'
        '\n\n'
        'def build_confirmation_message(\n'
        '    order_id: str, total: float, email: str\n'
        ') -> str:\n'
        '    """產生訂單確認訊息。"""\n'
        '    return (\n'
        '        f"訂單 {format_order_id(order_id)} 已確認。"\n'
        '        f"金額: {format_currency(total)}。"\n'
        '        f"確認信已寄至 {email}。"\n'
        '    )\n',
        encoding='utf-8',
    )

    # services/handlers/discount_handler.py（干擾 — 看起來相關但無 bug）
    (sandbox / 'services' / 'handlers' / 'discount_handler.py').write_text(
        'from __future__ import annotations\n'
        '\n'
        'from core.config import MAX_DISCOUNT_PERCENT\n'
        '\n\n'
        'def validate_discount_code(code: str) -> float:\n'
        '    """驗證折扣碼並回傳折扣百分比。\n'
        '\n'
        '    目前支援的折扣碼：\n'
        '    - SAVE10: 10% 折扣\n'
        '    - SAVE20: 20% 折扣\n'
        '    """\n'
        '    codes = {"SAVE10": 10.0, "SAVE20": 20.0}\n'
        '    discount = codes.get(code, 0.0)\n'
        '    if discount > MAX_DISCOUNT_PERCENT:\n'
        '        raise ValueError(f"折扣超過上限: {discount}%")\n'
        '    return discount\n',
        encoding='utf-8',
    )

    # ── 測試 ──

    # test_checkout.py — 失敗的測試（折扣計算結果不對）
    (sandbox / 'test_checkout.py').write_text(
        'from services.checkout import checkout_order\n'
        '\n\n'
        'def test_checkout_no_discount() -> None:\n'
        '    items = [{"name": "Book", "price": 100.0, "quantity": 2}]\n'
        '    result = checkout_order(items)\n'
        '    assert result["total"] == 200.0\n'
        '    assert result["status"] == "confirmed"\n'
        '\n\n'
        'def test_checkout_with_discount() -> None:\n'
        '    """10% 折扣：1000 元應變為 900 元。"""\n'
        '    items = [\n'
        '        {"name": "Laptop", "price": 500.0, "quantity": 1},\n'
        '        {"name": "Mouse", "price": 250.0, "quantity": 2},\n'
        '    ]\n'
        '    result = checkout_order(items, discount_percent=10)\n'
        '    # 500 + 250*2 = 1000, 10% off = 900\n'
        '    assert result["total"] == 900.0\n',
        encoding='utf-8',
    )

    # test_inventory.py — 通過的測試（干擾）
    (sandbox / 'test_inventory.py').write_text(
        'from services.inventory import add_stock, check_stock, reduce_stock\n'
        '\n\n'
        'def test_add_and_check_stock() -> None:\n'
        '    add_stock("Widget", 50)\n'
        '    assert check_stock("Widget") == 50\n'
        '\n\n'
        'def test_reduce_stock_success() -> None:\n'
        '    add_stock("Gadget", 10)\n'
        '    assert reduce_stock("Gadget", 3) is True\n'
        '    assert check_stock("Gadget") == 7\n'
        '\n\n'
        'def test_reduce_stock_insufficient() -> None:\n'
        '    assert reduce_stock("Nonexistent", 1) is False\n',
        encoding='utf-8',
    )

    # test_customer.py — 通過的測試（干擾）
    (sandbox / 'test_customer.py').write_text(
        'import pytest\n'
        '\n'
        'from services.customer_service import register_customer\n'
        '\n\n'
        'def test_register_valid_customer() -> None:\n'
        '    customer = register_customer("Alice", "alice@example.com")\n'
        '    assert customer.name == "Alice"\n'
        '\n\n'
        'def test_register_invalid_email() -> None:\n'
        '    with pytest.raises(ValueError, match="無效的 email"):\n'
        '        register_customer("Bob", "invalid")\n',
        encoding='utf-8',
    )


def evaluate(
    sandbox: Path,
    events: list[AgentEvent],
    conversation: list[MessageParam],
) -> EvalResult:
    """評估修復結果。"""
    details: dict[str, Any] = {}

    calculator_file = sandbox / 'core' / 'utils' / 'calculator.py'
    if not calculator_file.exists():
        return EvalResult(
            task_name=TASK_NAME,
            task_level=TASK_LEVEL,
            passed=False,
            score=0.0,
            details={'error': 'core/utils/calculator.py 不存在'},
        )

    content = calculator_file.read_text(encoding='utf-8')
    # 檢查百分比計算是否正確：除以 100 而非 1000
    details['has_correct_divisor'] = '/ 100' in content or '/100' in content

    # 計算 tool call 數量與類型
    tool_calls = [
        e['data']['name']
        for e in events
        if e['type'] == 'tool_call' and e['data'].get('status') == 'completed'
    ]
    details['tool_count'] = len(tool_calls)
    details['used_grep'] = 'grep_search' in tool_calls
    details['tool_sequence'] = tool_calls

    # 執行 pytest 驗證
    passed, output = run_pytest_in_sandbox(sandbox)
    details['pytest_passed'] = passed
    details['pytest_output'] = output[:1000]

    # 評分
    score = 0.0
    if details['has_correct_divisor']:
        score += 0.3
    if passed:
        score += 0.5
    # 效率加分：tool call 少於等於 13 次
    if details['tool_count'] <= 13:
        score += 0.2
    details['efficiency_bonus'] = details['tool_count'] <= 13

    return EvalResult(
        task_name=TASK_NAME,
        task_level=TASK_LEVEL,
        passed=passed,
        score=score,
        details=details,
    )
