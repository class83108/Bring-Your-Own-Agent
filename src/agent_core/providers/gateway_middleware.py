"""Gateway Middleware 介面定義。

定義 GatewayProvider 的 middleware protocol 與 RequestContext，
支援 OpenTelemetry 相容的 trace 結構。

agent_core 僅定義介面，不依賴 opentelemetry 套件。
具體實作（OTel 整合、SQLite cost storage 等）由應用層負責。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_core.providers.base import FinalMessage, UsageInfo


@dataclass
class RequestContext:
    """Gateway 請求上下文。

    在 middleware 鏈中傳遞，攜帶 trace 資訊與請求元資料。
    支援 OpenTelemetry 相容的 trace 結構，讓 middleware 實作
    可以橋接到 OTel SDK。

    Attributes:
        trace_id: 整個請求鏈的 ID（可由外部傳入，例如 HTTP traceparent）
        span_id: 本次 LLM 呼叫的 span ID（每次呼叫自動產生）
        parent_span_id: 父層 span ID（用於串接多次 LLM 呼叫）
        model: 模型名稱
        provider_type: 推斷出的 provider 類型
        user_id: 使用者 ID（由外部設定）
        timestamp: 請求開始時間（Unix timestamp）
        metadata: 自由擴充欄位（middleware 之間共享資料）
    """

    # Trace 結構（OTel 相容）
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: str | None = None

    # 請求資訊
    model: str = ''
    provider_type: str = ''
    user_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    # 擴充欄位
    metadata: dict[str, Any] = field(default_factory=lambda: {})


# --- 通用 Middleware Protocol ---


@runtime_checkable
class GatewayMiddleware(Protocol):
    """Gateway Middleware 介面。

    在每次 LLM 呼叫前後執行，用於橫切關注點：
    - Cost tracking（after_request 記錄用量）
    - Rate limiting（before_request 檢查限額）
    - Request tracing（before 開 span、after 關 span）
    - Budget alerts（after_request 檢查閾值）

    執行順序：
    - before_request: 依註冊順序正序執行
    - after_request: 依註冊順序逆序執行
    """

    async def before_request(self, context: RequestContext) -> None:
        """LLM 呼叫前執行。

        Args:
            context: 請求上下文（可讀寫 metadata 共享資料）
        """
        ...

    async def after_request(self, context: RequestContext, result: FinalMessage) -> None:
        """LLM 呼叫後執行。

        Args:
            context: 請求上下文
            result: LLM 回傳的最終訊息
        """
        ...


# --- 專用 Protocol（使用者可選擇只實作特定關注點） ---


@runtime_checkable
class CostTracker(Protocol):
    """費用追蹤介面。

    記錄每次 LLM 呼叫的 token 用量與費用，
    支援依 user_id 查詢累計費用。
    """

    async def record_usage(self, context: RequestContext, usage: UsageInfo) -> None:
        """記錄一次 LLM 呼叫的使用量。

        Args:
            context: 請求上下文（含 model、user_id 等）
            usage: token 使用量資訊
        """
        ...

    async def get_total_cost(self, user_id: str | None = None) -> float:
        """查詢累計費用。

        Args:
            user_id: 使用者 ID（None 表示全部使用者）

        Returns:
            累計費用（美元）
        """
        ...


@runtime_checkable
class RateLimiter(Protocol):
    """速率限制介面。

    檢查請求是否超過速率限制。
    超過限額時應 raise ProviderRateLimitError。
    """

    async def check_rate_limit(self, context: RequestContext) -> None:
        """檢查速率限制。

        Args:
            context: 請求上下文（含 user_id、model 等）

        Raises:
            ProviderRateLimitError: 超過速率限制
        """
        ...


@runtime_checkable
class RequestTracer(Protocol):
    """請求追蹤介面。

    設計為可橋接 OpenTelemetry SDK。
    應用層實作可在 start_trace 中建立 OTel Span，
    在 end_trace 中結束 Span 並記錄屬性。
    """

    async def start_trace(self, context: RequestContext) -> None:
        """開始追蹤（建立 span）。

        Args:
            context: 請求上下文（含 trace_id、span_id）
        """
        ...

    async def end_trace(self, context: RequestContext, result: FinalMessage) -> None:
        """結束追蹤（關閉 span）。

        Args:
            context: 請求上下文
            result: LLM 回傳的最終訊息
        """
        ...


@runtime_checkable
class BudgetGuard(Protocol):
    """預算控管介面。

    檢查是否超出預算上限。
    超過預算時應 raise BudgetExceededError。
    """

    async def check_budget(self, context: RequestContext) -> None:
        """檢查預算。

        Args:
            context: 請求上下文（含 user_id）

        Raises:
            BudgetExceededError: 超過預算上限
        """
        ...
