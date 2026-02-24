"""Provider 基礎介面定義。

定義 LLMProvider Protocol 與相關資料結構。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_core.types import ContentBlock, MessageParam, StopReason, ToolDefinition


@dataclass
class UsageInfo:
    """API 使用量資訊。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FinalMessage:
    """Provider 回傳的最終訊息（provider-agnostic）。

    Attributes:
        content: 內容區塊列表（ContentBlock 型別）
        stop_reason: 停止原因（"end_turn" | "tool_use" 等）
        usage: API 使用量資訊
    """

    content: list[ContentBlock]
    stop_reason: StopReason
    usage: UsageInfo | None = None


class StreamResult:
    """串流回應結果。

    封裝串流的 text iterator 與取得最終結果的方法。

    Attributes:
        text_stream: 逐步 yield text token 的 async iterator
        get_final_result: 取得最終結果的 async callable
    """

    def __init__(
        self,
        text_stream: AsyncIterator[str],
        get_final_result: Callable[[], Awaitable[FinalMessage]],
    ) -> None:
        self.text_stream = text_stream
        self.get_final_result = get_final_result


@runtime_checkable
class LLMProvider(Protocol):
    """LLM Provider 介面。

    所有 Provider 實作都必須符合此協定。
    """

    def stream(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> Any:
        """建立串流回應的 async context manager。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數

        Returns:
            AsyncContextManager[StreamResult]
        """
        ...

    async def count_tokens(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        """計算給定訊息的 token 數量。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數

        Returns:
            input token 數量
        """
        ...

    async def create(
        self,
        messages: list[MessageParam],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        """非串流呼叫，用於摘要等短回應場景。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            max_tokens: 最大回應 token 數

        Returns:
            完整的回應訊息
        """
        ...
