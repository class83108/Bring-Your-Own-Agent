"""型別定義模組。

定義專案中使用的自定義型別與資料結構。
透過 Literal discriminated union 讓 Pyright 自動窄化型別，減少 cast 需求。
"""

from __future__ import annotations

from typing import Any, Literal, Required, TypedDict

# --- Content Block（Discriminated Union） ---


class TextBlock(TypedDict):
    """文字內容區塊。"""

    type: Literal['text']
    text: str


class ToolUseBlock(TypedDict):
    """工具調用區塊。"""

    type: Literal['tool_use']
    id: str
    name: str
    input: dict[str, object]


class ToolResultBlock(TypedDict, total=False):
    """工具結果區塊。

    is_error 為選填欄位，僅在工具執行失敗時設為 True。
    """

    type: Required[Literal['tool_result']]
    tool_use_id: Required[str]
    content: Required[str]
    is_error: bool


class ImageBlock(TypedDict):
    """圖片內容區塊。"""

    type: Literal['image']
    source: dict[str, Any]


class DocumentBlock(TypedDict):
    """文件（PDF）內容區塊。"""

    type: Literal['document']
    source: dict[str, Any]


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock | DocumentBlock
"""所有內容區塊型別的聯合型別。"""


# --- Stop Reason ---

StopReason = Literal['end_turn', 'tool_use', 'max_tokens']
"""標準化的停止原因型別。

各 Provider 負責將 vendor 特定值映射為此型別：
- end_turn: 正常結束回應
- tool_use: 需要執行工具調用
- max_tokens: 達到 token 上限而截斷
"""


# --- Tool Definition ---


class ToolDefinition(TypedDict):
    """標準化的工具定義格式。

    Provider-agnostic 的工具描述，各 Provider 負責轉換為 vendor 格式。

    Attributes:
        name: 工具名稱
        description: 工具描述
        input_schema: JSON Schema 格式的參數定義
    """

    name: str
    description: str
    input_schema: dict[str, Any]


# --- Message ---


class MessageParam(TypedDict):
    """對話訊息型別。

    Attributes:
        role: 訊息角色（user 或 assistant）
        content: 訊息內容（純文字或內容區塊列表）
    """

    role: Literal['user', 'assistant']
    content: str | list[ContentBlock]


# --- Agent Event ---


class AgentEvent(TypedDict):
    """Agent 事件通知型別。

    用於串流回應中的工具調用狀態、compact 通知等。

    Attributes:
        type: 事件類型（tool_call、compact、preamble_end 等）
        data: 事件附帶資料
    """

    type: str
    data: dict[str, Any]


# --- Compact Result ---


class CompactResult(TypedDict):
    """上下文壓縮結果。

    Attributes:
        truncated: 被截斷的 tool_result 數量
        summarized: 是否執行了 LLM 摘要
        summary: 摘要文字（未摘要時為 None）
    """

    truncated: int
    summarized: bool
    summary: str | None
