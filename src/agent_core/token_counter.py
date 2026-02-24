"""Token 計數模組。

追蹤當前對話的 context window token 使用量，
為後續 Compact（上下文壓縮）功能提供觸發依據。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent_core.providers.base import UsageInfo

logger = logging.getLogger(__name__)

# 各模型的 context window 大小（tokens）
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic 模型
    'claude-sonnet-4-20250514': 200_000,
    'claude-haiku-4-20250514': 200_000,
    'claude-opus-4-20250514': 200_000,
    # OpenAI 模型
    'gpt-4o': 128_000,
    'gpt-4o-mini': 128_000,
    'gpt-4.1': 1_000_000,
    'gpt-4.1-mini': 1_000_000,
    'gpt-4.1-nano': 1_000_000,
    'o1': 200_000,
    'o1-mini': 128_000,
    'o3': 200_000,
    'o3-mini': 200_000,
    'o4-mini': 200_000,
}

# 找不到模型時的預設 context window
_DEFAULT_CONTEXT_WINDOW: int = 200_000


def get_context_window(model: str) -> int:
    """取得模型的 context window 大小。

    Args:
        model: 模型名稱

    Returns:
        context window token 數
    """
    return MODEL_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)


@dataclass
class TokenCounter:
    """Token 計數器。

    追蹤當前對話佔用的 context window token 數量。
    透過每次 API 回應的 UsageInfo 更新，
    input_tokens（含快取）+ output_tokens ≈ 回應後的 context 佔用量。

    Attributes:
        context_window: 模型的 context window 上限
    """

    context_window: int = _DEFAULT_CONTEXT_WINDOW
    _last_input_tokens: int = 0
    _last_output_tokens: int = 0

    @property
    def current_context_tokens(self) -> int:
        """估算當前 context window 的 token 佔用量。

        等於最近一次 API 呼叫的 input tokens（含快取）加上 output tokens。
        """
        return self._last_input_tokens + self._last_output_tokens

    @property
    def usage_percent(self) -> float:
        """Context window 使用百分比（0-100）。"""
        if self.context_window == 0:
            return 0.0
        return self.current_context_tokens / self.context_window * 100

    def update_from_usage(self, usage: UsageInfo) -> None:
        """從 API 回應的 UsageInfo 更新 token 計數。

        input_tokens + cache_creation + cache_read = 送入 API 的總 token 數
        output_tokens = 模型回應的 token 數
        兩者之和為回應後的 context window 估算佔用量。

        Args:
            usage: API 回應的使用量資訊
        """
        self._last_input_tokens = (
            usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        )
        self._last_output_tokens = usage.output_tokens

        logger.debug(
            'Token 計數已更新',
            extra={
                'context_tokens': self.current_context_tokens,
                'context_window': self.context_window,
                'usage_percent': round(self.usage_percent, 2),
            },
        )

    def update_from_count(self, input_tokens: int) -> None:
        """從精確 count_tokens API 結果更新。

        用於在 API 呼叫前精確計算目前 conversation 的 token 數。
        此時只有 input tokens（尚無 output）。

        Args:
            input_tokens: 精確計算的 input token 數
        """
        self._last_input_tokens = input_tokens
        self._last_output_tokens = 0

        logger.debug(
            'Token 計數已精確更新',
            extra={
                'context_tokens': self.current_context_tokens,
                'usage_percent': round(self.usage_percent, 2),
            },
        )

    def set_last_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """直接設定最近一次的 token 計數。

        用於從持久化記錄還原 token 狀態。

        Args:
            input_tokens: 最後一次的 input token 數（含快取）
            output_tokens: 最後一次的 output token 數
        """
        self._last_input_tokens = input_tokens
        self._last_output_tokens = output_tokens

    def get_status(self) -> dict[str, Any]:
        """取得 token 計數狀態。

        Returns:
            包含 context token 使用量資訊的字典
        """
        return {
            'current_tokens': self.current_context_tokens,
            'context_window': self.context_window,
            'usage_percent': round(self.usage_percent, 2),
        }
