"""Provider 通用例外模組。

定義 provider-agnostic 的例外類別，讓 Agent 不需要依賴特定 SDK 的例外。
"""

from __future__ import annotations


class ProviderError(Exception):
    """Provider 基礎例外。"""


class ProviderAuthError(ProviderError):
    """認證失敗（API Key 無效或過期）。"""


class ProviderConnectionError(ProviderError):
    """連線失敗。"""


class ProviderRateLimitError(ProviderError):
    """API 速率限制（429 Too Many Requests）。"""


class ProviderTimeoutError(ProviderError):
    """請求超時。"""
