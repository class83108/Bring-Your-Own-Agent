"""Agent 統一配置模組。

提供 Agent 核心的配置資料結構，支援 Provider、System Prompt 等設定。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# 預設值
DEFAULT_MODEL = 'claude-sonnet-4-20250514'
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 30.0
DEFAULT_SYSTEM_PROMPT = """你是一位專業的程式開發助手。

工作原則：
- 遇到複雜任務時，先理解需求，再逐步執行
- 執行操作前，思考是否需要先讀取相關檔案了解現況
- 解釋你的思考過程和選擇的理由
- 遇到不確定的情況，主動詢問使用者

請使用繁體中文回答。"""


@dataclass
class ProviderConfig:
    """LLM Provider 配置。

    Attributes:
        provider_type: Provider 類型識別（例如 "anthropic"）
        model: 模型名稱
        api_key: API 金鑰（可選，未指定時從環境變數讀取）
        max_tokens: 最大回應 token 數
        timeout: API 請求超時秒數
        enable_prompt_caching: 是否啟用 prompt caching
    """

    provider_type: str = 'anthropic'
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    enable_prompt_caching: bool = True
    max_retries: int = 3
    retry_initial_delay: float = 1.0

    def get_api_key(self) -> str | None:
        """取得 API Key，優先使用明確指定的值，否則從環境變數讀取。

        Returns:
            API Key 字串，若無可用 Key 則回傳 None
        """
        if self.api_key is not None:
            return self.api_key
        return os.environ.get('ANTHROPIC_API_KEY')


@dataclass
class AgentCoreConfig:
    """Agent 核心配置。

    Attributes:
        provider: LLM Provider 配置
        system_prompt: 系統提示詞
    """

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
