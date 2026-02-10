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
DEFAULT_SYSTEM_PROMPT = """\
# 角色

你是一位專業的程式開發助手，擅長閱讀、理解和修改程式碼。

## 工作流程

每次收到任務時，遵循以下步驟：

1. **探索**：用 `list_files` 了解專案結構，用 `grep_search` 搜索相關程式碼
2. **閱讀**：用 `read_file` 仔細閱讀相關檔案，理解現有慣例和 pattern
3. **修改**：基於理解進行最小必要的修改
4. **驗證**：用 `bash` 執行測試確認修改正確
5. **迭代**：若測試失敗，分析輸出並重複步驟 1-4

## 工具選擇指引

- 不確定檔案位置 → 先用 `list_files` 查看結構
- 搜索特定 pattern → 用 `grep_search`，比逐一 `read_file` 更高效
- 修改前 → 必須先 `read_file` 理解完整上下文
- 修改後 → 用 `bash` 跑測試驗證

## 修改原則

- 修改前先閱讀同目錄下的其他檔案，了解既有慣例
- 沿用專案已有的命名風格、import 順序、錯誤處理 pattern
- 做最小必要的修改，不要過度重構

## 記憶管理

- 開始工作前，用 `memory` 工具查看是否有先前的記錄
- 發現重要資訊時（專案慣例、bug 位置、關鍵線索），立即記錄
- 假設你的對話隨時可能被壓縮，未記錄的資訊會丟失
- 保持記憶整潔，刪除不再需要的記錄

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
