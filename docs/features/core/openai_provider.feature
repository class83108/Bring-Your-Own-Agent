# language: zh-TW
Feature: OpenAI Provider
  作為開發者
  我想要支援 OpenAI 模型作為 LLM Provider
  以便在 GPT-4o 等 OpenAI 模型與 Claude 之間自由切換

  Rule: OpenAI Provider 應實作 LLMProvider 介面

    Scenario: OpenAI Provider 串流回應
      Given 已建立 OpenAIProvider
      When 透過 Provider 發送訊息
      Then 應以 AsyncIterator 逐步回傳 token
      And 最終回傳包含 content 和 stop_reason 的結果

    Scenario: OpenAI Provider 處理工具調用
      Given 已建立 OpenAIProvider
      When GPT 回應包含 tool_calls
      Then Provider 應回傳 stop_reason 為 "tool_use"
      And content 應包含轉換後的 ToolUseBlock

    Scenario: OpenAI Provider 非串流呼叫
      Given 已建立 OpenAIProvider
      When 透過 Provider.create() 發送訊息
      Then 應回傳完整的 FinalMessage

  Rule: OpenAI Provider 應正確轉換訊息格式

    Scenario: system prompt 轉為 system role 訊息
      Given 已建立 OpenAIProvider
      When 建立 API 請求時帶有 system prompt
      Then 第一則訊息應為 {"role": "system", "content": "..."}

    Scenario: tool_result 區塊轉為 tool role 訊息
      Given 對話歷史包含 user 訊息中嵌入的 tool_result block
      When OpenAI Provider 轉換訊息格式
      Then tool_result 應變為獨立的 {"role": "tool"} 訊息
      And tool_call_id 應正確對應

    Scenario: ToolDefinition 轉為 OpenAI function 格式
      Given 有標準格式的 ToolDefinition
      When OpenAI Provider 建立 API 請求
      Then 工具應包裝為 {"type": "function", "function": {...}}
      And input_schema 應映射為 parameters

    Scenario: assistant 訊息中的 ToolUseBlock 轉為 tool_calls
      Given 對話歷史包含 assistant 訊息中的 ToolUseBlock
      When OpenAI Provider 轉換訊息格式
      Then ToolUseBlock 應轉為 OpenAI 的 tool_calls 格式

  Rule: OpenAI Provider 應映射 StopReason

    Scenario: "stop" 映射為 "end_turn"
      Given OpenAI API 回傳 finish_reason 為 "stop"
      Then FinalMessage 的 stop_reason 應為 "end_turn"

    Scenario: "tool_calls" 映射為 "tool_use"
      Given OpenAI API 回傳 finish_reason 為 "tool_calls"
      Then FinalMessage 的 stop_reason 應為 "tool_use"

    Scenario: "length" 映射為 "max_tokens"
      Given OpenAI API 回傳 finish_reason 為 "length"
      Then FinalMessage 的 stop_reason 應為 "max_tokens"

    Scenario: 未知 finish_reason 應回退為 end_turn
      Given OpenAI API 回傳 finish_reason 為 None 或未知值
      Then FinalMessage 的 stop_reason 應回退為 "end_turn"

  Rule: OpenAI Provider 應轉換特定例外為通用例外

    Scenario: API 金鑰無效
      Given OpenAI API 回傳 AuthenticationError
      When Provider 處理該錯誤
      Then 應拋出 ProviderAuthError

    Scenario: API 連線失敗
      Given OpenAI API 回傳 APIConnectionError
      When Provider 處理該錯誤
      Then 應拋出 ProviderConnectionError

    Scenario: API 回應超時
      Given OpenAI API 回傳 APITimeoutError
      When Provider 處理該錯誤
      Then 應拋出 ProviderTimeoutError

    Scenario: API 速率限制
      Given OpenAI API 回傳 429 RateLimitError
      When Provider 處理該錯誤
      Then 應拋出 ProviderRateLimitError

  Rule: OpenAI Provider 應支援 Prompt Caching 使用量回報

    Scenario: 回報快取命中的 token 數
      Given 已建立 OpenAIProvider
      When OpenAI API 回應包含 prompt_tokens_details.cached_tokens
      Then UsageInfo 的 cache_read_input_tokens 應對應該值

    Scenario: 無快取命中時 cache_read 為 0
      Given 已建立 OpenAIProvider
      When OpenAI API 回應不包含 cached_tokens
      Then UsageInfo 的 cache_read_input_tokens 應為 0

  Rule: OpenAI Provider 應支援 token 計數

    Scenario: 使用 tiktoken 計算 token 數
      Given 已建立 OpenAIProvider 使用 gpt-4o 模型
      When 呼叫 count_tokens
      Then 應使用 tiktoken 本地計算回傳正整數

  Rule: ProviderConfig 應根據 provider_type 讀取正確的環境變數

    Scenario: OpenAI provider 讀取 OPENAI_API_KEY
      Given provider_type 為 "openai"
      And 環境變數 OPENAI_API_KEY 已設定
      When 呼叫 get_api_key()
      Then 應回傳 OPENAI_API_KEY 的值

    Scenario: Anthropic provider 仍讀取 ANTHROPIC_API_KEY
      Given provider_type 為 "anthropic"
      And 環境變數 ANTHROPIC_API_KEY 已設定
      When 呼叫 get_api_key()
      Then 應回傳 ANTHROPIC_API_KEY 的值
