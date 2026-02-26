# language: zh-TW
Feature: Gemini Provider
  作為開發者
  我想要支援 Google Gemini 模型作為 LLM Provider
  以便在 Gemini 2.0 等 Google 模型與其他 Provider 之間自由切換

  Rule: Gemini Provider 應實作 LLMProvider 介面

    Scenario: Gemini Provider 串流回應
      Given 已建立 GeminiProvider
      When 透過 Provider 發送訊息
      Then 應以 AsyncIterator 逐步回傳 token
      And 最終回傳包含 content 和 stop_reason 的結果

    Scenario: Gemini Provider 處理工具調用
      Given 已建立 GeminiProvider
      When Gemini 回應包含 function_call Part
      Then Provider 應回傳 stop_reason 為 "tool_use"
      And content 應包含轉換後的 ToolUseBlock

    Scenario: Gemini Provider 非串流呼叫
      Given 已建立 GeminiProvider
      When 透過 Provider.create() 發送訊息
      Then 應回傳完整的 FinalMessage

  Rule: Gemini Provider 應正確轉換訊息格式

    Scenario: system prompt 轉為 system_instruction
      Given 已建立 GeminiProvider
      When 建立 API 請求時帶有 system prompt
      Then 應透過 GenerateContentConfig 的 system_instruction 傳入

    Scenario: assistant 角色轉為 model 角色
      Given 對話歷史包含 role 為 "assistant" 的訊息
      When Gemini Provider 轉換訊息格式
      Then 應轉為 Content 的 role 為 "model"

    Scenario: tool_result 區塊轉為 function_response Part
      Given 對話歷史包含 user 訊息中嵌入的 tool_result block
      When Gemini Provider 轉換訊息格式
      Then tool_result 應變為 Content(role='user') 中的 function_response Part

    Scenario: ToolDefinition 轉為 FunctionDeclaration
      Given 有標準格式的 ToolDefinition
      When Gemini Provider 建立 API 請求
      Then 工具應轉為 FunctionDeclaration
      And input_schema 應映射為 parameters

    Scenario: assistant 訊息中的 ToolUseBlock 轉為 function_call Part
      Given 對話歷史包含 assistant 訊息中的 ToolUseBlock
      When Gemini Provider 轉換訊息格式
      Then ToolUseBlock 應轉為 model 角色 Content 中的 function_call Part

  Rule: Gemini Provider 應映射 StopReason

    Scenario: "STOP" 映射為 "end_turn"
      Given Gemini API 回傳 finish_reason 為 "STOP"
      Then FinalMessage 的 stop_reason 應為 "end_turn"

    Scenario: 包含 function_call 時應回傳 "tool_use"
      Given Gemini API 回應包含 function_call Part
      Then FinalMessage 的 stop_reason 應為 "tool_use"
      # 注意：Gemini 無明確的 tool_use finish_reason，需透過檢查回應內容判斷

    Scenario: "MAX_TOKENS" 映射為 "max_tokens"
      Given Gemini API 回傳 finish_reason 為 "MAX_TOKENS"
      Then FinalMessage 的 stop_reason 應為 "max_tokens"

    Scenario: 安全過濾與未知 finish_reason 應回退為 end_turn
      Given Gemini API 回傳 finish_reason 為 "SAFETY" 或未知值
      Then FinalMessage 的 stop_reason 應回退為 "end_turn"

  Rule: Gemini Provider 應轉換特定例外為通用例外

    Scenario: API 金鑰無效
      Given Gemini API 回傳 401/403 錯誤
      When Provider 處理該錯誤
      Then 應拋出 ProviderAuthError

    Scenario: API 速率限制
      Given Gemini API 回傳 429 錯誤
      When Provider 處理該錯誤
      Then 應拋出 ProviderRateLimitError

    Scenario: API 連線失敗
      Given Gemini API 發生連線錯誤
      When Provider 處理該錯誤
      Then 應拋出 ProviderConnectionError

    Scenario: API 回應超時
      Given Gemini API 發生超時錯誤
      When Provider 處理該錯誤
      Then 應拋出 ProviderTimeoutError

  Rule: Gemini Provider 應支援 token 計數

    Scenario: 使用 Gemini API 計算 token 數
      Given 已建立 GeminiProvider
      When 呼叫 count_tokens
      Then 應透過 Gemini API 的 count_tokens 回傳正整數

  Rule: ProviderConfig 應根據 provider_type 讀取正確的環境變數

    Scenario: Gemini provider 讀取 GEMINI_API_KEY
      Given provider_type 為 "gemini"
      And 環境變數 GEMINI_API_KEY 已設定
      When 呼叫 get_api_key()
      Then 應回傳 GEMINI_API_KEY 的值
