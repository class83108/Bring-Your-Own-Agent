# language: zh-TW
Feature: Gateway Provider
  作為開發者
  我想要一個能根據模型名稱自動路由的 LLM Gateway
  以便統一入口使用任意 Provider，並支援 fallback 與 middleware 擴充

  Rule: GatewayProvider 應根據模型名稱自動路由至正確的 Provider

    Scenario: claude 模型路由至 Anthropic Provider
      Given 已建立 GatewayProvider 使用模型 "claude-sonnet-4-20250514"
      When 透過 Provider 發送訊息
      Then 應委派給 AnthropicProvider 處理

    Scenario: gpt 模型路由至 OpenAI Provider
      Given 已建立 GatewayProvider 使用模型 "gpt-4o"
      When 透過 Provider 發送訊息
      Then 應委派給 OpenAIProvider 處理

    Scenario: o 系列模型路由至 OpenAI Provider
      Given 已建立 GatewayProvider 使用模型 "o3-mini"
      When 透過 Provider 發送訊息
      Then 應委派給 OpenAIProvider 處理

    Scenario: gemini 模型路由至 Gemini Provider
      Given 已建立 GatewayProvider 使用模型 "gemini-2.0-flash"
      When 透過 Provider 發送訊息
      Then 應委派給 GeminiProvider 處理

    Scenario: 無法識別的模型名稱應拋出錯誤
      Given 模型名稱為 "unknown-model-xyz"
      When 嘗試建立 GatewayProvider
      Then 應拋出 ValueError 包含無法辨識模型的訊息

  Rule: GatewayProvider 應支援注入自訂 Provider

    Scenario: 注入自訂 Provider 覆蓋自動推斷
      Given 使用者注入了自訂的 LLMProvider 實例
      When 建立 GatewayProvider
      Then 應使用注入的 Provider 而非自動推斷

  Rule: GatewayProvider 應支援 Fallback 備援

    Scenario: 主 Provider 失敗時自動切換至 Fallback
      Given 已建立 GatewayProvider 並配置 fallback_models
      And 主 Provider 發生 ProviderError
      When 透過 Provider 發送訊息
      Then 應自動嘗試 fallback 模型對應的 Provider

    Scenario: 所有 Provider 皆失敗時拋出最後的錯誤
      Given 已建立 GatewayProvider 並配置 fallback_models
      And 主 Provider 與所有 fallback Provider 皆失敗
      When 透過 Provider 發送訊息
      Then 應拋出最後一個 ProviderError

  Rule: GatewayProvider 應實作 LLMProvider 介面

    Scenario: 串流回應委派
      Given 已建立 GatewayProvider
      When 呼叫 stream()
      Then 應委派給底層 Provider 並回傳 StreamResult

    Scenario: 非串流呼叫委派
      Given 已建立 GatewayProvider
      When 呼叫 create()
      Then 應委派給底層 Provider 並回傳 FinalMessage

    Scenario: Token 計數委派
      Given 已建立 GatewayProvider
      When 呼叫 count_tokens()
      Then 應委派給底層 Provider 計算並回傳正整數

  Rule: GatewayProvider 應支援 Middleware 鏈

    Scenario: Middleware 在 LLM 呼叫前後依序執行
      Given 已建立 GatewayProvider 並註冊多個 Middleware
      When 透過 Provider 發送訊息
      Then before_request 應依正序執行
      And after_request 應依逆序執行

    Scenario: Middleware 透過 RequestContext 共享資訊
      Given 已建立 GatewayProvider 並註冊 Middleware
      When 透過 Provider 發送訊息
      Then Middleware 的 before_request 和 after_request 應收到相同的 RequestContext

  Rule: RequestContext 應支援 OpenTelemetry 相容的 trace 結構

    Scenario: 自動產生 trace_id 和 span_id
      Given 未提供外部 trace context
      When 建立 RequestContext
      Then trace_id 和 span_id 應自動產生唯一值

    Scenario: 接受外部傳入的 trace context
      Given 外部提供了 trace_id 和 parent_span_id
      When 建立 RequestContext
      Then 應使用外部的 trace_id
      And span_id 應為新產生的值
      And parent_span_id 應為外部提供的值

  Rule: GatewayProvider 應正確傳遞 ProviderConfig

    Scenario: provider_type 應根據模型名稱自動設定
      Given 模型名稱為 "gpt-4o"
      When 建立 GatewayProvider
      Then 底層 ProviderConfig 的 provider_type 應為 "openai"
      And get_api_key() 應讀取 OPENAI_API_KEY

    Scenario: 使用者指定的 api_key 應直接傳遞
      Given 使用者在 ProviderConfig 中指定了 api_key
      When 建立 GatewayProvider
      Then 底層 Provider 應使用該 api_key
