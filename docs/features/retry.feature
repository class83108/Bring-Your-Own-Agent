# language: zh-TW
Feature: API 錯誤自動重試
  作為使用者
  我想要 Agent 在 API 呼叫失敗時自動重試
  以便暫時性錯誤不會中斷整輪對話

  Background:
    Given 已建立 AnthropicProvider
    And max_retries 設定為 3
    And retry_initial_delay 設定為 1.0 秒

  Rule: 可重試錯誤應自動重試並使用指數退避

    Scenario: 429 Rate Limit 錯誤觸發重試
      Given API 連續回傳 2 次 429 錯誤後成功
      When 透過 Provider 發送訊息
      Then 應在第 3 次嘗試成功回傳結果
      And 重試間隔應為指數退避（1秒、2秒）

    Scenario: 5xx 伺服器錯誤觸發重試
      Given API 連續回傳 1 次 500 錯誤後成功
      When 透過 Provider 發送訊息
      Then 應在第 2 次嘗試成功回傳結果

    Scenario: 網路超時觸發重試
      Given API 連續回傳 1 次 Timeout 錯誤後成功
      When 透過 Provider 發送訊息
      Then 應在第 2 次嘗試成功回傳結果

    Scenario: 連線失敗觸發重試
      Given API 連續回傳 1 次 ConnectionError 後成功
      When 透過 Provider 發送訊息
      Then 應在第 2 次嘗試成功回傳結果

  Rule: 不可重試錯誤應立即拋出

    Scenario: 401 認證錯誤不重試
      Given API 回傳 401 AuthenticationError
      When 透過 Provider 發送訊息
      Then 應立即拋出 ProviderAuthError
      And 不應有任何重試

    Scenario: 400 Bad Request 不重試
      Given API 回傳 400 錯誤
      When 透過 Provider 發送訊息
      Then 應立即拋出 ProviderError
      And 不應有任何重試

  Rule: 超過最大重試次數應拋出最後的錯誤

    Scenario: 重試耗盡後拋出例外
      Given API 持續回傳 429 錯誤
      And max_retries 設定為 3
      When 透過 Provider 發送訊息
      Then 應在 3 次重試後拋出 ProviderRateLimitError

  Rule: 重試應適用於所有 Provider 方法

    Scenario: stream() 方法支援重試
      Given API 在 stream 呼叫時回傳暫時性錯誤後成功
      When 透過 Provider.stream() 發送訊息
      Then 應自動重試並成功回傳串流結果

    Scenario: create() 方法支援重試
      Given API 在 create 呼叫時回傳暫時性錯誤後成功
      When 透過 Provider.create() 發送訊息
      Then 應自動重試並成功回傳結果

    Scenario: count_tokens() 方法支援重試
      Given API 在 count_tokens 呼叫時回傳暫時性錯誤後成功
      When 透過 Provider.count_tokens() 計算 token
      Then 應自動重試並成功回傳結果

  Rule: 重試過程可透過 SSE 通知前端

    Scenario: 重試時發送通知事件
      Given API 回傳暫時性錯誤後成功
      When Agent 的串流中觸發 Provider 重試
      Then 應透過 SSE 發送 retry 事件
      And 事件包含重試次數與錯誤類型
