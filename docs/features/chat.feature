# language: zh-TW
Feature: 基礎聊天功能
  作為使用者
  我想要與 Agent 進行對話
  以便獲得程式開發上的協助

  Background:
    Given Agent 已啟動
    And 系統提示詞已設定

  Rule: Agent 應驗證使用者輸入

    Scenario: 使用者發送空白訊息
      When 使用者輸入空白訊息
      Then Agent 應拋出 ValueError
      And 錯誤訊息應提示使用者輸入有效內容

  Rule: Agent 應維護對話歷史

    Scenario: 單輪對話後歷史正確記錄
      When 使用者發送一則訊息
      And Agent 回應完成
      Then 對話歷史應包含一組 user 和 assistant 訊息

    Scenario: 多輪對話後歷史正確累積
      Given 使用者已完成第一輪對話
      When 使用者發送第二則訊息
      And Agent 回應完成
      Then 對話歷史應包含兩組 user 和 assistant 訊息

    Scenario: 重設對話歷史
      Given 使用者已進行過對話
      When 呼叫重設對話功能
      Then 對話歷史應為空

  Rule: Agent 應正確處理錯誤情況

    Scenario: API 連線失敗
      Given API 服務無法連線
      When 使用者發送訊息
      Then Agent 應拋出 ConnectionError
      And 錯誤訊息應建議使用者稍後重試
      And 對話歷史不應被修改

    Scenario: API 金鑰無效
      Given API 金鑰設定錯誤
      When 使用者發送訊息
      Then Agent 應拋出 PermissionError
      And 錯誤訊息應說明如何設定 API 金鑰
      And 對話歷史不應被修改

    Scenario: API 回應超時
      Given API 回應超過超時閾值
      When 使用者發送訊息
      Then Agent 應拋出 TimeoutError
      And 對話歷史不應被修改

  Rule: Agent 應支援串流回應

    Scenario: 串流方式逐步回傳 token
      When 使用者發送訊息
      Then Agent 應以 AsyncIterator 逐步 yield 回應 token
      And 所有 token 組合後應為完整回應

    Scenario: 串流中斷時保留部分回應
      Given Agent 正在串流回應
      And 已收到部分 token
      When 串流連線意外中斷
      Then Agent 應將已收到的部分回應存入對話歷史
      And Agent 應拋出 ConnectionError 提示中斷
