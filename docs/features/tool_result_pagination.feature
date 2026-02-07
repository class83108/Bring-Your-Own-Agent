# language: zh-TW
Feature: Tool Result 分頁
  作為 Agent 系統
  我想要在工具結果過大時自動分頁
  以便避免單次工具結果衝破 context window 限制

  Background:
    Given ToolRegistry 已初始化
    And 結果大小上限設定為 MAX_RESULT_CHARS

  Rule: 小結果應直接回傳，不受影響

    Scenario: 工具結果未超過上限
      Given 已註冊一個工具，回傳 100 字元的結果
      When 執行該工具
      Then 應回傳完整的原始結果
      And 不應產生任何分頁暫存

  Rule: 超大結果應自動分頁並提供 read_more 機制

    Scenario: 工具結果超過上限時回傳第一頁
      Given 已註冊一個工具，回傳 60000 字元的結果
      When 執行該工具
      Then 回傳內容應只包含前 MAX_RESULT_CHARS 字元
      And 回傳內容應包含分頁提示（result_id、目前頁數、總頁數）
      And 完整結果應被儲存至暫存區

    Scenario: 透過 read_more 取得後續頁面
      Given 已有一筆分頁暫存結果（共 3 頁）
      When 呼叫 read_more 並指定 page=2
      Then 應回傳第 2 頁的內容
      And 回傳內容應包含分頁提示

    Scenario: 透過 read_more 取得最後一頁
      Given 已有一筆分頁暫存結果（共 3 頁）
      When 呼叫 read_more 並指定 page=3
      Then 應回傳第 3 頁的內容
      And 分頁提示應標示為最後一頁

  Rule: read_more 應處理無效請求

    Scenario: 查詢不存在的 result_id
      When 呼叫 read_more 並指定不存在的 result_id
      Then 應回傳錯誤訊息說明結果不存在或已過期

    Scenario: 查詢超出範圍的頁數
      Given 已有一筆分頁暫存結果（共 3 頁）
      When 呼叫 read_more 並指定 page=5
      Then 應回傳錯誤訊息說明頁數超出範圍

  Rule: 暫存區應有生命週期管理

    Scenario: 暫存結果在對話中可重複存取
      Given 已有一筆分頁暫存結果
      When 連續呼叫 read_more 取得第 1 頁兩次
      Then 兩次應回傳相同的內容

    Scenario: 清除暫存區
      Given 暫存區包含多筆分頁結果
      When 呼叫清除暫存區
      Then 暫存區應為空
      And 之前的 result_id 應不再可用
