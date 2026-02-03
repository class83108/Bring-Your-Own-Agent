# language: zh-TW
Feature: Agent 核心架構
  作為開發者
  我想要一個穩定的 Agent Loop 架構
  以便能夠擴展各種工具並處理複雜任務

  Background:
    Given Agent 已初始化
    And API 金鑰已設定

  Rule: Agent Loop 應持續運作直到任務完成

    Scenario: 單輪對話無工具調用
      Given 使用者輸入不需要工具的問題
      When Agent 處理該輸入
      Then Agent 應直接回傳 Claude 的回應
      And 對話歷史應包含一輪交互

    Scenario: 單輪對話有工具調用
      Given 使用者輸入需要工具的問題
      When Agent 處理該輸入
      Then Agent 應執行所需工具
      And Agent 應將工具結果傳回 Claude
      And Agent 應回傳最終回應

    Scenario: 多輪工具調用
      Given 使用者輸入需要多次工具調用的複雜任務
      When Agent 處理該輸入
      Then Agent 應依序執行每輪所需的工具
      And Agent 應在所有工具完成後回傳最終回應

  Rule: Agent 應支援並行工具執行

    Scenario: 同時執行多個獨立工具
      Given Claude 回應包含多個工具調用請求
      And 這些工具調用彼此獨立
      When Agent 執行這些工具
      Then 所有工具應並行執行
      And Agent 應收集所有結果後一併回傳給 Claude

    Scenario: 並行執行中部分工具失敗
      Given Claude 回應包含三個工具調用請求
      When Agent 並行執行這些工具
      And 其中一個工具執行失敗
      Then Agent 應收集成功的結果與失敗的錯誤訊息
      And Agent 應將所有結果回傳給 Claude 處理

  Rule: Agent 應維護對話歷史

    Scenario: 保持上下文連貫性
      Given 使用者已進行過多輪對話
      When 使用者提問涉及先前對話內容
      Then Agent 應能根據歷史上下文回答

    Scenario: 對話歷史包含工具調用記錄
      Given 使用者曾觸發工具調用
      When 查看對話歷史
      Then 歷史應包含工具調用請求與結果

  Rule: Tool Registry 應能動態管理工具

    Scenario: 註冊新工具
      Given Tool Registry 已初始化
      When 註冊一個新工具
      Then 該工具應出現在可用工具列表中
      And Claude 請求應包含該工具的定義

    Scenario: 執行已註冊的工具
      Given Tool Registry 包含 "read_file" 工具
      When Claude 請求執行 "read_file" 工具
      Then Agent 應找到並執行該工具
      And 應回傳工具執行結果

    Scenario: 處理未知工具調用
      Given Tool Registry 不包含 "unknown_tool" 工具
      When Claude 請求執行 "unknown_tool" 工具
      Then Agent 應回傳錯誤訊息
      And 錯誤訊息應說明工具不存在

  Rule: Agent 應使用 Prompt Caching 優化 API 呼叫

    Scenario: 在對話歷史最後添加緩存斷點
      Given Agent 已有對話歷史
      When Agent 建立 API 請求參數
      Then 對話歷史的最後一個 message 應包含 cache_control
      And cache_control 類型應為 ephemeral

    Scenario: 不修改原始對話歷史
      Given Agent 的對話歷史包含 3 則訊息
      When Agent 建立帶有 cache_control 的 API 請求參數
      Then 原始對話歷史應保持不變
      And 原始對話歷史不應包含任何 cache_control

    Scenario: 處理字串類型的 content
      Given 對話歷史最後一則訊息的 content 是字串
      When Agent 建立 API 請求參數
      Then 該 content 應轉換為 text block 格式
      And text block 應包含 cache_control

    Scenario: 處理列表類型的 content
      Given 對話歷史最後一則訊息的 content 是列表
      When Agent 建立 API 請求參數
      Then 列表的最後一個 block 應包含 cache_control
      And 其他 blocks 不應包含 cache_control
