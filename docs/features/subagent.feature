# language: zh-TW
Feature: Subagent 子代理機制
  作為開發者
  我想要讓 Agent 能建立子 Agent 來分工處理任務
  以便並行處理獨立的子任務並保持各自的 context 隔離

  Background:
    Given Agent 已初始化
    And 工具註冊表包含 "create_subagent" 工具

  Rule: Agent 應能建立子 Agent

    Scenario: 建立子 Agent 執行獨立任務
      When Agent 建立一個子 Agent 並指派任務 "列出所有 Python 檔案"
      Then 子 Agent 應被建立
      And 子 Agent 應執行指派的任務
      And 子 Agent 應回傳執行結果給父 Agent

    Scenario: 子 Agent 使用與父 Agent 相同的 Sandbox
      Given Agent 使用 LocalSandbox
      When Agent 建立子 Agent
      Then 子 Agent 應使用相同的 Sandbox 實例

  Rule: 子 Agent 的工具應受到限制

    Scenario: 子 Agent 預設不能建立子 Agent
      When Agent 建立子 Agent
      Then 子 Agent 的工具清單不應包含 "create_subagent"

    Scenario: 子 Agent 擁有其餘所有工具
      Given 父 Agent 的工具清單為 "read_file", "edit_file", "bash", "create_subagent"
      When Agent 建立子 Agent
      Then 子 Agent 的工具清單應為 "read_file", "edit_file", "bash"

  Rule: 子 Agent 應有獨立的 context

    Scenario: 子 Agent 的對話歷史與父 Agent 獨立
      When Agent 建立子 Agent 並指派任務
      Then 子 Agent 應有自己的對話歷史
      And 子 Agent 的對話歷史不應影響父 Agent

    Scenario: 子 Agent 完成後回傳摘要
      When Agent 建立子 Agent 並指派任務
      And 子 Agent 完成任務
      Then 父 Agent 應收到子 Agent 的執行結果摘要
      And 父 Agent 不應收到子 Agent 的完整對話歷史
