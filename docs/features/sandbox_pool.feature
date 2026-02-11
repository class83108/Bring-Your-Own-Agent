# language: zh-TW
Feature: RunnerPool 容器池
  作為應用層開發者
  我想要一個管理多個 ContainerRunner 生命週期的可選工具
  以便在多租戶場景下為每個使用者分配獨立的容器化 Agent 環境

  Background:
    Given RunnerPool 屬於應用層的可選 utility
    And 每個 ContainerRunner 對應一個容器化的 Agent 實例

  Rule: RunnerPool 應管理 Runner 的取得與釋放

    Scenario: 以 key 取得 Runner
      Given RunnerPool 已初始化
      When 以 key "user_a" 取得 Runner
      Then 應回傳一個已啟動的 ContainerRunner

    Scenario: 相同 key 取得相同 Runner
      Given RunnerPool 已初始化
      And 已以 key "user_a" 取得 Runner
      When 再次以 key "user_a" 取得 Runner
      Then 應回傳同一個 ContainerRunner

    Scenario: 不同 key 取得不同 Runner
      Given RunnerPool 已初始化
      When 以 key "user_a" 取得 Runner
      And 以 key "user_b" 取得 Runner
      Then 兩個 ContainerRunner 應為不同實例

    Scenario: 釋放 Runner
      Given 已以 key "user_a" 取得 Runner
      When 釋放 key "user_a" 的 Runner
      Then 該 Runner 的容器應被停止並清除
      And 再次以 key "user_a" 取得時應建立新的 Runner

  Rule: RunnerPool 應限制同時存在的容器數量

    Scenario: 超過上限時回收最久未使用的 Runner
      Given RunnerPool 上限為 2
      And 已以 key "user_a" 取得 Runner
      And 已以 key "user_b" 取得 Runner
      When 以 key "user_c" 取得 Runner
      Then 最久未使用的 Runner 應被回收
      And "user_c" 應取得新的 Runner

  Rule: RunnerPool 應支援閒置超時回收

    Scenario: 閒置超過設定時間的 Runner 應被回收
      Given RunnerPool 閒置超時為 1 秒
      And 已以 key "user_a" 取得 Runner
      When 等待超過 1 秒
      Then "user_a" 的 Runner 應被自動回收

  Rule: RunnerPool 應使用工廠函數建立 Runner

    Scenario: 使用自訂工廠函數
      Given RunnerPool 工廠函數指定映像與 mount 配置
      When 以 key "user_a" 取得 Runner
      Then 應回傳使用該配置的 ContainerRunner
