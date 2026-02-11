# language: zh-TW
Feature: Sandbox 沙箱環境
  作為開發者
  我想要一個可抽換的沙箱環境
  以便在不同部署模式下安全限制工具的操作範圍

  Background:
    Given Sandbox 提供兩個核心方法：validate_path 和 exec

  Rule: Sandbox 應驗證路徑不超出沙箱範圍

    Scenario: 驗證合法路徑
      Given 沙箱已初始化
      When 透過沙箱驗證路徑 "src/main.py"
      Then 應回傳正規化後的絕對路徑

    Scenario: 阻擋路徑穿越攻擊
      Given 沙箱已初始化
      When 透過沙箱驗證路徑 "../../../etc/passwd"
      Then 應拋出 PermissionError

    Scenario: 阻擋絕對路徑存取
      Given 沙箱已初始化
      When 透過沙箱驗證路徑 "/etc/passwd"
      Then 應拋出 PermissionError

  Rule: Sandbox 應在沙箱範圍內執行指令

    Scenario: 執行指令
      Given 沙箱已初始化
      When 透過沙箱執行指令 "echo hello"
      Then 應回傳 exit_code 為 0
      And stdout 應包含 "hello"

    Scenario: 指定工作目錄執行指令
      Given 沙箱內存在子目錄 "src"
      When 透過沙箱執行指令 "ls" 並指定工作目錄為 "src"
      Then stdout 應包含該目錄的檔案清單

    Scenario: 指令執行超時
      Given 沙箱已初始化
      When 透過沙箱執行耗時超過超時設定的指令
      Then 應拋出 TimeoutError

    Scenario: 工作目錄路徑穿越應被阻擋
      Given 沙箱已初始化
      When 透過沙箱執行指令並指定工作目錄為 "../../"
      Then 應拋出 PermissionError

  Rule: LocalSandbox 應在本地檔案系統的指定根目錄內操作

    Scenario: 在根目錄內執行指令
      Given LocalSandbox 根目錄為暫存目錄
      When 透過沙箱執行指令 "pwd"
      Then stdout 應包含該暫存目錄的路徑

    Scenario: validate_path 回傳根目錄
      Given LocalSandbox 根目錄為暫存目錄
      When 透過沙箱驗證路徑 "."
      Then 應回傳該暫存目錄的絕對路徑

    Scenario: 失敗指令回傳非零 exit_code
      Given LocalSandbox 已初始化
      When 透過沙箱執行指令 "false"
      Then exit_code 應不為 0

  Rule: Tool handler 應透過 Sandbox 介面操作

    Scenario: create_default_registry 接受 Sandbox 參數
      Given 一個 LocalSandbox 實例
      When 使用該 Sandbox 建立預設工具註冊表
      Then 所有檔案工具應已註冊
      And bash 工具應透過 Sandbox 限制的路徑範圍執行

    Scenario: 檔案工具使用 Sandbox 的路徑驗證
      Given 一個 LocalSandbox 實例的工具註冊表
      When 透過 read_file 工具讀取沙箱內的檔案
      Then 應正常回傳檔案內容

    Scenario: 檔案工具阻擋路徑穿越
      Given 一個 LocalSandbox 實例的工具註冊表
      When 透過 read_file 工具讀取 "../../../etc/passwd"
      Then 應拋出 PermissionError
