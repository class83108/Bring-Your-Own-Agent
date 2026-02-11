# language: zh-TW
Feature: ContainerRunner 容器化 Agent 執行
  作為應用層開發者
  我想要在 Docker 容器中啟動 Agent
  以便透過 OS 層級隔離確保 Agent 的操作不會影響 host 系統

  Background:
    Given ContainerRunner 屬於應用層，不在 agent_core 內
    And 容器內的 Agent 使用 agent_core 的 LocalSandbox 操作檔案
    And host 與容器之間透過 volume mount 分享檔案

  Rule: ContainerRunner 應管理容器生命週期

    Scenario: 啟動容器
      Given ContainerRunner 使用映像 "agent:latest"
      When 啟動容器
      Then 容器應在運行中
      And 容器內應有 Agent 程序

    Scenario: 停止容器
      Given ContainerRunner 已啟動容器
      When 停止容器
      Then 容器應已被移除

    Scenario: 支援 async context manager
      Given ContainerRunner 使用映像 "agent:latest"
      When 使用 "async with" 進入 runner
      Then 容器應自動啟動
      When 離開 "async with" 區塊
      Then 容器應自動停止並清除

  Rule: ContainerRunner 應透過 volume mount 分享檔案

    Scenario: 將工作目錄掛載到容器
      Given ContainerRunner 設定 mount "workspace" 為讀寫
      And 容器已啟動
      When 容器內的 Agent 在 "workspace" 寫入檔案
      Then host 端應能即時看到該檔案

    Scenario: 掛載唯讀目錄
      Given ContainerRunner 設定 mount "config" 為唯讀
      And 容器已啟動
      When 容器內的 Agent 嘗試在 "config" 寫入檔案
      Then 應寫入失敗

    Scenario: mount allowlist 控制可存取範圍
      Given ContainerRunner 設定 mount allowlist
      When 嘗試掛載不在 allowlist 中的目錄
      Then 應拋出 PermissionError

  Rule: ContainerRunner 應支援網路與資源配置

    Scenario: 預設斷開網路
      Given ContainerRunner 以預設設定啟動
      When 容器內的 Agent 嘗試存取外部網路
      Then 應連線失敗

    Scenario: 可選擇開啟網路
      Given ContainerRunner 設定 network 為 true
      And 容器已啟動
      When 容器內的 Agent 嘗試存取外部網路
      Then 應連線成功

    Scenario: 設定記憶體限制
      Given ContainerRunner 設定 memory_limit 為 "256m"
      And 容器已啟動
      When 查看容器的資源限制
      Then 記憶體上限應為 256MB

  Rule: ContainerRunner 應提供 IPC 與容器內 Agent 通訊

    Scenario: 傳送訊息給容器內的 Agent
      Given ContainerRunner 已啟動容器
      When 從 host 端傳送使用者訊息
      Then 容器內的 Agent 應接收到訊息
      And Agent 的回應應串流回 host 端

    Scenario: 串流回應包含工具事件
      Given ContainerRunner 已啟動容器
      When 從 host 端傳送需要使用工具的訊息
      Then 應以串流方式回傳 token 與工具調用事件

  Rule: 應用層應能選擇執行模式

    Scenario: Local 模式
      Given 應用層配置為 Local 模式
      When 處理聊天請求
      Then 應直接建立 Agent 並使用 LocalSandbox

    Scenario: Container 模式
      Given 應用層配置為 Container 模式
      When 處理聊天請求
      Then 應透過 ContainerRunner 將訊息轉發至容器內的 Agent
