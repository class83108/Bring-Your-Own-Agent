# language: zh-TW
Feature: Session 後端抽象
  作為開發者
  我想要可抽換的 Session 後端
  以便在不同環境使用不同的儲存方式

  Rule: 記憶體後端應支援基本操作

    Scenario: 儲存並讀取對話歷史
      Given 使用 MemorySessionBackend
      When 儲存一段對話歷史到 session "abc"
      And 讀取 session "abc" 的歷史
      Then 應回傳相同的對話歷史

    Scenario: 讀取不存在的 session
      Given 使用 MemorySessionBackend
      When 讀取 session "not-exist" 的歷史
      Then 應回傳空列表

    Scenario: 重設 session
      Given 使用 MemorySessionBackend
      And session "abc" 已有對話歷史
      When 重設 session "abc"
      Then 讀取 session "abc" 應回傳空列表

  Rule: Redis 後端應支援持久化

    Scenario: 對話歷史應有 TTL
      Given 使用 RedisSessionBackend 且 TTL 為 86400
      When 儲存一段對話歷史
      Then Redis key 的 TTL 應為 86400 秒

  Rule: SQLite 後端應支援基本 CRUD 操作

    Scenario: 儲存並讀取對話歷史
      Given 使用 SQLiteSessionBackend
      When 儲存一段對話歷史到 session "abc"
      And 讀取 session "abc" 的歷史
      Then 應回傳相同的對話歷史

    Scenario: 讀取不存在的 session
      Given 使用 SQLiteSessionBackend
      When 讀取 session "not-exist" 的歷史
      Then 應回傳空列表

    Scenario: 重設 session
      Given 使用 SQLiteSessionBackend
      And session "abc" 已有對話歷史
      When 重設 session "abc"
      Then 讀取 session "abc" 應回傳空列表

  Rule: SQLite 後端應跨程序存活

    Scenario: 關閉後重新開啟仍保留資料
      Given 使用 SQLiteSessionBackend 且指定資料庫檔案路徑
      And 已儲存一段對話歷史到 session "abc"
      When 關閉 backend 並以同一路徑重新建立
      And 讀取 session "abc" 的歷史
      Then 應回傳先前儲存的對話歷史

  Rule: SQLite 後端應支援多 session 隔離

    Scenario: 不同 session 的資料互不影響
      Given 使用 SQLiteSessionBackend
      When 儲存不同對話到 session "s1" 與 session "s2"
      Then 讀取 session "s1" 應只回傳 "s1" 的對話
      And 讀取 session "s2" 應只回傳 "s2" 的對話

    Scenario: 重設單一 session 不影響其他
      Given 使用 SQLiteSessionBackend
      And session "s1" 與 session "s2" 皆有對話歷史
      When 重設 session "s1"
      Then 讀取 session "s1" 應回傳空列表
      And 讀取 session "s2" 應仍有資料

  Rule: SQLite 後端應正確序列化複雜訊息結構

    Scenario: 儲存包含 tool_use 與 tool_result 的對話
      Given 使用 SQLiteSessionBackend
      When 儲存包含工具調用區塊的對話歷史
      And 讀取該 session 的歷史
      Then 工具調用區塊應完整保留（含 type、id、name、input）
      And 工具結果區塊應完整保留（含 tool_use_id、content、is_error）

  Rule: SQLite 後端應支援列出與刪除 session

    Scenario: 列出所有 session
      Given 使用 SQLiteSessionBackend
      And 已儲存 session "s1" 與 "s2" 的對話歷史
      When 列出所有 sessions
      Then 應回傳包含 "s1" 與 "s2" 的摘要列表
      And 每筆摘要應包含 session_id、created_at、updated_at、message_count

    Scenario: 無 session 時列出回傳空列表
      Given 使用 SQLiteSessionBackend
      When 列出所有 sessions
      Then 應回傳空列表

    Scenario: 刪除 session 同時清除對話與使用量
      Given 使用 SQLiteSessionBackend
      And session "abc" 已有對話歷史與使用量記錄
      When 刪除 session "abc"
      Then 讀取 session "abc" 對話歷史應回傳空列表
      And 讀取 session "abc" 使用量應回傳空列表
      And 列出所有 sessions 不應包含 "abc"

    Scenario: 刪除不存在的 session 不應報錯
      Given 使用 SQLiteSessionBackend
      When 刪除 session "not-exist"
      Then 不應拋出例外

  Rule: Session API 應支援 RESTful 操作

    Scenario: 建立新 session
      When POST /api/sessions
      Then 回應應包含新的 session_id
      And HTTP 狀態碼為 201

    Scenario: 列出所有 sessions
      Given 已有多個 session 存在
      When GET /api/sessions
      Then 回應應包含 session 摘要列表

    Scenario: 取得特定 session 歷史
      Given session "abc" 已有對話歷史
      When GET /api/sessions/abc
      Then 回應應包含該 session 的對話歷史

    Scenario: 取得不存在的 session
      When GET /api/sessions/not-exist
      Then HTTP 狀態碼為 404

    Scenario: 刪除特定 session
      Given session "abc" 已有對話歷史
      When DELETE /api/sessions/abc
      Then HTTP 狀態碼為 200
      And session "abc" 的資料應被清除
