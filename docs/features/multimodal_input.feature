# language: zh-TW
Feature: 多模態輸入（圖片與 PDF）
  作為使用者
  我想要在對話中傳送圖片或 PDF
  以便 Agent 能分析截圖、圖表、文件等視覺內容

  Rule: Agent 應支援接收圖片訊息

    Scenario: 使用者傳送 base64 圖片與文字
      Given Agent 已啟動
      When 使用者傳送包含 base64 圖片和文字的訊息
      Then Agent 應正確處理多模態內容
      And 對話歷史應包含 image 類型的內容區塊

    Scenario: 使用者傳送 URL 圖片與文字
      Given Agent 已啟動
      When 使用者傳送包含圖片 URL 和文字的訊息
      Then Agent 應正確處理多模態內容
      And 對話歷史應包含 image 類型的內容區塊

    Scenario: 使用者傳送多張圖片
      Given Agent 已啟動
      When 使用者傳送包含 2 張圖片和文字的訊息
      Then 對話歷史應包含 2 個 image 區塊和 1 個 text 區塊

  Rule: Agent 應支援接收 PDF 文件

    Scenario: 使用者傳送 base64 PDF 與文字
      Given Agent 已啟動
      When 使用者傳送包含 base64 PDF 和文字的訊息
      Then Agent 應正確處理多模態內容
      And 對話歷史應包含 document 類型的內容區塊

  Rule: 純文字訊息應向後相容

    Scenario: 使用者僅傳送文字
      Given Agent 已啟動
      When 使用者傳送純文字訊息
      Then Agent 應如同原本一樣正常回應
      And 對話歷史中使用者訊息的 content 應為字串

  Rule: 應驗證附件大小與格式

    Scenario: 圖片超過大小限制
      Given Agent 已啟動
      When 使用者傳送超過 20MB 的圖片
      Then 應拋出 ValueError 並提示檔案過大

    Scenario: PDF 超過大小限制
      Given Agent 已啟動
      When 使用者傳送超過 32MB 的 PDF
      Then 應拋出 ValueError 並提示檔案過大

    Scenario: 不支援的 media_type
      Given Agent 已啟動
      When 使用者傳送 media_type 為 "video/mp4" 的附件
      Then 應拋出 ValueError 並提示格式不支援

  Rule: API 層應支援附件欄位

    Scenario: 透過 JSON body 傳送 base64 圖片
      Given API 伺服器已啟動
      When 發送 POST /api/chat/stream 包含 attachments 欄位
      Then 應正確轉為 Anthropic content blocks 格式
      And 回傳 SSE 串流回應

    Scenario: 未附帶附件的請求（向後相容）
      Given API 伺服器已啟動
      When 發送 POST /api/chat/stream 僅包含 message 欄位
      Then 應正常處理純文字請求

  Rule: 對話歷史應正確保存多模態訊息

    Scenario: 多模態訊息持久化後可恢復
      Given 使用者已傳送包含圖片的訊息
      And Agent 已回應
      When 載入該 session 的對話歷史
      Then 歷史中應包含完整的圖片內容區塊
