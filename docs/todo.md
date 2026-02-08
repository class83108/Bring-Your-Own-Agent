# Agent Core - Feature Roadmap

## Priority 1: Conversation 持久化（必要）

### 1-1. SQLite Backend（內建預設）✅

- [x] 實作 `SQLiteSessionBackend`，符合現有 `SessionBackend` Protocol
- [x] 儲存：session ID、conversation messages、建立時間、更新時間
- [x] Server 重啟後自動恢復對話
- [x] 零依賴（Python 標準庫 `sqlite3`）
- [x] 使用量統計持久化（`load_usage` / `save_usage` / `reset_usage`）
- [x] `main.py` 已切換為 SQLite 後端（移除 Redis 依賴）

### 1-2. Session 管理與隔離 ✅

- [x] `SQLiteSessionBackend` 新增 `list_sessions()` 方法（回傳所有 session 摘要）
- [x] `SQLiteSessionBackend` 新增 `delete_session()` 方法（刪除 session 與其 usage）
- [x] RESTful Session API：
  - `POST /api/sessions` — 建立新 session
  - `GET /api/sessions` — 列出所有 sessions
  - `GET /api/sessions/{id}` — 取得特定 session 歷史
  - `DELETE /api/sessions/{id}` — 刪除特定 session
- [x] 移除舊 `/api/chat/reset`，由 `DELETE /api/sessions/{id}` 取代
- [x] 多個使用者/tab 不再共用同一個 conversation（每個 session 獨立 ID）

---

## Priority 2: Context Window 管理（必要）

沒有這個功能，長對話會直接超過 context window 限制而中斷。

### 2-0. Tool Result 分頁（防護）

- [x] 在 `ToolRegistry` 層統一攔截超大工具結果
- [x] 超過上限時自動分頁，儲存完整結果至暫存區
- [x] 提供 `read_more()` 方法，讓 Agent 可按需取得後續頁面
- [x] 暫存區生命週期管理（in-memory，隨 ToolRegistry 生命週期；過期回傳提示重新取得）

### 2-1. Token 計數 ✅

- [x] 新增 `TokenCounter` 模組，計算 conversation 目前的 token 使用量
- [x] 支援 Anthropic token counting API（或本地估算）
- [x] 在每次 API 回應後更新累計 token 數（利用現有 `UsageInfo`）
- [x] 透過 `/api/agent/status` 回傳目前 token 使用量

### 2-2. Compact（上下文壓縮）✅

- [x] 實作分層壓縮策略：先清除舊 tool_result 內容，再用 LLM 摘要早期對話
- [x] 定義 compact 觸發條件（例如：token 使用量超過 context window 的 80%）
- [x] Agent 在 `_stream_with_tool_loop()` 中自動檢查並觸發 compact
- [x] compact 過程對使用者透明（可選擇是否透過 SSE 通知前端）

---

## Priority 3: 錯誤恢復（重要）✅

API 呼叫失敗（429 rate limit、網路閃斷）很常見，目前一失敗就整輪中斷。

### 3-1. Retry with Backoff ✅

- [x] 在 `AnthropicProvider` 加入 retry 邏輯（指數退避）
- [x] 可設定 max_retries、初始等待時間
- [x] 區分可重試錯誤（429、5xx、網路超時）與不可重試錯誤（401、400）
- [x] 重試過程透過 SSE 通知前端（on_retry 回調）

---

## Priority 4: 多模態輸入（擴展能力）✅

支援 image 與 PDF 輸入，Anthropic API 本身已支援。

### 4-1. 圖片與 PDF 支援 ✅

- [x] `Agent.stream_message()` 支援接收圖片（base64 或 URL）與 PDF（base64）
- [x] API 層支援 base64 JSON（`attachments` 欄位）
- [x] conversation 歷史正確保存多模態訊息
- [x] 附件大小與格式驗證（圖片 20MB、PDF 32MB、不支援類型拒絕）

---

## 可以晚點做

| 功能 | 說明 | 備註 |
|------|------|------|
| Sub-Agent | 子 agent 隔離執行（compact 摘要、重量級工具調用） | 現有 `Agent` 架構可直接建立獨立實例，用 Haiku 降低成本 |
| Redis 中繼層 | 串流 buffer + 斷線復原 | SQLite 負責持久化，Redis 做短期快取，客戶端斷線後可從 Redis 補推未送完的 token |
| 多 Provider | 支援 OpenAI、Gemini 等 | Protocol 已設計好，需要時再加 |
| Cost tracking | 費用追蹤 | `UsageInfo` 已回傳 token 數，加累加器即可 |
| Guardrails | 輸入輸出過濾 | 可先透過 Skill 的 system prompt 做基本防護 |

---

## Monorepo 遷移（未來）

- [ ] 建立 uv workspace 結構
- [ ] `packages/core/` — 框架層（現在的 `src/agent_core`）
- [ ] `packages/app-*/` — 應用層 packages
- [ ] 根目錄共享工具鏈設定（ruff、pyright、pre-commit）
