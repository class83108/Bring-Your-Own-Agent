# Agent Core - Feature Roadmap

## Priority 1: Conversation 持久化（必要）

`SessionBackend` Protocol 已定義但無實作。Server 重啟對話全部消失。

### 1-1. SQLite Backend（內建預設）

- [ ] 實作 `SQLiteSessionBackend`，符合現有 `SessionBackend` Protocol
- [ ] 儲存：session ID、conversation messages、建立時間、更新時間
- [ ] Server 重啟後自動恢復對話
- [ ] 零依賴（Python 標準庫 `sqlite3`）

### 1-2. Session 隔離

- [ ] 每個對話有獨立 session ID
- [ ] API 層支援 session 管理（建立、切換、列出歷史對話）
- [ ] 多個使用者/tab 不再共用同一個 conversation

---

## Priority 2: Context Window 管理（必要）

沒有這個功能，長對話會直接超過 context window 限制而中斷。

### 2-0. Tool Result 分頁（防護）

- [x] 在 `ToolRegistry` 層統一攔截超大工具結果
- [x] 超過上限時自動分頁，儲存完整結果至暫存區
- [x] 提供 `read_more()` 方法，讓 Agent 可按需取得後續頁面
- [x] 暫存區生命週期管理（in-memory，隨 ToolRegistry 生命週期；過期回傳提示重新取得）

### 2-1. Token 計數

- [ ] 新增 `TokenCounter` 模組，計算 conversation 目前的 token 使用量
- [ ] 支援 Anthropic token counting API（或本地估算）
- [ ] 在每次 API 回應後更新累計 token 數（利用現有 `UsageInfo`）
- [ ] 透過 `/api/agent/status` 回傳目前 token 使用量

### 2-2. Compact（上下文壓縮）

- [ ] 實作分層壓縮策略：先清除舊 tool_result 內容，再用 LLM 摘要早期對話
- [ ] 定義 compact 觸發條件（例如：token 使用量超過 context window 的 80%）
- [ ] Agent 在 `_stream_with_tool_loop()` 中自動檢查並觸發 compact
- [ ] compact 過程對使用者透明（可選擇是否透過 SSE 通知前端）

---

## Priority 3: 錯誤恢復（重要）

API 呼叫失敗（429 rate limit、網路閃斷）很常見，目前一失敗就整輪中斷。

### 3-1. Retry with Backoff

- [ ] 在 `AnthropicProvider` 加入 retry 邏輯（指數退避）
- [ ] 可設定 max_retries、初始等待時間
- [ ] 區分可重試錯誤（429、5xx、網路超時）與不可重試錯誤（401、400）
- [ ] 重試過程透過 SSE 通知前端（可選）

---

## Priority 4: 多模態輸入（擴展能力）

支援 image 輸入（截圖、圖表），Anthropic API 本身已支援。

### 4-1. Image 支援

- [ ] `Agent.stream_message()` 支援接收圖片（base64 或 URL）
- [ ] API 層支援 multipart 上傳或 base64 JSON
- [ ] conversation 歷史正確保存圖片訊息

---

## 可以晚點做

| 功能 | 說明 | 備註 |
|------|------|------|
| Sub-Agent | 子 agent 隔離執行（compact 摘要、重量級工具調用） | 現有 `Agent` 架構可直接建立獨立實例，用 Haiku 降低成本 |
| 多 Provider | 支援 OpenAI、Gemini 等 | Protocol 已設計好，需要時再加 |
| Cost tracking | 費用追蹤 | `UsageInfo` 已回傳 token 數，加累加器即可 |
| Guardrails | 輸入輸出過濾 | 可先透過 Skill 的 system prompt 做基本防護 |

---

## Monorepo 遷移（未來）

- [ ] 建立 uv workspace 結構
- [ ] `packages/core/` — 框架層（現在的 `src/agent_core`）
- [ ] `packages/app-*/` — 應用層 packages
- [ ] 根目錄共享工具鏈設定（ruff、pyright、pre-commit）
