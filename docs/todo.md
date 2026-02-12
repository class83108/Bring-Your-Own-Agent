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

## Priority 5: Sandbox 沙箱環境 ✅

### 5-1. Sandbox ABC + LocalSandbox ✅

- [x] 定義 Sandbox ABC（`validate_path` + `exec`）
- [x] 實作 LocalSandbox：本地路徑驗證與 subprocess 執行
- [x] 重構 `create_default_registry` 接受 `Sandbox` 參數
- [x] 更新所有 18 個呼叫端
- [x] Feature specs 更新：sandbox、container_runner、sandbox_pool

架構決策：
- Sandbox 只負責路徑驗證與指令執行，檔案 I/O 由 handler 直接操作
- ContainerRunner 歸應用層（NanoClaw 模式：agent 在 container 內用 LocalSandbox）
- Agent Swarm 由應用層編排，不在 core framework 範圍

---

## Priority 6: agent_core 完善與發佈

目標：讓 `agent_core` 成為可獨立 `pip install` 的套件。

### 6-1. Subagent 子代理機制 ✅

- [x] 撰寫 Subagent 測試（紅燈）
- [x] 實作 `create_subagent` 工具
- [x] 子 Agent 使用與父 Agent 相同的 Sandbox
- [x] 子 Agent 預設排除 `create_subagent` 工具（防遞迴）
- [x] 子 Agent 有獨立 context，完成後回傳摘要
- [x] Smoke Test 驗證（真實 API）
- [x] Eval T13 — Subagent Parallel Fix（得分 1.0）

### 6-2. 分離 feature 與 test ✅

- [x] `docs/features/` 拆分為 `docs/features/core/` 和 `docs/features/app/`
- [x] `tests/` 拆分為 `tests/core/` 和 `tests/app/`
- [x] 更新 pytest 設定與 import 路徑

### 6-3. agent_core README 與專案架構文件 ✅

- [x] 撰寫 `src/agent_core/README.md`（安裝、快速上手、API 概覽）
- [x] 撰寫詳細的專案架構文件（`docs/architecture.md` + `docs/architecture.excalidraw`）

### 6-4. 發佈為 pip 套件 ✅

- [x] 檢查 `pyproject.toml`，確認 agent_core 的 package 設定
- [x] 套件改名為 `byoa-core`（import 路徑保持 `agent_core`）
- [x] 分離 core / app 的 dependencies（web、mcp 為 optional）
- [x] 刪除 `session/redis_backend.py`（不再維護）
- [x] 確認 `pip install` 可正常運作
- [ ] （可選）發佈到 PyPI 或私有 registry

---

## Priority 7: agent_app 應用層

待 agent_core 發佈後再處理。

### 7-1. ContainerRunner 容器化 Agent

- [ ] 設計 ContainerRunner（容器生命週期、volume mount、網路配置、IPC）
- [ ] 撰寫測試
- [ ] 實作 ContainerRunner

### 7-2. RunnerPool（可選 utility）

- [ ] 多租戶場景的 ContainerRunner 生命週期管理
- [ ] 上限控制、閒置超時回收、工廠函數

### 7-3. 應用層其他功能

| 功能 | 說明 | 備註 |
|------|------|------|
| Redis 中繼層 | 串流 buffer + 斷線復原 | SQLite 負責持久化，Redis 做短期快取 |
| 多 Provider | 支援 OpenAI、Gemini 等 | Protocol 已設計好，需要時再加 |
| Cost tracking | 費用追蹤 | `UsageInfo` 已回傳 token 數，加累加器即可 |
| Guardrails | 輸入輸出過濾 | 可先透過 Skill 的 system prompt 做基本防護 |
| Memory（agent.md） | 專案知識檔，Agent 啟動時自動載入 | 參考 Claude Code 的 CLAUDE.md pattern |
| Memory（跨 Session） | SQLite 持久化的結構化記憶 | 需先定義「什麼值得記住」 |

---

## Agent 優化（Eval 驅動）

v1-baseline 結果：9/10 通過、avg 0.91。詳細優化方向見 `.claude/plans/elegant-crafting-flame.md`。

| 優先級 | 方向 | 目標任務 | 狀態 |
|--------|------|----------|------|
| P0 | System Prompt 強化（markdown 結構 + 工作流程 + 工具指引 + TDD） | T7/T8/T9/T10 | ✅ |
| P1 | 工具描述優化（任務導向 / 情境式描述） | T7 | ✅ |
| P1 | Max Iterations 上限（防止失控迴圈，預設 25） | 全局 | ✅ |
| P2 | Working Memory 工具（view/write/delete） | T7/T9 | ✅ |
| P2 | Think 工具（無副作用推理記錄） | 全局 | ✅ |
| P2 | Web Fetch 工具（httpx + BeautifulSoup，含連結提取） | 全局 | ✅ |
| P2 | Web Search 工具（Tavily API） | 全局 | ✅ |
| P2 | T11 Web Crawler Eval（本地 HTTP 爬蟲任務） | 全局 | ✅ |
| P4 | 工具錯誤訊息增強（引導性建議） | 全局 | 待做 |
| P4 | 自動驗證提醒（多次 edit 後提示跑 pytest） | 全局 | 待做 |
| P2 | T12 迷宮探索 Eval（Memory + Compact 壓力測試） | 全局 | ✅ |
