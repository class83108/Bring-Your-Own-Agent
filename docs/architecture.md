# BYOA Core 專案架構

> 架構總覽圖見 [architecture.excalidraw](./architecture.excalidraw)

---

## 架構分層

BYOA Core 採用三層架構，每層透過 Protocol 解耦：

```
┌──────────────────────────────────────────────────┐
│          可插拔擴充層 (Extension Layer)            │
│   Tools  ·  Skills  ·  MCP                       │
├──────────────────────────────────────────────────┤
│          核心層 (Core Layer)                      │
│   Agent（串流 → 工具調用 → 迭代）                  │
│   + Compact · Subagent · Memory · TokenCounter   │
├──────────────────────────────────────────────────┤
│          基礎設施層 (Infrastructure Layer)         │
│   LLMProvider · SessionBackend · Sandbox · EventStore │
└──────────────────────────────────────────────────┘
```

### 設計原則

1. **Protocol-based 依賴注入**：所有外部依賴透過 Python Protocol（結構子型別）定義，使用者無需繼承 base class
2. **Async-first**：所有 I/O 操作皆為 async（串流、工具執行、儲存）
3. **Message-centric**：Agent 狀態 = 對話歷史，replay-safe
4. **最小介面**：每個 Protocol 只定義必要方法，實作門檻低

---

## 模組總覽

```
agent_core/
├── agent.py                  # Agent 核心（對話迴圈、工具調用）
├── config.py                 # 配置（ProviderConfig、AgentCoreConfig）
├── types.py                  # 型別定義（TypedDict discriminated union）
├── compact.py                # 上下文壓縮（兩階段策略）
├── token_counter.py          # Token 計數與 context window 追蹤
├── usage_monitor.py          # 使用量統計與費用估算
├── multimodal.py             # 多模態輸入（圖片、PDF）
├── tool_summary.py           # 工具呼叫的人類可讀摘要
├── memory.py                 # 工作記憶工具（檔案式）
│
├── providers/
│   ├── base.py               # LLMProvider Protocol
│   ├── anthropic_provider.py # Anthropic 實作（含 prompt caching、retry）
│   └── exceptions.py         # Provider 錯誤型別
│
├── tools/
│   ├── registry.py           # ToolRegistry（工具管理、分頁、鎖定）
│   ├── setup.py              # 內建工具工廠函數
│   ├── file_read.py          # 檔案讀取
│   ├── file_edit.py          # 檔案編輯
│   ├── file_list.py          # 目錄瀏覽
│   ├── grep_search.py        # 程式碼搜尋
│   ├── bash.py               # Bash 執行（含安全限制）
│   ├── think.py              # 無副作用推理記錄
│   ├── web_fetch.py          # 網頁抓取
│   ├── web_search.py         # 網路搜尋（Tavily）
│   ├── subagent.py           # 子代理建立
│   └── path_utils.py         # 路徑工具函數
│
├── skills/
│   ├── base.py               # Skill dataclass
│   └── registry.py           # SkillRegistry（兩階段載入）
│
├── mcp/
│   ├── client.py             # MCPClient Protocol + MCPServerConfig
│   └── adapter.py            # MCPToolAdapter（MCP → ToolRegistry 橋接）
│
├── sandbox/
│   ├── base.py               # Sandbox ABC
│   └── local.py              # LocalSandbox（本地路徑驗證 + subprocess）
│
├── session/
│   ├── base.py               # SessionBackend Protocol
│   ├── memory_backend.py     # 記憶體 Session（開發用）
│   └── sqlite_backend.py     # SQLite Session（生產用）
│
└── event_store/
    ├── base.py               # EventStore Protocol
    └── memory.py             # MemoryEventStore（含 TTL）
```

---

## 核心流程：Agent 對話迴圈

```
stream_message(content, attachments?, stream_id?)
│
├─ 1. 驗證輸入 + 建構 user message
├─ 2. 追加到 conversation
├─ 3. 進入 _stream_with_tool_loop()
│   │
│   ├─ 3a. _maybe_compact()
│   │   └─ token 使用率 ≥ 80% → Phase 1/2 壓縮
│   │
│   ├─ 3b. provider.stream()
│   │   ├─ yield 文字 token → 前端
│   │   └─ yield AgentEvent → 前端
│   │
│   ├─ 3c. 取得 final_message
│   │   ├─ 記錄 usage（token_counter、usage_monitor）
│   │   └─ 追加 assistant message 到 conversation
│   │
│   ├─ 3d. 檢查 stop_reason
│   │   ├─ stop_reason == 'end_turn' → 結束迴圈
│   │   └─ stop_reason == 'tool_use' → 繼續
│   │
│   └─ 3e. _execute_tool_calls()
│       ├─ 提取 tool_use blocks
│       ├─ asyncio.gather() 平行執行工具
│       ├─ 建構 tool_result blocks
│       ├─ 追加 user message（含 tool_result）
│       └─ 回到 3a（迭代至 max_tool_iterations）
│
├─ 4. EventStore 記錄事件（若有 stream_id）
└─ 5. 標記串流完成 / 失敗
```

### 工具迴圈限制

- `max_tool_iterations`（預設 25）：防止 Agent 失控迴圈
- 達到上限時自動停止，回傳已累積的回應

---

## Protocol 擴展點

### LLMProvider

```python
class LLMProvider(Protocol):
    async def stream(
        messages, system, tools, max_tokens, on_retry
    ) -> AsyncContextManager[StreamResult]

    async def count_tokens(messages, system, tools) -> int

    async def create(messages, system, tools, max_tokens) -> FinalMessage
```

**內建實作**：`AnthropicProvider`（含 prompt caching、指數退避重試、錯誤轉換）

**替換場景**：對接 OpenAI、Gemini、本地 Ollama 等

### SessionBackend

```python
class SessionBackend(Protocol):
    async def load(session_id: str) -> list[MessageParam]
    async def save(session_id: str, conversation: list[MessageParam]) -> None
    async def reset(session_id: str) -> None
```

**內建實作**：`MemorySessionBackend`（開發）、`SQLiteSessionBackend`（生產）

**替換場景**：Redis、DynamoDB、PostgreSQL 等

### EventStore

```python
class EventStore(Protocol):
    async def append(stream_id: str, event: StreamEvent) -> None
    async def read(stream_id: str, after: str | None, count: int) -> list[StreamEvent]
    async def get_status(stream_id: str) -> StreamStatus | None
    async def mark_complete(stream_id: str) -> None
    async def mark_failed(stream_id: str) -> None
```

**內建實作**：`MemoryEventStore`（含 TTL 自動清理）

**替換場景**：Redis Streams、Kafka 等

### Sandbox

```python
class Sandbox(ABC):
    @abstractmethod
    def validate_path(path: str) -> str

    @abstractmethod
    async def exec(command: str, timeout: int, working_dir: str | None) -> ExecResult
```

**內建實作**：`LocalSandbox`（本地路徑驗證 + subprocess）

**替換場景**：Docker Container、Firecracker microVM 等

### MCPClient

```python
class MCPClient(Protocol):
    server_name: str
    async def list_tools() -> list[MCPToolDefinition]
    async def call_tool(tool_name: str, arguments: dict) -> Any
    async def close() -> None
```

**橋接**：`MCPToolAdapter` 將 MCP 工具自動註冊到 `ToolRegistry`，名稱加 `{server}__{tool}` 前綴

---

## Agent 優化機制

### Compact（上下文壓縮）

```
compact_conversation(conversation, provider, usage_percent)
│
├─ usage < 80% → 不壓縮
│
├─ Phase 1: 截斷舊 tool_result
│   ├─ 掃描 user messages 中的 tool_result blocks
│   ├─ 將舊的 tool_result.content 替換為 '[已壓縮的工具結果]'
│   ├─ 保留最近 N 輪不截斷
│   └─ 若有截斷 → 返回（不進入 Phase 2）
│
└─ Phase 2: LLM 摘要
    ├─ 找安全分割點（不打斷 tool_use → tool_result 配對）
    ├─ 將早期對話送 LLM 摘要
    └─ 替換為 user/assistant 摘要訊息對
```

**關鍵設計**：
- Phase 1 無 API 呼叫，低成本
- 安全分割點偵測防止打斷工具鏈
- 最近輪次永遠保留

### Subagent（子代理）

```
create_subagent(task)
│
├─ 複製 ToolRegistry（排除 create_subagent）
├─ 建立新 Agent（獨立 conversation、無 usage/token 追蹤）
├─ 共享父 Agent 的 Sandbox（同一安全邊界）
├─ 執行任務，累積文字輸出
└─ 回傳 {'result': accumulated_text}
```

**關鍵設計**：
- 獨立 context 不污染父 Agent
- 排除 `create_subagent` 防止遞迴
- 可平行建立多個子代理

### Memory（工作記憶）

- 基於檔案系統的 key-value 儲存
- 支援 `view`（列表/讀取）、`write`（寫入）、`delete`（刪除）
- 路徑穿越防護（`resolve().is_relative_to(root)`）
- System prompt 引導 Agent 主動使用：「假設對話隨時可能被壓縮」

### TokenCounter

- 追蹤 context window 使用率（`input_tokens + output_tokens`）
- 每次 API 回應後更新
- 觸發 compact 的閾值：80%
- 支援多模型（200k context window）

---

## 工具系統

### ToolRegistry

```python
registry = ToolRegistry(lock_provider=None)

# 註冊
registry.register(name, description, parameters, handler, file_param=None)

# 查詢
registry.list_tools()           # 工具名稱列表
registry.get_tool_definitions() # LLM API 格式
registry.get_tool_summaries()   # 含 source 標籤

# 執行
await registry.execute(name, arguments)

# 進階
registry.clone(exclude=[...])        # 複製（用於子代理）
registry.read_more(result_id, page)  # 分頁讀取
```

### 大型結果分頁

- 工具回傳 > 30KB → 自動截斷
- 儲存完整結果到 UUID-keyed 快取
- 回傳第 1 頁 + `read_more(result_id, page=2)` 指引
- Agent 可按需取得後續頁面

### 分散式鎖定

- 工具有 `file_param` + registry 有 `lock_provider` → 自動加鎖
- 執行前 acquire、執行後 release（finally block）
- 防止平行工具呼叫的檔案競爭

### 內建工具安全

| 工具 | 安全機制 |
|------|---------|
| `bash` | 危險指令偵測（rm -rf /、dd、fork bomb）、輸出遮罩（API key、password） |
| `file_read` | 敏感檔案過濾（.env、id_rsa）、Sandbox 路徑驗證 |
| `file_edit` | Sandbox 路徑驗證、原始行保留 |
| `file_list` | Sandbox 路徑驗證、結果上限（100 項） |
| `grep_search` | Sandbox 路徑驗證 |

---

## Skill 系統

### 兩階段載入

| Phase | 時機 | 載入內容 | 影響 |
|-------|------|---------|------|
| Phase 1 | 每次 API 呼叫 | 所有 Skill 的 `name` + `description` | 輕量 system prompt |
| Phase 2 | 啟用後 | 僅啟用 Skill 的完整 `instructions` | 完整行為注入 |

### 可見性控制

```python
# 註冊（Phase 1 可見）
skill_registry.register(Skill(name='tdd', description='...', instructions='...'))

# 啟用（Phase 2 載入 instructions）
skill_registry.activate('tdd')

# 停用（回到 Phase 1）
skill_registry.deactivate('tdd')

# 隱藏模式（Phase 1 也不載入）
Skill(name='hidden', ..., disable_model_invocation=True)
```

---

## 錯誤處理

### Provider 錯誤階層

```
ProviderError (base)
├── ProviderAuthError         → 401, invalid API key
├── ProviderConnectionError   → 網路問題
├── ProviderRateLimitError    → 429
└── ProviderTimeoutError      → 請求超時
```

### Agent 層級處理

| 錯誤類型 | Agent 行為 |
|----------|-----------|
| `ProviderAuthError` | 彈出最後一則訊息，立即失敗 |
| `ProviderConnectionError` | 保留部分回應，重新拋出 |
| `ProviderTimeoutError` | 保留部分回應，重新拋出 |
| 工具執行錯誤 | 記錄日誌，回傳 `tool_result` 含 `is_error=True` |

### 重試策略（AnthropicProvider）

- **指數退避**：`delay = initial_delay × 2^attempt`
- **可重試**：429、5xx、timeout、connection error
- **不可重試**：401（auth）、400（bad request）
- **配置**：`max_retries`（預設 3）、`retry_initial_delay`（預設 1.0s）

---

## 型別系統

### Discriminated Union TypedDict

```python
# types.py — Pyright 自動窄化
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock | DocumentBlock

class TextBlock(TypedDict):
    type: Literal['text']
    text: str

class ToolUseBlock(TypedDict):
    type: Literal['tool_use']
    id: str
    name: str
    input: dict[str, object]

# 使用 block['type'] 觸發 narrowing
if block['type'] == 'tool_use':
    tool_name = block['name']  # Pyright 知道是 str
```

**設計決策**：
- 使用 `block['type']`（非 `.get('type')`）啟用 Pyright discriminated union narrowing
- `cast()` 僅用於反序列化邊界（`json.loads`、`model_dump`）
- 列表推導式窄化改用明確 for-loop

---

## 測試架構

### 三層策略

```
tests/
├── core/                 # agent_core 單元測試（27 files）
│   ├── test_agent.py     # Agent 核心邏輯
│   ├── test_compact.py   # 壓縮策略
│   ├── test_tool_*.py    # 各工具
│   └── ...
├── app/                  # agent_app 單元測試（1 file）
│   └── test_api.py       # FastAPI 端點
├── manual/               # Smoke tests（11 files，需真實 API）
│   ├── test_smoke.py
│   └── test_smoke_*.py
└── eval/                 # Agent 能力評估（13 tasks）
    ├── framework.py      # EvalRunner + EvalResult
    ├── store.py          # SQLite 結果持久化
    └── tasks/
        ├── t01_fix_syntax_error.py
        └── ...t13_subagent_parallel_fix.py
```

### Eval 系統

- 13 個任務覆蓋 easy → hard → special 難度
- 每個任務收集：通過率、分數、工具使用、token 消耗、耗時、自主驗證
- SQLite 持久化，支援跨版本對比與 A/B testing
- 可指定 model、system prompt、timeout

### 覆蓋率

```
整體覆蓋率：89%（393 tests passed）

核心模組：
  agent.py            90%
  tools/registry.py   97%
  compact.py          85%
  anthropic_provider   89%
  token_counter       100%
  skills/registry     94%
  event_store/memory  94%
  sandbox/local       96%
  multimodal          99%
  types               100%
```

---

## 設計模式

| Pattern | 位置 | 用途 |
|---------|------|------|
| Protocol | `base.py` files | Provider-agnostic 介面 |
| Discriminated Union | `types.py` | Pyright narrowing 優化 |
| Closure/Handler | `memory.py`, `subagent.py` | 捕獲上下文的 async handler |
| Context Manager | `provider.stream()` | 串流資源管理 |
| Clone | `registry.clone()` | 子代理工具隔離 |
| Two-Phase Loading | Skill system | 效能 vs 彈性 |
| Event Sourcing | EventStore | 可恢復串流 |
| Exponential Backoff | retry 邏輯 | API rate limit 處理 |
| Pagination | ToolRegistry | 大型結果處理 |
