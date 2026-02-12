# BYOA Core (Bring Your Own Agent)

可擴充的 AI Agent 核心框架。透過 API 直接與 Claude 互動，自由組裝 Tools、Skills、MCP 來打造**你自己的** Agent。

[![Test Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-393%20passed-brightgreen)]()
[![Eval](https://img.shields.io/badge/eval-13%2F13%20tasks-blue)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 為什麼選擇 BYOA Core？

### 你適合 BYOA Core，如果你⋯

| 使用情境 | 為什麼不是 Claude Code | BYOA Core 怎麼解決 |
|----------|----------------------|-------------------|
| **嵌入自己的產品** | Claude Code 是 CLI 工具，無法作為 library 嵌入 | BYOA Core 是 Python 套件，`import` 即用 |
| **自建 SaaS / 內部平台** | Claude Code 是單機單用戶，無法多租戶部署 | Session 隔離 + Sandbox 沙箱，天然支援多租戶 |
| **自訂工具鏈** | Claude Code 的 Tools 無法替換或深度客製 | Protocol-based 設計，Tools / Skills / MCP 自由拼裝 |
| **控制成本** | Claude Code 月費制，輕度使用不划算 | API 按量計費，附帶 token 追蹤與 prompt caching |
| **需要離線部署** | Claude Code 需要線上帳號 | 只要有 API Key，可部署在任何環境 |
| **Agent 品質調校** | Claude Code 的 prompt / 行為不可控 | System prompt、Skill、memory 完全可配置 |
| **接入私有模型** | Claude Code 僅支援 Anthropic 官方模型 | `LLMProvider` Protocol 可對接任意後端 |
| **自動化 / CI 流水線** | Claude Code 設計為互動式使用 | 純 API 呼叫，適合自動化場景 |

### 架構總覽

> 詳細架構圖見 [docs/architecture.excalidraw](../../docs/architecture.excalidraw)（可用 VS Code Excalidraw 擴充套件或 excalidraw.com 開啟）

**全部可配置**：Provider（模型）、Tools（工具）、Skills（行為）、MCP（外部服務）、Prompt（系統提示詞）、Session Backend（持久化）、Sandbox（沙箱）、EventStore（斷線恢復）。

---

## 安裝

```bash
# 基本安裝（核心功能）
uv add byoa-core

# 可選功能
uv add byoa-core[web]    # web_fetch + web_search 工具
uv add byoa-core[mcp]    # MCP 整合
uv add byoa-core[all]    # 全部安裝
```

或使用 pip：

```bash
pip install byoa-core
pip install byoa-core[all]
```

## 快速開始

### 最小範例（5 行啟動）

```python
import asyncio
from agent_core import Agent, AgentCoreConfig, AnthropicProvider

async def main():
    config = AgentCoreConfig()
    provider = AnthropicProvider(config.provider)
    agent = Agent(config=config, provider=provider)

    async for chunk in agent.stream_message("什麼是 Python？"):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### 完整配置範例

```python
import asyncio
from pathlib import Path
from agent_core import (
    Agent, AgentCoreConfig, AnthropicProvider,
    ProviderConfig, Skill, SkillRegistry, ToolRegistry,
)
from agent_core.tools.setup import create_default_registry
from agent_core.sandbox import LocalSandbox
from agent_core.session import SQLiteSessionBackend
from agent_core.event_store import MemoryEventStore
from agent_core.token_counter import TokenCounter

async def main():
    # ── 1. Provider 配置 ──
    config = AgentCoreConfig(
        provider=ProviderConfig(
            model="claude-sonnet-4-20250514",   # 可換任意 Claude 模型
            max_tokens=8192,
            enable_prompt_caching=True,          # 降低重複 prompt 成本
            max_retries=3,                       # 429/5xx 自動重試
        ),
        system_prompt="你是專業的程式開發助手。",
        max_tool_iterations=25,                  # 防止失控迴圈
    )
    provider = AnthropicProvider(config.provider)

    # ── 2. Sandbox 沙箱 ──
    sandbox = LocalSandbox(root=Path("./workspace"))

    # ── 3. Tools 工具 ──
    registry = create_default_registry(
        sandbox=sandbox,
        memory_dir=Path("./memory"),             # 啟用 memory 工具
        subagent_provider=provider,              # 啟用子代理
        subagent_config=config,
    )
    # 追加自訂工具
    registry.register(
        name="get_weather",
        description="查詢指定城市的天氣",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        handler=lambda city: f"{city} 25°C 多雲",
    )

    # ── 4. Skills 技能 ──
    skill_registry = SkillRegistry()
    skill_registry.register(Skill(
        name="code_review",
        description="程式碼審查模式",
        instructions="審查程式碼，以 markdown 表格輸出結果。",
    ))
    skill_registry.activate("code_review")

    # ── 5. Session 持久化 ──
    session = SQLiteSessionBackend("sessions.db")

    # ── 6. 斷線恢復 ──
    event_store = MemoryEventStore(ttl=300)

    # ── 7. Token 追蹤 ──
    token_counter = TokenCounter(model=config.provider.model)

    # ── 8. 組裝 Agent ──
    agent = Agent(
        config=config,
        provider=provider,
        tool_registry=registry,
        skill_registry=skill_registry,
        event_store=event_store,
        token_counter=token_counter,
    )

    # ── 9. 串流對話 ──
    async for chunk in agent.stream_message("請讀取 main.py 並審查程式碼"):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

---

## 專為 Agent 場景優化

BYOA Core 不只是一個 API wrapper，而是針對**自主 Agent** 場景做了大量優化。

### 1. Prompt Design — 引導式系統提示詞

內建的 system prompt 以**工作流程**結構引導 Agent：

```
1. 探索 → 用 list_files / grep_search 了解專案
2. 閱讀 → 用 read_file 理解上下文
3. 修改 → 基於理解做最小修改
4. 驗證 → 用 bash 執行測試
5. 迭代 → 測試失敗則重複
```

搭配「工具選擇指引」與「修改原則」，讓 Agent 的行為更可預測。當然，system prompt 完全可替換。

### 2. Resume 系統 — 斷線不中斷

`EventStore` 機制讓串流中斷後可以恢復：

```python
# 初次串流：帶 stream_id
async for chunk in agent.stream_message("任務...", stream_id="stream-001"):
    send_to_client(chunk)  # 如果斷線⋯

# 恢復串流：從最後收到的 event 繼續
events = await event_store.read("stream-001", after="last-event-id")
for event in events:
    send_to_client(event)
```

支援 `MemoryEventStore`（單機）或自行實作 `EventStore` Protocol（如 Redis）。

### 3. Compact — 智慧上下文壓縮

長對話不會因超出 context window 而中斷，框架自動管理：

```
Token 使用率 < 80%  →  正常運行
Token 使用率 ≥ 80%  →  Phase 1: 截斷舊 tool_result（無 API 呼叫）
仍然超標            →  Phase 2: LLM 摘要早期對話（一次 API 呼叫）
```

- 不會截斷最近的對話輪次（保護工作中的上下文）
- 安全分割點偵測，不會打斷 tool_use → tool_result 配對
- 壓縮過程透過 SSE 事件通知前端

### 4. Subagent — 不污染主 Context

子代理擁有**獨立的對話歷史**，完成任務後只回傳摘要：

```python
# Agent 自主決定何時使用 create_subagent
# 子代理：獨立 context → 執行任務 → 回傳摘要文字
# 父代理：主 context 不被子任務細節污染
```

- 共享父 Agent 的 Tools ��� Sandbox（同一安全邊界）
- 自動排除 `create_subagent` 工具（防止遞迴建立）
- 適合平行處理多個獨立子任務

### 5. Memory 系統 — 跨壓縮記憶

`memory` 工具讓 Agent 能在對話被壓縮後仍保留關鍵資訊：

```
memory view              → 查看所有記憶
memory write key content → 寫入記憶
memory delete key        → 刪除記憶
```

System prompt 引導 Agent：「假設你的對話隨時可能被壓縮，未記錄的資訊會丟失。」

### 6. Tool Description — 情境式工具描述

工具描述不只說「這個工具做什麼」，更指出「什麼時候該用」：

```
✗ 一般描述："搜尋程式碼"
✓ 情境描述："搜索特定 pattern → 用 grep_search，比逐一 read_file 更高效"
```

搭配工具錯誤訊息增強，讓 Agent 在遇到錯誤時知道如何調整：
- 路徑不存在 → 建議使用 `list_files` 確認
- 搜尋無結果 → 建議調整 pattern 或擴大搜尋範圍
- 指令超時 → 建議拆分指令或增加超時時間

### 7. 安全防護

- **危險指令偵測**：`rm -rf /`、`dd`、`mkfs`、fork bomb 等阻擋
- **系統修改警告**：`sudo`、`chmod 777` 等提示
- **敏感檔案保護**：`.env`、`id_rsa`、`credentials.json` 等過濾
- **輸出遮罩**：API key、token、password 自動遮蔽
- **路徑穿越防護**：Sandbox 驗證所有路徑在根目錄內

---

## 可配置元件一覽

| 元件 | Protocol | 內建實作 | 你可以⋯ |
|------|----------|---------|---------|
| **LLM Provider** | `LLMProvider` | `AnthropicProvider` | 對接 OpenAI、Gemini、本地模型 |
| **Tools** | `ToolRegistry` | 10+ 內建工具 | 註冊任意 sync/async handler |
| **Skills** | `SkillRegistry` | — | 定義行為模式，動態啟停 |
| **MCP** | `MCPClient` | — | 接入任意 MCP Server |
| **Session** | `SessionBackend` | `MemoryBackend`、`SQLiteBackend` | 換成 Redis、DynamoDB 等 |
| **EventStore** | `EventStore` | `MemoryEventStore` | 換成 Redis Streams 等 |
| **Sandbox** | `Sandbox` | `LocalSandbox` | 換成 Docker Container |
| **Config** | `AgentCoreConfig` | 預設值 | 調整 model、prompt、max iterations 等 |

### Protocol 設計

所有外部依賴都透過 Protocol（結構子型別）定義，不強制繼承：

```python
# 實作自己的 Session Backend — 只需符合 Protocol 簽名
class RedisSessionBackend:
    async def load(self, session_id: str) -> list[MessageParam]:
        return await self.redis.get(session_id)

    async def save(self, session_id: str, conversation: list[MessageParam]) -> None:
        await self.redis.set(session_id, conversation)

    async def reset(self, session_id: str) -> None:
        await self.redis.delete(session_id)

# 直接注入，無需繼承任何 base class
agent = Agent(config=config, provider=provider, session=RedisSessionBackend())
```

---

## 內建工具

| 工具 | 說明 | 類別 |
|------|------|------|
| `read_file` | 讀取檔案（支援行數範圍、語言偵測） | 檔案操作 |
| `edit_file` | 精確搜尋替換編輯（支援新建、備份） | 檔案操作 |
| `list_files` | 遞迴目錄列表（支援 pattern 過濾） | 檔案操作 |
| `grep_search` | 正則搜尋程式碼（支援上下文行數） | 搜尋 |
| `bash` | 執行 Shell 指令（含安全限制與輸出遮罩） | 執行 |
| `think` | 無副作用推理記錄（幫助 Agent 組織思路） | 推理 |
| `memory` | 工作記憶（view / write / delete） | 記憶 |
| `web_fetch` | 抓取網頁內容（需 `[web]` extra） | 網路 |
| `web_search` | 網路搜尋（Tavily API，需 `[web]` extra） | 網路 |
| `create_subagent` | 建立子代理執行獨立任務 | 代理 |

### 大型結果自動分頁

工具回傳超過 30KB 時，自動分頁處理：

```
Agent 調用 grep_search → 結果 50KB
→ 框架自動截斷，回傳第 1 頁 + 提示：
  "結果已分頁，使用 read_more(result_id, page=2) 取得下一頁"
→ Agent 可按需取得後續頁面
```

---

## Eval 系統

自製的 Eval 框架驗證 Agent 在真實程式開發場景的表現：

### 13 項評估任務

| 難度 | 任務 | 說明 |
|------|------|------|
| Easy | T1 Fix Syntax Error | 定位並修復 import 錯字 |
| Easy | T2 Fix Failing Test | 根據測試錯誤修復函數 |
| Medium | T3 Add Function | 從測試推導需求，實作 `slugify()` |
| Medium | T4 Add Error Handling | 為既有函數加入驗證與錯誤處理 |
| Hard | T5 Bug from Symptoms | 在 4 層呼叫鏈中找出隱藏 bug |
| Hard | T6 Implement from Tests | 無規格說明，純粹從測試檔反推實作 |
| Special | T7 Large Codebase | 在 15+ 檔案專案中導航並修復 bug |
| Special | T8 Ambiguous Requirements | 處理不完整、衝突的需求規格 |
| Special | T9 Self-Repair Loop | Agent 偵測測試失敗並自我修復 |
| Special | T10 Full Cycle TDD | 完整 TDD 流程：寫測試 → 實作 → 驗證 |
| Special | T11 Web Crawler | 建立含分頁、錯誤處理的網頁爬蟲 |
| Special | T12 Maze Exploration | 迷宮探索（Memory + Compact 壓力測試） |
| Special | T13 Subagent Parallel | 使用子代理平行修復多個 bug |

### 評估指標

每個任務收集：
- **通過率** — 所有驗證測試是否通過
- **分數** (0.0–1.0) — 多維度加權評分
- **工具使用** — 調用次數與順序
- **Token 消耗** — 衡量效率
- **耗時** — 執行速度
- **自主驗證** — Agent 是否主動跑測試

結果持久化至 SQLite，支援跨版本對比與 A/B testing（不同 system prompt、不同 model）。

```bash
# 執行 Eval
uv run pytest tests/eval --run-eval --eval-agent-version="v2" -v

# 指定模型與超時
uv run pytest tests/eval --run-eval --eval-model="claude-opus-4-20250805" --eval-timeout=600 -v

# 單一任務
uv run pytest tests/eval --run-eval --eval-task=t05_bug_from_symptoms -v
```

---

## 測試

```
393 tests passed | 89% coverage | 28 unit test files | 11 smoke test files | 30 Gherkin specs
```

### 三層測試策略

| 層級 | 位置 | 數量 | 說明 |
|------|------|------|------|
| **Unit Test** | `tests/core/` `tests/app/` | 28 files / 393 tests | 模擬 API，毫秒級執行 |
| **Smoke Test** | `tests/manual/` | 11 files | 真實 API 端對端驗證 |
| **Eval** | `tests/eval/` | 13 tasks | Agent 能力評估 |

```bash
# Unit tests（含覆蓋率報告）
uv run pytest

# Smoke tests（需要 API Key，會產生費用）
uv run pytest tests/manual --run-smoke -v

# 全部（含 Allure 報告）
uv run pytest --alluredir=allure-results
allure serve allure-results
```

### 覆蓋率摘要

| 模組 | 覆蓋率 |
|------|--------|
| `agent.py` (核心迴圈) | 90% |
| `tools/registry.py` (工具管理) | 97% |
| `compact.py` (上下文壓縮) | 85% |
| `providers/anthropic_provider.py` | 89% |
| `token_counter.py` | 100% |
| `skills/registry.py` | 94% |
| `event_store/memory.py` | 94% |
| `sandbox/local.py` | 96% |
| `multimodal.py` | 99% |
| `types.py` | 100% |
| **整體** | **89%** |

---

## API 概覽

### 核心類別

```python
from agent_core import Agent, AgentCoreConfig, ProviderConfig, AnthropicProvider
from agent_core import Skill, SkillRegistry, ToolRegistry

# 工具工廠
from agent_core.tools.setup import create_default_registry

# Sandbox
from agent_core.sandbox import LocalSandbox

# Session 持久化
from agent_core.session import MemorySessionBackend, SQLiteSessionBackend

# 斷線恢復
from agent_core.event_store import MemoryEventStore

# Token 追蹤
from agent_core.token_counter import TokenCounter

# MCP 整合
from agent_core.mcp import MCPToolAdapter, MCPToolDefinition

# 多模態
from agent_core.multimodal import Attachment, build_content_blocks

# 型別
from agent_core.types import MessageParam, ContentBlock, AgentEvent
```

### Agent 主要方法

```python
agent = Agent(config, provider, tool_registry, skill_registry, event_store, token_counter)

# 串流對話（主要入口）
async for chunk in agent.stream_message(content, attachments=None, stream_id=None):
    # chunk: str（文字片段）或 AgentEvent（工具調用等事件）
    ...

# 取得對話歷史
agent.conversation  # list[MessageParam]
```

### ToolRegistry

```python
registry = ToolRegistry()

# 註冊工具
registry.register(name, description, parameters, handler, file_param=None)

# 工具定義（傳給 LLM API）
registry.get_tool_definitions()  # list[dict]

# 複製（用於子代理）
child_registry = registry.clone(exclude=["create_subagent"])
```

### SkillRegistry

```python
skill_registry = SkillRegistry()
skill_registry.register(Skill(name, description, instructions))
skill_registry.activate(name)     # Phase 2：完整 instructions 注入
skill_registry.deactivate(name)   # 回到 Phase 1：僅描述
```

---

## Gherkin 驅動開發

每個功能都先撰寫 Gherkin 規格（`docs/features/`），再寫測試，最後實作：

```gherkin
# language: zh-TW
Feature: 上下文壓縮
  作為 Agent 框架
  我想要在 context window 快滿時自動壓縮
  以便長對話不會中斷

  Rule: Token 使用率超過 80% 時觸發壓縮

    Scenario: 截斷舊的工具結果
      Given 對話歷史中有 20 個 tool_result 區塊
      And token 使用率達到 85%
      When Agent 執行壓縮
      Then 舊的 tool_result 內容被替換為壓縮標記
      And 最近一輪的 tool_result 保持不變
```

共 30 個 Feature 規格檔，覆蓋 agent_core 的所有功能模組。

---

## License

MIT
