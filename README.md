# Agent Core

可擴充的 AI Agent 核心框架。透過 API 直接與 Claude 互動，自由組裝 Tools、Skills、MCP 來打造你自己的 Agent。

## 為什麼選擇 Agent Core？

| 特點 | 說明 |
|------|------|
| **API-first** | 直接呼叫 Anthropic API，不依賴 CLI 工具，無被封禁風險 |
| **Pay-per-use** | 按量計費，輕度使用比月費訂閱更划算 |
| **可組裝** | Tools、Skills、MCP 三種擴充機制，像樂高一樣自由拼裝 |
| **可嵌入** | 作為 library 嵌入你的應用，不是獨立的 CLI 工具 |

## 快速開始

### 安裝

```bash
# 前置需求：Python 3.12+、uv
uv sync
```

### 設定 API Key

```bash
export ANTHROPIC_API_KEY=your_api_key
```

或在程式碼中明確指定：

```python
from agent_core import AgentCoreConfig, ProviderConfig

config = AgentCoreConfig(
    provider=ProviderConfig(api_key='sk-ant-...'),
)
```

### 最小範例

```python
import asyncio
from agent_core import Agent, AgentCoreConfig, AnthropicProvider

async def main():
    config = AgentCoreConfig()
    provider = AnthropicProvider(config.provider)
    agent = Agent(config=config, provider=provider)

    async for chunk in agent.stream_message('什麼是 Python？'):
        if isinstance(chunk, str):
            print(chunk, end='', flush=True)

asyncio.run(main())
```

## 使用手冊

### 變更模型與參數

```python
from agent_core import AgentCoreConfig, ProviderConfig

config = AgentCoreConfig(
    provider=ProviderConfig(
        model='claude-sonnet-4-20250514',   # 變更模型
        max_tokens=4096,                     # 最大回應 token 數
        timeout=60.0,                        # API 超時秒數
        enable_prompt_caching=True,          # 啟用 prompt caching
    ),
    system_prompt='你是一位 Python 專家，請用繁體中文回答。',
)
```

### 自訂工具（Tools）

Tools 讓 Agent 能執行實際操作（讀檔、搜尋、API 呼叫等）。

```python
from agent_core import Agent, AgentCoreConfig, AnthropicProvider
from agent_core.tools.registry import ToolRegistry

# 定義自訂工具（支援同步與 async）
def get_weather(city: str) -> str:
    return f'{city} 目前 25°C，多雲'

# 註冊工具
registry = ToolRegistry()
registry.register(
    name='get_weather',
    description='查詢指定城市的天氣資訊',
    parameters={
        'type': 'object',
        'properties': {
            'city': {'type': 'string', 'description': '城市名稱'},
        },
        'required': ['city'],
    },
    handler=get_weather,
)

# 建立 Agent 並注入工具
config = AgentCoreConfig(
    system_prompt='你是助手。需要查天氣時使用 get_weather 工具。',
)
provider = AnthropicProvider(config.provider)
agent = Agent(config=config, provider=provider, tool_registry=registry)
```

**使用內建工具：**

框架提供 5 個內建開發工具，可透過 `create_default_registry()` 一次註冊：

```python
from pathlib import Path
from agent_core.tools.setup import create_default_registry

registry = create_default_registry(Path('./workspace'))
# 已註冊：read_file, edit_file, list_files, grep_search, bash
```

| 內建工具 | 說明 |
|----------|------|
| `read_file` | 讀取檔案內容，支援行數範圍與語言偵測 |
| `edit_file` | 精確搜尋替換編輯，支援新建檔案與備份 |
| `list_files` | 遞迴目錄列表，支援 pattern 過濾 |
| `grep_search` | 正則搜尋程式碼，支援上下文行數 |
| `bash` | 執行 Shell 指令（含安全限制） |

**混合使用內建 + 自訂工具：**

```python
registry = create_default_registry(Path('./workspace'))

# 追加自訂工具到同一個 registry
registry.register(
    name='calculator',
    description='計算數學表達式',
    parameters={
        'type': 'object',
        'properties': {
            'expression': {'type': 'string'},
        },
        'required': ['expression'],
    },
    handler=lambda expression: str(eval(expression)),
)
```

### 自訂技能（Skills）

Skills 透過 system prompt 注入來改變 Agent 的行為模式，採用**兩階段載入**：

- **Phase 1**：所有已註冊 Skill 的 `name` + `description` 注入 system prompt（讓 LLM 知道有哪些能力可用）
- **Phase 2**：只有**啟用**的 Skill 才載入完整 `instructions`

```python
from agent_core import Skill, SkillRegistry

skill_registry = SkillRegistry()

# 註冊 Skill
skill_registry.register(
    Skill(
        name='code_review',
        description='程式碼審查模式',
        instructions="""你現在是程式碼審查專家。審查時請注意：
1. 命名是否清楚
2. 是否有潛在 bug
3. 效能問題
4. 安全漏洞
以 markdown 表格格式輸出審查結果。""",
    )
)

# 啟用 Skill（觸發 Phase 2，完整 instructions 注入）
skill_registry.activate('code_review')

# 注入 Agent
agent = Agent(
    config=config,
    provider=provider,
    skill_registry=skill_registry,
)
```

**Skill 的可見性控制：**

```python
# 只註冊不啟用 → Phase 1（描述出現在 system prompt，instructions 不載入）
skill_registry.register(Skill(name='tdd', description='...', instructions='...'))

# 啟用 → Phase 2（完整 instructions 注入 system prompt）
skill_registry.activate('tdd')

# 停用 → 回到 Phase 1
skill_registry.deactivate('tdd')

# 隱藏模式 → Phase 1 也不載入描述（完全隱形）
Skill(name='hidden', description='...', instructions='...', disable_model_invocation=True)
```

**透過 API 啟用/停用（給終端使用者）：**

除了在程式碼中控制，也可以透過 REST API 讓使用者在對話過程中動態切換 Skill：

```bash
# 查看目前的 Skill 狀態
curl http://localhost:8000/api/agent/status
# → {"skills": {"registered": ["code_review", "tdd"], "active": []}}

# 啟用 Skill
curl -X POST http://localhost:8000/api/skills/code_review/activate
# → {"status": "ok", "skill": "code_review", "active": true}

# 停用 Skill
curl -X POST http://localhost:8000/api/skills/code_review/deactivate
# → {"status": "ok", "skill": "code_review", "active": false}
```

啟用後，下一次對話就會自動注入該 Skill 的 instructions 到 system prompt。

### MCP 整合

透過 MCP（Model Context Protocol）接入外部工具伺服器。框架定義了 `MCPClient` Protocol，只需實作此介面即可接入任何 MCP Server。

```python
from agent_core.mcp import MCPToolAdapter, MCPToolDefinition
from agent_core.tools.registry import ToolRegistry

# 實作 MCPClient Protocol（或使用 mcp SDK）
class MyMCPClient:
    server_name = 'weather'

    async def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name='get_forecast',
                description='取得天氣預報',
                input_schema={
                    'type': 'object',
                    'properties': {
                        'city': {'type': 'string'},
                    },
                    'required': ['city'],
                },
            ),
        ]

    async def call_tool(self, tool_name, arguments):
        return {'forecast': 'sunny', 'temperature': 25}

    async def close(self):
        pass

# 透過 Adapter 註冊到 ToolRegistry
registry = ToolRegistry()
adapter = MCPToolAdapter(MyMCPClient())
await adapter.register_tools(registry)

# 工具名稱自動加上 server 前綴
print(registry.list_tools())  # ['weather__get_forecast']
```

**MCP + 內建工具混合使用：**

```python
# 先建立內建工具
registry = create_default_registry(Path('./workspace'))

# 再追加 MCP 工具
adapter = MCPToolAdapter(my_mcp_client)
await adapter.register_tools(registry)

# Agent 同時擁有 read_file、edit_file... 和 MCP 工具
agent = Agent(config=config, provider=provider, tool_registry=registry)
```

### 完整組合範例

```python
import asyncio
from pathlib import Path
from agent_core import (
    Agent, AgentCoreConfig, AnthropicProvider,
    ProviderConfig, Skill, SkillRegistry,
)
from agent_core.tools.setup import create_default_registry

async def main():
    # 1. 配置
    config = AgentCoreConfig(
        provider=ProviderConfig(model='claude-sonnet-4-20250514'),
        system_prompt='你是專業的程式開發助手。',
    )

    # 2. 工具
    registry = create_default_registry(Path('./workspace'))

    # 3. 技能
    skill_registry = SkillRegistry()
    skill_registry.register(
        Skill(
            name='code_review',
            description='程式碼審查',
            instructions='審查程式碼並以表格輸出結果。',
        )
    )
    skill_registry.activate('code_review')

    # 4. 組裝 Agent
    provider = AnthropicProvider(config.provider)
    agent = Agent(
        config=config,
        provider=provider,
        tool_registry=registry,
        skill_registry=skill_registry,
    )

    # 5. 對話
    async for chunk in agent.stream_message('請讀取 main.py 並審查程式碼'):
        if isinstance(chunk, str):
            print(chunk, end='', flush=True)
    print()

asyncio.run(main())
```

## 架構

```
agent_core/
├── agent.py                 # Agent 核心（對話迴圈、工具調用）
├── config.py                # 配置（ProviderConfig、AgentCoreConfig）
├── main.py                  # FastAPI 應用（API 層，可選）
├── providers/
│   ├── base.py              # LLMProvider Protocol
│   ├── anthropic_provider.py # Anthropic 實作
│   └── exceptions.py        # Provider 錯誤型別
├── tools/
│   ├── registry.py          # ToolRegistry（工具管理與執行）
│   ├── setup.py             # 內建工具工廠
│   ├── file_read.py         # 檔案讀取
│   ├── file_edit.py         # 檔案編輯
│   ├── file_list.py         # 目錄瀏覽
│   ├── grep_search.py       # 程式碼搜尋
│   └── bash.py              # Bash 執行
├── skills/
│   ├── base.py              # Skill dataclass
│   └── registry.py          # SkillRegistry（兩階段載入）
├── mcp/
│   ├── client.py            # MCPClient Protocol + MCPServerConfig
│   └── adapter.py           # MCPToolAdapter（MCP → ToolRegistry 橋接）
└── session/
    ├── base.py              # SessionBackend Protocol
    ├── memory_backend.py    # 記憶體 Session（預設）
    └── redis_backend.py     # Redis Session
```

### 設計原則

**Protocol-based 依賴注入**：所有外部依賴都透過 Protocol 定義介面，使用者可自行替換實作。

| Protocol | 說明 | 內建實作 |
|----------|------|----------|
| `LLMProvider` | LLM API 介面 | `AnthropicProvider` |
| `MCPClient` | MCP Server 通訊 | 使用者自行實作 |
| `LockProvider` | 檔案操作鎖定 | 使用者自行實作 |
| `SessionBackend` | 對話持久化 | `MemoryBackend`、`RedisBackend` |

## API 端點

內建 FastAPI 應用提供 REST API，適合搭配前端使用：

```bash
uv run uvicorn agent_app.main:app --reload --port 8000
```

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/api/chat/stream` | SSE 串流對話 |
| GET | `/api/chat/history` | 取得對話歷史 |
| POST | `/api/chat/reset` | 清除對話 |
| GET | `/api/chat/usage` | Token 使用量統計 |
| POST | `/api/chat/usage/reset` | 重置使用量統計 |
| GET | `/api/agent/status` | Agent 配置狀態（model、tools、skills） |
| POST | `/api/skills/{name}/activate` | 啟用指定 Skill |
| POST | `/api/skills/{name}/deactivate` | 停用指定 Skill |
| GET | `/api/files/tree` | 沙箱目錄樹 |
| GET | `/api/files/content` | 取得檔案內容 |
| GET | `/health` | 健康檢查 |

## 開發

本專案採用 **Gherkin 驅動的 TDD**：功能規格（`docs/features/*.feature`）→ 紅燈測試 → 綠燈實作 → 重構。

```bash
# 測試
uv run pytest

# Lint + 格式化
uv run ruff check .
uv run ruff format .

# 型別檢查
uv run pyright

# Smoke test（需要 API Key，會產生費用）
uv run pytest tests/manual --run-smoke -v
```

## License

MIT
