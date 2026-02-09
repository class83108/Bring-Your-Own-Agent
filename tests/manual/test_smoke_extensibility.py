"""Agent 可擴充性 Smoke Test。

驗證使用者可以將 agent-core 當作 library 載入，
並自行新增 tools、skills、MCP 工具。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_extensibility.py -v --run-smoke
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.mcp.adapter import MCPToolAdapter
from agent_core.mcp.client import MCPToolDefinition
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.skills.base import Skill
from agent_core.skills.registry import SkillRegistry
from agent_core.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke


# =============================================================================
# 輔助函數
# =============================================================================


async def _collect_response(agent: Agent, message: str) -> str:
    """收集串流回應的完整文字。"""
    chunks: list[str] = []
    async for chunk in agent.stream_message(message):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return ''.join(chunks)


# =============================================================================
# Rule: 使用者可以當作 library 載入 Agent 並正常對話
# =============================================================================


@allure.feature('Agent 核心架構')
@allure.story('驗證 agent-core 可作為獨立 library 使用 (Smoke)')
class TestAgentAsLibrary:
    """驗證 agent-core 可作為獨立 library 使用。"""

    @allure.title('最小化使用：只需 AgentCoreConfig + AnthropicProvider 即可對話')
    async def test_minimal_agent_usage(self) -> None:
        """最小化使用：只需 AgentCoreConfig + AnthropicProvider 即可對話。"""
        config = AgentCoreConfig()
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider)

        response = await _collect_response(agent, '請只回答 "hello"')

        assert len(response) > 0
        assert len(agent.conversation) == 2

    @allure.title('使用者可以自訂 system prompt')
    async def test_custom_system_prompt(self) -> None:
        """使用者可以自訂 system prompt。"""
        config = AgentCoreConfig(
            system_prompt='你是一位數學老師。所有回答都必須包含數字。請用繁體中文回答。',
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider)

        response = await _collect_response(agent, '1 + 1 等於多少？')

        assert '2' in response

    @allure.title('使用者可以自訂 model 與 max_tokens')
    async def test_custom_model_config(self) -> None:
        """使用者可以自訂 model 與 max_tokens。"""
        config = AgentCoreConfig(
            provider=ProviderConfig(
                model='claude-haiku-4-5-20251001',
                max_tokens=256,
            ),
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider)

        response = await _collect_response(agent, '請回答 OK')

        assert len(response) > 0


# =============================================================================
# Rule: 使用者可以自行新增 Tool
# =============================================================================


@allure.feature('Agent 核心架構')
@allure.story('Tool Registry 應能動態管理工具 (Smoke)')
class TestCustomTools:
    """驗證使用者可以註冊自訂工具並讓 Agent 使用。"""

    @allure.title('Agent 可以使用使用者註冊的同步工具')
    async def test_sync_custom_tool(self) -> None:
        """Agent 可以使用使用者註冊的同步工具。"""
        registry = ToolRegistry()

        # 使用者自訂的計算機工具
        def calculator(expression: str) -> dict[str, Any]:
            """簡易計算機。"""
            # 安全限制：只允許數字和基本運算符
            allowed = set('0123456789+-*/(). ')
            if not all(c in allowed for c in expression):
                return {'error': '不允許的字元'}
            result = eval(expression)  # noqa: S307
            return {'expression': expression, 'result': result}

        registry.register(
            name='calculator',
            description='計算數學表達式。當使用者詢問數學計算時使用此工具。',
            parameters={
                'type': 'object',
                'properties': {
                    'expression': {
                        'type': 'string',
                        'description': '數學表達式，如 "2 + 3 * 4"',
                    },
                },
                'required': ['expression'],
            },
            handler=calculator,
        )

        config = AgentCoreConfig(
            system_prompt='你是計算助手。當使用者要求計算時，必須使用 calculator 工具。',
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider, tool_registry=registry)

        response = await _collect_response(agent, '請計算 15 * 7')

        # 應該有工具調用記錄
        assert len(agent.conversation) >= 4
        assert '105' in response

    @allure.title('Agent 可以使用使用者註冊的非同步工具')
    async def test_async_custom_tool(self) -> None:
        """Agent 可以使用使用者註冊的非同步工具。"""
        registry = ToolRegistry()

        # 使用者自訂的非同步查詢工具
        async def lookup_user(user_id: str) -> dict[str, Any]:
            """模擬使用者查詢（非同步）。"""
            # 模擬資料庫查詢
            users = {
                'u001': {'name': '小明', 'role': 'engineer'},
                'u002': {'name': '小華', 'role': 'designer'},
            }
            if user_id in users:
                return {'found': True, **users[user_id]}
            return {'found': False, 'error': f'找不到使用者 {user_id}'}

        registry.register(
            name='lookup_user',
            description='根據使用者 ID 查詢使用者資訊。',
            parameters={
                'type': 'object',
                'properties': {
                    'user_id': {
                        'type': 'string',
                        'description': '使用者 ID，如 "u001"',
                    },
                },
                'required': ['user_id'],
            },
            handler=lookup_user,
        )

        config = AgentCoreConfig(
            system_prompt='你是使用者管理助手。當需要查詢使用者時，使用 lookup_user 工具。',
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider, tool_registry=registry)

        response = await _collect_response(agent, '請查詢使用者 u001 的資訊')

        assert len(agent.conversation) >= 4
        assert '小明' in response or 'engineer' in response

    @allure.title('自訂工具可與內建工具共存')
    async def test_custom_tool_combined_with_builtin_tools(self, tmp_path: Path) -> None:
        """自訂工具可與內建工具共存。"""
        from agent_core.tools.setup import create_default_registry

        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()
        (sandbox / 'data.txt').write_text('價格: 100\n數量: 3\n', encoding='utf-8')

        registry = create_default_registry(sandbox)

        # 在內建工具的基礎上追加自訂工具
        def calculator(expression: str) -> dict[str, Any]:
            allowed = set('0123456789+-*/(). ')
            if not all(c in allowed for c in expression):
                return {'error': '不允許的字元'}
            result = eval(expression)  # noqa: S307
            return {'expression': expression, 'result': result}

        registry.register(
            name='calculator',
            description='計算數學表達式。',
            parameters={
                'type': 'object',
                'properties': {
                    'expression': {
                        'type': 'string',
                        'description': '數學表達式',
                    },
                },
                'required': ['expression'],
            },
            handler=calculator,
        )

        # 確認同時有內建與自訂工具
        tool_names = registry.list_tools()
        assert 'read_file' in tool_names
        assert 'calculator' in tool_names

        config = AgentCoreConfig(
            system_prompt='你是助手。讀取檔案時用 read_file，計算時用 calculator。',
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider, tool_registry=registry)

        response = await _collect_response(agent, '請讀取 data.txt，然後計算 價格 * 數量 的總額')

        # 應該有多次工具調用
        assert len(agent.conversation) >= 4
        assert '300' in response


# =============================================================================
# Rule: 使用者可以自行新增 Skill
# =============================================================================


@allure.feature('Agent 核心架構')
@allure.story('Agent 應整合 Skill 系統 (Smoke)')
class TestCustomSkills:
    """驗證使用者可以註冊 Skill 來擴充 Agent 行為。"""

    @allure.title('啟用的 Skill 應改變 Agent 的回應行為')
    async def test_skill_modifies_agent_behavior(self) -> None:
        """啟用的 Skill 應改變 Agent 的回應行為。"""
        skill_registry = SkillRegistry()
        skill_registry.register(
            Skill(
                name='json_mode',
                description='強制 Agent 以 JSON 格式回應',
                instructions="""你必須以有效的 JSON 格式回應所有問題。
回應格式：{"answer": "你的回答內容"}
不要在 JSON 之外輸出任何文字。""",
            )
        )
        skill_registry.activate('json_mode')

        config = AgentCoreConfig(system_prompt='你是助手。')
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            skill_registry=skill_registry,
        )

        response = await _collect_response(agent, '什麼是 Python？')

        # 回應應該是有效的 JSON
        parsed = json.loads(response)
        assert 'answer' in parsed

    @allure.title('未啟用的 Skill：描述會出現在 system prompt，但完整 instructions 不會')
    async def test_inactive_skill_does_not_inject_instructions(self) -> None:
        """未啟用的 Skill：描述會出現在 system prompt，但完整 instructions 不會。"""
        skill_registry = SkillRegistry()
        json_instructions = '你必須以有效的 JSON 格式回應。格式：{"answer": "..."}'
        skill_registry.register(
            Skill(
                name='json_mode',
                description='強制 Agent 以 JSON 格式回應',
                instructions=json_instructions,
            )
        )
        # 不啟用 — skill_registry.activate('json_mode')

        base_prompt = '你是助手。請用繁體中文回答。'

        # 直接驗證 system prompt 組合結果（確定性，不依賴 LLM）
        combined = skill_registry.get_combined_system_prompt(base_prompt)
        assert base_prompt in combined
        assert 'json_mode' in combined  # Phase 1: 描述出現
        assert json_instructions not in combined  # Phase 2: instructions 不出現

        # 再驗證 Agent 實際能對話（確保不會因 skill_registry 而出錯）
        config = AgentCoreConfig(system_prompt=base_prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            skill_registry=skill_registry,
        )

        response = await _collect_response(agent, '請說 "你好"')
        assert len(response) > 0

    @allure.title('Skill 與 Tool 可同時使用：skill instructions 注入 system prompt，工具正常執行')
    async def test_skill_combined_with_tools(self, tmp_path: Path) -> None:
        """Skill 與 Tool 可同時使用：skill instructions 注入 system prompt，工具正常執行。"""
        from agent_core.tools.setup import create_default_registry

        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()
        (sandbox / 'config.yaml').write_text('name: my-app\nversion: 1.0\n', encoding='utf-8')

        registry = create_default_registry(sandbox)
        skill_registry = SkillRegistry()
        concise_instructions = '回覆時盡量簡短，不超過 30 個字。直接回答重點。'
        skill_registry.register(
            Skill(
                name='concise',
                description='簡潔回覆模式',
                instructions=concise_instructions,
            )
        )
        skill_registry.activate('concise')

        # 驗證 system prompt 包含 skill instructions（確定性）
        base_prompt = '你是助手。'
        combined = skill_registry.get_combined_system_prompt(base_prompt)
        assert concise_instructions in combined

        config = AgentCoreConfig(system_prompt=base_prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            tool_registry=registry,
            skill_registry=skill_registry,
        )

        response = await _collect_response(agent, '請讀取 config.yaml 並告訴我 app 名稱')

        # 驗證工具確實被調用
        # conversation 至少 4 條：user, assistant+tool_use, tool_result, assistant
        assert len(agent.conversation) >= 4
        # 驗證工具結果被使用
        assert 'my-app' in response


# =============================================================================
# Rule: 使用者可以透過 MCP 接入外部工具
# =============================================================================


@allure.feature('Agent 核心架構')
@allure.story('驗證 MCP 工具可以透過 Adapter 接入 Agent 並實際被呼叫 (Smoke)')
class TestMCPIntegration:
    """驗證 MCP 工具可以透過 Adapter 接入 Agent 並實際被呼叫。"""

    @allure.title('Agent 能呼叫 MCP 工具並將結果納入回應')
    async def test_mcp_tool_used_by_agent(self) -> None:
        """Agent 能呼叫 MCP 工具並將結果納入回應。"""
        # 模擬 MCP Server 提供天氣查詢工具
        mock_client = AsyncMock()
        mock_client.server_name = 'weather'
        mock_client.list_tools = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name='get_weather',
                    description='查詢指定城市的即時天氣資訊，回傳溫度與天氣狀況。',
                    input_schema={
                        'type': 'object',
                        'properties': {
                            'city': {
                                'type': 'string',
                                'description': '城市名稱',
                            },
                        },
                        'required': ['city'],
                    },
                ),
            ]
        )
        mock_client.call_tool = AsyncMock(
            return_value={
                'city': '台北',
                'temperature': 25,
                'condition': '多雲',
            }
        )
        mock_client.close = AsyncMock()

        # 建立 registry 並註冊 MCP 工具
        registry = ToolRegistry()
        adapter = MCPToolAdapter(mock_client)
        await adapter.register_tools(registry)

        # 確認 MCP 工具已註冊
        assert 'weather__get_weather' in registry.list_tools()
        summaries = registry.get_tool_summaries()
        assert summaries[0]['source'] == 'mcp'

        config = AgentCoreConfig(
            system_prompt='你是天氣助手。當使用者詢問天氣時，使用 weather__get_weather 工具。',
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider, tool_registry=registry)

        response = await _collect_response(agent, '台北現在天氣如何？')

        # 驗證 MCP 工具被呼叫
        mock_client.call_tool.assert_called()
        call_args = mock_client.call_tool.call_args
        assert call_args[0][0] == 'get_weather'  # 原始工具名稱（不含前綴）

        # 回應應包含天氣資訊
        assert len(agent.conversation) >= 4
        assert '25' in response or '多雲' in response or '台北' in response

        await adapter.close()

    @allure.title('MCP 工具可與內建工具共存')
    async def test_mcp_tools_combined_with_native_tools(self, tmp_path: Path) -> None:
        """MCP 工具可與內建工具共存。"""
        from agent_core.tools.setup import create_default_registry

        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()
        (sandbox / 'cities.txt').write_text('台北\n東京\n紐約\n', encoding='utf-8')

        # 先建立內建工具
        registry = create_default_registry(sandbox)

        # 再追加 MCP 工具
        mock_client = AsyncMock()
        mock_client.server_name = 'translate'
        mock_client.list_tools = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name='translate',
                    description='將文字翻譯為英文。',
                    input_schema={
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': '要翻譯的文字'},
                        },
                        'required': ['text'],
                    },
                ),
            ]
        )
        mock_client.call_tool = AsyncMock(
            return_value={
                'translated': 'Taipei, Tokyo, New York',
            }
        )
        mock_client.close = AsyncMock()

        adapter = MCPToolAdapter(mock_client)
        await adapter.register_tools(registry)

        # 確認同時有內建與 MCP 工具
        tool_names = registry.list_tools()
        assert 'read_file' in tool_names
        assert 'translate__translate' in tool_names

        # 驗證 source 區分
        summaries = registry.get_tool_summaries()
        sources = {s['name']: s['source'] for s in summaries}
        assert sources['read_file'] == 'native'
        assert sources['translate__translate'] == 'mcp'

        await adapter.close()
