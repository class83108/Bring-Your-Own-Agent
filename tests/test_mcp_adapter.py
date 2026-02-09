"""MCP Adapter 測試模組。

根據 docs/features/mcp.feature 規格撰寫測試案例。
涵蓋：
- Rule: 應能探索 MCP Server 的工具
- Rule: MCP 工具應能註冊到 ToolRegistry
- Rule: MCP 連線生命週期

注意：實際 MCP Server 連線測試需要真實 server，
這裡使用 mock MCPClient 測試 Adapter 邏輯。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import allure

from agent_core.mcp import MCPServerConfig, MCPToolAdapter, MCPToolDefinition
from agent_core.tools.registry import ToolRegistry

# --- 輔助工具 ---


def _make_mock_client(
    server_name: str,
    tools: list[MCPToolDefinition],
) -> AsyncMock:
    """建立模擬的 MCPClient。"""
    client = AsyncMock()
    client.server_name = server_name
    client.list_tools = AsyncMock(return_value=tools)
    client.call_tool = AsyncMock(return_value={'result': 'ok'})
    client.close = AsyncMock()
    return client


# =============================================================================
# Rule: 應能探索 MCP Server 的工具
# =============================================================================


@allure.feature('MCP Server 整合')
@allure.story('應能探索 MCP Server 的工具')
class TestMCPToolDiscovery:
    """MCP 工具探索測試。"""

    @allure.title('列出 MCP Server 提供的工具')
    async def test_list_tools_from_server(self) -> None:
        """Scenario: 列出 MCP Server 提供的工具。"""
        tools = [
            MCPToolDefinition(
                name='get_forecast',
                description='取得天氣預報',
                input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            ),
            MCPToolDefinition(
                name='get_temperature',
                description='取得目前溫度',
                input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            ),
        ]
        client = _make_mock_client('weather', tools)

        result = await client.list_tools()

        assert len(result) == 2
        assert result[0].name == 'get_forecast'
        assert result[1].name == 'get_temperature'


# =============================================================================
# Rule: MCP 工具應能註冊到 ToolRegistry
# =============================================================================


@allure.feature('MCP Server 整合')
@allure.story('MCP 工具應能註冊到 ToolRegistry')
class TestMCPToolRegistration:
    """MCP 工具註冊到 ToolRegistry 測試。"""

    @allure.title('MCP 工具自動加上前綴')
    async def test_tools_registered_with_prefix(self) -> None:
        """Scenario: MCP 工具自動加上前綴。"""
        tools = [
            MCPToolDefinition(
                name='get_forecast',
                description='取得天氣預報',
                input_schema={'type': 'object', 'properties': {}},
            ),
        ]
        client = _make_mock_client('weather', tools)
        registry = ToolRegistry()

        adapter = MCPToolAdapter(client)
        await adapter.register_tools(registry)

        assert 'weather__get_forecast' in registry.list_tools()

    async def test_registered_tool_source_is_mcp(self) -> None:
        """工具的 source 應為 "mcp"。"""
        tools = [
            MCPToolDefinition(
                name='get_forecast',
                description='取得天氣預報',
                input_schema={'type': 'object', 'properties': {}},
            ),
        ]
        client = _make_mock_client('weather', tools)
        registry = ToolRegistry()

        adapter = MCPToolAdapter(client)
        await adapter.register_tools(registry)

        tool_defs = registry.get_tool_definitions()
        assert tool_defs[0]['name'] == 'weather__get_forecast'
        # 驗證內部 Tool 的 source（測試需要存取私有屬性）
        tool = registry._tools['weather__get_forecast']  # pyright: ignore[reportPrivateUsage]
        assert tool.source == 'mcp'

    @allure.title('多個 MCP 工具都應註冊成功')
    async def test_multiple_tools_registered(self) -> None:
        """多個 MCP 工具都應註冊成功。"""
        tools = [
            MCPToolDefinition(name='tool_a', description='A', input_schema={}),
            MCPToolDefinition(name='tool_b', description='B', input_schema={}),
            MCPToolDefinition(name='tool_c', description='C', input_schema={}),
        ]
        client = _make_mock_client('myserver', tools)
        registry = ToolRegistry()

        adapter = MCPToolAdapter(client)
        await adapter.register_tools(registry)

        names = registry.list_tools()
        assert 'myserver__tool_a' in names
        assert 'myserver__tool_b' in names
        assert 'myserver__tool_c' in names


# =============================================================================
# Rule: 執行 MCP 工具應委派給 Server
# =============================================================================


@allure.feature('MCP Server 整合')
@allure.story('執行 MCP 工具應委派給 Server')
class TestMCPToolExecution:
    """MCP 工具執行測試。"""

    @allure.title('執行 MCP 工具應委派給 Server')
    async def test_execute_delegates_to_client(self) -> None:
        """Scenario: 執行 MCP 工具應委派給 Server。"""
        tools = [
            MCPToolDefinition(
                name='get_forecast',
                description='取得天氣預報',
                input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            ),
        ]
        client = _make_mock_client('weather', tools)
        client.call_tool = AsyncMock(return_value={'forecast': 'sunny'})
        registry = ToolRegistry()

        adapter = MCPToolAdapter(client)
        await adapter.register_tools(registry)

        result = await registry.execute('weather__get_forecast', {'city': 'Taipei'})

        assert result == {'forecast': 'sunny'}
        # 確認是呼叫原始工具名稱（不帶前綴）
        client.call_tool.assert_called_once_with('get_forecast', {'city': 'Taipei'})


# =============================================================================
# Rule: MCP 連線生命週期
# =============================================================================


@allure.feature('MCP Server 整合')
@allure.story('MCP 連線生命週期')
class TestMCPLifecycle:
    """MCP 連線生命週期測試。"""

    @allure.title('關閉連線應清理資源')
    async def test_close_delegates_to_client(self) -> None:
        """Scenario: 關閉連線應清理資源。"""
        client = _make_mock_client('weather', [])

        adapter = MCPToolAdapter(client)
        await adapter.close()

        client.close.assert_called_once()

    @allure.title('MCPServerConfig 應正確儲存配置')
    def test_mcp_server_config(self) -> None:
        """MCPServerConfig 應正確儲存配置。"""
        config = MCPServerConfig(
            name='weather',
            command=['node', 'weather-server.js'],
            env={'API_KEY': 'test'},
        )

        assert config.name == 'weather'
        assert config.command == ['node', 'weather-server.js']
        assert config.env == {'API_KEY': 'test'}
