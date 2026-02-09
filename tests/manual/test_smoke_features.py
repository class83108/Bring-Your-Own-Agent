"""Feature-based Smoke Tests。

根據 docs/features/smoke_tests.feature 撰寫的端對端驗證測試。

涵蓋：
- Token Counter：真實 API 呼叫後追蹤 token 使用量
- Tool Result 分頁：大檔案結果自動分頁
- Compact（Phase 2）：LLM 摘要壓縮對話
- Multimodal：圖片輸入
- Skill：SEO Skill 改變回應行為
- MCP Server：透過真實 npx 啟動並列出工具

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_features.py --run-smoke -v
"""

from __future__ import annotations

import base64
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any, cast

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig
from agent_core.multimodal import Attachment
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.skills.base import Skill
from agent_core.skills.registry import SkillRegistry
from agent_core.token_counter import TokenCounter
from agent_core.tools.setup import create_default_registry

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


async def _collect_response_with_events(
    agent: Agent,
    message: str,
    attachments: list[Attachment] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """收集串流回應的完整文字與事件。"""
    chunks: list[str] = []
    events: list[dict[str, Any]] = []
    async for chunk in agent.stream_message(message, attachments=attachments):
        if isinstance(chunk, str):
            chunks.append(chunk)
        else:
            events.append(chunk)
    return ''.join(chunks), events


def _make_agent(
    system_prompt: str = '你是助手。請用繁體中文簡短回答。',
    token_counter: TokenCounter | None = None,
) -> Agent:
    """建立使用真實 API 的 Agent。"""
    config = AgentCoreConfig(system_prompt=system_prompt)
    provider = AnthropicProvider(config.provider)
    kwargs: dict[str, Any] = {'config': config, 'provider': provider}
    if token_counter is not None:
        kwargs['token_counter'] = token_counter
    return Agent(**kwargs)


def _find_pagination_marker(conversation: list[dict[str, Any]]) -> bool:
    """檢查對話歷史中的 tool_result 是否包含分頁標記。"""
    for msg in conversation:
        if msg.get('role') != 'user':
            continue
        content = msg.get('content')
        if not isinstance(content, list):
            continue
        for block in cast(list[dict[str, Any]], content):
            if block.get('type') != 'tool_result':
                continue
            result_text = str(block.get('content', ''))
            if '第 1 頁' in result_text and '共' in result_text:
                return True
    return False


def _create_test_png() -> bytes:
    """建立一個 10x10 的紅色 PNG 圖片（純 Python，無需 Pillow）。"""
    width, height = 10, 10
    # 每行：filter byte (0) + RGB pixels
    raw_data = b''
    for _ in range(height):
        raw_data += b'\x00'  # filter type: None
        for _ in range(width):
            raw_data += b'\xff\x00\x00'  # RGB: 紅色

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack('>I', len(data)) + c + crc

    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    idat_data = zlib.compress(raw_data)

    return (
        b'\x89PNG\r\n\x1a\n'
        + png_chunk(b'IHDR', ihdr_data)
        + png_chunk(b'IDAT', idat_data)
        + png_chunk(b'IEND', b'')
    )


# =============================================================================
# Rule: Token Counter 應在真實 API 呼叫後正確追蹤使用量
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('Token Counter 應在真實 API 呼叫後正確追蹤使用量 (Smoke)')
class TestTokenCounterSmoke:
    """Smoke test — 驗證 Token Counter 在真實 API 下的行為。"""

    @allure.title('對話後 token 使用量大於零')
    async def test_token_usage_greater_than_zero_after_response(self) -> None:
        """對話後 token 使用量大於零。

        Scenario: 對話後 token 使用量大於零
          Given Agent 已啟動且配置 TokenCounter
          When 使用者傳送一則訊息並取得回應
          Then token_counter.total_input_tokens 應大於 0
          And token_counter.total_output_tokens 應大於 0
          And token_counter.usage_percent 應大於 0
        """
        agent = _make_agent()
        assert agent.token_counter is not None
        assert agent.usage_monitor is not None

        await _collect_response(agent, '請回答 "OK"')

        # 透過 usage_monitor 驗證 input/output tokens 皆大於 0
        summary = agent.usage_monitor.get_summary()
        assert summary['tokens']['total_input'] > 0
        assert summary['tokens']['output'] > 0
        # 透過 token_counter 驗證 usage_percent 大於 0
        assert agent.token_counter.current_context_tokens > 0
        assert agent.token_counter.usage_percent > 0

    @allure.title('多輪對話後 token 使用量累計增長')
    async def test_token_usage_accumulates_across_turns(self) -> None:
        """多輪對話後 token 使用量累計增長。

        Scenario: 多輪對話後 token 使用量累計增長
          Given Agent 已啟動且配置 TokenCounter
          When 使用者連續傳送兩則訊息
          Then 第二輪後的 total_input_tokens 應大於第一輪
        """
        agent = _make_agent()
        assert agent.token_counter is not None
        assert agent.usage_monitor is not None

        # 第一輪
        await _collect_response(agent, '你好')
        first_context_tokens = agent.token_counter.current_context_tokens

        # 第二輪
        await _collect_response(agent, '今天天氣如何？')
        second_context_tokens = agent.token_counter.current_context_tokens

        # 第二輪的 context tokens 應大於第一輪（因為包含更多對話歷史）
        assert second_context_tokens > first_context_tokens


# =============================================================================
# Rule: Tool Result 分頁應在結果過大時自動觸發
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('Tool Result 分頁應在結果過大時自動觸發 (Smoke)')
class TestToolResultPaginationSmoke:
    """Smoke test — 驗證大型工具結果會自動分頁。"""

    @allure.title('讀取大檔案時自動分頁')
    async def test_large_file_auto_pagination(self, tmp_path: Path) -> None:
        """讀取大檔案時自動分頁。

        Scenario: 讀取大檔案時自動分頁
          Given Agent 已啟動且啟用內建工具
          And 沙箱中有一個超過分頁閾值的大檔案
          When 使用者要求讀取該大檔案
          Then Agent 應取得分頁後的結果（含 [第 1 頁/共 N 頁] 標記）
          And Agent 能正常回應檔案內容
        """
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()

        # 建立超過分頁閾值的大檔案
        large_content = ('這是第 {i} 行的測試內容。Python is great!\n' * 50).format(i=1)
        for i in range(200):
            large_content += f'第 {i + 1} 行：這是一段很長的測試資料，用來驗證分頁功能。\n'
        (sandbox / 'large_file.txt').write_text(large_content, encoding='utf-8')

        # 使用較小的 max_result_chars 以確保觸發分頁
        registry = create_default_registry(sandbox)
        registry.max_result_chars = 500

        config = AgentCoreConfig(
            system_prompt='你是助手。當被要求讀取檔案時，使用 read_file 工具。簡短描述檔案內容。',
        )
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider, tool_registry=registry)

        response = await _collect_response(agent, '請讀取 large_file.txt')

        # 檢查 conversation 中的 tool_result 包含分頁標記
        assert _find_pagination_marker(agent.conversation), '應在 tool_result 中找到分頁標記'
        assert len(response) > 0, 'Agent 應能正常回應檔案內容'


# =============================================================================
# Rule: Compact 應在 context window 使用率高時壓縮對話
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('Compact 應在 context window 使用率高時壓縮對話 (Smoke)')
class TestCompactSmoke:
    """Smoke test — 驗證 Compact Phase 2 LLM 摘要在真實 API 下的行為。"""

    @allure.title('Phase 2 LLM 摘要 — 純文字多輪對話觸發摘要')
    async def test_compact_phase2_summary(self) -> None:
        """Phase 2 LLM 摘要 — 純文字多輪對話觸發摘要。

        Scenario: Phase 2 LLM 摘要 — 純文字多輪對話觸發摘要
          Given Agent 已啟動且配置 TokenCounter
          And compact 閾值已調低以便測試觸發
          And 不使用任何工具（避免 Phase 1 截斷導致提早返回）
          When 使用者連續進行多輪純文字對話直到超過閾值
          Then compact 應被觸發
          And 對話歷史應被替換為摘要訊息加上保留的最近訊息
          And 摘要訊息的 role 應為 user 且內容包含「摘要」
        """
        token_counter = TokenCounter(context_window=200_000)
        agent = _make_agent(
            system_prompt='你是助手。請用繁體中文簡短回答，每次回覆不超過 20 個字。',
            token_counter=token_counter,
        )

        # 進行多輪純文字對話（無工具），累積足夠的對話歷史
        messages = [
            '你好，我叫小明',
            '我喜歡寫 Python 程式',
            '最近在學習 FastAPI 框架',
        ]
        for msg in messages:
            await _collect_response(agent, msg)

        # 此時 conversation 應有 6 則訊息（3 輪 × 2）
        assert len(agent.conversation) >= 6

        # 手動將 token 計數設為超過 80% 閾值，模擬高使用率
        # 這樣下一輪對話開始時就會觸發 compact
        token_counter.set_last_tokens(170_000, 10_000)
        assert token_counter.usage_percent >= 80.0

        # 第四輪：應觸發 compact
        _, events = await _collect_response_with_events(agent, '繼續聊天吧')

        # 驗證 compact 事件被發出
        compact_events = [e for e in events if e.get('type') == 'compact']
        assert len(compact_events) > 0, 'compact 應被觸發'

        # 驗證 compact 結果
        compact_data = compact_events[0]['data']
        assert compact_data['summarized'] is True, '應執行 Phase 2 摘要'
        assert compact_data['summary'] is not None, '摘要內容不應為空'

        # 驗證對話歷史已被替換為摘要訊息
        first_msg = agent.conversation[0]
        assert first_msg['role'] == 'user', '摘要訊息的 role 應為 user'
        assert '摘要' in str(first_msg['content']), '摘要訊息應包含「摘要」'


# =============================================================================
# Rule: Multimodal 應支援圖片輸入
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('Multimodal 應支援圖片輸入 (Smoke)')
class TestMultimodalSmoke:
    """Smoke test — 驗證圖片輸入功能。"""

    @allure.title('傳送圖片並取得描述')
    async def test_send_image_and_get_description(self) -> None:
        """傳送圖片並取得描述。

        Scenario: 傳送圖片並取得描述
          Given Agent 已啟動
          And 準備一張測試用 PNG 圖片（含可辨識的內容）
          When 使用者傳送包含該圖片的訊息並詢問圖片內容
          Then Agent 應回應包含圖片內容描述的文字
        """
        agent = _make_agent(system_prompt='你是圖片分析助手。請用繁體中文描述圖片。')

        # 建立測試用 PNG 圖片（10x10 紅色方塊）
        png_bytes = _create_test_png()
        png_b64 = base64.b64encode(png_bytes).decode('ascii')

        attachment = Attachment(media_type='image/png', data=png_b64)

        response, _ = await _collect_response_with_events(
            agent,
            '這張圖片是什麼顏色？請描述圖片內容。',
            attachments=[attachment],
        )

        # Agent 應回應圖片內容描述（紅色相關）
        assert len(response) > 0, 'Agent 應回應圖片內容'
        # 紅色的各種可能描述
        color_keywords = ['紅', '红', 'red', 'Red', 'RED']
        assert any(kw in response for kw in color_keywords), (
            f'回應應包含紅色相關描述，實際回應：{response[:200]}'
        )


# =============================================================================
# Rule: Skill 應能實際改變 Agent 的回應行為
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('Skill 應能實際改變 Agent 的回應行為 (Smoke)')
class TestSkillSmoke:
    """Smoke test — 驗證 Skill 能改變 Agent 回應行為。"""

    @allure.title('SEO Skill — 啟用後回應中應包含指定關鍵字')
    async def test_seo_skill_injects_keywords(self) -> None:
        """SEO Skill — 啟用後回應中應包含指定關鍵字。

        Scenario: SEO Skill — 啟用後回應中應包含指定關鍵字
          Given Agent 已啟動
          And 已註冊並啟用一個 SEO Skill，instructions 要求在每次回應中加入
              關鍵字 "AgentCore" 和 "框架"
          When 使用者詢問一個與關鍵字無關的問題（例如「天氣如何？」）
          Then Agent 的回應中應包含關鍵字 "AgentCore"
          And Agent 的回應中應包含關鍵字 "框架"
        """
        skill_registry = SkillRegistry()
        skill_registry.register(
            Skill(
                name='seo',
                description='SEO 優化技能，確保回應包含指定關鍵字',
                instructions=(
                    '你必須在每次回應中自然地加入以下關鍵字：\n'
                    '1. "AgentCore"\n'
                    '2. "框架"\n'
                    '無論使用者問什麼問題，你的回答中都必須包含這兩個關鍵字。'
                ),
            )
        )
        skill_registry.activate('seo')

        config = AgentCoreConfig(system_prompt='你是助手。請用繁體中文回答。')
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            skill_registry=skill_registry,
        )

        response = await _collect_response(agent, '天氣如何？')

        assert 'AgentCore' in response, f'回應應包含 "AgentCore"，實際回應：{response[:200]}'
        assert '框架' in response, f'回應應包含 "框架"，實際回應：{response[:200]}'


# =============================================================================
# Rule: MCP Server 應能透過真實 npx 啟動並提供工具
# =============================================================================


@allure.feature('冒煙測試')
@allure.story('MCP Server 應能透過真實 npx 啟動並提供工具 (Smoke)')
class TestMCPServerSmoke:
    """Smoke test — 驗證真實 MCP Server 連線與工具列表。"""

    @allure.title('連接真實 MCP Server 並列出工具')
    async def test_connect_real_mcp_server_and_list_tools(self) -> None:
        """連接真實 MCP Server 並列出工具。

        Scenario: 連接真實 MCP Server 並列出工具
          Given 系統已安裝 npx（若無則跳過此測試）
          And 啟動一個 MCP Server（例如 @modelcontextprotocol/server-memory）
          When 透過 MCPClient 取得工具列表
          Then 工具列表應不為空
          And 每個工具應有 name 和 description
        """
        # 檢查 npx 是否安裝
        if shutil.which('npx') is None:
            pytest.skip('npx 未安裝，跳過 MCP Server 測試')

        # 檢查 mcp SDK 是否可用
        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError:
            pytest.skip('mcp SDK 未安裝（pip install mcp），跳過 MCP Server 測試')

        # 透過 npx 啟動 @modelcontextprotocol/server-memory
        params = StdioServerParameters(  # type: ignore
            command='npx',
            args=['-y', '@modelcontextprotocol/server-memory'],
        )

        async with stdio_client(params) as streams:  # type: ignore
            read_stream, write_stream = streams  # type: ignore
            async with ClientSession(read_stream, write_stream) as session:  # type: ignore
                await session.initialize()  # type: ignore

                # 取得工具列表
                tools_result = await session.list_tools()  # type: ignore
                tools: list[Any] = tools_result.tools  # type: ignore

                # 驗證工具列表不為空
                assert len(tools) > 0, 'MCP Server 應提供至少一個工具'  # type: ignore

                # 驗證每個工具有 name 和 description
                for tool in tools:  # type: ignore
                    assert tool.name, f'工具應有 name，實際：{tool}'  # type: ignore
                    assert tool.description, f'工具應有 description，實際：{tool.name}'  # type: ignore

    @allure.title('MCP 工具應能透過 Adapter 註冊到 ToolRegistry')
    async def test_mcp_tools_register_to_agent(self) -> None:
        """MCP 工具應能透過 Adapter 註冊到 ToolRegistry。

        延伸驗證：MCP Server 的工具可透過 MCPToolAdapter 註冊到 Agent 的 ToolRegistry。
        """
        # 檢查 npx 是否安裝
        if shutil.which('npx') is None:
            pytest.skip('npx 未安裝，跳過 MCP Server 測試')

        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError:
            pytest.skip('mcp SDK 未安裝（pip install mcp），跳過 MCP Server 測試')

        from agent_core.mcp.adapter import MCPToolAdapter
        from agent_core.mcp.client import MCPToolDefinition
        from agent_core.tools.registry import ToolRegistry

        params = StdioServerParameters(  # type: ignore
            command='npx',
            args=['-y', '@modelcontextprotocol/server-memory'],
        )

        async with stdio_client(params) as streams:  # type: ignore
            read_stream, write_stream = streams  # type: ignore
            async with ClientSession(read_stream, write_stream) as session:  # type: ignore
                await session.initialize()  # type: ignore

                # 取得工具列表並轉換為 MCPToolDefinition
                tools_result = await session.list_tools()  # type: ignore
                mcp_tools: list[Any] = tools_result.tools  # type: ignore

                # 建立符合 MCPClient Protocol 的包裝
                tool_definitions: list[MCPToolDefinition] = [
                    MCPToolDefinition(
                        name=t.name,  # type: ignore
                        description=t.description or '',  # type: ignore
                        input_schema=t.inputSchema,  # type: ignore
                    )
                    for t in mcp_tools  # type: ignore
                ]

                # 建立簡易 MCP client 包裝，透過 MCPToolAdapter 註冊工具
                class _SessionWrapper:
                    """包裝 mcp SDK 的 ClientSession 為 MCPClient Protocol。"""

                    server_name: str = 'memory'

                    def __init__(self, tools: list[MCPToolDefinition]) -> None:
                        self._tools = tools

                    async def list_tools(self) -> list[MCPToolDefinition]:
                        return self._tools

                    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
                        return await session.call_tool(tool_name, arguments)  # type: ignore

                    async def close(self) -> None:
                        # MCPClient Protocol 要求實作 close，此處由外層管理連線
                        pass

                wrapper = _SessionWrapper(tool_definitions)
                adapter = MCPToolAdapter(wrapper)

                registry = ToolRegistry()
                await adapter.register_tools(registry)

                # 驗證工具已註冊
                tool_names = registry.list_tools()
                assert len(tool_names) > 0, '應有工具被註冊到 ToolRegistry'

                # 驗證工具名稱有 server 前綴
                for name in tool_names:
                    assert name.startswith('memory__'), f'工具名稱應有前綴，實際：{name}'

                # 驗證 source 為 mcp
                summaries = registry.get_tool_summaries()
                for summary in summaries:
                    assert summary['source'] == 'mcp'

                await adapter.close()
