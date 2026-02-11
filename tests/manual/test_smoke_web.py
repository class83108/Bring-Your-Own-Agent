"""Web 工具 Smoke Test。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- web_search 測試額外需要 TAVILY_API_KEY
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_web.py -v --run-smoke
"""

from __future__ import annotations

import http.server
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.sandbox import LocalSandbox
from agent_core.tools.setup import create_default_registry

pytestmark = pytest.mark.smoke

# --- 本地 HTTP 伺服器 ---

_TEST_PAGE = """\
<html>
<head><title>Smoke Test Page</title></head>
<body>
<h1>Hello from Smoke Test</h1>
<p>This page contains the secret code: EMERALD-42.</p>
<a href="/other">Other page</a>
</body>
</html>"""


class _TestHandler(http.server.BaseHTTPRequestHandler):
    """Smoke test 用的簡易 HTTP handler。"""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(_TEST_PAGE.encode('utf-8'))

    def log_message(self, format: str, *args: Any) -> None:
        """抑制請求日誌。"""


@pytest.fixture
def local_server() -> Iterator[tuple[str, http.server.HTTPServer]]:
    """啟動本地 HTTP 伺服器，回傳 (base_url, server)。"""
    server = http.server.HTTPServer(('127.0.0.1', 0), _TestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{port}', server
    server.shutdown()


# --- Web Fetch Smoke Test ---


@allure.feature('Web Fetch 工具')
@allure.story('驗證 Agent 能透過 web_fetch 擷取網頁 (Smoke)')
class TestSmokeWebFetch:
    """Smoke test - 驗證 Agent 能使用 web_fetch 工具。"""

    @allure.title('Agent 應能擷取本地網頁並回報內容')
    async def test_agent_fetches_local_page(
        self,
        tmp_path: Path,
        local_server: tuple[str, http.server.HTTPServer],
    ) -> None:
        """Agent 應能透過 web_fetch 擷取本地頁面並提取資訊。"""
        base_url, _ = local_server

        registry = create_default_registry(
            LocalSandbox(root=tmp_path),
            web_fetch_allowed_hosts=['127.0.0.1'],
        )
        prompt = '你是網頁分析助手。使用 web_fetch 工具擷取指定網頁。'
        config = AgentCoreConfig(system_prompt=prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message(
            f'請擷取 {base_url} 的內容，告訴我頁面中的 secret code 是什麼'
        ):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)

        # Agent 應能找到頁面中的 secret code
        assert 'EMERALD' in response or 'emerald' in response.lower()

        # 對話歷史應包含工具調用
        assert len(agent.conversation) >= 4


# --- Web Search Smoke Test ---


@allure.feature('Web Search 工具')
@allure.story('驗證 Agent 能透過 web_search 搜尋網路 (Smoke)')
class TestSmokeWebSearch:
    """Smoke test - 驗證 Agent 能使用 web_search 工具。"""

    @allure.title('Agent 應能搜尋並回報結果')
    @pytest.mark.skipif(
        not os.environ.get('TAVILY_API_KEY'),
        reason='需要 TAVILY_API_KEY 環境變數',
    )
    async def test_agent_searches_web(self, tmp_path: Path) -> None:
        """Agent 應能透過 web_search 搜尋並回傳結果。"""
        tavily_key = os.environ['TAVILY_API_KEY']

        registry = create_default_registry(
            LocalSandbox(root=tmp_path),
            tavily_api_key=tavily_key,
        )
        prompt = '你是搜尋助手。使用 web_search 工具搜尋資訊。'
        config = AgentCoreConfig(system_prompt=prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(
            config=config,
            provider=provider,
            tool_registry=registry,
        )

        chunks: list[str] = []
        async for chunk in agent.stream_message('請搜尋 "Python asyncio" 並告訴我它是什麼'):
            if isinstance(chunk, str):
                chunks.append(chunk)

        response = ''.join(chunks)

        # 回應應包含 asyncio 相關內容
        assert len(response) > 0
        assert 'asyncio' in response.lower() or 'async' in response.lower()
