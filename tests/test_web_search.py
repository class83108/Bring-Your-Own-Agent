"""Web Search Tool 測試模組。

涵蓋：
- 空查詢處理
- 缺少 API key 處理
- 透過 ToolRegistry 整合測試（mock）
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import allure

from agent_core.tools.web_search import web_search_handler

# =============================================================================
# Rule: 基本輸入驗證
# =============================================================================


@allure.feature('Web Search Tool')
@allure.story('應驗證輸入參數')
class TestWebSearchValidation:
    """web_search_handler 輸入驗證測試。"""

    @allure.title('空查詢應回傳錯誤')
    async def test_empty_query(self) -> None:
        result = await web_search_handler(query='', api_key='test-key')
        assert 'error' in result
        assert '空' in result['error']

    @allure.title('純空白查詢應回傳錯誤')
    async def test_whitespace_query(self) -> None:
        result = await web_search_handler(query='   ', api_key='test-key')
        assert 'error' in result

    @allure.title('缺少 API key 應回傳錯誤')
    async def test_missing_api_key(self) -> None:
        result = await web_search_handler(query='test', api_key='')
        assert 'error' in result
        assert 'TAVILY_API_KEY' in result['error']


# =============================================================================
# Rule: 搜尋功能（使用 mock）
# =============================================================================


@allure.feature('Web Search Tool')
@allure.story('應回傳結構化搜尋結果')
class TestWebSearchWithMock:
    """web_search_handler 搜尋功能測試（mock Tavily API）。"""

    @allure.title('成功搜尋應回傳結構化結果')
    async def test_successful_search(self) -> None:
        mock_response: dict[str, Any] = {
            'answer': 'Python 是一種程式語言',
            'results': [
                {
                    'title': 'Python 官網',
                    'url': 'https://python.org',
                    'content': 'Python is a programming language.',
                },
                {
                    'title': 'Python 教學',
                    'url': 'https://example.com/python',
                    'content': 'Learn Python basics.',
                },
            ],
        }

        with patch('agent_core.tools.web_search.AsyncTavilyClient') as mock_client_cls:
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            mock_client_cls.return_value = mock_instance

            result = await web_search_handler(
                query='什麼是 Python',
                api_key='test-key',
            )

        assert result['query'] == '什麼是 Python'
        assert result['answer'] == 'Python 是一種程式語言'
        assert result['result_count'] == 2
        assert result['results'][0]['title'] == 'Python 官網'
        assert result['results'][0]['url'] == 'https://python.org'

    @allure.title('API 錯誤應回傳錯誤訊息')
    async def test_api_error(self) -> None:
        with patch('agent_core.tools.web_search.AsyncTavilyClient') as mock_client_cls:
            mock_instance = AsyncMock()
            mock_instance.search.side_effect = Exception('API rate limit exceeded')
            mock_client_cls.return_value = mock_instance

            result = await web_search_handler(
                query='test query',
                api_key='test-key',
            )

        assert 'error' in result
        assert 'rate limit' in result['error']

    @allure.title('搜尋結果為空時應正常處理')
    async def test_empty_results(self) -> None:
        mock_response: dict[str, Any] = {
            'answer': '',
            'results': [],
        }

        with patch('agent_core.tools.web_search.AsyncTavilyClient') as mock_client_cls:
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            mock_client_cls.return_value = mock_instance

            result = await web_search_handler(
                query='極其罕見的搜尋詞',
                api_key='test-key',
            )

        assert result['result_count'] == 0
        assert result['results'] == []


# =============================================================================
# Rule: 應可透過 ToolRegistry 使用
# =============================================================================


@allure.feature('Web Search Tool')
@allure.story('應可透過 ToolRegistry 使用')
class TestWebSearchRegistration:
    """Web search 工具在 ToolRegistry 中的整合測試。"""

    @allure.title('提供 API key 時 create_default_registry 應包含 web_search')
    def test_web_search_in_registry(self, tmp_path: Any) -> None:
        from agent_core.sandbox import LocalSandbox
        from agent_core.tools.setup import create_default_registry

        registry = create_default_registry(LocalSandbox(root=tmp_path), tavily_api_key='test-key')
        assert 'web_search' in registry.list_tools()

    @allure.title('未提供 API key 時不應包含 web_search')
    def test_web_search_not_in_registry_without_key(self, tmp_path: Any) -> None:
        from agent_core.sandbox import LocalSandbox
        from agent_core.tools.setup import create_default_registry

        registry = create_default_registry(LocalSandbox(root=tmp_path))
        assert 'web_search' not in registry.list_tools()
