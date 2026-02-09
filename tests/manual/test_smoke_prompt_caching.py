"""Prompt Caching Smoke Test。

驗證 prompt caching 功能實際生效。

這些測試會呼叫真實的 Claude API，因此：
- 需要設定 ANTHROPIC_API_KEY 環境變數
- 每次執行會產生 API 費用
- 不應該在 CI 中自動執行

執行方式：
    uv run pytest tests/manual/test_smoke_prompt_caching.py --run-smoke -v

註：必須加上 --run-smoke 參數才會執行，否則會被自動跳過
"""

from __future__ import annotations

import time
from typing import Any

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig
from agent_core.providers.anthropic_provider import AnthropicProvider

pytestmark = pytest.mark.smoke


@allure.feature('Provider 抽象層')
@allure.story('Anthropic Provider 應支援 Prompt Caching (Smoke)')
class TestPromptCachingSmoke:
    """Smoke test - 驗證 Prompt Caching 實際生效。"""

    @allure.title('驗證第二次請求能使用緩存')
    async def test_cache_hit_on_second_request(self) -> None:
        """驗證第二次請求能使用緩存。

        Scenario:
        1. 發送第一次請求（創建緩存）
        2. 發送第二次請求（應該命中緩存）
        3. 檢查 usage 指標驗證緩存生效

        註：需要足夠的 tokens 才能觸發緩存（Sonnet 4 需要 >= 1024 tokens）
        """
        # 使用唯一的 system prompt 確保測試間緩存隔離
        unique_prompt = f'測試專用 Agent - Prompt Caching Test - {time.time()}'
        config = AgentCoreConfig(system_prompt=unique_prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider)

        # 添加足夠長的內容以觸發緩存（需要 >= 1024 tokens）
        long_context = (
            """
        這是一段很長的上下文資訊，用來確保 token 數量足夠觸發 prompt caching。
        Python 是一種高級編程語言，由 Guido van Rossum 創建。
        它具有簡潔明確的語法，易於學習和使用。
        Python 支援多種編程範式，包括面向對象、命令式、函數式和過程式編程。
        Python 擁有豐富的標準庫，涵蓋了從文件 I/O 到網絡編程的各種功能。
        許多知名公司如 Google、Facebook、Netflix 都在使用 Python。
        Python 在數據科學、機器學習、Web 開發等領域都有廣泛應用。
        常見的 Python 框架包括 Django、Flask、FastAPI 等。
        數據科學領域常用的庫有 NumPy、Pandas、Matplotlib、Scikit-learn。
        機器學習框架包括 TensorFlow、PyTorch、Keras 等。
        Python 的包管理工具包括 pip、conda、poetry、uv 等。
        虛擬環境工具有 venv、virtualenv、pipenv 等。
        Python 的版本管理工具有 pyenv、asdf 等。
        代碼格式化工具有 black、autopep8、yapf 等。
        Linting 工具有 pylint、flake8、ruff 等。
        類型檢查工具有 mypy、pyright、pyre 等。
        測試框架有 pytest、unittest、nose2 等。
        文檔生成工具有 Sphinx、MkDocs、pdoc 等。
        持續集成工具有 GitHub Actions、GitLab CI、Jenkins 等。
        Python 3.12 帶來了更好的錯誤信息和性能改進。
        Python 支援裝飾器、生成器、上下文管理器等高級特性。
        asyncio 提供了異步 I/O 支持，適合高並發場景。
        type hints 讓 Python 代碼更加健壯和可維護。
        """
            * 5
        )  # 重複多次以達到 >= 1024 tokens

        # 第一次請求 - 創建緩存
        first_chunks: list[str | dict[str, Any]] = []
        async for chunk in agent.stream_message(f'{long_context}\n\n請回答 "OK"'):
            first_chunks.append(chunk)

        # 檢查 usage_monitor 有記錄
        assert agent.usage_monitor is not None
        first_usage = agent.usage_monitor.get_summary()
        print('\n第一次請求 usage:', first_usage)

        # 第一次應該創建緩存
        assert first_usage['tokens']['cache_creation'] > 0
        assert first_usage['tokens']['cache_read'] == 0

        # 第二次請求 - 應該命中緩存
        second_chunks: list[str | dict[str, Any]] = []
        async for chunk in agent.stream_message('請再回答一次 "OK"'):
            second_chunks.append(chunk)

        # 獲取第二次請求後的總計 usage
        second_total_usage = agent.usage_monitor.get_summary()
        print('第二次請求後總計 usage:', second_total_usage)

        # 計算第二次請求的 usage（總計 - 第一次）
        second_request_cache_read = (
            second_total_usage['tokens']['cache_read'] - first_usage['tokens']['cache_read']
        )
        second_request_cache_creation = (
            second_total_usage['tokens']['cache_creation'] - first_usage['tokens']['cache_creation']
        )

        print(f'第二次請求 cache_read: {second_request_cache_read}')
        print(f'第二次請求 cache_creation: {second_request_cache_creation}')

        # 驗證：第二次請求應該有 cache read
        assert second_request_cache_read > 0, (
            f'第二次請求應該讀取緩存，但 cache_read_input_tokens={second_request_cache_read}'
        )

        # 驗證：第二次請求也會創建新的緩存（包含新的對話）
        assert second_request_cache_creation > 0, (
            f'第二次請求應該創建新緩存，但 cache_creation_input_tokens='
            f'{second_request_cache_creation}'
        )

    @allure.title('驗證對話歷史被正確緩存')
    async def test_cache_includes_conversation_history(self) -> None:
        """驗證對話歷史被正確緩存。

        Scenario:
        1. 進行多輪對話
        2. 每次請求都應該緩存之前的對話
        3. 檢查 cache_read_input_tokens 隨對話增長而增加

        註：需要足夠的 tokens 才能觸發緩存（Sonnet 4 需要 >= 1024 tokens）
        """
        # 使用唯一的 system prompt 確保測試間緩存隔離
        unique_prompt = f'測試專用 Agent - Conversation History Test - {time.time()}'
        config = AgentCoreConfig(system_prompt=unique_prompt)
        provider = AnthropicProvider(config.provider)
        agent = Agent(config=config, provider=provider)

        # 添加足夠長的內容以觸發緩存
        long_intro = (
            """
        我想和你進行一段對話。首先讓我自我介紹一下背景資訊。
        Python 是一種高級編程語言，具有簡潔的語法和強大的功能。
        它在數據科學、Web 開發、自動化等領域都有廣泛應用。
        我平時使用 Python 進行各種開發工作，包括 API 開發、數據分析等。
        我熟悉的框架包括 FastAPI、Django、Flask 等。
        我也使用 pytest 進行測試，使用 ruff 進行代碼檢查。
        在開發過程中，我會使用 Git 進行版本控制。
        我習慣使用 VS Code 作為編輯器，配合各種擴展提升效率。
        我認為代碼品質很重要，所以會注重測試和文檔。
        我也會關注最新的 Python 技術和最佳實踐。
        """
            * 5
        )

        # 第一輪對話
        async for _ in agent.stream_message(f'{long_intro}\n\n我的名字是小明'):
            pass

        assert agent.usage_monitor is not None
        first_usage = agent.usage_monitor.get_summary()
        print('\n第一輪 usage:', first_usage)

        # 第二輪對話
        async for _ in agent.stream_message('請重複我的名字'):
            pass

        assert agent.usage_monitor is not None
        second_total_usage = agent.usage_monitor.get_summary()
        print('第二輪後總計 usage:', second_total_usage)

        # 第二輪應該讀取第一輪的緩存
        second_cache_read = (
            second_total_usage['tokens']['cache_read'] - first_usage['tokens']['cache_read']
        )
        assert second_cache_read > 0, '第二輪對話應該讀取第一輪的緩存'

        # 第三輪對話
        async for _ in agent.stream_message('我的名字拼音怎麼寫？'):
            pass

        assert agent.usage_monitor is not None
        third_total_usage = agent.usage_monitor.get_summary()
        print('第三輪後總計 usage:', third_total_usage)

        # 第三輪的 cache_read 應該比第二輪多（因為包含更多對話歷史）
        third_cache_read = (
            third_total_usage['tokens']['cache_read'] - second_total_usage['tokens']['cache_read']
        )
        assert third_cache_read > 0, '第三輪對話應該讀取前兩輪的緩存'

        # 驗證：第三輪讀取的緩存應該比第二輪多（因為對話變長了）
        assert third_cache_read >= second_cache_read, (
            f'第三輪 cache_read ({third_cache_read}) 應該 >= 第二輪 ({second_cache_read})'
        )

        print(
            f'\n✅ 緩存隨對話增長：第二輪讀取 {second_cache_read} tokens，'
            f'第三輪讀取 {third_cache_read} tokens'
        )
