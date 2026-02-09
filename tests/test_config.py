"""配置系統測試模組。

根據 docs/features/config.feature 規格撰寫測試案例。
涵蓋：
- Rule: 應支援配置 LLM Provider
- Rule: 應支援配置 System Prompt
"""

from __future__ import annotations

import os
from unittest.mock import patch

import allure


@allure.feature('Agent 配置系統')
@allure.story('應支援配置 LLM Provider')
class TestDefaultConfig:
    """測試預設配置行為。"""

    @allure.title('使用預設配置建立 Agent — Provider 類型應為 anthropic')
    def test_default_provider_type(self) -> None:
        """Scenario: 使用預設配置建立 Agent — Provider 類型應為 anthropic。"""
        from agent_core.config import AgentCoreConfig

        config = AgentCoreConfig()
        assert config.provider.provider_type == 'anthropic'

    @allure.title('使用預設配置建立 Agent — 模型應為預設值')
    def test_default_model(self) -> None:
        """Scenario: 使用預設配置建立 Agent — 模型應為預設值。"""
        from agent_core.config import DEFAULT_MODEL, AgentCoreConfig

        config = AgentCoreConfig()
        assert config.provider.model == DEFAULT_MODEL


@allure.feature('Agent 配置系統')
@allure.story('應支援配置 LLM Provider')
class TestCustomConfig:
    """測試自訂配置。"""

    @allure.title('自訂模型與 API Key')
    def test_custom_model_and_api_key(self) -> None:
        """Scenario: 自訂模型與 API Key。"""
        from agent_core.config import AgentCoreConfig, ProviderConfig

        config = AgentCoreConfig(
            provider=ProviderConfig(
                model='claude-haiku-4-20250514',
                api_key='sk-test-key',
            ),
        )
        assert config.provider.model == 'claude-haiku-4-20250514'
        assert config.provider.api_key == 'sk-test-key'

    @allure.title('API Key 從環境變數讀取')
    def test_api_key_from_env(self) -> None:
        """Scenario: API Key 從環境變數讀取。"""
        from agent_core.config import ProviderConfig

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-env-key'}):
            provider_config = ProviderConfig()
            assert provider_config.get_api_key() == 'sk-env-key'

    @allure.title('明確指定的 API Key 應優先於環境變數')
    def test_explicit_api_key_overrides_env(self) -> None:
        """明確指定的 API Key 應優先於環境變數。"""
        from agent_core.config import ProviderConfig

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-env-key'}):
            provider_config = ProviderConfig(api_key='sk-explicit-key')
            assert provider_config.get_api_key() == 'sk-explicit-key'

    @allure.title('沒有 API Key 時應回傳 None')
    def test_no_api_key_available(self) -> None:
        """沒有 API Key 時應回傳 None。"""
        from agent_core.config import ProviderConfig

        with patch.dict(os.environ, {}, clear=True):
            provider_config = ProviderConfig()
            assert provider_config.get_api_key() is None


@allure.feature('Agent 配置系統')
@allure.story('應支援配置 System Prompt')
class TestSystemPromptConfig:
    """測試 System Prompt 配置。"""

    @allure.title('自訂 System Prompt')
    def test_custom_system_prompt(self) -> None:
        """Scenario: 自訂 System Prompt。"""
        from agent_core.config import AgentCoreConfig

        config = AgentCoreConfig(system_prompt='你是健身教練')
        assert config.system_prompt == '你是健身教練'

    @allure.title('預設 system prompt 應有值')
    def test_default_system_prompt(self) -> None:
        """預設 system prompt 應有值。"""
        from agent_core.config import DEFAULT_SYSTEM_PROMPT, AgentCoreConfig

        config = AgentCoreConfig()
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
