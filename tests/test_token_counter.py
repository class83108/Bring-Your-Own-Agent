"""Token 計數器測試模組。

對應 docs/features/token_counter.feature 中定義的驗收規格。
"""

from __future__ import annotations

import allure
import pytest

from agent_core.providers.base import UsageInfo
from agent_core.token_counter import (
    MODEL_CONTEXT_WINDOWS,
    TokenCounter,
    get_context_window,
)

_approx = pytest.approx  # type: ignore[reportUnknownMemberType]


# =============================================================================
# Rule: Token 計數器應在每次 API 回應後更新
# =============================================================================


@allure.feature('Token 計數')
@allure.story('Token 計數器應在每次 API 回應後更新')
class TestTokenCounterUpdate:
    """測試 TokenCounter 從 UsageInfo 更新的行為。"""

    @allure.title('初始化後 token 計數應為 0')
    def test_initial_state_is_zero(self) -> None:
        """初始化後 token 計數應為 0。"""
        counter = TokenCounter(context_window=200_000)
        assert counter.current_context_tokens == 0
        assert counter.usage_percent == _approx(0.0)

    @allure.title('首次 API 回應後更新 token 計數')
    def test_update_from_first_usage(self) -> None:
        """Scenario: 首次 API 回應後更新 token 計數。"""
        counter = TokenCounter(context_window=200_000)
        usage = UsageInfo(input_tokens=1000, output_tokens=500)

        counter.update_from_usage(usage)

        assert counter.current_context_tokens == 1500
        assert counter.usage_percent == _approx(0.75)

    @allure.title('多輪對話後 token 計數反映最新狀態')
    def test_update_replaces_previous_count(self) -> None:
        """Scenario: 多輪對話後 token 計數反映最新狀態。"""
        counter = TokenCounter(context_window=200_000)

        # 第一次更新
        counter.update_from_usage(UsageInfo(input_tokens=1000, output_tokens=500))
        assert counter.current_context_tokens == 1500

        # 第二次更新應覆蓋（非累加）
        counter.update_from_usage(UsageInfo(input_tokens=2000, output_tokens=800))
        assert counter.current_context_tokens == 2800


# =============================================================================
# Rule: 應正確計算含快取的 token 數
# =============================================================================


@allure.feature('Token 計數')
@allure.story('應正確計算含快取的 token 數')
class TestTokenCounterWithCache:
    """測試包含快取 token 的計算。"""

    @allure.title('API 回應包含快取 token')
    def test_cache_tokens_included_in_count(self) -> None:
        """Scenario: API 回應包含快取 token。"""
        counter = TokenCounter(context_window=200_000)
        usage = UsageInfo(
            input_tokens=500,
            output_tokens=400,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=200,
        )

        counter.update_from_usage(usage)

        # input: 500 + 300 + 200 = 1000, output: 400 → total: 1400
        assert counter.current_context_tokens == 1400
        assert counter.usage_percent == _approx(0.7)


# =============================================================================
# Rule: 應正確計算 context window 使用百分比
# =============================================================================


@allure.feature('Token 計數')
@allure.story('應正確計算 context window 使用百分比')
class TestUsagePercent:
    """測試 context window 使用百分比計算。"""

    @allure.title('低使用量')
    def test_low_usage(self) -> None:
        """Scenario: 低使用量。"""
        counter = TokenCounter(context_window=200_000)
        counter.update_from_usage(UsageInfo(input_tokens=10000, output_tokens=5000))

        assert counter.usage_percent == _approx(7.5)

    @allure.title('接近上限')
    def test_high_usage(self) -> None:
        """Scenario: 接近上限。"""
        counter = TokenCounter(context_window=200_000)
        counter.update_from_usage(UsageInfo(input_tokens=150000, output_tokens=20000))

        assert counter.usage_percent == _approx(85.0)

    @allure.title('context_window 為 0 時不應除以零')
    def test_zero_context_window(self) -> None:
        """context_window 為 0 時不應除以零。"""
        counter = TokenCounter(context_window=0)
        counter.update_from_usage(UsageInfo(input_tokens=100, output_tokens=50))

        assert counter.usage_percent == _approx(0.0)


# =============================================================================
# Rule: update_from_count 精確計數
# =============================================================================


@allure.feature('Token 計數')
@allure.story('支援精確 token 計數')
class TestUpdateFromCount:
    """測試從 count_tokens API 結果更新。"""

    @allure.title('透過 provider 的 count_tokens API 精確計數')
    def test_update_from_count(self) -> None:
        """Scenario: 透過 provider 的 count_tokens API 精確計數。"""
        counter = TokenCounter(context_window=200_000)

        counter.update_from_count(input_tokens=5000)

        # 精確計數只設定 input，output 為 0（current = input only）
        assert counter.current_context_tokens == 5000

    @allure.title('精確計數應覆蓋之前的 usage 更新')
    def test_update_from_count_overwrites_previous(self) -> None:
        """精確計數應覆蓋之前的 usage 更新。"""
        counter = TokenCounter(context_window=200_000)
        counter.update_from_usage(UsageInfo(input_tokens=1000, output_tokens=500))

        counter.update_from_count(input_tokens=3000)

        assert counter.current_context_tokens == 3000


# =============================================================================
# Rule: get_status 回傳格式
# =============================================================================


@allure.feature('Token 計數')
@allure.story('應透過 API 端點回傳 token 使用量')
class TestGetStatus:
    """測試 get_status 方法回傳的格式。"""

    @allure.title('get_status 應回傳包含必要欄位的字典')
    def test_get_status_structure(self) -> None:
        """get_status 應回傳包含必要欄位的字典。"""
        counter = TokenCounter(context_window=200_000)
        counter.update_from_usage(UsageInfo(input_tokens=10000, output_tokens=5000))

        status = counter.get_status()

        assert status['current_tokens'] == 15000
        assert status['context_window'] == 200_000
        assert status['usage_percent'] == _approx(7.5)

    @allure.title('初始狀態的 get_status')
    def test_get_status_empty(self) -> None:
        """初始狀態的 get_status。"""
        counter = TokenCounter(context_window=200_000)

        status = counter.get_status()

        assert status['current_tokens'] == 0
        assert status['context_window'] == 200_000
        assert status['usage_percent'] == _approx(0.0)


# =============================================================================
# Rule: get_context_window 輔助函數
# =============================================================================


@allure.feature('Token 計數')
@allure.story('應透過 API 端點回傳 token 使用量')
class TestGetContextWindow:
    """測試 get_context_window 輔助函數。"""

    @allure.title('已知模型應回傳對應的 context window')
    def test_known_model(self) -> None:
        """已知模型應回傳對應的 context window。"""
        for model, window in MODEL_CONTEXT_WINDOWS.items():
            assert get_context_window(model) == window

    @allure.title('未知模型應回傳預設值')
    def test_unknown_model_returns_default(self) -> None:
        """未知模型應回傳預設值。"""
        result = get_context_window('unknown-model')
        assert result == 200_000
