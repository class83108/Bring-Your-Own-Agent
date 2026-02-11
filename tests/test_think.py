"""Think Tool 測試模組。

涵蓋：
- 正常思考記錄
- 空白輸入處理
- 透過 ToolRegistry 整合測試
"""

from __future__ import annotations

from typing import Any

import allure
import pytest

from agent_core.tools.think import think_handler


@allure.feature('Think Tool')
@allure.story('應記錄思考內容')
class TestThinkHandler:
    """think_handler 單元測試。"""

    @allure.title('正常思考字串應回傳 recorded')
    def test_normal_thought(self) -> None:
        """正常思考字串應回傳 status=recorded。"""
        result = think_handler('我需要先讀取檔案，再修改程式碼')
        assert result['status'] == 'recorded'
        assert result['thought'] == '我需要先讀取檔案，再修改程式碼'

    @allure.title('空字串應回傳 empty')
    def test_empty_string(self) -> None:
        """空字串應回傳 status=empty。"""
        result = think_handler('')
        assert result['status'] == 'empty'
        assert result['thought'] == ''

    @allure.title('純空白字串應回傳 empty')
    def test_whitespace_only(self) -> None:
        """純空白字串應回傳 status=empty。"""
        result = think_handler('   \n\t  ')
        assert result['status'] == 'empty'
        assert result['thought'] == ''

    @allure.title('長文字思考應正常記錄')
    def test_long_thought(self) -> None:
        """長文字應完整保留。"""
        long_text = '分析步驟：\n' + '\n'.join(f'{i}. 步驟 {i}' for i in range(1, 20))
        result = think_handler(long_text)
        assert result['status'] == 'recorded'
        assert result['thought'] == long_text


@allure.feature('Think Tool')
@allure.story('應可透過 ToolRegistry 使用')
class TestThinkRegistration:
    """Think 工具在 ToolRegistry 中的整合測試。"""

    @allure.title('create_default_registry 應包含 think 工具')
    def test_think_in_default_registry(self, tmp_path: Any) -> None:
        """預設註冊表應包含 think。"""
        from agent_core.sandbox import LocalSandbox
        from agent_core.tools.setup import create_default_registry

        registry = create_default_registry(LocalSandbox(root=tmp_path))
        assert 'think' in registry.list_tools()

    @allure.title('透過 registry 執行 think 工具')
    @pytest.mark.asyncio
    async def test_execute_think_via_registry(self, tmp_path: Any) -> None:
        """透過 registry.execute 執行 think 應回傳正確結果。"""
        from agent_core.sandbox import LocalSandbox
        from agent_core.tools.setup import create_default_registry

        registry = create_default_registry(LocalSandbox(root=tmp_path))
        result: dict[str, Any] = await registry.execute('think', {'thought': '讓我想想這個問題'})
        assert result['status'] == 'recorded'
        assert result['thought'] == '讓我想想這個問題'
