"""Tool Registry 模組。

管理工具的註冊、查詢與執行。
支援大型工具結果的自動分頁機制。
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LockProvider(Protocol):
    """鎖提供者介面。

    用於檔案操作的分散式鎖定，避免競爭條件。
    """

    async def acquire(self, key: str) -> None:
        """取得指定 key 的鎖。"""
        ...

    async def release(self, key: str) -> None:
        """釋放指定 key 的鎖。"""
        ...


@dataclass
class Tool:
    """工具定義。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    file_param: str | None = None  # 指定哪個參數是檔案路徑
    source: str = 'native'  # 來源標記（native / skill / mcp）


# 預設結果大小上限（約 7500 tokens）
DEFAULT_MAX_RESULT_CHARS = 30000


@dataclass
class ToolRegistry:
    """工具註冊表。

    負責管理工具的註冊、查詢與執行。
    支援同步和非同步工具，以及並行執行。
    可注入 lock_provider 來避免檔案操作的競爭條件。
    超大工具結果會自動分頁，並提供 read_more 機制。
    """

    lock_provider: LockProvider | None = None
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    _tools: dict[str, Tool] = field(default_factory=lambda: {})
    _paginated_results: dict[str, str] = field(default_factory=lambda: {})
    _last_result_id: str = field(default='', init=False)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        file_param: str | None = None,
    ) -> None:
        """註冊新工具。

        Args:
            name: 工具名稱
            description: 工具描述
            parameters: JSON Schema 格式的參數定義
            handler: 工具執行函數
            file_param: 指定哪個參數是檔案路徑（用於鎖定）
        """
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            file_param=file_param,
        )
        logger.info('工具已註冊', extra={'tool_name': name, 'file_param': file_param})

    def set_tool_source(self, name: str, source: str) -> None:
        """設定工具的來源標記。

        Args:
            name: 工具名稱
            source: 來源標記（native / skill / mcp）

        Raises:
            KeyError: 工具不存在
        """
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 不存在")
        self._tools[name].source = source

    def clone(self, exclude: list[str] | None = None) -> ToolRegistry:
        """建立工具註冊表的副本，可選擇排除特定工具。

        用於建立子 Agent 的工具註冊表，共享 handler 閉包（即共享 Sandbox）。

        Args:
            exclude: 要排除的工具名稱列表（可選）

        Returns:
            新的 ToolRegistry 實例
        """
        exclude_set: set[str] = set(exclude) if exclude else set()
        new_registry = ToolRegistry(
            lock_provider=self.lock_provider,
            max_result_chars=self.max_result_chars,
        )
        for name, tool in self._tools.items():
            if name in exclude_set:
                continue
            new_registry._tools[name] = tool
        return new_registry

    def list_tools(self) -> list[str]:
        """列出所有已註冊的工具名稱。

        Returns:
            工具名稱列表
        """
        return list(self._tools.keys())

    def get_tool_summaries(self) -> list[dict[str, str]]:
        """取得所有工具的摘要資訊（含來源標記）。

        Returns:
            工具摘要列表，每項包含 name、description、source
        """
        return [
            {'name': t.name, 'description': t.description, 'source': t.source}
            for t in self._tools.values()
        ]

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """取得 LLM API 格式的工具定義。

        cache_control 等 provider 特定邏輯由 Provider 層處理。

        Returns:
            工具定義列表
        """
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'input_schema': tool.parameters,
            }
            for tool in self._tools.values()
        ]

    # -----------------------------------------------------------------
    # 分頁相關方法
    # -----------------------------------------------------------------

    def _result_to_str(self, result: Any) -> str:
        """將工具結果轉為字串。"""
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def _paginate_result(self, text: str) -> str:
        """將超大結果存入暫存區，回傳第一頁內容與分頁提示。"""
        result_id = uuid.uuid4().hex[:8]
        self._paginated_results[result_id] = text
        self._last_result_id = result_id

        total_pages = math.ceil(len(text) / self.max_result_chars)
        first_page = text[: self.max_result_chars]

        logger.info(
            '工具結果已分頁',
            extra={
                'result_id': result_id,
                'total_chars': len(text),
                'total_pages': total_pages,
            },
        )

        return (
            f'{first_page}\n\n'
            f'[第 1 頁 / 共 {total_pages} 頁] '
            f'使用 read_more(result_id="{result_id}", page=2) 取得下一頁'
        )

    def _maybe_paginate(self, result: Any) -> Any:
        """檢查結果大小，必要時自動分頁。"""
        text = self._result_to_str(result)
        if len(text) <= self.max_result_chars:
            return result
        return self._paginate_result(text)

    def read_more(self, result_id: str, page: int) -> str:
        """取得分頁結果的指定頁面。

        Args:
            result_id: 分頁結果 ID
            page: 頁碼（從 1 開始）

        Returns:
            該頁內容與分頁提示
        """
        if result_id not in self._paginated_results:
            return f'錯誤：結果 ID "{result_id}" 不存在或已過期。'

        text = self._paginated_results[result_id]
        total_pages = math.ceil(len(text) / self.max_result_chars)

        if page < 1 or page > total_pages:
            return f'錯誤：頁數超出範圍（共 {total_pages} 頁，請指定 1-{total_pages}）。'

        start = (page - 1) * self.max_result_chars
        end = start + self.max_result_chars
        page_content = text[start:end]

        if page == total_pages:
            return f'{page_content}\n\n[第 {page} 頁 / 共 {total_pages} 頁（最後一頁）]'

        return (
            f'{page_content}\n\n'
            f'[第 {page} 頁 / 共 {total_pages} 頁] '
            f'使用 read_more(result_id="{result_id}", page={page + 1}) 取得下一頁'
        )

    def get_paginated_result_count(self) -> int:
        """取得暫存區中的分頁結果數量。"""
        return len(self._paginated_results)

    def get_last_result_id(self) -> str:
        """取得最近一次分頁結果的 ID。"""
        return self._last_result_id

    def clear_paginated_results(self) -> None:
        """清除所有分頁暫存結果。"""
        self._paginated_results.clear()
        self._last_result_id = ''
        logger.debug('分頁暫存已清除')

    # -----------------------------------------------------------------
    # 工具執行
    # -----------------------------------------------------------------

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """執行指定工具。

        如果工具有指定 file_param 且有 lock_provider，
        會在執行前取得鎖，執行後釋放鎖。
        結果過大時會自動分頁。

        Args:
            name: 工具名稱
            arguments: 工具參數

        Returns:
            工具執行結果（過大時回傳第一頁與分頁提示）

        Raises:
            KeyError: 工具不存在
        """
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 不存在")

        tool = self._tools[name]
        logger.debug('執行工具', extra={'tool_name': name, 'arguments': arguments})

        # 判斷是否需要鎖定檔案
        lock_key: str | None = None
        if tool.file_param and self.lock_provider:
            lock_key = arguments.get(tool.file_param)

        # 取得鎖（如果需要）
        if lock_key and self.lock_provider:
            await self.lock_provider.acquire(lock_key)
            logger.debug('已取得檔案鎖', extra={'lock_key': lock_key})

        try:
            # 執行工具
            handler = tool.handler
            if inspect.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
            return self._maybe_paginate(result)
        finally:
            # 釋放鎖（如果有取得）
            if lock_key and self.lock_provider:
                await self.lock_provider.release(lock_key)
                logger.debug('已釋放檔案鎖', extra={'lock_key': lock_key})
