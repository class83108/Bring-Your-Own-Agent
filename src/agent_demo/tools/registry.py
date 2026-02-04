"""Tool Registry 模組。

管理工具的註冊、查詢與執行。
"""

from __future__ import annotations

import inspect
import logging
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


@dataclass
class ToolRegistry:
    """工具註冊表。

    負責管理工具的註冊、查詢與執行。
    支援同步和非同步工具，以及並行執行。
    可注入 lock_provider 來避免檔案操作的競爭條件。
    """

    lock_provider: LockProvider | None = None
    _tools: dict[str, Tool] = field(default_factory=lambda: {})

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

    def list_tools(self) -> list[str]:
        """列出所有已註冊的工具名稱。

        Returns:
            工具名稱列表
        """
        return list(self._tools.keys())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """取得 Claude API 格式的工具定義。

        Returns:
            工具定義列表，符合 Claude API tools 參數格式
        """
        tools = list(self._tools.values())
        definitions: list[dict[str, Any]] = []

        for i, tool in enumerate(tools):
            definition: dict[str, Any] = {
                'name': tool.name,
                'description': tool.description,
                'input_schema': tool.parameters,
            }
            # 在最後一個工具加上 cache_control，啟用 prompt caching
            if i == len(tools) - 1:
                definition['cache_control'] = {'type': 'ephemeral'}

            definitions.append(definition)

        return definitions

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """執行指定工具。

        如果工具有指定 file_param 且有 lock_provider，
        會在執行前取得鎖，執行後釋放鎖。

        Args:
            name: 工具名稱
            arguments: 工具參數

        Returns:
            工具執行結果

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
                return await handler(**arguments)
            else:
                return handler(**arguments)
        finally:
            # 釋放鎖（如果有取得）
            if lock_key and self.lock_provider:
                await self.lock_provider.release(lock_key)
                logger.debug('已釋放檔案鎖', extra={'lock_key': lock_key})
