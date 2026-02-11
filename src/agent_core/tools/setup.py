"""工具註冊工廠模組。

提供建立預設工具註冊表的工廠函數。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_core.memory import (
    MEMORY_TOOL_DESCRIPTION,
    MEMORY_TOOL_PARAMETERS,
    create_memory_handler,
)
from agent_core.sandbox.base import Sandbox
from agent_core.tools.bash import bash_handler
from agent_core.tools.file_edit import edit_file_handler
from agent_core.tools.file_list import list_files_handler
from agent_core.tools.file_read import read_file_handler
from agent_core.tools.grep_search import grep_search_handler
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.think import think_handler

logger = logging.getLogger(__name__)


def create_default_registry(
    sandbox: Sandbox,
    lock_provider: Any | None = None,
    memory_dir: Path | None = None,
    web_fetch_allowed_hosts: list[str] | None = None,
    tavily_api_key: str = '',
) -> ToolRegistry:
    """建立預設的工具註冊表，包含所有內建工具。

    Args:
        sandbox: Sandbox 實例，用於限制檔案操作範圍
        lock_provider: 鎖提供者（可選，用於避免檔案競爭）
        memory_dir: 記憶目錄（可選，提供時啟用 memory 工具）
        web_fetch_allowed_hosts: 允許存取的主機清單（提供時啟用 web_fetch 工具）
        tavily_api_key: Tavily API key（提供時啟用 web_search 工具）

    Returns:
        已註冊所有內建工具的 ToolRegistry
    """
    # 從 Sandbox 取得根目錄路徑，供現有 handler 使用
    sandbox_root = Path(sandbox.validate_path('.'))

    registry = ToolRegistry(lock_provider=lock_provider)

    # 註冊 read_file 工具
    _register_read_file(registry, sandbox_root)

    # 註冊 edit_file 工具
    _register_edit_file(registry, sandbox_root)

    # 註冊 list_files 工具
    _register_list_files(registry, sandbox_root)

    # 註冊 bash 工具
    _register_bash(registry, sandbox_root)

    # 註冊 grep_search 工具
    _register_grep_search(registry, sandbox_root)

    # 註冊 think 工具
    _register_think(registry)

    # 註冊 memory 工具（可選）
    if memory_dir is not None:
        _register_memory(registry, memory_dir)

    # 註冊 web_fetch 工具（可選）
    if web_fetch_allowed_hosts is not None:
        _register_web_fetch(registry, web_fetch_allowed_hosts)

    # 註冊 web_search 工具（可選）
    if tavily_api_key:
        _register_web_search(registry, tavily_api_key)

    logger.info('預設工具註冊表已建立', extra={'tools': registry.list_tools()})
    return registry


def _register_read_file(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 read_file 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """read_file handler 閉包，綁定 sandbox_root。"""
        return read_file_handler(
            path=path,
            sandbox_root=sandbox_root,
            start_line=start_line,
            end_line=end_line,
        )

    registry.register(
        name='read_file',
        description="""\
讀取專案中的檔案內容。

使用時機：
- 修改檔案前，先讀取理解完整上下文（**必須先讀再改**）
- 分析錯誤訊息中提到的檔案
- 檢查配置檔（pyproject.toml, package.json 等）
- 了解某個模組的 API 和用法

提示：若只需找出某個 pattern 的位置，用 grep_search 更高效。

回傳：檔案的文字內容、路徑與程式語言識別。""",
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': '檔案路徑（相對於 sandbox 根目錄）',
                },
                'start_line': {
                    'type': 'integer',
                    'description': '起始行號（可選，預設為 1）',
                },
                'end_line': {
                    'type': 'integer',
                    'description': '結束行號（可選，預設讀到檔尾）',
                },
            },
            'required': ['path'],
        },
        handler=_handler,
        file_param='path',
    )


def _register_edit_file(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 edit_file 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        path: str,
        old_content: str | None = None,
        new_content: str | None = None,
        create_if_missing: bool = False,
        backup: bool = False,
    ) -> dict[str, Any]:
        """edit_file handler 閉包，綁定 sandbox_root。"""
        return edit_file_handler(
            path=path,
            sandbox_root=sandbox_root,
            old_content=old_content,
            new_content=new_content,
            create_if_missing=create_if_missing,
            backup=backup,
        )

    registry.register(
        name='edit_file',
        description="""\
編輯或建立檔案內容。

使用時機：
- 修改程式碼（修 bug、重構、新增功能）
- 建立新檔案（設定 create_if_missing=true）
- 刪除程式碼片段（只提供 old_content，不提供 new_content）

**重要**：修改前必須先用 read_file 讀取檔案，確保 old_content 精確匹配。

編輯方式：
- 使用精確的字串匹配替換（old_content → new_content）
- old_content 必須在檔案中唯一存在（避免誤修改）
- 只能操作 sandbox 內的檔案

回傳：編輯結果（是否建立、是否修改、備份路徑）。""",
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': '檔案路徑（相對於 sandbox 根目錄）',
                },
                'old_content': {
                    'type': 'string',
                    'description': '要搜尋的舊內容（編輯時必須提供，建立新檔案時省略）',
                },
                'new_content': {
                    'type': 'string',
                    'description': '要替換的新內容（省略表示刪除 old_content）',
                },
                'create_if_missing': {
                    'type': 'boolean',
                    'description': '若檔案不存在是否建立（預設 false）',
                },
                'backup': {
                    'type': 'boolean',
                    'description': '是否在編輯前備份原始檔案（預設 false）',
                },
            },
            'required': ['path'],
        },
        handler=_handler,
        file_param='path',
    )


def _register_list_files(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 list_files 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        path: str = '.',
        recursive: bool = False,
        max_depth: int | None = None,
        pattern: str | None = None,
        exclude_dirs: list[str] | None = None,
        show_hidden: bool = False,
        show_details: bool = False,
    ) -> dict[str, Any]:
        """list_files handler 閉包，綁定 sandbox_root。"""
        return list_files_handler(
            path=path,
            sandbox_root=sandbox_root,
            recursive=recursive,
            max_depth=max_depth,
            pattern=pattern,
            exclude_dirs=exclude_dirs,
            show_hidden=show_hidden,
            show_details=show_details,
        )

    registry.register(
        name='list_files',
        description="""\
列出目錄中的檔案和子目錄。

使用時機：
- 接到任務時，第一步用此工具了解專案結構
- 不確定檔案在哪個目錄時，探索專案佈局
- 用 pattern 參數尋找特定類型檔案（如 "*.py", "test_*.py"）
- 修改前了解同目錄下有哪些相關檔案（學習既有慣例）

提示：若要搜尋檔案內容中的 pattern，用 grep_search 更適合。

回傳：檔案列表、目錄列表，以及可選的詳細資訊（大小、修改時間）。""",
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': '目錄路徑（相對於 sandbox 根目錄，預設為當前目錄 "."）',
                },
                'recursive': {
                    'type': 'boolean',
                    'description': '是否遞迴列出所有子目錄中的檔案（預設 false）',
                },
                'max_depth': {
                    'type': 'integer',
                    'description': '最大遞迴深度（僅在 recursive=true 時有效，預設無限制）',
                },
                'pattern': {
                    'type': 'string',
                    'description': '檔案名稱模式，如 "*.py", "test_*.py"（預設顯示所有檔案）',
                },
                'exclude_dirs': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': '要排除的目錄名稱列表，如 ["node_modules", ".git"]',
                },
                'show_hidden': {
                    'type': 'boolean',
                    'description': '是否顯示隱藏檔案（以 . 開頭的檔案，預設 false）',
                },
                'show_details': {
                    'type': 'boolean',
                    'description': '是否顯示詳細資訊（檔案大小、修改時間，預設 false）',
                },
            },
            'required': [],
        },
        handler=_handler,
        file_param=None,  # list_files 只讀取目錄結構，不需要檔案鎖定
    )


def _register_bash(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 bash 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> dict[str, Any]:
        """bash handler 閉包，綁定 sandbox_root。"""
        return bash_handler(
            command=command,
            sandbox_root=sandbox_root,
            timeout=timeout,
            working_dir=working_dir,
        )

    registry.register(
        name='bash',
        description="""\
執行 bash 命令來操作專案環境。

使用時機與對應指令：

**測試驗證**（修改程式碼後必做）：
- Python 專案：`pytest`、`pytest tests/test_xxx.py -v`
- Node 專案：`npm test`、`npx jest`

**程式碼品質檢查**：
- Python linting：`ruff check .`、`ruff format --check .`
- 型別檢查：`pyright`、`mypy`
- Node linting：`npx eslint .`

**套件管理**：
- Python (uv)：`uv add <pkg>`、`uv pip list`、`uv sync`
- Node：`npm install`、`npm ls`

**Git 操作**：
- 查看狀態：`git status`、`git diff`
- 查看歷史：`git log --oneline -10`

**環境資訊**：
- 版本確認：`python --version`、`node --version`
- 目錄結構：`tree -L 2`（可用，但 list_files 工具通常更適合）

優先使用專用工具：讀檔用 read_file、搜尋用 grep_search、列目錄用 list_files。
bash 適合跑測試、執行構建、安裝套件等需要 shell 的操作。

限制：僅可在 sandbox 目錄內執行，禁止危險命令（rm -rf /、sudo 等）。
回傳：命令執行結果（exit code、stdout、stderr）。""",
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': '要執行的 bash 命令',
                },
                'timeout': {
                    'type': 'integer',
                    'description': '超時時間（秒），預設 120 秒',
                },
                'working_dir': {
                    'type': 'string',
                    'description': '工作目錄（相對於 sandbox 根目錄，可選）',
                },
            },
            'required': ['command'],
        },
        handler=_handler,
        file_param=None,  # bash 不操作特定檔案，不需要鎖定
    )


def _register_grep_search(registry: ToolRegistry, sandbox_root: Path) -> None:
    """註冊 grep_search 工具。

    Args:
        registry: 工具註冊表
        sandbox_root: sandbox 根目錄
    """

    def _handler(
        pattern: str,
        path: str = '.',
        include: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        case_sensitive: bool = True,
        whole_word: bool = False,
        context_lines: int = 0,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """grep_search handler 閉包，綁定 sandbox_root。"""
        return grep_search_handler(
            pattern=pattern,
            sandbox_root=sandbox_root,
            path=path,
            include=include,
            exclude_dirs=exclude_dirs,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            context_lines=context_lines,
            max_results=max_results,
        )

    registry.register(
        name='grep_search',
        description="""\
在專案中搜尋程式碼。當你需要找到某段程式碼的位置時，這是最高效的工具。

使用時機：
- 找出函數、類別、變數的定義位置（如 `def calculate_total`）
- 追蹤某個函數被哪些地方呼叫（如搜尋 `calculate_total(` ）
- 搜尋 TODO、FIXME、HACK 等標記
- 找出特定 import、設定值、錯誤訊息出現的位置
- 理解程式碼之間的依賴關係

提示：比逐一 read_file 高效得多。用 include 參數限定檔案類型（如 ["*.py"]）可加速搜尋。

回傳：匹配結果清單，包含檔案路徑、行號、匹配內容。""",
        parameters={
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': '搜尋模式（支援正則表達式）',
                },
                'path': {
                    'type': 'string',
                    'description': '搜尋路徑（相對於 sandbox 根目錄，預設為 "."）',
                },
                'include': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': '只搜尋符合模式的檔案，如 ["*.py", "*.js"]',
                },
                'exclude_dirs': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': '額外要排除的目錄（預設已排除 node_modules、.git 等）',
                },
                'case_sensitive': {
                    'type': 'boolean',
                    'description': '是否區分大小寫（預設 true）',
                },
                'whole_word': {
                    'type': 'boolean',
                    'description': '是否全詞匹配（預設 false）',
                },
                'context_lines': {
                    'type': 'integer',
                    'description': '顯示匹配行前後的上下文行數（預設 0）',
                },
                'max_results': {
                    'type': 'integer',
                    'description': '最大結果數量（預設 100）',
                },
            },
            'required': ['pattern'],
        },
        handler=_handler,
        file_param=None,  # grep_search 是唯讀搜尋，不需要鎖定
    )


def _register_think(registry: ToolRegistry) -> None:
    """註冊 think 工具。

    Args:
        registry: 工具註冊表
    """
    registry.register(
        name='think',
        description="""\
記錄你的思考過程。

使用時機：
- 面對複雜任務時，拆解步驟前先整理思路
- 分析多個方案的優缺點
- 遇到歧義時，釐清假設和推理依據

此工具不會產生任何副作用，純粹用於結構化你的思考。
思考內容會保留在對話歷史中，有助於保持推理的連貫性。""",
        parameters={
            'type': 'object',
            'properties': {
                'thought': {
                    'type': 'string',
                    'description': '你的思考內容（推理步驟、分析、假設等）',
                },
            },
            'required': ['thought'],
        },
        handler=think_handler,
    )


def _register_memory(registry: ToolRegistry, memory_dir: Path) -> None:
    """註冊 memory 工具。

    Args:
        registry: 工具註冊表
        memory_dir: 記憶目錄路徑
    """
    handler = create_memory_handler(memory_dir)

    registry.register(
        name='memory',
        description=MEMORY_TOOL_DESCRIPTION,
        parameters=MEMORY_TOOL_PARAMETERS,
        handler=handler,
    )


def _register_web_fetch(registry: ToolRegistry, allowed_hosts: list[str]) -> None:
    """註冊 web_fetch 工具。

    Args:
        registry: 工具註冊表
        allowed_hosts: 允許存取的主機清單
    """
    from agent_core.tools.web_fetch import web_fetch_handler

    async def _handler(
        url: str,
        timeout: int = 30,
        max_size: int = 1_000_000,
    ) -> dict[str, Any]:
        return await web_fetch_handler(
            url=url,
            timeout=timeout,
            max_size=max_size,
            allowed_hosts=allowed_hosts,
        )

    registry.register(
        name='web_fetch',
        description="""\
擷取網頁內容並轉換為可讀文字。

使用時機：
- 查閱線上文件或 API 參考
- 從網頁擷取特定資訊
- 探索網站結構（回傳頁面中的連結清單）

回傳內容包含：頁面標題、純文字內容、頁面中的所有連結。
限制：僅支援 http/https，有大小（1MB）和超時（30秒）限制。""",
        parameters={
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': '要擷取的網頁 URL（http 或 https）',
                },
                'timeout': {
                    'type': 'integer',
                    'description': '超時秒數（預設 30）',
                },
            },
            'required': ['url'],
        },
        handler=_handler,
    )


def _register_web_search(registry: ToolRegistry, api_key: str) -> None:
    """註冊 web_search 工具。

    Args:
        registry: 工具註冊表
        api_key: Tavily API key
    """
    from agent_core.tools.web_search import web_search_handler

    async def _handler(
        query: str,
        max_results: int = 5,
        search_depth: str = 'basic',
        topic: str = 'general',
    ) -> dict[str, Any]:
        return await web_search_handler(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            api_key=api_key,
        )

    registry.register(
        name='web_search',
        description="""\
搜尋網路並回傳結構化結果。

使用時機：
- 查找特定主題的最新資訊
- 搜尋技術文件、套件用法、API 參考
- 獲取問題的背景知識

回傳內容包含：AI 摘要回答、搜尋結果清單（標題、URL、摘要）。
搭配 web_fetch 使用：先搜尋找到相關頁面，再用 web_fetch 深入閱讀。""",
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': '搜尋查詢字串',
                },
                'max_results': {
                    'type': 'integer',
                    'description': '最大結果數量（預設 5）',
                },
                'topic': {
                    'type': 'string',
                    'enum': ['general', 'news', 'finance'],
                    'description': '搜尋主題分類（預設 general）',
                },
            },
            'required': ['query'],
        },
        handler=_handler,
    )
