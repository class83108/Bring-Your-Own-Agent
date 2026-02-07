"""FastAPI 應用程序入口。

提供聊天 API 端點，支援 SSE 串流回應與會話管理。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.session import SessionManager
from agent_core.skills.registry import SkillRegistry
from agent_core.tools.file_read import detect_language, read_file_handler
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.setup import create_default_registry
from agent_core.types import ContentBlock

# 在匯入 Anthropic client 之前加載 .env
load_dotenv()

logger = logging.getLogger(__name__)

# --- 配置 ---
REDIS_URL = 'redis://localhost:6381'
STATIC_DIR = 'static'
SANDBOX_DIR = 'workspace/sandbox'
IS_PRODUCTION = os.environ.get('ENV') == 'production'

# --- 全局單例 ---
session_manager = SessionManager(redis_url=REDIS_URL)
tool_registry: ToolRegistry | None = None
skill_registry: SkillRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """應用程序生命週期管理。"""
    global tool_registry, skill_registry

    logger.info('應用程序啟動')

    # 啟動時建立工具註冊表與技能註冊表
    sandbox_root = Path(SANDBOX_DIR)
    tool_registry = create_default_registry(sandbox_root)
    skill_registry = SkillRegistry()

    yield

    await session_manager.close()
    logger.info('應用程序關閉')


app = FastAPI(title='Agent Chat API', lifespan=lifespan)


# --- 請求模型 ---
class ChatRequest(BaseModel):
    """聊天請求本體。"""

    message: str


# --- 輔助函數 ---
def _extract_text_from_content(content: Any) -> str | None:
    """從 content 中提取文字內容。

    Args:
        content: MessageParam 中的 content（可能是字串或 blocks 列表）

    Returns:
        提取的文字內容，若無文字則返回 None
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        # 使用 cast 告訴型別檢查器這是 dict 列表
        blocks = cast(list[dict[str, Any]], content)
        for item in blocks:
            # 檢查是否包含 type='text'
            if item.get('type') != 'text':
                continue
            # 轉換為 ContentBlock 以獲得更精確的型別提示
            block = cast(ContentBlock, item)
            text = block.get('text', '')
            if text:
                text_parts.append(text)
        if text_parts:
            return ''.join(text_parts)

    return None


def _convert_to_frontend_messages(
    conversation: list[Any],
) -> list[dict[str, str]]:
    """將 MessageParam 格式轉換為前端友善的格式。

    Args:
        conversation: MessageParam 列表

    Returns:
        前端訊息列表，每個包含 role 和 content
    """
    messages: list[dict[str, str]] = []

    for msg in conversation:
        role = msg.get('role', '')
        content = msg.get('content', '')

        text_content = _extract_text_from_content(content)
        if text_content is not None:
            messages.append({'role': role, 'content': text_content})

    return messages


# --- 生成會話 ID ---
def _get_or_create_session_id(session_id: str | None) -> tuple[str, bool]:
    """讀取或生成會話 ID。

    Args:
        session_id: Cookie 中的會話 ID

    Returns:
        會話 ID 與是否新建的旗標
    """
    if session_id:
        return session_id, False
    new_id = uuid.uuid4().hex
    logger.debug('生成新會話', extra={'session_id': new_id})
    return new_id, True


# --- SSE 事件格式化 ---
def _sse_event(event: str, data: Any) -> str:
    """格式化 SSE 事件。

    Args:
        event: 事件類型
        data: 事件數據（會自動 JSON 序列化）

    Returns:
        SSE 格式的字串
    """
    # 用 JSON 編碼確保換行符等特殊字元被正確傳輸
    encoded_data = json.dumps(data, ensure_ascii=False)
    return f'event: {event}\ndata: {encoded_data}\n\n'


def _extract_sse_events(conversation: list[Any]) -> list[dict[str, Any]]:
    """從對話歷史中提取工具回傳的 SSE 事件。

    Args:
        conversation: 對話歷史（MessageParam 列表）

    Returns:
        SSE 事件列表
    """
    events: list[dict[str, Any]] = []

    for msg in conversation:
        # 只處理 user role 的訊息（tool_result 在 user 訊息中）
        if msg.get('role') != 'user':
            continue

        content = msg.get('content', [])
        if not isinstance(content, list):
            continue

        # 轉換為明確型別的列表
        content_blocks = cast(list[Any], content)

        # 檢查每個 content block
        for item in content_blocks:
            if not isinstance(item, dict):
                continue

            block = cast(dict[str, Any], item)

            # 只處理 tool_result
            if block.get('type') != 'tool_result':
                continue

            # 嘗試解析 tool_result 的 content（可能是 JSON）
            tool_content = block.get('content', '')
            if not isinstance(tool_content, str):
                continue

            try:
                result_data = json.loads(tool_content)
                if isinstance(result_data, dict) and 'sse_events' in result_data:
                    if isinstance(result_data['sse_events'], list):
                        # 確保每個事件都是 dict
                        typed_events = cast(list[dict[str, Any]], result_data['sse_events'])
                        events.extend(typed_events)
            except json.JSONDecodeError:
                # content 不是 JSON，跳過
                continue

    return events


# --- 串流生成器 ---
async def _stream_chat(
    message: str,
    session_id: str,
) -> AsyncIterator[str]:
    """從 Agent 串流回應並格式化為 SSE 事件。

    Args:
        message: 使用者訊息
        session_id: 會話識別符

    Yields:
        格式化的 SSE 事件字串
    """
    # 從 Redis 讀取歷史和使用量統計
    conversation = await session_manager.load(session_id)
    usage_data = await session_manager.load_usage(session_id)

    # 建立 Agent（使用全局工具與技能註冊表，帶入歷史）
    config = AgentCoreConfig()
    provider = AnthropicProvider(config.provider)
    agent = Agent(
        config=config,
        provider=provider,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )
    agent.conversation = list(conversation)

    # 載入歷史使用量統計
    if agent.usage_monitor and usage_data:
        agent.usage_monitor.load_from_dicts(usage_data)

    try:
        async for item in agent.stream_message(message):
            if isinstance(item, str):
                # 文字 token
                yield _sse_event('token', item)
            else:
                # 事件通知（tool_call、preamble_end）
                yield _sse_event(item['type'], item.get('data', {}))

        # 提取並推送工具執行的 SSE 事件（file_open、file_change 等）
        for event in _extract_sse_events(agent.conversation):
            yield _sse_event(event['type'], event['data'])

        # 串流完成，儲存更新後的歷史和使用量統計
        await session_manager.save(session_id, agent.conversation)
        if agent.usage_monitor:
            await session_manager.save_usage(session_id, agent.usage_monitor.records)
        yield _sse_event('done', '')

    except (ValueError, Exception) as e:
        # 錯誤時傳出 SSE error 事件
        error_data = {'type': type(e).__name__, 'message': str(e)}
        yield _sse_event('error', error_data)


# --- API 路由 ---
@app.post('/api/chat/stream')
async def chat_stream(
    request: Request,
    session_id: Annotated[str | None, Cookie()] = None,
) -> StreamingResponse:
    """SSE 串流聊天端點。

    Args:
        request: HTTP 請求
        session_id: 會話 Cookie

    Returns:
        SSE 串流回應
    """
    body: dict[str, Any] = await request.json()
    chat_req = ChatRequest(**body)

    sid, _ = _get_or_create_session_id(session_id)

    response = StreamingResponse(
        _stream_chat(chat_req.message, sid),
        media_type='text/event-stream',
    )

    # 設定 Cookie（每次都設定，確保過期時間更新）
    response.set_cookie(
        key='session_id',
        value=sid,
        path='/',  # 明確指定根路徑
        httponly=True,
        samesite='lax',
        secure=IS_PRODUCTION,
        max_age=86400,  # 24 小時，與 Redis SESSION_TTL 一致
    )

    return response


@app.get('/api/chat/history')
async def chat_history(
    session_id: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    """取得會話歷史端點。

    Args:
        session_id: 會話 Cookie

    Returns:
        對話歷史列表
    """
    if not session_id:
        return JSONResponse({'messages': []})

    conversation = await session_manager.load(session_id)

    # 將 MessageParam 轉換為前端友善的格式
    messages = _convert_to_frontend_messages(conversation)

    logger.debug('取得會話歷史', extra={'session_id': session_id, 'messages': len(messages)})
    return JSONResponse({'messages': messages})


@app.post('/api/chat/reset')
async def chat_reset(
    session_id: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    """清除會話歷史端點。

    Args:
        session_id: 會話 Cookie

    Returns:
        清除結果
    """
    if not session_id:
        return JSONResponse({'status': 'ok', 'message': '無會話需清除'})

    await session_manager.reset(session_id)
    logger.info('會話歷史已清除', extra={'session_id': session_id})
    return JSONResponse({'status': 'ok', 'message': '歷史已清除'})


@app.get('/api/chat/usage')
async def chat_usage(
    session_id: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    """查看 API 使用量統計端點。

    Args:
        session_id: 會話 Cookie

    Returns:
        使用量統計摘要
    """
    if not session_id:
        return JSONResponse(
            {'error': '需要會話 ID'},
            status_code=400,
        )

    # 從 Redis 載入使用量統計並計算摘要
    from agent_core.usage_monitor import UsageMonitor

    usage_data = await session_manager.load_usage(session_id)
    monitor = UsageMonitor()
    if usage_data:
        monitor.load_from_dicts(usage_data)

    summary = monitor.get_summary()
    return JSONResponse(summary)


@app.post('/api/chat/usage/reset')
async def chat_usage_reset(
    session_id: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    """重設使用量統計端點。

    Args:
        session_id: 會話 Cookie

    Returns:
        重設結果
    """
    if not session_id:
        return JSONResponse(
            {'error': '需要會話 ID'},
            status_code=400,
        )

    await session_manager.reset_usage(session_id)
    return JSONResponse({'status': 'ok', 'message': '使用量統計已重設'})


@app.get('/api/agent/status')
async def agent_status() -> JSONResponse:
    """查詢 Agent 配置狀態端點。

    回傳目前的 model、工具清單（含來源）、技能清單（含啟用狀態）。

    Returns:
        Agent 配置狀態
    """
    config = AgentCoreConfig()

    # 工具摘要
    tools: list[dict[str, str]] = []
    if tool_registry:
        tools = tool_registry.get_tool_summaries()

    # 技能摘要
    skills: dict[str, list[str]] = {'registered': [], 'active': []}
    if skill_registry:
        skills = {
            'registered': skill_registry.list_skills(),
            'active': skill_registry.list_active_skills(),
        }

    return JSONResponse(
        {
            'model': config.provider.model,
            'max_tokens': config.provider.max_tokens,
            'tools': tools,
            'skills': skills,
        }
    )


@app.post('/api/skills/{name}/activate')
async def skill_activate(name: str) -> JSONResponse:
    """啟用指定 Skill。

    Args:
        name: Skill 名稱

    Returns:
        啟用結果
    """
    if not skill_registry:
        return JSONResponse(
            {'error': 'Skill Registry 未初始化'},
            status_code=404,
        )

    try:
        skill_registry.activate(name)
    except KeyError:
        return JSONResponse(
            {'error': f"Skill '{name}' 不存在"},
            status_code=404,
        )

    logger.info('Skill 已透過 API 啟用', extra={'skill_name': name})
    return JSONResponse({'status': 'ok', 'skill': name, 'active': True})


@app.post('/api/skills/{name}/deactivate')
async def skill_deactivate(name: str) -> JSONResponse:
    """停用指定 Skill。

    Args:
        name: Skill 名稱

    Returns:
        停用結果
    """
    if not skill_registry:
        return JSONResponse(
            {'error': 'Skill Registry 未初始化'},
            status_code=404,
        )

    # 檢查 Skill 是否存在
    if skill_registry.get(name) is None:
        return JSONResponse(
            {'error': f"Skill '{name}' 不存在"},
            status_code=404,
        )

    skill_registry.deactivate(name)
    logger.info('Skill 已透過 API 停用', extra={'skill_name': name})
    return JSONResponse({'status': 'ok', 'skill': name, 'active': False})


@app.get('/health')
async def health() -> JSONResponse:
    """健康檢查端點。"""
    return JSONResponse({'status': 'healthy'})


# --- 檔案瀏覽 API ---
def _build_tree(current_path: Path, sandbox_root: Path) -> list[dict[str, Any]]:
    """遞迴建立目錄樹結構。

    Args:
        current_path: 當前遍歷的路徑
        sandbox_root: sandbox 根目錄

    Returns:
        目錄樹結構列表
    """
    items: list[dict[str, Any]] = []

    # 跳過的目錄名稱
    skip_dirs = {'__pycache__', 'node_modules', '.git', '.venv', 'venv'}

    try:
        for entry in sorted(current_path.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
            # 跳過隱藏檔案和特定目錄
            if entry.name.startswith('.') or entry.name in skip_dirs:
                continue

            relative_path = str(entry.relative_to(sandbox_root))

            if entry.is_dir():
                items.append(
                    {
                        'name': entry.name,
                        'type': 'directory',
                        'path': relative_path,
                        'children': _build_tree(entry, sandbox_root),
                    }
                )
            else:
                items.append(
                    {
                        'name': entry.name,
                        'type': 'file',
                        'path': relative_path,
                        'language': detect_language(entry),
                    }
                )
    except PermissionError:
        logger.warning('無權限存取目錄', extra={'path': str(current_path)})

    return items


@app.get('/api/files/tree')
async def get_file_tree() -> JSONResponse:
    """取得 sandbox 目錄結構。

    Returns:
        包含目錄樹的 JSON 回應
    """
    sandbox_path = Path(SANDBOX_DIR).resolve()

    if not sandbox_path.exists():
        logger.warning('Sandbox 目錄不存在', extra={'path': SANDBOX_DIR})
        return JSONResponse({'root': SANDBOX_DIR, 'tree': []})

    tree = _build_tree(sandbox_path, sandbox_path)
    return JSONResponse({'root': SANDBOX_DIR, 'tree': tree})


@app.get('/api/files/content')
async def get_file_content(path: str) -> JSONResponse:
    """讀取檔案內容。

    Args:
        path: 檔案路徑（相對於 sandbox）

    Returns:
        包含檔案內容的 JSON 回應
    """
    sandbox_root = Path(SANDBOX_DIR).resolve()

    try:
        result = read_file_handler(path, sandbox_root)
        return JSONResponse(result)
    except FileNotFoundError as e:
        return JSONResponse({'error': str(e)}, status_code=404)
    except PermissionError as e:
        return JSONResponse({'error': str(e)}, status_code=403)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)


# --- 伺務前端靜態文件 ---
app.mount('/', StaticFiles(directory=STATIC_DIR, html=True), name='static')
