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
from typing import Any

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_demo.agent import Agent, AgentConfig
from agent_demo.session import SessionManager

# 在匯入 Anthropic client 之前加載 .env
load_dotenv()

logger = logging.getLogger(__name__)

# --- 配置 ---
REDIS_URL = 'redis://localhost:6381'
STATIC_DIR = 'static'
IS_PRODUCTION = os.environ.get('ENV') == 'production'

# --- 會話管理器（全局單例）---
session_manager = SessionManager(redis_url=REDIS_URL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """應用程序生命週期管理。"""
    logger.info('應用程序啟動')
    yield
    await session_manager.close()
    logger.info('應用程序關閉')


app = FastAPI(title='Agent Chat API', lifespan=lifespan)


# --- 請求模型 ---
class ChatRequest(BaseModel):
    """聊天請求本體。"""

    message: str


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
def _sse_event(event: str, data: str) -> str:
    """格式化 SSE 事件。

    Args:
        event: 事件類型
        data: 事件數據

    Returns:
        SSE 格式的字串
    """
    return f'event: {event}\ndata: {data}\n\n'


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
    # 從 Redis 讀取歷史
    conversation = await session_manager.load(session_id)

    # 建立 Agent（帶入歷史）
    agent = Agent(config=AgentConfig(), client=None)
    agent.conversation = list(conversation)

    try:
        async for token in agent.stream_message(message):
            yield _sse_event('token', token)

        # 串流完成，儲存更新後的歷史
        await session_manager.save(session_id, agent.conversation)
        yield _sse_event('done', '')

    except (ValueError, ConnectionError, PermissionError, TimeoutError, RuntimeError) as e:
        # 錯誤時傳出 SSE error 事件
        error_data = json.dumps({'type': type(e).__name__, 'message': str(e)}, ensure_ascii=False)
        yield _sse_event('error', error_data)


# --- API 路由 ---
@app.post('/api/chat/stream')
async def chat_stream(
    request: Request,
    session_id: str | None = Cookie(default=None),
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

    sid, is_new = _get_or_create_session_id(session_id)

    response = StreamingResponse(
        _stream_chat(chat_req.message, sid),
        media_type='text/event-stream',
    )

    # 若為新會話，設定 Cookie
    if is_new:
        response.set_cookie(
            key='session_id',
            value=sid,
            httponly=True,
            samesite='lax',
            secure=IS_PRODUCTION,
        )

    return response


@app.post('/api/chat/reset')
async def chat_reset(
    session_id: str | None = Cookie(default=None),
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


@app.get('/health')
async def health() -> JSONResponse:
    """健康檢查端點。"""
    return JSONResponse({'status': 'healthy'})


# --- 伺務前端靜態文件 ---
app.mount('/', StaticFiles(directory=STATIC_DIR, html=True), name='static')
