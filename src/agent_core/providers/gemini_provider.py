"""Gemini Provider 實作。

封裝 Google Gemini SDK 的串流呼叫，實作 LLMProvider 介面。
負責內部 MessageParam 格式與 Gemini API 格式之間的雙向轉換。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from agent_core.config import ProviderConfig
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo
from agent_core.providers.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)
from agent_core.types import (
    ContentBlock,
    MessageParam,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


# Gemini FinishReason 到標準 StopReason 的映射
_GEMINI_STOP_REASON_MAP: dict[str, StopReason] = {
    'STOP': 'end_turn',
    'MAX_TOKENS': 'max_tokens',
    'SAFETY': 'end_turn',
    'RECITATION': 'end_turn',
    'LANGUAGE': 'end_turn',
    'OTHER': 'end_turn',
    'BLOCKLIST': 'end_turn',
    'PROHIBITED_CONTENT': 'end_turn',
    'MALFORMED_FUNCTION_CALL': 'end_turn',
}

# 預設的 StopReason（當 vendor 回傳 None 或未知值時使用）
_DEFAULT_STOP_REASON: StopReason = 'end_turn'


class GeminiProvider:
    """Google Gemini LLM Provider。

    負責與 Gemini API 互動，將 SDK 特定邏輯封裝在此層。
    包含訊息格式轉換、例外轉換、重試等功能。
    """

    def __init__(
        self,
        config: ProviderConfig,
        client: genai.Client | None = None,
    ) -> None:
        """初始化 Gemini Provider。

        Args:
            config: Provider 配置
            client: 自訂 Gemini client（主要用於測試注入 mock）
        """
        self._config = config
        self._client: genai.Client = client or genai.Client(api_key=config.get_api_key())

    # --- 訊息格式轉換 ---

    def _convert_messages(
        self,
        messages: list[MessageParam],
    ) -> list[types.Content]:
        """將內部 MessageParam 格式轉換為 Gemini API 訊息格式。

        主要轉換：
        1. 'assistant' role → 'model' role
        2. TextBlock → Part(text=...)
        3. ToolUseBlock → Part(function_call=...)
        4. ToolResultBlock → Content(role='user', parts=[Part(function_response=...)])

        Args:
            messages: 內部格式的對話訊息列表

        Returns:
            Gemini API 格式的 Content 列表
        """
        result: list[types.Content] = []

        for msg in messages:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            content = msg['content']

            if isinstance(content, str):
                result.append(types.Content(role=role, parts=[types.Part(text=content)]))
                continue

            parts: list[types.Part] = []
            for block in content:
                if block['type'] == 'text':
                    parts.append(types.Part(text=block['text']))
                elif block['type'] == 'tool_use':
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=block['id'],
                                name=block['name'],
                                args=block['input'],
                            )
                        )
                    )
                elif block['type'] == 'tool_result':
                    # tool_result 需要包裝為 function_response
                    # 嘗試將 content 解析為 JSON，否則包裝為 {"result": content}
                    raw_content = block['content']
                    try:
                        response_data = json.loads(raw_content)
                    except (json.JSONDecodeError, TypeError):
                        response_data = {'result': raw_content}

                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=block.get('tool_use_id'),
                                name=self._find_tool_name(messages, block.get('tool_use_id', '')),
                                response=response_data,
                            )
                        )
                    )

            if parts:
                result.append(types.Content(role=role, parts=parts))

        return result

    @staticmethod
    def _find_tool_name(messages: list[MessageParam], tool_use_id: str) -> str:
        """從對話歷史中找出 tool_use_id 對應的工具名稱。

        Args:
            messages: 對話歷史
            tool_use_id: 工具調用 ID

        Returns:
            工具名稱，找不到時回傳空字串
        """
        for msg in messages:
            content = msg['content']
            if isinstance(content, list):
                for block in content:
                    if block['type'] == 'tool_use' and block['id'] == tool_use_id:
                        return block['name']
        return ''

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[types.Tool]:
        """將標準 ToolDefinition 轉換為 Gemini FunctionDeclaration 格式。

        Args:
            tools: 標準工具定義列表

        Returns:
            Gemini 格式的 Tool 列表
        """
        declarations: list[types.FunctionDeclaration] = []
        for t in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=t['name'],
                    description=t['description'],
                    parameters=cast(types.Schema, t['input_schema']),
                )
            )
        return [types.Tool(function_declarations=declarations)]

    # --- 回應解析 ---

    def _parse_response(self, response: types.GenerateContentResponse) -> FinalMessage:
        """將 Gemini GenerateContentResponse 轉為 FinalMessage。

        Args:
            response: Gemini SDK 的回應物件

        Returns:
            轉換後的 FinalMessage
        """
        content: list[ContentBlock] = []
        has_function_call = False

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        content.append(cast(ContentBlock, TextBlock(type='text', text=part.text)))
                    elif part.function_call:
                        has_function_call = True
                        fc = part.function_call
                        content.append(
                            cast(
                                ContentBlock,
                                ToolUseBlock(
                                    type='tool_use',
                                    id=fc.id or '',
                                    name=fc.name or '',
                                    input=dict(fc.args) if fc.args else {},
                                ),
                            )
                        )

        # Usage
        usage = self._extract_usage(response.usage_metadata)

        # Stop reason：有 function_call 時強制為 tool_use
        if has_function_call:
            stop_reason: StopReason = 'tool_use'
        else:
            stop_reason = self._map_finish_reason(response)

        return FinalMessage(content=content, stop_reason=stop_reason, usage=usage)

    @staticmethod
    def _extract_usage(
        usage_metadata: types.GenerateContentResponseUsageMetadata | None,
    ) -> UsageInfo:
        """從 Gemini usage_metadata 提取使用量資訊。"""
        if usage_metadata is None:
            return UsageInfo()
        return UsageInfo(
            input_tokens=usage_metadata.prompt_token_count or 0,
            output_tokens=usage_metadata.candidates_token_count or 0,
            cache_read_input_tokens=usage_metadata.cached_content_token_count or 0,
        )

    @staticmethod
    def _map_finish_reason(response: types.GenerateContentResponse) -> StopReason:
        """從回應中提取並映射 finish_reason。"""
        if not response.candidates:
            return _DEFAULT_STOP_REASON
        candidate = response.candidates[0]
        if candidate.finish_reason is None:
            return _DEFAULT_STOP_REASON
        reason_str = candidate.finish_reason.value if candidate.finish_reason else ''
        return _GEMINI_STOP_REASON_MAP.get(reason_str, _DEFAULT_STOP_REASON)

    # --- 錯誤轉換與重試 ---

    # 可重試的 HTTP 狀態碼
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    @staticmethod
    def _convert_error(error: APIError) -> ProviderError:
        """將 Gemini SDK 例外轉換為通用 Provider 例外。

        Args:
            error: Gemini SDK 例外

        Returns:
            對應的 Provider 例外
        """
        status = getattr(error, 'code', 0) or 0
        if status in (401, 403):
            return ProviderAuthError(
                'API 金鑰無效或已過期。請檢查 GEMINI_API_KEY 環境變數是否正確設定。'
            )
        if status == 429:
            return ProviderRateLimitError(f'API 速率限制: {error}')
        if isinstance(error, ServerError):
            return ProviderError(f'API 伺服器錯誤 ({status}): {error}')
        return ProviderError(str(error))

    def _is_retryable(self, error: Exception) -> bool:
        """判斷錯誤是否可重試。"""
        if isinstance(error, ServerError):
            return True
        if isinstance(error, ClientError):
            status = getattr(error, 'code', 0) or 0
            return status in self._RETRYABLE_STATUS_CODES
        return False

    def _check_retryable_or_raise(self, error: APIError, attempt: int) -> None:
        """檢查錯誤是否可重試，不可重試則直接拋出。"""
        if not self._is_retryable(error) or attempt >= self._config.max_retries:
            raise self._convert_error(error) from error

    async def _wait_for_retry(
        self,
        attempt: int,
        error: Exception,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> None:
        """等待指數退避延遲並通知回調。"""
        delay = self._config.retry_initial_delay * (2**attempt)
        logger.warning(
            '可重試錯誤，準備重試',
            extra={
                'attempt': attempt + 1,
                'max_retries': self._config.max_retries,
                'delay': delay,
                'error': str(error),
            },
        )
        if on_retry:
            on_retry(attempt + 1, error, delay)
        await asyncio.sleep(delay)

    # --- 公開 API ---

    @asynccontextmanager
    async def stream(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> AsyncIterator[StreamResult]:
        """建立串流回應，支援自動重試。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數
            on_retry: 重試回調函數（attempt, error, delay）

        Yields:
            StreamResult 包含 text_stream 和 get_final_result

        Raises:
            ProviderAuthError: API 認證失敗
            ProviderConnectionError: API 連線失敗
            ProviderTimeoutError: API 請求超時
            ProviderRateLimitError: API 速率限制，重試耗盡
            ProviderError: 其他 API 錯誤
        """
        gemini_messages = self._convert_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        if tools:
            # list invariance: list[Tool] 無法直接賦值給 list[Tool | Callable | ...]
            config.tools = cast(types.ToolListUnion, self._convert_tools(tools))

        for attempt in range(1 + self._config.max_retries):
            try:
                # SDK 內部 async generator 未標註型別，導致 pyright 判定為 partially unknown
                response_stream = await self._client.aio.models.generate_content_stream(  # type: ignore[reportUnknownMemberType]
                    model=self._config.model,
                    contents=gemini_messages,
                    config=config,
                )

                # 累積狀態（在 closure 中共享）
                _final_message: FinalMessage | None = None
                _text_content: str = ''
                _tool_calls: list[dict[str, Any]] = []
                _finish_reason: types.FinishReason | None = None
                _usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

                async def _text_stream_iter() -> AsyncIterator[str]:
                    nonlocal _text_content, _finish_reason, _usage_metadata
                    async for chunk in response_stream:
                        if not chunk.candidates:
                            if chunk.usage_metadata:
                                _usage_metadata = chunk.usage_metadata
                            continue

                        candidate = chunk.candidates[0]
                        if candidate.finish_reason:
                            _finish_reason = candidate.finish_reason

                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                # 文字 delta
                                if part.text:
                                    _text_content += part.text
                                    yield part.text
                                # function_call
                                elif part.function_call:
                                    fc = part.function_call
                                    _tool_calls.append(
                                        {
                                            'id': fc.id or '',
                                            'name': fc.name or '',
                                            'args': dict(fc.args) if fc.args else {},
                                        }
                                    )

                        if chunk.usage_metadata:
                            _usage_metadata = chunk.usage_metadata

                async def _get_final_result() -> FinalMessage:
                    nonlocal _final_message
                    if _final_message is not None:
                        return _final_message

                    content: list[ContentBlock] = []
                    if _text_content:
                        content.append(
                            cast(ContentBlock, TextBlock(type='text', text=_text_content))
                        )

                    has_function_call = False
                    for tc in _tool_calls:
                        has_function_call = True
                        content.append(
                            cast(
                                ContentBlock,
                                ToolUseBlock(
                                    type='tool_use',
                                    id=tc['id'],
                                    name=tc['name'],
                                    input=tc['args'],
                                ),
                            )
                        )

                    usage = self._extract_usage(_usage_metadata)

                    # 有 function_call 時強制為 tool_use
                    if has_function_call:
                        stop_reason: StopReason = 'tool_use'
                    elif _finish_reason:
                        reason_str = _finish_reason.value if _finish_reason else ''
                        stop_reason = _GEMINI_STOP_REASON_MAP.get(reason_str, _DEFAULT_STOP_REASON)
                    else:
                        stop_reason = _DEFAULT_STOP_REASON

                    _final_message = FinalMessage(
                        content=content,
                        stop_reason=stop_reason,
                        usage=usage,
                    )
                    return _final_message

                yield StreamResult(
                    text_stream=_text_stream_iter(),
                    get_final_result=_get_final_result,
                )
                return

            except (ClientError, ServerError) as e:
                self._check_retryable_or_raise(e, attempt)
                await self._wait_for_retry(attempt, e, on_retry)

    async def count_tokens(
        self,
        messages: list[MessageParam],
        system: str,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        """使用 Gemini API 計算 token 數。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            tools: 工具定義列表（可選）
            max_tokens: 最大回應 token 數（未使用，保持介面一致）

        Returns:
            input token 數量
        """
        gemini_messages = self._convert_messages(messages)
        # SDK 內部 async generator 未標註型別，導致 pyright 判定為 partially unknown
        response = await self._client.aio.models.count_tokens(  # type: ignore[reportUnknownMemberType]
            model=self._config.model,
            contents=gemini_messages,
        )
        return response.total_tokens or 0

    async def create(
        self,
        messages: list[MessageParam],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        """非串流呼叫，支援自動重試。

        Args:
            messages: 對話訊息列表
            system: 系統提示詞
            max_tokens: 最大回應 token 數

        Returns:
            完整的回應訊息

        Raises:
            ProviderAuthError: API 認證失敗
            ProviderConnectionError: API 連線失敗
            ProviderTimeoutError: API 請求超時
            ProviderError: 其他 API 錯誤
        """
        gemini_messages = self._convert_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        async def _call() -> FinalMessage:
            # SDK 內部 async generator 未標註型別，導致 pyright 判定為 partially unknown
            response = await self._client.aio.models.generate_content(  # type: ignore[reportUnknownMemberType]
                model=self._config.model,
                contents=gemini_messages,
                config=config,
            )
            return self._parse_response(response)

        for attempt in range(1 + self._config.max_retries):
            try:
                return await _call()
            except (ClientError, ServerError) as e:
                self._check_retryable_or_raise(e, attempt)
                await self._wait_for_retry(attempt, e)

        # 不應到達此處，但為了型別安全
        msg = '重試次數耗盡'
        raise ProviderError(msg)
