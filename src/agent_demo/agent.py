"""Agent 核心模組。

實作與 Claude API 互動的對話代理。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any  # 用於 client 類型

import anthropic
from anthropic import APIConnectionError, APIStatusError, AuthenticationError
from anthropic.types import MessageParam

logger = logging.getLogger(__name__)

# 預設配置
DEFAULT_MODEL = 'claude-sonnet-4-20250514'
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SYSTEM_PROMPT = '你是一位專業的程式開發助手。請使用繁體中文回答。'


@dataclass
class AgentConfig:
    """Agent 配置。"""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    timeout: float = 30.0


@dataclass
class Agent:
    """對話代理。

    負責管理與 Claude API 的對話互動，維護對話歷史，
    並支援串流回應。

    Attributes:
        config: Agent 配置
        client: Anthropic API 客戶端
        conversation: 對話歷史紀錄
    """

    config: AgentConfig = field(default_factory=AgentConfig)
    client: Any = None  # anthropic.AsyncAnthropic | MagicMock
    conversation: list[MessageParam] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """初始化 API 客戶端。"""
        if self.client is None:
            self.client = anthropic.AsyncAnthropic()
        logger.info('Agent 已初始化', extra={'model': self.config.model})

    def reset_conversation(self) -> None:
        """重設對話歷史。"""
        self.conversation = []
        logger.debug('對話歷史已重設')

    async def stream_message(
        self,
        content: str,
    ) -> AsyncIterator[str]:
        """以串流方式發送訊息並逐步取得回應。

        Args:
            content: 使用者訊息內容

        Yields:
            回應的每個 token

        Raises:
            ValueError: 訊息為空白
            ConnectionError: API 連線失敗
            PermissionError: API 認證失敗
            TimeoutError: API 回應超時
        """
        # 驗證輸入
        content = content.strip()
        if not content:
            raise ValueError('訊息不可為空白，請輸入有效內容')

        # 加入使用者訊息到對話歷史
        self.conversation.append({'role': 'user', 'content': content})
        logger.debug('收到使用者訊息 (串流模式)', extra={'content_length': len(content)})

        response_parts: list[str] = []

        try:
            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                messages=self.conversation,
                timeout=self.config.timeout,
            ) as stream:
                async for text in stream.text_stream:
                    response_parts.append(text)
                    yield text

            # 串流完成，加入助手回應到對話歷史
            response_text = ''.join(response_parts)
            self.conversation.append({'role': 'assistant', 'content': response_text})
            logger.debug('串流回應完成', extra={'response_length': len(response_text)})

        except AuthenticationError as e:
            self.conversation.pop()
            logger.error('API 認證失敗', extra={'error': str(e)})
            raise PermissionError(
                'API 金鑰無效或已過期。請檢查 ANTHROPIC_API_KEY 環境變數是否正確設定。'
            ) from e

        except anthropic.APITimeoutError as e:
            # 注意：APITimeoutError 繼承自 APIConnectionError，必須先捕獲
            if response_parts:
                # 有部分回應，保留 user 訊息並加入部分 assistant 回應
                partial = ''.join(response_parts)
                self.conversation.append({'role': 'assistant', 'content': partial})
                logger.warning('串流超時，已保留部分回應', extra={'partial_length': len(partial)})
            else:
                # 沒有回應，移除 user 訊息
                self.conversation.pop()
            logger.error('API 回應超時', extra={'error': str(e)})
            raise TimeoutError('串流回應超時。') from e

        except APIConnectionError as e:
            # 如果已有部分回應，保留 user 訊息並加入部分 assistant 回應
            if response_parts:
                partial = ''.join(response_parts)
                self.conversation.append({'role': 'assistant', 'content': partial})
                logger.warning('串流中斷，已保留部分回應', extra={'partial_length': len(partial)})
            else:
                # 沒有回應，移除 user 訊息
                self.conversation.pop()
            logger.error('API 連線失敗', extra={'error': str(e)})
            raise ConnectionError('串流連線中斷，請檢查網路連線並稍後重試。') from e

        except APIStatusError as e:
            self.conversation.pop()
            logger.error('API 錯誤', extra={'status_code': e.status_code, 'error': str(e)})
            raise RuntimeError(f'API 錯誤 ({e.status_code}): {e.message}') from e
