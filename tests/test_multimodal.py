"""多模態輸入測試模組。

根據 docs/features/multimodal_input.feature 規格撰寫測試案例。
涵蓋：
- Rule: Agent 應支援接收圖片訊息
- Rule: Agent 應支援接收 PDF 文件
- Rule: 純文字訊息應向後相容
- Rule: 應驗證附件大小與格式
- Rule: API 層應支援附件欄位
- Rule: 對話歷史應正確保存多模態訊息
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import allure
import pytest

from agent_core.agent import Agent
from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.multimodal import (
    SUPPORTED_MEDIA_TYPES,
    Attachment,
    build_content_blocks,
    validate_attachment,
)
from agent_core.providers.base import FinalMessage, StreamResult, UsageInfo

# --- Mock Helpers ---

# 1x1 白色 PNG（最小合法 PNG）
TINY_PNG_B64 = base64.b64encode(
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
    b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
    b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
).decode()

# 最小合法 PDF
TINY_PDF_B64 = base64.b64encode(b'%PDF-1.0 minimal').decode()


def _make_final_message(
    text: str = '回應內容',
    stop_reason: str = 'end_turn',
) -> FinalMessage:
    return FinalMessage(
        content=[{'type': 'text', 'text': text}],
        stop_reason=stop_reason,
        usage=UsageInfo(input_tokens=10, output_tokens=20),
    )


class MockProvider:
    """模擬的 LLM Provider。"""

    def __init__(self, responses: list[tuple[list[str], FinalMessage]]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.call_args_list: list[dict[str, Any]] = []

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamResult]:
        self.call_args_list.append({'messages': messages})

        text_chunks, final_msg = self._responses[self._call_count]
        self._call_count += 1

        async def _text_stream() -> AsyncIterator[str]:
            for chunk in text_chunks:
                yield chunk

        async def _get_final() -> FinalMessage:
            return final_msg

        yield StreamResult(text_stream=_text_stream(), get_final_result=_get_final)

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> int:
        return 0

    async def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int = 8192,
    ) -> FinalMessage:
        return FinalMessage(
            content=[{'type': 'text', 'text': ''}],
            stop_reason='end_turn',
            usage=UsageInfo(),
        )


async def _collect_stream(agent: Agent, message: str, **kwargs: Any) -> str:
    """收集串流回應文字。"""
    chunks: list[str] = []
    async for chunk in agent.stream_message(message, **kwargs):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return ''.join(chunks)


def _make_agent(provider: Any) -> Agent:
    config = AgentCoreConfig(provider=ProviderConfig(api_key='sk-test'))
    return Agent(config=config, provider=provider, token_counter=None, usage_monitor=None)


# =============================================================================
# Rule: Agent 應支援接收圖片訊息
# =============================================================================


@allure.feature('多模態輸入（圖片與 PDF）')
@allure.story('Agent 應支援接收圖片訊息')
class TestMultimodalImageInput:
    """Rule: Agent 應支援接收圖片訊息。"""

    @allure.title('使用者傳送 base64 圖片與文字')
    async def test_base64_image_with_text(self) -> None:
        """Scenario: 使用者傳送 base64 圖片與文字。"""
        provider = MockProvider([(['這是一張圖片'], _make_final_message('這是一張圖片'))])
        agent = _make_agent(provider)

        attachments = [Attachment(media_type='image/png', data=TINY_PNG_B64)]
        result = await _collect_stream(agent, '描述這張圖片', attachments=attachments)

        assert result == '這是一張圖片'
        # 對話歷史應包含 image 區塊
        user_msg = agent.conversation[0]
        assert user_msg['role'] == 'user'
        content: list[dict[str, Any]] = user_msg['content']
        assert isinstance(content, list)

        content_types = [b['type'] for b in content]
        assert 'image' in content_types
        assert 'text' in content_types

    @allure.title('使用者傳送 URL 圖片與文字')
    async def test_url_image_with_text(self) -> None:
        """Scenario: 使用者傳送 URL 圖片與文字。"""
        provider = MockProvider([(['看到了'], _make_final_message('看到了'))])
        agent = _make_agent(provider)

        attachments = [Attachment(media_type='image/png', url='https://example.com/img.png')]
        result = await _collect_stream(agent, '這張圖是什麼', attachments=attachments)

        assert result == '看到了'
        user_msg = agent.conversation[0]
        content: list[dict[str, Any]] = user_msg['content']
        assert isinstance(content, list)

        image_block = [b for b in content if b['type'] == 'image'][0]
        assert image_block['source']['type'] == 'url'
        assert image_block['source']['url'] == 'https://example.com/img.png'

    @allure.title('使用者傳送多張圖片')
    async def test_multiple_images(self) -> None:
        """Scenario: 使用者傳送多張圖片。"""
        provider = MockProvider([(['比較結果'], _make_final_message('比較結果'))])
        agent = _make_agent(provider)

        attachments = [
            Attachment(media_type='image/png', data=TINY_PNG_B64),
            Attachment(media_type='image/jpeg', data=TINY_PNG_B64),
        ]
        await _collect_stream(agent, '比較這兩張', attachments=attachments)

        user_msg = agent.conversation[0]
        content: list[dict[str, Any]] = user_msg['content']
        assert isinstance(content, list)

        image_blocks = [b for b in content if b['type'] == 'image']
        text_blocks = [b for b in content if b['type'] == 'text']
        assert len(image_blocks) == 2
        assert len(text_blocks) == 1


# =============================================================================
# Rule: Agent 應支援接收 PDF 文件
# =============================================================================


@allure.feature('多模態輸入（圖片與 PDF）')
@allure.story('Agent 應支援接收 PDF 文件')
class TestMultimodalPDFInput:
    """Rule: Agent 應支援接收 PDF 文件。"""

    @allure.title('使用者傳送 base64 PDF 與文字')
    async def test_base64_pdf_with_text(self) -> None:
        """Scenario: 使用者傳送 base64 PDF 與文字。"""
        provider = MockProvider([(['PDF 內容'], _make_final_message('PDF 內容'))])
        agent = _make_agent(provider)

        attachments = [Attachment(media_type='application/pdf', data=TINY_PDF_B64)]
        result = await _collect_stream(agent, '摘要這份文件', attachments=attachments)

        assert result == 'PDF 內容'
        user_msg = agent.conversation[0]
        content: list[dict[str, Any]] = user_msg['content']
        assert isinstance(content, list)

        doc_blocks = [b for b in content if b['type'] == 'document']
        assert len(doc_blocks) == 1
        assert doc_blocks[0]['source']['media_type'] == 'application/pdf'


# =============================================================================
# Rule: 純文字訊息應向後相容
# =============================================================================


@allure.feature('多模態輸入（圖片與 PDF）')
@allure.story('純文字訊息應向後相容')
class TestMultimodalBackwardCompatibility:
    """Rule: 純文字訊息應向後相容。"""

    @allure.title('使用者僅傳送文字')
    async def test_text_only_message(self) -> None:
        """Scenario: 使用者僅傳送文字。"""
        provider = MockProvider([(['Hello'], _make_final_message('Hello'))])
        agent = _make_agent(provider)

        result = await _collect_stream(agent, 'Hello')

        assert result == 'Hello'
        # 純文字時 content 應為字串，維持向後相容
        user_msg = agent.conversation[0]
        assert user_msg['content'] == 'Hello'

    @allure.title('空附件列表應視為純文字')
    async def test_empty_attachments_treated_as_text(self) -> None:
        """空附件列表應視為純文字。"""
        provider = MockProvider([(['OK'], _make_final_message('OK'))])
        agent = _make_agent(provider)

        result = await _collect_stream(agent, 'Hi', attachments=[])

        assert result == 'OK'
        user_msg = agent.conversation[0]
        assert user_msg['content'] == 'Hi'


# =============================================================================
# Rule: 應驗證附件大小與格式
# =============================================================================


@allure.feature('多模態輸入（圖片與 PDF）')
@allure.story('應驗證附件大小與格式')
class TestMultimodalValidation:
    """Rule: 應驗證附件大小與格式。"""

    @allure.title('圖片超過大小限制')
    def test_image_size_limit(self) -> None:
        """Scenario: 圖片超過大小限制。"""
        # 20MB = 20 * 1024 * 1024 bytes → base64 約 26.67MB 字元
        huge_data = base64.b64encode(b'\x00' * (20 * 1024 * 1024 + 1)).decode()
        attachment = Attachment(media_type='image/png', data=huge_data)

        with pytest.raises(ValueError, match='過大'):
            validate_attachment(attachment)

    @allure.title('PDF 超過大小限制')
    def test_pdf_size_limit(self) -> None:
        """Scenario: PDF 超過大小限制。"""
        huge_data = base64.b64encode(b'\x00' * (32 * 1024 * 1024 + 1)).decode()
        attachment = Attachment(media_type='application/pdf', data=huge_data)

        with pytest.raises(ValueError, match='過大'):
            validate_attachment(attachment)

    @allure.title('不支援的 media_type')
    def test_unsupported_media_type(self) -> None:
        """Scenario: 不支援的 media_type。"""
        attachment = Attachment(media_type='video/mp4', data='abc')

        with pytest.raises(ValueError, match='不支援'):
            validate_attachment(attachment)

    @allure.title('所有支援的 media_type 都應通過驗證')
    def test_supported_media_types(self) -> None:
        """所有支援的 media_type 都應通過驗證。"""
        for mt in SUPPORTED_MEDIA_TYPES:
            attachment = Attachment(media_type=mt, data=TINY_PNG_B64)
            validate_attachment(attachment)  # 不應拋出例外

    @allure.title('附件必須提供 data 或 url 其中之一')
    def test_attachment_must_have_data_or_url(self) -> None:
        """附件必須提供 data 或 url 其中之一。"""
        attachment = Attachment(media_type='image/png')

        with pytest.raises(ValueError, match='data.*url'):
            validate_attachment(attachment)


# =============================================================================
# Rule: build_content_blocks 正確組合
# =============================================================================


@allure.feature('多模態輸入（圖片與 PDF）')
@allure.story('API 層應支援附件欄位')
class TestBuildContentBlocks:
    """驗證 content blocks 組合邏輯。"""

    @allure.title('base64 圖片應產生正確的 content block')
    def test_image_base64_block(self) -> None:
        """base64 圖片應產生正確的 content block。"""
        attachments = [Attachment(media_type='image/png', data=TINY_PNG_B64)]
        blocks = build_content_blocks('描述圖片', attachments)
        assert isinstance(blocks, list)

        assert len(blocks) == 2
        assert blocks[0]['type'] == 'image'
        assert blocks[0]['source'] == {
            'type': 'base64',
            'media_type': 'image/png',
            'data': TINY_PNG_B64,
        }
        assert blocks[1] == {'type': 'text', 'text': '描述圖片'}

    @allure.title('URL 圖片應產生正確的 content block')
    def test_image_url_block(self) -> None:
        """URL 圖片應產生正確的 content block。"""
        attachments = [Attachment(media_type='image/jpeg', url='https://example.com/a.jpg')]
        blocks = build_content_blocks('看這張', attachments)
        assert isinstance(blocks, list)

        assert blocks[0]['type'] == 'image'
        assert blocks[0]['source'] == {
            'type': 'url',
            'url': 'https://example.com/a.jpg',
        }

    @allure.title('base64 PDF 應產生 document 類型的 content block')
    def test_pdf_base64_block(self) -> None:
        """base64 PDF 應產生 document 類型的 content block。"""
        attachments = [Attachment(media_type='application/pdf', data=TINY_PDF_B64)]
        blocks = build_content_blocks('摘要', attachments)
        assert isinstance(blocks, list)

        assert blocks[0]['type'] == 'document'
        assert blocks[0]['source'] == {
            'type': 'base64',
            'media_type': 'application/pdf',
            'data': TINY_PDF_B64,
        }

    @allure.title('混合圖片與 PDF 應正確產生各自的 block')
    def test_mixed_attachments(self) -> None:
        """混合圖片與 PDF 應正確產生各自的 block。"""
        attachments = [
            Attachment(media_type='image/png', data=TINY_PNG_B64),
            Attachment(media_type='application/pdf', data=TINY_PDF_B64),
        ]
        blocks = build_content_blocks('分析', attachments)
        assert isinstance(blocks, list)

        assert len(blocks) == 3
        assert blocks[0]['type'] == 'image'
        assert blocks[1]['type'] == 'document'
        assert blocks[2]['type'] == 'text'

    @allure.title('無附件時應回傳原始文字字串')
    def test_text_only_returns_string(self) -> None:
        """無附件時應回傳原始文字字串。"""
        result = build_content_blocks('Hello', None)
        assert result == 'Hello'

    @allure.title('空附件列表應回傳原始文字字串')
    def test_empty_attachments_returns_string(self) -> None:
        """空附件列表應回傳原始文字字串。"""
        result = build_content_blocks('Hello', [])
        assert result == 'Hello'


# =============================================================================
# Rule: 對話歷史應正確保存多模態訊息
# =============================================================================


@allure.feature('多模態輸入（圖片與 PDF）')
@allure.story('對話歷史應正確保存多模態訊息')
class TestMultimodalConversationHistory:
    """Rule: 對話歷史應正確保存多模態訊息。"""

    @allure.title('多模態訊息持久化後可恢復')
    async def test_multimodal_message_persisted_in_conversation(self) -> None:
        """Scenario: 多模態訊息持久化後可恢復。"""
        provider = MockProvider([(['分析完成'], _make_final_message('分析完成'))])
        agent = _make_agent(provider)

        attachments = [Attachment(media_type='image/png', data=TINY_PNG_B64)]
        await _collect_stream(agent, '分析圖片', attachments=attachments)

        # 對話應有 2 筆：user + assistant
        assert len(agent.conversation) == 2

        # user 訊息包含 image + text blocks
        user_msg = agent.conversation[0]
        content: list[dict[str, Any]] = user_msg['content']
        assert isinstance(content, list)
        assert any(b['type'] == 'image' for b in content)

        # assistant 回應正常
        assistant_msg = agent.conversation[1]
        assert assistant_msg['role'] == 'assistant'

    @allure.title('Provider 應收到完整的多模態 content blocks')
    async def test_provider_receives_multimodal_messages(self) -> None:
        """Provider 應收到完整的多模態 content blocks。"""
        provider = MockProvider([(['OK'], _make_final_message('OK'))])
        agent = _make_agent(provider)

        attachments = [Attachment(media_type='image/png', data=TINY_PNG_B64)]
        await _collect_stream(agent, '看圖', attachments=attachments)

        # 檢查 Provider 收到的 messages（第一個 user message）
        sent_messages = provider.call_args_list[0]['messages']
        user_msg = next(m for m in sent_messages if m['role'] == 'user')
        user_content: list[dict[str, Any]] = user_msg['content']
        assert isinstance(user_content, list)
        assert any(b['type'] == 'image' for b in user_content)
