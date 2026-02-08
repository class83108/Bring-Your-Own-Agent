"""多模態輸入模組。

處理圖片與 PDF 附件的驗證、轉換為 Anthropic content blocks。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 支援的 media types
SUPPORTED_IMAGE_TYPES = frozenset(
    {
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
    }
)

SUPPORTED_DOCUMENT_TYPES = frozenset(
    {
        'application/pdf',
    }
)

SUPPORTED_MEDIA_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_DOCUMENT_TYPES

# 大小限制（bytes）
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_PDF_SIZE = 32 * 1024 * 1024  # 32MB


@dataclass
class Attachment:
    """使用者附件（圖片或 PDF）。

    Attributes:
        media_type: MIME 類型（如 "image/png"、"application/pdf"）
        data: base64 編碼的內容（與 url 二擇一）
        url: 圖片 URL（與 data 二擇一，僅圖片支援）
    """

    media_type: str
    data: str | None = None
    url: str | None = None


def _estimate_decoded_size(base64_data: str) -> int:
    """估算 base64 資料解碼後的大小（bytes）。

    Args:
        base64_data: base64 編碼字串

    Returns:
        估算的解碼後大小
    """
    return math.ceil(len(base64_data) * 3 / 4)


def validate_attachment(attachment: Attachment) -> None:
    """驗證附件的格式與大小。

    Args:
        attachment: 要驗證的附件

    Raises:
        ValueError: 格式不支援、大小超過限制、或缺少 data/url
    """
    # 檢查 media_type
    if attachment.media_type not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(
            f'不支援的媒體類型：{attachment.media_type}。'
            f'支援的類型：{", ".join(sorted(SUPPORTED_MEDIA_TYPES))}'
        )

    # 檢查必須提供 data 或 url
    if attachment.data is None and attachment.url is None:
        raise ValueError('附件必須提供 data 或 url 其中之一')

    # 檢查大小限制（僅對 base64 data）
    if attachment.data is not None:
        decoded_size = _estimate_decoded_size(attachment.data)

        if attachment.media_type in SUPPORTED_IMAGE_TYPES:
            if decoded_size > MAX_IMAGE_SIZE:
                raise ValueError(
                    f'圖片檔案過大：{decoded_size / 1024 / 1024:.1f}MB，'
                    f'上限為 {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB'
                )
        elif attachment.media_type in SUPPORTED_DOCUMENT_TYPES:
            if decoded_size > MAX_PDF_SIZE:
                raise ValueError(
                    f'PDF 檔案過大：{decoded_size / 1024 / 1024:.1f}MB，'
                    f'上限為 {MAX_PDF_SIZE / 1024 / 1024:.0f}MB'
                )


def _attachment_to_block(attachment: Attachment) -> dict[str, Any]:
    """將附件轉換為 Anthropic content block。

    Args:
        attachment: 已驗證的附件

    Returns:
        Anthropic API 格式的 content block
    """
    # 決定 block type：圖片用 "image"，PDF 用 "document"
    if attachment.media_type in SUPPORTED_IMAGE_TYPES:
        block_type = 'image'
    else:
        block_type = 'document'

    # 建立 source
    if attachment.url is not None:
        source: dict[str, Any] = {
            'type': 'url',
            'url': attachment.url,
        }
    else:
        source = {
            'type': 'base64',
            'media_type': attachment.media_type,
            'data': attachment.data,
        }

    return {'type': block_type, 'source': source}


def build_content_blocks(
    text: str,
    attachments: list[Attachment] | None,
) -> str | list[dict[str, Any]]:
    """將文字與附件組合為 Anthropic messages content。

    無附件時回傳純文字字串（向後相容）。
    有附件時回傳 content blocks 列表。

    Args:
        text: 使用者文字訊息
        attachments: 附件列表（可選）

    Returns:
        字串或 content blocks 列表
    """
    if not attachments:
        return text

    blocks: list[dict[str, Any]] = []

    # 附件在前、文字在後（符合 Anthropic 最佳實踐）
    for attachment in attachments:
        validate_attachment(attachment)
        blocks.append(_attachment_to_block(attachment))

    blocks.append({'type': 'text', 'text': text})

    return blocks
