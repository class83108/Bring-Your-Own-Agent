"""API 使用量監控模組。

追蹤 LLM API 的 token 使用量，支援多模型定價。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# 多模型定價表（USD per million tokens）
MODEL_PRICING: dict[str, dict[str, float]] = {
    'claude-sonnet-4-20250514': {
        'input': 3.0,
        'output': 15.0,
        'cache_write': 3.75,
        'cache_read': 0.30,
    },
    'claude-haiku-4-20250514': {
        'input': 0.80,
        'output': 4.0,
        'cache_write': 1.0,
        'cache_read': 0.08,
    },
    'claude-opus-4-20250514': {
        'input': 15.0,
        'output': 75.0,
        'cache_write': 18.75,
        'cache_read': 1.50,
    },
    # OpenAI 模型（cache_write 為 0，OpenAI 快取是自動的無寫入成本）
    'gpt-4o': {
        'input': 2.50,
        'output': 10.0,
        'cache_write': 0.0,
        'cache_read': 1.25,
    },
    'gpt-4o-mini': {
        'input': 0.15,
        'output': 0.60,
        'cache_write': 0.0,
        'cache_read': 0.075,
    },
    'gpt-4.1': {
        'input': 2.0,
        'output': 8.0,
        'cache_write': 0.0,
        'cache_read': 0.50,
    },
    'gpt-4.1-mini': {
        'input': 0.40,
        'output': 1.60,
        'cache_write': 0.0,
        'cache_read': 0.10,
    },
    'gpt-4.1-nano': {
        'input': 0.10,
        'output': 0.40,
        'cache_write': 0.0,
        'cache_read': 0.025,
    },
    'o1': {
        'input': 15.0,
        'output': 60.0,
        'cache_write': 0.0,
        'cache_read': 7.50,
    },
    'o1-mini': {
        'input': 1.10,
        'output': 4.40,
        'cache_write': 0.0,
        'cache_read': 0.55,
    },
    'o3': {
        'input': 2.0,
        'output': 8.0,
        'cache_write': 0.0,
        'cache_read': 0.50,
    },
    'o3-mini': {
        'input': 1.10,
        'output': 4.40,
        'cache_write': 0.0,
        'cache_read': 0.55,
    },
    'o4-mini': {
        'input': 1.10,
        'output': 4.40,
        'cache_write': 0.0,
        'cache_read': 0.275,
    },
}

# 找不到模型時的預設定價
_DEFAULT_PRICING: dict[str, float] = {
    'input': 3.0,
    'output': 15.0,
    'cache_write': 3.75,
    'cache_read': 0.30,
}


@dataclass
class UsageRecord:
    """單次 API 呼叫的使用量記錄。"""

    timestamp: datetime
    input_tokens: int
    output_tokens: int
    # Prompt caching 相關欄位
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """計算總輸入 tokens（含快取）。"""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def cache_hit_rate(self) -> float:
        """計算快取命中率（0-1）。"""
        total = self.total_input_tokens
        if total == 0:
            return 0.0
        return self.cache_read_input_tokens / total

    def to_dict(self) -> dict[str, Any]:
        """轉換為可序列化的字典。"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cache_creation_input_tokens': self.cache_creation_input_tokens,
            'cache_read_input_tokens': self.cache_read_input_tokens,
            'total_input_tokens': self.total_input_tokens,
            'cache_hit_rate': round(self.cache_hit_rate * 100, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        """從字典反序列化。

        Args:
            data: 序列化的字典

        Returns:
            UsageRecord 實例
        """
        # ISO 格式字串轉為 datetime
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            timestamp=timestamp,
            input_tokens=data['input_tokens'],
            output_tokens=data['output_tokens'],
            cache_creation_input_tokens=data.get('cache_creation_input_tokens', 0),
            cache_read_input_tokens=data.get('cache_read_input_tokens', 0),
        )


@dataclass
class UsageMonitor:
    """API 使用量監控器。

    追蹤每次 API 呼叫的 token 使用量，並提供統計摘要。
    支援多模型定價。
    """

    # 使用的模型名稱（用於定價計算）
    model: str = 'claude-sonnet-4-20250514'
    # 使用量記錄列表
    records: list[UsageRecord] = field(default_factory=lambda: [])
    # 是否啟用監控
    enabled: bool = True

    @property
    def _pricing(self) -> dict[str, float]:
        """取得目前模型的定價。"""
        return MODEL_PRICING.get(self.model, _DEFAULT_PRICING)

    def record(self, usage: Any) -> UsageRecord | None:
        """記錄一次 API 呼叫的使用量。

        Args:
            usage: Anthropic API 回傳的 usage 物件

        Returns:
            建立的 UsageRecord，若監控已停用則回傳 None
        """
        if not self.enabled:
            return None

        record = UsageRecord(
            timestamp=datetime.now(),
            input_tokens=getattr(usage, 'input_tokens', 0),
            output_tokens=getattr(usage, 'output_tokens', 0),
            cache_creation_input_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0,
            cache_read_input_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
        )
        self.records.append(record)

        logger.info(
            'API 使用量已記錄',
            extra={
                'input_tokens': record.input_tokens,
                'output_tokens': record.output_tokens,
                'cache_creation': record.cache_creation_input_tokens,
                'cache_read': record.cache_read_input_tokens,
                'cache_hit_rate': round(record.cache_hit_rate * 100, 2),
            },
        )

        return record

    def get_summary(self) -> dict[str, Any]:
        """取得使用量統計摘要。

        Returns:
            包含統計資訊的字典
        """
        if not self.records:
            return {
                'total_requests': 0,
                'message': '尚無使用記錄',
            }

        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        total_cache_creation = sum(r.cache_creation_input_tokens for r in self.records)
        total_cache_read = sum(r.cache_read_input_tokens for r in self.records)
        total_all_input = sum(r.total_input_tokens for r in self.records)

        # 計算整體快取命中率
        overall_cache_hit_rate = total_cache_read / total_all_input if total_all_input > 0 else 0.0

        # 根據模型定價計算成本估算
        pricing = self._pricing
        cost_input = total_input * pricing['input'] / 1_000_000
        cost_output = total_output * pricing['output'] / 1_000_000
        cost_cache_write = total_cache_creation * pricing['cache_write'] / 1_000_000
        cost_cache_read = total_cache_read * pricing['cache_read'] / 1_000_000
        total_cost = cost_input + cost_output + cost_cache_write + cost_cache_read

        # 計算如果沒有快取的成本（所有輸入都按基礎價格計算）
        cost_without_cache = (
            total_all_input * pricing['input'] + total_output * pricing['output']
        ) / 1_000_000
        cost_saved = cost_without_cache - total_cost

        return {
            'total_requests': len(self.records),
            'tokens': {
                'input': total_input,
                'output': total_output,
                'cache_creation': total_cache_creation,
                'cache_read': total_cache_read,
                'total_input': total_all_input,
            },
            'cache': {
                'hit_rate_percent': round(overall_cache_hit_rate * 100, 2),
                'requests_with_cache_hit': sum(
                    1 for r in self.records if r.cache_read_input_tokens > 0
                ),
                'requests_with_cache_write': sum(
                    1 for r in self.records if r.cache_creation_input_tokens > 0
                ),
            },
            'cost_estimate_usd': {
                'input': round(cost_input, 6),
                'output': round(cost_output, 6),
                'cache_write': round(cost_cache_write, 6),
                'cache_read': round(cost_cache_read, 6),
                'total': round(total_cost, 6),
                'saved_by_cache': round(cost_saved, 6),
            },
            'recent_records': [r.to_dict() for r in self.records[-5:]],
        }

    def reset(self) -> None:
        """重設所有記錄。"""
        self.records = []
        logger.info('使用量記錄已重設')

    def load_from_dicts(self, data: list[dict[str, Any]]) -> None:
        """從字典列表載入記錄。

        Args:
            data: 序列化的記錄列表
        """
        self.records = [UsageRecord.from_dict(item) for item in data]
        logger.info('已載入使用量記錄', extra={'records': len(self.records)})
