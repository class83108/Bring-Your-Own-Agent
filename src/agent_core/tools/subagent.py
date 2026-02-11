"""Subagent 子代理工具模組。

提供 create_subagent 工具的 handler，讓父 Agent 可建立子 Agent
來分工處理獨立任務。子 Agent 共享父 Agent 的工具（Sandbox），
但排除 create_subagent 工具以防止遞迴。
"""

from __future__ import annotations

import logging
from typing import Any

from agent_core.config import AgentCoreConfig, ProviderConfig
from agent_core.providers.base import LLMProvider
from agent_core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# 子 Agent 的 system prompt
SUBAGENT_SYSTEM_PROMPT = """\
你是一個子代理，負責完成指定的任務。

規則：
- 專注於完成被指派的任務
- 完成後提供簡潔的結果摘要
- 使用可用的工具來完成任務
"""


async def create_subagent_handler(
    task: str,
    *,
    provider: LLMProvider,
    config: AgentCoreConfig,
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    """建立子 Agent 並執行任務。

    子 Agent 使用與父 Agent 相同的工具（共享 handler 閉包，即共享 Sandbox），
    但排除 create_subagent 工具以防止遞迴建立。

    Args:
        task: 要指派給子 Agent 的任務描述
        provider: LLM Provider 實例（與父 Agent 共享）
        config: 父 Agent 的配置（用於繼承 provider 設定）
        tool_registry: 父 Agent 的工具註冊表（會被 clone 並排除 create_subagent）

    Returns:
        包含子 Agent 執行結果的字典，格式為 {'result': str}
    """
    # 延遲 import 避免循環依賴
    from agent_core.agent import Agent

    # 複製 registry，排除 create_subagent 防止遞迴
    child_registry = tool_registry.clone(exclude=['create_subagent'])

    # 建立子 Agent 的配置
    child_config = AgentCoreConfig(
        provider=ProviderConfig(
            provider_type=config.provider.provider_type,
            model=config.provider.model,
            api_key=config.provider.api_key,
            max_tokens=config.provider.max_tokens,
            timeout=config.provider.timeout,
        ),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        max_tool_iterations=config.max_tool_iterations,
    )

    # 建立子 Agent（獨立 context，不追蹤 usage）
    child_agent = Agent(
        config=child_config,
        provider=provider,
        tool_registry=child_registry,
        usage_monitor=None,
        token_counter=None,
    )

    logger.info('子 Agent 已建立', extra={'task': task[:100]})

    # 執行任務並收集結果
    result_parts: list[str] = []
    async for token in child_agent.stream_message(task):
        if isinstance(token, str):
            result_parts.append(token)

    result_text = ''.join(result_parts)
    logger.info(
        '子 Agent 完成任務',
        extra={'task': task[:100], 'result_length': len(result_text)},
    )

    return {'result': result_text}
