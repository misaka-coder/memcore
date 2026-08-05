"""LLMClient 接口 —— 记忆库对模型的唯一依赖(model-agnostic 的关键)。

记忆库需要 3 类结构化调用(summary/semantic/reinforcement),
裸 `llm_call(system, user)` 会丢掉契约类型/超时/重试/错误态。所以注入这个接口,谁用谁喂自己的模型。

实现要求:失败必须**结构化返回**(ok=False + error + 用 fallback),
绝不抛裸异常打断主流程,也绝不返回半截 JSON。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    SUMMARY = "summary"
    SEMANTIC = "semantic"
    REINFORCEMENT = "reinforcement"


class ResponseFormat(str, Enum):
    JSON = "json"  # summary / semantic / reinforcement


@dataclass
class LLMResult:
    ok: bool
    data: Any = None  # JSON dict;失败时为 fallback
    error: str = ""
    latency_ms: int = 0
    attempts: int = 0
    degraded_to_fallback: bool = False


@dataclass
class LLMRequest:
    task_type: TaskType
    system_prompt: str
    user_prompt: str
    response_format: ResponseFormat = ResponseFormat.JSON
    timeout_s: float = 30.0
    max_retries: int = 1
    temperature: float = 0.2
    fallback: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def call(self, request: LLMRequest) -> LLMResult:
        """执行一次结构化调用;任何失败都要落到 LLMResult(ok=False) 上,不得抛裸异常。"""
        raise NotImplementedError
