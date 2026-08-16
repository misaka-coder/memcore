"""Official provider profile and normalized-event shape adapters.

The timeline remains provider-neutral. These adapters are the only public place
where a host validates provider-native tool-call/result shapes. Actual transport
conformance requires a host capture checked by ``validate_provider_wire_capture``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .projection import (
    ANTHROPIC_PROFILE,
    DEEPSEEK_PROFILE,
    OPENAI_PROFILE,
    OPENAI_RESPONSES_PROFILE,
)


@dataclass(frozen=True)
class NormalizedContextEvent:
    """Provider-neutral event used at the adapter boundary."""

    kind: str
    role: str
    content: Any = None
    tool_name: str = ""
    call_id: str = ""
    arguments: str = ""
    status: str = "success"


def normalize_context_event(event: Mapping[str, Any]) -> NormalizedContextEvent:
    """Normalize one host event without changing its ``event.*`` kind."""

    if not isinstance(event, Mapping):
        raise TypeError("context_event_must_be_object")
    kind = str(event.get("kind") or "").strip()
    if not kind.startswith("event."):
        raise ValueError("context_event_kind_must_start_with_event")
    payload = event.get("payload")
    if payload is None:
        payload = {key: value for key, value in event.items() if key not in {"kind", "role"}}
    return NormalizedContextEvent(
        kind=kind,
        role=str(event.get("role") or "user"),
        content=payload,
        status=str(event.get("status") or "success"),
    )


class ContextProviderAdapter(Protocol):
    profile: str

    def normalize(self, messages: Sequence[Mapping[str, Any]]) -> tuple[NormalizedContextEvent, ...]: ...


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        # Validate and return the original string: tool arguments are a wire
        # contract, not a semantic object to be reformatted twice.
        json.loads(value)
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


class OpenAIChatContextAdapter:
    profile = OPENAI_PROFILE

    def normalize(self, messages: Sequence[Mapping[str, Any]]) -> tuple[NormalizedContextEvent, ...]:
        result: list[NormalizedContextEvent] = []
        for message in messages:
            role = str(message.get("role") or "")
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                for call in message["tool_calls"]:
                    function = call.get("function") if isinstance(call, Mapping) else {}
                    function = function if isinstance(function, Mapping) else {}
                    result.append(
                        NormalizedContextEvent(
                            kind="tool_call",
                            role=role,
                            content=message.get("content"),
                            tool_name=str(function.get("name") or ""),
                            call_id=str(call.get("id") or ""),
                            arguments=_json_arguments(function.get("arguments")),
                        )
                    )
                continue
            if role == "tool":
                result.append(
                    NormalizedContextEvent(
                        kind="tool_result",
                        role=role,
                        content=message.get("content"),
                        call_id=str(message.get("tool_call_id") or ""),
                    )
                )
                continue
            result.append(NormalizedContextEvent(kind="message", role=role, content=message.get("content")))
        return tuple(result)


class DeepSeekChatContextAdapter(OpenAIChatContextAdapter):
    profile = DEEPSEEK_PROFILE


class AnthropicMessagesContextAdapter:
    profile = ANTHROPIC_PROFILE

    def normalize(self, messages: Sequence[Mapping[str, Any]]) -> tuple[NormalizedContextEvent, ...]:
        result: list[NormalizedContextEvent] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            blocks = content if isinstance(content, list) else []
            if role == "assistant" and blocks:
                for block in blocks:
                    if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                        continue
                    result.append(
                        NormalizedContextEvent(
                            kind="tool_call",
                            role=role,
                            content=content,
                            tool_name=str(block.get("name") or ""),
                            call_id=str(block.get("id") or ""),
                            arguments=_json_arguments(block.get("input")),
                        )
                    )
                if any(isinstance(block, Mapping) and block.get("type") == "tool_use" for block in blocks):
                    continue
            if role == "user" and blocks:
                for block in blocks:
                    if isinstance(block, Mapping) and block.get("type") == "tool_result":
                        result.append(
                            NormalizedContextEvent(
                                kind="tool_result",
                                role=role,
                                content=block.get("content"),
                                call_id=str(block.get("tool_use_id") or ""),
                                status="error" if block.get("is_error") else "success",
                            )
                        )
                if any(isinstance(block, Mapping) and block.get("type") == "tool_result" for block in blocks):
                    continue
            result.append(NormalizedContextEvent(kind="message", role=role, content=content))
        return tuple(result)


class OpenAIResponsesContextAdapter:
    profile = OPENAI_RESPONSES_PROFILE

    def normalize(self, messages: Sequence[Mapping[str, Any]]) -> tuple[NormalizedContextEvent, ...]:
        result: list[NormalizedContextEvent] = []
        for message in messages:
            kind = str(message.get("type") or "message")
            if kind == "function_call":
                result.append(
                    NormalizedContextEvent(
                        kind="tool_call",
                        role="assistant",
                        tool_name=str(message.get("name") or ""),
                        call_id=str(message.get("call_id") or message.get("id") or ""),
                        arguments=_json_arguments(message.get("arguments")),
                    )
                )
            elif kind == "function_call_output":
                result.append(
                    NormalizedContextEvent(
                        kind="tool_result",
                        role="tool",
                        content=message.get("output"),
                        call_id=str(message.get("call_id") or ""),
                    )
                )
            else:
                result.append(
                    NormalizedContextEvent(
                        kind="message",
                        role=str(message.get("role") or "user"),
                        content=message.get("content"),
                    )
                )
        return tuple(result)


def official_context_adapters() -> dict[str, ContextProviderAdapter]:
    return {
        OPENAI_PROFILE: OpenAIChatContextAdapter(),
        DEEPSEEK_PROFILE: DeepSeekChatContextAdapter(),
        ANTHROPIC_PROFILE: AnthropicMessagesContextAdapter(),
        OPENAI_RESPONSES_PROFILE: OpenAIResponsesContextAdapter(),
    }


__all__ = [
    "AnthropicMessagesContextAdapter",
    "ContextProviderAdapter",
    "DeepSeekChatContextAdapter",
    "NormalizedContextEvent",
    "normalize_context_event",
    "OpenAIChatContextAdapter",
    "OpenAIResponsesContextAdapter",
    "official_context_adapters",
]
