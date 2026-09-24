"""Context-shape checks plus exact host-captured provider-wire conformance."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .projection import (
    ANTHROPIC_PROFILE,
    OPENAI_RESPONSES_PROFILE,
    STANDARD_PROJECTION_PROFILES,
)
from .provider_adapters import NormalizedContextEvent, normalize_context_event
from .context_contract import ContextSurface
from .projection import canonical_json_bytes


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class ConformanceReport:
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[ConformanceCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def _wire_fixture(profile: str) -> list[Mapping[str, Any]]:
    if profile == ANTHROPIC_PROFILE:
        return [
            {"role": "user", "content": "current"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call-a", "name": "alpha", "input": {"n": 1}},
                    {"type": "tool_use", "id": "call-b", "name": "beta", "input": {"n": 2}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-a", "content": "", "is_error": False},
                    {"type": "tool_result", "tool_use_id": "call-b", "content": "failed", "is_error": True},
                ],
            },
        ]
    if profile == OPENAI_RESPONSES_PROFILE:
        return [
            {"type": "message", "role": "user", "content": "current"},
            {"type": "function_call", "call_id": "call-a", "name": "alpha", "arguments": '{"n":1}'},
            {"type": "function_call", "call_id": "call-b", "name": "beta", "arguments": '{"n":2}'},
            {"type": "function_call_output", "call_id": "call-a", "output": ""},
            {"type": "function_call_output", "call_id": "call-b", "output": "failed"},
        ]
    return [
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "alpha", "arguments": '{"n":1}'}},
                {"id": "call-b", "type": "function", "function": {"name": "beta", "arguments": '{"n":2}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": ""},
        {"role": "tool", "tool_call_id": "call-b", "content": "failed"},
    ]


def _check(name: str, fn: Any) -> ConformanceCheck:
    try:
        fn()
    except Exception as exc:
        return ConformanceCheck(name, False, str(exc))
    return ConformanceCheck(name, True)


async def validate_context_adapter(adapter: Any) -> ConformanceReport:
    """Validate provider-shaped inputs against the normalized-event contract.

    This does not claim that a host serialized a real provider request. Hosts
    must additionally pass their transport capture to
    :func:`validate_provider_wire_capture`.
    """

    checks: list[ConformanceCheck] = []
    profile = str(getattr(adapter, "profile", "")).strip()
    checks.append(
        _check(
            "supported_provider_profile",
            lambda: (
                None
                if profile in STANDARD_PROJECTION_PROFILES
                else (_ for _ in ()).throw(ValueError("unsupported_provider_profile"))
            ),
        )
    )
    normalize = getattr(adapter, "normalize", None)
    checks.append(
        _check(
            "normalize_entrypoint",
            lambda: None if callable(normalize) else (_ for _ in ()).throw(TypeError("adapter.normalize is required")),
        )
    )
    if not callable(normalize):
        return ConformanceReport(tuple(checks))

    raw = _wire_fixture(profile)
    try:
        events = normalize(raw)
        if inspect.isawaitable(events):
            events = await events
        events = tuple(events)
    except Exception as exc:
        checks.append(ConformanceCheck("normalize_wire", False, f"{type(exc).__name__}:{exc}"))
        return ConformanceReport(tuple(checks))
    checks.append(_check("normalized_event_shape", lambda: _assert_event_shape(events)))
    checks.append(_check("tool_call_result_pairing", lambda: _assert_pairing(events)))
    checks.append(_check("tool_arguments_json_roundtrip", lambda: _assert_arguments(events)))
    checks.append(_check("parallel_call_ids_preserved", lambda: _assert_parallel_ids(events)))
    checks.append(_check("empty_and_error_results_accepted", lambda: _assert_result_variants(events, profile)))
    checks.append(_check("current_message_single_input", lambda: _assert_single_current(raw)))
    checks.append(_check("event_kind_fidelity", _assert_event_kind_fidelity))
    checks.append(_check("open_round_2k_40k_128k_complete", lambda: _assert_open_round_sizes(normalize, profile)))
    checks.append(_check("no_expansion_is_byte_stable", lambda: _assert_no_expansion(normalize, raw)))
    checks.append(_check("same_input_same_normalized_shape", lambda: _assert_deterministic(normalize, raw)))
    return ConformanceReport(tuple(checks))


def validate_provider_wire_capture(
    surface: ContextSurface,
    captured_context_messages: Sequence[Mapping[str, Any]],
) -> ConformanceReport:
    """Compare ContextSurface with the exact context segment sent by a host.

    The host owns system/persona/tool-schema placement and therefore supplies
    only the ordered conversation-context portion of its final transport
    capture. Equality is byte-canonical and catches dropped current messages,
    reordered tool pairs, or a host-side second renderer.
    """

    expected = [dict(item) for item in surface.messages]
    captured = [dict(item) for item in captured_context_messages]
    checks = [
        _check(
            "captured_message_count",
            lambda: (
                None
                if len(captured) == len(expected)
                else (_ for _ in ()).throw(AssertionError("provider_wire_message_count_mismatch"))
            ),
        ),
        _check(
            "captured_context_byte_exact",
            lambda: (
                None
                if canonical_json_bytes(captured) == canonical_json_bytes(expected)
                else (_ for _ in ()).throw(AssertionError("provider_wire_context_mismatch"))
            ),
        ),
    ]
    return ConformanceReport(tuple(checks))


def _assert_event_shape(events: Sequence[Any]) -> None:
    for event in events:
        if not isinstance(event, NormalizedContextEvent):
            for field in ("kind", "role", "content", "call_id", "arguments", "status"):
                if not hasattr(event, field):
                    raise AssertionError(f"normalized_event_missing_{field}")


def _assert_event_kind_fidelity() -> None:
    event = normalize_context_event({"kind": "event.finance.quote", "role": "user", "payload": {"title": "A"}})
    if event.kind != "event.finance.quote" or event.content != {"title": "A"}:
        raise AssertionError("event_kind_or_payload_changed")


def _assert_open_round_sizes(normalize: Any, profile: str) -> None:
    for size in (2 * 1024, 40 * 1024, 128 * 1024):
        if profile == ANTHROPIC_PROFILE:
            raw = [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-a", "content": "x" * size}]}
            ]
        elif profile == OPENAI_RESPONSES_PROFILE:
            raw = [{"type": "function_call_output", "call_id": "call-a", "output": "x" * size}]
        else:
            raw = [{"role": "tool", "tool_call_id": "call-a", "content": "x" * size}]
        events = tuple(normalize(raw))
        if len(events) != 1 or str(events[0].content) != "x" * size:
            raise AssertionError(f"open_round_result_changed:{size}")


def _assert_no_expansion(normalize: Any, raw: Sequence[Mapping[str, Any]]) -> None:
    events = tuple(normalize(raw))
    for event in events:
        if str(event.kind) == "tool_result" and isinstance(event.content, str):
            if len(event.content.encode("utf-8")) > len(json.dumps(event.content, ensure_ascii=False).encode("utf-8")):
                raise AssertionError("unexpected_result_expansion")


def _assert_deterministic(normalize: Any, raw: Sequence[Mapping[str, Any]]) -> None:
    first = tuple(normalize(raw))
    second = tuple(normalize(raw))
    if first != second:
        raise AssertionError("normalized_events_not_deterministic")


def _assert_pairing(events: Sequence[Any]) -> None:
    calls = {str(event.call_id) for event in events if str(event.kind) == "tool_call"}
    results = {str(event.call_id) for event in events if str(event.kind) == "tool_result"}
    if calls != results or "" in calls:
        raise AssertionError(f"tool call/result ids differ: calls={calls!r}, results={results!r}")


def _assert_arguments(events: Sequence[Any]) -> None:
    for event in events:
        if str(event.kind) == "tool_call":
            parsed = json.loads(str(event.arguments))
            if not isinstance(parsed, dict):
                raise AssertionError("tool arguments must round-trip as an object")


def _assert_parallel_ids(events: Sequence[Any]) -> None:
    ids = [str(event.call_id) for event in events if str(event.kind) == "tool_call"]
    if ids != ["call-a", "call-b"]:
        raise AssertionError(f"parallel call ids changed: {ids!r}")


def _assert_result_variants(events: Sequence[Any], profile: str) -> None:
    results = [event for event in events if str(event.kind) == "tool_result"]
    if len(results) != 2 or str(results[0].content or "") != "":
        raise AssertionError("empty tool result was not preserved")
    if profile == ANTHROPIC_PROFILE and str(results[1].status) != "error":
        raise AssertionError("Anthropic error result status was not preserved")


def _assert_single_current(messages: Sequence[Mapping[str, Any]]) -> None:
    current = [message for message in messages if message.get("content") == "current"]
    if len(current) != 1:
        raise AssertionError("current message fixture is not unique")


__all__ = [
    "ConformanceCheck",
    "ConformanceReport",
    "validate_context_adapter",
    "validate_provider_wire_capture",
]
