"""Deterministic raw-timeline read views and lossless continuation cursors."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .namespace import Namespace
from .projection import RendererRegistry, canonical_json_bytes, sanitize_timeline_value, stable_projection_hash
from .rendering import render_timeline
from .timeline import TimelineEntry, TurnRole

TIMELINE_PROJECTIONS: tuple[str, ...] = ("conversation", "full", "tools")
_CURSOR_PREFIX = "timeline-v1"
_CURSOR_VERSION = 1
_OPERATION_ROOTS = frozenset({"material", "operation", "skill", "tool"})
_TOOLS_ROOTS = frozenset({"operation", "skill", "tool"})

StableKey = tuple[int, str, int, str]


@dataclass(frozen=True)
class TimelineReadUnit:
    unit_id: str
    sort_key: StableKey
    entries: tuple[TimelineEntry, ...]
    projected_messages: tuple[dict[str, Any], ...]
    compacted_entry_count: int


@dataclass(frozen=True)
class TimelinePage:
    messages: tuple[dict[str, Any], ...]
    complete: bool
    next_cursor: str
    logical_unit_count: int
    total_logical_unit_count: int
    entry_count: int
    total_entry_count: int
    compacted_entry_count: int
    oversized_unit: bool


def normalize_timeline_projection(value: Any) -> str:
    projection = str(value or "conversation").strip().lower()
    if projection not in TIMELINE_PROJECTIONS:
        raise ValueError(f"invalid_timeline_projection:{projection}")
    return projection


def build_timeline_read_units(
    entries: Sequence[TimelineEntry],
    *,
    projection: str,
    renderer_registry: RendererRegistry,
    timezone: str,
) -> list[TimelineReadUnit]:
    """Group raw entries into complete turns/standalone units, then project each unit."""

    resolved_projection = normalize_timeline_projection(projection)
    grouped: dict[str, list[TimelineEntry]] = {}
    for entry in entries:
        conversation = entry.namespace.conversation_id
        unit_id = (
            f"turn:{conversation}:{entry.turn_id}" if entry.turn_id else f"source:{conversation}:{entry.source_id}"
        )
        grouped.setdefault(unit_id, []).append(entry)

    units: list[TimelineReadUnit] = []
    for unit_id, raw_entries in grouped.items():
        ordered = tuple(sorted(raw_entries, key=_entry_key))
        projected: list[dict[str, Any]] = []
        compacted = 0
        for entry in ordered:
            message, detail = project_timeline_entry(
                entry,
                projection=resolved_projection,
                renderer_registry=renderer_registry,
                timezone=timezone,
            )
            if message is None:
                continue
            projected.append(message)
            compacted += int(detail == "compact")
        if not projected:
            continue
        units.append(
            TimelineReadUnit(
                unit_id=unit_id,
                sort_key=min(_entry_key(entry) for entry in ordered),
                entries=ordered,
                projected_messages=tuple(projected),
                compacted_entry_count=compacted,
            )
        )
    return sorted(units, key=lambda unit: (unit.sort_key, unit.unit_id))


def project_timeline_entry(
    entry: TimelineEntry,
    *,
    projection: str,
    renderer_registry: RendererRegistry,
    timezone: str,
    detail_override: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """Create a safe model-visible record without returning the raw stored payload."""

    resolved_projection = normalize_timeline_projection(projection)
    root = entry.kind.split(".", 1)[0].lower()
    is_operation = entry.turn_role in {TurnRole.ACTION, TurnRole.OBSERVATION} or root in _OPERATION_ROOTS
    is_tool = entry.turn_role in {TurnRole.ACTION, TurnRole.OBSERVATION} or root in _TOOLS_ROOTS
    if resolved_projection == "tools" and not is_tool:
        return None, ""
    detail = str(detail_override or "").strip().lower()
    if not detail:
        detail = "compact" if resolved_projection == "conversation" and is_operation else "full"
    if detail not in {"full", "compact"}:
        raise ValueError(f"invalid_entry_detail:{detail}")

    if detail == "full" and root == "message" and resolved_projection == "conversation":
        semantic_text, _ = sanitize_timeline_value(entry.semantic_text)
        extra_payload = dict(entry.payload)
        if str(extra_payload.get("text") or "") == entry.semantic_text:
            extra_payload.pop("text", None)
        extra_state: dict[str, Any] = {}
        if extra_payload:
            extra_state["payload"] = extra_payload
        if entry.trace_metadata:
            extra_state["trace"] = dict(entry.trace_metadata)
        safe_state, _ = sanitize_timeline_value(extra_state)
        content = str(semantic_text or "")
        if safe_state:
            content = f"{content}\nstate:\n{json.dumps(safe_state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        renderer_id = entry.renderer_id
        renderer_version = entry.renderer_version
    else:
        rendered = renderer_registry.render_detail(entry, timezone=timezone, detail=detail)
        content = rendered.text
        renderer_id = rendered.renderer_id
        renderer_version = rendered.renderer_version

    actor = entry.namespace.actor
    record = {
        "source_id": entry.source_id,
        "conversation_id": entry.namespace.conversation_id,
        "actor_id": actor.stable_id if actor else "",
        "actor_display_name": actor.display_name if actor else "",
        "target_actor_id": entry.target_actor.stable_id if entry.target_actor else "",
        "target_actor_display_name": entry.target_actor.display_name if entry.target_actor else "",
        "seq_no": entry.seq_no,
        "timestamp": entry.timestamp,
        "date_label": entry.date_label,
        "time_of_day": entry.time_of_day,
        "kind": entry.kind,
        "origin": entry.origin.value,
        "turn_role": entry.turn_role.value if entry.turn_role else "",
        "role": entry.compatibility_role,
        "turn_id": entry.turn_id,
        "reply_to_source_id": entry.reply_to_source_id,
        "correlation_id": entry.correlation_id,
        "relation_status": entry.relation_status,
        "detail": detail,
        "renderer_id": renderer_id,
        "renderer_version": renderer_version,
        "content": str(content or ""),
        "entry_type": "raw",
    }
    safe_record, _ = sanitize_timeline_value(record)
    return dict(safe_record), detail


def paginate_timeline_units(
    units: Sequence[TimelineReadUnit],
    *,
    page_token_budget: int,
    count_text: Callable[[str], int] | None,
    cursor_selector: Mapping[str, Any],
    projection: str,
    namespace: Namespace,
    after_key: StableKey | None = None,
    timezone: str,
) -> TimelinePage:
    remaining = [unit for unit in units if after_key is None or unit.sort_key > after_key]
    total_entry_count = sum(len(unit.projected_messages) for unit in remaining)
    budget = int(page_token_budget or 0)
    if budget < 0:
        raise ValueError("page_token_budget_must_be_non_negative_integer")
    if budget and count_text is None:
        raise ValueError("timeline_token_counter_required_for_page_budget")

    selected: list[TimelineReadUnit] = []
    oversized = False
    if not budget:
        selected = list(remaining)
    else:
        assert count_text is not None
        for unit in remaining:
            candidate_messages = [
                message for candidate in (*selected, unit) for message in candidate.projected_messages
            ]
            candidate_text = render_timeline(candidate_messages, tz=timezone)
            candidate_tokens = int(count_text(candidate_text))
            if candidate_tokens < 0:
                raise ValueError("TokenCounter.count_text() must return a non-negative int")
            if selected and candidate_tokens > budget:
                break
            if not selected and candidate_tokens > budget:
                selected.append(unit)
                oversized = True
                break
            selected.append(unit)

    complete = len(selected) == len(remaining)
    next_cursor = ""
    if selected and not complete:
        next_cursor = encode_timeline_cursor(
            selector=cursor_selector,
            projection=projection,
            page_token_budget=budget,
            last_unit_key=selected[-1].sort_key,
            namespace=namespace,
        )
    messages = tuple(message for unit in selected for message in unit.projected_messages)
    return TimelinePage(
        messages=messages,
        complete=complete,
        next_cursor=next_cursor,
        logical_unit_count=len(selected),
        total_logical_unit_count=len(remaining),
        entry_count=len(messages),
        total_entry_count=total_entry_count,
        compacted_entry_count=sum(unit.compacted_entry_count for unit in selected),
        oversized_unit=oversized,
    )


def encode_timeline_cursor(
    *,
    selector: Mapping[str, Any],
    projection: str,
    page_token_budget: int,
    last_unit_key: StableKey,
    namespace: Namespace,
) -> str:
    selector_payload = dict(selector)
    body = {
        "version": _CURSOR_VERSION,
        "scope_fingerprint": timeline_scope_fingerprint(namespace),
        "selector": selector_payload,
        "selector_fingerprint": stable_projection_hash(selector_payload),
        "projection": normalize_timeline_projection(projection),
        "direction": "forward",
        "page_token_budget": int(page_token_budget),
        "last_unit_key": list(last_unit_key),
    }
    encoded = _base64url(canonical_json_bytes(body))
    signature = _cursor_signature(canonical_json_bytes(body))
    return f"{_CURSOR_PREFIX}:{encoded}:{signature}"


def decode_timeline_cursor(cursor: str, *, namespace: Namespace) -> dict[str, Any]:
    token = str(cursor or "").strip()
    try:
        prefix, encoded, signature = token.split(":", 2)
        if prefix != _CURSOR_PREFIX:
            raise ValueError
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        if not hmac.compare_digest(signature, _cursor_signature(raw)):
            raise ValueError
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict) or canonical_json_bytes(body) != raw:
            raise ValueError
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_cursor") from exc

    try:
        version = int(body.get("version") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_cursor") from exc
    if version != _CURSOR_VERSION or body.get("direction") != "forward":
        raise ValueError("invalid_cursor")
    if str(body.get("scope_fingerprint") or "") != timeline_scope_fingerprint(namespace):
        raise ValueError("invalid_cursor_scope")
    selector = body.get("selector")
    if not isinstance(selector, dict) or str(body.get("selector_fingerprint") or "") != stable_projection_hash(
        selector
    ):
        raise ValueError("invalid_cursor")
    try:
        projection = normalize_timeline_projection(body.get("projection"))
    except ValueError as exc:
        raise ValueError("invalid_cursor") from exc
    try:
        budget = int(body.get("page_token_budget"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_cursor") from exc
    raw_key = body.get("last_unit_key")
    if not isinstance(raw_key, list) or len(raw_key) != 4:
        raise ValueError("invalid_cursor")
    try:
        key: StableKey = (int(raw_key[0]), str(raw_key[1]), int(raw_key[2]), str(raw_key[3]))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_cursor") from exc
    if budget <= 0 or not key[3]:
        raise ValueError("invalid_cursor")
    return {
        "selector": selector,
        "projection": projection,
        "page_token_budget": budget,
        "last_unit_key": key,
    }


def timeline_scope_fingerprint(namespace: Namespace) -> str:
    return stable_projection_hash(
        {
            "tenant_id": namespace.tenant_id,
            "user_id": namespace.user_id,
            "domain_id": namespace.domain_id,
            "conversation_id": namespace.conversation_id,
        }
    )


def timestamp_iso(value: int, *, timezone: str) -> str:
    if int(value or 0) <= 0:
        return ""
    return datetime.fromtimestamp(int(value), ZoneInfo(timezone)).isoformat(timespec="seconds")


def _entry_key(entry: TimelineEntry) -> StableKey:
    return (
        int(entry.timestamp),
        str(entry.namespace.conversation_id or ""),
        int(entry.seq_no),
        str(entry.source_id),
    )


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _cursor_signature(value: bytes) -> str:
    return hashlib.sha256(b"memcore.timeline.cursor.v1\0" + value).hexdigest()[:32]


__all__ = [
    "TIMELINE_PROJECTIONS",
    "TimelinePage",
    "TimelineReadUnit",
    "build_timeline_read_units",
    "decode_timeline_cursor",
    "encode_timeline_cursor",
    "normalize_timeline_projection",
    "paginate_timeline_units",
    "project_timeline_entry",
    "timestamp_iso",
    "timeline_scope_fingerprint",
]
