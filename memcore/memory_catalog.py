"""Compact navigation projections over MemCore's existing memory truth layers.

Catalog cards are derived views.  They never replace summary content or raw
lineage, and fallback text must remain deterministic so an old database is
immediately browsable without an LLM backfill pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .text_utils import normalize_text

CATALOG_SCHEMA_VERSION = 1
_MAX_TITLE_CHARS = 160
_MAX_HINT_CHARS = 500
_MAX_HEADING_CHARS = 120
_MAX_HEADINGS = 12


def normalize_catalog_fields(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize optional model-authored catalog fields without rejecting a valid summary."""

    source = payload if isinstance(payload, Mapping) else {}
    title = _compact_text(source.get("memory_title"), limit=_MAX_TITLE_CHARS)
    hint = _compact_text(source.get("catalog_hint"), limit=_MAX_HINT_CHARS)
    headings = [
        _compact_text(item, limit=_MAX_HEADING_CHARS)
        for item in _unique_strings(source.get("topic_headings"))[:_MAX_HEADINGS]
    ]
    return {
        "memory_title": title,
        "catalog_hint": hint,
        "topic_headings": headings,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION if title and hint else 0,
    }


def normalize_participant_refs(value: Any) -> list[dict[str, str]]:
    """Keep stable actor IDs with their latest non-empty display name."""

    if not isinstance(value, (list, tuple)):
        return []
    ordered_ids: list[str] = []
    names: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        actor_id = normalize_text(item.get("actor_id") or item.get("stable_id"))
        if not actor_id:
            continue
        if actor_id not in names:
            ordered_ids.append(actor_id)
            names[actor_id] = ""
        display_name = normalize_text(item.get("display_name") or item.get("actor_display_name"))
        if display_name:
            names[actor_id] = display_name
    return [{"actor_id": actor_id, "display_name": names[actor_id]} for actor_id in ordered_ids]


def build_memory_card(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one raw/episodic/semantic record into a compact navigation card."""

    node_type, memory_id = _node_identity(record)
    stored_title = _compact_text(record.get("memory_title"), limit=_MAX_TITLE_CHARS)
    stored_hint = _compact_text(record.get("catalog_hint"), limit=_MAX_HINT_CHARS)
    generated = bool(
        int(record.get("catalog_schema_version") or 0) >= CATALOG_SCHEMA_VERSION and stored_title and stored_hint
    )
    title = stored_title or _fallback_title(record, node_type=node_type)
    hint = stored_hint or _fallback_hint(record, title=title, node_type=node_type)
    source_ids = _source_ids(record, node_type=node_type)
    source_entry_count = _non_negative_int(record.get("source_entry_count"))
    if source_entry_count == 0 and node_type in {"episodic", "semantic"}:
        source_entry_count = len(source_ids)
    source_turn_count = _non_negative_int(record.get("source_turn_count"))
    return {
        "memory_id": memory_id,
        "node_type": node_type,
        "memory_title": title,
        "catalog_hint": hint,
        "topic_headings": [
            _compact_text(item, limit=_MAX_HEADING_CHARS)
            for item in _unique_strings(record.get("topic_headings"))[:_MAX_HEADINGS]
        ],
        "period_start_ts": _effective_timestamp(record.get("period_start_ts"), record.get("timestamp")),
        "period_end_ts": _effective_timestamp(record.get("period_end_ts"), record.get("timestamp")),
        "participant_refs": normalize_participant_refs(record.get("participant_refs")),
        "source_turn_count": source_turn_count,
        "source_entry_count": source_entry_count,
        "importance": _clamp01(record.get("importance")),
        "catalog_quality": "generated" if generated else "fallback",
        "has_content": _has_content(record, node_type=node_type),
        "has_sources": bool(source_ids),
    }


def collect_participant_refs(entries: list[Any]) -> list[dict[str, str]]:
    """Derive participants from generic source/target actors in frozen timeline entries."""

    refs: list[dict[str, str]] = []
    for entry in entries:
        namespace = getattr(entry, "namespace", None)
        actor = getattr(namespace, "actor", None)
        target = getattr(entry, "target_actor", None)
        for candidate in (actor, target):
            if candidate is None:
                continue
            refs.append(
                {
                    "actor_id": normalize_text(getattr(candidate, "stable_id", "")),
                    "display_name": normalize_text(getattr(candidate, "display_name", "")),
                }
            )
    return normalize_participant_refs(refs)


def count_logical_turns(entries: list[Any]) -> int:
    """Count complete source units without treating parallel entries as separate turns."""

    keys: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = normalize_text(getattr(entry, "turn_id", "")) or normalize_text(getattr(entry, "source_id", ""))
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return len(keys)


def _node_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    entry_type = normalize_text(record.get("entry_type")).lower()
    if entry_type == "summary" or (not entry_type and normalize_text(record.get("summary_id"))):
        return ("episodic", normalize_text(record.get("summary_id")))
    if entry_type == "semantic_summary" or (not entry_type and normalize_text(record.get("semantic_id"))):
        return ("semantic", normalize_text(record.get("semantic_id")))
    return ("raw", normalize_text(record.get("source_id")))


def _source_ids(record: Mapping[str, Any], *, node_type: str) -> list[str]:
    if node_type == "episodic":
        return _unique_strings(record.get("source_ids"))
    if node_type == "semantic":
        return _unique_strings(record.get("source_summary_ids"))
    return []


def _fallback_title(record: Mapping[str, Any], *, node_type: str) -> str:
    candidates: list[Any]
    if node_type == "semantic":
        candidates = [
            *_unique_strings(record.get("recurring_topics")),
            *_unique_strings(record.get("stable_facts")),
            record.get("semantic_summary"),
        ]
    elif node_type == "episodic":
        candidates = [
            record.get("period_label"),
            *_unique_strings(record.get("key_events")),
            record.get("diary_summary"),
        ]
    else:
        candidates = [record.get("semantic_text"), record.get("content"), record.get("kind")]
    for value in candidates:
        text = normalize_text(value)
        if text:
            return _compact_text(text, limit=_MAX_TITLE_CHARS)
    date_label = normalize_text(record.get("date_label"))
    if date_label:
        return f"{date_label} 的记忆"
    return {"episodic": "阶段记忆", "semantic": "长期记忆", "raw": "原始记录"}[node_type]


def _fallback_hint(record: Mapping[str, Any], *, title: str, node_type: str) -> str:
    candidates: list[Any]
    if node_type == "semantic":
        candidates = [
            record.get("semantic_summary"),
            *_unique_strings(record.get("stable_facts")),
            *_unique_strings(record.get("recurring_topics")),
        ]
    elif node_type == "episodic":
        candidates = [
            *_unique_strings(record.get("core_facts")),
            *_unique_strings(record.get("key_events")),
            record.get("diary_summary"),
        ]
    else:
        candidates = [record.get("semantic_text"), record.get("content")]
    for value in candidates:
        text = normalize_text(value)
        if text:
            return _compact_text(text, limit=_MAX_HINT_CHARS)
    return _compact_text(title, limit=_MAX_HINT_CHARS)


def _has_content(record: Mapping[str, Any], *, node_type: str) -> bool:
    if node_type == "semantic":
        return bool(
            normalize_text(record.get("semantic_summary"))
            or _unique_strings(record.get("stable_facts"))
            or _unique_strings(record.get("recurring_topics"))
            or _unique_strings(record.get("important_people"))
            or _unique_strings(record.get("open_loops"))
        )
    if node_type == "episodic":
        return bool(
            normalize_text(record.get("diary_summary"))
            or _unique_strings(record.get("key_events"))
            or _unique_strings(record.get("core_facts"))
        )
    return bool(normalize_text(record.get("semantic_text") or record.get("content")))


def _effective_timestamp(value: Any, fallback: Any) -> int:
    primary = _non_negative_int(value)
    return primary or _non_negative_int(fallback)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _compact_text(value: Any, *, limit: int) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "build_memory_card",
    "collect_participant_refs",
    "count_logical_turns",
    "normalize_catalog_fields",
    "normalize_participant_refs",
]
