"""Deterministic memory-catalog navigation and namespace-bound cursors.

The catalog is a projection over SQLite truth.  Cursors freeze the selector
and page size so a continuation cannot silently broaden scope or change view.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .namespace import Namespace
from .projection import canonical_json_bytes, stable_projection_hash

MEMORY_NODE_TYPES: tuple[str, ...] = ("episodic", "semantic")
MEMORY_VIEWS: tuple[str, ...] = ("card", "content", "sources")
MEMORY_DETAILS: tuple[str, ...] = ("full", "compact")
DEFAULT_MEMORY_PAGE_SIZE = 50
MAX_MEMORY_PAGE_SIZE = 200

_CURSOR_PREFIX = "memory-v1"
_CURSOR_VERSION = 1


@dataclass(frozen=True)
class MemoryPage:
    items: tuple[Any, ...]
    complete: bool
    next_cursor: str
    returned_count: int
    remaining_count: int


def normalize_page_size(value: Any) -> int:
    if value is None or value == "":
        return DEFAULT_MEMORY_PAGE_SIZE
    if isinstance(value, bool):
        raise ValueError("page_size_must_be_positive_integer")
    try:
        page_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("page_size_must_be_positive_integer") from exc
    if page_size <= 0 or page_size > MAX_MEMORY_PAGE_SIZE:
        raise ValueError(f"page_size_must_be_between_1_and_{MAX_MEMORY_PAGE_SIZE}")
    return page_size


def paginate_memory_items(
    items: Sequence[Any],
    *,
    item_keys: Sequence[Sequence[Any]],
    namespace: Namespace,
    mode: str,
    selector: Mapping[str, Any],
    page_size: int,
    after_key: tuple[Any, ...] | None = None,
) -> MemoryPage:
    resolved_size = normalize_page_size(page_size)
    if len(items) != len(item_keys):
        raise ValueError("memory_page_keys_mismatch")
    keyed = [(tuple(key), item) for key, item in zip(item_keys, items, strict=True)]
    if any(not key for key, _item in keyed):
        raise ValueError("memory_page_key_required")
    remaining = [(key, item) for key, item in keyed if after_key is None or key > after_key]
    selected_pairs = remaining[:resolved_size]
    selected = tuple(item for _key, item in selected_pairs)
    complete = len(selected_pairs) >= len(remaining)
    next_cursor = ""
    if selected and not complete:
        next_cursor = encode_memory_cursor(
            namespace=namespace,
            mode=mode,
            selector=selector,
            page_size=resolved_size,
            last_key=selected_pairs[-1][0],
        )
    return MemoryPage(
        items=selected,
        complete=complete,
        next_cursor=next_cursor,
        returned_count=len(selected),
        remaining_count=max(0, len(remaining) - len(selected_pairs)),
    )


def encode_memory_cursor(
    *,
    namespace: Namespace,
    mode: str,
    selector: Mapping[str, Any],
    page_size: int,
    last_key: Sequence[Any],
) -> str:
    selector_payload = dict(selector)
    body = {
        "version": _CURSOR_VERSION,
        "scope_fingerprint": memory_scope_fingerprint(namespace),
        "mode": str(mode),
        "selector": selector_payload,
        "selector_fingerprint": stable_projection_hash(selector_payload),
        "page_size": normalize_page_size(page_size),
        "last_key": list(last_key),
    }
    raw = canonical_json_bytes(body)
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = _cursor_signature(raw)
    return f"{_CURSOR_PREFIX}:{encoded}:{signature}"


def decode_memory_cursor(cursor: str, *, namespace: Namespace, expected_mode: str) -> dict[str, Any]:
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

    if int(body.get("version") or 0) != _CURSOR_VERSION:
        raise ValueError("invalid_cursor")
    if str(body.get("scope_fingerprint") or "") != memory_scope_fingerprint(namespace):
        raise ValueError("invalid_cursor_scope")
    if str(body.get("mode") or "") != str(expected_mode):
        raise ValueError("invalid_cursor_mode")
    selector = body.get("selector")
    if not isinstance(selector, dict) or str(body.get("selector_fingerprint") or "") != stable_projection_hash(
        selector
    ):
        raise ValueError("invalid_cursor")
    try:
        page_size = normalize_page_size(body.get("page_size"))
        last_key = body.get("last_key")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_cursor") from exc
    if not isinstance(last_key, list) or not last_key:
        raise ValueError("invalid_cursor")
    return {"selector": selector, "page_size": page_size, "last_key": tuple(last_key)}


def merge_intervals(intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    """Merge half-open timestamp intervals while ignoring malformed empty rows."""

    normalized: list[tuple[int, int]] = []
    for interval in intervals:
        try:
            start = int(interval.get("start_ts") or 0)
            end = int(interval.get("end_ts") or 0)
        except (TypeError, ValueError):
            continue
        if start > 0 and end > start:
            normalized.append((start, end))
    normalized.sort()
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [{"start_ts": start, "end_ts": end} for start, end in merged]


def memory_scope_fingerprint(namespace: Namespace) -> str:
    return stable_projection_hash(
        {
            "tenant_id": namespace.tenant_id,
            "user_id": namespace.user_id,
            "domain_id": namespace.domain_id,
            "conversation_id": namespace.conversation_id,
        }
    )


def _cursor_signature(value: bytes) -> str:
    return hashlib.sha256(b"memcore.memory.cursor.v1\0" + value).hexdigest()[:32]


__all__ = [
    "DEFAULT_MEMORY_PAGE_SIZE",
    "MAX_MEMORY_PAGE_SIZE",
    "MEMORY_DETAILS",
    "MEMORY_NODE_TYPES",
    "MEMORY_VIEWS",
    "MemoryPage",
    "decode_memory_cursor",
    "merge_intervals",
    "normalize_page_size",
    "paginate_memory_items",
]
