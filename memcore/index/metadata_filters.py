"""Helpers for index metadata fields used by pre-retrieval filtering."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_]+$")


def metadata_filter_key(prefix: str, value: str) -> str:
    """Return a stable scalar metadata field name for a configured enum value."""
    safe_prefix = _safe_part(prefix)
    raw = str(value or "").strip()
    if not raw:
        return f"{safe_prefix}__empty"
    if _SAFE_KEY.fullmatch(raw):
        return f"{safe_prefix}__{raw}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{safe_prefix}__h_{digest}"


def category_filter_key(category: str) -> str:
    return metadata_filter_key("memory_category", category)


def subject_scope_filter_key(scope: str) -> str:
    return metadata_filter_key("memory_scope", scope)


def metadata_filter_flags(metadata: dict[str, Any]) -> dict[str, bool]:
    """Build boolean index metadata flags for structured category/scope prefilters."""
    flags: dict[str, bool] = {}
    for category in _string_items(metadata.get("categories")):
        flags[category_filter_key(category)] = True
    for scope in _string_items(metadata.get("subject_scopes")):
        flags[subject_scope_filter_key(scope)] = True
    return flags


def _safe_part(value: str) -> str:
    raw = str(value or "").strip()
    if _SAFE_KEY.fullmatch(raw):
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"h_{digest}"


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
