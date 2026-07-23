"""Helpers for index metadata fields used by pre-retrieval filtering."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_]+$")
_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")
_KIND_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*$")

INDEX_SCHEMA_VERSION = 3
KIND_FLAG_SCHEMA_VERSION = 1
VISIBILITY_SCHEMA_VERSION = 1
ENTITY_FLAG_SCHEMA_VERSION = 1
INDEX_SCHEMA_KEY = (
    f"index_v{INDEX_SCHEMA_VERSION}:kind_v{KIND_FLAG_SCHEMA_VERSION}:"
    f"visibility_v{VISIBILITY_SCHEMA_VERSION}:entity_v{ENTITY_FLAG_SCHEMA_VERSION}"
)


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


def facet_filter_key(facet: str) -> str:
    return metadata_filter_key("memory_facet", facet)


def about_role_filter_key(role: str) -> str:
    return metadata_filter_key("memory_about_role", role)


def normalize_entity_anchor(value: Any) -> str:
    """Stable entity normalization without guessing aliases or stripping meaningful symbols."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.strip().casefold().split())


def entity_filter_key(entity: str) -> str:
    normalized = normalize_entity_anchor(entity)
    if not normalized:
        raise ValueError("empty_entity_anchor")
    digest = hashlib.sha256(f"entity_flag_v{ENTITY_FLAG_SCHEMA_VERSION}:{normalized}".encode("utf-8")).hexdigest()
    return f"memory_entity__v{ENTITY_FLAG_SCHEMA_VERSION}_{digest}"


def kind_prefixes(kind: str) -> tuple[str, ...]:
    """Return every dot-delimited prefix for one validated open kind."""
    normalized = str(kind or "").strip().lower()
    if not _KIND_PATTERN.fullmatch(normalized):
        raise ValueError("invalid_kind")
    parts = normalized.split(".")
    return tuple(".".join(parts[:index]) for index in range(1, len(parts) + 1))


def kind_filter_key(kind_prefix: str) -> str:
    """Use a versioned full digest so arbitrary future kinds remain safe Chroma keys."""
    normalized = str(kind_prefix or "").strip().lower()
    if not normalized or not all(_SAFE_KEY.fullmatch(part) for part in normalized.split(".")):
        raise ValueError("invalid_kind_prefix")
    digest = hashlib.sha256(f"kind_flag_v{KIND_FLAG_SCHEMA_VERSION}:{normalized}".encode("utf-8")).hexdigest()
    return f"memory_kind__v{KIND_FLAG_SCHEMA_VERSION}_{digest}"


def kind_filter_flags(kind: str) -> dict[str, bool]:
    return {kind_filter_key(prefix): True for prefix in kind_prefixes(kind)}


def normalize_kind_pattern(pattern: str) -> tuple[str, bool]:
    """Accept exact kinds or one trailing ``.*`` prefix; reject regex/mid-glob forms."""
    normalized = str(pattern or "").strip().lower()
    is_prefix = normalized.endswith(".*")
    kind = normalized[:-2] if is_prefix else normalized
    validator = _KIND_PREFIX_PATTERN if is_prefix else _KIND_PATTERN
    if not validator.fullmatch(kind):
        raise ValueError("invalid_kind_pattern")
    return kind, is_prefix


def metadata_filter_flags(metadata: dict[str, Any]) -> dict[str, bool]:
    """Build boolean index metadata flags for facet/role/entity prefilters."""
    flags: dict[str, bool] = {}
    for facet in _string_items(metadata.get("memory_facets")):
        flags[facet_filter_key(facet)] = True
    for role in _string_items(metadata.get("about_roles")):
        flags[about_role_filter_key(role)] = True
    for entity in _string_items(metadata.get("entity_anchors")):
        flags[entity_filter_key(entity)] = True
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
