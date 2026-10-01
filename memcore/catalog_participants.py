"""Exact participant selectors over system-recorded Actor identities."""

from __future__ import annotations

import unicodedata
from typing import Any

from .memory_catalog import normalize_participant_refs


def normalize_participant_selectors(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("participants_must_be_array")
    selectors: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("participants_must_contain_strings")
        text = item.strip()
        if not text or any(ord(char) < 32 for char in text):
            raise ValueError("invalid_participant_selector")
        if text not in selectors:
            selectors.append(text)
    return selectors


def participants_match(refs: Any, selectors: list[str], *, match: str) -> bool:
    """IDs stay case-sensitive; names use normalized equality, never contains."""
    participants = normalize_participant_refs(refs)
    ids = {ref["actor_id"] for ref in participants}
    names = {_name_key(ref["display_name"]) for ref in participants if ref["display_name"]}
    matches = [selector in ids or (":" not in selector and _name_key(selector) in names) for selector in selectors]
    return all(matches) if match == "all" else any(matches)


def _name_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
