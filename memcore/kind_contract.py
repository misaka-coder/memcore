"""Shared grammar for typed timeline kinds and kind-prefix filters."""

from __future__ import annotations

import re
from typing import Any

MAX_KIND_LENGTH = 160
KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")
KIND_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*$")


def normalize_kind(value: Any) -> str:
    return str(value or "").strip().lower()


def is_valid_kind(value: Any) -> bool:
    normalized = normalize_kind(value)
    return len(normalized) <= MAX_KIND_LENGTH and KIND_PATTERN.fullmatch(normalized) is not None


def is_valid_kind_prefix(value: Any) -> bool:
    normalized = normalize_kind(value)
    return len(normalized) <= MAX_KIND_LENGTH and KIND_PREFIX_PATTERN.fullmatch(normalized) is not None
