"""Explicit upgrade from legacy chat replay to semantic chat projection V5."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Sequence

from .namespace import Namespace
from .projection import PROJECTION_VERSION, ProjectionMessageInput
from .projection_migration import _settlement_plan_builder

CHAT_PROJECTION_KINDS = (
    "message.user",
    "message.user.observed",
    "message.user.voice",
    "message.assistant",
    "message.assistant.voice",
)


def _recover_legacy_emotion(entry: Any) -> Any:
    if str(getattr(entry, "kind", "") or "") not in {"message.assistant", "message.assistant.voice"}:
        return entry
    payload = dict(getattr(entry, "payload", {}) or {})
    if str(payload.get("emotion") or "").strip():
        return entry
    raw = str(payload.get("provider_output_raw") or "").strip()
    if not raw.startswith("{"):
        return entry
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return entry
    if not isinstance(parsed, dict):
        return entry
    emotion = str(parsed.get("emotion") or "").strip()
    if not emotion:
        return entry
    payload["emotion"] = emotion
    return replace(entry, payload=payload)


def _projection_builder(adapter: Any) -> Callable[[str, Sequence[Any]], list[ProjectionMessageInput]]:
    def build(provider_profile: str, entries: Sequence[Any]) -> list[ProjectionMessageInput]:
        recovered = [_recover_legacy_emotion(entry) for entry in entries]
        return list(
            adapter.project_entries(
                recovered,
                provider_profile=provider_profile,
                start_index=0,
            )
        )

    return build


def migrate_chat_projections_v5(
    *,
    store: Any,
    adapter: Any,
    namespace: Namespace,
    dry_run: bool = False,
    count_text: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Upgrade frozen chat rows once; request-time reads never call this path."""

    migrate = getattr(store, "migrate_chat_projections", None)
    if not callable(migrate):
        return _unsupported_report(dry_run)
    try:
        return migrate(
            namespace=namespace,
            target_version=PROJECTION_VERSION,
            projection_builder=_projection_builder(adapter),
            settlement_builder=_settlement_plan_builder(adapter, count_text=count_text),
            chat_kinds=CHAT_PROJECTION_KINDS,
            dry_run=bool(dry_run),
        )
    except NotImplementedError:
        return _unsupported_report(dry_run)


def _unsupported_report(dry_run: bool) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "target_projection_version": PROJECTION_VERSION,
        "scanned_projection_rows": 0,
        "affected_chat_rows": 0,
        "migrated": 0,
        "version_advanced_only": 0,
        "dry_run": bool(dry_run),
        "reason": "store_migration_unsupported",
    }
