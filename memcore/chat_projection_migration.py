"""Explicit upgrade to the V6 chat authorship projection."""

from __future__ import annotations

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


def _projection_builder(adapter: Any) -> Callable[[str, Sequence[Any]], list[ProjectionMessageInput]]:
    def build(provider_profile: str, entries: Sequence[Any]) -> list[ProjectionMessageInput]:
        return list(adapter.project_entries(entries, provider_profile=provider_profile, start_index=0))

    return build


def migrate_chat_projections_v6(
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
