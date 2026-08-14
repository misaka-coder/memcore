"""Explicit, idempotent migration of legacy path-omission projections.

Projection semantics V3 stop rewriting executable path evidence (tool
arguments, command output, operation results) with ``[local path omitted
from persistent history]`` markers.  Rows frozen under older semantics are
never silently rewritten during reads: hosts run
:func:`migrate_legacy_path_projections` at a controlled point and receive a
structured report.  Only marker-containing rows whose raw timeline sources
still exist are re-projected; everything else is preserved byte-for-byte and
reported.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .namespace import Namespace
from .projection import (
    LEGACY_PATH_OMISSION_MARKER,
    PROJECTION_VERSION,
    ProjectionMessageInput,
)

LEGACY_PATH_PROJECTION_TARGET_VERSION = PROJECTION_VERSION

ProjectionBuilder = Callable[[str, Sequence[Any]], Sequence[ProjectionMessageInput]]


def _adapter_projection_builder(adapter: Any) -> ProjectionBuilder:
    def build(provider_profile: str, entries: Sequence[Any]) -> list[ProjectionMessageInput]:
        return list(
            adapter.project_entries(
                entries,
                provider_profile=provider_profile,
                start_index=0,
            )
        )

    return build


def migrate_legacy_path_projections(
    *,
    store: Any,
    adapter: Any,
    namespace: Namespace,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-project legacy marker rows from raw sources and drop stale settled copies.

    Returns a structured report without any payload text.  Idempotent: rows
    migrated in a previous run carry the target projection version and no
    longer contain the legacy marker, so a second run leaves them untouched.
    """

    migrate = getattr(store, "migrate_legacy_path_projections", None)
    if not callable(migrate):
        return {
            "status": "unsupported",
            "target_projection_version": LEGACY_PATH_PROJECTION_TARGET_VERSION,
            "migrated": 0,
            "version_advanced_only": 0,
            "preserved_without_raw_source": 0,
            "preserved_shape_mismatch": 0,
            "preserved_reprojection_failed": 0,
            "settled_rebuilt": 0,
            "dry_run": bool(dry_run),
            "reason": "store_migration_unsupported",
        }
    try:
        return migrate(
            namespace=namespace,
            target_version=LEGACY_PATH_PROJECTION_TARGET_VERSION,
            projection_builder=_adapter_projection_builder(adapter),
            dry_run=bool(dry_run),
        )
    except NotImplementedError:
        return {
            "status": "unsupported",
            "target_projection_version": LEGACY_PATH_PROJECTION_TARGET_VERSION,
            "migrated": 0,
            "version_advanced_only": 0,
            "preserved_without_raw_source": 0,
            "preserved_shape_mismatch": 0,
            "preserved_reprojection_failed": 0,
            "settled_rebuilt": 0,
            "dry_run": bool(dry_run),
            "reason": "store_migration_unsupported",
        }


def migration_report_summary(report: Mapping[str, Any]) -> str:
    """Compact, non-sensitive one-line summary for host logs."""

    return (
        f"legacy_path_projection_migration status={str(report.get('status') or 'ok')} "
        f"target={report.get('target_projection_version')} migrated={report.get('migrated') or 0} "
        f"version_only={report.get('version_advanced_only') or 0} "
        f"preserved_no_raw={report.get('preserved_without_raw_source') or 0} "
        f"preserved_shape={report.get('preserved_shape_mismatch') or 0} "
        f"preserved_failed={report.get('preserved_reprojection_failed') or 0} "
        f"settled_rebuilt={report.get('settled_rebuilt') or 0} dry_run={bool(report.get('dry_run'))}"
    )
