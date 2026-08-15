"""Explicit, idempotent migration of legacy path-omission projections.

Projection semantics V3 stop rewriting executable path evidence (tool
arguments, command output, operation results) with ``[local path omitted
from persistent history]`` markers.  Rows frozen under older semantics are
never silently rewritten during reads: hosts run
:func:`migrate_legacy_path_projections` at an explicit pre-traffic
maintenance point and receive a structured report.

Three legacy damage patterns are scanned and reported separately:

- MemCore V2 omission marker (recoverable when raw sources are intact);
- Akane host redaction ``[local_path]`` (raw already damaged, irrecoverable);
- ``$TMPDIR`` alias (raw already damaged, irrecoverable).

Only records whose raw timeline sources still contain the original values
are re-projected.  Irrecoverable rows are preserved untouched and counted,
never guessed at.  Stale settlements (marker-containing or full-hash
mismatched) are rebuilt atomically through the regular settlement planner;
only cards that are truly re-committed count as rebuilt.
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

LEGACY_MEMCORE_OMISSION_MARKER = LEGACY_PATH_OMISSION_MARKER
LEGACY_HOST_LOCAL_PATH_MARKER = "[local_path]"
LEGACY_HOST_TMPDIR_ALIAS = "$TMPDIR"

LEGACY_PROJECTION_MARKERS: tuple[str, str, str] = (
    LEGACY_MEMCORE_OMISSION_MARKER,
    LEGACY_HOST_LOCAL_PATH_MARKER,
    LEGACY_HOST_TMPDIR_ALIAS,
)

# Markers whose presence in a projection payload proves the host already
# redacted the raw evidence before MemCore persisted it.
_HOST_DAMAGE_MARKERS = (LEGACY_HOST_LOCAL_PATH_MARKER, LEGACY_HOST_TMPDIR_ALIAS)

ProjectionBuilder = Callable[[str, Sequence[Any]], Sequence[ProjectionMessageInput]]
SettlementBuilder = Callable[..., Any]

_REPORT_TOTAL_KEYS = (
    "scanned_memcore_marker_rows",
    "scanned_host_local_path_rows",
    "scanned_host_tmpdir_rows",
    "migrated",
    "version_advanced_only",
    "preserved_irrecoverable_host_redaction",
    "preserved_without_raw_source",
    "preserved_shape_mismatch",
    "preserved_reprojection_failed",
    "settled_rebuilt",
    "settled_rebuilt_noop",
    "settled_rebuilt_fallback",
    "settled_rebuild_failed_dropped",
)


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


def _settlement_plan_builder(adapter: Any, *, count_text: Callable[[str], int] | None = None) -> SettlementBuilder:
    from functools import partial

    from .settlement import build_settlement_plan, classify_observation, settlement_config_hash

    def build(
        namespace: Any,
        turn_id: str,
        provider_profile: str,
        entries: Sequence[Any],
        full_messages: Sequence[Any],
        *,
        policy: str = "compact_after_terminal",
        min_inline_bytes: int = 256,
        required_savings_ratio: float = 0.5,
    ) -> Any:
        """Return a SettledProjectionPlan, or None when planning fails structurally.

        重建使用 turn 冻结的阈值, 不读运行中配置; 生成的 plan 携带同一
        配置指纹, 与已持久化 settlement 行不一致时由 store 结构化拒绝。
        """
        try:
            decider = partial(
                classify_observation,
                min_inline_bytes=int(min_inline_bytes),
                required_savings_ratio=float(required_savings_ratio),
                timezone=str(getattr(adapter, "timezone", "") or "").strip(),
            )
            return build_settlement_plan(
                adapter,
                entries,
                provider_profile=provider_profile,
                authoritative_messages=full_messages,
                count_text=count_text,
                observation_decider=decider,
                settlement_min_utf8_bytes=int(min_inline_bytes),
                settlement_min_saved_ratio=float(required_savings_ratio),
                settlement_config_hash=settlement_config_hash(
                    policy=policy,
                    min_utf8_bytes=int(min_inline_bytes),
                    saved_ratio=float(required_savings_ratio),
                ),
            )
        except Exception:
            return None

    return build


def migrate_legacy_path_projections(
    *,
    store: Any,
    adapter: Any,
    namespace: Namespace,
    dry_run: bool = False,
    count_text: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Re-project legacy marker rows from raw sources and rebuild stale settlements.

    Returns a structured report without any payload text.  Idempotent: rows
    migrated in a previous run carry the target projection version and no
    longer contain legacy markers, so a second run leaves them untouched.
    """

    migrate = getattr(store, "migrate_legacy_path_projections", None)
    if not callable(migrate):
        return _unsupported_report(dry_run)
    try:
        return migrate(
            namespace=namespace,
            target_version=LEGACY_PATH_PROJECTION_TARGET_VERSION,
            projection_builder=_adapter_projection_builder(adapter),
            settlement_builder=_settlement_plan_builder(adapter, count_text=count_text),
            markers=LEGACY_PROJECTION_MARKERS,
            dry_run=bool(dry_run),
        )
    except NotImplementedError:
        return _unsupported_report(dry_run)


def _unsupported_report(dry_run: bool) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "target_projection_version": LEGACY_PATH_PROJECTION_TARGET_VERSION,
        **{key: 0 for key in _REPORT_TOTAL_KEYS},
        "dry_run": bool(dry_run),
        "reason": "store_migration_unsupported",
    }


def migration_report_summary(report: Mapping[str, Any]) -> str:
    """Compact, non-sensitive one-line summary for host logs."""

    parts = [f"legacy_path_projection_migration status={str(report.get('status') or 'ok')}"]
    parts.append(f"target={report.get('target_projection_version')}")
    for key in _REPORT_TOTAL_KEYS:
        parts.append(f"{key}={report.get(key) or 0}")
    parts.append(f"dry_run={bool(report.get('dry_run'))}")
    return " ".join(parts)
