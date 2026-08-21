"""SQLiteMemoryStore —— MemoryStore 的默认实现(关系型真相源)。

三张记忆真相表:messages / summaries / semantic_summaries；
Timeline V2 另有 turn / projection / conversation coordination 表，不建立第二套 raw 真相源。

- 隔离采用 Namespace 五层(tenant/user/domain 硬隔离 + conversation 窗口 + actor 软标签)。
- index_status outbox 状态机(pending → indexed),向量 upsert 失败保持 pending,由 reindex 补做。
- 写入用单库事务保证原子;向量 upsert 由上层在事务外做。

时间字段(date_label / time_of_day)由调用方按 tz 算好后传入;store 不做时区换算。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from ..timeline import TurnProjectionSettlement

from ..errors import NamespaceError, SchemaError
from ..compaction_v2 import (
    CompactionSnapshot,
    SemanticBatchCommitResult,
    SemanticCommitInput,
    SemanticSnapshot,
    SummaryBatchCommitResult,
    SummaryRecordInput,
    TurnBundle,
)
from ..namespace import Namespace
from ..projection import (
    ProjectionAudit,
    ProjectionAuditInput,
    ProjectionMessage,
    ProjectionMessageInput,
    ProjectionStatus,
    RequestProjectionResult,
    canonical_json_bytes,
    merge_projection_status,
    stable_projection_hash,
)
from ..timeline import (
    AnnotationStatus,
    CompletionCommitResult,
    EntryOrigin,
    MemoryAnnotation,
    RetrievalPolicy,
    RetrievalVisibility,
    TimelineEntry,
    TimelineEntryInput,
    TurnAbortResult,
    TurnCompletion,
    TurnHandle,
    TurnRole,
    TurnStatus,
    resolve_retrieval_visibility,
)
from .base import LineageClosure, MemoryStore, RawTurnWindow
from .migrations import migrate_database, project_legacy_role

_JSON_FIELDS = {
    "messages": {
        "memory_metadata_json": "memory_metadata",
        "payload_json": "payload",
        "trace_metadata_json": "trace_metadata",
    },
    "summaries": {
        "key_events_json": "key_events",
        "core_facts_json": "core_facts",
        "topic_headings_json": "topic_headings",
        "participant_refs_json": "participant_refs",
        "semantic_tags_json": "semantic_tags",
        "memory_metadata_json": "memory_metadata",
        "source_ids_json": "source_ids",
        "trace_metadata_json": "trace_metadata",
    },
    "semantic_summaries": {
        "stable_facts_json": "stable_facts",
        "recurring_topics_json": "recurring_topics",
        "important_people_json": "important_people",
        "open_loops_json": "open_loops",
        "topic_headings_json": "topic_headings",
        "semantic_tags_json": "semantic_tags",
        "memory_metadata_json": "memory_metadata",
        "source_summary_ids_json": "source_summary_ids",
    },
    "turns": {
        "stimulus_source_ids_json": "stimulus_source_ids",
        "annotation_target_ids_json": "annotation_target_ids",
    },
    "prompt_projections": {
        "payload_json": "payload",
        "source_ids_json": "source_ids",
    },
}
_ENTRY_TYPE = {"messages": "raw", "summaries": "summary", "semantic_summaries": "semantic_summary"}
_JSON_OBJECT_FIELDS = frozenset({"memory_metadata", "payload", "trace_metadata"})


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _has_semantic_metadata(value: Any) -> bool:
    metadata = value if isinstance(value, dict) else {}
    for key in ("memory_facets", "about_roles", "entity_anchors", "topic_terms", "mood_tags"):
        if list(metadata.get(key) or []):
            return True
    return str(metadata.get("retrieval_priority") or "normal").strip().lower() in {"high", "critical"}


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = ":memory:") -> None:
        path_text = str(db_path or ":memory:")
        self._runtime_identity = (
            f"sqlite-memory:{uuid.uuid4().hex}"
            if path_text == ":memory:"
            else "sqlite-file:" + hashlib.sha256(os.path.normcase(os.path.abspath(path_text)).encode()).hexdigest()
        )
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        try:
            migrate_database(self._conn)
        except Exception:
            self._conn.close()
            raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row is not None else 0

    def runtime_identity(self) -> str:
        return self._runtime_identity

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        """Serialize snapshot validation before writes across SQLite connections."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    # --- 内部工具 ---

    @staticmethod
    def _row_to_record(row: sqlite3.Row, table: str) -> dict[str, Any]:
        record = dict(row)
        for json_col, key in _JSON_FIELDS.get(table, {}).items():
            raw = record.pop(json_col, None)
            try:
                record[key] = json.loads(raw) if raw else ({} if key in _JSON_OBJECT_FIELDS else [])
            except (TypeError, ValueError):
                record[key] = {} if key in _JSON_OBJECT_FIELDS else []
        if table in _ENTRY_TYPE:
            record["entry_type"] = _ENTRY_TYPE[table]
        return record

    @staticmethod
    def _scope_clause(namespace: Namespace, *, with_conversation: bool = False) -> tuple[str, list[Any]]:
        clause = "tenant_id = ? AND user_id = ? AND domain_id = ?"
        params: list[Any] = [namespace.tenant_id or "", namespace.user_id, namespace.domain_id or ""]
        if with_conversation:
            clause += " AND conversation_id = ?"
            params.append(namespace.conversation_id or "")
        return clause, params

    @staticmethod
    def _assert_namespace_owner(row: sqlite3.Row, namespace: Namespace, *, id_label: str) -> None:
        """既有记录若属于另一 owner,拒绝返回/覆盖 —— 防 source_id 跨会话/跨 namespace 泄漏。"""
        existing_key = (row["tenant_id"], row["user_id"], row["domain_id"], row["conversation_id"])
        wanted_key = (
            namespace.tenant_id or "",
            namespace.user_id,
            namespace.domain_id or "",
            namespace.conversation_id or "",
        )
        if existing_key != wanted_key:
            raise NamespaceError(
                f"{id_label} already exists under a different namespace/conversation; "
                "refusing to leak or overwrite across owner boundary"
            )
        if "actor_id" in row.keys():
            existing_actor = str(row["actor_id"] or "")
            wanted_actor = namespace.actor_id()
            if existing_actor != wanted_actor:
                raise NamespaceError(
                    f"{id_label} already exists under a different actor; refusing to reuse the record id"
                )

    @staticmethod
    def _assert_scope_owner(row: sqlite3.Row, namespace: Namespace, *, id_label: str) -> None:
        """Validate hard namespace + conversation; actor remains a soft readable label."""

        existing_key = (row["tenant_id"], row["user_id"], row["domain_id"], row["conversation_id"])
        wanted_key = (
            namespace.tenant_id or "",
            namespace.user_id,
            namespace.domain_id or "",
            namespace.conversation_id or "",
        )
        if existing_key != wanted_key:
            raise NamespaceError(f"{id_label} belongs to a different namespace/conversation")

    # --- 写 ---

    def begin_turn(
        self,
        *,
        namespace: Namespace,
        stimulus_entries: list[TimelineEntryInput],
        annotation_target_ids: list[str],
        turn_id: str = "",
        opened_at: int = 0,
        operation_projection_policy: str = "full_until_raw_compaction",
        operation_settlement_min_utf8_bytes: int = 256,
        operation_settlement_min_saved_ratio: float = 0.5,
    ) -> TurnHandle:
        if not stimulus_entries:
            raise SchemaError("turn_stimulus_required")
        if any(not isinstance(entry, TimelineEntryInput) for entry in stimulus_entries):
            raise TypeError("stimulus_entries must contain TimelineEntryInput values")
        if any(entry.turn_role is not TurnRole.STIMULUS for entry in stimulus_entries):
            raise SchemaError("turn_stimulus_role_required")

        normalized_turn_id = self._normalize_relation_id(turn_id or uuid.uuid4().hex, field="turn_id")
        opened = int(opened_at or time.time())
        prepared: list[TimelineEntryInput] = []
        seen_source_ids: set[str] = set()
        for entry in stimulus_entries:
            source_id = self._normalize_relation_id(entry.source_id or uuid.uuid4().hex, field="source_id")
            if source_id in seen_source_ids:
                raise SchemaError("turn_duplicate_stimulus_source_id")
            seen_source_ids.add(source_id)
            prepared.append(replace(entry, source_id=source_id, turn_id=normalized_turn_id))

        targets = tuple(
            self._normalize_relation_id(item, field="annotation_target_id") for item in annotation_target_ids
        )
        if len(set(targets)) != len(targets):
            raise SchemaError("turn_duplicate_annotation_target")
        if not set(targets).issubset(seen_source_ids):
            raise SchemaError("turn_annotation_target_not_in_stimuli")

        with self._lock, self._conn:
            existing_turn = self._conn.execute(
                "SELECT * FROM turns WHERE turn_id = ?", (normalized_turn_id,)
            ).fetchone()
            if existing_turn is not None:
                self._assert_scope_owner(existing_turn, namespace, id_label=f"turn_id={normalized_turn_id!r}")
                existing_stimuli = tuple(self._json_list(existing_turn["stimulus_source_ids_json"]))
                existing_targets = tuple(self._json_list(existing_turn["annotation_target_ids_json"]))
                if existing_stimuli != tuple(item.source_id for item in prepared) or existing_targets != targets:
                    raise SchemaError("turn_idempotency_conflict")
                return self._turn_handle_from_row(existing_turn)

            for entry in prepared:
                existing = self._conn.execute(
                    "SELECT * FROM messages WHERE source_id = ?", (entry.source_id,)
                ).fetchone()
                if existing is not None:
                    self._assert_scope_owner(existing, namespace, id_label=f"source_id={entry.source_id!r}")
                    raise SchemaError("turn_stimulus_source_id_conflict")

            self._conn.execute(
                """
                INSERT INTO turns(
                    turn_id, tenant_id, user_id, domain_id, conversation_id, status,
                    stimulus_source_ids_json, annotation_target_ids_json,
                    opened_at, operation_projection_policy,
                    operation_settlement_min_utf8_bytes, operation_settlement_min_saved_ratio,
                    row_version
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    normalized_turn_id,
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    _json_dumps([item.source_id for item in prepared]),
                    _json_dumps(list(targets)),
                    opened,
                    str(operation_projection_policy or "full_until_raw_compaction").strip(),
                    int(operation_settlement_min_utf8_bytes or 256),
                    float(
                        operation_settlement_min_saved_ratio
                        if operation_settlement_min_saved_ratio is not None
                        else 0.5
                    ),
                ),
            )
            self._ensure_conversation_state_locked(namespace=namespace, updated_at=opened)
            stored: list[TimelineEntry] = []
            for entry in prepared:
                record = self._add_timeline_entry_locked(namespace=namespace, entry=entry)
                stored.append(TimelineEntry.from_record(record))
            return TurnHandle(
                turn_id=normalized_turn_id,
                namespace=self._owner_namespace(namespace),
                status=TurnStatus.OPEN,
                stimuli=tuple(stored),
                annotation_target_ids=targets,
                opened_at=opened,
                operation_projection_policy=str(operation_projection_policy or "full_until_raw_compaction").strip(),
                operation_settlement_min_utf8_bytes=int(operation_settlement_min_utf8_bytes or 256),
                operation_settlement_min_saved_ratio=float(
                    operation_settlement_min_saved_ratio if operation_settlement_min_saved_ratio is not None else 0.5
                ),
            )

    def append_entry(self, *, namespace: Namespace, entry: TimelineEntryInput) -> TimelineEntry:
        if not isinstance(entry, TimelineEntryInput):
            raise TypeError("entry must be a TimelineEntryInput")
        if not entry.turn_id:
            raise SchemaError("timeline_entry_turn_id_required")
        if entry.turn_role is None:
            raise SchemaError("timeline_entry_turn_role_required")
        if entry.turn_role in {TurnRole.STIMULUS, TurnRole.FINAL}:
            raise SchemaError("timeline_entry_role_not_appendable")

        with self._lock, self._conn:
            turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (entry.turn_id,)).fetchone()
            if turn is None:
                raise SchemaError("turn_not_found")
            self._assert_scope_owner(turn, namespace, id_label=f"turn_id={entry.turn_id!r}")
            if str(turn["status"] or "") != TurnStatus.OPEN.value:
                raise SchemaError("turn_not_open")
            if entry.turn_role is TurnRole.ACTION:
                duplicate_action = self._conn.execute(
                    """
                    SELECT source_id FROM messages
                    WHERE turn_id = ? AND correlation_id = ? AND turn_role = 'action'
                    LIMIT 1
                    """,
                    (entry.turn_id, entry.correlation_id),
                ).fetchone()
                if duplicate_action is not None:
                    raise SchemaError("turn_duplicate_action_correlation")
            if entry.turn_role is TurnRole.OBSERVATION:
                action = self._conn.execute(
                    """
                    SELECT source_id FROM messages
                    WHERE turn_id = ? AND correlation_id = ? AND turn_role = 'action'
                    LIMIT 1
                    """,
                    (entry.turn_id, entry.correlation_id),
                ).fetchone()
                if action is None:
                    raise SchemaError("turn_observation_action_not_found")
            if entry.source_id:
                existing_entry = self._conn.execute(
                    "SELECT * FROM messages WHERE source_id = ?", (entry.source_id,)
                ).fetchone()
                if existing_entry is not None:
                    self._assert_scope_owner(
                        existing_entry,
                        namespace,
                        id_label=f"source_id={entry.source_id!r}",
                    )
                    raise SchemaError("timeline_entry_source_id_conflict")
            prepared = replace(entry, source_id=entry.source_id or uuid.uuid4().hex)
            record = self._add_timeline_entry_locked(namespace=namespace, entry=prepared)
            return TimelineEntry.from_record(record)

    def append_standalone_entry(self, *, namespace: Namespace, entry: TimelineEntryInput) -> TimelineEntry:
        if not isinstance(entry, TimelineEntryInput):
            raise TypeError("entry must be a TimelineEntryInput")
        if entry.turn_id or entry.turn_role is not None:
            raise SchemaError("standalone_entry_must_not_have_turn")
        if entry.reply_to_source_id or entry.correlation_id:
            raise SchemaError("standalone_entry_must_not_have_relations")

        with self._lock, self._conn:
            if entry.source_id:
                existing = self._conn.execute(
                    "SELECT * FROM messages WHERE source_id = ?",
                    (entry.source_id,),
                ).fetchone()
                if existing is not None:
                    self._assert_namespace_owner(
                        existing,
                        self._entry_namespace(namespace=namespace, actor=entry.actor),
                        id_label=f"source_id={entry.source_id!r}",
                    )
                    stored = TimelineEntry.from_record(self._row_to_record(existing, "messages"))
                    if (
                        stored.kind != entry.kind
                        or stored.semantic_text != entry.semantic_text
                        or stored.payload != dict(entry.payload)
                        or stored.memory_metadata != dict(entry.memory_metadata)
                        or stored.annotation_status is not entry.annotation_status
                        or stored.retrieval_policy is not entry.retrieval_policy
                        or stored.retrieval_visibility is not entry.retrieval_visibility
                        or stored.prompt_visible != bool(entry.prompt_visible)
                        or stored.trust is not entry.trust
                    ):
                        raise SchemaError("standalone_entry_source_id_conflict")
                    return stored
            prepared = replace(entry, source_id=entry.source_id or uuid.uuid4().hex)
            record = self._add_timeline_entry_locked(
                namespace=namespace,
                entry=prepared,
                relation_status="standalone",
            )
            return TimelineEntry.from_record(record)

    def commit_turn_completion(self, *, namespace: Namespace, completion: TurnCompletion) -> CompletionCommitResult:
        if not isinstance(completion, TurnCompletion):
            raise TypeError("completion must be a TurnCompletion")

        with self._lock, self._conn:
            turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (completion.turn_id,)).fetchone()
            if turn is None:
                return CompletionCommitResult(
                    status="not_found",
                    turn_id=completion.turn_id,
                    reason="turn_not_found",
                )
            self._assert_scope_owner(turn, namespace, id_label=f"turn_id={completion.turn_id!r}")
            turn_status = str(turn["status"] or "")
            if turn_status == TurnStatus.CLOSED.value:
                final = self._final_entry_for_turn_locked(turn)
                return CompletionCommitResult(
                    status="already_completed",
                    turn_id=completion.turn_id,
                    final_entry=final,
                    reason="turn_already_closed",
                )
            if turn_status == TurnStatus.ABORTED.value:
                return CompletionCommitResult(
                    status="conflict",
                    turn_id=completion.turn_id,
                    reason="turn_aborted",
                )

            target_ids = tuple(self._json_list(turn["annotation_target_ids_json"]))
            annotations_by_target: dict[str, MemoryAnnotation] = {}
            for annotation in completion.annotations:
                if not isinstance(annotation, MemoryAnnotation):
                    raise TypeError("completion.annotations must contain MemoryAnnotation values")
                if annotation.target_source_id in annotations_by_target:
                    return CompletionCommitResult(
                        status="invalid",
                        turn_id=completion.turn_id,
                        reason="duplicate_annotation_target",
                    )
                annotations_by_target[annotation.target_source_id] = annotation
            if set(annotations_by_target) != set(target_ids):
                return CompletionCommitResult(
                    status="invalid",
                    turn_id=completion.turn_id,
                    reason="annotation_targets_mismatch",
                )

            target_rows = self._target_rows_locked(
                namespace=namespace, turn_id=completion.turn_id, target_ids=target_ids
            )
            if len(target_rows) != len(target_ids):
                return CompletionCommitResult(
                    status="invalid",
                    turn_id=completion.turn_id,
                    reason="annotation_target_not_owned_stimulus",
                )

            pending = self._pending_correlations_locked(turn_id=completion.turn_id)
            if pending:
                return CompletionCommitResult(
                    status="pending_actions",
                    turn_id=completion.turn_id,
                    pending_correlations=tuple(pending),
                    reason="turn_has_pending_actions",
                )

            final_source_id = self._normalize_relation_id(
                completion.source_id or uuid.uuid4().hex,
                field="source_id",
            )
            existing_final_id = self._conn.execute(
                "SELECT source_id FROM messages WHERE source_id = ?", (final_source_id,)
            ).fetchone()
            if existing_final_id is not None:
                return CompletionCommitResult(
                    status="conflict",
                    turn_id=completion.turn_id,
                    reason="final_source_id_conflict",
                )

            updated_targets: list[TimelineEntry] = []
            target_visibilities: list[RetrievalVisibility] = []
            any_accepted = False
            for target_id in target_ids:
                annotation = annotations_by_target[target_id]
                target_row = next(row for row in target_rows if str(row["source_id"]) == target_id)
                policy = RetrievalPolicy(str(target_row["retrieval_policy"] or RetrievalPolicy.AUTO.value))
                visibility = resolve_retrieval_visibility(policy, annotation.status)
                target_visibilities.append(visibility)
                any_accepted = any_accepted or annotation.status.accepted
                annotation_metadata = self._completion_annotation_metadata(
                    target_row=target_row,
                    annotation=annotation,
                )
                self._conn.execute(
                    """
                    UPDATE messages
                    SET memory_metadata_json = ?, annotation_status = ?, annotation_source = ?,
                        retrieval_visibility = ?, row_version = row_version + 1,
                        index_status = 'pending'
                    WHERE source_id = ?
                    """,
                    (
                        _json_dumps(annotation_metadata),
                        annotation.status.value,
                        annotation.source,
                        visibility.value,
                        target_id,
                    ),
                )

            if RetrievalVisibility.DEFAULT in target_visibilities:
                final_visibility = RetrievalVisibility.DEFAULT
            elif RetrievalVisibility.EXPLICIT in target_visibilities or not target_visibilities:
                final_visibility = RetrievalVisibility.EXPLICIT
            else:
                final_visibility = RetrievalVisibility.NEVER

            final_payload = dict(completion.payload)
            final_payload["provider_output_raw"] = completion.provider_output_raw
            final_input = TimelineEntryInput(
                source_id=final_source_id,
                turn_id=completion.turn_id,
                kind=completion.kind,
                origin=EntryOrigin.ASSISTANT,
                turn_role=TurnRole.FINAL,
                semantic_text=completion.semantic_text,
                timestamp=completion.timestamp,
                payload=final_payload,
                reply_to_source_id=target_ids[0] if len(target_ids) == 1 else "",
                trace_metadata=completion.trace_metadata,
                annotation_status=(
                    AnnotationStatus.DERIVED_TURN_FINAL if any_accepted else AnnotationStatus.UNANNOTATED
                ),
                annotation_source="turn_completion" if any_accepted else "",
                retrieval_policy=RetrievalPolicy.AUTO,
                retrieval_visibility=final_visibility,
                semanticize=True,
                prompt_visible=True,
                date_label=completion.date_label,
                time_of_day=completion.time_of_day,
                compatibility_role="assistant",
            )
            final_record = self._add_timeline_entry_locked(namespace=namespace, entry=final_input)
            final_projection: ProjectionMessage | None = None
            if completion.final_projection is not None:
                declared_source_ids = tuple(completion.final_projection.source_ids)
                if declared_source_ids and declared_source_ids != (final_source_id,):
                    raise SchemaError("turn_completion_final_projection_source_mismatch")
                prepared_projection = replace(
                    completion.final_projection,
                    source_ids=(final_source_id,),
                )
                saved_projections = self._save_turn_projections_locked(
                    namespace=namespace,
                    turn_id=completion.turn_id,
                    projections=[prepared_projection],
                )
                final_projection = saved_projections[0]
            self._conn.execute(
                """
                UPDATE turns
                SET status = 'closed', final_source_id = ?, closed_at = ?, close_reason = ?,
                    row_version = row_version + 1
                WHERE turn_id = ?
                """,
                (final_source_id, completion.timestamp, completion.close_reason, completion.turn_id),
            )
            for target_id in target_ids:
                row = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (target_id,)).fetchone()
                if row is not None:
                    updated_targets.append(TimelineEntry.from_record(self._row_to_record(row, "messages")))
            return CompletionCommitResult(
                status="completed",
                turn_id=completion.turn_id,
                final_entry=TimelineEntry.from_record(final_record),
                updated_targets=tuple(updated_targets),
                final_projection=final_projection,
            )

    def abort_turn(self, *, namespace: Namespace, turn_id: str, reason: str, closed_at: int) -> TurnAbortResult:
        normalized = self._normalize_relation_id(turn_id, field="turn_id")
        with self._lock, self._conn:
            turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (normalized,)).fetchone()
            if turn is None:
                return TurnAbortResult(status="not_found", turn_id=normalized, reason="turn_not_found")
            self._assert_scope_owner(turn, namespace, id_label=f"turn_id={normalized!r}")
            status = str(turn["status"] or "")
            if status == TurnStatus.ABORTED.value:
                return TurnAbortResult(
                    status="already_aborted", turn_id=normalized, reason=str(turn["close_reason"] or "")
                )
            if status == TurnStatus.CLOSED.value:
                return TurnAbortResult(status="conflict", turn_id=normalized, reason="turn_already_closed")
            close_reason = str(reason or "aborted").strip()[:160] or "aborted"
            self._conn.execute(
                """
                UPDATE turns
                SET status = 'aborted', closed_at = ?, close_reason = ?, row_version = row_version + 1
                WHERE turn_id = ?
                """,
                (int(closed_at or time.time()), close_reason, normalized),
            )
            return TurnAbortResult(status="aborted", turn_id=normalized, reason=close_reason)

    def abort_stale_open_turns(
        self,
        *,
        namespace: Namespace,
        opened_before: int,
        reason: str,
        closed_at: int,
    ) -> tuple[TurnAbortResult, ...]:
        cutoff = max(0, int(opened_before or 0))
        closed = max(0, int(closed_at or time.time()))
        close_reason = str(reason or "stale_open_turn_recovered").strip()[:160] or "stale_open_turn_recovered"
        scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
        with self._lock, self._immediate_transaction():
            rows = self._conn.execute(
                f"""
                SELECT turn_id
                FROM turns
                WHERE {scope_clause} AND status = 'open' AND opened_at < ?
                ORDER BY opened_at, turn_id
                """,
                (*scope_params, cutoff),
            ).fetchall()
            turn_ids = tuple(str(row["turn_id"] or "") for row in rows if str(row["turn_id"] or ""))
            for turn_id in turn_ids:
                self._conn.execute(
                    """
                    UPDATE turns
                    SET status = 'aborted', closed_at = ?, close_reason = ?, row_version = row_version + 1
                    WHERE turn_id = ? AND status = 'open'
                    """,
                    (closed, close_reason, turn_id),
                )
        return tuple(TurnAbortResult(status="aborted", turn_id=turn_id, reason=close_reason) for turn_id in turn_ids)

    def save_turn_projections(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        projections: list[ProjectionMessageInput],
    ) -> tuple[ProjectionMessage, ...]:
        if any(not isinstance(item, ProjectionMessageInput) for item in projections):
            raise TypeError("projections must contain ProjectionMessageInput values")
        with self._lock, self._conn:
            return self._save_turn_projections_locked(
                namespace=namespace,
                turn_id=turn_id,
                projections=projections,
            )

    def commit_request_projection(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        projections: list[ProjectionMessageInput],
        audit: ProjectionAuditInput,
    ) -> RequestProjectionResult:
        if not isinstance(audit, ProjectionAuditInput):
            raise TypeError("audit must be a ProjectionAuditInput")
        if audit.turn_id != str(turn_id or "").strip():
            raise SchemaError("projection_audit_turn_mismatch")
        if any(item.provider_profile != audit.provider_profile for item in projections):
            raise SchemaError("projection_audit_profile_mismatch")

        with self._lock, self._conn:
            turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (audit.turn_id,)).fetchone()
            if turn is None:
                raise SchemaError("turn_not_found")
            self._assert_scope_owner(turn, namespace, id_label=f"turn_id={audit.turn_id!r}")
            stored = self._save_turn_projections_locked(
                namespace=namespace,
                turn_id=audit.turn_id,
                projections=projections,
                allow_open_request_replacement=True,
            )
            media_omitted = audit.media_omitted or any(
                item.projection_status is ProjectionStatus.MEDIA_OMITTED for item in stored
            )
            prepared_audit = replace(
                audit,
                media_omitted=media_omitted,
                created_at=int(audit.created_at or time.time()),
            )
            stored_audit = self._save_projection_audit_locked(namespace=namespace, audit=prepared_audit)
            return RequestProjectionResult(projections=stored, audit=stored_audit)

    def _save_turn_projections_locked(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        projections: list[ProjectionMessageInput],
        allow_open_request_replacement: bool = False,
    ) -> tuple[ProjectionMessage, ...]:
        normalized_turn_id = self._normalize_relation_id(turn_id, field="turn_id")
        if not projections:
            return ()
        if any(not isinstance(item, ProjectionMessageInput) for item in projections):
            raise TypeError("projections must contain ProjectionMessageInput values")
        if any(not item.source_ids for item in projections):
            raise SchemaError("projection_source_ids_required")
        profiles = {item.provider_profile for item in projections}
        if len(profiles) != 1:
            raise SchemaError("projection_mixed_provider_profiles")

        turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (normalized_turn_id,)).fetchone()
        if turn is not None:
            self._assert_scope_owner(turn, namespace, id_label=f"turn_id={normalized_turn_id!r}")
        elif not normalized_turn_id.startswith(("legacy.", "summary.", "semantic.")):
            raise SchemaError("turn_not_found")

        source_rows = self._projection_source_rows_locked(
            namespace=namespace,
            turn_id=normalized_turn_id,
            projections=projections,
            synthetic_turn=turn is None,
        )
        if not source_rows:
            raise SchemaError("projection_source_ids_required")

        provider_profile = next(iter(profiles))
        existing_rows = self._conn.execute(
            """
            SELECT * FROM prompt_projections
            WHERE tenant_id = ? AND user_id = ? AND domain_id = ? AND conversation_id = ?
              AND turn_id = ? AND provider_profile = ?
            ORDER BY projection_index
            """,
            (
                namespace.tenant_id or "",
                namespace.user_id,
                namespace.domain_id or "",
                namespace.conversation_id or "",
                normalized_turn_id,
                provider_profile,
            ),
        ).fetchall()
        existing_by_index = {int(row["projection_index"]): row for row in existing_rows}
        if sorted(existing_by_index) != list(range(len(existing_by_index))):
            raise SchemaError("projection_index_sequence_corrupt")
        source_projection_index: dict[str, int] = {}
        covered_source_ids: list[str] = []
        for row in existing_rows:
            record = self._row_to_record(row, "prompt_projections")
            for source_id in record.get("source_ids") or ():
                normalized_source_id = str(source_id)
                prior = source_projection_index.get(normalized_source_id)
                if prior is not None and prior != int(row["projection_index"]):
                    raise SchemaError("projection_source_already_mapped")
                source_projection_index[normalized_source_id] = int(row["projection_index"])
                covered_source_ids.append(normalized_source_id)
        if turn is None:
            expected_source_ids = tuple(
                dict.fromkeys(source_id for item in projections for source_id in item.source_ids)
            )
        else:
            expected_source_ids = tuple(
                str(row["source_id"])
                for row in self._conn.execute(
                    """
                    SELECT source_id FROM messages
                    WHERE turn_id = ? AND prompt_visible = 1
                    ORDER BY seq_no
                    """,
                    (normalized_turn_id,),
                ).fetchall()
            )
        if tuple(covered_source_ids) != expected_source_ids[: len(covered_source_ids)]:
            raise SchemaError("projection_history_not_append_only")
        append_offset = len(covered_source_ids)
        next_index = max(existing_by_index, default=-1) + 1
        resolved_indices: set[int] = set()
        stored: list[ProjectionMessage] = []
        now = int(time.time())

        for item in projections:
            projection_index = item.projection_index if item.projection_index >= 0 else next_index
            if projection_index in resolved_indices:
                raise SchemaError("projection_duplicate_index")
            resolved_indices.add(projection_index)
            source_ids = tuple(item.source_ids)
            if any(
                source_id in source_projection_index and source_projection_index[source_id] != projection_index
                for source_id in source_ids
            ):
                raise SchemaError("projection_source_already_mapped")
            payload = dict(item.payload)
            payload_hash = stable_projection_hash(payload)
            existing = existing_by_index.get(projection_index)
            if existing is not None:
                existing_record = self._row_to_record(existing, "prompt_projections")
                projection_changed = (
                    str(existing["payload_hash"] or "") != payload_hash
                    or tuple(existing_record.get("source_ids") or ()) != source_ids
                    or str(existing["projection_status"] or "") != item.projection_status.value
                    or int(existing["projection_version"] or 0) != item.projection_version
                )
                if projection_changed:
                    existing_status = str(existing["projection_status"] or "")
                    can_replace = bool(
                        allow_open_request_replacement
                        and turn is not None
                        and str(turn["status"] or "") == TurnStatus.OPEN.value
                        # Earlier messages in an open turn may already be audited
                        # while newly appended tool projections have not crossed a
                        # request boundary yet.  Freeze per projection, not per turn.
                        and existing_status != ProjectionStatus.REQUEST_FROZEN.value
                        and tuple(existing_record.get("source_ids") or ()) == source_ids
                        and int(existing["projection_version"] or 0) == item.projection_version
                    )
                    if not can_replace:
                        raise SchemaError("projection_immutable_conflict")
                    self._conn.execute(
                        """
                        UPDATE prompt_projections
                        SET payload_json = ?, payload_hash = ?, projection_status = ?, created_at = ?
                        WHERE projection_id = ?
                        """,
                        (
                            _json_dumps(payload),
                            payload_hash,
                            item.projection_status.value,
                            now,
                            str(existing["projection_id"] or ""),
                        ),
                    )
                    existing = self._conn.execute(
                        "SELECT * FROM prompt_projections WHERE projection_id = ?",
                        (str(existing["projection_id"] or ""),),
                    ).fetchone()
                    if existing is None:
                        raise SchemaError("projection_update_failed")
                    existing_record = self._row_to_record(existing, "prompt_projections")
                stored.append(ProjectionMessage.from_record(existing_record))
                continue

            if projection_index != next_index:
                raise SchemaError("projection_index_not_append_only")
            expected_segment = expected_source_ids[append_offset : append_offset + len(source_ids)]
            if source_ids != expected_segment:
                raise SchemaError("projection_source_not_next_append")
            append_offset += len(source_ids)
            next_index += 1

            for source_id in source_ids:
                source_projection_index[source_id] = projection_index

            projection_id = stable_projection_hash(
                {
                    "namespace": [
                        namespace.tenant_id or "",
                        namespace.user_id,
                        namespace.domain_id or "",
                        namespace.conversation_id or "",
                    ],
                    "turn_id": normalized_turn_id,
                    "provider_profile": provider_profile,
                    "projection_index": projection_index,
                }
            )
            self._conn.execute(
                """
                INSERT INTO prompt_projections(
                    projection_id, tenant_id, user_id, domain_id, conversation_id,
                    turn_id, projection_index, provider_profile, payload_json,
                    source_ids_json, payload_hash, projection_status,
                    projection_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    projection_id,
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    normalized_turn_id,
                    projection_index,
                    provider_profile,
                    _json_dumps(payload),
                    _json_dumps(list(source_ids)),
                    payload_hash,
                    item.projection_status.value,
                    item.projection_version,
                    now,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM prompt_projections WHERE projection_id = ?", (projection_id,)
            ).fetchone()
            if row is None:
                raise SchemaError("projection_insert_failed")
            stored.append(ProjectionMessage.from_record(self._row_to_record(row, "prompt_projections")))
        return tuple(stored)

    def _projection_source_rows_locked(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        projections: list[ProjectionMessageInput],
        synthetic_turn: bool,
    ) -> dict[str, sqlite3.Row]:
        source_ids = tuple(dict.fromkeys(source_id for item in projections for source_id in item.source_ids))
        if not source_ids:
            return {}
        placeholders = ",".join("?" for _ in source_ids)
        if turn_id.startswith("summary."):
            table, id_column = "summaries", "summary_id"
        elif turn_id.startswith("semantic."):
            table, id_column = "semantic_summaries", "semantic_id"
        else:
            table, id_column = "messages", "source_id"
        rows = self._conn.execute(
            f"SELECT * FROM {table} WHERE {id_column} IN ({placeholders})", list(source_ids)
        ).fetchall()
        by_source_id = {str(row[id_column]): row for row in rows}
        if set(by_source_id) != set(source_ids):
            raise SchemaError("projection_source_not_found")
        for source_id, row in by_source_id.items():
            self._assert_scope_owner(row, namespace, id_label=f"source_id={source_id!r}")
            if table != "messages":
                continue
            source_turn_id = str(row["turn_id"] or "")
            if synthetic_turn:
                if source_turn_id:
                    raise SchemaError("projection_source_turn_mismatch")
            elif source_turn_id != turn_id:
                raise SchemaError("projection_source_turn_mismatch")
        return by_source_id

    def _save_projection_audit_locked(
        self,
        *,
        namespace: Namespace,
        audit: ProjectionAuditInput,
    ) -> ProjectionAudit:
        key = (
            namespace.tenant_id or "",
            namespace.user_id,
            namespace.domain_id or "",
            namespace.conversation_id or "",
            audit.turn_id,
            audit.attempt,
            audit.provider_profile,
        )
        existing = self._conn.execute(
            """
            SELECT * FROM projection_audits
            WHERE tenant_id = ? AND user_id = ? AND domain_id = ? AND conversation_id = ?
              AND turn_id = ? AND attempt = ? AND provider_profile = ?
            """,
            key,
        ).fetchone()
        comparable = (
            audit.model_route_hash,
            audit.system_prefix_hash,
            audit.tool_schema_hash,
            audit.history_hash,
            audit.full_prefix_hash,
            audit.projection_version,
            int(audit.media_omitted),
        )
        if existing is not None:
            existing_comparable = (
                str(existing["model_route_hash"] or ""),
                str(existing["system_prefix_hash"] or ""),
                str(existing["tool_schema_hash"] or ""),
                str(existing["history_hash"] or ""),
                str(existing["full_prefix_hash"] or ""),
                int(existing["projection_version"] or 0),
                int(existing["media_omitted"] or 0),
            )
            if existing_comparable != comparable:
                raise SchemaError("projection_audit_immutable_conflict")
            return ProjectionAudit.from_record(dict(existing))

        self._conn.execute(
            """
            INSERT INTO projection_audits(
                tenant_id, user_id, domain_id, conversation_id, turn_id, attempt,
                provider_profile, model_route_hash, system_prefix_hash,
                tool_schema_hash, history_hash, full_prefix_hash,
                projection_version, media_omitted, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *key,
                audit.model_route_hash,
                audit.system_prefix_hash,
                audit.tool_schema_hash,
                audit.history_hash,
                audit.full_prefix_hash,
                audit.projection_version,
                int(audit.media_omitted),
                int(audit.created_at or time.time()),
            ),
        )
        row = self._conn.execute(
            """
            SELECT * FROM projection_audits
            WHERE tenant_id = ? AND user_id = ? AND domain_id = ? AND conversation_id = ?
              AND turn_id = ? AND attempt = ? AND provider_profile = ?
            """,
            key,
        ).fetchone()
        if row is None:
            raise SchemaError("projection_audit_insert_failed")
        return ProjectionAudit.from_record(dict(row))

    def list_compaction_bundles(self, *, namespace: Namespace) -> list[TurnBundle]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} AND is_summarized = 0 ORDER BY seq_no",
                params,
            ).fetchall()
            grouped: dict[str, list[sqlite3.Row]] = {}
            legacy_rows: list[sqlite3.Row] = []
            for row in rows:
                turn_id = str(row["turn_id"] or "")
                if turn_id:
                    grouped.setdefault(turn_id, []).append(row)
                else:
                    legacy_rows.append(row)

            bundles: list[TurnBundle] = []
            for turn_id, turn_rows in grouped.items():
                turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
                if turn is None:
                    raise SchemaError("compaction_turn_not_found")
                self._assert_scope_owner(turn, namespace, id_label=f"turn_id={turn_id!r}")
                all_rows = self._conn.execute(
                    "SELECT source_id, is_summarized FROM messages WHERE turn_id = ?",
                    (turn_id,),
                ).fetchall()
                if any(int(row["is_summarized"] or 0) for row in all_rows):
                    raise SchemaError("compaction_partial_turn")
                entries = tuple(TimelineEntry.from_record(self._row_to_record(row, "messages")) for row in turn_rows)
                bundles.append(
                    TurnBundle(
                        turn_id=turn_id,
                        status=TurnStatus(str(turn["status"] or TurnStatus.OPEN.value)),
                        entries=entries,
                        turn_row_version=int(turn["row_version"] or 0),
                        first_seq_no=min(entry.seq_no for entry in entries),
                        last_seq_no=max(entry.seq_no for entry in entries),
                    )
                )

            for row in legacy_rows:
                entry = TimelineEntry.from_record(self._row_to_record(row, "messages"))
                bundles.append(
                    TurnBundle(
                        turn_id=f"legacy.{stable_projection_hash({'source_id': entry.source_id})}",
                        status=TurnStatus.CLOSED,
                        entries=(entry,),
                        turn_row_version=entry.row_version,
                        first_seq_no=entry.seq_no,
                        last_seq_no=entry.seq_no,
                        legacy=True,
                    )
                )
        return sorted(bundles, key=lambda item: (item.first_seq_no, item.last_seq_no, item.turn_id))

    def commit_summary_batch(
        self,
        *,
        namespace: Namespace,
        snapshot: CompactionSnapshot,
        records: list[SummaryRecordInput],
    ) -> SummaryBatchCommitResult:
        if not isinstance(snapshot, CompactionSnapshot):
            raise TypeError("snapshot must be a CompactionSnapshot")
        if not records or any(not isinstance(item, SummaryRecordInput) for item in records):
            raise TypeError("records must contain SummaryRecordInput values")
        expected_namespace = (
            namespace.tenant_id or "",
            namespace.user_id,
            namespace.domain_id or "",
            namespace.conversation_id or "",
        )
        if snapshot.namespace_key != expected_namespace:
            raise NamespaceError("compaction_snapshot belongs to a different namespace/conversation")
        assigned: dict[str, str] = {}
        for item in records:
            if not item.source_ids:
                raise SchemaError("compaction_summary_sources_required")
            if str(item.record.get("summary_id") or item.summary_id) != item.summary_id:
                raise SchemaError("compaction_summary_id_mismatch")
            for source_id in item.source_ids:
                if source_id in assigned:
                    raise SchemaError("compaction_source_assignment_overlap")
                assigned[source_id] = item.summary_id
        if set(assigned) != set(snapshot.ordered_source_ids):
            raise SchemaError("compaction_source_partition_mismatch")

        with self._lock, self._immediate_transaction():
            placeholders = ",".join("?" for _ in snapshot.ordered_source_ids)
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE source_id IN ({placeholders})",
                list(snapshot.ordered_source_ids),
            ).fetchall()
            by_source = {str(row["source_id"]): row for row in rows}
            if set(by_source) != set(snapshot.ordered_source_ids):
                return SummaryBatchCommitResult(status="stale_batch", reason="source_missing")
            for source_id, row in by_source.items():
                self._assert_scope_owner(row, namespace, id_label=f"source_id={source_id!r}")

            if all(int(row["is_summarized"] or 0) for row in rows):
                if all(
                    str(by_source[source_id]["summary_id"] or "") == summary_id
                    for source_id, summary_id in assigned.items()
                ):
                    summary_rows = tuple(
                        self._conn.execute(
                            "SELECT * FROM summaries WHERE summary_id = ?", (item.summary_id,)
                        ).fetchone()
                        for item in records
                    )
                    if any(row is None for row in summary_rows):
                        return SummaryBatchCommitResult(status="stale_batch", reason="summary_missing")
                    summaries = tuple(self._row_to_record(row, "summaries") for row in summary_rows)
                    return SummaryBatchCommitResult(
                        status="already_committed",
                        summaries=summaries,
                        compaction_generation=self._compaction_generation_locked(namespace),
                    )
                return SummaryBatchCommitResult(status="stale_batch", reason="source_already_summarized")
            if any(int(row["is_summarized"] or 0) for row in rows):
                return SummaryBatchCommitResult(status="stale_batch", reason="partial_source_assignment")
            if self._compaction_generation_locked(namespace) != snapshot.compaction_generation:
                return SummaryBatchCommitResult(status="stale_batch", reason="generation_changed")
            expected_versions = dict(snapshot.message_row_versions)
            if any(
                int(row["row_version"] or 0) != expected_versions.get(source_id) for source_id, row in by_source.items()
            ):
                return SummaryBatchCommitResult(status="stale_batch", reason="source_version_changed")

            scope_clause, params = self._scope_clause(namespace, with_conversation=True)
            prefix_rows = self._conn.execute(
                f"SELECT source_id FROM messages WHERE {scope_clause} AND is_summarized = 0 ORDER BY seq_no LIMIT ?",
                [*params, len(snapshot.ordered_source_ids)],
            ).fetchall()
            if tuple(str(row["source_id"]) for row in prefix_rows) != snapshot.ordered_source_ids:
                return SummaryBatchCommitResult(status="stale_batch", reason="prefix_changed")

            for turn_id, row_version in snapshot.turn_row_versions:
                turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
                if turn is None or int(turn["row_version"] or 0) != row_version:
                    return SummaryBatchCommitResult(status="stale_batch", reason="turn_version_changed")
                try:
                    turn_status = TurnStatus(str(turn["status"] or ""))
                except ValueError:
                    return SummaryBatchCommitResult(status="stale_batch", reason="turn_status_invalid")
                if not turn_status.terminal:
                    return SummaryBatchCommitResult(status="stale_batch", reason="turn_not_terminal")

            current_hashes = self._projection_hashes_by_source_locked(
                namespace=namespace,
                provider_profile=snapshot.provider_profile,
                source_ids=snapshot.ordered_source_ids,
            )
            if (
                tuple((source_id, current_hashes.get(source_id, ())) for source_id, _ in snapshot.projection_hashes)
                != snapshot.projection_hashes
            ):
                return SummaryBatchCommitResult(status="stale_batch", reason="projection_changed")

            for item in records:
                existing = self._conn.execute(
                    "SELECT * FROM summaries WHERE summary_id = ?", (item.summary_id,)
                ).fetchone()
                if existing is not None:
                    self._assert_scope_owner(existing, namespace, id_label=f"summary_id={item.summary_id!r}")
                    return SummaryBatchCommitResult(status="stale_batch", reason="summary_id_conflict")

            saved: list[dict[str, Any]] = []
            for item in records:
                record = {**dict(item.record), "summary_id": item.summary_id, "source_ids": list(item.source_ids)}
                saved.append(self._add_summary_locked(namespace=namespace, record=record))
                record_placeholders = ",".join("?" for _ in item.source_ids)
                cursor = self._conn.execute(
                    f"""
                    UPDATE messages
                    SET is_summarized = 1, summary_id = ?, row_version = row_version + 1
                    WHERE source_id IN ({record_placeholders}) AND is_summarized = 0
                    """,
                    [item.summary_id, *item.source_ids],
                )
                if cursor.rowcount != len(item.source_ids):
                    raise SchemaError("compaction_atomic_source_update_failed")
            generation = self._increment_compaction_generation_locked(namespace)
            return SummaryBatchCommitResult(
                status="committed",
                summaries=tuple(saved),
                compaction_generation=generation,
            )

    def commit_semantic_batch(
        self,
        *,
        namespace: Namespace,
        commit: SemanticCommitInput,
    ) -> SemanticBatchCommitResult:
        if not isinstance(commit, SemanticCommitInput):
            raise TypeError("commit must be a SemanticCommitInput")
        snapshot = commit.snapshot
        if not isinstance(snapshot, SemanticSnapshot):
            raise TypeError("commit.snapshot must be a SemanticSnapshot")
        expected_namespace = (
            namespace.tenant_id or "",
            namespace.user_id,
            namespace.domain_id or "",
            namespace.conversation_id or "",
        )
        if snapshot.namespace_key != expected_namespace:
            raise NamespaceError("semantic_snapshot belongs to a different namespace/conversation")
        semantic_record = dict(commit.semantic_record)
        semantic_id = str(semantic_record.get("semantic_id") or "").strip()
        if not semantic_id:
            raise SchemaError("semantic_commit_id_required")

        with self._lock, self._immediate_transaction():
            placeholders = ",".join("?" for _ in snapshot.summary_ids)
            rows = self._conn.execute(
                f"SELECT * FROM summaries WHERE summary_id IN ({placeholders})",
                list(snapshot.summary_ids),
            ).fetchall()
            by_id = {str(row["summary_id"]): row for row in rows}
            if set(by_id) != set(snapshot.summary_ids):
                return SemanticBatchCommitResult(status="stale_batch", reason="summary_missing")
            for summary_id, row in by_id.items():
                self._assert_scope_owner(row, namespace, id_label=f"summary_id={summary_id!r}")
            if all(int(row["is_semanticized"] or 0) for row in rows):
                if all(str(row["semantic_id"] or "") == semantic_id for row in rows):
                    existing = self._conn.execute(
                        "SELECT * FROM semantic_summaries WHERE semantic_id = ?", (semantic_id,)
                    ).fetchone()
                    if existing is None:
                        return SemanticBatchCommitResult(status="stale_batch", reason="semantic_missing")
                    return SemanticBatchCommitResult(
                        status="already_committed",
                        semantic_record=self._row_to_record(existing, "semantic_summaries"),
                        compaction_generation=self._compaction_generation_locked(namespace),
                    )
                return SemanticBatchCommitResult(status="stale_batch", reason="summary_already_semanticized")
            if any(int(row["is_semanticized"] or 0) for row in rows):
                return SemanticBatchCommitResult(status="stale_batch", reason="partial_semantic_assignment")
            if self._compaction_generation_locked(namespace) != snapshot.compaction_generation:
                return SemanticBatchCommitResult(status="stale_batch", reason="generation_changed")
            expected_versions = dict(snapshot.summary_row_versions)
            if any(
                int(row["row_version"] or 0) != expected_versions.get(summary_id) for summary_id, row in by_id.items()
            ):
                return SemanticBatchCommitResult(status="stale_batch", reason="summary_version_changed")

            existing_semantic = self._conn.execute(
                "SELECT * FROM semantic_summaries WHERE semantic_id = ?", (semantic_id,)
            ).fetchone()
            if snapshot.reinforcement_target_id:
                if snapshot.reinforcement_target_id != semantic_id or existing_semantic is None:
                    return SemanticBatchCommitResult(status="stale_batch", reason="reinforcement_target_missing")
                self._assert_scope_owner(existing_semantic, namespace, id_label=f"semantic_id={semantic_id!r}")
                if int(existing_semantic["row_version"] or 0) != snapshot.reinforcement_target_row_version:
                    return SemanticBatchCommitResult(status="stale_batch", reason="reinforcement_target_changed")
            elif existing_semantic is not None:
                return SemanticBatchCommitResult(status="stale_batch", reason="semantic_id_conflict")

            if snapshot.reinforcement_target_id:
                semantic_turn_id = f"semantic.{stable_projection_hash({'semantic_id': semantic_id})}"
                scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
                self._conn.execute(
                    f"DELETE FROM prompt_projections WHERE {scope_clause} AND turn_id = ?",
                    [*scope_params, semantic_turn_id],
                )
                self._increment_projection_generation_locked(namespace)
            saved = self._add_semantic_summary_locked(namespace=namespace, record=semantic_record)
            cursor = self._conn.execute(
                f"""
                UPDATE summaries
                SET is_semanticized = 1, semantic_id = ?, row_version = row_version + 1
                WHERE summary_id IN ({placeholders}) AND is_semanticized = 0
                """,
                [semantic_id, *snapshot.summary_ids],
            )
            if cursor.rowcount != len(snapshot.summary_ids):
                raise SchemaError("semantic_atomic_source_update_failed")
            generation = self._increment_compaction_generation_locked(namespace)
            return SemanticBatchCommitResult(
                status="committed",
                semantic_record=saved,
                compaction_generation=generation,
            )

    def _projection_hashes_by_source_locked(
        self,
        *,
        namespace: Namespace,
        provider_profile: str,
        source_ids: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        wanted = set(source_ids)
        # settled 历史投影优先: 已冻结的 compact turn 的权威 hash 在 settled 账本,
        # 与 request builder / compaction 的 _freeze_bundle_projection 一致。
        settled_hashes: dict[str, list[str]] = {}
        for row in self._conn.execute(
            "SELECT source_ids_json, payload_hash FROM settled_prompt_projection WHERE provider_profile = ?",
            (provider_profile,),
        ).fetchall():
            for source_id in self._json_list(row["source_ids_json"]):
                if source_id in wanted:
                    settled_hashes.setdefault(source_id, []).append(str(row["payload_hash"] or ""))
        if set(settled_hashes) == wanted:
            return {source_id: tuple(values) for source_id, values in settled_hashes.items()}
        remaining = wanted - set(settled_hashes)
        hashes: dict[str, list[str]] = {source_id: list(values) for source_id, values in settled_hashes.items()}
        rows = self._conn.execute(
            f"""
            SELECT source_ids_json, payload_hash FROM prompt_projections
            WHERE {scope_clause} AND provider_profile = ?
            ORDER BY turn_id, projection_index
            """,
            [*params, provider_profile],
        ).fetchall()
        for row in rows:
            for source_id in self._json_list(row["source_ids_json"]):
                if source_id in remaining:
                    hashes.setdefault(source_id, []).append(str(row["payload_hash"] or ""))
        return {source_id: tuple(values) for source_id, values in hashes.items()}

    def _compaction_generation_locked(self, namespace: Namespace) -> int:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        row = self._conn.execute(
            f"SELECT compaction_generation FROM conversation_states WHERE {scope_clause}", params
        ).fetchone()
        return int(row["compaction_generation"] or 0) if row is not None else 0

    def _increment_compaction_generation_locked(self, namespace: Namespace) -> int:
        now = int(time.time())
        self._ensure_conversation_state_locked(namespace=namespace, updated_at=now)
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        self._conn.execute(
            f"""
            UPDATE conversation_states
            SET compaction_generation = compaction_generation + 1,
                row_version = row_version + 1, updated_at = ?
            WHERE {scope_clause}
            """,
            [now, *params],
        )
        return self._compaction_generation_locked(namespace)

    def _increment_projection_generation_locked(self, namespace: Namespace) -> int:
        now = int(time.time())
        self._ensure_conversation_state_locked(namespace=namespace, updated_at=now)
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        self._conn.execute(
            f"""
            UPDATE conversation_states
            SET projection_generation = projection_generation + 1,
                row_version = row_version + 1, updated_at = ?
            WHERE {scope_clause}
            """,
            [now, *params],
        )
        row = self._conn.execute(
            f"SELECT projection_generation FROM conversation_states WHERE {scope_clause}", params
        ).fetchone()
        return int(row["projection_generation"] or 0) if row is not None else 0

    def _add_timeline_entry_locked(
        self,
        *,
        namespace: Namespace,
        entry: TimelineEntryInput,
        relation_status: str = "linked",
    ) -> dict[str, Any]:
        entry_namespace = self._entry_namespace(namespace=namespace, actor=entry.actor)
        role = entry.compatibility_role or self._compatibility_role(entry)
        compatibility_metadata = self._compatibility_memory_metadata(entry)
        return self._add_message_locked(
            namespace=entry_namespace,
            role=role,
            content=entry.semantic_text,
            timestamp=entry.timestamp,
            fields={
                "source_id": entry.source_id,
                "date_label": entry.date_label,
                "time_of_day": entry.time_of_day,
                "memory_metadata": compatibility_metadata,
                "kind": entry.kind,
                "origin": entry.origin.value,
                "turn_id": entry.turn_id,
                "turn_role": entry.turn_role.value if entry.turn_role else "",
                "reply_to_source_id": entry.reply_to_source_id,
                "correlation_id": entry.correlation_id,
                "relation_status": relation_status,
                "target_actor_id": entry.target_actor.stable_id if entry.target_actor else "",
                "target_actor_display_name": entry.target_actor.display_name if entry.target_actor else "",
                "payload": dict(entry.payload),
                "semantic_text": entry.semantic_text,
                "trace_metadata": dict(entry.trace_metadata),
                "annotation_status": entry.annotation_status.value,
                "annotation_source": entry.annotation_source,
                "retrieval_policy": entry.retrieval_policy.value,
                "retrieval_visibility": entry.retrieval_visibility.value,
                "semanticize": entry.semanticize
                and entry.turn_role not in {TurnRole.ACTION, TurnRole.OBSERVATION}
                and not entry.kind.startswith("material."),
                "prompt_visible": entry.prompt_visible,
                "trust": entry.trust.value,
                "renderer_id": entry.renderer_id,
                "renderer_version": entry.renderer_version,
            },
        )

    def _target_rows_locked(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        target_ids: tuple[str, ...],
    ) -> list[sqlite3.Row]:
        if not target_ids:
            return []
        placeholders = ",".join("?" for _ in target_ids)
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        return self._conn.execute(
            f"""
            SELECT * FROM messages
            WHERE {scope_clause} AND turn_id = ? AND turn_role = 'stimulus'
              AND source_id IN ({placeholders})
            """,
            [*params, turn_id, *target_ids],
        ).fetchall()

    def _pending_correlations_locked(self, *, turn_id: str) -> list[str]:
        rows = self._conn.execute("SELECT * FROM messages WHERE turn_id = ? ORDER BY seq_no", (turn_id,)).fetchall()
        actions = {
            str(row["correlation_id"] or "")
            for row in rows
            if str(row["turn_role"] or "") == TurnRole.ACTION.value and str(row["correlation_id"] or "")
        }
        terminal: set[str] = set()
        for row in rows:
            if str(row["turn_role"] or "") != TurnRole.OBSERVATION.value:
                continue
            correlation_id = str(row["correlation_id"] or "")
            if not correlation_id:
                continue
            record = self._row_to_record(row, "messages")
            status = str((record.get("trace_metadata") or {}).get("status") or "").strip().lower()
            if status not in {"open", "pending", "running", "streaming"}:
                terminal.add(correlation_id)
        return sorted(actions - terminal)

    def _final_entry_for_turn_locked(self, turn: sqlite3.Row) -> TimelineEntry | None:
        final_source_id = str(turn["final_source_id"] or "")
        if not final_source_id:
            return None
        row = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (final_source_id,)).fetchone()
        return TimelineEntry.from_record(self._row_to_record(row, "messages")) if row is not None else None

    def _turn_handle_from_row(self, row: sqlite3.Row) -> TurnHandle:
        namespace = Namespace(
            tenant_id=str(row["tenant_id"] or ""),
            user_id=str(row["user_id"] or ""),
            domain_id=str(row["domain_id"] or ""),
            conversation_id=str(row["conversation_id"] or ""),
        )
        turn_id = str(row["turn_id"] or "")
        stimuli = tuple(self.get_turn_entries(namespace=namespace, turn_id=turn_id))
        stimulus_ids = set(self._json_list(row["stimulus_source_ids_json"]))
        return TurnHandle(
            turn_id=turn_id,
            namespace=namespace,
            status=TurnStatus(str(row["status"] or TurnStatus.OPEN.value)),
            stimuli=tuple(item for item in stimuli if item.source_id in stimulus_ids),
            annotation_target_ids=tuple(self._json_list(row["annotation_target_ids_json"])),
            opened_at=int(row["opened_at"] or 0),
            operation_projection_policy=str(row["operation_projection_policy"] or "full_until_raw_compaction").strip(),
            operation_settlement_min_utf8_bytes=int(row["operation_settlement_min_utf8_bytes"] or 256),
            operation_settlement_min_saved_ratio=float(
                row["operation_settlement_min_saved_ratio"]
                if row["operation_settlement_min_saved_ratio"] is not None
                else 0.5
            ),
        )

    def _ensure_conversation_state_locked(self, *, namespace: Namespace, updated_at: int) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO conversation_states(
                tenant_id, user_id, domain_id, conversation_id, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                namespace.tenant_id or "",
                namespace.user_id,
                namespace.domain_id or "",
                namespace.conversation_id or "",
                int(updated_at),
            ),
        )

    @staticmethod
    def _entry_namespace(*, namespace: Namespace, actor: Any) -> Namespace:
        return Namespace(
            tenant_id=namespace.tenant_id,
            user_id=namespace.user_id,
            domain_id=namespace.domain_id,
            conversation_id=namespace.conversation_id,
            actor=actor if actor is not None else namespace.actor,
        )

    @staticmethod
    def _owner_namespace(namespace: Namespace) -> Namespace:
        return Namespace(
            tenant_id=namespace.tenant_id,
            user_id=namespace.user_id,
            domain_id=namespace.domain_id,
            conversation_id=namespace.conversation_id,
        )

    @staticmethod
    def _compatibility_role(entry: TimelineEntryInput) -> str:
        if entry.kind == "message.user":
            return "user"
        if entry.kind == "message.assistant":
            return "assistant"
        if entry.kind.startswith("event."):
            return entry.kind
        if entry.kind.startswith("tool.") and entry.turn_role is TurnRole.ACTION:
            tool_name = entry.kind[len("tool.") :].removesuffix(".call")
            return f"assistant.tool_call {tool_name} {entry.correlation_id}".strip()
        if entry.kind.startswith("tool.") and entry.turn_role is TurnRole.OBSERVATION:
            tool_name = entry.kind[len("tool.") :].removesuffix(".result")
            return f"tool.{tool_name} {entry.correlation_id}".strip()
        return entry.kind

    @staticmethod
    def _compatibility_memory_metadata(entry: TimelineEntryInput) -> dict[str, Any]:
        """Project typed entries without reintroducing V1 trace categories."""

        return dict(entry.memory_metadata)

    @staticmethod
    def _completion_annotation_metadata(
        *,
        target_row: sqlite3.Row,
        annotation: MemoryAnnotation,
    ) -> dict[str, Any]:
        if annotation.status.accepted:
            return dict(annotation.memory_metadata)
        kind = str(target_row["kind"] or "")
        if not kind.startswith(("event.", "material.")):
            return dict(annotation.memory_metadata)
        try:
            prior = json.loads(str(target_row["memory_metadata_json"] or "{}"))
        except (TypeError, ValueError):
            prior = {}
        return dict(prior) if isinstance(prior, dict) else {}

    @staticmethod
    def _normalize_relation_id(value: Any, *, field: str) -> str:
        normalized = str(value or "").strip()
        if not normalized or len(normalized) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in normalized):
            raise SchemaError(f"timeline_invalid_{field}")
        return normalized

    @staticmethod
    def _json_list(value: Any) -> list[str]:
        try:
            parsed = json.loads(str(value or "[]"))
        except (TypeError, ValueError):
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []

    def add_message(
        self, *, namespace: Namespace, role: str, content: str, timestamp: int, **fields: Any
    ) -> dict[str, Any]:
        with self._lock, self._conn:  # 单库事务:seq 计算 + 插入原子完成
            return self._add_message_locked(
                namespace=namespace,
                role=role,
                content=content,
                timestamp=timestamp,
                fields=fields,
            )

    def _add_message_locked(
        self,
        *,
        namespace: Namespace,
        role: str,
        content: str,
        timestamp: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        source_id = str(fields.get("source_id") or "").strip() or uuid.uuid4().hex
        existing = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (source_id,)).fetchone()
        if existing is not None:  # 幂等:同 namespace 同 source_id 重复写入不产生重复记忆
            self._assert_namespace_owner(existing, namespace, id_label=f"message source_id={source_id!r}")
            return self._row_to_record(existing, "messages")

        scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
        row = self._conn.execute(
            f"SELECT COALESCE(MAX(seq_no), 0) AS m FROM messages WHERE {scope_clause}", scope_params
        ).fetchone()
        seq_no = int(row["m"]) + 1
        role_projection = project_legacy_role(role)
        memory_metadata = fields.get("memory_metadata") if isinstance(fields.get("memory_metadata"), dict) else {}
        kind = str(fields.get("kind") or role_projection["kind"])
        turn_role = (
            str(fields.get("turn_role") or "") if "turn_role" in fields else str(role_projection["turn_role"] or "")
        )
        turn_id = str(fields.get("turn_id") or "")
        if turn_id:
            turn_owner = self._relation_owner_row(turn_id=turn_id)
            if turn_owner is not None:
                self._assert_scope_owner(turn_owner, namespace, id_label=f"turn_id={turn_id!r}")
        annotation_status = str(fields.get("annotation_status") or "")
        if not annotation_status:
            annotation_status = (
                "accepted_host"
                if turn_role == "stimulus"
                and not kind.startswith(("material.", "tool."))
                and _has_semantic_metadata(memory_metadata)
                else "unannotated"
            )
        retrieval_visibility = str(fields.get("retrieval_visibility") or "")
        if not retrieval_visibility:
            retrieval_visibility = "default" if annotation_status.startswith("accepted_") else "explicit"
        trace_metadata = dict(role_projection["trace_metadata"])
        if isinstance(fields.get("trace_metadata"), dict):
            trace_metadata.update(fields["trace_metadata"])
        payload = fields.get("payload")
        if not isinstance(payload, dict):
            payload = {"text": str(content)}
        values = {
            "source_id": source_id,
            "tenant_id": namespace.tenant_id or "",
            "user_id": namespace.user_id,
            "domain_id": namespace.domain_id or "",
            "conversation_id": namespace.conversation_id or "",
            "actor_id": namespace.actor_id(),
            "actor_display_name": namespace.actor.display_name if namespace.actor else "",
            "seq_no": seq_no,
            "role": str(role),
            "content": str(content),
            "timestamp": int(timestamp),
            "date_label": str(fields.get("date_label") or ""),
            "time_of_day": str(fields.get("time_of_day") or ""),
            "memory_metadata_json": _json_dumps(memory_metadata),
            "index_status": "pending",
            "kind": kind,
            "origin": str(fields.get("origin") or role_projection["origin"]),
            "turn_id": turn_id,
            "turn_role": turn_role,
            "reply_to_source_id": str(fields.get("reply_to_source_id") or ""),
            "correlation_id": str(fields.get("correlation_id") or role_projection["correlation_id"]),
            "relation_status": str(fields.get("relation_status") or ("linked" if turn_id else "legacy_unlinked")),
            "target_actor_id": str(fields.get("target_actor_id") or ""),
            "target_actor_display_name": str(fields.get("target_actor_display_name") or ""),
            "payload_json": _json_dumps(payload),
            "semantic_text": str(fields.get("semantic_text") or content),
            "trace_metadata_json": _json_dumps(trace_metadata),
            "annotation_status": annotation_status,
            "annotation_source": str(
                fields.get("annotation_source") or ("host" if annotation_status == "accepted_host" else "")
            ),
            "retrieval_policy": str(fields.get("retrieval_policy") or "auto"),
            "retrieval_visibility": retrieval_visibility,
            "semanticize": int(
                fields.get(
                    "semanticize",
                    turn_role not in {"action", "observation"} and not kind.startswith("material."),
                )
            ),
            "prompt_visible": int(fields.get("prompt_visible", True)),
            "trust": str(fields.get("trust") or "untrusted_data"),
            "renderer_id": str(fields.get("renderer_id") or "canonical"),
            "renderer_version": int(fields.get("renderer_version") or 1),
            "row_version": 1,
            "index_schema_version": int(fields.get("index_schema_version") or 0),
            "index_key": str(fields.get("index_key") or ""),
        }
        columns = tuple(values)
        placeholders = ",".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT INTO messages ({','.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )
        return self._row_to_record(
            self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (source_id,)).fetchone(),
            "messages",
        )

    def add_summary(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._conn:
            return self._add_summary_locked(namespace=namespace, record=record)

    def _add_summary_locked(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        summary_id = str(record.get("summary_id") or "").strip() or uuid.uuid4().hex
        prior = self._conn.execute("SELECT * FROM summaries WHERE summary_id = ?", (summary_id,)).fetchone()
        if prior is not None:  # 同 namespace 内覆盖允许;跨 namespace 拒绝
            self._assert_namespace_owner(prior, namespace, id_label=f"summary_id={summary_id!r}")
        kind = str(record.get("kind") or "memory.episode_summary")
        pure_operation = kind == "memory.operation_digest"
        values = {
            "summary_id": summary_id,
            "tenant_id": namespace.tenant_id or "",
            "user_id": namespace.user_id,
            "domain_id": namespace.domain_id or "",
            "conversation_id": namespace.conversation_id or "",
            "timestamp": int(record.get("timestamp") or 0),
            "period_start_ts": int(record.get("period_start_ts") or 0),
            "period_end_ts": int(record.get("period_end_ts") or 0),
            "date_label": str(record.get("date_label") or ""),
            "time_of_day": str(record.get("time_of_day") or ""),
            "period_label": str(record.get("period_label") or ""),
            "event_type": str(record.get("event_type") or ""),
            "importance": float(record.get("importance") or 0.0),
            "diary_summary": str(record.get("diary_summary") or ""),
            "key_events_json": _json_dumps(record.get("key_events") or []),
            "core_facts_json": _json_dumps(record.get("core_facts") or []),
            "memory_title": str(record.get("memory_title") or ""),
            "catalog_hint": str(record.get("catalog_hint") or ""),
            "topic_headings_json": _json_dumps(record.get("topic_headings") or []),
            "participant_refs_json": _json_dumps(record.get("participant_refs") or []),
            "source_turn_count": max(0, int(record.get("source_turn_count") or 0)),
            "source_entry_count": max(0, int(record.get("source_entry_count") or 0)),
            "catalog_schema_version": max(0, int(record.get("catalog_schema_version") or 0)),
            "semantic_tags_json": _json_dumps(record.get("semantic_tags") or []),
            "memory_metadata_json": _json_dumps(record.get("memory_metadata") or {}),
            "source_ids_json": _json_dumps(record.get("source_ids") or []),
            "is_semanticized": int(record.get("is_semanticized") or 0),
            "semantic_id": str(record.get("semantic_id") or ""),
            "index_status": "pending",
            "kind": kind,
            "trace_metadata_json": _json_dumps(record.get("trace_metadata") or {}),
            "annotation_status": str(record.get("annotation_status") or "derived"),
            "retrieval_visibility": str(
                record.get("retrieval_visibility") or ("explicit" if pure_operation else "default")
            ),
            "semanticize": int(record.get("semanticize", not pure_operation)),
            "lineage_status": str(record.get("lineage_status") or "valid"),
            "compaction_schema_version": int(record.get("compaction_schema_version") or 1),
            "row_version": int(prior["row_version"] if prior is not None else 0) + 1,
            "index_schema_version": int(record.get("index_schema_version") or 0),
            "index_key": str(record.get("index_key") or ""),
        }
        columns = tuple(values)
        placeholders = ",".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO summaries ({','.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )
        return self._row_to_record(
            self._conn.execute("SELECT * FROM summaries WHERE summary_id = ?", (summary_id,)).fetchone(),
            "summaries",
        )

    def add_semantic_summary(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._conn:
            return self._add_semantic_summary_locked(namespace=namespace, record=record)

    def _add_semantic_summary_locked(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        semantic_id = str(record.get("semantic_id") or "").strip() or uuid.uuid4().hex
        ts = int(record.get("timestamp") or 0)
        prior = self._conn.execute("SELECT * FROM semantic_summaries WHERE semantic_id = ?", (semantic_id,)).fetchone()
        if prior is not None:  # 同 namespace 内覆盖(强化合并需要);跨 namespace 拒绝
            self._assert_namespace_owner(prior, namespace, id_label=f"semantic_id={semantic_id!r}")
        values = {
            "semantic_id": semantic_id,
            "tenant_id": namespace.tenant_id or "",
            "user_id": namespace.user_id,
            "domain_id": namespace.domain_id or "",
            "conversation_id": namespace.conversation_id or "",
            "timestamp": ts,
            "period_start_ts": int(record.get("period_start_ts") or 0),
            "period_end_ts": int(record.get("period_end_ts") or 0),
            "date_label": str(record.get("date_label") or ""),
            "time_of_day": str(record.get("time_of_day") or ""),
            "importance": float(record.get("importance") or 0.0),
            "semantic_summary": str(record.get("semantic_summary") or ""),
            "stable_facts_json": _json_dumps(record.get("stable_facts") or []),
            "recurring_topics_json": _json_dumps(record.get("recurring_topics") or []),
            "important_people_json": _json_dumps(record.get("important_people") or []),
            "open_loops_json": _json_dumps(record.get("open_loops") or []),
            "memory_title": str(record.get("memory_title") or ""),
            "catalog_hint": str(record.get("catalog_hint") or ""),
            "topic_headings_json": _json_dumps(record.get("topic_headings") or []),
            "catalog_schema_version": max(0, int(record.get("catalog_schema_version") or 0)),
            "semantic_tags_json": _json_dumps(record.get("semantic_tags") or []),
            "memory_metadata_json": _json_dumps(record.get("memory_metadata") or {}),
            "source_summary_ids_json": _json_dumps(record.get("source_summary_ids") or []),
            "reinforcement_count": int(record.get("reinforcement_count") or 1),
            "last_reinforced_ts": int(record.get("last_reinforced_ts") or ts),
            "index_status": "pending",
            "kind": str(record.get("kind") or "memory.semantic_summary"),
            "annotation_status": str(record.get("annotation_status") or "derived"),
            "retrieval_visibility": str(record.get("retrieval_visibility") or "default"),
            "lineage_status": str(record.get("lineage_status") or "valid"),
            "semantic_schema_version": int(record.get("semantic_schema_version") or 1),
            "row_version": int(prior["row_version"] if prior is not None else 0) + 1,
            "index_schema_version": int(record.get("index_schema_version") or 0),
            "index_key": str(record.get("index_key") or ""),
        }
        columns = tuple(values)
        placeholders = ",".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO semantic_summaries ({','.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )
        return self._row_to_record(
            self._conn.execute("SELECT * FROM semantic_summaries WHERE semantic_id = ?", (semantic_id,)).fetchone(),
            "semantic_summaries",
        )

    def mark_messages_summarized(self, source_ids: list[str], summary_id: str) -> None:
        if not source_ids:
            return
        placeholders = ",".join("?" for _ in source_ids)
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                UPDATE messages
                SET is_summarized = 1, summary_id = ?, row_version = row_version + 1
                WHERE source_id IN ({placeholders})
                """,
                [summary_id, *source_ids],
            )

    def mark_summaries_semanticized(self, summary_ids: list[str], semantic_id: str) -> None:
        if not summary_ids:
            return
        placeholders = ",".join("?" for _ in summary_ids)
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                UPDATE summaries
                SET is_semanticized = 1, semantic_id = ?, row_version = row_version + 1
                WHERE summary_id IN ({placeholders})
                """,
                [semantic_id, *summary_ids],
            )

    def set_index_status(self, source_id: str, status: str) -> None:
        self.set_index_state(source_id, status)

    def set_index_state(
        self,
        source_id: str,
        status: str,
        *,
        index_schema_version: int = 0,
        index_key: str = "",
    ) -> None:
        with self._lock, self._conn:
            for table, id_col in (
                ("messages", "source_id"),
                ("summaries", "summary_id"),
                ("semantic_summaries", "semantic_id"),
            ):
                cur = self._conn.execute(
                    f"""
                    UPDATE {table}
                    SET index_status = ?, index_schema_version = ?, index_key = ?
                    WHERE {id_col} = ?
                    """,
                    (status, int(index_schema_version or 0), str(index_key or ""), source_id),
                )
                if cur.rowcount:
                    return

    def update_message_memory_metadata(
        self, *, namespace: Namespace, source_id: str, memory_metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        sid = str(source_id or "").strip()
        if not sid:
            return None
        with self._lock, self._conn:
            existing = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (sid,)).fetchone()
            if existing is None:
                return None
            self._assert_namespace_owner(existing, namespace, id_label=f"message source_id={sid!r}")
            accepted = _has_semantic_metadata(memory_metadata)
            cur = self._conn.execute(
                """
                UPDATE messages
                SET memory_metadata_json = ?, annotation_status = ?, annotation_source = ?,
                    retrieval_visibility = ?, row_version = row_version + 1,
                    index_status = 'pending'
                WHERE source_id = ?
                """,
                (
                    _json_dumps(memory_metadata or {}),
                    "accepted_model" if accepted else "unannotated",
                    "model" if accepted else "",
                    "default" if accepted else "explicit",
                    sid,
                ),
            )
            if not cur.rowcount:
                return None
            row = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (sid,)).fetchone()
            return self._row_to_record(row, "messages") if row is not None else None

    def stage_message_memory_metadata(
        self, *, namespace: Namespace, source_id: str, memory_metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        sid = str(source_id or "").strip()
        if not sid:
            return None
        with self._lock, self._conn:
            existing = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (sid,)).fetchone()
            if existing is None:
                return None
            self._assert_namespace_owner(existing, namespace, id_label=f"message source_id={sid!r}")
            turn_id = str(existing["turn_id"] or "")
            if str(existing["turn_role"] or "") != TurnRole.STIMULUS.value or not turn_id:
                raise SchemaError("staged_annotation_requires_turn_stimulus")
            turn = self._conn.execute("SELECT status FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
            if turn is None or str(turn["status"] or "") != TurnStatus.OPEN.value:
                raise SchemaError("staged_annotation_requires_open_turn")
            self._conn.execute(
                """
                UPDATE messages
                SET memory_metadata_json = ?, annotation_status = 'unannotated',
                    annotation_source = 'model_staged',
                    retrieval_visibility = CASE
                        WHEN retrieval_policy = 'never' THEN 'never'
                        ELSE 'explicit'
                    END,
                    row_version = row_version + 1,
                    index_status = 'pending', index_schema_version = 0, index_key = ''
                WHERE source_id = ?
                """,
                (_json_dumps(memory_metadata or {}), sid),
            )
            row = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (sid,)).fetchone()
            return self._row_to_record(row, "messages") if row is not None else None

    # --- 读 ---

    def get_record_by_source_id(self, source_id: str) -> dict[str, Any] | None:
        with self._lock:
            for table, id_col in (
                ("messages", "source_id"),
                ("summaries", "summary_id"),
                ("semantic_summaries", "semantic_id"),
            ):
                row = self._conn.execute(f"SELECT * FROM {table} WHERE {id_col} = ?", (source_id,)).fetchone()
                if row is not None:
                    return self._row_to_record(row, table)
        return None

    def get_entry(self, *, namespace: Namespace, source_id: str) -> TimelineEntry | None:
        """Namespace-safe Timeline V2 raw lookup; unlike the legacy global lookup."""

        sid = str(source_id or "").strip()
        if not sid:
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (sid,)).fetchone()
            if row is None:
                return None
            self._assert_scope_owner(row, namespace, id_label=f"message source_id={sid!r}")
            return TimelineEntry.from_record(self._row_to_record(row, "messages"))

    def get_retrieval_record(
        self,
        *,
        namespace: Namespace,
        source_id: str,
        cross_conversation: bool = False,
    ) -> dict[str, Any] | None:
        sid = str(source_id or "").strip()
        if not sid:
            return None
        scope_clause, params = self._scope_clause(
            namespace,
            with_conversation=not bool(cross_conversation),
        )
        with self._lock:
            matches: list[dict[str, Any]] = []
            for table, id_col in (
                ("messages", "source_id"),
                ("summaries", "summary_id"),
                ("semantic_summaries", "semantic_id"),
            ):
                row = self._conn.execute(
                    f"SELECT * FROM {table} WHERE {scope_clause} AND {id_col} = ?",
                    [*params, sid],
                ).fetchone()
                if row is not None:
                    matches.append(self._row_to_record(row, table))
            if len(matches) > 1:
                raise SchemaError("ambiguous_memory_id")
            return matches[0] if matches else None

    def get_entries_by_source_ids(
        self,
        *,
        namespace: Namespace,
        source_ids: tuple[str, ...],
        cross_conversation: bool = False,
    ) -> list[TimelineEntry]:
        requested = tuple(dict.fromkeys(str(item or "").strip() for item in source_ids if str(item or "").strip()))
        if not requested:
            return []
        scope_clause, params = self._scope_clause(
            namespace,
            with_conversation=not bool(cross_conversation),
        )
        placeholders = ",".join("?" for _ in requested)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} AND source_id IN ({placeholders})",
                [*params, *requested],
            ).fetchall()
        by_id = {str(row["source_id"]): TimelineEntry.from_record(self._row_to_record(row, "messages")) for row in rows}
        return [by_id[source_id] for source_id in requested if source_id in by_id]

    def resolve_lineage_source_ids(
        self,
        *,
        namespace: Namespace,
        source_ids: tuple[str, ...],
        cross_conversation: bool = False,
        include_ancestors: bool = True,
    ) -> LineageClosure:
        requested = tuple(dict.fromkeys(str(item or "").strip() for item in source_ids if str(item or "").strip()))
        if not requested:
            return LineageClosure(status="resolved", requested_ids=())
        scope_clause, params = self._scope_clause(
            namespace,
            with_conversation=not bool(cross_conversation),
        )
        descendants: list[str] = []
        descendant_seen = set(requested)
        active: set[str] = set()
        resolved_children: dict[str, tuple[str, ...]] = {}
        failure_reason = "lineage_broken_or_cyclic"

        def read_children(source_id: str) -> tuple[str, ...] | None:
            cached = resolved_children.get(source_id)
            if cached is not None:
                return cached
            for table, id_column, lineage_column in (
                ("messages", "source_id", ""),
                ("summaries", "summary_id", "source_ids_json"),
                ("semantic_summaries", "semantic_id", "source_summary_ids_json"),
            ):
                select = lineage_column or id_column
                row = self._conn.execute(
                    f"SELECT {select} FROM {table} WHERE {scope_clause} AND {id_column} = ?",
                    [*params, source_id],
                ).fetchone()
                if row is None:
                    continue
                children = tuple(self._json_list(row[lineage_column])) if lineage_column else ()
                resolved_children[source_id] = children
                return children
            return None

        def walk_descendants(source_id: str) -> bool:
            nonlocal failure_reason
            if source_id in active:
                return False
            children = read_children(source_id)
            if children is None:
                failure_reason = "lineage_source_missing_or_out_of_scope"
                return False
            active.add(source_id)
            for child in children:
                if child in active:
                    return False
                if child not in descendant_seen:
                    descendant_seen.add(child)
                    descendants.append(child)
                    if not walk_descendants(child):
                        return False
            active.remove(source_id)
            return True

        with self._lock:
            if any(not walk_descendants(source_id) for source_id in requested):
                return LineageClosure(
                    status="invalid",
                    requested_ids=requested,
                    reason=failure_reason,
                )

            parents: dict[str, list[str]] = {}
            if include_ancestors:
                summary_rows = self._conn.execute(
                    f"SELECT summary_id, source_ids_json FROM summaries WHERE {scope_clause}",
                    params,
                ).fetchall()
                semantic_rows = self._conn.execute(
                    f"SELECT semantic_id, source_summary_ids_json FROM semantic_summaries WHERE {scope_clause}",
                    params,
                ).fetchall()
                for row in summary_rows:
                    parent = str(row["summary_id"])
                    for child in self._json_list(row["source_ids_json"]):
                        parents.setdefault(child, []).append(parent)
                for row in semantic_rows:
                    parent = str(row["semantic_id"])
                    for child in self._json_list(row["source_summary_ids_json"]):
                        parents.setdefault(child, []).append(parent)

        ancestors: list[str] = []
        if include_ancestors:
            ancestor_seen = set(requested)
            queue = list(requested)
            while queue:
                child = queue.pop(0)
                for parent in parents.get(child, ()):
                    if parent in ancestor_seen:
                        continue
                    ancestor_seen.add(parent)
                    ancestors.append(parent)
                    queue.append(parent)
        return LineageClosure(
            status="resolved",
            requested_ids=requested,
            descendant_ids=tuple(descendants),
            ancestor_ids=tuple(ancestors),
        )

    def get_turn(self, *, namespace: Namespace, turn_id: str) -> TurnHandle | None:
        normalized = str(turn_id or "").strip()
        if not normalized:
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (normalized,)).fetchone()
            if row is None:
                return None
            self._assert_scope_owner(row, namespace, id_label=f"turn_id={normalized!r}")
            return self._turn_handle_from_row(row)

    def upsert_turn_projection_settlement(
        self,
        *,
        namespace: Namespace,
        settlement: "TurnProjectionSettlement",
    ) -> bool:
        """Idempotently persist one frozen turn settlement (first write wins).

        Returns True when the row was inserted, False when an identical-key row already
        existed (重试幂等: 不覆盖已冻结 settlement, 不会生成不同 hash)。
        """
        with self._lock, self._conn:
            inserted = self._conn.execute(
                """
                INSERT INTO turn_projection_settlement(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                    policy, settlement_status, settlement_schema_version, terminal_source_id,
                    full_projection_hash, settled_projection_hash, first_changed_projection_index,
                    full_projected_tokens, settled_projected_tokens, token_count_quality,
                    reason, settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile
                ) DO NOTHING
                """,
                (
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    str(settlement.turn_id or "").strip(),
                    str(settlement.provider_profile or "").strip(),
                    str(settlement.policy or "full_until_raw_compaction").strip(),
                    str(settlement.settlement_status or "settled").strip(),
                    int(settlement.settlement_schema_version or 1),
                    str(settlement.terminal_source_id or "").strip(),
                    str(settlement.full_projection_hash or "").strip(),
                    str(settlement.settled_projection_hash or "").strip(),
                    int(
                        settlement.first_changed_projection_index
                        if settlement.first_changed_projection_index is not None
                        else -1
                    ),
                    int(settlement.full_projected_tokens or 0),
                    int(settlement.settled_projected_tokens or 0),
                    str(settlement.token_count_quality or "").strip(),
                    str(settlement.reason or "").strip(),
                    int(settlement.settled_at or 0),
                ),
            )
        return inserted.rowcount > 0

    def commit_turn_projection_settlement(
        self,
        *,
        namespace: Namespace,
        settlement: "TurnProjectionSettlement",
        projections: list[ProjectionMessageInput],
    ) -> bool:
        """Atomically publish the settlement row and every settled provider message."""

        if any(not isinstance(item, ProjectionMessageInput) for item in projections):
            raise TypeError("projections must contain ProjectionMessageInput values")
        turn_id = self._normalize_relation_id(str(settlement.turn_id or ""), field="turn_id")
        profile = str(settlement.provider_profile or "").strip().lower()
        policy = str(settlement.policy or "").strip()
        status = str(settlement.settlement_status or "").strip()
        if not profile:
            raise SchemaError("settlement_provider_profile_required")
        if status not in {"settled", "settled_noop", "full_fallback"}:
            raise SchemaError("settlement_status_invalid")
        if status == "settled" and not projections:
            raise SchemaError("settled_projection_rows_required")
        if status != "settled" and projections:
            raise SchemaError("non_settled_projection_rows_not_allowed")
        if projections:
            if any(item.provider_profile != profile for item in projections):
                raise SchemaError("settled_projection_profile_mismatch")
            if [item.projection_index for item in projections] != list(range(len(projections))):
                raise SchemaError("settled_projection_index_sequence_corrupt")
            if stable_projection_hash([item.payload for item in projections]) != str(
                settlement.settled_projection_hash or ""
            ):
                raise SchemaError("settled_projection_set_hash_mismatch")

        with self._lock, self._conn:
            turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
            if turn is None:
                raise SchemaError("turn_not_found")
            self._assert_scope_owner(turn, namespace, id_label=f"turn_id={turn_id!r}")
            if str(turn["status"] or "") != TurnStatus.CLOSED.value:
                raise SchemaError("settlement_turn_must_be_closed")
            if str(turn["operation_projection_policy"] or "full_until_raw_compaction") != policy:
                raise SchemaError("settlement_policy_mismatch")
            frozen_min = int(turn["operation_settlement_min_utf8_bytes"] or 256)
            frozen_ratio = float(turn["operation_settlement_min_saved_ratio"] or 0.5)
            if int(settlement.settlement_min_utf8_bytes or 256) != frozen_min:
                raise SchemaError("settlement_min_utf8_bytes_mismatch")
            if abs(float(settlement.settlement_min_saved_ratio or 0.5) - frozen_ratio) > 1e-9:
                raise SchemaError("settlement_min_saved_ratio_mismatch")
            existing = self._conn.execute(
                """
                SELECT 1 FROM turn_projection_settlement
                WHERE tenant_id = ? AND user_id = ? AND domain_id = ? AND conversation_id = ?
                  AND turn_id = ? AND provider_profile = ?
                """,
                (
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    turn_id,
                    profile,
                ),
            ).fetchone()
            if existing is not None:
                return False

            if projections:
                expected_source_ids = tuple(
                    str(row["source_id"])
                    for row in self._conn.execute(
                        """
                        SELECT source_id FROM messages
                        WHERE turn_id = ? AND prompt_visible = 1
                        ORDER BY seq_no
                        """,
                        (turn_id,),
                    ).fetchall()
                )
                covered_source_ids = tuple(source_id for item in projections for source_id in item.source_ids)
                if covered_source_ids != expected_source_ids:
                    raise SchemaError("settled_projection_source_coverage_mismatch")

            self._conn.execute(
                """
                INSERT INTO turn_projection_settlement(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                    policy, settlement_status, settlement_schema_version, terminal_source_id,
                    full_projection_hash, settled_projection_hash, first_changed_projection_index,
                    full_projected_tokens, settled_projected_tokens, token_count_quality,
                    reason, settled_at, settlement_min_utf8_bytes, settlement_min_saved_ratio,
                    settlement_config_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    turn_id,
                    profile,
                    policy,
                    status,
                    int(settlement.settlement_schema_version or 1),
                    str(settlement.terminal_source_id or "").strip(),
                    str(settlement.full_projection_hash or "").strip(),
                    str(settlement.settled_projection_hash or "").strip(),
                    int(
                        settlement.first_changed_projection_index
                        if settlement.first_changed_projection_index is not None
                        else -1
                    ),
                    int(settlement.full_projected_tokens or 0),
                    int(settlement.settled_projected_tokens or 0),
                    str(settlement.token_count_quality or "").strip(),
                    str(settlement.reason or "").strip(),
                    int(settlement.settled_at or 0),
                    int(settlement.settlement_min_utf8_bytes or 256),
                    float(settlement.settlement_min_saved_ratio or 0.5),
                    str(settlement.settlement_config_hash or "").strip(),
                ),
            )
            settlement_id = f"{turn_id}:{profile}"
            for item in projections:
                self._conn.execute(
                    """
                    INSERT INTO settled_prompt_projection(
                        settlement_id, projection_index, provider_profile,
                        source_ids_json, payload_json, payload_hash, projection_status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'settled')
                    """,
                    (
                        settlement_id,
                        int(item.projection_index),
                        profile,
                        _json_dumps(list(item.source_ids)),
                        _json_dumps(dict(item.payload)),
                        stable_projection_hash(item.payload),
                    ),
                )
        return True

    def get_turn_projection_settlement(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        provider_profile: str,
    ) -> dict[str, Any] | None:
        normalized_turn_id = str(turn_id or "").strip()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM turn_projection_settlement WHERE turn_id = ? AND provider_profile = ?",
                (normalized_turn_id, str(provider_profile or "").strip()),
            ).fetchone()
            if row is None:
                return None
            self._assert_scope_owner(row, namespace, id_label=f"settlement={normalized_turn_id!r}")
            return dict(row)

    def list_settled_projections(self, *, settlement_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM settled_prompt_projection WHERE settlement_id = ? ORDER BY projection_index",
                (str(settlement_id or "").strip(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_turn_projection_settlements(self, *, namespace: Namespace) -> list[dict[str, Any]]:
        scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM turn_projection_settlement WHERE {scope_clause} ORDER BY settled_at",
                tuple(scope_params),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_settled_prompt_projection(
        self,
        *,
        settlement_id: str,
        projection_index: int,
        provider_profile: str,
        message: Any,
        projection_status: str = "settled",
    ) -> None:
        payload = dict(getattr(message, "payload", {}) or {})
        payload_hash = stable_projection_hash(payload)
        source_ids = tuple(getattr(message, "source_ids", ()) or ())
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO settled_prompt_projection(
                    settlement_id, projection_index, provider_profile,
                    source_ids_json, payload_json, payload_hash, projection_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(settlement_id, projection_index) DO NOTHING
                """,
                (
                    str(settlement_id or "").strip(),
                    int(projection_index),
                    str(provider_profile or "").strip(),
                    _json_dumps([str(item) for item in source_ids]),
                    _json_dumps(payload),
                    str(payload_hash or "").strip(),
                    str(projection_status or "settled").strip(),
                ),
            )

    def get_turn_entries(self, *, namespace: Namespace, turn_id: str) -> list[TimelineEntry]:
        """Return one turn in sequence order without allowing cross-owner fallback."""

        normalized = str(turn_id or "").strip()
        if not normalized:
            return []
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            owner = self._relation_owner_row(turn_id=normalized)
            if owner is None:
                return []
            self._assert_scope_owner(owner, namespace, id_label=f"turn_id={normalized!r}")
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} AND turn_id = ? ORDER BY seq_no",
                [*params, normalized],
            ).fetchall()
        return [TimelineEntry.from_record(self._row_to_record(row, "messages")) for row in rows]

    def get_correlation_entries(
        self, *, namespace: Namespace, turn_id: str, correlation_id: str
    ) -> list[TimelineEntry]:
        """Return an ordered action/observation branch under a namespace-safe turn."""

        normalized_turn = str(turn_id or "").strip()
        normalized_correlation = str(correlation_id or "").strip()
        if not normalized_turn or not normalized_correlation:
            return []
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            owner = self._relation_owner_row(turn_id=normalized_turn)
            if owner is None:
                return []
            self._assert_scope_owner(owner, namespace, id_label=f"turn_id={normalized_turn!r}")
            rows = self._conn.execute(
                f"""
                SELECT * FROM messages
                WHERE {scope_clause} AND turn_id = ? AND correlation_id = ?
                ORDER BY seq_no
                """,
                [*params, normalized_turn, normalized_correlation],
            ).fetchall()
        return [TimelineEntry.from_record(self._row_to_record(row, "messages")) for row in rows]

    def list_prompt_visible_entries(self, *, namespace: Namespace) -> list[TimelineEntry]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM messages
                WHERE {scope_clause} AND is_summarized = 0 AND prompt_visible = 1
                ORDER BY seq_no
                """,
                params,
            ).fetchall()
        return [TimelineEntry.from_record(self._row_to_record(row, "messages")) for row in rows]

    def get_turn_projections(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        provider_profile: str,
    ) -> list[ProjectionMessage]:
        normalized_turn = str(turn_id or "").strip()
        normalized_profile = str(provider_profile or "").strip().lower()
        if not normalized_turn or not normalized_profile:
            return []
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            owner = self._relation_owner_row(turn_id=normalized_turn)
            if owner is not None:
                self._assert_scope_owner(owner, namespace, id_label=f"turn_id={normalized_turn!r}")
            rows = self._conn.execute(
                f"""
                SELECT * FROM prompt_projections
                WHERE {scope_clause} AND turn_id = ? AND provider_profile = ?
                ORDER BY projection_index
                """,
                [*params, normalized_turn, normalized_profile],
            ).fetchall()
        return [ProjectionMessage.from_record(self._row_to_record(row, "prompt_projections")) for row in rows]

    def list_projection_audits(
        self,
        *,
        namespace: Namespace,
        turn_id: str = "",
        provider_profile: str = "",
    ) -> list[ProjectionAudit]:
        normalized_turn = str(turn_id or "").strip()
        normalized_profile = str(provider_profile or "").strip().lower()
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        clause = scope_clause
        query_params = list(params)
        with self._lock:
            if normalized_turn:
                owner = self._relation_owner_row(turn_id=normalized_turn)
                if owner is not None:
                    self._assert_scope_owner(owner, namespace, id_label=f"turn_id={normalized_turn!r}")
                clause += " AND turn_id = ?"
                query_params.append(normalized_turn)
            if normalized_profile:
                clause += " AND provider_profile = ?"
                query_params.append(normalized_profile)
            rows = self._conn.execute(
                f"SELECT * FROM projection_audits WHERE {clause} ORDER BY created_at, attempt",
                query_params,
            ).fetchall()
        return [ProjectionAudit.from_record(dict(row)) for row in rows]

    def list_projection_namespaces(self) -> list[tuple[str, str, str, str]]:
        """Distinct (tenant, user, domain, conversation) scopes with frozen projections.

        Hosts use this for explicit pre-traffic maintenance passes such as the
        legacy path-projection migration.
        """

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT tenant_id, user_id, domain_id, conversation_id
                FROM prompt_projections
                ORDER BY tenant_id, user_id, domain_id, conversation_id
                """
            ).fetchall()
        return [
            (
                str(row["tenant_id"] or ""),
                str(row["user_id"] or ""),
                str(row["domain_id"] or ""),
                str(row["conversation_id"] or ""),
            )
            for row in rows
        ]

    def migrate_legacy_path_projections(
        self,
        *,
        namespace: Namespace,
        target_version: int,
        projection_builder: Callable[[str, list[TimelineEntry]], list[ProjectionMessageInput]],
        settlement_builder: Callable[..., Any] | None = None,
        markers: tuple[str, ...] = (),
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Re-project legacy path-damage rows from raw sources and rebuild stale settlements.

        Four legacy damage patterns are scanned and reported separately; only
        records whose raw timeline sources still hold the original values are
        rewritten.  Irrecoverable host-redacted rows are preserved untouched and
        counted.  Stale settlements (marker-containing or full-hash mismatched)
        are rebuilt atomically via the settlement planner; cards are only
        counted as rebuilt when truly re-committed.
        """

        target = int(target_version)
        if target < 1:
            raise ValueError("target_version must be positive")
        resolved_markers = tuple(str(item or "") for item in markers)
        marker_names = (
            "scanned_memcore_marker_rows",
            "scanned_host_local_path_rows",
            "scanned_host_tmpdir_rows",
            "scanned_memcore_secret_marker_rows",
        )
        report: dict[str, Any] = {
            "status": "dry_run" if dry_run else "ok",
            "target_projection_version": target,
            **{key: 0 for key in marker_names},
            "migrated": 0,
            "version_advanced_only": 0,
            "preserved_irrecoverable_host_redaction": 0,
            "preserved_without_raw_source": 0,
            "preserved_shape_mismatch": 0,
            "preserved_reprojection_failed": 0,
            "settled_rebuilt": 0,
            "settled_rebuilt_noop": 0,
            "settled_rebuilt_fallback": 0,
            "settled_rebuild_failed_dropped": 0,
            "dry_run": bool(dry_run),
        }
        if len(resolved_markers) > len(marker_names):
            raise ValueError("markers must contain at most four legacy patterns")
        with self._lock:
            with self._immediate_transaction():
                scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)

                # Read-only scan counts per legacy damage pattern.
                for marker, key in zip(resolved_markers, marker_names):
                    matched = self._conn.execute(
                        f"""
                        SELECT COUNT(*) FROM prompt_projections
                        WHERE {scope_clause} AND projection_version < ? AND payload_json LIKE ?
                        """,
                        [*scope_params, target, f"%{marker}%"],
                    ).fetchone()
                    report[key] = int(matched[0] or 0)

                candidate_rows: list[sqlite3.Row] = []
                seen_projection_ids: set[str] = set()
                for marker in resolved_markers:
                    for row in self._conn.execute(
                        f"""
                        SELECT * FROM prompt_projections
                        WHERE {scope_clause} AND projection_version < ? AND payload_json LIKE ?
                        """,
                        [*scope_params, target, f"%{marker}%"],
                    ).fetchall():
                        projection_id = str(row["projection_id"] or "")
                        if projection_id in seen_projection_ids:
                            continue
                        seen_projection_ids.add(projection_id)
                        candidate_rows.append(row)

                changed_turn_profiles: set[tuple[str, str]] = set()
                by_turn: dict[str, list[sqlite3.Row]] = {}
                for row in candidate_rows:
                    by_turn.setdefault(str(row["turn_id"] or "").strip(), []).append(row)
                for turn_id, turn_rows in by_turn.items():
                    entries = self.get_turn_entries(namespace=namespace, turn_id=turn_id)
                    entry_by_source = {str(entry.source_id): entry for entry in entries}
                    projectable = [entry for entry in entries if entry.prompt_visible]

                    def raw_contains_host_damage(source_ids: tuple[str, ...]) -> bool:
                        for source_id in source_ids:
                            entry = entry_by_source.get(str(source_id))
                            if entry is None:
                                continue
                            raw_text = f"{entry.semantic_text}\n{json.dumps(dict(entry.payload), ensure_ascii=False, sort_keys=True)}"
                            if any(damage in raw_text for damage in ("[local_path]", "$TMPDIR")):
                                return True
                        return False

                    profiles = sorted(
                        {
                            str(row["provider_profile"] or "").strip()
                            for row in turn_rows
                            if str(row["provider_profile"] or "").strip()
                        }
                    )
                    for profile in profiles:
                        profile_rows = [
                            row for row in turn_rows if str(row["provider_profile"] or "").strip() == profile
                        ]
                        try:
                            inputs = list(projection_builder(profile, projectable))
                        except Exception:
                            report["preserved_reprojection_failed"] += len(profile_rows)
                            continue
                        input_by_index = {int(item.projection_index): item for item in inputs}
                        for row in profile_rows:
                            projection_id = str(row["projection_id"] or "")
                            index = int(row["projection_index"] or 0)
                            row_source_ids = tuple(self._json_list(row["source_ids_json"]))
                            if any(source_id not in entry_by_source for source_id in row_source_ids):
                                report["preserved_without_raw_source"] += 1
                                continue
                            input_item = input_by_index.get(index)
                            if input_item is None or tuple(input_item.source_ids) != row_source_ids:
                                report["preserved_shape_mismatch"] += 1
                                continue
                            # Raw sources already contain host-redacted text: the
                            # original values no longer exist anywhere. Preserve
                            # untouched and report; never guess.
                            if raw_contains_host_damage(row_source_ids):
                                report["preserved_irrecoverable_host_redaction"] += 1
                                continue
                            old_payload = json.loads(str(row["payload_json"] or "{}"))
                            new_payload = dict(input_item.payload)
                            new_status = input_item.projection_status
                            if str(row["projection_status"] or "") == ProjectionStatus.REQUEST_FROZEN.value:
                                new_status = merge_projection_status(
                                    new_status,
                                    ProjectionStatus.REQUEST_FROZEN,
                                )
                            if canonical_json_bytes(new_payload) == canonical_json_bytes(old_payload):
                                if not dry_run:
                                    self._conn.execute(
                                        "UPDATE prompt_projections SET projection_version = ? WHERE projection_id = ?",
                                        (target, projection_id),
                                    )
                                report["version_advanced_only"] += 1
                            else:
                                if not dry_run:
                                    self._conn.execute(
                                        """
                                        UPDATE prompt_projections
                                        SET payload_json = ?, payload_hash = ?, projection_status = ?,
                                            projection_version = ?
                                        WHERE projection_id = ?
                                        """,
                                        (
                                            _json_dumps(new_payload),
                                            stable_projection_hash(new_payload),
                                            str(new_status),
                                            target,
                                            projection_id,
                                        ),
                                    )
                                report["migrated"] += 1
                            changed_turn_profiles.add((turn_id, profile))

                if settlement_builder is not None or changed_turn_profiles:
                    self._migrate_stale_settlements_locked(
                        namespace=namespace,
                        settlement_builder=settlement_builder,
                        changed_turn_profiles=changed_turn_profiles,
                        markers=resolved_markers,
                        dry_run=dry_run,
                        report=report,
                    )

                if not dry_run and (
                    report["migrated"]
                    or report["version_advanced_only"]
                    or report["settled_rebuilt"]
                    or report["settled_rebuilt_noop"]
                    or report["settled_rebuilt_fallback"]
                    or report["settled_rebuild_failed_dropped"]
                ):
                    self._increment_projection_generation_locked(namespace)
        return report

    def _migrate_stale_settlements_locked(
        self,
        *,
        namespace: Namespace,
        settlement_builder: Callable[..., Any] | None,
        changed_turn_profiles: set[tuple[str, str]],
        markers: tuple[str, ...],
        dry_run: bool,
        report: dict[str, Any],
    ) -> None:
        """Rebuild settlements that are marker-containing or whose full hash changed."""

        scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
        candidate_pairs: set[tuple[str, str]] = set(changed_turn_profiles)
        for marker in markers:
            rows = self._conn.execute(
                """
                SELECT DISTINCT t.turn_id, t.provider_profile
                FROM settled_prompt_projection s
                JOIN turn_projection_settlement t
                  ON t.turn_id || ':' || t.provider_profile = s.settlement_id
                WHERE t.tenant_id = ? AND t.user_id = ? AND t.domain_id = ? AND t.conversation_id = ?
                  AND s.payload_json LIKE ?
                """,
                (
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    f"%{marker}%",
                ),
            ).fetchall()
            for row in rows:
                turn_id = str(row["turn_id"] or "").strip()
                profile = str(row["provider_profile"] or "").strip()
                if turn_id and profile:
                    candidate_pairs.add((turn_id, profile))
        if not candidate_pairs:
            return

        for turn_id, profile in sorted(candidate_pairs):
            existing = self._conn.execute(
                f"""
                SELECT * FROM turn_projection_settlement
                WHERE {scope_clause} AND turn_id = ? AND provider_profile = ?
                """,
                [*scope_params, turn_id, profile],
            ).fetchone()
            if existing is None:
                continue
            full_messages = [
                ProjectionMessage.from_record(record)
                for record in (
                    self._row_to_record(row, "prompt_projections")
                    for row in self._conn.execute(
                        f"""
                        SELECT * FROM prompt_projections
                        WHERE {scope_clause} AND turn_id = ? AND provider_profile = ?
                        ORDER BY projection_index
                        """,
                        [*scope_params, turn_id, profile],
                    ).fetchall()
                )
            ]
            if not full_messages:
                # No frozen ledger left for this turn; the stale settlement
                # cannot be rebuilt and must not keep serving old content.
                self._drop_settlement_locked(namespace, turn_id, profile, dry_run=dry_run)
                report["settled_rebuild_failed_dropped"] += 1
                continue
            current_full_hash = stable_projection_hash([message.payload for message in full_messages])
            stale_by_hash = str(existing["full_projection_hash"] or "") != current_full_hash
            settlement_id = f"{turn_id}:{profile}"
            stale_by_marker = any(
                marker in str(row["payload_json"] or "")
                for row in self._conn.execute(
                    "SELECT payload_json FROM settled_prompt_projection WHERE settlement_id = ?",
                    (settlement_id,),
                ).fetchall()
                for marker in markers
            )
            if not stale_by_hash and not stale_by_marker:
                continue
            if settlement_builder is None:
                self._drop_settlement_locked(namespace, turn_id, profile, dry_run=dry_run)
                report["settled_rebuild_failed_dropped"] += 1
                continue
            entries = self.get_turn_entries(namespace=namespace, turn_id=turn_id)
            turn_row = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
            frozen_min = int(turn_row["operation_settlement_min_utf8_bytes"] or 256) if turn_row is not None else 256
            frozen_ratio = (
                float(turn_row["operation_settlement_min_saved_ratio"] or 0.5) if turn_row is not None else 0.5
            )
            frozen_policy = (
                str(turn_row["operation_projection_policy"] or "compact_after_terminal").strip()
                if turn_row is not None
                else "compact_after_terminal"
            )
            plan = None
            try:
                plan = settlement_builder(
                    namespace,
                    turn_id,
                    profile,
                    entries,
                    full_messages,
                    policy=frozen_policy,
                    min_inline_bytes=frozen_min,
                    required_savings_ratio=frozen_ratio,
                )
            except Exception:
                plan = None
            if plan is None:
                self._drop_settlement_locked(namespace, turn_id, profile, dry_run=dry_run)
                report["settled_rebuild_failed_dropped"] += 1
                continue
            status = str(getattr(plan, "settlement_status", "") or "").strip()
            messages = list(getattr(plan, "messages", ()) or ()) if status == "settled" else []
            if status == "settled" and not messages:
                status = "settled_noop"
            if status not in {"settled", "settled_noop", "full_fallback"}:
                self._drop_settlement_locked(namespace, turn_id, profile, dry_run=dry_run)
                report["settled_rebuild_failed_dropped"] += 1
                continue
            # A marker can be irrecoverable because it already exists in the
            # raw timeline source (for example an old host-side ``[local_path]``
            # rewrite).  Replanning then produces the exact same settlement.
            # Do not rewrite/count that row on every maintenance run: only a
            # changed full ledger, status, or settled hash is a real rebuild.
            if (
                not stale_by_hash
                and str(existing["settlement_status"] or "").strip() == status
                and str(existing["settled_projection_hash"] or "").strip()
                == str(getattr(plan, "settled_projection_hash", "") or "").strip()
            ):
                continue
            if not self._validate_settlement_plan_locked(namespace, existing, plan, messages):
                self._drop_settlement_locked(namespace, turn_id, profile, dry_run=dry_run)
                report["settled_rebuild_failed_dropped"] += 1
                continue
            if dry_run:
                self._count_planned_settlement(status, report)
                continue
            self._replace_settlement_locked(namespace, existing, plan, messages, status)
            self._count_planned_settlement(status, report)

    @staticmethod
    def _count_planned_settlement(status: str, report: dict[str, Any]) -> None:
        if status == "settled":
            report["settled_rebuilt"] += 1
        elif status == "settled_noop":
            report["settled_rebuilt_noop"] += 1
        else:
            report["settled_rebuilt_fallback"] += 1

    def _drop_settlement_locked(self, namespace: Namespace, turn_id: str, profile: str, *, dry_run: bool) -> None:
        if dry_run:
            return
        scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
        settlement_id = f"{turn_id}:{profile}"
        self._conn.execute(
            "DELETE FROM settled_prompt_projection WHERE settlement_id = ?",
            (settlement_id,),
        )
        self._conn.execute(
            f"""
            DELETE FROM turn_projection_settlement
            WHERE {scope_clause} AND turn_id = ? AND provider_profile = ?
            """,
            [*scope_params, turn_id, profile],
        )

    def _validate_settlement_plan_locked(
        self,
        namespace: Namespace,
        existing: sqlite3.Row,
        plan: Any,
        messages: list[ProjectionMessageInput],
    ) -> bool:
        turn_id = str(existing["turn_id"] or "")
        profile = str(existing["provider_profile"] or "")
        policy = str(existing["policy"] or "").strip()
        turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if turn is None:
            return False
        self._assert_scope_owner(turn, namespace, id_label=f"turn_id={turn_id!r}")
        if str(turn["status"] or "") != TurnStatus.CLOSED.value:
            return False
        if str(turn["operation_projection_policy"] or "full_until_raw_compaction") != policy:
            return False
        frozen_min = int(turn["operation_settlement_min_utf8_bytes"] or 256)
        frozen_ratio = float(turn["operation_settlement_min_saved_ratio"] or 0.5)
        plan_min = int(getattr(plan, "settlement_min_utf8_bytes", 256) or 256)
        plan_ratio = float(getattr(plan, "settlement_min_saved_ratio", 0.5) or 0.5)
        if plan_min != frozen_min or abs(plan_ratio - frozen_ratio) > 1e-9:
            return False
        # V6 之前的旧行没有 config hash; 只对已有非空 hash 的行做一致性校验,
        # 旧行由重建路径用冻结配置补齐后写回 plan 的 hash。
        existing_config_hash = str(existing["settlement_config_hash"] or "").strip()
        if (
            existing_config_hash
            and existing_config_hash != str(getattr(plan, "settlement_config_hash", "") or "").strip()
        ):
            return False
        if str(getattr(plan, "provider_profile", "") or "").strip() != profile:
            return False
        if not messages:
            return True
        if [int(item.projection_index) for item in messages] != list(range(len(messages))):
            return False
        if any(str(item.provider_profile or "").strip() != profile for item in messages):
            return False
        if stable_projection_hash([item.payload for item in messages]) != str(
            getattr(plan, "settled_projection_hash", "") or ""
        ):
            return False
        expected_source_ids = tuple(
            str(row["source_id"])
            for row in self._conn.execute(
                "SELECT source_id FROM messages WHERE turn_id = ? AND prompt_visible = 1 ORDER BY seq_no",
                (turn_id,),
            ).fetchall()
        )
        covered_source_ids = tuple(source_id for item in messages for source_id in item.source_ids)
        return covered_source_ids == expected_source_ids

    def _replace_settlement_locked(
        self,
        namespace: Namespace,
        existing: sqlite3.Row,
        plan: Any,
        messages: list[ProjectionMessageInput],
        status: str,
    ) -> None:
        scope_clause, scope_params = self._scope_clause(namespace, with_conversation=True)
        turn_id = str(existing["turn_id"] or "")
        profile = str(existing["provider_profile"] or "")
        settlement_id = f"{turn_id}:{profile}"
        self._conn.execute(
            "DELETE FROM settled_prompt_projection WHERE settlement_id = ?",
            (settlement_id,),
        )
        self._conn.execute(
            f"""
            DELETE FROM turn_projection_settlement
            WHERE {scope_clause} AND turn_id = ? AND provider_profile = ?
            """,
            [*scope_params, turn_id, profile],
        )
        self._conn.execute(
            """
            INSERT INTO turn_projection_settlement(
                tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                policy, settlement_status, settlement_schema_version, terminal_source_id,
                full_projection_hash, settled_projection_hash, first_changed_projection_index,
                full_projected_tokens, settled_projected_tokens, token_count_quality,
                reason, settled_at, settlement_min_utf8_bytes, settlement_min_saved_ratio,
                settlement_config_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                namespace.tenant_id or "",
                namespace.user_id,
                namespace.domain_id or "",
                namespace.conversation_id or "",
                turn_id,
                profile,
                str(existing["policy"] or "").strip(),
                status,
                int(existing["settlement_schema_version"] or 1),
                str(existing["terminal_source_id"] or "").strip(),
                str(getattr(plan, "full_projection_hash", "") or "").strip(),
                str(getattr(plan, "settled_projection_hash", "") or "").strip(),
                int(
                    getattr(plan, "first_changed_projection_index", -1)
                    if getattr(plan, "first_changed_projection_index", None) is not None
                    else -1
                ),
                int(getattr(plan, "full_projected_tokens", 0) or 0),
                int(getattr(plan, "settled_projected_tokens", 0) or 0),
                str(getattr(plan, "token_count_quality", "") or "").strip(),
                str(getattr(plan, "reason", "") or "").strip(),
                int(existing["settled_at"] or 0),
                int(getattr(plan, "settlement_min_utf8_bytes", 256) or 256),
                float(getattr(plan, "settlement_min_saved_ratio", 0.5) or 0.5),
                str(getattr(plan, "settlement_config_hash", "") or "").strip(),
            ),
        )
        for item in messages:
            self._conn.execute(
                """
                INSERT INTO settled_prompt_projection(
                    settlement_id, projection_index, provider_profile,
                    source_ids_json, payload_json, payload_hash, projection_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'settled')
                """,
                (
                    settlement_id,
                    int(item.projection_index),
                    profile,
                    _json_dumps(list(item.source_ids)),
                    _json_dumps(dict(item.payload)),
                    stable_projection_hash(item.payload),
                ),
            )

    def get_conversation_generations(self, *, namespace: Namespace) -> tuple[int, int]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            row = self._conn.execute(
                f"SELECT compaction_generation, projection_generation FROM conversation_states WHERE {scope_clause}",
                params,
            ).fetchone()
        if row is None:
            return (0, 0)
        return (int(row["compaction_generation"] or 0), int(row["projection_generation"] or 0))

    def _relation_owner_row(self, *, turn_id: str) -> sqlite3.Row | None:
        turn = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if turn is not None:
            return turn
        return self._conn.execute(
            "SELECT * FROM messages WHERE turn_id = ? ORDER BY seq_no LIMIT 1", (turn_id,)
        ).fetchone()

    def get_context_slice(self, *, namespace: Namespace, seq_no: int, window: int) -> list[dict[str, Any]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} AND seq_no BETWEEN ? AND ? ORDER BY seq_no",
                [*params, int(seq_no) - int(window), int(seq_no) + int(window)],
            ).fetchall()
        return [self._row_to_record(r, "messages") for r in rows]

    def get_raw_turn_window(
        self,
        *,
        namespace: Namespace,
        anchor_source_id: str,
        before_turns: int,
        after_turns: int,
    ) -> RawTurnWindow:
        anchor_id = str(anchor_source_id or "").strip()
        before = max(0, int(before_turns))
        after = max(0, int(after_turns))
        if not anchor_id:
            return RawTurnWindow(status="invalid", reason="anchor_source_id_required")
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            anchor = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} AND source_id = ?",
                [*params, anchor_id],
            ).fetchone()
            if anchor is None:
                return RawTurnWindow(
                    status="empty",
                    anchor_source_id=anchor_id,
                    before_turns=before,
                    after_turns=after,
                    reason="anchor_not_found_or_out_of_scope",
                )
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} ORDER BY seq_no",
                params,
            ).fetchall()

        groups: list[list[sqlite3.Row]] = []
        group_indexes: dict[tuple[str, str], int] = {}
        anchor_group = -1
        for row in rows:
            turn_id = str(row["turn_id"] or "").strip()
            key = ("turn", turn_id) if turn_id else ("source", str(row["source_id"]))
            group_index = group_indexes.get(key)
            if group_index is None:
                group_index = len(groups)
                group_indexes[key] = group_index
                groups.append([])
            groups[group_index].append(row)
            if str(row["source_id"]) == anchor_id:
                anchor_group = group_index

        if anchor_group < 0:
            return RawTurnWindow(
                status="empty",
                anchor_source_id=anchor_id,
                before_turns=before,
                after_turns=after,
                reason="anchor_not_found_or_out_of_scope",
            )
        start = max(0, anchor_group - before)
        end = min(len(groups), anchor_group + after + 1)
        selected_rows = [row for group in groups[start:end] for row in group]
        entries = tuple(TimelineEntry.from_record(self._row_to_record(row, "messages")) for row in selected_rows)
        return RawTurnWindow(
            status="found" if entries else "empty",
            entries=entries,
            anchor_source_id=anchor_id,
            before_turns=before,
            after_turns=after,
            reason="" if entries else "no_entries",
        )

    def get_unsummarized_messages(self, *, namespace: Namespace) -> list[dict[str, Any]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope_clause} AND is_summarized = 0 ORDER BY seq_no", params
            ).fetchall()
        return [self._row_to_record(r, "messages") for r in rows]

    def get_first_message_timestamp(self, *, namespace: Namespace, cross_conversation: bool = True) -> int | None:
        scope_clause, params = self._scope_clause(namespace, with_conversation=not cross_conversation)
        with self._lock:
            row = self._conn.execute(
                f"SELECT MIN(timestamp) AS m FROM messages WHERE {scope_clause}", params
            ).fetchone()
        return int(row["m"]) if row is not None and row["m"] is not None else None

    def get_messages_by_time_range(
        self,
        *,
        namespace: Namespace,
        start_ts: int | None = None,
        end_ts: int | None = None,
        time_periods: list[str] | None = None,
        cross_conversation: bool = False,
    ) -> list[dict[str, Any]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=not cross_conversation)
        clause = scope_clause
        params = list(params)
        if start_ts is not None:
            clause += " AND timestamp >= ?"
            params.append(int(start_ts))
        if end_ts is not None:
            clause += " AND timestamp < ?"
            params.append(int(end_ts))
        periods = [str(p) for p in (time_periods or []) if str(p or "").strip()]
        if periods:
            clause += f" AND time_of_day IN ({','.join('?' for _ in periods)})"
            params.extend(periods)
        with self._lock:
            rows = self._conn.execute(
                # 按真实时间戳排序:seq_no 是写入序且按会话重置,回填/跨会话会乱序。
                f"SELECT * FROM messages WHERE {clause} "
                "ORDER BY timestamp ASC, conversation_id ASC, seq_no ASC, source_id ASC",
                params,
            ).fetchall()
        return [self._row_to_record(r, "messages") for r in rows]

    def get_visible_episodic_summaries(
        self, *, namespace: Namespace, limit: int, cross_conversation: bool = False
    ) -> list[dict[str, Any]]:
        # 默认按 conversation;cross_conversation=True 时跨会话看同一用户(visible_memory_scope='user')。
        scope_clause, params = self._scope_clause(namespace, with_conversation=not cross_conversation)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM summaries WHERE {scope_clause} AND is_semanticized = 0 ORDER BY timestamp DESC LIMIT ?",
                [*params, int(limit)],
            ).fetchall()
        return [self._row_to_record(r, "summaries") for r in rows]

    def get_episodic_summaries_by_time_range(
        self,
        *,
        namespace: Namespace,
        start_ts: int,
        end_ts: int,
        cross_conversation: bool = False,
        include_explicit: bool = False,
    ) -> list[dict[str, Any]]:
        if isinstance(start_ts, bool) or isinstance(end_ts, bool):
            raise ValueError("catalog_time_range_requires_integer_timestamps")
        try:
            resolved_start = int(start_ts)
            resolved_end = int(end_ts)
        except (TypeError, ValueError) as exc:
            raise ValueError("catalog_time_range_requires_integer_timestamps") from exc
        if resolved_start < 0 or resolved_end <= resolved_start:
            raise ValueError("catalog_time_range_invalid")
        scope_clause, params = self._scope_clause(
            namespace,
            with_conversation=not bool(cross_conversation),
        )
        visibility_clause = "" if include_explicit else " AND retrieval_visibility = 'default'"
        effective_start = "CASE WHEN period_start_ts > 0 THEN period_start_ts ELSE timestamp END"
        effective_end = "CASE WHEN period_end_ts > 0 THEN period_end_ts ELSE timestamp END"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM summaries
                WHERE {scope_clause}
                  AND kind = 'memory.episode_summary'
                  {visibility_clause}
                  AND {effective_end} >= ?
                  AND {effective_start} < ?
                ORDER BY {effective_start} ASC, timestamp ASC, summary_id ASC
                """,
                [*params, resolved_start, resolved_end],
            ).fetchall()
        return [self._row_to_record(row, "summaries") for row in rows]

    def get_semantic_summaries_by_time_range(
        self,
        *,
        namespace: Namespace,
        start_ts: int,
        end_ts: int,
        cross_conversation: bool = False,
        include_explicit: bool = False,
    ) -> list[dict[str, Any]]:
        if isinstance(start_ts, bool) or isinstance(end_ts, bool):
            raise ValueError("catalog_time_range_requires_integer_timestamps")
        try:
            resolved_start = int(start_ts)
            resolved_end = int(end_ts)
        except (TypeError, ValueError) as exc:
            raise ValueError("catalog_time_range_requires_integer_timestamps") from exc
        if resolved_start < 0 or resolved_end <= resolved_start:
            raise ValueError("catalog_time_range_invalid")
        scope_clause, params = self._scope_clause(
            namespace,
            with_conversation=not bool(cross_conversation),
        )
        visibility_clause = "" if include_explicit else " AND retrieval_visibility = 'default'"
        effective_start = "CASE WHEN period_start_ts > 0 THEN period_start_ts ELSE timestamp END"
        effective_end = "CASE WHEN period_end_ts > 0 THEN period_end_ts ELSE timestamp END"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM semantic_summaries
                WHERE {scope_clause}
                  {visibility_clause}
                  AND {effective_end} >= ?
                  AND {effective_start} < ?
                ORDER BY {effective_start} ASC, timestamp ASC, semantic_id ASC
                """,
                [*params, resolved_start, resolved_end],
            ).fetchall()
        return [self._row_to_record(row, "semantic_summaries") for row in rows]

    def get_catalog_raw_coverage(
        self,
        *,
        namespace: Namespace,
        start_ts: int,
        end_ts: int,
        cross_conversation: bool = False,
    ) -> dict[str, Any]:
        if isinstance(start_ts, bool) or isinstance(end_ts, bool):
            raise ValueError("catalog_time_range_requires_integer_timestamps")
        resolved_start = int(start_ts)
        resolved_end = int(end_ts)
        if resolved_start < 0 or resolved_end <= resolved_start:
            raise ValueError("catalog_time_range_invalid")
        scope_clause, params = self._scope_clause(
            namespace,
            with_conversation=not bool(cross_conversation),
        )
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT conversation_id,
                       CASE
                         WHEN is_summarized = 0 THEN 'live'
                         WHEN summary_id != '' AND EXISTS (
                           SELECT 1 FROM summaries s
                           WHERE s.summary_id = messages.summary_id
                             AND s.tenant_id = messages.tenant_id
                             AND s.user_id = messages.user_id
                             AND s.domain_id = messages.domain_id
                             AND s.conversation_id = messages.conversation_id
                         ) THEN 'covered'
                         ELSE 'gap'
                       END AS coverage_state,
                       MIN(timestamp) AS start_ts,
                       MAX(timestamp) + 1 AS end_ts,
                       COUNT(*) AS source_count
                FROM messages
                WHERE {scope_clause} AND timestamp >= ? AND timestamp < ?
                GROUP BY conversation_id, coverage_state
                ORDER BY start_ts, conversation_id, coverage_state
                """,
                [*params, resolved_start, resolved_end],
            ).fetchall()
        result: dict[str, Any] = {
            "total_source_count": 0,
            "covered_source_count": 0,
            "live_source_count": 0,
            "gap_source_count": 0,
            "live_raw_intervals": [],
            "gap_intervals": [],
        }
        for row in rows:
            state = str(row["coverage_state"])
            count = int(row["source_count"] or 0)
            result["total_source_count"] += count
            result[f"{state}_source_count"] += count
            if state in {"live", "gap"}:
                target = "live_raw_intervals" if state == "live" else "gap_intervals"
                result[target].append(
                    {
                        "conversation_id": str(row["conversation_id"] or ""),
                        "start_ts": max(resolved_start, int(row["start_ts"] or resolved_start)),
                        "end_ts": min(resolved_end, int(row["end_ts"] or resolved_end)),
                        "source_count": count,
                        "interval_quality": "activity_envelope",
                    }
                )
        return result

    def get_recent_semantic_summaries(
        self, *, namespace: Namespace, limit: int | None = None, cross_conversation: bool = False
    ) -> list[dict[str, Any]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=not cross_conversation)
        sql = (
            f"SELECT * FROM semantic_summaries WHERE {scope_clause} "
            "ORDER BY last_reinforced_ts DESC, importance DESC, timestamp DESC"
        )
        if limit is not None:  # limit=None 取全部(供衰减排序对完整候选集生效,不预截断)
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r, "semantic_summaries") for r in rows]

    def get_uncompacted_episodic_summaries(
        self, *, namespace: Namespace, limit: int | None = None
    ) -> list[dict[str, Any]]:
        # 压缩始终按会话(与展示作用域无关),最老在前。
        scope_clause, params = self._scope_clause(namespace, with_conversation=True)
        sql = (
            f"SELECT * FROM summaries WHERE {scope_clause} "
            "AND is_semanticized = 0 AND semanticize = 1 ORDER BY timestamp ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r, "summaries") for r in rows]

    def list_pending_index(self, *, limit: int = 100) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for table in ("messages", "summaries", "semantic_summaries"):
                rows = self._conn.execute(
                    f"SELECT * FROM {table} WHERE index_status = 'pending' LIMIT ?", (int(limit),)
                ).fetchall()
                out.extend(self._row_to_record(r, table) for r in rows)
        return out

    def list_index_records(
        self, *, namespace: Namespace, limit: int | None = None, with_conversation: bool = False
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        max_rows = None if limit is None else max(0, int(limit))
        if max_rows == 0:
            return []
        scope_clause, params = self._scope_clause(namespace, with_conversation=with_conversation)
        tables = (
            ("messages", "ORDER BY timestamp ASC, conversation_id, seq_no"),
            ("summaries", "ORDER BY timestamp ASC, conversation_id, summary_id"),
            ("semantic_summaries", "ORDER BY timestamp ASC, conversation_id, semantic_id"),
        )
        with self._lock:
            for table, order_by in tables:
                if max_rows is not None:
                    remaining = max_rows - len(out)
                    if remaining <= 0:
                        break
                    rows = self._conn.execute(
                        f"SELECT * FROM {table} WHERE {scope_clause} AND index_status != 'skipped' {order_by} LIMIT ?",
                        [*params, remaining],
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT * FROM {table} WHERE {scope_clause} AND index_status != 'skipped' {order_by}",
                        params,
                    ).fetchall()
                out.extend(self._row_to_record(r, table) for r in rows)
        return out

    # --- 遗忘 / 合规 ---

    def delete_namespace(self, *, namespace: Namespace) -> list[str]:
        """定向遗忘:删该硬隔离边界下全部记忆,返回被删的 source_id(供上层同步清 VectorIndex)。"""
        scope_clause, params = self._scope_clause(namespace)
        deleted_ids: list[str] = []
        with self._lock, self._conn:
            for table, id_col in (
                ("messages", "source_id"),
                ("summaries", "summary_id"),
                ("semantic_summaries", "semantic_id"),
            ):
                rows = self._conn.execute(
                    f"SELECT {id_col} AS sid FROM {table} WHERE {scope_clause}", params
                ).fetchall()
                deleted_ids.extend(str(r["sid"]) for r in rows)
                self._conn.execute(f"DELETE FROM {table} WHERE {scope_clause}", params)
            for table in ("turns", "conversation_states", "prompt_projections", "projection_audits"):
                self._conn.execute(f"DELETE FROM {table} WHERE {scope_clause}", params)
        return deleted_ids
