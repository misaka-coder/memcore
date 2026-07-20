"""SQLiteMemoryStore —— MemoryStore 的默认实现(关系型真相源)。

三张记忆真相表:messages / summaries / semantic_summaries；
Timeline V2 另有 turn / projection / conversation coordination 表，不建立第二套 raw 真相源。

- 隔离采用 Namespace 五层(tenant/user/domain 硬隔离 + conversation 窗口 + actor 软标签)。
- index_status outbox 状态机(pending → indexed),向量 upsert 失败保持 pending,由 reindex 补做。
- 写入用单库事务保证原子;向量 upsert 由上层在事务外做。

时间字段(date_label / time_of_day)由调用方按 tz 算好后传入;store 不做时区换算。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any

from ..errors import NamespaceError
from ..namespace import Namespace
from .base import MemoryStore
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
        "semantic_tags_json": "semantic_tags",
        "memory_metadata_json": "memory_metadata",
        "source_summary_ids_json": "source_summary_ids",
    },
    "turns": {
        "stimulus_source_ids_json": "stimulus_source_ids",
        "annotation_target_ids_json": "annotation_target_ids",
    },
}
_ENTRY_TYPE = {"messages": "raw", "summaries": "summary", "semantic_summaries": "semantic_summary"}
_JSON_OBJECT_FIELDS = frozenset({"memory_metadata", "payload", "trace_metadata"})


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _has_semantic_metadata(value: Any) -> bool:
    metadata = value if isinstance(value, dict) else {}
    for key in ("keywords", "subject_scopes", "categories", "mood_tags"):
        if list(metadata.get(key) or []):
            return True
    for key in ("importance", "confidence"):
        try:
            if float(metadata.get(key) or 0.0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = ":memory:") -> None:
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

    def close(self) -> None:
        with self._lock:
            self._conn.close()

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

    def add_message(
        self, *, namespace: Namespace, role: str, content: str, timestamp: int, **fields: Any
    ) -> dict[str, Any]:
        source_id = str(fields.get("source_id") or "").strip() or uuid.uuid4().hex
        with self._lock, self._conn:  # 单库事务:seq 计算 + 插入原子完成
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
            turn_role = str(fields.get("turn_role") or role_projection["turn_role"])
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
        summary_id = str(record.get("summary_id") or "").strip() or uuid.uuid4().hex
        with self._lock, self._conn:
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
        semantic_id = str(record.get("semantic_id") or "").strip() or uuid.uuid4().hex
        ts = int(record.get("timestamp") or 0)
        with self._lock, self._conn:
            prior = self._conn.execute(
                "SELECT * FROM semantic_summaries WHERE semantic_id = ?", (semantic_id,)
            ).fetchone()
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
        with self._lock, self._conn:
            for table, id_col in (
                ("messages", "source_id"),
                ("summaries", "summary_id"),
                ("semantic_summaries", "semantic_id"),
            ):
                cur = self._conn.execute(f"UPDATE {table} SET index_status = ? WHERE {id_col} = ?", (status, source_id))
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

    def get_entry(self, *, namespace: Namespace, source_id: str) -> dict[str, Any] | None:
        """Namespace-safe Timeline V2 raw lookup; unlike the legacy global lookup."""

        sid = str(source_id or "").strip()
        if not sid:
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM messages WHERE source_id = ?", (sid,)).fetchone()
            if row is None:
                return None
            self._assert_scope_owner(row, namespace, id_label=f"message source_id={sid!r}")
            return self._row_to_record(row, "messages")

    def get_turn_entries(self, *, namespace: Namespace, turn_id: str) -> list[dict[str, Any]]:
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
        return [self._row_to_record(row, "messages") for row in rows]

    def get_correlation_entries(
        self, *, namespace: Namespace, turn_id: str, correlation_id: str
    ) -> list[dict[str, Any]]:
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
        return [self._row_to_record(row, "messages") for row in rows]

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

    def get_messages_by_date_range(
        self,
        *,
        namespace: Namespace,
        date_from: str = "",
        date_to: str = "",
        time_periods: list[str] | None = None,
        cross_conversation: bool = False,
    ) -> list[dict[str, Any]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=not cross_conversation)
        clause = scope_clause
        params = list(params)
        if date_from:
            clause += " AND date_label >= ?"
            params.append(str(date_from))
        if date_to:
            clause += " AND date_label <= ?"
            params.append(str(date_to))
        periods = [str(p) for p in (time_periods or []) if str(p or "").strip()]
        if periods:
            clause += f" AND time_of_day IN ({','.join('?' for _ in periods)})"
            params.extend(periods)
        with self._lock:
            rows = self._conn.execute(
                # 按真实时间戳排序:seq_no 是写入序且按会话重置,回填/跨会话会乱序。
                f"SELECT * FROM messages WHERE {clause} ORDER BY timestamp ASC, conversation_id, seq_no",
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
        sql = f"SELECT * FROM summaries WHERE {scope_clause} AND is_semanticized = 0 ORDER BY timestamp ASC"
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
