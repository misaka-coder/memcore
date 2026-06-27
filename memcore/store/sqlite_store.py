"""SQLiteMemoryStore —— MemoryStore 的默认实现(关系型真相源)。

三张记忆表:messages / summaries / semantic_summaries。

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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    source_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL,
    domain_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    actor_id TEXT NOT NULL DEFAULT '',
    actor_display_name TEXT NOT NULL DEFAULT '',
    seq_no INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    date_label TEXT NOT NULL DEFAULT '',
    time_of_day TEXT NOT NULL DEFAULT '',
    memory_metadata_json TEXT NOT NULL DEFAULT '{}',
    index_status TEXT NOT NULL DEFAULT 'pending',
    is_summarized INTEGER NOT NULL DEFAULT 0,
    summary_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_messages_scope_seq
ON messages(tenant_id, user_id, domain_id, conversation_id, seq_no);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL,
    domain_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    timestamp INTEGER NOT NULL,
    period_start_ts INTEGER NOT NULL DEFAULT 0,
    period_end_ts INTEGER NOT NULL DEFAULT 0,
    date_label TEXT NOT NULL DEFAULT '',
    time_of_day TEXT NOT NULL DEFAULT '',
    period_label TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0,
    diary_summary TEXT NOT NULL DEFAULT '',
    key_events_json TEXT NOT NULL DEFAULT '[]',
    core_facts_json TEXT NOT NULL DEFAULT '[]',
    semantic_tags_json TEXT NOT NULL DEFAULT '[]',
    memory_metadata_json TEXT NOT NULL DEFAULT '{}',
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    is_semanticized INTEGER NOT NULL DEFAULT 0,
    semantic_id TEXT NOT NULL DEFAULT '',
    index_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_summaries_scope_time
ON summaries(tenant_id, user_id, domain_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS semantic_summaries (
    semantic_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL,
    domain_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    timestamp INTEGER NOT NULL,
    period_start_ts INTEGER NOT NULL DEFAULT 0,
    period_end_ts INTEGER NOT NULL DEFAULT 0,
    date_label TEXT NOT NULL DEFAULT '',
    time_of_day TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0,
    semantic_summary TEXT NOT NULL DEFAULT '',
    stable_facts_json TEXT NOT NULL DEFAULT '[]',
    recurring_topics_json TEXT NOT NULL DEFAULT '[]',
    important_people_json TEXT NOT NULL DEFAULT '[]',
    open_loops_json TEXT NOT NULL DEFAULT '[]',
    semantic_tags_json TEXT NOT NULL DEFAULT '[]',
    memory_metadata_json TEXT NOT NULL DEFAULT '{}',
    source_summary_ids_json TEXT NOT NULL DEFAULT '[]',
    reinforcement_count INTEGER NOT NULL DEFAULT 1,
    last_reinforced_ts INTEGER NOT NULL DEFAULT 0,
    index_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_semantic_scope_recency
ON semantic_summaries(tenant_id, user_id, domain_id, last_reinforced_ts DESC, importance DESC, timestamp DESC);
"""

_JSON_FIELDS = {
    "messages": {"memory_metadata_json": "memory_metadata"},
    "summaries": {
        "key_events_json": "key_events",
        "core_facts_json": "core_facts",
        "semantic_tags_json": "semantic_tags",
        "memory_metadata_json": "memory_metadata",
        "source_ids_json": "source_ids",
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
}
_ENTRY_TYPE = {"messages": "raw", "summaries": "summary", "semantic_summaries": "semantic_summary"}


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- 内部工具 ---

    @staticmethod
    def _row_to_record(row: sqlite3.Row, table: str) -> dict[str, Any]:
        record = dict(row)
        for json_col, key in _JSON_FIELDS[table].items():
            raw = record.pop(json_col, None)
            try:
                record[key] = json.loads(raw) if raw else ([] if key != "memory_metadata" else {})
            except (TypeError, ValueError):
                record[key] = [] if key != "memory_metadata" else {}
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
            self._conn.execute(
                """
                INSERT INTO messages
                (source_id, tenant_id, user_id, domain_id, conversation_id, actor_id, actor_display_name,
                 seq_no, role, content, timestamp, date_label, time_of_day, memory_metadata_json, index_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'pending')
                """,
                (
                    source_id,
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    namespace.actor_id(),
                    namespace.actor.display_name if namespace.actor else "",
                    seq_no,
                    str(role),
                    str(content),
                    int(timestamp),
                    str(fields.get("date_label") or ""),
                    str(fields.get("time_of_day") or ""),
                    json.dumps(fields.get("memory_metadata") or {}, ensure_ascii=False),
                ),
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
            self._conn.execute(
                """
                INSERT OR REPLACE INTO summaries
                (summary_id, tenant_id, user_id, domain_id, conversation_id, timestamp, period_start_ts, period_end_ts,
                 date_label, time_of_day, period_label, event_type, importance, diary_summary,
                 key_events_json, core_facts_json, semantic_tags_json, memory_metadata_json, source_ids_json, index_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'pending')
                """,
                (
                    summary_id,
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    int(record.get("timestamp") or 0),
                    int(record.get("period_start_ts") or 0),
                    int(record.get("period_end_ts") or 0),
                    str(record.get("date_label") or ""),
                    str(record.get("time_of_day") or ""),
                    str(record.get("period_label") or ""),
                    str(record.get("event_type") or ""),
                    float(record.get("importance") or 0.0),
                    str(record.get("diary_summary") or ""),
                    json.dumps(record.get("key_events") or [], ensure_ascii=False),
                    json.dumps(record.get("core_facts") or [], ensure_ascii=False),
                    json.dumps(record.get("semantic_tags") or [], ensure_ascii=False),
                    json.dumps(record.get("memory_metadata") or {}, ensure_ascii=False),
                    json.dumps(record.get("source_ids") or [], ensure_ascii=False),
                ),
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
            self._conn.execute(
                """
                INSERT OR REPLACE INTO semantic_summaries
                (semantic_id, tenant_id, user_id, domain_id, conversation_id, timestamp, period_start_ts, period_end_ts,
                 date_label, time_of_day, importance, semantic_summary, stable_facts_json, recurring_topics_json,
                 important_people_json, open_loops_json, semantic_tags_json, memory_metadata_json,
                 source_summary_ids_json, reinforcement_count, last_reinforced_ts, index_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'pending')
                """,
                (
                    semantic_id,
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    ts,
                    int(record.get("period_start_ts") or 0),
                    int(record.get("period_end_ts") or 0),
                    str(record.get("date_label") or ""),
                    str(record.get("time_of_day") or ""),
                    float(record.get("importance") or 0.0),
                    str(record.get("semantic_summary") or ""),
                    json.dumps(record.get("stable_facts") or [], ensure_ascii=False),
                    json.dumps(record.get("recurring_topics") or [], ensure_ascii=False),
                    json.dumps(record.get("important_people") or [], ensure_ascii=False),
                    json.dumps(record.get("open_loops") or [], ensure_ascii=False),
                    json.dumps(record.get("semantic_tags") or [], ensure_ascii=False),
                    json.dumps(record.get("memory_metadata") or {}, ensure_ascii=False),
                    json.dumps(record.get("source_summary_ids") or [], ensure_ascii=False),
                    int(record.get("reinforcement_count") or 1),
                    int(record.get("last_reinforced_ts") or ts),
                ),
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
                f"UPDATE messages SET is_summarized = 1, summary_id = ? WHERE source_id IN ({placeholders})",
                [summary_id, *source_ids],
            )

    def mark_summaries_semanticized(self, summary_ids: list[str], semantic_id: str) -> None:
        if not summary_ids:
            return
        placeholders = ",".join("?" for _ in summary_ids)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE summaries SET is_semanticized = 1, semantic_id = ? WHERE summary_id IN ({placeholders})",
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
        self, *, namespace: Namespace, limit: int, cross_conversation: bool = False
    ) -> list[dict[str, Any]]:
        scope_clause, params = self._scope_clause(namespace, with_conversation=not cross_conversation)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM semantic_summaries WHERE {scope_clause} "
                f"ORDER BY last_reinforced_ts DESC, importance DESC, timestamp DESC LIMIT ?",
                [*params, int(limit)],
            ).fetchall()
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
        return deleted_ids
