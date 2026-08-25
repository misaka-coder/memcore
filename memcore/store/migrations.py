"""SQLite schema versioning and Timeline V2 foundation migrations.

SQLite remains the source of truth.  Migrations only mutate the relational
store; vector/index repair happens after the store has opened successfully.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable

from ..errors import SchemaError

CURRENT_SCHEMA_VERSION = 6

_CORE_TABLES = frozenset({"messages", "summaries", "semantic_summaries"})
_TRACE_CATEGORIES = frozenset({"event_trace", "material_trace", "tool_trace"})
_SAFE_KIND_SEGMENT = re.compile(r"[^a-z0-9_-]+")


LATEST_SCHEMA_STATEMENTS = (
    """
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
        summary_id TEXT NOT NULL DEFAULT '',
        kind TEXT NOT NULL DEFAULT '',
        origin TEXT NOT NULL DEFAULT '',
        turn_id TEXT NOT NULL DEFAULT '',
        turn_role TEXT NOT NULL DEFAULT '',
        reply_to_source_id TEXT NOT NULL DEFAULT '',
        correlation_id TEXT NOT NULL DEFAULT '',
        relation_status TEXT NOT NULL DEFAULT 'legacy_unlinked',
        target_actor_id TEXT NOT NULL DEFAULT '',
        target_actor_display_name TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        semantic_text TEXT NOT NULL DEFAULT '',
        trace_metadata_json TEXT NOT NULL DEFAULT '{}',
        annotation_status TEXT NOT NULL DEFAULT 'unannotated',
        annotation_source TEXT NOT NULL DEFAULT '',
        retrieval_policy TEXT NOT NULL DEFAULT 'auto',
        retrieval_visibility TEXT NOT NULL DEFAULT 'explicit',
        semanticize INTEGER NOT NULL DEFAULT 1,
        prompt_visible INTEGER NOT NULL DEFAULT 1,
        trust TEXT NOT NULL DEFAULT 'untrusted_data',
        renderer_id TEXT NOT NULL DEFAULT 'canonical',
        renderer_version INTEGER NOT NULL DEFAULT 1,
        row_version INTEGER NOT NULL DEFAULT 1,
        index_schema_version INTEGER NOT NULL DEFAULT 0,
        index_key TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_scope_seq
    ON messages(tenant_id, user_id, domain_id, conversation_id, seq_no)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_prompt_visible_scope_seq
    ON messages(tenant_id, user_id, domain_id, conversation_id, seq_no)
    WHERE is_summarized = 0 AND prompt_visible = 1
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_scope_turn
    ON messages(tenant_id, user_id, domain_id, conversation_id, turn_id, seq_no)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_scope_correlation
    ON messages(tenant_id, user_id, domain_id, conversation_id, correlation_id, seq_no)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_scope_date_visibility
    ON messages(
        tenant_id, user_id, domain_id, conversation_id,
        date_label, retrieval_visibility, timestamp
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_scope_timestamp
    ON messages(tenant_id, user_id, domain_id, conversation_id, timestamp, seq_no, source_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_hard_scope_timestamp
    ON messages(tenant_id, user_id, domain_id, timestamp, conversation_id, seq_no, source_id)
    """,
    """
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
        memory_title TEXT NOT NULL DEFAULT '',
        catalog_hint TEXT NOT NULL DEFAULT '',
        topic_headings_json TEXT NOT NULL DEFAULT '[]',
        participant_refs_json TEXT NOT NULL DEFAULT '[]',
        source_turn_count INTEGER NOT NULL DEFAULT 0,
        source_entry_count INTEGER NOT NULL DEFAULT 0,
        catalog_schema_version INTEGER NOT NULL DEFAULT 0,
        semantic_tags_json TEXT NOT NULL DEFAULT '[]',
        memory_metadata_json TEXT NOT NULL DEFAULT '{}',
        source_ids_json TEXT NOT NULL DEFAULT '[]',
        is_semanticized INTEGER NOT NULL DEFAULT 0,
        semantic_id TEXT NOT NULL DEFAULT '',
        index_status TEXT NOT NULL DEFAULT 'pending',
        kind TEXT NOT NULL DEFAULT 'memory.episode_summary',
        trace_metadata_json TEXT NOT NULL DEFAULT '{}',
        annotation_status TEXT NOT NULL DEFAULT 'derived',
        retrieval_visibility TEXT NOT NULL DEFAULT 'default',
        semanticize INTEGER NOT NULL DEFAULT 1,
        lineage_status TEXT NOT NULL DEFAULT 'valid',
        compaction_schema_version INTEGER NOT NULL DEFAULT 1,
        row_version INTEGER NOT NULL DEFAULT 1,
        index_schema_version INTEGER NOT NULL DEFAULT 0,
        index_key TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_summaries_scope_time
    ON summaries(tenant_id, user_id, domain_id, timestamp DESC)
    """,
    """
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
        memory_title TEXT NOT NULL DEFAULT '',
        catalog_hint TEXT NOT NULL DEFAULT '',
        topic_headings_json TEXT NOT NULL DEFAULT '[]',
        catalog_schema_version INTEGER NOT NULL DEFAULT 0,
        semantic_tags_json TEXT NOT NULL DEFAULT '[]',
        memory_metadata_json TEXT NOT NULL DEFAULT '{}',
        source_summary_ids_json TEXT NOT NULL DEFAULT '[]',
        reinforcement_count INTEGER NOT NULL DEFAULT 1,
        last_reinforced_ts INTEGER NOT NULL DEFAULT 0,
        index_status TEXT NOT NULL DEFAULT 'pending',
        kind TEXT NOT NULL DEFAULT 'memory.semantic_summary',
        annotation_status TEXT NOT NULL DEFAULT 'derived',
        retrieval_visibility TEXT NOT NULL DEFAULT 'default',
        lineage_status TEXT NOT NULL DEFAULT 'valid',
        semantic_schema_version INTEGER NOT NULL DEFAULT 1,
        row_version INTEGER NOT NULL DEFAULT 1,
        index_schema_version INTEGER NOT NULL DEFAULT 0,
        index_key TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_semantic_scope_recency
    ON semantic_summaries(
        tenant_id, user_id, domain_id,
        last_reinforced_ts DESC, importance DESC, timestamp DESC
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        turn_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL,
        domain_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'open',
        stimulus_source_ids_json TEXT NOT NULL DEFAULT '[]',
        annotation_target_ids_json TEXT NOT NULL DEFAULT '[]',
        final_source_id TEXT NOT NULL DEFAULT '',
        opened_at INTEGER NOT NULL DEFAULT 0,
        closed_at INTEGER NOT NULL DEFAULT 0,
        close_reason TEXT NOT NULL DEFAULT '',
        operation_projection_policy TEXT NOT NULL DEFAULT 'full_until_raw_compaction',
        operation_settlement_min_utf8_bytes INTEGER NOT NULL DEFAULT 256,
        operation_settlement_min_saved_ratio REAL NOT NULL DEFAULT 0.5,
        row_version INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_turns_scope_status
    ON turns(tenant_id, user_id, domain_id, conversation_id, status, opened_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_states (
        tenant_id TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL,
        domain_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        compaction_generation INTEGER NOT NULL DEFAULT 0,
        last_compacted_seq_no INTEGER NOT NULL DEFAULT 0,
        projection_generation INTEGER NOT NULL DEFAULT 0,
        row_version INTEGER NOT NULL DEFAULT 1,
        updated_at INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(tenant_id, user_id, domain_id, conversation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prompt_projections (
        projection_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL,
        domain_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        turn_id TEXT NOT NULL,
        projection_index INTEGER NOT NULL,
        provider_profile TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        source_ids_json TEXT NOT NULL DEFAULT '[]',
        payload_hash TEXT NOT NULL DEFAULT '',
        projection_status TEXT NOT NULL DEFAULT 'complete',
        projection_version INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL DEFAULT 0,
        UNIQUE(
            tenant_id, user_id, domain_id, conversation_id,
            turn_id, provider_profile, projection_index
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prompt_projections_scope_turn
    ON prompt_projections(
        tenant_id, user_id, domain_id, conversation_id,
        turn_id, provider_profile, projection_index
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS projection_audits (
        tenant_id TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL,
        domain_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        turn_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        provider_profile TEXT NOT NULL,
        model_route_hash TEXT NOT NULL DEFAULT '',
        system_prefix_hash TEXT NOT NULL DEFAULT '',
        tool_schema_hash TEXT NOT NULL DEFAULT '',
        history_hash TEXT NOT NULL DEFAULT '',
        full_prefix_hash TEXT NOT NULL DEFAULT '',
        projection_version INTEGER NOT NULL DEFAULT 1,
        media_omitted INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(
            tenant_id, user_id, domain_id, conversation_id,
            turn_id, attempt, provider_profile
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turn_projection_settlement (
        tenant_id TEXT NOT NULL DEFAULT '',
        user_id TEXT NOT NULL,
        domain_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        turn_id TEXT NOT NULL,
        provider_profile TEXT NOT NULL,
        policy TEXT NOT NULL DEFAULT 'full_until_raw_compaction',
        settlement_status TEXT NOT NULL DEFAULT 'settled',
        settlement_schema_version INTEGER NOT NULL DEFAULT 1,
        terminal_source_id TEXT NOT NULL DEFAULT '',
        full_projection_hash TEXT NOT NULL DEFAULT '',
        settled_projection_hash TEXT NOT NULL DEFAULT '',
        first_changed_projection_index INTEGER NOT NULL DEFAULT -1,
        full_projected_tokens INTEGER NOT NULL DEFAULT 0,
        settled_projected_tokens INTEGER NOT NULL DEFAULT 0,
        token_count_quality TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        settled_at INTEGER NOT NULL DEFAULT 0,
        settlement_min_utf8_bytes INTEGER NOT NULL DEFAULT 256,
        settlement_min_saved_ratio REAL NOT NULL DEFAULT 0.5,
        settlement_config_hash TEXT NOT NULL DEFAULT '',
        PRIMARY KEY(
            tenant_id, user_id, domain_id, conversation_id,
            turn_id, provider_profile
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_settlement_scope_turn
    ON turn_projection_settlement(
        tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settled_prompt_projection (
        settlement_id TEXT NOT NULL,
        projection_index INTEGER NOT NULL,
        provider_profile TEXT NOT NULL DEFAULT '',
        source_ids_json TEXT NOT NULL DEFAULT '[]',
        payload_json TEXT NOT NULL DEFAULT '{}',
        payload_hash TEXT NOT NULL DEFAULT '',
        projection_status TEXT NOT NULL DEFAULT 'settled',
        PRIMARY KEY(settlement_id, projection_index)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_settled_projection_profile
    ON settled_prompt_projection(settlement_id, projection_index, provider_profile)
    """,
)

_CATALOG_INDEX_STATEMENTS = (
    """
    CREATE INDEX IF NOT EXISTS idx_summaries_scope_period
    ON summaries(
        tenant_id, user_id, domain_id, conversation_id,
        period_start_ts, period_end_ts, timestamp, summary_id
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_summaries_hard_scope_period
    ON summaries(
        tenant_id, user_id, domain_id,
        period_start_ts, period_end_ts, timestamp, conversation_id, summary_id
    )
    """,
)


_V2_MESSAGE_COLUMNS = {
    "kind": "TEXT NOT NULL DEFAULT ''",
    "origin": "TEXT NOT NULL DEFAULT ''",
    "turn_id": "TEXT NOT NULL DEFAULT ''",
    "turn_role": "TEXT NOT NULL DEFAULT ''",
    "reply_to_source_id": "TEXT NOT NULL DEFAULT ''",
    "correlation_id": "TEXT NOT NULL DEFAULT ''",
    "relation_status": "TEXT NOT NULL DEFAULT 'legacy_unlinked'",
    "target_actor_id": "TEXT NOT NULL DEFAULT ''",
    "target_actor_display_name": "TEXT NOT NULL DEFAULT ''",
    "payload_json": "TEXT NOT NULL DEFAULT '{}'",
    "semantic_text": "TEXT NOT NULL DEFAULT ''",
    "trace_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    "annotation_status": "TEXT NOT NULL DEFAULT 'unannotated'",
    "annotation_source": "TEXT NOT NULL DEFAULT ''",
    "retrieval_policy": "TEXT NOT NULL DEFAULT 'auto'",
    "retrieval_visibility": "TEXT NOT NULL DEFAULT 'explicit'",
    "semanticize": "INTEGER NOT NULL DEFAULT 1",
    "prompt_visible": "INTEGER NOT NULL DEFAULT 1",
    "trust": "TEXT NOT NULL DEFAULT 'untrusted_data'",
    "renderer_id": "TEXT NOT NULL DEFAULT 'canonical'",
    "renderer_version": "INTEGER NOT NULL DEFAULT 1",
    "row_version": "INTEGER NOT NULL DEFAULT 1",
    "index_schema_version": "INTEGER NOT NULL DEFAULT 0",
    "index_key": "TEXT NOT NULL DEFAULT ''",
}

_V2_SUMMARY_COLUMNS = {
    "kind": "TEXT NOT NULL DEFAULT 'memory.episode_summary'",
    "trace_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    "annotation_status": "TEXT NOT NULL DEFAULT 'derived'",
    "retrieval_visibility": "TEXT NOT NULL DEFAULT 'default'",
    "semanticize": "INTEGER NOT NULL DEFAULT 1",
    "lineage_status": "TEXT NOT NULL DEFAULT 'valid'",
    "compaction_schema_version": "INTEGER NOT NULL DEFAULT 1",
    "row_version": "INTEGER NOT NULL DEFAULT 1",
    "index_schema_version": "INTEGER NOT NULL DEFAULT 0",
    "index_key": "TEXT NOT NULL DEFAULT ''",
}

_V2_SEMANTIC_COLUMNS = {
    "kind": "TEXT NOT NULL DEFAULT 'memory.semantic_summary'",
    "annotation_status": "TEXT NOT NULL DEFAULT 'derived'",
    "retrieval_visibility": "TEXT NOT NULL DEFAULT 'default'",
    "lineage_status": "TEXT NOT NULL DEFAULT 'valid'",
    "semantic_schema_version": "INTEGER NOT NULL DEFAULT 1",
    "row_version": "INTEGER NOT NULL DEFAULT 1",
    "index_schema_version": "INTEGER NOT NULL DEFAULT 0",
    "index_key": "TEXT NOT NULL DEFAULT ''",
}

_V4_SUMMARY_COLUMNS = {
    "memory_title": "TEXT NOT NULL DEFAULT ''",
    "catalog_hint": "TEXT NOT NULL DEFAULT ''",
    "topic_headings_json": "TEXT NOT NULL DEFAULT '[]'",
    "participant_refs_json": "TEXT NOT NULL DEFAULT '[]'",
    "source_turn_count": "INTEGER NOT NULL DEFAULT 0",
    "source_entry_count": "INTEGER NOT NULL DEFAULT 0",
    "catalog_schema_version": "INTEGER NOT NULL DEFAULT 0",
}

_V4_SEMANTIC_COLUMNS = {
    "memory_title": "TEXT NOT NULL DEFAULT ''",
    "catalog_hint": "TEXT NOT NULL DEFAULT ''",
    "topic_headings_json": "TEXT NOT NULL DEFAULT '[]'",
    "catalog_schema_version": "INTEGER NOT NULL DEFAULT 0",
}

_REQUIRED_COLUMNS = {
    "messages": frozenset(
        {
            "source_id",
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "actor_id",
            "actor_display_name",
            "seq_no",
            "role",
            "content",
            "timestamp",
            "date_label",
            "time_of_day",
            "memory_metadata_json",
            "index_status",
            "is_summarized",
            "summary_id",
            *_V2_MESSAGE_COLUMNS,
        }
    ),
    "summaries": frozenset(
        {
            "summary_id",
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "timestamp",
            "period_start_ts",
            "period_end_ts",
            "date_label",
            "time_of_day",
            "period_label",
            "event_type",
            "importance",
            "diary_summary",
            "key_events_json",
            "core_facts_json",
            *_V4_SUMMARY_COLUMNS,
            "semantic_tags_json",
            "memory_metadata_json",
            "source_ids_json",
            "is_semanticized",
            "semantic_id",
            "index_status",
            *_V2_SUMMARY_COLUMNS,
        }
    ),
    "semantic_summaries": frozenset(
        {
            "semantic_id",
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "timestamp",
            "period_start_ts",
            "period_end_ts",
            "date_label",
            "time_of_day",
            "importance",
            "semantic_summary",
            "stable_facts_json",
            "recurring_topics_json",
            "important_people_json",
            "open_loops_json",
            *_V4_SEMANTIC_COLUMNS,
            "semantic_tags_json",
            "memory_metadata_json",
            "source_summary_ids_json",
            "reinforcement_count",
            "last_reinforced_ts",
            "index_status",
            *_V2_SEMANTIC_COLUMNS,
        }
    ),
    "turns": frozenset(
        {
            "turn_id",
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "status",
            "stimulus_source_ids_json",
            "annotation_target_ids_json",
            "final_source_id",
            "row_version",
        }
    ),
    "conversation_states": frozenset(
        {
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "compaction_generation",
            "projection_generation",
            "row_version",
        }
    ),
    "prompt_projections": frozenset(
        {
            "projection_id",
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "turn_id",
            "projection_index",
            "provider_profile",
            "payload_json",
            "source_ids_json",
            "payload_hash",
            "projection_status",
            "projection_version",
            "created_at",
        }
    ),
    "projection_audits": frozenset(
        {
            "tenant_id",
            "user_id",
            "domain_id",
            "conversation_id",
            "turn_id",
            "attempt",
            "provider_profile",
            "model_route_hash",
            "system_prefix_hash",
            "tool_schema_hash",
            "history_hash",
            "full_prefix_hash",
            "projection_version",
            "media_omitted",
            "created_at",
        }
    ),
}


def migrate_database(connection: sqlite3.Connection) -> int:
    """Create or migrate a database to the latest schema in one transaction."""

    if connection.in_transaction:
        raise SchemaError("sqlite_schema_migration_requires_idle_connection")

    connection.execute("BEGIN IMMEDIATE")
    try:
        version = _user_version(connection)
        tables = _table_names(connection)
        has_core = _CORE_TABLES.intersection(tables)

        if version > CURRENT_SCHEMA_VERSION:
            raise SchemaError("sqlite_schema_version_is_newer_than_runtime")

        if not has_core:
            if tables:
                raise SchemaError("sqlite_schema_unknown_unversioned_database")
            _execute_statements(connection, LATEST_SCHEMA_STATEMENTS)
        else:
            if has_core != _CORE_TABLES:
                raise SchemaError("sqlite_schema_partial_core_tables")
            if version not in {0, 1, 2, 3, 4, 5, CURRENT_SCHEMA_VERSION}:
                raise SchemaError("sqlite_schema_unsupported_version")
            if version < 2:
                _migrate_v1_to_v2(connection)
            else:
                _execute_statements(connection, LATEST_SCHEMA_STATEMENTS)
            if version < 3:
                _migrate_v2_to_v3(connection)
            if version < 4:
                _migrate_v3_to_v4(connection)
            if version < 5:
                _migrate_v4_to_v5(connection)
            if version < 6:
                _migrate_v5_to_v6(connection)

        _execute_statements(connection, _CATALOG_INDEX_STATEMENTS)
        _validate_latest_schema(connection)
        connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
        connection.commit()
        return CURRENT_SCHEMA_VERSION
    except SchemaError:
        connection.rollback()
        raise
    except sqlite3.Error as exc:
        connection.rollback()
        raise SchemaError(f"sqlite_schema_migration_failed:{type(exc).__name__}") from exc
    except Exception as exc:
        connection.rollback()
        raise SchemaError(f"sqlite_schema_backfill_failed:{type(exc).__name__}") from exc


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    _add_missing_columns(connection, "messages", _V2_MESSAGE_COLUMNS)
    _add_missing_columns(connection, "summaries", _V2_SUMMARY_COLUMNS)
    _add_missing_columns(connection, "semantic_summaries", _V2_SEMANTIC_COLUMNS)
    _execute_statements(connection, LATEST_SCHEMA_STATEMENTS)
    _backfill_messages(connection)
    _backfill_summaries(connection)
    _backfill_semantic_summaries(connection)


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Replace the V1 metadata JSON contract and invalidate every old index entry."""

    for table, id_column in (
        ("messages", "source_id"),
        ("summaries", "summary_id"),
        ("semantic_summaries", "semantic_id"),
    ):
        rows = connection.execute(
            f"SELECT {id_column}, memory_metadata_json FROM {table} ORDER BY {id_column}"
        ).fetchall()
        for row in rows:
            connection.execute(
                f"""
                UPDATE {table}
                SET memory_metadata_json = ?, index_status = 'pending',
                    index_schema_version = 0, index_key = ''
                WHERE {id_column} = ?
                """,
                (_json_dumps(migrate_legacy_memory_metadata(_load_json_object(row[1]))), str(row[0])),
            )

    message_rows = connection.execute(
        "SELECT source_id, annotation_status, trace_metadata_json, memory_metadata_json FROM messages"
    ).fetchall()
    for row in message_rows:
        source_id = str(row[0])
        annotation_status = str(row[1] or "")
        trace_metadata = _load_json_object(row[2])
        trace_metadata.pop("legacy_categories", None)
        metadata = _load_json_object(row[3])
        if annotation_status == "accepted_legacy":
            accepted = _has_target_semantic_metadata(metadata)
            connection.execute(
                """
                UPDATE messages
                SET annotation_status = ?, annotation_source = ?, retrieval_visibility = ?,
                    trace_metadata_json = ?
                WHERE source_id = ?
                """,
                (
                    "accepted_host" if accepted else "unannotated",
                    "schema_v3_migration" if accepted else "",
                    "default" if accepted else "explicit",
                    _json_dumps(trace_metadata),
                    source_id,
                ),
            )
        else:
            connection.execute(
                "UPDATE messages SET trace_metadata_json = ? WHERE source_id = ?",
                (_json_dumps(trace_metadata), source_id),
            )

    summary_rows = connection.execute("SELECT summary_id, trace_metadata_json FROM summaries").fetchall()
    for row in summary_rows:
        trace_metadata = _load_json_object(row[1])
        trace_metadata.pop("legacy_categories", None)
        connection.execute(
            "UPDATE summaries SET trace_metadata_json = ? WHERE summary_id = ?",
            (_json_dumps(trace_metadata), str(row[0])),
        )


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """Add catalog projections without calling an LLM or rewriting memory truth."""

    _add_missing_columns(connection, "summaries", _V4_SUMMARY_COLUMNS)
    _add_missing_columns(connection, "semantic_summaries", _V4_SEMANTIC_COLUMNS)
    _backfill_summary_catalog_metrics(connection)


_V5_TURN_COLUMNS = {"operation_projection_policy": "TEXT NOT NULL DEFAULT 'full_until_raw_compaction'"}


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """Add the settled-compact projection schema without rewriting memory truth."""

    _add_missing_columns(connection, "turns", _V5_TURN_COLUMNS)
    _execute_statements(connection, LATEST_SCHEMA_STATEMENTS)


_V6_TURN_COLUMNS = {
    "operation_settlement_min_utf8_bytes": "INTEGER NOT NULL DEFAULT 256",
    "operation_settlement_min_saved_ratio": "REAL NOT NULL DEFAULT 0.5",
}
_V6_SETTLEMENT_COLUMNS = {
    "settlement_min_utf8_bytes": "INTEGER NOT NULL DEFAULT 256",
    "settlement_min_saved_ratio": "REAL NOT NULL DEFAULT 0.5",
    "settlement_config_hash": "TEXT NOT NULL DEFAULT ''",
}


def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
    """Freeze settlement byte/ratio thresholds per turn without rewriting truth.

    Defaults (256 / 0.5) reproduce the historical classify_observation behavior
    byte-for-byte, so upgraded databases keep their old semantics.
    """

    _add_missing_columns(connection, "turns", _V6_TURN_COLUMNS)
    _add_missing_columns(connection, "turn_projection_settlement", _V6_SETTLEMENT_COLUMNS)
    _execute_statements(connection, LATEST_SCHEMA_STATEMENTS)


def _backfill_summary_catalog_metrics(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT summary_id, tenant_id, user_id, domain_id, conversation_id, source_ids_json
        FROM summaries ORDER BY summary_id
        """
    ).fetchall()
    for summary_id, tenant_id, user_id, domain_id, conversation_id, encoded_source_ids in rows:
        source_ids = _load_json_list(encoded_source_ids)
        source_ids = list(dict.fromkeys(str(item or "").strip() for item in source_ids if str(item or "").strip()))
        by_source: dict[str, tuple[str, str, str, str, str]] = {}
        for offset in range(0, len(source_ids), 500):
            batch = source_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            if not placeholders:
                continue
            source_rows = connection.execute(
                f"""
                SELECT source_id, turn_id, actor_id, actor_display_name,
                       target_actor_id, target_actor_display_name
                FROM messages
                WHERE source_id IN ({placeholders})
                  AND tenant_id = ? AND user_id = ? AND domain_id = ? AND conversation_id = ?
                """,
                [
                    *batch,
                    str(tenant_id or ""),
                    str(user_id or ""),
                    str(domain_id or ""),
                    str(conversation_id or ""),
                ],
            ).fetchall()
            for row in source_rows:
                by_source[str(row[0])] = (
                    str(row[1] or ""),
                    str(row[2] or ""),
                    str(row[3] or ""),
                    str(row[4] or ""),
                    str(row[5] or ""),
                )

        logical_turns: set[str] = set()
        participant_order: list[str] = []
        participant_names: dict[str, str] = {}
        for source_id in source_ids:
            values = by_source.get(source_id)
            if values is None:
                continue
            turn_id, actor_id, actor_name, target_id, target_name = values
            logical_turns.add(turn_id or source_id)
            for participant_id, display_name in ((actor_id, actor_name), (target_id, target_name)):
                if not participant_id:
                    continue
                if participant_id not in participant_names:
                    participant_order.append(participant_id)
                    participant_names[participant_id] = ""
                if display_name:
                    participant_names[participant_id] = display_name
        participants = [
            {"actor_id": actor_id, "display_name": participant_names[actor_id]} for actor_id in participant_order
        ]
        connection.execute(
            """
            UPDATE summaries
            SET participant_refs_json = ?, source_turn_count = ?, source_entry_count = ?
            WHERE summary_id = ?
            """,
            (_json_dumps(participants), len(logical_turns), len(source_ids), str(summary_id)),
        )


def migrate_legacy_memory_metadata(data: Any) -> dict[str, Any]:
    """One-time schema-2/import adapter; active runtime code must not dual-read old keys."""

    metadata = dict(data) if isinstance(data, dict) else {}
    old_categories = _unique_strings(metadata.get("categories"))
    facets = _unique_strings(metadata.get("memory_facets"))
    facet_mapping = {
        "preference": "preference",
        "personal_profile": "profile",
        "plan_goal": "plan",
        "relationship": "relationship",
        "emotion_state": "state",
        "life_event": "event",
    }
    facets = _unique_strings([*facets, *(facet_mapping[item] for item in old_categories if item in facet_mapping)])
    allowed_facets = {
        "profile",
        "preference",
        "viewpoint",
        "relationship",
        "event",
        "state",
        "plan",
        "decision",
        "constraint",
        "knowledge",
        "procedure",
    }
    facets = [item for item in facets if item in allowed_facets]

    roles = _unique_strings(metadata.get("about_roles"))
    for scope in _unique_strings(metadata.get("subject_scopes")):
        if scope in {"user", "assistant"}:
            roles.append(scope)
        elif scope == "other":
            roles.extend(("third_party", "external"))
    roles = [item for item in _unique_strings(roles) if item in {"user", "assistant", "third_party", "external"}]

    turn_intent = str(metadata.get("turn_intent") or "").strip()
    if "memory_query" in old_categories:
        turn_intent = "memory_query"
    if turn_intent != "memory_query":
        turn_intent = ""

    priority = str(metadata.get("retrieval_priority") or "").strip().lower()
    if priority not in {"low", "normal", "high", "critical"}:
        priority = _priority_from_legacy_importance(metadata.get("importance"))

    return {
        "turn_intent": turn_intent,
        "memory_facets": facets,
        "about_roles": roles,
        "entity_anchors": _unique_strings(metadata.get("entity_anchors")),
        "topic_terms": _unique_strings(
            [*(_unique_strings(metadata.get("topic_terms"))), *(_unique_strings(metadata.get("keywords")))]
        ),
        "retrieval_priority": priority,
        "mood_tags": _unique_strings(metadata.get("mood_tags"))[:3],
    }


def _priority_from_legacy_importance(value: Any) -> str:
    try:
        importance = float(value)
    except (TypeError, ValueError):
        importance = 0.0
    if importance >= 0.85:
        return "critical"
    if importance >= 0.65:
        return "high"
    if importance >= 0.25:
        return "normal"
    return "low"


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _has_target_semantic_metadata(metadata: dict[str, Any]) -> bool:
    if any(
        _unique_strings(metadata.get(key))
        for key in ("memory_facets", "about_roles", "entity_anchors", "topic_terms", "mood_tags")
    ):
        return True
    return str(metadata.get("retrieval_priority") or "normal") in {"high", "critical"}


def _add_missing_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = _table_columns(connection, table)
    for name, declaration in columns.items():
        if name not in existing:
            connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declaration}')


def _backfill_messages(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT source_id, role, content, memory_metadata_json FROM messages ORDER BY source_id"
    ).fetchall()
    for row in rows:
        source_id = str(row[0])
        role = str(row[1] or "")
        content = str(row[2] or "")
        metadata = _load_json_object(row[3])
        semantic_metadata, trace_categories = _split_trace_categories(metadata)
        projection = project_legacy_role(role)
        accepted = (
            projection["turn_role"] == "stimulus"
            and not str(projection["kind"]).startswith(("material.", "tool."))
            and _has_semantic_metadata(semantic_metadata)
        )
        annotation_status = "accepted_legacy" if accepted else "unannotated"
        visibility = "default" if accepted else "explicit"
        trace_metadata = dict(projection["trace_metadata"])
        if trace_categories:
            trace_metadata["legacy_categories"] = trace_categories
        semanticize = (
            0
            if projection["turn_role"] in {"action", "observation"} or str(projection["kind"]).startswith("material.")
            else 1
        )
        connection.execute(
            """
            UPDATE messages
            SET kind = ?, origin = ?, turn_role = ?, correlation_id = ?,
                relation_status = 'legacy_unlinked', payload_json = ?, semantic_text = ?,
                trace_metadata_json = ?, memory_metadata_json = ?,
                annotation_status = ?, annotation_source = ?,
                retrieval_policy = 'auto', retrieval_visibility = ?, semanticize = ?,
                prompt_visible = 1, trust = 'untrusted_data', renderer_id = 'canonical',
                renderer_version = 1, row_version = 1,
                index_schema_version = 0, index_key = '', index_status = 'pending'
            WHERE source_id = ?
            """,
            (
                projection["kind"],
                projection["origin"],
                projection["turn_role"],
                projection["correlation_id"],
                _json_dumps({"text": content}),
                content,
                _json_dumps(trace_metadata),
                # Foundation migration keeps V1 trace categories until retrieval/compaction
                # consumers cut over to kind + trace_metadata in the same later slice.
                _json_dumps(metadata),
                annotation_status,
                "legacy_migration" if accepted else "",
                visibility,
                semanticize,
                source_id,
            ),
        )


def _backfill_summaries(connection: sqlite3.Connection) -> None:
    rows = connection.execute("SELECT summary_id, memory_metadata_json FROM summaries ORDER BY summary_id").fetchall()
    for row in rows:
        summary_id = str(row[0])
        metadata = _load_json_object(row[1])
        semantic_metadata, trace_categories = _split_trace_categories(metadata)
        pure_trace = bool(trace_categories) and not _has_semantic_metadata(semantic_metadata)
        connection.execute(
            """
            UPDATE summaries
            SET kind = ?, trace_metadata_json = ?, memory_metadata_json = ?,
                annotation_status = 'derived', retrieval_visibility = ?, semanticize = ?,
                lineage_status = 'valid', compaction_schema_version = 1,
                row_version = 1, index_schema_version = 0, index_key = '', index_status = 'pending'
            WHERE summary_id = ?
            """,
            (
                "memory.operation_digest" if pure_trace else "memory.episode_summary",
                _json_dumps({"legacy_categories": trace_categories} if trace_categories else {}),
                _json_dumps(metadata),
                "explicit" if pure_trace else "default",
                0 if pure_trace else 1,
                summary_id,
            ),
        )


def _backfill_semantic_summaries(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT semantic_id, memory_metadata_json FROM semantic_summaries ORDER BY semantic_id"
    ).fetchall()
    for row in rows:
        semantic_id = str(row[0])
        metadata = _load_json_object(row[1])
        connection.execute(
            """
            UPDATE semantic_summaries
            SET kind = 'memory.semantic_summary', memory_metadata_json = ?,
                annotation_status = 'derived', retrieval_visibility = 'default',
                lineage_status = 'valid', semantic_schema_version = 1,
                row_version = 1, index_schema_version = 0, index_key = '', index_status = 'pending'
            WHERE semantic_id = ?
            """,
            (_json_dumps(metadata), semantic_id),
        )


def project_legacy_role(role: Any) -> dict[str, Any]:
    """Map the package's known V1 role encodings without inventing turns."""

    raw = str(role or "").strip()
    if raw == "user":
        return _legacy_projection("message.user", "user", "stimulus")
    if raw == "assistant":
        return _legacy_projection("message.assistant", "assistant", "final")
    if raw.startswith("event.") and " " not in raw:
        return _legacy_projection(_safe_namespaced_kind(raw, fallback_prefix="event"), "environment", "stimulus")

    parts = raw.split()
    if len(parts) >= 2 and parts[0] == "assistant.tool_call":
        tool_name = _safe_kind_segment(parts[1])
        call_id = parts[2] if len(parts) >= 3 else ""
        return _legacy_projection(
            f"tool.{tool_name}.call",
            "assistant",
            "action",
            correlation_id=call_id,
            trace_metadata={"tool_name": tool_name, "call_id": call_id},
        )
    if len(parts) >= 1 and parts[0].startswith("tool."):
        tool_name = _safe_kind_segment(parts[0][len("tool.") :])
        call_id = parts[1] if len(parts) >= 2 else ""
        return _legacy_projection(
            f"tool.{tool_name}.result",
            "environment",
            "observation",
            correlation_id=call_id,
            trace_metadata={"tool_name": tool_name, "call_id": call_id},
        )
    if len(parts) >= 1 and parts[0] == "user.attachment":
        material_kind = _safe_kind_segment(parts[1]) if len(parts) >= 2 else "file"
        file_id = parts[2] if len(parts) >= 3 else ""
        return _legacy_projection(
            "material.reference",
            "user",
            "stimulus",
            trace_metadata={"material_kind": material_kind, "file_id": file_id},
        )
    if len(parts) >= 1 and parts[0] == "system.material_cleanup":
        material_kind = _safe_kind_segment(parts[1]) if len(parts) >= 2 else "file"
        file_id = parts[2] if len(parts) >= 3 else ""
        return _legacy_projection(
            "material.cleanup",
            "environment",
            "observation",
            trace_metadata={"material_kind": material_kind, "file_id": file_id},
        )

    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    return _legacy_projection(f"legacy.{digest}", "environment", "")


def _legacy_projection(
    kind: str,
    origin: str,
    turn_role: str,
    *,
    correlation_id: str = "",
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "origin": origin,
        "turn_role": turn_role,
        "correlation_id": str(correlation_id or ""),
        "trace_metadata": dict(trace_metadata or {}),
    }


def _safe_namespaced_kind(value: str, *, fallback_prefix: str) -> str:
    parts = [_safe_kind_segment(part) for part in str(value or "").split(".")]
    cleaned = ".".join(part for part in parts if part)
    return cleaned or f"{fallback_prefix}.unknown"


def _safe_kind_segment(value: Any) -> str:
    normalized = _SAFE_KIND_SEGMENT.sub("_", str(value or "").strip().lower()).strip("_")
    return normalized[:80] or "unknown"


def _split_trace_categories(metadata: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    clean = dict(metadata)
    categories = [str(item) for item in list(clean.get("categories") or []) if str(item or "").strip()]
    trace = sorted({item for item in categories if item in _TRACE_CATEGORIES})
    clean["categories"] = [item for item in categories if item not in _TRACE_CATEGORIES]
    return clean, trace


def _has_semantic_metadata(metadata: dict[str, Any]) -> bool:
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


def _load_json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _load_json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _execute_statements(connection: sqlite3.Connection, statements: Iterable[str]) -> None:
    for statement in statements:
        connection.execute(statement)


def _user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]) for row in rows}


def _validate_latest_schema(connection: sqlite3.Connection) -> None:
    tables = _table_names(connection)
    missing_tables = sorted(set(_REQUIRED_COLUMNS) - tables)
    if missing_tables:
        raise SchemaError("sqlite_schema_missing_tables:" + ",".join(missing_tables))
    for table, required in _REQUIRED_COLUMNS.items():
        missing = sorted(required - _table_columns(connection, table))
        if missing:
            raise SchemaError(f"sqlite_schema_missing_columns:{table}:" + ",".join(missing))


__all__ = ["CURRENT_SCHEMA_VERSION", "LATEST_SCHEMA_STATEMENTS", "migrate_database", "project_legacy_role"]
