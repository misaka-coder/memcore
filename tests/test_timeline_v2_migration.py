"""Timeline V2 schema foundation: migration, backfill, and owner-safe reads."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    NamespaceError,
    SQLiteMemoryStore,
    SchemaError,
)
from memcore.namespace import Actor
from memcore.store.migrations import CURRENT_SCHEMA_VERSION, _migrate_v1_to_v2


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


_V1_SCHEMA = """
CREATE TABLE messages (
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
CREATE TABLE summaries (
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
CREATE TABLE semantic_summaries (
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
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _create_v1_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_V1_SCHEMA)
        messages = (
            (
                "user-1",
                1,
                "user",
                "我喜欢无糖可乐。",
                {"keywords": ["可乐"], "categories": ["preference"], "importance": 0.8},
            ),
            (
                "call-1",
                2,
                "assistant.tool_call web_search call_001",
                'input:\n{"query":"天气"}',
                {"categories": ["tool_trace"]},
            ),
            (
                "result-1",
                3,
                "tool.web_search call_001",
                "output:\n晴天。",
                {"categories": ["tool_trace"]},
            ),
            (
                "event-1",
                4,
                "event.qq.poke",
                "张三戳了戳助手。",
                {"keywords": ["戳一戳"], "categories": ["event_trace", "social_event"]},
            ),
            ("legacy-1", 5, "Unexpected Role/With Spaces", "旧内容必须保留。", {}),
            (
                "material-1",
                6,
                "user.attachment image file_img_1",
                "file_id: file_img_1",
                {"keywords": ["photo.jpg"], "categories": ["material_trace"]},
            ),
        )
        for source_id, seq_no, role, content, metadata in messages:
            connection.execute(
                """
                INSERT INTO messages(
                    source_id, tenant_id, user_id, domain_id, conversation_id,
                    actor_id, actor_display_name, seq_no, role, content, timestamp,
                    date_label, time_of_day, memory_metadata_json, index_status
                ) VALUES (?, 'tenant', 'user', 'domain', 'conversation', '', '', ?, ?, ?, ?,
                          '2026-07-20', 'afternoon', ?, 'indexed')
                """,
                (source_id, seq_no, role, content, 1000 + seq_no, _json(metadata)),
            )
        connection.execute(
            """
            INSERT INTO summaries(
                summary_id, tenant_id, user_id, domain_id, conversation_id, timestamp,
                diary_summary, memory_metadata_json, source_ids_json, index_status
            ) VALUES ('mixed-summary', 'tenant', 'user', 'domain', 'conversation', 2000,
                      '对话事实与工具过程混合。', ?, '["user-1","call-1"]', 'indexed')
            """,
            (_json({"keywords": ["可乐"], "categories": ["preference", "tool_trace"]}),),
        )
        connection.execute(
            """
            INSERT INTO summaries(
                summary_id, tenant_id, user_id, domain_id, conversation_id, timestamp,
                diary_summary, memory_metadata_json, source_ids_json, index_status
            ) VALUES ('trace-summary', 'tenant', 'user', 'domain', 'conversation', 2001,
                      '只记录工具运行。', ?, '["call-1","result-1"]', 'indexed')
            """,
            (_json({"categories": ["tool_trace"]}),),
        )
        connection.execute(
            """
            INSERT INTO semantic_summaries(
                semantic_id, tenant_id, user_id, domain_id, conversation_id, timestamp,
                semantic_summary, memory_metadata_json, source_summary_ids_json, index_status
            ) VALUES ('semantic-1', 'tenant', 'user', 'domain', 'conversation', 3000,
                      '用户偏好无糖饮料。', ?, '["mixed-summary"]', 'indexed')
            """,
            (_json({"categories": ["preference", "tool_trace"], "keywords": ["无糖饮料"]}),),
        )
        connection.commit()
    finally:
        connection.close()


def _create_v2_database(path: Path) -> None:
    _create_v1_database(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _migrate_v1_to_v2(connection)
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()}


class LatestSchemaTests(unittest.TestCase):
    def test_new_database_starts_at_latest_schema_and_v2_writes_are_populated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "new.sqlite3"
            store = SQLiteMemoryStore(str(path))
            self.assertEqual(store.schema_version, CURRENT_SCHEMA_VERSION)
            namespace = Namespace(user_id="u1", tenant_id="t1", domain_id="d1", conversation_id="c1")
            record = store.add_message(
                namespace=namespace,
                role="user",
                content="我喜欢热咖啡。",
                timestamp=100,
                memory_metadata={
                    "entity_anchors": ["咖啡"],
                    "memory_facets": ["preference"],
                    "about_roles": ["user"],
                },
            )
            operation = store.add_summary(
                namespace=namespace,
                record={"summary_id": "op-1", "kind": "memory.operation_digest", "timestamp": 101},
            )
            material = store.add_message(
                namespace=namespace,
                role="user.attachment image file-1",
                content="file_id: file-1",
                timestamp=102,
                memory_metadata={},
            )
            store.close()

            self.assertEqual(record["kind"], "message.user")
            self.assertEqual(record["origin"], "user")
            self.assertEqual(record["turn_role"], "stimulus")
            self.assertEqual(record["payload"], {"text": "我喜欢热咖啡。"})
            self.assertEqual(record["semantic_text"], "我喜欢热咖啡。")
            self.assertEqual(record["annotation_status"], "accepted_host")
            self.assertEqual(record["retrieval_visibility"], "default")
            self.assertEqual(operation["retrieval_visibility"], "explicit")
            self.assertEqual(operation["semanticize"], 0)
            self.assertEqual(material["kind"], "material.reference")
            self.assertEqual(material["annotation_status"], "unannotated")
            self.assertEqual(material["retrieval_visibility"], "explicit")
            self.assertEqual(material["semanticize"], 0)

            connection = sqlite3.connect(path)
            try:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
                self.assertTrue(
                    {
                        "messages",
                        "summaries",
                        "semantic_summaries",
                        "turns",
                        "conversation_states",
                        "prompt_projections",
                        "projection_audits",
                    }.issubset(tables)
                )
                self.assertTrue(
                    {
                        "kind",
                        "turn_id",
                        "payload_json",
                        "trace_metadata_json",
                        "retrieval_visibility",
                        "row_version",
                    }.issubset(_columns(connection, "messages"))
                )
            finally:
                connection.close()


class V1MigrationTests(unittest.TestCase):
    def test_v2_database_is_converted_once_and_every_layer_is_marked_for_reindex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v2.sqlite3"
            _create_v2_database(path)

            store = SQLiteMemoryStore(str(path))
            records = [
                store.get_record_by_source_id("user-1"),
                store.get_record_by_source_id("mixed-summary"),
                store.get_record_by_source_id("semantic-1"),
            ]
            store.close()

            for record in records:
                self.assertIsNotNone(record)
                metadata = record["memory_metadata"]
                self.assertFalse(
                    {"keywords", "categories", "subject_scopes", "importance", "confidence"} & set(metadata)
                )
                self.assertEqual(record["index_status"], "pending")
                self.assertEqual(record["index_schema_version"], 0)
                self.assertEqual(record["index_key"], "")

            reopened = SQLiteMemoryStore(str(path))
            self.assertEqual(reopened.schema_version, 3)
            self.assertEqual(
                reopened.get_record_by_source_id("user-1")["memory_metadata"],
                records[0]["memory_metadata"],
            )
            reopened.close()

    def test_v1_database_is_backfilled_without_losing_content_or_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            _create_v1_database(path)

            store = SQLiteMemoryStore(str(path))
            self.assertEqual(store.schema_version, CURRENT_SCHEMA_VERSION)
            user = store.get_record_by_source_id("user-1")
            call = store.get_record_by_source_id("call-1")
            result = store.get_record_by_source_id("result-1")
            event = store.get_record_by_source_id("event-1")
            legacy = store.get_record_by_source_id("legacy-1")
            material = store.get_record_by_source_id("material-1")
            mixed = store.get_record_by_source_id("mixed-summary")
            trace = store.get_record_by_source_id("trace-summary")
            semantic = store.get_record_by_source_id("semantic-1")
            store.close()

            self.assertEqual(user["kind"], "message.user")
            self.assertEqual(user["semantic_text"], "我喜欢无糖可乐。")
            self.assertEqual(user["annotation_status"], "accepted_host")
            self.assertEqual(user["annotation_source"], "schema_v3_migration")
            self.assertEqual(user["memory_metadata"]["memory_facets"], ["preference"])
            self.assertEqual(user["memory_metadata"]["topic_terms"], ["可乐"])
            self.assertEqual(user["memory_metadata"]["retrieval_priority"], "high")
            self.assertEqual(user["retrieval_visibility"], "default")

            self.assertEqual(call["kind"], "tool.web_search.call")
            self.assertEqual(call["turn_role"], "action")
            self.assertEqual(result["kind"], "tool.web_search.result")
            self.assertEqual(result["turn_role"], "observation")
            self.assertEqual(call["correlation_id"], "call_001")
            self.assertEqual(result["correlation_id"], "call_001")
            self.assertEqual(call["memory_metadata"]["memory_facets"], [])
            self.assertNotIn("legacy_categories", call["trace_metadata"])
            self.assertEqual(call["retrieval_visibility"], "explicit")

            self.assertEqual(event["kind"], "event.qq.poke")
            self.assertEqual(event["memory_metadata"]["topic_terms"], ["戳一戳"])
            self.assertNotIn("legacy_categories", event["trace_metadata"])
            self.assertEqual(event["annotation_status"], "accepted_host")

            self.assertTrue(legacy["kind"].startswith("legacy."))
            self.assertEqual(legacy["content"], "旧内容必须保留。")
            self.assertEqual(legacy["semantic_text"], "旧内容必须保留。")

            self.assertEqual(material["kind"], "material.reference")
            self.assertEqual(material["trace_metadata"]["material_kind"], "image")
            self.assertEqual(material["trace_metadata"]["file_id"], "file_img_1")
            self.assertEqual(material["retrieval_visibility"], "explicit")
            self.assertEqual(material["semanticize"], 0)

            self.assertEqual(mixed["kind"], "memory.episode_summary")
            self.assertEqual(mixed["retrieval_visibility"], "default")
            self.assertEqual(mixed["memory_metadata"]["memory_facets"], ["preference"])
            self.assertEqual(mixed["memory_metadata"]["topic_terms"], ["可乐"])
            self.assertNotIn("legacy_categories", mixed["trace_metadata"])
            self.assertEqual(trace["kind"], "memory.operation_digest")
            self.assertEqual(trace["retrieval_visibility"], "explicit")
            self.assertEqual(trace["semanticize"], 0)
            self.assertEqual(semantic["memory_metadata"]["memory_facets"], ["preference"])
            self.assertEqual(semantic["memory_metadata"]["topic_terms"], ["无糖饮料"])
            self.assertEqual(semantic["annotation_status"], "derived")
            self.assertEqual(
                {item["index_status"] for item in (user, call, result, event, material, mixed, trace, semantic)},
                {"pending"},
            )

    def test_migration_is_idempotent_across_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            _create_v1_database(path)
            first = SQLiteMemoryStore(str(path))
            first.close()
            second = SQLiteMemoryStore(str(path))
            self.assertEqual(second.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(
                len(
                    second.get_messages_by_date_range(
                        namespace=Namespace(
                            user_id="user", tenant_id="tenant", domain_id="domain", conversation_id="conversation"
                        )
                    )
                ),
                6,
            )
            self.assertEqual(second.get_record_by_source_id("call-1")["kind"], "tool.web_search.call")
            second.close()

    def test_retrieval_v2_does_not_admit_v1_trace_category_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            _create_v1_database(path)
            store = SQLiteMemoryStore(str(path))
            embedding = HashedEmbeddingProvider()
            index = InMemoryVectorIndex(embedding=embedding)
            mem = MemorySystem(
                llm=_NoopLLM(),
                namespace=Namespace(
                    user_id="user",
                    tenant_id="tenant",
                    domain_id="domain",
                    conversation_id="conversation",
                ),
                timezone="Asia/Shanghai",
                store=store,
                index=index,
                embedding=embedding,
                config=MemoryConfig(enable_verifier=False),
            )
            try:
                mem.reindex_all()
                default_hits = mem.retrieve("晴天")
                with self.assertRaises(TypeError):
                    mem.retrieve("晴天", categories=["tool_trace"])
                explicit = mem.retrieve_structured(
                    "晴天",
                    include_explicit=True,
                    kind_patterns=["tool.web_search.*"],
                )

                self.assertFalse(any("晴天" in item for item in default_hits))
                self.assertEqual(explicit.status, "found")
                self.assertTrue(any("晴天" in item for item in explicit.rendered_texts))
            finally:
                mem.close()
                store.close()

    def test_failed_migration_rolls_back_all_alterations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.sqlite3"
            _create_v1_database(path)
            connection = sqlite3.connect(path)
            connection.execute("CREATE VIEW turns AS SELECT source_id AS turn_id FROM messages")
            connection.commit()
            connection.close()

            with self.assertRaises(SchemaError) as raised:
                SQLiteMemoryStore(str(path))
            self.assertIn("sqlite_schema_migration_failed", str(raised.exception))

            connection = sqlite3.connect(path)
            try:
                self.assertEqual(int(connection.execute("PRAGMA user_version").fetchone()[0]), 0)
                self.assertNotIn("kind", _columns(connection, "messages"))
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 6)
            finally:
                connection.close()

    def test_partial_unversioned_core_is_rejected_without_fake_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "partial.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE messages(source_id TEXT PRIMARY KEY)")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(SchemaError, "sqlite_schema_partial_core_tables"):
                SQLiteMemoryStore(str(path))


class NamespaceSafeRelationReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.ns = Namespace(
            user_id="user",
            tenant_id="tenant",
            domain_id="domain",
            conversation_id="group",
            actor=Actor(stable_id="qq:1", display_name="张三"),
        )
        self.other_actor = Namespace(
            user_id="user",
            tenant_id="tenant",
            domain_id="domain",
            conversation_id="group",
            actor=Actor(stable_id="qq:2", display_name="李四"),
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_actor_is_soft_for_reads_but_namespace_and_conversation_are_hard(self) -> None:
        entry = self.store.add_message(
            namespace=self.ns,
            role="user",
            content="张三的问题",
            timestamp=100,
            source_id="stimulus",
            turn_id="turn-1",
        )
        self.assertEqual(
            self.store.get_entry(namespace=self.other_actor, source_id=entry["source_id"]).namespace.actor.stable_id,
            "qq:1",
        )

        wrong_user = Namespace(user_id="other", tenant_id="tenant", domain_id="domain", conversation_id="group")
        wrong_conversation = Namespace(
            user_id="user", tenant_id="tenant", domain_id="domain", conversation_id="other-group"
        )
        with self.assertRaises(NamespaceError):
            self.store.get_entry(namespace=wrong_user, source_id="stimulus")
        with self.assertRaises(NamespaceError):
            self.store.get_entry(namespace=wrong_conversation, source_id="stimulus")

    def test_turn_and_correlation_reads_preserve_order_and_reject_cross_owner(self) -> None:
        self.store.add_message(
            namespace=self.ns,
            role="assistant.tool_call web_search call-a",
            content="call",
            timestamp=100,
            source_id="call",
            turn_id="turn-tools",
            correlation_id="call-a",
            turn_role="action",
        )
        self.store.add_message(
            namespace=self.other_actor,
            role="tool.web_search call-a",
            content="result",
            timestamp=101,
            source_id="result",
            turn_id="turn-tools",
            correlation_id="call-a",
            turn_role="observation",
        )
        self.store.add_message(
            namespace=self.ns,
            role="assistant",
            content="final",
            timestamp=102,
            source_id="final",
            turn_id="turn-tools",
            turn_role="final",
        )

        turn = self.store.get_turn_entries(namespace=self.other_actor, turn_id="turn-tools")
        branch = self.store.get_correlation_entries(namespace=self.ns, turn_id="turn-tools", correlation_id="call-a")
        self.assertEqual([item.source_id for item in turn], ["call", "result", "final"])
        self.assertEqual([item.source_id for item in branch], ["call", "result"])

        wrong = Namespace(user_id="other", tenant_id="tenant", domain_id="domain", conversation_id="group")
        with self.assertRaises(NamespaceError):
            self.store.get_turn_entries(namespace=wrong, turn_id="turn-tools")
        with self.assertRaises(NamespaceError):
            self.store.get_correlation_entries(namespace=wrong, turn_id="turn-tools", correlation_id="call-a")
        with self.assertRaises(NamespaceError):
            self.store.add_message(
                namespace=wrong,
                role="tool.web_search call-a",
                content="cross-owner pollution",
                timestamp=103,
                source_id="pollution",
                turn_id="turn-tools",
                correlation_id="call-a",
                turn_role="observation",
            )


if __name__ == "__main__":
    unittest.main()
