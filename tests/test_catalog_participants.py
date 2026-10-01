"""Source-derived participant inheritance, exact catalog filters and safe paging."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from memcore import (
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)
from memcore.store.migrations import CURRENT_SCHEMA_VERSION, migrate_database


class _ParticipantLLM(LLMClient):
    def call(self, request):
        return LLMResult(
            ok=True,
            data={
                "diary_summary": "讨论答辩。",
                "semantic_summary": "持续准备答辩。",
                "importance": 0.8,
                "memory_title": "答辩计划",
                "catalog_hint": "答辩准备记录",
                "stable_facts": ["准备答辩"],
                "memory_metadata": {"entity_anchors": ["答辩项目"], "memory_facets": ["plan"]},
                "participant_refs": [{"actor_id": "qq:999", "display_name": "模型猜测"}],
                "important_people": ["正文提及者"],
            },
        )


class CatalogParticipantTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.namespace = Namespace(tenant_id="t", user_id="u", domain_id="d", conversation_id="group")
        self.embedding = HashedEmbeddingProvider()
        self.mem = self.system(self.namespace)
        self.addCleanup(self.store.close)
        self.addCleanup(self.mem.close)

    def system(self, namespace, **config):
        return MemorySystem(
            namespace=namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            llm=_ParticipantLLM(),
            embedding=self.embedding,
            index=InMemoryVectorIndex(embedding=self.embedding),
            config=MemoryConfig(**config),
        )

    def raw(self, sid, actor_id="qq:1", name="小林", *, namespace=None, timestamp=100, **fields):
        namespace = replace(namespace or self.namespace, actor=Actor(stable_id=actor_id, display_name=name))
        return self.store.add_message(
            namespace=namespace,
            source_id=sid,
            role="user",
            content="提到了旁观者，但不是参与者名单。",
            timestamp=timestamp,
            annotation_status="missing",
            **fields,
        )

    def episode(self, sid, sources=(), *, namespace=None, timestamp=200, **fields):
        return self.store.add_summary(
            namespace=namespace or self.namespace,
            record={
                "summary_id": sid,
                "timestamp": timestamp,
                "diary_summary": "讨论答辩，也提到了旁观者。",
                "source_ids": list(sources),
                "memory_metadata": {"entity_anchors": ["答辩项目"], "memory_facets": ["plan"]},
                **fields,
            },
        )

    def ids(self, **args):
        result = self.mem.browse_memory(**args)
        self.assertIn(result["status"], ("ok", "empty"), result)
        return [card["memory_id"] for card in result["cards"]]

    def test_raw_compaction_collects_latest_name_and_target_without_model_guesses(self):
        mem = self.system(self.namespace, raw_token_trigger=1, episodic_compact_trigger_count=99)
        self.addCleanup(mem.close)
        handle = mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="开始讨论答辩。",
                    actor=Actor(stable_id="qq:1", display_name="旧昵称"),
                    target_actor=Actor(stable_id="qq:2", display_name="小周"),
                    timestamp=100,
                )
            ]
        )
        for name, timestamp in (("新昵称", 110), ("", 120)):
            mem.append_entry(
                TimelineEntryInput(
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.INTERMEDIATE,
                    semantic_text="补充答辩安排。",
                    actor=Actor(stable_id="qq:1", display_name=name),
                    timestamp=timestamp,
                ),
                turn_id=handle.turn_id,
            )
        self.assertTrue(
            mem.complete_turn(
                turn_id=handle.turn_id, semantic_text="收到。", provider_output_raw="收到。", timestamp=130
            ).completed
        )
        result = mem.compact_due_sync()
        self.assertEqual(result["status"], "compacted", result)
        saved = self.store.get_visible_episodic_summaries(namespace=self.namespace, limit=10)[0]
        self.assertEqual(
            saved["participant_refs"],
            [
                {"actor_id": "qq:1", "display_name": "新昵称"},
                {"actor_id": "qq:2", "display_name": "小周"},
            ],
        )
        card = mem.browse_memory(participants=["qq:1"])["cards"][0]
        self.assertEqual(card["participant_refs"], saved["participant_refs"])
        self.assertEqual(self.ids(participants=["模型猜测"]), [])

    def test_exact_names_ids_any_all_and_keyword_date_intersection(self):
        self.raw("r1", name="Ａlice")
        self.raw("r2", actor_id="qq:2", name="小周", timestamp=101)
        self.episode("both", ["r2", "r1"], timestamp=200)
        self.raw("r3", actor_id="qq:3", name="Alice")
        self.episode("same-name", ["r3"], timestamp=300)
        self.assertEqual(self.ids(participants=["alice"]), ["both", "same-name"])
        self.assertEqual(self.ids(participants=["Ali"]), [])
        self.assertEqual(self.ids(participants=["qq:1"]), ["both"])
        self.assertEqual(self.ids(participants=["QQ:1"]), [])
        self.assertEqual(self.ids(participants=["Alice", "小周"], participant_match="all"), ["both"])
        self.assertEqual(self.ids(participants=["qq:3", "小周"]), ["both", "same-name"])
        self.assertEqual(self.ids(participants=["qq:1"], keywords=["答辩"]), ["both"])
        self.assertEqual(self.ids(participants=["qq:1"], keywords=["天气"]), [])
        self.assertEqual(
            self.ids(
                participants=["qq:1"],
                time_range={
                    "start_at": "1970-01-01 08:03:20",
                    "end_at": "1970-01-01 08:03:21",
                },
            ),
            ["both"],
        )
        self.assertEqual(self.ids(participants=["qq:1"], date_from="1970-01-02"), [])
        self.assertEqual(self.ids(participants=["旁观者"]), [])

    def test_legacy_source_order_latest_nonempty_name_and_open_cards_agree(self):
        self.raw("later", name="新名", timestamp=120)
        self.raw("early", name="旧名", timestamp=100)
        self.raw("empty", name="", timestamp=130)
        self.raw("same-name", actor_id="qq:2", name="新名", timestamp=140)
        self.episode("old", ["empty", "same-name", "later", "early"])
        card = self.mem.browse_memory(participants=["新名"])["cards"][0]
        self.assertEqual(
            card["participant_refs"],
            [
                {"actor_id": "qq:1", "display_name": "新名"},
                {"actor_id": "qq:2", "display_name": "新名"},
            ],
        )
        self.assertEqual(self.ids(participants=["旧名"]), [])
        self.assertEqual(self.mem.open_memory(memory_id="old")["result"]["participant_refs"], card["participant_refs"])
        self.assertEqual(
            self.mem.open_memory(memory_ids=["old"], view="content")["result"]["items"][0]["result"]["card"][
                "participant_refs"
            ],
            card["participant_refs"],
        )

    def test_independent_semantic_batches_preserve_own_system_identity(self):
        mem = self.system(
            self.namespace, raw_token_trigger=100000, episodic_compact_trigger_count=3, episodic_compact_batch_size=2
        )
        self.addCleanup(mem.close)
        for i in range(3):
            self.raw(f"r{i}", actor_id=f"qq:{i + 1}", name=f"成员{i}", timestamp=100 + i)
            self.episode(f"e{i}", [f"r{i}"], timestamp=200 + i)
        first = mem.compact_due_sync()
        self.assertEqual(first["semantic_created"], 1, first)
        semantic = self.store.get_recent_semantic_summaries(namespace=self.namespace)[0]
        self.assertEqual([ref["actor_id"] for ref in semantic["participant_refs"]], ["qq:1", "qq:2"])
        self.raw("new-name", name="更新昵称", timestamp=400)
        self.episode("e3", ["new-name"], timestamp=500)
        self.episode("e4", timestamp=600)
        second = mem.compact_due_sync()
        self.assertEqual(second["semantic_created"], 1, second)
        saved = self.store.get_recent_semantic_summaries(namespace=self.namespace)[0]
        self.assertEqual(
            saved["participant_refs"],
            [
                {"actor_id": "qq:3", "display_name": "成员2"},
                {"actor_id": "qq:1", "display_name": "更新昵称"},
            ],
        )
        self.assertEqual(
            self.ids(participants=["更新昵称", "qq:3"], participant_match="all", node_types=["semantic"]),
            [saved["semantic_id"]],
        )
        children = mem.open_memory(memory_id=saved["semantic_id"], view="sources")["result"]["sources"]
        self.assertTrue(any(card["participant_refs"] for card in children))
        self.assertNotIn("qq:999", json.dumps(saved["participant_refs"]))

    def test_legacy_semantic_refs_follow_children_and_survive_missing_sources(self):
        self.raw("r")
        self.episode("e", ["r"])
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={
                "semantic_id": "s",
                "timestamp": 300,
                "semantic_summary": "答辩",
                "source_summary_ids": ["e", "missing"],
            },
        )
        self.assertEqual(self.ids(participants=["qq:1"], node_types=["semantic"]), ["s"])
        saved = self.store.get_retrieval_record(namespace=self.namespace, source_id="s")
        saved["participant_refs"] = [{"actor_id": "qq:2", "display_name": "保留者"}]
        self.store.add_semantic_summary(namespace=self.namespace, record=saved)
        self.store._conn.execute("DELETE FROM messages WHERE source_id = 'r'")
        self.store._conn.commit()
        self.assertEqual(self.ids(participants=["qq:1"], node_types=["semantic"]), [])
        self.assertEqual(self.ids(participants=["保留者"], node_types=["semantic"]), ["s"])

    def test_raw_summary_link_never_substitutes_parent_participants(self):
        self.raw("a", actor_id="qq:1", name="小林")
        self.raw("b", actor_id="qq:2", name="小周")
        self.episode("episode", ["a", "b"])
        self.store.mark_messages_summarized(["a", "b"], "episode")
        single = self.mem.open_memory(memory_id="a", view="card")
        batch = self.mem.open_memory(memory_ids=["a", "episode"], view="card")
        self.assertEqual(single["node_type"], "raw")
        self.assertEqual(single["result"]["participant_refs"], [])
        self.assertEqual(batch["result"]["items"][0], single)
        self.assertEqual(len(batch["result"]["items"][1]["result"]["participant_refs"]), 2)

    def test_lineage_cannot_import_foreign_conversation_namespace_or_trace_actors(self):
        other_conversation = replace(self.namespace, conversation_id="other")
        other_user = replace(self.namespace, user_id="other-user")
        self.raw("foreign-conversation", name="别的会话", namespace=other_conversation)
        self.raw("foreign-user", name="别的用户", namespace=other_user)
        self.raw("hidden", name="隐藏者", retrieval_visibility="never")
        self.raw("trace", name="操作人员", kind="tool.search.result", retrieval_visibility="explicit")
        self.raw("material", name="材料人员", kind="material.image.reference", retrieval_visibility="default")
        self.episode("bad-lineage", ["foreign-conversation", "foreign-user", "hidden", "trace", "material"])
        for name in ("别的会话", "别的用户", "隐藏者", "操作人员", "材料人员"):
            self.assertEqual(self.ids(participants=[name]), [])
        self.assertEqual(self.ids(participants=["别的会话"], cross_conversation=True), ["bad-lineage"])
        for name in ("别的用户", "隐藏者", "操作人员", "材料人员"):
            self.assertEqual(self.ids(participants=[name], cross_conversation=True), [])

    def test_native_schema_strict_filters_receipt_and_cursor_continuation(self):
        for i in range(3):
            self.raw(f"r{i}")
            self.episode(f"e{i}", [f"r{i}"], timestamp=200 + i)
        first = dispatch_native_memory_tool(
            "browse_memory",
            {
                "participants": ["qq:1"],
                "participant_match": "all",
                "page_size": 1,
            },
            mem=self.mem,
        )
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["receipt"]["selector"]["participants"], ["qq:1"])
        cursor = first["result"]["next_cursor"]
        second = dispatch_native_memory_tool(
            "browse_memory",
            {
                "cursor": cursor,
                "participants": None,
                "participant_match": None,
            },
            mem=self.mem,
        )
        self.assertEqual(second["result"]["cards"][0]["memory_id"], "e1")
        self.assertEqual(second["receipt"]["selector"]["participant_match"], "all")
        self.assertFalse(second["result"]["coverage"]["participant_filtered"])
        for arguments in (
            {"participants": "小林"},
            {"participants": [1]},
            {"participants": [" "]},
            {"participants": ["qq:1"], "participant_match": "fuzzy"},
            {"participant_match": "all"},
            {"cursor": cursor, "participants": []},
            {"cursor": cursor, "participant_match": "any"},
        ):
            direct = self.mem.browse_memory(**arguments)
            native = dispatch_native_memory_tool("browse_memory", arguments, mem=self.mem)
            self.assertEqual(direct["status"], "invalid_filter", direct)
            self.assertFalse(native["ok"], native)
        for tool_format in ("plain", "openai", "openai_responses", "anthropic"):
            spec = next(
                spec
                for spec in build_native_memory_tool_specs(tool_format=tool_format)
                if (spec.get("name") or spec.get("function", {}).get("name")) == "browse_memory"
            )
            self.assertIn("participant_match", json.dumps(spec))

    def test_participant_cursor_detects_identity_repair_and_empty_is_honest(self):
        self.raw("a")
        self.raw("b")
        self.episode("e1", ["a"], timestamp=200)
        self.episode("e2", ["b"], timestamp=300)
        first = self.mem.browse_memory(participants=["qq:1"], page_size=1)
        self.store._conn.execute("UPDATE messages SET actor_display_name = '新名' WHERE source_id = 'b'")
        self.store._conn.commit()
        self.assertEqual(
            self.mem.browse_memory(cursor=first["next_cursor"])["reason"], "catalog_changed_restart_required"
        )
        empty = self.mem.browse_memory(participants=["不存在的人"])
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(empty["reason"], "no_catalog_participant_matches")
        self.assertEqual(empty["coverage"]["total_source_count"], 2)
        self.assertIsNone(empty["coverage"]["requested_range"]["start_ts"])

    def test_identity_queries_batch_without_reading_bodies_or_calling_models(self):
        for i in range(405):
            self.raw(f"r{i}", actor_id=f"qq:{i + 1}", name=f"成员{i}")
        self.episode("big", [f"r{i}" for i in range(405)])
        statements = []
        self.store._conn.set_trace_callback(statements.append)
        try:
            with (
                patch.object(self.mem.llm, "call", side_effect=AssertionError("LLM read")),
                patch.object(self.embedding, "embed_query", side_effect=AssertionError("embedding read")),
            ):
                refs = self.store.get_catalog_participant_refs(namespace=self.namespace, memory_ids=("big",))
        finally:
            self.store._conn.set_trace_callback(None)
        self.assertEqual(len(refs["big"]), 405)
        reads = [query for query in statements if query.startswith("SELECT")]
        self.assertEqual(len(reads), 4)
        self.assertTrue(
            all("SELECT *" not in query and "semantic_text" not in query and "content" not in query for query in reads)
        )
        for query in reads:
            plan = [row[3] for row in self.store._conn.execute("EXPLAIN QUERY PLAN " + query)]
            self.assertTrue(any("USING INDEX" in step and "_id=?" in step for step in plan), plan)

    def test_unsupported_and_ambiguous_identity_reads_never_succeed_broadly(self):
        self.episode("e")
        with patch.object(self.store, "get_catalog_participant_refs", side_effect=NotImplementedError):
            self.assertEqual(self.mem.browse_memory(participants=["小林"])["status"], "unavailable")
            self.assertEqual(
                self.mem.browse_memory(
                    time_range={
                        "start_at": "1970-01-01 08:00:00",
                        "end_at": "1970-01-01 09:00:00",
                    }
                )["status"],
                "ok",
            )
        self.store.add_semantic_summary(namespace=self.namespace, record={"semantic_id": "e", "timestamp": 300})
        self.assertEqual(
            self.mem.browse_memory(participants=["小林"], node_types=["episodic", "semantic"])["status"], "unavailable"
        )

    def test_schema_7_upgrade_is_additive_idempotent_and_keeps_legacy_evidence(self):
        self.raw("r")
        self.episode("e", ["r"])
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={
                "semantic_id": "s",
                "timestamp": 300,
                "semantic_summary": "原有正文",
                "source_summary_ids": ["e"],
            },
        )
        connection = self.store._conn
        connection.execute("ALTER TABLE semantic_summaries DROP COLUMN participant_refs_json")
        connection.execute("PRAGMA user_version = 7")
        connection.commit()
        migrate_database(connection)
        migrate_database(connection)
        self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], CURRENT_SCHEMA_VERSION)
        stored = self.store.get_retrieval_record(namespace=self.namespace, source_id="s")
        self.assertEqual(stored["semantic_summary"], "原有正文")
        self.assertEqual(stored["participant_refs"], [])
        self.assertEqual(self.ids(participants=["qq:1"], node_types=["semantic"]), ["s"])


if __name__ == "__main__":
    unittest.main()
