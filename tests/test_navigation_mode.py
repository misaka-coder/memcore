"""Navigation works over real SQLite truth without any embedding/search index."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from memcore import (
    Actor,
    ConfigError,
    EntryOrigin,
    HashedEmbeddingProvider,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)
from tests.test_slice4_compaction import CannedLLM, _complete_turn


class NavigationModeTests(unittest.TestCase):
    def memory(self, *, store=None, user="u", conversation="c", **settings):
        store = store or SQLiteMemoryStore(":memory:")
        self.addCleanup(store.close)
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id=user, conversation_id=conversation),
            timezone="Asia/Shanghai",
            store=store,
            config=MemoryConfig(retrieval_mode="navigation", **settings),
        )
        self.addCleanup(mem.close)
        return mem

    def history(self, mem, count=5):
        text = "原话定位记号:我想保留这段完整记录,包括冷门词和标点。" * 6
        for number in range(count):
            _complete_turn(
                mem, number, user_text=text, timestamp=1775782800 + number * 60, actor=Actor("qq:123456", "小周")
            )
            stats = mem.compact_due_sync()
        return text, stats

    def test_no_model_or_index_is_constructed(self):
        with (
            patch("memcore.memory_system.InMemoryVectorIndex", side_effect=AssertionError("index constructed")),
            patch(
                "memcore.memory_system.MemorySystem._resolve_embedding", side_effect=AssertionError("model resolved")
            ),
        ):
            mem = self.memory()
            _complete_turn(mem, 0, user_text="无向量写入", timestamp=1775782800)
            mem.build_context_surface(provider_profile="openai_chat")
            mem.read_timeline(date_from="2026-04-10")
        self.assertIsNone(mem.embedding)
        self.assertIsNone(mem.index)
        self.assertEqual(mem.retrieval_mode, "navigation")
        self.assertEqual(mem.embedding_status()["status"], "disabled")
        self.assertFalse(mem.embedding_status()["degraded"])
        self.assertEqual(mem.verify_embedding()["reason"], "retrieval_disabled")

    def test_hybrid_remains_default_and_requires_embedding(self):
        self.assertEqual(MemoryConfig().retrieval_mode, "hybrid")
        with self.assertRaisesRegex(ValueError, "embedding is required"):
            MemorySystem(llm=CannedLLM(), namespace=Namespace(user_id="u"), timezone="Asia/Shanghai")

    def test_invalid_mode_and_conflicting_dependencies_fail_construction(self):
        for mode in ("", "keyword", "Navigation", False, None):
            with self.subTest(mode=mode), self.assertRaises(ConfigError):
                MemoryConfig(retrieval_mode=mode)
            with self.subTest(mode=mode), self.assertRaises(ConfigError):
                build_native_memory_tool_specs(retrieval_mode=mode)
        for dependency in ({"embedding": "a-model"}, {"index": object()}):
            with self.subTest(dependency=next(iter(dependency))), self.assertRaisesRegex(ValueError, "navigation mode"):
                MemorySystem(
                    llm=CannedLLM(),
                    namespace=Namespace(user_id="u"),
                    timezone="Asia/Shanghai",
                    config=MemoryConfig(retrieval_mode="navigation"),
                    **dependency,
                )

    def test_recent_context_timeline_and_complete_original_text_remain_visible(self):
        mem = self.memory()
        text = "这段原话必须完整保留。" * 50
        _complete_turn(mem, 0, user_text=text, timestamp=1775782800, actor=Actor("qq:123456", "小周"))
        context = mem.build_prompt_context(current={"timestamp": 1775782802})
        self.assertIn(text, mem.render_prompt_context(context))
        result = dispatch_native_memory_tool(
            "read_timeline", {"date_from": "2026-04-10", "page_token_budget": 1}, mem=mem
        )
        self.assertTrue(result["ok"])
        self.assertIn(text, result["result"]["text"])
        self.assertIn("小周", result["result"]["text"])

    def test_real_raw_and_semantic_compaction_work_with_disabled_index_status(self):
        mem = self.memory(raw_token_trigger=1, episodic_compact_trigger_count=3, episodic_compact_batch_size=2)
        text, stats = self.history(mem)
        self.assertEqual(stats["index_status"], "disabled")
        semantic = mem.browse_memory(keywords=["学习"], node_types=["semantic"])
        self.assertGreater(semantic["matched_card_count"], 0)
        card = semantic["cards"][0]
        self.assertEqual(mem.open_memory(memory_id=card["memory_id"], view="content")["status"], "ok")
        timeline = mem.read_timeline(date_from="2026-04-10")
        self.assertIn(text, timeline["text"])

    def test_keyword_directory_opens_matching_original_utterances_and_continues(self):
        mem = self.memory(raw_token_trigger=1, episodic_compact_trigger_count=99)
        text, _ = self.history(mem, 4)
        first = mem.browse_memory(keywords=["原话定位记号"], page_size=1)
        self.assertGreater(first["matched_card_count"], 1)
        self.assertFalse(first["page_complete"])
        second = mem.browse_memory(cursor=first["next_cursor"])
        self.assertNotEqual(first["cards"][0]["memory_id"], second["cards"][0]["memory_id"])
        matched = mem.open_memory(match_ref=first["cards"][0]["match_ref"])
        self.assertEqual(matched["status"], "found")
        self.assertIn(text, matched["text"])
        self.assertIn("小周", matched["text"])

    def test_native_specs_and_explicit_fuzzy_calls_report_disabled(self):
        mem = self.memory()
        current = mem.record_user_turn("待查询的内容", timestamp=1775782800)
        for format_name, name_key in (
            ("plain", "name"),
            ("anthropic", "name"),
            ("openai_responses", "name"),
            ("openai", "function"),
        ):
            specs = build_native_memory_tool_specs(
                retrieval_mode=mem.retrieval_mode, tool_format=format_name, include_material_tool=False
            )
            names = [item[name_key]["name"] if name_key == "function" else item[name_key] for item in specs]
            self.assertEqual(names, ["browse_memory", "open_memory", "read_timeline"])
        for result in (
            mem.retrieve_structured("原话"),
            mem.retrieve_for_turn_structured(current=current, query="原话"),
        ):
            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.reason, "retrieval_disabled")
            self.assertEqual(result.matches, ())
            self.assertIn("browse_memory", " ".join(result.suggested_next_actions))
        dispatched = dispatch_native_memory_tool("retrieve_for_turn", {"query": "原话"}, mem=mem, current=current)
        self.assertEqual(dispatched["result"]["status"], "unavailable")
        self.assertEqual(dispatched["result"]["reason"], "retrieval_disabled")

    def test_semantic_history_repair_and_rollback_preserve_truth_without_index(self):
        mem = self.memory(episodic_compact_trigger_count=3, episodic_compact_batch_size=2)
        for number in range(6):
            mem.store.add_message(
                namespace=mem.namespace,
                role="user",
                source_id=f"r{number}",
                content=f"日{number} 原话",
                timestamp=100 + number,
            )
            mem.store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": f"e{number}",
                    "timestamp": 200 + number,
                    "period_start_ts": 100 + number,
                    "period_end_ts": 100 + number,
                    "source_ids": [f"r{number}"],
                    "diary_summary": f"日{number} 摘要",
                },
            )
        for semantic_id, sources in (("old-a", ["e0", "e2", "e4"]), ("old-b", ["e1", "e3", "e5"])):
            mem.store.add_semantic_summary(
                namespace=mem.namespace,
                record={
                    "semantic_id": semantic_id,
                    "timestamp": 205,
                    "semantic_summary": "旧交错历史",
                    "source_summary_ids": sources,
                    "reinforcement_count": 2,
                },
            )
            mem.store.mark_summaries_semanticized(sources, semantic_id)
        before = mem.store.get_semantic_history_snapshot(namespace=mem.namespace)
        plan = mem.plan_semantic_history_repair()
        repaired = mem.repair_semantic_history(plan)
        self.assertEqual(repaired["status"], "committed")
        self.assertEqual(repaired["index_status"], "disabled")
        self.assertEqual(repaired["index_repaired"], 0)
        rolled_back = mem.rollback_semantic_history_repair(plan["repair_id"])
        self.assertEqual(rolled_back["status"], "rolled_back")
        self.assertEqual(rolled_back["index_status"], "disabled")
        after = mem.store.get_semantic_history_snapshot(namespace=mem.namespace)
        self.assertEqual(after["raw_versions"], before["raw_versions"])
        self.assertEqual({r["semantic_id"] for r in after["semantics"]}, {"old-a", "old-b"})
        self.assertEqual(mem.retrieve("原话"), [])

    def test_index_maintenance_does_not_scan_truth_or_claim_repaired_entries(self):
        mem = self.memory()
        raw = mem.record_user_turn("保持待重建", timestamp=1775782800)
        self.assertEqual(raw["index_status"], "pending")
        with (
            patch.object(mem.store, "list_index_records", side_effect=AssertionError("full scan")),
            patch.object(mem.store, "list_pending_index", side_effect=AssertionError("pending scan")),
        ):
            for stats in (mem.reindex_all(), mem.reindex_pending()):
                self.assertEqual(stats["status"], "disabled")
                self.assertEqual(stats["scanned"], 0)
        self.assertEqual(mem.store.get_record_by_source_id(raw["source_id"])["index_status"], "pending")

    def test_metadata_staging_updates_and_explicit_opt_out_remain_truthful(self):
        mem = self.memory()
        handle = mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="原始内容",
                    timestamp=1775782800,
                )
            ]
        )
        raw_id = handle.stimuli[0].source_id
        staged = mem.stage_turn_metadata(raw_id, {"entity_anchors": ["项目甲"]})
        self.assertEqual(staged["status"], "staged")
        self.assertEqual(staged["index_status"], "disabled")
        completed = mem.complete_turn(
            turn_id=handle.turn_id, semantic_text="已记下", provider_output_raw="已记下", timestamp=1775782801
        )
        self.assertTrue(completed.completed)
        updated = mem.update_turn_metadata(raw_id, {"entity_anchors": ["项目乙"]})
        self.assertEqual(updated["status"], "updated")
        self.assertEqual(updated["index_status"], "disabled")
        self.assertEqual(mem.store.get_record_by_source_id(raw_id)["memory_metadata"]["entity_anchors"], ["项目乙"])
        event = mem.record_user_turn("停用索引的记录", index_in_vector=False)
        self.assertEqual(event["index_status"], "skipped")
        self.assertEqual(mem.update_turn_metadata(event["source_id"], {})["index_status"], "skipped")

    def test_hard_namespace_isolation_still_applies_without_an_index(self):
        mem = self.memory(raw_token_trigger=1, episodic_compact_trigger_count=99)
        self.history(mem, 2)
        card = mem.browse_memory(keywords=["原话定位记号"])["cards"][0]
        foreign = self.memory(store=mem.store, user="foreign")
        self.assertEqual(foreign.browse_memory(keywords=["原话定位记号"], cross_conversation=True)["status"], "empty")
        self.assertNotEqual(foreign.open_memory(memory_id=card["memory_id"], view="sources")["status"], "ok")

    def test_restart_and_hybrid_rebuild_restore_existing_raw_and_summary_records(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite3"
            store = SQLiteMemoryStore(str(path))
            mem = self.memory(store=store, raw_token_trigger=1, episodic_compact_trigger_count=99)
            text, _ = self.history(mem, 3)
            mem.close()
            store.close()
            reopened = SQLiteMemoryStore(str(path))
            navigation = self.memory(store=reopened)
            self.assertGreater(navigation.browse_memory(keywords=["原话定位记号"])["matched_card_count"], 0)
            hybrid = MemorySystem(
                llm=CannedLLM(),
                namespace=navigation.namespace,
                timezone="Asia/Shanghai",
                store=reopened,
                embedding=HashedEmbeddingProvider(),
            )
            self.addCleanup(hybrid.close)
            try:
                stats = hybrid.reindex_all()
                self.assertGreater(stats["reindexed"], 0)
                self.assertEqual(stats["failed"], 0)
                result = hybrid.retrieve_structured("原话定位记号", source_layers=["raw"])
                self.assertEqual(result.status, "found")
                self.assertIn(text, "\n".join(result.rendered_texts))
            finally:
                hybrid.close()
                navigation.close()
                reopened.close()

    def test_forgetting_without_an_index_deletes_only_owned_truth(self):
        mem = self.memory()
        foreign = self.memory(store=mem.store, user="foreign")
        mem.record_user_turn("本人的记忆")
        foreign.record_user_turn("其他人的记忆")
        result = mem.forget_namespace()
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["index_status"], "disabled")
        self.assertFalse(result["index_deleted"])
        self.assertEqual(len(foreign.store.get_unsummarized_messages(namespace=foreign.namespace)), 1)


if __name__ == "__main__":
    unittest.main()
