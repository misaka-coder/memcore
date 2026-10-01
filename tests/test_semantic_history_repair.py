"""Atomic repair of interleaved semantic sources and guarded recovery."""

import unittest
from dataclasses import replace

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
)
from memcore.llm.base import LLMResult
from memcore.semantic_history import contiguous_episode_ids
from tests.test_slice4_compaction import CannedLLM


class FaultIndex(InMemoryVectorIndex):
    fail_delete = False

    def delete(self, source_ids):
        if self.fail_delete:
            raise RuntimeError("delete unavailable")
        return super().delete(source_ids)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.ns = Namespace(user_id="repair", conversation_id="group-a")
        self.store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        self.index = FaultIndex(embedding=embedding)
        self.mem = MemorySystem(
            namespace=self.ns,
            timezone="Asia/Shanghai",
            store=self.store,
            llm=CannedLLM(),
            embedding=embedding,
            index=self.index,
            config=MemoryConfig(episodic_compact_trigger_count=3, episodic_compact_batch_size=2),
        )
        self.addCleanup(self.store.close)
        self.addCleanup(self.mem.close)
        for i in range(6):
            self.store.add_message(
                namespace=self.ns, role="user", source_id=f"r{i}", content=f"日{i} 对话", timestamp=100 + i
            )
            self.store.add_summary(
                namespace=self.ns,
                record={
                    "summary_id": f"e{i}",
                    "timestamp": 200 + i,
                    "period_start_ts": 100 + i,
                    "period_end_ts": 100 + i,
                    "source_ids": [f"r{i}"],
                    "diary_summary": f"日{i} 摘要",
                },
            )
        for sid, sources in (("old-a", ["e0", "e2", "e4"]), ("old-b", ["e1", "e3", "e5"])):
            self.store.add_semantic_summary(
                namespace=self.ns,
                record={
                    "semantic_id": sid,
                    "timestamp": 205,
                    "semantic_summary": "旧交错历史",
                    "source_summary_ids": sources,
                    "reinforcement_count": 2,
                },
            )
            self.store.mark_summaries_semanticized(sources, sid)
        self.mem.reindex_all()

    def test_source_partition_references_cursors_and_rollback(self):
        before = self.store.get_semantic_history_snapshot(namespace=self.ns)
        plan = self.mem.plan_semantic_history_repair()
        self.assertEqual(plan["groups"], [["e0", "e1"], ["e2", "e3"]])
        self.assertEqual(plan["pending_summary_ids"], ["e4", "e5"])
        cursor = self.mem.browse_memory(
            time_range={"start_at": "1970-01-01T00:00:00Z", "end_at": "1970-01-02T00:00:00Z"}, page_size=1
        )["next_cursor"]
        result = self.mem.repair_semantic_history(plan)
        self.assertEqual(result["status"], "committed", result)
        self.assertEqual(result["index_status"], "indexed")
        after = self.store.get_semantic_history_snapshot(namespace=self.ns)
        self.assertEqual(after["raw_versions"], before["raw_versions"])
        self.assertEqual(
            [r["diary_summary"] for r in after["episodes"]], [r["diary_summary"] for r in before["episodes"]]
        )
        sources = [sid for row in after["semantics"] for sid in row["source_summary_ids"]]
        self.assertEqual(sources + plan["pending_summary_ids"], plan["affected_summary_ids"])
        for row in after["semantics"]:
            self.assertTrue(contiguous_episode_ids(after["episodes"], row["source_summary_ids"]))
            self.assertEqual(row["reinforcement_count"], 1)
        self.assertEqual(self.mem.open_memory(memory_id="old-a")["status"], "replaced")
        self.assertEqual(self.mem.open_memory(memory_id="old-b", view="sources")["reason"], "semantic_history_repaired")
        self.assertEqual(self.mem.browse_memory(cursor=cursor)["reason"], "catalog_changed_restart_required")
        self.assertIsNone(
            self.store.get_memory_replacements(namespace=replace(self.ns, conversation_id="other"), memory_id="old-a")
        )
        self.assertEqual(self.mem.plan_semantic_history_repair()["status"], "clean")
        self.store.add_message(namespace=self.ns, role="user", source_id="new", content="后来聊天", timestamp=300)
        rolled = self.mem.rollback_semantic_history_repair(plan["repair_id"])
        self.assertEqual(rolled["status"], "rolled_back", rolled)
        self.assertIsNotNone(self.store.get_entry(namespace=self.ns, source_id="new"))
        self.assertEqual(self.mem.open_memory(memory_id=plan["new_memory_ids"][0])["status"], "replaced")
        self.assertEqual(
            {row["semantic_id"] for row in self.store.get_semantic_history_snapshot(namespace=self.ns)["semantics"]},
            {"old-a", "old-b"},
        )

    def test_generation_failure_publishes_nothing(self):
        plan = self.mem.plan_semantic_history_repair()
        before = self.store.get_semantic_history_snapshot(namespace=self.ns)
        original = self.mem._compaction.llm.call
        calls = 0

        def fail_second(request):
            nonlocal calls
            calls += 1
            return original(request) if calls == 1 else LLMResult(ok=False, error="unavailable")

        self.mem._compaction.llm.call = fail_second
        result = self.mem.repair_semantic_history(plan)
        self.assertEqual(result["status"], "generation_failed", result)
        self.assertEqual(result["prepared_count"], 1)
        self.assertEqual(self.store.get_semantic_history_snapshot(namespace=self.ns), before)

    def test_separate_preparation_never_publishes_and_rechecks_at_commit(self):
        before = self.store.get_semantic_history_snapshot(namespace=self.ns)
        staged = self.mem.prepare_semantic_history_repair(self.mem.plan_semantic_history_repair())
        self.assertEqual(staged["status"], "prepared")
        self.assertEqual(self.store.get_semantic_history_snapshot(namespace=self.ns), before)
        self.store.mark_summaries_semanticized(["e1"], "changed")
        self.assertEqual(self.mem.commit_prepared_semantic_history_repair(staged)["status"], "stale_plan")

    def test_version_conflict_and_raw_source_loss_refuse_publication(self):
        plan = self.mem.plan_semantic_history_repair()
        self.store._conn.execute("UPDATE summaries SET row_version = row_version + 1 WHERE summary_id = 'e2'")
        self.store._conn.commit()
        self.assertEqual(self.mem.repair_semantic_history(plan)["status"], "stale_plan")
        self.store._conn.execute("DELETE FROM messages WHERE source_id = 'r2'")
        self.store._conn.commit()
        self.assertEqual(self.mem.plan_semantic_history_repair()["reason"], "missing_raw_sources")

    def test_index_delete_outbox_and_sqlite_truth_prevent_stale_recall(self):
        plan = self.mem.plan_semantic_history_repair()
        self.index.fail_delete = True
        result = self.mem.repair_semantic_history(plan)
        self.assertEqual(result["status"], "committed")
        self.assertEqual(result["index_status"], "pending")
        self.assertEqual(result["obsolete_index_failed"], 2)
        self.assertFalse(
            {"old-a", "old-b"}.intersection(
                m.source_id for m in self.mem.retrieve_structured(query="旧交错历史").matches
            )
        )
        self.index.fail_delete = False
        self.assertEqual(self.mem.reindex_pending()["obsolete_index_deleted"], 2)
        self.assertEqual(self.store.list_semantic_index_cleanup(namespace=self.ns), [])
        self.store.mark_summaries_semanticized(["e4"], "another")
        self.assertEqual(self.mem.rollback_semantic_history_repair(plan["repair_id"])["status"], "stale_repair")

    def test_namespace_forget_removes_recovery_history(self):
        plan = self.mem.plan_semantic_history_repair()
        self.mem.repair_semantic_history(plan)
        self.mem.forget_namespace()
        self.assertIsNone(self.store.get_memory_replacements(namespace=self.ns, memory_id="old-a"))
        self.assertEqual(self.mem.rollback_semantic_history_repair(plan["repair_id"])["status"], "empty")

    def test_stale_cleanup_cannot_erase_an_index_restored_by_rollback(self):
        plan = self.mem.plan_semantic_history_repair()
        delete = self.index.delete
        rolled = False

        def delayed_delete(ids):
            nonlocal rolled
            if not rolled and set(ids) == set(plan["affected_memory_ids"]):
                rolled = True
                result = self.mem.rollback_semantic_history_repair(plan["repair_id"])
                self.assertEqual(result["status"], "rolled_back")
            delete(ids)

        self.index.delete = delayed_delete
        self.mem.repair_semantic_history(plan)
        hits = self.index.keyword_search(
            query_text="旧交错历史", entity_anchors=[], topic_terms=[], where={}, n_results=10
        )
        self.assertEqual({hit["source_id"] for hit in hits}, {"old-a", "old-b"})
