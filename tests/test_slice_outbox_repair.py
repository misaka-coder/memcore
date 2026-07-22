"""outbox 自愈:向量后端故障时记录仍安全落库(pending),reindex_pending 后补齐索引。"""

from __future__ import annotations

import unittest

from memcore import HashedEmbeddingProvider, MemoryConfig, MemorySystem, Namespace, SQLiteMemoryStore
from memcore.index.memory_index import InMemoryVectorIndex
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


class _SummaryLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            return LLMResult(ok=True, data={"diary_summary": "压缩了一段", "importance": 0.6, "core_facts": ["事实"]})
        return LLMResult(ok=True, data={})


class FlakyIndex(InMemoryVectorIndex):
    """可开关的故障索引:fail=True 时 upsert 抛错,模拟向量后端不可用。"""

    fail = True

    def upsert(self, entries):
        if self.fail:
            raise RuntimeError("vector backend down")
        super().upsert(entries)


class OutboxResilience(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.emb = HashedEmbeddingProvider()
        self.index = FlakyIndex(embedding=self.emb)

    def tearDown(self) -> None:
        self.store.close()

    def _mem(self, llm=None, config=None, namespace=None, index=None) -> MemorySystem:
        return MemorySystem(
            llm=llm or _StubLLM(),
            namespace=namespace or Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=index or self.index,
            embedding=self.emb,
            config=config,
        )

    def test_record_does_not_raise_when_index_down(self) -> None:
        mem = self._mem()
        rec = mem.record_user_turn("索引挂了也要存住", timestamp=1000, source_id="s1")  # 不应抛错
        self.assertEqual(rec["index_status"], "pending")
        self.assertIsNotNone(self.store.get_record_by_source_id("s1"))  # 消息安全落库
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})

    def test_vector_opt_out_keeps_raw_and_never_enters_outbox(self) -> None:
        mem = self._mem()
        rec = mem.record_user_turn(
            "这条只保留原始时间线",
            timestamp=1000,
            source_id="raw-only",
            index_in_vector=False,
        )

        self.assertEqual(rec["index_status"], "skipped")
        stored = self.store.get_record_by_source_id("raw-only")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["content"], "这条只保留原始时间线")
        self.assertEqual(self.store.list_pending_index(), [])
        self.assertEqual(self.index.count(), 0)

        self.index.fail = False
        self.assertEqual(mem.reindex_pending(), {"scanned": 0, "repaired": 0, "failed": 0})
        self.assertEqual(mem.reindex_all(), {"scanned": 0, "reindexed": 0, "failed": 0})

    def test_metadata_update_preserves_vector_opt_out(self) -> None:
        mem = self._mem()
        mem.record_user_turn("原始但不参与检索", timestamp=1000, source_id="raw-only", index_in_vector=False)
        self.index.fail = False

        out = mem.update_turn_metadata("raw-only", {"keywords": ["时间线"]})

        self.assertTrue(out["ok"])
        self.assertEqual(out["index_status"], "skipped")
        self.assertEqual(out["reason"], "index_in_vector_disabled")
        self.assertEqual(self.store.get_record_by_source_id("raw-only")["index_status"], "skipped")
        self.assertEqual(self.index.count(), 0)

    def test_reindex_pending_heals_after_backend_recovers(self) -> None:
        mem = self._mem()
        mem.record_user_turn("待补索引", timestamp=1000, source_id="s1")
        self.index.fail = False  # 后端恢复
        out = mem.reindex_pending()
        self.assertEqual(out["repaired"], 1)
        self.assertEqual(out["failed"], 0)
        self.assertEqual(self.store.list_pending_index(), [])  # 已自愈
        self.assertEqual(self.index.count(), 1)  # 向量补上了

    def test_reindex_reports_failure_if_still_down(self) -> None:
        mem = self._mem()
        mem.record_user_turn("还没好", timestamp=1000, source_id="s1")
        out = mem.reindex_pending()  # 仍故障
        self.assertEqual(out["failed"], 1)
        self.assertEqual(out["repaired"], 0)
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})  # 仍 pending,不丢

    def test_update_turn_metadata_pending_when_index_down_then_heals(self) -> None:
        mem = self._mem()
        self.index.fail = False
        mem.record_user_turn("先安全落库", timestamp=1000, source_id="s1")
        self.index.fail = True
        out = mem.update_turn_metadata("s1", {"keywords": ["补标签"]})
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "pending")
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})
        self.assertEqual(self.store.get_record_by_source_id("s1")["memory_metadata"]["keywords"], ["补标签"])
        self.index.fail = False
        healed = mem.reindex_pending()
        self.assertEqual(healed["repaired"], 1)
        self.assertEqual(self.store.list_pending_index(), [])

    def test_compaction_summary_survives_index_down_then_heals(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=100, episodic_compact_trigger_count=99)
        mem = self._mem(llm=_SummaryLLM(), config=cfg)
        for i in range(4):
            mem.record_user_turn(f"消息{i}", timestamp=1000 + i, source_id=f"m{i}")
        out = mem.compact_due_sync()  # 索引故障下压缩:摘要应已存库(pending),不抛错
        self.assertEqual(out["summaries_created"], 1)
        self.index.fail = False
        mem.reindex_pending()
        self.assertEqual(self.store.list_pending_index(), [])  # raw + summary 全部补齐

    def test_reindex_all_rebuilds_fresh_index_from_store(self) -> None:
        ns = Namespace(user_id="u1", conversation_id="c1")
        self.store.add_message(
            namespace=ns,
            role="user",
            content="用户喜欢喝可乐",
            timestamp=1000,
            source_id="raw1",
            memory_metadata={
                "keywords": ["可乐"],
                "categories": ["preference"],
                "subject_scopes": ["user"],
                "importance": 0.7,
            },
        )
        self.store.add_summary(
            namespace=ns,
            record={
                "summary_id": "sum1",
                "timestamp": 1001,
                "diary_summary": "用户表达了可乐偏好",
                "memory_metadata": {"keywords": ["可乐"], "categories": ["preference"]},
            },
        )
        self.store.add_semantic_summary(
            namespace=ns,
            record={
                "semantic_id": "sem1",
                "timestamp": 1002,
                "semantic_summary": "用户偏好可乐",
                "memory_metadata": {"keywords": ["可乐"], "subject_scopes": ["user"]},
            },
        )
        fresh_index = InMemoryVectorIndex(embedding=self.emb)
        mem = self._mem(namespace=ns, index=fresh_index)

        out = mem.reindex_all()

        self.assertEqual(out, {"scanned": 3, "reindexed": 3, "failed": 0})
        self.assertEqual(fresh_index.count(), 3)
        self.assertEqual(self.store.list_pending_index(), [])
        hits = fresh_index.keyword_search(
            query_text="可乐",
            keywords=["可乐"],
            where={"tenant_id": "", "user_id": "u1", "domain_id": ""},
            n_results=10,
        )
        self.assertEqual({hit["source_id"] for hit in hits}, {"raw1", "sum1", "sem1"})

    def test_reindex_all_respects_hard_namespace(self) -> None:
        ns = Namespace(user_id="u1", conversation_id="c1")
        other = Namespace(user_id="u2", conversation_id="c1")
        self.store.add_message(namespace=ns, role="user", content="u1 可乐", timestamp=1000, source_id="u1-raw")
        self.store.add_message(namespace=other, role="user", content="u2 可乐", timestamp=1000, source_id="u2-raw")
        fresh_index = InMemoryVectorIndex(embedding=self.emb)
        mem = self._mem(namespace=ns, index=fresh_index)

        out = mem.reindex_all()

        self.assertEqual(out["scanned"], 1)
        self.assertEqual(fresh_index.count(), 1)
        hits = fresh_index.keyword_search(
            query_text="可乐",
            keywords=["可乐"],
            where={"tenant_id": "", "user_id": "u2", "domain_id": ""},
            n_results=10,
        )
        self.assertEqual(hits, [])

    def test_reindex_all_can_limit_to_current_conversation(self) -> None:
        ns = Namespace(user_id="u1", conversation_id="c1")
        other_conversation = Namespace(user_id="u1", conversation_id="c2")
        self.store.add_message(namespace=ns, role="user", content="c1 可乐", timestamp=1000, source_id="c1-raw")
        self.store.add_message(
            namespace=other_conversation, role="user", content="c2 可乐", timestamp=1001, source_id="c2-raw"
        )
        fresh_index = InMemoryVectorIndex(embedding=self.emb)
        mem = self._mem(namespace=ns, index=fresh_index)

        out = mem.reindex_all(current_conversation_only=True)

        self.assertEqual(out, {"scanned": 1, "reindexed": 1, "failed": 0})
        self.assertEqual(fresh_index.count(), 1)
        self.assertEqual(set(fresh_index._entries), {"c1-raw"})  # noqa: SLF001 - white-box scope guard

    def test_reindex_all_reports_failure_and_marks_pending(self) -> None:
        mem = self._mem()
        self.store.add_message(namespace=mem.namespace, role="user", content="索引仍挂", timestamp=1000, source_id="s1")
        self.store.set_index_status("s1", "indexed")

        out = mem.reindex_all()

        self.assertEqual(out, {"scanned": 1, "reindexed": 0, "failed": 1})
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})


if __name__ == "__main__":
    unittest.main()
