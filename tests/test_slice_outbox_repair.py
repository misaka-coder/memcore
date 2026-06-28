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

    def _mem(self, llm=None, config=None) -> MemorySystem:
        return MemorySystem(
            llm=llm or _StubLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.emb,
            config=config,
        )

    def test_record_does_not_raise_when_index_down(self) -> None:
        mem = self._mem()
        rec = mem.record_user_turn("索引挂了也要存住", timestamp=1000, source_id="s1")  # 不应抛错
        self.assertEqual(rec["index_status"], "pending")
        self.assertIsNotNone(self.store.get_record_by_source_id("s1"))  # 消息安全落库
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})

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
        cfg = MemoryConfig(raw_trigger_count=4, summary_batch_size=2, episodic_compact_trigger_count=99)
        mem = self._mem(llm=_SummaryLLM(), config=cfg)
        for i in range(4):
            mem.record_user_turn(f"消息{i}", timestamp=1000 + i, source_id=f"m{i}")
        out = mem.compact_due_sync()  # 索引故障下压缩:摘要应已存库(pending),不抛错
        self.assertEqual(out["summaries_created"], 1)
        self.index.fail = False
        mem.reindex_pending()
        self.assertEqual(self.store.list_pending_index(), [])  # raw + summary 全部补齐


if __name__ == "__main__":
    unittest.main()
