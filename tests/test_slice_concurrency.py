"""并发/性能加固:RRF 保序去重、内存索引线程安全、压缩异步化(对齐 Akane)。"""

from __future__ import annotations

import threading
import unittest

from memcore import HashedEmbeddingProvider, InMemoryVectorIndex, MemoryConfig, MemorySystem, Namespace, fuse_with_rrf
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType


class CannedSummaryLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            return LLMResult(ok=True, data={"diary_summary": "d", "importance": 0.6, "core_facts": ["f"]})
        return LLMResult(ok=True, data={})


class RRFDedup(unittest.TestCase):
    def test_dedup_preserves_order_no_duplicates(self) -> None:
        semantic = [{"source_id": "a", "semantic_score": 0.9}, {"source_id": "b", "semantic_score": 0.8}]
        keyword = [{"source_id": "b", "tag_score": 2.0}, {"source_id": "c", "tag_score": 1.0}]
        fused = fuse_with_rrf(semantic, keyword)
        ids = [h["source_id"] for h in fused]
        self.assertEqual(len(ids), 3)  # b 只出现一次
        self.assertEqual(set(ids), {"a", "b", "c"})


class IndexThreadSafety(unittest.TestCase):
    def test_concurrent_upserts_keep_all_entries(self) -> None:
        idx = InMemoryVectorIndex(embedding=HashedEmbeddingProvider())

        def writer(start: int) -> None:
            for i in range(start, start + 50):
                idx.upsert([{"source_id": f"s{i}", "text": f"内容{i}", "metadata": {"user_id": "u1"}}])

        threads = [threading.Thread(target=writer, args=(base,)) for base in (0, 50, 100, 150)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(idx.count(), 200)  # 并发写入无丢失


class AsyncCompaction(unittest.TestCase):
    def _mem(self) -> MemorySystem:
        return MemorySystem(
            llm=CannedSummaryLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
            config=MemoryConfig(raw_trigger_count=4, summary_batch_size=2, episodic_compact_trigger_count=99),
        )

    def test_background_compaction_runs_and_returns_future(self) -> None:
        mem = self._mem()
        for i in range(4):
            mem.record_user_turn(f"消息{i}", timestamp=1000 + i)
        future = mem.compact_due_background()
        out = future.result(timeout=5)  # 后台跑完
        self.assertEqual(out["summaries_created"], 1)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 2)
        mem.close()

    def test_close_is_idempotent(self) -> None:
        mem = self._mem()
        mem.compact_due_background().result(timeout=5)
        mem.close()
        mem.close()  # 再次 close 不报错


if __name__ == "__main__":
    unittest.main()
