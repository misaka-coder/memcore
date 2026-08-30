"""outbox 自愈:向量后端故障时记录仍安全落库(pending),reindex_pending 后补齐索引。"""

from __future__ import annotations

import unittest

from memcore import HashedEmbeddingProvider, MemoryConfig, MemorySystem, Namespace, SQLiteMemoryStore
from memcore.index.memory_index import InMemoryVectorIndex
from memcore.index.metadata_filters import kind_filter_key
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


class BatchRecordingIndex(InMemoryVectorIndex):
    def __init__(self, *, embedding):
        super().__init__(embedding=embedding)
        self.batch_sizes: list[int] = []

    def upsert(self, entries):
        self.batch_sizes.append(len(entries))
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

        out = mem.update_turn_metadata("raw-only", {"topic_terms": ["时间线"]})

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
        out = mem.update_turn_metadata("s1", {"topic_terms": ["补标签"]})
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "pending")
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})
        self.assertEqual(self.store.get_record_by_source_id("s1")["memory_metadata"]["topic_terms"], ["补标签"])
        self.index.fail = False
        healed = mem.reindex_pending()
        self.assertEqual(healed["repaired"], 1)
        self.assertEqual(self.store.list_pending_index(), [])

    def test_compaction_summary_survives_index_down_then_heals(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
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
                "entity_anchors": ["可乐"],
                "memory_facets": ["preference"],
                "about_roles": ["user"],
                "retrieval_priority": "high",
            },
        )
        self.store.add_summary(
            namespace=ns,
            record={
                "summary_id": "sum1",
                "timestamp": 1001,
                "diary_summary": "用户表达了可乐偏好",
                "memory_metadata": {"entity_anchors": ["可乐"], "memory_facets": ["preference"]},
            },
        )
        self.store.add_semantic_summary(
            namespace=ns,
            record={
                "semantic_id": "sem1",
                "timestamp": 1002,
                "semantic_summary": "用户偏好可乐",
                "memory_metadata": {"entity_anchors": ["可乐"], "about_roles": ["user"]},
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
            entity_anchors=["可乐"],
            topic_terms=[],
            where={"tenant_id": "", "user_id": "u1", "domain_id": ""},
            n_results=10,
        )
        self.assertEqual({hit["source_id"] for hit in hits}, {"raw1", "sum1", "sem1"})

    def test_reindex_all_preserves_batch_calls_for_remote_providers(self) -> None:
        ns = Namespace(user_id="u1", conversation_id="c1")
        for index in range(5):
            self.store.add_message(
                namespace=ns,
                role="user",
                content=f"记忆 {index}",
                timestamp=1000 + index,
                source_id=f"raw-{index}",
            )
        recording_index = BatchRecordingIndex(embedding=self.emb)
        mem = self._mem(namespace=ns, index=recording_index)

        out = mem.reindex_all(batch_size=2)

        self.assertEqual(out, {"scanned": 5, "reindexed": 5, "failed": 0})
        self.assertEqual(recording_index.batch_sizes, [2, 2, 1])

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
            entity_anchors=["可乐"],
            topic_terms=[],
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

    def test_reindex_all_isolates_one_malformed_legacy_kind(self) -> None:
        fresh_index = InMemoryVectorIndex(embedding=self.emb)
        mem = self._mem(index=fresh_index)
        self.store.add_message(
            namespace=mem.namespace,
            role="user",
            content="合法记录仍应完成热加载",
            timestamp=1000,
            source_id="good",
        )
        self.store.add_message(
            namespace=mem.namespace,
            role="user",
            content="模拟升级前遗留坏 kind",
            timestamp=1001,
            source_id="legacy-bad",
        )
        with self.store._lock, self.store._conn:  # noqa: SLF001 - migration-corruption fixture
            self.store._conn.execute(  # noqa: SLF001 - migration-corruption fixture
                "UPDATE messages SET kind = ?, index_status = 'indexed' WHERE source_id = ?",
                ("bad kind", "legacy-bad"),
            )

        out = mem.reindex_all(batch_size=64)

        self.assertEqual(out, {"scanned": 2, "reindexed": 1, "failed": 1})
        self.assertEqual(fresh_index.count(), 1)
        self.assertEqual({row["source_id"] for row in self.store.list_pending_index()}, {"legacy-bad"})

    def test_hyphenated_mcp_kind_survives_restart_reindex_and_retrieval(self) -> None:
        ns = Namespace(user_id="u1", conversation_id="c1")
        self.store.add_message(
            namespace=ns,
            role="tool",
            content="GitHub issue 123 is open",
            timestamp=1000,
            source_id="mcp-result",
            kind="tool.github-mcp.get-issue.result",
            retrieval_visibility="default",
        )
        fresh_index = InMemoryVectorIndex(embedding=self.emb)
        mem = self._mem(namespace=ns, index=fresh_index)

        out = mem.reindex_all(current_conversation_only=True)
        hits = fresh_index.keyword_search(
            query_text="GitHub issue 123",
            entity_anchors=[],
            topic_terms=[],
            where={kind_filter_key("tool.github-mcp"): True},
            n_results=10,
        )

        self.assertEqual(out, {"scanned": 1, "reindexed": 1, "failed": 0})
        self.assertEqual([hit["source_id"] for hit in hits], ["mcp-result"])


if __name__ == "__main__":
    unittest.main()
