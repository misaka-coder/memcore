"""切片 3 单测:SQLiteMemoryStore 的写入一致性、隔离、可见窗口、outbox、遗忘。

用 :memory: 单连接,不落盘、不依赖外部。验证设计文档 §B/§10 的承重行为。
"""

from __future__ import annotations

import unittest

from memcore import Namespace, NamespaceError, SQLiteMemoryStore
from memcore.namespace import Actor


class StoreSliceBase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.ns = Namespace(user_id="u1", tenant_id="t1", domain_id="fin", conversation_id="c1")

    def tearDown(self) -> None:
        self.store.close()


class WriteAndIdempotency(StoreSliceBase):
    def test_seq_increments_per_conversation(self) -> None:
        a = self.store.add_message(namespace=self.ns, role="user", content="一", timestamp=100)
        b = self.store.add_message(namespace=self.ns, role="assistant", content="二", timestamp=101)
        self.assertEqual(a["seq_no"], 1)
        self.assertEqual(b["seq_no"], 2)

    def test_idempotent_source_id(self) -> None:
        r1 = self.store.add_message(namespace=self.ns, role="user", content="hi", timestamp=100, source_id="fixed")
        r2 = self.store.add_message(
            namespace=self.ns, role="user", content="hi-again", timestamp=101, source_id="fixed"
        )
        self.assertEqual(r1["seq_no"], r2["seq_no"])  # 重复写入 = 同一条,不产生重复记忆
        self.assertEqual(self.store.get_record_by_source_id("fixed")["content"], "hi")

    def test_new_message_is_pending_then_indexed(self) -> None:
        rec = self.store.add_message(namespace=self.ns, role="user", content="x", timestamp=100, source_id="s1")
        self.assertEqual(rec["index_status"], "pending")
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})
        self.store.set_index_status("s1", "indexed")
        self.assertEqual(self.store.list_pending_index(), [])

    def test_memory_metadata_roundtrip(self) -> None:
        rec = self.store.add_message(
            namespace=self.ns,
            role="user",
            content="x",
            timestamp=100,
            memory_metadata={"entity_anchors": ["可乐"], "retrieval_priority": "high"},
        )
        got = self.store.get_record_by_source_id(rec["source_id"])
        self.assertEqual(got["memory_metadata"]["entity_anchors"], ["可乐"])

    def test_update_message_memory_metadata_sets_pending(self) -> None:
        self.store.add_message(namespace=self.ns, role="user", content="x", timestamp=100, source_id="s1")
        self.store.set_index_status("s1", "indexed")
        updated = self.store.update_message_memory_metadata(
            namespace=self.ns,
            source_id="s1",
            memory_metadata={"entity_anchors": ["英伟达"], "retrieval_priority": "high"},
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["memory_metadata"]["entity_anchors"], ["英伟达"])
        self.assertEqual(updated["index_status"], "pending")  # metadata 变了,raw index 必须重建
        self.assertEqual({r["source_id"] for r in self.store.list_pending_index()}, {"s1"})


class Isolation(StoreSliceBase):
    def test_hard_isolation_between_users(self) -> None:
        other = Namespace(user_id="u2", tenant_id="t1", domain_id="fin", conversation_id="c1")
        self.store.add_message(namespace=self.ns, role="user", content="u1 私密", timestamp=100)
        self.store.add_message(namespace=other, role="user", content="u2 私密", timestamp=100)
        u1_msgs = self.store.get_unsummarized_messages(namespace=self.ns)
        contents = {m["content"] for m in u1_msgs}
        self.assertIn("u1 私密", contents)
        self.assertNotIn("u2 私密", contents)  # u2 绝不串到 u1

    def test_actor_stable_id_and_display_name_stored(self) -> None:
        ns = Namespace(user_id="u1", actor=Actor(stable_id="qq-123", display_name="张三"))
        rec = self.store.add_message(namespace=ns, role="user", content="hi", timestamp=100)
        got = self.store.get_record_by_source_id(rec["source_id"])
        self.assertEqual(got["actor_id"], "qq-123")
        self.assertEqual(got["actor_display_name"], "张三")

    def test_update_message_memory_metadata_rejects_cross_user(self) -> None:
        other = Namespace(user_id="u2", tenant_id="t1", domain_id="fin", conversation_id="c1")
        self.store.add_message(namespace=self.ns, role="user", content="u1 私密", timestamp=100, source_id="s1")
        with self.assertRaises(NamespaceError):
            self.store.update_message_memory_metadata(
                namespace=other,
                source_id="s1",
                memory_metadata={"entity_anchors": ["污染"]},
            )

    def test_update_message_memory_metadata_rejects_cross_conversation(self) -> None:
        other = Namespace(user_id="u1", tenant_id="t1", domain_id="fin", conversation_id="c2")
        self.store.add_message(namespace=self.ns, role="user", content="c1 私密", timestamp=100, source_id="s1")
        with self.assertRaises(NamespaceError):
            self.store.update_message_memory_metadata(
                namespace=other,
                source_id="s1",
                memory_metadata={"entity_anchors": ["污染"]},
            )


class WindowsAndCompaction(StoreSliceBase):
    def test_context_slice_window(self) -> None:
        for i in range(1, 6):
            self.store.add_message(
                namespace=self.ns, role="user", content=f"m{i}", timestamp=100 + i, source_id=f"s{i}"
            )
        rows = self.store.get_context_slice(namespace=self.ns, seq_no=3, window=1)
        self.assertEqual([r["content"] for r in rows], ["m2", "m3", "m4"])

    def test_episodic_visibility_excludes_semanticized(self) -> None:
        s1 = self.store.add_summary(
            namespace=self.ns, record={"summary_id": "sum1", "timestamp": 100, "diary_summary": "a"}
        )
        self.store.add_summary(namespace=self.ns, record={"summary_id": "sum2", "timestamp": 200, "diary_summary": "b"})
        self.store.mark_summaries_semanticized(["sum1"], "sem1")
        visible = self.store.get_visible_episodic_summaries(namespace=self.ns, limit=10)
        ids = {s["summary_id"] for s in visible}
        self.assertEqual(ids, {"sum2"})  # 已语义化的不再可见
        self.assertEqual(visible[0]["diary_summary"], "b")
        self.assertEqual(s1["entry_type"], "summary")

    def test_semantic_recency_ordering(self) -> None:
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={"semantic_id": "x", "timestamp": 100, "last_reinforced_ts": 100, "importance": 0.5},
        )
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={"semantic_id": "y", "timestamp": 200, "last_reinforced_ts": 300, "importance": 0.4},
        )
        recent = self.store.get_recent_semantic_summaries(namespace=self.ns, limit=10)
        self.assertEqual(recent[0]["semantic_id"], "y")  # last_reinforced_ts 更新者优先


class Forgetting(StoreSliceBase):
    def test_delete_namespace_removes_all_layers(self) -> None:
        self.store.add_message(namespace=self.ns, role="user", content="x", timestamp=100)
        self.store.add_summary(namespace=self.ns, record={"summary_id": "s", "timestamp": 100})
        self.store.add_semantic_summary(namespace=self.ns, record={"semantic_id": "z", "timestamp": 100})
        deleted = self.store.delete_namespace(namespace=self.ns)
        self.assertEqual(len(deleted), 3)  # 返回被删的 source_id 列表
        self.assertEqual(self.store.get_unsummarized_messages(namespace=self.ns), [])

    def test_delete_is_scoped(self) -> None:
        other = Namespace(user_id="u2")
        self.store.add_message(namespace=self.ns, role="user", content="keep-u1", timestamp=100)
        self.store.add_message(namespace=other, role="user", content="del-u2", timestamp=100)
        self.store.delete_namespace(namespace=other)
        self.assertEqual(len(self.store.get_unsummarized_messages(namespace=self.ns)), 1)


if __name__ == "__main__":
    unittest.main()
