"""Repair pass 回归:堵住 review 找到的 6 个接缝。

每个测试对应一个已修问题,防止回潮。
"""

from __future__ import annotations

import unittest

from memcore import (
    HashedEmbeddingProvider,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemorySystem,
    Namespace,
    NamespaceError,
    SQLiteMemoryStore,
)
from memcore.rendering import record_time_range, render_summary_snippet


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


class SourceIdNamespaceLeak(unittest.TestCase):
    """#1:同一 source_id 跨 namespace 必须拒绝,不能返回别人的记录。"""

    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_cross_namespace_source_id_rejected(self) -> None:
        u1 = Namespace(user_id="u1")
        u2 = Namespace(user_id="u2")
        self.store.add_message(namespace=u1, role="user", content="u1 私密", timestamp=100, source_id="shared")
        with self.assertRaises(NamespaceError):
            self.store.add_message(namespace=u2, role="user", content="u2", timestamp=100, source_id="shared")

    def test_same_namespace_source_id_still_idempotent(self) -> None:
        u1 = Namespace(user_id="u1")
        a = self.store.add_message(namespace=u1, role="user", content="hi", timestamp=100, source_id="s")
        b = self.store.add_message(namespace=u1, role="user", content="hi2", timestamp=101, source_id="s")
        self.assertEqual(a["seq_no"], b["seq_no"])

    def test_same_user_different_conversation_source_id_rejected(self) -> None:
        c1 = Namespace(user_id="u1", conversation_id="c1")
        c2 = Namespace(user_id="u1", conversation_id="c2")
        self.store.add_message(namespace=c1, role="user", content="c1 私密", timestamp=100, source_id="shared")
        with self.assertRaises(NamespaceError):
            self.store.add_message(namespace=c2, role="user", content="c2", timestamp=101, source_id="shared")

    def test_summary_cross_namespace_overwrite_rejected(self) -> None:
        u1 = Namespace(user_id="u1")
        u2 = Namespace(user_id="u2")
        self.store.add_summary(namespace=u1, record={"summary_id": "shared", "timestamp": 100, "diary_summary": "u1"})
        with self.assertRaises(NamespaceError):
            self.store.add_summary(namespace=u2, record={"summary_id": "shared", "timestamp": 100})

    def test_summary_cross_conversation_overwrite_rejected(self) -> None:
        c1 = Namespace(user_id="u1", conversation_id="c1")
        c2 = Namespace(user_id="u1", conversation_id="c2")
        self.store.add_summary(namespace=c1, record={"summary_id": "shared", "timestamp": 100, "diary_summary": "c1"})
        with self.assertRaises(NamespaceError):
            self.store.add_summary(
                namespace=c2, record={"summary_id": "shared", "timestamp": 101, "diary_summary": "c2"}
            )


class VisibleScopedToConversation(unittest.TestCase):
    """#2:可见 episodic/semantic 不跨 conversation。"""

    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_episodic_visible_does_not_cross_conversation(self) -> None:
        c1 = Namespace(user_id="u1", conversation_id="c1")
        c2 = Namespace(user_id="u1", conversation_id="c2")
        self.store.add_summary(namespace=c1, record={"summary_id": "s1", "timestamp": 100, "diary_summary": "c1事"})
        self.store.add_summary(namespace=c2, record={"summary_id": "s2", "timestamp": 100, "diary_summary": "c2事"})
        visible = self.store.get_visible_episodic_summaries(namespace=c1, limit=10)
        self.assertEqual({s["summary_id"] for s in visible}, {"s1"})

    def test_semantic_visible_does_not_cross_conversation(self) -> None:
        c1 = Namespace(user_id="u1", conversation_id="c1")
        c2 = Namespace(user_id="u1", conversation_id="c2")
        self.store.add_semantic_summary(namespace=c1, record={"semantic_id": "x", "timestamp": 100})
        self.store.add_semantic_summary(namespace=c2, record={"semantic_id": "y", "timestamp": 100})
        visible = self.store.get_recent_semantic_summaries(namespace=c1, limit=10)
        self.assertEqual({s["semantic_id"] for s in visible}, {"x"})


class DefaultTimestampRendering(unittest.TestCase):
    """#3:period_start/end 默认 0 不该渲染成 1970。"""

    def test_zero_ts_is_treated_as_missing(self) -> None:
        self.assertEqual(record_time_range({"period_start_ts": 0, "period_end_ts": 0, "timestamp": 0}), (None, None))

    def test_no_1970_in_snippet(self) -> None:
        rec = {"diary_summary": "无时间的摘要", "period_start_ts": 0, "period_end_ts": 0, "timestamp": 0}
        out = render_summary_snippet(rec, tz="Asia/Shanghai")
        self.assertNotIn("1970", out)


class DeleteReturnsIds(unittest.TestCase):
    """#4:delete_namespace 返回被删 source_id 列表(供清向量索引)。"""

    def test_delete_returns_source_ids(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        ns = Namespace(user_id="u1")
        m = store.add_message(namespace=ns, role="user", content="x", timestamp=100, source_id="m1")
        store.add_summary(namespace=ns, record={"summary_id": "s1", "timestamp": 100})
        ids = store.delete_namespace(namespace=ns)
        self.assertIn(m["source_id"], ids)
        self.assertIn("s1", ids)
        store.close()


class TimezoneValidation(unittest.TestCase):
    """#6:非法时区构造时就报错。"""

    def test_invalid_timezone_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MemorySystem(llm=_StubLLM(), namespace=Namespace(user_id="u1"), timezone="Not/AZone")

    def test_valid_timezone_ok(self) -> None:
        mem = MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1"),
            timezone="America/New_York",
            embedding=HashedEmbeddingProvider(),
        )
        self.assertEqual(mem.timezone, "America/New_York")


if __name__ == "__main__":
    unittest.main()
