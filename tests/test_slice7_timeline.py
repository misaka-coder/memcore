"""切片 7 单测:时间线工具 —— 按时间精确读原始对话(不走向量),与向量检索互补。"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import HashedEmbeddingProvider, MemorySystem, Namespace, SQLiteMemoryStore
from memcore.index.memory_index import InMemoryVectorIndex
from memcore.llm.base import LLMClient, LLMRequest, LLMResult


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


def _ts(y, mo, d, h, mi=0, tz="Asia/Shanghai") -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp())


class Timeline(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.emb = HashedEmbeddingProvider()
        self.index = InMemoryVectorIndex(embedding=self.emb)

    def tearDown(self) -> None:
        self.store.close()

    def _mem(self, conversation: str, user: str = "u1") -> MemorySystem:
        return MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id=user, conversation_id=conversation),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.emb,
        )

    def test_reads_messages_in_date_range_ordered(self) -> None:
        mem = self._mem("c1")
        mem.record_user_turn("4月10日上午的话", timestamp=_ts(2026, 4, 10, 9))
        mem.record_user_turn("4月10日晚上的话", timestamp=_ts(2026, 4, 10, 22))
        mem.record_user_turn("4月20日的话", timestamp=_ts(2026, 4, 20, 12))
        out = mem.read_timeline(date_from="2026-04-10", date_to="2026-04-10")
        contents = [m["content"] for m in out["messages"]]
        self.assertEqual(contents, ["4月10日上午的话", "4月10日晚上的话"])  # 按时间排序,排除 4-20
        self.assertEqual(out["status"], "ok")
        self.assertIn("日期 2026-04-10", out["text"])
        self.assertIn("日期 2026-04-10 周五", out["text"])

    def test_single_day_when_date_to_omitted(self) -> None:
        mem = self._mem("c1")
        mem.record_user_turn("当天", timestamp=_ts(2026, 4, 10, 9))
        mem.record_user_turn("另一天", timestamp=_ts(2026, 4, 11, 9))
        out = mem.read_timeline(date_from="2026-04-10")
        self.assertEqual([m["content"] for m in out["messages"]], ["当天"])

    def test_time_period_filter_with_alias(self) -> None:
        mem = self._mem("c1")
        mem.record_user_turn("上午说的", timestamp=_ts(2026, 4, 10, 9))
        mem.record_user_turn("晚上说的", timestamp=_ts(2026, 4, 10, 22))
        out = mem.read_timeline(date_from="2026-04-10", time_periods=["晚上"])  # 中文别名
        self.assertEqual([m["content"] for m in out["messages"]], ["晚上说的"])

    def test_exact_local_time_range_reads_only_requested_minutes(self) -> None:
        mem = self._mem("c1")
        mem.record_user_turn("上午较早的大段记录" * 3000, timestamp=_ts(2026, 4, 10, 8, 19))
        mem.record_user_turn("十点五十九", timestamp=_ts(2026, 4, 10, 10, 59))
        mem.record_user_turn("我们俩一起来玩的", timestamp=_ts(2026, 4, 10, 11, 47))
        mem.record_user_turn("十二点边界", timestamp=_ts(2026, 4, 10, 12, 0))

        out = mem.read_timeline(
            time_range={
                "start_at": "2026-04-10 11:00",
                "end_at": "2026-04-10 12:00",
            }
        )

        self.assertEqual([m["content"] for m in out["messages"]], ["我们俩一起来玩的"])
        self.assertEqual(out["selector_mode"], "time_range")
        self.assertEqual(out["time_range"]["start_at"], "2026-04-10T11:00:00+08:00")
        self.assertEqual(out["time_range"]["end_at"], "2026-04-10T12:00:00+08:00")

    def test_exact_time_range_accepts_explicit_offset_and_crosses_date(self) -> None:
        mem = self._mem("c1")
        mem.record_user_turn("范围之前", timestamp=_ts(2026, 4, 10, 22, 59))
        mem.record_user_turn("范围内一", timestamp=_ts(2026, 4, 10, 23, 30))
        mem.record_user_turn("范围内二", timestamp=_ts(2026, 4, 11, 0, 30))
        mem.record_user_turn("范围之后", timestamp=_ts(2026, 4, 11, 1, 0))

        out = mem.read_timeline(
            time_range={
                "start_at": "2026-04-10T15:00:00Z",
                "end_at": "2026-04-10T17:00:00Z",
            }
        )

        self.assertEqual([m["content"] for m in out["messages"]], ["范围内一", "范围内二"])
        self.assertEqual(out["time_range"]["start_at"], "2026-04-10T23:00:00+08:00")
        self.assertEqual(out["time_range"]["end_at"], "2026-04-11T01:00:00+08:00")

    def test_empty_range(self) -> None:
        mem = self._mem("c1")
        mem.record_user_turn("有话", timestamp=_ts(2026, 4, 10, 9))
        out = mem.read_timeline(date_from="2026-05-01", date_to="2026-05-31")
        self.assertEqual(out["status"], "empty")
        self.assertEqual(out["messages"], [])
        self.assertEqual(out["text"], "")

    def test_default_scope_is_current_conversation(self) -> None:
        self._mem("c2").record_user_turn("c2 的话", timestamp=_ts(2026, 4, 10, 9))
        mem_c1 = self._mem("c1")
        mem_c1.record_user_turn("c1 的话", timestamp=_ts(2026, 4, 10, 10))
        out = mem_c1.read_timeline(date_from="2026-04-10")
        self.assertEqual([m["content"] for m in out["messages"]], ["c1 的话"])
        # 跨会话开关:能读到该用户全部会话
        out_all = mem_c1.read_timeline(date_from="2026-04-10", cross_conversation=True)
        self.assertEqual({m["content"] for m in out_all["messages"]}, {"c1 的话", "c2 的话"})

    def test_hard_isolation_other_user_excluded(self) -> None:
        self._mem("c1", user="u2").record_user_turn("u2 私密", timestamp=_ts(2026, 4, 10, 9))
        mem_u1 = self._mem("c1", user="u1")
        mem_u1.record_user_turn("u1 的话", timestamp=_ts(2026, 4, 10, 10))
        out = mem_u1.read_timeline(date_from="2026-04-10", cross_conversation=True)
        contents = [m["content"] for m in out["messages"]]
        self.assertIn("u1 的话", contents)
        self.assertNotIn("u2 私密", contents)

    def test_backfill_out_of_order_sorts_by_timestamp(self) -> None:
        # 先写晚上(22:00)再写上午(09:00):必须按真实时间排序,不是写入顺序。
        mem = self._mem("c1")
        mem.record_user_turn("晚上写的", timestamp=_ts(2026, 4, 10, 22), source_id="late")
        mem.record_user_turn("上午写的", timestamp=_ts(2026, 4, 10, 9), source_id="early")
        out = mem.read_timeline(date_from="2026-04-10")
        self.assertEqual([m["content"] for m in out["messages"]], ["上午写的", "晚上写的"])

    def test_cross_conversation_same_day_sorted_by_time(self) -> None:
        self._mem("c2").record_user_turn("c2 早上", timestamp=_ts(2026, 4, 10, 8))
        mem_c1 = self._mem("c1")
        mem_c1.record_user_turn("c1 中午", timestamp=_ts(2026, 4, 10, 13))
        out = mem_c1.read_timeline(date_from="2026-04-10", cross_conversation=True)
        self.assertEqual([m["content"] for m in out["messages"]], ["c2 早上", "c1 中午"])


class TimelineStrictFilters(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.emb = HashedEmbeddingProvider()
        self.index = InMemoryVectorIndex(embedding=self.emb)
        self.mem = MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.emb,
        )
        self.mem.record_user_turn("有话", timestamp=_ts(2026, 4, 10, 9))

    def tearDown(self) -> None:
        self.store.close()

    def test_invalid_date_rejected_not_relaxed(self) -> None:
        out = self.mem.read_timeline(date_from="2026/04/10")
        self.assertEqual(out["status"], "invalid_filter")
        self.assertEqual(out["messages"], [])

    def test_empty_date_from_rejected(self) -> None:
        # 不能因为 date_from 空就查出全部消息。
        out = self.mem.read_timeline(date_from="")
        self.assertEqual(out["status"], "invalid_filter")

    def test_date_from_after_date_to_rejected(self) -> None:
        out = self.mem.read_timeline(date_from="2026-04-20", date_to="2026-04-10")
        self.assertEqual(out["status"], "invalid_filter")

    def test_unknown_time_period_rejected_not_relaxed(self) -> None:
        # "中午"未知:不能静默归一成 [] 然后查整天。
        out = self.mem.read_timeline(date_from="2026-04-10", time_periods=["中午"])
        self.assertEqual(out["status"], "invalid_filter")
        self.assertIn("中午", out["reason"])

    def test_exact_time_range_rejects_missing_or_reversed_boundaries(self) -> None:
        missing = self.mem.read_timeline(time_range={"start_at": "2026-04-10 11:00"})
        reversed_range = self.mem.read_timeline(
            time_range={"start_at": "2026-04-10 12:00", "end_at": "2026-04-10 11:00"}
        )

        self.assertEqual(missing["status"], "invalid_filter")
        self.assertEqual(missing["reason"], "time_range_end_at_required")
        self.assertEqual(reversed_range["status"], "invalid_filter")
        self.assertEqual(reversed_range["reason"], "time_range_start_must_be_before_end")

    def test_exact_time_range_is_mutually_exclusive_with_legacy_and_anchor_modes(self) -> None:
        exact = {"start_at": "2026-04-10 08:00", "end_at": "2026-04-10 10:00"}

        mixed_legacy = self.mem.read_timeline(time_range=exact, date_from="2026-04-10")
        mixed_anchor = self.mem.read_timeline(time_range=exact, anchor_source_id="missing")

        self.assertEqual(mixed_legacy["reason"], "timeline_modes_are_mutually_exclusive")
        self.assertEqual(mixed_anchor["reason"], "timeline_modes_are_mutually_exclusive")


class PromptOverridesTypeCheck(unittest.TestCase):
    def test_wrong_type_rejected_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            MemorySystem(
                llm=_StubLLM(),
                namespace=Namespace(user_id="u1"),
                timezone="UTC",
                embedding=HashedEmbeddingProvider(),
                prompt_overrides={"persona_text": "应该用 PromptOverrides 而不是 dict"},  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
