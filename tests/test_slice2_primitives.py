"""切片 2 单测:时间锚点(带时区)、渲染、hashed embedding、内存索引混合检索 + RRF。

全部不依赖 LLM / chroma / 网络。验证设计文档里几条承重行为。
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from memcore import HashedEmbeddingProvider, InMemoryVectorIndex, fuse_with_rrf
from memcore.rendering import render_raw_snippet, render_semantic_snippet, render_summary_snippet
from memcore.time_anchor import (
    format_time_range_label,
    infer_time_of_day,
    render_relative_time_anchor_line,
)


def _ts(y, mo, d, h, mi=0, tz="Asia/Shanghai") -> int:
    from zoneinfo import ZoneInfo

    return int(datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp())


class TimeAnchor(unittest.TestCase):
    def test_timezone_changes_bucket(self) -> None:
        # 同一 UTC 时刻,上海是上午、纽约是深夜 —— 证明换算吃显式时区。
        utc_noon = int(datetime(2026, 4, 10, 4, 0, tzinfo=timezone.utc).timestamp())  # 12:00 上海
        self.assertEqual(infer_time_of_day(utc_noon, "Asia/Shanghai"), "afternoon")
        self.assertEqual(infer_time_of_day(utc_noon, "America/New_York"), "midnight")

    def test_buckets(self) -> None:
        self.assertEqual(infer_time_of_day(_ts(2026, 4, 10, 8), "Asia/Shanghai"), "morning")
        self.assertEqual(infer_time_of_day(_ts(2026, 4, 10, 14), "Asia/Shanghai"), "afternoon")
        self.assertEqual(infer_time_of_day(_ts(2026, 4, 10, 22), "Asia/Shanghai"), "night")
        self.assertEqual(infer_time_of_day(_ts(2026, 4, 10, 2), "Asia/Shanghai"), "midnight")

    def test_same_day_vs_cross_day_range(self) -> None:
        same = format_time_range_label(start_ts=_ts(2026, 4, 10, 9), end_ts=_ts(2026, 4, 10, 11), tz="Asia/Shanghai")
        self.assertEqual(same, "2026-04-10 09:00 ~ 11:00")
        cross = format_time_range_label(start_ts=_ts(2026, 4, 10, 23), end_ts=_ts(2026, 4, 11, 1), tz="Asia/Shanghai")
        self.assertEqual(cross, "2026-04-10 23:00 ~ 2026-04-11 01:00")

    def test_relative_anchor_only_when_relative_words_present(self) -> None:
        self.assertTrue(render_relative_time_anchor_line(text="昨天聊到的事", time_range_label="2026-04-10"))
        self.assertEqual(render_relative_time_anchor_line(text="复习微积分", time_range_label="2026-04-10"), "")


class Rendering(unittest.TestCase):
    def test_summary_snippet_carries_time_and_flavor_gate(self) -> None:
        rec = {
            "diary_summary": "复习了微积分",
            "period_label": "夜间学习",
            "event_type": "学习",
            "timestamp": _ts(2026, 4, 10, 22),
            "key_events": ["泰勒展开"],
            "core_facts": ["用户在复习高数"],
            "memory_metadata": {"mood_tags": ["proud"]},
        }
        with_flavor = render_summary_snippet(rec, tz="Asia/Shanghai", enable_flavor=True)
        self.assertIn("2026-04-10", with_flavor)
        self.assertIn("记忆情绪", with_flavor)
        no_flavor = render_summary_snippet(rec, tz="Asia/Shanghai", enable_flavor=False)
        self.assertNotIn("记忆情绪", no_flavor)

    def test_semantic_snippet_adds_anchor_for_relative_words(self) -> None:
        rec = {
            "semantic_summary": "最近一直在推进复习",
            "importance": 0.8,
            "period_start_ts": _ts(2026, 4, 1, 9),
            "period_end_ts": _ts(2026, 4, 10, 22),
            "stable_facts": ["持续关注学习"],
        }
        out = render_semantic_snippet(rec, tz="Asia/Shanghai")
        self.assertIn("相对时间锚点", out)  # "最近" 触发

    def test_raw_snippet_lists_turns(self) -> None:
        rows = [
            {"role": "user", "content": "在吗", "timestamp": _ts(2026, 4, 10, 9), "time_of_day": "morning"},
            {"role": "assistant", "content": "在的", "timestamp": _ts(2026, 4, 10, 9, 1), "time_of_day": "morning"},
        ]
        out = render_raw_snippet(rows, tz="Asia/Shanghai")
        self.assertIn("在吗", out)
        self.assertIn("在的", out)


class HybridRetrieval(unittest.TestCase):
    def _index(self) -> InMemoryVectorIndex:
        idx = InMemoryVectorIndex(embedding=HashedEmbeddingProvider())
        idx.upsert(
            [
                {
                    "source_id": "m1",
                    "text": "主人喜欢喝可乐",
                    "metadata": {"user_id": "u1", "memory_keywords_text": "可乐 饮料", "entry_type": "raw"},
                },
                {
                    "source_id": "m2",
                    "text": "今天讨论了项目排期",
                    "metadata": {"user_id": "u1", "entry_type": "raw"},
                },
                {
                    "source_id": "m3",
                    "text": "另一个用户的私密记录",
                    "metadata": {"user_id": "u2", "entry_type": "raw"},
                },
            ]
        )
        return idx

    def test_where_hard_isolation(self) -> None:
        idx = self._index()
        hits = idx.keyword_search(query_text="可乐", keywords=["可乐"], where={"user_id": "u1"})
        ids = {h["source_id"] for h in hits}
        self.assertIn("m1", ids)
        self.assertNotIn("m3", ids)  # u2 的记忆绝不串到 u1

    def test_keyword_via_metadata_tags(self) -> None:
        idx = self._index()
        # "饮料"只出现在 m1 的标签里,正文没有 —— 证明关键词侧吃了多维标签。
        hits = idx.keyword_search(query_text="饮料", keywords=["饮料"], where={"user_id": "u1"})
        self.assertEqual(hits[0]["source_id"], "m1")

    def test_rrf_prefers_dual_hit(self) -> None:
        semantic = [{"source_id": "a", "semantic_score": 0.5}, {"source_id": "b", "semantic_score": 0.4}]
        keyword = [{"source_id": "b", "tag_score": 2.0}, {"source_id": "c", "tag_score": 1.0}]
        fused = fuse_with_rrf(semantic, keyword)
        self.assertEqual(fused[0]["source_id"], "b")  # 双命中排第一
        self.assertTrue(fused[0]["dual_hit"])

    def test_delete_and_count(self) -> None:
        idx = self._index()
        self.assertEqual(idx.count(), 3)
        idx.delete(["m3"])
        self.assertEqual(idx.count(), 2)


if __name__ == "__main__":
    unittest.main()
