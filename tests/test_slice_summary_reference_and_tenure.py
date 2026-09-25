"""摘要参考上下文(防冲突/保持一致)与认识第 N 天(opt-in 陪伴向)的行为校验。

router 已从 memcore 移除:接入方把 retrieve/read_timeline 暴露给聊天模型,走"聊天模型自驱检索"。
"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import HashedEmbeddingProvider, MemoryConfig, MemorySystem, Namespace, SQLiteMemoryStore
from memcore.compaction import Compaction
from memcore.index.memory_index import InMemoryVectorIndex
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType
from memcore.prompts import build_summary_prompts


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


class CapturingSummaryLLM(LLMClient):
    def __init__(self) -> None:
        self.summary_user_prompts: list[str] = []

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            self.summary_user_prompts.append(request.user_prompt)
            return LLMResult(ok=True, data={"diary_summary": "d", "importance": 0.6, "core_facts": ["f"]})
        return LLMResult(ok=True, data={})


def _ts(y, mo, d, h=12) -> int:
    return int(datetime(y, mo, d, h, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


class ReferenceSummaryContext(unittest.TestCase):
    def test_build_summary_includes_reference_block(self) -> None:
        _, user = build_summary_prompts(transcript="x", batch_size=1, reference_summary_text="- [2026-04-10] 旧摘要A")
        self.assertIn("旧摘要A", user)
        self.assertIn("避免与既有摘要", user)  # 防冲突措辞在

    def test_build_summary_no_reference_when_none(self) -> None:
        _, user = build_summary_prompts(transcript="x", batch_size=1)
        self.assertNotIn("可参考的既有阶段摘要", user)

    def test_compaction_feeds_existing_summaries_as_reference(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        index = InMemoryVectorIndex(embedding=HashedEmbeddingProvider())
        ns = Namespace(user_id="u1", conversation_id="c1")
        # 预置一条既有摘要,作为后续摘要的参考。
        store.add_summary(
            namespace=ns,
            record={
                "summary_id": "old",
                "timestamp": 100,
                "date_label": "2026-04-10",
                "diary_summary": "之前聊过登山计划",
            },
        )
        llm = CapturingSummaryLLM()
        cfg = MemoryConfig(raw_token_trigger=1)
        comp = Compaction(store=store, index=index, llm=llm, config=cfg, timezone="Asia/Shanghai")
        for i in range(2):
            store.add_message(namespace=ns, role="user", content=f"m{i}", timestamp=200 + i, source_id=f"m{i}")
        comp.run_due(namespace=ns)
        self.assertTrue(llm.summary_user_prompts)
        self.assertIn("登山计划", llm.summary_user_prompts[0])  # 既有摘要作为参考喂进去了
        store.close()


class AcquaintanceNote(unittest.TestCase):
    def _mem(self):
        store = SQLiteMemoryStore(":memory:")
        emb = HashedEmbeddingProvider()
        return MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=store,
            index=InMemoryVectorIndex(embedding=emb),
            embedding=emb,
        )

    def test_empty_when_no_history(self) -> None:
        mem = self._mem()
        self.assertEqual(mem.acquaintance_note(), "")
        mem.store.close()

    def test_reports_first_date_and_days(self) -> None:
        mem = self._mem()
        mem.record_user_turn("第一天", timestamp=_ts(2026, 4, 1))
        mem.record_user_turn("后来", timestamp=_ts(2026, 4, 5))
        note = mem.acquaintance_note(now_ts=_ts(2026, 4, 10))
        self.assertIn("2026-04-01", note)  # 最早一条
        self.assertIn("第 10 天", note)  # 4-01 到 4-10 = 第10天
        mem.store.close()

    def test_cross_conversation_first_contact(self) -> None:
        mem = self._mem()
        # 另一个会话更早的记录也算"认识起点"(陪伴跨会话)。
        other = Namespace(user_id="u1", conversation_id="c0")
        mem.store.add_message(namespace=other, role="user", content="更早", timestamp=_ts(2026, 3, 20))
        mem.record_user_turn("c1 的话", timestamp=_ts(2026, 4, 1))
        note = mem.acquaintance_note(now_ts=_ts(2026, 4, 1))
        self.assertIn("2026-03-20", note)
        mem.store.close()


if __name__ == "__main__":
    unittest.main()
