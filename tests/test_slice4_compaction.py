"""切片 4 单测:写侧压缩链(raw→摘要→语义)+ 强化合并 + outbox 索引。

用 CannedLLM 替代真实模型,让压缩链在无网络下可单测。
"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import (
    Actor,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TokenCounter,
)
from memcore.compaction import Compaction
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType


def _ts(y: int, mo: int, d: int, h: int, mi: int = 0, tz: str = "Asia/Shanghai") -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp())


class CannedLLM(LLMClient):
    """按 task_type 返回固定 JSON;语义/强化共享 recurring_topics 以触发重叠合并。"""

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            data = {
                "diary_summary": "复习了微积分",
                "period_label": "夜间学习",
                "event_type": "学习",
                "importance": 0.7,
                "key_events": ["泰勒展开"],
                "core_facts": ["用户在复习高数"],
                "memory_metadata": {"keywords": ["学习"], "categories": ["plan_goal"], "importance": 0.7},
            }
        elif request.task_type == TaskType.SEMANTIC:
            data = {
                "semantic_summary": "用户长期在推进学习",
                "importance": 0.8,
                "stable_facts": ["持续关注学习"],
                "recurring_topics": ["学习", "复习"],
                "important_people": [],
                "open_loops": [],
                "memory_metadata": {"keywords": ["学习"], "importance": 0.8},
            }
        elif request.task_type == TaskType.REINFORCEMENT:
            data = {
                "semantic_summary": "用户长期在推进学习(已融合)",
                "importance": 0.85,
                "stable_facts": ["持续关注学习"],
                "recurring_topics": ["学习", "复习"],
                "important_people": [],
                "open_loops": [],
            }
        else:
            data = {}
        return LLMResult(ok=True, data=data, attempts=1)


class LengthTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(text)


class FailThenSummaryLLM(CannedLLM):
    def __init__(self) -> None:
        self.summary_calls = 0

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            self.summary_calls += 1
            if self.summary_calls == 1:
                return LLMResult(ok=False, data=None, error="temporary failure", attempts=1)
        return super().call(request)


class FailThenSemanticLLM(CannedLLM):
    def __init__(self) -> None:
        self.semantic_calls = 0

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SEMANTIC:
            self.semantic_calls += 1
            if self.semantic_calls == 1:
                return LLMResult(ok=False, data=None, error="temporary failure", attempts=1)
        return super().call(request)


class CapturingLLM(CannedLLM):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def call(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        return super().call(request)


class SummaryCycleViaFacade(unittest.TestCase):
    def test_raw_compacts_to_summary_with_differential(self) -> None:
        cfg = MemoryConfig(raw_trigger_count=4, summary_batch_size=2, episodic_compact_trigger_count=99)
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        for i in range(4):
            mem.record_user_turn(f"消息{i}", timestamp=1000 + i)
        out = mem.compact_due_sync()
        self.assertEqual(out["summaries_created"], 1)
        # 触发 4 → 总结最老 2 → 剩 2(差值关系)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertEqual(len(remaining), 2)
        # 摘要进了向量索引,且 outbox 置 indexed
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["index_status"], "indexed")

    def test_trace_raw_does_not_count_but_is_included_in_count_compaction_span(self) -> None:
        cfg = MemoryConfig(raw_trigger_count=4, summary_batch_size=2, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.record_user_turn("普通消息0", timestamp=1000, source_id="m0")
        tool = mem.record_tool_exchange(
            tool_name="web_search",
            tool_call_id="call_001",
            tool_input={"query": "北京天气"},
            result="long search output",
            timestamp=1001,
            source_id_prefix="tool1",
            keywords=["搜索结果"],
        )
        material = mem.record_material_reference(
            file_id="file_img_001",
            kind="image",
            filename="photo.jpg",
            mime_type="image/jpeg",
            file_status="ready",
            derived_status="ocr_ready",
            timestamp=1003,
            source_id="mat1",
            keywords=["题目图片"],
        )
        mem.record_user_turn("普通消息1", timestamp=1004, source_id="m1")
        mem.record_user_turn("普通消息2", timestamp=1005, source_id="m2")
        self.assertEqual(tool["tool_use"]["role"], "assistant.tool_call web_search call_001")
        self.assertEqual(tool["tool_result"]["role"], "tool.web_search call_001")
        self.assertIn("input:", tool["tool_use"]["content"])
        self.assertIn("source: web_search\noutput:\nlong search output", tool["tool_result"]["content"])
        self.assertEqual(tool["tool_result"]["memory_metadata"]["categories"], ["tool_trace"])
        self.assertEqual(material["role"], "user.attachment image file_img_001")
        self.assertEqual(material["memory_metadata"]["categories"], ["material_trace"])

        first = mem.compact_due_sync()
        self.assertEqual(first["summaries_created"], 0)

        mem.record_user_turn("普通消息3", timestamp=1006, source_id="m3")
        second = mem.compact_due_sync()

        self.assertEqual(second["summaries_created"], 1)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertEqual([m["source_id"] for m in remaining], ["m2", "m3"])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(visible[0]["source_ids"], ["m0", "tool1:tool_use", "tool1:tool_result", "mat1", "m1"])
        self.assertIn("tool_trace", visible[0]["memory_metadata"]["categories"])
        self.assertIn("material_trace", visible[0]["memory_metadata"]["categories"])
        self.assertIn("file_img_001", visible[0]["memory_metadata"]["keywords"])
        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertIn("assistant.tool_call web_search call_001\ninput:", summary_requests[0].user_prompt)
        self.assertIn("user.attachment image file_img_001\nsource: attachment", summary_requests[0].user_prompt)

    def test_raw_token_policy_waits_until_last_message_is_assistant(self) -> None:
        cfg = MemoryConfig(
            raw_compaction_policy="token",
            raw_token_trigger=10,
            raw_token_batch_ratio=0.6,
            episodic_compact_trigger_count=99,
        )
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
            token_counter=LengthTokenCounter(),
        )
        mem.record_user_turn("aaaa", timestamp=1000, source_id="u1")
        mem.record_assistant_turn("bb", timestamp=1001, source_id="a1")
        mem.record_user_turn("cccc", timestamp=1002, source_id="u2")

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 0)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 3)

    def test_raw_token_policy_compacts_oldest_assistant_boundary_and_keeps_tail(self) -> None:
        cfg = MemoryConfig(
            raw_compaction_policy="token",
            raw_token_trigger=10,
            raw_token_batch_ratio=0.6,
            raw_token_min_remainder_messages=1,
            episodic_compact_trigger_count=99,
        )
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
            token_counter=LengthTokenCounter(),
        )
        mem.record_user_turn("aaaa", timestamp=1000, source_id="u1")
        mem.record_assistant_turn("bb", timestamp=1001, source_id="a1")
        mem.record_user_turn("cccc", timestamp=1002, source_id="u2")
        mem.record_assistant_turn("dd", timestamp=1003, source_id="a2")

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 1)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertEqual([m["source_id"] for m in remaining], ["u2", "a2"])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(visible[0]["source_ids"], ["u1", "a1"])

    def test_raw_token_policy_user_cutpoint_aligns_to_next_assistant_when_tail_exists(self) -> None:
        cfg = MemoryConfig(
            raw_compaction_policy="token",
            raw_token_trigger=12,
            raw_token_batch_ratio=0.5,
            raw_token_min_remainder_messages=1,
            episodic_compact_trigger_count=99,
        )
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
            token_counter=LengthTokenCounter(),
        )
        mem.record_user_turn("aa", timestamp=1000, source_id="u1")
        mem.record_assistant_turn("bb", timestamp=1001, source_id="a1")
        mem.record_user_turn("ccc", timestamp=1002, source_id="u2")  # cumulative crosses target here
        mem.record_assistant_turn("d", timestamp=1003, source_id="a2")  # cutpoint aligns here
        mem.record_user_turn("eeee", timestamp=1004, source_id="u3")
        mem.record_assistant_turn("f", timestamp=1005, source_id="a3")

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 1)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertEqual([m["source_id"] for m in remaining], ["u3", "a3"])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(visible[0]["source_ids"], ["u1", "a1", "u2", "a2"])

    def test_raw_token_policy_compacts_first_long_turn_without_tail(self) -> None:
        cfg = MemoryConfig(
            raw_compaction_policy="token",
            raw_token_trigger=10,
            raw_token_batch_ratio=0.67,
            raw_token_min_remainder_messages=1,
            episodic_compact_trigger_count=99,
        )
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
            token_counter=LengthTokenCounter(),
        )
        mem.record_user_turn("x" * 50, timestamp=1000, source_id="u1")
        mem.record_assistant_turn("ok", timestamp=1001, source_id="a1")

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 1)
        self.assertEqual(mem.store.get_unsummarized_messages(namespace=mem.namespace), [])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(visible[0]["source_ids"], ["u1", "a1"])

    def test_raw_is_indexed_on_record(self) -> None:
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
        )
        rec = mem.record_user_turn("在吗", timestamp=1000)
        self.assertEqual(rec["index_status"], "indexed")  # 返回值与 store 同步
        self.assertEqual(mem.store.get_record_by_source_id(rec["source_id"])["index_status"], "indexed")
        self.assertEqual(mem.index.count(), 1)

    def test_raw_metadata_is_coerced_before_indexing(self) -> None:
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
            enable_flavor=False,
        )
        rec = mem.record_user_turn(
            "在吗",
            timestamp=1000,
            memory_metadata={
                "keywords": ["可乐", "饮料", "可乐", "a", "b", "c"],
                "categories": ["not_a_category"],
                "mood_tags": ["warm"],
                "importance": "oops",
            },
        )
        metadata = rec["memory_metadata"]
        self.assertEqual(metadata["categories"], [])
        self.assertEqual(metadata["mood_tags"], [])
        self.assertEqual(metadata["importance"], 0.0)
        self.assertEqual(len(metadata["keywords"]), 4)
        self.assertEqual(mem.store.get_record_by_source_id(rec["source_id"])["index_status"], "indexed")

    def test_summary_failure_keeps_raw_for_retry(self) -> None:
        cfg = MemoryConfig(raw_trigger_count=2, summary_batch_size=1, episodic_compact_trigger_count=99)
        llm = FailThenSummaryLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.record_user_turn("第一条重要事实", timestamp=1000)
        mem.record_user_turn("第二条重要事实", timestamp=1001)

        first = mem.compact_due_sync()
        self.assertEqual(first["summaries_created"], 0)
        self.assertEqual(first["summary_retry_pending"], 1)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 2)
        self.assertEqual(mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10), [])

        second = mem.compact_due_sync()
        self.assertEqual(second["summaries_created"], 1)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 1)

    def test_compaction_passes_configured_llm_retries(self) -> None:
        cfg = MemoryConfig(
            raw_trigger_count=2,
            summary_batch_size=1,
            episodic_compact_trigger_count=99,
            llm_max_retries=4,
        )
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.record_user_turn("第一条重要事实", timestamp=1000)
        mem.record_user_turn("第二条重要事实", timestamp=1001)

        mem.compact_due_sync()

        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertEqual(summary_requests[0].max_retries, 4)

    def test_summary_prompt_carries_weekday_anchor(self) -> None:
        cfg = MemoryConfig(raw_trigger_count=2, summary_batch_size=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.record_user_turn("上周二说的事情还记得吗", timestamp=_ts(2026, 4, 10, 9))
        mem.record_assistant_turn("记得,我们可以继续整理。", timestamp=_ts(2026, 4, 10, 9, 1))

        mem.compact_due_sync()

        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertIn("2026-04-10 周五", summary_requests[0].user_prompt)
        self.assertIn("上周二", summary_requests[0].user_prompt)

    def test_summary_prompt_keeps_group_actor_attribution(self) -> None:
        cfg = MemoryConfig(raw_trigger_count=2, summary_batch_size=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="group-1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.record_user_turn(
            "我下周三要复盘基金组合", actor=Actor(stable_id="qq-1", display_name="张三"), timestamp=1000
        )
        mem.record_user_turn("我周五看风险报告", actor=Actor(stable_id="qq-2", display_name="李四"), timestamp=1001)

        mem.compact_due_sync()

        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertIn("user(张三): 我下周三要复盘基金组合", summary_requests[0].user_prompt)


class SemanticAndReinforcement(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.index = InMemoryVectorIndex(embedding=HashedEmbeddingProvider())
        self.ns = Namespace(user_id="u1", conversation_id="c1")
        self.cfg = MemoryConfig(episodic_compact_trigger_count=2, episodic_compact_batch_size=1)
        self.compaction = Compaction(
            store=self.store, index=self.index, llm=CannedLLM(), config=self.cfg, timezone="Asia/Shanghai"
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_episodic_compacts_to_semantic_and_reinforces(self) -> None:
        for i in range(4):
            self.store.add_summary(
                namespace=self.ns,
                record={
                    "summary_id": f"ep{i}",
                    "timestamp": 100 + i,
                    "period_start_ts": 100 + i,
                    "period_end_ts": 100 + i,
                    "diary_summary": f"第{i}段",
                    "core_facts": [f"事实{i}"],
                },
            )
        out = self.compaction.run_due(namespace=self.ns)
        # 第一条语义新建,后续重叠(共享 recurring_topics)被融合
        self.assertEqual(out["semantic_created"], 1)
        self.assertGreaterEqual(out["reinforced"], 1)
        recent = self.store.get_recent_semantic_summaries(namespace=self.ns, limit=10)
        self.assertEqual(len(recent), 1)  # 都融进同一条
        self.assertGreater(recent[0]["reinforcement_count"], 1)
        self.assertEqual(recent[0]["index_status"], "indexed")

    def test_no_pending_index_left_after_compaction(self) -> None:
        for i in range(2):
            self.store.add_summary(namespace=self.ns, record={"summary_id": f"ep{i}", "timestamp": 100 + i})
        self.compaction.run_due(namespace=self.ns)
        # 新建的语义记忆都已 indexed(原始 ep 摘要是直接塞库的,本测试只关心语义层)
        pending_semantic = [r for r in self.store.list_pending_index() if r["entry_type"] == "semantic_summary"]
        self.assertEqual(pending_semantic, [])

    def test_semantic_failure_keeps_episodic_for_retry(self) -> None:
        compaction = Compaction(
            store=self.store,
            index=self.index,
            llm=FailThenSemanticLLM(),
            config=self.cfg,
            timezone="Asia/Shanghai",
        )
        for i in range(2):
            self.store.add_summary(
                namespace=self.ns,
                record={"summary_id": f"ep{i}", "timestamp": 100 + i, "diary_summary": f"第{i}段"},
            )

        first = compaction.run_due(namespace=self.ns)
        self.assertEqual(first["semantic_created"], 0)
        self.assertEqual(first["semantic_retry_pending"], 1)
        self.assertEqual(len(self.store.get_uncompacted_episodic_summaries(namespace=self.ns)), 2)

        second = compaction.run_due(namespace=self.ns)
        self.assertEqual(second["semantic_created"], 1)
        self.assertEqual(len(self.store.get_uncompacted_episodic_summaries(namespace=self.ns)), 1)


if __name__ == "__main__":
    unittest.main()
