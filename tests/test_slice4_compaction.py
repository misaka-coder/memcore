"""切片 4 单测:写侧压缩链(raw→摘要→语义)+ 强化合并 + outbox 索引。

用 CannedLLM 替代真实模型,让压缩链在无网络下可单测。
"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import (
    AnnotationStatus,
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TokenCounter,
    TurnRole,
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


def _complete_turn(
    mem: MemorySystem,
    number: int,
    *,
    user_text: str = "问题",
    assistant_text: str = "回答",
    timestamp: int = 1000,
    actor: Actor | None = None,
) -> str:
    turn_id = f"turn-{number}"
    mem.begin_turn(
        turn_id=turn_id,
        opened_at=timestamp,
        stimuli=[
            TimelineEntryInput(
                source_id=f"u{number}",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text=user_text,
                payload={"text": user_text},
                timestamp=timestamp,
                actor=actor,
                compatibility_role="user",
            )
        ],
    )
    result = mem.complete_turn(
        turn_id=turn_id,
        semantic_text=assistant_text,
        provider_output_raw=assistant_text,
        memory_annotation={"keywords": [user_text[:20]], "categories": ["plan_goal"]},
        annotation_status=AnnotationStatus.ACCEPTED_HOST,
        timestamp=timestamp + 1,
        source_id=f"a{number}",
    )
    if not result.completed:
        raise AssertionError(result)
    return turn_id


class SummaryCycleViaFacade(unittest.TestCase):
    def test_raw_compacts_by_ratio_without_splitting_turns(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        mem = MemorySystem(
            llm=CannedLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        for i in range(4):
            _complete_turn(mem, i, user_text=f"消息{i}", timestamp=1000 + i * 10)
        out = mem.compact_due_sync()
        self.assertEqual(out["summaries_created"], 1)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertGreater(len(remaining), 0)
        self.assertEqual(len(remaining) % 2, 0)
        remaining_counts: dict[str, int] = {}
        for item in remaining:
            remaining_counts[item["turn_id"]] = remaining_counts.get(item["turn_id"], 0) + 1
        self.assertTrue(all(count == 2 for count in remaining_counts.values()))
        # 摘要进了向量索引,且 outbox 置 indexed
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["index_status"], "indexed")

    def test_standalone_backlog_compacts_one_batch_per_run_and_preserves_lineage(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=100, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        for i in range(10):
            mem.record_user_turn(f"消息{i}", timestamp=1000 + i, source_id=f"m{i}")

        first = mem.compact_due_sync()

        self.assertEqual(first["summaries_created"], 1)
        self.assertLess(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 10)
        self.assertEqual(
            len([request for request in llm.requests if request.task_type == TaskType.SUMMARY]),
            1,
        )

        results = [first]
        for _ in range(5):
            current = mem.compact_due_sync()
            results.append(current)
            if current["summaries_created"] == 0:
                break

        self.assertEqual(results[-1]["summaries_created"], 0)
        self.assertTrue(all(item["summaries_created"] <= 1 for item in results))
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertEqual([message["source_id"] for message in remaining], ["m9"])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        summarized_source_ids = {source_id for summary in visible for source_id in summary.get("source_ids", [])}
        self.assertEqual(summarized_source_ids, {f"m{i}" for i in range(9)})

    def test_tool_exchange_joins_one_turn_and_compacts_as_operation_partition(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.begin_turn(
            turn_id="tool-turn",
            stimuli=[
                TimelineEntryInput(
                    source_id="tool-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="查两个方向",
                    payload={"text": "查两个方向"},
                    timestamp=999,
                )
            ],
        )
        first_tool = mem.record_tool_exchange(
            turn_id="tool-turn",
            tool_name="web_search",
            tool_call_id="call_001",
            tool_input={"query": "第一条搜索"},
            result="first search output",
            timestamp=1000,
            source_id_prefix="tool1",
            keywords=["第一条搜索"],
        )
        second_tool = mem.record_tool_exchange(
            turn_id="tool-turn",
            tool_name="quote_snapshot",
            tool_call_id="call_002",
            tool_input={"code": "NIKKEI225.INDEX"},
            result="second tool output",
            timestamp=1002,
            source_id_prefix="tool2",
            keywords=["日经225"],
        )
        mem.complete_turn(
            turn_id="tool-turn",
            semantic_text="两个方向都查完了",
            provider_output_raw="两个方向都查完了",
            annotation_status=AnnotationStatus.MISSING,
            timestamp=1004,
            source_id="tool-final",
        )

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 2)
        self.assertEqual(first_tool["tool_result"]["memory_metadata"]["categories"], ["tool_trace"])
        self.assertEqual(second_tool["tool_result"]["memory_metadata"]["categories"], ["tool_trace"])
        self.assertEqual(mem.store.get_unsummarized_messages(namespace=mem.namespace), [])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        operation = next(item for item in visible if item["kind"] == "memory.operation_digest")
        self.assertEqual(
            set(operation["source_ids"]),
            {"tool1:tool_use", "tool1:tool_result", "tool2:tool_use", "tool2:tool_result"},
        )
        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertNotIn("first search output", summary_requests[0].user_prompt)

    def test_external_event_and_response_compact_as_one_annotated_turn(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        handle = mem.begin_turn(
            turn_id="event-turn",
            stimuli=[
                TimelineEntryInput(
                    kind="event.finance",
                    origin=EntryOrigin.ENVIRONMENT,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="只用于测试",
                    payload={"source": "public_news", "title": "虚构事件", "summary": "只用于测试"},
                    timestamp=1000,
                    source_id="event-1",
                    compatibility_role="event.finance",
                )
            ],
        )
        completed = mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="条件式分析",
            provider_output_raw="条件式分析",
            memory_annotation={"keywords": ["虚构事件"], "categories": ["plan_goal"]},
            annotation_status=AnnotationStatus.ACCEPTED_HOST,
            timestamp=1001,
            source_id="assistant-1",
        )

        out = mem.compact_due_sync()

        self.assertTrue(completed.completed)
        self.assertEqual(completed.updated_targets[0].kind, "event.finance")
        self.assertEqual(out["summaries_created"], 1)
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(visible[0]["source_ids"], ["event-1", "assistant-1"])
        self.assertIn("event.finance", llm.requests[0].user_prompt)
        self.assertIn("source: public_news", llm.requests[0].user_prompt)
        self.assertIn("title: 虚构事件", llm.requests[0].user_prompt)

    def test_material_entry_compacts_with_its_turn_without_polluting_episode(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        mem.begin_turn(
            turn_id="material-turn",
            stimuli=[
                TimelineEntryInput(
                    source_id="m0",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="看看这张图",
                    payload={"text": "看看这张图"},
                    timestamp=1000,
                    actor=Actor(stable_id="qq-1", display_name="张三"),
                )
            ],
        )
        material = mem.append_entry(
            TimelineEntryInput(
                source_id="mat1",
                kind="material.image.reference",
                origin=EntryOrigin.ENVIRONMENT,
                turn_role=TurnRole.INTERMEDIATE,
                semantic_text="图片材料已就绪",
                payload={"file_id": "file_img_001", "filename": "photo.jpg", "status": "ready"},
                timestamp=1001,
                memory_metadata={"categories": ["material_trace"], "keywords": ["题目图片"]},
                semanticize=False,
            ),
            turn_id="material-turn",
        )
        mem.complete_turn(
            turn_id="material-turn",
            semantic_text="图片里是一道题",
            provider_output_raw="图片里是一道题",
            annotation_status=AnnotationStatus.MISSING,
            timestamp=1002,
            source_id="material-final",
        )

        out = mem.compact_due_sync()

        self.assertEqual(material.memory_metadata["categories"], ["material_trace"])
        self.assertEqual(out["summaries_created"], 2)
        self.assertEqual(mem.store.get_unsummarized_messages(namespace=mem.namespace), [])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        episode = next(item for item in visible if item["kind"] == "memory.episode_summary")
        operation = next(item for item in visible if item["kind"] == "memory.operation_digest")
        self.assertEqual(episode["source_ids"], ["m0", "material-final"])
        self.assertEqual(operation["source_ids"], ["mat1"])

    def test_open_turn_blocks_compaction_instead_of_splitting_it(self) -> None:
        cfg = MemoryConfig(
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
        mem.begin_turn(
            turn_id="open-turn",
            stimuli=[
                TimelineEntryInput(
                    source_id="u1",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="x" * 50,
                    payload={"text": "x" * 50},
                    timestamp=1000,
                )
            ],
        )

        out = mem.compact_due_sync()

        self.assertEqual(out["status"], "blocked_by_open_turn")
        self.assertEqual(out["summaries_created"], 0)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 1)

    def test_token_policy_compacts_oldest_complete_turn_and_keeps_recent_turn(self) -> None:
        cfg = MemoryConfig(
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
        _complete_turn(mem, 1, user_text="aaaa", assistant_text="bb", timestamp=1000)
        _complete_turn(mem, 2, user_text="cccc", assistant_text="dd", timestamp=1010)

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 1)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        self.assertEqual([m["source_id"] for m in remaining], ["u2", "a2"])
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(visible[0]["source_ids"], ["u1", "a1"])

    def test_ratio_cutpoint_never_partially_summarizes_a_turn(self) -> None:
        cfg = MemoryConfig(
            raw_token_trigger=12,
            raw_token_batch_ratio=0.5,
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
        _complete_turn(mem, 1, user_text="aa", assistant_text="bb", timestamp=1000)
        _complete_turn(mem, 2, user_text="ccc", assistant_text="d", timestamp=1010)
        _complete_turn(mem, 3, user_text="eeee", assistant_text="f", timestamp=1020)

        out = mem.compact_due_sync()

        self.assertEqual(out["summaries_created"], 1)
        remaining = mem.store.get_unsummarized_messages(namespace=mem.namespace)
        counts: dict[str, int] = {}
        for item in remaining:
            counts[item["turn_id"]] = counts.get(item["turn_id"], 0) + 1
        self.assertTrue(counts)
        self.assertTrue(all(count == 2 for count in counts.values()))
        visible = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        self.assertEqual(len(visible[0]["source_ids"]) % 2, 0)

    def test_token_policy_compacts_first_long_turn_without_tail(self) -> None:
        cfg = MemoryConfig(
            raw_token_trigger=10,
            raw_token_batch_ratio=0.67,
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
        _complete_turn(mem, 1, user_text="x" * 50, assistant_text="ok", timestamp=1000)

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
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        llm = FailThenSummaryLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        _complete_turn(mem, 1, user_text="第一条重要事实", assistant_text="已记住", timestamp=1000)

        first = mem.compact_due_sync()
        self.assertEqual(first["summaries_created"], 0)
        self.assertEqual(first["summary_retry_pending"], 1)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 2)
        self.assertEqual(mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10), [])

        second = mem.compact_due_sync()
        self.assertEqual(second["summaries_created"], 1)
        self.assertEqual(len(mem.store.get_unsummarized_messages(namespace=mem.namespace)), 0)

    def test_compaction_passes_configured_llm_retries(self) -> None:
        cfg = MemoryConfig(
            raw_token_trigger=1,
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
        _complete_turn(mem, 1, user_text="第一条重要事实", assistant_text="已记住", timestamp=1000)

        mem.compact_due_sync()

        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertEqual(summary_requests[0].max_retries, 4)

    def test_summary_prompt_carries_weekday_anchor(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        _complete_turn(
            mem,
            1,
            user_text="上周二说的事情还记得吗",
            assistant_text="记得,我们可以继续整理。",
            timestamp=_ts(2026, 4, 10, 9),
        )

        mem.compact_due_sync()

        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertIn("2026-04-10 周五", summary_requests[0].user_prompt)
        self.assertIn("上周二", summary_requests[0].user_prompt)

    def test_summary_prompt_keeps_group_actor_attribution(self) -> None:
        cfg = MemoryConfig(raw_token_trigger=1, episodic_compact_trigger_count=99)
        llm = CapturingLLM()
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="group-1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=cfg,
            embedding=HashedEmbeddingProvider(),
        )
        _complete_turn(
            mem,
            1,
            user_text="我下周三要复盘基金组合",
            assistant_text="到时一起复盘",
            timestamp=1000,
            actor=Actor(stable_id="qq-1", display_name="张三"),
        )

        mem.compact_due_sync()

        summary_requests = [req for req in llm.requests if req.task_type == TaskType.SUMMARY]
        self.assertEqual(len(summary_requests), 1)
        self.assertIn("user(张三;id=qq-1): 我下周三要复盘基金组合", summary_requests[0].user_prompt)


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
        llm = CapturingLLM()
        compaction = Compaction(store=self.store, index=self.index, llm=llm, config=self.cfg, timezone="Asia/Shanghai")
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

        first = compaction.run_due(namespace=self.ns)

        self.assertEqual(first["semantic_created"], 1)
        self.assertEqual(first["reinforced"], 0)
        self.assertEqual(len(self.store.get_uncompacted_episodic_summaries(namespace=self.ns)), 3)
        self.assertEqual(
            len([request for request in llm.requests if request.task_type == TaskType.SEMANTIC]),
            1,
        )

        second = compaction.run_due(namespace=self.ns)
        third = compaction.run_due(namespace=self.ns)
        fourth = compaction.run_due(namespace=self.ns)

        self.assertEqual(second["semantic_created"], 0)
        self.assertEqual(second["reinforced"], 1)
        self.assertEqual(third["semantic_created"], 0)
        self.assertEqual(third["reinforced"], 1)
        self.assertEqual(fourth["semantic_created"], 0)
        self.assertEqual(fourth["reinforced"], 0)
        self.assertEqual(len(self.store.get_uncompacted_episodic_summaries(namespace=self.ns)), 1)
        recent = self.store.get_recent_semantic_summaries(namespace=self.ns, limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["reinforcement_count"], 3)
        self.assertEqual(recent[0]["source_summary_ids"], ["ep0", "ep1", "ep2"])
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
