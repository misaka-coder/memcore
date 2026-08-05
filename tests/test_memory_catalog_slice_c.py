"""Slice C: hard lineage-scoped semantic retrieval."""

from __future__ import annotations

import copy
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


class _RecordingIndex(InMemoryVectorIndex):
    def __init__(self, *, embedding: HashedEmbeddingProvider) -> None:
        super().__init__(embedding=embedding)
        self.count_wheres: list[dict] = []
        self.semantic_wheres: list[dict] = []
        self.keyword_wheres: list[dict] = []

    def count_candidates(self, **kwargs) -> int:
        self.count_wheres.append(copy.deepcopy(kwargs["where"]))
        return super().count_candidates(**kwargs)

    def semantic_search(self, **kwargs) -> list[dict]:
        self.semantic_wheres.append(copy.deepcopy(kwargs["where"]))
        return super().semantic_search(**kwargs)

    def keyword_search(self, **kwargs) -> list[dict]:
        self.keyword_wheres.append(copy.deepcopy(kwargs["where"]))
        return super().keyword_search(**kwargs)


class _IgnoringLineageIndex(_RecordingIndex):
    def __init__(self, *, embedding: HashedEmbeddingProvider, leaked_source_id: str) -> None:
        super().__init__(embedding=embedding)
        self.leaked_source_id = leaked_source_id

    def count_candidates(self, **kwargs) -> int:
        self.count_wheres.append(copy.deepcopy(kwargs["where"]))
        return 1

    def semantic_search(self, **kwargs) -> list[dict]:
        self.semantic_wheres.append(copy.deepcopy(kwargs["where"]))
        return [{"source_id": self.leaked_source_id, "semantic_score": 1.0}]

    def keyword_search(self, **kwargs) -> list[dict]:
        self.keyword_wheres.append(copy.deepcopy(kwargs["where"]))
        return []


def _ts(day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(2026, 8, day, hour, minute, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


class LineageScopedRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = HashedEmbeddingProvider()
        self.store = SQLiteMemoryStore(":memory:")
        self.index = _RecordingIndex(embedding=self.embedding)
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="u1", conversation_id="group-1"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.embedding,
            config=MemoryConfig(),
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()

    def _episode(self, *, raw_id: str, episode_id: str, text: str, timestamp: int) -> None:
        self.mem.record_user_turn(text, timestamp=timestamp, source_id=raw_id)
        self.store.add_summary(
            namespace=self.mem.namespace,
            record={
                "summary_id": episode_id,
                "timestamp": timestamp,
                "period_start_ts": timestamp,
                "period_end_ts": timestamp + 1,
                "diary_summary": text,
                "memory_title": "早茶同行与账单",
                "catalog_hint": "可回答早茶同行人员和账单细节。",
                "source_ids": [raw_id],
                # Keep the episode out of the ordinary visible-summary window,
                # so retrieve_for_turn can deliberately reopen its raw evidence.
                "is_semanticized": 1,
            },
        )
        self.store.mark_messages_summarized([raw_id], episode_id)
        self.mem.reindex_all(current_conversation_only=True)

    def test_unknown_answer_is_found_only_inside_selected_episode(self) -> None:
        self._episode(
            raw_id="target-raw",
            episode_id="target-episode",
            text="扬州早茶账单已经结清，同行的人叫李嘉图。",
            timestamp=_ts(2, 9),
        )
        self._episode(
            raw_id="outside-raw",
            episode_id="outside-episode",
            text="广州早茶账单已经结清，同行的人叫张三。",
            timestamp=_ts(3, 9),
        )

        result = self.mem.retrieve_structured(
            "早茶 同行 账单",
            within_memory_id="target-episode",
            source_layers=["raw"],
            cross_conversation=True,
        )

        self.assertEqual(result.status, "found")
        self.assertIn("李嘉图", "\n".join(result.rendered_texts))
        self.assertNotIn("张三", "\n".join(result.rendered_texts))
        self.assertEqual(
            result.lineage_scope,
            {
                "status": "resolved",
                "within_memory_id": "target-episode",
                "candidate_source_count": 2,
            },
        )

    def test_entity_relaxation_keeps_lineage_scope_on_every_index_operation(self) -> None:
        self._episode(
            raw_id="target-raw",
            episode_id="target-episode",
            text="扬州早茶账单已经结清，同行的人叫李嘉图。",
            timestamp=_ts(2, 9),
        )
        self._episode(
            raw_id="outside-raw",
            episode_id="outside-episode",
            text="扬州早茶账单已经结清，同行的人叫另一个人。",
            timestamp=_ts(3, 9),
        )
        self.index.count_wheres.clear()
        self.index.semantic_wheres.clear()
        self.index.keyword_wheres.clear()

        result = self.mem.retrieve_structured(
            "早茶 同行 账单",
            entity_anchors=["尚不知道的同行者"],
            within_memory_id="target-episode",
            source_layers=["raw"],
            cross_conversation=True,
        )

        self.assertEqual(result.status, "found")
        self.assertTrue(result.entity_filter_relaxed)
        self.assertEqual(
            result.relaxation_steps,
            ("drop_entity_requirement_after_zero_candidates:raw",),
        )
        for where in [*self.index.count_wheres, *self.index.semantic_wheres, *self.index.keyword_wheres]:
            self.assertEqual(where["source_id"]["$in"], ["target-episode", "target-raw"])

    def test_empty_selected_layer_never_falls_back_to_global_candidates(self) -> None:
        self._episode(
            raw_id="target-raw",
            episode_id="target-episode",
            text="目标段只有原始对话。",
            timestamp=_ts(2, 9),
        )
        self.store.add_semantic_summary(
            namespace=self.mem.namespace,
            record={
                "semantic_id": "outside-semantic",
                "timestamp": _ts(3, 9),
                "semantic_summary": "早茶同行账单长期记忆",
                "source_summary_ids": [],
            },
        )
        self.mem.reindex_all(current_conversation_only=True)

        result = self.mem.retrieve_structured(
            "早茶 同行 账单",
            within_memory_id="target-episode",
            source_layers=["semantic_summary"],
            cross_conversation=True,
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.reason, "no_match")
        self.assertEqual(result.candidate_counts["derived_effective"], 0)
        self.assertEqual(result.lineage_scope["within_memory_id"], "target-episode")

    def test_backend_ignoring_lineage_filter_is_rejected_as_unavailable(self) -> None:
        self._episode(
            raw_id="target-raw",
            episode_id="target-episode",
            text="目标早茶记录。",
            timestamp=_ts(2, 9),
        )
        self._episode(
            raw_id="outside-raw",
            episode_id="outside-episode",
            text="不应泄漏的另一段早茶记录。",
            timestamp=_ts(3, 9),
        )
        unsafe = MemorySystem(
            llm=_NoopLLM(),
            namespace=self.mem.namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            index=_IgnoringLineageIndex(embedding=self.embedding, leaked_source_id="outside-raw"),
            embedding=self.embedding,
            config=MemoryConfig(),
        )

        result = unsafe.retrieve_structured(
            "早茶",
            within_memory_id="target-episode",
            source_layers=["raw"],
            cross_conversation=True,
        )

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.reason, "index_filter_unsupported")
        self.assertEqual(result.lineage_scope["within_memory_id"], "target-episode")
        unsafe.close()

    def test_missing_and_broken_lineage_are_structured_without_broadening(self) -> None:
        missing = self.mem.retrieve_structured(
            "任何内容",
            within_memory_id="missing",
            cross_conversation=True,
        )
        self.store.add_summary(
            namespace=self.mem.namespace,
            record={
                "summary_id": "broken",
                "timestamp": _ts(4),
                "diary_summary": "来源缺失",
                "source_ids": ["missing-raw"],
            },
        )
        broken = self.mem.retrieve_structured(
            "任何内容",
            within_memory_id="broken",
            cross_conversation=True,
        )

        self.assertEqual(missing.status, "invalid")
        self.assertEqual(missing.reason, "within_memory_id_not_found_or_out_of_scope")
        self.assertEqual(broken.status, "unavailable")
        self.assertEqual(broken.reason, "lineage_source_missing_or_out_of_scope")

    def test_selected_node_respects_host_conversation_scope_and_hard_user_namespace(self) -> None:
        other_conversation = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="u1", conversation_id="group-2"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.embedding,
            config=MemoryConfig(),
        )
        other_conversation.record_user_turn(
            "跨会话早茶同行是李嘉图。",
            timestamp=_ts(2, 9),
            source_id="cross-raw",
        )
        self.store.add_summary(
            namespace=other_conversation.namespace,
            record={
                "summary_id": "cross-episode",
                "timestamp": _ts(2, 9),
                "diary_summary": "跨会话早茶",
                "source_ids": ["cross-raw"],
                "is_semanticized": 1,
            },
        )
        self.store.mark_messages_summarized(["cross-raw"], "cross-episode")
        other_conversation.reindex_all(current_conversation_only=True)
        foreign_user = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="u2", conversation_id="group-1"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.embedding,
            config=MemoryConfig(),
        )
        foreign_user.record_user_turn(
            "别的用户的私密早茶记录。",
            timestamp=_ts(2, 10),
            source_id="foreign-raw",
        )
        self.store.add_summary(
            namespace=foreign_user.namespace,
            record={
                "summary_id": "foreign-episode",
                "timestamp": _ts(2, 10),
                "diary_summary": "私密早茶",
                "source_ids": ["foreign-raw"],
                "is_semanticized": 1,
            },
        )

        conversation_blocked = self.mem.retrieve_structured(
            "早茶同行",
            within_memory_id="cross-episode",
            source_layers=["raw"],
            cross_conversation=False,
        )
        conversation_allowed = self.mem.retrieve_structured(
            "早茶同行",
            within_memory_id="cross-episode",
            source_layers=["raw"],
            cross_conversation=True,
        )
        foreign_blocked = self.mem.retrieve_structured(
            "早茶",
            within_memory_id="foreign-episode",
            cross_conversation=True,
        )

        self.assertEqual(conversation_blocked.status, "invalid")
        self.assertEqual(conversation_allowed.status, "found")
        self.assertIn("李嘉图", "\n".join(conversation_allowed.rendered_texts))
        self.assertEqual(foreign_blocked.status, "invalid")
        self.assertEqual(foreign_blocked.reason, "within_memory_id_not_found_or_out_of_scope")
        other_conversation.close()
        foreign_user.close()

    def test_native_tool_schema_and_dispatch_preserve_the_selected_memory_id(self) -> None:
        self._episode(
            raw_id="target-raw",
            episode_id="target-episode",
            text="早茶同行的人叫李嘉图。",
            timestamp=_ts(2, 9),
        )
        schema = build_native_memory_tool_specs(tool_format="plain")[0]["parameters"]

        result = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {
                "query": "早茶同行",
                "within_memory_id": "target-episode",
                "source_layers": ["raw"],
            },
            mem=self.mem,
            current={"source_id": "current", "timestamp": _ts(5)},
        )
        invalid = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "早茶同行", "within_memory_id": ["target-episode"]},
            mem=self.mem,
            current={"source_id": "current", "timestamp": _ts(5)},
        )

        self.assertIn("within_memory_id", schema["properties"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["lineage_scope"]["within_memory_id"], "target-episode")
        self.assertIn("李嘉图", "\n".join(result["result"]["snippets"]))
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["reason"], "within_memory_id_must_be_string")


if __name__ == "__main__":
    unittest.main()
