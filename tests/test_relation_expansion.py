"""Relation-aware Retrieval V2 expansion and atomic token-budget tests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from memcore import (
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TokenCounter,
    TurnRole,
)
from memcore.index.entry_builder import build_semantic_entry, build_summary_entry
from memcore.index.metadata_filters import INDEX_SCHEMA_KEY, INDEX_SCHEMA_VERSION


class NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


class CharacterTokenCounter(TokenCounter):
    @property
    def quality(self) -> str:
        return "exact-test-counter"

    def count_text(self, text: str) -> int:
        return len(str(text or ""))


def _memory(*, token_counter: TokenCounter | None = None) -> tuple[MemorySystem, SQLiteMemoryStore]:
    store = SQLiteMemoryStore(":memory:")
    embedding = HashedEmbeddingProvider()
    mem = MemorySystem(
        llm=NoopLLM(),
        namespace=Namespace(user_id="user", conversation_id="c1"),
        timezone="Asia/Shanghai",
        config=MemoryConfig(enable_verifier=False),
        store=store,
        index=InMemoryVectorIndex(embedding=embedding),
        embedding=embedding,
        token_counter=token_counter,
    )
    return mem, store


def _complete_message_turn(
    mem: MemorySystem,
    *,
    turn_id: str,
    source_id: str,
    text: str,
    final_id: str,
    final_text: str,
    timestamp: int,
) -> None:
    mem.begin_turn(
        stimuli=[
            TimelineEntryInput(
                source_id=source_id,
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text=text,
                payload={"text": text},
                timestamp=timestamp,
            )
        ],
        turn_id=turn_id,
        opened_at=timestamp,
    )
    mem.complete_turn(
        turn_id=turn_id,
        semantic_text=final_text,
        provider_output_raw=json.dumps({"speech": final_text}, ensure_ascii=False),
        memory_annotation={"keywords": ["滑雪"], "categories": ["preference"]},
        annotation_status="accepted",
        source_id=final_id,
        timestamp=timestamp + 1,
    )


class RelationExpansionTests(unittest.TestCase):
    def test_default_hit_expands_to_stimulus_and_final_without_tool_trace(self) -> None:
        mem, store = _memory()
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="question",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="我想周末去滑雪",
                        timestamp=1000,
                    )
                ],
                turn_id="turn-default",
                opened_at=1000,
            )
            mem.append_entry(
                TimelineEntryInput(
                    source_id="weather-call",
                    kind="tool.weather.call",
                    origin=EntryOrigin.ASSISTANT,
                    turn_role=TurnRole.ACTION,
                    semantic_text="查询雪场天气",
                    correlation_id="weather",
                    timestamp=1001,
                ),
                turn_id="turn-default",
            )
            mem.append_entry(
                TimelineEntryInput(
                    source_id="weather-result",
                    kind="tool.weather.result",
                    origin=EntryOrigin.ENVIRONMENT,
                    turn_role=TurnRole.OBSERVATION,
                    semantic_text="雪场晴天",
                    correlation_id="weather",
                    trace_metadata={"status": "success"},
                    timestamp=1002,
                ),
                turn_id="turn-default",
            )
            mem.complete_turn(
                turn_id="turn-default",
                semantic_text="记得带护目镜。",
                provider_output_raw='{"speech":"记得带护目镜。"}',
                memory_annotation={"keywords": ["滑雪"], "categories": ["preference"]},
                annotation_status="accepted",
                source_id="answer",
                timestamp=1003,
            )

            result = mem.retrieve_structured("周末滑雪", keywords=["滑雪"])

            self.assertEqual(result.status, "found")
            self.assertEqual(len(result.matches), 1)
            self.assertEqual(result.matches[0].source_ids, ("question", "answer"))
            self.assertIn("周末去滑雪", result.matches[0].rendered_text)
            self.assertIn("带护目镜", result.matches[0].rendered_text)
            self.assertNotIn("weather-call", result.matches[0].source_ids)
            self.assertNotIn("雪场晴天", result.matches[0].rendered_text)
        finally:
            mem.close()
            store.close()

    def test_explicit_parallel_tool_hit_returns_only_its_complete_correlation_branch(self) -> None:
        mem, store = _memory()
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="parallel-question",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="并行查天气和汇率",
                        timestamp=2000,
                    )
                ],
                turn_id="turn-parallel",
                opened_at=2000,
            )
            for source_id, kind, correlation, text, timestamp in (
                ("weather-call", "tool.weather.call", "weather", "查北京天气", 2001),
                ("fx-call", "tool.fx.call", "fx", "查美元汇率", 2002),
                ("fx-result", "tool.fx.result", "fx", "汇率是 7.2", 2003),
                ("weather-result", "tool.weather.result", "weather", "北京晴天 25 度", 2004),
            ):
                role = TurnRole.ACTION if kind.endswith(".call") else TurnRole.OBSERVATION
                mem.append_entry(
                    TimelineEntryInput(
                        source_id=source_id,
                        kind=kind,
                        origin=EntryOrigin.ASSISTANT if role is TurnRole.ACTION else EntryOrigin.ENVIRONMENT,
                        turn_role=role,
                        semantic_text=text,
                        correlation_id=correlation,
                        trace_metadata={} if role is TurnRole.ACTION else {"status": "success"},
                        timestamp=timestamp,
                    ),
                    turn_id="turn-parallel",
                )
            mem.complete_turn(
                turn_id="turn-parallel",
                semantic_text="天气与汇率都查好了。",
                provider_output_raw='{"speech":"天气与汇率都查好了。"}',
                annotation_status="accepted",
                source_id="parallel-final",
                timestamp=2005,
            )

            result = mem.retrieve_structured(
                "北京晴天",
                keywords=["北京晴天"],
                include_explicit=True,
                kind_patterns=["tool.weather.*"],
            )

            self.assertEqual(result.status, "found")
            self.assertEqual(len(result.matches), 1)
            self.assertEqual(result.matches[0].source_ids, ("weather-call", "weather-result"))
            self.assertEqual(result.matches[0].correlation_id, "weather")
            self.assertIn("北京晴天", result.matches[0].rendered_text)
            self.assertNotIn("汇率是", result.matches[0].rendered_text)
            self.assertNotIn("parallel-final", result.matches[0].source_ids)
        finally:
            mem.close()
            store.close()

    def test_explicit_event_expands_to_event_and_its_explicit_final(self) -> None:
        mem, store = _memory()
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="finance-event",
                        kind="event.finance.flash",
                        origin=EntryOrigin.ENVIRONMENT,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="虚构政策快讯",
                        timestamp=2500,
                    )
                ],
                turn_id="turn-event",
                opened_at=2500,
            )
            mem.complete_turn(
                turn_id="turn-event",
                semantic_text="这只是方向信号。",
                provider_output_raw='{"speech":"这只是方向信号。"}',
                annotation_status="missing",
                source_id="event-final",
                timestamp=2501,
            )

            result = mem.retrieve_structured(
                "政策快讯",
                keywords=["政策快讯"],
                include_explicit=True,
                kind_patterns=["event.finance.*"],
            )

            self.assertEqual(result.status, "found")
            self.assertEqual(result.matches[0].source_ids, ("finance-event", "event-final"))
            self.assertIn("方向信号", result.matches[0].rendered_text)
        finally:
            mem.close()
            store.close()

    def test_incomplete_tool_branch_is_never_returned_as_a_partial_pair(self) -> None:
        mem, store = _memory()
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="pending-question",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="发起未完成查询",
                        timestamp=2600,
                    )
                ],
                turn_id="turn-pending",
                opened_at=2600,
            )
            mem.append_entry(
                TimelineEntryInput(
                    source_id="pending-call",
                    kind="tool.pending.call",
                    origin=EntryOrigin.ASSISTANT,
                    turn_role=TurnRole.ACTION,
                    semantic_text="未完成查询参数",
                    correlation_id="pending",
                    timestamp=2601,
                ),
                turn_id="turn-pending",
            )

            result = mem.retrieve_structured(
                "未完成查询参数",
                keywords=["未完成查询参数"],
                include_explicit=True,
                kind_patterns=["tool.pending.*"],
            )

            self.assertEqual(result.status, "empty")
            self.assertEqual(result.rejected_counts["incomplete_relation"], 1)
            self.assertEqual(result.matches, ())
        finally:
            mem.close()
            store.close()

    def test_semantic_lineage_suppresses_summary_and_raw_duplicates(self) -> None:
        mem, store = _memory()
        try:
            raw = mem.record_user_turn("我喜欢滑雪", timestamp=3000, source_id="raw-ski")
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "summary-ski",
                    "timestamp": 3001,
                    "diary_summary": "用户喜欢滑雪",
                    "source_ids": [raw["source_id"]],
                    "memory_metadata": {"keywords": ["滑雪"]},
                },
            )
            semantic = store.add_semantic_summary(
                namespace=mem.namespace,
                record={
                    "semantic_id": "semantic-ski",
                    "timestamp": 3002,
                    "semantic_summary": "稳定偏好：滑雪",
                    "source_summary_ids": [summary["summary_id"]],
                    "memory_metadata": {"keywords": ["滑雪"]},
                },
            )
            mem.index.upsert([build_summary_entry(summary), build_semantic_entry(semantic)])
            for source_id in ("summary-ski", "semantic-ski"):
                store.set_index_state(
                    source_id,
                    "indexed",
                    index_schema_version=INDEX_SCHEMA_VERSION,
                    index_key=INDEX_SCHEMA_KEY,
                )

            result = mem.retrieve_structured("滑雪", keywords=["滑雪"])

            self.assertEqual(result.status, "found")
            self.assertEqual([match.source_id for match in result.matches], ["semantic-ski"])
            self.assertEqual(result.matches[0].lineage, ("summary-ski", "raw-ski"))
        finally:
            mem.close()
            store.close()

    def test_broken_cross_namespace_lineage_is_removed_and_marked_invalid(self) -> None:
        mem, store = _memory()
        other = Namespace(user_id="other", conversation_id="c1")
        try:
            store.add_message(
                namespace=other,
                role="user",
                content="其他用户私密滑雪记录",
                timestamp=4000,
                source_id="other-raw",
            )
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "broken-summary",
                    "timestamp": 4001,
                    "diary_summary": "错误跨域滑雪摘要",
                    "source_ids": ["other-raw"],
                    "memory_metadata": {"keywords": ["滑雪"]},
                },
            )
            mem.index.upsert([build_summary_entry(summary)])
            store.set_index_state(
                "broken-summary",
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )

            result = mem.retrieve_structured("滑雪", keywords=["滑雪"])
            stored = store.get_retrieval_record(namespace=mem.namespace, source_id="broken-summary")

            self.assertEqual(result.status, "empty")
            self.assertEqual(result.rejected_counts["invalid_lineage"], 1)
            self.assertEqual(stored["index_status"], "invalid_lineage")
            self.assertNotIn("其他用户私密", json.dumps(result.to_dict(), ensure_ascii=False))
        finally:
            mem.close()
            store.close()

    def test_cyclic_lineage_is_structurally_invalid(self) -> None:
        mem, store = _memory()
        try:
            store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "cycle-a",
                    "timestamp": 4500,
                    "diary_summary": "cycle a",
                    "source_ids": ["cycle-b"],
                },
            )
            store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "cycle-b",
                    "timestamp": 4501,
                    "diary_summary": "cycle b",
                    "source_ids": ["cycle-a"],
                },
            )

            closure = store.resolve_lineage_source_ids(
                namespace=mem.namespace,
                source_ids=("cycle-a",),
                include_ancestors=False,
            )

            self.assertEqual(closure.status, "invalid")
            self.assertEqual(closure.reason, "lineage_broken_or_cyclic")
        finally:
            mem.close()
            store.close()

    def test_visible_raw_excludes_hidden_derived_ancestor_before_scoring(self) -> None:
        mem, store = _memory()
        try:
            current = mem.record_user_turn("当前可见滑雪消息", timestamp=5000, source_id="visible-raw")
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "hidden-derived",
                    "timestamp": 5001,
                    "diary_summary": "不应重复返回的滑雪摘要",
                    "source_ids": ["visible-raw"],
                    "memory_metadata": {"keywords": ["滑雪"]},
                },
            )
            mem.index.upsert([build_summary_entry(summary)])
            store.set_index_state(
                "hidden-derived",
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )
            store.mark_summaries_semanticized(["hidden-derived"], "hidden-semantic-placeholder")

            result = mem.retrieve_for_turn_structured(
                current=current,
                query="滑雪",
                keywords=["滑雪"],
            )

            self.assertEqual(result.status, "empty")
            self.assertNotIn("hidden-derived", [match.source_id for match in result.matches])
        finally:
            mem.close()
            store.close()

    def test_visible_lineage_store_failure_is_structured(self) -> None:
        mem, store = _memory()
        try:
            current = mem.record_user_turn("当前消息", timestamp=5500, source_id="visible-current")
            with patch.object(store, "resolve_lineage_source_ids", side_effect=NotImplementedError):
                result = mem.retrieve_for_turn_structured(current=current, query="过去")
            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.reason, "lineage_store_unsupported")
        finally:
            mem.close()
            store.close()


class AtomicBudgetTests(unittest.TestCase):
    def test_explicit_budget_without_token_counter_is_structured_unavailable(self) -> None:
        mem, store = _memory()
        try:
            mem.record_user_turn("预算测试", timestamp=5900, source_id="budget-no-counter")
            result = mem.retrieve_structured("预算测试", result_token_budget=100)
            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.reason, "token_counter_required")
        finally:
            mem.close()
            store.close()

    def test_oversized_tool_branch_becomes_one_explicitly_truncated_group(self) -> None:
        mem, store = _memory(token_counter=CharacterTokenCounter())
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="budget-question",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="读取超长工具结果",
                        timestamp=6000,
                    )
                ],
                turn_id="turn-budget",
                opened_at=6000,
            )
            mem.append_entry(
                TimelineEntryInput(
                    source_id="budget-call",
                    kind="tool.long.call",
                    origin=EntryOrigin.ASSISTANT,
                    turn_role=TurnRole.ACTION,
                    semantic_text="调用长结果工具",
                    correlation_id="long",
                    timestamp=6001,
                ),
                turn_id="turn-budget",
            )
            mem.append_entry(
                TimelineEntryInput(
                    source_id="budget-result",
                    kind="tool.long.result",
                    origin=EntryOrigin.ENVIRONMENT,
                    turn_role=TurnRole.OBSERVATION,
                    semantic_text="超长结果" + "甲" * 1200,
                    correlation_id="long",
                    trace_metadata={"status": "success"},
                    timestamp=6002,
                ),
                turn_id="turn-budget",
            )
            mem.complete_turn(
                turn_id="turn-budget",
                semantic_text="处理完成。",
                provider_output_raw='{"speech":"处理完成。"}',
                annotation_status="accepted",
                source_id="budget-final",
                timestamp=6003,
            )

            result = mem.retrieve_structured(
                "超长结果",
                keywords=["超长结果"],
                include_explicit=True,
                kind_patterns=["tool.long.*"],
                result_token_budget=520,
            )

            self.assertEqual(result.status, "found")
            self.assertTrue(result.truncated)
            self.assertEqual(result.omitted_match_count, 0)
            self.assertLessEqual(result.token_usage, 520)
            self.assertEqual(result.matches[0].source_ids, ("budget-call", "budget-result"))
            self.assertTrue(result.matches[0].truncated)
            self.assertEqual(result.matches[0].semantic_text, "")
            self.assertIn("检索原子组摘录", result.matches[0].rendered_text)
        finally:
            mem.close()
            store.close()

    def test_budget_omits_a_whole_second_turn_instead_of_splitting_it(self) -> None:
        mem, store = _memory(token_counter=CharacterTokenCounter())
        try:
            _complete_message_turn(
                mem,
                turn_id="turn-one",
                source_id="one-user",
                text="共同主题滑雪 第一轮",
                final_id="one-final",
                final_text="第一轮回复",
                timestamp=7000,
            )
            _complete_message_turn(
                mem,
                turn_id="turn-two",
                source_id="two-user",
                text="共同主题滑雪 第二轮",
                final_id="two-final",
                final_text="第二轮回复",
                timestamp=7100,
            )
            full = mem.retrieve_structured(
                "共同主题滑雪",
                keywords=["共同主题滑雪"],
                result_token_budget=100_000,
            )
            self.assertEqual(len(full.matches), 2)
            first_count = len(
                json.dumps(full.matches[0].to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )

            limited = mem.retrieve_structured(
                "共同主题滑雪",
                keywords=["共同主题滑雪"],
                result_token_budget=first_count,
            )

            self.assertEqual(limited.status, "found")
            self.assertEqual(len(limited.matches), 1)
            self.assertFalse(limited.matches[0].truncated)
            self.assertEqual(len(limited.matches[0].source_ids), 2)
            self.assertEqual(limited.omitted_match_count, 1)
            self.assertTrue(limited.truncated)
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
