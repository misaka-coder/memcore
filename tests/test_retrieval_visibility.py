"""Retrieval V2 admission, hard-filter, and structured-result acceptance tests."""

from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from memcore import (
    ConfigError,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    RetrievalPolicy,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
    VectorIndex,
)
from memcore.index.entry_builder import build_raw_entry
from memcore.index.metadata_filters import (
    INDEX_SCHEMA_KEY,
    INDEX_SCHEMA_VERSION,
    kind_filter_flags,
    kind_filter_key,
    kind_prefixes,
)


class NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


class WhereSpyIndex(VectorIndex):
    def __init__(self) -> None:
        self.semantic_wheres: list[dict[str, Any]] = []
        self.keyword_wheres: list[dict[str, Any]] = []

    def upsert(self, entries: list[dict[str, Any]]) -> None:
        return None

    def semantic_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.semantic_wheres.append(dict(kwargs["where"]))
        return []

    def keyword_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.keyword_wheres.append(dict(kwargs["where"]))
        return []

    def count_candidates(self, **kwargs: Any) -> int:
        return 1

    def delete(self, source_ids: list[str]) -> None:
        return None

    def count(self) -> int:
        return 0


class ZeroScoreIndex(WhereSpyIndex):
    def semantic_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.semantic_wheres.append(dict(kwargs["where"]))
        return [
            {
                "source_id": "zero-user",
                "semantic_score": 0.0,
                "metadata": {},
                "document": "完全无关",
            }
        ]


class NonFiniteScoreIndex(ZeroScoreIndex):
    def semantic_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        hits = super().semantic_search(**kwargs)
        hits[0]["semantic_score"] = float("nan")
        return hits


class FixedHitIndex(WhereSpyIndex):
    def __init__(self, source_id: str) -> None:
        super().__init__()
        self.source_id = source_id

    def semantic_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.semantic_wheres.append(dict(kwargs["where"]))
        return [
            {
                "source_id": self.source_id,
                "semantic_score": 1.0,
                "metadata": {},
                "document": "malicious backend hit",
            }
        ]


def _backends() -> tuple[SQLiteMemoryStore, InMemoryVectorIndex, HashedEmbeddingProvider]:
    embedding = HashedEmbeddingProvider()
    return SQLiteMemoryStore(":memory:"), InMemoryVectorIndex(embedding=embedding), embedding


def _memory(
    store: SQLiteMemoryStore,
    index: VectorIndex,
    embedding: HashedEmbeddingProvider,
    *,
    conversation: str = "c1",
) -> MemorySystem:
    return MemorySystem(
        llm=NoopLLM(),
        namespace=Namespace(user_id="user", conversation_id=conversation),
        timezone="Asia/Shanghai",
        config=MemoryConfig(),
        store=store,
        index=index,
        embedding=embedding,
    )


def _complete_message_turn(
    mem: MemorySystem,
    *,
    turn_id: str,
    source_id: str,
    text: str,
    timestamp: int,
    policy: RetrievalPolicy = RetrievalPolicy.AUTO,
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
                retrieval_policy=policy,
                timestamp=timestamp,
            )
        ],
        annotation_target_ids=[source_id],
        turn_id=turn_id,
        opened_at=timestamp,
    )
    mem.complete_turn(
        turn_id=turn_id,
        semantic_text="收到。",
        provider_output_raw='{"speech":"收到。"}',
        memory_annotation={
            "entity_anchors": ["可乐"],
            "memory_facets": ["preference"],
            "about_roles": ["user"],
            "retrieval_priority": "high",
        },
        annotation_status="accepted",
        timestamp=timestamp + 1,
        source_id=f"{source_id}-final",
    )


class KindMetadataTests(unittest.TestCase):
    def test_retrieval_score_thresholds_reject_negative_and_non_finite_values(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(retrieval_min_dense_score=-0.1)
        with self.assertRaises(ConfigError):
            MemoryConfig(retrieval_min_bm25_score=float("inf"))

    def test_open_kind_prefixes_and_filter_keys_are_stable(self) -> None:
        self.assertEqual(
            kind_prefixes("event.finance.flash"),
            ("event", "event.finance", "event.finance.flash"),
        )
        key = kind_filter_key("event.finance")
        self.assertEqual(key, kind_filter_key("event.finance"))
        self.assertRegex(key, r"^memory_kind__v1_[a-f0-9]{64}$")

    def test_hyphenated_kind_uses_the_shared_grammar_for_index_flags(self) -> None:
        kind = "tool.github-mcp.get-issue.result"
        self.assertEqual(
            kind_prefixes(kind),
            ("tool", "tool.github-mcp", "tool.github-mcp.get-issue", kind),
        )
        key = kind_filter_key("tool.github-mcp.get-issue")
        self.assertRegex(key, r"^memory_kind__v1_[a-f0-9]{64}$")
        self.assertTrue(kind_filter_flags(kind)[key])

    def test_index_entry_contains_visibility_kind_and_generation_scalars(self) -> None:
        entry = build_raw_entry(
            {
                "source_id": "entry-1",
                "tenant_id": "",
                "user_id": "user",
                "domain_id": "",
                "conversation_id": "c1",
                "kind": "event.finance.flash",
                "semantic_text": "新闻",
                "annotation_status": "accepted_model",
                "retrieval_visibility": "default",
                "trust": "untrusted_data",
            }
        )
        metadata = entry["metadata"]
        self.assertEqual(metadata["kind_exact"], "event.finance.flash")
        self.assertTrue(metadata[kind_filter_key("event.finance")])
        self.assertEqual(metadata["retrieval_visibility"], "default")
        self.assertEqual(metadata["annotation_status"], "accepted_model")
        self.assertEqual(metadata["index_schema_version"], INDEX_SCHEMA_VERSION)
        self.assertEqual(metadata["index_schema_key"], INDEX_SCHEMA_KEY)


class StructuredAdmissionTests(unittest.TestCase):
    def test_missing_metadata_does_not_hide_an_ordinary_completed_turn(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="missing-user",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="周四把蓝色账本交给林舟",
                        payload={"text": "周四把蓝色账本交给林舟"},
                        timestamp=1800,
                    )
                ],
                turn_id="missing-turn",
            )
            completed = mem.complete_turn(
                turn_id="missing-turn",
                semantic_text="好，我记下了。",
                provider_output_raw="好，我记下了。",
                memory_annotation=None,
                annotation_status="missing",
                timestamp=1801,
                source_id="missing-final",
            )

            self.assertEqual(completed.updated_targets[0].retrieval_visibility.value, "default")
            result = mem.retrieve_structured("蓝色账本 林舟")
            self.assertEqual(result.status, "found")
            self.assertTrue(any(match.source_id == "missing-user" for match in result.matches))
        finally:
            mem.close()
            store.close()

    def test_accepted_v2_message_returns_structured_match_and_persists_index_generation(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            _complete_message_turn(
                mem,
                turn_id="turn-message",
                source_id="message-user",
                text="我喜欢无糖可乐",
                timestamp=1000,
            )
            result = mem.retrieve_structured("无糖可乐", entity_anchors=["无糖可乐"])
            self.assertEqual(result.status, "found")
            self.assertTrue(any(match.kind == "message.user" for match in result.matches))
            self.assertTrue(any("无糖可乐" in match.rendered_text for match in result.matches))
            stored = store.get_entry(namespace=mem.namespace, source_id="message-user")
            self.assertIsNotNone(stored)
            record = store.get_retrieval_record(namespace=mem.namespace, source_id="message-user")
            self.assertEqual(record["index_schema_version"], INDEX_SCHEMA_VERSION)
            self.assertEqual(record["index_key"], INDEX_SCHEMA_KEY)
        finally:
            mem.close()
            store.close()

    def test_explicit_tool_requires_kind_pattern_and_does_not_open_other_tools(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="tool-user",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="查天气和计算",
                        timestamp=2000,
                    )
                ],
                turn_id="tool-turn",
            )
            for correlation, kind, text in (
                ("weather", "tool.web_search", "北京晴天 25 度"),
                ("calc", "tool.calculator", "结果是 42"),
            ):
                mem.append_entry(
                    TimelineEntryInput(
                        source_id=f"{correlation}-call",
                        kind=f"{kind}.call",
                        origin=EntryOrigin.ASSISTANT,
                        turn_role=TurnRole.ACTION,
                        semantic_text=f"call {correlation}",
                        correlation_id=correlation,
                        timestamp=2001,
                    ),
                    turn_id="tool-turn",
                )
                mem.append_entry(
                    TimelineEntryInput(
                        source_id=f"{correlation}-result",
                        kind=f"{kind}.result",
                        origin=EntryOrigin.ENVIRONMENT,
                        turn_role=TurnRole.OBSERVATION,
                        semantic_text=text,
                        correlation_id=correlation,
                        trace_metadata={"status": "success"},
                        timestamp=2002,
                    ),
                    turn_id="tool-turn",
                )

            self.assertEqual(mem.retrieve_structured("北京晴天", entity_anchors=["北京晴天"]).status, "empty")
            invalid = mem.retrieve_structured("北京晴天", include_explicit=True)
            self.assertEqual(invalid.status, "invalid")
            self.assertEqual(invalid.reason, "explicit_kind_patterns_required")

            result = mem.retrieve_structured(
                "北京晴天",
                entity_anchors=["北京晴天"],
                include_explicit=True,
                kind_patterns=["tool.web_search.*"],
            )
            self.assertEqual(result.status, "found")
            self.assertTrue(all(match.kind.startswith("tool.web_search.") for match in result.matches))
            self.assertFalse(any("42" in match.semantic_text for match in result.matches))
        finally:
            mem.close()
            store.close()

    def test_protocol_neutral_action_is_still_a_trace_not_ordinary_memory(self) -> None:
        entry = build_raw_entry(
            {
                "source_id": "operation-action",
                "tenant_id": "",
                "user_id": "user",
                "domain_id": "",
                "conversation_id": "c1",
                "kind": "operation.catalog.request",
                "turn_role": "action",
                "semantic_text": "load catalog",
                "retrieval_visibility": "default",
            }
        )
        self.assertTrue(entry["metadata"]["is_trace_kind"])

    def test_annotated_event_uses_the_same_default_admission_as_a_message(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="finance-event",
                        kind="event.finance.flash",
                        origin=EntryOrigin.ENVIRONMENT,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="虚构科创债政策事件",
                        timestamp=2500,
                    )
                ],
                turn_id="event-turn",
            )
            mem.complete_turn(
                turn_id="event-turn",
                semantic_text="这条事件值得后续观察。",
                provider_output_raw='{"speech":"这条事件值得后续观察。"}',
                memory_annotation={
                    "entity_anchors": ["科创债"],
                    "memory_facets": ["event"],
                    "about_roles": ["external"],
                },
                annotation_status="accepted",
                timestamp=2501,
                source_id="finance-event-final",
            )
            result = mem.retrieve_structured("科创债", entity_anchors=["科创债"])
            self.assertEqual(result.status, "found")
            self.assertTrue(any(match.kind == "event.finance.flash" for match in result.matches))
        finally:
            mem.close()
            store.close()

    def test_cross_conversation_scope_is_hard_and_host_selected(self) -> None:
        store, index, embedding = _backends()
        current = _memory(store, index, embedding, conversation="c1")
        other = _memory(store, index, embedding, conversation="c2")
        try:
            _complete_message_turn(
                other,
                turn_id="other-turn",
                source_id="other-user",
                text="我喜欢无糖可乐",
                timestamp=3000,
            )
            local = current.retrieve_structured("无糖可乐", entity_anchors=["无糖可乐"])
            cross = current.retrieve_structured(
                "无糖可乐",
                entity_anchors=["无糖可乐"],
                cross_conversation=True,
            )
            self.assertEqual(local.status, "empty")
            self.assertEqual(cross.status, "found")
            self.assertTrue(all(match.source_id.startswith("other-") for match in cross.matches))
        finally:
            current.close()
            other.close()
            store.close()

    def test_accepted_legacy_is_not_admitted_by_v2(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            record = store.add_message(
                namespace=mem.namespace,
                role="user",
                content="旧兼容可乐",
                timestamp=4000,
                source_id="legacy-user",
                kind="message.user",
                semantic_text="旧兼容可乐",
                annotation_status="accepted_legacy",
                retrieval_visibility="default",
            )
            index.upsert([build_raw_entry(record)])
            store.set_index_state(
                "legacy-user",
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )
            result = mem.retrieve_structured("旧兼容可乐", entity_anchors=["旧兼容可乐"])
            self.assertEqual(result.status, "empty")
        finally:
            mem.close()
            store.close()

    def test_never_policy_stays_out_even_after_a_valid_annotation(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            _complete_message_turn(
                mem,
                turn_id="never-turn",
                source_id="never-user",
                text="不可检索秘密",
                timestamp=4500,
                policy=RetrievalPolicy.NEVER,
            )
            result = mem.retrieve_structured("不可检索秘密", entity_anchors=["不可检索秘密"])
            self.assertEqual(result.status, "empty")
            stored = store.get_retrieval_record(namespace=mem.namespace, source_id="never-user")
            self.assertEqual(stored["retrieval_visibility"], "never")
            self.assertEqual(stored["index_status"], "skipped")
        finally:
            mem.close()
            store.close()

    def test_invalid_mid_glob_is_structured_invalid_not_a_broad_search(self) -> None:
        store, index, embedding = _backends()
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured(
                "天气",
                include_explicit=True,
                kind_patterns=["tool.*.result"],
            )
            self.assertEqual(result.status, "invalid")
            self.assertEqual(result.reason, "invalid_kind_pattern")
        finally:
            mem.close()
            store.close()

    def test_backend_that_ignores_hard_namespace_filter_is_rejected_as_unavailable(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        other_namespace = Namespace(user_id="other-user", conversation_id="c1")
        record = store.add_message(
            namespace=other_namespace,
            role="user",
            content="其他用户私密内容",
            timestamp=4600,
            source_id="other-private",
            kind="message.user",
            semantic_text="其他用户私密内容",
            annotation_status="accepted_host",
            retrieval_visibility="default",
        )
        store.set_index_state(
            "other-private",
            "indexed",
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_key=INDEX_SCHEMA_KEY,
        )
        index = FixedHitIndex(record["source_id"])
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured("私密内容")
            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.reason, "index_filter_unsupported")
            self.assertEqual(result.matches, ())
        finally:
            mem.close()
            store.close()

    def test_backend_that_ignores_other_hard_filters_is_rejected_as_unavailable(self) -> None:
        store, normal_index, embedding = _backends()
        writer = _memory(store, normal_index, embedding)
        try:
            _complete_message_turn(
                writer,
                turn_id="hard-filter-turn",
                source_id="hard-filter-user",
                text="只用于硬过滤测试",
                timestamp=4700,
            )
        finally:
            writer.close()

        for filters in (
            {"source_layers": ["summary"]},
            {"exclude_source_ids": ["hard-filter-user"]},
        ):
            with self.subTest(filters=filters):
                mem = _memory(store, FixedHitIndex("hard-filter-user"), embedding)
                try:
                    result = mem.retrieve_structured("硬过滤测试", **filters)
                    self.assertEqual(result.status, "unavailable")
                    self.assertEqual(result.reason, "index_filter_unsupported")
                    self.assertEqual(result.matches, ())
                finally:
                    mem.close()
        store.set_index_state("hard-filter-user", "indexed", index_schema_version=1, index_key="stale")
        mem = _memory(store, FixedHitIndex("hard-filter-user"), embedding)
        try:
            result = mem.retrieve_structured("硬过滤测试")
            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.reason, "index_filter_unsupported")
        finally:
            mem.close()
            store.close()

    def test_zero_similarity_candidate_is_not_returned_to_fill_top_k(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        index = ZeroScoreIndex()
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured("无关问题")
            self.assertEqual(result.status, "empty")
            self.assertGreater(result.rejected_counts["below_dense_score"], 0)
        finally:
            mem.close()
            store.close()

    def test_non_finite_similarity_candidate_is_rejected(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        index = NonFiniteScoreIndex()
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured("无关问题")
            self.assertEqual(result.status, "empty")
            self.assertGreater(result.rejected_counts["below_dense_score"], 0)
        finally:
            mem.close()
            store.close()


class HardFilterPlanningTests(unittest.TestCase):
    def test_facets_and_roles_are_never_silently_relaxed(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        index = WhereSpyIndex()
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured(
                "金融风险",
                memory_facets=["preference"],
                about_roles=["user"],
                include_explicit=True,
                kind_patterns=["event.finance.*"],
            )
            self.assertEqual(result.status, "empty")
            self.assertEqual(result.relaxation_steps, ())
            self.assertEqual(len(index.semantic_wheres), 2)
            for where in index.semantic_wheres:
                self.assertEqual(where["conversation_id"], "c1")
                self.assertEqual(where["index_schema_version"], INDEX_SCHEMA_VERSION)
                self.assertEqual(where["index_schema_key"], INDEX_SCHEMA_KEY)
                self.assertIn({kind_filter_key("event.finance"): True}, where["$and"])
                self.assertTrue(
                    any(
                        "$or" in clause and any(child.get("$and") for child in clause["$or"] if isinstance(child, dict))
                        for clause in where["$and"]
                    )
                )
        finally:
            mem.close()
            store.close()

    def test_exact_time_hint_is_in_both_index_queries_before_scoring(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        index = WhereSpyIndex()
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured(
                "misaka 和谁同行",
                entity_anchors=["misaka"],
                topic_terms=["同行"],
                time_hint={
                    "start_at": "2026-08-03 11:00",
                    "end_at": "2026-08-03 12:00",
                },
            )

            self.assertEqual(result.status, "empty")
            start_ts = int(datetime(2026, 8, 3, 11, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
            end_ts = int(datetime(2026, 8, 3, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
            expected = {"$gte": start_ts, "$lt": end_ts}
            self.assertTrue(index.semantic_wheres)
            self.assertEqual(index.semantic_wheres, index.keyword_wheres)
            self.assertTrue(all(where["timestamp"] == expected for where in index.semantic_wheres))
        finally:
            mem.close()
            store.close()

    def test_invalid_exact_time_hint_never_degrades_to_an_unfiltered_search(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        index = WhereSpyIndex()
        mem = _memory(store, index, embedding)
        try:
            result = mem.retrieve_structured(
                "misaka 和谁同行",
                time_hint={
                    "start_at": "2026-08-03 12:00",
                    "end_at": "2026-08-03 11:00",
                },
            )

            self.assertEqual(result.status, "invalid")
            self.assertEqual(result.reason, "time_range_start_must_be_before_end")
            self.assertEqual(index.semantic_wheres, [])
            self.assertEqual(index.keyword_wheres, [])
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
