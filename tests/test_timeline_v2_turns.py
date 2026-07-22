"""Unified Timeline V2 turn lifecycle and atomic completion tests."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from memcore import (
    AnnotationStatus,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MAX_OPERATION_RETENTION_ANCHOR_BYTES,
    MemoryAnnotation,
    MemorySystem,
    Namespace,
    NamespaceError,
    RetrievalPolicy,
    RetrievalVisibility,
    SQLiteMemoryStore,
    SchemaError,
    TimelineEntryInput,
    TurnRole,
    TurnStatus,
    VectorIndex,
    build_action_entry,
    build_observation_entry,
)
from memcore.namespace import Actor


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


class _FailingIndex(VectorIndex):
    def upsert(self, entries: list[dict[str, Any]]) -> None:
        raise RuntimeError("index unavailable")

    def semantic_search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def keyword_search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, source_ids: list[str]) -> None:
        return None

    def count(self) -> int:
        return 0


def _stimulus(
    text: str,
    *,
    source_id: str = "",
    kind: str = "message.user",
    actor: Actor | None = None,
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.AUTO,
) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind=kind,
        origin=EntryOrigin.USER if kind == "message.user" else EntryOrigin.ENVIRONMENT,
        turn_role=TurnRole.STIMULUS,
        semantic_text=text,
        payload={"text": text},
        actor=actor,
        retrieval_policy=retrieval_policy,
    )


def _tool_action(call_id: str, *, source_id: str = "") -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="tool.web_search.call",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.ACTION,
        semantic_text=f"search call {call_id}",
        payload={"query": call_id},
        correlation_id=call_id,
        trace_metadata={"tool_name": "web_search", "status": "running"},
    )


def _tool_result(call_id: str, status: str, text: str, *, source_id: str = "") -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="tool.web_search.result",
        origin=EntryOrigin.ENVIRONMENT,
        turn_role=TurnRole.OBSERVATION,
        semantic_text=text,
        payload={"output": text},
        correlation_id=call_id,
        trace_metadata={"tool_name": "web_search", "status": status},
    )


class TurnLifecycleBase(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = Namespace(
            user_id="user",
            tenant_id="tenant",
            domain_id="domain",
            conversation_id="conversation",
        )
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.index = InMemoryVectorIndex(embedding=self.embedding)
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=self.namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.embedding,
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()


class TimelineContractsTests(unittest.TestCase):
    def test_open_namespaced_kind_is_allowed_without_business_enum(self) -> None:
        entry = TimelineEntryInput(
            kind="event.some_future_channel.signal",
            origin="environment",
            turn_role="stimulus",
            semantic_text="future event",
        )
        self.assertEqual(entry.kind, "event.some_future_channel.signal")
        self.assertEqual(entry.origin, EntryOrigin.ENVIRONMENT)

    def test_invalid_kind_and_unpaired_action_are_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "timeline_entry_invalid_kind"):
            _stimulus("bad", kind="single")
        with self.assertRaisesRegex(SchemaError, "timeline_entry_correlation_id_required"):
            TimelineEntryInput(
                kind="tool.web_search.call",
                origin="assistant",
                turn_role="action",
                semantic_text="missing correlation",
            )

    def test_generic_operation_builders_keep_protocol_open_and_bound_retention_anchor(self) -> None:
        action = build_action_entry(
            kind="operation.catalog.request",
            correlation_id="catalog-1",
            semantic_text='{"action":"list"}',
            payload={"protocol": "host_json", "request": {"action": "list"}},
            retention_anchor={"catalog_version": "v3", "schema_hash": "sha256:abc"},
        )
        observation = build_observation_entry(
            kind="operation.catalog.response",
            correlation_id="catalog-1",
            semantic_text="catalog returned",
            payload={"items": ["search", "weather"]},
            status="success",
        )

        self.assertEqual(action.turn_role, TurnRole.ACTION)
        self.assertEqual(action.payload["protocol"], "host_json")
        self.assertEqual(action.trace_metadata["retention_anchor"]["schema_hash"], "sha256:abc")
        self.assertEqual(observation.turn_role, TurnRole.OBSERVATION)
        self.assertEqual(observation.trace_metadata["status"], "success")
        with self.assertRaisesRegex(SchemaError, "operation_retention_anchor_too_large"):
            build_action_entry(
                kind="operation.catalog.request",
                correlation_id="too-large",
                retention_anchor={"value": "x" * (MAX_OPERATION_RETENTION_ANCHOR_BYTES + 1)},
            )


class BasicTurnCompletionTests(TurnLifecycleBase):
    def test_partial_v2_store_reports_structured_unsupported_error(self) -> None:
        with mock.patch.object(self.store, "get_turn", side_effect=NotImplementedError):
            with self.assertRaisesRegex(SchemaError, "store_timeline_v2_unsupported"):
                self.mem.begin_turn(
                    stimuli=[_stimulus("partial store", source_id="partial-store-stimulus")],
                    turn_id="partial-store-turn",
                )

    def test_begin_turn_is_idempotent_only_for_the_same_declared_sources(self) -> None:
        first = self.mem.begin_turn(
            stimuli=[_stimulus("same", source_id="stable-stimulus")],
            turn_id="stable-turn",
            opened_at=90,
        )
        repeated = self.mem.begin_turn(
            stimuli=[_stimulus("same", source_id="stable-stimulus")],
            turn_id="stable-turn",
            opened_at=91,
        )
        self.assertEqual(first.turn_id, repeated.turn_id)
        self.assertEqual([entry.source_id for entry in repeated.stimuli], ["stable-stimulus"])
        with self.assertRaisesRegex(SchemaError, "turn_idempotency_conflict"):
            self.mem.begin_turn(
                stimuli=[_stimulus("different", source_id="different-stimulus")],
                turn_id="stable-turn",
            )

    def test_begin_and_complete_atomically_materialize_target_and_final(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                _stimulus(
                    "我喜欢无糖可乐。",
                    source_id="stimulus-1",
                    actor=Actor(stable_id="qq:1", display_name="张三"),
                )
            ],
            turn_id="turn-1",
            opened_at=100,
        )
        self.assertEqual(handle.status, TurnStatus.OPEN)
        self.assertEqual(handle.annotation_target_ids, ("stimulus-1",))
        self.assertEqual(handle.stimuli[0].namespace.actor.stable_id, "qq:1")
        self.assertEqual(handle.stimuli[0].index_status, "indexed")

        result = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="记住了，你喜欢无糖可乐。",
            provider_output_raw='{"speech":"记住了，你喜欢无糖可乐。","memory_metadata":{}}',
            memory_annotation={
                "keywords": ["可乐", "无糖饮料"],
                "categories": ["preference", "tool_trace"],
                "subject_scopes": ["user"],
                "importance": 0.8,
                "confidence": 0.95,
            },
            annotation_status="accepted",
            timestamp=101,
            source_id="final-1",
        )

        self.assertTrue(result.completed)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.updated_targets[0].annotation_status, AnnotationStatus.ACCEPTED_MODEL)
        self.assertEqual(result.updated_targets[0].retrieval_visibility, RetrievalVisibility.DEFAULT)
        self.assertEqual(result.updated_targets[0].memory_metadata["keywords"], ["可乐", "无糖饮料"])
        self.assertEqual(result.updated_targets[0].memory_metadata["categories"], ["preference"])
        self.assertEqual(result.final_entry.annotation_status, AnnotationStatus.DERIVED_TURN_FINAL)
        self.assertEqual(result.final_entry.retrieval_visibility, RetrievalVisibility.DEFAULT)
        self.assertEqual(result.final_entry.reply_to_source_id, "stimulus-1")
        self.assertIn("provider_output_raw", result.final_entry.payload)
        self.assertEqual(result.final_entry.index_status, "indexed")
        self.assertEqual(self.store.get_turn(namespace=self.namespace, turn_id="turn-1").status, TurnStatus.CLOSED)

        repeated = self.mem.complete_turn(
            turn_id="turn-1",
            semantic_text="a retry must not append another final",
            provider_output_raw="retry",
            memory_annotation={},
            annotation_status="missing",
            timestamp=102,
            source_id="different-final-id",
        )
        self.assertEqual(repeated.status, "already_completed")
        self.assertEqual(repeated.final_entry.source_id, "final-1")
        self.assertEqual(
            [entry.source_id for entry in self.store.get_turn_entries(namespace=self.namespace, turn_id="turn-1")],
            ["stimulus-1", "final-1"],
        )

    def test_missing_annotation_stays_explicit_instead_of_becoming_empty_accepted(self) -> None:
        handle = self.mem.begin_turn(stimuli=[_stimulus("普通输入", source_id="s1")], turn_id="turn-missing")
        result = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="普通回复",
            provider_output_raw="普通回复",
            memory_annotation=None,
            annotation_status="missing",
            timestamp=200,
        )
        self.assertEqual(result.updated_targets[0].annotation_status, AnnotationStatus.MISSING)
        self.assertEqual(result.updated_targets[0].retrieval_visibility, RetrievalVisibility.EXPLICIT)
        self.assertEqual(result.final_entry.annotation_status, AnnotationStatus.UNANNOTATED)
        self.assertEqual(result.final_entry.retrieval_visibility, RetrievalVisibility.EXPLICIT)

    def test_valid_empty_annotation_remains_accepted_and_differs_from_missing(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("没有关键词也可以是有效标注", source_id="empty-accepted")],
            turn_id="turn-empty-accepted",
        )
        result = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="收到。",
            provider_output_raw="raw",
            memory_annotation={},
            annotation_status="accepted",
        )
        self.assertEqual(result.updated_targets[0].annotation_status, AnnotationStatus.ACCEPTED_MODEL)
        self.assertEqual(result.updated_targets[0].memory_metadata["keywords"], [])
        self.assertEqual(result.updated_targets[0].retrieval_visibility, RetrievalVisibility.DEFAULT)

    def test_unannotated_event_keeps_v1_explicit_trace_compatibility(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("张三戳了戳助手", source_id="poke", kind="event.qq.poke")],
            turn_id="turn-event-missing",
        )
        before = self.store.get_entry(namespace=self.namespace, source_id="poke")
        self.assertIn("event_trace", before.memory_metadata["categories"])
        result = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="我感觉到了。",
            provider_output_raw="raw",
            annotation_status="missing",
        )
        self.assertEqual(result.updated_targets[0].retrieval_visibility, RetrievalVisibility.EXPLICIT)
        self.assertIn("event_trace", result.updated_targets[0].memory_metadata["categories"])

    def test_policy_always_and_never_override_auto_admission(self) -> None:
        always = self.mem.begin_turn(
            stimuli=[_stimulus("always", source_id="always", retrieval_policy=RetrievalPolicy.ALWAYS)],
            turn_id="turn-always",
        )
        always_result = self.mem.complete_turn(
            turn_id=always.turn_id,
            semantic_text="reply",
            provider_output_raw="reply",
            memory_annotation=None,
            annotation_status="missing",
        )
        self.assertEqual(always_result.updated_targets[0].retrieval_visibility, RetrievalVisibility.DEFAULT)
        self.assertEqual(always_result.final_entry.retrieval_visibility, RetrievalVisibility.DEFAULT)

        never = self.mem.begin_turn(
            stimuli=[_stimulus("never", source_id="never", retrieval_policy=RetrievalPolicy.NEVER)],
            turn_id="turn-never",
        )
        never_result = self.mem.complete_turn(
            turn_id=never.turn_id,
            semantic_text="reply",
            provider_output_raw="reply",
            memory_annotation={"keywords": ["never"]},
            annotation_status="accepted",
        )
        self.assertEqual(never_result.updated_targets[0].retrieval_visibility, RetrievalVisibility.NEVER)
        self.assertEqual(never_result.updated_targets[0].index_status, "skipped")
        self.assertEqual(never_result.final_entry.retrieval_visibility, RetrievalVisibility.NEVER)
        self.assertEqual(never_result.final_entry.index_status, "skipped")
        indexed_ids = {
            item["source_id"]
            for item in self.index.semantic_search(
                query_text="never",
                where={
                    "tenant_id": "tenant",
                    "user_id": "user",
                    "domain_id": "domain",
                },
                n_results=20,
            )
        }
        self.assertNotIn("never", indexed_ids)
        self.assertNotIn(never_result.final_entry.source_id, indexed_ids)


class ParallelToolTurnTests(TurnLifecycleBase):
    def test_memory_system_helpers_append_generic_non_tool_operations(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("加载目录", source_id="generic-question")],
            turn_id="generic-operation-turn",
        )
        action = self.mem.append_action(
            turn_id=handle.turn_id,
            kind="operation.catalog.request",
            correlation_id="catalog",
            semantic_text='{"action":"list"}',
            payload={"action": "list"},
            source_id="generic-action",
        )
        observation = self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="operation.catalog.response",
            correlation_id="catalog",
            semantic_text="search, weather",
            payload={"items": ["search", "weather"]},
            status="success",
            source_id="generic-observation",
        )

        self.assertEqual(action.kind, "operation.catalog.request")
        self.assertEqual(observation.kind, "operation.catalog.response")
        branch = self.store.get_correlation_entries(
            namespace=self.namespace,
            turn_id=handle.turn_id,
            correlation_id="catalog",
        )
        self.assertEqual([entry.source_id for entry in branch], ["generic-action", "generic-observation"])

    def test_out_of_order_parallel_results_remain_correlated_and_block_early_final(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("查两个方向", source_id="question")],
            turn_id="turn-tools",
            opened_at=300,
        )
        self.mem.append_entry(_tool_action("call-a", source_id="action-a"), turn_id=handle.turn_id)
        self.mem.append_entry(_tool_action("call-b", source_id="action-b"), turn_id=handle.turn_id)
        self.mem.append_entry(_tool_result("call-b", "success", "B done", source_id="result-b"), turn_id=handle.turn_id)
        self.mem.append_entry(
            _tool_result("call-a", "running", "A progress", source_id="progress-a"), turn_id=handle.turn_id
        )

        pending = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="too early",
            provider_output_raw="too early",
            memory_annotation={"keywords": ["两个方向"]},
            annotation_status="accepted",
            timestamp=310,
            source_id="must-not-exist",
        )
        self.assertEqual(pending.status, "pending_actions")
        self.assertEqual(pending.pending_correlations, ("call-a",))
        self.assertIsNone(self.store.get_entry(namespace=self.namespace, source_id="must-not-exist"))
        target_before = self.store.get_entry(namespace=self.namespace, source_id="question")
        self.assertEqual(target_before.annotation_status, AnnotationStatus.UNANNOTATED)
        self.assertEqual(self.store.get_turn(namespace=self.namespace, turn_id=handle.turn_id).status, TurnStatus.OPEN)

        self.mem.append_entry(_tool_result("call-a", "success", "A done", source_id="result-a"), turn_id=handle.turn_id)
        completed = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="两个方向都查完了。",
            provider_output_raw="final raw",
            memory_annotation={"keywords": ["两个方向"]},
            annotation_status="accepted",
            timestamp=312,
            source_id="tools-final",
        )
        self.assertEqual(completed.status, "completed")
        branch_a = self.store.get_correlation_entries(
            namespace=self.namespace, turn_id=handle.turn_id, correlation_id="call-a"
        )
        branch_b = self.store.get_correlation_entries(
            namespace=self.namespace, turn_id=handle.turn_id, correlation_id="call-b"
        )
        self.assertEqual([entry.source_id for entry in branch_a], ["action-a", "progress-a", "result-a"])
        self.assertEqual([entry.source_id for entry in branch_b], ["action-b", "result-b"])
        self.assertTrue(all(entry.retrieval_visibility is RetrievalVisibility.EXPLICIT for entry in branch_a))
        self.assertTrue(all("tool_trace" in entry.memory_metadata["categories"] for entry in branch_a))

    def test_duplicate_action_correlation_is_rejected(self) -> None:
        handle = self.mem.begin_turn(stimuli=[_stimulus("q")], turn_id="turn-duplicate")
        self.mem.append_entry(_tool_action("same"), turn_id=handle.turn_id)
        with self.assertRaisesRegex(SchemaError, "turn_duplicate_action_correlation"):
            self.mem.append_entry(_tool_action("same"), turn_id=handle.turn_id)

    def test_observation_must_reference_an_existing_action(self) -> None:
        handle = self.mem.begin_turn(stimuli=[_stimulus("q")], turn_id="turn-orphan-result")
        with self.assertRaisesRegex(SchemaError, "turn_observation_action_not_found"):
            self.mem.append_entry(
                _tool_result("missing-action", "success", "orphan result"),
                turn_id=handle.turn_id,
            )


class MultiTargetAndRollbackTests(TurnLifecycleBase):
    def test_multiple_targets_require_complete_explicit_annotation_mapping(self) -> None:
        stimuli = [
            _stimulus("张三喜欢可乐", source_id="u1", actor=Actor(stable_id="qq:1", display_name="张三")),
            _stimulus("李四喜欢咖啡", source_id="u2", actor=Actor(stable_id="qq:2", display_name="李四")),
        ]
        handle = self.mem.begin_turn(
            stimuli=stimuli,
            annotation_target_ids=["u1", "u2"],
            turn_id="turn-bundle",
        )
        invalid = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="收到",
            provider_output_raw="raw",
            annotations=[
                MemoryAnnotation(
                    target_source_id="u1",
                    status=AnnotationStatus.ACCEPTED_MODEL,
                    memory_metadata={"keywords": ["可乐"]},
                )
            ],
            annotation_status="accepted",
            timestamp=401,
        )
        self.assertEqual(invalid.status, "invalid")
        self.assertEqual(invalid.reason, "annotation_targets_mismatch")
        self.assertEqual(
            self.store.get_entry(namespace=self.namespace, source_id="u1").annotation_status,
            AnnotationStatus.UNANNOTATED,
        )

        completed = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="我分别记住了。",
            provider_output_raw="raw",
            annotations=[
                MemoryAnnotation(
                    target_source_id="u1",
                    status=AnnotationStatus.ACCEPTED_MODEL,
                    memory_metadata={"keywords": ["可乐"], "categories": ["preference"]},
                ),
                MemoryAnnotation(
                    target_source_id="u2",
                    status=AnnotationStatus.ACCEPTED_MODEL,
                    memory_metadata={"keywords": ["咖啡"], "categories": ["preference"]},
                ),
            ],
            annotation_status="accepted",
            timestamp=402,
            source_id="bundle-final",
        )
        targets = {entry.source_id: entry for entry in completed.updated_targets}
        self.assertEqual(targets["u1"].memory_metadata["keywords"], ["可乐"])
        self.assertEqual(targets["u2"].memory_metadata["keywords"], ["咖啡"])
        self.assertEqual(completed.final_entry.reply_to_source_id, "")
        repeated = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="retry",
            provider_output_raw="retry",
        )
        self.assertEqual(repeated.status, "already_completed")
        self.assertEqual(repeated.final_entry.source_id, "bundle-final")

    def test_final_source_conflict_rolls_back_annotation_and_close(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("原始问题", source_id="rollback-target")],
            turn_id="turn-rollback",
        )
        self.store.add_message(
            namespace=self.namespace,
            role="user",
            content="unrelated",
            timestamp=500,
            source_id="occupied-final-id",
        )
        conflict = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="should roll back",
            provider_output_raw="raw",
            memory_annotation={"keywords": ["污染"]},
            annotation_status="accepted",
            timestamp=501,
            source_id="occupied-final-id",
        )
        self.assertEqual(conflict.status, "conflict")
        target = self.store.get_entry(namespace=self.namespace, source_id="rollback-target")
        self.assertEqual(target.annotation_status, AnnotationStatus.UNANNOTATED)
        self.assertEqual(target.memory_metadata, {})
        self.assertEqual(self.store.get_turn(namespace=self.namespace, turn_id=handle.turn_id).status, TurnStatus.OPEN)


class AbortIsolationAndOutboxTests(TurnLifecycleBase):
    def test_abort_is_terminal_idempotent_and_prevents_more_entries(self) -> None:
        handle = self.mem.begin_turn(stimuli=[_stimulus("q")], turn_id="turn-abort")
        self.mem.append_entry(_tool_action("pending"), turn_id=handle.turn_id)
        aborted = self.mem.abort_turn(handle.turn_id, reason="provider_failed", closed_at=600)
        self.assertEqual(aborted.status, "aborted")
        repeated = self.mem.abort_turn(handle.turn_id, reason="again", closed_at=601)
        self.assertEqual(repeated.status, "already_aborted")
        with self.assertRaisesRegex(SchemaError, "turn_not_open"):
            self.mem.append_entry(_tool_result("pending", "cancelled", "cancelled"), turn_id=handle.turn_id)
        conflict = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="late",
            provider_output_raw="late",
            memory_annotation={},
            annotation_status="missing",
        )
        self.assertEqual(conflict.status, "conflict")
        self.assertEqual(conflict.reason, "turn_aborted")

    def test_cross_namespace_cannot_read_append_complete_or_abort_turn(self) -> None:
        handle = self.mem.begin_turn(stimuli=[_stimulus("private")], turn_id="turn-private")
        other = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(
                user_id="other",
                tenant_id="tenant",
                domain_id="domain",
                conversation_id="conversation",
            ),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.embedding,
        )
        try:
            with self.assertRaises(NamespaceError):
                self.store.get_turn(namespace=other.namespace, turn_id=handle.turn_id)
            with self.assertRaises(NamespaceError):
                other.append_entry(_tool_action("x"), turn_id=handle.turn_id)
            with self.assertRaises(NamespaceError):
                other.complete_turn(
                    turn_id=handle.turn_id,
                    semantic_text="steal",
                    provider_output_raw="steal",
                    memory_annotation={},
                    annotation_status="missing",
                )
            with self.assertRaises(NamespaceError):
                other.abort_turn(handle.turn_id, reason="steal")
        finally:
            other.close()

    def test_index_failure_keeps_completed_sql_facts_pending(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="failure", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=store,
            index=_FailingIndex(),
            embedding=HashedEmbeddingProvider(),
        )
        try:
            handle = mem.begin_turn(stimuli=[_stimulus("must persist", source_id="persist-target")])
            result = mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text="persisted final",
                provider_output_raw="raw",
                memory_annotation={"keywords": ["persist"]},
                annotation_status="accepted",
                source_id="persist-final",
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.updated_targets[0].index_status, "pending")
            self.assertEqual(result.final_entry.index_status, "pending")
            self.assertEqual(store.get_turn(namespace=mem.namespace, turn_id=handle.turn_id).status, TurnStatus.CLOSED)
            self.assertEqual(
                store.get_entry(namespace=mem.namespace, source_id="persist-final").semantic_text, "persisted final"
            )
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
