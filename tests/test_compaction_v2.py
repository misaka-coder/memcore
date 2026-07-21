"""Closed-turn/token/atomic compaction and shared runtime acceptance tests."""

from __future__ import annotations

import threading
import unittest
from typing import Any

from memcore import (
    CANONICAL_PROFILE,
    CompactionSnapshot,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemCoreRuntime,
    MemoryConfig,
    MemorySystem,
    Namespace,
    OPENAI_PROFILE,
    ProjectionMessageInput,
    SQLiteMemoryStore,
    SchemaError,
    SummaryRecordInput,
    TimelineEntryInput,
    TokenCounter,
    TurnBundle,
    TurnRole,
    TurnStatus,
    canonical_json_bytes,
)
from memcore.llm.base import TaskType


class LengthTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(text)


class CountingLLM(LLMClient):
    def __init__(self) -> None:
        self.summary_calls = 0
        self.semantic_calls = 0
        self._lock = threading.Lock()

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            with self._lock:
                self.summary_calls += 1
            return LLMResult(
                ok=True,
                data={
                    "diary_summary": "完整轮次摘要",
                    "importance": 0.7,
                    "key_events": ["完成对话"],
                    "core_facts": ["输入与终态成对保留"],
                    "memory_metadata": {
                        "keywords": ["完整轮次"],
                        "categories": ["plan_goal"],
                        "subject_scopes": ["user"],
                    },
                },
                attempts=1,
            )
        if request.task_type == TaskType.SEMANTIC:
            with self._lock:
                self.semantic_calls += 1
            return LLMResult(
                ok=True,
                data={
                    "semantic_summary": "长期事实",
                    "importance": 0.8,
                    "stable_facts": ["事实"],
                    "recurring_topics": ["主题"],
                },
                attempts=1,
            )
        return LLMResult(ok=True, data={}, attempts=1)


class FailSecondSummaryStore(SQLiteMemoryStore):
    def __init__(self) -> None:
        super().__init__(":memory:")
        self.summary_writes = 0

    def _add_summary_locked(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        self.summary_writes += 1
        if self.summary_writes == 2:
            raise SchemaError("forced_summary_write_failure")
        return super()._add_summary_locked(namespace=namespace, record=record)


def _projected_config(**overrides: Any) -> MemoryConfig:
    values: dict[str, Any] = {
        "compaction_policy": "projected_tokens",
        "max_prompt_history_tokens": 260,
        "target_prompt_history_tokens": 140,
        "reserved_current_turn_tokens": 0,
        "reserved_retrieval_tokens": 0,
        "compaction_min_recent_turns": 1,
        "episodic_compact_trigger_count": 99,
        "projection_profile": OPENAI_PROFILE,
    }
    values.update(overrides)
    return MemoryConfig(**values)


def _count_config(**overrides: Any) -> MemoryConfig:
    values: dict[str, Any] = {
        "compaction_policy": "count_compat",
        "raw_trigger_count": 4,
        "summary_batch_size": 2,
        "compaction_min_recent_turns": 1,
        "episodic_compact_trigger_count": 99,
        "projection_profile": OPENAI_PROFILE,
    }
    values.update(overrides)
    return MemoryConfig(**values)


def _stimulus(
    text: str, *, source_id: str, timestamp: int, payload: dict[str, Any] | None = None
) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="message.user",
        origin=EntryOrigin.USER,
        turn_role=TurnRole.STIMULUS,
        semantic_text=text,
        payload=payload or {"text": text},
        timestamp=timestamp,
    )


def _complete_simple_turn(mem: MemorySystem, number: int, *, payload: dict[str, Any] | None = None) -> str:
    turn_id = f"turn-{number}"
    source_id = f"user-{number}"
    mem.begin_turn(
        stimuli=[
            _stimulus(
                f"问题{number}",
                source_id=source_id,
                timestamp=1000 + number * 10,
                payload=payload,
            )
        ],
        turn_id=turn_id,
        opened_at=1000 + number * 10,
    )
    mem.complete_turn(
        turn_id=turn_id,
        semantic_text=f"回答{number}",
        provider_output_raw=f'{{"speech":"回答{number}"}}',
        memory_annotation={
            "keywords": [f"问题{number}"],
            "categories": ["plan_goal"],
            "subject_scopes": ["user"],
        },
        annotation_status="accepted",
        timestamp=1001 + number * 10,
        source_id=f"final-{number}",
    )
    return turn_id


def _append_tool_turn(mem: MemorySystem, *, turn_id: str = "tool-turn") -> None:
    mem.begin_turn(
        stimuli=[_stimulus("查两个方向", source_id=f"{turn_id}-user", timestamp=2000)],
        turn_id=turn_id,
        opened_at=2000,
    )
    for call_id in ("a", "b"):
        mem.append_entry(
            TimelineEntryInput(
                source_id=f"{turn_id}-action-{call_id}",
                kind="tool.web_search.call",
                origin=EntryOrigin.ASSISTANT,
                turn_role=TurnRole.ACTION,
                semantic_text=f"call {call_id}",
                payload={"query": call_id},
                correlation_id=call_id,
                trace_metadata={"tool_name": "web_search", "status": "running"},
                timestamp=2001,
            ),
            turn_id=turn_id,
        )
    for call_id in ("b", "a"):
        mem.append_entry(
            TimelineEntryInput(
                source_id=f"{turn_id}-result-{call_id}",
                kind="tool.web_search.result",
                origin=EntryOrigin.ENVIRONMENT,
                turn_role=TurnRole.OBSERVATION,
                semantic_text=f"result {call_id}",
                payload={"output": call_id},
                correlation_id=call_id,
                trace_metadata={"tool_name": "web_search", "status": "success"},
                timestamp=2002,
            ),
            turn_id=turn_id,
        )
    mem.complete_turn(
        turn_id=turn_id,
        semantic_text="两个方向都完成",
        provider_output_raw='{"speech":"两个方向都完成"}',
        memory_annotation={"keywords": ["两个方向"], "categories": ["plan_goal"]},
        annotation_status="accepted",
        timestamp=2003,
        source_id=f"{turn_id}-final",
    )


def _snapshot_for_first_bundle(
    mem: MemorySystem,
    store: SQLiteMemoryStore,
) -> tuple[CompactionSnapshot, TurnBundle]:
    projection = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
    bundle = store.list_compaction_bundles(namespace=mem.namespace)[0]
    generation, _ = store.get_conversation_generations(namespace=mem.namespace)
    hashes = {item.source_id: item.payload_hashes for item in projection.entry_projection_hashes}
    return (
        CompactionSnapshot(
            namespace_key=("", "user", "", "conversation"),
            provider_profile=OPENAI_PROFILE,
            compaction_generation=generation,
            bundles=(bundle,),
            ordered_source_ids=bundle.source_ids,
            message_row_versions=tuple((entry.source_id, entry.row_version) for entry in bundle.entries),
            turn_row_versions=((bundle.turn_id, bundle.turn_row_version),),
            projection_hashes=tuple((source_id, hashes.get(source_id, ())) for source_id in bundle.source_ids),
            before_projected_tokens=100,
            selected_projected_tokens=100,
            token_count_quality="exact",
        ),
        bundle,
    )


class CompactionV2Base(unittest.TestCase):
    def make_mem(
        self,
        *,
        config: MemoryConfig,
        store: SQLiteMemoryStore | None = None,
        llm: CountingLLM | None = None,
        runtime: MemCoreRuntime | None = None,
        with_token_counter: bool = True,
    ) -> tuple[MemorySystem, SQLiteMemoryStore, CountingLLM]:
        resolved_store = store or SQLiteMemoryStore(":memory:")
        resolved_llm = llm or CountingLLM()
        embedding = HashedEmbeddingProvider()
        mem = MemorySystem(
            llm=resolved_llm,
            namespace=Namespace(user_id="user", conversation_id="conversation"),
            timezone="Asia/Shanghai",
            config=config,
            store=resolved_store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
            token_counter=LengthTokenCounter() if with_token_counter else None,
            runtime=runtime,
        )
        return mem, resolved_store, resolved_llm


class ClosedTurnPlanningTests(CompactionV2Base):
    def test_explicit_provider_profile_uses_actual_frozen_request_size(self) -> None:
        mem, store, _ = self.make_mem(
            config=_projected_config(
                max_prompt_history_tokens=4_000,
                target_prompt_history_tokens=2_000,
                projection_profile=CANONICAL_PROFILE,
            )
        )
        try:
            first = mem.begin_turn(
                stimuli=[_stimulus("简短问题", source_id="user-0", timestamp=1000)],
                turn_id="turn-0",
                opened_at=1000,
            )
            actual_user = {"role": "user", "content": "真实 provider 请求中的大块" + "x" * 12_000}
            mem.record_request_projection(
                turn_id=first.turn_id,
                provider_profile=OPENAI_PROFILE,
                turn_messages=[
                    ProjectionMessageInput(
                        provider_profile=OPENAI_PROFILE,
                        payload=actual_user,
                        source_ids=("user-0",),
                    )
                ],
                history_messages=[actual_user],
            )
            mem.complete_turn(
                turn_id=first.turn_id,
                semantic_text="简短回答",
                provider_output_raw='{"speech":"简短回答"}',
                memory_annotation={},
                annotation_status="accepted",
                timestamp=1001,
                source_id="final-0",
                provider_profile=OPENAI_PROFILE,
                provider_projection={"role": "assistant", "content": '{"speech":"简短回答"}'},
            )
            _complete_simple_turn(mem, 1)

            canonical_projection = mem.build_context_projection(provider_profile=CANONICAL_PROFILE)
            openai_projection = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            canonical_result = mem.compact_due_sync()

            self.assertEqual(canonical_result["status"], "not_due")
            self.assertEqual(canonical_result["provider_profile"], CANONICAL_PROFILE)
            self.assertGreater(
                len(canonical_json_bytes(openai_projection.payloads)),
                len(canonical_json_bytes(canonical_projection.payloads)) + 10_000,
            )

            provider_result = mem.compact_due_sync(provider_profile=OPENAI_PROFILE)
            self.assertEqual(provider_result["status"], "compacted")
            self.assertEqual(provider_result["provider_profile"], OPENAI_PROFILE)
            self.assertGreater(provider_result["before_projected_tokens"], 12_000)
            self.assertIn("user-0", provider_result["summary_source_ids"])
            self.assertIn("final-0", provider_result["summary_source_ids"])
        finally:
            mem.close()
            store.close()

    def test_projected_token_compaction_replaces_only_complete_old_turns(self) -> None:
        mem, store, _ = self.make_mem(config=_projected_config())
        try:
            for number in range(3):
                _complete_simple_turn(mem, number)
            before = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "compacted")
            self.assertGreater(result["before_projected_tokens"], result["after_projected_tokens"])
            self.assertEqual(result["token_count_quality"], "exact")
            self.assertGreaterEqual(result["source_turn_count"], 1)
            self.assertEqual(result["source_entry_count"] % 2, 0)

            summarized = set(result["summary_source_ids"])
            for number in range(3):
                pair = {f"user-{number}", f"final-{number}"}
                self.assertIn(len(pair & summarized), {0, 2})
            after = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            repeated = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            actual_after_tokens = sum(
                len(canonical_json_bytes(message.payload).decode("utf-8")) + 4 for message in after.messages
            )
            self.assertEqual(result["after_projected_tokens"], actual_after_tokens)
            self.assertEqual(result["compaction_generation"], after.compaction_generation)
            self.assertEqual(result["index_status"], "indexed")
            self.assertNotEqual(before.stable_prefix_hash, after.stable_prefix_hash)
            self.assertEqual(after.payloads, repeated.payloads)
            self.assertEqual(after.compaction_generation, 1)
            self.assertTrue(any(message.turn_id.startswith("summary.") for message in after.messages))
        finally:
            mem.close()
            store.close()

    def test_open_oldest_turn_blocks_compaction_without_splitting_it(self) -> None:
        mem, store, llm = self.make_mem(
            config=_projected_config(max_prompt_history_tokens=100, target_prompt_history_tokens=60)
        )
        try:
            mem.begin_turn(
                stimuli=[
                    _stimulus(
                        "open",
                        source_id="open-user",
                        timestamp=1000,
                        payload={"blob": "x" * 1000},
                    )
                ],
                turn_id="open-turn",
            )
            _complete_simple_turn(mem, 1)
            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "blocked_by_open_turn")
            self.assertEqual(result["reason"], "open_turn_in_prefix")
            self.assertEqual(llm.summary_calls, 0)
            self.assertEqual(store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10), [])
            self.assertEqual(len(store.get_unsummarized_messages(namespace=mem.namespace)), 3)
        finally:
            mem.close()
            store.close()

    def test_large_structured_payload_counts_even_when_semantic_text_is_short(self) -> None:
        mem, store, _ = self.make_mem(
            config=_projected_config(max_prompt_history_tokens=300, target_prompt_history_tokens=160)
        )
        try:
            _complete_simple_turn(mem, 0, payload={"blob": "z" * 1200})
            _complete_simple_turn(mem, 1)
            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "compacted")
            self.assertGreater(result["before_projected_tokens"], 1200)
            self.assertIn("user-0", result["summary_source_ids"])
            self.assertIn("final-0", result["summary_source_ids"])
        finally:
            mem.close()
            store.close()

    def test_projected_token_fallback_is_explicitly_marked_estimated(self) -> None:
        mem, store, _ = self.make_mem(
            config=_projected_config(max_prompt_history_tokens=200, target_prompt_history_tokens=100),
            with_token_counter=False,
        )
        try:
            _complete_simple_turn(mem, 0, payload={"blob": "z" * 1200})
            _complete_simple_turn(mem, 1)
            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "compacted")
            self.assertEqual(result["token_count_quality"], "estimated")
            self.assertGreater(result["before_projected_tokens"], result["after_projected_tokens"])
        finally:
            mem.close()
            store.close()

    def test_aborted_oldest_turn_compacts_without_fabricating_a_reply(self) -> None:
        mem, store, llm = self.make_mem(
            config=_projected_config(max_prompt_history_tokens=100, target_prompt_history_tokens=60)
        )
        try:
            mem.begin_turn(
                stimuli=[
                    _stimulus(
                        "aborted",
                        source_id="aborted-user",
                        timestamp=1000,
                        payload={"blob": "x" * 1000},
                    )
                ],
                turn_id="aborted-turn",
            )
            mem.abort_turn("aborted-turn", reason="test_abort", closed_at=1001)
            _complete_simple_turn(mem, 1)

            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "compacted")
            self.assertEqual(llm.summary_calls, 1)
            self.assertEqual(result["summary_source_ids"], ["aborted-user"])
            self.assertEqual(
                [item["source_id"] for item in store.get_unsummarized_messages(namespace=mem.namespace)],
                ["user-1", "final-1"],
            )
            turn = store.get_turn(namespace=mem.namespace, turn_id="aborted-turn")
            self.assertIsNotNone(turn)
            self.assertEqual(turn.status, TurnStatus.ABORTED)
            summaries = store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["source_ids"], ["aborted-user"])
        finally:
            mem.close()
            store.close()


class ToolPartitionAndAtomicityTests(CompactionV2Base):
    def test_parallel_tool_turn_commits_episode_and_operation_partitions_together(self) -> None:
        mem, store, _ = self.make_mem(config=_count_config(raw_trigger_count=6, summary_batch_size=4))
        try:
            _append_tool_turn(mem)
            _complete_simple_turn(mem, 9)
            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "compacted")
            summaries = store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
            by_kind = {item["kind"]: item for item in summaries}
            self.assertEqual(set(by_kind), {"memory.episode_summary", "memory.operation_digest"})
            episode_sources = set(by_kind["memory.episode_summary"]["source_ids"])
            operation_sources = set(by_kind["memory.operation_digest"]["source_ids"])
            self.assertFalse(episode_sources & operation_sources)
            self.assertIn("tool-turn-user", episode_sources)
            self.assertIn("tool-turn-final", episode_sources)
            self.assertEqual(
                operation_sources,
                {
                    "tool-turn-action-a",
                    "tool-turn-action-b",
                    "tool-turn-result-b",
                    "tool-turn-result-a",
                },
            )
            self.assertEqual(by_kind["memory.operation_digest"]["retrieval_visibility"], "explicit")
            self.assertEqual(by_kind["memory.operation_digest"]["semanticize"], 0)
            uncompacted = store.get_uncompacted_episodic_summaries(namespace=mem.namespace)
            self.assertTrue(all(item["kind"] != "memory.operation_digest" for item in uncompacted))
        finally:
            mem.close()
            store.close()

    def test_second_summary_write_failure_rolls_back_both_partitions_and_source_marks(self) -> None:
        store = FailSecondSummaryStore()
        mem, _, _ = self.make_mem(
            config=_count_config(raw_trigger_count=6, summary_batch_size=4),
            store=store,
        )
        try:
            _append_tool_turn(mem, turn_id="atomic-tool")
            _complete_simple_turn(mem, 8)
            result = mem.compact_due_sync()
            self.assertEqual(result["status"], "failed")
            self.assertIn("forced_summary_write_failure", result["reason"])
            self.assertEqual(store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10), [])
            self.assertTrue(
                all(not item["is_summarized"] for item in store.get_unsummarized_messages(namespace=mem.namespace))
            )
            self.assertEqual(store.get_conversation_generations(namespace=mem.namespace)[0], 0)
        finally:
            mem.close()
            store.close()

    def test_stale_snapshot_cannot_commit_a_second_summary(self) -> None:
        mem, store, _ = self.make_mem(config=_count_config())
        try:
            _complete_simple_turn(mem, 0)
            snapshot, bundle = _snapshot_for_first_bundle(mem, store)
            store.update_message_memory_metadata(
                namespace=mem.namespace,
                source_id="user-0",
                memory_metadata={"keywords": ["changed"]},
            )
            summary_id = "stable-summary-id"
            committed = store.commit_summary_batch(
                namespace=mem.namespace,
                snapshot=snapshot,
                records=[
                    SummaryRecordInput(
                        summary_id=summary_id,
                        summary_profile="test",
                        source_ids=bundle.source_ids,
                        record={"summary_id": summary_id, "diary_summary": "must not commit"},
                    )
                ],
            )
            self.assertEqual(committed.status, "stale_batch")
            self.assertEqual(store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10), [])
            self.assertTrue(
                all(not item["is_summarized"] for item in store.get_unsummarized_messages(namespace=mem.namespace))
            )
        finally:
            mem.close()
            store.close()

    def test_idempotent_retry_reports_missing_summary_row_as_stale(self) -> None:
        mem, store, _ = self.make_mem(config=_count_config())
        try:
            _complete_simple_turn(mem, 0)
            snapshot, bundle = _snapshot_for_first_bundle(mem, store)
            summary_id = "retry-summary-id"
            records = [
                SummaryRecordInput(
                    summary_id=summary_id,
                    summary_profile="test",
                    source_ids=bundle.source_ids,
                    record={"summary_id": summary_id, "diary_summary": "committed once"},
                )
            ]
            first = store.commit_summary_batch(
                namespace=mem.namespace,
                snapshot=snapshot,
                records=records,
            )
            self.assertEqual(first.status, "committed")
            with store._lock, store._conn:
                store._conn.execute("DELETE FROM summaries WHERE summary_id = ?", (summary_id,))

            retry = store.commit_summary_batch(
                namespace=mem.namespace,
                snapshot=snapshot,
                records=records,
            )
            self.assertEqual(retry.status, "stale_batch")
            self.assertEqual(retry.reason, "summary_missing")
            self.assertEqual(retry.summaries, ())
        finally:
            mem.close()
            store.close()


class SharedRuntimeTests(CompactionV2Base):
    def test_two_memory_systems_share_one_compaction_job_and_external_runtime_survives_close(self) -> None:
        runtime = MemCoreRuntime(compaction_workers=2)
        store = SQLiteMemoryStore(":memory:")
        llm = CountingLLM()
        config = _count_config()
        embedding = HashedEmbeddingProvider()
        index = InMemoryVectorIndex(embedding=embedding)
        namespace = Namespace(user_id="user", conversation_id="conversation")
        mem1 = MemorySystem(
            llm=llm,
            namespace=namespace,
            timezone="Asia/Shanghai",
            config=config,
            store=store,
            index=index,
            embedding=embedding,
            token_counter=LengthTokenCounter(),
            runtime=runtime,
        )
        mem2 = MemorySystem(
            llm=llm,
            namespace=namespace,
            timezone="Asia/Shanghai",
            config=config,
            store=store,
            index=index,
            embedding=embedding,
            token_counter=LengthTokenCounter(),
            runtime=runtime,
        )
        try:
            for number in range(3):
                _complete_simple_turn(mem1, number)
            first = mem1.compact_due_background()
            second = mem2.compact_due_background()
            results = [first.result(timeout=5), second.result(timeout=5)]
            self.assertEqual(llm.summary_calls, 1)
            self.assertIn("compacted", {item["status"] for item in results})
            self.assertTrue({item["status"] for item in results} & {"busy", "not_due"})
            mem1.close()
            mem2.close()
            self.assertEqual(runtime.submit_index_repair(lambda: 7).result(timeout=5), 7)
        finally:
            runtime.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
