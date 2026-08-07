"""Slice 2: 确定性 settlement(文档 §7/§12/§14.3)。

覆盖: compact_after_terminal 建立 settled / settled_noop / full_fallback,
通用 renderer 的 inline_full/compact_reloadable 分类与 no-expansion,
原 request projection 不可变, 以及串行/并行/pending/retry/restart。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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
    ProjectionMessageInput,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
)
from memcore.settlement import (
    COMPACT_RELOAD_MIN_INLINE_BYTES,
    COMPACT_RELOADABLE,
    INLINE_FULL,
    classify_observation,
    stable_settlement_id,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _shared_mem(
    policy: str = "compact_after_terminal",
    conversation: str = "c1",
    store: SQLiteMemoryStore | None = None,
):
    emb = HashedEmbeddingProvider()
    resolved_store = store or SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    mem = MemorySystem(
        llm=_NoopLLM(),
        namespace=Namespace(user_id="u1", conversation_id=conversation),
        timezone="Asia/Shanghai",
        store=resolved_store,
        index=index,
        embedding=emb,
        config=MemoryConfig(operation_projection_policy=policy),
    )
    return mem, resolved_store


def _turn_with_tools(
    mem: MemorySystem,
    *,
    turn_id: str = "t1",
    results: list[tuple[str, str, str]],
) -> list[str]:
    mem.begin_turn(
        turn_id=turn_id,
        opened_at=1000,
        stimuli=[
            TimelineEntryInput(
                source_id=f"{turn_id}-u",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text="查一下",
                payload={"text": "查一下"},
                timestamp=1000,
                compatibility_role="user",
            )
        ],
    )
    ids: list[str] = []
    for index, (tool, call_id, result) in enumerate(results):
        exchange = mem.record_tool_exchange(
            turn_id=turn_id,
            tool_name=tool,
            tool_call_id=call_id,
            tool_input={"query": f"q{index}"},
            result=result,
            source="web",
            source_id_prefix=f"tooltrace:{turn_id}-{index}",
        )
        ids.append(exchange["tool_result"]["source_id"])
    return ids


def _complete(mem: MemorySystem, turn_id: str, *, profile: str = "openai_chat", text: str = "ok"):
    """生产链路中 turn 投影在工具轮内已冻结, 这里先 build_context_projection 再 complete。"""
    mem.build_context_projection(provider_profile=profile)
    return mem.complete_turn(
        turn_id=turn_id,
        semantic_text=text,
        provider_output_raw=text,
        provider_profile=profile,
        provider_projection={"role": "assistant", "content": text},
    )


def _settlement_row(mem: MemorySystem, turn_id: str, profile: str = "openai_chat"):
    return mem.store.get_turn_projection_settlement(
        namespace=mem.namespace,
        turn_id=turn_id,
        provider_profile=profile,
    )


def _settled_payloads(mem: MemorySystem, turn_id: str, profile: str = "openai_chat") -> list[dict]:
    settlement_id = stable_settlement_id(turn_id=turn_id, provider_profile=profile)
    rows = mem.store.list_settled_projections(settlement_id=settlement_id)
    return [json.loads(row["payload_json"]) for row in rows]


LONG_BODY = "网页正文 " + "记忆引擎上下文管理策略说明内容填充。" * 200  # 远大于 256 bytes
SHORT_BODY = "音乐已暂停"


class SettledLifecycleTests(unittest.TestCase):
    def test_long_result_is_compacted_and_short_result_stays_inline(self) -> None:
        mem, store = _shared_mem()
        long_sid = _turn_with_tools(mem, results=[("web_search", "call_long", LONG_BODY)])[0]
        _complete(mem, "t1", text="查完了。")
        row = _settlement_row(mem, "t1")
        self.assertIsNotNone(row)
        self.assertEqual(row["settlement_status"], "settled")
        self.assertGreaterEqual(row["first_changed_projection_index"], 0)
        self.assertGreater(row["full_projected_tokens"], row["settled_projected_tokens"])
        payloads = _settled_payloads(mem, "t1")
        tool_payload = next(p for p in payloads if p.get("role") == "tool")
        self.assertIn("[compact_reloadable]", tool_payload["content"])
        self.assertIn("reload: open_memory", tool_payload["content"])
        self.assertIn(long_sid, tool_payload["content"])
        store.close()

    def test_short_results_produce_settled_noop_and_leave_history_unchanged(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("music_control", "call_a", SHORT_BODY), ("probe", "call_b", "已保存")])
        _complete(mem, "t1")
        row = _settlement_row(mem, "t1")
        self.assertIsNotNone(row)
        self.assertEqual(row["settlement_status"], "settled_noop")
        self.assertEqual(row["first_changed_projection_index"], -1)
        self.assertEqual(row["full_projection_hash"], row["settled_projection_hash"])
        payloads = _settled_payloads(mem, "t1")
        self.assertEqual(payloads, [])
        full_projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = [p for p in full_projection.payloads if p.get("role") == "tool"]
        self.assertNotIn("[compact_reloadable]", "".join(str(p.get("content")) for p in tool_payloads))
        self.assertTrue(any(SHORT_BODY in str(p.get("content")) for p in tool_payloads))
        store.close()

    def test_default_policy_never_creates_settlement(self) -> None:
        mem, store = _shared_mem(policy="full_until_raw_compaction")
        _turn_with_tools(mem, results=[("web_search", "call_long", LONG_BODY)])
        _complete(mem, "t1")
        self.assertIsNone(_settlement_row(mem, "t1"))
        store.close()

    def test_no_expansion_keeps_bodies_that_cannot_shrink(self) -> None:
        mem, store = _shared_mem()
        tiny = "x" * (COMPACT_RELOAD_MIN_INLINE_BYTES - 20)
        _turn_with_tools(mem, results=[("probe", "call_tiny", tiny)])
        _complete(mem, "t1")
        payloads = _settled_payloads(mem, "t1")
        self.assertEqual(payloads, [])
        full_projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payload = next(p for p in full_projection.payloads if p.get("role") == "tool")
        self.assertNotIn("[compact_reloadable]", tool_payload["content"])
        self.assertIn(tiny, tool_payload["content"])
        store.close()

    def test_open_turn_never_settles(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_long", LONG_BODY)])
        self.assertIsNone(_settlement_row(mem, "t1"))
        store.close()

    def test_anthropic_tool_result_block_is_compacted_and_use_id_kept(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_ant", LONG_BODY)])
        _complete(mem, "t1", profile="anthropic_messages")
        row = _settlement_row(mem, "t1", profile="anthropic_messages")
        self.assertIsNotNone(row)
        self.assertEqual(row["settlement_status"], "settled")
        payloads = _settled_payloads(mem, "t1", profile="anthropic_messages")
        user_payload = next(
            p
            for p in payloads
            if p.get("role") == "user"
            and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in p.get("content", []))
        )
        blocks = user_payload["content"]
        tool_result = next(b for b in blocks if b.get("type") == "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "call_ant")
        self.assertIn("[compact_reloadable]", tool_result["content"])


class SettledParallelPendingTests(unittest.TestCase):
    def test_parallel_tools_compact_all_long_results_without_reordering(self) -> None:
        mem, store = _shared_mem()
        bodies_a = LONG_BODY + "_AAA"
        bodies_b = LONG_BODY + "_BBB"
        _turn_with_tools(
            mem,
            results=[
                ("web_search", "call_pa", bodies_a),
                ("web_search", "call_pb", bodies_b),
            ],
        )
        _complete(mem, "t1")
        payloads = _settled_payloads(mem, "t1")
        tool_payloads = [p for p in payloads if p.get("role") == "tool"]
        self.assertEqual(len(tool_payloads), 2)
        for tool_payload in tool_payloads:
            self.assertIn("[compact_reloadable]", tool_payload["content"])
        self.assertNotIn(bodies_a, "".join(str(p.get("content")) for p in tool_payloads))
        self.assertNotIn(bodies_b, "".join(str(p.get("content")) for p in tool_payloads))
        store.close()

    def test_pending_correlation_blocks_completion_and_settlement(self) -> None:
        mem, store = _shared_mem()
        mem.begin_turn(
            turn_id="pending-t",
            opened_at=1000,
            stimuli=[
                TimelineEntryInput(
                    source_id="pu",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="hi",
                    payload={"text": "hi"},
                    timestamp=1000,
                    compatibility_role="user",
                )
            ],
        )
        mem.append_action(
            turn_id="pending-t",
            kind="tool.web_search.call",
            correlation_id="call_pending",
            semantic_text="tool call",
            payload={"input": {"query": "q"}},
            timestamp=1001,
        )
        result = mem.complete_turn(turn_id="pending-t", semantic_text="reply", provider_output_raw="reply")
        self.assertFalse(result.completed)
        self.assertIn("pending", str(result.status))
        self.assertIsNone(_settlement_row(mem, "pending-t"))
        store.close()


class SettledRetryRestartTests(unittest.TestCase):
    def test_retry_complete_is_idempotent_and_never_overwrites_settlement(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_retry", LONG_BODY)])
        first = _complete(mem, "t1")
        self.assertTrue(first.completed)
        row_before = _settlement_row(mem, "t1")
        payload_count_before = len(_settled_payloads(mem, "t1"))
        second = _complete(mem, "t1")
        self.assertTrue(second.completed or second.status == "already_completed")
        row_after = _settlement_row(mem, "t1")
        self.assertEqual(row_before["settled_projection_hash"], row_after["settled_projection_hash"])
        self.assertEqual(payload_count_before, len(_settled_payloads(mem, "t1")))
        store.close()

    def test_turn_uses_policy_frozen_at_begin_not_mutated_runtime_config(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        _turn_with_tools(mem, results=[("web_search", "call_frozen", LONG_BODY)])
        mem.config.operation_projection_policy = "full_until_raw_compaction"
        _complete(mem, "t1")
        self.assertEqual(_settlement_row(mem, "t1")["settlement_status"], "settled")
        store.close()

        mem, store = _shared_mem(policy="full_until_raw_compaction")
        _turn_with_tools(mem, results=[("web_search", "call_frozen_full", LONG_BODY)])
        mem.config.operation_projection_policy = "compact_after_terminal"
        _complete(mem, "t1")
        self.assertIsNone(_settlement_row(mem, "t1"))
        store.close()

    def test_settlement_preserves_actual_request_frozen_action_wire(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_wire", LONG_BODY)])
        generated = mem.build_context_projection(provider_profile="openai_chat")
        turn_messages: list[ProjectionMessageInput] = []
        for message in generated.messages:
            payload = dict(message.payload)
            if payload.get("role") == "assistant" and payload.get("tool_calls"):
                payload["provider_extension"] = {"cache_control": "ephemeral"}
            turn_messages.append(
                ProjectionMessageInput(
                    provider_profile="openai_chat",
                    payload=payload,
                    source_ids=message.source_ids,
                    projection_index=message.projection_index,
                )
            )
        mem.record_request_projection(
            turn_id="t1",
            provider_profile="openai_chat",
            turn_messages=turn_messages,
            history_messages=[dict(message.payload) for message in turn_messages],
            attempt=1,
        )
        _complete(mem, "t1")
        payloads = _settled_payloads(mem, "t1")
        action = next(payload for payload in payloads if payload.get("tool_calls"))
        self.assertEqual(action["provider_extension"], {"cache_control": "ephemeral"})
        self.assertIn("[compact_reloadable]", next(p for p in payloads if p.get("role") == "tool")["content"])
        store.close()

    def test_projection_write_failure_rolls_back_partial_rows_before_full_fallback(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_atomic", LONG_BODY)])
        store._conn.execute(
            """
            CREATE TRIGGER fail_settled_projection_insert
            BEFORE INSERT ON settled_prompt_projection
            BEGIN
                SELECT RAISE(ABORT, 'forced settlement projection failure');
            END
            """
        )
        result = _complete(mem, "t1")
        self.assertTrue(result.completed)
        row = _settlement_row(mem, "t1")
        self.assertEqual(row["settlement_status"], "full_fallback")
        self.assertEqual(_settled_payloads(mem, "t1"), [])
        self.assertIsNotNone(result.final_entry)
        store.close()

    def test_restart_reloads_same_frozen_settlement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.sqlite3"
            mem, store = _shared_mem(store=SQLiteMemoryStore(str(path)))
            _turn_with_tools(mem, results=[("web_search", "call_restart", LONG_BODY)])
            _complete(mem, "t1")
            expected_hash = _settlement_row(mem, "t1")["settled_projection_hash"]
            store.close()

            fresh, fresh_store = _shared_mem(store=SQLiteMemoryStore(str(path)))
            row = fresh_store.get_turn_projection_settlement(
                namespace=fresh.namespace, turn_id="t1", provider_profile="openai_chat"
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["settled_projection_hash"], expected_hash)
            self.assertEqual(row["settlement_status"], "settled")
            fresh_store.close()

    def test_generation_failure_falls_back_to_full_without_losing_final(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_fail", LONG_BODY)])
        with patch("memcore.settlement.build_settlement_plan", side_effect=RuntimeError("boom")):
            result = _complete(mem, "t1")
        self.assertTrue(result.completed)
        row = _settlement_row(mem, "t1")
        self.assertIsNotNone(row)
        self.assertEqual(row["settlement_status"], "full_fallback")
        self.assertIn("RuntimeError", row["reason"])
        store.close()


class ObservationKindClassificationTests(unittest.TestCase):
    def test_classifier_labels(self) -> None:
        mem, store = _shared_mem()
        _turn_with_tools(mem, results=[("web_search", "call_a", LONG_BODY), ("probe", "call_b", SHORT_BODY)])
        entries = store.get_turn_entries(namespace=mem.namespace, turn_id="t1")
        observations = [e for e in entries if e.turn_role is TurnRole.OBSERVATION]
        self.assertEqual(len(observations), 2)
        long_kind, long_content = classify_observation(observations[0], LONG_BODY)
        short_kind, short_content = classify_observation(observations[1], SHORT_BODY)
        self.assertEqual(long_kind, COMPACT_RELOADABLE)
        self.assertEqual(short_kind, INLINE_FULL)
        self.assertLess(len(long_content), len(LONG_BODY))
        self.assertEqual(short_content, SHORT_BODY)
        store.close()


if __name__ == "__main__":
    unittest.main()
