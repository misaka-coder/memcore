"""Slice 3: 历史投影选择与回读闭环(文档 §8/§9/§10/§14.3/§14.5)。

覆盖: request builder 对 closed turn 选择 settled/full, settled_noop/full_fallback
保持 full, has_compact_history 能力标志, 前缀稳定与重启一致, 以及 open_memory
回读当轮全量、终局后再沉降的闭环。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
    TurnRole,
    dispatch_native_memory_tool,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _shared_mem(
    policy: str = "compact_after_terminal",
    store: SQLiteMemoryStore | None = None,
):
    emb = HashedEmbeddingProvider()
    resolved_store = store or SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    mem = MemorySystem(
        llm=_NoopLLM(),
        namespace=Namespace(user_id="u1", conversation_id="c1"),
        timezone="Asia/Shanghai",
        store=resolved_store,
        index=index,
        embedding=emb,
        config=MemoryConfig(operation_projection_policy=policy),
    )
    return mem, resolved_store


def _begin(mem: MemorySystem, turn_id: str, text: str = "问题") -> None:
    mem.begin_turn(
        turn_id=turn_id,
        opened_at=1000,
        stimuli=[
            TimelineEntryInput(
                source_id=f"{turn_id}-u",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text=text,
                payload={"text": text},
                timestamp=1000,
                compatibility_role="user",
            )
        ],
    )


def _add_tool(mem: MemorySystem, turn_id: str, call_id: str, result: str, tool: str = "web_search") -> str:
    exchange = mem.record_tool_exchange(
        turn_id=turn_id,
        tool_name=tool,
        tool_call_id=call_id,
        tool_input={"query": "q"},
        result=result,
        source="web",
        source_id_prefix=f"tooltrace:{turn_id}-{call_id}",
    )
    return exchange["tool_result"]["source_id"]


def _complete(mem: MemorySystem, turn_id: str, profile: str = "openai_chat") -> None:
    mem.build_context_projection(provider_profile=profile)
    result = mem.complete_turn(
        turn_id=turn_id,
        semantic_text="ok",
        provider_output_raw="ok",
        provider_profile=profile,
        provider_projection={"role": "assistant", "content": "ok"},
    )
    assert result.completed


def _tool_payloads(projection: object, profile: str = "openai_chat") -> list[dict]:
    payloads = list(projection.payloads)
    if profile == "anthropic_messages":
        out: list[dict] = []
        for p in payloads:
            if p.get("role") != "user":
                continue
            for block in p.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(block)
        return out
    return [p for p in payloads if p.get("role") == "tool"]


LONG_BODY = "网页正文 " + "记忆引擎上下文管理策略说明内容填充。" * 200
SHORT_BODY = "音乐已暂停"


class SettledHistorySelectionTests(unittest.TestCase):
    def test_closed_settled_turn_is_projected_from_settled_ledger(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_a", LONG_BODY)
        _complete(mem, "t1")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(projection)
        self.assertEqual(len(tool_payloads), 1)
        self.assertIn("[compact_reloadable]", tool_payloads[0]["content"])
        self.assertNotIn(LONG_BODY, tool_payloads[0]["content"])
        self.assertTrue(projection.has_compact_history)
        store.close()

    def test_settled_noop_turn_stays_full_in_history(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_b", SHORT_BODY)
        _complete(mem, "t1")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(projection)
        self.assertEqual(len(tool_payloads), 1)
        self.assertNotIn("[compact_reloadable]", tool_payloads[0]["content"])
        self.assertIn(SHORT_BODY, tool_payloads[0]["content"])
        self.assertFalse(projection.has_compact_history)
        store.close()

    def test_full_policy_keeps_full_projection_even_with_long_bodies(self) -> None:
        mem, store = _shared_mem(policy="full_until_raw_compaction")
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_c", LONG_BODY)
        _complete(mem, "t1")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(projection)
        self.assertNotIn("[compact_reloadable]", tool_payloads[0]["content"])
        self.assertIn(LONG_BODY, tool_payloads[0]["content"])
        self.assertFalse(projection.has_compact_history)
        store.close()

    def test_mixed_history_mixes_settled_and_full_in_order(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_d", LONG_BODY)
        _complete(mem, "t1")
        _begin(mem, "t2")
        _add_tool(mem, "t2", "call_e", SHORT_BODY)
        _complete(mem, "t2")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(projection)
        self.assertEqual(len(tool_payloads), 2)
        self.assertIn("[compact_reloadable]", tool_payloads[0]["content"])
        self.assertIn(SHORT_BODY, tool_payloads[1]["content"])
        indices = [p.get("role") for p in projection.payloads]
        self.assertEqual(indices.count("tool"), 2)
        store.close()

    def test_anthropic_settled_history_keeps_tool_use_shape(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_f", LONG_BODY)
        _complete(mem, "t1", profile="anthropic_messages")
        projection = mem.build_context_projection(provider_profile="anthropic_messages")
        tool_payloads = _tool_payloads(projection, profile="anthropic_messages")
        self.assertEqual(len(tool_payloads), 1)
        self.assertEqual(tool_payloads[0]["tool_use_id"], "call_f")
        self.assertIn("[compact_reloadable]", tool_payloads[0]["content"])
        self.assertTrue(projection.has_compact_history)
        store.close()


class SettledStabilityTests(unittest.TestCase):
    def test_settled_projection_is_stable_across_builds(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_g", LONG_BODY)
        _complete(mem, "t1")
        first = mem.build_context_projection(provider_profile="openai_chat")
        second = mem.build_context_projection(provider_profile="openai_chat")
        self.assertEqual(first.stable_prefix_hash, second.stable_prefix_hash)
        self.assertEqual([p for p in first.payloads], [p for p in second.payloads])
        store.close()

    def test_restart_rebuilds_same_settled_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slice3.sqlite3"
            mem, store = _shared_mem(store=SQLiteMemoryStore(str(path)))
            _begin(mem, "t1")
            _add_tool(mem, "t1", "call_h", LONG_BODY)
            _complete(mem, "t1")
            first = mem.build_context_projection(provider_profile="openai_chat")
            expected_hash = first.stable_prefix_hash
            store.close()

            fresh, fresh_store = _shared_mem(store=SQLiteMemoryStore(str(path)))
            rebuilt = fresh.build_context_projection(provider_profile="openai_chat")
            self.assertEqual(rebuilt.stable_prefix_hash, expected_hash)
            tool_payloads = _tool_payloads(rebuilt)
            self.assertIn("[compact_reloadable]", tool_payloads[0]["content"])
            fresh_store.close()


class ReadBackLoopTests(unittest.TestCase):
    def test_open_memory_readback_is_full_then_settles_again(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        source_sid = _add_tool(mem, "t1", "call_i", LONG_BODY)
        _complete(mem, "t1")

        # 新一轮: 模型按 observation source_id 回读旧结果
        settlement = store.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t1", provider_profile="openai_chat"
        )
        self.assertEqual(settlement["settlement_status"], "settled")
        readback = dispatch_native_memory_tool(
            "open_memory", {"memory_id": source_sid, "view": "content", "detail": "full"}, mem=mem
        )
        self.assertTrue(readback["ok"])
        self.assertIn(LONG_BODY, readback["result"]["text"])

        _begin(mem, "t2")
        _add_tool(mem, "t2", "call_j", readback["result"]["text"], tool="open_memory")
        # complete 前: 回读结果当轮全量可见
        open_projection = mem.build_context_projection(provider_profile="openai_chat")
        open_tool_payloads = _tool_payloads(open_projection)
        self.assertIn(LONG_BODY, open_tool_payloads[-1]["content"])
        _complete(mem, "t2")
        # 终局后再沉降
        t2_settlement = store.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t2", provider_profile="openai_chat"
        )
        self.assertEqual(t2_settlement["settlement_status"], "settled")
        final_projection = mem.build_context_projection(provider_profile="openai_chat")
        final_tool_payloads = _tool_payloads(final_projection)
        self.assertEqual(len(final_tool_payloads), 2)
        for tool_payload in final_tool_payloads:
            self.assertIn("[compact_reloadable]", tool_payload["content"])
        store.close()

    def test_batch_readback_and_missing_node_are_structured(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        sid_a = _add_tool(mem, "t1", "call_k", LONG_BODY + "_A")
        sid_b = _add_tool(mem, "t1", "call_l", LONG_BODY + "_B")
        _complete(mem, "t1")
        batch = dispatch_native_memory_tool(
            "open_memory", {"memory_ids": [sid_a, sid_b], "view": "content", "detail": "full"}, mem=mem
        )
        self.assertTrue(batch["ok"])
        top = batch["result"]["text"]
        self.assertIn(LONG_BODY + "_A", top)
        self.assertIn(LONG_BODY + "_B", top)
        missing = dispatch_native_memory_tool(
            "open_memory", {"memory_ids": [sid_a, "no-such-id"], "view": "content", "detail": "full"}, mem=mem
        )
        self.assertEqual(missing["result"]["status"], "partial")
        self.assertEqual(missing["result"]["reason"], "some_memory_nodes_unavailable")
        store.close()


if __name__ == "__main__":
    unittest.main()
