"""Settlement card readable time + byte-preservation regression (Phase 4/5 items 8-12).

Covers, without adding any tool-specific renderer:
- the generic [compact_reloadable] card gains one readable ``time`` line
  derived from the entry's existing timestamp;
- the card never duplicates the tool input (kept in the adjacent action);
- OpenAI ``function.arguments`` and Anthropic ``tool_use.input`` stay
  byte-identical across settlement;
- an open tool round keeps the full result (no early compaction);
- open_memory(detail="full") restores the full result together with its time.
"""

from __future__ import annotations

import json
import unittest

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
from memcore.time_anchor import timestamp_to_datetime_label, timestamp_to_datetime_weekday_label


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


FIXED_TS = 1755000000
LONG_BODY = "网页正文 " + "记忆引擎上下文管理策略说明内容填充。" * 200


def _shared_mem(policy: str = "compact_after_terminal"):
    emb = HashedEmbeddingProvider()
    store = SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    mem = MemorySystem(
        llm=_NoopLLM(),
        namespace=Namespace(user_id="u1", conversation_id="c1"),
        timezone="Asia/Shanghai",
        store=store,
        index=index,
        embedding=emb,
        config=MemoryConfig(operation_projection_policy=policy),
    )
    return mem, store


def _begin(mem: MemorySystem, turn_id: str, text: str = "问题") -> None:
    mem.begin_turn(
        turn_id=turn_id,
        opened_at=FIXED_TS,
        stimuli=[
            TimelineEntryInput(
                source_id=f"{turn_id}-u",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text=text,
                payload={"text": text},
                timestamp=FIXED_TS,
                compatibility_role="user",
            )
        ],
    )


def _add_tool(
    mem: MemorySystem,
    turn_id: str,
    call_id: str,
    result: str,
    tool: str = "web_search",
    tool_input: object | None = None,
    timestamp: int = FIXED_TS,
) -> str:
    exchange = mem.record_tool_exchange(
        turn_id=turn_id,
        tool_name=tool,
        tool_call_id=call_id,
        tool_input=tool_input if tool_input is not None else {"query": "q"},
        result=result,
        source="web",
        timestamp=timestamp,
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
        for payload in payloads:
            if payload.get("role") != "user":
                continue
            for block in payload.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(block)
        return out
    return [payload for payload in payloads if payload.get("role") == "tool"]


def _openai_action_arguments(projection: object) -> list[str]:
    out: list[str] = []
    for payload in projection.payloads:
        if payload.get("role") != "assistant":
            continue
        for call in payload.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            if function.get("name") == "web_search":
                out.append(function.get("arguments") or "")
    return out


def _anthropic_tool_use_inputs(projection: object) -> list[object]:
    out: list[object] = []
    for payload in projection.payloads:
        if payload.get("role") != "assistant":
            continue
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append(block.get("input"))
    return out


class SettlementCardTimeTests(unittest.TestCase):
    def test_card_carries_readable_time_and_no_tool_input_duplication(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        secret_input = {"query": "绝密搜索词", "command": "echo no-duplicate"}
        _add_tool(mem, "t1", "call_t", LONG_BODY, tool_input=secret_input)
        _complete(mem, "t1")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(projection)
        self.assertEqual(len(tool_payloads), 1)
        content = str(tool_payloads[0]["content"])
        self.assertIn("[compact_reloadable]", content)
        # One generic readable time line, derived from the entry timestamp.
        expected_time = timestamp_to_datetime_label(FIXED_TS + 1, "Asia/Shanghai")
        self.assertIn(f"time: {expected_time}", content)
        self.assertIn("tool: web_search", content)
        self.assertIn("call_id: call_t", content)
        self.assertIn("status: success", content)
        self.assertIn("source_id: tooltrace:t1-call_t:tool_result", content)
        self.assertIn("stored_chars:", content)
        self.assertIn("reload: open_memory(memory_id=", content)
        # The tool input stays in the adjacent action and is never copied into
        # the result card.
        self.assertNotIn("绝密搜索词", content)
        self.assertNotIn("echo no-duplicate", content)
        self.assertNotIn(LONG_BODY, content)
        store.close()

    def test_old_cards_without_timezone_stay_byte_identical(self) -> None:
        # The generic renderer must stay backwards compatible: without a
        # timezone there is no time line, so historical cards are unchanged.
        from types import SimpleNamespace

        from memcore.settlement import render_compact_reload

        entry = SimpleNamespace(
            kind="tool.web_search.result",
            source_id="tooltrace:x:tool_result",
            correlation_id="call_old",
            timestamp=FIXED_TS,
            payload={"output": LONG_BODY},
            trace_metadata={"tool_name": "web_search", "status": "success"},
        )
        legacy = render_compact_reload(entry, full_content=LONG_BODY)
        self.assertNotIn("time:", legacy)
        with_time = render_compact_reload(entry, full_content=LONG_BODY, timezone="Asia/Shanghai")
        self.assertIn(f"time: {timestamp_to_datetime_label(FIXED_TS, 'Asia/Shanghai')}", with_time)
        self.assertTrue(with_time.startswith("[compact_reloadable]\ntime: "))

    def test_openai_function_arguments_byte_identical_after_settlement(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        tool_input = {"query": "字节保持", "limit": 3, "tags": ["a", "b"]}
        _add_tool(mem, "t1", "call_args", LONG_BODY, tool_input=tool_input)
        full = mem.build_context_projection(provider_profile="openai_chat")
        _complete(mem, "t1")
        settled = mem.build_context_projection(provider_profile="openai_chat")
        full_arguments = _openai_action_arguments(full)
        settled_arguments = _openai_action_arguments(settled)
        self.assertEqual(full_arguments, settled_arguments)
        self.assertEqual(
            settled_arguments,
            [json.dumps(tool_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))],
        )
        store.close()

    def test_anthropic_tool_use_input_unchanged_after_settlement(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        tool_input = {"command": "ls -la", "cwd": "/tmp"}
        _add_tool(mem, "t1", "call_ant", LONG_BODY, tool_input=tool_input)
        full = mem.build_context_projection(provider_profile="anthropic_messages")
        _complete(mem, "t1", profile="anthropic_messages")
        settled = mem.build_context_projection(provider_profile="anthropic_messages")
        full_inputs = _anthropic_tool_use_inputs(full)
        settled_inputs = _anthropic_tool_use_inputs(settled)
        self.assertEqual(full_inputs, settled_inputs)
        self.assertEqual(
            [json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in settled_inputs],
            [json.dumps(tool_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))],
        )
        store.close()

    def test_open_tool_round_keeps_full_result_no_early_compaction(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        _add_tool(mem, "t1", "call_open", LONG_BODY)
        # The turn is still open: the current tool round must keep the full
        # result in the projection, never an early compact card.
        open_projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(open_projection)
        self.assertEqual(len(tool_payloads), 1)
        self.assertIn(LONG_BODY, str(tool_payloads[0]["content"]))
        self.assertNotIn("[compact_reloadable]", str(tool_payloads[0]["content"]))
        store.close()

    def test_open_memory_restores_full_result_with_time(self) -> None:
        mem, store = _shared_mem()
        _begin(mem, "t1")
        source_sid = _add_tool(mem, "t1", "call_read", LONG_BODY)
        _complete(mem, "t1")
        # The settled card points at the stored observation via source_id.
        projection = mem.build_context_projection(provider_profile="openai_chat")
        tool_payloads = _tool_payloads(projection)
        self.assertIn(source_sid, str(tool_payloads[0]["content"]))
        readback = dispatch_native_memory_tool(
            "open_memory", {"memory_id": source_sid, "view": "content", "detail": "full"}, mem=mem
        )
        self.assertTrue(readback["ok"])
        text = str(readback["result"]["text"])
        self.assertIn(LONG_BODY, text)
        # The restored full result carries its own readable time anchor.
        self.assertIn(timestamp_to_datetime_weekday_label(FIXED_TS + 1, "Asia/Shanghai"), text)
        store.close()


if __name__ == "__main__":
    unittest.main()
