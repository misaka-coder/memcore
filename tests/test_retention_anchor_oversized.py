"""Oversized retention_anchor is non-fatal: receipts are preserved, never lost.

Regression for the production ``operation_retention_anchor_too_large`` failures:
an auxiliary metadata field exceeding the byte budget used to raise SchemaError
during entry construction and drop the whole tool record. It now stores the full
anchor, marks ``retention_anchor_status=oversized`` with a byte metric, and the
entry constructs and persists normally.
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
    build_action_entry,
    build_observation_entry,
    MAX_OPERATION_RETENTION_ANCHOR_BYTES,
)


class NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


def _memory() -> tuple[MemorySystem, SQLiteMemoryStore]:
    store = SQLiteMemoryStore(":memory:")
    embedding = HashedEmbeddingProvider()
    mem = MemorySystem(
        llm=NoopLLM(),
        namespace=Namespace(user_id="u1", conversation_id="s1"),
        timezone="Asia/Shanghai",
        config=MemoryConfig(),
        store=store,
        index=InMemoryVectorIndex(embedding=embedding),
        embedding=embedding,
    )
    return mem, store


def _open_turn(mem: MemorySystem, *, turn_id: str = "turn-1", ts: int = 100) -> None:
    mem.begin_turn(
        stimuli=[
            TimelineEntryInput(
                source_id="user-1",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text="查一下记忆",
                timestamp=ts,
                payload={"text": "查一下记忆"},
            )
        ],
        turn_id=turn_id,
        opened_at=ts,
    )


def _anchor_bytes(anchor: dict) -> int:
    return len(json.dumps(anchor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _host_style_pair(tool: str, correlation_id: str, prefix: str, ts: int, result: str, anchor: dict | None):
    """Mirror the host's manager._build_tool_entry_pair (TimelineEntryInput path)."""
    action = TimelineEntryInput(
        source_id=f"{prefix}:tool_use",
        kind=f"tool.{tool}.call",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.ACTION,
        semantic_text=f"call {tool}",
        timestamp=ts,
        payload={"input": {"q": "x"}},
        trace_metadata={"tool_name": tool, "status": "running"},
        compatibility_role=f"assistant.tool_call {tool} {correlation_id}",
        correlation_id=correlation_id,
        memory_metadata={},
    )
    observation_trace = {"tool_name": tool, "status": "success"}
    if anchor is not None:
        observation_trace["retention_anchor"] = dict(anchor)
    observation = TimelineEntryInput(
        source_id=f"{prefix}:tool_result",
        kind=f"tool.{tool}.result",
        origin=EntryOrigin.ENVIRONMENT,
        turn_role=TurnRole.OBSERVATION,
        semantic_text=f"result of {tool}: {result}",
        timestamp=ts + 1,
        payload={"output": result, "source": "synthetic"},
        trace_metadata=observation_trace,
        compatibility_role=f"tool.{tool} {correlation_id}",
        correlation_id=correlation_id,
        memory_metadata={},
    )
    return action, observation


class OversizedAnchorNonFatalTests(unittest.TestCase):
    def test_oversized_anchor_still_saves_action_and_observation(self) -> None:
        mem, store = _memory()
        try:
            _open_turn(mem)
            big = {"operation": "read_timeline", "cursor": "c-" + ("9" * 9000), "covered_ids": ["id-%05d" % i for i in range(2000)]}
            self.assertGreater(_anchor_bytes(big), MAX_OPERATION_RETENTION_ANCHOR_BYTES)

            action, observation = _host_style_pair("read_timeline", "call-1", "m1", 101, "timeline rows", big)
            stored_action = mem.append_entry(action, turn_id="turn-1")
            stored_observation = mem.append_entry(observation, turn_id="turn-1")

            self.assertIsNotNone(stored_action)
            self.assertIsNotNone(stored_observation)
            self.assertEqual(stored_observation.payload.get("output"), "timeline rows")
            self.assertEqual(stored_observation.semantic_text, "result of read_timeline: timeline rows")
            trace = stored_observation.trace_metadata
            self.assertEqual(trace["retention_anchor"], big)
            self.assertEqual(trace["retention_anchor_status"], "oversized")
            self.assertGreater(trace["retention_anchor_bytes"], MAX_OPERATION_RETENTION_ANCHOR_BYTES)
        finally:
            mem.close()
            store.close()

    def test_oversized_anchor_is_not_truncated(self) -> None:
        mem, _store = _memory()
        try:
            big = {"value": "x" * (MAX_OPERATION_RETENTION_ANCHOR_BYTES + 2048)}
            entry = build_observation_entry(
                kind="tool.foo.result",
                correlation_id="call-big",
                semantic_text="result body",
                retention_anchor=dict(big),
            )
            self.assertEqual(entry.trace_metadata["retention_anchor"], big)
            self.assertEqual(entry.trace_metadata["retention_anchor_status"], "oversized")
        finally:
            mem.close()

    def test_parallel_batch_with_one_oversized_anchor_saves_all(self) -> None:
        mem, store = _memory()
        try:
            _open_turn(mem, turn_id="turn-2", ts=200)
            big = {"operation": "read_timeline", "cursor": "c-" + ("9" * 9000), "covered_ids": ["id-%05d" % i for i in range(1500)]}
            pairs = [
                _host_style_pair("web_search", "call-p1", "p1", 201, "search rows", {"cursor": "c1"}),
                _host_style_pair("open_memory", "call-p2", "p2", 203, "card content", big),
                _host_style_pair("read_timeline", "call-p3", "p3", 205, "timeline rows", {"cursor": "c3"}),
            ]
            actions = [mem.append_entry(a, turn_id="turn-2") for a, _ in pairs]
            observations = [mem.append_entry(o, turn_id="turn-2") for _, o in pairs]
            self.assertEqual(len(actions), 3)
            self.assertEqual(len(observations), 3)
            saved_obs = {o.source_id: o for o in observations}
            self.assertIn("p2:tool_result", saved_obs)
            self.assertEqual(saved_obs["p2:tool_result"].trace_metadata["retention_anchor_status"], "oversized")
            self.assertEqual(saved_obs["p1:tool_result"].trace_metadata.get("retention_anchor_status"), None)
        finally:
            mem.close()
            store.close()

    def test_normal_anchor_metadata_is_unchanged(self) -> None:
        mem, _store = _memory()
        try:
            entry = build_observation_entry(
                kind="tool.foo.result",
                correlation_id="call-small",
                semantic_text="result body",
                retention_anchor={"operation": "open_memory", "returned_memory_ids": ["e1"]},
            )
            trace = entry.trace_metadata
            self.assertEqual(trace["retention_anchor"], {"operation": "open_memory", "returned_memory_ids": ["e1"]})
            self.assertNotIn("retention_anchor_status", trace)
            self.assertNotIn("retention_anchor_bytes", trace)
        finally:
            mem.close()


if __name__ == "__main__":
    unittest.main()
