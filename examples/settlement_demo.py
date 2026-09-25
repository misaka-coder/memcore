"""60-second MemCore Settlement demo.

This demo has no external model or vector-database dependency. It uses the
explicitly degraded hashed embedder only to keep the demo deterministic.

Run:
    python examples/settlement_demo.py
"""

from __future__ import annotations

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


class NoopMemoryLLM(LLMClient):
    """The demo never triggers semantic compaction, so fallback data is enough."""

    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def tool_contents(projection: object) -> list[str]:
    return [str(payload.get("content") or "") for payload in projection.payloads if payload.get("role") == "tool"]


def main() -> None:
    embedding = HashedEmbeddingProvider()
    store = SQLiteMemoryStore(":memory:")
    mem = MemorySystem(
        llm=NoopMemoryLLM(),
        namespace=Namespace(user_id="demo-user", conversation_id="demo-chat"),
        timezone="UTC",
        store=store,
        index=InMemoryVectorIndex(embedding=embedding),
        embedding=embedding,
        config=MemoryConfig(operation_projection_policy="compact_after_terminal"),
    )

    # A deliberately large tool result, representative of logs/web/code output.
    raw_result = (
        "TRACE: worker failed while loading deployment state\n"
        + "stack frame -> service/context/runtime.py:241\n" * 700
        + "ROOT_CAUSE=stale_projection_generation\n"
    )

    try:
        handle = mem.begin_turn(
            turn_id="demo-turn",
            stimuli=[
                TimelineEntryInput(
                    source_id="demo-user-message",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="Inspect this failure and tell me the root cause.",
                    payload={"text": "Inspect this failure and tell me the root cause."},
                    compatibility_role="user",
                )
            ],
        )

        exchange = mem.record_tool_exchange(
            turn_id=handle.turn_id,
            tool_name="shell",
            tool_call_id="call-demo",
            tool_input={"command": "cat service.log"},
            result=raw_result,
            source="local-shell",
            source_id_prefix="demo-shell",
        )
        result_source_id = exchange["tool_result"]["source_id"]

        # While the turn is still open, the full tool result remains resident.
        open_projection = mem.build_context_projection(provider_profile="openai_chat")
        open_tool_text = tool_contents(open_projection)[-1]
        assert raw_result in open_tool_text

        print("1) OPEN TURN: full evidence stays in context")
        print(f"   resident tool payload: {len(open_tool_text):,} characters")
        print(f"   raw source_id: {result_source_id}")

        # Freeze the provider-visible turn, then commit the terminal response.
        completed = mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="The root cause was a stale projection generation.",
            provider_output_raw="The root cause was a stale projection generation.",
            provider_profile="openai_chat",
            provider_projection={
                "role": "assistant",
                "content": "The root cause was a stale projection generation.",
            },
        )
        assert completed.completed

        # Closed history now projects the deterministic reload card, not the raw body.
        settled_projection = mem.build_context_projection(provider_profile="openai_chat")
        settled_tool_text = tool_contents(settled_projection)[-1]
        assert "[compact_reloadable]" in settled_tool_text
        assert raw_result not in settled_tool_text

        metric = mem.settlement_metrics()[0]
        print("\n2) TERMINAL SETTLEMENT: retired evidence pages out")
        print(f"   settled card: {len(settled_tool_text):,} characters")
        print(f"   projected token saving: {metric['saved_ratio']:.1%}")
        print("   card preview:")
        print("   " + settled_tool_text.replace("\n", "\n   ")[:700])

        # The exact raw observation is still available by stable source ID.
        readback = dispatch_native_memory_tool(
            "open_memory",
            {"memory_id": result_source_id, "view": "content", "detail": "full"},
            mem=mem,
        )
        assert readback["ok"]
        restored = str(readback["result"]["text"])
        assert raw_result in restored

        print("\n3) ON-DEMAND READBACK: exact evidence pages back in")
        print(f"   restored payload: {len(restored):,} characters")
        root_cause_recovered = "ROOT_CAUSE=stale_projection_generation" in restored
        print(f"   exact root cause recovered: {root_cause_recovered}")
    finally:
        mem.close()
        store.close()


if __name__ == "__main__":
    main()
