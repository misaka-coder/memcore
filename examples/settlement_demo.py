"""Zero-API-key demo of MemCore's terminal settlement.

Run from a source checkout after installing the package:

    python examples/settlement_demo.py

The demo keeps a long tool result fully visible while the turn is open, then
settles it into a compact reloadable card after the final answer. The original
observation remains in SQLite and is reopened by its stable source_id.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

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
)


class NoopMemoryLLM(LLMClient):
    """Deterministic adapter; no external API is called in this demo."""

    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def main() -> None:
    with TemporaryDirectory() as tmp:
        embedding = HashedEmbeddingProvider()  # demo/test only
        store = SQLiteMemoryStore(str(Path(tmp) / "memcore.sqlite3"))
        mem = MemorySystem(
            llm=NoopMemoryLLM(),
            namespace=Namespace(user_id="demo-user", conversation_id="settlement-demo"),
            timezone="UTC",
            store=store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
            config=MemoryConfig(operation_projection_policy="compact_after_terminal"),
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
                        semantic_text="Inspect this large diagnostic output and summarize the failure.",
                        payload={"text": "Inspect this large diagnostic output and summarize the failure."},
                    )
                ],
            )

            long_result = (
                "Traceback: worker failed while rebuilding the search index. "
                "ROOT_CAUSE=stale_projection_generation\n"
                + ("diagnostic context line; cache state and retry metadata follow.\n" * 500)
            )
            exchange = mem.record_tool_exchange(
                turn_id=handle.turn_id,
                tool_name="diagnostics",
                tool_call_id="call-001",
                tool_input={"target": "index-rebuild"},
                result=long_result,
                source="demo",
                source_id_prefix="demo-tool",
            )
            result_source_id = exchange["tool_result"]["source_id"]

            # Open-turn provider history: the full tool output is still resident.
            before = mem.build_context_projection(provider_profile="openai_chat")
            before_text = str(before.payloads)
            assert "ROOT_CAUSE=stale_projection_generation" in before_text

            completed = mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text="The rebuild failed because the projection generation was stale.",
                provider_output_raw="The rebuild failed because the projection generation was stale.",
                provider_profile="openai_chat",
                provider_projection={
                    "role": "assistant",
                    "content": "The rebuild failed because the projection generation was stale.",
                },
            )
            assert completed.completed

            # Later provider history: the long body is replaced by a reloadable card.
            after = mem.build_context_projection(provider_profile="openai_chat")
            after_text = str(after.payloads)
            assert "[compact_reloadable]" in after_text
            assert "ROOT_CAUSE=stale_projection_generation" not in after_text

            reopened = mem.open_memory(
                memory_id=result_source_id,
                view="content",
                detail="full",
            )
            assert "ROOT_CAUSE=stale_projection_generation" in str(reopened)

            metrics = mem.settlement_metrics()
            row = metrics[-1]

            print("MemCore settlement demo")
            print("-----------------------")
            print(f"full projected tokens:    {row['full_projected_tokens']}")
            print(f"settled projected tokens: {row['settled_projected_tokens']}")
            print(f"saved projected tokens:   {row['saved_projected_tokens']}")
            print(f"saved ratio:              {row['saved_ratio']:.1%}")
            print(f"reload source_id:          {result_source_id}")
            print("reload verified:           yes")
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    main()
