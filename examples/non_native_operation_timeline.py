"""Host-neutral action/result timeline example.

This demonstrates a host whose chat model emits custom XML-like tags instead of
provider-native tool calls. MemCore does not parse or execute the protocol. The
host maps the model action and environment result into the same append-only turn,
then freezes the provider-visible messages that were actually sent.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from memcore import (
    OPENAI_PROFILE,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemorySystem,
    Namespace,
    ProjectionMessageInput,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
    is_strict_message_prefix,
)


class NoopMemoryLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def main() -> None:
    with TemporaryDirectory() as tmp:
        embedding = HashedEmbeddingProvider()  # demo only; use a real embedding provider in production
        store = SQLiteMemoryStore(str(Path(tmp) / "memcore.sqlite3"))
        mem = MemorySystem(
            llm=NoopMemoryLLM(),
            namespace=Namespace(user_id="demo", conversation_id="custom-protocol"),
            timezone="Asia/Shanghai",
            store=store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
        )
        try:
            handle = mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="列出当前可以使用的能力",
                        payload={"text": "列出当前可以使用的能力"},
                        source_id="user-1",
                    )
                ],
                turn_id="turn-1",
            )

            # The model emitted this custom action. The host parsed it; MemCore only records it.
            model_action = '<action name="list_capabilities" id="catalog-1" />'
            mem.append_action(
                turn_id=handle.turn_id,
                kind="operation.catalog.request",
                correlation_id="catalog-1",
                semantic_text=model_action,
                payload={"name": "list_capabilities"},
                source_id="action-1",
                retention_anchor={"catalog_version": "v3", "schema_hash": "sha256:abc"},
            )

            # The host executed the action and returned a result through its own protocol.
            host_result = '<result id="catalog-1">search,weather</result>'
            mem.append_observation(
                turn_id=handle.turn_id,
                kind="operation.catalog.response",
                correlation_id="catalog-1",
                semantic_text=host_result,
                payload={"items": ["search", "weather"]},
                status="success",
                source_id="observation-1",
                retention_anchor={"catalog_ref": "catalog:v3"},
            )

            actual_history = [
                {"role": "user", "content": "列出当前可以使用的能力"},
                {"role": "assistant", "content": model_action},
                {"role": "user", "content": host_result},
            ]
            mem.record_request_projection(
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
                turn_messages=[
                    ProjectionMessageInput(
                        provider_profile=OPENAI_PROFILE,
                        payload=payload,
                        source_ids=(source_id,),
                        projection_index=index,
                    )
                    for index, (payload, source_id) in enumerate(
                        zip(actual_history, ("user-1", "action-1", "observation-1"))
                    )
                ],
                history_messages=actual_history,
                model_route="custom-tag-model",
                system_prefix="stable host prompt",
                tool_schema=[],
            )
            before_final = mem.build_context_projection(provider_profile=OPENAI_PROFILE)

            mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text="当前可以使用 search 和 weather。",
                provider_output_raw="当前可以使用 search 和 weather。",
                annotation_status="accepted",
                memory_annotation={"topic_terms": ["能力目录"]},
                source_id="final-1",
                provider_profile=OPENAI_PROFILE,
                provider_projection={"role": "assistant", "content": "当前可以使用 search 和 weather。"},
            )
            after_final = mem.build_context_projection(provider_profile=OPENAI_PROFILE)

            print("strict_prefix:", is_strict_message_prefix(before_final.payloads, after_final.payloads))
            print("provider_history:")
            for message in after_final.payloads:
                print(message)
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    main()
