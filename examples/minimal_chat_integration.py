"""Minimal memcore chat integration demo.

Run in an environment containing the released `memcore` package:

    python examples/minimal_chat_integration.py

This file is intentionally small and copyable. Replace `DemoMemoryLLM`,
`fake_chat_model`, and `HashedEmbeddingProvider` with your real model adapters in
production. Hashed embeddings are demo/test-only; they do not provide real
semantic recall.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

from memcore import (
    AnnotationStatus,
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
    TaskType,
    TimelineEntryInput,
    TurnRole,
    build_chat_output_contract_prompt,
    parse_chat_output,
)


def ts(year: int, month: int, day: int, hour: int, minute: int = 0, tz: str = "Asia/Shanghai") -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(tz)).timestamp())


class DemoMemoryLLM(LLMClient):
    """Deterministic LLM adapter for memcore's internal structured calls."""

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            data = {
                "diary_summary": "用户在询问记忆,并提到了饮料偏好。",
                "period_label": "一次记忆查询",
                "event_type": "memory_query",
                "importance": 0.55,
                "key_events": ["用户询问自己之前喜欢喝什么"],
                "core_facts": ["用户关心自己过往表达过的偏好"],
                "memory_metadata": {
                    "turn_intent": "memory_query",
                    "memory_facets": ["preference"],
                    "about_roles": ["user"],
                    "entity_anchors": ["可乐"],
                    "topic_terms": ["饮料", "偏好"],
                    "retrieval_priority": "normal",
                },
            }
            return LLMResult(ok=True, data=data, attempts=1)

        if request.task_type == TaskType.SEMANTIC:
            data = {
                "semantic_summary": "用户会回头询问自己表达过的偏好。",
                "importance": 0.65,
                "stable_facts": ["用户重视历史偏好的连续记忆"],
                "recurring_topics": ["偏好", "记忆查询"],
                "important_people": [],
                "open_loops": [],
                "memory_metadata": {
                    "memory_facets": ["preference"],
                    "about_roles": ["user"],
                    "entity_anchors": ["可乐"],
                    "topic_terms": ["偏好", "饮料", "记忆查询"],
                    "retrieval_priority": "high",
                },
            }
            return LLMResult(ok=True, data=data, attempts=1)

        if request.task_type == TaskType.REINFORCEMENT:
            return LLMResult(ok=True, data=request.fallback or {}, attempts=1)

        return LLMResult(ok=False, data=request.fallback or {}, error="unsupported_task", attempts=1)


def fake_chat_model(
    *,
    user_text: str,
    visible_memory: str,
    retrieved_memory: list[str],
    timeline_text: str,
    output_contract: str,
) -> str:
    """Stand in for the host application's final chat model call.

    A real host would send `visible_memory`, tool results, and `output_contract`
    to the chat model. If the model needs more memory, it should call the
    host-exposed wrappers around `retrieve_for_turn` and `read_timeline` first.
    """

    _ = (user_text, visible_memory, timeline_text, output_contract)
    knows_coke = any("可乐" in item for item in retrieved_memory)
    speech = "你之前说过自己喜欢喝无糖可乐。" if knows_coke else "我这边没有找到你之前提过的饮料偏好。"
    return json.dumps(
        {
            "speech": speech,
            "memory_metadata": {
                "turn_intent": "memory_query",
                "memory_facets": [],
                "about_roles": ["user"],
                "entity_anchors": ["可乐"],
                "topic_terms": ["饮料", "偏好"],
                "retrieval_priority": "low",
                "mood_tags": [],
            },
        },
        ensure_ascii=False,
    )


def main() -> None:
    with TemporaryDirectory() as tmp:
        embedding = HashedEmbeddingProvider()
        store = SQLiteMemoryStore(str(Path(tmp) / "memcore.sqlite3"))
        index = InMemoryVectorIndex(embedding=embedding)
        config = MemoryConfig(
            raw_token_trigger=1200,
            raw_token_batch_ratio=0.67,
            episodic_compact_trigger_count=99,
            enable_flavor=False,
        )
        llm = DemoMemoryLLM()

        old_mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="demo-user", conversation_id="old-chat"),
            timezone="Asia/Shanghai",
            store=store,
            index=index,
            embedding=embedding,
            config=config,
        )
        mem = MemorySystem(
            llm=llm,
            namespace=Namespace(user_id="demo-user", conversation_id="live-chat"),
            timezone="Asia/Shanghai",
            store=store,
            index=index,
            embedding=embedding,
            config=config,
        )

        try:
            old_handle = old_mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="我平常喜欢喝无糖可乐,别太甜。",
                        payload={"text": "我平常喜欢喝无糖可乐,别太甜。"},
                        timestamp=ts(2026, 4, 10, 9, 0),
                    )
                ]
            )
            old_mem.complete_turn(
                turn_id=old_handle.turn_id,
                semantic_text="记住了,你偏好无糖可乐。",
                provider_output_raw="记住了,你偏好无糖可乐。",
                memory_annotation={
                    "memory_facets": ["preference"],
                    "about_roles": ["user"],
                    "entity_anchors": ["可乐", "无糖可乐"],
                    "topic_terms": ["饮料", "无糖"],
                    "retrieval_priority": "high",
                },
                annotation_status=AnnotationStatus.ACCEPTED_HOST,
                timestamp=ts(2026, 4, 10, 9, 1),
            )

            reminder = mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="上午提醒我看一下持仓风险。",
                        payload={"text": "上午提醒我看一下持仓风险。"},
                        timestamp=ts(2026, 4, 10, 10, 0),
                    )
                ]
            )
            mem.complete_turn(
                turn_id=reminder.turn_id,
                semantic_text="好,我会按上午来理解这个提醒。",
                provider_output_raw="好,我会按上午来理解这个提醒。",
                annotation_status=AnnotationStatus.MISSING,
                timestamp=ts(2026, 4, 10, 10, 1),
            )

            user_text = "我之前说过自己喜欢喝什么吗?"
            handle = mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text=user_text,
                        payload={"text": user_text},
                        timestamp=ts(2026, 4, 10, 22, 0),
                    )
                ]
            )
            cur = handle.stimuli[0].to_record()

            context = mem.build_prompt_context(current=cur)
            visible_memory = mem.render_prompt_context(context)

            def retrieve_memory(query: str, **filters: Any) -> list[str]:
                return mem.retrieve_for_turn(current=cur, query=query, **filters)

            def read_timeline(**filters: Any) -> dict[str, Any]:
                return mem.read_timeline(**filters)

            retrieved = retrieve_memory(
                "用户喜欢喝什么",
                entity_anchors=["可乐"],
                topic_terms=["饮料"],
                memory_facets=["preference"],
                about_roles=["user"],
            )
            timeline = read_timeline(
                time_range={"start_at": "2026-04-10 09:30", "end_at": "2026-04-10 10:30"},
                cross_conversation=True,
            )
            output_contract = build_chat_output_contract_prompt(
                enable_flavor=config.enable_flavor,
                enable_sentence_segments=True,
            )

            raw_output = fake_chat_model(
                user_text=user_text,
                visible_memory=visible_memory,
                retrieved_memory=retrieved,
                timeline_text=str(timeline.get("text") or ""),
                output_contract=output_contract,
            )
            parsed = parse_chat_output(
                raw_output,
                mode="memcore_json",
                enable_flavor=config.enable_flavor,
            )
            if not parsed.ok:
                print({"status": parsed.status, "reason": parsed.reason})
                return

            completed = mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text=parsed.speech,
                provider_output_raw=raw_output,
                memory_annotation=parsed.memory_metadata,
                annotation_status=AnnotationStatus.ACCEPTED_MODEL,
                timestamp=ts(2026, 4, 10, 22, 1),
            )
            if not completed.completed:
                print({"status": completed.status, "reason": completed.reason})
                return

            compact_stats = mem.compact_due_background().result(timeout=10)

            print("visible memory prompt:")
            print(visible_memory)
            print("\nretrieve_for_turn result:")
            print("\n---\n".join(retrieved) or "(empty)")
            print("\nread_timeline text:")
            print(timeline.get("text") or "(empty)")
            print("\nassistant speech:")
            print(parsed.speech)
            print("\nspeech segments:")
            print(parsed.segments)
            print("\nturn completion:")
            print(completed)
            print("\nbackground compaction:")
            print(compact_stats)
        finally:
            old_mem.close()
            mem.close()
            store.close()


if __name__ == "__main__":
    main()
