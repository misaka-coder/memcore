from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from memcore import (
    ANTHROPIC_PROFILE,
    DEEPSEEK_PROFILE,
    OPENAI_PROFILE,
    OPENAI_RESPONSES_PROFILE,
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemorySystem,
    MemCoreContextSession,
    Namespace,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
    official_context_adapters,
    validate_context_adapter,
    validate_provider_wire_capture,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


class ContextContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(tenant_id="t", user_id="u", domain_id="d", conversation_id="c"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=InMemoryVectorIndex(embedding=self.embedding),
            embedding=self.embedding,
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()

    def test_surface_keeps_current_message_once_and_active_round_separate(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="current-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="hello",
                    payload={"text": "hello"},
                )
            ],
            turn_id="turn-1",
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="hi",
            provider_output_raw="hi",
            timestamp=1_724_000_000,
        )
        surface = self.mem.build_context_surface(
            session_id="session-1",
            provider_profile=OPENAI_PROFILE,
            current_source_id="current-user",
            active_turn_messages=({"role": "assistant", "tool_calls": []},),
        )
        self.assertEqual(surface.version, "context_surface_v1")
        self.assertEqual(surface.current_message["role"], "user")
        self.assertIn("hello", str(surface.current_message["content"]))
        self.assertEqual(sum(1 for item in surface.messages if item == surface.current_message), 1)
        self.assertEqual(surface.active_turn_messages[0]["role"], "assistant")
        self.assertEqual(len(surface.message_projection_metadata), len(surface.messages))
        current_index = len(surface.history_messages)
        current_metadata = surface.message_projection_metadata[current_index]
        self.assertEqual(current_metadata["turn_id"], "turn-1")
        self.assertEqual(current_metadata["source_ids"], ["current-user"])
        self.assertEqual(current_metadata["projection_index"], 0)
        self.assertGreaterEqual(current_metadata["projection_version"], 1)
        self.assertEqual(
            surface.as_dict()["message_projection_metadata"][current_index],
            current_metadata,
        )
        self.assertTrue(surface.projection_hash)
        self.assertEqual(
            surface.projection_hash,
            self.mem.build_context_surface(
                session_id="session-1",
                provider_profile=OPENAI_PROFILE,
                current_source_id="current-user",
                active_turn_messages=({"role": "assistant", "tool_calls": []},),
            ).projection_hash,
        )

    def test_surface_reads_visible_entries_once_and_uses_projection_batch(self) -> None:
        for index in range(3):
            handle = self.mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id=f"batch-user-{index}",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text=f"hello-{index}",
                        payload={"text": f"hello-{index}"},
                    )
                ],
                turn_id=f"batch-turn-{index}",
            )
            self.mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text=f"reply-{index}",
                provider_output_raw=f"reply-{index}",
                timestamp=1_724_000_000 + index,
            )
        # Freeze the rows once; the measured rebuild must only read them.
        self.mem.build_context_surface(
            provider_profile=OPENAI_PROFILE,
            current_source_id="batch-user-2",
        )

        with (
            patch.object(
                self.store,
                "list_prompt_visible_entries",
                wraps=self.store.list_prompt_visible_entries,
            ) as visible_read,
            patch.object(
                self.store,
                "get_context_projection_rows",
                wraps=self.store.get_context_projection_rows,
            ) as batch_read,
            patch.object(
                self.store,
                "get_turn_projections",
                side_effect=AssertionError("scalar projection read must not run"),
            ),
            patch.object(
                self.store,
                "get_turn_projection_settlement",
                side_effect=AssertionError("scalar settlement read must not run"),
            ),
        ):
            surface = self.mem.build_context_surface(
                provider_profile=OPENAI_PROFILE,
                current_source_id="batch-user-2",
            )

        self.assertTrue(surface.projection_hash)
        self.assertEqual(visible_read.call_count, 1)
        self.assertEqual(batch_read.call_count, 1)

    def test_projection_batch_fallback_is_byte_equivalent(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="fallback-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="hello",
                    payload={"text": "hello"},
                )
            ],
            turn_id="fallback-turn",
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="reply",
            provider_output_raw="reply",
            timestamp=1_724_000_000,
        )
        batched = self.mem.build_context_surface(
            provider_profile=OPENAI_PROFILE,
            current_source_id="fallback-user",
        )
        with patch.object(self.store, "get_context_projection_rows", side_effect=NotImplementedError):
            fallback = self.mem.build_context_surface(
                provider_profile=OPENAI_PROFILE,
                current_source_id="fallback-user",
            )
        self.assertEqual(fallback.as_dict(), batched.as_dict())

    def test_prompt_visible_query_uses_partial_scope_index(self) -> None:
        plan = self.store._conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM messages
            WHERE tenant_id = ? AND user_id = ? AND domain_id = ? AND conversation_id = ?
              AND is_summarized = 0 AND prompt_visible = 1
            ORDER BY seq_no
            """,
            ("t", "u", "d", "c"),
        ).fetchall()
        self.assertIn(
            "idx_messages_prompt_visible_scope_seq",
            " ".join(str(column) for row in plan for column in row),
        )

    def test_event_profile_is_not_downgraded_to_context_inject(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="finance-1",
                    kind="event.finance.quote",
                    origin=EntryOrigin.ENVIRONMENT,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="涨停",
                    payload={"source": "东方财富", "title": "A", "summary": "涨停"},
                )
            ],
            turn_id="turn-event",
        )
        surface = self.mem.build_context_surface(
            provider_profile=DEEPSEEK_PROFILE,
            current_source_id="finance-1",
        )
        self.assertIsNotNone(surface.current_message)
        self.assertIn("event.finance", str(surface.current_message))
        self.assertNotIn("context.inject", str(surface.current_message))

    def test_open_turn_keeps_current_then_native_tool_round(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="open-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="run it",
                    payload={"text": "run it"},
                )
            ],
            turn_id="open-turn",
        )
        self.mem.append_action(
            turn_id=handle.turn_id,
            kind="tool.exec.call",
            correlation_id="call-open",
            semantic_text='{"path":"x"}',
            payload={"name": "exec", "arguments": '{"path":"x"}'},
            source_id="open-action",
        )
        self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="tool.exec.result",
            correlation_id="call-open",
            semantic_text="done",
            payload={"output": "done", "status": "success"},
            source_id="open-observation",
            status="success",
        )
        surface = self.mem.build_context_surface(
            provider_profile=OPENAI_PROFILE,
            current_source_id="open-user",
        )
        self.assertEqual(surface.messages[0]["role"], "user")
        self.assertEqual(surface.messages[1]["role"], "assistant")
        self.assertEqual(surface.messages[2]["role"], "tool")
        self.assertEqual(surface.messages[2]["tool_call_id"], "call-open")

    def test_current_responses_message_keeps_non_text_blocks(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="image-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="describe it",
                    payload={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe it"},
                            {"type": "input_image", "image_url": "attachment://img-1"},
                        ],
                    },
                )
            ],
            turn_id="image-turn",
        )
        surface = self.mem.build_context_surface(
            provider_profile=OPENAI_RESPONSES_PROFILE,
            current_source_id="image-user",
        )
        blocks = surface.current_message["content"]
        self.assertIn("User: describe it", blocks[0]["text"])
        self.assertEqual(blocks[1], {"type": "input_image", "image_url": "attachment://img-1"})


class _Session:
    session_id = "session-1"
    session_settings = None

    def __init__(self) -> None:
        self.items: list[dict] = []

    async def get_items(self, limit=None):
        return list(self.items if limit is None else self.items[-limit:])

    async def add_items(self, items):
        self.items.extend(dict(item) for item in items)

    async def pop_item(self):
        return self.items.pop() if self.items else None

    async def clear_session(self):
        self.items.clear()


class ContextSessionWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="u", conversation_id="session-1"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=InMemoryVectorIndex(embedding=self.embedding),
            embedding=self.embedding,
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()

    def test_wrap_hides_turn_lifecycle_and_returns_authoritative_history(self) -> None:
        wrapped = MemCoreContextSession.wrap(_Session(), memory=self.mem)
        asyncio.run(wrapped.add_items([{"role": "user", "content": "hello"}]))
        asyncio.run(wrapped.add_items([{"role": "assistant", "content": "hi"}]))
        items = asyncio.run(wrapped.get_items())
        self.assertEqual(wrapped.status.mode, "authoritative")
        self.assertEqual(len([item for item in items if item.get("role") == "user"]), 1)
        self.assertEqual(len([item for item in items if item.get("role") == "assistant"]), 1)
        self.assertIn("User: hello", str(items[0].get("content")))

    def test_wrap_requires_memory_instead_of_silent_storage_only(self) -> None:
        with self.assertRaisesRegex(TypeError, "authoritative wrapping cannot be storage-only"):
            MemCoreContextSession.wrap(_Session())

    def test_wrap_rejects_memory_from_another_conversation(self) -> None:
        other = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="u", conversation_id="another-session"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=InMemoryVectorIndex(embedding=self.embedding),
            embedding=self.embedding,
        )
        try:
            with self.assertRaisesRegex(ValueError, "conversation_id must match"):
                MemCoreContextSession.wrap(_Session(), memory=other)
        finally:
            other.close()

    def test_runner_order_callback_projects_current_before_first_request(self) -> None:
        session = _Session()
        wrapped = MemCoreContextSession.wrap(session, memory=self.mem)
        new_item = {"type": "message", "role": "user", "content": "hello from runner"}

        provider_items = asyncio.run(wrapped.input_callback([], [new_item]))

        self.assertEqual(len(provider_items), 1)
        self.assertEqual(provider_items[0]["type"], "message")
        self.assertIn("User: hello from runner", str(provider_items[0]["content"]))
        asyncio.run(wrapped.add_items([new_item, {"type": "message", "role": "assistant", "content": "hi"}]))
        persisted = asyncio.run(wrapped.get_items())
        self.assertEqual(len(persisted), 2)

    def test_existing_session_history_is_bootstrapped_once(self) -> None:
        session = _Session()
        session.items = [
            {"type": "message", "role": "user", "content": "older question"},
            {"type": "message", "role": "assistant", "content": "older answer"},
        ]
        wrapped = MemCoreContextSession.wrap(session, memory=self.mem)

        first = asyncio.run(wrapped.get_items())
        second = asyncio.run(wrapped.get_items())

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertIn("older question", str(first[0]))
        self.assertIn("older answer", str(first[1]))

    def test_first_limited_read_bootstraps_the_complete_host_history(self) -> None:
        session = _Session()
        session.items = [
            {"type": "message", "role": "user", "content": "question one"},
            {"type": "message", "role": "assistant", "content": "answer one"},
            {"type": "message", "role": "user", "content": "question two"},
            {"type": "message", "role": "assistant", "content": "answer two"},
        ]
        wrapped = MemCoreContextSession.wrap(session, memory=self.mem)

        limited = asyncio.run(wrapped.get_items(limit=1))
        complete = asyncio.run(wrapped.get_items())

        self.assertEqual(len(limited), 1)
        self.assertEqual(len(complete), 4)
        self.assertIn("question one", str(complete[0]))

    def test_input_callback_surface_failure_keeps_native_history_and_current_input(self) -> None:
        session = _Session()
        session.items = [
            {"type": "message", "role": "user", "content": "older question"},
            {"type": "message", "role": "assistant", "content": "older answer"},
        ]
        wrapped = MemCoreContextSession.wrap(session, memory=self.mem)
        asyncio.run(wrapped.get_items())
        new_item = {"type": "message", "role": "user", "content": "new question"}

        with patch.object(self.mem, "build_context_surface", side_effect=RuntimeError("offline")):
            visible = asyncio.run(wrapped.input_callback([], [new_item]))

        self.assertEqual(visible, [*session.items, new_item])
        self.assertEqual(wrapped.status.mode, "degraded")
        self.assertIn("context_surface_unavailable", [item.reason for item in wrapped.status.diagnostics])

    def test_unsupported_existing_item_degrades_without_emptying_history(self) -> None:
        session = _Session()
        session.items = [{"type": "reasoning", "id": "reasoning-1", "summary": []}]
        wrapped = MemCoreContextSession.wrap(session, memory=self.mem)

        visible = asyncio.run(wrapped.get_items())
        first_request = asyncio.run(
            wrapped.input_callback([], [{"type": "message", "role": "user", "content": "continue"}])
        )

        self.assertEqual(visible, session.items)
        self.assertEqual(first_request, [*session.items, {"type": "message", "role": "user", "content": "continue"}])
        self.assertEqual(wrapped.status.mode, "degraded")

    def test_bootstrap_uses_host_timestamp_when_present(self) -> None:
        session = _Session()
        session.items = [
            {"type": "message", "role": "user", "content": "dated", "created_at": 1_700_000_000},
            {"type": "message", "role": "assistant", "content": "done", "created_at": 1_700_000_001},
        ]
        wrapped = MemCoreContextSession.wrap(session, memory=self.mem)

        visible = asyncio.run(wrapped.get_items())

        self.assertIn("[2023-11-15 06:13] User: dated", str(visible[0]))

    def test_frozen_projection_renders_ordinary_chat_messages_with_time_anchors(self) -> None:
        """The ledger-frozen openai family must use the same ordinary chat
        rendering as the live surface: `[YYYY-MM-DD HH:MM] User:/Assistant: …`.
        A frozen history whose user rows render in the structured canonical
        format would break hosts that read history_records directly from
        context.build (thin adapters never re-render payloads)."""
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="frozen", conversation_id="frozen"),
            timezone="Asia/Shanghai",
            store=store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
        )
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="frozen-user",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="hello world",
                        payload={"text": "hello world"},
                        timestamp=1_700_000_000,
                        compatibility_role="user",
                    )
                ],
                turn_id="frozen-turn",
            )
            # First build freezes the turn; the second must serve the frozen
            # payload — still in the ordinary chat format.
            mem.build_context_surface(
                provider_profile=OPENAI_PROFILE,
                current_source_id="frozen-user",
            )
            projection = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            user_payload = projection.payloads[0]
            self.assertEqual(user_payload["role"], "user")
            self.assertIn("[2023-11-15 06:13] User: hello world", str(user_payload["content"]))
            self.assertNotIn("message.user", str(user_payload["content"]))
        finally:
            mem.close()
            store.close()

    def test_attributed_group_message_keeps_actor_target_and_mentions(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="group-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="【李四】你怎么看？",
                    payload={
                        "text": "【李四】你怎么看？",
                        "mentioned_actors": [{"actor_id": "qq:3", "display_name": "天为"}],
                    },
                    actor=Actor(stable_id="qq:2", display_name="李四"),
                    target_actor=Actor(stable_id="assistant"),
                    timestamp=1_700_000_000,
                    compatibility_role="user",
                )
            ],
            turn_id="group-turn",
        )

        for profile in (OPENAI_PROFILE, OPENAI_RESPONSES_PROFILE, ANTHROPIC_PROFILE):
            with self.subTest(profile=profile):
                projection = self.mem.build_context_projection(provider_profile=profile)
                content = str(projection.payloads[0].get("content") or "")
                self.assertIn("message.user", content)
                self.assertIn("actor: 李四 (id=qq:2)", content)
                self.assertIn("target_actor: assistant", content)
                self.assertIn('"actor_id":"qq:3"', content)
                self.assertNotIn("User: 【李四】你怎么看？", content)

    def test_pop_rebuilds_memcore_and_removes_the_popped_item_from_context(self) -> None:
        wrapped = MemCoreContextSession.wrap(_Session(), memory=self.mem)
        asyncio.run(wrapped.add_items([{"type": "message", "role": "user", "content": "hello"}]))
        asyncio.run(wrapped.add_items([{"type": "message", "role": "assistant", "content": "hi"}]))

        popped = asyncio.run(wrapped.pop_item())
        visible = asyncio.run(wrapped.get_items())

        self.assertEqual(popped["content"], "hi")
        self.assertNotIn("Assistant: hi", str(visible))
        self.assertIn("session_pop_rebuilt_memcore", [item.reason for item in wrapped.status.diagnostics])

    def test_wrap_preserves_native_tool_call_and_result(self) -> None:
        wrapped = MemCoreContextSession.wrap(_Session(), memory=self.mem)
        asyncio.run(wrapped.add_items([{"type": "message", "role": "user", "content": "run"}]))
        asyncio.run(
            wrapped.add_items(
                [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "exec",
                        "arguments": '{"x":1}',
                    }
                ]
            )
        )
        asyncio.run(wrapped.add_items([{"type": "function_call_output", "call_id": "call-1", "output": "ok"}]))
        asyncio.run(wrapped.add_items([{"type": "message", "role": "assistant", "content": "done"}]))
        items = asyncio.run(wrapped.get_items())
        tool_call = next(item for item in items if item.get("type") == "function_call")
        tool_result = next(item for item in items if item.get("type") == "function_call_output")
        self.assertEqual(tool_call["arguments"], '{"x":1}')
        self.assertEqual(tool_result["output"], "ok")


class ProviderConformanceTests(unittest.TestCase):
    def test_official_adapters_pass_provider_shape_conformance(self) -> None:
        for profile, adapter in official_context_adapters().items():
            with self.subTest(profile=profile):
                report = asyncio.run(validate_context_adapter(adapter))
                self.assertTrue(report.passed, report.failures)

    def test_supported_profiles_are_explicit(self) -> None:
        self.assertEqual(
            set(official_context_adapters()),
            {OPENAI_PROFILE, DEEPSEEK_PROFILE, ANTHROPIC_PROFILE, OPENAI_RESPONSES_PROFILE},
        )

    def test_provider_wire_capture_requires_exact_surface_messages(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="wire", conversation_id="wire"),
            timezone="Asia/Shanghai",
            store=store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
        )
        try:
            mem.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id="wire-user",
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text="hello",
                        payload={"type": "message", "role": "user", "content": "hello"},
                    )
                ],
                turn_id="wire-turn",
            )
            surface = mem.build_context_surface(
                provider_profile=OPENAI_RESPONSES_PROFILE,
                current_source_id="wire-user",
            )
            self.assertTrue(validate_provider_wire_capture(surface, surface.messages).passed)
            changed = [dict(item) for item in surface.messages]
            changed[0]["content"] = "host rewrote it"
            self.assertFalse(validate_provider_wire_capture(surface, changed).passed)
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
