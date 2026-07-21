"""Timeline V2 deterministic rendering, provider adapters, and projection ledger tests."""

from __future__ import annotations

import unittest
from typing import Any

from memcore import (
    ANTHROPIC_PROFILE,
    CANONICAL_PROFILE,
    OPENAI_PROFILE,
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemorySystem,
    Namespace,
    NamespaceError,
    ProjectionMessageInput,
    ProjectionStatus,
    RendererRegistry,
    SQLiteMemoryStore,
    SchemaError,
    TimelineEntryInput,
    TurnRole,
    TurnStatus,
    canonical_json_bytes,
    is_strict_message_prefix,
    stable_projection_hash,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _stimulus(
    text: str,
    *,
    source_id: str,
    kind: str = "message.user",
    payload: dict[str, Any] | None = None,
    actor: Actor | None = None,
) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind=kind,
        origin=EntryOrigin.USER if kind == "message.user" else EntryOrigin.ENVIRONMENT,
        turn_role=TurnRole.STIMULUS,
        semantic_text=text,
        payload=payload or {"text": text},
        actor=actor,
    )


def _action(call_id: str, *, source_id: str) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="tool.web_search.call",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.ACTION,
        semantic_text=f"search {call_id}",
        payload={"query": call_id},
        correlation_id=call_id,
        trace_metadata={"tool_name": "web_search", "status": "running"},
    )


def _intermediate(text: str, *, source_id: str) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="message.assistant.intermediate",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.INTERMEDIATE,
        semantic_text=text,
        payload={"text": text},
    )


def _observation(
    call_id: str,
    *,
    source_id: str,
    text: str,
    status: str = "success",
) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="tool.web_search.result",
        origin=EntryOrigin.ENVIRONMENT,
        turn_role=TurnRole.OBSERVATION,
        semantic_text=text,
        payload={"output": text},
        correlation_id=call_id,
        trace_metadata={"tool_name": "web_search", "status": status},
    )


class ProjectionBase(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = Namespace(
            tenant_id="tenant",
            user_id="user",
            domain_id="domain",
            conversation_id="conversation",
        )
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=self.namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            index=InMemoryVectorIndex(embedding=self.embedding),
            embedding=self.embedding,
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()


class CanonicalRenderingTests(ProjectionBase):
    def test_renderer_resolution_prefers_exact_then_longest_prefix(self) -> None:
        registry = RendererRegistry()

        def broad(_entry: Any, _timezone: str) -> str:
            return "broad"

        def narrow(_entry: Any, _timezone: str) -> str:
            return "narrow"

        def exact(_entry: Any, _timezone: str) -> str:
            return "exact"

        registry.register_prefix("event", renderer_id="broad", version=1, renderer=broad)
        registry.register_prefix("event.custom", renderer_id="narrow", version=1, renderer=narrow)
        registry.register_exact("event.custom.one", renderer_id="exact", version=1, renderer=exact)
        self.assertEqual(registry.select("event.other"), ("broad", 1))
        self.assertEqual(registry.select("event.custom.two"), ("narrow", 1))
        self.assertEqual(registry.select("event.custom.one"), ("exact", 1))

    def test_unknown_kind_uses_deterministic_safe_canonical_fallback(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                _stimulus(
                    "未来事件",
                    source_id="unknown-event",
                    kind="event.future.signal",
                    payload={"z": 2, "a": 1},
                    actor=Actor(stable_id="qq:1", display_name="张三"),
                )
            ],
            turn_id="unknown-turn",
            opened_at=1_753_000_000,
        )
        first = self.mem.build_context_projection(provider_profile=CANONICAL_PROFILE)
        second = self.mem.build_context_projection(provider_profile=CANONICAL_PROFILE)
        self.assertEqual(first.payloads, second.payloads)
        self.assertEqual(first.stable_prefix_hash, second.stable_prefix_hash)
        content = str(first.payloads[0]["content"])
        self.assertIn("event.future.signal", content)
        self.assertIn("张三 (id=qq:1)", content)
        self.assertLess(content.index('"a":1'), content.index('"z":2'))
        self.assertEqual(first.messages[0].projection_status, ProjectionStatus.CANONICAL_FALLBACK)

    def test_finance_renderer_is_neutral_and_field_order_is_stable(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                _stimulus(
                    "财经摘要",
                    source_id="finance-event",
                    kind="event.finance",
                    payload={
                        "url": "https://example.com/news",
                        "summary": "摘要",
                        "title": "标题",
                        "published_at": "08:52",
                        "source": "wire",
                    },
                )
            ],
            turn_id="finance-turn",
        )
        self.assertEqual(handle.stimuli[0].renderer_id, "event.finance")
        content = str(self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads[0]["content"])
        self.assertNotIn("插件", content)
        self.assertNotIn("系统事件", content)
        positions = [content.index(f"{key}:") for key in ("source", "published_at", "title", "summary", "url")]
        self.assertEqual(positions, sorted(positions))

    def test_missing_historical_renderer_falls_back_without_changing_provider_role(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="missing-renderer",
                    kind="event.missing.renderer",
                    origin=EntryOrigin.ENVIRONMENT,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="safe data",
                    payload={"value": 1},
                    renderer_id="removed-renderer",
                    renderer_version=9,
                )
            ],
            turn_id="missing-renderer-turn",
        )
        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(projection.payloads[0]["role"], "user")
        self.assertIn("event.missing.renderer", projection.payloads[0]["content"])
        self.assertEqual(projection.messages[0].projection_status, ProjectionStatus.CANONICAL_FALLBACK)

    def test_renderer_upgrade_only_affects_new_entries(self) -> None:
        registry = RendererRegistry()

        def render_v1(_entry: Any, _timezone: str) -> str:
            return "renderer-v1"

        def render_v2(_entry: Any, _timezone: str) -> str:
            return "renderer-v2"

        registry.register_prefix("event.custom", renderer_id="custom", version=1, renderer=render_v1)
        mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="renderer", conversation_id="c"),
            timezone="Asia/Shanghai",
            store=SQLiteMemoryStore(":memory:"),
            index=InMemoryVectorIndex(embedding=HashedEmbeddingProvider()),
            embedding=HashedEmbeddingProvider(),
            renderer_registry=registry,
        )
        try:
            first = mem.begin_turn(
                stimuli=[_stimulus("one", source_id="renderer-one", kind="event.custom.one")],
                turn_id="renderer-turn-one",
            )
            before = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            registry.register_prefix("event.custom", renderer_id="custom", version=2, renderer=render_v2)
            second = mem.begin_turn(
                stimuli=[_stimulus("two", source_id="renderer-two", kind="event.custom.two")],
                turn_id="renderer-turn-two",
            )
            after = mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            self.assertEqual(first.stimuli[0].renderer_version, 1)
            self.assertEqual(second.stimuli[0].renderer_version, 2)
            self.assertEqual(before.payloads[0]["content"], "renderer-v1")
            self.assertEqual(after.payloads[0]["content"], "renderer-v1")
            self.assertEqual(after.payloads[1]["content"], "renderer-v2")
        finally:
            mem.store.close()
            mem.close()


class ProviderAdapterTests(ProjectionBase):
    def _build_tool_turn(self) -> str:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("查两个方向", source_id="tool-question")],
            turn_id="provider-tools",
        )
        self.mem.append_entry(_action("call-a", source_id="action-a"), turn_id=handle.turn_id)
        self.mem.append_entry(_action("call-b", source_id="action-b"), turn_id=handle.turn_id)
        self.mem.append_entry(
            _observation("call-b", source_id="result-b", text="B result"),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("call-a", source_id="result-a", text="A result"),
            turn_id=handle.turn_id,
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="查完了",
            provider_output_raw='{"speech":"查完了"}',
            memory_annotation={"keywords": ["两个方向"]},
            annotation_status="accepted",
            source_id="tool-final",
        )
        return handle.turn_id

    def test_openai_parallel_tools_are_one_call_batch_with_correlated_results(self) -> None:
        self._build_tool_turn()
        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        payloads = projection.payloads
        self.assertEqual([item["role"] for item in payloads], ["user", "assistant", "tool", "tool", "assistant"])
        calls = payloads[1]["tool_calls"]
        self.assertEqual([item["id"] for item in calls], ["call-a", "call-b"])
        self.assertEqual([payloads[2]["tool_call_id"], payloads[3]["tool_call_id"]], ["call-b", "call-a"])
        self.assertEqual(payloads[-1]["content"], '{"speech":"查完了"}')
        repeated = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(projection.payloads, repeated.payloads)

    def test_anthropic_parallel_tools_round_trip_without_openai_shape_reuse(self) -> None:
        self._build_tool_turn()
        projection = self.mem.build_context_projection(provider_profile=ANTHROPIC_PROFILE)
        payloads = projection.payloads
        self.assertEqual([item["role"] for item in payloads], ["user", "assistant", "user", "assistant"])
        self.assertEqual([block["type"] for block in payloads[1]["content"]], ["tool_use", "tool_use"])
        self.assertEqual(
            [block["tool_use_id"] for block in payloads[2]["content"]],
            ["call-b", "call-a"],
        )
        self.assertNotIn("tool_calls", str(payloads))
        openai = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertNotEqual(openai.stable_prefix_hash, projection.stable_prefix_hash)

    def test_provider_tool_results_preserve_content_and_anthropic_terminal_errors(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("查失败路径", source_id="error-question")],
            turn_id="provider-tool-errors",
        )
        self.mem.append_entry(_action("call-error", source_id="action-error"), turn_id=handle.turn_id)
        self.mem.append_entry(_action("call-cancelled", source_id="action-cancelled"), turn_id=handle.turn_id)
        self.mem.append_entry(
            _observation(
                "call-cancelled",
                source_id="result-cancelled",
                text="cancelled verbatim",
                status="cancelled",
            ),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("call-error", source_id="result-error", text="error verbatim", status="error"),
            turn_id=handle.turn_id,
        )

        openai = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads
        anthropic = self.mem.build_context_projection(provider_profile=ANTHROPIC_PROFILE).payloads

        self.assertEqual([item["content"] for item in openai[-2:]], ["cancelled verbatim", "error verbatim"])
        self.assertEqual(
            anthropic[-1]["content"],
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-cancelled",
                    "content": "cancelled verbatim",
                    "is_error": True,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call-error",
                    "content": "error verbatim",
                    "is_error": True,
                },
            ],
        )

    def test_tool_preface_and_parallel_calls_share_the_original_assistant_message(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("同时查", source_id="preface-question")],
            turn_id="preface-tool-turn",
        )
        self.mem.append_entry(
            _intermediate("我一起查一下。", source_id="preface-assistant"),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(_action("preface-a", source_id="preface-action-a"), turn_id=handle.turn_id)
        self.mem.append_entry(_action("preface-b", source_id="preface-action-b"), turn_id=handle.turn_id)

        openai = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads
        anthropic = self.mem.build_context_projection(provider_profile=ANTHROPIC_PROFILE).payloads

        self.assertEqual([item["role"] for item in openai], ["user", "assistant"])
        self.assertEqual(openai[1]["content"], "我一起查一下。")
        self.assertEqual([item["id"] for item in openai[1]["tool_calls"]], ["preface-a", "preface-b"])
        self.assertEqual([item["role"] for item in anthropic], ["user", "assistant"])
        self.assertEqual(anthropic[1]["content"][0], {"type": "text", "text": "我一起查一下。"})
        self.assertEqual(
            [item["id"] for item in anthropic[1]["content"][1:]],
            ["preface-a", "preface-b"],
        )

    def test_environment_media_between_tool_rounds_stays_user_input(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("先加载图片再查资料", source_id="media-loop-user")],
            turn_id="media-between-tools",
        )
        self.mem.append_entry(_action("load-image", source_id="media-load-action"), turn_id=handle.turn_id)
        self.mem.append_entry(
            _observation("load-image", source_id="media-load-result", text="图片已加载"),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            TimelineEntryInput(
                source_id="media-model-input",
                kind="material.model_input",
                origin=EntryOrigin.ENVIRONMENT,
                turn_role=TurnRole.INTERMEDIATE,
                semantic_text="工具为当前模型请求加载了图片：img_001。",
                payload={"items": [{"attachment_handle": "img_001", "mime_type": "image/png"}]},
                semanticize=False,
            ),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(_action("search-next", source_id="media-search-action"), turn_id=handle.turn_id)
        self.mem.append_entry(
            _observation("search-next", source_id="media-search-result", text="查到补充资料"),
            turn_id=handle.turn_id,
        )

        openai = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads
        anthropic = self.mem.build_context_projection(provider_profile=ANTHROPIC_PROFILE).payloads

        self.assertEqual(
            [message["role"] for message in openai],
            ["user", "assistant", "tool", "user", "assistant", "tool"],
        )
        self.assertIn("material.model_input", str(openai[3]["content"]))
        self.assertIsNone(openai[4]["content"])
        self.assertEqual(openai[4]["tool_calls"][0]["id"], "search-next")
        self.assertEqual(
            [message["role"] for message in anthropic],
            ["user", "assistant", "user", "user", "assistant", "user"],
        )
        self.assertIn("material.model_input", str(anthropic[3]["content"]))
        self.assertEqual(anthropic[4]["content"][0]["id"], "search-next")

    def test_open_tool_loop_projection_only_appends_new_batches(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("逐步查", source_id="loop-question")],
            turn_id="open-tool-loop",
        )
        stimulus = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.mem.append_entry(_action("loop-a", source_id="loop-action-a"), turn_id=handle.turn_id)
        self.mem.append_entry(_action("loop-b", source_id="loop-action-b"), turn_id=handle.turn_id)
        actions = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(is_strict_message_prefix(stimulus.payloads, actions.payloads))
        self.assertEqual(len(actions.payloads[-1]["tool_calls"]), 2)

        self.mem.append_entry(
            _observation("loop-b", source_id="loop-result-b", text="B"),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("loop-a", source_id="loop-result-a", text="A"),
            turn_id=handle.turn_id,
        )
        results = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(is_strict_message_prefix(actions.payloads, results.payloads))
        self.assertEqual([item["tool_call_id"] for item in results.payloads[-2:]], ["loop-b", "loop-a"])

        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="第一轮完成",
            provider_output_raw="first final raw",
            memory_annotation={"keywords": ["工具"]},
            annotation_status="accepted",
            source_id="loop-final",
        )
        closed = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.mem.begin_turn(
            stimuli=[_stimulus("普通下一问", source_id="after-tool-question")],
            turn_id="after-tool-turn",
        )
        next_turn = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(is_strict_message_prefix(closed.payloads, next_turn.payloads))
        self.assertEqual(next_turn.payloads[-1]["role"], "user")
        self.assertIn("普通下一问", next_turn.payloads[-1]["content"])


class PrefixAndLedgerTests(ProjectionBase):
    def test_explicit_final_projection_requires_prior_turn_prefix_and_rolls_back(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("missing request ledger", source_id="missing-prefix-user")],
            turn_id="missing-prefix-turn",
        )
        with self.assertRaisesRegex(SchemaError, "projection_source_not_next_append"):
            self.mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text="must not close",
                provider_output_raw="raw",
                memory_annotation={"keywords": ["prefix"]},
                annotation_status="accepted",
                source_id="missing-prefix-final",
                provider_profile=OPENAI_PROFILE,
                provider_projection={"role": "assistant", "content": "raw"},
            )
        self.assertIsNone(self.store.get_entry(namespace=self.namespace, source_id="missing-prefix-final"))
        self.assertEqual(
            self.store.get_turn(namespace=self.namespace, turn_id=handle.turn_id).status,
            TurnStatus.OPEN,
        )

    def test_actual_final_projection_is_atomic_and_history_grows_by_strict_prefix(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("第一问", source_id="prefix-user-1")],
            turn_id="prefix-turn-1",
        )
        user_payload = {"role": "user", "content": "actual-user-1"}
        request = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=user_payload,
                    source_ids=("prefix-user-1",),
                )
            ],
            history_messages=[user_payload],
            attempt=1,
            model_route="gpt",
            system_prefix="stable system",
            tool_schema=[{"name": "memory"}],
        )
        before_final = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(before_final.payloads, (user_payload,))
        self.assertEqual(request.projections[0].projection_status, ProjectionStatus.REQUEST_FROZEN)

        completed = self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="第一答",
            provider_output_raw='{"speech":"第一答"}',
            memory_annotation={"keywords": ["第一问"]},
            annotation_status="accepted",
            source_id="prefix-final-1",
            provider_profile=OPENAI_PROFILE,
            provider_projection={"role": "assistant", "content": '{"speech":"第一答"}'},
        )
        self.assertIsNotNone(completed.final_projection)
        after_final = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(is_strict_message_prefix(before_final.payloads, after_final.payloads))

        self.mem.begin_turn(
            stimuli=[_stimulus("第二问", source_id="prefix-user-2")],
            turn_id="prefix-turn-2",
        )
        next_turn = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(is_strict_message_prefix(after_final.payloads, next_turn.payloads))
        self.assertNotEqual(after_final.stable_prefix_hash, next_turn.stable_prefix_hash)

    def test_request_audits_hash_dynamic_system_and_tools_without_rewriting_history(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("audit", source_id="audit-user")],
            turn_id="audit-turn",
        )
        payload = {"role": "user", "content": "audit"}
        message = ProjectionMessageInput(
            provider_profile=OPENAI_PROFILE,
            payload=payload,
            source_ids=("audit-user",),
        )
        first = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[message],
            history_messages=[payload],
            attempt=1,
            model_route={"model": "gpt"},
            system_prefix="system-v1",
            tool_schema=[{"name": "a"}],
        )
        repeated = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[message],
            history_messages=[payload],
            attempt=1,
            model_route={"model": "gpt"},
            system_prefix="system-v1",
            tool_schema=[{"name": "a"}],
        )
        changed = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[message],
            history_messages=[payload],
            attempt=2,
            model_route={"model": "gpt"},
            system_prefix="system-v2",
            tool_schema=[{"name": "a"}, {"name": "b"}],
        )
        self.assertEqual(first.audit.full_prefix_hash, repeated.audit.full_prefix_hash)
        self.assertNotEqual(first.audit.full_prefix_hash, changed.audit.full_prefix_hash)
        rows = self.store.get_turn_projections(
            namespace=self.namespace,
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].payload, payload)
        self.assertEqual(len(self.store.list_projection_audits(namespace=self.namespace)), 2)

    def test_first_real_request_replaces_unfrozen_open_turn_projection(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("canonical", source_id="replace-user")],
            turn_id="replace-turn",
        )
        canonical = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(len(canonical.messages), 1)
        actual_payload = {"role": "user", "content": "actual provider-visible message"}

        result = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=actual_payload,
                    source_ids=("replace-user",),
                )
            ],
            history_messages=[actual_payload],
            attempt=1,
        )

        self.assertEqual(result.projections[0].payload, actual_payload)
        self.assertEqual(result.projections[0].projection_status, ProjectionStatus.REQUEST_FROZEN)
        rebuilt = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(rebuilt.payloads, (actual_payload,))

    def test_request_audit_can_hash_wire_items_without_changing_chat_projection(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("tool", source_id="wire-user")],
            turn_id="wire-turn",
        )
        chat_payload = {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}]}
        wire_payload = {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"}
        result = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=chat_payload,
                    source_ids=("wire-user",),
                )
            ],
            history_messages=[chat_payload],
            audit_history_messages=[wire_payload],
            attempt=1,
        )

        self.assertEqual(result.projections[0].payload, chat_payload)
        self.assertEqual(result.audit.history_hash, stable_projection_hash([wire_payload]))

    def test_request_projection_cannot_change_after_first_audit(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("first", source_id="locked-user")],
            turn_id="locked-turn",
        )
        first_payload = {"role": "user", "content": "first wire"}
        self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=first_payload,
                    source_ids=("locked-user",),
                )
            ],
            history_messages=[first_payload],
            attempt=1,
        )
        changed_payload = {"role": "user", "content": "changed wire"}
        with self.assertRaisesRegex(SchemaError, "projection_immutable_conflict"):
            self.mem.record_request_projection(
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
                turn_messages=[
                    ProjectionMessageInput(
                        provider_profile=OPENAI_PROFILE,
                        payload=changed_payload,
                        source_ids=("locked-user",),
                    )
                ],
                history_messages=[changed_payload],
                attempt=2,
            )
        stored = self.store.get_turn_projections(
            namespace=self.namespace,
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
        )
        self.assertEqual(stored[0].payload, first_payload)
        self.assertEqual(len(self.store.list_projection_audits(namespace=self.namespace)), 1)

    def test_tool_projections_appended_after_first_audit_freeze_once(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("run tools", source_id="append-user")],
            turn_id="append-tools-turn",
        )
        first = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        first_payload = {"role": "user", "content": "actual current request"}
        self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=first_payload,
                    source_ids=first.messages[0].source_ids,
                    projection_index=first.messages[0].projection_index,
                )
            ],
            history_messages=[first_payload],
            attempt=1,
        )
        self.mem.append_entry(_action("call-late", source_id="append-action"), turn_id=handle.turn_id)
        self.mem.append_entry(
            _observation("call-late", source_id="append-observation", text="late result"),
            turn_id=handle.turn_id,
        )
        expanded = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(len(expanded.messages), 3)
        self.assertEqual(expanded.messages[0].projection_status, ProjectionStatus.REQUEST_FROZEN)
        self.assertNotEqual(expanded.messages[1].projection_status, ProjectionStatus.REQUEST_FROZEN)

        second = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=dict(message.payload),
                    source_ids=message.source_ids,
                    projection_index=message.projection_index,
                    projection_status=message.projection_status,
                    projection_version=message.projection_version,
                )
                for message in expanded.messages
            ],
            history_messages=[dict(message.payload) for message in expanded.messages],
            attempt=2,
        )

        self.assertEqual(
            [item.projection_status for item in second.projections],
            [
                ProjectionStatus.REQUEST_FROZEN,
                ProjectionStatus.REQUEST_FROZEN,
                ProjectionStatus.REQUEST_FROZEN,
            ],
        )
        changed = [dict(message.payload) for message in second.projections]
        changed[-1] = {**changed[-1], "content": "changed late result"}
        with self.assertRaisesRegex(SchemaError, "projection_immutable_conflict"):
            self.mem.record_request_projection(
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
                turn_messages=[
                    ProjectionMessageInput(
                        provider_profile=OPENAI_PROFILE,
                        payload=payload,
                        source_ids=message.source_ids,
                        projection_index=message.projection_index,
                        projection_status=message.projection_status,
                        projection_version=message.projection_version,
                    )
                    for message, payload in zip(second.projections, changed)
                ],
                history_messages=changed,
                attempt=3,
            )
        self.assertEqual(len(self.store.list_projection_audits(namespace=self.namespace)), 2)

    def test_request_projection_mismatch_rolls_back_projection_and_audit(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("mismatch", source_id="mismatch-user")],
            turn_id="mismatch-turn",
        )
        declared = ProjectionMessageInput(
            provider_profile=OPENAI_PROFILE,
            payload={"role": "user", "content": "declared"},
            source_ids=("mismatch-user",),
        )
        with self.assertRaisesRegex(SchemaError, "projection_actual_history_mismatch"):
            self.mem.record_request_projection(
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
                turn_messages=[declared],
                history_messages=[{"role": "user", "content": "actual"}],
                attempt=1,
            )
        self.assertEqual(
            self.store.get_turn_projections(
                namespace=self.namespace,
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
            ),
            [],
        )
        self.assertEqual(self.store.list_projection_audits(namespace=self.namespace), [])

    def test_final_projection_conflict_rolls_back_final_annotation_and_close(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("rollback", source_id="projection-rollback-user")],
            turn_id="projection-rollback-turn",
        )
        payload = {"role": "user", "content": "rollback"}
        self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=payload,
                    source_ids=("projection-rollback-user",),
                    projection_index=0,
                )
            ],
            history_messages=[payload],
            attempt=1,
        )
        with self.assertRaisesRegex(SchemaError, "projection_immutable_conflict"):
            self.mem.complete_turn(
                turn_id=handle.turn_id,
                semantic_text="must roll back",
                provider_output_raw="raw",
                memory_annotation={"keywords": ["pollution"]},
                annotation_status="accepted",
                source_id="projection-rollback-final",
                provider_projection=ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload={"role": "assistant", "content": "conflict"},
                    projection_index=0,
                ),
            )
        self.assertIsNone(self.store.get_entry(namespace=self.namespace, source_id="projection-rollback-final"))
        self.assertEqual(
            self.store.get_entry(
                namespace=self.namespace, source_id="projection-rollback-user"
            ).annotation_status.value,
            "unannotated",
        )
        self.assertEqual(
            self.store.get_turn(namespace=self.namespace, turn_id=handle.turn_id).status,
            TurnStatus.OPEN,
        )


class SafetyAndIsolationTests(ProjectionBase):
    def test_compacted_context_projects_summary_without_silently_dropping_it(self) -> None:
        self.store.add_summary(
            namespace=self.namespace,
            record={
                "summary_id": "legacy-summary",
                "timestamp": 100,
                "diary_summary": "must not disappear silently",
            },
        )
        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(len(projection.messages), 1)
        self.assertTrue(projection.messages[0].turn_id.startswith("summary."))
        self.assertIn("must not disappear silently", str(projection.payloads[0]["content"]))

    def test_media_secret_and_local_path_are_never_persisted_in_projection_payload(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("图片", source_id="media-user")],
            turn_id="media-turn",
        )
        secret_value = "sk-" + "not-a-real-key-123456"
        local_path = "X:" + "\\private\\image.png"
        raw_payload = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAASECRET"}},
                {"type": "text", "text": f"{secret_value} {local_path}"},
            ],
        }
        declared = ProjectionMessageInput(
            provider_profile=OPENAI_PROFILE,
            payload=raw_payload,
            source_ids=("media-user",),
        )
        result = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[declared],
            history_messages=[raw_payload],
            attempt=1,
        )
        persisted = str(result.projections[0].payload)
        self.assertNotIn("AAAASECRET", persisted)
        self.assertNotIn(secret_value, persisted)
        self.assertNotIn(local_path, persisted)
        self.assertIn("omitted from persistent history", persisted)
        self.assertTrue(result.audit.media_omitted)

    def test_system_messages_cannot_be_persisted_as_timeline_projection(self) -> None:
        with self.assertRaisesRegex(SchemaError, "projection_system_message_not_persistable"):
            ProjectionMessageInput(
                provider_profile=OPENAI_PROFILE,
                payload={"role": "system", "content": "do not persist me"},
                source_ids=("source",),
            )

    def test_projection_reads_enforce_namespace_and_conversation(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("private", source_id="private-user")],
            turn_id="private-projection-turn",
        )
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        wrong = Namespace(
            tenant_id="tenant",
            user_id="other",
            domain_id="domain",
            conversation_id="conversation",
        )
        with self.assertRaises(NamespaceError):
            self.store.get_turn_projections(
                namespace=wrong,
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
            )

    def test_legacy_unlinked_entry_gets_safe_synthetic_projection_group(self) -> None:
        self.store.add_message(
            namespace=self.namespace,
            role="user",
            content="legacy message",
            timestamp=100,
            source_id="legacy-source",
        )
        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(len(projection.messages), 1)
        self.assertTrue(projection.messages[0].turn_id.startswith("legacy."))
        self.assertEqual(projection.messages[0].source_ids, ("legacy-source",))


class CanonicalBytesTests(unittest.TestCase):
    def test_canonical_json_ignores_mapping_insertion_order(self) -> None:
        left = {"role": "user", "content": {"b": 2, "a": 1}}
        right = {"content": {"a": 1, "b": 2}, "role": "user"}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))


if __name__ == "__main__":
    unittest.main()
