"""Timeline V2 deterministic rendering, provider adapters, and projection ledger tests."""

from __future__ import annotations

import json
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
    PROJECTION_VERSION,
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
from memcore.projection import sanitize_projection_payload


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
        self.assertIn("content:\n未来事件", content)
        self.assertLess(content.index("a: 1"), content.index("z: 2"))
        self.assertNotIn("data:", content)
        self.assertEqual(first.messages[0].projection_status, ProjectionStatus.CANONICAL_FALLBACK)

    def test_generic_event_deduplicates_structured_semantic_text_and_payload(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                _stimulus(
                    "source: task_workspace\ntask_id: task_1\nstatus: completed\nmessage: 已完成",
                    source_id="task-event",
                    kind="event.task.completed",
                    payload={
                        "message": "已完成",
                        "status": "completed",
                        "task_id": "task_1",
                        "source": "task_workspace",
                    },
                )
            ],
            turn_id="task-event-turn",
        )

        content = str(self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads[0]["content"])

        self.assertEqual(content.count("source: task_workspace"), 1)
        self.assertEqual(content.count("task_id: task_1"), 1)
        self.assertEqual(content.count("status: completed"), 1)
        self.assertEqual(content.count("message: 已完成"), 1)
        self.assertNotIn("content:", content)
        self.assertNotIn("data:", content)

    def test_generic_material_merges_readable_anchor_fields_without_json_copy(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                _stimulus(
                    "source: attachment\nfile_id: file_img_001\nkind: image\nfilename: photo.jpg\n"
                    "mime: image/jpeg\nfile_status: ready\nderived_status: ocr_ready",
                    source_id="material-event",
                    kind="material.reference",
                    payload={
                        "file_id": "file_img_001",
                        "kind": "image",
                        "filename": "photo.jpg",
                        "mime_type": "image/jpeg",
                        "file_status": "ready",
                        "derived_status": "ocr_ready",
                    },
                )
            ],
            turn_id="material-event-turn",
        )

        content = str(self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads[0]["content"])

        self.assertEqual(content.count("file_id: file_img_001"), 1)
        self.assertEqual(content.count("mime: image/jpeg"), 1)
        self.assertNotIn("mime_type:", content)
        self.assertNotIn("content:", content)
        self.assertNotIn("data:", content)

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

    def test_renderer_can_expose_compact_read_view_without_changing_full_projection(self) -> None:
        registry = RendererRegistry()

        def render_full(_entry: Any, _timezone: str) -> str:
            return "full-provider-history"

        def render_compact(_entry: Any, _timezone: str) -> str:
            return "compact-evidence"

        registry.register_prefix(
            "event.custom",
            renderer_id="custom-detail",
            version=1,
            renderer=render_full,
            compact_renderer=render_compact,
        )
        mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="renderer-detail", conversation_id="c"),
            timezone="Asia/Shanghai",
            store=SQLiteMemoryStore(":memory:"),
            index=InMemoryVectorIndex(embedding=HashedEmbeddingProvider()),
            embedding=HashedEmbeddingProvider(),
            renderer_registry=registry,
        )
        try:
            handle = mem.begin_turn(
                stimuli=[_stimulus("event", source_id="detail-entry", kind="event.custom.one")],
                turn_id="detail-turn",
            )

            self.assertEqual(
                registry.render(handle.stimuli[0], timezone="Asia/Shanghai").text,
                "full-provider-history",
            )
            self.assertEqual(
                registry.render_detail(handle.stimuli[0], timezone="Asia/Shanghai", detail="compact").text,
                "compact-evidence",
            )
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
            memory_annotation={"topic_terms": ["两个方向"]},
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
        self.assertIn("speech:\n查完了", payloads[-1]["content"])
        self.assertNotIn("provider_output_raw", payloads[-1]["content"])
        self.assertNotIn("tool_call", payloads[-1]["content"])
        repeated = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(projection.payloads, repeated.payloads)

    def test_typed_assistant_final_keeps_only_model_relevant_voice_state(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="voice-user",
                    kind="message.user.voice",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="继续说。",
                    payload={"text": "继续说。", "voice_turn_id": "voice-turn-1"},
                )
            ],
            turn_id="typed-final-turn",
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="好，我接着说。",
            provider_output_raw="",
            memory_annotation={},
            annotation_status="accepted",
            source_id="voice-assistant",
            kind="message.assistant.voice",
            payload={
                "voice_turn_id": "voice-turn-1",
                "delivery_status": "interrupted",
                "delivered_units": [0],
                "interrupted_units": [1],
            },
        )

        payloads = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads

        self.assertIn("medium: voice", payloads[0]["content"])
        self.assertIn("text:\n继续说。", payloads[0]["content"])
        assistant_content = payloads[1]["content"]
        self.assertIn("medium: voice", assistant_content)
        self.assertIn("delivery: interrupted", assistant_content)
        self.assertIn("delivered_units: [0]", assistant_content)
        self.assertIn("interrupted_units: [1]", assistant_content)
        self.assertIn("speech:\n好，我接着说。", assistant_content)
        self.assertNotIn("voice_turn_id", assistant_content)
        self.assertNotIn("provider_output_raw", payloads[1]["content"])

        anthropic = self.mem.build_context_projection(provider_profile=ANTHROPIC_PROFILE).payloads
        anthropic_content = anthropic[1]["content"][0]["text"]
        self.assertEqual(anthropic_content, assistant_content)

    def test_assistant_json_is_audit_only_while_emotion_and_speech_are_projected(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("小灵聪明", source_id="v5-user")],
            turn_id="v5-final-turn",
            opened_at=1_700_000_000,
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="那可不，链路都顺。",
            provider_output_raw=(
                '{"tool_call":null,"status":"final","choices":[],"emotion":"得意",'
                '"speech":"那可不，链路都顺。","state_request":null}'
            ),
            memory_annotation={},
            annotation_status="accepted",
            timestamp=1_700_000_001,
            source_id="v5-assistant",
            payload={"emotion": "得意"},
        )

        payloads = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads
        assistant_content = payloads[-1]["content"]

        self.assertEqual(payloads[-1]["role"], "assistant")
        self.assertIn("time: 2023-11-15 06:13", assistant_content)
        self.assertIn("emotion: 得意", assistant_content)
        self.assertIn("speech:\n那可不，链路都顺。", assistant_content)
        for obsolete in ("tool_call", "status", "choices", "state_request", "Assistant:"):
            self.assertNotIn(obsolete, assistant_content)

    def test_explicit_v5_migration_rewrites_frozen_chat_once_and_recovers_emotion(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("测试迁移", source_id="migration-user")],
            turn_id="migration-turn",
            opened_at=1_700_000_000,
        )
        self.store.save_turn_projections(
            namespace=self.namespace,
            turn_id=handle.turn_id,
            projections=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload={"role": "user", "content": "[2023-11-15 06:13] User: 测试迁移"},
                    source_ids=("migration-user",),
                    projection_index=0,
                    projection_version=4,
                )
            ],
        )
        raw_final = (
            '{"tool_call":null,"status":"final","choices":[],"emotion":"得意",'
            '"speech":"迁移完成。","state_request":null}'
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="迁移完成。",
            provider_output_raw=raw_final,
            memory_annotation={},
            annotation_status="accepted",
            timestamp=1_700_000_001,
            source_id="migration-final",
            provider_profile=OPENAI_PROFILE,
            provider_projection=ProjectionMessageInput(
                provider_profile=OPENAI_PROFILE,
                payload={"role": "assistant", "content": raw_final},
                source_ids=("migration-final",),
                projection_index=1,
                projection_version=4,
            ),
        )
        before = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(before.payloads[-1]["content"], raw_final)
        generation_before = before.projection_generation

        dry_run = self.mem.migrate_chat_projections_v5(dry_run=True)
        self.assertEqual(dry_run["affected_chat_rows"], 2)
        self.assertEqual(self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads, before.payloads)

        applied = self.mem.migrate_chat_projections_v5()
        after = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(applied["migrated"], 2)
        self.assertEqual(after.projection_version, PROJECTION_VERSION)
        self.assertEqual(after.projection_generation, generation_before + 1)
        self.assertIn("text:\n测试迁移", after.payloads[0]["content"])
        self.assertIn("emotion: 得意", after.payloads[-1]["content"])
        self.assertIn("speech:\n迁移完成。", after.payloads[-1]["content"])
        self.assertEqual(
            self.store.get_entry(namespace=self.namespace, source_id="migration-final").payload["provider_output_raw"],
            raw_final,
        )

        repeated = self.mem.migrate_chat_projections_v5()
        self.assertEqual(repeated["migrated"], 0)
        self.assertEqual(repeated["version_advanced_only"], 0)
        self.assertEqual(
            self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads,
            after.payloads,
        )

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

    def test_structured_tool_result_is_projected_once_without_data_output_echo(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("打开记忆", source_id="structured-question")],
            turn_id="structured-result-turn",
        )
        self.mem.append_entry(_action("call-open", source_id="structured-action"), turn_id=handle.turn_id)
        self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="tool.open_memory.result",
            correlation_id="call-open",
            semantic_text='output:\n{"status":"ok","text":"七月证据"}',
            payload={"output": {"status": "ok", "text": "七月证据"}},
            source_id="structured-result",
            status="success",
            retention_anchor={"operation": "open_memory", "returned_memory_ids": ["episode-july"]},
        )

        openai = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE).payloads
        anthropic = self.mem.build_context_projection(provider_profile=ANTHROPIC_PROFILE).payloads

        expected = '{"status":"ok","text":"七月证据"}'
        self.assertEqual(openai[-1]["content"], expected)
        self.assertEqual(openai[-1]["content"].count("七月证据"), 1)
        self.assertNotIn("data", openai[-1]["content"])
        self.assertEqual(anthropic[-1]["content"][0]["content"], expected)

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
            memory_annotation={"topic_terms": ["工具"]},
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

    def test_open_turn_projection_matches_full_projection_suffix_without_other_history(self) -> None:
        first = self.mem.begin_turn(
            stimuli=[_stimulus("旧问题", source_id="focused-old-user")],
            turn_id="focused-old-turn",
        )
        self.mem.complete_turn(
            turn_id=first.turn_id,
            semantic_text="旧答案",
            provider_output_raw='{"speech":"旧答案"}',
            memory_annotation={},
            annotation_status="accepted",
            source_id="focused-old-final",
        )
        current = self.mem.begin_turn(
            stimuli=[_stimulus("继续查", source_id="focused-current-user")],
            turn_id="focused-current-turn",
        )
        self.mem.append_entry(
            _action("focused-call", source_id="focused-action"),
            turn_id=current.turn_id,
        )
        self.mem.append_entry(
            _observation("focused-call", source_id="focused-result", text="真实结果"),
            turn_id=current.turn_id,
        )

        full = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        focused = self.mem.build_open_turn_projection(
            turn_id=current.turn_id,
            provider_profile=OPENAI_PROFILE,
        )

        full_current = tuple(message.payload for message in full.messages if message.turn_id == current.turn_id)
        self.assertEqual(focused.payloads, full_current)
        self.assertEqual(
            [message.turn_id for message in focused.messages],
            [current.turn_id] * len(focused.messages),
        )
        self.assertNotIn("旧答案", str(focused.payloads))
        self.assertFalse(focused.has_compact_history)

    def test_open_turn_projection_rejects_closed_and_missing_turns(self) -> None:
        closed = self.mem.begin_turn(
            stimuli=[_stimulus("关闭", source_id="focused-closed-user")],
            turn_id="focused-closed-turn",
        )
        self.mem.complete_turn(
            turn_id=closed.turn_id,
            semantic_text="完成",
            provider_output_raw='{"speech":"完成"}',
            memory_annotation={},
            annotation_status="accepted",
            source_id="focused-closed-final",
        )
        with self.assertRaisesRegex(SchemaError, "open_turn_projection_turn_not_open"):
            self.mem.build_open_turn_projection(
                turn_id=closed.turn_id,
                provider_profile=OPENAI_PROFILE,
            )
        with self.assertRaisesRegex(SchemaError, "open_turn_projection_turn_not_found"):
            self.mem.build_open_turn_projection(
                turn_id="missing-turn",
                provider_profile=OPENAI_PROFILE,
            )


class PrefixAndLedgerTests(ProjectionBase):
    def test_non_native_json_or_tag_protocol_can_freeze_its_actual_history(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("列出能力", source_id="custom-user")],
            turn_id="custom-protocol-turn",
        )
        self.mem.append_action(
            turn_id=handle.turn_id,
            kind="operation.catalog.request",
            correlation_id="catalog-1",
            semantic_text='{"action":"list_capabilities"}',
            payload={"action": "list_capabilities"},
            source_id="custom-action",
        )
        self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="operation.catalog.response",
            correlation_id="catalog-1",
            semantic_text="search and weather are available",
            payload={"items": ["search", "weather"]},
            status="success",
            source_id="custom-observation",
        )
        actual = [
            {"role": "user", "content": "列出能力"},
            {"role": "assistant", "content": '<action name="list_capabilities" id="catalog-1" />'},
            {"role": "user", "content": '<result id="catalog-1">search,weather</result>'},
        ]
        frozen = self.mem.record_request_projection(
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
                    zip(actual, ("custom-user", "custom-action", "custom-observation"))
                )
            ],
            history_messages=actual,
            model_route="custom-tag-model",
            system_prefix="stable",
            tool_schema=[],
        )

        self.assertEqual([item.payload for item in frozen.projections], actual)
        before_final = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(list(before_final.payloads), actual)
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="已经列出了可用能力。",
            provider_output_raw="已经列出了可用能力。",
            annotation_status="accepted",
            memory_annotation={"topic_terms": ["能力目录"]},
            source_id="custom-final",
            provider_profile=OPENAI_PROFILE,
            provider_projection={"role": "assistant", "content": "已经列出了可用能力。"},
        )
        after_final = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(is_strict_message_prefix(before_final.payloads, after_final.payloads))

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
                memory_annotation={"topic_terms": ["prefix"]},
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

    def test_request_projection_verifies_noncontiguous_active_message_indexes(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("active question", source_id="indexed-user")],
            turn_id="indexed-active-turn",
        )
        self.mem.append_action(
            turn_id=handle.turn_id,
            kind="tool.read.call",
            correlation_id="indexed-read",
            semantic_text='{"path":"README.md"}',
            payload={"name": "read", "arguments": '{"path":"README.md"}'},
            source_id="indexed-action",
        )
        self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="tool.read.result",
            correlation_id="indexed-read",
            semantic_text="active result",
            payload={"output": "active result"},
            status="success",
            source_id="indexed-result",
        )
        actual = [
            {"role": "user", "content": "active question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "indexed-read",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "user", "content": "standalone material"},
            {"role": "tool", "tool_call_id": "indexed-read", "content": "active result"},
        ]
        result = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=actual[0],
                    source_ids=("indexed-user",),
                    projection_index=0,
                ),
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=actual[1],
                    source_ids=("indexed-action",),
                    projection_index=1,
                ),
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=actual[3],
                    source_ids=("indexed-result",),
                    projection_index=2,
                ),
            ],
            history_messages=actual,
            history_message_indexes=[0, 1, 3],
        )

        self.assertEqual([item.payload for item in result.projections], [actual[0], actual[1], actual[3]])
        with self.assertRaisesRegex(SchemaError, "projection_actual_history_mismatch"):
            self.mem.record_request_projection(
                turn_id=handle.turn_id,
                provider_profile=OPENAI_PROFILE,
                turn_messages=[
                    ProjectionMessageInput(
                        provider_profile=OPENAI_PROFILE,
                        payload=actual[0],
                        source_ids=("indexed-user",),
                        projection_index=0,
                    )
                ],
                history_messages=actual,
                history_message_indexes=[2],
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
            memory_annotation={"topic_terms": ["第一问"]},
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
                memory_annotation={"topic_terms": ["pollution"]},
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
        self.assertIn("omitted from persistent history", persisted)
        self.assertTrue(result.audit.media_omitted)

    def test_plain_user_supplied_path_stays_in_projection(self) -> None:
        path = "C:" + "\\Users\\Public\\Documents\\notes.txt"
        payload, status = sanitize_projection_payload(
            {"role": "user", "content": f"请整理 {path} 和 /opt/akane/notes.txt"}
        )
        self.assertEqual(status, ProjectionStatus.COMPLETE)
        self.assertIn(path, payload["content"])
        self.assertIn("/opt/akane/notes.txt", payload["content"])

    def test_tool_arguments_keep_json_shape_when_private_paths_are_sanitized(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("run", source_id="path-tool-user")],
            turn_id="path-tool-turn",
        )
        arguments = {
            "command": "cd /tmp && python /opt/akane/private.py --out /tmp/result.txt",
            "output_globs": ["result.txt"],
        }
        raw_payload = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-path",
                    "type": "function",
                    "function": {
                        "name": "exec_run",
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }
        result = self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=OPENAI_PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=raw_payload,
                    source_ids=("path-tool-user",),
                )
            ],
            history_messages=[raw_payload],
            attempt=1,
        )

        stored_arguments = result.projections[0].payload["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(stored_arguments)
        self.assertEqual(parsed["output_globs"], ["result.txt"])
        # Executable command evidence is preserved verbatim; no marker or
        # alias text may be injected into the command.
        self.assertEqual(
            parsed["command"],
            "cd /tmp && python /opt/akane/private.py --out /tmp/result.txt",
        )
        self.assertNotIn("[local path omitted", stored_arguments)
        self.assertNotIn("$TMPDIR", stored_arguments)

    def test_legacy_malformed_native_tool_call_downgrades_at_provider_read_boundary(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("run", source_id="legacy-path-user")],
            turn_id="legacy-path-tool-turn",
        )
        self.mem.append_entry(
            _action("legacy-path-call", source_id="legacy-path-action"),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation(
                "legacy-path-call",
                source_id="legacy-path-result",
                text="command completed before the old projection was persisted",
            ),
            turn_id=handle.turn_id,
        )
        self.store.save_turn_projections(
            namespace=self.namespace,
            turn_id=handle.turn_id,
            projections=[
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload={"role": "user", "content": "run"},
                    source_ids=("legacy-path-user",),
                    projection_index=0,
                ),
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "legacy-path-call",
                                "type": "function",
                                "function": {
                                    "name": "exec_run",
                                    "arguments": "[local path omitted from persistent history]",
                                },
                            }
                        ],
                    },
                    source_ids=("legacy-path-action",),
                    projection_index=1,
                ),
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload={
                        "role": "tool",
                        "tool_call_id": "legacy-path-call",
                        "content": "command completed before the old projection was persisted",
                    },
                    source_ids=("legacy-path-result",),
                    projection_index=2,
                ),
            ],
        )

        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        payloads = projection.payloads

        self.assertEqual([payload["role"] for payload in payloads], ["user", "assistant", "user"])
        self.assertNotIn("tool_calls", payloads[1])
        self.assertNotIn("tool_call_id", payloads[2])
        self.assertIn("exec_run", payloads[1]["content"])
        self.assertIn("legacy-path-call", payloads[1]["content"])
        self.assertIn("legacy-path-action", payloads[1]["content"])
        self.assertIn("command completed", payloads[2]["content"])
        self.assertNotIn("[local path omitted from persistent history]", repr(payloads))
        self.assertEqual(projection.messages[1].source_ids, ("legacy-path-action",))
        self.assertEqual(projection.messages[2].source_ids, ("legacy-path-result",))

        repeated = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(projection.payloads, repeated.payloads)
        self.assertEqual(projection.stable_prefix_hash, repeated.stable_prefix_hash)

    def test_unquoted_private_path_with_spaces_does_not_leak_tail(self) -> None:
        payload, status = sanitize_projection_payload(
            {
                "role": "assistant",
                "content": "cat /opt/akane/Private Folder/secret.txt && echo done",
            }
        )

        self.assertEqual(status, ProjectionStatus.COMPLETE)
        self.assertEqual(payload["content"], "cat /opt/akane/Private Folder/secret.txt && echo done")

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
