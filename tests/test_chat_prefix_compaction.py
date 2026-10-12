"""Actual-prefix eligibility, open-tool publication, lineage and append races."""

from __future__ import annotations

import tempfile
import threading
import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from memcore import (
    ChatContextSnapshot,
    LLMClient,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    bind_request_projection_messages,
)
from memcore.chat_compaction import paired_tool_boundary
from tests.test_compaction_v2 import LengthTokenCounter
from tests.test_slice3_settled_history import _begin, _add_tool, _complete, LONG_BODY


class MemoryLLM(LLMClient):
    def __init__(self):
        self.calls = []
        self.during_call = None
        self.fail = False

    def call(self, request):
        self.calls.append(request)
        if self.during_call:
            callback, self.during_call = self.during_call, None
            callback()
        if self.fail:
            return LLMResult(ok=False, data={}, error="provider_unavailable")
        if request.chat_prefix and request.chat_prefix.purpose == "working":
            return LLMResult(
                ok=True, data={"summary": "A has completed. Continue with B; the pending job ID remains valid."}
            )
        if request.task_type.value == "semantic":
            return LLMResult(ok=True, data={"semantic_summary": "long-term continuity"})
        return LLMResult(ok=True, data={"diary_summary": "Completed work and conversation."})


def make_memory(policy="compact_after_terminal", store=None, persona_text="", **overrides):
    llm = MemoryLLM()
    cfg = dict(
        retrieval_mode="navigation",
        operation_projection_policy=policy,
        raw_token_trigger=999999,
        working_token_trigger=1000,
        compaction_request_mode="chat_prefix",
        episodic_compact_trigger_count=99,
        projection_profile="openai_chat",
    )
    cfg.update(overrides)
    mem = MemorySystem(
        llm=llm,
        namespace=Namespace(user_id="u", conversation_id="c"),
        timezone="Asia/Shanghai",
        store=store or SQLiteMemoryStore(),
        token_counter=LengthTokenCounter(),
        config=MemoryConfig(**cfg),
        persona_text=persona_text,
    )
    return mem, llm


def capture(mem):
    projection = mem.build_context_projection(provider_profile="openai_chat")
    messages = (
        {"role": "system", "content": "stable persona/tools " + mem.prompt_overrides.persona_text},
        *projection.payloads,
    )
    groups = ((), *(message.source_ids for message in projection.messages))
    mem.capture_chat_context(ChatContextSnapshot("openai_chat", tuple(messages), tuple(groups), object()))
    return projection


class ChatPrefixCompactionTests(unittest.TestCase):
    def memory(self, **kwargs):
        mem, llm = make_memory(**kwargs)
        self.addCleanup(mem.store.close)
        self.addCleanup(mem.close)
        return mem, llm

    def test_actual_prefix_crops_new_tail_and_rejects_changed_source(self):
        mem, _ = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        projection = capture(mem)
        selected = list(projection.messages)[1:]
        prefix = mem._compaction._select_chat_prefix(selected, "working")
        self.assertIsNotNone(prefix)
        self.assertEqual(prefix.prefix_end, 4)
        self.assertEqual(prefix.source_start, 2)
        from dataclasses import replace

        changed = [replace(selected[0], payload={**selected[0].payload, "content": "changed"}), *selected[1:]]
        self.assertIsNone(mem._compaction._select_chat_prefix(changed, "working"))

    def test_cards_then_summary_then_concurrently_appended_b(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        source = _add_tool(mem, "t", "a", LONG_BODY)
        before = capture(mem)
        llm.during_call = lambda: _add_tool(mem, "t", "b", "B finished after the snapshot")
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "committed", stats)
        after = mem.build_context_projection(provider_profile="openai_chat")
        text = str(after.payloads)
        self.assertIn("compact_reloadable", text)
        summary_index = next(i for i, m in enumerate(after.messages) if "[working summary;" in str(m.payload))
        b_index = next(i for i, m in enumerate(after.messages) if "B finished after" in str(m.payload))
        self.assertLess(summary_index, b_index)
        self.assertEqual(after.payloads[0], before.payloads[0])
        entry = mem.store.get_entries_by_source_ids(namespace=mem.namespace, source_ids=(source,))[0]
        self.assertIn(LONG_BODY, str(entry.to_record()))
        surface = mem.build_context_surface(provider_profile="openai_chat", current_source_id="t-u")
        metadata = [
            {**dict(meta), "payload": dict(payload)}
            for payload, meta in zip(surface.messages, surface.message_projection_metadata)
        ]
        binding = bind_request_projection_messages(metadata, active_turn_id="t")
        self.assertTrue(binding.ok)
        self.assertTrue(all(not m.source_turn_id.startswith("working.") for m in binding.active_group.messages))
        self.assertEqual(mem.build_context_projection(provider_profile="openai_chat").payloads, after.payloads)
        self.assertEqual(len(llm.calls), 1)

    def test_full_history_goes_directly_to_summary_without_cards(self):
        mem, _ = self.memory(policy="full_until_raw_compaction")
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        capture(mem)
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "committed", stats)
        projection = mem.build_context_projection(provider_profile="openai_chat")
        payloads = projection.payloads
        self.assertFalse(projection.has_compact_history)
        self.assertIn("[working summary;", str(payloads))
        self.assertNotIn("compact_reloadable", str(payloads))
        self.assertFalse(any(m.get("role") == "tool" for m in payloads))

    def test_failure_keeps_originals_and_never_publishes_empty_summary(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        before = capture(mem)
        llm.fail = True
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats["reason"], "working_summary_retry_pending")
        self.assertEqual(mem.build_context_projection(provider_profile="openai_chat").payloads, before.payloads)

    def test_pending_native_batch_is_not_cut(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        mem.append_action(
            turn_id="t",
            kind="tool.test.call",
            correlation_id="pending",
            payload={"input": {"content": "x" * 5000}},
            trace_metadata={"tool_name": "test"},
        )
        capture(mem)
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "no_paired_segment")
        self.assertFalse(llm.calls)
        self.assertFalse(paired_tool_boundary([{"role": "assistant", "tool_calls": [{"id": "a"}]}]))

    def test_oversized_summary_keeps_originals(self):
        mem, llm = self.memory(policy="full_until_raw_compaction")
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        before = capture(mem)
        llm.call = lambda request: LLMResult(ok=True, data={"summary": LONG_BODY * 3})
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "no_savings", stats)
        self.assertEqual(mem.build_context_projection(provider_profile="openai_chat").payloads, before.payloads)
        self.assertFalse(mem.store.get_working_segments(namespace=mem.namespace, provider_profile="openai_chat"))

    def test_unsupported_working_commit_preserves_originals(self):
        mem, _ = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        before = capture(mem)
        with patch.object(mem.store, "commit_working_segment", side_effect=NotImplementedError):
            stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "unavailable", stats)
        self.assertEqual(stats.get("reason"), "store_working_compaction_unsupported", stats)
        self.assertEqual(mem.build_context_projection(provider_profile="openai_chat").payloads, before.payloads)

    def test_selected_source_mutation_rejects_overlay(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        sid = _add_tool(mem, "t", "a", LONG_BODY)
        capture(mem)

        def mutate():
            with mem.store._lock, mem.store._conn:
                mem.store._conn.execute("UPDATE messages SET row_version = row_version + 1 WHERE source_id = ?", (sid,))

        llm.during_call = mutate
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "stale_batch", stats)
        self.assertFalse(mem.store.get_working_segments(namespace=mem.namespace, provider_profile="openai_chat"))

    def test_deleted_turn_during_summary_is_not_recreated(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        capture(mem)
        llm.during_call = mem.forget_namespace
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "stale_batch", stats)
        self.assertFalse(mem.store.get_turn(namespace=mem.namespace, turn_id="t"))
        self.assertFalse(mem.store.get_working_segments(namespace=mem.namespace, provider_profile="openai_chat"))

    def test_restart_preserves_summary_position_and_new_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.db")
            store = SQLiteMemoryStore(path)
            mem, _ = make_memory(store=store)
            _begin(mem, "t")
            _add_tool(mem, "t", "a", LONG_BODY)
            capture(mem)
            mem.compact_due_sync(provider_profile="openai_chat")
            _add_tool(mem, "t", "b", "later B")
            expected = mem.build_context_projection(provider_profile="openai_chat").payloads
            mem.close()
            store.close()
            restarted, _ = make_memory(store=SQLiteMemoryStore(path))
            try:
                self.assertEqual(restarted.build_context_projection(provider_profile="openai_chat").payloads, expected)
            finally:
                restarted.close()
                restarted.store.close()

    def test_completed_large_latest_turn_catches_up_in_one_batch(self):
        persona = "fixture character with a diary voice"
        mem, llm = self.memory(raw_token_trigger=1200, working_token_trigger=0, persona_text=persona)
        _begin(mem, "old", "older dialogue" * 80)
        _complete(mem, "old")
        _begin(mem, "new", "large completed task" * 150)
        _complete(mem, "new")
        capture(mem)
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats["status"], "compacted", stats)
        self.assertEqual(stats["source_turn_count"], 2)
        self.assertEqual(stats["summary_request_mode"], "chat_prefix")
        self.assertEqual(len(llm.calls), 1)
        request = llm.calls[0]
        self.assertIn(persona, request.chat_prefix.snapshot.messages[0]["content"])
        self.assertNotIn(persona, request.system_prompt)
        self.assertNotIn("CHARACTER MEMORY SELF", request.system_prompt)
        self.assertIn("diary_summary", request.system_prompt)
        self.assertIn("memory_metadata 标注本次选中的历史内容", request.system_prompt)
        self.assertEqual(mem.compact_due_sync(provider_profile="openai_chat")["status"], "not_due")

    def test_raw_prefix_excludes_runtime_context_inside_selected_range_without_rewriting_sources(self):
        mem, llm = self.memory(raw_token_trigger=100, working_token_trigger=0)
        _begin(mem, "t", "a recorded conversation " * 50)
        _complete(mem, "t")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        messages = (
            {"role": "system", "content": "stable persona/tools"},
            projection.payloads[0],
            {"role": "user", "content": "ephemeral runtime/retrieval context"},
            *projection.payloads[1:],
        )
        groups = ((), projection.messages[0].source_ids, (), *(m.source_ids for m in projection.messages[1:]))
        snapshot = ChatContextSnapshot("openai_chat", messages, groups, object())
        mem.capture_chat_context(snapshot)
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats["summary_request_mode"], "chat_prefix", stats)
        request = llm.calls[0]
        prefix = request.chat_prefix
        self.assertEqual((prefix.source_start, prefix.prefix_end), (1, len(messages)))
        self.assertIn(
            "request messages 3 are background, not memory sources", prefix.instruction(request.system_prompt)
        )
        self.assertEqual(prefix.snapshot.messages, messages)
        self.assertEqual(prefix.snapshot.message_source_ids, groups)
        episode = mem.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=1, cross_conversation=False)[
            0
        ]
        self.assertEqual(set(episode["source_ids"]), {sid for m in projection.messages for sid in m.source_ids})

    def test_provider_wait_does_not_lock_append_or_context_read(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        capture(mem)
        entered, release = threading.Event(), threading.Event()

        def wait():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("fixture timed out")

        llm.during_call = wait
        future = mem.compact_due_background(provider_profile="openai_chat")
        try:
            self.assertTrue(entered.wait(2))
            _add_tool(mem, "t", "b", "B during provider wait")
            projection = mem.build_context_projection(provider_profile="openai_chat")
            self.assertIn("B during provider wait", str(projection.payloads))
            self.assertNotIn("[working summary;", str(projection.payloads))
        finally:
            release.set()
        self.assertEqual(future.result(timeout=3)["working_status"], "committed")
        self.assertIn(
            "B during provider wait", str(mem.build_context_projection(provider_profile="openai_chat").payloads)
        )

    def test_later_segment_does_not_fold_old_sources_twice_and_terminal_compaction_consumes_overlay(self):
        mem, llm = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY)
        capture(mem)
        self.assertEqual(mem.compact_due_sync(provider_profile="openai_chat")["working_status"], "committed")
        _add_tool(mem, "t", "b", LONG_BODY)
        capture(mem)
        self.assertEqual(mem.compact_due_sync(provider_profile="openai_chat")["working_status"], "committed")
        segments = mem.store.get_working_segments(namespace=mem.namespace, provider_profile="openai_chat")
        self.assertEqual(len(segments), 2)
        self.assertFalse(set(segments[0]["source_ids"]).intersection(segments[1]["source_ids"]))
        _complete(mem, "t")
        capture(mem)
        mem.config.raw_token_trigger = 100
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats["status"], "compacted", stats)
        self.assertEqual(stats["summary_request_mode"], "chat_prefix")
        self.assertEqual(len(llm.calls), 3)
        self.assertFalse(mem.store.get_working_segments(namespace=mem.namespace, provider_profile="openai_chat"))
        self.assertEqual(mem.compact_due_sync(provider_profile="openai_chat")["status"], "not_due")

    def test_terminal_fallback_keeps_working_note_before_later_dialogue(self):
        for mode in ("chat_prefix", "independent"):
            with self.subTest(mode=mode):
                persona = "fixture standalone character"
                mem, llm = self.memory(persona_text=persona)
                _begin(mem, "t", "original task")
                _add_tool(mem, "t", "a", LONG_BODY)
                capture(mem)
                self.assertEqual(mem.compact_due_sync()["working_status"], "committed")
                _add_tool(mem, "t", "b", LONG_BODY)
                _complete(mem, "t")
                # No refreshed capture: this is the actual former gap.
                mem.config.compaction_request_mode = mode
                mem.config.raw_token_trigger = 100
                stats = mem.compact_due_sync()
                self.assertEqual(stats["summary_request_mode"], "independent", stats)
                self.assertEqual(stats["status"], "compacted", stats)
                transcript = llm.calls[-1].user_prompt
                self.assertIn(persona, llm.calls[-1].system_prompt)
                self.assertIn("CHARACTER MEMORY SELF", llm.calls[-1].system_prompt)
                note = "A has completed. Continue with B; the pending job ID remains valid."
                self.assertEqual(transcript.count(note), 1)
                self.assertLess(transcript.index("original task"), transcript.index(note))
                self.assertLess(transcript.index(note), transcript.rindex("ok"))
                self.assertNotIn(LONG_BODY, transcript)
                self.assertFalse(
                    mem.store.get_working_segments(namespace=mem.namespace, provider_profile="openai_chat")
                )

    def test_terminal_preparation_is_validated_and_cannot_overwrite_new_capture(self):
        for refresh_kind in ("valid", "stale", "failed", "new_capture"):
            with self.subTest(refresh_kind=refresh_kind):
                mem, llm = self.memory(working_token_trigger=0)
                _begin(mem, "t", "task" * 400)
                capture(mem)
                old = mem._compaction._chat_context
                refresh_calls = []
                latest = []

                def refresh(context, projections):
                    refresh_calls.append(True)
                    if refresh_kind == "failed":
                        raise RuntimeError("fixture refresh failure")
                    if refresh_kind == "stale":
                        return context
                    prepared = replace(
                        context,
                        messages=(context.messages[0], *(m.payload for m in projections)),
                        message_source_ids=((), *(m.source_ids for m in projections)),
                    )
                    if refresh_kind == "new_capture":
                        newer = replace(prepared, transport=object())
                        mem.capture_chat_context(newer)
                        latest.append(newer)
                    return prepared

                mem.capture_chat_context(replace(old, terminal_context_refresh=refresh))
                _complete(mem, "t")
                mem.config.raw_token_trigger = 100
                stats = mem.compact_due_sync()
                self.assertEqual(stats["status"], "compacted", stats)
                expected = "chat_prefix" if refresh_kind in {"valid", "new_capture"} else "independent"
                self.assertEqual(stats["summary_request_mode"], expected, stats)
                self.assertEqual(len(refresh_calls), 1)
                self.assertEqual(len(llm.calls), 1)
                if latest:
                    self.assertIs(mem._compaction._chat_context, latest[0])
                    self.assertIs(llm.calls[0].chat_prefix.snapshot, latest[0])

    def test_semantic_prefix_requires_selected_episodes_to_be_present_and_keeps_lineage(self):
        for visible in (8, 1):
            with self.subTest(visible=visible):
                mem, llm = self.memory(
                    episodic_visible_max=visible,
                    episodic_compact_trigger_count=3,
                    episodic_compact_batch_size=2,
                    working_token_trigger=0,
                    persona_text="fixture semantic character",
                )
                for i in range(3):
                    sid, eid = f"raw-{i}", f"episode-{i}"
                    mem.store.add_message(
                        namespace=mem.namespace, source_id=sid, role="user", content=f"day {i}", timestamp=100 + i
                    )
                    mem.store.add_summary(
                        namespace=mem.namespace,
                        record={
                            "summary_id": eid,
                            "diary_summary": f"day {i} events",
                            "source_ids": [sid],
                            "timestamp": 100 + i,
                        },
                    )
                    mem.store.mark_messages_summarized([sid], eid)
                capture(mem)
                stats = mem.compact_due_sync(provider_profile="openai_chat")
                self.assertEqual(stats["semantic_created"], 1, stats)
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(llm.calls[0].chat_prefix is not None, visible == 8)
                request = llm.calls[0]
                self.assertEqual("fixture semantic character" in request.system_prompt, visible == 1)
                self.assertEqual("CHARACTER MEMORY SELF" in request.system_prompt, visible == 1)
                self.assertIn("memory_metadata 标注本次选中的历史内容", request.system_prompt)
                self.assertIn("stable_facts", request.system_prompt)
                semantic = mem.store.get_recent_semantic_summaries(
                    namespace=mem.namespace, limit=1, cross_conversation=False
                )[0]
                self.assertEqual(semantic["source_summary_ids"], ["episode-0", "episode-1"])

    def test_responses_boundary_and_canonical_source_pairing(self):
        self.assertFalse(paired_tool_boundary([{"type": "function_call", "call_id": "a"}]))
        self.assertTrue(
            paired_tool_boundary(
                [
                    {"type": "function_call", "call_id": "a"},
                    {"type": "function_call_output", "call_id": "a", "output": "done"},
                ]
            )
        )
        mem, llm = self.memory(projection_profile="canonical_user_assistant")
        _begin(mem, "t")
        mem.append_action(
            turn_id="t",
            kind="tool.test.call",
            correlation_id="pending",
            payload={"input": {"content": "x" * 5000}},
            trace_metadata={"tool_name": "test"},
        )
        projection = mem.build_context_projection(provider_profile="canonical_user_assistant")
        mem.capture_chat_context(
            ChatContextSnapshot(
                "canonical_user_assistant",
                tuple(projection.payloads),
                tuple(m.source_ids for m in projection.messages),
                object(),
            )
        )
        stats = mem.compact_due_sync(provider_profile="canonical_user_assistant")
        self.assertEqual(stats.get("working_status"), "no_paired_segment")
        self.assertFalse(llm.calls)

    def test_pending_jobs_created_only_in_b_do_not_enter_a_summary(self):
        mem, _ = self.memory()
        _begin(mem, "t")
        _add_tool(mem, "t", "a", LONG_BODY + " running-job-A")
        _add_tool(mem, "t", "b", "running-job-B was just queued")
        projection = mem.build_context_projection(provider_profile="openai_chat")
        mem.capture_chat_context(
            ChatContextSnapshot(
                "openai_chat",
                tuple(projection.payloads),
                tuple(m.source_ids for m in projection.messages),
                object(),
                ("running-job-A", "running-job-B"),
            )
        )
        stats = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(stats.get("working_status"), "committed", stats)
        summary = next(
            m.payload["content"]
            for m in mem.build_context_projection(provider_profile="openai_chat").messages
            if "[working summary;" in str(m.payload)
        )
        self.assertIn("running-job-A", summary)
        self.assertNotIn("running-job-B", summary)

    def test_v10_database_gains_working_table_without_changing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "v10.db")
            mem, _ = make_memory(store=SQLiteMemoryStore(path))
            _begin(mem, "t")
            _add_tool(mem, "t", "a", LONG_BODY)
            expected = mem.build_context_projection(provider_profile="openai_chat").payloads
            mem.close()
            mem.store.close()
            with sqlite3.connect(path) as connection:
                connection.execute("DROP TABLE working_compaction_segments")
                connection.execute("PRAGMA user_version = 10")
            connection.close()
            restarted, _ = make_memory(store=SQLiteMemoryStore(path))
            try:
                self.assertEqual(restarted.build_context_projection(provider_profile="openai_chat").payloads, expected)
                self.assertFalse(
                    restarted.store.get_working_segments(namespace=restarted.namespace, provider_profile="openai_chat")
                )
            finally:
                restarted.close()
                restarted.store.close()
