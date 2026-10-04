"""Completed history imports remain atomic, scoped and append-only."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from memcore import (
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    LLMClient,
    LLMResult,
    MemoryAnnotation,
    MemoryConfig,
    MemorySystem,
    Namespace,
    NamespaceError,
    SQLiteMemoryStore,
    SchemaError,
    TimelineEntryInput,
    TurnRole,
)
from memcore.rendering import render_raw_snippet, render_summary_chat_entry


class NoopLLM(LLMClient):
    def call(self, request):
        return LLMResult(ok=True, data=request.fallback or {})


class CompletedTurnImportTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.mem = MemorySystem(
            llm=NoopLLM(),
            embedding=HashedEmbeddingProvider(),
            store=self.store,
            namespace=Namespace(user_id="group", conversation_id="room", domain_id="character"),
            timezone="Asia/Shanghai",
        )
        self.owner = Actor(stable_id="owner", display_name="主人")
        self.user = TimelineEntryInput(
            source_id="copy.u",
            kind="message.user",
            origin=EntryOrigin.USER,
            turn_role=TurnRole.STIMULUS,
            semantic_text="周末做蛋糕",
            timestamp=1770000000,
            actor=self.owner,
            payload={"channel": "private_app"},
        )
        self.final = TimelineEntryInput(
            source_id="copy.a",
            kind="message.assistant",
            origin=EntryOrigin.ASSISTANT,
            turn_role=TurnRole.FINAL,
            semantic_text="一起做吧",
            timestamp=1770000010,
            target_actor=self.owner,
            payload={"channel": "private_app"},
        )

    def tearDown(self):
        self.mem.close()
        self.store.close()

    def imported(self, **kwargs):
        options = dict(
            turn_id="copy.turn",
            stimuli=[self.user],
            final=self.final,
            provider_output_raw='{"speech":"一起做吧"}',
            annotations=[MemoryAnnotation("copy.u", "accepted_host", {"topic_terms": ["蛋糕"]})],
        )
        options.update(kwargs)
        return self.mem.import_completed_turn(**options)

    def test_import_indexes_both_dialogue_records_and_reply_lineage(self):
        result = self.imported()
        self.assertTrue(result.completed)
        entries = self.mem.get_turn_entries("copy.turn")
        self.assertEqual([entry.seq_no for entry in entries], [1, 2])
        self.assertEqual(entries[1].reply_to_source_id, "copy.u")
        self.assertEqual(entries[1].target_actor, self.owner)
        self.assertEqual([entry.retrieval_visibility.value for entry in entries], ["default", "default"])
        self.assertEqual([entry.index_status for entry in entries], ["indexed", "indexed"])
        self.assertEqual(self.mem.get_turn("copy.turn").status.value, "closed")

    def test_duplicate_and_changed_retry(self):
        self.imported()
        self.assertEqual(self.imported().status, "already_completed")
        self.assertEqual(len(self.mem.get_turn_entries("copy.turn")), 2)
        with self.assertRaisesRegex(SchemaError, "idempotency_conflict"):
            self.imported(stimuli=[replace(self.user, semantic_text="different")])

    def test_deferred_import_keeps_index_outbox_and_supports_normal_repair(self):
        with patch.object(self.mem.embedding, "embed_text", side_effect=AssertionError("must not index inside import")):
            self.assertTrue(self.imported(defer_index=True).completed)
        self.assertEqual(
            [entry.index_status for entry in self.mem.get_turn_entries("copy.turn")], ["pending", "pending"]
        )
        self.mem.reindex_pending(limit=10, batch_size=2)
        self.assertEqual(
            [entry.index_status for entry in self.mem.get_turn_entries("copy.turn")], ["indexed", "indexed"]
        )

    def test_selective_turn_read_excludes_tool_bodies_and_keeps_user_steers(self):
        handle = self.mem.begin_turn(stimuli=[self.user])
        self.mem.append_entry(
            replace(self.user, source_id="steer", turn_role=TurnRole.INTERMEDIATE), turn_id=handle.turn_id
        )
        self.mem.record_tool_exchange(
            turn_id=handle.turn_id,
            tool_name="search",
            tool_call_id="call",
            tool_input={"query": "private tool body"},
            result="tool output",
        )
        self.mem.complete_turn(turn_id=handle.turn_id, semantic_text="final", provider_output_raw="final")
        self.assertEqual(len(self.mem.get_turn_entries(handle.turn_id, roles=(TurnRole.FINAL,))), 1)
        messages = self.mem.get_turn_entries(handle.turn_id, kinds=("message.user", "message.user.voice"))
        self.assertEqual([entry.source_id for entry in messages], ["copy.u", "steer"])
        with self.assertRaisesRegex(SchemaError, "kinds_invalid"):
            self.mem.get_turn_entries(handle.turn_id, kinds=())
        with self.assertRaisesRegex(SchemaError, "roles_invalid"):
            self.mem.get_turn_entries(handle.turn_id, roles=("final",))

    def test_mixed_compaction_keeps_private_source_lineage_and_read_channels(self):
        from tests.test_compaction_v2 import CountingLLM, LengthTokenCounter
        from memcore.llm.base import TaskType

        llm = CountingLLM()
        mem = MemorySystem(
            llm=llm,
            embedding=HashedEmbeddingProvider(),
            store=self.store,
            namespace=self.mem.namespace,
            timezone=self.mem.timezone,
            token_counter=LengthTokenCounter(),
            config=MemoryConfig(
                raw_token_trigger=260,
                raw_token_batch_ratio=0.67,
                compaction_min_recent_turns=1,
                episodic_compact_trigger_count=99,
            ),
        )
        self.addCleanup(mem.close)
        self.imported()
        for index in range(4):
            handle = mem.begin_turn(
                stimuli=[
                    replace(
                        self.user,
                        source_id=f"group.u{index}",
                        actor=Actor("member", "群友"),
                        payload={"channel": "group_app"},
                        semantic_text="群内讨论桌游计划。" * 4,
                        timestamp=1770000030 + index,
                    )
                ]
            )
            mem.complete_turn(
                turn_id=handle.turn_id, semantic_text="群里周末玩桌游。" * 4, provider_output_raw="群里周末玩桌游。" * 4
            )
        with patch.object(llm, "call", wraps=llm.call) as calls:
            result = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(result["status"], "compacted", result)
        self.assertTrue({"copy.u", "copy.a"}.issubset(result["summary_source_ids"]))
        requests = [call.args[0] for call in calls.call_args_list if call.args[0].task_type == TaskType.SUMMARY]
        self.assertTrue(requests)
        rendered = requests[0].user_prompt
        self.assertIn("channel: private_app", rendered)
        self.assertIn("target: 主人", rendered)
        episodes = self.store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        episode = next(row for row in episodes if "copy.u" in row["source_ids"])
        opened = mem.open_memory(memory_id=episode["summary_id"], view="sources")
        self.assertIn("private_app", opened["text"])
        self.assertIn("copy.u", opened["text"])
        self.assertEqual(mem.get_turn("copy.turn").status.value, "closed")

    def test_invalid_completion_rolls_back_the_whole_turn(self):
        with self.assertRaises(SchemaError):
            self.imported(annotations=[])
        self.assertIsNone(self.mem.get_turn("copy.turn"))
        self.assertEqual(self.mem.get_turn_entries("copy.turn"), ())
        self.imported()

    def test_mid_write_failure_does_not_commit_half_import(self):
        original = self.store._add_timeline_entry_locked

        def fail_final(*, namespace, entry):
            if entry.turn_role is TurnRole.FINAL:
                raise RuntimeError("disk_failure")
            return original(namespace=namespace, entry=entry)

        with patch.object(self.store, "_add_timeline_entry_locked", side_effect=fail_final):
            with self.assertRaisesRegex(RuntimeError, "disk_failure"):
                self.imported()
        self.assertIsNone(self.mem.get_turn("copy.turn"))
        self.assertEqual(self.mem.get_turn_entries("copy.turn"), ())

    def test_existing_provider_prefix_is_unchanged_for_late_timestamps(self):
        handle = self.mem.begin_turn(stimuli=[replace(self.user, source_id="group.u", timestamp=1770000030)])
        self.mem.complete_turn(turn_id=handle.turn_id, semantic_text="群里回复", provider_output_raw="群里回复")
        before = self.mem.build_context_projection(provider_profile="openai_chat").payloads
        self.imported()
        after = self.mem.build_context_projection(provider_profile="openai_chat").payloads
        self.assertEqual(after[: len(before)], before)
        self.assertEqual(after[-1]["content"], '{"speech":"一起做吧"}')
        self.assertIn("channel: private_app", after[-2]["content"])

    def test_compaction_and_timeline_preserve_both_sides_source(self):
        self.imported()
        entries = self.mem.get_turn_entries("copy.turn")
        for entry in entries:
            self.assertIn("channel: private_app", render_summary_chat_entry(entry, tz=self.mem.timezone))
        self.assertIn("target: 主人", render_summary_chat_entry(entries[1], tz=self.mem.timezone))
        timeline = self.mem.read_timeline(time_range={"start_at": "2026-02-02", "end_at": "2026-02-04"})
        self.assertIn("private_app", timeline["text"])
        self.assertIn("private_app", render_raw_snippet([entry.to_record() for entry in entries], tz=self.mem.timezone))

    def test_cannot_reuse_a_turn_from_another_namespace(self):
        self.imported()
        other = MemorySystem(
            llm=NoopLLM(),
            embedding=HashedEmbeddingProvider(),
            store=self.store,
            namespace=Namespace(user_id="another", conversation_id="room"),
            timezone="Asia/Shanghai",
        )
        try:
            with self.assertRaises(NamespaceError):
                other.import_completed_turn(
                    turn_id="copy.turn",
                    stimuli=[self.user],
                    final=self.final,
                    annotations=[MemoryAnnotation("copy.u", "missing")],
                )
            with self.assertRaises(NamespaceError):
                other.get_turn_entries("copy.turn")
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
