"""Host completion without fabricated assistant history."""

from dataclasses import replace

from memcore import MemorySystem, Namespace, NamespaceError, SchemaError, TurnStatus
from tests.test_compaction_v2 import CountingLLM
from tests.test_timeline_v2_turns import TurnLifecycleBase, _stimulus, _tool_action, _tool_result


class CompletionWithoutResponseTests(TurnLifecycleBase):
    def complete(self, turn_id="t", **kwargs):
        return self.mem.complete_turn(
            turn_id=turn_id, semantic_text="", provider_output_raw="", append_final=False, **kwargs
        )

    def test_closes_without_final_row_and_preserves_pairs_annotations_and_idempotence(self):
        handle = self.mem.begin_turn(stimuli=[_stimulus("戳我", source_id="u")], turn_id="t")
        self.mem.append_entry(_tool_action("poke"), turn_id="t")
        pending = self.complete()
        self.assertEqual(pending.status, "pending_actions")
        self.mem.append_entry(_tool_result("poke", "success", "已戳"), turn_id="t")
        completed = self.complete(memory_annotation={"topic_terms": ["戳一戳"]}, annotation_status="accepted_host")
        self.assertTrue(completed.completed)
        self.assertIsNone(completed.final_entry)
        self.assertEqual(completed.updated_targets[0].annotation_status.value, "accepted_host")
        turn = self.store.get_turn(namespace=self.namespace, turn_id=handle.turn_id)
        self.assertEqual(turn.status, TurnStatus.CLOSED)
        self.assertEqual(self.complete().status, "already_completed")
        entries = self.store.get_turn_entries(namespace=self.namespace, turn_id="t")
        self.assertEqual(len(entries), 3)
        self.assertFalse(any(e.turn_role.value == "final" for e in entries))
        self.assertIn("已戳", str(self.mem.build_context_projection(provider_profile="openai_chat").payloads))

    def test_empty_normal_final_and_contradictory_output_remain_errors(self):
        self.mem.begin_turn(stimuli=[_stimulus("hi")], turn_id="t")
        with self.assertRaisesRegex(SchemaError, "empty_final"):
            self.mem.complete_turn(turn_id="t", semantic_text="", provider_output_raw="")
        with self.assertRaisesRegex(SchemaError, "unexpected_final"):
            self.mem.complete_turn(turn_id="t", semantic_text="fake", provider_output_raw="", append_final=False)
        with self.assertRaisesRegex(SchemaError, "unexpected_final"):
            self.complete(provider_projection={"role": "assistant", "content": ""})

    def test_namespace_is_not_bypassed(self):
        self.mem.begin_turn(stimuli=[_stimulus("hi")], turn_id="t")
        original = self.mem.namespace
        self.mem.namespace = Namespace(user_id="other", conversation_id="elsewhere")
        try:
            with self.assertRaises(NamespaceError):
                self.complete()
        finally:
            self.mem.namespace = original

    def test_settlement_and_raw_compaction_accept_closed_turn_without_final(self):
        config = replace(
            self.mem.config,
            operation_projection_policy="compact_after_terminal",
            raw_token_trigger=100,
            compaction_min_recent_turns=1,
        )
        self.mem.close()
        self.mem = MemorySystem(
            llm=CountingLLM(),
            namespace=self.namespace,
            store=self.store,
            index=self.index,
            embedding=self.embedding,
            config=config,
            timezone="Asia/Shanghai",
        )
        self.mem.begin_turn(stimuli=[_stimulus("操作一下", source_id="u")], turn_id="t")
        self.mem.record_tool_exchange(
            turn_id="t",
            tool_name="onebot_action",
            tool_call_id="poke",
            tool_input={"action": "poke"},
            result="成功 " * 3000,
            source="test",
        )
        self.mem.build_context_projection(provider_profile="openai_chat")
        self.assertTrue(self.complete(provider_profile="openai_chat").completed)
        settlement = self.store.get_turn_projection_settlement(
            namespace=self.namespace, turn_id="t", provider_profile="openai_chat"
        )
        self.assertEqual(settlement["settlement_status"], "settled")
        self.assertEqual(settlement["terminal_source_id"], "")
        result = self.mem.compact_due_sync(provider_profile="openai_chat")
        self.assertEqual(result["status"], "compacted", result)
