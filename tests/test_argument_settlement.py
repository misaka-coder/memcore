"""Native structure, immutable readback and frozen terminal argument settlement."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from memcore import ConfigError, MemoryConfig, Namespace, SQLiteMemoryStore, SchemaError
from memcore.settlement import build_settlement_plan, compact_argument_value, settlement_config_hash
from tests.test_settlement_threshold_config import _complete, _shared_mem, _turn_with_tools

POLICY = "compact_arguments_after_terminal"


class ArgumentSettlementTests(unittest.TestCase):
    def test_parallel_calls_and_reversed_failed_results_preserve_pairing(self):
        for profile in ("openai_chat", "anthropic_messages"):
            with self.subTest(profile=profile):
                mem, store = _shared_mem(policy=POLICY)
                self.addCleanup(store.close)
                self.addCleanup(mem.close)
                _turn_with_tools(mem, results=[])
                for call_id in ("a", "b"):
                    mem.append_action(
                        turn_id="t1",
                        kind="tool.write.call",
                        correlation_id=call_id,
                        source_id=f"action-{call_id}",
                        payload={"input": {"content": call_id * 2000}},
                        trace_metadata={"tool_name": "write"},
                    )
                for call_id in ("b", "a"):
                    mem.append_observation(
                        turn_id="t1",
                        kind="tool.write.result",
                        correlation_id=call_id,
                        source_id=f"result-{call_id}",
                        payload={"output": "failure " * 300},
                        status="failed",
                    )
                _complete(mem, "t1", profile=profile)
                projection = mem.build_context_projection(provider_profile=profile)
                if profile == "openai_chat":
                    calls = next(m.payload["tool_calls"] for m in projection.messages if m.payload.get("tool_calls"))
                    self.assertEqual([call["id"] for call in calls], ["a", "b"])
                    self.assertTrue(all("truncated" in call["function"]["arguments"] for call in calls))
                    observations = [m.payload for m in projection.messages if m.payload.get("role") == "tool"]
                    self.assertEqual([p["tool_call_id"] for p in observations], ["b", "a"])
                    self.assertTrue(all("status: failed" in p["content"] for p in observations))
                else:
                    calls = [
                        b
                        for m in projection.messages
                        for b in m.payload.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    ]
                    self.assertEqual([b["id"] for b in calls], ["a", "b"])
                    observations = [
                        b
                        for m in projection.messages
                        for b in m.payload.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    ]
                    self.assertEqual([b["tool_use_id"] for b in observations], ["b", "a"])
                    self.assertTrue(all(b["is_error"] for b in observations))

    def test_malformed_wire_falls_back_and_aborted_turn_does_not_settle(self):
        mem, store, _, _ = self.open_long_turn()
        mem.build_context_projection(provider_profile="openai_chat")
        # Simulate a structurally unusable frozen native argument string.
        ledger = store._conn.execute(
            "SELECT projection_id,payload_json FROM prompt_projections WHERE turn_id='t1'"
        ).fetchall()
        from memcore import stable_projection_hash

        for row in ledger:
            payload = json.loads(row["payload_json"])
            if payload.get("tool_calls"):
                payload["tool_calls"][0]["function"]["arguments"] = "not json"
                store._conn.execute(
                    "UPDATE prompt_projections SET payload_json=?,payload_hash=? WHERE projection_id=?",
                    (json.dumps(payload), stable_projection_hash(payload), row["projection_id"]),
                )
        store._conn.commit()
        _complete(mem, "t1")
        settlement = store.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t1", provider_profile="openai_chat"
        )
        self.assertEqual(settlement["settlement_status"], "full_fallback")
        self.assertIn("arguments_invalid", settlement["reason"])
        self.assertFalse(mem.build_context_projection(provider_profile="openai_chat").has_compact_history)
        mem2, store2, _, _ = self.open_long_turn()
        mem2.abort_turn(turn_id="t1", reason="cancelled")
        self.assertIsNone(
            store2.get_turn_projection_settlement(
                namespace=mem2.namespace, turn_id="t1", provider_profile="openai_chat"
            )
        )

    def test_v9_migration_preserves_old_settlement_and_generic_store_history_flag(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = Path(directory.name) / "v9.db"
        mem, store, _, _ = self.open_long_turn(
            store=SQLiteMemoryStore(str(database)),
            config=MemoryConfig(operation_projection_policy="compact_after_terminal"),
        )
        _complete(mem, "t1")
        row = store.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t1", provider_profile="openai_chat"
        )
        mem.close()
        store.close()
        with sqlite3.connect(database) as connection:
            self.addCleanup(connection.close)
            for table, names in {
                "turns": ("operation_argument_min_utf8_bytes", "operation_argument_preview_chars"),
                "turn_projection_settlement": (
                    "settlement_argument_min_utf8_bytes",
                    "settlement_argument_preview_chars",
                ),
            }.items():
                for name in names:
                    connection.execute(f"ALTER TABLE {table} DROP COLUMN {name}")
            connection.execute("PRAGMA user_version = 9")
        reopened = SQLiteMemoryStore(str(database))
        self.addCleanup(reopened.close)
        updated = reopened.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t1", provider_profile="openai_chat"
        )
        for key, value in row.items():
            self.assertEqual(updated[key], value)
        self.assertEqual(reopened._conn.execute("PRAGMA user_version").fetchone()[0], 10)
        mem2, store2, _, _ = self.open_long_turn()
        _complete(mem2, "t1")

        def no_batch(**kwargs):
            raise NotImplementedError

        store2.get_context_projection_rows = no_batch
        self.assertTrue(mem2.build_context_projection(provider_profile="openai_chat").has_compact_arguments_history)

    def open_long_turn(self, *, store=None, config=None):
        mem, store = _shared_mem(store=store, config=config or MemoryConfig(operation_projection_policy=POLICY))
        self.addCleanup(store.close)
        self.addCleanup(mem.close)
        _turn_with_tools(mem, results=[])
        args = {
            "path": "notes.md",
            "content": "中文正文" * 500,
            "mode": "create",
            "count": 2,
            "nested": {"patch": "+line\n" * 400},
            "array": ["keep" * 500],
        }
        exchange = mem.record_tool_exchange(
            turn_id="t1",
            tool_name="workspace_write",
            tool_call_id="call1",
            tool_input=args,
            result="saved",
            source_id_prefix="long-action",
        )
        return mem, store, args, exchange["tool_use"]["source_id"]

    def test_native_shape_action_only_readback_and_open_turn_full(self):
        mem, store, args, source_id = self.open_long_turn()
        open_projection = mem.build_context_projection(provider_profile="openai_chat")
        call = next(m for m in open_projection.messages if m.payload.get("tool_calls"))
        self.assertEqual(json.loads(call.payload["tool_calls"][0]["function"]["arguments"]), args)
        _complete(mem, "t1")
        row = store.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t1", provider_profile="openai_chat"
        )
        self.assertEqual(row["settlement_status"], "settled")
        self.assertLess(row["settled_projected_tokens"], row["full_projected_tokens"])
        closed = mem.build_context_projection(provider_profile="openai_chat")
        call = next(m for m in closed.messages if m.payload.get("tool_calls"))
        native = call.payload["tool_calls"][0]
        compact = json.loads(native["function"]["arguments"])
        self.assertEqual(native["id"], "call1")
        self.assertEqual(native["function"]["name"], "workspace_write")
        for key in ("path", "mode", "count", "array"):
            self.assertEqual(compact[key], args[key])
        self.assertIn(f"[truncated source_id={source_id}]", compact["content"])
        self.assertIn(f"[truncated source_id={source_id}]", compact["nested"]["patch"])
        self.assertEqual(store.get_entry(namespace=mem.namespace, source_id=source_id).payload["input"], args)
        opened = mem.open_memory(memory_id=source_id, view="content", detail="full")
        self.assertEqual(opened["status"], "ok")
        self.assertIn(args["content"], str(opened))
        self.assertTrue(closed.has_compact_arguments_history)
        self.assertEqual(
            store.get_retrieval_records(
                namespace=Namespace(user_id="u1", conversation_id="other"), source_ids=(source_id,)
            ),
            {},
        )
        self.assertEqual(closed.payloads, mem.build_context_projection(provider_profile="openai_chat").payloads)
        _turn_with_tools(mem, turn_id="next", results=[])
        next_projection = mem.build_context_projection(provider_profile="openai_chat")
        self.assertEqual(next_projection.payloads[: len(closed.payloads)], closed.payloads)

    def test_byte_threshold_preview_and_strict_half_savings(self):
        for value in ("x" * 1023, "中" * 341):
            self.assertFalse(compact_argument_value(value, source_id="a")[1])
        self.assertTrue(compact_argument_value("x" * 1024, source_id="a")[1])
        preview = "x" + "\n[truncated source_id=a]"
        exact_half = "x" * (len(preview.encode()) * 2)
        self.assertFalse(compact_argument_value(exact_half, source_id="a", min_utf8_bytes=1, preview_chars=1)[1])
        self.assertTrue(compact_argument_value(exact_half + "x", source_id="a", min_utf8_bytes=1, preview_chars=1)[1])
        for field in ("operation_argument_min_utf8_bytes", "operation_argument_preview_chars"):
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ConfigError):
                    MemoryConfig(**{field: value})

    def test_native_parallel_extensions_unchanged_calls_and_signatures(self):
        for profile in ("openai_chat", "anthropic_messages", "canonical_user_assistant"):
            with self.subTest(profile=profile):
                mem, store, _, _ = self.open_long_turn()
                mem.record_tool_exchange(
                    turn_id="t1",
                    tool_name="search",
                    tool_call_id="short",
                    tool_input={"query": "small"},
                    result="ok",
                    source_id_prefix="short",
                )
                projection = mem.build_context_projection(provider_profile=profile)
                originals = list(projection.messages)
                if profile == "openai_chat":
                    for index, message in enumerate(originals):
                        if message.payload.get("tool_calls"):
                            payload = {**message.payload, "provider_extension": {"custom": 7}}
                            for call in payload["tool_calls"]:
                                if call["id"] == "short":
                                    call["function"]["arguments"] = '{ "query" : "small" }'
                            originals[index] = replace(message, payload=payload)
                plan = build_settlement_plan(
                    mem._projection,
                    store.get_turn_entries(namespace=mem.namespace, turn_id="t1"),
                    provider_profile=profile,
                    authoritative_messages=originals,
                    compact_arguments=True,
                )
                self.assertEqual(plan.settlement_status, "settled")
                if profile == "openai_chat":
                    calls = [m.payload for m in plan.messages if m.payload.get("tool_calls")]
                    self.assertTrue(all(p["provider_extension"] == {"custom": 7} for p in calls))
                    short = next(c for p in calls for c in p["tool_calls"] if c["id"] == "short")
                    self.assertEqual(short["function"]["arguments"], '{ "query" : "small" }')
                    signed = [
                        replace(m, payload={**m.payload, "extra_content": {"thought_signature": "opaque"}})
                        if m.payload.get("tool_calls")
                        else m
                        for m in originals
                    ]
                    with self.assertRaisesRegex(SchemaError, "signed_action"):
                        build_settlement_plan(
                            mem._projection,
                            store.get_turn_entries(namespace=mem.namespace, turn_id="t1"),
                            provider_profile=profile,
                            authoritative_messages=signed,
                            compact_arguments=True,
                        )

    def test_frozen_settings_restart_and_idempotence(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = Path(directory.name) / "memory.db"
        config = MemoryConfig(operation_projection_policy=POLICY, operation_argument_preview_chars=37)
        mem, store, _, _ = self.open_long_turn(store=SQLiteMemoryStore(str(database)), config=config)
        mem.config.operation_argument_min_utf8_bytes = 99999
        mem.config.operation_argument_preview_chars = 200
        _complete(mem, "t1")
        before = mem.build_context_projection(provider_profile="openai_chat").payloads
        row = store.get_turn_projection_settlement(
            namespace=mem.namespace, turn_id="t1", provider_profile="openai_chat"
        )
        self.assertEqual(row["settlement_argument_preview_chars"], 37)
        self.assertEqual(row["settlement_argument_min_utf8_bytes"], 1024)
        _complete(mem, "t1")
        self.assertEqual(before, mem.build_context_projection(provider_profile="openai_chat").payloads)
        mem.close()
        store.close()
        reopened, store2 = _shared_mem(store=SQLiteMemoryStore(str(database)), policy="full_until_raw_compaction")
        self.addCleanup(store2.close)
        self.addCleanup(reopened.close)
        self.assertEqual(before, reopened.build_context_projection(provider_profile="openai_chat").payloads)
        self.assertTrue(reopened.build_context_surface(provider_profile="openai_chat").has_compact_arguments_history)
        handle = store2.get_turn(namespace=reopened.namespace, turn_id="t1")
        self.assertEqual(handle.operation_argument_preview_chars, 37)
        # Existing policy fingerprints must keep their published algorithm.
        old_hash = hashlib.sha256(b"compact_after_terminal|256|0.500000").hexdigest()
        self.assertEqual(
            settlement_config_hash(
                policy="compact_after_terminal", min_utf8_bytes=256, saved_ratio=0.5, argument_preview_chars=37
            ),
            old_hash,
        )
