"""Source batches: speaker evidence, independent cursors and shared limits."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from memcore import (
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TokenCounter,
    TurnRole,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)
from memcore.errors import ConfigError
from memcore.memory_sources import project_source_result
from memcore.memory_navigation import encode_memory_cursor
from memcore.timeline_read import project_timeline_entry


class _NoReadLLM(LLMClient):
    def call(self, request):
        raise AssertionError("exact source navigation must not call an LLM")


class _ByteCounter(TokenCounter):
    def count_text(self, text):
        return len(text.encode("utf-8"))


class MemorySourcesBatchTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.namespace = Namespace(tenant_id="t", user_id="u", domain_id="d", conversation_id="group")
        self.mem = self.system(self.namespace)
        self.addCleanup(self.store.close)

    def system(self, namespace, **options):
        mem = MemorySystem(
            llm=_NoReadLLM(),
            namespace=namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            embedding=self.embedding,
            index=InMemoryVectorIndex(embedding=self.embedding),
            config=MemoryConfig(**options),
        )
        self.addCleanup(mem.close)
        return mem

    def raw(self, sid, *, actor="qq:1", name="小林", text=None, timestamp=100, namespace=None, **fields):
        ns = namespace or self.namespace
        ns = replace(ns, actor=Actor(actor, name) if actor else None)
        return self.store.add_message(
            namespace=ns,
            source_id=sid,
            role=fields.pop("role", "user"),
            content=text or f"{sid} 的完整原话",
            timestamp=timestamp,
            **fields,
        )

    def episode(self, sid, sources, *, namespace=None):
        self.store.add_summary(
            namespace=namespace or self.namespace,
            record={
                "summary_id": sid,
                "timestamp": 300,
                "memory_title": "发言讨论",
                "diary_summary": "摘要正文不应再次喂给模型",
                "source_ids": sources,
            },
        )

    def native(self, **args):
        return dispatch_native_memory_tool("open_memory", args, mem=self.mem)

    @staticmethod
    def entries(result):
        return [entry for unit in result["result"]["source_units"] for entry in unit["entries"]]

    def test_actual_turn_speakers_exclude_assistant_events_and_receivers(self):
        # An assistant may share the namespace Actor. Attribution comes from
        # the actual user entry, never the shared namespace or target Actor.
        self.mem.namespace = replace(self.namespace, actor=Actor("qq:1", "机器人所在域"))
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="第一人的原始发言",
                    source_id="u1",
                    timestamp=100,
                    actor=Actor("qq:1", "旧昵称"),
                    target_actor=Actor("qq:2", "同名"),
                    payload={"text": "第一人的原始发言", "host_note": "不要作为发言返回"},
                ),
                TimelineEntryInput(
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="第二人的原始发言",
                    source_id="u2",
                    timestamp=101,
                    actor=Actor("qq:2", "同名"),
                ),
            ],
            annotation_target_ids=["u1"],
            turn_id="real-turn",
            opened_at=100,
        )
        self.mem.append_entry(
            TimelineEntryInput(
                kind="event.qq.poke",
                origin=EntryOrigin.ENVIRONMENT,
                turn_role=TurnRole.INTERMEDIATE,
                semantic_text="事件不是原始发言",
                source_id="poke",
                timestamp=102,
                actor=Actor("qq:1", "同名"),
                target_actor=Actor("qq:2", "同名"),
            ),
            turn_id=handle.turn_id,
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="助手的回应",
            provider_output_raw="助手的回应",
            source_id="assistant",
            timestamp=103,
        )
        self.raw("renamed", name="新昵称", timestamp=110)
        self.episode("episode", ["u1", "u2", "poke", "assistant", "renamed"])
        with patch.object(self.embedding, "embed_text", side_effect=AssertionError("no read embedding")):
            filtered = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1"])
            both = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1", "qq:2"])
            whole = self.mem.open_memory(memory_id="episode", view="sources")
        self.assertEqual([row["source_id"] for row in self.entries(filtered)], ["u1", "renamed"])
        self.assertEqual([row["source_id"] for row in self.entries(both)], ["u1", "u2", "renamed"])
        self.assertEqual(filtered["read_mode"], "speaker_filtered")
        self.assertIn("旧昵称", filtered["text"])
        self.assertIn("新昵称", filtered["text"])
        self.assertEqual(filtered["text"].count('actor_id="qq:1"'), 2)
        self.assertNotIn("助手的回应", filtered["text"])
        self.assertNotIn("事件不是原始发言", filtered["text"])
        self.assertNotIn("不要作为发言返回", filtered["text"])
        self.assertIn("助手的回应", whole["text"])
        self.assertIn("事件不是原始发言", whole["text"])
        self.assertEqual(whole["read_mode"], "complete_sources")

    def test_exact_ids_do_not_resolve_nicknames_mentions_or_inconsistent_assistant_origin(self):
        self.raw("one", name="同名")
        self.raw("observed", kind="message.user.observed", retrieval_visibility="never", timestamp=105)
        self.raw("voice", kind="message.user.voice", timestamp=106)
        self.raw("two", actor="qq:2", name="同名", target_actor_id="qq:1", timestamp=110)
        self.raw("assistant", role="assistant", kind="message.user", origin="assistant", timestamp=120)
        self.episode("episode", ["one", "observed", "voice", "two", "assistant"])
        for selector in (["同名"], ["qq:"], ["QQ:1"], ["1"]):
            result = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=selector)
            self.assertEqual(result["status"], "empty", result)
            self.assertEqual(result["reason"], "no_matching_speaker_utterances")
        result = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1", "qq:1"])
        self.assertEqual([row["source_id"] for row in self.entries(result)], ["one", "observed", "voice"])

    def test_missing_sources_unknown_identity_and_real_no_match_are_distinct(self):
        self.raw("known", actor="qq:2")
        self.raw("unknown", actor="", timestamp=110)
        for sid, sources, status, reason in (
            ("no-match", ["known"], "empty", "no_matching_speaker_utterances"),
            ("unknown", ["unknown"], "partial", "speaker_identity_incomplete"),
            ("missing", ["gone"], "partial", "source_lineage_incomplete"),
        ):
            # Root IDs must not overlap the raw source IDs.
            root = sid + "-episode"
            self.episode(root, sources)
            result = self.mem.open_memory(memory_id=root, view="sources", speaker_ids=["qq:1"])
            self.assertEqual((result["status"], result["reason"]), (status, reason))
        self.assertEqual(result["result"]["missing_source_ids"], ["gone"])

    def test_batch_order_failure_isolation_and_cross_card_source_body_dedup(self):
        self.raw("shared", text="共享原始发言")
        self.raw("a", text="甲卡片独有发言", timestamp=110)
        self.raw("b", text="乙卡片独有发言", timestamp=120)
        self.episode("episode-a", ["shared", "a"])
        self.episode("episode-b", ["shared", "b"])
        result = self.native(memory_ids=["episode-b", "missing", "episode-a", "episode-b"], view="sources")
        self.assertTrue(result["ok"])
        body = result["result"]
        self.assertEqual([item["memory_id"] for item in body["result"]["items"]], ["episode-b", "missing", "episode-a"])
        self.assertEqual(body["text"].count("共享原始发言"), 1)
        self.assertIn("shared source_ids: shared", body["text"])
        self.assertNotIn("摘要正文不应再次喂给模型", json.dumps(body, ensure_ascii=False))
        self.assertNotIn("共享原始发言", json.dumps(body["result"], ensure_ascii=False))
        self.assertEqual(result["receipt"]["returned_memory_ids"], ["episode-b", "episode-a"])
        self.assertEqual([node["status"] for node in result["receipt"]["source_nodes"]], ["ok", "empty", "ok"])

    def test_each_cursor_freezes_one_tree_and_page_is_standalone(self):
        for i in range(3):
            self.raw(f"raw-{i}", timestamp=100 + i * 10)
        self.episode("one", ["raw-0", "raw-1"])
        self.episode("two", ["raw-0", "raw-2"])
        first = self.native(memory_ids=["one", "two"], view="sources", speaker_ids=["qq:1"], page_size=1)
        items = first["result"]["result"]["items"]
        self.assertFalse(first["result"]["result"]["page_complete"])
        self.assertNotEqual(items[0]["result"]["next_cursor"], items[1]["result"]["next_cursor"])
        for item, expected in zip(items, ("raw-1", "raw-2"), strict=True):
            cursor = item["result"]["next_cursor"]
            continued = self.native(cursor=cursor)
            self.assertEqual(continued["result"]["memory_id"], item["memory_id"])
            self.assertIn(expected + " 的完整原话", continued["result"]["text"])
            self.assertTrue(continued["result"]["result"]["page_complete"])
            self.assertEqual(continued, self.native(cursor=cursor))
            self.assertFalse(self.native(cursor=cursor, speaker_ids=["qq:2"])["ok"])
            self.assertFalse(self.native(cursor=cursor + "x")["ok"])

    def test_cursor_detects_body_actor_membership_changes_and_deleted_root(self):
        self.raw("a")
        self.raw("b", timestamp=110)
        self.episode("episode", ["a", "b"])
        cursor = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1"], page_size=1)["result"][
            "next_cursor"
        ]
        with self.store._conn:
            self.store._conn.execute("UPDATE messages SET semantic_text='改过的原话' WHERE source_id='b'")
        changed = self.mem.open_memory(cursor=cursor)
        self.assertEqual(changed["reason"], "source_changed_restart_required")
        self.assertEqual(changed["suggested_next_actions"][0]["speaker_ids"], ["qq:1"])
        cursor = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1"], page_size=1)["result"][
            "next_cursor"
        ]
        with self.store._conn:
            self.store._conn.execute("UPDATE messages SET actor_id='qq:2' WHERE source_id='b'")
        self.assertEqual(self.mem.open_memory(cursor=cursor)["reason"], "source_changed_restart_required")
        with self.store._conn:
            self.store._conn.execute("UPDATE messages SET actor_id='qq:1' WHERE source_id='b'")
        cursor = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1"], page_size=1)["result"][
            "next_cursor"
        ]
        self.raw("c", timestamp=120)
        with self.store._conn:
            self.store._conn.execute(
                'UPDATE summaries SET source_ids_json=\'["a","b","c"]\' WHERE summary_id=\'episode\''
            )
        self.assertEqual(self.mem.open_memory(cursor=cursor)["reason"], "source_changed_restart_required")
        with self.store._conn:
            self.store._conn.execute("DELETE FROM summaries WHERE summary_id='episode'")
        self.assertEqual(self.mem.open_memory(cursor=cursor)["reason"], "source_changed_restart_required")

    def test_legacy_source_cursor_returns_reload_action_instead_of_losing_its_root(self):
        self.raw("a")
        self.raw("b", timestamp=110)
        self.episode("episode", ["a", "b"])
        old = encode_memory_cursor(
            namespace=self.namespace,
            mode="open_sources",
            page_size=1,
            selector={
                "memory_id": "episode",
                "view": "sources",
                "detail": "full",
                "projection": "conversation",
                "cross_conversation": False,
            },
            last_key=(100, "group", 1, "a"),
        )
        result = self.mem.open_memory(cursor=old)
        self.assertEqual(result["reason"], "source_cursor_upgrade_restart_required")
        reloaded = self.mem.open_memory(**result["suggested_next_actions"][0])
        self.assertEqual(reloaded["status"], "ok")
        self.assertIn("a 的完整原话", reloaded["text"])

    def test_conversation_and_hard_namespace_are_enforced_on_roots_sources_and_cursors(self):
        other = replace(self.namespace, conversation_id="other")
        private = replace(self.namespace, user_id="other-user")
        self.raw("local")
        self.raw("elsewhere", namespace=other, timestamp=110)
        self.raw("private", namespace=private, timestamp=120)
        self.episode("mixed", ["local", "elsewhere", "private"])
        scoped = self.mem.open_memory(memory_id="mixed", view="sources", speaker_ids=["qq:1"])
        self.assertEqual([row["source_id"] for row in self.entries(scoped)], ["local"])
        self.assertEqual(scoped["result"]["missing_source_ids"], ["elsewhere", "private"])
        cross = self.mem.open_memory(
            memory_id="mixed", view="sources", cross_conversation=True, speaker_ids=["qq:1"], page_size=1
        )
        cursor = cross["result"]["next_cursor"]
        continued = self.mem.open_memory(cursor=cursor)
        self.assertEqual([row["source_id"] for row in self.entries(continued)], ["elsewhere"])
        outsider = self.system(private)
        self.assertEqual(outsider.open_memory(cursor=cursor)["reason"], "invalid_cursor_scope")
        self.assertEqual(self.system(other).open_memory(memory_id="mixed", view="sources")["status"], "empty")

    def test_semantic_sources_remain_child_cards_and_unsupported_speaker_filter_is_per_node(self):
        self.raw("raw", text="禁止自动深入的原话")
        self.episode("episode", ["raw"])
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={
                "semantic_id": "semantic",
                "timestamp": 300,
                "semantic_summary": "长期记忆",
                "source_summary_ids": ["episode"],
            },
        )
        result = self.native(memory_ids=["semantic", "episode"], view="sources")
        self.assertEqual(result["result"]["result"]["items"][0]["result"]["sources"][0]["memory_id"], "episode")
        semantic = self.native(memory_id="semantic", view="sources")
        self.assertNotIn("禁止自动深入的原话", json.dumps(semantic, ensure_ascii=False))
        filtered = self.native(memory_ids=["semantic", "episode"], view="sources", speaker_ids=["qq:1"])
        self.assertEqual(filtered["result"]["result"]["items"][0]["reason"], "speaker_filter_requires_episodic_sources")
        self.assertIn("禁止自动深入的原话", filtered["result"]["text"])

    def test_invalid_filters_never_fall_back_to_complete_dialogue_and_strict_null_is_valid(self):
        self.raw("raw")
        self.episode("episode", ["raw"])
        for value in ([], "qq:1", [None], [1], [""], [" "], [" qq:1"], ["qq:1\n"], ["x" * 257]):
            out = self.native(memory_id="episode", view="sources", speaker_ids=value)
            self.assertFalse(out["ok"], out)
            self.assertNotIn("raw 的完整原话", json.dumps(out))
        for options in (
            {"view": "card"},
            {"view": "content"},
            {"view": "sources", "projection": "full"},
            {"view": "sources", "projection": "tools"},
        ):
            self.assertFalse(self.native(memory_id="episode", speaker_ids=["qq:1"], **options)["ok"])
        self.assertTrue(self.native(memory_id="episode", view="sources", speaker_ids=None)["ok"])
        for fmt in ("openai", "openai_responses", "anthropic", "plain"):
            spec = next(
                s
                for s in build_native_memory_tool_specs(tool_format=fmt, include_material_tool=False)
                if (s.get("function") or s).get("name") == "open_memory"
            )
            spec = spec.get("function") or spec
            schema = spec.get("parameters") or spec["input_schema"]
            self.assertIn("speaker_ids", schema["properties"])
        out = self.native(memory_ids=[str(i) for i in range(9)], view="sources")
        self.assertEqual(out["reason"], "sources_batch_exceeds_8_cards")

    def test_shared_token_budget_for_one_four_eight_cards_and_fair_first_groups(self):
        for i in range(8):
            ids = []
            for j in range(2):
                sid = f"raw-{i}-{j}"
                ids.append(sid)
                self.raw(sid, text=f"card {i} utterance {j} " + "原话" * 180, timestamp=100 + i * 20 + j * 10)
            self.episode(f"episode-{i}", ids)
        self.mem.token_counter = _ByteCounter()
        for cards in (1, 4, 8):
            result = self.mem.open_memory(
                memory_ids=[f"episode-{i}" for i in range(cards)], view="sources", page_token_budget=7500
            )
            self.assertLessEqual(result["coverage"]["returned_projected_token_count"], 7500)
            returned = [item["result"]["returned_logical_unit_count"] for item in result["result"]["items"]]
            if 0 in returned:
                first_unread = returned.index(0)
                self.assertTrue(all(count == 0 for count in returned[first_unread:]), returned)
                pending = next(
                    item for item in result["result"]["items"] if not item["result"]["returned_logical_unit_count"]
                )
                self.assertEqual(pending["result"]["suggested_next_actions"][0]["view"], "sources")
            first_groups = self.mem.open_memory(
                memory_ids=[f"episode-{i}" for i in range(cards)], view="sources", page_size=1
            )
            # Frozen budget/page-size fields make continuation tokens slightly
            # larger than an unlimited diagnostic page's tokens.
            enough_for_first_groups = first_groups["coverage"]["returned_projected_token_count"] + 200 * cards
            fair = self.mem.open_memory(
                memory_ids=[f"episode-{i}" for i in range(cards)],
                view="sources",
                page_token_budget=enough_for_first_groups,
            )
            self.assertTrue(all(item["result"]["returned_logical_unit_count"] for item in fair["result"]["items"]))

    def test_one_oversized_complete_group_is_returned_once_with_flag(self):
        self.raw("question", text="完整问题" * 250, turn_id="turn", turn_role="stimulus")
        self.raw("answer", text="完整回答" * 250, role="assistant", timestamp=110, turn_id="turn", turn_role="final")
        self.raw("later", timestamp=120)
        self.episode("big", ["question", "answer", "later"])
        result = self.mem.open_memory(memory_ids=["big"], view="sources", page_token_budget=200)
        self.assertTrue(result["coverage"]["oversized_unit"])
        self.assertEqual(result["result"]["items"][0]["result"]["returned_logical_unit_count"], 1)
        self.assertIn("完整问题" * 250, result["text"])
        self.assertIn("完整回答" * 250, result["text"])
        self.assertNotIn("later 的完整原话", result["text"])

    def test_byte_limit_reports_unreadable_group_without_cutting_other_card(self):
        self.raw("huge", text="不可切断的原话" * 2500)
        self.raw("small", text="另一个卡片仍然成功", timestamp=110)
        self.episode("huge-card", ["huge"])
        self.episode("small-card", ["small"])
        result = self.mem.open_memory(memory_ids=["huge-card", "small-card"], view="sources", page_byte_budget=4000)
        items = result["result"]["items"]
        self.assertEqual(items[0]["reason"], "source_unit_exceeds_byte_budget")
        self.assertEqual(items[0]["suggested_next_actions"][0]["memory_id"], "huge")
        self.assertIn("另一个卡片仍然成功", result["text"])
        self.assertNotIn("不可切断的原话", result["text"])
        self.assertLessEqual(len(json.dumps(project_source_result(result), ensure_ascii=False).encode("utf-8")), 4000)

    def test_native_cursor_cannot_lift_frozen_budget_and_smaller_current_limit_wins(self):
        for i in range(4):
            self.raw(f"r-{i}", text="完整发言" * 100, timestamp=100 + i * 10)
        self.episode("episode", [f"r-{i}" for i in range(4)])
        self.mem.config.native_timeline_page_token_budget = 1500
        cursor = self.native(memory_id="episode", view="sources", page_size=1)["result"]["result"]["next_cursor"]
        self.mem.config.native_timeline_page_token_budget = 5000
        result = self.mem.open_memory(cursor=cursor, page_token_budget=5000)
        self.assertEqual(result["coverage"]["page_token_budget"], 1500)
        self.mem.config.native_timeline_page_token_budget = 900
        result = self.mem.open_memory(cursor=cursor, page_token_budget=900)
        self.assertEqual(result["coverage"]["page_token_budget"], 900)

    def test_headers_batch_only_matching_bodies_once_and_more_than_400_sources(self):
        ids = []
        for i in range(405):
            sid = f"source-{i}"
            ids.append(sid)
            self.raw(sid, actor="qq:1" if i % 200 == 0 else "qq:2", timestamp=100 + i)
        self.episode("one", ids)
        self.episode("two", ids)
        queries = []
        self.store._conn.set_trace_callback(queries.append)
        try:
            result = self.mem.open_memory(memory_ids=["one", "two"], view="sources", speaker_ids=["qq:1"])
        finally:
            self.store._conn.set_trace_callback(None)
        reads = [q for q in queries if q.startswith("SELECT") and ("FROM messages" in q or "JOIN messages" in q)]
        headers = [q for q in reads if q.startswith("SELECT source_id, tenant_id")]
        bodies = [q for q in reads if q.startswith(("SELECT * FROM messages", "SELECT messages.*"))]
        self.assertEqual(len(headers), 2, reads)
        # One is the roots lookup, the other loads three matching bodies for
        # both trees together. Unselected people are absent from the body read.
        self.assertEqual(len(bodies), 2, reads)
        self.assertNotIn("'source-1'", bodies[-1])
        self.assertEqual(result["text"].count("source-0 的完整原话"), 1)
        loaded = self.store.get_entries_by_source_ids(namespace=self.namespace, source_ids=tuple(ids))
        self.assertEqual([entry.source_id for entry in loaded], ids)

    def test_optional_header_store_fallback_still_filters_actual_evidence(self):
        self.raw("one")
        self.raw("two", actor="qq:2", timestamp=110)
        self.episode("episode", ["one", "two"])
        with patch.object(self.store, "get_source_entry_headers", side_effect=NotImplementedError):
            result = self.mem.open_memory(memory_id="episode", view="sources", speaker_ids=["qq:1"])
        self.assertEqual([entry["source_id"] for entry in self.entries(result)], ["one"])

    def test_unread_groups_are_not_projected_and_shared_group_is_projected_once(self):
        sources = []
        for i in range(200):
            sid = f"raw-{i}"
            sources.append(sid)
            self.raw(sid, timestamp=100 + i)
        self.episode("one", sources)
        self.episode("two", sources)
        with patch("memcore.timeline_read.project_timeline_entry", wraps=project_timeline_entry) as projector:
            first = self.mem.open_memory(memory_ids=["one", "two"], view="sources", page_size=1)
            self.assertEqual(projector.call_count, 1)
            self.assertEqual(first["text"].count("raw-0 的完整原话"), 1)
            self.assertNotIn("raw-1 的完整原话", first["text"])
        cursor = first["result"]["items"][0]["result"]["next_cursor"]
        with patch("memcore.timeline_read.project_timeline_entry", wraps=project_timeline_entry) as projector:
            continued = self.mem.open_memory(cursor=cursor)
            self.assertEqual(projector.call_count, 1)
            self.assertIn("raw-1 的完整原话", continued["text"])

    def test_receipt_is_bounded_and_never_copies_lineage_or_bodies(self):
        for i in range(8):
            sources = []
            for j in range(2):
                sid = f"raw-{i}-{j}"
                sources.append(sid)
                self.raw(sid, text="不应该复制进回执的原话", timestamp=100 + j * 10)
            self.episode(f"episode-{i}", sources)
        result = self.native(
            memory_ids=[f"episode-{i}" for i in range(8)], view="sources", speaker_ids=["qq:1"], page_size=1
        )
        serialized = json.dumps(result["receipt"], ensure_ascii=False)
        self.assertLessEqual(len(serialized.encode("utf-8")), 4096)
        self.assertNotIn("不应该复制进回执的原话", serialized)
        self.assertEqual(result["receipt"]["selector"]["speaker_ids"], ["qq:1"])
        self.assertEqual(result["receipt"]["selector"]["read_mode"], "speaker_filtered")
        self.assertTrue(all(item["result"]["next_cursor"] for item in result["result"]["result"]["items"]))
        self.assertTrue(result["receipt"]["navigation_omitted"])

    def test_source_config_rejects_unlimited_native_limits(self):
        for fields in (
            {"native_sources_max_cards": 0},
            {"native_sources_max_cards": 33},
            {"native_sources_max_cards": True},
            {"native_sources_page_utf8_budget": 0},
            {"native_sources_page_utf8_budget": True},
        ):
            with self.assertRaises(ConfigError):
                MemoryConfig(**fields)


if __name__ == "__main__":
    unittest.main()
