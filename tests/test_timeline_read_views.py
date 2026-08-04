"""Timeline read views: compact evidence, complete-unit paging, and exact entry expansion."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TokenCounter,
)
from memcore.timeline import EntryOrigin, TimelineEntryInput, TurnRole


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


class _CharCounter(TokenCounter):
    @property
    def quality(self) -> str:
        return "estimated"

    def count_text(self, text: str) -> int:
        return len(text)


def _ts(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 8, 3, hour, minute, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


class TimelineReadViews(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.index = InMemoryVectorIndex(embedding=self.embedding)
        self.mem = self._mem("c1")

    def tearDown(self) -> None:
        self.store.close()

    def _mem(self, conversation: str, *, user: str = "u1", counter: TokenCounter | None = None) -> MemorySystem:
        return MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id=user, conversation_id=conversation),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.embedding,
            token_counter=counter,
        )

    def _record_large_tool_turn(self, *, turn_id: str = "tool-turn") -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="question",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="帮我查一下",
                    timestamp=_ts(11, 5),
                )
            ],
            turn_id=turn_id,
            opened_at=_ts(11, 5),
        )
        self.mem.append_action(
            turn_id=handle.turn_id,
            kind="tool.web_search.call",
            correlation_id="search-1",
            semantic_text="搜索同行信息",
            payload={"query": "同行信息"},
            source_id="search-call",
            timestamp=_ts(11, 5),
        )
        self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="tool.web_search.result",
            correlation_id="search-1",
            semantic_text="巨型搜索正文" * 1000,
            payload={"status": "completed", "result": "巨型搜索正文" * 1000},
            status="completed",
            source_id="search-result",
            timestamp=_ts(11, 6),
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="查完了",
            provider_output_raw='{"speech":"查完了"}',
            memory_annotation={},
            annotation_status="accepted",
            source_id="tool-final",
            timestamp=_ts(11, 7),
        )

    def test_conversation_projection_compacts_large_tool_result_but_keeps_later_dialogue(self) -> None:
        self._record_large_tool_turn()
        self.mem.record_user_turn(
            "misaka 和李嘉图一起来玩的",
            timestamp=_ts(11, 47),
            source_id="target",
        )

        result = self.mem.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            projection="conversation",
        )

        self.assertEqual(result["status"], "ok")
        self.assertIn("misaka 和李嘉图一起来玩的", result["text"])
        self.assertIn("source_id: search-result", result["text"])
        self.assertIn("read_entry", result["text"])
        self.assertNotIn("巨型搜索正文巨型搜索正文", result["text"])
        self.assertEqual(result["coverage"]["compacted_entry_count"], 2)
        self.assertTrue(result["coverage"]["complete"])
        self.assertEqual(result["coverage"]["next_cursor"], "")

        expanded = self.mem.read_entry(source_id="search-result", detail="full")
        self.assertEqual(expanded["status"], "ok")
        self.assertIn("巨型搜索正文巨型搜索正文", expanded["text"])

    def test_full_and_tools_projections_are_explicit_and_deterministic(self) -> None:
        self._record_large_tool_turn()
        self.mem.record_user_turn("后续对话", timestamp=_ts(11, 47), source_id="later")

        full = self.mem.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            projection="full",
        )
        tools = self.mem.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            projection="tools",
        )

        self.assertIn("巨型搜索正文巨型搜索正文", full["text"])
        self.assertEqual(
            [entry["source_id"] for entry in tools["messages"]],
            ["search-call", "search-result"],
        )
        self.assertNotIn("后续对话", tools["text"])
        self.assertEqual(tools["projection"], "tools")

    def test_explicit_page_budget_never_splits_a_turn_and_cursor_is_lossless(self) -> None:
        paged = self._mem("paged", counter=_CharCounter())
        self.mem = paged
        self._record_large_tool_turn(turn_id="oversized-turn")
        paged.record_user_turn("下一完整单元", timestamp=_ts(11, 47), source_id="next-unit")

        first = paged.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            projection="full",
            page_token_budget=50,
        )
        second = paged.read_timeline(cursor=first["coverage"]["next_cursor"])

        self.assertEqual(
            [entry["source_id"] for entry in first["messages"]],
            ["question", "search-call", "search-result", "tool-final"],
        )
        self.assertTrue(first["coverage"]["oversized_unit"])
        self.assertFalse(first["coverage"]["complete"])
        self.assertTrue(first["coverage"]["next_cursor"].startswith("timeline-v1:"))
        self.assertEqual([entry["source_id"] for entry in second["messages"]], ["next-unit"])
        self.assertTrue(second["coverage"]["complete"])
        self.assertEqual(second["coverage"]["next_cursor"], "")

    def test_cursor_tampering_and_cross_namespace_reuse_are_rejected(self) -> None:
        paged = self._mem("paged", counter=_CharCounter())
        self.mem = paged
        self._record_large_tool_turn(turn_id="oversized-turn")
        paged.record_user_turn("下一完整单元", timestamp=_ts(11, 47), source_id="next-unit")
        first = paged.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            projection="full",
            page_token_budget=50,
        )
        cursor = first["coverage"]["next_cursor"]
        tampered = cursor[:-1] + ("a" if cursor[-1] != "a" else "b")

        bad = paged.read_timeline(cursor=tampered)
        other = self._mem("other", counter=_CharCounter()).read_timeline(cursor=cursor)

        self.assertEqual(bad["status"], "invalid_filter")
        self.assertEqual(bad["reason"], "invalid_cursor")
        self.assertEqual(other["status"], "invalid_filter")
        self.assertEqual(other["reason"], "invalid_cursor_scope")

    def test_no_page_budget_has_no_hidden_truncation(self) -> None:
        self._record_large_tool_turn()
        self.mem.record_user_turn("最后一条", timestamp=_ts(11, 47), source_id="last")

        result = self.mem.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            projection="full",
        )

        self.assertTrue(result["coverage"]["complete"])
        self.assertEqual(result["coverage"]["next_cursor"], "")
        self.assertEqual(result["coverage"]["entry_count"], 5)

    def test_read_entry_is_current_conversation_only_and_sanitizes_model_visible_data(self) -> None:
        self.mem.record_user_turn(
            "凭据 sk-abcdefghijklmnopqrstuvwxyz 和 C:\\Users\\alice\\secret.txt",
            timestamp=_ts(11),
            source_id="safe-entry",
        )
        other = self._mem("other")

        own = self.mem.read_entry(source_id="safe-entry", detail="full")
        out_of_scope = other.read_entry(source_id="safe-entry", detail="full")

        self.assertEqual(own["status"], "ok")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", own["text"])
        self.assertNotIn("C:\\Users\\alice", own["text"])
        self.assertEqual(out_of_scope["status"], "empty")
        self.assertEqual(out_of_scope["reason"], "entry_not_found_or_out_of_scope")


if __name__ == "__main__":
    unittest.main()
