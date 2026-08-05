"""Slice B: deterministic catalog browse and lineage-aware memory opening."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import (
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
    TimelineEntryInput,
    TurnRole,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _ts(day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(2026, 8, day, hour, minute, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


class MemoryCatalogNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        embedding = HashedEmbeddingProvider()
        self.store = SQLiteMemoryStore(":memory:")
        self.index = InMemoryVectorIndex(embedding=embedding)
        self.namespace = Namespace(user_id="u1", conversation_id="group-1")
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=self.namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=embedding,
            config=MemoryConfig(),
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()

    def _raw(self, source_id: str, text: str, timestamp: int, **fields: object) -> None:
        self.store.add_message(
            namespace=self.namespace,
            role=str(fields.pop("role", "user")),
            content=text,
            timestamp=timestamp,
            source_id=source_id,
            **fields,
        )

    def _episode(self, summary_id: str, source_ids: list[str], start: int, end: int, title: str) -> None:
        self.store.add_summary(
            namespace=self.namespace,
            record={
                "summary_id": summary_id,
                "timestamp": end,
                "period_start_ts": start,
                "period_end_ts": end,
                "diary_summary": f"{title}的完整摘要",
                "memory_title": title,
                "catalog_hint": f"可回答{title}相关问题",
                "catalog_schema_version": 1,
                "source_ids": source_ids,
            },
        )
        self.store.mark_messages_summarized(source_ids, summary_id)

    def test_browse_returns_lossless_card_pages_and_explicit_raw_coverage(self) -> None:
        self._raw("day22", "二十二日聊了行程", _ts(22, 10))
        self._raw("day23", "二十三日聊了早茶", _ts(23, 11))
        self._raw("live", "二十四日尚未压缩的尾部", _ts(24, 12))
        self._episode("episode-22", ["day22"], _ts(22, 10), _ts(22, 10, 1), "行程安排")
        self._episode("episode-23", ["day23"], _ts(23, 11), _ts(23, 11, 1), "扬州早茶")

        first = self.mem.browse_memory(date_from="2026-08-22", date_to="2026-08-24", page_size=1)
        second = self.mem.browse_memory(cursor=first["next_cursor"])

        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["matched_card_count"], 2)
        self.assertEqual(first["returned_card_count"], 1)
        self.assertFalse(first["page_complete"])
        self.assertTrue(first["next_cursor"].startswith("memory-v1:"))
        self.assertEqual([card["memory_id"] for card in first["cards"]], ["episode-22"])
        self.assertEqual([card["memory_id"] for card in second["cards"]], ["episode-23"])
        self.assertTrue(second["page_complete"])
        self.assertEqual(first["coverage"]["total_source_count"], 3)
        self.assertEqual(first["coverage"]["covered_source_count"], 2)
        self.assertEqual(first["coverage"]["live_source_count"], 1)
        self.assertEqual(first["coverage"]["gap_source_count"], 0)
        self.assertTrue(first["coverage"]["complete"])

    def test_catalog_cursor_cannot_cross_namespace_or_change_selector(self) -> None:
        self._raw("one", "一", _ts(22, 10))
        self._raw("two", "二", _ts(23, 10))
        self._episode("episode-one", ["one"], _ts(22, 10), _ts(22, 10, 1), "一")
        self._episode("episode-two", ["two"], _ts(23, 10), _ts(23, 10, 1), "二")
        first = self.mem.browse_memory(date_from="2026-08-22", date_to="2026-08-23", page_size=1)
        other = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="u1", conversation_id="other"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=HashedEmbeddingProvider(),
            config=MemoryConfig(),
        )

        changed = self.mem.browse_memory(cursor=first["next_cursor"], date_from="2026-08-22")
        out_of_scope = other.browse_memory(cursor=first["next_cursor"])

        self.assertEqual(changed["status"], "invalid_filter")
        self.assertEqual(changed["reason"], "cursor_options_are_embedded")
        self.assertEqual(out_of_scope["status"], "invalid_filter")
        self.assertEqual(out_of_scope["reason"], "invalid_cursor_scope")
        other.close()

    def test_catalog_cursor_uses_stable_key_when_earlier_card_is_inserted(self) -> None:
        self._raw("one", "一", _ts(22, 10))
        self._raw("two", "二", _ts(23, 10))
        self._episode("episode-one", ["one"], _ts(22, 10), _ts(22, 10, 1), "一")
        self._episode("episode-two", ["two"], _ts(23, 10), _ts(23, 10, 1), "二")
        first = self.mem.browse_memory(date_from="2026-08-21", date_to="2026-08-23", page_size=1)
        self._raw("inserted", "后来回填的更早记录", _ts(21, 10))
        self._episode("episode-inserted", ["inserted"], _ts(21, 10), _ts(21, 10, 1), "更早")

        second = self.mem.browse_memory(cursor=first["next_cursor"])

        self.assertEqual([card["memory_id"] for card in first["cards"]], ["episode-one"])
        self.assertEqual([card["memory_id"] for card in second["cards"]], ["episode-two"])

    def test_open_episode_content_then_exact_complete_raw_unit(self) -> None:
        self._raw("question", "和谁一起吃早茶？", _ts(23, 11), turn_id="turn-1", turn_role="stimulus")
        self._raw(
            "answer",
            "和李嘉图一起。",
            _ts(23, 11, 1),
            role="assistant",
            turn_id="turn-1",
            turn_role="final",
        )
        self._episode(
            "episode-breakfast",
            ["question", "answer"],
            _ts(23, 11),
            _ts(23, 11, 1),
            "早茶同行人员",
        )

        content = self.mem.open_memory(memory_id="episode-breakfast", view="content")
        sources = self.mem.open_memory(memory_id="episode-breakfast", view="sources")

        self.assertEqual(content["status"], "ok")
        self.assertIn("早茶同行人员的完整摘要", content["text"])
        self.assertEqual(sources["status"], "ok")
        self.assertEqual(sources["result"]["source_count"], 2)
        self.assertEqual(sources["result"]["logical_unit_count"], 1)
        self.assertEqual(
            [entry["source_id"] for entry in sources["result"]["source_units"][0]["entries"]],
            ["question", "answer"],
        )
        self.assertIn("和李嘉图一起", sources["text"])

    def test_open_episode_sources_default_to_dialogue_with_reloadable_tool_evidence(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="question-with-tool",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="帮我查完以后告诉我结论",
                    timestamp=_ts(23, 13),
                )
            ],
            turn_id="turn-with-tool",
            opened_at=_ts(23, 13),
        )
        self.mem.append_action(
            turn_id=handle.turn_id,
            kind="tool.web_search.call",
            correlation_id="search-breakfast",
            semantic_text="搜索早茶资料",
            payload={"query": "扬州早茶"},
            source_id="search-call",
            timestamp=_ts(23, 13, 1),
        )
        self.mem.append_observation(
            turn_id=handle.turn_id,
            kind="tool.web_search.result",
            correlation_id="search-breakfast",
            semantic_text="巨型工具正文" * 1000,
            payload={"status": "completed", "result": "巨型工具正文" * 1000},
            status="completed",
            source_id="search-result",
            timestamp=_ts(23, 13, 2),
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="结论是和李嘉图一起吃了早茶。",
            provider_output_raw='{"speech":"结论是和李嘉图一起吃了早茶。"}',
            memory_annotation={},
            annotation_status="accepted",
            source_id="answer-with-tool",
            timestamp=_ts(23, 13, 3),
        )
        source_ids = ["question-with-tool", "search-call", "search-result", "answer-with-tool"]
        self._episode(
            "episode-with-tool",
            source_ids,
            _ts(23, 13),
            _ts(23, 13, 3),
            "早茶工具轮",
        )

        sources = self.mem.open_memory(memory_id="episode-with-tool", view="sources")

        self.assertEqual(sources["projection"], "conversation")
        self.assertEqual(sources["result"]["compacted_entry_count"], 2)
        self.assertIn("帮我查完以后告诉我结论", sources["text"])
        self.assertIn("结论是和李嘉图一起吃了早茶", sources["text"])
        self.assertIn("tool.web_search.call", sources["text"])
        self.assertIn("correlation_id: search-breakfast", sources["text"])
        self.assertIn('open_memory(memory_id="search-result"', sources["text"])
        self.assertNotIn("巨型工具正文巨型工具正文", sources["text"])

        expanded = self.mem.open_memory(memory_id="search-result", view="content")
        explicit_full = self.mem.open_memory(
            memory_id="episode-with-tool",
            view="sources",
            projection="full",
        )

        self.assertIn("巨型工具正文巨型工具正文", expanded["text"])
        self.assertIn("巨型工具正文巨型工具正文", explicit_full["text"])

    def test_open_semantic_sources_returns_exact_episode_cards(self) -> None:
        self._raw("source-a", "聊了出行", _ts(22, 10))
        self._episode("episode-a", ["source-a"], _ts(22, 10), _ts(22, 10, 1), "出行")
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={
                "semantic_id": "semantic-a",
                "timestamp": _ts(25),
                "period_start_ts": _ts(22, 10),
                "period_end_ts": _ts(22, 10, 1),
                "semantic_summary": "长期出行主线",
                "source_summary_ids": ["episode-a"],
            },
        )

        opened = self.mem.open_memory(memory_id="semantic-a", view="sources")

        self.assertEqual(opened["status"], "ok")
        self.assertEqual(opened["result"]["source_node_type"], "episodic")
        self.assertEqual([item["memory_id"] for item in opened["result"]["sources"]], ["episode-a"])

    def test_open_sources_cursor_pages_only_between_complete_units(self) -> None:
        self._raw("source-a", "第一轮", _ts(22, 10))
        self._raw("source-b", "第二轮", _ts(22, 11))
        self._episode(
            "episode-two-units",
            ["source-a", "source-b"],
            _ts(22, 10),
            _ts(22, 11),
            "两轮对话",
        )

        first = self.mem.open_memory(memory_id="episode-two-units", view="sources", page_size=1)
        second = self.mem.open_memory(cursor=first["result"]["next_cursor"])

        self.assertFalse(first["result"]["page_complete"])
        self.assertEqual(
            [entry["source_id"] for entry in first["result"]["source_units"][0]["entries"]],
            ["source-a"],
        )
        self.assertEqual(
            [entry["source_id"] for entry in second["result"]["source_units"][0]["entries"]],
            ["source-b"],
        )
        self.assertTrue(second["result"]["page_complete"])

    def test_broken_source_lineage_is_reported_not_silently_dropped(self) -> None:
        self.store.add_summary(
            namespace=self.namespace,
            record={
                "summary_id": "broken",
                "timestamp": _ts(23),
                "diary_summary": "含缺失来源",
                "source_ids": ["missing"],
            },
        )

        opened = self.mem.open_memory(memory_id="broken", view="sources")

        self.assertEqual(opened["status"], "partial")
        self.assertEqual(opened["reason"], "source_lineage_incomplete")
        self.assertEqual(opened["result"]["missing_source_ids"], ["missing"])

    def test_browse_reports_broken_summarized_raw_as_a_real_gap(self) -> None:
        self._raw("orphan", "被错误标记为已摘要", _ts(23, 9))
        self.store.mark_messages_summarized(["orphan"], "missing-summary")

        result = self.mem.browse_memory(date_from="2026-08-23")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["coverage"]["gap_source_count"], 1)
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(result["coverage"]["gap_intervals"][0]["source_count"], 1)

    def test_ambiguous_id_across_truth_layers_is_structured_unavailable(self) -> None:
        self._raw("duplicate", "raw", _ts(23, 10))
        self.store.add_summary(
            namespace=self.namespace,
            record={"summary_id": "duplicate", "timestamp": _ts(23, 10), "diary_summary": "summary"},
        )

        opened = self.mem.open_memory(memory_id="duplicate", view="content")

        self.assertEqual(opened["status"], "unavailable")
        self.assertEqual(opened["reason"], "ambiguous_memory_id")

    def test_read_entry_is_a_raw_only_open_memory_compatibility_adapter(self) -> None:
        self._raw("raw", "原始内容", _ts(23, 12))
        self._episode("episode", ["raw"], _ts(23, 12), _ts(23, 12, 1), "摘要")

        raw = self.mem.read_entry(source_id="raw", detail="full")
        derived = self.mem.read_entry(source_id="episode", detail="full")

        self.assertEqual(raw["status"], "ok")
        self.assertIn("原始内容", raw["text"])
        self.assertEqual(derived["status"], "empty")
        self.assertEqual(derived["reason"], "raw_entry_required")


if __name__ == "__main__":
    unittest.main()
