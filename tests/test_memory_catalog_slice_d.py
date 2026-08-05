"""Slice D: bounded native raw reads and compact operation receipts."""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    build_memory_operation_receipt,
    dispatch_native_memory_tool,
)


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


def _ts(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 8, 3, hour, minute, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


class MemoryCatalogSliceD(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=Namespace(user_id="group-1", conversation_id="busy-group"),
            timezone="Asia/Shanghai",
            store=self.store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
            config=MemoryConfig(native_timeline_page_token_budget=600),
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()

    def _record_busy_hour(self, count: int = 80) -> None:
        for index in range(count):
            self.mem.record_user_turn(
                f"群聊消息 {index} " + "很多上下文" * 30,
                timestamp=_ts(11, index % 60),
                source_id=f"busy-{index:03d}",
            )

    def test_native_omitted_or_zero_budget_uses_configured_finite_limit(self) -> None:
        seen: list[int] = []

        class _FakeMem:
            config = SimpleNamespace(native_timeline_page_token_budget=321)

            def read_timeline(self, **kwargs):
                seen.append(kwargs["page_token_budget"])
                return {"status": "empty", "reason": "no_timeline_entries", "coverage": {"complete": True}}

        for requested in ({}, {"page_token_budget": 0}, {"page_token_budget": 999}, {"page_token_budget": 100}):
            out = dispatch_native_memory_tool(
                "read_timeline",
                {"date_from": "2026-08-03", **requested},
                mem=_FakeMem(),
            )
            self.assertTrue(out["ok"])

        self.assertEqual(seen, [321, 321, 321, 100])

    def test_busy_native_range_is_partial_with_total_volume_and_lossless_cursor(self) -> None:
        self._record_busy_hour()

        first = dispatch_native_memory_tool(
            "read_timeline",
            {"time_range": {"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"}},
            mem=self.mem,
        )
        result = first["result"]

        self.assertTrue(first["ok"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["reason"], "page_boundary")
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["selected_logical_unit_count"], 80)
        self.assertLess(result["returned_logical_unit_count"], 80)
        self.assertGreater(result["selected_projected_token_count"], result["returned_projected_token_count"])
        self.assertEqual(result["coverage"]["token_count_quality"], "estimated")
        self.assertEqual(result["suggested_next_actions"], ["continue_page", "browse_memory_for_overview"])
        self.assertTrue(result["next_cursor"].startswith("timeline-v1:"))
        self.assertLess(len(result["text"]), 5000)
        self.assertNotIn("messages", result)
        self.assertLess(len(json.dumps(first, ensure_ascii=False)), 12000)

        second = dispatch_native_memory_tool("read_timeline", {"cursor": result["next_cursor"]}, mem=self.mem)
        first_ids = set(first["receipt"]["returned_logical_unit_ids"])
        second_ids = set(second["receipt"]["returned_logical_unit_ids"])
        self.assertTrue(second["ok"])
        self.assertTrue(second_ids)
        self.assertFalse(first_ids & second_ids)
        self.assertEqual(second["result"]["selected_logical_unit_count"], 80)

    def test_precise_small_range_stays_complete_and_direct_zero_remains_unlimited(self) -> None:
        for index in range(3):
            self.mem.record_user_turn(
                f"精确短对话 {index}",
                timestamp=_ts(11, index),
                source_id=f"small-{index}",
            )

        native = dispatch_native_memory_tool(
            "read_timeline",
            {"time_range": {"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"}},
            mem=self.mem,
        )
        direct = self.mem.read_timeline(
            time_range={"start_at": "2026-08-03 11:00", "end_at": "2026-08-03 12:00"},
            page_token_budget=0,
        )

        self.assertEqual(native["result"]["status"], "ok")
        self.assertTrue(native["result"]["coverage_complete"])
        self.assertEqual(native["result"]["message_count"], 3)
        self.assertEqual(direct["status"], "ok")
        self.assertTrue(direct["coverage"]["complete"])
        self.assertEqual(direct["coverage"]["page_token_budget"], 0)
        self.assertEqual(direct["coverage"]["token_count_quality"], "estimated")

    def test_receipt_is_deterministic_and_never_copies_result_bodies_or_sensitive_values(self) -> None:
        arguments = {
            "query": "读取 C:\\Users\\alice\\private.txt 的 sk-abcdefghijklmnopqrstuvwxyz",
            "within_memory_id": "episode-1",
        }
        dispatch_result = {
            "ok": True,
            "status": "ok",
            "tool_name": "retrieve_for_turn",
            "result": {
                "status": "found",
                "matches": [
                    {
                        "source_id": "raw-1",
                        "source_ids": ["raw-1", "raw-2"],
                        "rendered_text": "巨型检索正文" * 1000,
                        "semantic_text": "sk-abcdefghijklmnopqrstuvwxyz C:\\Users\\alice\\private.txt",
                    }
                ],
                "snippets": ["巨型检索正文" * 1000],
            },
        }

        first = build_memory_operation_receipt("retrieve_for_turn", arguments, dispatch_result)
        second = build_memory_operation_receipt("retrieve_for_turn", arguments, dispatch_result)
        blob = json.dumps(first, ensure_ascii=False)

        self.assertEqual(first, second)
        self.assertEqual(first["returned_source_ids"], ["raw-1", "raw-2"])
        self.assertNotIn("巨型检索正文", blob)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", blob)
        self.assertNotIn("C:\\Users\\alice", blob)
        self.assertEqual(len(first["result_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
