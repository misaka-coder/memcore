"""Evidence-bound ID and timestamp comparison for real-host replay."""

from __future__ import annotations

import copy
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from examples.research_pilot.akane_replay import ReplayMapper, ReplayMismatch

_OLD_TS = int(datetime(2026, 9, 6, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
_NEW_TS = _OLD_TS + 3600


def memory_row(
    source_id: str,
    *,
    kind: str = "message.assistant",
    timestamp: int = _OLD_TS,
    seq_no: int = 1,
) -> dict:
    return {
        "source_id": source_id,
        "timestamp": timestamp,
        "kind": kind,
        "turn_role": (
            "terminal"
            if kind == "message.assistant"
            else "observation"
            if kind == "tool.result"
            else "action"
            if kind == "tool.call"
            else "stimulus"
        ),
        "turn_id": "turn-stable-1",
        "correlation_id": "call-stable-1" if kind.startswith("tool.") else None,
        "seq_no": seq_no,
    }


def projection(source_id: str, payload: dict, *, index: int = 0) -> dict:
    return {
        "source_ids": [source_id],
        "turn_id": "turn-stable-1",
        "projection_index": index,
        "projection_status": "complete",
        "projection_version": 1,
        "payload": copy.deepcopy(payload),
    }


def snapshot(rows: list[dict], payloads: list[dict] | None = None) -> dict:
    if payloads is None:
        payloads = [{"role": "assistant", "content": "Exact original synthetic answer"} for _ in rows]
    result = {
        "memory": copy.deepcopy(rows),
        "projection": [
            projection(row["source_id"], payload, index=index)
            for index, (row, payload) in enumerate(zip(rows, payloads, strict=True))
        ],
    }
    for item in result["projection"]:
        payload = item["payload"]
        if payload.get("role") == "tool" and str(payload.get("content", "")).startswith("[compact_reloadable]\n"):
            item["projection_status"] = "settled"
    return result


def card(stamp: int, *, source_id: str = "tool-result-stable-1", correlation_id: str = "call-stable-1") -> dict:
    label = datetime.fromtimestamp(stamp, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    return {
        "role": "tool",
        "tool_call_id": correlation_id,
        "content": (
            f"[compact_reloadable]\ntime: {label}\nsource_id: {source_id}\n"
            "kind: tool.result\nsummary: Original fixture date 2026-08-01 remains exact."
        ),
    }


class ReplayMapperTests(unittest.TestCase):
    def bind_card(self) -> ReplayMapper:
        mapper = ReplayMapper()
        mapper.bind(
            snapshot([memory_row("tool-result-stable-1", kind="tool.result")], [card(_OLD_TS)]),
            snapshot([memory_row("tool-result-stable-1", kind="tool.result", timestamp=_NEW_TS)], [card(_NEW_TS)]),
        )
        return mapper

    def test_exact_snapshot_and_request_need_no_allowed_differences(self) -> None:
        mapper = ReplayMapper()
        source = snapshot([memory_row("assistant-source-1")])
        self.assertEqual(mapper.compare_snapshot(source, copy.deepcopy(source))["normalized_equal"], True)
        request = {"model": "synthetic-model", "temperature": 0.8, "messages": [source["projection"][0]["payload"]]}
        self.assertTrue(mapper.compare_request(request, copy.deepcopy(request)))
        self.assertEqual(mapper.allowed_differences, [])
        self.assertEqual(mapper.request_checks[-1], {"raw_equal": True, "normalized_equal": True})

    def test_generated_assistant_id_changes_only_typed_projection_identity(self) -> None:
        mapper = ReplayMapper()
        expected = snapshot([memory_row("assistant-source-old")])
        actual = snapshot([memory_row("assistant-source-new", timestamp=_NEW_TS)])
        result = mapper.compare_snapshot(expected, actual)
        self.assertFalse(result["raw_equal"])
        self.assertTrue(result["normalized_equal"])
        self.assertEqual(mapper.id_map["assistant-source-old"], "assistant-source-new")
        self.assertEqual(expected["projection"][0]["payload"], actual["projection"][0]["payload"])

    def test_provider_authored_id_references_are_never_rewritten(self) -> None:
        mapper = ReplayMapper()
        mapper.bind(snapshot([memory_row("assistant-source-old")]), snapshot([memory_row("assistant-source-new")]))
        left = [{"role": "assistant", "content": "Look up assistant-source-old"}]
        right = [{"role": "assistant", "content": "Look up assistant-source-new"}]
        self.assertFalse(mapper.compare_payloads(left, right))
        self.assertFalse(
            mapper.compare_request({"model": "synthetic", "messages": left}, {"model": "synthetic", "messages": right})
        )
        expected = snapshot([memory_row("assistant-source-old")], left)
        actual = snapshot([memory_row("assistant-source-new")], right)
        with self.assertRaisesRegex(ReplayMismatch, "replay_assistant_body_identity_unproven"):
            mapper.compare_snapshot(expected, actual)

    def test_changed_assistant_id_requires_nonempty_identical_associated_payloads(self) -> None:
        for omit in ("expected", "actual", "both"):
            expected = snapshot([memory_row("assistant-source-old")])
            actual = snapshot([memory_row("assistant-source-new")])
            if omit in ("expected", "both"):
                expected["projection"] = []
            if omit in ("actual", "both"):
                actual["projection"] = []
            with (
                self.subTest(omit=omit),
                self.assertRaisesRegex(ReplayMismatch, "replay_assistant_body_identity_unproven"),
            ):
                ReplayMapper().bind(expected, actual)

    def test_source_content_and_annotation_hashes_cannot_change(self) -> None:
        for field in ("content_hash", "annotation_hash"):
            expected = snapshot([memory_row("tool-stable", kind="tool.result")])
            expected["memory"][0][field] = "synthetic-original-hash"
            actual = copy.deepcopy(expected)
            actual["memory"][0][field] = "synthetic-changed-hash"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ReplayMismatch, "replay_source_content_or_annotation_changed"),
            ):
                ReplayMapper().bind(expected, actual)

    def test_native_tool_arguments_and_call_ids_are_exact(self) -> None:
        mapper = ReplayMapper()
        mapper.bind(snapshot([memory_row("assistant-source-old")]), snapshot([memory_row("assistant-source-new")]))
        left = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-stable-1",
                    "type": "function",
                    "function": {"name": "open_memory", "arguments": '{"memory_id":"assistant-source-old"}'},
                }
            ],
        }
        changed_argument = copy.deepcopy(left)
        changed_argument["tool_calls"][0]["function"]["arguments"] = '{"memory_id":"assistant-source-new"}'
        self.assertFalse(mapper.compare_payloads([left], [changed_argument]))
        changed_call = copy.deepcopy(left)
        changed_call["tool_calls"][0]["id"] = "call-changed"
        self.assertFalse(mapper.compare_payloads([left], [changed_call]))

    def test_only_proven_compact_card_time_line_may_differ(self) -> None:
        mapper = self.bind_card()
        self.assertTrue(mapper.compare_payloads([card(_OLD_TS)], [card(_NEW_TS)]))
        self.assertTrue(
            mapper.compare_request(
                {"model": "synthetic", "temperature": 0.8, "messages": [card(_OLD_TS)]},
                {"model": "synthetic", "temperature": 0.8, "messages": [card(_NEW_TS)]},
            )
        )
        self.assertEqual(mapper.request_checks[-1], {"raw_equal": False, "normalized_equal": True})

    def test_compact_time_without_public_evidence_is_rejected(self) -> None:
        self.assertFalse(ReplayMapper().compare_payloads([card(_OLD_TS)], [card(_NEW_TS)]))

    def test_card_looking_full_projection_and_nonobservation_cannot_relax_dates(self) -> None:
        for case in ("not_settled", "not_observation"):
            expected = snapshot([memory_row("tool-result-stable-1", kind="tool.result")], [card(_OLD_TS)])
            actual = snapshot(
                [memory_row("tool-result-stable-1", kind="tool.result", timestamp=_NEW_TS)], [card(_NEW_TS)]
            )
            for evidence in (expected, actual):
                if case == "not_settled":
                    evidence["projection"][0]["projection_status"] = "complete"
                else:
                    evidence["memory"][0]["turn_role"] = "action"
            mapper = ReplayMapper()
            mapper.bind(expected, actual)
            with self.subTest(case=case):
                self.assertFalse(mapper.compare_payloads([card(_OLD_TS)], [card(_NEW_TS)]))

    def test_compact_time_must_match_both_proven_timestamps_and_correlation(self) -> None:
        mapper = self.bind_card()
        for left, right in (
            (card(_OLD_TS + 60), card(_NEW_TS)),
            (card(_OLD_TS), card(_NEW_TS + 60)),
            (card(_OLD_TS, source_id="unproven-source"), card(_NEW_TS, source_id="unproven-source")),
            (card(_OLD_TS, correlation_id="wrong-call"), card(_NEW_TS, correlation_id="wrong-call")),
            (card(_OLD_TS), card(_NEW_TS, correlation_id="changed-call")),
        ):
            with self.subTest(left=left, right=right):
                self.assertFalse(mapper.compare_payloads([left], [right]))

    def test_dates_in_fixture_or_model_text_and_system_clock_are_exact(self) -> None:
        mapper = self.bind_card()
        left, right = card(_OLD_TS), card(_NEW_TS)
        right["content"] = right["content"].replace("2026-08-01", "2026-08-02")
        self.assertFalse(mapper.compare_payloads([left], [right]))
        for role in ("assistant", "user", "system"):
            with self.subTest(role=role):
                self.assertFalse(
                    mapper.compare_payloads(
                        [{"role": role, "content": "Current date: 2026-09-06 10:00"}],
                        [{"role": role, "content": "Current date: 2026-09-06 11:00"}],
                    )
                )
        left, right = card(_OLD_TS), card(_NEW_TS)
        left["role"] = right["role"] = "assistant"
        self.assertFalse(mapper.compare_payloads([left], [right]))

    def test_generation_parameters_and_schema_remain_exact(self) -> None:
        mapper = self.bind_card()
        left = {
            "model": "synthetic",
            "temperature": 0.8,
            "messages": [card(_OLD_TS)],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        }
        right = copy.deepcopy(left)
        right["messages"] = [card(_NEW_TS)]
        right["temperature"] = 0
        self.assertFalse(mapper.compare_request(left, right))
        right["temperature"] = 0.8
        right["tools"][0]["function"]["name"] = "different_lookup"
        self.assertFalse(mapper.compare_request(left, right))

    def test_stable_user_tool_and_turn_identity_changes_are_rejected(self) -> None:
        for kind in ("message.user", "tool.call", "tool.result"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ReplayMismatch, "stable_replay_source_identifier_changed"),
            ):
                ReplayMapper().bind(
                    snapshot([memory_row("stable-old", kind=kind)]), snapshot([memory_row("stable-new", kind=kind)])
                )
        for key in ("kind", "turn_role", "turn_id", "correlation_id", "seq_no"):
            left = snapshot([memory_row("assistant-stable")])
            right = copy.deepcopy(left)
            right["memory"][0][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ReplayMismatch, "replay_memory_identity_mismatch"):
                ReplayMapper().bind(left, right)

    def test_user_timestamp_and_unapproved_event_timestamp_changes_are_rejected(self) -> None:
        for kind, error in (
            ("message.user", "replay_user_timestamp_changed"),
            ("event.synthetic", "unsupported_replay_timestamp_difference"),
        ):
            with self.subTest(kind=kind), self.assertRaisesRegex(ReplayMismatch, error):
                ReplayMapper().bind(
                    snapshot([memory_row("stable", kind=kind)]),
                    snapshot([memory_row("stable", kind=kind, timestamp=_NEW_TS)]),
                )

    def test_existing_id_mapping_cannot_revert_to_identity(self) -> None:
        mapper = ReplayMapper()
        expected = snapshot([memory_row("assistant-old")])
        mapper.bind(expected, snapshot([memory_row("assistant-new")]))
        with self.assertRaisesRegex(ReplayMismatch, "replay_identifier_mapping_changed"):
            mapper.bind(expected, copy.deepcopy(expected))

    def test_existing_timestamp_mapping_cannot_revert_to_identity(self) -> None:
        mapper = self.bind_card()
        expected = snapshot([memory_row("tool-result-stable-1", kind="tool.result")], [card(_OLD_TS)])
        with self.assertRaisesRegex(ReplayMismatch, "replay_timestamp_mapping_changed"):
            mapper.bind(expected, copy.deepcopy(expected))

    def test_assistant_source_mapping_is_bijective_including_unchanged_ids(self) -> None:
        expected = snapshot([memory_row("assistant-a", seq_no=1), memory_row("assistant-b", seq_no=2)])
        actual = snapshot([memory_row("assistant-b", seq_no=1), memory_row("assistant-b", seq_no=2)])
        with self.assertRaises(ReplayMismatch):
            ReplayMapper().bind(expected, actual)

    def test_distinct_assistant_id_chain_is_mapped_once_only_in_typed_ids(self) -> None:
        expected = snapshot([memory_row("assistant-a", seq_no=1), memory_row("assistant-b", seq_no=2)])
        actual = snapshot([memory_row("assistant-b", seq_no=1), memory_row("assistant-c", seq_no=2)])
        mapper = ReplayMapper()
        self.assertTrue(mapper.compare_snapshot(expected, actual)["normalized_equal"])
        self.assertEqual(mapper.id_map, {"assistant-a": "assistant-b", "assistant-b": "assistant-c"})

    def test_source_count_missing_typed_evidence_and_invalid_timestamps_are_rejected(self) -> None:
        for malformed in ({}, {"memory": "not rows"}, {"memory": ["not a record"]}):
            with (
                self.subTest(malformed=malformed),
                self.assertRaisesRegex(ReplayMismatch, "typed_replay_memory_evidence_required"),
            ):
                ReplayMapper().bind(malformed, {"memory": []})
        with self.assertRaisesRegex(ReplayMismatch, "replay_memory_source_count_mismatch"):
            ReplayMapper().bind(snapshot([memory_row("source")]), snapshot([]))
        for timestamp in (None, True, float(_OLD_TS), "2026-09-06"):
            left = snapshot([memory_row("source")])
            right = copy.deepcopy(left)
            right["memory"][0]["timestamp"] = timestamp
            with (
                self.subTest(timestamp=timestamp),
                self.assertRaisesRegex(ReplayMismatch, "replay_source_timestamp_missing"),
            ):
                ReplayMapper().bind(left, right)

    def test_changed_projection_identity_or_payload_is_rejected(self) -> None:
        expected = snapshot([memory_row("assistant-stable")])
        for key in ("source_ids", "turn_id", "projection_index", "projection_status", "projection_version"):
            actual = copy.deepcopy(expected)
            actual["projection"][0][key] = ["changed"] if key == "source_ids" else "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ReplayMismatch, "replay_projection_identity_mismatch"):
                ReplayMapper().compare_snapshot(expected, actual)
        actual = copy.deepcopy(expected)
        actual["projection"][0]["payload"]["content"] = "Different synthetic answer"
        with self.assertRaisesRegex(ReplayMismatch, "replay_projection_body_mismatch"):
            ReplayMapper().compare_snapshot(expected, actual)


if __name__ == "__main__":
    unittest.main()
