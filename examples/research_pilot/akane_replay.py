"""Strict replay comparison with evidence-bound host-generated differences.

User, turn and native tool identifiers must remain identical. Only generated
assistant identifiers are mapped. A compact tool card's time line may differ
when each timestamp is proven by that source's public raw record. Dates inside
model-authored text and original fixture bodies are never normalized.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .runner import digest


class ReplayMismatch(RuntimeError):
    pass


def _rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = snapshot.get("memory")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ReplayMismatch("typed_replay_memory_evidence_required")
    return rows


class ReplayMapper:
    def __init__(self) -> None:
        self.id_map: dict[str, str] = {}
        self.timestamp_pairs: dict[str, tuple[int, int, str]] = {}
        self.allowed_differences: list[dict[str, Any]] = []
        self.request_checks: list[dict[str, Any]] = []
        self.compact_sources: set[str] = set()

    @staticmethod
    def _source_payloads(snapshot: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
        return [item["payload"] for item in snapshot.get("projection", []) if source_id in item.get("source_ids", [])]

    def bind(self, expected: dict[str, Any], actual: dict[str, Any]) -> None:
        self.compact_sources.clear()
        before, after = _rows(expected), _rows(actual)
        if len(before) != len(after):
            raise ReplayMismatch("replay_memory_source_count_mismatch")
        for left, right in zip(before, after, strict=True):
            for key in (
                "kind",
                "turn_role",
                "turn_id",
                "correlation_id",
                "seq_no",
                "origin",
                "role",
                "entry_type",
                "renderer_id",
                "renderer_version",
                "annotation_hash_available",
            ):
                if left.get(key) != right.get(key):
                    raise ReplayMismatch("replay_memory_identity_mismatch")
            old_id, new_id = left.get("source_id"), right.get("source_id")
            if not isinstance(old_id, str) or not isinstance(new_id, str):
                raise ReplayMismatch("replay_source_identifier_missing")
            if old_id in self.id_map and self.id_map[old_id] != new_id:
                raise ReplayMismatch("replay_identifier_mapping_changed")
            if new_id in self.id_map.values() and self.id_map.get(old_id) != new_id:
                raise ReplayMismatch("replay_identifier_mapping_not_bijective")
            if old_id != new_id:
                if left.get("kind") != "message.assistant":
                    raise ReplayMismatch("stable_replay_source_identifier_changed")
                old_payloads = self._source_payloads(expected, old_id)
                new_payloads = self._source_payloads(actual, new_id)
                if not old_payloads or old_payloads != new_payloads:
                    raise ReplayMismatch("replay_assistant_body_identity_unproven")
                if old_id not in self.id_map:
                    self.allowed_differences.append(
                        {"kind": "generated_assistant_source_id", "source": old_id, "replay": new_id}
                    )
            self.id_map[old_id] = new_id
            old_ts, new_ts = left.get("timestamp"), right.get("timestamp")
            if type(old_ts) is not int or type(new_ts) is not int:
                raise ReplayMismatch("replay_source_timestamp_missing")
            prior = self.timestamp_pairs.get(new_id)
            pair = (old_ts, new_ts, str(right.get("correlation_id") or ""))
            if prior is not None and prior != pair:
                raise ReplayMismatch("replay_timestamp_mapping_changed")
            if old_ts != new_ts:
                if left.get("kind") == "message.user":
                    raise ReplayMismatch("replay_user_timestamp_changed")
                if not (str(left.get("kind", "")).startswith("tool.") or left.get("kind") == "message.assistant"):
                    raise ReplayMismatch("unsupported_replay_timestamp_difference")
                if prior is None:
                    self.allowed_differences.append(
                        {"kind": "host_wall_clock_timestamp", "source_id": new_id, "source": old_ts, "replay": new_ts}
                    )
            self.timestamp_pairs[new_id] = pair
            for key in ("content_hash", "annotation_hash"):
                if key in left or key in right:
                    if left.get(key) != right.get(key):
                        raise ReplayMismatch("replay_source_content_or_annotation_changed")
            for record, stamp in ((left, old_ts), (right, new_ts)):
                local = datetime.fromtimestamp(stamp, ZoneInfo("Asia/Shanghai"))
                period = (
                    "morning"
                    if 5 <= local.hour < 12
                    else "afternoon"
                    if 12 <= local.hour < 18
                    else "night"
                    if local.hour >= 18
                    else "midnight"
                )
                for key, value in (("date_label", local.strftime("%Y-%m-%d")), ("time_of_day", period)):
                    if key in record and record[key] != value:
                        raise ReplayMismatch("replay_source_time_metadata_unproven")
            if str(left.get("kind", "")).startswith("tool.") and left.get("turn_role") == "observation":
                old_projections = [p for p in expected.get("projection", []) if old_id in p.get("source_ids", [])]
                new_projections = [p for p in actual.get("projection", []) if new_id in p.get("source_ids", [])]
                if (
                    len(old_projections) == len(new_projections) == 1
                    and old_projections[0].get("projection_status") == "settled"
                    and new_projections[0].get("projection_status") == "settled"
                ):
                    self.compact_sources.add(new_id)
            if set(left) != set(right):
                raise ReplayMismatch("replay_source_metadata_fields_changed")
            for key in set(left) - {"source_id", "timestamp", "date_label", "time_of_day"}:
                if key in {"projection_content_hash", "projection_payload_hashes"} and new_id in self.compact_sources:
                    continue
                if left[key] != right[key]:
                    raise ReplayMismatch("replay_source_metadata_changed")

    def _tool_card(self, expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
        if expected.get("role") != "tool" or actual.get("role") != "tool":
            return expected
        left, right = expected.get("content"), actual.get("content")
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            return expected
        if not left.startswith("[compact_reloadable]\ntime: ") or not right.startswith("[compact_reloadable]\ntime: "):
            return expected
        old_lines, new_lines = left.splitlines(), right.splitlines()
        if len(old_lines) != len(new_lines):
            return expected
        source_lines = [line[11:] for line in old_lines if line.startswith("source_id: ")]
        if len(source_lines) != 1:
            return expected
        pair = self.timestamp_pairs.get(source_lines[0])
        if (
            pair is None
            or source_lines[0] not in self.compact_sources
            or expected.get("tool_call_id") != pair[2]
            or actual.get("tool_call_id") != pair[2]
        ):
            return expected
        old_ts, new_ts, _ = pair

        def date_label(stamp: int) -> str:
            return datetime.fromtimestamp(stamp, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")

        if old_lines[1] != "time: " + date_label(old_ts) or new_lines[1] != "time: " + date_label(new_ts):
            return expected
        if old_lines[:1] + old_lines[2:] != new_lines[:1] + new_lines[2:]:
            return expected
        result = dict(expected)
        old_lines[1] = new_lines[1]
        result["content"] = "\n".join(old_lines)
        return result

    def compare_payloads(self, expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
        if len(expected) != len(actual):
            return False
        return all(self._tool_card(left, right) == right for left, right in zip(expected, actual, strict=True))

    def compare_request(self, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        raw_equal = expected == actual
        left, right = copy.deepcopy(expected), copy.deepcopy(actual)
        left_messages, right_messages = left.pop("messages", None), right.pop("messages", None)
        equivalent = (
            left == right
            and isinstance(left_messages, list)
            and isinstance(right_messages, list)
            and self.compare_payloads(left_messages, right_messages)
        )
        self.request_checks.append({"raw_equal": raw_equal, "normalized_equal": equivalent})
        return equivalent

    def compare_snapshot(self, expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
        self.bind(expected, actual)
        left, right = expected.get("projection", []), actual.get("projection", [])
        if len(left) != len(right):
            raise ReplayMismatch("replay_projection_count_mismatch")
        for old, new in zip(left, right, strict=True):
            for projected in (old, new):
                if "payload_hash" in projected and projected["payload_hash"] != digest(projected["payload"]):
                    raise ReplayMismatch("replay_payload_hash_invalid")
            for key in ("source_ids", "turn_id", "projection_index", "projection_status", "projection_version"):
                expected_value = old.get(key)
                if key == "source_ids":
                    expected_value = [self.id_map.get(source, source) for source in expected_value or []]
                if expected_value != new.get(key):
                    raise ReplayMismatch("replay_projection_identity_mismatch")
        expected_payloads = [item["payload"] for item in left]
        actual_payloads = [item["payload"] for item in right]
        normalized_equal = self.compare_payloads(expected_payloads, actual_payloads)
        if not normalized_equal:
            raise ReplayMismatch("replay_projection_body_mismatch")
        old_metrics, new_metrics = copy.deepcopy(expected.get("metrics", {})), copy.deepcopy(actual.get("metrics", {}))
        for metrics, payloads in ((old_metrics, expected_payloads), (new_metrics, actual_payloads)):
            prefix_hash = metrics.pop("stable_prefix_hash", None)
            if prefix_hash is not None and prefix_hash != digest(payloads):
                raise ReplayMismatch("replay_projection_hash_invalid")
        if old_metrics != new_metrics:
            raise ReplayMismatch("replay_projection_metrics_changed")
        old_state, new_state = (
            copy.deepcopy(expected.get("host_state", {})),
            copy.deepcopy(actual.get("host_state", {})),
        )
        old_session, new_session = old_state.get("session"), new_state.get("session")
        if isinstance(old_session, dict) and isinstance(new_session, dict):
            old_updated, new_updated = old_session.get("updated_at"), new_session.get("updated_at")
            if old_updated != new_updated:
                old_dialogue = [
                    record for record in _rows(expected) if str(record.get("kind", "")).startswith("message.")
                ]
                new_dialogue = [
                    record for record in _rows(actual) if str(record.get("kind", "")).startswith("message.")
                ]
                old_latest = max(old_dialogue, key=lambda record: record["seq_no"]) if old_dialogue else {}
                new_latest = max(new_dialogue, key=lambda record: record["seq_no"]) if new_dialogue else {}
                if old_updated != old_latest.get("timestamp") or new_updated != new_latest.get("timestamp"):
                    raise ReplayMismatch("replay_session_timestamp_unproven")
                old_session["updated_at"] = new_updated
        if old_state != new_state:
            raise ReplayMismatch("replay_host_state_changed")
        return {
            "raw_equal": left == right,
            "normalized_equal": normalized_equal,
            "allowed_differences": copy.deepcopy(self.allowed_differences),
            "request_checks": copy.deepcopy(self.request_checks),
            "host_state_equal_after_proven_session_clock_mapping": True,
            "projection_metrics_equal": True,
        }
