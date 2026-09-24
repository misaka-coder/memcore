"""Offline rejection checks for the Phase 0 research pack boundary."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from examples.research_pilot.validation import validate_pack


_PACK_DIR = Path(__file__).resolve().parents[1] / "docs" / "research" / "pilot_v1"
_FILES = {"scenarios": "scenarios.json", "answer_key": "answer_key.json", "manifest": "run_manifest.json"}


class ResearchPilotValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.pack = {
            key: json.loads((_PACK_DIR / filename).read_text(encoding="utf-8")) for key, filename in _FILES.items()
        }

    def _write(self) -> None:
        for key, filename in _FILES.items():
            (self.directory / filename).write_text(json.dumps(self.pack[key], ensure_ascii=False), encoding="utf-8")

    def _invalid(self, message: str) -> None:
        self._write()
        with self.assertRaisesRegex(ValueError, message):
            validate_pack(self.directory)

    def test_current_pack_is_parsed_without_network_or_credentials(self) -> None:
        self._write()
        with (
            patch("socket.create_connection", side_effect=AssertionError("network forbidden")),
            patch("os.getenv", side_effect=AssertionError("credential lookup forbidden")),
        ):
            validated = validate_pack(self.directory)
        self.assertEqual(validated, self.pack)
        self.assertEqual(set(validated), {"scenarios", "answer_key", "manifest"})
        self.assertEqual(validated["answer_key"]["audience"], "evaluator_only")

    def test_cross_scenario_step_allowlist_is_rejected(self) -> None:
        scenario = self.pack["scenarios"]["scenarios"][0]
        foreign = self.pack["scenarios"]["scenarios"][1]["fixtures"][0]["fixture_id"]
        scenario["fixture_availability_by_step"]["g_task_v1"] = [foreign]
        self._invalid("cross-scenario")

    def test_cross_scenario_owned_allowlist_is_rejected(self) -> None:
        self.pack["scenarios"]["scenarios"][0]["allowed_fixture_ids"].append("notifier_doc_v1")
        self._invalid("owned fixtures")

    def test_duplicate_run_id_is_rejected(self) -> None:
        runs = self.pack["manifest"]["runs"]
        runs.append(copy.deepcopy(runs[0]))
        self._invalid("duplicate run_id")

    def test_duplicate_run_pair_cannot_hide_behind_another_id(self) -> None:
        runs = self.pack["manifest"]["runs"]
        runs[1] = copy.deepcopy(runs[0])
        runs[1]["run_id"] += "_other"
        self._invalid("duplicate scenario/condition")

    def test_missing_condition_run_is_rejected(self) -> None:
        self.pack["manifest"]["runs"].pop()
        self._invalid("four runs")

    def test_future_fixture_access_is_rejected(self) -> None:
        self.pack["scenarios"]["scenarios"][0]["fixture_availability_by_step"]["g_intro"] = ["gallery_doc_v2"]
        self._invalid("future version")

    def test_probe_cannot_refetch_expired_v1(self) -> None:
        self.pack["scenarios"]["scenarios"][0]["fixture_availability_by_step"]["g_probe_history"] = ["gallery_doc_v1"]
        self._invalid("expired v1")

    def test_missing_availability_step_is_rejected(self) -> None:
        del self.pack["scenarios"]["scenarios"][0]["fixture_availability_by_step"]["g_gap_1"]
        self._invalid("exactly every step")

    def test_broken_gold_fixture_reference_is_rejected(self) -> None:
        self.pack["answer_key"]["cases"][0]["probes"][0]["expected_fixture_ids"] = ["missing_fixture"]
        self._invalid("unknown or foreign fixture")

    def test_gold_cannot_use_another_probe_as_evidence(self) -> None:
        self.pack["answer_key"]["cases"][0]["probes"][1]["expected_user_step_ids"] = ["g_probe_history"]
        self._invalid("predate the probe snapshot")

    def test_gold_task_cannot_substitute_v2_for_v1(self) -> None:
        self.pack["answer_key"]["cases"][0]["task_checks"][0]["required_fixture_id"] = "gallery_doc_v2"
        self._invalid("matching version")

    def test_gold_must_cover_all_probes(self) -> None:
        self.pack["answer_key"]["cases"][0]["probes"].pop()
        self._invalid("gold probe coverage")

    def test_duplicate_step_id_is_rejected(self) -> None:
        steps = self.pack["scenarios"]["scenarios"][0]["steps"]
        steps[1]["step_id"] = steps[0]["step_id"]
        self._invalid("duplicate step_id")

    def test_fixture_document_id_must_match_recorded_identity(self) -> None:
        self.pack["scenarios"]["scenarios"][0]["fixtures"][0]["result"]["document_id"] = "gallery_doc_v2"
        self._invalid("result.document_id")

    def test_fixture_query_bindings_must_be_unique(self) -> None:
        fixtures = self.pack["scenarios"]["scenarios"][0]["fixtures"]
        fixtures[1]["arguments"] = dict(fixtures[0]["arguments"])
        self._invalid("query_key must bind")

    def test_query_schema_cannot_expose_unbound_fixtures(self) -> None:
        schema = self.pack["scenarios"]["fixture_tool_spec"]["input_schema"]
        schema["properties"]["query_key"]["enum"].append("secret_v1")
        self._invalid("query enum")

    def test_timestamps_must_be_ordered_and_offset_aware(self) -> None:
        steps = self.pack["scenarios"]["scenarios"][0]["steps"]
        original = steps[1]["timestamp"]
        for timestamp, expected in (
            (steps[0]["timestamp"], "strictly increase"),
            ("2026-08-10T19:05:00", "UTC offset"),
            ("not-a-timestamp", "invalid ISO"),
        ):
            with self.subTest(timestamp=timestamp):
                steps[1]["timestamp"] = timestamp
                self._invalid(expected)
        steps[1]["timestamp"] = original

    def test_care_cannot_be_enabled_or_coerced_from_falsey_number(self) -> None:
        for enabled in (True, 0):
            with self.subTest(enabled=enabled):
                self.pack["manifest"]["fixed_controls"]["care"]["enabled"] = enabled
                self._invalid("care.enabled")

    def test_model_budget_currency_and_policy_are_fixed(self) -> None:
        mutations = (
            ("authorized_planning", "model", "other-model", "authorized_planning.model"),
            ("authorized_planning", "currency", "USD", "authorized_planning.currency"),
            ("authorized_planning", "first_stage_spend_ceiling", 100, "budget must be 50"),
            ("proposed_runner_limits", "max_paid_spend_cny", 100, "budget must be 50"),
        )
        for section, field, value, error in mutations:
            with self.subTest(field=field):
                old = self.pack["manifest"][section][field]
                self.pack["manifest"][section][field] = value
                self._invalid(error)
                self.pack["manifest"][section][field] = old
        self.pack["manifest"]["condition_definitions"]["card"]["operation_projection_policy"] = "unknown_policy"
        self._invalid("unsupported projection policy")

    def test_prepared_pack_cannot_contain_completed_results(self) -> None:
        self.pack["manifest"]["runs"][0]["result"] = {"score": 1}
        self._invalid("unrun result")

    def test_version_and_evaluator_audience_are_checked(self) -> None:
        for key, field, value, error in (
            ("scenarios", "format", "unknown", "unsupported format"),
            ("manifest", "status", "completed", "only prepared"),
            ("answer_key", "audience", "model", "evaluator-only"),
        ):
            with self.subTest(field=field):
                old = self.pack[key][field]
                self.pack[key][field] = value
                self._invalid(error)
                self.pack[key][field] = old

    def test_bad_json_missing_files_and_wrong_shapes_raise_value_error(self) -> None:
        self._write()
        path = self.directory / "answer_key.json"
        for body in ('{"broken":', "[]", '{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}'):
            with self.subTest(body=body):
                path.write_text(body, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_pack(self.directory)
        path.unlink()
        with self.assertRaises(ValueError):
            validate_pack(self.directory)

    def test_wrong_nested_shapes_raise_value_error(self) -> None:
        self.pack["scenarios"]["scenarios"][0]["fixtures"][0]["arguments"] = []
        self._invalid("expected an object")


if __name__ == "__main__":
    unittest.main()
