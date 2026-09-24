"""Evidence extraction must not promote string matches into experimental scores."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.research_pilot.evaluation import evaluate_trace


_PACK_DIR = Path(__file__).resolve().parents[1] / "docs" / "research" / "pilot_v1"


class ResearchPilotEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = json.loads((_PACK_DIR / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
        self.gold_cases = json.loads((_PACK_DIR / "answer_key.json").read_text(encoding="utf-8"))["cases"]
        self.scenario = self.scenarios[0]
        self.gold = self.gold_cases[0]
        self.trace = {
            "status": "completed",
            "setup": [
                {
                    "step_id": "g_task_v1",
                    "status": "completed",
                    "speech": "推荐松岚厅，上限12人，安静。",
                    "assistant_messages": [{"role": "assistant", "content": "推荐松岚厅。"}],
                    "tool_calls": [
                        {
                            "name": "lookup_fixture",
                            "arguments": {"query_key": "gallery_v1"},
                            "result": copy.deepcopy(self.scenario["fixtures"][0]["result"]),
                            "source_id": "actual-source-v1",
                        }
                    ],
                }
            ],
            "branches": [
                {
                    "step_id": "g_probe_history",
                    "status": "completed",
                    "speech": "第一版为RZ-417，钥匙应还到东侧服务台。",
                    "assistant_messages": [],
                    "tool_calls": [],
                },
                {"step_id": "g_probe_memory", "status": "completed", "speech": "周四晚上带蓝色笔记本。"},
                {"step_id": "g_probe_unknown", "status": "completed", "speech": "资料没有提供停车费信息。"},
            ],
        }

    def _evaluate(self) -> dict:
        return evaluate_trace(self.scenario, self.gold, self.trace)

    def _node(self, *, source: str = "actual-source-v1", status: str = "ok", view: str = "content") -> dict:
        fixture_result = self.scenario["fixtures"][0]["result"]
        return {
            "memory_id": source,
            "status": status,
            "view": view,
            "detail": "full",
            "node_type": "raw",
            "result": {"source_id": source},
            "text": "raw metadata\ndata:\n" + json.dumps({"output": json.dumps(fixture_result, ensure_ascii=False)}),
        }

    def _set_open(self, node: dict, *, arguments: dict | None = None) -> None:
        self.trace["branches"][0]["tool_calls"] = [
            {
                "name": "open_memory",
                "arguments": arguments or {"memory_id": "actual-source-v1", "view": "content"},
                "result": {"ok": True, "status": "ok", "tool_name": "open_memory", "result": node},
                "source_id": "probe-observation",
            }
        ]

    def test_evaluation_is_offline_and_does_not_mutate_inputs(self) -> None:
        before = copy.deepcopy((self.scenario, self.gold, self.trace))
        with (
            patch("socket.create_connection", side_effect=AssertionError("network forbidden")),
            patch("os.getenv", side_effect=AssertionError("credential lookup forbidden")),
            patch("builtins.open", side_effect=AssertionError("filesystem forbidden")),
        ):
            result = self._evaluate()
        self.assertEqual(before, (self.scenario, self.gold, self.trace))
        self.assertTrue(result["requires_review"])
        self.assertFalse(result["automatic_correctness_scoring"])
        self.assertIsNone(result["correct_probe_count"])
        self.assertIsNone(result["correctness_rate"])
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_literal_matches_are_candidates_not_correctness_or_evidence_grades(self) -> None:
        result = self._evaluate()
        probe = result["probes"][0]
        self.assertTrue(probe["attempts"][0]["candidate_fields"]["entry_code"]["literal_present"])
        self.assertIsNone(probe["correct"])
        self.assertIsNone(probe["evidence_grade"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["completed_probe_count"], 3)
        self.assertEqual(len(result["manual_review_table"]), 3)

    def test_paraphrase_without_literal_is_not_automatically_wrong(self) -> None:
        self.trace["branches"][1]["speech"] = "礼拜四夜里带那本蓝色的笔记本。"
        attempt = self._evaluate()["probes"][1]["attempts"][0]
        self.assertFalse(attempt["candidate_fields"]["time"]["literal_present"])
        self.assertIsNone(attempt["correct"])

    def test_numeric_candidates_do_not_match_larger_or_decimal_values(self) -> None:
        self.scenario, self.gold = self.scenarios[1], self.gold_cases[1]
        self.trace = {"branches": [{"step_id": "n_probe_history", "speech": "117秒、50秒、17.5秒"}]}
        fields = self._evaluate()["probes"][0]["attempts"][0]["candidate_fields"]
        self.assertFalse(fields["backoff_seconds"]["literal_present"])
        self.assertFalse(fields["jitter_upper_bound_seconds"]["literal_present"])
        self.trace["branches"][0]["speech"] = "固定退避17秒，抖动上界5秒。"
        fields = self._evaluate()["probes"][0]["attempts"][0]["candidate_fields"]
        self.assertTrue(all(candidate["literal_present"] for candidate in fields.values()))

    def test_generic_absence_is_not_a_missing_field_candidate(self) -> None:
        for speech in ("没有。", "停车费是10元，没有别的问题。", "资料没有提供钥匙编号。"):
            with self.subTest(speech=speech):
                self.trace["branches"][2]["speech"] = speech
                evidence = self._evaluate()["probes"][2]["attempts"][0]["missing_field_evidence"]
                self.assertFalse(evidence["explicit_subject_absence_candidate"])
                self.assertIsNone(evidence["correct"])

    def test_specific_missing_field_and_numeric_risk_are_independent_review_aids(self) -> None:
        self.trace["branches"][2]["speech"] = "资料没有提供停车费信息，不能推断为17元。"
        evidence = self._evaluate()["probes"][2]["attempts"][0]["missing_field_evidence"]
        self.assertTrue(evidence["explicit_subject_absence_candidate"])
        self.assertEqual(evidence["numeric_unit_mentions_for_review"][0]["value"], "17元")
        self.scenario, self.gold = self.scenarios[1], self.gold_cases[1]
        self.trace = {"branches": [{"step_id": "n_probe_unknown", "speech": "HTTP连接超时未提供。17秒是退避。"}]}
        evidence = self._evaluate()["probes"][2]["attempts"][0]["missing_field_evidence"]
        self.assertTrue(evidence["explicit_subject_absence_candidate"])

    def test_tool_and_user_text_are_not_assistant_answer_exposure(self) -> None:
        self.trace["setup"][0]["assistant_messages"].append({"role": "user", "content": "RZ-417 东侧服务台"})
        result = self._evaluate()
        self.assertFalse(result["probes"][0]["setup_answer_exposure_candidates"])
        self.assertFalse(result["setup_audit"]["task_forbidden_literal_candidates"])

    def test_exposure_audit_covers_gap_finals_retained_reasoning_and_tool_arguments(self) -> None:
        self.trace["setup"].append(
            {
                "step_id": "g_gap_2",
                "speech": "钥匙是东侧服务台。",
                "assistant_messages": [
                    {
                        "role": "assistant",
                        "reasoning_content": "此前核验码RZ-417。",
                        "content": "v2是RZ-982。",
                        "tool_calls": [{"function": {"name": "retrieve_for_turn", "arguments": '{"query":"RZ-417"}'}}],
                    }
                ],
            }
        )
        probe = self._evaluate()["probes"][0]
        paths = {hit["path"] for hit in probe["setup_answer_exposure_candidates"]}
        self.assertIn("speech", paths)
        self.assertIn("assistant_messages[0].reasoning_content", paths)
        self.assertIn("assistant_messages[0].tool_calls[0].function.arguments", paths)
        self.assertTrue(probe["setup_alternate_version_exposure_candidates"])

    def test_correct_old_new_comparison_is_flagged_but_never_automatically_failed(self) -> None:
        self.trace["branches"][0]["speech"] = "旧版是RZ-417、东侧服务台；新版才是RZ-982、北门值班柜台。"
        attempt = self._evaluate()["probes"][0]["attempts"][0]
        self.assertEqual(len(attempt["alternate_version_mentions"]), 2)
        self.assertTrue(all(item["literal_present"] for item in attempt["candidate_fields"].values()))
        self.assertIsNone(attempt["correct"])

    def test_escaped_json_and_separately_recorded_tool_arguments_are_audited(self) -> None:
        step = self.trace["setup"][0]
        step["assistant_messages"] = [{"role": "assistant", "content": json.dumps({"speech": "东侧服务台"})}]
        step["tool_calls"].append({"name": "retrieve_for_turn", "arguments": {"query": "RZ-417"}})
        paths = {hit["path"] for hit in self._evaluate()["probes"][0]["setup_answer_exposure_candidates"]}
        self.assertIn("assistant_messages[0].content.decoded.speech", paths)
        self.assertIn("tool_calls[1].arguments.query", paths)

    def test_actual_source_mapping_and_exact_returned_body_prove_content_readback(self) -> None:
        self._set_open(self._node())
        readback = self._evaluate()["probes"][0]["attempts"][0]["readback"]
        self.assertEqual(readback["expected_fixture_source_ids"], {"actual-source-v1": ["gallery_doc_v1"]})
        self.assertEqual(readback["successful_expected_content_source_ids"], ["actual-source-v1"])
        self.assertEqual(readback["exact_fixture_body_verified_source_ids"], ["actual-source-v1"])

    def test_fixture_document_id_is_not_an_invented_memcore_source_id(self) -> None:
        self._set_open(
            self._node(source="gallery_doc_v1"), arguments={"memory_id": "gallery_doc_v1", "view": "content"}
        )
        readback = self._evaluate()["probes"][0]["attempts"][0]["readback"]
        self.assertFalse(readback["successful_expected_content_source_ids"])

    def test_card_failed_and_compact_results_are_not_full_content_readback(self) -> None:
        for mutation in ({"view": "card"}, {"status": "empty"}, {"detail": "compact"}):
            with self.subTest(mutation=mutation):
                node = self._node()
                node.update(mutation)
                self._set_open(node)
                readback = self._evaluate()["probes"][0]["attempts"][0]["readback"]
                self.assertFalse(readback["successful_expected_content_source_ids"])
                self.assertFalse(readback["exact_fixture_body_verified_source_ids"])

    def test_successful_status_without_body_is_not_verified_full_body(self) -> None:
        node = self._node()
        node["text"] = "RZ-417"
        self._set_open(node)
        readback = self._evaluate()["probes"][0]["attempts"][0]["readback"]
        self.assertTrue(readback["successful_expected_content_source_ids"])
        self.assertFalse(readback["exact_fixture_body_verified_source_ids"])

    def test_native_partial_batch_maps_bodies_and_status_per_node(self) -> None:
        node = self._node()
        body = node.pop("text")
        batch = {
            "status": "partial",
            "view": "content",
            "memory_ids": ["actual-source-v1", "missing"],
            "result": {"items": [node, {"memory_id": "missing", "status": "empty", "view": "content"}]},
            "text": f"[memory_id=actual-source-v1 status=ok]\n{body}\n\n[memory_id=missing status=empty]\nreason=missing",
        }
        self._set_open(batch, arguments={"memory_ids": ["actual-source-v1", "missing"], "view": "content"})
        readback = self._evaluate()["probes"][0]["attempts"][0]["readback"]
        self.assertEqual(readback["exact_fixture_body_verified_source_ids"], ["actual-source-v1"])
        batch["result"]["items"][0]["status"] = "empty"
        self._set_open(batch, arguments={"memory_ids": ["actual-source-v1", "missing"], "view": "content"})
        self.assertFalse(
            self._evaluate()["probes"][0]["attempts"][0]["readback"]["exact_fixture_body_verified_source_ids"]
        )

    def test_missing_fixture_mapping_is_explicit_and_not_guessed(self) -> None:
        self.trace["setup"][0]["tool_calls"][0].pop("source_id")
        self._set_open(self._node())
        readback = self._evaluate()["probes"][0]["attempts"][0]["readback"]
        self.assertEqual(readback["missing_fixture_source_mappings"], ["gallery_doc_v1"])
        self.assertFalse(readback["exact_fixture_body_verified_source_ids"])

    def test_failed_missing_duplicate_and_extra_branches_are_retained(self) -> None:
        self.trace["status"] = "failed"
        self.trace["branches"] = [
            {"step_id": "g_probe_history", "status": "failed", "error": {"reason": "provider_error"}},
            {"step_id": "g_probe_memory", "status": "completed", "speech": "A"},
            {"step_id": "g_probe_memory", "status": "completed", "speech": "B"},
            {"step_id": "foreign", "status": "completed", "speech": "C"},
        ]
        result = self._evaluate()
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["present_probe_count"], 2)
        self.assertEqual(result["completed_probe_count"], 0)
        self.assertEqual([probe["branch_status"] for probe in result["probes"]], ["failed", "duplicate", "missing"])
        self.assertEqual(result["probes"][0]["attempts"][0]["error"], {"reason": "provider_error"})
        self.assertEqual(result["manual_review_table"][1]["answers"], ["A", "B"])
        self.assertEqual(result["unmatched_branch_indices"], [3])

    def test_missing_status_is_unknown_not_success_and_incomplete_setup_is_visible(self) -> None:
        self.trace["branches"][0].pop("status")
        self.trace["setup"][0].pop("assistant_messages")
        result = self._evaluate()
        self.assertEqual(result["probes"][0]["branch_status"], "unknown")
        self.assertEqual(result["completed_probe_count"], 2)
        self.assertIn("g_intro", result["setup_audit"]["missing_setup_step_ids"])
        self.assertEqual(result["setup_audit"]["steps_missing_assistant_messages"], ["g_task_v1"])

    def test_gold_and_trace_scope_or_denominator_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same scenario"):
            evaluate_trace(self.scenario, self.gold_cases[1], self.trace)
        self.trace["scenario_id"] = "notifier_history"
        with self.assertRaisesRegex(ValueError, "different scenario"):
            self._evaluate()
        del self.trace["scenario_id"]
        self.gold["probes"].pop()
        with self.assertRaisesRegex(ValueError, "every planned probe"):
            self._evaluate()


if __name__ == "__main__":
    unittest.main()
