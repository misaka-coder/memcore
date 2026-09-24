"""Integration checks for offline pilot isolation and truthful experiment logs."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.research_pilot.runner import (
    EventLog,
    FixtureService,
    PilotFailure,
    SmokeSession,
    canonical,
    offline_plan,
    run_smoke,
)
from examples.research_pilot.validation import validate_pack


class PilotRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = validate_pack(Path(__file__).resolve().parents[1] / "docs/research/pilot_v1")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.sessions: list[SmokeSession] = []

    def tearDown(self) -> None:
        for session in self.sessions:
            session.close()
        self.temp.cleanup()

    def session(self, scope: str = "test", condition: str = "card", manifest: dict | None = None) -> SmokeSession:
        manifest = copy.deepcopy(manifest or self.pack["manifest"])
        session = SmokeSession(
            path=self.directory / f"{scope}.sqlite3",
            scope=scope,
            scenario=self.pack["scenarios"]["scenarios"][0],
            policy=manifest["condition_definitions"][condition]["operation_projection_policy"],
            specification=self.pack["scenarios"],
            manifest=manifest,
            log=EventLog(self.directory / f"{scope}.jsonl"),
        )
        self.sessions.append(session)
        return session

    def test_four_runs_complete_without_network_gold_or_fake_scores(self) -> None:
        pack = copy.deepcopy(self.pack)
        sentinel = "EVALUATOR_ONLY_NEVER_SEND_5dc491"
        pack["answer_key"] = {"secret_gold": sentinel}
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            result = run_smoke(pack, self.directory / "output", {})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(sum(len(r["branches"]) for r in result["runs"]), 12)
        self.assertIsNone(result["model_scores"])
        self.assertIsNone(result["provider_usage"])
        self.assertFalse(result["live_transport_verified"])
        self.assertFalse(result["akane_runtime_verified"])
        self.assertEqual(result["paid_cost_cny"], 0)
        self.assertTrue(result["forced_routing"])
        events_text = (self.directory / "output/events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(sentinel, events_text)
        events = [json.loads(line) for line in events_text.splitlines()]
        requests = [e for e in events if e["event"] == "simulated_request"]
        self.assertEqual(len(requests), result["simulated_requests"])
        for row in result["runs"]:
            self.assertTrue(all(b["baseline_equal"] and b["model_score"] is None for b in row["branches"]))
            self.assertTrue(all(c["open_full"] and c["reload_equal"] for c in row["observation_checks"]))
            self.assertTrue(all(c["closed_compact"] == (row["condition"] == "card") for c in row["observation_checks"]))
        self.assertEqual(len({r["control_hash"] for r in result["runs"]}), 1)
        for scenario in pack["scenarios"]["scenarios"]:
            for probe in [s for s in scenario["steps"] if s["stage"] == "probe"]:
                setup_requests = [e for e in requests if e["scope"].endswith("__setup")]
                self.assertTrue(all(probe["user_text"] not in canonical(e["request"]) for e in setup_requests))

    def test_old_fixture_and_cross_scenario_queries_are_unavailable(self) -> None:
        scenario = self.pack["scenarios"]["scenarios"][0]
        service = FixtureService(scenario)
        self.assertEqual(service.lookup("g_task_v1", {"query_key": "gallery_v1"})["status"], "ok")
        self.assertEqual(service.lookup("g_task_v1", {"query_key": "gallery_v2"})["status"], "unavailable")
        self.assertEqual(service.lookup("g_probe_history", {"query_key": "gallery_v1"})["status"], "unavailable")
        self.assertEqual(service.lookup("g_probe_history", {"query_key": "notifier_v1"})["status"], "unavailable")
        result = service.lookup("g_task_v1", {"query_key": "gallery_v1"})
        result["content"] = "mutated"
        self.assertNotEqual(service.lookup("g_task_v1", {"query_key": "gallery_v1"})["content"], "mutated")

    def test_branch_writes_cannot_reach_sibling_reads(self) -> None:
        first, second = self.session("first"), self.session("second")
        intro = first.scenario["steps"][0]
        for session in (first, second):
            session.run_turn(intro, offline_plan(intro, session.scenario, {}), replay=True)
        self.assertEqual(first.history(), second.history())
        marker = "branch_only_marker_c8d647"
        step = {
            "step_id": "sentinel",
            "stage": "conversation",
            "timestamp": "2026-08-10T19:01:00+08:00",
            "user_text": marker,
        }
        first.run_turn(step, offline_plan(step, first.scenario, {}))
        self.assertNotIn(marker, canonical(second.history()))
        self.assertEqual(second.mem.open_memory(memory_id="sentinel:user", view="content")["status"], "empty")
        self.assertNotIn(marker, canonical(second.mem.read_timeline(date_from="2026-08-10")))
        self.assertNotIn(marker, canonical(second.mem.browse_memory(date_from="2026-08-10")))
        self.assertNotIn(marker, canonical(second.mem.retrieve(query=marker)))

    def test_context_limit_aborts_without_fake_request_or_final(self) -> None:
        manifest = copy.deepcopy(self.pack["manifest"])
        manifest["proposed_runner_limits"]["research_context_budget_tokens"] = 2049
        session = self.session(manifest=manifest)
        step = session.scenario["steps"][0]
        with self.assertRaisesRegex(PilotFailure, "estimated_context_budget_exceeded"):
            session.run_turn(step, offline_plan(step, session.scenario, {}))
        self.assertEqual(session.requests, 0)
        result = session.mem.complete_turn(turn_id=step["step_id"], semantic_text="bad", provider_output_raw="bad")
        self.assertEqual(result.status, "conflict")

    def test_unknown_tool_aborts_and_is_not_executed(self) -> None:
        session = self.session()
        step = session.scenario["steps"][0]
        response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "bad_call", "function": {"name": "read_file", "arguments": "{}"}}],
        }
        with self.assertRaisesRegex(PilotFailure, "unapproved_tool"):
            session.run_turn(step, [response])
        self.assertNotIn("bad_call", canonical(session.history()))

    def test_tool_limit_and_invalid_final_preserve_failure(self) -> None:
        manifest = copy.deepcopy(self.pack["manifest"])
        manifest["proposed_runner_limits"]["max_tool_calls_per_turn"] = 1
        session = self.session(manifest=manifest)
        step = session.scenario["steps"][1]
        responses = offline_plan(step, session.scenario, {})
        second_call = copy.deepcopy(responses[0])
        second_call["tool_calls"][0]["id"] += "_again"
        with self.assertRaisesRegex(PilotFailure, "tool_call_limit_exceeded"):
            session.run_turn(step, [responses[0], second_call, responses[1]])
        self.assertNotIn(step["step_id"] + ":final", canonical(session.history()))
        other = self.session("invalid_final")
        with self.assertRaisesRegex(PilotFailure, "invalid_final_json"):
            other.run_turn(other.scenario["steps"][0], [{"role": "assistant", "content": "broken JSON"}])

    def test_run_failure_is_retained_and_existing_output_is_not_overwritten(self) -> None:
        output = self.directory / "failure"
        with patch("examples.research_pilot.runner.offline_plan", side_effect=PilotFailure("injected_failure")):
            result = run_smoke(self.pack, output, {})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["runs"]), 4)
        self.assertTrue(all(r["error"] == "injected_failure" for r in result["runs"]))
        before = (output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_smoke(self.pack, output, {})
        self.assertEqual((output / "report.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
