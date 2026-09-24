"""TEST_ONLY scripted transport checks of the public MemCore live-run assembly.

These tests inject hashed embeddings and never contact a provider. Their
temporary traces prove plumbing only; they are not model-evaluation evidence.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from memcore import HashedEmbeddingProvider

from examples.research_pilot.akane_prompt import FORMAT, _hash
from examples.research_pilot.live import LiveSession, run_live
from examples.research_pilot.preflight import MODEL
from examples.research_pilot.runner import EventLog, FixtureService, PilotFailure, canonical
from examples.research_pilot.validation import validate_pack


def _bundle(specification: dict[str, Any]) -> dict[str, Any]:
    # Valid composition schema, deliberately not an actual Akane builder export.
    bundle = {
        "format": FORMAT,
        "shared": {
            "system_prompt": "TEST_ONLY stable system contract",
            "system_extra_blocks": ["TEST_ONLY final JSON contract"],
            "history_prefix_messages": [{"role": "user", "content": "TEST_ONLY fixed persona fixture"}],
        },
        "steps": {
            step["step_id"]: {
                "user_prompt": f"time: {step['timestamp']}\nUser: {step['user_text']}",
                "ephemeral_turns": [{"role": "user", "content": f"TEST_ONLY ephemeral {step['step_id']}"}],
            }
            for scenario in specification["scenarios"]
            for step in scenario["steps"]
        },
        "evidence": {"actual_builder_called": False, "run_mode": "TEST_ONLY"},
    }
    bundle["bundle_hash"] = _hash(bundle)
    return bundle


def _final(speech: str = "TEST_ONLY scripted reply; no model accuracy claim.") -> dict[str, Any]:
    return {"role": "assistant", "content": canonical({"speech": speech, "memory_metadata": {}})}


def _call(name: str, arguments: str, *, index: int = 0) -> dict[str, Any]:
    return {
        "index": index,
        "id": f"TEST_ONLY_call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class _OfflineLedger:
    def snapshot(self) -> dict[str, Any]:
        return {"mode": "TEST_ONLY", "paid_cost_cny": "0", "pending_reservations": 0}


class _OfflineClient:
    """In-memory provider-shape fake; script routing is explicit test-only data."""

    def __init__(self, script: Callable[..., dict[str, Any]]):
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self.ledger = _OfflineLedger()
        self._live = False
        self.counts: Counter = Counter()

    @property
    def is_live(self) -> bool:
        return False

    def call(self, request: dict[str, Any], *, scope: str, step_id: str) -> dict[str, Any]:
        call_index = self.counts[(scope, step_id)]
        self.counts[(scope, step_id)] += 1
        row = {
            "request": copy.deepcopy(request),
            "scope": scope,
            "step_id": step_id,
            "run_mode": "TEST_ONLY_offline_script",
            "status": "running",
        }
        self.requests.append(row)
        try:
            message = self.script(request, scope, step_id, call_index)
            response = {
                "model": MODEL,
                "choices": [
                    {"finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": message}
                ],
            }
            row.update(status="passed", response=copy.deepcopy(response))
            return response
        except Exception:
            row["status"] = "failed"
            raise


class ResearchPilotLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = validate_pack(Path(__file__).resolve().parents[1] / "docs/research/pilot_v1")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.sessions: list[LiveSession] = []
        self.bundle = _bundle(self.pack["scenarios"])
        self.embedding = HashedEmbeddingProvider()  # TEST_ONLY: never production evidence.
        self.network_guard = patch("socket.socket.connect", side_effect=AssertionError("TEST_ONLY network forbidden"))
        self.network_guard.start()

    def tearDown(self) -> None:
        for session in self.sessions:
            session.close()
        self.network_guard.stop()
        self.temp.cleanup()

    def session(self, client: _OfflineClient, *, scope: str, condition: str = "card") -> LiveSession:
        session = LiveSession(
            path=self.directory / f"{scope}.sqlite3",
            scope=scope,
            scenario=self.pack["scenarios"]["scenarios"][0],
            policy=self.pack["manifest"]["condition_definitions"][condition]["operation_projection_policy"],
            specification=self.pack["scenarios"],
            manifest=self.pack["manifest"],
            log=EventLog(self.directory / f"{scope}.jsonl", run_mode="TEST_ONLY_offline_script"),
            embedding=self.embedding,
            prompt_bundle=self.bundle,
            client=client,
        )
        self.sessions.append(session)
        return session

    def _standard_script(self, request: dict, scope: str, step_id: str, call_index: int) -> dict:
        step = next(
            step
            for scenario in self.pack["scenarios"]["scenarios"]
            for step in scenario["steps"]
            if step["step_id"] == step_id
        )
        if step["stage"] == "task" and call_index == 0:
            prefix = "gallery" if step_id.startswith("g_") else "notifier"
            version = step_id.rsplit("_", 1)[1]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("lookup_fixture", canonical({"query_key": f"{prefix}_{version}"}))],
            }
        return _final(f"TEST_ONLY scripted final for {step_id}")

    def _run(self, client: _OfflineClient, *, pack: dict | None = None, name: str = "output") -> dict:
        return run_live(
            copy.deepcopy(pack or self.pack),
            self.directory / name,
            {"run_mode": "TEST_ONLY", "frozen_source_verified": False},
            prompt_bundle=self.bundle,
            embedding=self.embedding,
            embedding_report={"status": "TEST_ONLY", "production_verified": False},
            client=client,
        )

    def test_batched_native_message_preserves_content_indexes_arguments_and_open_full_results(self) -> None:
        first_arguments = ' { "query_key" : "gallery_v1" } '
        second_arguments = ' {\n  "date_from" : "2026-08-10"\n } '
        authored = {
            "role": "assistant",
            "content": "TEST_ONLY 我先同时读取两个信息源。",
            "reasoning_content": "",
            "tool_calls": [
                _call("lookup_fixture", first_arguments, index=0),
                _call("browse_memory", second_arguments, index=1),
            ],
        }
        client = _OfflineClient(lambda request, scope, step, index: copy.deepcopy(authored) if index == 0 else _final())
        session = self.session(client, scope="batched")
        step = session.scenario["steps"][1]
        trace = session.run_turn(step)
        self.assertEqual(trace["status"], "passed", trace.get("error"))
        self.assertEqual(len(client.requests), 2)
        second_request = client.requests[1]["request"]["messages"]
        authored_index = second_request.index(authored)
        self.assertEqual(second_request[authored_index], authored)
        self.assertEqual(trace["assistant_messages"][0], authored)
        self.assertEqual([call["call_id"] for call in trace["tool_calls"]], ["TEST_ONLY_call_0", "TEST_ONLY_call_1"])
        for offset, tool in enumerate(trace["tool_calls"], start=1):
            self.assertEqual(
                second_request[authored_index + offset],
                {"role": "tool", "tool_call_id": tool["call_id"], "content": tool["body"]},
            )
        expected_user = self.bundle["steps"][step["step_id"]]["user_prompt"]
        self.assertEqual(sum(message.get("content") == expected_user for message in second_request), 1)
        self.assertTrue(all(check["open_full"] and check["reload_equal"] for check in trace["observation_checks"]))
        closed = session.history()
        self.assertIn(authored, closed)
        self.assertNotIn("TEST_ONLY ephemeral", canonical(closed))
        fixture_call = trace["tool_calls"][0]
        closed_tool = next(message for message in closed if message.get("tool_call_id") == fixture_call["call_id"])
        self.assertIn("[compact_reloadable]", closed_tool["content"])
        self.assertIn(fixture_call["source_id"], closed_tool["content"])
        self.assertNotIn(session.scenario["fixtures"][0]["result"]["content"], closed_tool["content"])
        reopened = session.mem.open_memory(memory_id=fixture_call["source_id"], view="content", detail="full")
        self.assertEqual(reopened["status"], "ok")
        recovered = json.loads(reopened["text"].partition("\ndata:\n")[2])
        self.assertEqual(recovered["output"], fixture_call["body"])

    def test_full_policy_keeps_exact_fixture_result_after_final(self) -> None:
        session = self.session(_OfflineClient(self._standard_script), scope="full", condition="full")
        trace = session.run_turn(session.scenario["steps"][1])
        self.assertEqual(trace["status"], "passed", trace.get("error"))
        tool = trace["tool_calls"][0]
        result = next(message for message in session.history() if message.get("tool_call_id") == tool["call_id"])
        self.assertEqual(result["content"], tool["body"])
        self.assertFalse(trace["observation_checks"][0]["closed_compact"])

    def test_public_journal_replay_never_reinvokes_model_or_tools_and_branches_stay_isolated(self) -> None:
        base_client = _OfflineClient(self._standard_script)
        base = self.session(base_client, scope="replay_base")
        task = base.scenario["steps"][1]
        trace = base.run_turn(task)
        self.assertEqual(trace["status"], "passed", trace.get("error"))
        baseline = base.history()
        branches = []
        for branch_name in ("replay_first", "replay_second"):
            client = _OfflineClient(lambda *args: (_ for _ in ()).throw(AssertionError("model called during replay")))
            branch = self.session(client, scope=branch_name)
            branches.append(branch)
            with (
                patch.object(branch, "_tool_result", side_effect=AssertionError("tool dispatched during replay")),
                patch.object(
                    branch.fixtures, "lookup", side_effect=AssertionError("fixture service rerun during replay")
                ),
            ):
                restored = branch.run_turn(task, replay_trace=trace)
            self.assertEqual(restored["status"], "passed", restored.get("error"))
            self.assertEqual(restored["request_hashes"], trace["request_hashes"])
            self.assertEqual(branch.history(), baseline)
            self.assertEqual(branch.source_map, base.source_map)
            self.assertEqual(client.requests, [])
            self.assertEqual(branch.requests, 0)
        self.assertEqual(len({session.mem.namespace.user_id for session in [base, *branches]}), 3)
        marker = "TEST_ONLY_branch_specific_answer_8de43"
        branches[0].client = _OfflineClient(lambda *args: _final(marker))
        probe = next(step for step in base.scenario["steps"] if step["step_id"] == "g_probe_memory")
        written = branches[0].run_turn(probe)
        self.assertEqual(written["status"], "passed", written.get("error"))
        self.assertIn(marker, canonical(branches[0].history()))
        self.assertEqual(branches[1].history(), baseline)
        self.assertNotIn(marker, canonical(base.history()))
        self.assertEqual(
            branches[1].mem.open_memory(memory_id="g_probe_memory:final", view="content")["status"], "empty"
        )
        self.assertNotIn(marker, canonical(branches[1].mem.read_timeline(date_from="2026-08-11")))
        self.assertNotIn(marker, canonical(branches[1].mem.retrieve(query=marker)))

    def test_invalid_tool_arguments_return_errors_without_broadening_retrieval(self) -> None:
        authored = {
            "role": "assistant",
            "content": "TEST_ONLY invalid argument cases",
            "tool_calls": [
                _call("retrieve_for_turn", "{", index=0),
                _call("retrieve_for_turn", "[]", index=1),
                _call("retrieve_for_turn", '{"query":"secret","source_layers":["not-a-layer"]}', index=2),
            ],
        }
        client = _OfflineClient(lambda request, scope, step, index: copy.deepcopy(authored) if index == 0 else _final())
        session = self.session(client, scope="invalid_arguments")
        with patch.object(
            session.mem, "retrieve_for_turn_structured", side_effect=AssertionError("invalid filter widened to search")
        ) as retrieval:
            trace = session.run_turn(session.scenario["steps"][0])
        self.assertEqual(trace["status"], "passed", trace.get("error"))
        retrieval.assert_not_called()
        self.assertEqual(len(trace["tool_calls"]), 3)
        results = [tool["result"] for tool in trace["tool_calls"]]
        self.assertEqual([result["status"] for result in results], ["invalid", "invalid", "invalid_arguments"])
        self.assertTrue(all(result.get("reason") for result in results))
        self.assertFalse(results[2]["ok"])
        tool_messages = [message for message in client.requests[1]["request"]["messages"] if message["role"] == "tool"]
        self.assertEqual([json.loads(message["content"]) for message in tool_messages], results)

    def test_expired_and_foreign_fixture_requests_stay_unavailable_under_card_policy(self) -> None:
        authored = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call("lookup_fixture", '{"query_key":"gallery_v1"}', index=0),
                _call("lookup_fixture", '{"query_key":"notifier_v1"}', index=1),
            ],
        }
        client = _OfflineClient(lambda request, scope, step, index: copy.deepcopy(authored) if index == 0 else _final())
        session = self.session(client, scope="unavailable_fixtures")
        step = next(step for step in session.scenario["steps"] if step["step_id"] == "g_probe_history")
        trace = session.run_turn(step)
        self.assertEqual(trace["status"], "passed", trace.get("error"))
        self.assertEqual([tool["status"] for tool in trace["tool_calls"]], ["unavailable", "unavailable"])
        self.assertEqual(trace["tool_calls"][0]["result"], session.scenario["unavailable_fixture_result"])
        self.assertEqual(trace["tool_calls"][1]["result"]["reason"], "fixture_outside_scenario")
        self.assertNotIn("gallery_doc_v1", session.source_map)
        self.assertNotIn("notifier_doc_v1", session.source_map)
        self.assertTrue(all(check["reload_equal"] for check in trace["observation_checks"]))

    def test_four_runs_have_twelve_isolated_probes_no_gold_and_no_replay_model_or_fixture_calls(self) -> None:
        pack = copy.deepcopy(self.pack)
        sentinel = "TEST_ONLY_EVALUATOR_GOLD_DO_NOT_SEND_1cde40"
        pack["answer_key"] = {"private_gold": sentinel}
        client = _OfflineClient(self._standard_script)
        actual_lookup = FixtureService.lookup
        observed_lookups = []

        def counted_lookup(service: FixtureService, step_id: str, arguments: dict) -> dict:
            observed_lookups.append((step_id, copy.deepcopy(arguments)))
            return actual_lookup(service, step_id, arguments)

        with patch.object(FixtureService, "lookup", counted_lookup):
            report = self._run(client, pack=pack)
        failures = [
            (run["run_id"], run["error"], [(branch["step_id"], branch["error"]) for branch in run["branches"]])
            for run in report["runs"]
            if run["status"] != "passed"
        ]
        self.assertEqual(report["status"], "passed", failures)
        self.assertEqual(len(report["runs"]), 4)
        self.assertEqual(sum(len(run["branches"]) for run in report["runs"]), 12)
        self.assertEqual(len(client.requests), 40)  # 4 * (5 setup finals + 2 fixture calls + 3 probe finals).
        self.assertEqual(len(observed_lookups), 8)  # Replay must consume saved observations only.
        self.assertIsNone(report["scores"])
        self.assertEqual(report["run_mode"], "offline_live_runner_test")
        self.assertEqual(report["actual_model"], "offline_injected_not_deepseek")
        self.assertTrue(report["forced_routing"])
        self.assertFalse(report["akane_prompt_verified"])
        self.assertFalse(report["akane_engine_verified"])
        self.assertEqual(report["ledger"]["paid_cost_cny"], "0")
        self.assertNotIn(sentinel, canonical(client.requests))
        self.assertTrue(
            all(run["fixtures_observed"] and all(run["fixtures_observed"].values()) for run in report["runs"])
        )
        self.assertTrue(all(branch["baseline_equal"] for run in report["runs"] for branch in run["branches"]))
        self.assertEqual(len({run["control_hash"] for run in report["runs"]}), 1)
        for scenario in self.pack["scenarios"]["scenarios"]:
            probes = [step for step in scenario["steps"] if step["stage"] == "probe"]
            setup_requests = [row for row in client.requests if row["scope"].endswith("__setup")]
            for probe in probes:
                self.assertTrue(all(probe["user_text"] not in canonical(row["request"]) for row in setup_requests))
                own_requests = [row for row in client.requests if row["step_id"] == probe["step_id"]]
                for other_probe in probes:
                    if other_probe is not probe:
                        self.assertTrue(
                            all(other_probe["user_text"] not in canonical(row["request"]) for row in own_requests)
                        )
        persisted = json.loads((self.directory / "output" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["runs"], report["runs"])

    def test_failed_probe_is_preserved_and_later_independent_probes_still_run(self) -> None:
        pack = copy.deepcopy(self.pack)
        pack["manifest"]["runs"] = pack["manifest"]["runs"][:1]

        def script(request: dict, scope: str, step: str, index: int) -> dict:
            if step == "g_probe_history":
                raise PilotFailure("TEST_ONLY_provider_failure")
            return self._standard_script(request, scope, step, index)

        client = _OfflineClient(script)
        report = self._run(client, pack=pack)
        self.assertEqual(report["status"], "failed")
        branches = report["runs"][0]["branches"]
        self.assertEqual(len(branches), 3)
        self.assertEqual([branch["status"] for branch in branches], ["failed", "passed", "passed"])
        self.assertEqual(branches[0]["error"], "TEST_ONLY_provider_failure")
        self.assertIsNone(branches[0]["speech"])
        self.assertTrue(all(branch["baseline_equal"] for branch in branches))
        persisted = json.loads((self.directory / "output" / report["runs"][0]["run_id"] / "trace.json").read_text())
        self.assertEqual(persisted["branches"], branches)

    def test_setup_failure_marks_all_probe_slots_not_run_instead_of_dropping_them(self) -> None:
        pack = copy.deepcopy(self.pack)
        pack["manifest"]["runs"] = pack["manifest"]["runs"][:1]

        def script(request: dict, scope: str, step: str, index: int) -> dict:
            if step == "g_task_v1":
                return {"role": "assistant", "content": "TEST_ONLY broken JSON"}
            return self._standard_script(request, scope, step, index)

        client = _OfflineClient(script)
        report = self._run(client, pack=pack)
        row = report["runs"][0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], "setup_turn_failed:invalid_final_json")
        self.assertEqual(len(row["setup"]), 2)
        self.assertEqual(row["setup"][-1]["status"], "failed")
        self.assertEqual(len(row["branches"]), 3)
        self.assertTrue(all(branch["status"] == "not_run" for branch in row["branches"]))
        self.assertTrue(all(branch["error"] == "setup_not_complete" for branch in row["branches"]))
        self.assertTrue(all("_probe_" not in request["step_id"] for request in client.requests))


if __name__ == "__main__":
    unittest.main()
