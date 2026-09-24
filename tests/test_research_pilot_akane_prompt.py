"""Prompt composition invariants; optional real Akane builder acceptance.

Set AKANE_TEST_SOURCE_ROOT and, when needed, AKANE_TEST_PYTHON to exercise the
existing host checkout in an isolated child. These tests never call a model.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.research_pilot.akane_prompt import (
    FORMAT,
    AkanePromptError,
    _hash,
    _isolated_environment,
    _normalize_steps,
    compose_akane_messages,
    export_akane_prompt_bundle,
    load_akane_prompt_bundle,
)


def composition_fixture() -> dict:
    # A transport-layout fixture, never labelled as an actual Akane export.
    bundle = {
        "format": FORMAT,
        "shared": {
            "system_prompt": "Fixture system contract",
            "system_extra_blocks": ["Fixture final contract"],
            "history_prefix_messages": [{"role": "user", "content": "Fixed persona fixture"}],
        },
        "steps": {
            "probe": {
                "user_prompt": "time: 2026-09-06 周日 09:00\nUser: current-sentinel",
                "ephemeral_turns": [{"role": "user", "content": "current-time-ephemeral"}],
            }
        },
        "evidence": {"actual_builder_called": False},
    }
    bundle["bundle_hash"] = _hash(bundle)
    return bundle


class AkanePromptCompositionTests(unittest.TestCase):
    def test_current_user_is_replaced_once_and_native_wire_remains_exact(self) -> None:
        bundle = composition_fixture()
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "MemCore current projection", "extension": {"kept": True}},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "original-call-id",
                        "type": "function",
                        "function": {"name": "open_memory", "arguments": '{ "memory_id" : "original-id" }'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "original-call-id", "content": "full original observation"},
        ]
        original_history, original_bundle = copy.deepcopy(history), copy.deepcopy(bundle)
        result = compose_akane_messages(
            bundle,
            history_messages=history,
            step_id="probe",
            current_message_index=2,
            extra_system_blocks=["readback rule"],
        )
        self.assertEqual(history, original_history)
        self.assertEqual(bundle, original_bundle)
        self.assertEqual(result["history_payloads"][:2], history[:2])
        self.assertEqual(result["history_payloads"][3:], history[3:])
        self.assertEqual(result["history_payloads"][2]["extension"], {"kept": True})
        self.assertEqual(result["history_payloads"][2]["content"], bundle["steps"]["probe"]["user_prompt"])
        self.assertEqual(sum("current-sentinel" in str(m.get("content")) for m in result["messages"]), 1)
        for index, payload in zip(result["history_message_indexes"], result["history_payloads"]):
            self.assertEqual(result["messages"][index], payload)
        current_slot = result["history_message_indexes"][2]
        self.assertEqual(result["messages"][current_slot + 1], result["ephemeral_messages"][0])
        self.assertEqual(result["messages"][current_slot + 2], history[3])
        self.assertNotIn("current-time-ephemeral", json.dumps(result["history_payloads"]))
        self.assertNotIn("Fixed persona fixture", json.dumps(result["history_payloads"]))
        result["messages"][-1]["content"] = "mutated request copy"
        self.assertEqual(result["history_payloads"][-1], history[-1])

    def test_invalid_stimulus_index_and_mutated_bundle_fail(self) -> None:
        bundle = composition_fixture()
        history = [{"role": "user", "content": "current"}, {"role": "tool", "content": "body"}]
        for index in (-1, 1, 2, True):
            with self.subTest(index=index), self.assertRaisesRegex(AkanePromptError, "current_stimulus_index"):
                compose_akane_messages(bundle, history_messages=history, step_id="probe", current_message_index=index)
        bundle["shared"]["system_prompt"] = "edited after export"
        with self.assertRaisesRegex(AkanePromptError, "bundle_hash_mismatch"):
            compose_akane_messages(bundle, history_messages=history, step_id="probe", current_message_index=0)

    def test_step_boundary_excludes_gold_and_fixture_payloads(self) -> None:
        step = {"step_id": "a", "user_text": "question", "timestamp": "2026-09-06T09:00:00+08:00"}
        self.assertEqual(len(_normalize_steps([step, dict(step)])), 1)
        with self.assertRaisesRegex(AkanePromptError, "conflicting_prompt_step_id"):
            _normalize_steps([step, {**step, "user_text": "another question"}])
        with patch("subprocess.run", side_effect=AssertionError("must reject before starting child")):
            for name in ("gold", "answer_key", "fixture", "memory", "care_state"):
                with self.subTest(name=name), self.assertRaisesRegex(AkanePromptError, "current_input_only"):
                    export_akane_prompt_bundle(Path("unused"), Path("unused"), [{**step, name: "hidden"}])

    def test_subprocess_environment_drops_credentials_and_runtime_overrides(self) -> None:
        env = _isolated_environment(
            Path("new-export"),
            {
                "PATH": "interpreter-path",
                "SystemRoot": "os-root",
                "DEEPSEEK_API_KEY": "credential-sentinel",
                "TEXT_API_KEY": "credential-sentinel",
                "AKANE_DATA_ROOT": "daily-instance",
                "AKANE_ENV_FILE": "daily.env",
                "PERSONA_CONFIG_PATH": "private-persona",
                "PYTHONPATH": "arbitrary-code",
            },
        )
        self.assertEqual(env["PATH"], "interpreter-path")
        self.assertFalse(any(name.endswith("API_KEY") for name in env))
        self.assertNotIn("credential-sentinel", json.dumps(env))
        self.assertNotIn("daily-instance", json.dumps(env))
        self.assertNotIn("PERSONA_CONFIG_PATH", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["MEMCORE_ENABLE_FLAVOR"], "false")

    def test_worker_guard_blocks_private_reads_writes_and_network_before_io(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            akane, output = root / "source", root / "export"
            akane.mkdir()
            output.mkdir()
            private = akane / ".env"
            private.write_text("synthetic-private-sentinel", encoding="utf-8")
            code = """
import json, pathlib, socket, sys
from examples.research_pilot.akane_prompt import _install_export_audit_guard, AkanePromptError
akane, output = map(pathlib.Path, sys.argv[1:])
counts = _install_export_audit_guard(akane.resolve(), output.resolve())
observed = []
for action in (lambda: (akane / '.env').read_text(), lambda: (akane / 'bad.txt').write_text('bad'),
               lambda: socket.getaddrinfo('127.0.0.1', 1)):
    try:
        action()
    except AkanePromptError as exc:
        observed.append(exc.code)
print(json.dumps({'observed': observed, 'counts': counts}))
"""
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(akane), str(output)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )
            record = json.loads(result.stdout)
            self.assertEqual(len(record["observed"]), 3)
            self.assertEqual(record["counts"], {"network_attempts": 1, "blocked_reads": 1, "blocked_writes": 1})
            self.assertFalse((akane / "bad.txt").exists())


@unittest.skipUnless(os.environ.get("AKANE_TEST_SOURCE_ROOT"), "set AKANE_TEST_SOURCE_ROOT for real host builder")
class RealAkanePromptExportTests(unittest.TestCase):
    def test_real_builder_profile_persona_time_and_isolation(self) -> None:
        steps = [
            {"step_id": "first", "user_text": "prompt-export-first-sentinel", "timestamp": "2026-09-06T09:00:00+08:00"},
            {
                "step_id": "second",
                "user_text": "prompt-export-second-sentinel",
                "timestamp": "2026-09-07T10:00:00+08:00",
            },
        ]
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "never-inherit-credential"}),
        ):
            output = Path(temp) / "export"
            bundle = export_akane_prompt_bundle(
                Path(os.environ["AKANE_TEST_SOURCE_ROOT"]),
                output,
                steps,
                python_executable=os.environ.get("AKANE_TEST_PYTHON"),
            )
            self.assertEqual(load_akane_prompt_bundle(output / "akane_prompt_bundle.json"), bundle)
            evidence = bundle["evidence"]
            self.assertTrue(evidence["actual_builder_called"])
            self.assertEqual(evidence["actual_builder_calls"], 4)
            self.assertTrue(evidence["imported_source_modules_verified"])
            self.assertEqual(evidence["profile"]["id"], "desktop_pet")
            self.assertFalse(evidence["care_enabled"])
            self.assertFalse(evidence["akane_engine_verified"])
            self.assertEqual(evidence["audit_guard"], {"network_attempts": 0, "blocked_reads": 0, "blocked_writes": 0})
            self.assertEqual(evidence["paid_requests"], 0)
            self.assertIn("2026-09-06 周日 09:00", bundle["steps"]["first"]["user_prompt"])
            self.assertIn("2026-09-07 周一 10:00", bundle["steps"]["second"]["user_prompt"])
            self.assertNotIn("state_request", bundle["shared"]["system_prompt"])
            self.assertNotIn("affinity", bundle["shared"]["system_prompt"])
            self.assertTrue(any("# Akane" in row["content"] for row in bundle["shared"]["history_prefix_messages"]))
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertNotIn("never-inherit-credential", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
