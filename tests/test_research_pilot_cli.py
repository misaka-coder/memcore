"""Offline CLI checks for frozen inputs, ledger identity, and production gating."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from examples.research_pilot import __main__ as cli
from examples.research_pilot.budget import BudgetLedger
from examples.research_pilot.production_embedding import PROBE_TEXTS
from examples.research_pilot.runner import digest, write_json


def embedding_report() -> dict:
    return {
        "format": "memcore_pilot_production_embedding_preflight_v1",
        "status": "passed",
        "reason": None,
        "model_id": "BAAI/bge-m3",
        "provider": "HuggingFaceEmbeddingProvider",
        "adapter_version": "st-local-v1",
        "device": "cuda",
        "local_files_only": True,
        "hashed_fallback": False,
        "paid_embedding_api_calls": 0,
        "downloaded_model": False,
        "dimension": 1024,
        "finite_normalized_vectors": True,
        "synthetic_texts": list(PROBE_TEXTS),
        "similar_score": 0.88,
        "unrelated_score": 0.41,
        "semantic_gap": 0.47,
        "required_gap": 0.05,
        "repeat_score": 1.0,
        "vector_fingerprint": "a" * 64,
        "probe_scope": "small_synthetic_health_check_not_retrieval_benchmark",
        "network_attempt_count": 0,
    }


class PilotCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.original = self.directory / "original"
        self.original.mkdir()
        for name in ("memcore/stub.py", "examples/research_pilot/__main__.py", "docs/research/pilot_v1/stub.json"):
            path = self.original / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if name.endswith(".json") else "# source fixture\n", encoding="utf-8")
        self.stage_ledger = self.original / ".research-runs/stage1-budget.sqlite3"
        self.stage_ledger.parent.mkdir()
        with BudgetLedger(self.stage_ledger) as ledger:
            self.initial_ledger = ledger.snapshot()
        self.bundle_path = self.directory / "prompt_bundle.json"
        bundle = {"format": "akane_research_prompt_bundle_v1", "shared": {}, "steps": {}}
        bundle["bundle_hash"] = digest(bundle)
        write_json(self.bundle_path, bundle)
        self.report_path = self.directory / "embedding_report.json"
        write_json(self.report_path, embedding_report())
        self.pack = {"manifest": {"runs": [{"run_id": "synthetic"}]}, "scenarios": {"scenarios": []}}
        self.frozen = self.original / ".research-runs/frozen"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def invoke(self, root: Path, *arguments: str) -> tuple[int, dict]:
        output = io.StringIO()
        with (
            patch.object(cli, "__file__", str(root / "examples/research_pilot/__main__.py")),
            patch("sys.argv", ["research_pilot", *map(str, arguments)]),
            patch.object(cli, "validate_pack", return_value=self.pack),
            patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
            contextlib.redirect_stdout(output),
        ):
            code = cli.main()
        return code, json.loads(output.getvalue())

    def freeze(self, *, include_inputs: bool = True) -> None:
        arguments = ["freeze", "--output", str(self.frozen)]
        if include_inputs:
            arguments += ["--prompt-bundle", str(self.bundle_path), "--embedding-report", str(self.report_path)]
        code, result = self.invoke(self.original, *arguments)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "frozen")

    def live(self, output: Path | None = None) -> tuple[int, dict]:
        return self.invoke(self.frozen, "live", "--output", output or self.directory / "output")

    def test_freeze_copies_exact_bytes_and_binds_inputs_to_existing_stage(self) -> None:
        self.freeze()
        frozen = json.loads((self.frozen / "frozen_source.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen["stage_budget"]["stage_id"], self.initial_ledger["stage_id"])
        for name, source in (
            ("inputs/akane_prompt_bundle.json", self.bundle_path),
            ("inputs/embedding_preflight.json", self.report_path),
        ):
            self.assertEqual((self.frozen / name).read_bytes(), source.read_bytes())
            self.assertEqual(frozen["input_hashes"][name], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertNotIn(str(self.directory), json.dumps(frozen))

    def test_freeze_requires_both_optional_live_inputs_together(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.invoke(self.original, "freeze", "--output", self.frozen, "--prompt-bundle", self.bundle_path)
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.frozen.exists())

    def test_live_refuses_freeze_without_live_inputs_before_loading_model(self) -> None:
        self.freeze(include_inputs=False)
        with patch("examples.research_pilot.production_embedding.create_production_embedding") as factory:
            code, result = self.live()
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "frozen_live_inputs_required")
        factory.assert_not_called()
        self.assertFalse((self.directory / "output").exists())

    def test_live_detects_source_or_input_tampering_before_any_model_load(self) -> None:
        self.freeze()
        source = self.frozen / "memcore/stub.py"
        original = source.read_bytes()
        source.write_bytes(original + b"# changed\n")
        with patch("examples.research_pilot.production_embedding.create_production_embedding") as factory:
            code, result = self.live()
            self.assertEqual(code, 1)
            self.assertEqual(result["error"], "frozen_source_changed")
            source.write_bytes(original)
            with (self.frozen / "inputs/akane_prompt_bundle.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            code, result = self.live()
            self.assertEqual(code, 1)
            self.assertEqual(result["error"], "frozen_live_input_hash_mismatch")
            factory.assert_not_called()

    def test_missing_bound_ledger_cannot_create_a_new_budget(self) -> None:
        self.freeze()
        metadata_path = self.frozen / "frozen_source.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["stage_budget"]["path_from_source"] = "../wrong-ledger.sqlite3"
        write_json(metadata_path, metadata)
        with patch("examples.research_pilot.production_embedding.create_production_embedding") as factory:
            code, result = self.live()
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "existing_budget_ledger_required")
        factory.assert_not_called()
        self.assertFalse((self.frozen.parent / "wrong-ledger.sqlite3").exists())

    def test_success_wires_frozen_inputs_runtime_only_path_and_same_ledger_without_paid_calls(self) -> None:
        self.freeze()
        embedding = SimpleNamespace(name="BAAI/bge-m3", dimension=1024)
        client = Mock()
        current_health = embedding_report()
        current_health.pop("network_attempt_count")  # This run's health probe is not the old socket audit.
        current_health.update(similar_score=0.8801, semantic_gap=0.4701, vector_fingerprint="b" * 64)

        def fake_run(pack, output, frozen, **kwargs):
            output.mkdir()
            result = {"status": "passed", "embedding_verification": kwargs["embedding_report"]}
            write_json(output / "report.json", result)
            return result

        with (
            patch(
                "examples.research_pilot.production_embedding.create_production_embedding", return_value=embedding
            ) as factory,
            patch(
                "examples.research_pilot.production_embedding.verify_production_embedding", return_value=current_health
            ) as verify,
            patch("examples.research_pilot.transport.BudgetedTransport", return_value=client) as transport,
            patch("examples.research_pilot.live.run_live", side_effect=fake_run) as run,
            patch.dict("os.environ", {"PILOT_EMBEDDING_MODEL_PATH": "runtime-only-path-sentinel"}),
        ):
            code, result = self.live()
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ledger"], self.initial_ledger)
        factory.assert_called_once_with(local_model_path="runtime-only-path-sentinel", device="cuda")
        verify.assert_called_once_with(embedding)
        self.assertEqual(transport.call_args.kwargs["capture_dir"], self.directory / "output/requests")
        self.assertIs(run.call_args.kwargs["embedding"], embedding)
        self.assertIs(run.call_args.kwargs["client"], client)
        self.assertEqual(run.call_args.args[0], self.pack)
        self.assertEqual(run.call_args.args[2]["stage_budget"]["stage_id"], self.initial_ledger["stage_id"])
        self.assertNotIn("runtime-only-path-sentinel", json.dumps(result))
        self.assertNotIn("runtime-only-path-sentinel", json.dumps(run.call_args.kwargs["embedding_report"]))
        report = run.call_args.kwargs["embedding_report"]
        self.assertEqual(report["vector_fingerprint"], "a" * 64)
        self.assertEqual(report["runtime_verification"]["vector_fingerprint"], "b" * 64)
        self.assertEqual(report["runtime_verification"]["similar_score"], 0.8801)
        self.assertNotIn("network_attempt_count", report["runtime_verification"])
        self.assertRegex(report["runtime_package_versions"]["python"], r"\A[0-9]+\.[0-9]+\.[0-9]+\Z")
        self.assertIn("torch", report["runtime_package_versions"])
        with BudgetLedger(self.stage_ledger, expected_stage_id=self.initial_ledger["stage_id"]) as ledger:
            self.assertEqual(ledger.snapshot(), self.initial_ledger)

    def test_loaded_embedding_mismatch_stops_before_transport_and_output_creation(self) -> None:
        self.freeze()
        with (
            patch(
                "examples.research_pilot.production_embedding.create_production_embedding",
                return_value=SimpleNamespace(name="BAAI/bge-m3", dimension=768),
            ),
            patch("examples.research_pilot.transport.BudgetedTransport") as transport,
        ):
            code, result = self.live()
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "embedding_identity_mismatch")
        transport.assert_not_called()
        self.assertFalse((self.directory / "output").exists())
        self.assertEqual(result["ledger"], self.initial_ledger)

    def test_failed_or_degraded_embedding_report_is_rejected_before_load(self) -> None:
        report = embedding_report()
        report["hashed_fallback"] = True
        write_json(self.report_path, report)
        self.freeze()
        with patch("examples.research_pilot.production_embedding.create_production_embedding") as factory:
            code, result = self.live()
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "invalid_embedding_report")
        factory.assert_not_called()

    def test_current_health_failure_or_score_drift_stops_before_transport(self) -> None:
        self.freeze()
        cases = [
            ({"status": "failed", "reason": "semantic_health_check_failed"}, "runtime_embedding_verification_failed"),
            ({"similar_score": 0.881, "semantic_gap": 0.471}, "runtime_embedding_preflight_mismatch"),
            ({"unrelated_score": 0.411, "semantic_gap": 0.469}, "runtime_embedding_preflight_mismatch"),
            ({"repeat_score": 0.9998}, "runtime_embedding_preflight_mismatch"),
        ]
        for changes, expected_error in cases:
            current_health = {**embedding_report(), **changes}
            with (
                self.subTest(changes=changes),
                patch(
                    "examples.research_pilot.production_embedding.create_production_embedding",
                    return_value=SimpleNamespace(name="BAAI/bge-m3", dimension=1024),
                ),
                patch(
                    "examples.research_pilot.production_embedding.verify_production_embedding",
                    return_value=current_health,
                ) as verify,
                patch("examples.research_pilot.transport.BudgetedTransport") as transport,
            ):
                code, result = self.live()
                self.assertEqual(code, 1)
                self.assertEqual(result["error"], expected_error)
                self.assertEqual(result["ledger"], self.initial_ledger)
                self.assertFalse((self.directory / "output").exists())
                verify.assert_called_once()
                transport.assert_not_called()

    def test_runtime_versions_use_loaded_safe_module_values_without_paths(self) -> None:
        with patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(__version__="2.11.0+cu128"),
                "numpy": SimpleNamespace(__version__="unapproved/path/sentinel"),
                "sentence_transformers": None,
            },
        ):
            versions = cli._runtime_package_versions()
        self.assertEqual(versions["torch"], "2.11.0+cu128")
        self.assertIsNone(versions["numpy"])
        self.assertIsNone(versions["sentence-transformers"])
        self.assertNotIn("sentinel", json.dumps(versions))

    def test_runtime_failure_does_not_echo_configuration_or_exception_details(self) -> None:
        self.freeze()
        with patch(
            "examples.research_pilot.production_embedding.create_production_embedding",
            side_effect=RuntimeError("secret-config-and-cache-path-sentinel"),
        ):
            code, result = self.live()
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "live_command_failed")
        self.assertNotIn("secret-config", json.dumps(result))

    def test_live_output_must_be_new(self) -> None:
        self.freeze()
        output = self.directory / "output"
        output.mkdir()
        sentinel = output / "existing.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with patch("examples.research_pilot.production_embedding.create_production_embedding") as factory:
            code, result = self.live(output)
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "live_output_already_exists")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        factory.assert_not_called()

    def test_live_cannot_replace_frozen_scenarios_with_an_external_pack(self) -> None:
        self.freeze()
        with patch("examples.research_pilot.production_embedding.create_production_embedding") as factory:
            code, result = self.invoke(
                self.frozen,
                "live",
                "--output",
                self.directory / "output",
                "--pack",
                self.original / "docs/research/pilot_v1",
            )
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], "live_pack_must_be_frozen")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
