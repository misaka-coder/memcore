"""Offline preparation and separately bounded paid preflight/scenario commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .runner import run_smoke, source_snapshot, write_json
from .validation import validate_pack

_LIVE_INPUTS = ("inputs/akane_prompt_bundle.json", "inputs/embedding_preflight.json")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LIVE_LIBRARY_ERRORS = {
    "existing_budget_ledger_required",
    "budget_stage_id_missing",
    "budget_stage_id_mismatch",
    "invalid_expected_stage_id",
    "invalid_budget_stage_id",
    "unsupported_budget_schema",
    "budget_limit_mismatch",
    "pending_reservation_blocks_dispatch",
    "budget_limit_exceeded",
    "local_embedding_model_missing",
    "production_embedding_load_failed",
    "embedding_invalid_dimension",
    "invalid_akane_prompt_bundle",
    "akane_prompt_bundle_hash_mismatch",
}


class PilotCLIError(RuntimeError):
    """Fixed CLI diagnostic, with no paths, provider output, or configuration."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _embedding_report(path: Path) -> dict[str, Any]:
    """Validate the production health evidence and forward only safe fields."""
    return _safe_embedding_report(json.loads(path.read_text(encoding="utf-8")), require_network_evidence=True)


def _safe_embedding_report(value: Any, *, require_network_evidence: bool) -> dict[str, Any]:
    from .production_embedding import PROBE_TEXTS

    if not isinstance(value, dict):
        raise PilotCLIError("invalid_embedding_report")
    fixed = {
        "format": "memcore_pilot_production_embedding_preflight_v1",
        "status": "passed",
        "reason": None,
        "provider": "HuggingFaceEmbeddingProvider",
        "adapter_version": "st-local-v1",
        "device": "cuda",
        "local_files_only": True,
        "hashed_fallback": False,
        "paid_embedding_api_calls": 0,
        "downloaded_model": False,
        "finite_normalized_vectors": True,
        "synthetic_texts": list(PROBE_TEXTS),
        "required_gap": 0.05,
        "probe_scope": "small_synthetic_health_check_not_retrieval_benchmark",
    }
    if require_network_evidence:
        fixed["network_attempt_count"] = 0
    if any(value.get(key) != expected or type(value.get(key)) is not type(expected) for key, expected in fixed.items()):
        raise PilotCLIError("invalid_embedding_report")
    if (
        not isinstance(value.get("model_id"), str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value["model_id"]) is None
        or type(value.get("dimension")) is not int
        or value["dimension"] <= 0
        or not isinstance(value.get("vector_fingerprint"), str)
        or _HASH.fullmatch(value["vector_fingerprint"]) is None
    ):
        raise PilotCLIError("invalid_embedding_report")
    scores = ("similar_score", "unrelated_score", "semantic_gap", "repeat_score")
    if any(type(value.get(key)) not in {int, float} or not math.isfinite(value[key]) for key in scores):
        raise PilotCLIError("invalid_embedding_report")
    if (
        value["similar_score"] <= 0
        or value["semantic_gap"] <= 0.05
        or value["repeat_score"] <= 0.999
        or abs(value["similar_score"] - value["unrelated_score"] - value["semantic_gap"]) > 0.00001
    ):
        raise PilotCLIError("invalid_embedding_report")
    safe = {**fixed, **{key: value[key] for key in (*scores, "model_id", "dimension", "vector_fingerprint")}}
    versions = value.get("package_versions")
    packages = {"sentence-transformers", "transformers", "torch", "numpy", "huggingface-hub"}
    if (
        isinstance(versions, dict)
        and set(versions) <= packages
        and all(
            isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", version)
            for version in versions.values()
        )
    ):
        safe["package_versions"] = versions
    if value.get("runtime") == "existing_Akane_environment":
        safe["runtime"] = value["runtime"]
    elapsed = value.get("elapsed_seconds")
    if type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0:
        safe["elapsed_seconds"] = elapsed
    return safe


def _runtime_package_versions() -> dict[str, str | None]:
    """Describe the Python process and loaded modules, without importing models."""
    versions: dict[str, str | None] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    modules = {
        "sentence-transformers": "sentence_transformers",
        "transformers": "transformers",
        "torch": "torch",
        "numpy": "numpy",
        "huggingface-hub": "huggingface_hub",
    }
    for package, module_name in modules.items():
        version = getattr(sys.modules.get(module_name), "__version__", None)
        versions[package] = (
            str(version) if isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", version) else None
        )
    return versions


def _run_live_command(root: Path, args: argparse.Namespace) -> int:
    from .akane_prompt import load_akane_prompt_bundle
    from .budget import BudgetLedger
    from .production_embedding import create_production_embedding, verify_production_embedding

    ledger = None
    ledger_state = None
    status, error = "failed", None
    report_path = None
    try:
        if args.output is None:
            raise PilotCLIError("live_output_required")
        if args.output.exists():
            raise PilotCLIError("live_output_already_exists")
        frozen_file = root / "frozen_source.json"
        if not frozen_file.is_file():
            raise PilotCLIError("frozen_source_required")
        frozen = json.loads(frozen_file.read_text(encoding="utf-8"))
        if not isinstance(frozen, dict):
            raise PilotCLIError("invalid_frozen_source")
        if not args.pack.resolve().is_relative_to(root.resolve()):
            raise PilotCLIError("live_pack_must_be_frozen")
        actual_source = source_snapshot(root, args.pack)
        if actual_source["source_file_hashes"] != frozen.get("source_file_hashes") or actual_source[
            "source_fingerprint"
        ] != frozen.get("source_fingerprint"):
            raise PilotCLIError("frozen_source_changed")
        hashes = frozen.get("input_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(_LIVE_INPUTS):
            raise PilotCLIError("frozen_live_inputs_required")
        for name in _LIVE_INPUTS:
            path = root / name
            if not path.is_file():
                raise PilotCLIError("frozen_live_input_missing")
            if (
                not isinstance(hashes[name], str)
                or _HASH.fullmatch(hashes[name]) is None
                or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]
            ):
                raise PilotCLIError("frozen_live_input_hash_mismatch")
        prompt_bundle = load_akane_prompt_bundle(root / _LIVE_INPUTS[0])
        embedding_report = _embedding_report(root / _LIVE_INPUTS[1])
        pack = validate_pack(args.pack)
        binding = frozen.get("stage_budget")
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("stage_id"), str)
            or not isinstance(binding.get("path_from_source"), str)
            or Path(binding["path_from_source"]).is_absolute()
        ):
            raise PilotCLIError("invalid_frozen_budget_binding")
        ledger = BudgetLedger(root / binding["path_from_source"], limit_cny="50", expected_stage_id=binding["stage_id"])
        ledger.assert_ready()
        embedding = create_production_embedding(
            local_model_path=os.environ.get("PILOT_EMBEDDING_MODEL_PATH"), device="cuda"
        )
        if embedding.name != embedding_report["model_id"] or embedding.dimension != embedding_report["dimension"]:
            raise PilotCLIError("embedding_identity_mismatch")
        current_health = verify_production_embedding(embedding)
        try:
            current_health = _safe_embedding_report(current_health, require_network_evidence=False)
        except PilotCLIError:
            raise PilotCLIError("runtime_embedding_verification_failed") from None
        if (
            current_health["model_id"] != embedding_report["model_id"]
            or current_health["dimension"] != embedding_report["dimension"]
            or any(
                abs(Decimal(str(current_health[key])) - Decimal(str(embedding_report[key]))) > Decimal("0.0001")
                for key in ("similar_score", "unrelated_score", "repeat_score")
            )
        ):
            raise PilotCLIError("runtime_embedding_preflight_mismatch")
        embedding_report["runtime_verification"] = current_health
        embedding_report["runtime_package_versions"] = _runtime_package_versions()
        from .live import run_live
        from .transport import BudgetedTransport

        args.output.mkdir(parents=True, exist_ok=False)
        client = BudgetedTransport(ledger, capture_dir=args.output / "requests")
        report = run_live(
            pack,
            args.output / "run",
            frozen,
            prompt_bundle=prompt_bundle,
            embedding=embedding,
            embedding_report=embedding_report,
            client=client,
        )
        if not isinstance(report, dict) or report.get("status") not in {"passed", "failed"}:
            raise PilotCLIError("invalid_live_report")
        status = report["status"]
        error = None if status == "passed" else "live_run_failed"
    except BaseException as exc:
        if isinstance(exc, PilotCLIError):
            error = exc.code
        elif isinstance(exc, (KeyboardInterrupt, SystemExit)):
            error = "interrupted"
        else:
            candidate = getattr(exc, "code", None)
            if candidate is None and type(exc).__name__ == "AkanePromptError":
                candidate = str(exc)
            error = candidate if candidate in _LIVE_LIBRARY_ERRORS else "live_command_failed"
    finally:
        if ledger is not None:
            try:
                ledger_state = ledger.snapshot()
            except Exception:
                status, error = "failed", "budget_ledger_failed"
            finally:
                ledger.close()
        if args.output is not None and (args.output / "run/report.json").is_file():
            report_path = str((args.output / "run/report.json").resolve())
    print(
        json.dumps(
            {"status": status, "error": error, "report": report_path, "ledger": ledger_state}, ensure_ascii=False
        )
    )
    return 0 if status == "passed" else 1


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Synthetic pilot: validate/smoke/freeze are offline; preflight/live are paid."
    )
    parser.add_argument("command", choices=["validate", "smoke", "init-budget", "freeze", "preflight", "live"])
    parser.add_argument("--pack", type=Path, default=root / "docs/research/pilot_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt-bundle", type=Path)
    parser.add_argument("--embedding-report", type=Path)
    args = parser.parse_args()
    if args.command == "live":
        return _run_live_command(root, args)
    pack = validate_pack(args.pack)
    stage_ledger = root / ".research-runs/stage1-budget.sqlite3"
    if args.command == "init-budget":
        from .budget import BudgetLedger

        if (root / "frozen_source.json").exists():
            parser.error("initialize the stage budget only from the original repository")
        stage_ledger.parent.mkdir(parents=True, exist_ok=True)
        with BudgetLedger(stage_ledger, limit_cny="50") as ledger:
            print(json.dumps({"status": "budget_ready", "ledger": ledger.snapshot(), "paid_cost_cny": 0}))
        return 0
    if args.command == "validate":
        print(json.dumps({"status": "valid", "runs": len(pack["manifest"]["runs"]), "paid_cost_cny": 0}))
        return 0
    if args.command == "freeze":
        from .budget import BudgetLedger

        if args.output is None:
            parser.error("freeze requires --output for a new source-only directory")
        if not args.output.resolve().is_relative_to((root / ".research-runs").resolve()):
            parser.error("freeze output must stay under the repository's .research-runs directory")
        if not stage_ledger.is_file():
            parser.error("initialize the fixed stage ledger with init-budget before freezing")
        if (args.prompt_bundle is None) != (args.embedding_report is None):
            parser.error("freeze requires --prompt-bundle and --embedding-report together")
        if args.prompt_bundle is not None and (not args.prompt_bundle.is_file() or not args.embedding_report.is_file()):
            parser.error("freeze prompt bundle and embedding report must be existing files")
        snapshot = source_snapshot(root, args.pack)
        with BudgetLedger(stage_ledger, limit_cny="50") as ledger:
            ledger.assert_ready()
            snapshot["stage_budget"] = {
                "stage_id": ledger.snapshot()["stage_id"],
                "path_from_source": os.path.relpath(stage_ledger, args.output).replace("\\", "/"),
            }
        args.output.mkdir(parents=True, exist_ok=False)
        for name in snapshot["source_file_hashes"]:
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        if args.prompt_bundle is not None:
            snapshot["input_hashes"] = {}
            for source, name in zip((args.prompt_bundle, args.embedding_report), _LIVE_INPUTS):
                target = args.output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                body = source.read_bytes()
                target.write_bytes(body)
                snapshot["input_hashes"][name] = hashlib.sha256(body).hexdigest()
        write_json(args.output / "frozen_source.json", snapshot)
        print(
            json.dumps({"status": "frozen", "source_fingerprint": snapshot["source_fingerprint"], "paid_cost_cny": 0})
        )
        return 0
    if args.command == "preflight":
        from .budget import BudgetLedger
        from .preflight import run_preflight

        if args.output is None:
            parser.error("preflight requires --output")
        frozen_file = root / "frozen_source.json"
        if not frozen_file.is_file():
            parser.error("run preflight from a source directory created by freeze")
        frozen = json.loads(frozen_file.read_text(encoding="utf-8"))
        if source_snapshot(root, args.pack)["source_file_hashes"] != frozen["source_file_hashes"]:
            parser.error("frozen source changed; create a fresh freeze before dispatch")
        binding = frozen["stage_budget"]
        ledger = BudgetLedger(root / binding["path_from_source"], limit_cny="50", expected_stage_id=binding["stage_id"])
        try:
            report = run_preflight(args.output, ledger, frozen)
        finally:
            ledger.close()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "run_mode": report["run_mode"],
                    "report": str(args.output / "report.md"),
                    "error": report["error"],
                    "tariff_estimated_cost_cny": report["tariff_estimated_cost_cny"],
                    "ledger": report["ledger"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["status"] == "passed" else 1
    output = args.output or root / ".research-runs" / datetime.now().strftime("pilot-smoke-%Y%m%d-%H%M%S-%f")
    report = run_smoke(pack, output, source_snapshot(root, args.pack))
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_mode": "offline_smoke",
                "report": str(output / "report.md"),
                "paid_cost_cny": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
