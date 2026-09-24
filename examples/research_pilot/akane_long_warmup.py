"""Production-embedding restart preflight on a copy of synthetic research data."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .akane_host import AkaneHostSession, initialize_isolated_akane
from .akane_long_artifacts import PlanWorkspace
from .akane_long_observation import PublicMemoryObserver
from .akane_long_run import _restart_comparable, load_long_pack
from .akane_provenance import freeze_sources, verify_sources
from .akane_run import _contains_sensitive, _error_code
from .production_embedding import create_production_embedding, verify_production_embedding
from .runner import digest, write_json


class NoModelCalls:
    """No provider transport; the process network audit denies every connection."""

    @staticmethod
    def attach_runtime(runtime: Any) -> tuple[str, ...]:
        return ()


def run(args: Any, root: Path) -> int:
    source = args.source_trajectory.resolve()
    output = args.output.resolve()
    if not source.is_relative_to(root / ".research-runs") or output.exists():
        raise RuntimeError("fresh_output_and_synthetic_source_required")
    before = json.loads((source / "phases/before_restart/report.json").read_text(encoding="utf-8"))
    if before["status"] != "passed" or before["host_close"]["status"] != "stopped":
        raise RuntimeError("completed_synthetic_source_required")
    pack = root / "docs/research/pilot_v4"
    inputs = load_long_pack(pack)
    model_path = os.environ.get("PILOT_EMBEDDING_MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("prebound_embedding_required")
    evidence = freeze_sources(root, args.akane_root, pack, output / "source")
    for path in (source / "host").rglob("*"):
        if not path.resolve().is_relative_to(source / "host"):
            raise RuntimeError("synthetic_copy_symlink_escape")
    shutil.copytree(source / "host", output / "host")
    shutil.copyfile(source / "plans.json", output / "plans.json")
    initialize_isolated_akane(
        args.akane_root,
        output,
        read_roots=(Path(model_path),),
        raw_token_trigger=inputs["manifest"]["raw_token_trigger"],
        embedding_reindex_batch_size=inputs["manifest"]["embedding_reindex_batch_size"],
    )
    report: dict[str, Any] = {
        "format": "production_embedding_restart_preflight_v1",
        "status": "running",
        "source_run_id": before["run_id"],
        "source_evidence": evidence,
        "synthetic_data_copy": True,
        "model_calls_allowed": False,
        "error": None,
    }
    host = None
    with PublicMemoryObserver().installed() as observer:
        try:
            embedding = create_production_embedding(local_model_path=model_path, device="cuda")
            health = verify_production_embedding(embedding)
            report["embedding"] = health
            if health["status"] != "passed":
                raise RuntimeError("production_embedding_health_failed")
            case = next(c for c in inputs["scenarios"]["scenarios"] if c["scenario_id"] == before["scenario_id"])
            workspace = PlanWorkspace(output / "plans.json", case["plan_requirements"])
            host = AkaneHostSession(
                output / "host",
                policy=before["operation_policy"],
                embedding=embedding,
                transport=NoModelCalls(),
                fixture_resolver=lambda step, args: {"status": "forbidden_in_restart_preflight"},
                stable_system_blocks_provider=lambda: (inputs["scenarios"]["shared_model_instruction"],),
                user_id=f"long-{before['scenario_id']}-user",
                session_id=f"long-{before['scenario_id']}-session",
                reopen_existing=True,
                extra_handlers_factory=workspace.handlers,
            )
            started = time.monotonic()
            host.snapshot()
            observer.wait_idle(timeout=inputs["manifest"]["startup_index_wait_seconds"])
            snapshot = host.snapshot()
            report["index_ready_elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            report["host_evidence"] = host.evidence()
            report["restored_snapshot_hash"] = digest(_restart_comparable(snapshot))
            report["original_snapshot_hash"] = digest(_restart_comparable(before["final_snapshot"]))
            report["exact_saved_context_restored"] = (
                report["restored_snapshot_hash"] == report["original_snapshot_hash"]
            )
            report["exact_saved_artifacts_restored"] = digest(workspace.snapshot()) == digest(before["saved_artifacts"])
            verify_sources(evidence, root, args.akane_root, pack)
            audit = report["host_evidence"]["audit_guard"]
            passed = (
                report["exact_saved_context_restored"]
                and report["exact_saved_artifacts_restored"]
                and observer.index_runs
                and all(row["status"] == "finished" and not row["result"]["failed"] for row in observer.index_runs)
                and audit["authorized_network_events"] == 0
                and audit["blocked_network"] == 0
            )
            report["status"] = "passed" if passed else "failed"
        except BaseException as exc:
            report.update(status="failed", error=_error_code(exc))
        finally:
            if host is not None:
                report["host_close"] = host.close()
                if report["host_close"]["status"] != "stopped":
                    report["status"] = "failed"
            report["maintenance"] = observer.evidence()
            if _contains_sensitive(report, ""):
                raise RuntimeError("sensitive_preflight_evidence_detected")
            write_json(output / "report.json", report)
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--akane-root", type=Path, required=True)
    args = parser.parse_args()
    args.akane_root = args.akane_root.resolve()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run(args, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    raise SystemExit(main())
