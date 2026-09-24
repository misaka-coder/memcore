"""Coordinate fresh-process long-horizon experiments, preserving every attempt."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .akane_long_run import load_long_pack, run_long_phase
from .akane_provenance import freeze_sources, verify_sources
from .akane_run import _error_code
from .budget import BudgetLedger
from .runner import digest, write_json


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _worker(args: Any, root: Path, pack: Path) -> int:
    from .akane_host import allow_model_network, initialize_isolated_akane
    from .akane_long_observation import LongRunTransport, PublicMemoryObserver
    from .production_embedding import create_production_embedding, verify_production_embedding

    inputs = load_long_pack(pack)
    plan = next(row for row in inputs["manifest"]["runs"] if row["run_id"] == args.run_id)
    source = _read(args.output / "source/source_evidence.json")
    verify_sources(source, root, args.akane_root, pack)
    trajectory = args.output / "trajectories" / plan["run_id"]
    ledger_path = (
        args.output / "TEST-ONLY-budget.sqlite3" if args.offline_test else root / ".research-runs/stage1-budget.sqlite3"
    )
    budget_metadata = _read(args.output / "budget_binding.json")
    if args.offline_test:
        secret, model_path = "TEST-ONLY-not-a-real-key", ""
    else:
        secret, model_path = os.environ.get("DEEPSEEK_API_KEY", ""), os.environ.get("PILOT_EMBEDDING_MODEL_PATH", "")
        if not secret or not model_path:
            raise RuntimeError("prebound_credential_and_embedding_required")
    # Open the existing ledger before installing the host filesystem audit.
    with BudgetLedger(ledger_path, expected_stage_id=budget_metadata["stage_id"]) as ledger:
        initialize_isolated_akane(
            args.akane_root,
            trajectory,
            read_roots=() if args.offline_test else (Path(model_path),),
            write_paths=(ledger_path,),
            raw_token_trigger=inputs["manifest"]["raw_token_trigger"],
            embedding_reindex_batch_size=inputs["manifest"]["embedding_reindex_batch_size"],
        )
        if args.offline_test:
            from .akane_long_mock import TestOnlyEmbedding, TestOnlyLongTransport

            embedding, health, transport_class = (
                TestOnlyEmbedding(),
                {"status": "passed", "mode": "TEST_ONLY_hashed_offline_embedding"},
                TestOnlyLongTransport,
            )
        else:
            embedding = create_production_embedding(local_model_path=model_path, device="cuda")
            health = verify_production_embedding(embedding)
            if health["status"] != "passed":
                raise RuntimeError("production_embedding_health_failed")
            baseline = _read(root / "docs/research/pilot_v3/actual_results.json")["embedding"]
            if any(health[key] != baseline[key] for key in ("model_id", "dimension")) or any(
                abs(health[key] - baseline[key]) > 0.0001
                for key in ("similar_score", "unrelated_score", "repeat_score")
            ):
                raise RuntimeError("long_experiment_embedding_baseline_changed")
            transport_class = LongRunTransport
        with PublicMemoryObserver().installed() as observer:
            ledger.assert_ready()
            transport = transport_class(
                ledger,
                run_id=plan["run_id"],
                capture_dir=trajectory / "requests" / args.phase,
                api_key_provider=lambda: secret,
                network_context_factory=None if args.offline_test else allow_model_network,
            )
            if args.offline_test:
                case = next(
                    case for case in inputs["scenarios"]["scenarios"] if case["scenario_id"] == plan["scenario_id"]
                )
                transport.bind_case(case, phase=args.phase)
            result = run_long_phase(
                inputs,
                plan,
                args.phase,
                trajectory,
                transport=transport,
                observer=observer,
                embedding=embedding,
                embedding_evidence=health,
                source_check=lambda: verify_sources(source, root, args.akane_root, pack),
                source_evidence=source,
                secret=secret,
                offline=args.offline_test,
            )
            return 0 if result["status"] in {"passed", "completed_with_turn_failures"} else 1


def _trajectory_verification(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    first_sources = {
        completion["final_entry"]["source_id"]
        for row in before["steps"][:3]
        for completion in row.get("acceptance", [])
    }
    first_sources.update(
        entry["source_id"]
        for row in before["steps"][:3]
        for entry in row.get("public_appended_entries", [])
        if entry["turn_role"] == "observation"
    )
    end_visible = {row["source_id"] for row in before["final_snapshot"]["memory"]}
    jobs = [*before["maintenance_evidence"]["compactions"], *after["maintenance_evidence"]["compactions"]]
    completed = [row["stats"] for row in jobs if row.get("stats", {}).get("status") == "compacted"]
    generations = sorted({row["compaction_generation"] for row in completed})
    first_fixture_calls = [call for call in before["steps"][1]["tool_calls"] if call["name"] == "lookup_fixture"]
    readbacks = []
    for original in first_fixture_calls:
        original_content = json.loads(original["provider_result_body"]).get("content", "")
        matches = [
            call
            for row in after["steps"]
            for call in row["tool_calls"]
            if call["name"] == "open_memory"
            and call["delivered_to_model"]
            and original["source_id"] in [call["arguments"].get("memory_id"), *call["arguments"].get("memory_ids", [])]
            and all(line in (call.get("result") or "") for line in original_content.splitlines() if line)
        ]
        readbacks.append({"source_id": original["source_id"], "original_content_lines_delivered": bool(matches)})
    events = [event for row in [*before["steps"], *after["steps"]] for event in row["artifact_events"]]
    checks = [event for event in events if event["operation"] == "check"]
    v2_statuses = [event["result"]["status"] for event in checks if event["revision"] == "draft_v2"]
    saved_v1 = next(
        (event["plan"] for event in events if event["operation"] == "save" and event["revision"] == "draft_v1"), None
    )
    source_readback = [*before["early_source_readback"], *after["early_source_readback"]]
    audits = [phase["final_host_evidence"]["audit_guard"] for phase in (before, after)]
    return {
        "actual_raw_compaction_generations": generations,
        "at_least_two_actual_compactions": len(generations) >= 2,
        "early_sources_checked": sorted(first_sources),
        "early_sources_no_longer_in_raw_visible_context": bool(first_sources) and not (first_sources & end_visible),
        "restart": after["restart_verification"],
        "all_turns_accepted_and_committed": all(
            row["status"] == "passed" for row in [*before["steps"], *after["steps"]]
        ),
        "background_failures": [
            row
            for row in jobs
            if row.get("status") == "failed" or row.get("stats", {}).get("status") in {"failed", "cancelled"}
        ],
        "saved_artifact_hash": digest(after["saved_artifacts"]),
        "all_early_fixture_bodies_preserved_across_compaction_and_restart": bool(source_readback)
        and all(row["exact_body_preserved"] for row in source_readback),
        "model_original_fixture_readbacks": readbacks,
        "all_original_fixture_bodies_delivered_to_model_after_restart": bool(readbacks)
        and all(row["original_content_lines_delivered"] for row in readbacks),
        "all_three_artifact_revisions_checked_passed": all(
            any(event["revision"] == revision and event["result"]["status"] == "passed" for event in checks)
            for revision in ("draft_v1", "draft_v2", "final")
        ),
        "initial_artifact_revision_preserved": saved_v1 is not None
        and after["saved_artifacts"]["plans"].get("draft_v1") == saved_v1,
        "offline_wrong_artifact_rejected_then_corrected": "failed" in v2_statuses
        and "passed" in v2_statuses[v2_statuses.index("failed") + 1 :],
        "offline_malformed_output_repaired": before["steps"][0]["request_count_by_role"].get("chat") == 2
        and before["steps"][0]["status"] == "passed",
        "network_audit": audits,
        "offline_no_real_network_events": all(
            audit["authorized_network_events"] == 0 and audit["blocked_network"] == 0 for audit in audits
        ),
    }


def _controller(args: Any, root: Path, pack: Path) -> int:
    if args.output.exists():
        raise RuntimeError("fresh_long_experiment_output_required")
    inputs = load_long_pack(pack)
    source = freeze_sources(root, args.akane_root, pack, args.output / "source")
    if not args.offline_test:
        if args.offline_gate is None:
            raise RuntimeError("long_experiment_offline_gate_required")
        gate = _read(args.offline_gate)
        if gate.get("status") != "passed" or not gate.get("actual_engine_compaction_and_process_restart"):
            raise RuntimeError("long_experiment_offline_gate_not_passed")
        if gate["source_evidence"]["file_hashes"] != source["file_hashes"]:
            raise RuntimeError("long_experiment_gate_source_changed")
    ledger_path = (
        args.output / "TEST-ONLY-budget.sqlite3" if args.offline_test else root / ".research-runs/stage1-budget.sqlite3"
    )
    with BudgetLedger(
        ledger_path, expected_stage_id=None if args.offline_test else inputs["manifest"]["stage_id"]
    ) as ledger:
        ledger.assert_ready()
        budget_before = ledger.snapshot()
    write_json(
        args.output / "budget_binding.json", {"stage_id": budget_before["stage_id"], "test_only": args.offline_test}
    )
    selected = inputs["manifest"]["runs"][:2] if args.offline_test else inputs["manifest"]["runs"]
    report: dict[str, Any] = {
        "format": "akane_actual_long_horizon_run_v1",
        "status": "running",
        "test_mode": "actual_engine_mock_http_TEST_ONLY_ledger" if args.offline_test else "actual_engine_real_model",
        "source_evidence": source,
        "planned_trajectories": len(selected),
        "planned_user_turns": len(selected) * 24,
        "budget_before": budget_before,
        "runs": [],
    }
    halted = False
    for plan in selected:
        row: dict[str, Any] = {**plan, "status": "not_run", "phases": {}, "error": None}
        report["runs"].append(row)
        if halted:
            row["error"] = "earlier_infrastructure_interruption"
            write_json(args.output / "report.json", report)
            continue
        for phase in inputs["manifest"]["phases"]:
            verify_sources(source, root, args.akane_root, pack)
            command = [
                sys.executable,
                "-B",
                "-m",
                "examples.research_pilot.akane_long",
                "--worker",
                "--akane-root",
                str(args.akane_root),
                "--output",
                str(args.output),
                "--run-id",
                plan["run_id"],
                "--phase",
                phase,
            ]
            if args.offline_test:
                command.append("--offline-test")
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=os.environ.copy(),
                    capture_output=True,
                    timeout=900,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
                phase_file = args.output / "trajectories" / plan["run_id"] / "phases" / phase / "report.json"
                phase_result = _read(phase_file) if phase_file.is_file() else None
                row["phases"][phase] = {
                    "status": phase_result["status"] if phase_result else "startup_failed",
                    "process_exit_code": completed.returncode,
                    "report": phase_file.relative_to(args.output).as_posix(),
                    "error": phase_result.get("error") if phase_result else "phase_report_missing",
                }
                if completed.returncode or phase_result is None or phase_result["status"] == "interrupted":
                    halted = True
                    row.update(status="interrupted", error=row["phases"][phase]["error"])
                    break
            except BaseException as exc:
                halted = True
                row.update(status="interrupted", error=_error_code(exc))
                break
            write_json(args.output / "report.json", report)
        if not halted:
            before, after = (
                _read(args.output / row["phases"][phase]["report"]) for phase in inputs["manifest"]["phases"]
            )
            row["verification"] = _trajectory_verification(before, after)
            row["status"] = "completed"
        write_json(args.output / "report.json", report)
    with BudgetLedger(ledger_path, expected_stage_id=budget_before["stage_id"]) as ledger:
        report["budget_after"] = ledger.snapshot()
    report["status"] = "completed" if not halted else "incomplete"
    write_json(args.output / "report.json", report)
    if args.offline_test:
        passed = not halted and all(
            row["verification"]["at_least_two_actual_compactions"]
            and row["verification"]["early_sources_no_longer_in_raw_visible_context"]
            and row["verification"]["all_turns_accepted_and_committed"]
            and not row["verification"]["background_failures"]
            and all(
                row["verification"][name]
                for name in (
                    "all_early_fixture_bodies_preserved_across_compaction_and_restart",
                    "all_original_fixture_bodies_delivered_to_model_after_restart",
                    "all_three_artifact_revisions_checked_passed",
                    "initial_artifact_revision_preserved",
                    "offline_wrong_artifact_rejected_then_corrected",
                    "offline_malformed_output_repaired",
                    "offline_no_real_network_events",
                )
            )
            for row in report["runs"]
        )
        write_json(
            args.output / "acceptance.json",
            {
                "status": "passed" if passed else "failed",
                "actual_engine_compaction_and_process_restart": passed,
                "test_only_mock_http": True,
                "real_network_calls": sum(
                    audit["authorized_network_events"]
                    for row in report["runs"]
                    for audit in row.get("verification", {}).get("network_audit", [])
                ),
                "source_evidence": source,
                "runs": report["runs"],
            },
        )
        return 0 if passed else 1
    return 0 if not halted else 1


def main() -> int:
    import memcore

    root = Path(__file__).resolve().parents[2]
    if not Path(memcore.__file__).resolve().is_relative_to(root / "memcore"):
        raise RuntimeError("unexpected_memcore_runtime_source")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--akane-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline-gate", type=Path)
    parser.add_argument("--offline-test", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--phase")
    args = parser.parse_args()
    args.akane_root, args.output = args.akane_root.resolve(), args.output.resolve()
    if args.offline_gate:
        args.offline_gate = args.offline_gate.resolve()
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            return (
                _worker(args, root, root / "docs/research/pilot_v4")
                if args.worker
                else _controller(args, root, root / "docs/research/pilot_v4")
            )
    except BaseException as exc:
        failure = {"status": "startup_failed", "error": _error_code(exc)}
        target = args.output / "trajectories" / args.run_id / "phases" / args.phase if args.worker else args.output
        if target.exists() or args.output.exists():
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "startup_failure.json", failure)
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
