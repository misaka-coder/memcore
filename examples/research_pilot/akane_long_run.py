"""Real, autonomous Akane trajectories with compaction and process restart."""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .akane_host import AkaneHostSession
from .akane_long_artifacts import PlanWorkspace
from .akane_long_observation import LongRunTransport, PublicMemoryObserver
from .akane_run import _assistant_messages, _contains_sensitive, _error_code, _tool_calls
from .akane_transport import AkaneTransportInterrupted
from .runner import FixtureService, digest, write_json


TOOLS = {
    "retrieve_memory",
    "read_memory_timeline",
    "browse_memory",
    "open_memory",
    "lookup_fixture",
    "save_research_plan",
    "check_research_plan",
}


def load_long_pack(pack: Path) -> dict[str, Any]:
    """The scoring answer_key is not opened or passed to a worker."""
    manifest = json.loads((pack / "run_manifest.json").read_text(encoding="utf-8"))
    scenarios = json.loads((pack / "scenarios.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "akane_actual_long_horizon_manifest_v1" or scenarios.get("synthetic") is not True:
        raise RuntimeError("invalid_long_horizon_pack")
    if len(manifest["runs"]) != 8 or len(scenarios["scenarios"]) != 2:
        raise RuntimeError("unexpected_long_horizon_denominator")
    if len({row["run_id"] for row in manifest["runs"]}) != 8:
        raise RuntimeError("duplicate_long_horizon_run")
    for case in scenarios["scenarios"]:
        if len(case["steps"]) != 24 or len({step["step_id"] for step in case["steps"]}) != 24:
            raise RuntimeError("unexpected_long_horizon_turns")
    return {"manifest": manifest, "scenarios": scenarios}


def _restart_comparable(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: snapshot[key] for key in ("projection", "memory", "metrics", "host_state")}


def _accepted(completions: list[dict[str, Any]], speech: Any) -> list[dict[str, Any]]:
    if not isinstance(speech, str) or not speech.strip():
        return []
    matched = []
    for row in completions:
        if not row["completed"] or not row["final_entry"] or row["final_entry"]["kind"] != "message.assistant":
            continue
        exact_speech = row["submitted_speech"] == speech
        projection = row.get("final_projection") or {}
        try:
            exact_projection = json.loads(projection.get("content", ""))["speech"] == speech
        except (TypeError, ValueError, KeyError):
            exact_projection = False
        if exact_speech or exact_projection:
            matched.append(
                {
                    **copy.deepcopy(row),
                    "speech_matches_commit": exact_speech,
                    "speech_matches_projection": exact_projection,
                }
            )
    return matched


def _readback_evidence(system: Any, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read recorded IDs through MemorySystem, not the rendered host adapter."""
    selected = [row for row in entries if row["kind"] == "tool.lookup_fixture.result"]
    if not selected:
        return []
    opened = system.open_memory(memory_ids=[row["source_id"] for row in selected], view="content")
    returned = {
        item["memory_id"]: digest(item["result"]["content"])
        for item in opened.get("result", {}).get("items", [])
        if item.get("status") == "ok" and isinstance(item.get("result", {}).get("content"), str)
    }
    return [
        {
            "source_id": row["source_id"],
            "initial_public_content_hash": row["initial_public_content_hash"],
            "reopened_public_content_hash": returned.get(row["source_id"]),
            "exact_body_preserved": returned.get(row["source_id"]) == row["initial_public_content_hash"],
            "purpose": "read_only_diagnostic_not_delivered_to_model",
        }
        for row in selected
    ]


def run_long_phase(
    inputs: dict[str, Any],
    plan: dict[str, Any],
    phase: str,
    trajectory_root: Path,
    *,
    transport: LongRunTransport,
    observer: PublicMemoryObserver,
    embedding: Any,
    embedding_evidence: dict[str, Any],
    source_check: Callable[[], None],
    source_evidence: dict[str, Any],
    secret: str,
    offline: bool = False,
) -> dict[str, Any]:
    manifest = inputs["manifest"]
    if phase not in manifest["phases"]:
        raise RuntimeError("invalid_long_horizon_phase")
    output = trajectory_root / "phases" / phase
    output.mkdir(parents=True, exist_ok=False)
    case = next(case for case in inputs["scenarios"]["scenarios"] if case["scenario_id"] == plan["scenario_id"])
    fixture = FixtureService(case)
    workspace = PlanWorkspace(trajectory_root / "plans.json", case["plan_requirements"])
    report: dict[str, Any] = {
        "format": "akane_actual_long_horizon_phase_v1",
        "status": "running",
        "test_mode": "actual_engine_mock_http_TEST_ONLY_ledger" if offline else "actual_engine_real_model",
        "run_id": plan["run_id"],
        "scenario_id": plan["scenario_id"],
        "condition": plan["condition"],
        "phase": phase,
        "process_identity": uuid.uuid4().hex,
        "process_id": os.getpid(),
        "operation_policy": manifest["condition_definitions"][plan["condition"]],
        "raw_token_trigger": manifest["raw_token_trigger"],
        "input_token_budget": transport.input_token_budget,
        "embedding_reindex_batch_size": manifest["embedding_reindex_batch_size"],
        "source_evidence": source_evidence,
        "embedding_evidence": embedding_evidence,
        "budget_before": transport.ledger.snapshot(),
        "steps": [],
        "error": None,
    }
    first, last = (manifest["phases"][phase][name] for name in ("first_step", "last_step"))
    steps = case["steps"][first - 1 : last]
    host = None
    checkpoint = trajectory_root / "restart_checkpoint.json"
    try:
        source_check()
        previous = json.loads(checkpoint.read_text(encoding="utf-8")) if phase == "after_restart" else None
        if previous is not None and (
            previous["run_id"] != plan["run_id"]
            or previous["host_close"]["status"] != "stopped"
            or previous["source_fingerprint"] != source_evidence["source_fingerprint"]
            or previous["process_id"] == report["process_id"]
        ):
            raise AkaneTransportInterrupted("restart_checkpoint_or_process_mismatch")
        host = AkaneHostSession(
            trajectory_root / "host",
            policy=report["operation_policy"],
            embedding=embedding,
            transport=transport,
            fixture_resolver=lambda step, args: fixture.lookup(step["step_id"], dict(args)),
            stable_system_blocks_provider=lambda: (inputs["scenarios"]["shared_model_instruction"],),
            user_id=f"long-{plan['scenario_id']}-user",
            session_id=f"long-{plan['scenario_id']}-session",
            reopen_existing=phase == "after_restart",
            extra_handlers_factory=workspace.handlers,
        )
        # Creating the public context starts Akane's normal index warmup.
        startup_started = time.monotonic()
        host.snapshot()
        observer.wait_idle(timeout=manifest["startup_index_wait_seconds"])
        report["startup_index_ready_elapsed_ms"] = round((time.monotonic() - startup_started) * 1000, 3)
        startup = host.snapshot()
        report["startup_snapshot"] = startup
        report["host_evidence"] = host.evidence()
        if previous is not None:
            context_equal = digest(_restart_comparable(startup)) == previous["snapshot_hash"]
            artifacts_equal = digest(workspace.snapshot()) == previous["artifact_hash"]
            report["restart_verification"] = {
                "new_process": True,
                "provider_projection_and_metadata_exactly_equal": context_equal,
                "artifact_state_exactly_equal": artifacts_equal,
                "previous_process_identity": previous["process_identity"],
                "restored_snapshot_hash": digest(_restart_comparable(startup)),
                "previous_snapshot_hash": previous["snapshot_hash"],
                "index_warmup": copy.deepcopy(observer.index_runs),
            }
            if not context_equal or not artifacts_equal:
                raise AkaneTransportInterrupted("restart_saved_state_changed")

        def request_observer(row: dict[str, Any]) -> None:
            source_check()
            evidence = host.snapshot() if row["bundle_role"] == "chat" else {"kind": "infrastructure_request"}
            if _contains_sensitive({"request": row["request"], "evidence": evidence}, secret):
                raise AkaneTransportInterrupted("sensitive_request_evidence_detected")
            names = {tool.get("function", {}).get("name") for tool in row["request"].get("tools", [])}
            if names - TOOLS:
                raise AkaneTransportInterrupted("unexpected_long_horizon_tool_exposure")
            row["memory_evidence"] = evidence

        transport.request_observer = request_observer
        for step in steps:
            source_check()
            transport.current_step = step["step_id"]
            start_request, start_completion, start_entry = (
                len(transport.requests),
                len(observer.completions),
                len(observer.entries),
            )
            start_compaction, start_artifact = len(observer.compactions), len(workspace.events)
            row: dict[str, Any] = {
                "step_id": step["step_id"],
                "stage": step["stage"],
                "status": "running",
                "error": None,
            }
            report["steps"].append(row)
            started = time.monotonic()
            try:
                with transport.scope(plan["run_id"], step["step_id"]):
                    result = host.process_turn({**step, "timestamp": int(time.time())})
                row["host_return_elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
                speech = result["output"].get("speech") if isinstance(result["output"], dict) else None
                commits = _accepted(observer.completions[start_completion:], speech)
                row.update(result)
                row.update(speech=speech, acceptance=commits, status="passed" if len(commits) == 1 else "failed")
                if row["status"] == "failed":
                    row["error"] = "nonempty_speech_not_bound_to_one_public_completion"
                # Waiting belongs to the script's pacing, after the real host
                # returned speech. It does not replace background scheduling.
                row["maintenance_idle"] = observer.wait_idle()
                row["after_maintenance"] = host.snapshot()
            except BaseException as exc:
                row.update(status="interrupted", error=_error_code(exc))
            captured = transport.requests[start_request:]
            row["request_ids"] = [item["request_id"] for item in captured]
            row["request_count_by_role"] = {
                role: sum(item["bundle_role"] == role for item in captured)
                for role in sorted({item["bundle_role"] for item in captured})
            }
            row["assistant_messages"] = _assistant_messages(
                [item for item in captured if item["bundle_role"] == "chat"]
            )
            recorded = observer.entries[start_entry:]
            row["tool_calls"] = _tool_calls(
                [item for item in captured if item["bundle_role"] == "chat"],
                {"memory": recorded, "before": {"memory": []}},
            )
            row["public_appended_entries"] = copy.deepcopy(recorded)
            row["compaction_jobs"] = copy.deepcopy(observer.compactions[start_compaction:])
            row["artifact_events"] = copy.deepcopy(workspace.events[start_artifact:])
            row["step_with_maintenance_elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            if _contains_sensitive(row, secret):
                row.clear()
                row.update(step_id=step["step_id"], status="interrupted", error="sensitive_step_evidence_detected")
            report["budget_latest"] = transport.ledger.snapshot()
            write_json(output / "report.json", report)
            if row["status"] == "interrupted" or any(item["status"] == "interrupted" for item in captured):
                raise AkaneTransportInterrupted(row.get("error") or "captured_transport_interruption")
        observer.wait_idle()
        report["final_snapshot"] = host.snapshot()
        report["saved_artifacts"] = workspace.snapshot()
        early_entries = [entry for row in report["steps"][:3] for entry in row.get("public_appended_entries", [])]
        if previous is not None:
            early_entries = previous["early_fixture_entries"]
        report["early_source_readback"] = _readback_evidence(observer.system, early_entries)
        report["maintenance_evidence"] = observer.evidence()
        report["final_host_evidence"] = host.evidence()
        report["status"] = (
            "passed" if all(step["status"] == "passed" for step in report["steps"]) else "completed_with_turn_failures"
        )
    except BaseException as exc:
        report.update(status="interrupted", error=_error_code(exc))
    finally:
        if host is not None:
            report["host_close"] = host.close()
            if report["host_close"].get("status") != "stopped":
                report["status"] = "interrupted"
                report["shutdown_error"] = "long_phase_host_close_failed"
                if report["error"] is None:
                    report["error"] = report["shutdown_error"]
        report["maintenance_evidence"] = observer.evidence()
        transport.request_observer = None
        report["budget_after"] = transport.ledger.snapshot()
        report["request_index"] = [
            {
                key: item.get(key)
                for key in (
                    "request_id",
                    "capture_file",
                    "scope",
                    "step_id",
                    "bundle_role",
                    "run_mode",
                    "status",
                    "dispatched",
                )
            }
            for item in transport.requests
        ]
        report["actual_model_http_requests"] = (
            0 if offline else sum(item["dispatched"] is True for item in transport.requests)
        )
        report["mock_http_requests"] = sum(item["dispatched"] is True for item in transport.requests) if offline else 0
        if _contains_sensitive(report, secret):
            raise RuntimeError("sensitive_phase_evidence_detected")
        write_json(output / "report.json", report)
    if phase == "before_restart" and report["status"] in {"passed", "completed_with_turn_failures"}:
        write_json(
            checkpoint,
            {
                "run_id": plan["run_id"],
                "process_identity": report["process_identity"],
                "process_id": report["process_id"],
                "source_fingerprint": source_evidence["source_fingerprint"],
                "snapshot_hash": digest(_restart_comparable(report["final_snapshot"])),
                "artifact_hash": digest(report["saved_artifacts"]),
                "early_fixture_entries": [
                    entry
                    for row in report["steps"][:3]
                    for entry in row.get("public_appended_entries", [])
                    if entry["kind"] == "tool.lookup_fixture.result"
                ],
                "host_close": report["host_close"],
            },
        )
    return report
