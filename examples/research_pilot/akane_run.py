"""Four-run retest through the real Akane Engine and budgeted SDK transport.

Scenario fixtures are accessible only through their current-step resolver. The
evaluator answer key is never loaded by this runner. Every probe reconstructs
its own setup with original provider responses; replay cannot fall back to HTTP.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .akane_provenance import freeze_sources, load_pack, verify_sources
from .akane_replay import ReplayMapper, ReplayMismatch
from .runner import FixtureService, digest, write_json


_LOCAL_PATH = re.compile(r"(?i)(?:(?<![a-z0-9_])[a-z]:[\\/]|file://|\\\\[a-z0-9_.-]{2,}\\[a-z0-9$_.-]{2,})")
_MEMORY_TOOL_NAMES = {"retrieve_memory", "read_memory_timeline", "browse_memory", "open_memory"}


class _SetupIncomplete(RuntimeError):
    def __init__(self, trace: dict[str, Any]):
        self.code = trace["error"] or "setup_not_complete"
        self.status = trace["status"]
        self.stop_all = trace.get("stop_all", False)
        super().__init__(self.code)


def _contains_sensitive(value: Any, secret: str) -> bool:
    # Decode structured strings before inspecting path syntax. Timeline bodies
    # can contain JSON inside rendered text, including several escaped layers.
    if isinstance(value, str):
        if secret and secret in value:
            return True
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            decoded = None
        if isinstance(decoded, (dict, list, str)):
            return _contains_sensitive(decoded, secret)
        if _LOCAL_PATH.search(value):
            return True
        decoder = json.JSONDecoder()
        position = 0
        while (start := value.find('"', position)) >= 0:
            try:
                quoted, position = decoder.raw_decode(value, start)
            except (ValueError, RecursionError):
                position = start + 1
                continue
            if isinstance(quoted, str) and _contains_sensitive(quoted, secret):
                return True
        return False
    if isinstance(value, dict):
        return any(_contains_sensitive(item, secret) for pair in value.items() for item in pair)
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive(item, secret) for item in value)
    return False


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_:-]{1,160}", code):
        return code
    if isinstance(exc, ReplayMismatch):
        return str(exc)
    return type(exc).__name__


def _assistant_messages(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in requests:
        response = row.get("response")
        if not isinstance(response, dict):
            continue
        for choice in response.get("choices", []):
            message = choice.get("message")
            if isinstance(message, dict):
                result.append(copy.deepcopy(message))
    return result


def _tool_calls(requests: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for request_index, row in enumerate(requests):
        for message in _assistant_messages([row]):
            for batch_index, tool in enumerate(message.get("tool_calls") or []):
                function = tool.get("function", {})
                raw_args = function.get("arguments", "")
                try:
                    arguments = json.loads(raw_args)
                except (TypeError, ValueError):
                    arguments = raw_args
                calls.append(
                    {
                        "call_id": tool["id"],
                        "name": function.get("name"),
                        "arguments": arguments,
                        "request_index": request_index,
                        "batch_index": batch_index,
                        "result": None,
                        "source_id": None,
                        "delivered_to_model": False,
                    }
                )
    previous_sources = {record["source_id"] for record in snapshot.get("before", {}).get("memory", [])}
    for call in calls:
        # Retain separate native invocations even if a provider reuses an ID.
        # Such an ID cannot be assigned an unambiguous raw source by guessing.
        duplicate = sum(item["call_id"] == call["call_id"] for item in calls) > 1
        call["duplicate_native_call_id"] = duplicate
        for row in requests[call["request_index"] + 1 :]:
            matching = [
                message
                for message in (row.get("request") or {}).get("messages", [])
                if message.get("role") == "tool" and message.get("tool_call_id") == call["call_id"]
            ]
            if matching:
                body = matching[-1].get("content")
                try:
                    result = json.loads(body)
                except (TypeError, ValueError):
                    result = body
                call.update(
                    result=result,
                    delivered_to_model=True,
                    provider_result_body=body,
                    result_binding_ambiguous=duplicate,
                )
                break
        candidates = [
            record["source_id"]
            for record in snapshot.get("memory", [])
            if record.get("correlation_id") == call["call_id"]
            and record.get("turn_role") == "observation"
            and record["source_id"] not in previous_sources
        ]
        call["source_id_candidates"] = candidates
        call["source_binding_ambiguous"] = duplicate or len(candidates) != 1
        if not call["source_binding_ambiguous"]:
            call["source_id"] = candidates[0]
    return calls


def _snapshot_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {key: turn[key] for key in ("projection", "memory", "host_state", "metrics")}


def _run_step(
    session: Any,
    transport: Any,
    step: dict[str, Any],
    *,
    scope: str,
    source_check: Callable[[], None],
    secret: str,
    replay_records: list[dict[str, Any]] | None = None,
    mapper: ReplayMapper | None = None,
    preserve_input_timestamp: bool = False,
    request_check: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from .akane_transport import AkaneTransportInterrupted

    replay = replay_records is not None
    actual_step = {key: step[key] for key in ("step_id", "user_text", "timestamp")}
    if not replay and not preserve_input_timestamp:
        actual_step["timestamp"] = int(time.time())
    trace: dict[str, Any] = {
        "step_id": step["step_id"],
        "status": "running",
        "error": None,
        "speech": None,
        "scenario_timestamp": step["timestamp"],
        "input_timestamp": actual_step["timestamp"],
        "run_mode": "replay" if replay else "actual_akane",
    }
    start = len(transport.requests)
    replay_index = 0

    def observer(row: dict[str, Any]) -> None:
        nonlocal replay_index
        snapshot = session.snapshot()
        if _contains_sensitive({"request": row["request"], "snapshot": snapshot}, secret):
            raise AkaneTransportInterrupted("sensitive_request_evidence_detected")
        tools = row["request"].get("tools", [])
        names = {tool.get("function", {}).get("name") for tool in tools}
        if names - (_MEMORY_TOOL_NAMES | {"lookup_fixture"}):
            raise AkaneTransportInterrupted("unexpected_model_tool_exposure")
        row["memory_evidence"] = snapshot
        if request_check is not None:
            request_check(row)
        if replay:
            if mapper is None or replay_records is None or replay_index >= len(replay_records):
                raise ReplayMismatch("replay_observer_record_missing")
            try:
                mapper.bind(replay_records[replay_index]["memory_evidence"], snapshot)
            except ReplayMismatch as exc:
                raise AkaneTransportInterrupted(str(exc)) from None
            replay_index += 1

    previous_observer = transport.request_observer
    transport.request_observer = observer
    started = time.monotonic()
    try:
        source_check()
        with transport.scope(scope, step["step_id"]):
            if replay:
                assert replay_records is not None and mapper is not None
                # No provider body or response ID substitution. Only the
                # mapper's source-bound compact-card time line can differ.
                with transport.replay(replay_records, request_matcher=mapper.compare_request):
                    result = session.process_turn(actual_step)
            else:
                result = session.process_turn(actual_step)
        trace.update(result)
        output = result["output"]
        speech = output.get("speech") if isinstance(output, dict) else None
        trace["speech"] = speech if isinstance(speech, str) else None
        new_sources = {item["source_id"] for item in result["memory"]} - {
            item["source_id"] for item in result["before"]["memory"]
        }
        new_turn_ids = {
            item["turn_id"]
            for item in result["memory"]
            if item["source_id"] in new_sources and item.get("kind") == "message.user"
        }
        finals = [
            item
            for item in result["memory"]
            if item["source_id"] in new_sources
            and item.get("kind") == "message.assistant"
            and item.get("turn_role") == "final"
            and item.get("turn_id") in new_turn_ids
        ]
        trace["assistant_recorded"] = bool(finals)
        trace["status"] = "passed" if isinstance(speech, str) and speech.strip() and finals else "failed"
        if trace["status"] == "failed":
            trace["error"] = "host_did_not_accept_and_record_nonempty_answer"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            trace.update(status="interrupted", error="interrupted", stop_all=True)
        elif isinstance(exc, AkaneTransportInterrupted):
            trace.update(status="interrupted", error=exc.code, stop_all=exc.code == "interrupted")
        else:
            trace.update(status="failed", error=_error_code(exc))
    finally:
        transport.request_observer = previous_observer
        requests = transport.requests[start:]
        trace["request_ids"] = [row["request_id"] for row in requests]
        trace["request_count"] = len(requests)
        trace["paid_requests"] = sum(row.get("dispatched") is True for row in requests)
        if replay:
            trace["original_response_bytes_preserved"] = all(
                row.get("replay_id_map") == {}
                and row.get("response_body_sha256") == row.get("source_response_body_sha256")
                for row in requests
                if row.get("status") == "replayed"
            )
        trace["assistant_messages"] = _assistant_messages(requests)
        trace["tool_calls"] = _tool_calls(requests, trace)
        trace["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
    return trace


def run_actual_host(
    pack: dict[str, Any],
    output: Path,
    *,
    embedding: Any,
    transport: Any,
    source_check: Callable[[], None],
    source_evidence: dict[str, Any],
    embedding_evidence: dict[str, Any],
    gate_evidence: dict[str, Any],
    secret: str,
) -> dict[str, Any]:
    from .akane_host import AkaneHostSession

    output.mkdir(parents=True, exist_ok=False)
    specification = pack["scenarios"]
    scenarios = {case["scenario_id"]: case for case in specification["scenarios"]}
    report: dict[str, Any] = {
        "format": "akane_actual_host_retest_v2",
        "status": "running",
        "source_evidence": source_evidence,
        "host_equivalence_gate": gate_evidence,
        "embedding_evidence": embedding_evidence,
        "runs": [],
        "budget_before": transport.ledger.snapshot(),
        "scores": None,
        "baseline_equivalence": "source_mapped_with_explicit_host_timestamp_differences",
        "run_mode": "actual_akane_http" if getattr(transport, "is_live", True) else "offline_control_flow_test",
    }
    write_json(output / "report.json", report)
    stop_all = False
    for plan in pack["manifest"]["runs"]:
        if stop_all:
            report["runs"].append(
                {
                    **plan,
                    "status": "not_run",
                    "error": "experiment_interrupted",
                    "setup": [],
                    "branches": [
                        {
                            "step_id": probe["step_id"],
                            "status": "not_run",
                            "speech": None,
                            "error": "experiment_interrupted",
                        }
                        for probe in scenarios[plan["scenario_id"]]["steps"]
                        if probe["stage"] == "probe"
                    ],
                }
            )
            continue
        case = scenarios[plan["scenario_id"]]
        fixture_service = FixtureService(case)
        run_dir = output / plan["run_id"]
        run_dir.mkdir()
        row: dict[str, Any] = {**plan, "status": "running", "setup": [], "branches": [], "error": None}
        report["runs"].append(row)
        sessions = []

        def make_session(label: str) -> Any:
            session = AkaneHostSession(
                run_dir / label,
                policy=pack["manifest"]["condition_definitions"][plan["condition"]]["operation_projection_policy"],
                embedding=embedding,
                transport=transport,
                fixture_resolver=lambda current, arguments: fixture_service.lookup(current["step_id"], dict(arguments)),
                stable_system_blocks_provider=lambda: (specification["shared_model_instruction"],),
            )
            sessions.append(session)
            return session

        def save() -> None:
            write_json(run_dir / "trace.json", row)
            report["budget_latest"] = transport.ledger.snapshot()
            write_json(output / "report.json", report)

        try:
            base = make_session("setup")
            for step in case["steps"]:
                if step["stage"] == "probe":
                    continue
                turn = _run_step(base, transport, step, scope=plan["run_id"], source_check=source_check, secret=secret)
                row["setup"].append(turn)
                save()
                if turn["status"] != "passed":
                    raise _SetupIncomplete(turn)
            baseline = base.snapshot()
            row["host_evidence"] = base.evidence()
            row["setup_snapshot"] = baseline
            row["setup_projection_hash"] = digest(baseline["projection"])
            row["setup_close"] = base.close()
            sessions.remove(base)
            if not isinstance(row["setup_close"], dict) or row["setup_close"].get("status") != "stopped":
                raise RuntimeError("setup_host_close_not_stopped")
            paid_by_id = {
                request["request_id"]: request for request in transport.requests if request["run_mode"] != "replay"
            }
            steps_by_id = {step["step_id"]: step for step in case["steps"]}
            for probe in (step for step in case["steps"] if step["stage"] == "probe"):
                branch_result: dict[str, Any] = {
                    "step_id": probe["step_id"],
                    "status": "not_run",
                    "speech": None,
                    "error": None,
                    "replay": [],
                }
                row["branches"].append(branch_result)
                if stop_all:
                    branch_result["error"] = "experiment_interrupted"
                    continue
                branch = None
                try:
                    branch = make_session(probe["step_id"])
                    mapper = ReplayMapper()
                    for setup_turn in row["setup"]:
                        replay_step = dict(steps_by_id[setup_turn["step_id"]])
                        replay_step["timestamp"] = setup_turn["input_timestamp"]
                        restored = _run_step(
                            branch,
                            transport,
                            replay_step,
                            scope=f"{plan['run_id']}:{probe['step_id']}",
                            source_check=source_check,
                            secret=secret,
                            replay_records=[paid_by_id[key] for key in setup_turn["request_ids"]],
                            mapper=mapper,
                        )
                        branch_result["replay"].append(restored)
                        if restored["status"] != "passed":
                            raise ReplayMismatch("setup_replay_not_complete")
                        mapper.compare_snapshot(_snapshot_from_turn(setup_turn), _snapshot_from_turn(restored))
                    branch_result["baseline_comparison"] = mapper.compare_snapshot(baseline, branch.snapshot())
                    probe_turn = _run_step(
                        branch,
                        transport,
                        probe,
                        scope=f"{plan['run_id']}:{probe['step_id']}",
                        source_check=source_check,
                        secret=secret,
                    )
                    branch_result.update(probe_turn)
                    stop_all = bool(probe_turn.get("stop_all"))
                    branch_result["host_evidence"] = branch.evidence()
                except BaseException as exc:
                    branch_result.update(status="not_run", error=_error_code(exc))
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        stop_all = True
                        branch_result["error"] = "interrupted"
                finally:
                    if branch is not None:
                        branch_result["host_close"] = branch.close()
                        sessions.remove(branch)
                        if branch_result["host_close"].get("status") != "stopped":
                            branch_result.update(status="failed", error="probe_host_close_not_stopped")
                    save()
            row["status"] = (
                "interrupted"
                if stop_all or any(b["status"] == "interrupted" for b in row["branches"])
                else ("passed" if all(branch["status"] == "passed" for branch in row["branches"]) else "failed")
            )
        except BaseException as exc:
            row.update(status=getattr(exc, "status", "failed"), error=_error_code(exc))
            stop_all = bool(getattr(exc, "stop_all", False)) or isinstance(exc, (KeyboardInterrupt, SystemExit))
            if stop_all:
                row["status"] = "interrupted"
            present = {branch["step_id"] for branch in row["branches"]}
            for probe in (
                step for step in case["steps"] if step["stage"] == "probe" and step["step_id"] not in present
            ):
                row["branches"].append(
                    {"step_id": probe["step_id"], "status": "not_run", "speech": None, "error": "setup_not_complete"}
                )
        finally:
            for session in sessions:
                row.setdefault("cleanup", []).append(session.close())
            save()
    report["status"] = (
        "interrupted"
        if any(row["status"] == "interrupted" for row in report["runs"])
        else ("passed" if all(row["status"] == "passed" for row in report["runs"]) else "failed")
    )
    report["budget_after"] = transport.ledger.snapshot()
    report["request_index"] = [
        {
            key: row.get(key)
            for key in ("request_id", "capture_file", "scope", "step_id", "run_mode", "status", "dispatched")
        }
        for row in transport.requests
    ]
    write_json(output / "report.json", report)
    return report


def main() -> int:
    import memcore
    from .akane_host import allow_model_network, initialize_isolated_akane
    from .akane_transport import AkaneBudgetedTransport
    from .budget import BudgetLedger
    from .production_embedding import create_production_embedding, verify_production_embedding

    root = Path(__file__).resolve().parents[2]
    if not Path(memcore.__file__).resolve().is_relative_to(root / "memcore"):
        raise RuntimeError("unexpected_memcore_runtime_source")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--akane-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--equivalence-evidence", type=Path, required=True)
    parser.add_argument("--pack", type=Path, default=root / "docs/research/pilot_v2")
    args = parser.parse_args()
    args.akane_root, args.output, args.pack = args.akane_root.resolve(), args.output.resolve(), args.pack.resolve()
    gate = json.loads(args.equivalence_evidence.read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("actual_engine_exercised") is not True:
        raise RuntimeError("actual_host_equivalence_gate_required")
    if args.output.exists():
        raise RuntimeError("new_retest_output_required")
    secret = os.environ.get("DEEPSEEK_API_KEY", "")
    model_path = os.environ.get("PILOT_EMBEDDING_MODEL_PATH", "")
    if not secret or not model_path:
        raise RuntimeError("prebound_credential_and_local_embedding_required")
    embedding_baseline = json.loads(
        (root / ".research-runs/embedding-preflight/preflight.json").read_text(encoding="utf-8")
    )
    pack = load_pack(args.pack)
    source = freeze_sources(root, args.akane_root, args.pack, args.output / "source")
    gate_hashes = gate.get("source_hashes", {})
    required_gate_sources = {
        "akane/companion_v01/engine.py",
        "akane/companion_v01/llm_runtime.py",
        "akane/services/llm_client.py",
        "memcore/examples/research_pilot/akane_host.py",
        "memcore/examples/research_pilot/akane_transport.py",
    }
    if not isinstance(gate_hashes, dict) or not required_gate_sources <= set(gate_hashes):
        raise RuntimeError("host_equivalence_source_binding_required")
    if any(source["file_hashes"].get(name) != expected_hash for name, expected_hash in gate_hashes.items()):
        raise RuntimeError("host_equivalence_sources_changed")
    ledger_path = root / ".research-runs/stage1-budget.sqlite3"
    with BudgetLedger(ledger_path, expected_stage_id=pack["manifest"]["external_limits"]["stage_id"]) as ledger:
        ledger.assert_ready()
        initialize_isolated_akane(
            args.akane_root, args.output, read_roots=(Path(model_path),), write_paths=(ledger_path,)
        )
        embedding = create_production_embedding(local_model_path=model_path, device="cuda")
        health = verify_production_embedding(embedding)
        if health["status"] != "passed":
            raise RuntimeError("production_embedding_health_failed")
        if any(health[key] != embedding_baseline[key] for key in ("model_id", "dimension")) or any(
            abs(health[key] - embedding_baseline[key]) > 0.0001
            for key in ("similar_score", "unrelated_score", "repeat_score")
        ):
            raise RuntimeError("production_embedding_baseline_changed")
        health["previous_health_probe_matched"] = True
        health["runtime_package_versions"] = {
            name: getattr(sys.modules.get(module), "__version__", None)
            for name, module in {
                "sentence-transformers": "sentence_transformers",
                "transformers": "transformers",
                "torch": "torch",
                "numpy": "numpy",
                "huggingface-hub": "huggingface_hub",
            }.items()
        }
        transport = AkaneBudgetedTransport(
            ledger,
            capture_dir=args.output / "requests",
            api_key_provider=lambda: secret,
            network_context_factory=allow_model_network,
        )
        report = run_actual_host(
            pack,
            args.output / "run",
            embedding=embedding,
            transport=transport,
            source_check=lambda: verify_sources(source, root, args.akane_root, args.pack),
            source_evidence=source,
            embedding_evidence=health,
            gate_evidence=gate,
            secret=secret,
        )
        print(json.dumps({"status": report["status"], "budget": report["budget_after"]}, ensure_ascii=False))
        return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
