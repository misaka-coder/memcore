"""A fixed-history, fixed-tool-choice contrast of Akane's empty-read guidance.

The recorded initial probe response is replayed once, then the unchanged actual
Engine continues with paid model requests. Only the empty-result text constant
is varied. This measures continuation behavior after a known empty retrieval;
it does not measure autonomous tool selection or independent history samples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

from .akane_provenance import freeze_sources, verify_sources
from .akane_replay import ReplayMapper, ReplayMismatch
from .akane_run import _run_step, _snapshot_from_turn
from .akane_transport import AkaneBudgetedTransport, AkaneTransportInterrupted
from .runner import FixtureService, digest, write_json


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrefixReplayTransport(AkaneBudgetedTransport):
    """Explicitly replay one recorded choice, then permit budgeted continuation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._choice: ContextVar[dict[str, Any] | None] = ContextVar("fixed_probe_choice", default=None)
        self._choice_active = False

    @contextmanager
    def fixed_choice(self, record: dict[str, Any], matcher: Callable) -> Iterator[None]:
        if self._choice_active:
            raise AkaneTransportInterrupted("nested_fixed_choice_forbidden")
        state = {"record": record, "matcher": matcher, "used": False}
        token = self._choice.set(state)
        self._choice_active = True
        try:
            yield
            if not state["used"]:
                raise AkaneTransportInterrupted("fixed_choice_not_consumed")
        finally:
            self._choice.reset(token)
            self._choice_active = False

    def _send(self, original: Callable, request: Any, args: tuple, kwargs: dict, role: str) -> Any:
        choice = self._choice.get()
        if self._choice_active and choice is None:
            raise AkaneTransportInterrupted("fixed_choice_context_missing_in_background")
        if choice is not None and not choice["used"]:
            with self.replay([choice["record"]], request_matcher=choice["matcher"]):
                response = super()._send(original, request, args, kwargs, role)
            choice["used"] = True
            return response
        return super()._send(original, request, args, kwargs, role)


def normalized_continuation(
    request: dict[str, Any], *, call_id: str, expected_body: str, guidance: str
) -> dict[str, Any]:
    """Permit only the predeclared guidance tail of one bound tool result."""
    normalized = copy.deepcopy(request)
    candidates = [
        item for item in normalized["messages"] if item.get("role") == "tool" and item.get("tool_call_id") == call_id
    ]
    if len(candidates) != 1 or candidates[0].get("content") != expected_body:
        raise AkaneTransportInterrupted("unexpected_empty_receipt_body_or_binding")
    prefix, separator, tail = expected_body.partition("\n\n")
    if not separator or tail != guidance or "status=empty；returned=0" not in prefix:
        raise AkaneTransportInterrupted("unexpected_empty_receipt_status_or_guidance")
    candidates[0]["content"] = prefix + "\n\n<PREDECLARED_EMPTY_RETRIEVAL_GUIDANCE>"
    return normalized


def load_inputs(pack: Path, previous: Path) -> dict[str, Any]:
    manifest = _read(pack / "run_manifest.json")
    if _sha(previous / "run/report.json") != manifest["conditioning"]["report_sha256"]:
        raise RuntimeError("conditioning_report_changed")
    source = _read(previous / "run/report.json")
    for name, expected in manifest["conditioning"]["capture_sha256"].items():
        if Path(name).name != name or _sha(previous / "requests" / name) != expected:
            raise RuntimeError("conditioning_capture_changed")
    selected = next(row for row in source["runs"] if row["run_id"] == manifest["conditioning"]["run_id"])
    probe = next(row for row in selected["branches"] if row["step_id"] == manifest["conditioning"]["probe_id"])
    records = {
        item["request_id"]: _read(previous / "requests" / Path(item["capture_file"]).name)
        for item in source["request_index"]
        if item["request_id"] in {key for row in [*selected["setup"], probe] for key in row["request_ids"]}
    }
    original = records[probe["request_ids"][0]]
    if original["response_body_sha256"] != manifest["conditioning"]["initial_response_sha256"]:
        raise RuntimeError("conditioning_response_changed")
    continuation = records[probe["request_ids"][1]]
    choices = original["response"]["choices"]
    calls = choices[0]["message"]["tool_calls"]
    if len(calls) != 1 or calls[0]["function"]["name"] != "retrieve_memory":
        raise RuntimeError("conditioning_choice_must_be_single_retrieve_memory")
    call_id = calls[0]["id"]
    bodies = [
        message["content"]
        for message in continuation["request"]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    ]
    if len(bodies) != 1 or "status=empty；returned=0" not in bodies[0]:
        raise RuntimeError("conditioning_empty_result_required")
    if digest(original["request"]) != manifest["conditioning"]["initial_request_hash"]:
        raise RuntimeError("conditioning_request_changed")
    scenarios = _read(pack / "scenarios.json")
    if scenarios.get("synthetic") is not True or len(scenarios["scenarios"]) != 1:
        raise RuntimeError("one_synthetic_fixed_history_required")
    if manifest["runs"] != [
        {"pair": pair, "condition": condition, "run_id": f"empty_receipt__{condition}__r{pair}"}
        for pair in range(1, 6)
        for condition in (("old", "new") if pair % 2 else ("new", "old"))
    ]:
        raise RuntimeError("predeclared_ten_run_order_required")
    return {
        "manifest": manifest,
        "scenarios": scenarios,
        "setup": selected["setup"],
        "baseline": selected["setup_snapshot"],
        "probe": probe,
        "records": records,
        "original_choice": original,
        "old_body": bodies[0],
        "old_guidance": bodies[0].partition("\n\n")[2],
        "call_id": call_id,
        "reference_continuation": continuation,
        "embedding_baseline": source["embedding_evidence"],
    }


def run_contrast(
    inputs: dict[str, Any],
    output: Path,
    *,
    transport: PrefixReplayTransport,
    embedding: Any,
    source_check: Callable[[], None],
    source_evidence: dict[str, Any],
    secret: str,
    embedding_evidence: dict[str, Any],
) -> dict[str, Any]:
    from companion_v01 import retrieval_engine

    from .akane_host import AkaneHostSession

    output.mkdir(parents=True, exist_ok=False)
    new_guidance = retrieval_engine.EMPTY_RETRIEVAL_FOLLOWUP_GUIDANCE
    if new_guidance == inputs["old_guidance"]:
        raise RuntimeError("distinct_guidance_conditions_required")
    new_body = inputs["old_body"].partition("\n\n")[0] + "\n\n" + new_guidance
    reference = normalized_continuation(
        inputs["reference_continuation"]["request"],
        call_id=inputs["call_id"],
        expected_body=inputs["old_body"],
        guidance=inputs["old_guidance"],
    )
    report: dict[str, Any] = {
        "format": "akane_fixed_empty_retrieval_contrast_v1",
        "status": "running",
        "method": "Actual Engine continuation, original setup and initial tool choice replayed; one tool-result guidance tail varied.",
        "autonomous_tool_selection_measured": False,
        "independent_histories": 1,
        "planned_conditioned_samples": 10,
        "source_evidence": source_evidence,
        "embedding_evidence": embedding_evidence,
        "conditioning": inputs["manifest"]["conditioning"],
        "guidance": {"old": inputs["old_guidance"], "new": new_guidance},
        "normalized_first_continuation_hash": digest(reference),
        "budget_before": transport.ledger.snapshot(),
        "runs": [],
        "scores": None,
    }
    case = inputs["scenarios"]["scenarios"][0]
    steps = {step["step_id"]: step for step in case["steps"]}
    probe_id = inputs["manifest"]["conditioning"]["probe_id"]
    halted = False

    def save() -> None:
        report["budget_latest"] = transport.ledger.snapshot()
        write_json(output / "report.json", report)

    for plan in inputs["manifest"]["runs"]:
        row: dict[str, Any] = {**plan, "status": "not_run", "error": None, "setup_replay": []}
        report["runs"].append(row)
        if halted:
            row["error"] = "earlier_transport_or_control_interruption"
            save()
            continue
        host = None
        try:
            source_check()
            fixture = FixtureService(case)
            host = AkaneHostSession(
                output / plan["run_id"],
                policy="full_until_raw_compaction",
                embedding=embedding,
                transport=transport,
                fixture_resolver=lambda current, arguments: fixture.lookup(current["step_id"], dict(arguments)),
                stable_system_blocks_provider=lambda: (inputs["scenarios"]["shared_model_instruction"],),
            )
            mapper = ReplayMapper()
            guidance = inputs["old_guidance"] if plan["condition"] == "old" else new_guidance
            body = inputs["old_body"] if plan["condition"] == "old" else new_body
            with patch.object(retrieval_engine, "EMPTY_RETRIEVAL_FOLLOWUP_GUIDANCE", guidance):
                for original in inputs["setup"]:
                    step = {**steps[original["step_id"]], "timestamp": original["input_timestamp"]}
                    restored = _run_step(
                        host,
                        transport,
                        step,
                        scope=plan["run_id"],
                        source_check=source_check,
                        secret=secret,
                        replay_records=[inputs["records"][key] for key in original["request_ids"]],
                        mapper=mapper,
                    )
                    row["setup_replay"].append(restored)
                    if restored["status"] != "passed":
                        raise ReplayMismatch("conditioning_setup_replay_failed")
                    mapper.compare_snapshot(_snapshot_from_turn(original), _snapshot_from_turn(restored))
                row["baseline_comparison"] = mapper.compare_snapshot(inputs["baseline"], host.snapshot())
                # The original probe was itself an independent replay branch.
                # Its generated assistant IDs differ from the original setup's
                # IDs, so it owns a separate proven bijection to this branch.
                choice_mapper = ReplayMapper()
                first_paid_checked = False

                def check_request(capture: dict[str, Any]) -> None:
                    nonlocal first_paid_checked
                    if capture["run_mode"] == "replay":
                        try:
                            choice_mapper.bind(inputs["original_choice"]["memory_evidence"], capture["memory_evidence"])
                        except ReplayMismatch as exc:
                            raise AkaneTransportInterrupted(str(exc)) from None
                        return
                    if first_paid_checked:
                        return
                    normalized = normalized_continuation(
                        capture["request"], call_id=inputs["call_id"], expected_body=body, guidance=guidance
                    )
                    if normalized != reference:
                        raise AkaneTransportInterrupted("first_continuation_not_guidance_only_contrast")
                    capture["contrast_control"] = {
                        "only_guidance_changed_from_reference": True,
                        "normalized_request_hash": digest(normalized),
                        "condition": plan["condition"],
                    }
                    first_paid_checked = True

                with transport.fixed_choice(inputs["original_choice"], choice_mapper.compare_request):
                    row["probe"] = _run_step(
                        host,
                        transport,
                        {**steps[probe_id], "timestamp": inputs["probe"]["input_timestamp"]},
                        scope=plan["run_id"],
                        source_check=source_check,
                        secret=secret,
                        preserve_input_timestamp=True,
                        request_check=check_request,
                    )
                row["first_paid_continuation_control_passed"] = first_paid_checked
                row["status"] = row["probe"]["status"]
                row["error"] = row["probe"]["error"]
                row["host_evidence"] = host.evidence()
                if row["status"] == "interrupted" or not first_paid_checked:
                    halted = True
        except BaseException as exc:
            row.update(
                status="interrupted",
                error=row.get("probe", {}).get("error") or getattr(exc, "code", type(exc).__name__),
            )
            halted = True
        finally:
            if host is not None:
                row["host_close"] = host.close()
                if row["host_close"].get("status") != "stopped":
                    row.update(status="interrupted", error="contrast_host_close_failed")
                    halted = True
            save()
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["runs"]) else "incomplete"
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
    from .budget import BudgetLedger
    from .production_embedding import create_production_embedding, verify_production_embedding

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--akane-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline-gate", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not Path(memcore.__file__).resolve().is_relative_to(root / "memcore"):
        raise RuntimeError("unexpected_memcore_runtime_source")
    pack = root / "docs/research/pilot_v3"
    previous = root / ".research-runs/akane-retest-live-20260906-attempt02"
    output, akane = args.output.resolve(), args.akane_root.resolve()
    if output.exists():
        raise RuntimeError("fresh_contrast_output_required")
    inputs = load_inputs(pack, previous)
    secret, model_path = os.environ.get("DEEPSEEK_API_KEY", ""), os.environ.get("PILOT_EMBEDDING_MODEL_PATH", "")
    if not secret or not model_path:
        raise RuntimeError("prebound_credential_and_embedding_required")
    gate = _read(args.offline_gate)
    if gate.get("status") != "passed" or gate.get("actual_engine_exercised") is not True:
        raise RuntimeError("actual_contrast_offline_gate_required")
    source = freeze_sources(root, akane, pack, output / "source")
    if source["file_hashes"] != gate["source_evidence"]["file_hashes"]:
        raise RuntimeError("contrast_gate_source_changed")
    ledger_path = root / ".research-runs/stage1-budget.sqlite3"
    with BudgetLedger(ledger_path, expected_stage_id=inputs["manifest"]["stage_id"]) as ledger:
        ledger.assert_ready()
        initialize_isolated_akane(akane, output, read_roots=(Path(model_path),), write_paths=(ledger_path,))
        embedding = create_production_embedding(local_model_path=model_path, device="cuda")
        health = verify_production_embedding(embedding)
        if health["status"] != "passed":
            raise RuntimeError("production_embedding_health_failed")
        baseline = inputs["embedding_baseline"]
        if any(health[key] != baseline[key] for key in ("model_id", "dimension")) or any(
            abs(health[key] - baseline[key]) > 0.0001 for key in ("similar_score", "unrelated_score", "repeat_score")
        ):
            raise RuntimeError("conditioning_embedding_baseline_changed")
        health["conditioning_health_probe_matched"] = True
        transport = PrefixReplayTransport(
            ledger,
            capture_dir=output / "requests",
            api_key_provider=lambda: secret,
            network_context_factory=allow_model_network,
        )
        result = run_contrast(
            inputs,
            output / "run",
            transport=transport,
            embedding=embedding,
            source_check=lambda: verify_sources(source, root, akane, pack),
            source_evidence=source,
            secret=secret,
            embedding_evidence=health,
        )
        print(json.dumps({"status": result["status"], "budget": result["budget_after"]}, ensure_ascii=False))
        return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
