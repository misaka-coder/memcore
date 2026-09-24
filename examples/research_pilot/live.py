"""Autonomous synthetic pilot using public MemCore and actual Akane prompt blocks.

The evaluator's answers are never passed to a session or model. Journal replay
restores actual model messages and tool bodies without asking the model again.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from memcore import (
    EntryOrigin,
    ProjectionMessageInput,
    TimelineEntryInput,
    TurnRole,
    dispatch_native_memory_tool,
    parse_chat_output,
)

from .akane_prompt import compose_akane_messages
from .preflight import MODEL, base_request, response_message
from .runner import MEMORY_TOOLS, PROFILE, EventLog, PilotFailure, SmokeSession, canonical, digest, write_json


def error_code(exc: BaseException) -> str:
    if hasattr(exc, "code"):
        return str(exc.code)
    return str(exc) if isinstance(exc, PilotFailure) else type(exc).__name__


def continuation_handles(value: Any) -> list[str]:
    """Find active namespace-bound cursors that cannot be copied to a new branch."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "next_cursor" and isinstance(item, str) and item:
                found.append(item)
            else:
                found.extend(continuation_handles(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(continuation_handles(item))
    return found


def projected_result_body(message: Any, item: dict[str, Any]) -> tuple[str | None, bool]:
    """Read either native content or MemCore's exact documented safety wrapper."""
    payload = message.payload
    if payload.get("role") == "tool" and payload.get("tool_call_id") == item["call_id"]:
        content = payload.get("content")
        return (content if isinstance(content, str) else None), False
    prefix = "\n".join(
        [
            "[historical tool result retained as canonical trace]",
            "call_id: " + item["call_id"],
            "source_ids: " + ", ".join(message.source_ids),
            "result:",
            "",
        ]
    )
    content = payload.get("content")
    if payload.get("role") == "user" and isinstance(content, str) and content.startswith(prefix):
        return content[len(prefix) :], True
    return None, False


class LiveSession(SmokeSession):
    """One independent hard namespace, store and index; one shared real embedder."""

    def __init__(self, *, prompt_bundle: dict[str, Any], client: Any, **kwargs: Any):
        if kwargs.get("embedding") is None:
            raise PilotFailure("live_embedding_required")
        super().__init__(**kwargs)
        self.prompt_bundle, self.client = prompt_bundle, client
        self.wire: list[dict[str, Any]] = []
        self.turn_request_hashes: list[str] = []

    def request(self, handle: Any, step: dict[str, Any], *, replay: bool) -> dict[str, Any]:
        # Read MemCore's complete authoritative history each time. Override only
        # the active suffix with original provider messages not yet frozen.
        projected = self.mem.build_context_projection(provider_profile=PROFILE)
        closed = [m.payload for m in projected.messages if m.turn_id != handle.turn_id]
        current_index = len(closed)
        history = closed + [item["payload"] for item in self.wire]
        if self.wire[0]["source_ids"] != (handle.stimuli[0].source_id,):
            raise PilotFailure("current_stimulus_mapping_changed")
        if {t["function"]["name"] for t in self.tools} != MEMORY_TOOLS | {"lookup_fixture"}:
            raise PilotFailure("model_tool_allowlist_mismatch")
        composed = compose_akane_messages(
            self.prompt_bundle,
            history_messages=history,
            step_id=step["step_id"],
            current_message_index=current_index,
            extra_system_blocks=(self.system,),
        )
        actual_history = composed["history_payloads"]
        if actual_history[:current_index] != closed:
            raise PilotFailure("host_changed_closed_memory_history")
        if actual_history[current_index + 1 :] != history[current_index + 1 :]:
            raise PilotFailure("host_changed_native_tool_suffix")
        self.wire[0]["payload"] = copy.deepcopy(actual_history[current_index])
        actual_messages = composed["messages"]
        declared_indexes = composed["history_message_indexes"][current_index:]
        self.mem.record_request_projection(
            turn_id=handle.turn_id,
            provider_profile=PROFILE,
            turn_messages=[
                ProjectionMessageInput(
                    provider_profile=PROFILE,
                    payload=actual_history[current_index + index],
                    source_ids=item["source_ids"],
                    projection_index=index,
                )
                for index, item in enumerate(self.wire)
            ],
            history_messages=actual_messages,
            history_message_indexes=declared_indexes,
            audit_history_messages=actual_messages,
            model_route=MODEL,
            system_prefix=composed["system_prefix"],
            tool_schema=self.tools,
            created_at=int(datetime.fromisoformat(step["timestamp"]).timestamp()),
        )
        request = base_request(actual_messages, self.tools)
        request["tool_choice"] = "auto"
        # The stable prompt specifies final JSON; native tool calls remain native.
        self.turn_request_hashes.append(digest(request))
        if not replay:
            self.requests += 1
            self.log.emit(
                "model_request_prepared",
                scope=self.scope,
                step_id=step["step_id"],
                request_hash=digest(request),
                current_stimulus_occurrences=1,
                host_history_message_indexes=composed["history_message_indexes"],
                ephemeral_message_count=len(composed["ephemeral_messages"]),
                tool_schema_hash=digest(self.tools),
            )
        return request

    def _tool_result(self, name: str, arguments: Any, step: dict[str, Any], handle: Any) -> tuple[dict, Any]:
        if not isinstance(arguments, dict):
            return {"status": "invalid", "reason": "tool_arguments_must_be_object"}, None
        if name == "lookup_fixture":
            return self.fixtures.lookup(step["step_id"], arguments), None
        dispatched = dispatch_native_memory_tool(name, arguments, mem=self.mem, current=handle.stimuli[0].to_record())
        return {k: v for k, v in dispatched.items() if k != "receipt"}, dispatched.get("receipt")

    def run_turn(self, step: dict[str, Any], *, replay_trace: dict[str, Any] | None = None) -> dict[str, Any]:
        replay = replay_trace is not None
        stamp = int(datetime.fromisoformat(step["timestamp"]).timestamp())
        turn_id = step["step_id"]
        trace: dict[str, Any] = {
            "step_id": turn_id,
            "stage": step["stage"],
            "status": "running",
            "speech": None,
            "assistant_messages": [],
            "tool_calls": [],
            "request_hashes": [],
            "observation_checks": [],
            "error": None,
        }
        handle = self.mem.begin_turn(
            turn_id=turn_id,
            opened_at=stamp,
            stimuli=[
                TimelineEntryInput(
                    source_id=f"{turn_id}:user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text=step["user_text"],
                    payload={"text": step["user_text"]},
                    timestamp=stamp,
                    compatibility_role="user",
                )
            ],
        )
        self.source_map[turn_id] = {"source_id": handle.stimuli[0].source_id}
        self.wire = [
            {
                "payload": {"role": "user", "content": step["user_text"]},
                "source_ids": (handle.stimuli[0].source_id,),
            }
        ]
        self.turn_request_hashes = []
        completed, calls_used, response_index, replay_tool_index = False, 0, 0, 0
        observations: list[dict[str, Any]] = []
        try:
            # A successful tool round consumes at least one of the finite calls.
            for _round in range(self.manifest["proposed_runner_limits"]["max_tool_calls_per_turn"] + 1):
                request = self.request(handle, step, replay=replay)
                if replay:
                    saved = replay_trace["assistant_messages"]
                    if response_index >= len(saved):
                        raise PilotFailure("replay_responses_exhausted")
                    response = copy.deepcopy(saved[response_index])
                    finish = "tool_calls" if response.get("tool_calls") else "stop"
                else:
                    raw = self.client.call(request, scope=self.scope, step_id=turn_id)
                    finish = raw["choices"][0]["finish_reason"]
                    response = response_message(raw, finish)
                response_index += 1
                trace["assistant_messages"].append(copy.deepcopy(response))
                if not replay:
                    self.log.emit(
                        "model_response",
                        scope=self.scope,
                        step_id=turn_id,
                        message=response,
                        forced_routing=False,
                    )
                calls = response.get("tool_calls") or []
                if calls:
                    if finish != "tool_calls" or not isinstance(calls, list):
                        raise PilotFailure("tool_calls_finish_mismatch")
                    calls_used += len(calls)
                    if calls_used > self.manifest["proposed_runner_limits"]["max_tool_calls_per_turn"]:
                        raise PilotFailure("tool_call_limit_exceeded")
                    seen_ids = {item["call_id"] for item in trace["tool_calls"]}
                    actions: list[tuple[dict, str, Any]] = []
                    for call in calls:
                        if not isinstance(call, dict) or call.get("type") != "function":
                            raise PilotFailure("invalid_native_tool_call")
                        function, call_id = call.get("function"), call.get("id")
                        if (
                            not isinstance(function, dict)
                            or function.get("name") not in MEMORY_TOOLS | {"lookup_fixture"}
                            or not isinstance(function.get("arguments"), str)
                            or not isinstance(call_id, str)
                            or not call_id
                            or call_id in seen_ids
                        ):
                            raise PilotFailure("invalid_or_unapproved_tool_call")
                        seen_ids.add(call_id)
                        try:
                            arguments = json.loads(function["arguments"])
                        except ValueError:
                            arguments = None
                        source_id = f"{turn_id}:action:{len(trace['tool_calls']) + len(actions)}"
                        self.mem.append_action(
                            turn_id=turn_id,
                            kind=f"tool.{function['name']}.call",
                            correlation_id=call_id,
                            payload={"input": function["arguments"]},
                            source_id=source_id,
                            timestamp=stamp,
                            trace_metadata={"tool_name": function["name"]},
                        )
                        actions.append((call, source_id, arguments))
                    # One full provider message owns every action in this batch.
                    self.wire.append(
                        {
                            "payload": copy.deepcopy(response),
                            "source_ids": tuple(source_id for _, source_id, _ in actions),
                        }
                    )
                    for call, _action_id, arguments in actions:
                        name, call_id = call["function"]["name"], call["id"]
                        if replay:
                            saved_tools = replay_trace["tool_calls"]
                            if replay_tool_index >= len(saved_tools):
                                raise PilotFailure("replay_tool_results_exhausted")
                            saved_tool = saved_tools[replay_tool_index]
                            replay_tool_index += 1
                            if (
                                saved_tool["name"] != name
                                or saved_tool["call_id"] != call_id
                                or saved_tool["arguments"] != arguments
                            ):
                                raise PilotFailure("replay_tool_binding_mismatch")
                            result, receipt = copy.deepcopy(saved_tool["result"]), copy.deepcopy(saved_tool["receipt"])
                            if digest(canonical(result)) != saved_tool["body_hash"]:
                                raise PilotFailure("replay_tool_body_hash_mismatch")
                        else:
                            result, receipt = self._tool_result(name, arguments, step, handle)
                        body = canonical(result)
                        nested = result.get("result")
                        nested_status = nested.get("status") if isinstance(nested, dict) else None
                        status = str(nested_status or result.get("status", "failed"))
                        source_id = f"{turn_id}:result:{len(trace['tool_calls'])}"
                        if replay and source_id != saved_tool["source_id"]:
                            raise PilotFailure("replay_tool_source_id_mismatch")
                        observed = self.mem.append_observation(
                            turn_id=turn_id,
                            kind=f"tool.{name}.result",
                            correlation_id=call_id,
                            payload={"output": body},
                            source_id=source_id,
                            timestamp=stamp,
                            status=status,
                            retention_anchor=receipt,
                            trace_metadata={"tool_name": name},
                        )
                        tool_trace = {
                            "name": name,
                            "call_id": call_id,
                            "arguments": arguments,
                            "result": result,
                            "receipt": receipt,
                            "source_id": observed.source_id,
                            "body": body,
                            "body_hash": digest(body),
                            "status": status,
                        }
                        trace["tool_calls"].append(tool_trace)
                        observations.append(tool_trace)
                        self.wire.append(
                            {
                                "payload": {"role": "tool", "tool_call_id": call_id, "content": body},
                                "source_ids": (observed.source_id,),
                            }
                        )
                        if name == "lookup_fixture" and status == "ok":
                            self.source_map[result["document_id"]] = {
                                "source_id": observed.source_id,
                                "body_hash": digest(body),
                                "call_id": call_id,
                            }
                        if not replay:
                            self.log.emit("tool_result", scope=self.scope, step_id=turn_id, **tool_trace)
                    continue
                if finish != "stop":
                    raise PilotFailure("final_finish_reason_not_stop")
                parsed = parse_chat_output(response.get("content", ""), mode="memcore_json", enable_flavor=False)
                if not parsed.ok:
                    raise PilotFailure("invalid_final_json")
                trace["speech"] = parsed.speech
                open_projection = self.mem.build_context_projection(provider_profile=PROFILE)
                for item in observations:
                    matching = [
                        m
                        for m in open_projection.messages
                        if m.turn_id == turn_id and item["source_id"] in m.source_ids
                    ]
                    if len(matching) != 1 or projected_result_body(matching[0], item)[0] != item["body"]:
                        raise PilotFailure("open_loop_result_changed_or_duplicated")
                result = self.mem.complete_turn(
                    turn_id=turn_id,
                    semantic_text=parsed.speech,
                    provider_output_raw=response["content"],
                    source_id=f"{turn_id}:final",
                    timestamp=stamp,
                    memory_annotation=parsed.memory_metadata,
                    annotation_status="accepted_model" if parsed.metadata_status == "accepted" else "missing",
                    provider_profile=PROFILE,
                    provider_projection=response,
                )
                if not result.completed:
                    raise PilotFailure("final_commit_" + result.status)
                completed = True
                closed_projection = self.mem.build_context_projection(provider_profile=PROFILE)
                for item in observations:
                    matching = [
                        m
                        for m in closed_projection.messages
                        if m.turn_id == turn_id and item["source_id"] in m.source_ids
                    ]
                    if len(matching) != 1:
                        raise PilotFailure("closed_tool_wire_changed")
                    closed_body, canonical_fallback = projected_result_body(matching[0], item)
                    if closed_body is None:
                        raise PilotFailure("closed_result_projection_unknown")
                    opened = self.mem.open_memory(memory_id=item["source_id"], view="content")
                    _, separator, raw_data = opened.get("text", "").partition("\ndata:\n")
                    readback = json.loads(raw_data) if separator else {}
                    if opened.get("status") != "ok" or readback.get("output") != item["body"]:
                        raise PilotFailure("readback_body_mismatch")
                    compact = "[compact_reloadable]" in closed_body
                    if self.policy == "full_until_raw_compaction" and closed_body != item["body"]:
                        raise PilotFailure("full_condition_changed_result")
                    if (
                        self.policy == "compact_after_terminal"
                        and item["name"] == "lookup_fixture"
                        and item["status"] == "ok"
                        and not compact
                    ):
                        raise PilotFailure("fixture_not_actually_settled")
                    trace["observation_checks"].append(
                        {
                            "source_id": item["source_id"],
                            "name": item["name"],
                            "open_full": True,
                            "reload_equal": True,
                            "closed_compact": compact,
                            "canonical_safety_fallback": canonical_fallback,
                        }
                    )
                if replay:
                    if response_index != len(replay_trace["assistant_messages"]):
                        raise PilotFailure("replay_unused_responses")
                    if replay_tool_index != len(replay_trace["tool_calls"]):
                        raise PilotFailure("replay_unused_tool_results")
                    if self.turn_request_hashes != replay_trace["request_hashes"]:
                        raise PilotFailure("replay_request_hash_mismatch")
                trace["status"] = "passed"
                break
            if not completed:
                raise PilotFailure("request_round_limit_exceeded")
        except Exception as exc:
            trace.update(status="failed", error=error_code(exc))
        finally:
            if not completed:
                self.mem.abort_turn(turn_id, reason="research_pilot_failed", closed_at=stamp)
            trace["request_hashes"] = list(self.turn_request_hashes)
            if not replay:
                self.log.emit(
                    "turn_finished",
                    scope=self.scope,
                    step_id=turn_id,
                    status=trace["status"],
                    error=trace["error"],
                    settlement_metrics=self.mem.settlement_metrics(),
                    observation_checks=trace["observation_checks"],
                )
        return trace


def run_live(
    pack: dict[str, Any],
    output: Path,
    snapshot: dict[str, Any],
    *,
    prompt_bundle: dict[str, Any],
    embedding: Any,
    embedding_report: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    """Run every configured condition once and each probe on a separate replay."""
    output.mkdir(parents=True, exist_ok=False)
    live_transport = getattr(client, "is_live", False) is True
    mode = "autonomous_model_pilot" if live_transport else "offline_live_runner_test"
    log = EventLog(output / "events.jsonl", run_mode=mode)
    specification, manifest = pack["scenarios"], pack["manifest"]
    scenarios = {s["scenario_id"]: s for s in specification["scenarios"]}
    report: dict[str, Any] = {
        "format": "akane_memcore_live_pilot_v1",
        "run_mode": mode,
        "status": "running",
        "actual_model": MODEL if live_transport else "offline_injected_not_deepseek",
        "forced_routing": not live_transport,
        "source_snapshot": snapshot,
        "embedding_verification": embedding_report,
        "embedding_name": embedding.name,
        "embedding_dimension": embedding.dimension,
        "akane_prompt_verified": bool(prompt_bundle.get("evidence", {}).get("imported_source_modules_verified")),
        "akane_engine_verified": False,
        "prompt_bundle_hash": digest(prompt_bundle),
        "no_maintenance_calls": True,
        "runs": [],
        "scores": None,
    }
    write_json(output / "report.json", report)
    for run in manifest["runs"]:
        scenario = scenarios[run["scenario_id"]]
        run_dir = output / run["run_id"]
        run_dir.mkdir()
        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "scenario_id": run["scenario_id"],
            "condition": run["condition"],
            "status": "running",
            "setup": [],
            "branches": [],
            "error": None,
        }
        report["runs"].append(row)
        sessions: list[LiveSession] = []
        policy = manifest["condition_definitions"][run["condition"]]["operation_projection_policy"]
        started = time.monotonic()

        def session(branch: str) -> LiveSession:
            result = LiveSession(
                path=run_dir / f"{branch}.sqlite3",
                scope=f"{run['run_id']}__{branch}",
                scenario=scenario,
                policy=policy,
                specification=specification,
                manifest=manifest,
                log=log,
                embedding=embedding,
                prompt_bundle=prompt_bundle,
                client=client,
            )
            sessions.append(result)
            return result

        try:
            base = session("setup")
            row["control_hash"] = digest({"system": base.system, "tools": base.tools, "prompt": prompt_bundle})
            for step in scenario["steps"]:
                if step["stage"] == "probe":
                    continue
                trace = base.run_turn(step)
                row["setup"].append(trace)
                write_json(run_dir / "trace.json", row)
                if trace["status"] != "passed":
                    raise PilotFailure("setup_turn_failed:" + trace["error"])
            baseline = base.history()
            row["setup_history_hash"] = digest(baseline)
            row["setup_history_utf8_bytes"] = len(canonical(baseline).encode("utf-8"))
            row["source_map"] = copy.deepcopy(base.source_map)
            row["settlement_metrics"] = base.mem.settlement_metrics()
            row["fixtures_observed"] = {
                f["fixture_id"]: f["fixture_id"] in base.source_map for f in scenario["fixtures"]
            }
            cursor_count = sum(
                len(continuation_handles(tool["result"])) + len(continuation_handles(tool["receipt"]))
                for turn in row["setup"]
                for tool in turn["tool_calls"]
            )
            row["checkpoint_portability"] = {
                "status": "supported" if cursor_count == 0 else "unsupported_namespace_bound_cursor",
                "active_cursor_count": cursor_count,
            }
            write_json(
                run_dir / "setup_checkpoint.json",
                {
                    "setup": row["setup"],
                    "history": baseline,
                    "source_map": base.source_map,
                },
            )
            step_by_id = {s["step_id"]: s for s in scenario["steps"]}
            if cursor_count:
                raise PilotFailure("checkpoint_contains_namespace_bound_cursor")
            for step in scenario["steps"]:
                if step["stage"] != "probe":
                    continue
                branch = session(step["step_id"])
                try:
                    for setup_trace in row["setup"]:
                        restored = branch.run_turn(step_by_id[setup_trace["step_id"]], replay_trace=setup_trace)
                        if restored["status"] != "passed":
                            raise PilotFailure("branch_replay_failed:" + str(restored["error"]))
                    if branch.history() != baseline or branch.source_map != base.source_map:
                        raise PilotFailure("probe_replay_baseline_mismatch")
                    log.emit("branch_replayed", scope=branch.scope, baseline_hash=digest(baseline), paid_requests=0)
                    branch_trace = branch.run_turn(step)
                    branch_trace["baseline_equal"] = True
                except Exception as exc:
                    branch_trace = {
                        "step_id": step["step_id"],
                        "status": "failed",
                        "speech": None,
                        "assistant_messages": [],
                        "tool_calls": [],
                        "error": error_code(exc),
                        "baseline_equal": False,
                    }
                row["branches"].append(branch_trace)
                write_json(run_dir / "trace.json", row)
            row["status"] = "passed" if all(b["status"] == "passed" for b in row["branches"]) else "failed"
        except Exception as exc:
            row.update(status="failed", error=error_code(exc))
            present = {b["step_id"] for b in row["branches"]}
            for step in scenario["steps"]:
                if step["stage"] == "probe" and step["step_id"] not in present:
                    row["branches"].append(
                        {
                            "step_id": step["step_id"],
                            "status": "not_run",
                            "speech": None,
                            "assistant_messages": [],
                            "tool_calls": [],
                            "error": "setup_not_complete",
                            "baseline_equal": False,
                        }
                    )
        finally:
            row["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            for opened in sessions:
                opened.close()
            write_json(run_dir / "trace.json", row)
            write_json(output / "report.json", report)
    report["status"] = "passed" if all(r["status"] == "passed" for r in report["runs"]) else "failed"
    report["request_accounting"] = client.requests
    report["ledger"] = client.ledger.snapshot()
    write_json(output / "report.json", report)
    return report
