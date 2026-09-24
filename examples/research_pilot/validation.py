"""Offline validation for the fixed, not-yet-run Phase 0 experiment pack.

This module reads only the three named JSON documents. It never reads credentials
or invokes a model. ``answer_key`` is evaluator-only: callers must not pass the
returned pack (or its gold fields) to a model or expose it through model tools.
Validation establishes structural prerequisites, not successful runtime isolation.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")
_POLICIES = {"full": "full_until_raw_compaction", "card": "compact_after_terminal"}


def _require(condition: bool, location: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{location}: {message}")


def _object(value: Any, location: str) -> dict:
    _require(isinstance(value, dict), location, "expected an object")
    return value


def _list(value: Any, location: str) -> list:
    _require(isinstance(value, list), location, "expected an array")
    return value


def _text(value: Any, location: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), location, "expected nonempty text")
    return value


def _id(value: Any, location: str) -> str:
    value = _text(value, location)
    _require(bool(_IDENTIFIER.fullmatch(value)), location, "invalid identifier")
    return value


def _ids(value: Any, location: str) -> list[str]:
    values = [_id(item, location) for item in _list(value, location)]
    _require(len(values) == len(set(values)), location, "duplicate identifier")
    return values


def _texts(value: Any, location: str) -> list[str]:
    return [_text(item, location) for item in _list(value, location)]


def _indexed(value: Any, field: str, location: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for raw in _list(value, location):
        item = _object(raw, location)
        key = _id(item.get(field), f"{location}.{field}")
        _require(key not in result, location, f"duplicate {field}: {key}")
        result[key] = item
    return result


def _json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result), "JSON", "nonfinite number")
    return result


def _read(path: Path) -> dict:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{path.name}: cannot read valid JSON ({exc})") from exc
    return _object(raw, path.name)


def _timestamp(value: Any, location: str) -> datetime:
    value = _text(value, location)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{location}: invalid ISO timestamp") from exc
    _require(parsed.utcoffset() is not None, location, "timestamp must include a UTC offset")
    return parsed


def _positive_int(value: Any, location: str, *, minimum: int = 1) -> int:
    _require(type(value) is int and value >= minimum, location, f"expected integer >= {minimum}")
    return value


def _validate_scenario(scenario: dict, gold: dict) -> tuple[set[str], set[str], set[str]]:
    sid = scenario["scenario_id"]
    fixtures = _indexed(scenario.get("fixtures"), "fixture_id", f"{sid}.fixtures")
    _require(len(fixtures) == 2, sid, "Phase 0 requires exactly two fixture versions")
    allowed = _ids(scenario.get("allowed_fixture_ids"), f"{sid}.allowed_fixture_ids")
    _require(set(allowed) == set(fixtures), sid, "fixture allowlist must contain exactly the owned fixtures")
    queries: set[str] = set()
    versions: dict[int, str] = {}
    for fid, fixture in fixtures.items():
        args = _object(fixture.get("arguments"), f"{fid}.arguments")
        _require(set(args) == {"query_key"}, fid, "fixture arguments must contain only query_key")
        query = _id(args["query_key"], f"{fid}.query_key")
        _require(query not in queries, fid, "query_key must bind to one fixture")
        queries.add(query)
        match = re.fullmatch(r"(.+)_v([12])", query)
        _require(match is not None, fid, "expected a v1 or v2 query_key")
        version = int(match.group(2))
        _require(version not in versions, sid, "duplicate fixture version")
        versions[version] = fid
        result = _object(fixture.get("result"), f"{fid}.result")
        _require(result.get("document_id") == fid, fid, "result.document_id must match fixture_id")
        _require(result.get("status") == "ok" and result.get("synthetic") is True, fid, "expected synthetic ok result")
        _text(result.get("content"), f"{fid}.content")
    _require(len({query.rsplit("_v", 1)[0] for query in queries}) == 1, sid, "versions must share a query family")

    steps = _indexed(scenario.get("steps"), "step_id", f"{sid}.steps")
    positions = {step_id: i for i, step_id in enumerate(steps)}
    previous_time: datetime | None = None
    probe_started = False
    task_ids: list[str] = []
    probe_ids: list[str] = []
    for step_id, step in steps.items():
        stage = step.get("stage")
        _require(stage in ("conversation", "task", "probe"), step_id, "unsupported stage")
        _text(step.get("user_text"), f"{step_id}.user_text")
        timestamp = _timestamp(step.get("timestamp"), f"{step_id}.timestamp")
        _require(previous_time is None or timestamp > previous_time, step_id, "timestamps must strictly increase")
        previous_time = timestamp
        _require(not probe_started or stage == "probe", step_id, "all probes must follow the setup")
        if stage == "task":
            task_ids.append(step_id)
        if stage == "probe":
            probe_started = True
            probe_ids.append(step_id)
    _require(len(task_ids) == 2 and len(probe_ids) == 3, sid, "expected two tasks and three probes")
    _require(positions[task_ids[1]] < positions[probe_ids[0]] - 1, sid, "post-task gap is required before probes")
    task_checks = _indexed(gold.get("task_checks"), "step_id", f"{sid}.task_checks")
    _require(set(task_checks) == set(task_ids), sid, "gold task coverage must match task steps")
    for number, step_id in enumerate(task_ids, start=1):
        check = task_checks[step_id]
        _require(
            check.get("required_fixture_id") == versions[number], step_id, "gold task must use the matching version"
        )
        _require(bool(_texts(check.get("criteria"), step_id)), step_id, "task criteria must not be empty")
        _texts(check.get("must_not_volunteer"), step_id)

    availability = _object(scenario.get("fixture_availability_by_step"), f"{sid}.fixture_availability_by_step")
    _require(set(availability) == set(steps), sid, "availability must cover exactly every step")
    for step_id, raw_allowed in availability.items():
        available = _ids(raw_allowed, f"{step_id}.availability")
        _require(set(available) <= set(fixtures), step_id, "cross-scenario or unknown fixture in allowlist")
        version = 1 if positions[step_id] < positions[task_ids[1]] else 2
        _require(available == [versions[version]], step_id, "future version access or expired v1 re-fetch")
    unavailable = _object(scenario.get("unavailable_fixture_result"), f"{sid}.unavailable_fixture_result")
    _require(unavailable.get("status") == "unavailable", sid, "unavailable fixtures must not return success")
    _text(unavailable.get("reason"), sid)
    _require(bool(_texts(scenario.get("probe_preconditions"), sid)), sid, "probe preconditions are required")

    probes = _indexed(gold.get("probes"), "step_id", f"{sid}.gold.probes")
    _require(set(probes) == set(probe_ids), sid, "gold probe coverage must match probe steps")
    kinds: list[str] = []
    for step_id, probe in probes.items():
        kind = probe.get("kind")
        _require(
            kind in ("historical_tool_detail", "conversation_fact", "explicitly_missing_field"),
            step_id,
            "invalid probe kind",
        )
        kinds.append(kind)
        expected = _object(probe.get("expected"), f"{step_id}.expected")
        _require(bool(expected), step_id, "expected answer must not be empty")
        fixture_refs = _ids(probe.get("expected_fixture_ids"), f"{step_id}.expected_fixture_ids")
        user_refs = _ids(probe.get("expected_user_step_ids"), f"{step_id}.expected_user_step_ids")
        _require(set(fixture_refs) <= set(fixtures), step_id, "gold references an unknown or foreign fixture")
        _require(set(user_refs) <= set(steps), step_id, "gold references an unknown or foreign user step")
        _require(
            all(positions[ref] < positions[probe_ids[0]] and steps[ref]["stage"] != "probe" for ref in user_refs),
            step_id,
            "gold user evidence must predate the probe snapshot",
        )
        if kind == "conversation_fact":
            _require(not fixture_refs and bool(user_refs), step_id, "conversation gold requires only user evidence")
        else:
            _require(
                fixture_refs == [versions[1]] and not user_refs, step_id, "historical gold requires only v1 evidence"
            )
        if kind == "explicitly_missing_field":
            _require(expected.get("answerable") is False, step_id, "missing field must be unanswerable")
        _require(bool(_texts(probe.get("criteria"), step_id)), step_id, "probe criteria must not be empty")
    _require(len(set(kinds)) == 3, sid, "expected one probe of each kind")
    _text(gold.get("pre_probe_leakage_audit"), sid)
    return set(fixtures), queries, set(steps)


def _validate_manifest(manifest: dict, scenario_ids: set[str]) -> None:
    _require(manifest.get("status") == "prepared_not_run", "manifest", "only prepared packs are supported")
    _require(manifest.get("scenarios_file") == "scenarios.json", "manifest", "unexpected scenario file")
    _require(manifest.get("evaluator_only_file") == "answer_key.json", "manifest", "unexpected evaluator file")
    conditions = _object(manifest.get("condition_definitions"), "condition_definitions")
    _require(set(conditions) == set(_POLICIES), "condition_definitions", "only full and card are supported")
    for name, policy in _POLICIES.items():
        condition = _object(conditions[name], name)
        _require(
            condition == {"operation_projection_policy": policy}, name, "unsupported projection policy or condition"
        )
    controls = _object(manifest.get("fixed_controls"), "fixed_controls")
    _require(controls.get("timezone") == "Asia/Shanghai", "fixed_controls", "unexpected timezone")
    _require(
        controls.get("compaction") == "do_not_call_raw_or_semantic_maintenance_in_phase_0",
        "fixed_controls",
        "maintenance must remain disabled",
    )
    _require(controls.get("model_gold_access") == "forbidden", "fixed_controls", "gold must be evaluator-only")
    care = _object(controls.get("care"), "care")
    for field, expected in (("enabled", False), ("disable_live_client_state", True), ("disable_autonomous_push", True)):
        _require(care.get(field) is expected, f"care.{field}", "unexpected care control")
    _require(care.get("time") == "scripted_timestamps_not_wall_clock_elapsed", "care", "scripted time required")
    _require(care.get("random_actions") == "not_exposed", "care", "random actions must remain unavailable")
    planning = _object(manifest.get("authorized_planning"), "authorized_planning")
    for field, expected in (
        ("provider", "DeepSeek official API"),
        ("base_url", "https://api.deepseek.com"),
        ("model", "deepseek-v4-flash"),
        ("currency", "CNY"),
    ):
        _require(planning.get(field) == expected, f"authorized_planning.{field}", "unexpected authorized setting")
    _require(
        type(planning.get("first_stage_spend_ceiling")) is int and planning["first_stage_spend_ceiling"] == 50,
        "authorized_planning",
        "first-stage budget must be 50 CNY",
    )
    limits = _object(manifest.get("proposed_runner_limits"), "proposed_runner_limits")
    _require(
        type(limits.get("max_paid_spend_cny")) is int and limits["max_paid_spend_cny"] == 50,
        "proposed_runner_limits",
        "global budget must be 50 CNY",
    )
    for field in (
        "research_context_budget_tokens",
        "output_reservation_tokens",
        "max_tool_calls_per_turn",
        "max_inflight_requests",
    ):
        _positive_int(limits.get(field), field)
    _positive_int(limits.get("max_transport_retries"), "max_transport_retries", minimum=0)
    _require(
        limits["research_context_budget_tokens"] > limits["output_reservation_tokens"],
        "proposed_runner_limits",
        "context must exceed output reservation",
    )

    runs = _indexed(manifest.get("runs"), "run_id", "runs")
    expected_pairs = {(sid, condition) for sid in scenario_ids for condition in _POLICIES}
    seen = set()
    for run_id, run in runs.items():
        sid = _id(run.get("scenario_id"), run_id)
        condition = _id(run.get("condition"), run_id)
        _require((sid, condition) in expected_pairs, run_id, "unknown scenario or condition")
        _require((sid, condition) not in seen, run_id, "duplicate scenario/condition run")
        seen.add((sid, condition))
        _require(type(run.get("repeat")) is int and run["repeat"] == 1, run_id, "only repeat 1 is supported")
        _require(run_id == f"{sid}__{condition}__r1", run_id, "run ID must match its scenario and condition")
        _require(
            run.get("status") == "not_run" and "result" in run and run["result"] is None,
            run_id,
            "expected an unrun result",
        )
    _require(seen == expected_pairs and len(runs) == 4, "runs", "four runs must cover the scenario/condition product")


def validate_pack(pack_dir: Path) -> dict:
    """Return parsed ``scenarios``, evaluator-only ``answer_key``, and ``manifest``.

    Malformed JSON and unsupported/inconsistent pack contracts raise ValueError.
    The return value is host data, never a model prompt or tool result.
    """
    _require(isinstance(pack_dir, Path), "pack_dir", "expected pathlib.Path")
    pack = {
        "scenarios": _read(pack_dir / "scenarios.json"),
        "answer_key": _read(pack_dir / "answer_key.json"),
        "manifest": _read(pack_dir / "run_manifest.json"),
    }
    for name, suffix in (("scenarios", "scenarios"), ("answer_key", "answer_key"), ("manifest", "run_manifest")):
        _require(pack[name].get("format") == f"akane_memcore_pilot_{suffix}_v1", name, "unsupported format")
    scenarios = pack["scenarios"]
    answer_key = pack["answer_key"]
    _require(scenarios.get("status") == "prepared_not_run", "scenarios", "only prepared packs are supported")
    _require(
        scenarios.get("synthetic") is True and answer_key.get("synthetic") is True,
        "pack",
        "synthetic fixtures required",
    )
    _require(scenarios.get("timezone") == "Asia/Shanghai", "scenarios", "unexpected timezone")
    _require(answer_key.get("audience") == "evaluator_only", "answer_key", "gold must be evaluator-only")
    owned = _indexed(scenarios.get("scenarios"), "scenario_id", "scenarios")
    gold = _indexed(answer_key.get("cases"), "scenario_id", "answer_key.cases")
    _require(len(owned) == 2 and set(owned) == set(gold), "pack", "expected two scenarios with matching gold cases")
    global_ids: tuple[set[str], set[str], set[str]] = (set(), set(), set())
    for sid, scenario in owned.items():
        local_ids = _validate_scenario(scenario, gold[sid])
        for all_ids, ids in zip(global_ids, local_ids):
            _require(not all_ids.intersection(ids), sid, "fixture, query, and step IDs must be unique across scenarios")
            all_ids.update(ids)
    tool = _object(scenarios.get("fixture_tool_spec"), "fixture_tool_spec")
    _require(tool.get("name") == "lookup_fixture", "fixture_tool_spec", "unexpected tool name")
    schema = _object(tool.get("input_schema"), "fixture_tool_spec.input_schema")
    _require(
        schema.get("type") == "object" and schema.get("additionalProperties") is False,
        "fixture_tool_spec",
        "strict object schema required",
    )
    _require(schema.get("required") == ["query_key"], "fixture_tool_spec", "query_key must be required")
    properties = _object(schema.get("properties"), "fixture_tool_spec.properties")
    _require(set(properties) == {"query_key"}, "fixture_tool_spec", "only query_key is supported")
    query_schema = _object(properties["query_key"], "fixture_tool_spec.query_key")
    _require(query_schema.get("type") == "string", "fixture_tool_spec", "query_key must be a string")
    enum = _ids(query_schema.get("enum"), "fixture_tool_spec.enum")
    _require(set(enum) == global_ids[1], "fixture_tool_spec", "query enum must match exactly the fixture bindings")
    _validate_manifest(pack["manifest"], set(owned))
    return pack
