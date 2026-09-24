"""Offline evidence extraction for the pilot's mandatory manual evaluation.

This module never grades correctness. Literal matches can be negated, assigned
to the wrong field/version, or paraphrased, and therefore remain review aids.
Pass one scenario, its evaluator-only gold case, and one condition's trace.
Do not put this module's output or the answer key in tested model context.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any, Iterator


_COMPLETED = {"completed", "already_completed", "ok", "success", "passed"}
_FIELD_PATTERNS = {
    "entry_code": r"入场核验码[：:]\s*([A-Z0-9-]+)",
    "key_return_location": r"钥匙归还位置[：:]\s*([^。\n]+)",
    "backoff_seconds": r"backoff_seconds\s*=\s*(\d+)",
    "jitter_upper_bound_seconds": r"jitter_seconds\s*=\s*(\d+)",
}
_MISSING_SUBJECTS = {
    "gallery_history": r"停车(?:费|计费|收费|费用)",
    "notifier_history": r"(?:HTTP\s*)?连接超时|HTTP\s*connect(?:ion)?\s*timeout",
}
_EXPLICIT_ABSENCE = (
    r"(?:未|没有|没)(?:提供|给出|注明|列出|写明|记载|说明|提及|提到)"
    r"|(?:没有|没|缺少|不含|不包含).{0,16}(?:信息|数据|金额|标准|说明|参数|字段|记录)"
    r"|(?:资料|文档|正文|配置|记录).{0,12}(?:没有|不含|未写)"
)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def _matches(text: str, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, bool) or value is None:
        return []
    token = str(value)
    if not token:
        return []
    pattern = re.escape(token)
    if isinstance(value, (int, float)):
        # Chinese unit words must remain legal neighbours, but 17 != 117/17.5.
        pattern = rf"(?<![\d.]){pattern}(?![\d.])"
    return [
        {
            "start": match.start(),
            "end": match.end(),
            "excerpt": text[max(0, match.start() - 45) : match.end() + 65],
        }
        for match in re.finditer(pattern, text)
    ]


def _field_candidates(text: str, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        field: {"expected": value, "literal_present": bool(hits), "matches": hits}
        for field, value in expected.items()
        if not isinstance(value, bool)
        for hits in [_matches(text, value)]
    }


def _assistant_outputs(setup: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []

    def append_strings(common: dict[str, Any], value: Any, path: str) -> None:
        for leaf_path, text in _strings(value, path):
            outputs.append({**common, "path": leaf_path, "text": text})
            # Provider content and function arguments can be JSON with escaped
            # Unicode. Audit decoded authored values as well as their wire text.
            try:
                decoded = json.loads(text)
            except (ValueError, TypeError):
                continue
            if isinstance(decoded, (dict, list)):
                for decoded_path, decoded_text in _strings(decoded, f"{leaf_path}.decoded"):
                    outputs.append({**common, "path": decoded_path, "text": decoded_text})

    for setup_index, step in enumerate(setup):
        common = {"setup_index": setup_index, "step_id": step.get("step_id")}
        if isinstance(step.get("speech"), str):
            outputs.append({**common, "path": "speech", "text": step["speech"]})
        for message_index, message in enumerate(step.get("assistant_messages", [])):
            if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
                continue
            # Include retained reasoning, content blocks, metadata JSON, and
            # authored function arguments, not only the final speech string.
            append_strings(common, message, f"assistant_messages[{message_index}]")
        for call_index, call in enumerate(step.get("tool_calls", [])):
            append_strings(common, call.get("arguments"), f"tool_calls[{call_index}].arguments")
    return outputs


def _exposures(outputs: list[dict[str, Any]], values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**{key: item[key] for key in ("setup_index", "step_id", "path")}, "field": field, "value": value, **hit}
        for item in outputs
        for field, value in values.items()
        for hit in _matches(item["text"], value)
    ]


def _alternate_values(scenario: dict[str, Any], probe: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    if probe.get("kind") != "historical_tool_detail":
        return values
    for fixture in scenario["fixtures"]:
        if fixture["fixture_id"] in probe.get("expected_fixture_ids", []):
            continue
        for field, expected in probe["expected"].items():
            pattern = _FIELD_PATTERNS.get(field)
            found = re.search(pattern, fixture["result"].get("content", "")) if pattern else None
            if found and found.group(1) != str(expected):
                value = int(found.group(1)) if isinstance(expected, int) else found.group(1)
                values.append({"fixture_id": fixture["fixture_id"], "field": field, "value": value})
    return values


def _missing_field_evidence(scenario_id: str, speech: str) -> dict[str, Any]:
    subject = _MISSING_SUBJECTS.get(scenario_id)
    evidence = []
    if subject:
        for clause in re.finditer(r"[^。！？!?;；\n]+", speech):
            if re.search(subject, clause.group(), re.IGNORECASE) and re.search(_EXPLICIT_ABSENCE, clause.group()):
                evidence.append({"start": clause.start(), "end": clause.end(), "excerpt": clause.group()})
    return {
        "explicit_subject_absence_candidate": bool(evidence),
        "matches": evidence,
        "numeric_unit_mentions_for_review": [
            {"value": match.group(), "excerpt": speech[max(0, match.start() - 30) : match.end() + 40]}
            for match in re.finditer(r"\d+(?:\.\d+)?\s*(?:元|块|秒|小时|CNY|RMB)", speech, re.IGNORECASE)
        ],
        "correct": None,
    }


def _fixture_returns(scenario: dict[str, Any], setup: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixtures = {item["fixture_id"]: item for item in scenario["fixtures"]}
    observed = []
    for setup_index, step in enumerate(setup):
        for call_index, call in enumerate(step.get("tool_calls", [])):
            if call.get("name", call.get("tool")) != "lookup_fixture":
                continue
            result = _object(call.get("result"))
            fixture_id = result.get("document_id")
            fixture = fixtures.get(fixture_id)
            observed.append(
                {
                    "setup_index": setup_index,
                    "step_id": step.get("step_id"),
                    "step_status": step.get("status"),
                    "call_index": call_index,
                    "fixture_id": fixture_id,
                    "source_id": call.get("source_id"),
                    "status": result.get("status"),
                    "full_result_matches_fixture": bool(fixture and result == fixture["result"]),
                }
            )
    return observed


def _has_fixture_body(value: Any, fixture_result: dict[str, Any], depth: int = 0) -> bool:
    if depth > 8:
        return False
    if value == fixture_result:
        return True
    if isinstance(value, str):
        # Raw content is rendered as metadata + data JSON. Its output may itself
        # be the serialized exact provider fixture result.
        candidates = [value]
        if "\ndata:\n" in value:
            candidates.append(value.partition("\ndata:\n")[2])
        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if decoded != value and _has_fixture_body(decoded, fixture_result, depth + 1):
                return True
    elif isinstance(value, dict):
        return any(_has_fixture_body(item, fixture_result, depth + 1) for item in value.values())
    elif isinstance(value, list):
        return any(_has_fixture_body(item, fixture_result, depth + 1) for item in value)
    return False


def _open_nodes(result: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _object(result)
    if root.get("ok") is False:
        return root, []
    if isinstance(root.get("result"), dict) and "view" not in root:
        root = root["result"]
    items = _object(root.get("result")).get("items")
    if not isinstance(items, list):
        return root, [root] if root.get("memory_id") else []
    # Native batch results keep node status separately but render bodies once
    # in the aggregate text. Bind each section back to its own memory_id.
    sections = {}
    text = root.get("text", "")
    if isinstance(text, str):
        headers = list(re.finditer(r"(?m)^\[memory_id=(.*?) status=([^\]\n]+)\]\n", text))
        for index, header in enumerate(headers):
            stop = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            sections[header.group(1)] = text[header.end() : stop].strip()
    nodes = []
    for item in items:
        if isinstance(item, dict):
            node = dict(item)
            if not node.get("text") and node.get("memory_id") in sections:
                node["text"] = sections[node["memory_id"]]
            nodes.append(node)
    return root, nodes


def _readback_evidence(
    branch: dict[str, Any], scenario: dict[str, Any], probe: dict[str, Any], returns: list[dict[str, Any]]
) -> dict[str, Any]:
    fixtures = {item["fixture_id"]: item["result"] for item in scenario["fixtures"]}
    expected_sources: dict[str, list[str]] = {}
    for item in returns:
        if item["fixture_id"] in probe.get("expected_fixture_ids", []) and item["status"] == "ok":
            if isinstance(item["source_id"], str) and item["source_id"]:
                expected_sources.setdefault(item["source_id"], []).append(item["fixture_id"])
    calls = []
    matched, verified = set(), set()
    for index, call in enumerate(branch.get("tool_calls", [])):
        if call.get("name", call.get("tool")) != "open_memory":
            continue
        args = _object(call.get("arguments"))
        requested = args.get("memory_ids", [args.get("memory_id")])
        requested = [item for item in requested if isinstance(item, str)] if isinstance(requested, list) else []
        root, nodes = _open_nodes(call.get("result"))
        node_rows = []
        for node in nodes:
            sid = node.get("memory_id")
            relevant = expected_sources.get(sid, []) if sid in requested else []
            content_ok = bool(
                relevant
                and node.get("status") == "ok"
                and node.get("view", args.get("view", "card")) == "content"
                and node.get("detail", args.get("detail", "full")) == "full"
            )
            body_matches = [fid for fid in relevant if _has_fixture_body(node, fixtures[fid])]
            if content_ok:
                matched.add(sid)
            if content_ok and body_matches:
                verified.add(sid)
            node_rows.append(
                {
                    "memory_id": sid,
                    "status": node.get("status"),
                    "view": node.get("view", args.get("view", "card")),
                    "expected_fixture_ids": relevant,
                    "successful_expected_full_content_open": content_ok,
                    "exact_fixture_bodies_present": body_matches,
                }
            )
        calls.append(
            {
                "call_index": index,
                "requested_memory_ids": requested,
                "view": args.get("view", "card"),
                "status": root.get("status"),
                "requested_expected_source_ids": sorted(set(requested) & expected_sources.keys()),
                "nodes": node_rows,
            }
        )
    return {
        "expected_fixture_source_ids": expected_sources,
        "missing_fixture_source_mappings": sorted(
            set(probe.get("expected_fixture_ids", [])) - {fid for ids in expected_sources.values() for fid in ids}
        ),
        "open_memory_calls": calls,
        "successful_expected_content_source_ids": sorted(matched),
        "exact_fixture_body_verified_source_ids": sorted(verified),
        "other_memory_tool_calls": [
            {"call_index": index, "name": call.get("name", call.get("tool"))}
            for index, call in enumerate(branch.get("tool_calls", []))
            if call.get("name", call.get("tool")) in {"retrieve_for_turn", "read_timeline", "browse_memory"}
        ],
        "evidence_grade": None,
    }


def evaluate_trace(scenario: dict[str, Any], gold: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic review aids, preserving every expected probe.

    ``setup`` and ``branches`` are lists of step records with ``step_id``,
    ``status``, ``speech``, ``assistant_messages`` and ``tool_calls``. Each tool
    call carries ``name``, ``arguments``, ``result`` and its observation
    ``source_id``. Missing statuses are reported as unknown, never successful.
    Duplicate probe branches remain separate attempts under one planned probe.
    """
    scenario_id = scenario.get("scenario_id")
    if not scenario_id or gold.get("scenario_id") != scenario_id:
        raise ValueError("scenario and evaluator gold must identify the same scenario")
    if trace.get("scenario_id", scenario_id) != scenario_id:
        raise ValueError("trace belongs to a different scenario")
    planned = {step["step_id"] for step in scenario["steps"] if step["stage"] == "probe"}
    gold_ids = [probe["step_id"] for probe in gold["probes"]]
    if len(set(gold_ids)) != len(gold_ids) or set(gold_ids) != planned:
        raise ValueError("gold must cover every planned probe exactly once")
    setup = trace.get("setup", [])
    branches = trace.get("branches", [])
    outputs = _assistant_outputs(setup)
    returns = _fixture_returns(scenario, setup)
    probes, review = [], []
    for probe in gold["probes"]:
        exposures = _exposures(outputs, probe["expected"])
        alternates = _alternate_values(scenario, probe)
        attempts = []
        for index, branch in enumerate(branches):
            if branch.get("step_id") != probe["step_id"]:
                continue
            speech = branch.get("speech") if isinstance(branch.get("speech"), str) else ""
            attempts.append(
                {
                    "branch_index": index,
                    "status": branch.get("status", "unknown"),
                    "speech": speech,
                    "error": copy.deepcopy(branch.get("error")),
                    "candidate_fields": _field_candidates(speech, probe["expected"]),
                    "alternate_version_mentions": [
                        {**item, "matches": hits} for item in alternates if (hits := _matches(speech, item["value"]))
                    ],
                    "missing_field_evidence": (
                        _missing_field_evidence(scenario_id, speech)
                        if probe["kind"] == "explicitly_missing_field"
                        else None
                    ),
                    "readback": _readback_evidence(branch, scenario, probe, returns),
                    "correct": None,
                    "requires_review": True,
                }
            )
        row = {
            "step_id": probe["step_id"],
            "kind": probe["kind"],
            "expected": copy.deepcopy(probe["expected"]),
            "criteria": list(probe.get("criteria", [])),
            "expected_user_step_ids": list(probe.get("expected_user_step_ids", [])),
            "branch_status": attempts[0]["status"]
            if len(attempts) == 1
            else "missing"
            if not attempts
            else "duplicate",
            "attempts": attempts,
            "setup_answer_exposure_candidates": exposures,
            "setup_alternate_version_exposure_candidates": [
                {"fixture_id": item["fixture_id"], **hit}
                for item in alternates
                for hit in _exposures(outputs, {item["field"]: item["value"]})
            ],
            "correct": None,
            "evidence_grade": None,
            "requires_review": True,
        }
        probes.append(row)
        review.append(
            {
                "step_id": row["step_id"],
                "branch_status": row["branch_status"],
                "expected": copy.deepcopy(row["expected"]),
                "answers": [attempt["speech"] for attempt in attempts],
                "literal_candidate_fields": [
                    [field for field, candidate in attempt["candidate_fields"].items() if candidate["literal_present"]]
                    for attempt in attempts
                ],
                "setup_answer_exposure_candidate": bool(exposures),
                "exact_fixture_body_verified_source_ids": [
                    attempt["readback"]["exact_fixture_body_verified_source_ids"] for attempt in attempts
                ],
                "criteria": row["criteria"],
                "correct": None,
                "evidence_grade": None,
                "review_note": "",
            }
        )
    setup_counts = Counter(step.get("step_id") for step in setup)
    expected_setup = [step["step_id"] for step in scenario["steps"] if step["stage"] != "probe"]
    return {
        "format": "akane_memcore_pilot_evaluation_evidence_v1",
        "scenario_id": scenario_id,
        "run_status": trace.get("status", "unknown"),
        "requires_review": True,
        "automatic_correctness_scoring": False,
        "denominator": len(gold_ids),
        "present_probe_count": sum(bool(row["attempts"]) for row in probes),
        "completed_probe_count": sum(
            len(row["attempts"]) == 1 and row["branch_status"] in _COMPLETED and bool(row["attempts"][0]["speech"])
            for row in probes
        ),
        "correct_probe_count": None,
        "correctness_rate": None,
        "fixture_returns": returns,
        "setup_audit": {
            "scope": "Every supplied assistant message string leaf, setup speech and tool arguments; results excluded.",
            "capture_completeness": "requires_host_and_manual_verification",
            "missing_setup_step_ids": [sid for sid in expected_setup if not setup_counts[sid]],
            "duplicate_setup_step_ids": sorted(sid for sid, count in setup_counts.items() if count > 1),
            "steps_missing_assistant_messages": [
                step.get("step_id") for step in setup if "assistant_messages" not in step
            ],
            "authored_output_fragments_checked": len(outputs),
            "task_forbidden_literal_candidates": [
                {"task_step_id": check["step_id"], **hit}
                for check in gold.get("task_checks", [])
                for literal in check.get("must_not_volunteer", [])
                for hit in _exposures(outputs, {"must_not_volunteer": literal})
            ],
        },
        "unmatched_branch_indices": [
            index for index, branch in enumerate(branches) if branch.get("step_id") not in planned
        ],
        "probes": probes,
        "manual_review_table": review,
        "limitations": [
            "Literal presence does not judge meaning, negation, attribution, or correct association of values with fields.",
            "Version comparison may correctly mention v1 and v2; alternate values are review flags, not failures.",
            "No candidate exposure is not proof of no semantic/paraphrased exposure or complete provider capture.",
            "Content readback is separate from answer correctness and from whether readback was necessary.",
            "Other navigation tools require manual evidence review; source-ID matching does not grade their results.",
            "Failed, missing and duplicate probes remain in the planned denominator; no observed answer is auto-scored.",
        ],
    }
