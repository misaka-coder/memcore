"""Compact, deterministic receipts for model-facing memory operations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .projection import sanitize_timeline_value, stable_projection_hash

_COVERAGE_KEYS = (
    "complete",
    "coverage_complete",
    "requested_start",
    "requested_end",
    "returned_start",
    "returned_end",
    "selected_logical_unit_count",
    "returned_logical_unit_count",
    "remaining_logical_unit_count",
    "selected_entry_count",
    "returned_entry_count",
    "remaining_entry_count",
    "selected_projected_token_count",
    "returned_projected_token_count",
    "remaining_projected_token_count",
    "token_count_quality",
    "page_token_budget",
    "oversized_unit",
    "total_source_count",
    "covered_source_count",
    "live_source_count",
    "gap_source_count",
)


def build_memory_operation_receipt(
    tool_name: str,
    arguments: Any,
    dispatch_result: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Build the small cross-round record for one memory-tool operation.

    This receipt contains navigation IDs, coverage and hashes only. Hosts may
    retain it beside the complete model-visible observation as a compact
    reload anchor that survives operation compaction; it never replaces the
    observation body.
    """

    operation = str(tool_name or "").strip()
    parsed_arguments = _parse_arguments(arguments)
    safe_arguments, _ = sanitize_timeline_value(parsed_arguments)
    raw_dispatch = dict(dispatch_result) if isinstance(dispatch_result, Mapping) else {"result": dispatch_result}
    safe_dispatch, _ = sanitize_timeline_value(raw_dispatch)
    safe_dispatch = dict(safe_dispatch) if isinstance(safe_dispatch, Mapping) else {"result": safe_dispatch}

    result = raw_dispatch.get("result") if isinstance(raw_dispatch.get("result"), Mapping) else {}
    result = dict(result)
    status = str(result.get("status") or raw_dispatch.get("status") or "failed")
    reason = str(result.get("reason") or raw_dispatch.get("reason") or "")
    coverage = _compact_coverage(result)
    next_cursor = _next_cursor(result)

    receipt: dict[str, Any] = {
        "receipt_schema_version": 1,
        "operation": operation,
        "status": status,
        "reason": reason,
        "selector": _selector_for(operation, parsed_arguments, result),
        "returned_memory_ids": _returned_memory_ids(operation, result),
        "returned_source_ids": _returned_source_ids(operation, result),
        "returned_logical_unit_ids": _returned_logical_unit_ids(operation, result),
        "coverage": coverage,
        "next_cursor": next_cursor,
        "request_hash": stable_projection_hash({"operation": operation, "arguments": safe_arguments}),
        "result_hash": stable_projection_hash(safe_dispatch),
    }
    safe_receipt, _ = sanitize_timeline_value(receipt)
    return dict(safe_receipt)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _compact_coverage(result: Mapping[str, Any]) -> dict[str, Any]:
    source = result.get("coverage") if isinstance(result.get("coverage"), Mapping) else {}
    compact = {key: source[key] for key in _COVERAGE_KEYS if key in source}
    requested_range = source.get("requested_range")
    if isinstance(requested_range, Mapping):
        compact["requested_range"] = {
            key: requested_range[key] for key in ("start_ts", "end_ts", "start_at", "end_at") if key in requested_range
        }
    if "page_complete" in result:
        compact["page_complete"] = bool(result.get("page_complete"))
    return compact


def _selector_for(operation: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    if operation == "read_timeline":
        selector = {
            key: result[key]
            for key in ("selector_mode", "time_range", "date_from", "date_to", "time_periods", "anchor_source_id")
            if key in result
        }
        selector["projection"] = str(result.get("projection") or arguments.get("projection") or "conversation")
        return selector
    if operation == "browse_memory":
        coverage = result.get("coverage") if isinstance(result.get("coverage"), Mapping) else {}
        return {
            "requested_range": dict(coverage.get("requested_range") or {}),
            "node_types": list(result.get("node_types") or []),
            "cross_conversation": bool(result.get("cross_conversation")),
        }
    if operation in {"open_memory", "read_entry"}:
        return {
            "memory_id": str(result.get("memory_id") or arguments.get("memory_id") or arguments.get("source_id") or ""),
            "memory_ids": list(result.get("memory_ids") or arguments.get("memory_ids") or []),
            "view": str(result.get("view") or arguments.get("view") or "content"),
            "detail": str(result.get("detail") or arguments.get("detail") or "full"),
            "projection": str(result.get("projection") or arguments.get("projection") or "conversation"),
        }
    if operation == "retrieve_for_turn":
        return {
            "within_memory_id": str(arguments.get("within_memory_id") or ""),
            "time_hint": dict(arguments.get("time_hint") or {})
            if isinstance(arguments.get("time_hint"), Mapping)
            else {},
            "source_layers": list(arguments.get("source_layers") or []),
            "include_explicit": bool(arguments.get("include_explicit")),
        }
    if operation == "load_material":
        return {
            "file_id": str(arguments.get("file_id") or ""),
            "kind": str(arguments.get("kind") or ""),
            "preferred_source": str(arguments.get("preferred_source") or "auto"),
        }
    return {}


def _returned_memory_ids(operation: str, result: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    if operation == "browse_memory":
        values.extend(card.get("memory_id") for card in _mapping_items(result.get("cards")))
    elif operation in {"open_memory", "read_entry"}:
        if result.get("memory_id"):
            values.append(result.get("memory_id"))
        payload = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        values.extend(
            item.get("memory_id")
            for item in _mapping_items(payload.get("items"))
            if str(item.get("status") or "") == "ok"
        )
        values.extend(card.get("memory_id") for card in _mapping_items(payload.get("sources")))
    return _unique_strings(values)


def _returned_source_ids(operation: str, result: Mapping[str, Any]) -> list[str]:
    if operation != "retrieve_for_turn":
        return []
    values: list[Any] = []
    for match in _mapping_items(result.get("matches")):
        values.extend(match.get("source_ids") or [])
        values.append(match.get("source_id"))
    return _unique_strings(values)


def _returned_logical_unit_ids(operation: str, result: Mapping[str, Any]) -> list[str]:
    if operation == "read_timeline":
        coverage = result.get("coverage") if isinstance(result.get("coverage"), Mapping) else {}
        return _unique_strings(coverage.get("returned_logical_unit_ids") or [])
    if operation == "open_memory":
        payload = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        returned_ids = payload.get("returned_logical_unit_ids")
        if returned_ids:
            return _unique_strings(returned_ids)
        return _unique_strings(unit.get("unit_id") for unit in _mapping_items(payload.get("source_units")))
    return []


def _next_cursor(result: Mapping[str, Any]) -> str:
    direct = str(result.get("next_cursor") or "")
    if direct:
        return direct
    coverage = result.get("coverage") if isinstance(result.get("coverage"), Mapping) else {}
    nested = str(coverage.get("next_cursor") or "")
    if nested:
        return nested
    payload = result.get("result") if isinstance(result.get("result"), Mapping) else {}
    return str(payload.get("next_cursor") or "")


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _unique_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


__all__ = ["build_memory_operation_receipt"]
