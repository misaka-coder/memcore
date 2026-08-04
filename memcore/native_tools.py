"""Provider-native tool helpers for host chat integrations.

The helpers in this module do not call a chat model and do not store files.
They provide JSON-schema tool definitions plus a strict dispatcher that host
apps can wire into their model provider's native tool-calling loop.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schema import ABOUT_ROLES, MEMORY_FACETS

SOURCE_LAYERS: tuple[str, ...] = ("raw", "summary", "semantic_summary")
MATERIAL_SOURCE_PREFERENCES: tuple[str, ...] = ("auto", "original", "derived")
ENTRY_DETAILS: tuple[str, ...] = ("full", "compact")
TIMELINE_PROJECTIONS: tuple[str, ...] = ("conversation", "full", "tools")
NATIVE_MEMORY_TOOL_NAMES: tuple[str, ...] = (
    "retrieve_for_turn",
    "read_timeline",
    "read_entry",
    "load_material",
)

MaterialLoader = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolDispatchPolicy:
    """Host-owned permissions; model arguments can only narrow this grant."""

    allow_explicit_trace: bool = False
    allowed_kind_prefixes: tuple[str, ...] = ()


def build_native_memory_tool_specs(
    *,
    include_material_tool: bool = True,
    tool_format: str = "openai",
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Build provider-native tool declarations for memcore memory tools.

    Supported formats:
    - "openai": Chat Completions-compatible {"type":"function","function":...}
    - "openai_responses": Responses-compatible {"type":"function","name":...}
    - "anthropic": {"name","description","input_schema"}
    - "plain": {"name","description","parameters"}
    """

    tools = [
        _tool_spec("retrieve_for_turn", _retrieve_description(), _retrieve_schema()),
        _tool_spec("read_timeline", _timeline_description(), _timeline_schema()),
        _tool_spec("read_entry", _entry_description(), _entry_schema()),
    ]
    if include_material_tool:
        tools.append(_tool_spec("load_material", _material_description(), _material_schema()))
    return [_format_tool(tool, tool_format=tool_format, strict=strict) for tool in tools]


def dispatch_native_memory_tool(
    tool_name: str,
    arguments: Any,
    *,
    mem: Any,
    current: dict[str, Any] | None = None,
    material_loader: MaterialLoader | None = None,
    policy: ToolDispatchPolicy | None = None,
) -> dict[str, Any]:
    """Dispatch a native tool call to memcore or a host material loader.

    Invalid filters return structured errors instead of being relaxed into broad
    searches. ``material_loader`` is host-provided and is responsible for reading
    original files, OCR text, image descriptions, document chunks, or status.
    """

    name = str(tool_name or "").strip()
    args, error = _coerce_arguments(arguments)
    if error:
        return _err(name, "invalid_arguments", error)
    if name == "retrieve_for_turn":
        return _dispatch_retrieve(args, mem=mem, current=current, policy=policy or ToolDispatchPolicy())
    if name == "read_timeline":
        return _dispatch_timeline(args, mem=mem)
    if name == "read_entry":
        return _dispatch_entry(args, mem=mem)
    if name == "load_material":
        return _dispatch_material(args, material_loader=material_loader)
    return _err(name, "unknown_tool", f"unsupported_tool:{name}")


def _dispatch_retrieve(
    args: dict[str, Any],
    *,
    mem: Any,
    current: dict[str, Any] | None,
    policy: ToolDispatchPolicy,
) -> dict[str, Any]:
    unknown = _unknown_keys(
        args,
        {
            "query",
            "entity_anchors",
            "topic_terms",
            "source_layers",
            "memory_facets",
            "about_roles",
            "time_hint",
            "include_explicit",
            "kind_patterns",
        },
    )
    if unknown:
        return _err("retrieve_for_turn", "invalid_arguments", f"unknown_arguments:{unknown}")
    query = _required_string(args, "query")
    if not query:
        return _err("retrieve_for_turn", "invalid_arguments", "query_required")
    if not isinstance(current, dict) or not str(current.get("source_id") or "").strip():
        return _err("retrieve_for_turn", "invalid_state", "current_turn_required")

    entity_anchors, error = _string_list(args.get("entity_anchors"), "entity_anchors")
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    topic_terms, error = _string_list(args.get("topic_terms"), "topic_terms")
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    source_layers, error = _enum_list(args.get("source_layers"), "source_layers", SOURCE_LAYERS)
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    memory_facets, error = _enum_list(args.get("memory_facets"), "memory_facets", MEMORY_FACETS)
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    about_roles, error = _enum_list(args.get("about_roles"), "about_roles", ABOUT_ROLES)
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    time_hint, error = _time_hint(args.get("time_hint"))
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    include_explicit = args.get("include_explicit", False)
    if not isinstance(include_explicit, bool):
        return _err("retrieve_for_turn", "invalid_arguments", "include_explicit_must_be_boolean")
    kind_patterns, error = _string_list(args.get("kind_patterns"), "kind_patterns")
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    authorized_patterns, error_status, error = _authorize_kind_patterns(
        include_explicit=include_explicit,
        kind_patterns=kind_patterns,
        policy=policy,
    )
    if error:
        return _err("retrieve_for_turn", error_status, error)

    filters: dict[str, Any] = {
        "entity_anchors": entity_anchors,
        "topic_terms": topic_terms,
        "source_layers": source_layers,
        "memory_facets": memory_facets,
        "about_roles": about_roles,
        # Search every conversation owned by this user, while the current
        # prompt's visible lineage is still excluded below. Namespace
        # ownership remains a hard filter and cannot be widened by the model.
        "cross_conversation": True,
    }
    if time_hint:
        filters["time_hint"] = time_hint
    filters["include_explicit"] = include_explicit
    filters["kind_patterns"] = authorized_patterns

    try:
        result = mem.retrieve_for_turn_structured(current=current, query=query, **filters)
    except ValueError as exc:
        return _err("retrieve_for_turn", "invalid_filter", str(exc) or "invalid_filter")
    except Exception:
        return _err("retrieve_for_turn", "failed", "internal_error")
    payload = result.to_dict() if hasattr(result, "to_dict") else {"status": "failed", "reason": "invalid_result"}
    snippets = list(getattr(result, "rendered_texts", ()) or ())
    payload["snippets"] = snippets
    payload["count"] = len(snippets)
    return _ok("retrieve_for_turn", payload)


def _dispatch_timeline(args: dict[str, Any], *, mem: Any) -> dict[str, Any]:
    unknown = _unknown_keys(
        args,
        {
            "time_range",
            "date_from",
            "date_to",
            "time_periods",
            "anchor_source_id",
            "before_turns",
            "after_turns",
            "cross_conversation",
            "projection",
            "page_token_budget",
            "cursor",
        },
    )
    if unknown:
        return _err("read_timeline", "invalid_arguments", f"unknown_arguments:{unknown}")
    time_range, error = _timeline_time_range(args.get("time_range"))
    if error:
        return _err("read_timeline", "invalid_arguments", error)
    date_from = _required_string(args, "date_from")
    date_to = _optional_string(args.get("date_to"))
    time_periods, error = _string_list(args.get("time_periods"), "time_periods")
    if error:
        return _err("read_timeline", "invalid_arguments", error)
    cross = args.get("cross_conversation", False)
    if cross is None:
        cross = False
    if not isinstance(cross, bool):
        return _err("read_timeline", "invalid_arguments", "cross_conversation_must_be_boolean")
    anchor_source_id = _optional_string(args.get("anchor_source_id"))
    projection = _optional_string(args.get("projection")) or "conversation"
    if projection not in TIMELINE_PROJECTIONS:
        return _err("read_timeline", "invalid_arguments", f"invalid_timeline_projection:{projection}")
    cursor = _optional_string(args.get("cursor"))
    try:
        before_turns = int(args.get("before_turns") or 0)
        after_turns = int(args.get("after_turns") or 0)
        if isinstance(args.get("page_token_budget"), bool):
            raise ValueError
        page_token_budget = int(args.get("page_token_budget") or 0)
    except (TypeError, ValueError):
        return _err("read_timeline", "invalid_arguments", "timeline_integer_argument_invalid")
    if page_token_budget < 0:
        return _err("read_timeline", "invalid_arguments", "page_token_budget_must_be_non_negative_integer")
    if cursor and any(
        (
            time_range is not None,
            bool(date_from),
            bool(date_to),
            bool(time_periods),
            bool(anchor_source_id),
            bool(before_turns),
            bool(after_turns),
            bool(cross),
            projection != "conversation",
            bool(page_token_budget),
        )
    ):
        return _err("read_timeline", "invalid_arguments", "cursor_options_are_embedded")
    result = mem.read_timeline(
        time_range=time_range,
        date_from=date_from,
        date_to=date_to,
        time_periods=time_periods,
        anchor_source_id=anchor_source_id,
        before_turns=before_turns,
        after_turns=after_turns,
        cross_conversation=cross,
        projection=projection,
        page_token_budget=page_token_budget,
        cursor=cursor,
    )
    if result.get("status") == "invalid_filter":
        return _err("read_timeline", "invalid_filter", str(result.get("reason") or "invalid_filter"), result=result)
    return _ok("read_timeline", result)


def _dispatch_entry(args: dict[str, Any], *, mem: Any) -> dict[str, Any]:
    unknown = _unknown_keys(args, {"source_id", "detail"})
    if unknown:
        return _err("read_entry", "invalid_arguments", f"unknown_arguments:{unknown}")
    source_id = _required_string(args, "source_id")
    if not source_id:
        return _err("read_entry", "invalid_arguments", "source_id_required")
    detail = _optional_string(args.get("detail")) or "full"
    if detail not in ENTRY_DETAILS:
        return _err("read_entry", "invalid_arguments", f"invalid_entry_detail:{detail}")
    result = mem.read_entry(source_id=source_id, detail=detail)
    if result.get("status") == "invalid_filter":
        return _err("read_entry", "invalid_filter", str(result.get("reason") or "invalid_filter"), result=result)
    return _ok("read_entry", result)


def _dispatch_material(args: dict[str, Any], *, material_loader: MaterialLoader | None) -> dict[str, Any]:
    unknown = _unknown_keys(args, {"file_id", "kind", "preferred_source", "purpose"})
    if unknown:
        return _err("load_material", "invalid_arguments", f"unknown_arguments:{unknown}")
    file_id = _required_string(args, "file_id")
    if not file_id:
        return _err("load_material", "invalid_arguments", "file_id_required")
    preferred = _optional_string(args.get("preferred_source")) or "auto"
    if preferred not in MATERIAL_SOURCE_PREFERENCES:
        return _err("load_material", "invalid_arguments", f"invalid_preferred_source:{preferred}")
    payload = {
        "file_id": file_id,
        "kind": _optional_string(args.get("kind")),
        "preferred_source": preferred,
        "purpose": _optional_string(args.get("purpose")),
    }
    if material_loader is None:
        return _err("load_material", "unavailable", "material_loader_not_configured")
    try:
        result = material_loader(payload)
    except Exception as exc:  # pragma: no cover - exact host failures vary
        return _err("load_material", "loader_failed", str(exc) or exc.__class__.__name__)
    material_payload, error = _material_result_payload(result, requested_file_id=file_id)
    if error:
        return _err("load_material", "file_id_mismatch", error, result=material_payload)
    return _ok("load_material", material_payload)


def _coerce_arguments(arguments: Any) -> tuple[dict[str, Any], str]:
    if isinstance(arguments, dict):
        return arguments, ""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {}, "arguments_must_be_json_object"
        if isinstance(parsed, dict):
            return parsed, ""
    return {}, "arguments_must_be_object"


def _material_result_payload(result: Any, *, requested_file_id: str) -> tuple[dict[str, Any], str]:
    if not isinstance(result, dict):
        return {"requested_file_id": requested_file_id, "content": result}, ""
    payload = dict(result)
    returned_file_id = str(payload.get("file_id") or "").strip()
    if returned_file_id and returned_file_id != requested_file_id:
        return payload, f"material_file_id_mismatch:requested={requested_file_id},returned={returned_file_id}"
    payload.setdefault("requested_file_id", requested_file_id)
    return payload, ""


def _required_string(args: dict[str, Any], key: str) -> str:
    return str(args.get(key) or "").strip()


def _authorize_kind_patterns(
    *,
    include_explicit: bool,
    kind_patterns: list[str],
    policy: ToolDispatchPolicy,
) -> tuple[list[str], str, str]:
    if not include_explicit:
        return ([], "invalid_arguments", "kind_patterns_require_include_explicit") if kind_patterns else ([], "", "")
    if not kind_patterns:
        return [], "invalid_arguments", "explicit_kind_patterns_required"
    if not policy.allow_explicit_trace:
        return [], "forbidden", "explicit_trace_not_authorized"
    allowed = [str(item or "").strip().lower().rstrip(".*") for item in policy.allowed_kind_prefixes]
    allowed = [item for item in allowed if item]
    normalized: list[str] = []
    for raw in kind_patterns:
        pattern = str(raw or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*(?:\.\*)?", pattern):
            return [], "invalid_arguments", "invalid_kind_pattern"
        base = pattern[:-2] if pattern.endswith(".*") else pattern
        if not any(base == prefix or base.startswith(prefix + ".") for prefix in allowed):
            return [], "forbidden", "kind_pattern_not_authorized"
        if pattern not in normalized:
            normalized.append(pattern)
    return normalized[:8], "", ""


def _optional_string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any, name: str) -> tuple[list[str], str]:
    if value is None:
        return [], ""
    if not isinstance(value, list):
        return [], f"{name}_must_be_array"
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out, ""


def _enum_list(value: Any, name: str, allowed: tuple[str, ...]) -> tuple[list[str], str]:
    values, error = _string_list(value, name)
    if error:
        return [], error
    allowed_set = set(allowed)
    invalid = [item for item in values if item not in allowed_set]
    if invalid:
        return [], f"invalid_{name}:{invalid}"
    return values, ""


def _time_hint(value: Any) -> tuple[dict[str, Any], str]:
    if value is None:
        return {}, ""
    if not isinstance(value, dict):
        return {}, "time_hint_must_be_object"
    unknown = _unknown_keys(value, {"date_label", "time_of_day", "start_ts", "end_ts"})
    if unknown:
        return {}, f"unknown_time_hint_keys:{unknown}"
    out: dict[str, Any] = {}
    for key in ("date_label", "time_of_day"):
        text = str(value.get(key) or "").strip()
        if text:
            out[key] = text
    for key in ("start_ts", "end_ts"):
        if value.get(key) is not None:
            try:
                out[key] = int(value[key])
            except (TypeError, ValueError):
                return {}, f"{key}_must_be_int"
    if out.get("start_ts") is not None and out.get("end_ts") is not None and out["start_ts"] > out["end_ts"]:
        return {}, "time_hint_start_after_end"
    return out, ""


def _timeline_time_range(value: Any) -> tuple[dict[str, str] | None, str]:
    if value is None:
        return None, ""
    if not isinstance(value, dict):
        return None, "time_range_must_be_object"
    unknown = _unknown_keys(value, {"start_at", "end_at"})
    if unknown:
        return None, f"unknown_time_range_keys:{unknown}"
    out: dict[str, str] = {}
    for key in ("start_at", "end_at"):
        raw = value.get(key)
        if raw is None:
            continue
        if not isinstance(raw, str):
            return None, f"time_range_{key}_must_be_string"
        out[key] = raw.strip()
    return out, ""


def _unknown_keys(args: dict[str, Any], allowed: set[str]) -> list[str]:
    return sorted(str(key) for key in args if str(key) not in allowed)


def _ok(tool_name: str, result: Any) -> dict[str, Any]:
    return {"ok": True, "status": "ok", "tool_name": tool_name, "result": result}


def _err(tool_name: str, status: str, reason: str, *, result: Any | None = None) -> dict[str, Any]:
    payload = {"ok": False, "status": status, "tool_name": tool_name, "reason": reason}
    if result is not None:
        payload["result"] = result
    return payload


def _tool_spec(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": parameters}


def _format_tool(tool: dict[str, Any], *, tool_format: str, strict: bool) -> dict[str, Any]:
    fmt = str(tool_format or "").strip().lower()
    if fmt == "plain":
        return dict(tool)
    if fmt == "openai":
        function = dict(tool)
        if strict:
            function["parameters"] = _openai_strict_schema(tool["parameters"])
        function["strict"] = bool(strict)
        return {"type": "function", "function": function}
    if fmt == "openai_responses":
        formatted = dict(tool)
        if strict:
            formatted["parameters"] = _openai_strict_schema(tool["parameters"])
        return {"type": "function", **formatted, "strict": bool(strict)}
    if fmt == "anthropic":
        return {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
    raise ValueError(f"unsupported tool_format: {tool_format!r}")


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _strictify_schema(schema, nullable=False)


def _strictify_schema(schema: dict[str, Any], *, nullable: bool) -> dict[str, Any]:
    out = dict(schema)
    if "properties" in out:
        original_required = {str(item) for item in out.get("required", [])}
        properties = dict(out.get("properties") or {})
        out["properties"] = {
            name: _strictify_schema(prop, nullable=name not in original_required)
            for name, prop in properties.items()
            if isinstance(prop, dict)
        }
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if isinstance(out.get("items"), dict):
        out["items"] = _strictify_schema(out["items"], nullable=False)
    if nullable:
        out["type"] = _nullable_type(out.get("type"))
        if isinstance(out.get("enum"), list) and None not in out["enum"]:
            out["enum"] = [*out["enum"], None]
    return out


def _nullable_type(value: Any) -> Any:
    if isinstance(value, list):
        return value if "null" in value else [*value, "null"]
    if isinstance(value, str):
        return [value, "null"] if value != "null" else value
    return ["null"]


def _retrieve_description() -> str:
    return (
        "Fuzzy memory search for preferences, plans, people, old facts, relationships, "
        "promises, and material/tool trace anchors. Excludes visible context for the current turn. "
        "Use include_explicit with a precise kind_patterns value only when tool, event, skill, or material records are needed."
    )


def _timeline_description() -> str:
    return (
        "Exact raw timeline lookup. When concrete hours or minutes are known, pass time_range.start_at/end_at "
        "as ISO 8601 or local date-time strings; no Unix timestamp calculation is needed. Legacy date fields remain "
        "available for whole-day or coarse-period reads. Conversation view keeps dialogue/events full and returns "
        "reloadable compact evidence for large operation/material records. If coverage is incomplete, call again with "
        "only next_cursor. A raw retrieval source id can anchor complete nearby turns."
    )


def _entry_description() -> str:
    return (
        "Expand one raw timeline source_id in the current authorized conversation. Use this when read_timeline "
        "returned a compact tool, operation, skill, or material evidence block and its full stored content is needed."
    )


def _material_description() -> str:
    return (
        "Load current host-managed material content or status by file_id. The host may return original file data, "
        "OCR text, image descriptions, document chunks, or an expired/unavailable status."
    )


def _retrieve_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Natural-language memory query."},
            "entity_anchors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only exact entities already known from the question/context. Never guess an unknown answer entity; omit when unsure.",
            },
            "topic_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Actions, properties, or supporting topic terms; these do not replace query.",
            },
            "source_layers": {"type": "array", "items": {"type": "string", "enum": list(SOURCE_LAYERS)}},
            "memory_facets": {
                "type": "array",
                "items": {"type": "string", "enum": list(MEMORY_FACETS)},
                "description": "What kind of historical answer is needed. Omit rather than guess.",
            },
            "about_roles": {
                "type": "array",
                "items": {"type": "string", "enum": list(ABOUT_ROLES)},
                "description": "Who or what the target historical content is about, not who spoke.",
            },
            "time_hint": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "date_label": {"type": "string", "description": "YYYY-MM-DD date label."},
                    "time_of_day": {"type": "string", "description": "morning/afternoon/night/midnight if known."},
                    "start_ts": {"type": "integer"},
                    "end_ts": {"type": "integer"},
                },
            },
            "include_explicit": {
                "type": "boolean",
                "description": "Whether this query intentionally needs explicit trace/event/material records.",
            },
            "kind_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact kinds or trailing-wildcard prefixes authorized by the host.",
            },
        },
    }


def _timeline_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {
            "time_range": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start_at", "end_at"],
                "description": "Exact start-inclusive/end-exclusive range. Omit unknown answer entities; read the raw range instead.",
                "properties": {
                    "start_at": {
                        "type": "string",
                        "description": "ISO 8601 or local YYYY-MM-DD HH:MM[:SS].",
                    },
                    "end_at": {
                        "type": "string",
                        "description": "Exclusive ISO 8601 or local YYYY-MM-DD HH:MM[:SS].",
                    },
                },
            },
            "date_from": {"type": "string", "description": "YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "YYYY-MM-DD; omit for a single day."},
            "time_periods": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional localized periods such as morning/afternoon/night.",
            },
            "anchor_source_id": {
                "type": "string",
                "description": "Raw source_id returned by retrieve_for_turn; mutually exclusive with date fields.",
            },
            "before_turns": {"type": "integer", "minimum": 0},
            "after_turns": {"type": "integer", "minimum": 0},
            "cross_conversation": {"type": "boolean", "description": "Only true if host policy allows it."},
            "projection": {
                "type": "string",
                "enum": list(TIMELINE_PROJECTIONS),
                "description": "conversation (default), full, or tools.",
            },
            "page_token_budget": {
                "type": "integer",
                "minimum": 0,
                "description": "Explicit caller budget only; omit/0 means no MemCore pagination or hidden truncation.",
            },
            "cursor": {
                "type": "string",
                "description": "Opaque next_cursor from a prior incomplete result. Send it alone; selector/view/budget are embedded.",
            },
        },
    }


def _entry_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_id"],
        "properties": {
            "source_id": {"type": "string", "description": "Raw source_id from timeline/retrieval evidence."},
            "detail": {"type": "string", "enum": list(ENTRY_DETAILS)},
        },
    }


def _material_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["file_id"],
        "properties": {
            "file_id": {"type": "string", "description": "Host material/file identifier from material_trace."},
            "kind": {"type": "string", "description": "Optional material kind such as image/pdf/file."},
            "preferred_source": {"type": "string", "enum": list(MATERIAL_SOURCE_PREFERENCES)},
            "purpose": {"type": "string", "description": "Why the model needs the material."},
        },
    }
