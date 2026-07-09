"""Provider-native tool helpers for host chat integrations.

The helpers in this module do not call a chat model and do not store files.
They provide JSON-schema tool definitions plus a strict dispatcher that host
apps can wire into their model provider's native tool-calling loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from .schema import DEFAULT_CATEGORIES, SUBJECT_SCOPES

SOURCE_LAYERS: tuple[str, ...] = ("raw", "summary", "semantic_summary")
MATERIAL_SOURCE_PREFERENCES: tuple[str, ...] = ("auto", "original", "derived")
NATIVE_MEMORY_TOOL_NAMES: tuple[str, ...] = ("retrieve_for_turn", "read_timeline", "load_material")

MaterialLoader = Callable[[dict[str, Any]], Any]


def build_native_memory_tool_specs(
    *,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
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

    cats = _clean_categories(categories)
    tools = [
        _tool_spec("retrieve_for_turn", _retrieve_description(), _retrieve_schema(cats)),
        _tool_spec("read_timeline", _timeline_description(), _timeline_schema()),
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
        return _dispatch_retrieve(args, mem=mem, current=current)
    if name == "read_timeline":
        return _dispatch_timeline(args, mem=mem)
    if name == "load_material":
        return _dispatch_material(args, material_loader=material_loader)
    return _err(name, "unknown_tool", f"unsupported_tool:{name}")


def _dispatch_retrieve(args: dict[str, Any], *, mem: Any, current: dict[str, Any] | None) -> dict[str, Any]:
    unknown = _unknown_keys(
        args,
        {
            "query",
            "keywords",
            "source_layers",
            "categories",
            "subject_scopes",
            "importance_min",
            "time_hint",
        },
    )
    if unknown:
        return _err("retrieve_for_turn", "invalid_arguments", f"unknown_arguments:{unknown}")
    query = _required_string(args, "query")
    if not query:
        return _err("retrieve_for_turn", "invalid_arguments", "query_required")
    if not isinstance(current, dict) or not str(current.get("source_id") or "").strip():
        return _err("retrieve_for_turn", "invalid_state", "current_turn_required")

    keywords, error = _string_list(args.get("keywords"), "keywords")
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    source_layers, error = _enum_list(args.get("source_layers"), "source_layers", SOURCE_LAYERS)
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    categories, error = _enum_list(args.get("categories"), "categories", tuple(mem.config.categories))
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    subject_scopes, error = _enum_list(args.get("subject_scopes"), "subject_scopes", SUBJECT_SCOPES)
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    importance_min, error = _importance_min(args.get("importance_min"))
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)
    time_hint, error = _time_hint(args.get("time_hint"))
    if error:
        return _err("retrieve_for_turn", "invalid_arguments", error)

    filters: dict[str, Any] = {
        "keywords": keywords,
        "source_layers": source_layers,
        "categories": categories,
        "subject_scopes": subject_scopes,
    }
    if importance_min is not None:
        filters["importance_min"] = importance_min
    if time_hint:
        filters["time_hint"] = time_hint

    result = mem.retrieve_for_turn(current=current, query=query, **filters)
    return _ok("retrieve_for_turn", {"snippets": result, "count": len(result)})


def _dispatch_timeline(args: dict[str, Any], *, mem: Any) -> dict[str, Any]:
    unknown = _unknown_keys(args, {"date_from", "date_to", "time_periods", "cross_conversation"})
    if unknown:
        return _err("read_timeline", "invalid_arguments", f"unknown_arguments:{unknown}")
    date_from = _required_string(args, "date_from")
    if not date_from:
        return _err("read_timeline", "invalid_arguments", "date_from_required")
    date_to = _optional_string(args.get("date_to"))
    time_periods, error = _string_list(args.get("time_periods"), "time_periods")
    if error:
        return _err("read_timeline", "invalid_arguments", error)
    cross = args.get("cross_conversation", False)
    if not isinstance(cross, bool):
        return _err("read_timeline", "invalid_arguments", "cross_conversation_must_be_boolean")
    result = mem.read_timeline(
        date_from=date_from,
        date_to=date_to,
        time_periods=time_periods,
        cross_conversation=cross,
    )
    if result.get("status") == "invalid_filter":
        return _err("read_timeline", "invalid_filter", str(result.get("reason") or "invalid_filter"), result=result)
    return _ok("read_timeline", result)


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
    return _ok("load_material", result if isinstance(result, dict) else {"content": result})


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


def _required_string(args: dict[str, Any], key: str) -> str:
    return str(args.get(key) or "").strip()


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


def _importance_min(value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "importance_min_must_be_number"
    if not 0.0 <= number <= 1.0:
        return None, "importance_min_must_be_between_0_and_1"
    return number, ""


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


def _unknown_keys(args: dict[str, Any], allowed: set[str]) -> list[str]:
    return sorted(str(key) for key in args if str(key) not in allowed)


def _ok(tool_name: str, result: Any) -> dict[str, Any]:
    return {"ok": True, "status": "ok", "tool_name": tool_name, "result": result}


def _err(tool_name: str, status: str, reason: str, *, result: Any | None = None) -> dict[str, Any]:
    payload = {"ok": False, "status": status, "tool_name": tool_name, "reason": reason}
    if result is not None:
        payload["result"] = result
    return payload


def _clean_categories(categories: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in categories:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


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
        "promises, and material/tool trace anchors. Excludes visible context for the current turn."
    )


def _timeline_description() -> str:
    return "Exact raw timeline lookup for dates, date ranges, relative-time resolutions, and attribution questions."


def _material_description() -> str:
    return (
        "Load current host-managed material content or status by file_id. The host may return original file data, "
        "OCR text, image descriptions, document chunks, or an expired/unavailable status."
    )


def _retrieve_schema(categories: list[str]) -> dict[str, Any]:
    category_items: dict[str, Any] = {"type": "string"}
    if categories:
        category_items["enum"] = categories
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Natural-language memory query."},
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "Reusable recall tags."},
            "source_layers": {"type": "array", "items": {"type": "string", "enum": list(SOURCE_LAYERS)}},
            "categories": {"type": "array", "items": category_items},
            "subject_scopes": {"type": "array", "items": {"type": "string", "enum": list(SUBJECT_SCOPES)}},
            "importance_min": {"type": "number", "minimum": 0.0, "maximum": 1.0},
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
        },
    }


def _timeline_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["date_from"],
        "properties": {
            "date_from": {"type": "string", "description": "YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "YYYY-MM-DD; omit for a single day."},
            "time_periods": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional localized periods such as morning/afternoon/night.",
            },
            "cross_conversation": {"type": "boolean", "description": "Only true if host policy allows it."},
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
