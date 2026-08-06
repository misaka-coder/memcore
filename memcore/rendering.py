"""记忆片段渲染:把记录渲染成喂回模型的文本,带时间标签、相对时间锚点、心情行。

见设计文档 §6(检索回填带回时间标签)。渲染按显式 tz;时间范围从记录自身的
period_start_ts/period_end_ts/timestamp 取(跨 store 的 seq 解析属于 store,渲染不碰)。
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .text_utils import normalize_text
from .time_anchor import (
    TIME_PERIOD_LABELS,
    format_time_range_label,
    render_relative_time_anchor_line,
    timestamp_to_date_weekday_label,
    timestamp_to_datetime_weekday_label,
)


def _positive_ts(value: Any) -> int | None:
    """把 <=0 / 非法值当"缺失"——SQLite 默认 0 不该被渲染成 1970。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def record_time_range(record: dict[str, Any]) -> tuple[int | None, int | None]:
    start = _positive_ts(record.get("period_start_ts"))
    end = _positive_ts(record.get("period_end_ts")) or _positive_ts(record.get("timestamp"))
    if start is None and end is None:
        ts = _positive_ts(record.get("timestamp"))
        return (ts, ts)
    return (start, end)


def render_mood_line(record: dict[str, Any], *, enable_flavor: bool) -> str:
    if not enable_flavor:
        return ""
    metadata = record.get("memory_metadata") if isinstance(record.get("memory_metadata"), dict) else {}
    mood_tags = metadata.get("mood_tags") if isinstance(metadata, dict) else []
    rendered = [normalize_text(item) for item in (mood_tags or []) if normalize_text(item)]
    if not rendered:
        return ""
    return "记忆情绪:" + " / ".join(rendered[:3])


def _anchor_line(record: dict[str, Any], parts_text: list[str], time_range_label: str) -> str:
    return render_relative_time_anchor_line(
        text=" ".join(str(p or "") for p in parts_text),
        time_range_label=time_range_label,
    )


def _memory_open_hint(memory_id: str, *, down_label: str) -> str:
    encoded = json.dumps(str(memory_id), ensure_ascii=False)
    return (
        f"expand: 需要{down_label}时调用 open_memory(memory_id={encoded}, view=\"sources\")；"
        f"完整卡片用 view=\"content\"。"
    )


def render_summary_snippet(record: dict[str, Any], *, tz: str, enable_flavor: bool = False) -> str:
    start, end = record_time_range(record)
    time_label = format_time_range_label(start_ts=start, end_ts=end, tz=tz)
    labels: list[str] = []
    if time_label:
        labels.append(time_label)
    if record.get("period_label"):
        labels.append(f"阶段:{record['period_label']}")
    if record.get("event_type"):
        labels.append(f"类型:{record['event_type']}")
    prefix = f"【摘要回忆】[{' | '.join(labels)}] " if labels else "【摘要回忆】"
    parts = [f"{prefix}{record.get('diary_summary', '')}"]
    anchor = _anchor_line(
        record,
        [record.get("diary_summary"), *(record.get("key_events") or []), *(record.get("core_facts") or [])],
        time_label,
    )
    if anchor:
        parts.append(anchor)
    mood = render_mood_line(record, enable_flavor=enable_flavor)
    if mood:
        parts.append(mood)
    if record.get("key_events"):
        parts.append("关键事件:" + ";".join(str(e) for e in record["key_events"]))
    if record.get("core_facts"):
        parts.append("核心事实:" + ";".join(str(f) for f in record["core_facts"]))
    summary_id = str(record.get("summary_id") or "").strip()
    if summary_id:
        parts.append(_memory_open_hint(summary_id, down_label="该阶段原始对话"))
    return "\n".join(parts)


def render_semantic_snippet(record: dict[str, Any], *, tz: str, enable_flavor: bool = False) -> str:
    start, end = record_time_range(record)
    time_label = format_time_range_label(start_ts=start, end_ts=end, tz=tz)
    labels: list[str] = []
    if time_label:
        labels.append(time_label)
    if record.get("importance") is not None:
        labels.append(f"重要度:{float(record.get('importance') or 0.0):.2f}")
    prefix = f"【长期语义记忆】[{' | '.join(labels)}] " if labels else "【长期语义记忆】"
    parts = [f"{prefix}{record.get('semantic_summary', '')}"]
    anchor = _anchor_line(
        record,
        [
            record.get("semantic_summary"),
            *(record.get("stable_facts") or []),
            *(record.get("recurring_topics") or []),
            *(record.get("important_people") or []),
            *(record.get("open_loops") or []),
        ],
        time_label,
    )
    if anchor:
        parts.append(anchor)
    mood = render_mood_line(record, enable_flavor=enable_flavor)
    if mood:
        parts.append(mood)
    for field_key, label in (
        ("stable_facts", "稳定事实"),
        ("recurring_topics", "反复话题"),
        ("important_people", "重要人物"),
        ("open_loops", "待续线索"),
    ):
        if record.get(field_key):
            parts.append(f"{label}:" + ";".join(str(x) for x in record[field_key]))
    semantic_id = str(record.get("semantic_id") or "").strip()
    if semantic_id:
        parts.append(_memory_open_hint(semantic_id, down_label="该长期记忆的下级记忆"))
    return "\n".join(parts)


def render_timeline(messages: list[dict[str, Any]], *, tz: str) -> str:
    """时间线工具:把按日期范围读出的原始对话渲染成"按天分组、带时刻"的可读文本。"""
    return _render_grouped_raw_messages(messages, tz=tz, title="【按时间读取的原始对话(精确记录,非摘要)】")


def render_visible_raw(messages: list[dict[str, Any]], *, tz: str) -> str:
    """固定可见 raw:按日期分组,每条只渲染时刻/时间段,减少重复时间标签。"""
    return _render_grouped_raw_messages(messages, tz=tz, title="【近期原始对话(未摘要)】")


_SPEAKER_LABEL_MAX_CHARS = 80
_SPEAKER_STRUCTURAL_CHARS = re.compile(r"[:：\[\]\(\)（）{}<>]")
_WHITESPACE = re.compile(r"\s+")


def _sanitize_speaker_part(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in text)
    text = _SPEAKER_STRUCTURAL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if len(text) > _SPEAKER_LABEL_MAX_CHARS:
        text = text[:_SPEAKER_LABEL_MAX_CHARS].rstrip() + "..."
    return text


def render_speaker_label(row: dict[str, Any]) -> str:
    """渲染 role、说话人和明确对象,保留多方对话的归属与寻址。"""
    role = _sanitize_speaker_part(row.get("role"))
    actor_name = _sanitize_speaker_part(row.get("actor_display_name"))
    actor_id = _sanitize_speaker_part(row.get("actor_id"))
    if actor_name and actor_id and actor_name != actor_id:
        actor = f"{actor_name};id={actor_id}"
    else:
        actor = actor_name or actor_id
    if role and actor:
        source = f"{role}({actor})"
    else:
        source = role or actor
    target_name = _sanitize_speaker_part(row.get("target_actor_display_name"))
    target_id = _sanitize_speaker_part(row.get("target_actor_id"))
    if target_name and target_id and target_name != target_id:
        target = f"{target_name};id={target_id}"
    else:
        target = target_name or target_id
    return f"{source} -> {target}" if source and target else source or target


def _is_trace_event(row: dict[str, Any]) -> bool:
    kind_root = str(row.get("kind") or "").strip().lower().split(".", 1)[0]
    return kind_root in {"event", "tool", "material", "skill"}


def _format_tool_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_format_tool_value(item) for item in value)
    if isinstance(value, (dict, bool, int, float)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_text(value)


def _format_event_scalar(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def render_tool_use_text(
    *,
    tool_input: Any = None,
) -> str:
    """把跨轮保留的工具调用内容渲染成稳定 input 块。"""
    lines = ["input:"]
    lines.append(_format_tool_value(tool_input if tool_input is not None else {}))
    return "\n".join(lines)


def render_tool_result_text(
    *,
    result: Any,
    source: str = "",
) -> str:
    """把跨轮保留的工具结果内容渲染成稳定 source/output 块。"""
    lines: list[str] = []
    src = normalize_text(source)
    if src:
        lines.append(f"source: {src}")
    lines.append("output:")
    formatted = _format_tool_value(result)
    if formatted:
        lines.append(formatted)
    return "\n".join(lines)


_EVENT_FIELD_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_EVENT_FIELD_ORDER = {name: index for index, name in enumerate(("published_at", "title", "summary", "url"))}


def render_external_event_text(
    *,
    event_type: str,
    fields: dict[str, Any] | None = None,
    source: str = "",
) -> str:
    """Render one external event as a stable data block, never as instructions."""

    if not _format_event_scalar(event_type):
        return ""
    lines: list[str] = []
    rendered_source = _format_event_scalar(source)
    if rendered_source:
        lines.append(f"source: {rendered_source}")
    rendered_fields: list[tuple[str, str]] = []
    for raw_key, value in dict(fields or {}).items():
        key = str(raw_key or "").strip()
        if key in {"event_type", "source"} or _EVENT_FIELD_KEY.fullmatch(key) is None:
            continue
        rendered = _format_event_scalar(value)
        if rendered:
            rendered_fields.append((key, rendered))
    rendered_fields.sort(key=lambda item: (_EVENT_FIELD_ORDER.get(item[0], len(_EVENT_FIELD_ORDER)), item[0]))
    lines.extend(f"{key}: {rendered}" for key, rendered in rendered_fields)
    return "\n".join(lines)


def render_prompt_message(record: dict[str, Any], *, tz: str) -> str:
    """Render one raw record exactly as a provider history/current turn."""

    ts = record.get("timestamp")
    full_stamp = timestamp_to_datetime_weekday_label(ts, tz) if ts is not None else ""
    period = TIME_PERIOD_LABELS.get(str(record.get("time_of_day") or ""), "")
    head = " | ".join(part for part in (full_stamp, period) if part)
    speaker = render_speaker_label(record)
    lines: list[str] = []
    _append_raw_message_line(
        lines,
        head=head,
        speaker=speaker,
        content=normalize_text(record.get("content")),
        is_trace_event=_is_trace_event(record),
    )
    return "\n".join(lines)


def render_material_reference_text(
    *,
    file_id: str,
    kind: str,
    filename: str = "",
    mime_type: str = "",
    file_status: str = "",
    derived_status: str = "",
    source: str = "attachment",
) -> str:
    """把附件/文件引用渲染成模型可读的材料事件,不包含文件本体。"""
    fields = [
        ("source", source),
        ("file_id", file_id),
        ("kind", kind),
        ("filename", filename),
        ("mime", mime_type),
        ("file_status", file_status),
        ("derived_status", derived_status),
    ]
    return "\n".join(f"{key}: {text}" for key, value in fields if (text := _format_event_scalar(value)))


def render_material_cleanup_text(
    *,
    file_id: str,
    kind: str = "",
    filename: str = "",
    file_status: str = "deleted",
    derived_status: str = "",
    reason: str = "",
    source: str = "attachment_cleanup",
) -> str:
    """把材料清理事件渲染成可追踪但不承诺文件仍可读取的块。"""
    fields = [
        ("source", source),
        ("file_id", file_id),
        ("kind", kind),
        ("filename", filename),
        ("file_status", file_status),
        ("derived_status", derived_status),
        ("reason", reason),
    ]
    return "\n".join(f"{key}: {text}" for key, value in fields if (text := _format_event_scalar(value)))


def _append_raw_message_line(lines: list[str], *, head: str, speaker: str, content: str, is_trace_event: bool) -> None:
    prefix = f"[{head}] {speaker}".rstrip()
    if is_trace_event:
        lines.append(prefix)
        if content:
            lines.append(content)
        return
    lines.append(f"{prefix}: {content}".rstrip())


def _render_grouped_raw_messages(messages: list[dict[str, Any]], *, tz: str, title: str) -> str:
    if not messages:
        return ""
    lines: list[str] = [title]
    current_header = ""
    for row in messages:
        ts = row.get("timestamp")
        date_label = str(row.get("date_label") or "")
        header_label = timestamp_to_date_weekday_label(ts, tz) if ts is not None else date_label
        if header_label and header_label != current_header:
            lines.append(f"[日期 {header_label}]")
            current_header = header_label
        stamp = timestamp_to_datetime_weekday_label(ts, tz).rsplit(" ", 1)[-1] if ts is not None else ""  # HH:MM
        period = TIME_PERIOD_LABELS.get(str(row.get("time_of_day") or ""), "")
        head = " | ".join(p for p in (stamp, period) if p)
        speaker = render_speaker_label(row)
        _append_raw_message_line(
            lines,
            head=head,
            speaker=speaker,
            content=normalize_text(row.get("content")),
            is_trace_event=_is_trace_event(row),
        )
    return "\n".join(lines)


def render_prompt_context(context: dict[str, Any], *, tz: str, enable_flavor: bool = False) -> str:
    """把 build_prompt_context 的结构化三层渲染成可直接放进聊天模型 prompt 的文本。"""
    sections: list[str] = []
    raw_text = render_visible_raw(list(context.get("raw") or []), tz=tz)
    if raw_text:
        sections.append(raw_text)

    episodic = [
        render_summary_snippet(row, tz=tz, enable_flavor=enable_flavor) for row in list(context.get("episodic") or [])
    ]
    semantic = [
        render_semantic_snippet(row, tz=tz, enable_flavor=enable_flavor) for row in list(context.get("semantic") or [])
    ]
    sections.extend(text for text in episodic + semantic if text)
    return "\n\n".join(sections)


def render_raw_snippet(context_rows: list[dict[str, Any]], *, tz: str) -> str:
    """raw 上下文扩窗后的多条消息渲染成一段带时间标签的对话。"""
    lines: list[str] = ["【原始对话片段】"]
    for row in context_rows:
        ts = row.get("timestamp")
        stamp = timestamp_to_datetime_weekday_label(ts, tz) if ts is not None else ""
        period = TIME_PERIOD_LABELS.get(str(row.get("time_of_day") or ""), "")
        speaker = render_speaker_label(row)
        head = " | ".join(p for p in (stamp, period) if p)
        _append_raw_message_line(
            lines,
            head=head,
            speaker=speaker,
            content=normalize_text(row.get("content")),
            is_trace_event=_is_trace_event(row),
        )
    return "\n".join(lines)
