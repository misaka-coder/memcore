"""记忆片段渲染:把记录渲染成喂回模型的文本,带时间标签、相对时间锚点、心情行。

见设计文档 §6(检索回填带回时间标签)。渲染按显式 tz;时间范围从记录自身的
period_start_ts/period_end_ts/timestamp 取(跨 store 的 seq 解析属于 store,渲染不碰)。
"""

from __future__ import annotations

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
    return "\n".join(parts)


def render_timeline(messages: list[dict[str, Any]], *, tz: str) -> str:
    """时间线工具:把按日期范围读出的原始对话渲染成"按天分组、带时刻"的可读文本。"""
    return _render_grouped_raw_messages(messages, tz=tz, title="【按时间读取的原始对话(精确记录,非摘要)】")


def render_visible_raw(messages: list[dict[str, Any]], *, tz: str) -> str:
    """固定可见 raw:按日期分组,每条只渲染时刻/时间段,减少重复时间标签。"""
    return _render_grouped_raw_messages(messages, tz=tz, title="【近期原始对话(未摘要)】")


_SPEAKER_LABEL_MAX_CHARS = 40
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
    """渲染 role + actor 显示名,让群聊/多方场景不丢"谁说的"。"""
    role = _sanitize_speaker_part(row.get("role"))
    actor_name = _sanitize_speaker_part(row.get("actor_display_name"))
    actor_id = _sanitize_speaker_part(row.get("actor_id"))
    if actor_name and actor_id and actor_name != actor_id:
        actor = f"{actor_name};id={actor_id}"
    else:
        actor = actor_name or actor_id
    if role and actor:
        return f"{role}({actor})"
    return role or actor


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
        lines.append(f"[{head}] {speaker}: {normalize_text(row.get('content'))}".rstrip())
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
        lines.append(f"[{head}] {speaker}: {normalize_text(row.get('content'))}".rstrip())
    return "\n".join(lines)
