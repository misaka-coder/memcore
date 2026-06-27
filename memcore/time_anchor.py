"""时间锚点系统(焊死规则,但 timezone 是必填输入)。

见设计文档 §5。所有"时间戳 → 日期/时间段/相对转绝对"换算必须按显式 timezone 进行,
绝不用服务器本地时区。存储统一 UTC 时间戳,渲染时套该会话 tz。
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

# 时间段标签(中文展示用)
TIME_PERIOD_LABELS = {
    "morning": "上午",
    "afternoon": "下午",
    "night": "晚上",
    "midnight": "凌晨",
}
TIME_PERIOD_ORDER = ("midnight", "morning", "afternoon", "night")

# 检索出的旧记忆若含这些相对词,需追加锚点行,提醒按该记忆自身时间解释。
RELATIVE_TIME_TOKENS = (
    "今天",
    "今日",
    "今晚",
    "今早",
    "明天",
    "明早",
    "明晚",
    "昨天",
    "昨晚",
    "前天",
    "后天",
    "上周",
    "下周",
    "上个月",
    "下个月",
    "去年",
    "今年",
    "最近",
    "这段时间",
    "当前",
    "现在",
    "刚才",
    "一会儿",
    "过几天",
)


_PERIOD_ALIASES = {
    "morning": "morning",
    "上午": "morning",
    "早上": "morning",
    "清晨": "morning",
    "afternoon": "afternoon",
    "下午": "afternoon",
    "午后": "afternoon",
    "night": "night",
    "evening": "night",
    "晚上": "night",
    "夜晚": "night",
    "夜里": "night",
    "midnight": "midnight",
    "凌晨": "midnight",
    "半夜": "midnight",
}


def normalize_time_periods(values: object) -> list[str]:
    """把上午/下午/night 等别名归一成规范时间段,按一天顺序去重。"""
    if not isinstance(values, (list, tuple)):
        return []
    seen: set[str] = set()
    for value in values:
        canonical = _PERIOD_ALIASES.get(str(value or "").strip().lower())
        if canonical:
            seen.add(canonical)
    return [p for p in TIME_PERIOD_ORDER if p in seen]


def _dt(timestamp: float, tz: str) -> datetime:
    return datetime.fromtimestamp(float(timestamp), ZoneInfo(tz))


def infer_time_of_day(timestamp: float, tz: str) -> str:
    """按显式时区把时间戳归入 上午/下午/晚上/凌晨(确定性,不靠模型)。"""
    hour = _dt(timestamp, tz).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "night"
    return "midnight"


def timestamp_to_date_label(timestamp: float, tz: str) -> str:
    return _dt(timestamp, tz).strftime("%Y-%m-%d")


def timestamp_to_datetime_label(timestamp: float, tz: str) -> str:
    return _dt(timestamp, tz).strftime("%Y-%m-%d %H:%M")


def format_time_range_label(*, start_ts: float | None, end_ts: float | None, tz: str) -> str:
    """同日渲染 `YYYY-MM-DD HH:MM ~ HH:MM`,跨日渲染两端完整日期。"""
    if start_ts is None and end_ts is None:
        return ""
    if start_ts is None:
        return timestamp_to_datetime_label(end_ts, tz)  # type: ignore[arg-type]
    if end_ts is None:
        return timestamp_to_datetime_label(start_ts, tz)

    start_dt = _dt(start_ts, tz)
    end_dt = _dt(end_ts, tz)
    if start_dt.date() == end_dt.date():
        return f"{start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%H:%M')}"
    return f"{start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%Y-%m-%d %H:%M')}"


def render_relative_time_anchor_line(*, text: str, time_range_label: str) -> str:
    """旧记忆含相对时间时,锁定其按自身时间范围解释,不按当前日期重算。"""
    if not time_range_label:
        return ""
    normalized = str(text or "")
    if not any(token in normalized for token in RELATIVE_TIME_TOKENS):
        return ""
    return f"相对时间锚点:本条记忆中的今天/明天/昨天/最近等说法,均以 {time_range_label} 为准,不按当前日期重算。"
