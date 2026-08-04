"""时间锚点系统(焊死规则,但 timezone 是必填输入)。

见设计文档 §5。所有"时间戳 → 日期/时间段/相对转绝对"换算必须按显式 timezone 进行,
绝不用服务器本地时区。存储统一 UTC 时间戳,渲染时套该会话 tz。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone as datetime_timezone
from typing import Any
from zoneinfo import ZoneInfo

# 时间段标签(中文展示用)
TIME_PERIOD_LABELS = {
    "morning": "上午",
    "afternoon": "下午",
    "night": "晚上",
    "midnight": "凌晨",
}
TIME_PERIOD_ORDER = ("midnight", "morning", "afternoon", "night")
WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

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
    "本周",
    "这周",
    "周一",
    "周二",
    "周三",
    "周四",
    "周五",
    "周六",
    "周日",
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
    "星期天",
    "礼拜一",
    "礼拜二",
    "礼拜三",
    "礼拜四",
    "礼拜五",
    "礼拜六",
    "礼拜日",
    "礼拜天",
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


@dataclass(frozen=True)
class TimelineTimeSelector:
    """Normalized, timezone-explicit bounds for deterministic raw timeline reads."""

    start_ts: int
    end_ts: int
    start_at: str
    end_at: str
    time_periods: tuple[str, ...] = ()
    mode: str = "time_range"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_at": self.start_at,
            "end_at": self.end_at,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
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


def normalize_timeline_time_selector(
    *,
    timezone: str,
    time_range: object = None,
    date_from: str = "",
    date_to: str = "",
    time_periods: list[str] | None = None,
) -> TimelineTimeSelector:
    """Normalize exact or legacy timeline selectors into one timestamp-bounded contract.

    Exact bounds accept ISO 8601 strings with an explicit offset, or local wall
    time strings interpreted using ``timezone``. Legacy date/period arguments
    are aliases that enter the same timestamp SQL path.
    """

    zone = ZoneInfo(str(timezone))
    has_exact = time_range is not None
    has_legacy = bool(str(date_from or "").strip() or str(date_to or "").strip() or list(time_periods or []))
    if has_exact and has_legacy:
        raise ValueError("timeline_modes_are_mutually_exclusive")

    if has_exact:
        if not isinstance(time_range, dict):
            raise ValueError("time_range_must_be_object")
        unknown = sorted(str(key) for key in time_range if str(key) not in {"start_at", "end_at"})
        if unknown:
            raise ValueError(f"unknown_time_range_keys:{unknown}")
        start_raw = _required_time_range_text(time_range, "start_at")
        end_raw = _required_time_range_text(time_range, "end_at")
        start = _parse_timeline_datetime(start_raw, field="start_at", zone=zone)
        end = _parse_timeline_datetime(end_raw, field="end_at", zone=zone)
        if start >= end:
            raise ValueError("time_range_start_must_be_before_end")
        return TimelineTimeSelector(
            start_ts=int(start.timestamp()),
            end_ts=int(end.timestamp()),
            start_at=start.isoformat(timespec="seconds"),
            end_at=end.isoformat(timespec="seconds"),
        )

    start_text = str(date_from or "").strip()
    end_text = str(date_to or "").strip() or start_text
    if not start_text:
        raise ValueError("timeline_selector_required")
    try:
        start_date = date.fromisoformat(start_text)
        end_date = date.fromisoformat(end_text)
    except ValueError as exc:
        raise ValueError("date_must_be_YYYY-MM-DD") from exc
    if start_date > end_date:
        raise ValueError("date_from_after_date_to")

    raw_periods = [str(value).strip() for value in (time_periods or []) if str(value or "").strip()]
    unknown_periods = [value for value in raw_periods if not normalize_time_periods([value])]
    if unknown_periods:
        raise ValueError(f"unknown_time_periods:{unknown_periods}")
    periods = tuple(normalize_time_periods(raw_periods))
    start = _localize_wall_time(datetime.combine(start_date, time.min), zone=zone, field="date_from")
    exclusive_end = _localize_wall_time(
        datetime.combine(end_date + timedelta(days=1), time.min), zone=zone, field="date_to"
    )
    return TimelineTimeSelector(
        start_ts=int(start.timestamp()),
        end_ts=int(exclusive_end.timestamp()),
        start_at=start.isoformat(timespec="seconds"),
        end_at=exclusive_end.isoformat(timespec="seconds"),
        time_periods=periods,
        mode="date",
    )


def normalize_retrieval_time_hint(*, timezone: str, value: object = None) -> dict[str, Any]:
    """Normalize fuzzy-retrieval time hints through the timeline selector.

    The model-facing exact form is ``{start_at, end_at}``. Legacy
    ``date_label/time_of_day`` and epoch ``start_ts/end_ts`` remain accepted as
    aliases, but selector modes cannot be mixed. Exact and dated hints use the
    same timezone/DST rules as :func:`normalize_timeline_time_selector`.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("time_hint_must_be_object")
    if not value:
        return {}
    allowed = {"start_at", "end_at", "date_label", "time_of_day", "start_ts", "end_ts"}
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"unknown_time_hint_keys:{unknown}")

    has_exact = any(key in value and value.get(key) is not None for key in ("start_at", "end_at"))
    has_dated = any(str(value.get(key) or "").strip() for key in ("date_label", "time_of_day"))
    has_epoch = any(key in value and value.get(key) is not None for key in ("start_ts", "end_ts"))
    if sum((has_exact, has_dated, has_epoch)) > 1:
        raise ValueError("time_hint_modes_are_mutually_exclusive")

    if has_exact:
        selector = normalize_timeline_time_selector(
            timezone=timezone,
            time_range={"start_at": value.get("start_at"), "end_at": value.get("end_at")},
        )
        return selector.to_dict()

    if has_dated:
        date_label = str(value.get("date_label") or "").strip()
        time_of_day = str(value.get("time_of_day") or "").strip()
        periods: list[str] = []
        if time_of_day:
            periods = normalize_time_periods([time_of_day])
            if not periods:
                raise ValueError(f"unknown_time_periods:{[time_of_day]}")
        if not date_label:
            return {"time_periods": periods}
        selector = normalize_timeline_time_selector(
            timezone=timezone,
            date_from=date_label,
            date_to=date_label,
            time_periods=periods,
        )
        result = selector.to_dict()
        if selector.time_periods:
            result["time_periods"] = list(selector.time_periods)
        return result

    if has_epoch:
        result: dict[str, Any] = {}
        for key in ("start_ts", "end_ts"):
            raw = value.get(key)
            if raw is None:
                continue
            if isinstance(raw, bool):
                raise ValueError(f"time_hint_{key}_must_be_integer")
            try:
                result[key] = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"time_hint_{key}_must_be_integer") from exc
        if "start_ts" in result and "end_ts" in result and result["start_ts"] >= result["end_ts"]:
            raise ValueError("time_hint_start_must_be_before_end")
        return result

    return {}


def _required_time_range_text(value: dict[object, object], field: str) -> str:
    raw = value.get(field)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError(f"time_range_{field}_required")
    if not isinstance(raw, str):
        raise ValueError(f"time_range_{field}_must_be_string")
    return raw.strip()


def _parse_timeline_datetime(value: str, *, field: str, zone: ZoneInfo) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"time_range_{field}_invalid") from exc
    if parsed.microsecond:
        raise ValueError(f"time_range_{field}_fractional_seconds_unsupported")
    if parsed.tzinfo is None:
        return _localize_wall_time(parsed, zone=zone, field=field)
    return parsed.astimezone(zone)


def _localize_wall_time(value: datetime, *, zone: ZoneInfo, field: str) -> datetime:
    candidates: list[datetime] = []
    for fold in (0, 1):
        candidate = value.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(datetime_timezone.utc).astimezone(zone)
        if round_trip.replace(tzinfo=None) == value and round_trip.fold == fold:
            candidates.append(candidate)
    distinct_offsets = {candidate.utcoffset() for candidate in candidates}
    if not candidates:
        raise ValueError(f"time_range_{field}_nonexistent_local_time")
    if len(distinct_offsets) > 1:
        raise ValueError(f"time_range_{field}_ambiguous_local_time_requires_offset")
    return candidates[0]


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


def timestamp_to_weekday_label(timestamp: float, tz: str) -> str:
    return WEEKDAY_LABELS[_dt(timestamp, tz).weekday()]


def timestamp_to_date_weekday_label(timestamp: float, tz: str) -> str:
    local = _dt(timestamp, tz)
    return f"{local.strftime('%Y-%m-%d')} {WEEKDAY_LABELS[local.weekday()]}"


def timestamp_to_datetime_label(timestamp: float, tz: str) -> str:
    return _dt(timestamp, tz).strftime("%Y-%m-%d %H:%M")


def timestamp_to_datetime_weekday_label(timestamp: float, tz: str) -> str:
    local = _dt(timestamp, tz)
    return f"{local.strftime('%Y-%m-%d')} {WEEKDAY_LABELS[local.weekday()]} {local.strftime('%H:%M')}"


def format_time_range_label(*, start_ts: float | None, end_ts: float | None, tz: str) -> str:
    """同日渲染 `YYYY-MM-DD 周X HH:MM ~ HH:MM`,跨日渲染两端完整日期。"""
    if start_ts is None and end_ts is None:
        return ""
    if start_ts is None:
        return timestamp_to_datetime_weekday_label(end_ts, tz)  # type: ignore[arg-type]
    if end_ts is None:
        return timestamp_to_datetime_weekday_label(start_ts, tz)

    start_dt = _dt(start_ts, tz)
    end_dt = _dt(end_ts, tz)
    start_weekday = WEEKDAY_LABELS[start_dt.weekday()]
    end_weekday = WEEKDAY_LABELS[end_dt.weekday()]
    if start_dt.date() == end_dt.date():
        return (
            f"{start_dt.strftime('%Y-%m-%d')} {start_weekday} {start_dt.strftime('%H:%M')} ~ {end_dt.strftime('%H:%M')}"
        )
    return (
        f"{start_dt.strftime('%Y-%m-%d')} {start_weekday} {start_dt.strftime('%H:%M')} ~ "
        f"{end_dt.strftime('%Y-%m-%d')} {end_weekday} {end_dt.strftime('%H:%M')}"
    )


def render_relative_time_anchor_line(*, text: str, time_range_label: str) -> str:
    """旧记忆含相对时间时,锁定其按自身时间范围解释,不按当前日期重算。"""
    if not time_range_label:
        return ""
    normalized = str(text or "")
    if not any(token in normalized for token in RELATIVE_TIME_TOKENS):
        return ""
    return (
        "相对时间锚点:本条记忆中的今天/明天/昨天/上周二/下周三/最近等说法,"
        f"均以 {time_range_label} 为准,不按当前日期重算。"
    )
