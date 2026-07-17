"""焊死的输出字段契约 —— 这是 memcore 的皇冠明珠。

提示词、存储、检索过滤、片段渲染**四方共用**这套字段名与类型。改一个字段名,四处静默崩。
因此:

- **枚举焊死**(categories / subject_scopes / mood_tags):领域可换具体词表(经 MemoryConfig 校验),
  但取值必须来自固定枚举,模型不许自由发挥。
- **数值焊死**(importance / confidence 必须是 0..1)。
- **校验即插槽**:外部/模型给的数据先过 `coerce_*`,非法值被丢弃或回退,而不是污染记忆。

本切片只定义契约与校验;真正的入库/检索在后续切片消费这些契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import SchemaError

# --- 焊死枚举(默认词表;领域可经 MemoryConfig 覆盖,但仍是"固定枚举") ---

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "casual",
    "preference",
    "personal_profile",
    "plan_goal",
    "project_work",
    "relationship",
    "emotion_state",
    "life_event",
    "memory_query",
    "system_meta",
    "tool_trace",
    "material_trace",
)

TRACE_CATEGORIES: tuple[str, ...] = ("tool_trace", "material_trace")

SUBJECT_SCOPES: tuple[str, ...] = ("user", "assistant", "other")

MOOD_TAGS: tuple[str, ...] = (
    "calm",
    "warm",
    "affectionate",
    "happy",
    "playful",
    "curious",
    "thoughtful",
    "touched",
    "proud",
    "worried",
    "lonely",
    "sad",
    "embarrassed",
    "tense",
    "annoyed",
    "determined",
)

# 各列表字段的上限(与设计文档一致:keywords 0-4、mood 0-3)
MAX_KEYWORDS = 4
MAX_MOOD_TAGS = 3


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def _coerce_str_list(value: Any, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _coerce_enum_list(value: Any, allowed: Iterable[str], *, limit: int | None = None) -> list[str]:
    """只保留合法枚举值(非法值丢弃,不报错——保持记忆可检索,符合"软过滤"原则)。"""
    allowed_set = {str(item) for item in allowed}
    out: list[str] = []
    seen: set[str] = set()
    for item in value if isinstance(value, (list, tuple)) else []:
        text = str(item or "").strip()
        if text in allowed_set and text not in seen:
            seen.add(text)
            out.append(text)
            if limit is not None and len(out) >= limit:
                break
    return out


@dataclass
class MemoryMetadata:
    """检索入库元数据 —— 检索过滤靠它,贯穿全链。"""

    keywords: list[str] = field(default_factory=list)
    subject_scopes: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    mood_tags: list[str] = field(default_factory=list)
    importance: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "subject_scopes": list(self.subject_scopes),
            "categories": list(self.categories),
            "mood_tags": list(self.mood_tags),
            "importance": self.importance,
            "confidence": self.confidence,
        }


def coerce_memory_metadata(
    data: Any,
    *,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    enable_flavor: bool = False,
) -> MemoryMetadata:
    """把模型/外部给的 metadata 校验、回退成合法 MemoryMetadata。

    - 枚举外的值丢弃;数值 clamp 到 0..1;列表去重截断。
    - `enable_flavor=False` 时强制清空 mood_tags(温度层关闭,客观字段不受污染)。
    """
    src = data if isinstance(data, dict) else {}
    mood = _coerce_enum_list(src.get("mood_tags"), MOOD_TAGS, limit=MAX_MOOD_TAGS) if enable_flavor else []
    return MemoryMetadata(
        keywords=_coerce_str_list(src.get("keywords"), limit=MAX_KEYWORDS),
        subject_scopes=_coerce_enum_list(src.get("subject_scopes"), SUBJECT_SCOPES),
        categories=_coerce_enum_list(src.get("categories"), categories),
        mood_tags=mood,
        importance=_clamp01(src.get("importance")),
        confidence=_clamp01(src.get("confidence")),
    )


@dataclass
class SummaryRecord:
    """阶段摘要契约(raw → 摘要)。"""

    diary_summary: str
    period_label: str
    event_type: str
    importance: float
    key_events: list[str] = field(default_factory=list)
    core_facts: list[str] = field(default_factory=list)
    memory_metadata: MemoryMetadata = field(default_factory=MemoryMetadata)


@dataclass
class SemanticRecord:
    """长期语义记忆契约(摘要 → 长期事实 / 强化合并)。"""

    semantic_summary: str
    importance: float
    stable_facts: list[str] = field(default_factory=list)
    recurring_topics: list[str] = field(default_factory=list)
    important_people: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    memory_metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    reinforcement_count: int = 1


# 焊死的"必须存在"字段集合(供后续切片校验模型输出用)
SUMMARY_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"diary_summary", "period_label", "event_type", "importance", "key_events", "core_facts", "memory_metadata"}
)
SEMANTIC_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "semantic_summary",
        "importance",
        "stable_facts",
        "recurring_topics",
        "important_people",
        "open_loops",
        "memory_metadata",
    }
)


def require_fields(data: dict[str, Any], required: frozenset[str], *, context: str) -> None:
    """焊死字段契约:缺字段直接结构化报错(用于校验模型输出,非用户可配项)。"""
    if not isinstance(data, dict):
        raise SchemaError(f"{context}: expected a JSON object, got {type(data).__name__}")
    missing = required - set(data.keys())
    if missing:
        raise SchemaError(f"{context}: missing required fields {sorted(missing)}")
