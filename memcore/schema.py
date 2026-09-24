"""MemCore 唯一的模型可写记忆元数据契约。

提示词、存储、压缩、索引和检索工具都必须从这里读取字段名与枚举，宿主不能
另写一套同义 schema。系统掌握的 namespace/kind/lineage 等事实不属于模型元数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import SchemaError

# --- 固定协议枚举 ---

MEMORY_FACETS: tuple[str, ...] = (
    "profile",
    "preference",
    "viewpoint",
    "relationship",
    "event",
    "state",
    "plan",
    "decision",
    "constraint",
    "knowledge",
    "procedure",
)

ABOUT_ROLES: tuple[str, ...] = ("user", "assistant", "third_party", "external")
RETRIEVAL_PRIORITIES: tuple[str, ...] = ("low", "normal", "high", "critical")
TURN_INTENTS: tuple[str, ...] = ("memory_query",)

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

# mood 是有限表现枚举；实体、主题和 facet 不设隐藏长度/token 裁剪。
MAX_MOOD_TAGS = 3


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

    turn_intent: str = ""
    memory_facets: list[str] = field(default_factory=list)
    about_roles: list[str] = field(default_factory=list)
    entity_anchors: list[str] = field(default_factory=list)
    topic_terms: list[str] = field(default_factory=list)
    retrieval_priority: str = "normal"
    mood_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_intent": self.turn_intent,
            "memory_facets": list(self.memory_facets),
            "about_roles": list(self.about_roles),
            "entity_anchors": list(self.entity_anchors),
            "topic_terms": list(self.topic_terms),
            "retrieval_priority": self.retrieval_priority,
            "mood_tags": list(self.mood_tags),
        }


def coerce_memory_metadata(
    data: Any,
    *,
    enable_flavor: bool = False,
) -> MemoryMetadata:
    """把模型/外部给的 metadata 校验、回退成合法 MemoryMetadata。

    - 枚举外的值丢弃；自由字符串列表 trim 并按首次出现去重。
    - `enable_flavor=False` 时强制清空 mood_tags(温度层关闭,客观字段不受污染)。
    - 不读取旧 keywords/categories/subject_scopes/importance/confidence；旧库由 schema migration 一次转换。
    """
    src = data if isinstance(data, dict) else {}
    mood = _coerce_enum_list(src.get("mood_tags"), MOOD_TAGS, limit=MAX_MOOD_TAGS) if enable_flavor else []
    turn_intent = str(src.get("turn_intent") or "").strip()
    if turn_intent not in TURN_INTENTS:
        turn_intent = ""
    priority = str(src.get("retrieval_priority") or "normal").strip().lower()
    if priority not in RETRIEVAL_PRIORITIES:
        priority = "normal"
    return MemoryMetadata(
        turn_intent=turn_intent,
        memory_facets=_coerce_enum_list(src.get("memory_facets"), MEMORY_FACETS),
        about_roles=_coerce_enum_list(src.get("about_roles"), ABOUT_ROLES),
        entity_anchors=_coerce_str_list(src.get("entity_anchors")),
        topic_terms=_coerce_str_list(src.get("topic_terms")),
        retrieval_priority=priority,
        mood_tags=mood,
    )


def memory_metadata_has_signal(data: Any) -> bool:
    """Return whether normalized metadata contains a non-default retrieval signal."""

    metadata = data.to_dict() if isinstance(data, MemoryMetadata) else coerce_memory_metadata(data).to_dict()
    if metadata["turn_intent"] or metadata["retrieval_priority"] != "normal":
        return True
    return any(metadata[key] for key in ("memory_facets", "about_roles", "entity_anchors", "topic_terms", "mood_tags"))


def build_memory_metadata_instruction(
    *,
    enable_flavor: bool = False,
    require_disabled_mood_field: bool = False,
) -> str:
    """Build the one model-facing metadata explanation used by every writer.

    Compaction prompts omit the optional flavor field entirely while flavor is
    disabled. The chat-output JSON contract has a fixed shape, so it can ask
    for an empty ``mood_tags`` array through ``require_disabled_mood_field``.
    Both forms still come from this single field contract.
    """

    mood_rule = ""
    if enable_flavor:
        mood_rule = f"[情感温度(已启用)] mood_tags：情感余温，可从以下枚举选择 0-3 个：{' / '.join(MOOD_TAGS)}。"
    elif require_disabled_mood_field:
        mood_rule = "mood_tags：输出空数组。"
    return (
        "memory_metadata 标注宿主指定的本轮记忆目标，不描述回复；无关字段留空。"
        "turn_intent：目标在查询历史时填 memory_query，否则留空。"
        f"memory_facets：内容未来可回答的问题类型，选 {' / '.join(MEMORY_FACETS)}。"
        f"about_roles：内容主要描述谁或什么（不是发言参与者），选 {' / '.join(ABOUT_ROLES)}。"
        "entity_anchors：已知且未来正常聊天可能追问的准确名称或别名；未知答案和宽泛上位词不填。"
        "topic_terms：优先保留具体、便于检索的名词和主题短语，也可保留关键动作、关系、属性，不写句子。"
        "两组词去重；先保留原内容的具体名称，再少量补充有依据的同义词或上位词（放入 topic_terms），不要用泛词替代专名。"
        f"retrieval_priority：未来召回价值，选 {' / '.join(RETRIEVAL_PRIORITIES)}。"
        f"{mood_rule}"
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
    memory_title: str = ""
    catalog_hint: str = ""
    topic_headings: list[str] = field(default_factory=list)


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
    memory_title: str = ""
    catalog_hint: str = ""
    topic_headings: list[str] = field(default_factory=list)


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
