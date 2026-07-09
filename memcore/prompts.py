"""记忆压缩链的承重提示词(焊死骨架 + 人格文本插槽)。

覆盖 summary / semantic / reinforcement(写侧)与 verifier(读侧)。
焊死的是:"你就是当前角色整理自己的记忆 + 不许编造 + 只输出 JSON + 字段固定 + importance 是 0-1 数字 +
时间锚点(相对转绝对)"。可配置的只有 persona_text(填进 [CHARACTER MEMORY SELF] 插槽)。

提示词治理:焊死骨架 + 校验插槽 —— 见 PromptOverrides 与 _weld,插槽只能补充、不可移除骨架。
具体的人格内容由接入方运行时通过 PromptOverrides 注入,不内置于本库。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PromptError
from .schema import MOOD_TAGS

MEMORY_TIME_ANCHOR_RULES = (
    "[时间锚点规则]\n"
    "记录里的日期、时间范围都是真实时间锚点。整理任何会入库的记忆字段时,遇到"
    "“今天/明天/昨天/最近/刚才/这段时间”等相对时间,必须按源消息或源摘要的时间锚点"
    "改写为绝对日期或绝对日期范围,不要在入库字段里留下未锚定的相对时间。"
    "日期旁的星期几(如 2026-04-10 周五)也是时间锚点的一部分,解析“上周二/下周三”等说法时必须使用它。"
)

MULTI_ACTOR_MEMORY_RULES = (
    "[多方/群聊归因规则]\n"
    "原始对话若出现 user(昵称) / user(昵称;id=稳定ID) / assistant(昵称) / other(昵称) 这类发言人标签,"
    "整理事实时必须保留事实主体:谁表达了偏好、谁提出计划、谁承诺行动、谁情绪变化。"
    "同一稳定ID代表同一发言人;昵称只用于显示,不要把不同稳定ID的人合并。"
    "不要把不同发言人的事实笼统写成“用户说/大家说”。"
)

MEMORY_METADATA_RULES = (
    "[memory_metadata 标注规则]\n"
    "memory_metadata 是后续检索前置过滤的索引信号。keywords 写 0-4 个可复用检索标签,不要写整句或短句;"
    "优先选择用户未来正常聊天里可能会用来追问的自然短词,如具体实体、别名、主题、计划、偏好、风险等;"
    "上位词/领域词/意图词只在常见且能提高召回时补充,不要机械泛化成太宽的标签。"
    "例如“可乐”可补“饮料/偏好”,但具体项目名通常保留项目名、别名和真实议题即可;"
    "subject_scopes 标事实主体(user/assistant/other),categories 只从固定枚举选;"
    "importance 按长期价值评分,confidence 按你对标注正确性的把握评分。"
    "没有明确长期价值时宁可低分或空数组,不要为了填字段而编造标签。"
)

# 插槽文本上限:领域插槽只能补充,不能塞进一整套替代提示词。
MAX_SLOT_CHARS = 4000


@dataclass(frozen=True)
class PromptOverrides:
    """提示词治理:只暴露"可补充的领域插槽",焊死骨架(契约 + 时间锚点)永远不可被外部移除。

    - persona_text:角色身份文本,填进 [CHARACTER MEMORY SELF]。
    - extra_*_guidance:各任务的领域补充指引,只追加在骨架之后,不能改写骨架。
    构造时校验类型与长度;非法直接报错(让"写坏"在接口层提交不进来)。
    """

    persona_text: str = ""
    extra_summary_guidance: str = ""
    extra_semantic_guidance: str = ""
    extra_reinforcement_guidance: str = ""

    def __post_init__(self) -> None:
        for name in (
            "persona_text",
            "extra_summary_guidance",
            "extra_semantic_guidance",
            "extra_reinforcement_guidance",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise PromptError(f"{name} must be a string, got {type(value).__name__}")
            if len(value) > MAX_SLOT_CHARS:
                raise PromptError(f"{name} too long ({len(value)} > {MAX_SLOT_CHARS}); slots may only supplement")


def _mood_instruction() -> str:
    """温度启用时注入:让模型在 mood_tags 写情感余温。枚举取自焊死的 MOOD_TAGS。"""
    return (
        "[情感温度(已启用)]\n"
        "在 memory_metadata.mood_tags 里写 0-3 个你记住这件事时的情感余温,只能从固定枚举里选:"
        f"{' / '.join(MOOD_TAGS)}。\n"
        "mood_tags 是你的记忆感受,不要污染 core_facts / stable_facts 等客观字段。"
    )


def _weld(base_system: str, *, persona_text: str = "", extra_guidance: str = "", enable_flavor: bool = False) -> str:
    """焊死骨架装配:base(契约)永远在前,质量规则永远追加;插槽/温度只能填在中间(只增不改)。

    enable_flavor 动态决定是否注入 mood 指令——温度关闭时三层提示词一个字都不提 mood。
    """
    parts = [base_system.rstrip()]
    persona = str(persona_text or "").strip()
    if persona:
        parts.append(
            "[CHARACTER MEMORY SELF]\n"
            "下面是你此刻的身份与表达侧面;整理记忆时就按这个身份记。"
            "角色设定只决定记忆口吻、在意点和情感余温,不是事实本身。\n"
            f"{persona}"
        )
    if enable_flavor:
        parts.append(_mood_instruction())
    extra = str(extra_guidance or "").strip()
    if extra:
        parts.append(f"[领域补充指引(只补充,不得改写以上任何规则与字段契约)]\n{extra}")
    parts.append(MEMORY_TIME_ANCHOR_RULES)
    parts.append(MULTI_ACTOR_MEMORY_RULES)
    parts.append(MEMORY_METADATA_RULES)
    return "\n\n".join(parts)


SUMMARY_SYSTEM = (
    "你就是当前角色,正在整理自己较早的一段记忆。请严格输出 JSON。\n"
    "字段固定为 diary_summary, period_label, event_type, importance, key_events, core_facts, memory_metadata。\n"
    "diary_summary 是你的日记式回忆,可带一点语气和心情。\n"
    "core_facts 要客观、稳定、适合后续检索;不要把角色设定当成事实写进去。\n"
    "memory_metadata 只用于检索入库:keywords 0-4 个短标签,按未来正常聊天里可能命中的问法选词,"
    "不要写成短句,也不要机械补太宽泛的上位词;subject_scopes 从 user/assistant/other 选,categories 从固定枚举选。\n"
    "不要编造对话里没有出现的事实。importance 必须是 0.0 到 1.0 之间的数字,不要写“高/中/低”。\n"
    "key_events 和 core_facts 必须是 JSON 数组。只输出一个合法 JSON 对象,不要解释或代码块。"
)

SUMMARY_USER_TEMPLATE = (
    "请总结下面这段较早的 {batch_size} 条对话:\n{transcript}\n\n"
    "要求:简洁、连贯、适合后续记忆检索。再次强调:importance 只能是数字。"
)

SEMANTIC_SYSTEM = (
    "你就是当前角色,正在把阶段回忆沉淀成长期记忆。请把收到的较早阶段摘要进一步压缩成更稳定的语义记忆,"
    "并严格输出 JSON。\n"
    "字段固定为 semantic_summary, importance, stable_facts, recurring_topics, important_people, open_loops, memory_metadata。\n"
    "stable_facts 要稳定、客观、适合长期保留;不要把角色设定写进去。\n"
    "recurring_topics 抓反复出现的话题;important_people 只留明显重要或反复出现的人;open_loops 记仍在推进的事项。\n"
    "不要编造摘要里没有的长期结论。importance 必须是 0.0 到 1.0 之间的数字。\n"
    "stable_facts/recurring_topics/important_people/open_loops 必须是 JSON 数组。只输出一个合法 JSON 对象。"
)

SEMANTIC_USER_TEMPLATE = (
    "请把下面这组更早的阶段摘要,再压缩成一条长期语义记忆:\n{source_text}\n\n"
    "要求:保留长期稳定事实、反复话题、重要人物和未完成线索;不要写成流水账。importance 只能是 0 到 1 的数字。"
)

REINFORCEMENT_SYSTEM = (
    "你就是当前角色,正在重新整理一条自己的长期记忆。你会收到一条已有长期语义记忆,以及一组新的阶段摘要压缩结果。\n"
    "如果它们明显属于同一长期主线,请输出一条融合后的长期语义记忆,严格输出 JSON,字段同语义记忆。\n"
    "尽量保留已有稳定事实,同时自然吸收新近重复出现的内容。不要因为新内容只出现一次就推翻旧的稳定印象。\n"
    "不要写成流水账。importance 必须是 0.0 到 1.0 之间的数字。各列表字段必须是 JSON 数组。只输出一个合法 JSON 对象。"
)

REINFORCEMENT_USER_TEMPLATE = (
    "已有长期语义记忆:\n{existing_text}\n\n新的阶段摘要压缩结果:\n{incoming_text}\n\n"
    "请输出融合后的长期记忆,尽量保留旧的稳定信息,同时吸收新的重复线索。"
)


VERIFIER_SYSTEM = (
    "你是记忆检索校验器,只判断检索到的片段是否足以回答用户问题。\n"
    "必须输出 NDJSON,每行一个合法 JSON 对象,不要输出解释。\n"
    '第一行 decision 事件:{"type":"decision","match_result":"match|mismatch"}。\n'
    '若 match,第二行 selection 事件:{"type":"selection","selected_indexes":[1,2]}'
    "(编号从 1 开始,对应展示的片段)。\n"
    "只选对回答当前问题有直接帮助的片段;若片段只是重复用户提问或没有新事实,判 mismatch。"
)

VERIFIER_USER_TEMPLATE = "用户问题:\n{query}\n\n检索到的记忆片段(编号从 1 开始):\n{snippets}"


def build_verifier_prompts(*, query: str, snippets_text: str) -> tuple[str, str]:
    return (VERIFIER_SYSTEM, VERIFIER_USER_TEMPLATE.format(query=query, snippets=snippets_text))


def build_summary_prompts(
    *,
    transcript: str,
    batch_size: int,
    overrides: PromptOverrides | None = None,
    enable_flavor: bool = False,
    reference_summary_text: str = "",
) -> tuple[str, str]:
    ov = overrides or PromptOverrides()
    user = SUMMARY_USER_TEMPLATE.format(transcript=transcript, batch_size=int(batch_size))
    reference = str(reference_summary_text or "").strip()
    if reference:
        # 既有摘要作参考:保持人物/项目/时间线/口吻一致,并避免与既有摘要冲突;但不引入新事实。
        user = (
            f"{user}\n\n"
            "可参考的既有阶段摘要(仅用于保持人物关系、项目脉络、时间线、记忆口吻一致,"
            "并避免与既有摘要相互冲突;不要把参考里出现、但本段对话没出现的内容写成本段的新事实):\n"
            f"{reference}"
        )
    return (
        _weld(
            SUMMARY_SYSTEM,
            persona_text=ov.persona_text,
            extra_guidance=ov.extra_summary_guidance,
            enable_flavor=enable_flavor,
        ),
        user,
    )


def build_semantic_prompts(
    *, source_text: str, overrides: PromptOverrides | None = None, enable_flavor: bool = False
) -> tuple[str, str]:
    ov = overrides or PromptOverrides()
    return (
        _weld(
            SEMANTIC_SYSTEM,
            persona_text=ov.persona_text,
            extra_guidance=ov.extra_semantic_guidance,
            enable_flavor=enable_flavor,
        ),
        SEMANTIC_USER_TEMPLATE.format(source_text=source_text),
    )


def build_reinforcement_prompts(
    *, existing_text: str, incoming_text: str, overrides: PromptOverrides | None = None, enable_flavor: bool = False
) -> tuple[str, str]:
    ov = overrides or PromptOverrides()
    return (
        _weld(
            REINFORCEMENT_SYSTEM,
            persona_text=ov.persona_text,
            extra_guidance=ov.extra_reinforcement_guidance,
            enable_flavor=enable_flavor,
        ),
        REINFORCEMENT_USER_TEMPLATE.format(existing_text=existing_text, incoming_text=incoming_text),
    )
