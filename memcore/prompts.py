"""记忆压缩链的承重提示词(焊死骨架 + 人格文本插槽)。

覆盖 summary / semantic / reinforcement 三个记忆压缩任务。
焊死的是:"你就是当前角色整理自己的记忆 + 不许编造 + 只输出 JSON + 字段固定 + importance 是 0-1 数字 +
时间锚点(相对转绝对)"。可配置的只有 persona_text(填进 [CHARACTER MEMORY SELF] 插槽)。

提示词治理:焊死骨架 + 校验插槽 —— 见 PromptOverrides 与 _weld,插槽只能补充、不可移除骨架。
具体的人格内容由接入方运行时通过 PromptOverrides 注入,不内置于本库。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PromptError
from .schema import build_memory_metadata_instruction

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
    "若发言标签含“-> 接收方”或 target_actor,事实、请求、计划或承诺与接收对象有关时必须保留接收方。"
    "没有明确接收方不等于发给助手;旁观到的群消息不得改写成对助手的请求、承诺或共同经历。"
)

MEMORY_METADATA_RULES = "[memory_metadata 标注规则]\n" + build_memory_metadata_instruction(enable_flavor=False)

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
    extra = str(extra_guidance or "").strip()
    if extra:
        parts.append(f"[领域补充指引(只补充,不得改写以上任何规则与字段契约)]\n{extra}")
    parts.append(MEMORY_TIME_ANCHOR_RULES)
    parts.append(MULTI_ACTOR_MEMORY_RULES)
    parts.append("[memory_metadata 标注规则]\n" + build_memory_metadata_instruction(enable_flavor=enable_flavor))
    return "\n\n".join(parts)


SUMMARY_SYSTEM = (
    "你就是当前角色,正在整理自己较早的一段记忆。请严格输出 JSON。\n"
    "字段固定为 diary_summary, period_label, event_type, importance, key_events, core_facts, memory_metadata,"
    " memory_title, catalog_hint, topic_headings。\n"
    "diary_summary 是你的日记式回忆,可带一点语气和心情。\n"
    "memory_title 是便于以后翻找这段记忆的简短具体标题;catalog_hint 用一句话说明这段记忆能回答什么;"
    "topic_headings 在本段确有多个主题时列出少量主题短语,否则输出空数组。"
    "它们只能概括本段内容,不能补写没出现的人名、事件或结论。\n"
    "core_facts 要客观、稳定、适合后续检索;不要把角色设定当成事实写进去。\n"
    "memory_metadata 只用于检索入库，字段和含义遵循后附的统一标注规则。\n"
    "从本段完整来源提取 entity_anchors/topic_terms，覆盖实际谈到的具体对象和次要话题；"
    "即使原始消息没有标签，也要依据内容生成，不要只从总标题取词。\n"
    "不要编造对话里没有出现的事实。importance 必须是 0.0 到 1.0 之间的数字,不要写“高/中/低”。\n"
    "若后续消息明确说某个任务/材料已清理、取消、不再需要或已经结束,必须保留这个关闭状态;"
    "更早的失败、等待确认或待处理只能作为历史经过,不能继续写成当前未完成事项。\n"
    "key_events、core_facts 和 topic_headings 必须是 JSON 数组。只输出一个合法 JSON 对象,不要解释或代码块。"
)

SUMMARY_USER_TEMPLATE = (
    "请总结下面这段较早的 {batch_size} 条对话:\n{transcript}\n\n"
    "要求:简洁、连贯、适合后续记忆检索。再次强调:importance 只能是数字。"
)

SEMANTIC_SYSTEM = (
    "你就是当前角色,正在把阶段回忆沉淀成长期记忆。请把收到的较早阶段摘要进一步压缩成更稳定的语义记忆,"
    "并严格输出 JSON。\n"
    "字段固定为 semantic_summary, importance, stable_facts, recurring_topics, important_people, open_loops,"
    " memory_metadata, memory_title, catalog_hint, topic_headings。\n"
    "memory_title 是便于以后翻找这条长期记忆的简短具体标题;catalog_hint 用一句话说明它能回答什么;"
    "topic_headings 只列少量稳定主题短语。它们不能引入来源摘要里没有的事实。\n"
    "entity_anchors/topic_terms 从全部来源摘要提取并去重，优先保留有辨识度的名称和具体主题，不要只从总标题取词。\n"
    "stable_facts 要稳定、客观、适合长期保留;不要把角色设定写进去。\n"
    "recurring_topics 抓反复出现的话题;important_people 只留明显重要或反复出现的人;open_loops 记仍在推进的事项。\n"
    "只有来源摘要最新状态仍明确在推进的事项才能进入 open_loops;已清理、取消、不再需要或已经结束的事项不能进入 open_loops,"
    "旧失败也不能覆盖后来的关闭状态。\n"
    "不要编造摘要里没有的长期结论。importance 必须是 0.0 到 1.0 之间的数字。\n"
    "stable_facts/recurring_topics/important_people/open_loops/topic_headings 必须是 JSON 数组。"
    "只输出一个合法 JSON 对象。"
)

SEMANTIC_USER_TEMPLATE = (
    "请把下面这组更早的阶段摘要,再压缩成一条长期语义记忆:\n{source_text}\n\n"
    "要求:保留长期稳定事实、反复话题、重要人物和未完成线索;不要写成流水账。importance 只能是 0 到 1 的数字。"
)

REINFORCEMENT_SYSTEM = (
    "你就是当前角色,正在重新整理一条自己的长期记忆。你会收到一条已有长期语义记忆,以及一组新的阶段摘要压缩结果。\n"
    "如果它们明显属于同一长期主线,请输出一条融合后的长期语义记忆,严格输出 JSON,字段同语义记忆。\n"
    "同时更新 memory_title、catalog_hint 和 topic_headings,让标题覆盖融合后的长期主线,但不能引入新事实。\n"
    "尽量保留已有稳定事实,同时自然吸收新近重复出现的内容。不要因为新内容只出现一次就推翻旧的稳定印象。\n"
    "但状态更新必须以后来的明确记录为准:新内容若说明任务已清理、取消、不再需要或结束,"
    "应移除已有 open_loops 中对应待办,只可把它保留为历史经过。\n"
    "不要写成流水账。importance 必须是 0.0 到 1.0 之间的数字。各列表字段必须是 JSON 数组。只输出一个合法 JSON 对象。"
)

REINFORCEMENT_USER_TEMPLATE = (
    "已有长期语义记忆:\n{existing_text}\n\n新的阶段摘要压缩结果:\n{incoming_text}\n\n"
    "请输出融合后的长期记忆,尽量保留旧的稳定信息,同时吸收新的重复线索。"
)


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
