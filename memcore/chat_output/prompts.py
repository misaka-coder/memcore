"""Prompt snippets for the optional memcore chat-output contract."""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import DEFAULT_CATEGORIES, MOOD_TAGS


def build_chat_output_contract_prompt(
    *,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    enable_flavor: bool = False,
    enable_sentence_segments: bool = False,
) -> str:
    cats = " / ".join(str(item).strip() for item in categories if str(item).strip())
    mood_line = (
        "mood_tags 可写 0-3 个,只能从固定枚举选择:" + " / ".join(MOOD_TAGS) + "。"
        if enable_flavor
        else "mood_tags 必须输出为空数组。"
    )
    lines = [
        "请只输出一个合法 JSON 对象,不要输出代码块或解释。",
        "字段固定为 speech, memory_metadata。",
        "speech 是给用户看的最终回复,必须是字符串。",
        "memory_metadata 用于本轮用户原始消息的记忆检索标注,不要把它当作给用户看的内容。",
        "工具调用阶段不适用本 JSON 契约;需要调用工具时请正常使用宿主项目的工具调用机制。",
        "只有在所有工具调用完成、准备给用户最终回复时,才按本契约只输出一个合法 JSON 对象。",
        "memory_metadata 字段固定为 keywords, subject_scopes, categories, mood_tags, importance, confidence。",
        "keywords 最多 4 个可复用检索标签,按用户未来正常聊天里可能命中的问法选词;例如可乐可补饮料/偏好,但不要机械补太宽泛的上位词。不要写整句或短句。",
        "subject_scopes 标注本轮原始消息涉及的事实主体,只能从 user/assistant/other 中选择;群聊中不要把别人的事实归到 user。",
        f"categories 只能从当前枚举选择:{cats}。",
        "importance 表示这条原始消息未来是否值得检索,confidence 表示你对 metadata 标注的把握;都必须是 0.0 到 1.0 的数字。",
        "如果用户提到“昨天/上周/上周二/最近”等相对时间,请结合 prompt 中的日期与星期锚点理解;需要精确日期范围时优先调用 read_timeline,需要模糊事实时调用 retrieve。",
        "多方/群聊场景请保留谁说的、谁的偏好、谁的计划。若宿主传入 actor/昵称信息,不要把不同发言人的事实混成同一个人。",
        mood_line,
        "不确定 metadata 时可以输出空数组和较低 confidence,不要为了填字段编造标签。",
    ]
    if enable_sentence_segments:
        lines.append(
            "请把 speech 写成自然短句。每个完整句子请用 。！？.!? 或换行结尾,"
            "方便系统按句分段展示或播放。不要为了分段把同一句话硬拆碎。"
        )
    return "\n".join(lines)
