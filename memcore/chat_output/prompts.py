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
        "keywords 最多 4 个短词;subject_scopes 只能从 user/assistant/other 中选择。",
        f"categories 只能从当前枚举选择:{cats}。",
        mood_line,
        "importance/confidence 必须是 0.0 到 1.0 的数字。",
    ]
    if enable_sentence_segments:
        lines.append(
            "请把 speech 写成自然短句。每个完整句子请用 。！？.!? 或换行结尾,"
            "方便系统按句分段展示或播放。不要为了分段把同一句话硬拆碎。"
        )
    return "\n".join(lines)
