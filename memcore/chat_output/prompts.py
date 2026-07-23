"""Prompt snippets for the optional memcore chat-output contract."""

from __future__ import annotations

from ..schema import build_memory_metadata_instruction


def build_chat_output_contract_prompt(
    *,
    enable_flavor: bool = False,
    enable_sentence_segments: bool = False,
) -> str:
    lines = [
        "请只输出一个合法 JSON 对象,不要输出代码块或解释。",
        "字段固定为 speech, memory_metadata。",
        "speech 是给用户看的最终回复,必须是字符串。",
        "memory_metadata 用于宿主指定的本轮记忆标注目标,目标可能是用户消息或触发回复的外部事件;不要把它当作给用户看的内容。",
        "工具调用阶段不适用本 JSON 契约;需要调用工具时请正常使用宿主项目的工具调用机制。",
        "只有在所有工具调用完成、准备给用户最终回复时,才按本契约只输出一个合法 JSON 对象。",
        "把可用工具当作你的能力和结构化信息通道,不是摆设。凡是答案依赖当前 prompt 没有明确给出的旧记忆、精确时间线、人物归因、偏好、关系、承诺或平台事件时,请主动调用合适工具;一次结果不够时可以继续补查。",
        "工具结果应保留工具调用边界,不要伪装成普通用户文本;legacy 文本 followup 只应作为兼容方案。",
        "memory_metadata 字段固定为 turn_intent, memory_facets, about_roles, entity_anchors, topic_terms, retrieval_priority, mood_tags。",
        build_memory_metadata_instruction(
            enable_flavor=enable_flavor,
            require_disabled_mood_field=True,
        ),
        "如果用户提到“昨天/上周/上周二/最近”等相对时间,请结合 prompt 中的日期与星期锚点理解;需要精确日期范围时优先调用 read_timeline,需要模糊事实时调用 retrieve。",
        '如果用户追问历史图片、附件或 PDF 内容,先从可见上下文或时间线找 file_id;需要检索显式材料轨迹时,使用宿主授权的 retrieve_for_turn(include_explicit=true, kind_patterns=["material.*"]),再调用 load_material 读取当前可用内容或清理状态。',
        "非多模态接入中,当前可见上下文里只要有明确绑定同一 file_id 的 OCR、视觉描述、文档正文或其它 derived 内容,无论来自当前轮还是历史工具结果,都可以据此回答并说明证据范围。只有 file_id、filename、derived_status 或 pending 状态不算看到了内容;内容缺失、已清理、已过期、归属有歧义,或问题必须重新读取原件细节时,再调用 load_material/视觉工具,不要拿其它 file_id 的材料猜。",
        "历史对话、摘要、工具轨迹和材料轨迹只说明过去发生过什么,不自动构成当前待办;后出现的已清理/已取消/不用了/已结束优先于更早的失败或等待确认。除非用户当前追问,或宿主当前任务工作区明确仍活跃,不要主动续报旧附件、旧转写、旧工具失败或旧交付请求。",
        "多方/群聊场景请保留谁说的、谁的偏好、谁的计划。若宿主传入 actor/昵称/稳定ID 信息,稳定ID相同才视为同一人,不要把不同发言人的事实混成同一个人。",
        "回答谁说的、谁戳的、谁答应的、谁负责的这类归因问题时,必须依据可见原文或工具结果;没有明确记录就说没看到明确记录,不要猜名字。",
        "若当前可见记忆没有明确证据,工具仍无证据时说没有看到明确记录,不要为了显得记得而编造。",
    ]
    if enable_sentence_segments:
        lines.append(
            "请把 speech 写成自然短句。每个完整句子请用 。！？.!? 或换行结尾,"
            "方便系统按句分段展示或播放。不要为了分段把同一句话硬拆碎。"
        )
    return "\n".join(lines)
