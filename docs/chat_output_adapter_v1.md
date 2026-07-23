# Chat Output Adapter V1

Chat Output Adapter 是 MemCore 的可选最终回复契约。它只负责把聊天模型最终输出拆成：

- 给用户看的 `speech`；
- 写回本轮标注目标的 `memory_metadata`。

工具调用、权限、文件加载和 provider 消息循环仍由宿主负责。中间工具轮不能套最终 JSON 契约。

## 最终 JSON

```json
{
  "speech": "记得，你之前说过喜欢无糖可乐。",
  "memory_metadata": {
    "turn_intent": "memory_query",
    "memory_facets": [],
    "about_roles": ["user"],
    "entity_anchors": ["可乐", "无糖可乐"],
    "topic_terms": ["饮料", "偏好"],
    "retrieval_priority": "low",
    "mood_tags": []
  }
}
```

`memory_metadata` 标注宿主指定的本轮目标，通常是触发回复的用户消息或外部事件，不是 assistant 回复本身。

字段含义：

- `turn_intent`：仅在当前内容明确查询历史记忆时填 `memory_query`，普通内容为空。
- `memory_facets`：内容未来能回答哪类问题；固定枚举为 `profile / preference / viewpoint / relationship / event / state / plan / decision / constraint / knowledge / procedure`。
- `about_roles`：内容主要在陈述 `user / assistant / third_party / external` 中的谁或什么，不表示谁发了消息。
- `entity_anchors`：明确出现或能够确定的准确名称与别名。
- `topic_terms`：动作、属性和辅助主题短词，不写整句。
- `retrieval_priority`：`low / normal / high / critical`，只表达未来找回价值，不决定是否保存 raw。
- `mood_tags`：只有 flavor 开启时从固定枚举选择；关闭时为空数组。

旧 `keywords / subject_scopes / categories / importance / confidence` 不再是运行时写入接口。旧 SQLite 数据只在 schema 迁移时转换一次。

## 生成提示词

```python
from memcore import build_chat_output_contract_prompt

contract = build_chat_output_contract_prompt(
    enable_flavor=False,
    enable_sentence_segments=True,
)
```

提示词与摘要、长期记忆共用 `memcore.schema` 的同一字段说明，宿主不应再复制另一份枚举。`enable_sentence_segments` 只引导自然句末，不能要求模型机械拆句。

推荐 prompt 顺序：

```text
稳定人格/安全规则
稳定工具 schema 与 Chat Output 契约
稳定领域说明
动态时间锚点与 MemCore 可见三层
当前用户消息
原生 tool result（按需）
最终 JSON
```

工具 schema、字段顺序和固定说明应保持字面稳定；当前时间、可见记忆、附件和请求 ID 放在稳定前缀之后。

## 非流式解析

```python
from memcore import parse_chat_output

parsed = parse_chat_output(
    raw_model_output,
    mode="memcore_json",
    enable_flavor=False,
)

if not parsed.ok:
    handle_error(status=parsed.status, reason=parsed.reason)
else:
    completed = mem.complete_turn(
        turn_id=handle.turn_id,
        semantic_text=parsed.speech,
        provider_output_raw=raw_model_output,
        memory_annotation=parsed.memory_metadata,
        annotation_status="accepted_model",
    )
```

支持的模式：

- `memcore_json`：严格要求完整 JSON 和 `speech`。
- `legacy_text`：把普通文本作为 speech；只用于宿主明确选择的迁移路径。

解析失败必须返回结构化 `status/reason`。不得把损坏 JSON 原样发给用户，也不得伪造一份“成功解析”的空 metadata。

校验行为：

- 枚举外值丢弃；自由字符串 trim、按首次出现去重。
- 非法 `turn_intent` 变为空；非法 priority 回到 `normal`。
- flavor 关闭时清空 `mood_tags`。
- 不读取旧 metadata 键，也不做隐藏字符/token 截断。

## 流式 speech

```python
from memcore import ChatOutputStreamParser

parser = ChatOutputStreamParser(mode="memcore_json", enable_flavor=False)
for chunk in provider_chunks:
    for event in parser.feed(chunk):
        if event.type == "speech_delta":
            render_or_play(event.text)

for event in parser.finish():
    if event.type == "metadata":
        final_metadata = event.memory_metadata
```

流式解析器只在确定内容属于 JSON `speech` 字符串后发出增量；转义、半个 Unicode 序列和不完整 JSON 都必须等后续 chunk。最终 metadata 只在完整对象校验通过后交给宿主。

若流在已经展示部分 speech 后失败，宿主应报告真实的 partial/failed 状态，并按产品策略决定是否保留已展示文本；不能把失败伪装成完整成功。

## 与工具调用的边界

正确循环：

1. 模型按 provider 原生工具或宿主自定义协议发出 action。
2. 宿主执行并把 observation 追加到当前 turn。
3. 模型可以继续调用更多工具。
4. 所有工具完成后，模型只输出一次 Chat Output JSON。
5. adapter 解析最终 speech/metadata，`complete_turn()` 原子提交最终回复与标注。

MemCore 可以记录原生工具、JSON、XML、标签和 Skill 请求/结果，但不会解析或执行宿主协议。动作和结果通过 `kind / turn_role / correlation_id` 保持边界，不伪装成用户消息，也不塞进 metadata facet。

## 最小验收

接入至少覆盖：

- 合法 JSON、缺失 speech、损坏 JSON、legacy_text；
- flavor 开/关及非法 metadata 枚举；
- 多 chunk 转义、中文、句末和流结束；
- 工具轮不会提前触发最终 JSON parser；
- 解析失败不会关闭 turn 或写入错误 metadata；
- 用户实际只看到干净 speech，不看到 metadata、代码块或内部错误对象。

可运行闭环见 [`../examples/minimal_chat_integration.py`](../examples/minimal_chat_integration.py)。
