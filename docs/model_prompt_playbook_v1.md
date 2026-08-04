# Model Prompt Playbook V1

这篇文档给接入方用:当 memcore 的存储、检索、时间线、metadata 前置过滤都已经接好后,最终效果很大一部分取决于聊天模型是否被讲清楚。

memcore 不替宿主写完整人格 prompt,但建议把下面这些规则拼到最终聊天模型 system/developer prompt 中。

## 最小推荐提示词

```text
你可以看到 memcore 提供的可见三层记忆,并可使用记忆工具:
- retrieve_for_turn: 按语义/关键词/metadata 模糊检索长期或历史记忆,并排除当前 prompt 已经可见的记忆与本轮消息。
- read_timeline: 按 ISO 或本地 `start_at/end_at` 精确到小时/分钟读取原始对话，无需计算 epoch；也可按旧日期/粗时段读取，或用 retrieve 返回的 raw source_id 读取前后完整轮次。
- read_entry: 使用时间线返回的 raw source_id 展开一条紧凑工具/材料凭据；只在正文确实影响答案时调用。
- load_material: 若宿主支持图片/文件,按 file_id 读取当前可用的原文件、OCR、视觉描述、文档 chunks 或清理状态。

把工具当作你的可用能力和结构化信息通道,不是摆设。凡是答案依赖未在当前 prompt 中明确可见的事实、旧记忆、精确时间线、人物归因、承诺、偏好、关系或平台事件时,请主动调用合适的工具求证。一次工具结果不够时,可以根据结果继续调用工具补查,直到足以回答或确认没有明确记录。

当用户提到“昨天/今天/明天/上周/上周二/最近/刚才”等相对时间时,必须结合 prompt 里的日期、星期和时间段锚点理解。
如果需要精确日期或时间范围,优先调用 read_timeline;如果是偏好、计划、人物关系、长期事实等模糊问题,调用 retrieve_for_turn。
如果问题依赖“刚才那张图/之前那个文件/PDF 第几页”等材料内容,先从可见 raw 或 `read_timeline` 找到 file_id；需要检索显式材料轨迹时,使用宿主授权的 `retrieve_for_turn(include_explicit=true, kind_patterns=["material.*"])`,再调用 `load_material`。
非多模态接入里,只有当前轮 provider 请求已经附了原生图片输入,或 load_material/视觉工具返回了同一 file_id 的 OCR、视觉描述、文档 chunks 时,才可以回答材料内容。只有 file_id、filename、derived_status 或 pending 状态不算看到了材料,不要用旧附件或旧工具结果猜。
人格亲近感不能替代证据;不要为了显得记得、懂得或反应快而跳过工具编造。
历史对话、摘要、工具轨迹和材料轨迹只说明过去发生过什么,不自动构成当前待办。后出现的“已清理/已取消/不用了/已结束”优先于更早的失败、等待确认或待续描述。除非用户当前追问,或宿主当前任务工作区明确仍活跃,不要主动续报旧附件、旧转写、旧工具失败或旧交付请求。

不要重复检索当前 prompt 已经可见的记忆;宿主会用 retrieve_for_turn 排除可见三层和本轮消息。

如果处在群聊/多人场景,请保留“谁说的、谁的偏好、谁的计划、谁的承诺”。不要把不同发言人的事实混写成“用户说/大家说”。
回答“谁说的/谁戳的/谁答应的/谁负责的”这类归因问题时,必须以当前可见原文或工具结果中的发言人/事件记录为依据。没有明确记录时,请先调用 read_timeline 或 retrieve;仍没有证据就说明没看到明确记录,不要猜名字。

最终回复若启用 memcore_json,只在所有工具调用完成后输出 JSON:
{"speech":"给用户看的回复","memory_metadata":{...}}
memory_metadata 标注的是宿主指定的本轮记忆目标,可能是用户消息或触发回复的外部事件,不是你的回复。它用于后续检索前置过滤,不要展示给用户。
```

## 缓存友好 Prompt 布局

许多模型服务会对重复输入前缀做缓存(例如按完整前缀单元命中)。接入 memcore 时,要让稳定内容形成尽可能长、尽可能不变的前缀,把每轮变化的记忆和用户消息放到后面。

推荐顺序:

```text
1. system/developer: 固定人格、合规边界、工具使用总规则
2. system/developer: 固定 memcore 工具说明、metadata 标注规则、JSON 输出契约
3. system/developer 或 user: 固定领域规则/长期不变的大段资料(如同一份研报/财报原文)
4. user/developer: 本轮动态时间锚点、render_prompt_context(ctx) 的可见三层记忆
5. user: 本轮用户输入
6. tool result: retrieve_for_turn / read_timeline 的工具返回
7. assistant final: 最终回复;若启用 memcore_json,只在这里输出 JSON
```

注意:

- 固定前缀要保持字面稳定:顺序、空格、换行、字段名、工具 schema 不要每轮重排。
- 不要把当前时间、可见三层记忆、检索结果、随机 request id 混进固定 system prompt 前部。
- `render_prompt_context(ctx)` 每轮都会变,应放在稳定规则之后;它的存在不影响前面的固定规则命中缓存。
- 长文本问答里,若同一份财报/研报会被连续追问,把原文放在问题之前并保持完全一致;后续问题只追加在原文之后。
- 需要变更人格、工具契约或领域枚举时,把它当成 prompt 版本升级;版本稳定后不要频繁微调标点。

## 时间锚点

memcore 会把 raw、summary、semantic、timeline 渲染成带日期和星期的文本,例如:

```text
[日期 2026-04-10 周五]
[09:00 | 上午] user(张三;id=qq-1): 上周二那份报告我还没看完
```

模型应理解:

- “上周二”要以这条消息自己的时间锚点 `2026-04-10 周五` 为参照。
- 摘要或长期记忆里不要留下未锚定的“昨天/上周/最近”;需要写入记忆时改成绝对日期或日期范围。
- 用户问精确时间问题时,`read_timeline` 比向量检索更可靠。
- 用户问“我是不是说过喜欢什么/计划过什么/谁负责什么”这类模糊事实时,`retrieve_for_turn` 更合适。

## 工具选择

推荐给聊天模型的工具说明:

```text
retrieve_for_turn(query, entity_anchors?, topic_terms?, source_layers?, memory_facets?, about_roles?, time_hint?)
用于模糊检索。query 始终参与检索；准确且已知的实体使用 entity_anchors，正在询问的未知人物/答案不能先猜成 anchor；动作、关系、属性和主题使用 topic_terms。已知具体时间时可传 time_hint={start_at,end_at}，使用与 read_timeline 相同的本地/ISO 时间规则，并在相似度计算前硬过滤；memory_facets/about_roles 也会在评分前裁剪候选。

read_timeline(time_range?, date_from?, date_to?, time_periods?, anchor_source_id?, before_turns?, after_turns?, projection?, page_token_budget?, cursor?)
time_range 使用 start_at/end_at 做起点包含、终点不包含的精确读取；日期模式读取整天或粗时段；anchor 模式从一条 raw 命中扩展前后完整 turn。conversation/full/tools 决定读取密度。默认不分页；只有调用者显式提供页面预算时才返回 continuation。若 coverage.complete=false，下一次只传 cursor，选择器、投影和预算都已封装其中。

read_entry(source_id, detail?)
用于展开 read_timeline conversation 视图里的紧凑 raw 凭据。当前会话之外、summary 或 semantic ID 都不会被当作 raw 返回。

load_material(file_id, kind?, preferred_source?, purpose?)
用于读取宿主保存的图片/附件/PDF 当前可用内容或状态。先从可见 raw、`read_timeline` 或宿主授权的 `retrieve_for_turn(include_explicit=true, kind_patterns=["material.*"])` 找到 file_id,再调用它。
```

常见选择:

| 用户意图 | 推荐工具 |
|---|---|
| “昨天晚上我说了什么?” | `read_timeline(date_from=昨天日期,time_periods=["night"])` |
| “昨天 11 点到 12 点和我一起来的是谁?” | `read_timeline(time_range={"start_at":"昨天日期 11:00","end_at":"昨天日期 12:00"})`；从原话识别人名，不猜未知 entity anchor |
| “大约那一小时、和 misaka 同行的人是谁，但还需按关系模糊找?” | `retrieve_for_turn(query="和 misaka 一起来玩的另一个人", entity_anchors=["misaka"], topic_terms=["同行","一起来玩"], time_hint={"start_at":"日期 11:00","end_at":"日期 12:00"})`；不填写未知答案的人名 |
| “我之前是不是说过喜欢可乐?” | `retrieve_for_turn(query="喜欢 可乐", entity_anchors=["可乐"], memory_facets=["preference"], about_roles=["user"])` |
| “上周二那件事后来怎么样了?” | 先用时间锚点算日期,再 `read_timeline`;必要时补 `retrieve_for_turn` |
| “谁负责基金复盘?” | 群聊场景优先带人物/计划关键词 `retrieve_for_turn`,必要时读时间线 |
| “刚才谁戳你了?” | 优先 `read_timeline` 读取最近/当天平台事件;只根据明确事件记录回答 |
| “之前那张图/PDF 里是什么?” | 先从可见上下文、时间线或授权的显式 `material.*` 检索拿 `file_id`,再 `load_material(file_id=..., preferred_source="derived")` |
| “你还记得我生日吗?” | 当前可见记忆没有明确生日时必须 `retrieve_for_turn`,查不到就说没看到明确记录 |

## 结构化上下文与工具通道

模型服务原生 tool calling / tool result 通常能直接保留调用 ID、参数和返回值边界，
但 MemCore 不要求宿主必须使用它。宿主也可以使用稳定的 JSON、XML、标签或其他
结构化协议，只要宿主负责解析、权限和执行，并用明确 `correlation_id` 把模型动作
与系统结果追加到同一 Timeline。不要把结果伪装成用户本人说的话。MemCore 提供
`build_native_memory_tool_specs(...)` / `dispatch_native_memory_tool(...)` 作为可选原生
工具辅助，也提供 `append_action(...)` / `append_observation(...)` 记录通用动作和结果。

推荐分层:

1. `tool_use/tool_result`:当前轮工具调用的结构化通道。模型能区分工具结果和用户文本,也能关联结果属于哪次调用。
2. operation entries:宿主把工具调用/结果作为同一开放 turn 的 action/observation，用 `correlation_id` 关联，用于“刚才那个搜索结果/上次读的文件”类追问。它们参与统一 token 生命周期并压缩为 operation digest；普通检索默认排除，只有显式授权与 kind pattern 才检索。
3. material entries:宿主可把图片/文件上传、解析状态、清理状态追加为当前 turn intermediate 或 typed standalone entry。只记录 file_id、文件名、类型和状态；文件本体与 OCR/视觉描述/文档 chunks 留在宿主存储。材料进入 operation 分区，普通检索默认排除。
4. `render_prompt_context(ctx)`:memcore 的可见 raw/summary/semantic 记忆,用于长期连续性和可见上下文。
5. `retrieve_for_turn/read_timeline/load_material`:需要更多记忆证据或材料内容时由模型主动调用。

如果使用文本协议，请用稳定结构明确标出动作、结果、调用 ID 和时间，并告诉模型
这段来自宿主环境而非用户原话。实际 provider messages 应通过
`record_request_projection(...)` 冻结，避免下一轮被默认投影换成另一种表示。协议选择
由宿主和模型能力决定，MemCore 不据此改变存储、检索或缓存边界。

工具结果写入 raw 时，优先使用 `append_action()` / `append_observation()`；
`record_tool_exchange(turn_id=...)` 是等价薄适配。不要创建无 turn_id 的 flat 工具行。

```python
mem.record_tool_exchange(
    turn_id=handle.turn_id,
    tool_name="web_search",
    tool_call_id="call_001",
    tool_input={"query": "北京天气"},
    result="北京今天 25°C,晴天。",
    timestamp=now_ts,
)
```

它会写入带同一 `turn_id` 与 `correlation_id` 的 action/observation，并按 provider
profile 渲染成稳定 tool call/result；并行结果乱序也不会串线。

```text
[20:31 | 晚上] assistant.tool_call web_search call_001
input:
{"query": "北京天气"}

[20:31 | 晚上] tool.web_search call_001
source: web_search
output:
北京今天 25°C,晴天。
```

这样工具使用与工具返回仍在同一条线性事件流里，但边界由 typed role 和调用 ID
确定；工具身份不伪装成语义 metadata。

材料引用写入 raw 时,优先使用 `record_material_reference(...)`;材料被容量策略或用户操作清理时,使用 `record_material_cleanup(...)`。这两个事件不会保存文件本体:

```python
mem.record_material_reference(
    file_id="file_img_001",
    kind="image",
    actor=Actor(stable_id="qq-1", display_name="张三"),  # 群聊/多人上传时保留上传者归因
    filename="photo.jpg",
    mime_type="image/jpeg",
    file_status="ready",
    derived_status="ocr_ready",
    timestamp=now_ts,
    topic_terms=["题目图片"],
)
```

材料身份和显式检索依据是 `kind=material.*`、visibility 与宿主授权，不占用
`memory_facets`。可见 raw 中渲染为稳定块:

```text
[20:30 | 晚上] user.attachment image file_img_001
source: attachment
file_id: file_img_001
kind: image
filename: photo.jpg
mime: image/jpeg
file_status: ready
derived_status: ocr_ready
```

多模态模型当前轮需要看图时,宿主可以在 provider 请求里直接附图片;非多模态模型则应先等待视觉/OCR/文档解析完成,或让 `load_material` 返回明确的 pending/unavailable 状态,不要让最终聊天模型和视觉模型赛跑。历史追问时,模型先通过可见 raw、`read_timeline` 或宿主授权的 `retrieve_for_turn(include_explicit=true, kind_patterns=["material.*"])` 找到材料锚点,再调用宿主暴露的 `load_material` 工具读取可用内容。若材料已清理且没有保留解析结果,模型必须说明无法确认,不要假装看到了原文件。

## 工具使用纪律

人格可以自然亲近,但不能把亲近感当作事实来源。聊天模型应自主判断是否需要工具,并遵守下面的通用纪律:

1. 先判断当前可见 raw/summary/semantic 是否足以支持回答。
2. 如果答案依赖不可见或不确定的信息,主动调用工具;不要等用户追问“你查了吗”。
3. 精确时间线、最近事件、平台事件优先 `read_timeline`;模糊长期事实、偏好、关系、计划、承诺优先 `retrieve_for_turn`。
4. 工具结果不足时,可以基于已得到的线索继续调用另一个工具或换查询词补查。
5. 工具结果仍没有明确证据时,回复“我没有看到明确记录/我不确定”,可以邀请用户补充,但不要猜一个看似合理的答案。
6. 如果用户随后纠正,按用户纠正更新本轮 metadata;不要把前一轮未证实的猜测当成事实继续强化。

## memory_metadata 标注

`memory_metadata` 是检索前置过滤的上游信号。字段越稳,后续检索越准、越省计算。

```json
{
  "turn_intent": "",
  "memory_facets": ["preference"],
  "about_roles": ["user"],
  "entity_anchors": ["可乐", "无糖可乐"],
  "topic_terms": ["饮料", "偏好"],
  "retrieval_priority": "high",
  "mood_tags": []
}
```

标注原则:

- `turn_intent`: 只有当前内容在查询历史记忆时写 `memory_query`；它不表示想检索的历史内容类别。
- `memory_facets`: 内容未来能回答哪类问题，从 `profile / preference / viewpoint / relationship / event / state / plan / decision / constraint / knowledge / procedure` 中选择。不确定就留空，不要猜一个大桶。
- `about_roles`: 内容主要在陈述 `user / assistant / third_party / external` 中的谁或什么，不表示谁参加了对话，也不自动等于发言者。
- `entity_anchors`: 只填问题和上下文中已经明确知道的准确名称、项目名和别名，如 `Fable / 雅可比猜想 / NVDA`；正在询问“是谁”的未知对象不是 anchor，不要猜测答案后填写，也不要机械补宽泛上位词。
- `topic_terms`: 填动作、属性和辅助主题短词，如 `反例 / 风险 / 无糖`；不要写整句或短句。
- `retrieval_priority`: 未来重新找回的价值，使用 `low / normal / high / critical`。它不决定是否入库，也不会让低值 raw 消失。
- `mood_tags`: 只有 `enable_flavor=True` 时才写;关闭时必须空数组。

检索时传入的是“想找的历史内容”。例如当前消息“我们聊过 Fable 吗”可标为
`turn_intent=memory_query`，但工具调用应使用 `memory_facets=["knowledge"]`、
`about_roles=["external"]`、`entity_anchors=["Fable"]`，不能拿 `memory_query` 去过滤历史。

## 群聊 / 多人

memcore 的硬隔离是 `tenant_id / user_id / domain_id`;`actor` 是同一记忆池里的发言人软标签。

接入方要做:

- 群聊消息调用 `record_user_turn(..., actor=Actor(stable_id=平台稳定ID, display_name=当前昵称))`。
- 稳定 ID 用平台不会变的 ID;昵称只做显示。
- 最终 prompt 告诉模型保留发言人归因。

模型要做:

- 看见 `user(张三;id=qq-1): ...` 时,记成“qq-1/张三说、计划或偏好”,不要写成“用户都喜欢”。
- 逐行绑定稳定 ID、昵称和内容。稳定 ID 不同的人不要因为昵称、相邻消息、上下文语气或同一话题被合并。
- 多人出现相同偏好时可以合并,但要保留人名或主体。
- 不要把其他人的风险偏好、资产偏好、任务承诺归到当前用户身上。
- 对戳一戳、撤回、入群、改名等平台事件,优先相信结构化事件记录;没有事件记录时不要凭聊天语气推断是谁。

## JSON 输出边界

如果启用 `memcore_json`,只让最终回复使用 JSON 契约。工具调用阶段仍走宿主模型工具机制。

正确流程:

1. 用户消息先 `record_user_turn`。
2. 宿主拼可见三层 + 工具说明 + 输出契约。
3. 模型按需调用 `retrieve` / `read_timeline`。
4. 工具结果回到模型。
5. 模型输出最终 JSON。
6. adapter 解析 `speech`,并把 `memory_metadata` 回写本轮 raw。

不要让工具调用中间步骤输出 memcore JSON,也不要把 `memory_metadata` 当作给用户看的内容。
