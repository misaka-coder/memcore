# Model Prompt Playbook V1

这篇文档给接入方用:当 memcore 的存储、检索、时间线、metadata 前置过滤都已经接好后,最终效果很大一部分取决于聊天模型是否被讲清楚。

memcore 不替宿主写完整人格 prompt,但建议把下面这些规则拼到最终聊天模型 system/developer prompt 中。

## 最小推荐提示词

```text
你可以看到 memcore 提供的可见三层记忆,并可使用两个记忆工具:
- retrieve: 按语义/关键词/metadata 模糊检索长期或历史记忆。
- read_timeline: 按日期/时间段精确读取原始对话,适合处理“昨天/上周二/4月10日晚上”等时间问题。

当用户提到“昨天/今天/明天/上周/上周二/最近/刚才”等相对时间时,必须结合 prompt 里的日期、星期和时间段锚点理解。
如果需要精确日期或时间范围,优先调用 read_timeline;如果是偏好、计划、人物关系、长期事实等模糊问题,调用 retrieve。

不要重复检索当前 prompt 已经可见的记忆;宿主会用 retrieve_for_turn 排除可见三层和本轮消息。

如果处在群聊/多人场景,请保留“谁说的、谁的偏好、谁的计划、谁的承诺”。不要把不同发言人的事实混写成“用户说/大家说”。

最终回复若启用 memcore_json,只在所有工具调用完成后输出 JSON:
{"speech":"给用户看的回复","memory_metadata":{...}}
memory_metadata 标注的是本轮用户原始消息,不是你的回复。它用于后续检索前置过滤,不要展示给用户。
```

## 时间锚点

memcore 会把 raw、summary、semantic、timeline 渲染成带日期和星期的文本,例如:

```text
[日期 2026-04-10 周五]
[09:00 | 上午] user(张三): 上周二那份报告我还没看完
```

模型应理解:

- “上周二”要以这条消息自己的时间锚点 `2026-04-10 周五` 为参照。
- 摘要或长期记忆里不要留下未锚定的“昨天/上周/最近”;需要写入记忆时改成绝对日期或日期范围。
- 用户问精确时间问题时,`read_timeline` 比向量检索更可靠。
- 用户问“我是不是说过喜欢什么/计划过什么/谁负责什么”这类模糊事实时,`retrieve` 更合适。

## 工具选择

推荐给聊天模型的工具说明:

```text
retrieve(query, keywords?, source_layers?, categories?, subject_scopes?, importance_min?, time_hint?)
用于模糊检索。可以传 categories/subject_scopes/importance_min 缩小候选,系统会先做 metadata 前置过滤再算相似度。

read_timeline(date_from, date_to?, time_periods?)
用于精确读取某天或日期范围的原始对话。日期必须是 YYYY-MM-DD。
```

常见选择:

| 用户意图 | 推荐工具 |
|---|---|
| “昨天晚上我说了什么?” | `read_timeline(date_from=昨天日期,time_periods=["night"])` |
| “我之前是不是说过喜欢可乐?” | `retrieve(query="喜欢 可乐", categories=["preference"], subject_scopes=["user"])` |
| “上周二那件事后来怎么样了?” | 先用时间锚点算日期,再 `read_timeline`;必要时补 `retrieve` |
| “谁负责基金复盘?” | 群聊场景优先带人物/计划关键词 `retrieve`,必要时读时间线 |

## memory_metadata 标注

`memory_metadata` 是检索前置过滤的上游信号。字段越稳,后续检索越准、越省计算。

```json
{
  "keywords": ["可乐", "饮料"],
  "subject_scopes": ["user"],
  "categories": ["preference"],
  "mood_tags": [],
  "importance": 0.7,
  "confidence": 0.9
}
```

标注原则:

- `keywords`: 0-4 个短词,写实体、主题、计划、偏好词,不要写整句。
- `subject_scopes`: 事实主体。用户自己的偏好/计划用 `user`;助手自己的设定或承诺用 `assistant`;群聊其他人用 `other`。
- `categories`: 必须从当前 `MemoryConfig.categories` 枚举里选。金融领域应换成稳定领域枚举,如 `risk_profile / investment_goal / asset_preference / compliance_preference`。
- `importance`: 未来是否值得检索。闲聊寒暄低,稳定偏好/身份/计划/风险约束高。
- `confidence`: 模型对自己标注是否准确的把握。不确定就低分,不要硬填。
- `mood_tags`: 只有 `enable_flavor=True` 时才写;关闭时必须空数组。

## 群聊 / 多人

memcore 的硬隔离是 `tenant_id / user_id / domain_id`;`actor` 是同一记忆池里的发言人软标签。

接入方要做:

- 群聊消息调用 `record_user_turn(..., actor=Actor(stable_id=平台稳定ID, display_name=当前昵称))`。
- 稳定 ID 用平台不会变的 ID;昵称只做显示。
- 最终 prompt 告诉模型保留发言人归因。

模型要做:

- 看见 `user(张三): ...` 时,记成“张三说/张三计划/张三偏好”,不要写成“用户都喜欢”。
- 多人出现相同偏好时可以合并,但要保留人名或主体。
- 不要把其他人的风险偏好、资产偏好、任务承诺归到当前用户身上。

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

