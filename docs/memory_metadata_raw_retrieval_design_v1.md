# MemCore 元数据与 Raw 主检索方案 V1

> 状态：MemCore package 与 Akane 宿主薄适配均已实现。MemCore 全量 320/320、Akane 记忆真实链路 282/282 通过。
>
> 日期：2026-07-23

## 1. 目标

本方案解决两个直接影响使用体验的问题：

1. 模型无法稳定理解现有 `categories / subject_scopes / keywords / confidence` 的含义，导致写入标签和检索参数采用不同口径。
2. 三层记忆在检索中职责不清，摘要和长期记忆可能与原始对话争抢结果名额，却不能像原始对话一样可靠扩展附近时间线。

目标不是建立包罗万象的知识本体，而是提供一套模型容易标注、容易反向查询、能真实裁剪检索候选的元数据协议。

核心结论：

```text
raw 原始对话是语义检索、元数据前置过滤和时间线扩窗的主力。
阶段摘要是 raw 的短期压缩表示，只作为可见上下文和辅助检索结果。
长期记忆是固定窗口内的常驻信息，只作为可见上下文和辅助检索结果。
只有 raw 检索结果可以作为时间线扩窗锚点。
```

## 2. 设计原则

### 2.1 保留前置过滤

Namespace、会话授权、可见性、kind、时间、删除状态、lineage、索引 generation 等安全边界继续在评分前硬过滤。

模型传入的记忆维度同样用于评分前候选裁剪。不能先对全库计算向量/BM25，再在结果末尾过滤元数据。

### 2.2 不让一个枚举解释所有信息

单一 `categories` 无法同时稳定表达“内容性质、内容关于谁、涉及什么实体、属于什么话题”。强行这样做最终会产生 `fact / casual / project_work` 一类巨大兜底桶。

目标元数据拆成四个互补问题：

1. 这段内容以后能回答哪类问题？
2. 内容主要在讲谁或什么？
3. 出现了哪些准确名称？
4. 还有哪些辅助主题词？

### 2.3 不为每层记忆补齐同等能力

摘要和长期记忆可以直接回答时就使用；不足以回答时不沿 lineage 强行下钻原文，也不增加摘要内部重排、多级锚点和歧义分支。

需要准确证据和附近上下文时，应依赖 raw 的正常语义检索结果。

### 2.4 提示词短、稳定、单一来源

写入标注、阶段摘要、长期记忆和检索工具必须引用同一份字段契约。工具 schema、字段枚举和提示词说明不能分别手写成多套口径。

提示词只解释模型需要作出的判断，不向模型暴露 lineage、索引 flag、压缩 generation 等内部实现。

### 2.5 先验证能力，再增加限制

新元数据、raw 主检索和时间线配合尚未通过真实链路验收前，不新增分层结果配额、锚点扩窗 token 配额或其他为了“可控”而预设的隐藏限制。

只有真实运行已经证明存在明确的成本、延迟或上下文溢出问题时，才能针对该问题提出限制；限制必须可观测、可配置，并在截断时返回结构化状态，不能静默损失信息。

现有 raw token 差值压缩属于已经运行的基础记忆生命周期，不是本方案新增的检索限制。模型供应商的物理上下文上限由宿主传输层负责，不能反过来成为 MemCore 在效果尚未验证前缩小检索和扩窗能力的理由。

## 3. 三层记忆职责

| 层 | 主要职责 | 语义检索角色 | 能否时间线扩窗 |
| --- | --- | --- | --- |
| `raw` | 精确保存真实消息、事件与最终回复 | 主结果 | 可以 |
| `summary` | 压缩一段旧 raw，短期保留更多信息 | 辅助结果，内容够用就直接使用 | 不可以 |
| `semantic_summary` | 固定窗口内常驻的长期信息 | 辅助结果，内容够用就直接使用 | 不可以 |

三层结果不能直接混成一个总榜，让更长、更概括的摘要挤掉 raw。

检索应先形成并返回 raw 主结果，再附加 derived 辅助结果；初版沿用已有调用结果数量，不新增分层硬配额。具体是否需要额外限制，必须等真实链路证明存在问题后再讨论。

同一 lineage 的多个层同时命中时，普通记忆查询优先保留 raw，并去掉代表相同来源的 summary/semantic 副本。

## 4. 目标元数据契约

```json
{
  "turn_intent": "",
  "memory_facets": [],
  "about_roles": [],
  "entity_anchors": [],
  "topic_terms": [],
  "retrieval_priority": "normal",
  "mood_tags": []
}
```

`turn_intent` 属于本轮标注信息，但不进入历史内容 facets；当前只定义可选值 `memory_query`，普通消息留空。

`metadata_confidence` 不进入目标契约。

系统生成的以下字段也不允许模型填写：

```text
namespace / conversation / kind / visibility / timestamp
source_id / turn_id / correlation_id / lineage
annotation_status / retrieval_policy / index generation
```

### 4.1 `memory_facets`

`memory_facets` 回答“这段内容以后能回答哪类问题”。它允许少量多选，但不要求每条消息必须填写。

固定枚举建议如下：

| 枚举 | 明确定义 | 典型问题 |
| --- | --- | --- |
| `profile` | 主体相对稳定的身份、背景、属性或能力 | “我是什么专业？” |
| `preference` | 喜欢、讨厌、习惯或表达风格偏好 | “我喜欢喝什么？” |
| `viewpoint` | 主体对某件事的看法、评价或判断 | “我之前怎么看这个方案？” |
| `relationship` | 主体之间的关系、角色、归属或绑定 | “谁是群管理员？” |
| `event` | 已发生或正在发生的事情、动作或经历 | “什么时候任命的？” |
| `state` | 某个时点或阶段的状态 | “Bot 当时是不是离线？” |
| `plan` | 未来计划、目标、承诺、待办或开放事项 | “我们接下来准备做什么？” |
| `decision` | 已确认的选择、方案或结论 | “最后决定用哪个模型？” |
| `constraint` | 明确要求、边界、必须或禁止事项 | “我说过哪些不能改？” |
| `knowledge` | 讨论、查证或学习到的外部知识 | “Fable 和雅可比猜想是什么情况？” |
| `procedure` | 方法、步骤、操作方式或工作流程 | “这个部署流程怎么做？” |

明确取消以下目标类别：

- `fact`：几乎所有陈述都可以被归为 fact，无法有效裁剪候选。
- `casual`：没有检索价值的闲聊应使用空 facets，不进入巨大兜底桶。
- `project_work`：项目是实体或主题，不是内容性质。
- `personal_profile / plan_goal / emotion_state / life_event / system_meta`：分别由 `profile / plan / state / event` 和其他维度表达。

项目、系统和领域名称写入 `entity_anchors`，不再通过类别表达。例如 Akane 配置可以按具体内容标为 `state / decision / constraint / procedure`，实体锚点填写 `Akane`。

### 4.2 当前消息的 `memory_query`

“我们之前聊过 Fable 吗”这句话本身是一次记忆查询，但想检索的历史内容不是 `memory_query`。

目标设计将它作为当前轮意图，而不是历史内容 facet：

```json
{
  "memory_metadata": {
    "turn_intent": "memory_query",
    "memory_facets": [],
    "about_roles": ["external"],
    "entity_anchors": ["Fable"],
    "topic_terms": [],
    "retrieval_priority": "low",
    "mood_tags": []
  }
}
```

检索参数描述目标历史内容，因此通常使用 `knowledge / event` 等 facets，而不是把 `memory_query` 传入历史候选过滤。

旧 SQLite 数据中的 `categories=["memory_query"]` 只会在 schema 2→3 一次迁移时转译为当前轮意图；运行时新写入与检索 API 不再接收旧字段。

### 4.3 `about_roles`

`about_roles` 回答“内容主要在陈述谁或什么”，不表示谁参加了这轮对话。

固定枚举：

| 枚举 | 定义 |
| --- | --- |
| `user` | 内容在陈述当前用户的属性、行为、观点、计划等 |
| `assistant` | 内容在陈述当前助手自身 |
| `third_party` | 内容在陈述群友、作者或其他人物/人群 |
| `external` | 内容在陈述项目、系统、组织、物品、概念或外部事件 |

允许多选。

关键规则：

- 发言者不自动成为内容主体。
- 群聊其他人的事实不能归到 `user`。
- 不再使用含义过宽的 `other`。
- 不建立“模型/项目/产品/系统”等实体类型硬枚举；这些边界本身容易歧义，准确名称更有检索价值。

示例：

```text
“Fable 5 构造了雅可比猜想反例”
about_roles = [external]

“Levent 使用 Fable 构造反例”
about_roles = [third_party, external]

“我把你设成群管理员”
about_roles = [user, assistant]
```

### 4.4 `entity_anchors`

`entity_anchors` 只保存内容中明确出现或可以确定的准确名称和常用别名，例如：

```json
["Fable", "Fable 5", "雅可比猜想", "Levent Alpoge"]
```

禁止把自由联想、上位词或近期上下文里的无关限定写成实体锚点。例如原文只说 Fable 时，不能猜测并加入 `flash / pro`。

实体不按“人/事/物”建立硬类型。对于检索来说，`Fable 5` 这个准确名称比“它是模型还是项目”更稳定、更有区分度。

### 4.5 `topic_terms`

`topic_terms` 保存动作、属性、议题和有助于语义匹配的短词：

```json
["证伪", "反例", "多项式映射"]
```

它不参与实体精确候选过滤，只参与 BM25、向量查询构造和结果排序。它不能与 `entity_anchors` 同权。

### 4.6 `retrieval_priority`

建议使用离散枚举：

```text
low / normal / high / critical
```

它表示这段内容未来被重新找回的价值，不表示模型对事实真假的信心。

允许用途：

- 指导摘要保留明确的重要事实；
- 在检索相关度基本相同时作为弱排序信号；
- 用户明确询问“重要决定/关键约束”时作为可选条件；
- 为后续受控的记忆资源管理提供信号。

禁止用途：

- 普通检索默认设置最低 priority；
- 因 priority 低而禁止 raw 入库或索引；
- 用 priority 压过实体、主题和文本相关性；
- 默认替代长期记忆固定可见窗口的 recency 规则。

新 native retrieval tool 默认不暴露 `priority_min`。只有产品明确需要“只看关键记忆”时才开放。

### 4.7 删除 `metadata_confidence`

当前 `confidence` 无法明确区分“事实真实性、模型推理把握、标签准确性”，且没有形成稳定的检索或压缩行为。

目标方案不再要求模型输出该字段。事实来源与可信边界使用系统掌握的客观字段表达：

```text
origin / annotation_status / retrieval_policy / trust
tool provenance / source lineage / material status
```

旧数据库字段可以在迁移窗口内读取但不再使用；目标实现不保留两套权威语义。

### 4.8 `mood_tags`

`mood_tags` 只在宿主启用 flavor 时使用，只影响情感余温和表现，不进入候选前置过滤。

## 5. 实体锚点的候选机制

查询端有明确 `entity_anchors` 时，采用两阶段但仍然评分前过滤的机制。

### 5.1 第一阶段：实体候选过滤

```text
安全硬边界
 memory_facets
 about_roles
 entity_anchors（同维度 OR）
 → 候选集合
 → dense / BM25 / RRF
```

不同维度之间为 AND，同一维度多个值为 OR。

实体规范化只处理稳定的大小写、空白、标点和已记录别名，不凭空推导实体类型或相关项目。

### 5.2 没有实体候选时

如果第一阶段实体候选为零：

1. 只移除 `entity_anchors` 的强制候选条件；
2. `memory_facets / about_roles / 安全边界` 继续保留；
3. 实体词继续作为高权重 BM25 与语义查询信号；
4. `topic_terms` 仍为普通辅助信号；
5. 结果必须记录实体过滤发生了放宽，不能静默忽略。

这样可以兼容旧记录漏标实体、别名不同但正文存在准确名称的情况。

实体锚点即使不再作为强制条件，也不会降成与普通主题词完全同权。

### 5.3 Facet 与主体过滤

`memory_facets` 和 `about_roles` 是模型明确选择的受控候选条件。目标方案不在同一次调用中静默删除这两个条件。

模型不确定时应在调用时省略对应字段；查询为空时返回结构化 diagnostics，模型可以根据结果显式调整参数再次调用。这样模型始终知道自己实际搜索了什么。

这不会改变前置过滤原则：每一次重试仍先形成候选集合，再计算向量/BM25。

## 6. Raw 主检索流程

### 6.1 查询输入

模型传入的参数描述“想找的历史内容”，不是照抄当前用户消息的标签：

```json
{
  "query": "我们以前是否讨论过 Fable 证伪雅可比猜想",
  "memory_facets": ["knowledge", "event"],
  "about_roles": ["external"],
  "entity_anchors": ["Fable", "雅可比猜想"],
  "topic_terms": ["证伪", "反例"]
}
```

当前消息可以是 `turn_intent=memory_query`，但检索目标不能因此被过滤成 memory_query。

当已知具体时间有助于缩小模糊事件检索时，`retrieve_for_turn` 使用与时间线相同的
本地/ISO 时间输入；省略偏移时按 `MemorySystem.timezone` 解释，并在 dense/BM25
评分前执行起点包含、终点不包含的硬过滤：

```json
{
  "query": "和 misaka 一起来玩的另一个人是谁",
  "entity_anchors": ["misaka"],
  "topic_terms": ["同行", "一起来玩"],
  "time_hint": {
    "start_at": "2026-08-03 11:00",
    "end_at": "2026-08-03 12:00"
  }
}
```

“另一个人”是待查询答案，不得先猜成 `entity_anchors`。准确时间范围足够小时仍应
优先 `read_timeline` 直接读取原话；时间、人物或事件位置只有部分已知时，再由
`retrieve_for_turn` 使用已知证据发现候选。

### 6.2 候选与评分顺序

```text
1. 编译不可放宽的安全 HardFilterPlan
2. 排除当前 prompt 已可见记录及完整 lineage closure
3. 按 memory_facets / about_roles / entity_anchors 前置裁剪
4. 分别在 raw 与 derived 辅助池中执行 dense/BM25
5. raw 结果优先返回，summary/semantic 作为不与 raw 竞争排序位置的辅助结果
6. 同 lineage 多层命中时优先 raw
7. 按 raw-first 与确定性分数选取已经合法的候选
8. 返回结构化结果、实际生效条件和诊断
```

### 6.3 结果层级行为

每个结果必须带明确 `source_layer`：

```text
raw              → 可直接回答，也可作为 read_timeline 锚点
summary          → 只按当前摘要内容使用
semantic_summary → 只按当前长期记忆内容使用
```

摘要或长期记忆不足以回答时，模型忽略该结果或使用其他 raw 结果；不能沿它的 lineage 自动回原文。

## 7. 时间线工具配合

`read_timeline` 保留三种互斥入口。

### 7.1 精确时间范围模式

用户明确给出小时或分钟时，直接使用本地时间或 ISO 8601，不要求模型计算 epoch：

```json
{
  "time_range": {
    "start_at": "2026-07-22 11:00",
    "end_at": "2026-07-22 12:00"
  }
}
```

`start_at` 包含、`end_at` 不包含；无显式偏移时使用 `MemorySystem.timezone`。

### 7.2 日期/粗时段兼容模式

用于用户明确给出昨天、上周二、某个日期或时间段：

```json
{
  "date_from": "2026-07-22",
  "date_to": "2026-07-22",
  "time_periods": ["afternoon"]
}
```

旧日期字段会归一成 timestamp 起止范围，再进入与精确模式相同的 Store 查询路径。

### 7.3 Raw 锚点扩窗模式

用于语义检索已经找到相关 raw，但一轮内容不足以看清前因后果：

```json
{
  "anchor_source_id": "raw-source-id",
  "before_turns": 4,
  "after_turns": 8
}
```

规则：

- `anchor_source_id` 必须解析为当前授权 Namespace 内可读的 raw entry；
- summary/semantic id 返回 `invalid_filter/raw_anchor_required`；
- 以完整 turn/relation component 为窗口原子，不截断一半工具分支或一半用户轮次；
- 只在同一获授权 conversation scope 内扩展；
- 初版不新增 MemCore 锚点扩窗 token 配额，按模型明确请求的前后 turn 读取；
- 若宿主最终遇到供应商物理上下文上限，应结构化报告实际截断，不允许 MemCore 提前静默缩窗；
- 返回内容按 `source_id` 去重并保持真实时间顺序。

## 8. 去重与可见排除

`retrieve_for_turn` 必须自动排除：

```text
当前可见 raw
当前可见阶段摘要
当前可见长期记忆
当前用户输入
以上记录连接的全部上下游 lineage closure
```

这是基于来源关系的去重，不是文本相似度去重。

检索结果内部如果 raw、summary、semantic 代表同一来源，普通查询保留 raw。不能让包含当前可见 raw 的摘要从 derived 层旁路返回。

显式调用 `read_timeline(anchor_source_id=<raw>)` 属于模型主动索取附近证据，不等同于普通检索重复召回；其返回内部仍按 raw `source_id` 去重。

## 9. 压缩与长期可见窗口

### 9.1 Raw 压缩

继续使用当前唯一 token 差值策略：

```text
raw_projected_tokens >= raw_token_trigger
→ 从最旧前缀选择约 raw_token_trigger * raw_token_batch_ratio
→ 只在完整 terminal turn/relation component 边界切割
```

不恢复 count policy、消息数量上限或按 category 排除 raw 的第二套压缩路径。

### 9.2 阶段摘要

阶段摘要是 raw 的压缩表示，不承担精确证据恢复职责。

metadata 继承必须逐字段处理：模型生成的某一字段为空时，只补该字段的合法来源值；不能因为 `entity_anchors` 非空就阻止 `memory_facets/about_roles/topic_terms` 的继承。

### 9.3 长期记忆窗口

长期记忆使用固定可见窗口，建议继续以当前默认 `semantic_visible_limit=5` 为初始值：

- 新条目进入后，最旧条目退出常驻 prompt；
- 退出常驻窗口不等于删除，旧长期记忆仍可作为辅助结果被检索；
- 默认按 recency/最近强化时间决定常驻窗口，不默认启用 importance decay；
- 被检索或出现同主题新证据时，可以按明确规则重新强化或提升。

### 9.4 强化合并

淘汰相邻不代表语义相同，不能只因两条记忆先后退出窗口就强制合并。

强化至少需要：

- 存在明确相同的实体锚点；
- facets 兼容；
- 核心命题相同、补充或构成有时间顺序的更新；
- 不能仅因为都出现 `user / assistant / Akane` 等常见主体就合并。

不满足条件时，两条记忆分别进入冷存储。

合并必须保留完整 `source_summary_ids` lineage。来源链不能复用普通展示列表的 8 条截断上限。

## 10. 给聊天模型的提示词草案

### 10.1 Metadata 标注

```text
memory_metadata 标注宿主指定的本轮记忆目标，不是你的回复。
turn_intent 只在当前内容明确查询历史记忆时填写 memory_query，普通消息留空；它不描述想找的历史内容。
memory_facets 表示这段内容以后能回答哪类问题，只能从固定枚举选择；没有长期检索价值时留空。
about_roles 表示内容主要在陈述谁或什么，不表示谁参加了对话；可选 user、assistant、third_party、external。
entity_anchors 只填写明确出现或能够确定的名称和别名，不要猜测相关实体。
topic_terms 填写有助于未来查询的动作或主题短词，不要写整句。
retrieval_priority 表示未来重新找回的价值，使用 low、normal、high、critical。
不确定时宁可留空，不要为了填字段编造标签。
```

固定 facet 枚举及一句话定义由同一份契约生成并保持稳定，不能由不同宿主提示词各写一版。

### 10.2 工具使用

```text
不知道具体时间但需要回忆旧事时，使用语义检索；检索参数描述想找的历史内容，不要照抄当前“是否记得”的问句类型。
语义检索以 raw 原始对话为主要证据。命中 raw 但缺少前因后果时，可用它的 source_id 读取附近时间线。
summary 和 semantic_summary 只按返回内容使用；内容不足时不要沿其来源链展开。
用户给出明确日期或相对日期时，可直接使用时间线工具。
工具仍没有明确证据时，说明没有找到或无法确定，不要猜测。
```

## 11. 示例

### 11.1 Fable / 雅可比猜想

写入：

```json
{
  "memory_facets": ["knowledge"],
  "about_roles": ["third_party", "external"],
  "entity_anchors": ["Fable", "Fable 5", "雅可比猜想", "Levent Alpoge"],
  "topic_terms": ["证伪", "反例"],
  "retrieval_priority": "normal",
  "mood_tags": []
}
```

查询：

```json
{
  "query": "以前关于 Fable 证伪雅可比猜想聊了什么",
  "memory_facets": ["knowledge", "event"],
  "about_roles": ["external"],
  "entity_anchors": ["Fable", "雅可比猜想"],
  "topic_terms": ["证伪", "反例"]
}
```

### 11.2 群管理员

```json
{
  "memory_facets": ["event", "relationship"],
  "about_roles": ["user", "assistant"],
  "entity_anchors": ["misaka", "Akane"],
  "topic_terms": ["群管理员", "任命", "册封"],
  "retrieval_priority": "normal",
  "mood_tags": []
}
```

### 11.3 用户的硬性设计要求

```json
{
  "memory_facets": ["constraint", "decision"],
  "about_roles": ["user", "external"],
  "entity_anchors": ["MemCore"],
  "topic_terms": ["前置过滤", "硬过滤", "检索成本"],
  "retrieval_priority": "critical",
  "mood_tags": []
}
```

## 12. 迁移原则

本方案通过审阅后，实施时采用一次权威切换，不长期维护新旧两套 metadata 逻辑。

建议顺序：

1. 固化唯一 metadata contract 和枚举；
2. 更新 schema、Chat Output Adapter、摘要/长期提示词和 native tool schema；
3. 更新 index metadata flags 与实体倒排字段，提升 index schema generation；
4. 更新 raw-first 分层检索和实体零候选放宽；
5. 增加 `read_timeline(raw anchor)` 模式；
6. 修复逐字段 metadata 继承、严格强化和完整 lineage；
7. 对旧 metadata 做一次迁移和全量 reindex；
8. 删除迁移 adapter，不保留永久双写或双读。

旧字段映射：

```text
preference       → preference
personal_profile → profile
plan_goal        → plan
relationship     → relationship
emotion_state    → state
life_event       → event
```

`casual / project_work / system_meta` 不能仅靠名称可靠映射，需要根据原内容重新标注，不能用机械映射制造错误 metadata。

旧 `confidence` 丢弃；旧 `importance` 只能经过明确阈值映射为离散 priority，不能直接假装原浮点具有统一标尺。

## 13. 验收标准

### 13.1 标签稳定性

固定测试集至少覆盖：

- 外部知识与人物：Fable / 雅可比猜想；
- 关系与事件：群管理员任命；
- 用户偏好；
- 计划、承诺和开放事项；
- 决策与硬性约束；
- 当前状态；
- 群聊第三方人物；
- 当前轮 memory query 与目标历史 facet 的区别。

同一测试集在聊天标注、阶段摘要、长期记忆和检索参数生成中必须使用同一字段含义。

### 13.2 前置过滤

测试必须证明不满足 `memory_facets / about_roles / entity_anchors` 的 entry 没有进入 dense cosine 和 BM25 文档统计，而不只是最终结果中看不到它。

实体过滤为零时，只放宽实体强制条件；实体词继续获得高权重，且 diagnostics 明确记录放宽。

### 13.3 Raw 主检索

- summary/semantic 不能在统一排序中压掉相关 raw；
- 同 lineage 多层命中时优先 raw；
- 当前可见三层及 lineage closure 不能旁路重复返回；
- raw 命中保留真实 source_id、timestamp、actor、turn 关系。

### 13.4 时间线配合

- 日期模式继续精确读取；
- raw source id 可以读取前后完整 turn；
- summary/semantic id 作为 anchor 时结构化拒绝；
- 扩窗不跨越 Namespace/conversation 权限；
- 返回按 source_id 去重并保持真实顺序。

### 13.5 压缩与长期记忆

- raw 继续按 token 差值与完整关系边界压缩；
- metadata 缺失字段逐项继承；
- 长期可见窗口保持固定且可配置；
- 不同主题不能因常见人物或相邻淘汰而强化合并；
- lineage 来源数量不受展示列表上限截断。

### 13.6 真实模型验收

除了单元测试，还必须记录真实聊天模型在固定问题集中的：

```text
模型看到的稳定提示词
生成的 metadata
发出的检索参数
实际候选裁剪数量
raw / summary / semantic 命中层
是否正确调用时间线扩窗
最终回答是否有足够证据
```

验收目标不是“字段能够解析”，而是模型能稳定理解写入与查询采用同一套规则。

## 14. 本方案明确不做

- 不增加每轮 router LLM；
- 不把 metadata 过滤改成评分后的后置过滤；
- 不建立复杂实体类型本体；
- 不从 summary/semantic 自动沿 lineage 下钻原文；
- 不把相邻退出窗口的长期记忆强制合并；
- 不让低 priority 原始对话失去入库或检索资格；
- 不长期保留新旧 metadata 双权威；
- 不用大量抽象提示词代替明确字段定义和示例。

## 15. 2026-07-23 代码审计结论

当前问题不是单个提示词写错，而是旧契约同时存在于 package 和 Akane 宿主的多个入口。实施必须按调用链整体切换，不能只修改 `schema.py`。

真实读写链如下：

```text
聊天模型最终输出
  -> memcore.chat_output / Akane final_output_engine 解析 memory_metadata
  -> MemorySystem.complete_turn / MemcoreManager.stage_turn_metadata
  -> SQLite messages.memory_metadata_json
  -> entry_builder 构建 index metadata 与 BM25 文档
  -> ReadPipeline 编译前置过滤并检索
  -> MemorySystem.retrieve_for_turn_structured
  -> MemcoreManager.retrieve_memory
  -> Akane retrieve_memory 工具结果

raw 压缩
  -> Compaction 生成阶段摘要及 metadata
  -> SQLite summaries + summary index
  -> 阶段摘要压缩/强化
  -> SQLite semantic_summaries + semantic index
```

审计确认的旧权威与缺口：

| 位置 | 当前事实 | 目标动作 |
| --- | --- | --- |
| `memcore/schema.py` | 仍使用 `keywords/subject_scopes/categories/importance/confidence` | 改为唯一新契约并导出枚举、校验和提示词片段 |
| `memcore/chat_output/prompts.py`、`memcore/prompts.py` | 分别手写旧字段说明 | 都从 schema 契约生成说明，不再复制字段定义 |
| `memcore/index/metadata_filters.py` | 仅有 category/scope boolean flag | 改为 facet/role/entity flag，实体 flag 使用稳定规范化与 hash |
| `memcore/index/entry_builder.py` | BM25 tag 文档仍拼旧字段 | 只拼 `entity_anchors/topic_terms/facets/roles/mood`，priority 只作弱信号 |
| `memcore/index/memory_index.py`、`chroma_index.py` | `keywords` 会替代原 query；实体无独立权重 | query 始终保留，实体词高权重复用，topic 为普通辅助词 |
| `memcore/retrieval.py` | 三层共榜；自动丢 importance/category/scope；默认 token 裁剪 | raw-first；只允许实体零候选放宽；默认不裁剪 |
| `memcore/native_tools.py` | 对模型暴露旧检索字段；时间线只支持日期 | 切换新检索字段并增加 raw anchor 模式 |
| `memcore/store/base.py`、`sqlite_store.py` | 只有物理 `seq_no +/- window` | 增加按完整 turn 分组的 namespace-safe raw window API |
| `memcore/compaction.py` | metadata 全有或全无继承；强化条件过宽；lineage 复用 8 条截断 | 逐字段继承；严格命题兼容；lineage 无展示上限 |
| `memcore/store/migrations.py` | SQLite schema version 2，未迁移新 metadata | 升 version 3，一次转换 JSON 并把三层 index 标为 pending |
| Akane `final_output_engine.py` | 宿主自行归一化旧 metadata | 改为调用 MemCore 唯一契约 |
| Akane `capability_registry.py`、`tool_runtime.py` | 两份旧 retrieve schema 和一份手写提示 | capability registry 保留宿主 ToolSpec 外壳，字段 schema/说明由 MemCore 生成；runtime 只做薄适配 |
| Akane `memcore_integration/manager.py` | 再次翻译旧字段，并默认启用 2000 token 结果预算 | 透传新字段，默认预算设为 0，不再维护旧枚举 |
| Akane `prompt_blocks.py`、`persona_profiles.toml` | 仍向聊天模型解释旧字段 | 引用/注入同一新契约，删除重复说明 |

`companion_v01/memcore_integration/tools.py` 当前只是 future-only 占位，不能再成为第四个权威入口；本次实施要删除占位语义或让它只 re-export package 工具定义。

## 16. 目标 Python 契约

### 16.1 Schema

`memcore/schema.py` 公开以下常量：

```python
MEMORY_FACETS = (
    "profile", "preference", "viewpoint", "relationship", "event",
    "state", "plan", "decision", "constraint", "knowledge", "procedure",
)
ABOUT_ROLES = ("user", "assistant", "third_party", "external")
RETRIEVAL_PRIORITIES = ("low", "normal", "high", "critical")
TURN_INTENTS = ("memory_query",)
```

目标 dataclass：

```python
@dataclass
class MemoryMetadata:
    turn_intent: str = ""
    memory_facets: list[str] = field(default_factory=list)
    about_roles: list[str] = field(default_factory=list)
    entity_anchors: list[str] = field(default_factory=list)
    topic_terms: list[str] = field(default_factory=list)
    retrieval_priority: str = "normal"
    mood_tags: list[str] = field(default_factory=list)
```

校验行为：

- 枚举外值丢弃；字符串列表 trim、按首次出现去重；
- `turn_intent` 为空或固定枚举；
- `retrieval_priority` 非法时回到 `normal`；
- `enable_flavor=False` 时清空 `mood_tags`；
- 不再输出旧键，也不保留 `confidence`；
- 本阶段不新增 metadata token 配额、总字符配额或隐藏裁剪。

`MemoryConfig.categories` 随旧契约删除。facet/role 是协议常量，不允许宿主换成另一套词表，否则写入与检索会再次分叉。

### 16.2 Retrieval

`RetrievalRequest` 目标字段：

```python
query: str
entity_anchors: tuple[str, ...] = ()
topic_terms: tuple[str, ...] = ()
source_layers: tuple[str, ...] = ()
memory_facets: tuple[str, ...] = ()
about_roles: tuple[str, ...] = ()
time_hint: Mapping[str, Any] = {}
kind_patterns: tuple[str, ...] = ()
include_explicit: bool = False
cross_conversation: bool = False
exclude_source_ids: tuple[str, ...] = ()
max_matches: int = 0
result_token_budget: int = 0
```

`result_token_budget=0` 必须表示不做 MemCore 结果裁剪。`MemoryConfig` 和 Akane 配置默认值都改为 `0`。保留显式正值只作为其他宿主主动选择的兼容能力，不在 native tool 中暴露，也不由 MemCore 自动推断供应商窗口。

`RetrievalResult` 增加可观测 diagnostics，而不是让模型猜系统做过什么：

```python
candidate_counts: Mapping[str, int]
entity_filter_relaxed: bool
effective_filters: Mapping[str, Any]
```

旧 `relaxation_steps` 不再记录自动删除 facet/role/priority；目标实现只允许出现 `drop_entity_requirement_after_zero_candidates`。

### 16.3 Raw 时间线扩窗

`memcore/store/base.py` 增加通用结果类型和接口：

```python
@dataclass(frozen=True)
class RawTurnWindow:
    status: str
    entries: tuple[TimelineEntry, ...] = ()
    anchor_source_id: str = ""
    before_turns: int = 0
    after_turns: int = 0
    reason: str = ""

def get_raw_turn_window(
    *, namespace: Namespace, anchor_source_id: str,
    before_turns: int, after_turns: int,
) -> RawTurnWindow: ...
```

SQLite 选择算法固定如下：

1. 在当前完整 Namespace（含 conversation）内查 `messages.source_id`；
2. 查不到返回 `empty/anchor_not_found_or_out_of_scope`，不能跨会话猜同名 ID；
3. 非 raw ID 不会落入 messages；MemorySystem 若发现该 ID 属于 summary/semantic，返回 `invalid_filter/raw_anchor_required`；
4. 非空 `turn_id` 的全部 messages 是一个窗口原子；空 `turn_id` 的 standalone entry 自成一组；
5. 组按最小 `seq_no` 排序，选择 anchor 组前后完整组；
6. 组内按 `seq_no` 返回，最终按真实会话顺序去重；
7. 不因 `is_summarized=1` 排除旧 raw，因为时间线工具本来就是精确读取原始记录；
8. 不在 Store 层加入 token 裁剪。

`MemorySystem.read_timeline()` 的精确时间、旧日期和 anchor 模式互斥：

```python
read_timeline(
    time_range=None,
    date_from="", date_to="", time_periods=None,
    anchor_source_id="", before_turns=0, after_turns=0,
    cross_conversation=False,
    projection="conversation", page_token_budget=0, cursor="",
)
```

同时传多个 selector 模式返回 `invalid_filter/timeline_modes_are_mutually_exclusive`；三个 selector 模式都不传返回 `invalid_filter/timeline_selector_required`。`projection` 和显式页面预算会被冻结进 cursor，续页时不能重复或篡改。

当前实现按 `turn_id`（无 turn 的记录按 source_id）组成分页逻辑单元。直接 Python API 的 `page_token_budget=0` 是可信宿主使用的显式无限路径；provider-native dispatcher 则把省略/0 解析为 `MemoryConfig.native_timeline_page_token_budget`，并把模型传入的更大值限制在该宿主上限内。正预算优先使用宿主注入的 TokenCounter；没有 tokenizer 时使用明确标记为 `estimated` 的 UTF-8 粗略估算，不会令工具失效。单个 turn 超预算时仍完整返回并标记 `oversized_unit=true`。cursor 校验版本、selector 指纹、投影、方向、有效页面预算、最后一个稳定 unit key 和当前 namespace 指纹，重新查询时仍以当前 MemorySystem namespace 为唯一授权来源。

## 17. 索引与检索执行细节

### 17.1 Index schema 3

`INDEX_SCHEMA_VERSION` 升为 3。目标 metadata：

```text
memory_facet__<safe-or-hash>       bool
memory_about_role__<safe-or-hash>  bool
memory_entity__v1_<sha256>         bool
memory_entity_text                 normalized entity anchors
memory_topic_text                  topic terms
memory_facets_text                 facets
memory_about_roles_text            roles
memory_priority                    low/normal/high/critical
```

实体规范化只做 Unicode NFKC、trim、合并空白和不区分大小写；不去掉会改变专名的内部符号，不做模糊实体推断。flag key 对规范化实体做 SHA-256，数据库/日志和 Chroma metadata key 不泄漏原始专名。

BM25 文档由正文加上述稳定 tag 文本组成。查询词由三部分合并且不得互相替代：

```text
原始 query（始终保留）
+ entity_anchors（高权重，可通过重复 token 或后续独立 boost 实现）
+ topic_terms（普通辅助权重）
```

第一版可以用确定性的 token 重复实现实体高权重；必须有测试证明 topic terms 不会覆盖原 query，也不能与实体同权。

### 17.2 Raw-first 计划

一次检索编译两个候选池：

```text
raw pool     : entry_type=raw + 安全边界 + facets + roles + entities
derived pool : entry_type in (summary, semantic_summary) + 同样边界
```

执行顺序：

1. 分别计算加实体条件后的候选数量；
2. 某个池实体候选为零时，仅在该池移除实体 flag 条件并记录 diagnostics；
3. 在实际候选池内运行 dense 与 BM25，再 RRF；
4. 不再追加 LLM verifier；相关性问题必须在 query、索引、确定性分数与 diagnostics 中可解释；
5. 先按相关度填充 raw；还有现有 `max_matches` 空位时才附加 derived；
6. derived 的 lineage closure 与已选 raw 相交时丢弃 derived 副本；
7. derived 不能挤掉 raw，也不自动下钻；
8. facets/roles 空时表示模型不确定，不加对应条件；非空时绝不在同一次调用中静默删除。

这里沿用现有 `max_matches/retrieval_limit` 作为工具返回条目数量，不新增 raw/summary/semantic 固定配额。

### 17.3 可见内容排除

`retrieve_for_turn_structured()` 继续在评分前调用 `visible_lineage_source_ids()`，但测试必须覆盖：

- 可见 raw 的 summary/semantic ancestor 被排除；
- 可见 summary 的 raw descendants 和 semantic ancestor 被排除；
- 可见 semantic 的 summary/raw descendants 被排除；
- 当前 input source ID 被排除；
- lineage 缺失或成环时返回结构化 unavailable，不退化成可能重复的宽搜。

## 18. SQLite metadata 一次迁移

SQLite schema version 从 2 升到 3。三个表的 JSON 统一迁移，不新增重复列：

```text
messages.memory_metadata_json
summaries.memory_metadata_json
semantic_summaries.memory_metadata_json
```

安全机械映射：

```text
categories.preference       -> memory_facets.preference
categories.personal_profile -> memory_facets.profile
categories.plan_goal        -> memory_facets.plan
categories.relationship     -> memory_facets.relationship
categories.emotion_state    -> memory_facets.state
categories.life_event       -> memory_facets.event
categories.memory_query     -> turn_intent.memory_query

subject_scopes.user         -> about_roles.user
subject_scopes.assistant    -> about_roles.assistant
subject_scopes.other        -> about_roles.third_party + external（仅迁移兼容）

keywords                    -> topic_terms
confidence                  -> 删除
```

`casual/project_work/system_meta` 不机械映射 facet。旧关键词不能可靠区分专名和主题，因此不得伪装成 `entity_anchors`；准确名称仍存在正文中，查询实体零候选时会进入明确的实体放宽路径。

旧浮点 importance 仅在迁移中使用固定阈值：

```text
>= 0.85 critical
>= 0.65 high
>= 0.25 normal
<  0.25 low
```

迁移完成后：

- JSON 中不再保留旧字段；
- 所有三层记录 `index_status='pending'`、`index_schema_version=0`、`index_key=''`；
- MemorySystem/Akane 启动时现有 namespace warmup 重建 schema 3 索引；
- 迁移函数可重复运行但 version 3 不再重复变换；
- 不保留长期双写、双读或“新字段为空时再偷读旧字段”的路径。

## 19. Akane 回填与权威删除

package 先完成并通过测试，Akane 再做薄适配：

1. `final_output_engine.normalize_memory_metadata()` 改为 MemCore contract adapter，输出只含新字段；
2. `engine._memory_metadata_has_signal()` 检查新列表字段、turn intent、非默认 priority 或 mood；
3. `MemcoreManager.retrieve_memory()` 参数改为 facets/roles/entities/topics，不再翻译旧枚举；
4. `capability_registry.py` 的 ToolSpec 保留 Akane capability id、风险和可见端配置，但 input schema 由 MemCore builder 生成；
5. `tool_runtime.py` 删除重复 input schema 和重复枚举，只负责 JSON/原生调用统一成同一个参数字典；
6. `prompt_blocks.py/persona_profiles.toml` 删除旧字段解释，聊天输出提示使用 MemCore 导出的简短规则；
7. `config.py` 的 `MEMCORE_RETRIEVAL_RESULT_TOKEN_BUDGET` 默认改为 0；
8. 事件、工具、材料仍通过 `kind/origin/turn_role/retrieval_visibility` 表达，不再塞 `event_trace/tool_trace/material_trace` facet；
9. `extract_memory_keywords` 改为读取 `entity_anchors + topic_terms` 的薄兼容命名，随后重命名调用点，不能继续成为旧 keywords 权威。

旧 Akane legacy/dual backend 属于既有迁移开关，本方案不扩大其能力；MemCore backend 的新写入链不能因 legacy 兼容而继续产生旧 metadata。

### 19.1 Akane active path 的精确替换表

Akane 回填不是新增第二套 memory API。下列位置是 active MemCore 路径中需要被替换的旧权威；完成后，宿主只保留产品命名、权限、Namespace 和交付编排。

| Akane 位置 | 回填后的职责 | 旧实现最终状态 |
| --- | --- | --- |
| `final_output_engine.normalize_memory_metadata()` | 调用 `memcore.coerce_memory_metadata()`，再输出 `MemoryMetadata.to_dict()` | `thin adapter`；删除 aliases、旧枚举、浮点 confidence/importance 归一化 |
| `final_output_engine.extract_memory_keywords()` | 删除；调用点直接读取 `entity_anchors + topic_terms`，仅用于旧 UI/日志所需的扁平标签时才局部投影 | `deleted` |
| `llm_runtime._memory_metadata_has_signal()` 与 `engine._memory_metadata_has_signal()` | 统一调用 `memcore.memory_metadata_has_signal()` | `thin adapter` 或直接导入；不再各自判断字段 |
| `capability_registry.RETRIEVE_MEMORY_TOOL_SPEC` / `READ_MEMORY_TIMELINE_TOOL_SPEC` | 从 `build_native_memory_tool_specs(tool_format="plain", include_material_tool=False)` 读取 description/schema，再换成 Akane 的 capability id、权限与 visible client 配置 | schema `thin adapter`；MemCore 是唯一字段权威 |
| `tool_runtime` 的两个 `*_INPUT_SCHEMA` | 删除，`ToolMetadata` 与 handler 均引用 capability registry 中的 package-backed schema | `deleted` |
| `RetrieveMemoryToolHandler.normalize_call()` | 只接收 MemCore 新字段并做形状归一化；不翻译 `keywords/categories/subject_scopes/importance_min` | `thin adapter`，旧参数不双读 |
| `ReadMemoryTimelineToolHandler.normalize_call()` | 透传精确 `time_range`、旧日期别名或 raw anchor 模式；冲突交给 MemCore 返回结构化 `invalid_filter` | `thin adapter` |
| `MemcoreManager.retrieve_memory()` / `shadow_retrieve_memory()` | 透传 `query/entity_anchors/topic_terms/source_layers/memory_facets/about_roles/time_hint/include_explicit/kind_patterns` | `thin adapter`；不做旧枚举翻译，不加隐藏结果限制 |
| `MemcoreManager.read_memory_timeline()` | 透传 `time_range`、旧 `date_from/date_to/time_periods` 或 `anchor_source_id/before_turns/after_turns` | `thin adapter` |
| `MemcoreManager._build_system()` | 不再读取 `DEFAULT_CATEGORIES` 或传 `MemoryConfig.categories`；结果 token budget 默认 `0` | 旧 category 配置 `deleted` |
| 事件、工具与材料写入点 | 依赖 `kind/origin/turn_role/retrieval_policy/retrieval_visibility` 表达协议事实；`memory_metadata` 仅在确有可检索语义标注时填写新契约 | 旧 `event_trace/tool_trace/material_trace` category `deleted` |
| `prompt_blocks.py` | 注入 `build_memory_metadata_instruction()` 和 MemCore native tool schema，不再手写字段解释 | `thin adapter` |
| `persona_profiles.toml` | 只保留 Akane 人设与产品输出结构；metadata 字段说明、枚举与摘要/强化契约由 MemCore prompt builder 提供 | 重复 schema `deleted` |
| `memcore_integration/tools.py` | 若保留，仅 re-export 当前 package tool specs/availability，不描述 future-only 能力 | `thin adapter` 或 `deleted` |

### 19.2 参数与命名映射

Akane 对模型保留产品工具名 `retrieve_memory` 和 `read_memory_timeline`，但它们分别映射到 MemCore 的 `retrieve_for_turn` 与 `read_timeline` 契约。名称映射不能复制 schema：

```text
MemCore plain spec name=retrieve_for_turn
  -> Akane capability_id=retrieve_memory
  -> RetrieveMemoryToolHandler
  -> MemcoreManager.retrieve_memory
  -> MemorySystem.retrieve_for_turn_structured

MemCore plain spec name=read_timeline
  -> Akane capability_id=read_memory_timeline
  -> ReadMemoryTimelineToolHandler
  -> MemcoreManager.read_memory_timeline
  -> MemorySystem.read_timeline
```

`limit` 不再暴露给模型。Akane 可以沿用已经存在的系统级 `retrieval_limit/max_matches` 作为宿主默认返回条目数，但本次不新增 raw/derived 固定名额、扩窗 token 配额或按消息类型裁剪。`source_layers` 省略表示由 MemCore 使用正常 raw-first 策略，不由 Akane 擅自补成某一层。

### 19.3 typed trace 的写入规则

工具、事件、材料不是历史内容 facet。它们使用稳定类型字段：

```text
event.*      -> kind=event.<domain>, origin=environment, turn_role=stimulus/standalone
tool.*       -> kind=tool.<tool_name>.<call|result>, turn_role=action/observation
material.*   -> kind=material.<event_type>, origin=user/environment, retrieval_visibility=explicit
```

系统生成的 trace 默认使用空的新 metadata 契约；只有事件正文确实包含以后值得语义检索的事实、且模型或宿主能够准确标注时，才填写 `memory_facets/about_roles/entity_anchors/topic_terms/retrieval_priority`。不能为了让 trace 可识别而伪造 facet。显式检索继续由 `include_explicit + kind_patterns` 控制。

### 19.4 真实验收点

回填完成不仅检查字段存在，还要验证模型实际看到和调用的是同一契约：

1. 最终回复 JSON 只输出七个新字段，不出现旧键；空 metadata 仍是合法输出，不影响正常回复。
2. 模型询问旧事实时，`retrieve_memory` 的审计参数保留原始 query，并可传准确实体、主题、facet 和主体；零实体候选时结果明确显示 entity relaxation。
3. raw 命中不足以回答时，模型能把该结果的 `source_id` 传给 `read_memory_timeline(anchor_source_id=...)`，完整取回前后 turn；摘要 id 会被结构化拒绝。
4. 工具调用、外部事件和附件记录仍按 typed kind 出现在 raw 时间线中，但默认普通对话检索不会因为伪造 category 把它们召回。
5. Akane 配置默认不截断 MemCore 检索结果 token；若某部署显式设置正值，截断仍由现有可观测机制处理，本次不扩大该兼容路径。

## 20. 实施切片与验证矩阵

### Slice A：唯一 metadata 契约

修改：schema、chat output、prompts、config、三层解析测试。

验收：同一 JSON 在聊天输出、阶段摘要、长期记忆中得到完全相同的新字段；旧键不出现在新写入中；不新增长度/token 裁剪。

### Slice B：index schema 3 与迁移

修改：metadata filters、entry builder、两个 index、SQLite migration、reindex 测试。

验收：迁移后旧 JSON 消失；索引只接受 schema 3；候选前置过滤测试直接统计 dense/BM25 收到的候选数。

### Slice C：raw-first retrieval

修改：RetrievalRequest/Plan/Result、双候选池、实体放宽、lineage 去重、native retrieve tool。

验收：Fable 固定数据集能以 raw 为主命中；错误 entity 触发可见 diagnostics；错误 facet/role 不被静默删除；summary 不能挤掉 raw。

### Slice D：raw anchor timeline

修改：Store interface、SQLite 实现、MemorySystem、native timeline tool、rendering。

验收：完整多工具 turn 不被截半；standalone 正确成组；越权/derived anchor 结构化拒绝；日期模式不回归。

### Slice E：compaction 与 lineage

修改：逐字段继承、严格强化候选、完整 `source_summary_ids`。

验收：Fable 不再与天气/VRChat 等仅有常见主体重叠的记忆合并；单字段缺失能独立继承；超过 8 个来源仍完整保留。

### Slice F：Akane 回填

修改：final output、工具 schema/runtime、manager、prompt、config 和相关测试。

验收：Akane 真实模型看到的新说明与 MemCore native tool 一致；入库和检索 audit 都只出现新字段；用户实际追问旧事实能够先 raw 检索、再按需要用 raw anchor 扩窗。

每个 slice 至少运行：

```text
相关 unittest
python -m ruff check <changed files>
python -m ruff format --check <changed files>
python -m build
git diff --check
```

最终再运行 MemCore 全量 unittest、Akane 记忆相关测试，并检查两个仓库的 `git status --short`、`git diff --stat`、`git diff --cached --name-only`。测试若暴露效果问题，先修正契约或实现，不用新增 token 配额掩盖问题。

## 21. 2026-07-23 实施落点与交接状态

本方案已经从设计切换为运行实现，当前权威链路如下：

```text
模型输出 memory_metadata
  -> memcore.chat_output 解析
  -> memcore.coerce_memory_metadata
  -> Akane final_output_engine 薄适配
  -> MemcoreManager stage/complete turn
  -> SQLite schema 3 + schema 3 index

模型调用 retrieve_memory / read_memory_timeline
  -> Akane capability 名称映射
  -> package 生成的 description + input schema
  -> Akane handler 只做调用形状归一化和权限上下文注入
  -> MemcoreManager 透传
  -> MemorySystem raw-first retrieval / raw anchor timeline
```

### 21.1 package 已落地文件

- `memcore/schema.py`：七字段唯一 metadata、统一归一化、signal 判断、统一提示词片段。
- `memcore/native_tools.py`：检索和时间线的模型可见 description/schema；宿主不再复制字段定义。
- `memcore/retrieval.py`：raw/derived 分池、评分前 metadata 过滤、实体零候选单独放宽、lineage 去重与 diagnostics。
- `memcore/memory_system.py`：structured retrieval、raw anchor timeline、完整 turn 扩窗。
- `memcore/compaction.py`：逐字段 metadata 继承、严格长期强化候选、完整 lineage。
- `memcore/store/migrations.py`：SQLite schema 3 数据迁移，以及供 Akane 明确 legacy import 使用的一次转换函数。
- `memcore/index/*`：schema 3 metadata 过滤和 query/entity/topic 检索构造。
- `memcore/rendering.py`：message/event/tool/material 的稳定 raw 渲染。

### 21.2 Akane 已落地文件

- `final_output_engine.py`、`llm_runtime.py`、`engine.py`：输出归一化与 metadata signal 委托 MemCore；事件/工具不再伪造 facet。
- `capability_registry.py`、`tool_runtime.py`：模型看到的 memory tool schema 来自 package；删除旧字段和模型可控 `limit`。
- `memcore_integration/manager.py`：新检索参数直接透传；raw source_id、诊断信息和结构化错误保留；默认结果 token budget 为 `0`。
- `memcore_integration/timeline.py`：日期查询 profile-wide，raw anchor 固定当前 conversation；MemCore 返回的 `invalid_filter/invalid_range/failed/unavailable` 不再被覆盖成泛化失败。
- `retrieval_engine.py`：MemCore active path 只使用新契约；旧字段投影仅存在于明确的 legacy/dual backend 分支。
- `prompt_blocks.py`、`prompt_builder.py`、`persona_profiles.toml`、`prompt_profiles.py`：写入、摘要、强化和聊天输出共用新契约。
- `config.py`、`.env.example`：`MEMCORE_RETRIEVAL_RESULT_TOKEN_BUDGET=0`。
- `memcore_integration/tools.py`：future-only 重复占位已删除。

### 21.3 legacy 数据边界

运行时新写入不双读旧键。旧 `keywords/categories/subject_scopes/importance/confidence` 只允许出现在两种位置：

1. SQLite schema 2→3 迁移输入；
2. Akane 显式 `legacy_import=True` 的旧库导入输入。

两条路径都立即转换成新 JSON，不能把旧键继续写回 active MemCore。Akane 的 legacy/dual 检索后端仍有保守字段投影，这是既有迁移开关，不是模型可见 schema，也不进入 MemCore backend。

### 21.4 已验证行为

- capability schema 的 properties/required 与 package builder 完全一致；
- 模型 schema 不包含 `keywords/categories/subject_scopes/importance_min/limit`；
- 工具 action/observation 默认 metadata 没有伪造 facet、role、entity 或 topic；
- raw anchor 强制使用当前 `session_id` 且 `cross_conversation=False`；
- 日期与 anchor 混用时保留 `invalid_filter: timeline_modes_are_mutually_exclusive`，不会降级成不明所以的 unavailable；
- 当前事件与后续历史使用同一 typed raw 渲染；
- 默认不对检索结果增加 token 裁剪。

验证结果：

```text
MemCore: 320 tests passed
Akane memory/metadata/prompt/tool/timeline focused regression: 282 tests passed
Akane full repository regression: 1688 tests executed; 本轮暴露并修正 1 个旧 memory 文本断言，仍有 3 个与本方案无关的既有失败（desktop satellite capability 选择 2 项、settings catalog 漂移 1 项）
```

本轮未部署云端、未启动 Bot，也未用线上对话替代离线验收。下一阶段若要做真实使用效果验收，应观察模型生成的新 metadata、raw-first 命中和必要时的 anchor 扩窗；不要先增加结果名额或 token 限制。
