# MemCore Unified Timeline V2 设计与实现说明

> Timeline V2 建设过程的历史设计记录。metadata 与检索章节停留在 schema 2；
> 当前字段、raw-first 检索和 raw anchor 规则以
> [`memory_metadata_raw_retrieval_design_v1.md`](memory_metadata_raw_retrieval_design_v1.md)
> 为准。旧字段描述只解释迁移来源，不是运行时兼容承诺。

状态（2026-07-22）: package 与 Akane 回填均已完成。Timeline V2 是唯一运行权威；Schema foundation、Turn lifecycle、Projection ledger、Compaction V2、Retrieval admission、Relation expansion、native memory tool schema/dispatch 和 Akane 读写切换均已落地。provider-native transport、产品 prompt assembly 与工具执行继续由宿主负责，这是稳定架构边界，不是待迁移能力。

当前实现检查点（2026-07-22）：

- Slice 1 `Schema foundation` 已提交（`4b82e2d`）：正式 migration runner、V2 物理列/表、namespace-safe relation reads 与 V1 trace compatibility window；
- Slice 2 `Turn lifecycle` 已提交（`24f1c1b`）：`begin_turn -> append_entry -> complete_turn / abort_turn`、显式 annotation target、并行 action/observation correlation、原子终态提交、visibility 物化与索引 outbox；
- Slice 3 `Projection ledger` 已提交（`38727e1`）：versioned renderer registry、canonical/OpenAI/Anthropic adapter、不可变 projection rows、请求 hash audit、final projection 原子提交和 strict-prefix 验收；
- Slice 4 `Compaction V2` 已提交（`41d55b4`）：共享 `MemCoreRuntime`、terminal-turn/token planning、summary/semantic 两阶段原子提交、episode/operation lineage 分离，以及压缩后 summary/semantic projection；
- 后续压缩可靠性修复已提交（`23213d9`、`006ead2`、`d97afbc`），并在差值策略 repair 中收口：按实际 provider projection 规划一次 raw generation；token 模式按比例选足完整 component，不再复用旧条目批次或独立 source 上限；单次 `run_due()` 仍只提交一个 raw generation 和一个 semantic batch；
- Slice 5 `Retrieval admission` 已提交（`df09a50`）：结构化 query/result、visibility/annotation/kind/conversation/index-generation 硬准入、开放 kind prefix flags、评分前过滤与有界 semantic relaxation；
- Slice 6 `Relation expansion` 已提交（`3523c26`）：Namespace-safe lineage closure、stimulus/final 原子组、并行 correlation branch、派生层去重、visible lineage 排除与精确 token budget；
- Akane cutover primitives 已实现：无模型回复的 typed standalone entry，以及不提前授予检索准入的 staged annotation；
- 宿主中立的 `append_action` / `append_observation` 薄接口已实现；JSON、XML、
  标签等非原生协议可冻结真实 request projection，operation 可逐条选择小型
  retention anchor，不要求宿主采用固定能力加载协议；
- `kind` 仍是开放 namespaced 字符串；关系完整性只约束通用 action/observation，不枚举工具、事件、Skill 或 Bot；
- accepted empty annotation 与 missing/invalid annotation 已分开，只有 accepted target 自动进入 default retrieval；
- final 继续保留真实 `provider_output_raw`；宿主提供的真实 final projection 会与 annotation/final/turn close 同事务保存，没有提供时可由标准 adapter 产生显式 `canonical_fallback`；
- episode/semantic summary 已进入同一不可变 projection ledger；`before/after_projected_tokens` 按目标 provider payload 计数，不再只统计正文；
- `add_summary() + mark_*()` 分事务 Store 方法只为测试 fixture、历史导入和维护兼容保留；V2 runtime 不调用它们，唯一在线压缩权威是原子的 `commit_summary_batch()` / `commit_semantic_batch()`；
- package 的 V2 能力已回填 Akane；Akane 仍保留产品级 prompt assembly、provider transport 和旧 `record_*` 薄适配。旧 API 不再作为 MemCore 内部新能力的扩展入口。

本文定义 MemCore 从“通用三层记忆内核”演进为“统一时间线、稳定上下文投影与记忆读取内核”的目标形态。它不改变 MemCore 与宿主的基本边界：宿主仍负责渠道、权限、工具执行、文件本体、模型选择和最终请求；MemCore 负责把模型实际经历的输入、输出、工具与事件可靠地记录、投影、检索和压缩。

## 1. 已确定的核心原则

1. MemCore 同时承担两类职责，但只保留一个真相源:
   - 宿主侧的底层时间线存储、压缩和稳定 prompt 投影；
   - 模型侧少量、清晰、可主动调用的记忆读取工具。
2. 用户消息、助手回复、外部事件、工具调用/结果、Skill 加载结果、材料状态等都按实际发生顺序追加到底部。
3. “统一进入时间线”不等于“统一进入普通检索或长期语义记忆”。检索准入、语义化和 prompt 可见性是彼此独立的策略。
4. 记录类型使用开放的 namespaced `kind`，例如 `message.user`、`event.finance`、`event.qq.poke`、`tool.web_search.result`、`skill.loaded`。新增 kind 不要求修改 MemCore 枚举。
5. 有效终态模型 JSON 产生的 `memory_metadata` 是一条输入具有普通记忆价值的强信号。事件不需要特殊开门；只要它触发完整模型回复并获得有效记忆标注，就可以按配置进入普通检索。
6. 工具调用和结果通常只有运行轨迹元数据，不得冒充模型产生的记忆标注，也不得污染当前输入的 `memory_metadata`。
7. 所有影响候选集合的硬过滤必须在向量/BM25 评分前完成。前置过滤是不可回退的核心能力。
8. 普通检索的上下文扩窗按逻辑轮次和关联关系完成，不再按物理 `seq_no ± N` 猜测上下文。
9. 缓存稳定是正式契约：除明确压缩或显式迁移外，旧记录的 provider-visible 投影必须保持字节级稳定，新内容只能追加在尾部。

## 2. 边界

### 2.1 MemCore 负责

- append-only 统一时间线与数据库迁移；
- Namespace/Actor 隔离与参与者归因；
- 轮次、回复、并行调用和因果关联；
- 结构化 entry 的通用与专用渲染；
- provider-neutral entry 到 provider messages 的稳定投影；
- raw/episodic/semantic 记忆生命周期；
- token 预算、完整轮次压缩与开放轮次保护；
- 混合检索、前置过滤、确定性诊断和关系扩窗；
- `retrieve_for_turn`、`read_timeline` 及可选材料读取工具的 schema、分发和结构化状态；
- 稳定前缀 hash、renderer/projection 版本和可诊断的缓存审计。

### 2.2 宿主负责

- QQ、桌宠、HTTP 等渠道接入；
- 判断发生了什么业务事件，并提交开放 `kind` 与结构化 payload；
- 工具和 Skill 的权限、信任与实际执行；
- 图片、文件本体、OCR、视觉描述和文档 chunks；
- 人格、安全规则、工具集合、模型参数和最终 API 请求；
- 决定一个输入是否触发模型轮次，以及哪些输入是该轮的记忆标注目标。

宿主决定“发生什么”；MemCore 决定“如何可靠地记录、展示、检索、关联和压缩”。

## 3. TimelineEntry V2

逻辑记录结构如下；第 16 节给出代码审计后确定并已实现的 SQLite 列拆分。未来内部索引可以演进，但不能删减这些公共语义。

```text
source_id                    # 继续作为权威 entry id；公共语义可称 entry_id，但不新增第二套 ID
namespace: tenant_id / user_id / domain_id / conversation_id
seq_no / timestamp

kind                         # 开放 namespaced 字符串
origin                       # user / assistant / environment
turn_id
turn_role                    # stimulus / intermediate / action / observation / final
reply_to_id
correlation_id               # 工具 call、并行任务或异步结果关联
actor / target_actor

payload_json                 # 权威结构化数据
semantic_text                # 用于检索、摘要和可读时间线的正文
provider_projection_ref      # 指向实际 provider-visible 投影；多工具批次可由一条投影关联多个 source_id

trace_metadata               # source、tool_name、call_id、状态等运行信息
retrieval_policy             # auto / always / explicit / never
semanticize                  # 是否允许进入个人/领域语义记忆
prompt_visible               # 是否进入正常上下文投影
trust                        # untrusted_data / trusted_instruction
renderer_id / renderer_version
```

`origin` 和 `turn_role` 是 MemCore 的语义，不等于 provider role。一个工具结果即使因兼容性通过 provider `user` role 返回，它仍然是 `origin=environment`、`turn_role=observation`，绝不能因此被当成用户原话。

`actor / target_actor` 会进入通用时间线渲染，宿主可用开放的 namespaced `kind`（例如旁观消息）区分消息语义，而不需要把渠道字段塞进 MemCore。说话人与明确对象不能在清洗 provider 控制片段时一起丢失。

没有触发模型回复的外部事实也可以作为 standalone entry 追加，`turn_id/turn_role` 留空并按自身策略决定 visibility；只有进入模型循环的输入才创建 turn。助手在工具前发出的非终态说明使用 `turn_role=intermediate`，不能冒充 final，也不要求像 action 一样必须有 correlation。

## 4. 轮次与记忆标注归属

每次模型轮次开始前，宿主与 MemCore 必须确定:

```text
turn_id
stimulus_entry_ids
annotation_target_ids
```

普通单输入轮次通常只有一个 target:

```text
turn-001
├── message.user                 stimulus + annotation target
├── assistant.tool_call A        action
├── assistant.tool_call B        action
├── tool.B.result                observation, correlation_id=B
├── tool.A.result                observation, correlation_id=A
└── assistant.final              final
```

事件轮次使用同一结构:

```text
turn-002
├── event.qq.poke                stimulus + annotation target
└── assistant.final              final
```

### 4.1 终态提交规则

- 只有宿主确认的终态模型输出才能提交 `memory_annotation`。
- 中间工具轮、修复轮或子模型输出即使偶然包含同名字段，也不能覆盖 annotation target。
- annotation target 由 `entry_id` 显式确定，不能通过“最后一条 provider role=user”推断。
- 一份单目标 `memory_metadata` 只写回一个明确 target。若宿主批量合并多个独立输入，必须创建明确的 input bundle，或使用逐 target annotations；不得把一份 metadata 静默涂到所有输入。
- `assistant.final` 通过 `turn_id/reply_to_id` 与 target 关联，因此普通检索可以同时返回输入和最终回复。

### 4.2 两类 metadata 必须拆开

`memory_annotation` 保存终态模型 JSON 的语义标注:

```text
keywords / subject_scopes / categories / mood_tags / importance / confidence
annotation_target_id / annotation_status / annotator
```

`trace_metadata` 保存运行与过滤信息:

```text
source / tool_name / call_id / status / mime / file_id / provider / host tags
```

`event_trace/tool_trace/material_trace` 只可能作为旧接入兼容 metadata 保留，
不再承担事件、工具或材料的权威身份与检索开关；V2 使用
`kind + trace_metadata + retrieval_visibility`。

### 4.3 检索自动准入

`retrieval_policy=auto` 时:

- 接收到有效、已接受的 `memory_annotation` 后，记录进入普通检索候选；
- 没有有效 annotation 时，使用宿主/namespace 的未标注输入策略，默认只允许显式检索；
- fallback、解析失败后补出的空 metadata 不等于有效模型 annotation；
- `always/explicit/never` 用于宿主的明确覆盖，不需要 MemCore 穷举 event kind。

因此普通消息和事件可以一视同仁；真正的区别是它是否完成了一个具有记忆标注的模型轮次，而不是它的 kind 是否叫 `event.*`。

## 5. 稳定渲染与模型投影

### 5.1 通用兜底渲染

未知 kind 必须无需注册即可安全工作:

```text
[2026-07-20 14:32 | 下午] event.some_new_type
source: qq
payload:
{"foo":"bar","value":123}
```

通用规则:

- 时间戳、时区、Actor 和 kind 结构固定；
- JSON key、空值、换行、Unicode 与长文本处理确定性；
- `untrusted_data` 始终渲染成数据块，不允许内容伪造 system/tool 边界；
- 未知 payload 类型退化为规范 JSON 或稳定文本，不能静默丢弃；
- renderer 缺失时使用通用兜底，而不是报“未支持事件类型”。

### 5.2 可选 renderer registry

宿主或包可以为 `tool.*`、`material.*`、`skill.*` 或特定 kind 注册美化 renderer。注册器只提升可读性，不决定功能是否可用。

每条记录保存 `renderer_id/version` 或写入时的 canonical projection。升级 renderer 默认只影响新记录；旧投影不自动重写，避免历史缓存前缀持续变化。显式迁移可以重写，但必须被记录为一次预期缓存失效。

### 5.3 Provider projection

同一 TimelineEntry 可以投影成不同 provider 协议:

- 原生 tool calling 的 assistant tool calls + tool results；
- Anthropic 风格 tool use/result；
- 只支持 user/assistant 的结构化 environment/user-return 块；
- Skill 加载后的可信 instruction result；
- 外部事件的普通数据块。

MemCore 不发送 HTTP 请求，但标准 projection adapter 应输出确定性的 provider messages。宿主自定义 adapter 必须返回最终采用的投影，以便 MemCore 保存 projection ledger。

### 5.4 终态 JSON 的双表示

终态助手输出同时保留:

- `provider_output_raw`: 模型实际产生的完整 JSON，用于精确 provider 历史重放和缓存；
- `semantic_text`: 解析后的 `speech`，用于检索、压缩和可读时间线；
- `memory_annotation`: 独立写回 annotation target，不塞进工具轮或自然语言正文。

## 6. 缓存稳定契约

MemCore 对自己的上下文段提供以下保证:

1. 已投影历史 append-only；新 entry 只追加到底部。
2. 相同 entry、renderer version、timezone 与 projection profile 必须产生相同字节。
3. 旧 entry 不因当前时间、debug 状态、动态工具可用性或新 renderer 发布而改变。
4. 工具调用、工具结果、事件和终态 JSON 都保存模型实际看到的稳定投影。
5. 压缩只替换完整、已关闭的旧前缀；压缩是显式、可审计的一次缓存破坏。
6. 每次投影返回:

```text
projection_version
stable_prefix_hash
entry_projection_hashes
compaction_generation
```

核心验收性质:

```text
未压缩时，下一轮输入历史以前一轮请求历史 + 前一轮真实输出为完整前缀。
```

MemCore 不能替 provider 承诺缓存服务一定命中，但必须保证传入 provider 的历史前缀可证明地相同，并在真实链路上验收 hit/miss tokens。

## 7. Token 压缩与并行工具

- 主触发条件使用实际 projected token，而不是消息条数；count policy 只保留为兼容模式。
- 压缩原子单位是已关闭的完整 turn，不能切断 stimulus/final 或 tool call/result。
- 有未返回并行调用、未完成 Skill 或尚无 final 的 turn 保持 open，不进入压缩批次。
- 多个并行结果按真实到达顺序追加，通过 `correlation_id` 关联，不依赖物理相邻。
- 普通记忆压缩视图优先使用 stimulus + assistant final；中间调用只提供必要的证据/结果摘要，不得成为用户稳定事实。
- 运行轨迹可以形成独立的 operation digest，但不能直接流入 personal semantic facts。
- 明确的“已结束/已取消/已清理”状态关闭旧 open loop，后来的状态优先。

## 8. 检索架构

### 8.1 一个真相源，多种检索视图

SQLite TimelineEntry 是真相源。向量/BM25 索引是可重建的加速层。索引可以为同一 entry/turn 建立不同视图:

- 普通记忆视图：获得普通检索准入的 stimulus、assistant final、episodic 和 semantic；
- 显式运行视图：尚未获得普通检索准入的事件，以及工具、材料、Skill 和其他 trace；
- 精确时间线视图：按真实时间和关系读取，不依赖向量相似度。

### 8.2 前置过滤是不变式

以下硬过滤必须在 semantic/BM25 评分前下推:

- hard namespace: `tenant_id/user_id/domain_id`；
- 当前 prompt 已可见 source ids 与当前输入排除；
- `retrieval_policy`/是否允许普通或显式检索；
- 用户明确指定的精确时间范围；
- 用户明确指定的 kind/kind prefix；
- 隐私、删除和保留状态；
- source layer 的硬限制（若调用方明确要求）。

可以逐级放宽的只应是软过滤:

- importance 下限；
- 语义 categories；
- subject scope；
- 未被调用方声明为硬要求的 source layer 偏好。

硬 namespace、当前可见排除、隐私状态和默认/显式检索边界永远不能因候选不足而放宽。`include_explicit=true` 是切换候选池，不是关闭其他过滤。

### 8.3 开放 kind 的前置过滤

开放 kind 不能退化成评分后的字符串过滤。Chroma metadata 只可靠支持标量，所以不能直接保存字符串数组 `kind_path`；索引 entry 应为每级祖先生成通用布尔 flag，例如:

```text
kind = tool.web_search.result
kind_prefix__tool = true
kind_prefix__tool_web_search = true
kind_exact__tool_web_search_result = true
```

真实字段名由 `kind_filter_key()` 做安全转义或稳定 hash。调用 `kind_patterns=["tool.web_search.*"]` 时直接转换为 `kind_filter_key("tool.web_search") == true` 的索引前置条件。该机制不需要预先枚举 tool、event 或 Skill 的具体类型，也适用于 InMemory 和 Chroma 两种索引。

### 8.4 关系扩窗

V2 不再使用 `seq_no ± 1/2` 猜邻居。规则:

- 普通命中 stimulus 或 final：返回同 turn 的 stimulus + final，跳过默认不可检索的中间 trace；
- 显式命中 tool call/result：返回对应 correlation branch，必要时补充 stimulus 和 final；
- 显式命中 event：返回事件及其关联 final；
- 并行工具不要求彼此物理相邻；
- 扩窗预算按 token，原子单位为 turn 或完整 call/result pair；
- 扩窗中的邻居继续遵守 retrieval visibility，不能绕过默认前置过滤把 trace 偷带回来；
- 无可靠 relations 的记录只保留自身，不做物理邻居 fallback；需要完整关系的宿主直接写 V2 turn/correlation。

## 9. 给模型的工具面

模型工具应该少、稳定、名字与用途明确。宿主写入接口不作为模型工具暴露。

### 9.1 `retrieve_for_turn`

用途：模糊回忆偏好、计划、人物、关系、主题、过去说法或无法确定具体日期的事件。

推荐的模型侧核心参数:

```json
{
  "query": "自然语言检索问题，必填",
  "keywords": ["可选关键词"],
  "categories": ["可选的受控语义分类"],
  "subject_scopes": ["user", "assistant", "other"],
  "time_hint": {
    "date_from": "YYYY-MM-DD",
    "date_to": "YYYY-MM-DD",
    "time_periods": ["morning", "afternoon", "evening", "night"]
  },
  "include_explicit": false,
  "kind_patterns": ["仅在需要工具/事件/Skill 轨迹时使用，例如 tool.web_search.*"]
}
```

公共 Python API 可以继续支持 source layers、importance 等高级参数，但默认 native tool schema 不应一次把所有内部旋钮暴露给模型。

模型说明必须明确:

- 当前可见上下文已有答案时不要重复检索；
- 不确定具体日期的旧偏好/关系/计划使用本工具；
- 默认只搜索普通记忆候选；
- 查工具、事件、Skill 过程时设置 `include_explicit=true`，并给出宿主授权的精确 kind/prefix pattern；
- 第一轮证据不足时允许调整 query 或过滤条件再次调用；
- 无证据时承认没有检索到，不得把空结果当成支持。

### 9.2 `read_timeline`

用途：读取昨天、上周二、某晚或明确日期范围内的精确时间线。

推荐参数:

```json
{
  "date_from": "YYYY-MM-DD，必填",
  "date_to": "YYYY-MM-DD，可选",
  "time_periods": ["morning", "afternoon", "evening", "night"],
  "include_explicit": false,
  "kind_patterns": ["可选，例如 event.finance.* 或 tool.web_search.*"]
}
```

默认返回可读的普通记忆时间线；需要核对工具、事件或 Skill 执行细节时显式打开 trace。精确日期过滤和 namespace 是硬过滤，不能因结果少而放宽。

### 9.3 `load_material`（宿主可选）

材料工具保持独立：先从可见上下文、检索或时间线获得 `file_id`，再由宿主 loader 读取原文件或 derived 内容。只有锚点、filename 或 pending 状态不代表模型已经看到了材料。

### 9.4 结构化工具结果

工具不应只返回字符串数组。推荐统一 envelope:

```json
{
  "status": "found",
  "reason": "",
  "effective_filters": {},
  "matches": [
    {
      "source_id": "...",
      "kind": "message.user",
      "timestamp": 0,
      "actor": {},
      "text": "...",
      "turn_id": "...",
      "relation_ids": []
    }
  ]
}
```

失败状态至少区分:

```text
found / empty / invalid / forbidden / unavailable / failed
```

具体错误通过稳定 reason code 区分，例如 `invalid_filter/invalid_time_range/forbidden_scope/internal_error`。非法过滤不能静默扩大成无过滤成功。返回给模型的 reason 应简短、可操作，不能包含密钥、路径或内部异常堆栈。

### 9.5 模型调用决策

| 问题 | 工具 |
| --- | --- |
| 当前可见上下文已有明确证据 | 不调用 |
| 偏好、关系、计划、人物、模糊旧事 | `retrieve_for_turn` |
| 昨天、上周二、某晚、明确日期 | `read_timeline` |
| 过去工具/事件/Skill 执行细节 | 对应工具并打开 `include_explicit` |
| 历史图片、文件、PDF 内容 | 先找 material anchor，再 `load_material` |

Native tool call 本身不包在最终表现 JSON 中；只有没有待处理工具调用的终态用户可见回复使用 chat output JSON contract。

## 10. 建议的 MemorySystem 公共生命周期

```python
turn = mem.begin_turn(
    stimuli=[...],
    annotation_target_ids=[...],
)

mem.append_action(..., turn_id=turn.turn_id, correlation_id=call_id)
mem.append_observation(..., turn_id=turn.turn_id, correlation_id=call_id)
# 高级接入仍可直接使用 append_entry(TimelineEntryInput(...))。
# record_tool_exchange() 同样必须接收当前开放的 V2 turn_id。

projection = mem.build_context_projection(provider_profile="openai_chat")

# 宿主执行模型和工具循环。

completed = mem.complete_turn(
    turn_id=turn.turn_id,
    provider_output_raw=raw_final_json,
    semantic_text=parsed.speech,
    memory_annotation=parsed.memory_metadata,
    annotation_status=parsed.metadata_status,
    provider_projection=actual_final_projection,
)

mem.compact_due_background()
```

独立消息、事件和材料便捷 API 已是 `append_standalone_entry()` 的薄适配；
`record_tool_exchange` 必须提供开放 `turn_id` 并委托 action/observation。权威实现只有统一 TimelineEntry/Turn API。

## 11. Schema migration 与兼容

Timeline V2 已有正式 schema version/migration runner。后续迁移应满足:

- 原数据库原地升级，失败时事务回滚；
- 旧 role 文本尽可能解析为 `kind`、call id 与 correlation；
- 旧 `event_trace/tool_trace/material_trace` 映射到 trace metadata，不自动伪造 model annotation；
- 旧记录没有 turn 关系时保留 legacy relation status，使用受限 fallback 扩窗；
- 不因升级重写旧 provider projection；
- SQLite 仍是真相源，向量索引可以删除后重建；
- 迁移报告结构化返回 migrated/skipped/ambiguous/failed 数量和 reason。

## 12. 分阶段实现记录

本节记录已经完成的落地顺序，不是当前待办。当前运行状态以文档顶部检查点和
`public_capabilities_v1.md` 为准。

### A. 当前版本稳定与真实发布

- 完成并提交当前附件串线防护与旧任务关闭规则；
- 建立真实 package version 和 release manifest；
- 保证本地、云端与测试安装的是同一构建。

### B. Timeline V2 与 migration foundation

- schema version；
- typed entry、turn、reply/correlation、actor/target；
- 开放 kind、payload、trace metadata；
- 旧 API 薄适配。

### C. Annotation 与终态提交

- 拆分 memory annotation/trace metadata；
- annotation target；
- terminal-only commit；
- 事件与普通输入统一的 annotation-driven retrieval admission。

### D. Projection ledger 与缓存验收

- canonical renderer + registry；
- provider projection adapter；
- raw final JSON/semantic text 双表示；
- prefix hash 与真实 provider cache acceptance。

### E. Retrieval/expansion V2 与模型工具

- 新 visibility/kind scalar flags 前置过滤；
- relation-aware expansion；
- token-bounded snippets；
- 新工具 schema、结构化结果与模型调用说明。

### F. Akane 回填与旧权威删除

- 收薄 Akane MemcoreManager；
- 删除 legacy `prompt_envelope_text` 权威路径；
- 不再由 Akane 私自渲染三层记忆或调用 MemCore 私有 reindex；
- 旧逻辑进入 deleted/thin adapter/documented migration window 三者之一。

## 13. 验收矩阵

### 存储与关系

- 普通消息、事件、Skill、工具和材料未知 kind 都可写入并稳定重放；
- 并行工具结果乱序到达仍能正确关联；
- final metadata 只写回明确 annotation target；
- provider user-role 的工具结果不会变成用户记忆。

### 检索与前置过滤

- 有效 annotation 的事件可按配置进入普通检索，无需事件白名单；
- 只有 trace metadata 的工具默认不进入普通检索；
- `include_explicit` 配合宿主授权的 kind pattern 可检索开放 kind；
- namespace、visible source ids、visibility 和 kind 硬过滤发生在向量/BM25 评分前；
- 被硬过滤的候选不会计算 embedding 相似度或 BM25；
- 软过滤放宽不突破硬边界。

### 扩窗

- 普通命中只返回 stimulus + final，不偷带中间 tool trace；
- 显式工具检索返回完整 call/result branch；
- 多工具并行不被固定邻居窗口切断；
- token 预算不足时按完整原子组裁剪并返回 truncated 状态。

### 缓存

- 未压缩连续轮次的 projected history 具有严格增长前缀；
- 普通消息、事件、工具结果交织不会重渲染旧前缀；
- renderer 升级不改变旧 entry projection；
- 压缩只造成一次可解释 miss；
- 真实普通对话、主动事件和多工具链路均记录 provider cache hit/miss tokens。

### 模型工具可用性

- 模型能区分模糊检索与精确时间线；
- 模型能在证据不足时主动调用并允许多轮工具使用；
- 默认工具 schema 不暴露无必要的内部旋钮；
- invalid filter、empty、unavailable 均有明确状态，模型不会把失败当证据；
- 普通回复、事件回复与工具后的终态回复都遵守唯一 final JSON contract。

## 14. 当前实现边界

V2 核心能力已经在 package 中可用；下面是仍然真实存在的兼容边界,不是待
实现的虚假占位:

- 旧 V1 raw 记录仍以 role/content 兼容字段保存,没有完整 turn/correlation 的
  历史数据会保留 `legacy_unlinked` 关系状态；
- `record_assistant_turn(..., in_reply_to=...)` 只生成 standalone entry，不会替旧数据凭物理相邻关系伪造 lineage；需要关系语义时使用 `complete_turn()`；
- standalone `record_external_event()` 使用显式 trace visibility；turn event 是否
  进入普通检索由 annotation target/status 决定；
- Akane 仍负责产品级 prompt assembly、provider transport、工具实际执行和权限；
  MemCore 提供 projection ledger 与 native memory tool schema/dispatch,不是渠道网关；
- 旧数据不会被静默重写 provider projection 或删除可追溯 lineage；没有 turn_id
  的记录作为 closed standalone component 进入同一个 V2 planner。

## 15. 代码审计后的实现修正

本节不是新的愿景，而是逐文件核对当前实现后，对前文方案作出的落地修正。

### 15.1 不新增第二套 raw 真相源

当前 `messages` 表已经承担以下权威职责:

- conversation 内原子递增 `seq_no`；
- 全局幂等 `source_id` 与跨 namespace/actor 冲突拒绝；
- raw 的 `index_status` outbox；
- `is_summarized/summary_id` 压缩 lineage；
- 精确日期读取、未摘要窗口和 namespace 删除。

因此 V2 不创建长期并行的 `timeline_entries` 表。物理上原地升级 `messages`，逻辑上把它定义为 TimelineEntry 表；`source_id` 继续作为唯一 entry id。公共 API 可以使用 `entry_id` 术语，但实现和返回值不能同时维护两套 ID。

`summaries` 与 `semantic_summaries` 继续保留，不重写三层记忆。它们的 `source_ids_json/source_summary_ids_json` 继续作为压缩 lineage，但需要增加可解析 lineage closure 的公共 Store API。

### 15.2 现有能力直接复用

以下实现应保留并扩展，而不是替换:

- `SQLiteMemoryStore` 的 RLock、单连接事务、source id owner 校验；
- pending/indexed/skipped 的 outbox 思路；
- `InMemoryVectorIndex._candidate_entries()` 先过滤再 cosine/BM25；
- Chroma where 的递归 `$and/$or` 翻译；
- category/scope 的布尔 metadata flags；
- RRF 双路融合与确定性分数/关系完整性降级；
- timezone、日期/星期/时间段渲染；
- Actor stable id/display name 分离；
- `StreamingSpeechParser` 只流式展示、最终完成才提交 metadata；
- `load_material` 的宿主 loader 与 `file_id` 一致性检查。

### 15.3 已从当前实现移除的旧机制

1. `role` 拼接 `event/tool/call_id` 作为权威结构。
2. `memory_metadata.categories` 同时表达语义类别和 trace 身份。
3. 请求任意 trace category 就关闭全部默认 trace 排除。
4. soft category 放宽顺带放开 trace 候选池。
5. raw 命中后按 `seq_no ± 1/2` 扩窗，且邻居不重新执行 retrieval visibility。
6. summary 确定性继承 `tool_trace/event_trace/material_trace` category；混合摘要会因此整条退出普通检索。
7. token policy 只数 `content`，并用 `role == "assistant"` 猜完整轮次。
8. `record_assistant_turn(..., in_reply_to=...)` 接收但忽略关系。
9. 每个 MemorySystem 懒建独立压缩线程，且 Compaction 锁只在单实例内有效。
10. 模型可直接传 `cross_conversation=true`，但 dispatcher 没有宿主权限对象再次约束。
11. prompt envelope 与 provider-native tool history 的权威实现仍留在 Akane。

### 15.4 审计时发现并已关闭的可靠性问题

以下描述的是 V2 实施前的审计结果，不是当前仍存在的缺口：

- `add_summary()` 与 `mark_messages_summarized()` 是两个事务；进程在中间退出可能留下重复摘要。semantic commit 同理。
- `get_record_by_source_id()` 是全局回表；检索当前依赖索引 hard where 保证安全，V2 关系读取必须再带 namespace owner 校验。
- visible-source 排除只排除当前可见 summary/semantic 自己的 ID，没有排除它们覆盖的 raw lineage，可能重复检索 prompt 已经概括的旧 raw。
- Chat Output Parser 在 `memcore_json` 中只强制 `speech`，缺少 `memory_metadata` 也会被 coerce 成空对象，无法区分有效模型 annotation 与 fallback/缺字段。
- OpenAI strict tool schema 会把所有可选字段变成 required + nullable；模型侧 schema 参数越多，请求越冗长且越难正确调用。
- Chroma collection 当前只按 embedding key 命名；索引 metadata schema 升级后缺少明确 index generation。

这些问题已在下列实现切片中收口，没有另开第二套重构主线。

## 16. SQLite V2 物理实现

### 16.1 Migration runner

新增 `memcore/store/migrations.py`，使用 `PRAGMA user_version` 作为数据库 schema version。SQLiteMemoryStore 初始化顺序固定为:

```python
open connection
-> BEGIN IMMEDIATE
-> detect/create latest base schema
-> run ordered migrations
-> PRAGMA user_version = target
-> COMMIT
```

要求:

- 当前无 `user_version` 但存在三张旧表的数据库识别为 schema v1；
- 每个 migration 先通过 `PRAGMA table_info` 检查，保证重复启动幂等；
- 任一步失败整次回滚，Store 构造返回结构化 `SchemaError`；
- 新建数据库直接创建 latest schema，不先建 v1 再 ALTER；
- migration 只改变 SQLite 真相源，不在事务中调用 embedding/vector backend；
- migration 完成后把受影响记录标为新 index generation 的 pending。

### 16.2 `messages` 新增列

建议在 v1 -> v2 中增加:

```text
kind TEXT NOT NULL DEFAULT ''
origin TEXT NOT NULL DEFAULT ''
turn_id TEXT NOT NULL DEFAULT ''
turn_role TEXT NOT NULL DEFAULT ''
reply_to_source_id TEXT NOT NULL DEFAULT ''
correlation_id TEXT NOT NULL DEFAULT ''
relation_status TEXT NOT NULL DEFAULT 'legacy_unlinked'

target_actor_id TEXT NOT NULL DEFAULT ''
target_actor_display_name TEXT NOT NULL DEFAULT ''

payload_json TEXT NOT NULL DEFAULT '{}'
semantic_text TEXT NOT NULL DEFAULT ''
trace_metadata_json TEXT NOT NULL DEFAULT '{}'

annotation_status TEXT NOT NULL DEFAULT 'unannotated'
annotation_source TEXT NOT NULL DEFAULT ''
retrieval_policy TEXT NOT NULL DEFAULT 'auto'
retrieval_visibility TEXT NOT NULL DEFAULT 'explicit'
semanticize INTEGER NOT NULL DEFAULT 1
prompt_visible INTEGER NOT NULL DEFAULT 1
trust TEXT NOT NULL DEFAULT 'untrusted_data'

renderer_id TEXT NOT NULL DEFAULT 'canonical'
renderer_version INTEGER NOT NULL DEFAULT 1
row_version INTEGER NOT NULL DEFAULT 1
index_schema_version INTEGER NOT NULL DEFAULT 0
index_key TEXT NOT NULL DEFAULT ''
```

兼容列语义:

- `content` 继续作为旧调用方可读的兼容列；V2 权威输入是 `payload_json + semantic_text`；
- `role` 继续保留给旧 adapter 和旧记录，不再承载新记录的 call id/payload；
- `memory_metadata_json` 表示 memory annotation；旧 standalone/迁移记录可能保留 trace category 兼容副本，但 V2 不以它判断 trace 身份或检索准入；
- `source_id` 就是 entry id，不添加重复主键。

当前使用的主要索引:

```sql
CREATE INDEX idx_messages_scope_turn
ON messages(tenant_id, user_id, domain_id, conversation_id, turn_id, seq_no);

CREATE INDEX idx_messages_scope_correlation
ON messages(tenant_id, user_id, domain_id, conversation_id, correlation_id, seq_no);

CREATE INDEX idx_messages_scope_date_visibility
ON messages(tenant_id, user_id, domain_id, conversation_id, date_label, retrieval_visibility, timestamp);
```

### 16.3 Derived memory 表新增列

`summaries` 继续承载 episodic/operation 两类 derived record，增加：

```text
kind TEXT NOT NULL DEFAULT 'memory.episode_summary'
trace_metadata_json TEXT NOT NULL DEFAULT '{}'
annotation_status TEXT NOT NULL DEFAULT 'derived'
retrieval_visibility TEXT NOT NULL DEFAULT 'default'
semanticize INTEGER NOT NULL DEFAULT 1
lineage_status TEXT NOT NULL DEFAULT 'valid'
compaction_schema_version INTEGER NOT NULL DEFAULT 1
row_version INTEGER NOT NULL DEFAULT 1
index_schema_version INTEGER NOT NULL DEFAULT 0
index_key TEXT NOT NULL DEFAULT ''
```

`kind=memory.operation_digest` 时 migration/default resolver 强制 `retrieval_visibility=explicit`、`semanticize=0`。`memory.episode_summary` 的 visibility 由有效 source lineage 推导，不能把 source categories 的 trace 并集直接复制过来。

`semantic_summaries` 增加：

```text
kind TEXT NOT NULL DEFAULT 'memory.semantic_summary'
annotation_status TEXT NOT NULL DEFAULT 'derived'
retrieval_visibility TEXT NOT NULL DEFAULT 'default'
lineage_status TEXT NOT NULL DEFAULT 'valid'
semantic_schema_version INTEGER NOT NULL DEFAULT 1
row_version INTEGER NOT NULL DEFAULT 1
index_schema_version INTEGER NOT NULL DEFAULT 0
index_key TEXT NOT NULL DEFAULT ''
```

derived record 的 `memory_metadata_json` 是摘要器产生并经 schema 校验的语义 metadata，不等于某条 raw 的 model annotation；`annotation_status=derived` 使检索能显式区分二者。任何 source/summary lineage 缺失、跨 Namespace 或成环时写 `lineage_status=invalid` 并从 index 排除。

### 16.4 `turns` 表

```text
turn_id TEXT PRIMARY KEY
tenant_id / user_id / domain_id / conversation_id
status: open / closed / aborted
stimulus_source_ids_json
annotation_target_ids_json
final_source_id
opened_at / closed_at
close_reason
row_version INTEGER NOT NULL DEFAULT 1
```

Store 写 turn/entry 时显式验证 owner；关系查询必须同时带 Namespace，不提供新的全局 `get_turn(turn_id)` 公共入口。

不在 turn row 维护第二份 pending correlation 列表。`commit_turn_completion()` 在同一事务中从 `messages` 的 action/observation relation 计算未闭合 correlation，避免 entry 与 turn 状态漂移。

### 16.5 `conversation_states` 表

新增一张仅保存协调状态、不保存记忆正文的表：

```text
tenant_id / user_id / domain_id / conversation_id
compaction_generation INTEGER NOT NULL DEFAULT 0
last_compacted_seq_no INTEGER NOT NULL DEFAULT 0
projection_generation INTEGER NOT NULL DEFAULT 0
row_version INTEGER NOT NULL DEFAULT 1
updated_at INTEGER NOT NULL

PRIMARY KEY(namespace...)
```

`commit_summary_batch()` 与 source 标记在同一事务中递增 `compaction_generation`；projection 显式迁移/重写时递增 `projection_generation`。普通 append 不改旧 generation。该表让缓存审计和 compaction snapshot 不需要用 `MAX(summary.timestamp)` 猜当前代数。

### 16.6 `prompt_projections` 表

一条 provider message 可能聚合多个并行 tool call，所以 projection 不能简单塞进单个 message 列。新增:

```text
projection_id TEXT PRIMARY KEY
tenant_id / user_id / domain_id / conversation_id
turn_id
projection_index INTEGER
provider_profile TEXT
payload_json TEXT
source_ids_json TEXT
payload_hash TEXT
projection_status: complete / canonical_fallback / media_omitted
projection_version INTEGER
created_at

UNIQUE(namespace..., turn_id, provider_profile, projection_index)
```

`payload_json` 保存安全、可持久化的单条 provider message 结构，例如 role/content/tool_calls/tool_call_id；不保存 API key、二进制图片、base64 或宿主内部路径字段（cached_path/storage_relpath/database_path 等）。投影语义 V3 起，模型完成任务所需的可执行路径证据（工具参数中的命令与路径、工具结果正文、assistant final 提到的路径）原样保留，不再用 omission marker 改写命令。相同 turn/profile 的多条 provider message 依 `projection_index` 排序。`record_request_projection()` 不重复保存每次请求的完整历史副本，而是验证/补齐这些 source-attributed messages，并把整次实际请求的 hashes 写入 audit。带不可持久化媒体的请求标记 `media_omitted`，该轮允许成为一次明确的缓存断点，但文字时间线和 derived material anchor 仍正常保存。

### 16.8 投影语义 V3:路径证据保真与旧投影迁移

V2 的路径投影把“所有本地绝对路径都不进入 provider projection”当作全局规则，导致模型发现并生成的命令（PowerShell counter、Windows/POSIX 路径、注册表命名空间）在投影里被换成 `[local path omitted from persistent history]`，下一轮无法原样复用。V3 改为:

- 可以进入模型历史:用户提供的路径、assistant 生成的命令与脚本参数、`exec_run/exec_status` 等工具结果正文里的操作路径、命令发现的路径。
- 仍然隐藏:API key/token/password/cookie；MemCore 数据库路径、宿主缓存与 run log 物理路径；`cached_path`/`storage_relpath`/`database_path` 等明确宿主内部字段；二进制媒体与 base64。
- 不再对所有字符串做绝对路径正则替换，也不再向 `/tmp` 注入 `$TMPDIR` 别名。宿主内部错误必须在进入 semantic tool result 前结构化，不依赖 MemCore 正则清洗全文。

迁移:`MemorySystem.migrate_legacy_path_projections(dry_run=False)` 在受控维护点(接流量前)显式执行,不在用户请求路径上运行。它扫描并分别报告三类旧数据:MemCore omission marker、宿主 `[local_path]` 脱敏、`$TMPDIR` 别名;只对 raw source 仍含原值的记录重投影(更新 payload/hash/status,`projection_version` 提升到 3),raw 已被宿主脱敏的记录保持原样并计入 `preserved_irrecoverable_host_redaction`,不猜测恢复。含旧 marker 或 full hash 已变化的 settlement 通过同一 settlement planner 原子重建,只有真正重新提交的卡片才计入 `settled_rebuilt`(noop/fallback 分别计数,重建失败则丢弃陈旧 settlement 回退到迁移后的 full ledger)。无 raw source 的行保持原样并计数报告;未受影响行字节不变;重复执行幂等。返回结构只含数量与版本,不含正文或路径。读取旧 frozen projection 时不做静默替换。主机通过 `list_projection_namespaces()` 枚举全部 namespace 做全库维护。

### 16.7 `projection_audits` 表

只保存 hash 和计数，不保存完整 system/persona/tool schema:

```text
tenant_id / user_id / domain_id / conversation_id
turn_id / attempt / provider_profile
model_route_hash
system_prefix_hash
tool_schema_hash
history_hash
full_prefix_hash
projection_version
media_omitted
created_at

PRIMARY KEY(namespace..., turn_id, attempt, provider_profile)
```

这张表用于解释缓存为什么变化，不参与记忆检索。

### 16.8 旧数据回填

只解析 MemCore 自己已知的旧 role:

- `user` -> `message.user`；
- `assistant` -> `message.assistant`；
- `event.X` -> 原 kind；
- `assistant.tool_call <name> <id>` -> `tool.<name>.call`；
- `tool.<name> <id>` -> `tool.<name>.result`；
- `user.attachment ...` -> `material.reference`；
- `system.material_cleanup ...` -> `material.cleanup`。

未知 role 保留原文并映射为安全的 `legacy.*` kind。迁移不得靠物理相邻猜造 turn；能从 call id 确定的 tool pair 只回填 correlation，其他记录保持 `relation_status=legacy_unlinked`，读取时走受限 legacy fallback。

旧 metadata 回填规则:

- Schema foundation 先把 trace category **复制**到 `trace_metadata_json`，annotation status 仍按非 trace 语义字段判断；
- 普通 user/event raw 中已有非 trace 语义 metadata 的记录标为 `accepted_legacy`；
- trace category 可以保留在旧 `memory_metadata.categories` 中作为兼容副本，但 V2 compaction/retrieval 只以 typed role、`kind + trace_metadata`、visibility 和 policy 为权威；
- 兼容副本不会打开候选池，也不会影响 V2 relation/compaction 分区；未来删除旧便捷 API 时可以单独清理，不要求破坏性重写历史记录；
- 不因迁移伪造模型 annotation。

这是明确的 compatibility boundary，不是双权威：

```text
兼容数据: 旧记录与 standalone 便捷 API 可保留 trace category 副本
运行权威: typed role + kind + trace_metadata_json + retrieval_visibility
准入保证: category 本身不能打开 explicit tool/event/material 候选
守护验证: 旧数据仍可迁移、压缩和经授权显式读取
```

## 17. Store 与 MemorySystem API 的具体改造

### 17.1 新类型文件

新增 `memcore/timeline.py`:

```text
TimelineEntryInput
TimelineEntry
TurnHandle
TurnCompletion
ProjectionMessage
RetrievalVisibility
TurnStatus
AnnotationStatus
```

`kind` 只校验安全 namespaced 语法和长度，不校验业务枚举。`turn_role/origin/retrieval_policy/trust` 是少量稳定的运行语义枚举。

`AnnotationStatus` 至少区分：

```text
unannotated
accepted_model / accepted_host / accepted_legacy
derived_turn_final / derived
missing / invalid / plain / fallback / rejected
```

只有 accepted 状态可以让 stimulus 自动进入普通检索；`derived_turn_final` 只由 `complete_turn()` 在 accepted target 的 final 上生成，`derived` 只用于通过 lineage 校验的 summary/semantic。其他状态不得被空对象或默认值提升为 accepted。

### 17.2 MemoryStore 新公共能力

在 `memcore/store/base.py` 增加:

```python
begin_turn(namespace, stimulus_entries, annotation_target_ids) -> TurnHandle
append_entry(namespace, entry) -> TimelineEntry
append_standalone_entry(namespace, entry) -> TimelineEntry
stage_message_memory_metadata(namespace, source_id, memory_metadata) -> record
commit_turn_completion(namespace, completion) -> CompletionCommitResult
get_entry(namespace, source_id) -> TimelineEntry | None
get_turn_entries(namespace, turn_id) -> list[TimelineEntry]
get_correlation_entries(namespace, turn_id, correlation_id) -> list[TimelineEntry]
get_compactable_turns(namespace, ...) -> list[TurnBundle]
resolve_lineage_source_ids(namespace, source_ids, cross_conversation, include_ancestors) -> LineageClosure
commit_summary_batch(namespace, records, source_assignments) -> CommitResult
commit_semantic_batch(namespace, records, summary_assignments) -> CommitResult
```

`get_record_by_source_id()` 暂时保留给 migration/compatibility；新运行路径不得用它绕过 namespace owner 校验。

Schema foundation 阶段，新增的 namespace-safe relation methods 在 `MemoryStore` 基类提供显式 `NotImplementedError` 默认实现，而不立即变成 abstract method；这样现有第三方 V1 Store 不会仅因升级包就在启动时失效。宿主启用 V2 runtime 前必须做 capability preflight，缺少这些实现时返回 `store_timeline_v2_unsupported`，不能退回全局 source-id 查询。SQLite 默认实现从 Slice 1 起完整支持。

### 17.3 原子终态提交

`commit_turn_completion()` 在一个 SQLite 事务中完成:

1. 校验 turn 是 open 且 owner 匹配；
2. 校验 annotation targets 属于同一 turn/namespace；
3. 校验每个必须完成的 action 已有对应 observation/error/cancelled；
4. 写 assistant final entry，并在 target accepted 时物化 `annotation_status=derived_turn_final`；
5. 把有效 memory annotation 写到明确 targets；
6. 解析并物化 targets/final 的 retrieval visibility；
7. 写 final provider projection；
8. 将受影响 entry 的 index status/version 置 pending；
9. 关闭 turn。

若还有未结束 correlation，返回 `status=pending_actions`，不伪造 final 和 metadata。向量 upsert 仍在 SQL commit 后执行；失败只留下 pending，不回滚已经完成的对话事实。

### 17.4 原子压缩提交

当前 add-summary 与 mark-source 分离的问题由 Store 原子 API 修复:

- 一次 batch 可以提交 episode summary 和 operation digest 多条 derived record；它们的 source assignments 必须构成不重叠的 source partition；
- 每条 summary id 根据 namespace + ordered source ids + compaction schema version + summary profile 稳定生成，重试幂等；
- SQL commit 前确认所有 source 仍未摘要且属于相同合法批次；
- 同一事务插入全部 derived records、标记全部 source 并递增 conversation compaction generation；
- 冲突返回 `status=stale_batch`，Compaction 重新读取，不重复生成第二条摘要；
- semantic commit 使用同样方式处理 summary lineage 与 reinforcement update。

### 17.5 MemorySystem 新生命周期

```python
turn = mem.begin_turn(
    stimuli=[TimelineEntryInput(...)],
    annotation_target_ids=[...],
)

mem.append_entry(..., turn_id=turn.turn_id)
mem.append_entry(
    TimelineEntryInput(
        kind="tool.web_search.call",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.ACTION,
        semantic_text="搜索北京天气",
        correlation_id=call_id,
        trace_metadata={"tool_name": "web_search"},
    ),
    turn_id=turn.turn_id,
)

result = mem.complete_turn(
    turn_id=turn.turn_id,
    semantic_text=parsed.speech,
    provider_output_raw=raw_final,
    memory_annotation=parsed.memory_metadata,
    annotation_status=parsed.metadata_status,
    provider_projection=actual_final_projection,
)
```

上例是当前 V2 API。`provider_projection` 会把实际 provider-visible final message 与
annotation/final/turn close 一起写入 projection ledger；若宿主没有提供实际投影,
应明确省略它或返回结构化不可用状态,不要伪造 provider history。

Akane 的 live model turn 已切到本节 V2 API。package 内仍保留的独立消息/事件/
材料便捷名字只调用 typed standalone writer；它们不是第二套权威。工具便捷入口
必须关联开放 turn。

### 17.6 Standalone entry 与 staged annotation

两种宿主行为不应伪造成完整 turn：

- 被动群消息、材料清理、状态同步等不触发模型回复的事实使用 `append_standalone_entry()`；其 `turn_id/turn_role/correlation_id` 为空，`relation_status=standalone`，不会制造永远 open 的假 turn；
- final JSON 已解析但 assistant final 尚未原子提交时，使用 `stage_turn_metadata()`；它只更新 open stimulus 的规范化 metadata，并保持 `annotation_status=unannotated + retrieval_visibility=explicit`，不会提前让半轮对话进入普通检索；
- `complete_turn()` 仍是唯一把 annotation、assistant final、visibility 与 turn close 一起提交的权威；staged metadata 不替代 completion annotation。

SQLite standalone 写入对相同 source id + 相同 typed payload 幂等；同 id 不同内容/策略结构化拒绝。staged annotation 只接受当前 Namespace/actor 下仍 open 的 stimulus，standalone、intermediate、final 或 closed turn 不得假成功。

### 17.7 Retrieval admission 的物化

存储 `retrieval_policy` 表达调用方意图，同时存储 `retrieval_visibility` 供索引硬过滤。默认解析:

```text
always -> default
explicit -> explicit
never -> never
auto + accepted_model/accepted_host/accepted_legacy annotation -> default
auto + unannotated/fallback/rejected -> namespace 配置，默认 explicit
```

当 target 获得 default visibility，同一 turn 的 assistant final 继承 default，并获得 `derived_turn_final` 状态，因此既可以独立参与语义评分，也可以通过关系扩窗与 target 成对返回；中间 intermediate/action/observation 保持 explicit。配置变化若需要重算旧记录，必须调用结构化 maintenance API 重新物化 visibility 并 reindex，不能在查询时悄悄改变历史行为。

## 18. 渲染与 Projection Ledger 实现

### 18.1 文件安排

- 保留 `memcore/rendering.py` 中已有时间、Actor、summary/semantic 渲染；
- 在其中增加 `RendererRegistry` 与 canonical fallback，或在代码超过单文件边界时拆为 `memcore/rendering/` package；
- 新增 `memcore/projection.py`，定义 provider-neutral `ProjectionAdapter` 与标准 profile；
- 标准首批 profile: `canonical_user_assistant`、`openai_chat`、`anthropic_messages`。

### 18.2 Renderer resolution

解析顺序:

```text
exact kind renderer
-> longest registered prefix renderer
-> canonical fallback
```

renderer 输出必须包含稳定 header、Actor、kind 和 payload data block。现有 `event` renderer 中财经字段的特殊优先级移到 `event.finance` renderer；canonical fallback 对未知字段采用稳定 key sort。

`trusted_instruction` 只能由宿主已授权的 Skill/指令加载路径设置；未知 kind 默认 `untrusted_data`。通用 renderer 必须把 untrusted payload 当数据引用，不允许 payload 文本改变 system/tool 边界。

### 18.3 两阶段投影记录

MemCore 不组装宿主完整 system prompt，因此采用两阶段协议:

1. `build_context_projection()` 从已保存 projection/canonical renderer 生成历史。
2. 宿主把人格、工具、动态当前上下文组合成最终 provider request。
3. provider adapter 完成 role/content/tool call 归一化、真正发出请求前，宿主调用 `record_request_projection()`，传入当前 turn 实际发送的安全 conversation messages 与完整请求 hash。
4. 模型终态返回后，`complete_turn()` 保存 raw final output、semantic speech 和 final provider projection。

这样 MemCore 不接管模型 HTTP，却保存真正影响下一轮历史的 provider message。system/persona/tool schema 只保存 hash，不复制敏感 prompt。

请求冻结以 **projection message** 为粒度，而不是以整个 turn 为粒度。某条 projection 第一次经过真实 transport
边界时，可以用 observer 看到的安全实际 payload 替换 canonical fallback，并标为 `request_frozen`；之后该条
projection 不得再改变。同一 open turn 在后续工具轮追加的 action/observation/material projection 尚未经过请求
边界，因此仍可分别完成自己的第一次冻结。这样第一轮 user 已冻结，不会阻止第二轮新增工具消息进入 ledger，
也不会允许后续请求反过来改写已发送的 user/tool payload。

### 18.4 Provider 切换

结构化 `payload_json + relations` 是跨 provider 权威。若下一轮使用不同 profile:

- 优先读取该 profile 已保存的 projection；
- 没有时从结构化 entry 重新投影；
- profile 变化定义为新的 cache family，首次 miss 是预期行为；
- 不能拿 OpenAI tool message 原样伪装成 Anthropic history。

### 18.5 Chat Output Adapter 调整

`ChatOutputParseResult` 增加:

```text
metadata_status: accepted / missing / invalid / plain
metadata_present: bool
```

`mode=memcore_json` 时 speech 的交付状态与 annotation truth 分开：合法 object（包括模型明确输出的空 object）为 `accepted`，字段缺失为 `missing`，错误类型为 `invalid`；字段值继续通过 coerce 约束。缺失或错误 metadata 不得冒充有效 annotation，但也不阻断已经生成的 speech、流式显示或 TTS。宿主 fallback 的模板空对象保持 `missing`，只有宿主确实归一化出有效记忆信号时才可提升为 `accepted_host`。`auto/plain` 可以得到 speech，annotation status 为 `plain/missing`。

Prompt 文案从“本轮用户原始消息”改为“本轮由宿主指定的记忆标注目标（可能是用户消息或触发回复的外部事件）”。工具中间轮仍不使用 final JSON contract。

### 18.6 媒体与缓存例外

图片/二进制不得为了缓存完整性写入 MemCore。带原生图片的 turn:

- 持久化安全文本 envelope、material source id 与 derived anchor；
- projection 标记 `media_omitted`；
- 缓存审计把该轮列为明确断点；
- 后续历史使用安全文字/derived projection，不假装仍附有原图。

若图片由工具在本轮中产生或加载，宿主必须保持最初 stimulus/user projection 不变，并在线性时间线中按
`tool action -> tool observation -> material.model_input` 追加媒体输入。真实图片块只附在最后这条新增
provider message 上；observer 冻结时把图片替换为稳定 omission marker。禁止为了省事把工具图片重新塞回
最初 user message，因为那会改写已冻结投影并破坏缓存前缀与审计真实性。

这是一条明确安全边界，不用本地路径或 base64 换取表面上的 100% 前缀一致。

### 18.7 Slice 3 当前实现

已新增 `memcore/projection.py`，并由 `MemorySystem` 暴露：

```python
projection = mem.build_context_projection(provider_profile="openai_chat")

recorded = mem.record_request_projection(
    turn_id=turn.turn_id,
    provider_profile="openai_chat",
    turn_messages=current_turn_actual_messages,
    history_messages=actual_history_messages,
    attempt=1,
    model_route=model_route,
    system_prefix=system_prompt,
    tool_schema=tools,
)

mem.complete_turn(
    ...,
    provider_profile="openai_chat",
    provider_projection=actual_final_assistant_message,
)
```

`record_request_projection()` 的 `turn_messages` 必须是当前 turn 到本次请求为止的完整、带 source attribution 的 provider messages，而不是只传本次新增长度。Store 逐 profile 强制每个 prompt-visible source 只映射一次、projection index 连续、source coverage 只能按时间线前缀增长；任何覆盖旧 row、跳号或只写 final 的请求都在事务内拒绝。

生成重试也属于真实请求边界：同一 MemCore turn 的重试必须复用相同的 current user payload、历史和动态上下文，
不得在 user 尾部临时拼接 retry note。重试原因可以进入宿主日志/指标，但不能偷偷改变可持久化 conversation。

`system_prefix`、`tool_schema` 与 `model_route` 只参与 hash audit，不复制进数据库。source-attributed `system/developer` message 禁止持久化；base64/原生媒体、密钥形态与宿主内部路径字段（cached_path/storage_relpath/database_path 等）会被稳定 omission marker 替代并产生 `media_omitted/skipped_unsafe` 状态；可执行路径证据（V3 起）原样保留。不同 provider profile 是不同 cache family，OpenAI tool calls 和 Anthropic tool use/result 不互相冒充。

Slice 3 最初只冻结未摘要 V2 raw timeline 与安全 legacy-unlinked 单条记录；Slice 4 已补齐 episode/semantic summary 投影、provider payload token 预算与 compaction generation。Akane 当前主链已读取 MemCore provider projection；产品级 system/persona/tool schema 仍由宿主在稳定历史之前组装。

## 19. Compaction V2 与共享 Runtime

### 19.1 压缩原子从 message 改为 terminal turn

当前按消息数量和 `role == "assistant"` 猜测轮次结束的实现不能继续扩展。V2 的可压缩单位是 `TurnBundle`：

```text
TurnBundle
├── turn row(status=closed)
├── ordered stimulus entries
├── zero or more action/observation branches
└── exactly one final entry，或显式 aborted terminal
```

一个 turn 只有同时满足以下条件才可进入 compaction snapshot：

- `turns.status` 是 `closed` 或允许摘要的 `aborted`；
- 所有 action correlation 都已经以 observation/error/cancelled 结束；
- turn 中不存在 `index_status=writing` 或未提交的 projection；
- turn 的所有 entry 都属于同一 Namespace 和 conversation；
- turn 没有被当前请求、检索结果 pin 或 maintenance job 锁定；
- turn 位于可压缩的连续旧前缀中，而不是从时间线中间任意挖洞。

`open` turn、pending correlation 和正在重试的终态输出永不压缩。旧数据没有可靠 `turn_id` 时，只允许 migration 产生的保守 `LegacyTurnBundle`：明确的单条 user + assistant pair 可以成组，其余未知结构按单条不可拆单元处理，不靠相邻 role 猜造并行工具关系。

### 19.2 Token 预算按实际投影计算

`MemoryConfig` 的 raw 投影预算配置：

```text
raw_token_trigger
raw_token_batch_ratio
compaction_min_recent_turns
projection_profile
```

唯一的 raw planner 使用目标 provider profile 的 tokenizer 计算下列完整内容：

- role/header；
- canonical/provider content；
- tool call 名称、参数、call id；
- tool result envelope；
- 终态 raw JSON；
- message framing 的固定开销。

不能只统计 `entry.content`。没有 provider tokenizer 时使用明确的保守估算，并在结果中返回 `token_count_quality=estimated`；不存在 count compatibility planner。

当未摘要 raw projection 达到 `raw_token_trigger` 时，压缩从最老的连续
terminal-turn 前缀（`closed` 或 `aborted`）开始，累计到
`raw_token_trigger * raw_token_batch_ratio`，并以完整 relation component 为实际
切点。token 只负责规划，条目/turn/component 负责原子边界；
任何独立消息 batch cap 都不得提前截断本轮目标。通常保留至少一个可配置的近期完整 turn；若单个完整 terminal
history 自身已经超线且不存在合法尾部，允许整块压缩，避免永久超预算。
`aborted` 只总结真实已提交条目，不补造 assistant final；若第一个 `open` turn
已导致超预算，返回结构化 `status=blocked_by_open_turn`，不得切断该 turn 来伪装成功。

### 19.3 两阶段原子提交与幂等重试

一次压缩按以下顺序执行：

1. 在 conversation compaction lock 下读取连续候选 turn，生成不可变 snapshot；
2. snapshot 保存 ordered source ids、row versions、projection hashes 和 compaction generation；
3. 释放 SQLite 写事务后调用摘要模型，避免长事务占锁；
4. 重新获取锁并调用一次 `commit_summary_batch(records, source_assignments)`；
5. Store 在单个 `BEGIN IMMEDIATE` 事务中重新验证 snapshot；
6. 插入 summary、标记全部 source、更新 generation 和 index pending 状态；
7. SQL commit 后异步 upsert index；失败保留可修复的 pending 状态。

每条 summary id 稳定计算：

```text
sha256(namespace canonical bytes
       + conversation_id
       + ordered source_ids
       + compaction_schema_version
       + summary_profile)
```

相同 snapshot 重试得到相同 id；任意 source 已被其他任务处理、row version 改变或 prefix 不再连续时，返回 `status=stale_batch`，调用方重新规划，不能再插入一条内容相近的新 summary。

`commit_semantic_batch()` 对 episodic -> semantic 的 lineage 使用同样协议。现有 `add_summary()` 后再调用 `mark_messages_summarized()` 的分事务路径进入 documented migration window，V2 主链路不再调用。

### 19.4 语义摘要和运行摘要分离

V2 不再把 source metadata 的 category 并集原样继承给 summary。尤其不能因为一个批次含有 `tool_trace` 或 `event_trace`，就把包含普通对话事实的整条 summary 变成默认不可检索。

压缩输出逻辑上分为两种：

- `memory.episode_summary`：总结 annotation targets、助手终态结论和对人物/偏好/计划有意义的事实，可进入 episodic/semantic 生命周期；
- `memory.operation_digest`：记录工具名称、执行状态、材料来源和错误等运行轨迹，默认 `retrieval_visibility=explicit`、`semanticize=false`。

同一个 turn 可同时贡献两种输出，但二者使用不同且不重叠的 source lineage 和 index entry：stimulus/intermediate/final 归 episode，action/observation/material trace 归 operation digest；两条 record 在同一 `commit_summary_batch()` 事务提交，保证整个 terminal turn 要么一起被替换，要么都不替换。个人语义强化只读取 accepted annotation、final semantic text 和已有 semantic lineage；原始 tool observation 不能仅因物理上夹在用户与 final 之间就进入 personal semantic memory。

工具结果中真正被模型采用的事实通常已体现在 assistant final；若宿主需要让某个结构化材料长期可查，应把它保存为有自身语义策略的 `material.*` entry，而不是把所有 tool result 提升为普通记忆。

### 19.5 新增共享 `MemCoreRuntime`

新增 `memcore/runtime.py`：

```python
class MemCoreRuntime:
    compaction_executor: Executor
    index_executor: Executor
    lock_registry: ConversationLockRegistry

    def submit_compaction(...): ...
    def submit_index_repair(...): ...
    def close(...): ...
```

`MemorySystem` 接受显式 `runtime` 注入；未传入时可为简单单实例应用创建 owned runtime，但必须记录 ownership，只有 owner 才能 close。Akane 同一进程中的全部 bot/manager 注入同一个 runtime，不再每个 `MemorySystem` 各建一个压缩线程。

conversation lock key 必须包含：

```text
store_identity + tenant_id + user_id + domain_id + conversation_id
```

其中 `store_identity` 是规范化数据库身份或 Store 提供的稳定实例 identity，不能只用 conversation id，也不能包含要写进日志/prompt 的绝对路径。锁保护 compaction plan/commit 和同 conversation 的 projection generation 更新；不同 conversation 可并行。

SQLite 约束和 snapshot revalidation 仍是最终一致性边界，本地进程锁只是减少重复工作。多进程部署必须依赖数据库原子提交判断胜者，不能把 Python lock 当分布式锁。

### 19.6 压缩结果与可观测性

`CompactionResult` 至少返回：

```text
status: compacted / not_due / busy / stale_batch / blocked_by_open_turn / failed
source_turn_count / source_entry_count
before_projected_tokens / after_projected_tokens
token_count_quality
summary_source_ids
compaction_generation
index_status
reason
```

日志只记录 namespace 的不可逆短 hash、数量、版本和状态，不记录正文、宿主内部路径或 provider payload。投影语义 V3 起，模型完成任务所需的操作路径可以进入工具结果与会话历史，但不进入日志与诊断快照。

### 19.7 当前实现检查点（2026-07-20）

- `MemorySystem` 可注入进程级共享 runtime；未注入时创建 owned runtime，只有 owner 的 `close()` 会关闭 executor；
- conversation lock 使用安全 `store_identity + namespace + conversation`，SQLite 文件 identity 只保存规范化路径 hash，不暴露绝对路径；
- V2 压缩只选择最老的连续 terminal-turn 前缀，并至少保留配置的近期完整 turn；`aborted` 可按真实条目压缩但不伪造 final，只有 `open` 前缀返回结构化阻塞状态；
- projected-token 口径包含 provider role/content、结构化 payload、工具批次和固定 framing；无 tokenizer 时明确返回 `estimated`；
- `commit_summary_batch()` 与 `commit_semantic_batch()` 使用 `BEGIN IMMEDIATE`，在单事务内重验 generation、row version、turn status、连续前缀和 projection hash；
- 普通 episode 与 operation digest 使用互不重叠的 source lineage 同批提交；operation digest 默认 explicit 且不进入长期语义压缩；
- 压缩后的 summary/semantic 继续走统一 projection ledger，`after_projected_tokens` 与实际可见 provider payload 使用同一计数口径；
- 索引失败保留 pending outbox 并在结果中返回 `index_status=pending`，不回滚已提交的 SQL 事实；
- V1 `record_*` 历史数据在没有 V2 turn 时作为 closed standalone component 进入同一 V2 planner；Akane、QQ、桌宠、金融和个人 Bot 的当前主链均已切换，状态见文档顶部检查点；
- Retrieval admission 与 Relation expansion 已完成：visibility/kind/policy/index generation 在评分前硬过滤，随后按 turn/correlation/lineage 扩成受 token budget 约束的原子结果；Native Memory Tools 的 schema/dispatch 已完成，宿主 provider transport 仍由接入方负责。

## 20. Retrieval V2 的具体实现

### 20.1 查询先编译成不可放宽的 QueryPlan

新增或扩展 `memcore/retrieval.py` 类型：

```text
RetrievalRequest
RetrievalQueryPlan
HardFilterPlan
SemanticFilterPlan
RelationExpansionPlan
RetrievalMatch
RetrievalResult
```

入口先完成权限和参数校验，再把查询编译成 plan。`HardFilterPlan` 一旦生成，在 dense、BM25、RRF、relaxation 和关系扩展各阶段都不可改变：

- 完整 Namespace；
- conversation scope 和宿主授权的 cross-conversation scope；
- `retrieval_visibility`；
- `annotation_status` 对应的准入规则；
- kind exact/prefix flags；
- trust/trace admission；
- 精确 time range；
- visible source ids 及其 lineage closure；
- 已删除、已替代和当前不可见 generation。

允许逐级放宽的只有 `SemanticFilterPlan`：categories、importance、subject scopes、keywords 与相似度阈值。放宽语义条件不得关闭 Namespace、kind、visibility、时间或 lineage 过滤。

### 20.2 Chroma 使用标量 kind flags

Chroma metadata 不能可靠表达数组层级包含，因此不采用 `kind_path=[...]`。`memcore/index/metadata_filters.py` 增加纯函数：

```python
kind_prefixes("event.finance.flash")
# ["event", "event.finance", "event.finance.flash"]

kind_filter_key("event.finance")
# 确定性的安全标量 metadata key
```

每条 index document 写入：

```text
kind_exact = "event.finance.flash"
<kind_filter_key("event")> = true
<kind_filter_key("event.finance")> = true
<kind_filter_key("event.finance.flash")> = true
```

`kind_filter_key()` 使用固定 schema version、规范化 prefix 和无碰撞风险的完整 digest 编码；查询端使用同一函数生成 Chroma `where`。开放的新 kind 自动得到 flags，不要求修改枚举或发布新 MemCore 版本。

模型/公共 API 的 `kind_patterns` V2 只接受两种可前置过滤语法：精确 kind 和尾部 `.*` 的 prefix，例如 `tool.web_search.result`、`event.finance.*`。任意中段 glob/regex 返回 `invalid_filter`，不允许先召回全库再用 Python 后过滤。

### 20.3 visibility 与 explicit trace 的准入

普通候选的准入规则：

- raw stimulus 必须是 `retrieval_visibility=default` 且 annotation status 为 `accepted_model/accepted_host`，或由宿主明确设置 `retrieval_policy=always`；`accepted_legacy` 不进入 V2；
- assistant final 在 accepted turn 中物化为 `retrieval_visibility=default + annotation_status=derived_turn_final`，可以独立评分，也可通过 turn relation expansion 随 stimulus 返回；
- derived summary 必须是 `annotation_status=derived`，并拥有已验证 lineage 和独立的 derived visibility；
- `never` 永远不可检索；
- `explicit` 只有请求与宿主 policy 同时允许时进入候选。

`include_explicit=true` 只是把 visibility 候选池从 `default` 切换为 `default + explicit`，不是“关闭所有 trace 排除”。若要检索 `tool.*`、`material.*` 或其他 trace，调用方还必须提供匹配的 `kind_patterns`，且该 prefix 在 `ToolDispatchPolicy.allowed_kind_prefixes` 中。未指定 kind 时，explicit 只增加非 trace 的显式记忆，不把任意工具轨迹混入候选。

categories 不再承担 trace 开关。即使用户显式传入一个旧 `tool_trace` category，也只能由 compatibility layer 转译为一个具体、获授权的 kind filter；它不能导致其他 tool/event/material 轨迹一起进入候选。

### 20.4 真正的评分前过滤

一次检索按以下顺序执行：

1. Store/Index 根据 HardFilterPlan 生成合法候选集合；
2. dense query 使用 Chroma `where`/ids 在合法集合内计算距离；
3. BM25 只从同一合法集合构建/读取 postings；
4. 两路结果进入 RRF；
5. 对语义条件执行有界 relaxation；
6. 按 raw-first 规则确定性选取候选；
7. 关系扩窗并执行 token 裁剪。

不得先对全 Namespace 计算 cosine，再用 Python 删除 tool trace。若 Chroma 方言或候选 ids 数量无法表达所需 hard filter，返回 `status=unavailable, reason=index_filter_unsupported`，或走同样先由 SQLite 建候选集的安全实现；不能静默退化为后过滤。

BM25/dense 均应有最低有效分数或空查询诊断。“零相似度候选仍返回”的问题由 `min_dense_score/min_bm25_score/min_fused_score` 与 `diagnostics.rejected_counts` 显式处理。阈值允许配置，但 `score=0` 不得仅因为 top-k 未填满而成为有效记忆；不能再增加一次模型调用替确定性检索收尾。

### 20.5 Lineage closure

`MemoryStore.resolve_lineage_source_ids()` 已从当前授权 Namespace 的 raw/episodic/semantic 记录计算向下与向上的传递闭包：

```text
semantic -> source episodic summaries
episodic summary -> source raw entries/turns
replacement -> replaced generation
```

当一个高层记录已经代表某组 source 时，同一次结果默认排除其全部下层 lineage，避免 summary 和原文重复占用 token。visible-source exclusion 同时包含上下游 closure，避免当前已可见 raw 从 summary/semantic 旁路重复返回。反过来，当调用方明确请求 raw timeline 时，可以排除 derived 层并保留 raw。

visible-source 过滤对 raw、summary、semantic 使用同一 closure；不能只过滤 raw source id，却让包含该 source 的 summary 从另一层绕回来。发现断裂/跨 Namespace lineage 时，该 derived record 标记 `index_status=invalid_lineage` 并排除，不猜测修复。

### 20.6 关系扩窗替代 `seq_no ± N`

普通命中的最小返回原子是：

```text
annotation target stimulus + 对应 assistant final
```

若命中的是 final，则反向解析同一 turn 的 annotation target。默认不夹带中间 tool branch；这样用户问“我们之前谈过什么”时不会被参数和工具回包污染。

显式请求工具轨迹时，扩窗单位变为完整 correlation branch：

```text
assistant action(call_id=X)
-> observation/error/cancelled(call_id=X)
```

并行工具按 turn 中的真实 `seq_no` 展示，但关联按 `correlation_id`，不按结果返回先后猜测。请求完整 turn 时可返回 stimulus + 所有完整 branch + final。

扩窗受独立 `result_token_budget` 限制，并按原子组裁剪：一个 stimulus/final pair 或 call/result branch 要么完整保留，要么整体舍弃；绝不截掉 tool result 只留下 call。若单个原子组本身超预算，返回保留完整 `source_ids` 的 stable excerpt/material anchor 和 `truncated=true`，不伪造完整结果。只有注入真实 `TokenCounter` 才执行预算；显式设置预算却缺少 counter 时返回 `unavailable/token_counter_required`，不按字符比例猜 token。

### 20.7 结构化检索结果

MemorySystem 内部和 native dispatcher 使用结构化结果，不再先渲染成 `list[str]`：

```text
RetrievalResult
├── status: found / empty / invalid / unavailable / failed
├── matches[]
│   ├── source_ids / turn_id / correlation_id
│   ├── kind / layer / timestamp / score breakdown
│   ├── semantic_text / rendered_text
│   └── lineage / truncated
├── effective_filters
├── relaxation_steps
├── rejected_counts
├── token_usage / truncated
└── reason
```

旧 `retrieve_for_turn() -> list[str]` 保留薄 adapter，只读取 `matches[].rendered_text`。新工具、Akane 回填和测试使用结构化 API，防止 found/empty/error 再被混成一个空列表。

### 20.8 Index schema version 与重建

Index key 至少包含：

```text
embedding_profile
index_schema_version
kind_flag_schema_version
visibility_schema_version
renderer/semantic_text schema version
```

SQLite 保存当前 index generation 和每条 entry 的 indexed generation。kind flags/visibility 语义改变时创建新 Chroma collection generation 或执行明确的全量重建；新 generation warmup 完成并通过 count/hash 检查后再原子切换。不得在旧 collection 中部分覆盖，让相同查询同时看到两种过滤语义。

### 20.9 当前实现检查点（2026-07-21）

- `RetrievalRequest -> RetrievalQueryPlan -> RetrievalResult` 已成为唯一检索算法；旧 `retrieve()/retrieve_for_turn() -> list[str]` 只渲染结构化 matches，不再维护第二套搜索或过滤逻辑；
- HardFilterPlan 固定 Namespace、conversation scope、time range、visibility、annotation、kind patterns、trust、lineage 状态、visible source ids 与 index schema generation；semantic relaxation 只允许依次移除 importance、categories、subject scopes；
- `kind_patterns` 仅接受 exact kind 或尾部 `.*` prefix；kind flags 使用 versioned SHA-256 完整 digest，新增业务 kind 不要求修改 MemCore 枚举；
- 普通 default admission 接受有效 model/host annotation、derived summary/final 和宿主明确 `always` policy；`tool.*`/`material.*` 只能通过 `include_explicit + kind_patterns` 打开；
- 旧 `tool_trace/event_trace/material_trace` category 不再能扩大候选池，迁移记录的 `accepted_legacy/legacy_categories` 也不进入 V2 检索；
- index entry 已物化 visibility、annotation、kind、trust、lineage 和 schema-generation 标量；Chroma collection 名包含 index/kind/visibility schema generation，避免新旧过滤语义混在同一 collection；
- dense/BM25 的零分候选不会为了填满 top-k 被返回；配置阈值与 rejected counts 已进入结构化结果；
- Store 回取使用 Namespace-safe raw/summary/semantic lookup。若第三方 index 忽略 hard where 并返回越界 source，整次查询返回 `unavailable/index_filter_unsupported`，不会靠后置删除伪装成功；
- Retrieval admission 提交点仍以单记录 seed 为输入；实际返回原子组、lineage closure 和 result token budget 已由下节 Relation expansion 接管；
- 该切片提交时没有切换 Akane 或云端 Bot；当前状态见文档顶部检查点。

### 20.10 Relation expansion 实现检查点（2026-07-21）

- raw seed 按真实 turn/correlation 关系扩成原子组，默认对话只返回 annotation target stimulus + assistant final，不夹带 intermediate/action/observation；候选取舍由分数、硬过滤和关系完整性确定，不追加 LLM verifier；
- 显式 `tool.*` seed 只返回同一 `correlation_id` 下的完整 action/terminal observation branch；未闭合 branch 计入 `incomplete_relation` 并整体拒绝；
- `event.*` seed 返回事件及其关联 final；关系邻居仍重新检查 Namespace、conversation、trust、visibility、annotation 与 never policy；
- summary/semantic 使用 Store lineage closure；semantic 命中会压掉其 source summary/raw，断裂、循环或跨 Namespace lineage 会从 index 隔离并计入结构化 diagnostics；
- `retrieve_for_turn` 的 visible exclusion 改为非对称 lineage：排除可见记录及其上层 derived 副本，避免已见 raw 从 derived 层绕回；可见 summary/semantic 的下游 raw 证据不再被永久排除，可从历史中语义找回具体原话（2026-08-06 修正生产"已摘要即不可命中"问题）；
- token budget 对序列化后的完整 match 计数。可容纳时保留完整组，剩余预算不足时整体省略；单组自身超预算时只返回显式 truncated anchor，`semantic_text` 不保留未计费副本；
- 没有 relations 的旧记录不会再做 `seq_no ± N` 推测，只返回自身；Akane 与云端 Bot 当前已经使用这套关系扩窗语义。

## 21. 给模型的 Native Memory Tools

本节能力已经落地在 `memcore/native_tools.py`。MemCore 生成 provider-specific
schema 并做参数/权限边界检查，宿主仍负责把返回值送回模型 provider 的原生
tool-result 通道。

### 21.1 宿主权限对象，而不是模型自授权

当前实现的宿主授权对象是：

```python
@dataclass(frozen=True)
class ToolDispatchPolicy:
    allow_explicit_trace: bool = False
    allowed_kind_prefixes: tuple[str, ...] = ()
```

模型即使生成任意 `kind_patterns`,也只能得到宿主授予的前缀范围；namespace、
当前 turn 和结果预算由 `MemorySystem`/宿主上下文持有,不由模型参数自授权。

### 21.2 默认 schema 精简

`retrieve_for_turn` schema 当前暴露：

```text
query
keywords
categories
subject_scopes
time_hint
include_explicit
kind_patterns
```

`read_timeline` schema 当前暴露：

```text
date_from
date_to
time_periods
cross_conversation
```

`retrieve_for_turn` 的 `source_layers`、`importance_min`、`include_explicit` 和
`kind_patterns` 都是可选的显式过滤参数；`read_timeline` 只接受日期、时间段和
`cross_conversation`。schema 由同一份实现生成,避免 prompt 手抄另一套参数列表。

`load_material` 继续是可选宿主工具：MemCore 可以返回 material anchor/file id，但文件本体读取和权限仍由宿主执行。宿主可在 `build_native_memory_tool_specs(include_material_tool=False)` 时不注册它；若注册但未注入 loader,dispatcher 返回结构化 `unavailable`,不假装已读到文件。

### 21.3 清晰的工具说明

模型说明必须明确区分：

- 想找“与当前问题相关的过去事实、约定、偏好” -> `retrieve_for_turn`；
- 想按日期回看“某天/某段时间发生了什么” -> `read_timeline`；
- 普通问题已有当前上下文答案 -> 不调用记忆工具；
- 需要工具调用细节时才设置 `include_explicit=true`，并提供具体 `kind_patterns`；
- empty 表示合法查询没有结果，不应反复换同义词无限重试；
- unavailable/failed 表示系统能力问题，应自然说明暂时无法读取，不能假装没发生过。

工具说明与真实 schema 从同一 spec 生成，prompt playbook 不再手抄另一套参数列表。

### 21.4 Dispatcher 结构化失败

dispatcher 边界捕获参数错误、权限拒绝、Store/Index 不可用和未知异常，并统一返回：

```json
{
  "status": "found | empty | invalid | forbidden | unavailable | failed",
  "reason": "stable_reason_code",
  "effective_filters": {},
  "matches": [],
  "truncated": false
}
```

未知异常写安全日志后返回 `failed/internal_error`，不让 Promise/Future exception 破坏主回复，也不能吞掉异常后伪装成 `empty`。结果正文按 policy token budget 裁剪，裁剪信息显式进入 `truncated` 和 `omitted_match_count`。

### 21.5 模型可用性验收

单元测试验证 schema 只是最低要求；还需要使用真实 provider tool calling 做固定决策集：

- 当前上下文足够时不误调；
- 询问既往偏好时主动用 retrieve；
- 指定昨天/上周时选择 timeline；
- 想看某次搜索过程时使用 explicit + 精确 kind；
- empty 后正常回答，不循环；
- unavailable 后不编造记忆；
- 多工具同轮时能与其他宿主工具并行，并保留各自 call id。

验收记录模型输入、tool call、结构化 tool result 和最终回复，但清除密钥、宿主内部路径字段和敏感正文；模型完成任务所需的操作路径证据保留原样。

## 22. Akane 文件级回填结果与旧权威删除

本节保留文件级职责和当时的切换顺序，便于审计为什么这些逻辑归属于 MemCore
或宿主。当前回填已经完成；下文描述的是现有边界和历史迁移记录，不是要求接入方
再次复制一遍 Akane 的迁移过程。

### 22.1 `companion_v01/memcore_integration/manager.py`

`MemcoreManager` 仍是 Akane 到包的薄适配层，但改为：

- 从进程级 provider/container 获取同一个 `MemCoreRuntime` 并注入每个 `MemorySystem`；
- 暴露 `begin_turn/append_entry/record_tool_exchange/record_request_projection/complete_turn`；
- 把 Akane identity/config 转成 Namespace、Actor、ToolDispatchPolicy；
- 保留 legacy import 和结构化状态转换；
- 删除 `_render_prompt_context_layers()`、`_project_timeline_result()` 等与包重复的私有渲染；
- index warmup/repair 走 MemCore runtime，不再持有独立 `_index_warmup_executor`；
- `record_user_turn/record_external_event/...` 只做 typed standalone API 的薄适配，不维护第二套行为。

一个 Akane 进程无论挂载多少 bot，manager 只携带实例配置和 Namespace，不复制一套能力逻辑。bot 间记忆隔离由 Namespace 数据决定，不能靠不同 manager 分叉代码。

### 22.2 `companion_v01/engine.py`

普通消息、群聊事件、金融主动事件都使用同一轮次生命周期：

1. 渠道层确定 `kind/payload/actor`；
2. 输入落库并 `begin_turn()`，得到 `turn_id` 和 annotation target；
3. 每轮 provider 请求前记录实际安全 projection；
4. `_execute_and_record_tool_batch()` 给每个并行 call 传相同 turn id、独立 correlation id；
5. result 无论成功、失败或取消都闭合对应 correlation；
6. 终态 JSON 解析后调用一次 `complete_turn()`；
7. complete 成功后才安排 compaction。

当前主链不再分别调用 `_record_memcore_input_turn()`、若干 tool trace 写入、`_update_memcore_turn_metadata()` 和 `_record_memcore_assistant_turn()` 来拼出一个“看起来完整”的轮次。遗留名字只允许是 typed turn API 的薄 adapter，不得恢复为第二条写路径。

工具执行、QQ 发送、图片理解、GPT-SoVITS 和金融业务判断仍留在 Akane；移入 MemCore 的只是通用 action/observation 的结构化记录和 provider projection。

### 22.3 原生多工具 history 移入 projection adapter

当前结果：

- Akane 工具编排器继续产生 provider-neutral `ToolExchangeBatch`；
- MemCore `ProjectionAdapter` 把 batch 投影成 OpenAI/Anthropic 的原生 assistant calls + tool results；
- Akane 只把 adapter 输出追加给 provider，并在发送前回传 ledger；
- 下一轮优先读取相同 profile 的已保存 projection，恢复完全相同的 tool call id、参数顺序和 result 结构；
- provider-specific `_append_native_*` 已退出权威实现；宿主只保留调用 MemCore adapter 和附加当轮不可持久化媒体的薄步骤。

当前 Akane repair pass 已把 provider-specific 拼装收敛为一个薄的 projection 选择/当轮媒体附加步骤：工具
call/result 的 role、call id、参数和顺序由 MemCore `ProjectionAdapter` 产生；Akane 只按 source id 取回
当前新增 projection，并把不可持久化图片附到对应 `material.model_input`。不得重新长出第二套 OpenAI/Anthropic renderer。

### 22.4 终态 raw output 接口

`companion_v01/llm_runtime.py` 的流式路径已有 `ChatJSONStreamResult.raw_text`，直接传给 `complete_turn(provider_output_raw=...)`。

同步链路已经提供 `call_chat_json_result()`，返回 parsed 与真实 raw output：

```text
ChatJSONResult
├── parsed
├── raw_text
├── provider_profile/model_route
└── usage/cache fields
```

旧 `call_chat_json()` 返回 `.parsed` 保持兼容，MemCore 主链使用 `call_chat_json_result()`。不能从 parsed dict 再 `json.dumps()` 冒充模型原始输出，否则字段顺序、空白和 provider 历史都可能变化，破坏下一轮前缀。

### 22.5 `response_builder.py` 与 prompt envelope 清理结果

`companion_v01/engine_services/response_builder.py` 调用 MemCore
`build_context_projection(provider_profile=...)`，然后再由 Akane 添加人格、当前动态
上下文和本轮工具 schema。以下逻辑已经退出运行权威并删除：

- `_attach_message_prompt_envelopes()`；
- `_sync_current_message_prompt_envelope()`；
- Akane 自己按 source id 拼历史 envelope 的路径。

`companion_v01/store/core.py` 只保留 `chat_messages.prompt_envelope_text` 数据库兼容列，
不再接受运行时新写入，也没有 writer/reader/pruner 权威。数据库列可留到后续独立
schema cleanup，不需要为了表面整洁立即做破坏性 DROP。运行时只有 MemCore
projection 一个历史权威，不能恢复双写后再靠“优先取非空”决定。

### 22.6 Akane 历史切换顺序

回填当时严格按以下顺序执行，每一步都可独立回滚到上一步：

1. 升级 MemCore schema/runtime/types，但 Akane 仍走旧主链；
2. 只 shadow 记录新 turn/projection，比较 source count、关系和 hash，不给 prompt 使用；
3. 切换普通私聊/群聊的历史读取为 MemCore projection；
4. 切换并行工具原生 projection；
5. 切换外部事件/金融主动推送；
6. 切换 compaction 与 retrieval V2；
7. 删除 Akane prompt envelope 和重复渲染/检索权威；
8. 再扩大到其他 bot 实例，配置差异只留在 profile/plugin/namespace。

shadow 阶段没有双发消息、双执行工具或把 shadow summary 注入模型。迁移过程中每步失败均要求返回结构化状态并保持旧可用链，不写 fake success。

## 23. 已完成的文件改动矩阵

### 23.1 MemCore package

| 文件 | 改动 | 完成条件 |
|---|---|---|
| `memcore/timeline.py` | 新增 entry/turn/projection/visibility 类型 | 类型不依赖 Akane 业务枚举 |
| `memcore/runtime.py` | 共享 executor、lock registry、ownership | 多 MemorySystem 同库同会话不会并发重复压缩 |
| `memcore/store/migrations.py` | `PRAGMA user_version` runner、幂等 migration | 旧 DB 可重复启动且数据不丢 |
| `memcore/store/base.py` | 新 Turn、relation、projection、atomic commit API | 所有新关系读取显式 Namespace |
| `memcore/store/sqlite_store.py` | V2 columns/tables/index/transactions | stale snapshot 不产生重复 summary |
| `memcore/memory_system.py` | 新生命周期、runtime 注入、结构化 retrieval | 旧 API 只剩薄 adapter |
| `memcore/projection.py` | canonical/OpenAI/Anthropic adapter 与 ledger | 同一 entry/profile 字节稳定 |
| `memcore/rendering.py` | registry、prefix resolution、canonical fallback | 未知 kind 安全可渲染 |
| `memcore/compaction.py` | terminal-turn/token/lineage 原子压缩 | 不切断并行工具 branch，不让 aborted 永久阻塞 |
| `memcore/retrieval.py` | QueryPlan、hard filters、关系扩窗、结构化结果 | 被过滤记录从未进入评分 |
| `memcore/index/metadata_filters.py` | kind flags 与 filter compiler | 新 kind prefix 可前置过滤 |
| `memcore/index/entry_builder.py` | V2 flags/visibility/generation metadata | Chroma metadata 全为支持的标量 |
| `memcore/index/chroma_index.py` | generation/schema-aware collection | schema 变化可安全重建切换 |
| `memcore/native_tools.py` | 精简 schema、policy、结构化 dispatcher | 模型不能扩大权限 |
| `memcore/chat_output/*` | raw/semantic 双表示、metadata status | 缺 metadata 不再算 accepted |
| `memcore/config.py` | token/visibility/runtime/index version 配置 | 旧 count 配置有明确兼容映射 |

### 23.2 Akane 回填

| 文件 | 改动 | 最终应删除/变薄的旧权威 |
|---|---|---|
| `companion_v01/memcore_integration/manager.py` | Turn/Projection/Policy adapter | 私有渲染、独立 executor、旧分段写轮次 |
| `companion_v01/engine.py` | 一个 TurnSession 贯穿输入、并行工具和 final | 分散的 metadata/assistant 拼接逻辑 |
| `companion_v01/llm_runtime.py` | 同步 raw-result API | parsed 后重序列化路径 |
| `companion_v01/engine_services/response_builder.py` | 使用 MemCore projection | attach/sync prompt envelope |
| `companion_v01/store/core.py` | legacy envelope 一次导入及停写 | envelope writer/reader/pruner |

金融插件、个人 bot、QQ 群聊和桌宠不各写一套 MemCore integration；它们只提交不同 kind/payload、加载不同插件/profile，并由同一生命周期处理。

## 24. 测试、真实链路验收与实施记录

### 24.1 主要测试文件

当前主要覆盖文件：

```text
tests/test_timeline_v2_migration.py
tests/test_timeline_v2_turns.py
tests/test_projection_cache.py
tests/test_retrieval_visibility.py
tests/test_relation_expansion.py
tests/test_compaction_v2.py
tests/test_slice_native_tools.py
tests/test_slice_concurrency.py
```

现有 slice tests 继续保留用于兼容回归；新语义不继续塞进一个越来越大的 slice 文件。

### 24.2 必须覆盖的自动化用例

存储与迁移：

- 从当前真实 schema fixture 升级，数据、source id、summary lineage 不丢；
- migration 中断后重启幂等；
- Namespace 不匹配时 turn/source/correlation 全部不可读取；
- 未知 legacy role 安全落为 `legacy.*`，不伪造关系；
- 同一 summary snapshot 重试只产生一个 id。

轮次与并行工具：

- annotation 精确写回宿主指定 target；
- 中间工具 JSON 不能覆盖终态 annotation；
- 两个以上并行 call 乱序返回仍按 correlation 成对；
- success/error/cancelled 都能闭合 branch；
- pending branch 阻止 final/compaction；
- event 与 message 在有效 annotation 下得到相同检索准入。

投影与缓存：

- unknown kind canonical 渲染稳定；
- key 顺序、当前时间、debug flag、工具可用性变化不改旧 projection；
- OpenAI/Anthropic 原生多工具 projection 可 round-trip；
- 同 profile 连续请求满足严格增长前缀；
- provider 切换形成新 cache family；
- media omission 形成可解释断点且不泄漏 base64/path；
- 工具新加载的图片位于 tool result 之后，不能改写已经 `request_frozen` 的原始 user projection；
- 同一 turn 的生成重试保持 current user payload 字节一致；
- final 保存真实 raw output，不是 parsed 后重新序列化。

检索与扩窗：

- Namespace/kind/visibility/time/visible ids 在 cosine/BM25 前过滤；
- semantic relaxation 永不放宽 hard filters；
- explicit tool 查询只打开授权 prefix，不带入其他 trace；
- 混合 trace 的 episode summary 仍可按自身语义检索；
- summary/semantic lineage closure 不重复返回 raw；
- 普通命中返回 stimulus/final，默认不夹工具；
- 显式工具命中返回完整 call/result branch；
- token 裁剪不拆原子组；
- 零相似度、empty、index unavailable 得到不同状态。

Runtime 与故障：

- 同 DB/Namespace/conversation 的两个 MemorySystem 只执行一次压缩 commit；
- 不同 conversation 可并行；
- SQL commit 后 index 失败留下 pending 并能 repair；
- shutdown 不关闭外部注入 runtime；
- dispatcher 任意异常结构化返回，不影响主回复。

### 24.3 真实 provider 缓存验收

自动 hash 测试不能替代真实 API。每个 provider/profile 使用独立测试会话，保存安全 usage audit，并至少覆盖：

1. 连续普通对话三轮；
2. 普通对话与结构化 event 交织；
3. 一轮单工具；
4. 一轮两个以上并行工具且乱序返回；
5. 工具轮后继续普通对话；
6. 一次明确 compaction 前后；
7. 媒体轮前后。

验收同时比较：

```text
上一轮实际发送 conversation + 上一轮实际 raw final
是否为下一轮 conversation 的字节级前缀

prompt_cache_hit_tokens
prompt_cache_miss_tokens
命中率
projection/full-prefix hash 变化原因
```

普通对话、事件和工具交织时只要没有压缩/provider 切换/媒体例外，旧前缀必须保持；新增尾部产生 miss 是正常增长，不等于旧缓存失效。缓存服务“尽力而为”不能成为结构不稳定的借口，先用 hash 证明请求正确，再看 provider usage。

Akane 最终验收必须分别走个人 bot 普通私聊、个人群聊、金融私聊主动推送和金融群聊对话真实入口；不能用本地直接调用 MemCore 的高命中替代真实链路。

### 24.4 已完成的实施切片

各切片均按一个可验证边界推进，并在通过后形成聚焦 commit：

1. **Schema foundation（已完成）**：migration runner、V2 columns、turn/projection tables、namespace-safe reads；
2. **Turn lifecycle（已完成）**：begin/append/complete/abort、annotation status、并行 correlation 与原子终态，尚不切 Akane；
3. **Projection ledger（已完成）**：renderer registry、provider adapters、immutable ledger、request audit、strict-prefix tests；
4. **Compaction V2（已提交：`41d55b4`）**：shared runtime、terminal-turn/token planning、atomic summary/semantic commits；
5. **Retrieval admission（已提交：`df09a50`）**：visibility、kind flags、新 index generation、hard-filter tests；
6. **Relation expansion（已提交：`3523c26`）**：turn/correlation/lineage closure、structured results、atomic token budget；
7. **Akane cutover primitives（已完成）**：typed standalone entry、staged annotation；
8. **Native tools V2（已完成）**：policy、精简 schema、dispatcher 与模型决策验收；
9. **Akane V2 write cutover（已完成）**：普通输入、event、intermediate、并行工具和 final 切到同一 turn；
10. **Akane read cutover（已完成）**：普通对话、tools、event/finance 使用同一权威，旧读取权威已删除或收薄；
11. **真实 provider acceptance（已完成当前部署验收）**：缓存、失败降级和多 bot 同能力已走真实入口；provider 缓存仍按服务商规则持续观测；
12. **Cleanup（已完成）**：删除 prompt envelope writer/private renderer/旧 dual-write，并更新公开文档。

每个切片必须执行相关单测、全量测试、lint/format/build 和 `git diff --check`。Akane 回填切片还要说明用户实际会感觉到的变化，并验证 QQ 文本、图片、TTS/表情等表现层没有因主回复或工具错误被连带破坏。

### 24.5 本设计明确不做的事

- 不新增第二套长期 raw timeline 表；
- 不让 MemCore 执行 Akane 工具、发送 QQ、读取文件本体或决定 bot 权限；
- 不把 system/persona/tool schema 全文复制进记忆库；
- 不按 Bot、金融/个人或每种 event kind 分叉实现；
- 不用 provider role 代替 origin/turn role；
- 不让 include_explicit/category 放宽关闭全部前置过滤；
- 不为了缓存保存密钥、宿主内部路径、base64 或伪造 provider 历史；
- 不长期并行维护 Akane prompt envelope 与 MemCore projection 两个权威；
- 不在本设计文档完成前直接开始大范围功能改造。
