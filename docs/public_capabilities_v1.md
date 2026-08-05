# memcore 对外能力总览

`memcore` 不是一个把聊天记录塞进向量库的简单记忆插件。它是一个
provider-neutral 的记忆与上下文基础设施：把普通消息、模型回复、外部事件、
工具调用/结果、Skill 结果和材料状态放进同一条可追溯时间线，再分别决定它们
如何展示、检索、压缩和进入长期记忆。

它不替宿主接 QQ、桌宠或 HTTP，也不替宿主执行工具。它负责把宿主已经确认
发生的事实，可靠地保存成模型下一轮能理解、能检索、能稳定复用的上下文。

## 一张能力地图

```text
宿主输入 / 模型输出 / 事件 / 工具 / 材料
                    │
                    ▼
       Unified Timeline + Namespace 隔离
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  稳定投影       可见上下文      结构化检索工具
  provider       raw/summary/    retrieve + timeline
  messages       semantic        + material loader
       │            │            │
       └────────────┼────────────┘
                    ▼
       token 差值压缩、lineage、outbox 自愈
```

## 已实现能力

### 1. 统一时间线，而不是“聊天记忆”和“工具记忆”两套系统

Timeline V2 支持：

- `message.*`、`event.*`、`tool.*`、`material.*`、`skill.*` 等开放的 namespaced
  `kind`；新增业务类型不需要修改 MemCore 枚举；
- `begin_turn → append_entry → complete_turn / abort_turn` 的完整轮次生命周期；
- 多个并行 action/observation 通过 `correlation_id` 关联，结果乱序也不会串线；
- 外部事件既可以作为独立记录，也可以作为一个模型轮次的 stimulus；
- `actor` 只表达群聊归因，`tenant/user/domain` 才是硬隔离边界。

`record_user_turn`、`record_assistant_turn`、独立事件和材料便捷 API 只会生成
V2 typed standalone entry，不再直写 flat V1 路径；有模型回复的请求必须使用
完整 turn API。`record_tool_exchange` 需要开放的 `turn_id`。

### 2. 稳定的 provider-visible 投影

MemCore 不把内部数据库行直接拼进 prompt，而是通过 renderer registry 和
projection ledger 生成确定性的 provider messages：

- canonical、OpenAI、Anthropic 三种投影 profile；
- 工具调用、工具结果、事件、材料状态的中性结构化渲染；
- renderer/version、entry projection hash、compaction generation 和真实请求
  audit；
- 旧记录的投影默认不会因为 renderer 升级而被静默重写；
- 新内容只追加在尾部，模型请求的历史前缀可以做严格 prefix 验收。

这保证的是“传给 provider 的前缀可证明稳定”，不是替任何 provider 承诺
缓存一定命中。最终 hit/miss 仍由 provider 的缓存策略决定。

### 3. 三层记忆生命周期

写侧采用：

```text
raw working memory → episodic summaries → semantic long-term facts
```

- raw 保留近期原始细节和完整 source lineage；
- episodic 摘要压缩一段完整内容；
- semantic 只有在共享明确实体、兼容 facet 且共享命题/主题时才强化合并，常见人物重叠不会把无关主题焊在一起；
- 每次后台 `run_due()` 只推进一个 raw batch 和一个 semantic batch，下一次调度
  继续，避免高活跃 namespace 在一次维护中连续调用模型清仓；
- LLM 失败、索引失败或并发冲突都返回结构化状态，不提交空摘要、不伪造成功。

V2 使用实际 provider projection 的 token 预算规划完整 turn。这是唯一 raw
压缩策略，不存在 count/flat 备选路径。宿主可注入真实 tokenizer；没有 counter
时仍可压缩，但结果明确标为 `token_count_quality=estimated`。

### 4. 互补的记忆读取与证据导航工具

- `retrieve_for_turn`：面向偏好、计划、关系、人物、主题和长期事实的模糊检索；支持用本地/ISO `time_hint.start_at/end_at` 在评分前硬过滤，也可用 `within_memory_id` 把 dense/BM25 候选硬限制在已选节点及其精确后代中，允许只凭已知事件、关系和时间发现未知答案；段内为空不会放宽到全库；
- `browse_memory`：面向多日/宽范围概览，按 SQLite 时间重叠返回小型 episodic/semantic 卡片，而不是把整段群聊 raw 塞回模型；结果同时报告已摘要、未摘要 live tail、broken lineage 和无损 cursor；
- `open_memory`：统一打开 raw/episodic/semantic ID；`card` 看导航信息，`content` 看完整节点正文，`sources` 沿精确 lineage 返回子摘要卡或完整 raw 逻辑单元；
- `read_timeline`：既可按 `start_at/end_at` 精确到小时/分钟读取，也可按旧日期/粗时段读取，或把 raw 检索命中扩成前后完整 turn；`conversation/full/tools` 决定证据密度。模型侧 native dispatch 使用 `native_timeline_page_token_budget` 的有限上限，返回完整逻辑单元、所选/本页 token 与条目总量、`partial/page_boundary`、导航建议和 namespace-safe cursor；无真实 tokenizer 时使用并明确标记估算，不会让工具失效。native result 只保留一份渲染正文，避免和结构化 messages 重复耗费 provider token；直接 Python API 保留双视图与显式无限诊断路径；
- `dispatch_native_memory_tool` 同时返回完整当前轮 result 与紧凑 `receipt`；receipt 只保留选择器、返回 ID、coverage、cursor 和经过清理的稳定 hash，供宿主跨轮写入 operation observation，避免把大段 raw/摘要/检索正文复制进时间线；非原生适配可直接调用 `build_memory_operation_receipt`；
- `read_entry`：兼容期 raw-only 薄适配，新接入使用 `open_memory(view="content")`；
- `load_material`：只负责调用宿主提供的材料 loader，不保存文件本体；
- 普通检索只接纳 `retrieval_visibility=default` 的记录；没有有效 annotation 的
  standalone 事件、operation 和 material 轨迹默认是 `explicit`，不会混入普通候选；
- 这不是“事件永不检索”：V2 `event.*` 如果作为轮次 stimulus 并获得有效
  `accepted_model/accepted_host` annotation,默认准入规则与普通消息相同；
  关联的模型 final 会通过 stimulus/final relation 一起返回；
- standalone `record_external_event()` 或没有有效 annotation 的事件保持 explicit,
  需要 `include_explicit=true`、明确 `kind_patterns` 和宿主授权才能检索；
- 模型 final 不是一律独立成长期记忆 seed：V2 默认按它所属轮次和 stimulus
  成组返回；独立 `record_assistant_turn()` 写入的 standalone assistant raw 则按其显式
  retrieval policy 处理；
- 需要工具/事件/材料轨迹时，模型必须设置 `include_explicit=true`、提供明确的
  kind pattern，并接受宿主
  `ToolDispatchPolicy` 的授权；
- namespace、conversation、visibility、kind 和 source lineage 等硬过滤在
  embedding/BM25 评分前完成；facet/role 同样前置。准确实体导致候选为零时只移除
  实体强制条件，query 与实体权重仍保留，并返回候选数和放宽 diagnostics；
- raw 与 summary/semantic 分池检索，raw 先占用现有结果位；derived 只补充，不能挤掉原始证据，也不会自动沿 lineage 下钻；
- relation-aware expansion 会返回完整 stimulus/final 或 call/result 关系，
  不用物理相邻消息猜窗口，也不会把中间工具轨迹偷偷混入普通命中。

### 5. 工具、事件和材料的通用接缝

- `append_action()` / `append_observation()` 是开放的动作—结果薄接口，宿主可用
  原生 tool calling、JSON、XML、标签或其他协议；MemCore 不解析协议、不执行工具；
- 非原生协议可以通过 `record_request_projection()` 冻结宿主真正发送的 provider
  messages，不会被默认 native-tool fallback 改写；
- operation entry 可逐条附带小型 `retention_anchor`，在 raw 压缩后保留资源 ID、
  版本、schema hash 或结果引用；无 anchor 的旧行为不变，完整结果仍留在宿主存储；
- `record_tool_exchange(turn_id=...)` 生成关联到同一 turn 的 action/observation；
- `record_external_event()` 生成 `event.*` 中性结构化块；
- `record_material_reference()` / `record_material_cleanup()` 只保存 file_id、
  文件类型和状态，原图、OCR、视觉描述和文档 chunks 留在宿主存储；
- renderer registry 可为特定 kind 增加可读性 renderer，但不会借渲染器越权
  改变检索准入、工具权限或信任边界；
- 未知 namespaced kind 有安全的 canonical fallback，不会伪造 `system` 或
  provider tool 边界。

这不规定宿主必须动态加载工具。工具少且稳定时，把详细 schema 固定放在系统
提示词仍是合理选择；动作—结果 Timeline 只是提供可选的统一记录与投影机制。

### 6. Chat Output Adapter

这不是未来草案，而是已实现的可选接入层：

- `memcore_json` 最终回复契约；
- `speech` 提取、流式 speech 解析和中文/英文标点分段；
- `memory_metadata` 校验、状态区分和回写当前 annotation target；
- 写入、摘要、长期记忆、索引与检索共用 `turn_intent / memory_facets /
  about_roles / entity_anchors / topic_terms / retrieval_priority / mood_tags`
  唯一契约；工具/事件/材料身份由 typed kind 表达；
- 工具中间轮不套最终回复 JSON，只有没有待处理工具调用的终态回复进入
  final contract；
- 缺字段、非法 JSON、fallback 和 rejected 都有显式状态，不把坏 JSON 当成
  用户可见 speech 或有效记忆。

### 7. 存储可靠性与可维护性

- SQLite 是事实源，向量索引只是检索加速层；
- source_id 幂等、namespace owner 校验、定向遗忘和事务提交；
- index outbox 在向量后端故障时保留 pending，恢复后可 `reindex_pending()`；
- `reindex_all()` 可从 SQLite 真相源重建当前 hard namespace 的 raw、summary、
  semantic 索引；
- `MemCoreRuntime` 提供共享 namespace 锁和受控后台维护，避免同一进程多个
  Bot/MemorySystem 重复压缩同一 namespace。

## 为什么它不是普通向量记忆

| 常见做法 | memcore 的区别 |
|---|---|
| 聊天历史、工具轨迹各存一份 | 一条统一时间线，检索准入和长期语义再独立决策 |
| 直接把数据库行拼进 prompt | provider-specific projection + immutable ledger + prefix audit |
| 先向量搜索再过滤 | namespace/kind/visibility/lineage/facet/role/entity 在评分前裁剪候选 |
| 摘要和 raw 混成一个榜 | raw-first 分池，摘要和长期记忆只补充 |
| 用相邻物理行扩窗 | raw source_id 按完整 turn 分组扩窗，工具轮不会截半 |
| 压缩失败仍标记完成 | 结构化 retry/deferred，原始 source 保留，不假成功 |
| 工具类型写死在核心枚举 | 开放 namespaced kind + 可插拔 renderer |

## 宿主必须提供的部分

MemCore 不包含也不推断这些产品资产：

- 最终聊天模型、API key、provider 路由和模型参数；
- QQ/桌宠/HTTP 渠道、权限和工具实际执行；
- persona 与领域补充提示词；固定 metadata facet/role 契约不能由宿主换成另一套同义词；
- 文件本体、OCR、视觉模型和 derived store；
- 生产环境 embedding 模型选择。

通过 `LLMClient`、`EmbeddingProvider`、`MemoryStore`、`VectorIndex`、
`RendererRegistry`、`TokenCounter` 和 `material_loader` 注入这些边界，
MemCore 不需要为每个 Bot 复制一套业务逻辑。

## 仍然明确存在的兼容边界

- SQLite schema 3 会把旧 metadata 一次转换并将三层索引全部标记 pending；运行时
  不再双读或双写旧 `keywords/categories/subject_scopes`；
- 更早的数据可能没有完整 turn/correlation lineage；它们作为 standalone component
  进入唯一 V2 planner，不靠物理相邻位置猜造关系；
- 产品级 prompt assembly 和 provider transport 按架构边界继续由宿主负责，
  这不是 V2 未完成或需要复制的第二套记忆实现；
- `HashedEmbeddingProvider` 只用于测试，不代表生产语义检索质量；
- MemCore 保证稳定前缀和可诊断审计，不保证 PinAI、DeepSeek 或其他 provider
  的缓存服务必然命中；
- 历史 standalone 数据通过有界维护逐步压缩，不会被一次性清仓或静默丢弃。

## 最小入口

正常接入先读：

1. [`README.md`](../README.md)；
2. [`usage_flow_v1.md`](usage_flow_v1.md)；
3. [`model_prompt_playbook_v1.md`](model_prompt_playbook_v1.md)；
4. [`memory_metadata_raw_retrieval_design_v1.md`](memory_metadata_raw_retrieval_design_v1.md)。

测试、构建和授权信息以仓库根目录 README 为准。
