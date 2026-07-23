# Design Highlights V1

这篇文档用于向接入方、客户或协作者说明 memcore 的设计价值。它不是 API 手册,而是解释这套记忆系统为什么这样做。

## 一句话

memcore 是一个领域无关的分层记忆内核:把原始对话、阶段摘要、长期语义记忆分层沉淀,用时间锚点和结构化 metadata 让聊天模型可见、可查、可控地使用长期记忆。

## 核心亮点

### 1. 三层记忆,不是简单聊天记录

memcore 把记忆分成:

- raw: 精确原始对话,保留近期上下文和可追溯性。
- episodic summary: 阶段摘要,压缩一段对话里的事件、事实、时间范围。
- semantic memory: 长期语义记忆,保留稳定事实、反复主题、重要人物、未完成线索。

价值:近期细节不丢,长期画像不靠无限堆 raw,上下文窗口压力可控。

### 2. Token 差值窗口 + 完整关系边界

raw 达到 `raw_token_trigger` 后按 `raw_token_batch_ratio` 规划旧前缀，实际切点只能落在完整 terminal turn/relation component；episodic → semantic 仍保持批次小于触发线。

价值:压缩后仍保留近期完整 raw，用户输入、并行工具和最终回复不会被截成半轮；同时摘要层不会和 raw 层长期重复。

### 3. 时间锚点和星期感知

raw、summary、semantic、timeline 都带真实时间锚点,包括星期几。

价值:模型处理“昨天/上周二/最近/下周三”时不再靠猜。总结入库时也能把相对时间转成绝对日期或日期范围。

### 4. 双工具:模糊检索 + 精确时间线

memcore 提供两类读工具:

- `retrieve`: 向量/关键词混合检索,适合偏好、计划、长期事实、人物关系。
- `read_timeline`: 按日期/时间段精确读 raw；也可用 raw source_id 扩展前后完整 turn。

价值:不用把所有记忆问题都塞进向量检索。精确时间问题走精确工具,长期语义问题走检索工具。

### 5. 不内置 router,让聊天模型自驱检索

memcore 不每轮额外调用一个 router LLM 判断要不要搜。

价值:减少成本和延迟,避免“不是当事模型”的误判。聊天模型自己知道当前对话意图,由它决定是否调用 `retrieve/read_timeline`。

### 6. metadata 前置过滤

`memory_facets / about_roles / entity_anchors / source_layers / time_hint` 会下推到 index 的 `where`。

价值:先裁候选,再算向量/BM25。模型传入 `memory_facets=["preference"]`、`about_roles=["user"]` 时,系统只和满足条件的记忆计算相似度,不是全量算完再后置过滤。

### 7. 只放宽零候选实体,不突破调用方边界

检索先严格使用模型传入的过滤参数。准确实体条件令某个 raw/derived 池候选为零时，
只在该池移除实体强制 flag；原始 query 和实体高权重仍参与排序，并返回 strict/effective
候选数。facet、role、`source_layers` 与 Namespace、conversation、time、visibility、
annotation、kind、lineage 和 index generation 都不静默放宽。

价值:旧记录缺少准确实体标签时仍有召回机会，但模型猜错 facet/主体不会被系统
悄悄改成宽搜，调用方能从 diagnostics 看见实际发生了什么。

### 8. Raw-first 分池，而不是三层共榜

raw 先按相关度填充现有 `max_matches`，还有位置时才加入 summary/semantic。相同
lineage 同时命中时保留 raw；derived 内容够用就直接回答，不足时也不会自动下钻原文。

价值:摘要文本更长、更概括也不能挤掉可核对、可扩窗的原始对话。

### 9. 向量数据库可选

默认 `InMemoryVectorIndex` 支持语义 cosine、BM25、metadata where、RRF。安装 `memcore[speed]` 后可用 NumPy float32 加速和降内存。

价值:中小规模无需强制部署向量数据库;需要 Chroma 等外部后端时再替换 `VectorIndex`。

### 10. SQLite 是真相源,index 是可重建加速层

所有原文、时间、metadata、摘要都在 SQLite 真相源。向量 index 只做检索加速。

价值:向量后端故障不会导致记忆丢失。`reindex_pending()` 修复 pending outbox;`reindex_all()` 可从 SQLite 补建/热加载空内存索引。

### 11. Chat Output Adapter 让聊天模型本人给 metadata

可选 `memcore_json` 契约让最终聊天模型输出:

```json
{"speech":"给用户看的回复","memory_metadata":{}}
```

价值:不需要每轮再调一个轻量模型给 raw 打标签。metadata 来自真正理解当前对话的聊天模型本人,并且 `speech` 可以流式提前解析。

### 12. 群聊 actor 软标签

硬隔离仍是 `tenant/user/domain`;`actor` 用来记录同一会话/群聊里“谁说的”。

价值:群聊里能保留发言人归因,但不会把每个发言人误做成硬隔离用户。需要硬隔离时,接入方应映射到不同 `user_id`。

### 13. 提示词治理:焊死骨架 + 可控插槽

压缩提示词里的字段契约、JSON 格式、时间锚点、多方归因、metadata 标注规则不可被外部覆盖。接入方只能通过 `PromptOverrides` 追加 persona 或领域补充。

价值:领域可扩展,但不会把记忆系统的承重规则写坏。

### 14. 不假成功

压缩 LLM 失败不会提交空摘要;向量 upsert 失败保留 pending;非法时间线过滤返回结构化错误;未实现或坏配置在边界处报错。

价值:宁可显式失败或降级,不把坏数据悄悄写进长期记忆。

### 15. 稳定 provider 投影,而不是每轮重渲染整段历史

Timeline entry 通过 renderer registry 和 projection ledger 生成 canonical、OpenAI
或 Anthropic provider message。每条旧投影有 renderer/version 和 projection hash,
请求还有实际 history 的 audit。

价值:普通消息、主动事件、工具结果交织时,旧前缀仍可做严格 prefix 验收;压缩或
显式迁移之外不会偷偷改写历史。MemCore 保证前缀稳定和可诊断,不虚假承诺 provider
缓存一定命中。

### 16. Raw 锚点按完整 turn 扩窗,不是物理相邻消息窗口

V2 使用 raw source_id 找到它所属的完整 turn，并按前后 turn 数读取窗口。空 turn_id
的 standalone entry 自成一组；summary/semantic 不自动沿 lineage 下钻 raw。

价值:并行工具乱序、事件回复和压缩后的 derived memory 不会因为 `seq_no ± N`
猜错关系,也不会把已经在 prompt 可见的中间轨迹重复塞回模型。

### 17. 开放 kind,稳定边界

`event.finance`、`event.qq.poke`、`tool.web_search.result`、`skill.loaded` 都是
合法的 namespaced kind。renderer 可以扩展可读性,但不会因为新增业务类型就复制
一套 MemCore 逻辑,也不会借渲染器绕过 namespace、visibility 或工具授权。

通用 `event.*` 与 `material.*` 会把结构化字段按稳定顺序渲染一次；若
`semantic_text` 已经只是同一 payload 的 `key: value` 可读投影，不会再附一份
重复 `data` JSON。无法由 payload 表达的自由文本仍以 `content` 保留，不以变短为由丢失语义。

### 18. 后台维护有界,但不把差值切成连续小压缩

一次 `compact_due` 只提交一个 raw compaction generation 和一个 semantic batch。
projected-token 模式会在这一个 generation 内按配置比例选足最旧的完整
turn/component；不会复用 count 批次或独立 source 上限，把同一个差值拆成聊天
过程中连续发生的小压缩。失败返回结构化 retry/deferred，不提交半批 lineage。

价值:高活跃群不会因为旧条目限制连续改写 prompt 前缀，同时模型调用、延迟和
数据提交边界仍然可观测、可回滚。

### 19. 动作—结果开放接缝，不接管宿主工具架构

`append_action` / `append_observation` 只固定轮次角色和 `correlation_id` 关系，
不固定工具数量、目录加载方式、模型输出语法或业务 payload。原生 tool calling、
JSON、XML、标签都能进入同一 Timeline，真实 provider 表示可写入 projection ledger。

operation 压缩默认仍是轻量 digest；只有宿主显式提供的小型 retention anchor 才保留
资源 ID、版本、schema hash 或结果引用。这样既不会无限复制工具结果，也不会要求
所有宿主实现一套“能力加载协议”。

## 适合什么场景

- 长期陪伴型 AI:用户偏好、关系、计划、相处时间、情绪余温。
- 金融/投顾/客服助手:风险偏好、投资目标、约束条件、历史决策和回访记录。
- 群聊/团队助手:谁提出需求、谁承诺行动、事项何时发生、后续有没有完成。
- 需要私有化部署的产品:SQLite + 内存索引即可起步,后续可换向量库。

## 边界

- memcore 不提供具体人格、金融建议规则或合规判断;这些由接入方 prompt / policy / 工具链负责。
- memcore 不替宿主做最终 prompt token 预算;它提供可见三层、检索工具和压缩策略。
- memcore 不保证模型一定会聪明使用工具;接入方应参考 `docs/model_prompt_playbook_v1.md` 给模型明确说明。
- `reindex_all()` 是补 upsert / 热加载,不是清空外部向量库的管理工具。
