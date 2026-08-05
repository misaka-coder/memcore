# memcore

领域无关、可扩展、可授权的**统一时间线与分层记忆内核**。

> memcore 不是“把聊天记录塞进向量库”。它把消息、事件、工具、材料和模型
> 回复放进同一条可追溯时间线，再分别处理稳定投影、检索准入、关系扩窗和三层
> 压缩。对外能力地图见 [`docs/public_capabilities_v1.md`](docs/public_capabilities_v1.md)。

- **写侧**:working(原始对话)→ episodic(阶段摘要)→ semantic(长期事实)三层压缩 + 强化合并。
- **读侧**:显式 `retrieve` / `browse_memory` / `open_memory` / `read_timeline` 互补工具 + 可选原生 `load_material` 分发 + 向量/关键词混合检索 + RRF。raw 先于摘要和长期记忆占用结果位；facet/role 保持前置过滤，只有准确实体导致候选为零时才单独放宽实体条件，并返回可解释 diagnostics。读侧不再追加一次 LLM verifier 调用。
- **贯穿**:时间锚点(带时区)、命名空间硬隔离、可选 flavor 层、提示词注入防线。

实现说明按以下公开文档维护：[`usage_flow_v1.md`](docs/usage_flow_v1.md)、
[`memory_read_api_v1.md`](docs/memory_read_api_v1.md)、
[`memory_metadata_raw_retrieval_design_v1.md`](docs/memory_metadata_raw_retrieval_design_v1.md) 和
[`model_prompt_playbook_v1.md`](docs/model_prompt_playbook_v1.md)。

## 当前进度

- **切片 1(边界)✅**:字段契约 `schema` + `MemoryConfig` + `Namespace`(五层)+ 四类 base 接口
  (`LLMClient` / `MemoryStore` / `VectorIndex` / `EmbeddingProvider`)+ `MemorySystem` 空壳。
- **切片 2(检索原语)✅**:`time_anchor`(带时区)+ `rendering` + `embedding`(hashed/HF)
  + `index`(RRF + 内存索引语义/BM25 + Chroma 懒加载)。可单测,不依赖 LLM。
- **切片 3(存储)✅**:`SQLiteMemoryStore` —— SQLite 事实源、Namespace 硬隔离、
  Timeline/summary/semantic/projection/audit 表、index_status outbox、幂等 source_id、
  事务与定向遗忘。
- **repair pass ✅**:堵住跨 namespace 泄漏、可见层跨会话、默认时间渲染 1970、删除回传 ID、
  跨 namespace 覆盖、时区校验六处接缝(见 tests/test_slice3b_repair.py)。
- **切片 4(写侧)✅**:`compaction` 三层压缩(raw→摘要→语义)+ 主题重叠强化合并(注入 LLMClient,
  按 namespace 加锁,outbox 索引);`MemorySystem` 写侧 record/compact 接通。压缩 LLM 失败时不会提交空摘要,
  会返回 `summary_retry_pending` / `semantic_retry_pending`,并保留原记录供下一轮后台压缩重试。
- **切片 5(读侧)✅**:`retrieval` —— 显式 retrieve 工具 → metadata 前置过滤 → raw/derived 分池混合检索 + RRF → 确定性分数与关系完整性检查；原始 query 始终保留，`entity_anchors` 高权重，`topic_terms` 只作普通辅助；`time_hint.start_at/end_at` 复用时间线的本地/ISO 解析并在评分前执行起点包含、终点不包含的硬过滤，未知答案不需要也不允许伪装成实体锚点；
  `build_prompt_context` 只拼可见三层,是否检索交给聊天模型调用工具决定。**读写侧全闭环。**
- **时间线工具 ✅**:`read_timeline(...)` 支持无需 epoch 的 `time_range.start_at/end_at` 小时/分钟级读取，旧日期/时间段字段归一到同一 timestamp 路径，也支持以 raw `source_id` 为锚点读取前后完整 turn；默认 `conversation` 投影保留完整对话/事件并把大工具与材料轨迹变成可展开凭据，`full/tools` 可显式切换。模型侧 native dispatcher 默认使用可配置的有限页面预算，按完整 turn 无损分页，并返回所选总量、本页总量、`partial/page_boundary`、后续动作和可校验 `next_cursor`；native result 只发送一份渲染正文，不再把等价 `messages` 正文重复喂给模型。可信宿主直接调用 Python API 时仍保留结构化 messages/text 双视图，并可显式使用 `page_token_budget=0` 完整读取。
- **记忆目录导航 ✅**:`browse_memory(...)` 按确定性时间范围返回有界卡片目录与 raw 覆盖状态；`open_memory(memory_id, view=card/content/sources)` 可从摘要正文继续展开精确子摘要或完整 raw 逻辑单元。`sources` 默认使用 `conversation` 投影：完整保留对话/事件，工具、Skill 与材料正文只显示可按 `source_id` 重载的紧凑轨迹；只有显式 `projection=full/tools` 或打开单条 raw `content` 才返回工具正文。`retrieve_for_turn(within_memory_id=...)` 可在已选节点的精确 lineage 内继续做 dense/BM25 模糊检索，实体条件放宽不会移除该边界，段内为空也不会退回全库。分页使用 namespace-safe 稳定键 cursor，不截断卡片/turn，缺失 lineage 明确返回 `partial`。
- **精确条目展开 ✅**:`read_entry(source_id, detail)` 是 `open_memory(view="content")` 的 current-conversation raw-only 兼容适配；summary/semantic、越权 ID、密钥和本地路径不会伪装成 raw 正文。
- **embedding 三条路 + 自检 ✅**:`HuggingFaceEmbeddingProvider`(本地 BGE-M3)/ `HTTPEmbeddingProvider`(OpenAI 兼容 API,纯 stdlib 零依赖)/ `HashedEmbeddingProvider`(仅测试)。
  `EmbeddingProvider` 同时提供 `embed_query/embed_queries` 与
  `embed_document/embed_documents`；对称模型默认复用旧 `embed_text(s)`，Jina 等非对称模型可分别实现 query/passage，内存与 Chroma 索引会走正确通道。
  `RoleAwareHTTPEmbeddingProvider` 接受宿主提供的 query/document 请求体扩展，不内置厂商 task 名；全量重建与 pending 修复按批调用远程 provider。
  `verify_embedding()` 自检语义是否真有效(近义词应明显更近),hashed/弱模型会被响亮标记。**不捆绑任何模型权重。**
- **outbox 自愈 ✅**:向量后端故障时记录仍安全落库(pending),`reindex_pending()` 恢复后补齐索引;
  `reindex_all()` 可从 SQLite 真相源补建/热加载当前 hard namespace 的三层索引,记录/压缩都不被向量故障阻断。
- **提示词治理 ✅**:焊死骨架 + 校验插槽(`PromptOverrides`);插槽只能补充、不可移除契约/时间锚点。
- **importance 衰减 ✅**:`enable_importance_decay` 开启后,长期记忆可见窗口按"随时间衰减的重要度"排序(久未强化的记忆淡出),衰减对**全部**候选生效、不静默截断。
- **raw token 差值压缩 ✅**:唯一 raw 压缩策略按目标 provider 的完整投影 token 触发，按比例选择最旧的完整 turn/relation component；可注入精确或明确标记为 `estimated` 的 `TokenCounter`。
- **星期感知时间锚点 ✅**:raw、摘要、语义与时间线渲染会由 `timestamp + timezone` 自动派生 `周一..周日`,
  让“上周二/下周三”这类相对表达在压缩与检索回填时有明确参照。
- **内存索引加速 ✅**:默认不强制向量数据库;`InMemoryVectorIndex` 会先按 namespace/time/exclude 与可前置 metadata 过滤候选,
  再计算分数。安装 `memcore[speed]` 后向量以 float32 存储,语义 cosine 自动走可选 NumPy 批量计算。
- **模型协作提示词 ✅**:标准输出契约与压缩链提示词已补充时间锚点、工具选择、群聊归因、metadata 标注规则。
  接入方提示词指南见 `docs/model_prompt_playbook_v1.md`。
- **工具轨迹 ✅**:`append_action(...)` / `append_observation(...)` 把调用与结果追加到同一开放 turn，
  通过 `correlation_id` 支持并行和乱序返回；`record_tool_exchange(turn_id=...)` 只是这两个 V2 API 的薄适配。
  工具轨迹参与同一 token 生命周期，但压缩为独立 operation digest；普通检索仍默认排除，显式授权后可检索。
- **材料轨迹类别 ✅**:材料引用/清理使用 typed standalone entry，或作为当前 turn 的 `material.*` intermediate。
  只保存 file_id、文件名、类型和状态；原始文件与 OCR/视觉描述/文档 chunks 由宿主存储。压缩时材料进入 operation 分区，不污染对话摘要；普通检索默认排除。
- 可配置:`raw_token_trigger`、`raw_token_batch_ratio`、`retrieval_result_token_budget`、`native_timeline_page_token_budget`、`visible_memory_scope`、`enable_flavor`、`enable_importance_decay`。
- 压缩重试:`llm_max_retries` 会传给注入的 `LLMClient`;最终仍失败时压缩层不标记已完成,下一轮继续重试。
- **Chat Output Adapter ✅**:标准 JSON 输出契约、`speech` 流式解析、普通文本尽力分段、raw metadata 回写流程见 `docs/chat_output_adapter_v1.md`;工具调用阶段不套该 JSON,只在最终回复阶段输出 memcore JSON。
- **稳定投影与缓存审计 ✅**:canonical/OpenAI/Anthropic provider projection、renderer/version、strict-prefix 验收、projection hash 与真实请求 audit;MemCore 保证前缀稳定,不替 provider 承诺缓存必命中。
- **开放动作/结果时间线 ✅**:`append_action(...)` / `append_observation(...)` 可记录原生工具、JSON、XML、标签或宿主自定义协议；模型实际看到的完整结果与调用一起保留到统一 raw token 压缩，可选小型 `retention_anchor` 在 operation 压缩后继续保留资源 ID、版本、hash 等重载锚点。实际 provider 消息可冻结回 projection ledger。
- **有界后台维护 ✅**:每次 `compact_due` 只提交一个 raw compaction generation 和一个 semantic batch；
  token 模式按配置比例一次选足最旧的完整 turn/component，不再被旧条目批次或独立 source 上限提前截断。

核心 + 评测台 + Timeline V2 + provider projection + embedding 三路 + outbox 自愈
+ Chat Output Adapter + importance 衰减均已完成。flat V1 compactor 已删除；无 turn_id 的历史记录由 V2 规划器包装为 closed standalone component 后原子压缩。
可选扩展(按需):更大语料的 BM25/向量后端；`enable_flavor` 已作为可配置能力落地,
不会强迫纯事实型接入启用口吻或情绪字段。

可运行的最小接入样板见 `examples/minimal_chat_integration.py`。它演示一轮聊天里
`begin_turn` → 可见三层 → `retrieve_for_turn` / `read_timeline` 工具 → final JSON 解析 →
`complete_turn` 原子提交 metadata 与回复 → 后台压缩的完整闭环。动作/结果时间线的
独立可运行示例见 `examples/non_native_operation_timeline.py`。

原生 tool calling 接入可用 `build_native_memory_tool_specs(...)` 生成工具 schema,再用
`dispatch_native_memory_tool(...)` 分发 `retrieve_for_turn` / `browse_memory` / `open_memory` / `read_timeline` / `load_material`。旧 `read_entry` 只保留 Python/dispatcher 兼容适配，不再进入模型可见的生成工具列表；新接入统一使用 `open_memory`。
`load_material` 只调用宿主传入的 `material_loader` 回调,用于读取 file_store/derived_store 中的原图、
OCR、视觉描述、文档 chunks 或当前清理状态;memcore 不保存文件本体。
非多模态接入不要让最终聊天模型和视觉/OCR 解析赛跑:要么先等宿主 derived_store 写入同一
`file_id` 的摘要/OCR,要么让 `load_material` 返回 pending/unavailable,避免模型只凭附件锚点或旧结果猜图。

设计亮点说明见 `docs/design_highlights_v1.md`;接入聊天模型时建议先读 `docs/model_prompt_playbook_v1.md`。
如果让 AI 编码助手接入本库,请先把根目录 `AGENTS.md` 交给它读;独立接入流程见
`docs/usage_flow_v1.md`。
模型服务前缀缓存友好的 prompt 拼接顺序见 `docs/model_prompt_playbook_v1.md` 的“缓存友好 Prompt 布局”。
宿主中立的动作/结果接入、非原生协议真实投影与可选压缩锚点见
[`docs/operation_timeline_v1.md`](docs/operation_timeline_v1.md)，可运行示例见
[`examples/non_native_operation_timeline.py`](examples/non_native_operation_timeline.py)。

## 端到端用法

```python
mem = MemorySystem(llm=MyLLMClient(), namespace=Namespace(user_id="u1", conversation_id="c1"),
                   timezone="Asia/Shanghai", embedding="BAAI/bge-m3")
handle = mem.begin_turn(stimuli=[TimelineEntryInput(
    kind="message.user",
    origin=EntryOrigin.USER,
    turn_role=TurnRole.STIMULUS,
    semantic_text="我之前说过爱喝什么",
    payload={"text": "我之前说过爱喝什么"},
)])
cur = handle.stimuli[0].to_record()
ctx = mem.build_prompt_context(current=cur)   # 可见三层;是否检索由聊天模型自行调用 retrieve/read_timeline
ctx_text = mem.render_prompt_context(ctx)     # 推荐文本渲染:近期 raw 按日期分组,自带星期/时间段
# ...用 ctx + memcore 的原生记忆工具拼你自己的最终聊天 prompt、调你自己的聊天模型...
# 当前轮工具包装推荐用 retrieve_for_turn(current=cur, ...),避免把 prompt 已可见三层重复检索回来。
mem.record_tool_exchange(
    turn_id=handle.turn_id,
    tool_name="web_search",
    tool_call_id="call_001",
    tool_input={"query": "北京天气"},
    result="北京今天 25°C,晴天",
)
mem.append_entry(TimelineEntryInput(
    kind="material.image.reference",
    origin=EntryOrigin.ENVIRONMENT,
    turn_role=TurnRole.INTERMEDIATE,
    semantic_text="图片材料已就绪",
    payload={"file_id": "file_img_001", "filename": "photo.jpg", "status": "ocr_ready"},
    semanticize=False,
), turn_id=handle.turn_id)
mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text=reply,
    provider_output_raw=raw_model_output,
    memory_annotation=parsed.memory_metadata,
    annotation_status="accepted_model",
)
future = mem.compact_due_background()          # 聊天链路推荐后台沉淀,不阻塞用户可见回复
# 可忽略 future 做 fire-and-forget;测试/脚本可 future.result() 读取压缩统计
```

宿主应在模型或交付异常的 `finally` 路径立即调用 `abort_turn()`。为处理进程崩溃、
强制关机等无法执行 `finally` 的情况，启动或新一轮开始前可按产品超时策略调用：

```python
recovered = mem.recover_stale_open_turns(max_age_seconds=1800)
```

该操作只会原子终止当前 conversation namespace 内早于截止时间的 `open` turn，
不会删除消息、摘要或已完成轮次。超时窗口由宿主选择；不能用它代替正常异常路径的即时 abort。

`complete_turn()` 默认写入 `message.assistant`。语音、具身或其他宿主定义的
typed final 可显式传入开放 namespaced kind，例如
`kind="message.assistant.voice"`。普通 assistant final 继续使用原始纯文本
provider 投影；typed final 会把自然回复保留为顶层 `speech`，并把 kind 与结构化
payload 收进 `host_state`。这样交付、打断等宿主状态在后续模型上下文中仍然可见，
又不会让模型把内部状态字段误认成下一轮最终回复格式。

原生工具循环示意:

```python
import json

tools = build_native_memory_tool_specs()

tool_payload = dispatch_native_memory_tool(
    tool_name=tool_call.name,
    arguments=tool_call.arguments,
    mem=mem,
    current=cur,
    material_loader=load_material_from_host_store,  # 宿主实现
    policy=ToolDispatchPolicy(
        allow_explicit_trace=True,
        allowed_kind_prefixes=("material",),  # 只开放产品已授权的 kind 前缀
    ),
)

# receipt 是宿主侧导航锚点；把其余完整结果作为 provider 原生 tool_result 回给模型。
provider_result = {key: value for key, value in tool_payload.items() if key != "receipt"}
provider_result_text = json.dumps(provider_result, ensure_ascii=False, sort_keys=True)
# 将模型实际看到的同一份结果写入 observation。它在后续回合继续可见，直到统一
# raw token 差值压缩；receipt 只作为 operation digest 可保留的小型重载锚点。
mem.append_observation(
    turn_id=handle.turn_id,
    kind=f"operation.memory.{tool_call.name}.result",
    correlation_id=tool_call.id,
    semantic_text=provider_result_text,
    payload={"output": provider_result_text},
    retention_anchor=tool_payload["receipt"],
    status=tool_payload["receipt"]["status"],
)
```

宿主不使用原生 tool calling 时也不需要另建历史系统。宿主解析模型自己的
JSON/标签后，可将请求和结果追加到同一轮：

```python
mem.append_action(
    turn_id=turn_id,
    kind="operation.catalog.request",
    correlation_id="catalog-1",
    payload={"action": "list_capabilities"},
)
mem.append_observation(
    turn_id=turn_id,
    kind="operation.catalog.response",
    correlation_id="catalog-1",
    payload={"items": ["search", "weather"]},
    status="success",
    retention_anchor={"catalog_ref": "catalog:v3", "schema_hash": "sha256:abc"},
)
```

`kind` 和 payload 均由宿主定义；MemCore 不解析协议、不决定权限、不执行工具。
若实际 provider history 使用普通 JSON/XML/标签消息，用 `record_request_projection(...)`
冻结真实消息即可，后续严格前缀不会被默认原生工具投影改写。

进程重启、换一个新的内存索引实例,或升级索引 metadata 字段后,可以从 SQLite 真相源补建/热加载索引:

```python
stats = mem.reindex_all()  # 默认 upsert 当前 tenant/user/domain 下全部会话的 raw/summary/semantic
# stats: {"scanned": 42, "reindexed": 42, "failed": 0}
```

`reindex_all()` 不会清空已有 index 里的陈旧条目;它适合空内存索引冷启动/补 upsert。若外部向量库已污染,
应先用后端管理工具清理对应集合或新建空 index。

`compact_due_sync()` 是同步确定性入口,适合单测、CLI、管理脚本或进程退出前 flush。在线聊天产品默认应使用
`compact_due_background()`。

如果宿主按请求动态选择 provider,应把本轮实际投影 profile 传给压缩入口,例如
`mem.compact_due_background(provider_profile="openai_chat")`。这样 `projected_tokens` 预算按模型真正收到的
provider history 计算；不传时继续使用 `MemoryConfig.projection_profile`,兼容固定 provider 的宿主。

raw token 压缩只影响 raw → episodic 的触发/批次选择，不改变检索条目结构。若宿主能提供与模型对齐的 tokenizer，
注入 `TokenCounter` 可得到 exact 口径；没有 counter 时 MemCore 使用内置保守估算，并在结果中明确返回
`token_count_quality="estimated"`，不会伪装成精确 tokenizer：

```python
class MyTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return count_with_your_model_tokenizer(text)

cfg = MemoryConfig(raw_token_trigger=12000, raw_token_batch_ratio=0.67)
mem = MemorySystem(..., config=cfg, token_counter=MyTokenCounter())
```

压缩统计目标 provider 的完整 raw projection，并以完整 terminal turn/relation
component 为切点。系统中不存在 count policy、独立消息 batch cap 或第二套 flat compactor。

## 公开边界:本库提供什么 / 接入方自备什么

memcore 是**纯机制**:它不含任何具体人格、领域调教或模型权重。这条边界让它可以放心交付/授权。

| memcore 提供(机制) | 接入方自备(你的资产) |
|---|---|
| 三层记忆、压缩、强化、时间锚点 | 具体**人格文本**(经 `persona_text` / `PromptOverrides` 运行时注入) |
| raw-first 混合检索、可观测实体放宽、核心读工具与原生工具分发辅助 | 你的**聊天模型**(`LLMClient` 只用于三层压缩，不介入读侧筛选) |
| 统一 metadata 契约 + 提示词骨架 + 校验插槽 | **领域补充说明**与**调参**(窗口/阈值)；不能替换固定 facet/role 协议 |
| embedding 接口 + 三路适配器 + 自检 | **embedding 模型**(本地 / API / 自有) |
| 隔离、outbox 自愈、遗忘、评测台 | 领域**合规规则**(memcore 只保证记忆不越权变指令) |

> 焊死项(时间锚点、字段契约、"只输出 JSON"等)无法被外部覆盖;插槽只能补充。详见 `prompts.PromptOverrides`。

### ⚠️ 隔离单位 = `tenant_id / user_id / domain_id`,不是 actor

记忆的硬隔离边界是 Namespace 的 `hard_key`(tenant/user/domain)。**`actor` 是软标签**(群聊里"谁说的"),
**不进硬隔离、同一 user_id 下所有 actor 共享记忆池**。要"按人隔离",必须把"人"映射到 **user_id**,不能指望 actor。
(检索 where 按 hard_key 过滤;`ChromaVectorIndex` 已内置 where 方言适配,把多条件转成 Chroma 的 `$and` 语法。)

## 公共 API

`import memcore` 暴露:`MemorySystem`、`MemoryConfig`、`Namespace`/`Actor`、`PromptOverrides`、
`LLMClient`/`LLMRequest`/`LLMResult`、`MemoryStore`/`VectorIndex`/`EmbeddingProvider`/`TokenCounter` 接口、
默认实现 `SQLiteMemoryStore`/`InMemoryVectorIndex`/`HashedEmbeddingProvider`/`HuggingFaceEmbeddingProvider`/`HTTPEmbeddingProvider`、
`verify_embedding`、`build_native_memory_tool_specs` / `dispatch_native_memory_tool` / `build_memory_operation_receipt`、
`build_action_entry` / `build_observation_entry`、Timeline V2 与 projection 契约、
`MemoryMetadata`/`SummaryRecord`/`SemanticRecord`、异常类。

## 跑测试

```bash
uv run --extra dev python -m unittest discover -s tests -v
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
```

## 授权

**专有软件,默认保留所有权利(All Rights Reserved)。** 见 [LICENSE](LICENSE)。

- 本库**不开源**;未经书面商业授权,不得使用、复制、修改、再分发。
- **商用需单独签授权协议**,授权范围与费用另行约定;对外仅按商业授权交付 wheel / 源码副本。
- 如日后发布开源版本,将另行声明其许可证;本仓库不授予任何开源权利。
- 本库不含受私有许可约束的人格/权重内容(人格由接入方运行时注入),因此可作为纯机制独立授权。
