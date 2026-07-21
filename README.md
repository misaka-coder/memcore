# memcore

领域无关、可扩展、可授权的**分层记忆内核**。

- **写侧**:working(原始对话)→ episodic(阶段摘要)→ semantic(长期事实)三层压缩 + 强化合并。
- **读侧**:显式 `retrieve` / `read_timeline` 核心读工具 + 可选原生 `load_material` 分发 + 向量/关键词混合检索 + RRF + verifier + 五级自动放宽。
- **贯穿**:时间锚点(带时区)、命名空间硬隔离、温度层可关、注入防线。

设计规格见桌面《可复用记忆库_设计文档_v1.md》(v1.1),提示词原文见《Akane记忆系统提示词原文.md》。
本仓库按该文档路线图分片实现。

## 当前进度

- **切片 1(边界)✅**:字段契约 `schema` + `MemoryConfig` + `Namespace`(五层)+ 四类 base 接口
  (`LLMClient` / `MemoryStore` / `VectorIndex` / `EmbeddingProvider`)+ `MemorySystem` 空壳。
- **切片 2(检索原语)✅**:`time_anchor`(带时区)+ `rendering` + `embedding`(hashed/HF)
  + `index`(RRF + 内存索引语义/BM25 + Chroma 懒加载)。可单测,不依赖 LLM。
- **切片 3(存储)✅**:`SQLiteMemoryStore` —— 三张记忆表、Namespace 五层隔离、index_status outbox、
  幂等 source_id + 事务、定向遗忘。
- **repair pass ✅**:堵住跨 namespace 泄漏、可见层跨会话、默认时间渲染 1970、删除回传 ID、
  跨 namespace 覆盖、时区校验六处接缝(见 tests/test_slice3b_repair.py)。
- **切片 4(写侧)✅**:`compaction` 三层压缩(raw→摘要→语义)+ 主题重叠强化合并(注入 LLMClient,
  按 namespace 加锁,outbox 索引);`MemorySystem` 写侧 record/compact 接通。压缩 LLM 失败时不会提交空摘要,
  会返回 `summary_retry_pending` / `semantic_retry_pending`,并保留原记录供下一轮后台压缩重试。
- **切片 5(读侧)✅**:`retrieval` —— 显式 retrieve 工具 → 混合检索 + RRF + 五级放宽 + raw 扩窗 → verifier 门;
  `build_prompt_context` 只拼可见三层,是否检索交给聊天模型调用工具决定。**读写侧全闭环。**
- **时间线工具 ✅**:`read_timeline(date_from, date_to, time_periods)` —— 按时间精确读原始对话(不走向量),
  与 `retrieve`(向量模糊检索)互补,构成核心读工具对。
- **embedding 三条路 + 自检 ✅**:`HuggingFaceEmbeddingProvider`(本地 BGE-M3)/ `HTTPEmbeddingProvider`(OpenAI 兼容 API,纯 stdlib 零依赖)/ `HashedEmbeddingProvider`(仅测试)。
  `verify_embedding()` 自检语义是否真有效(近义词应明显更近),hashed/弱模型会被响亮标记。**不捆绑任何模型权重。**
- **outbox 自愈 ✅**:向量后端故障时记录仍安全落库(pending),`reindex_pending()` 恢复后补齐索引;
  `reindex_all()` 可从 SQLite 真相源补建/热加载当前 hard namespace 的三层索引,记录/压缩都不被向量故障阻断。
- **提示词治理 ✅**:焊死骨架 + 校验插槽(`PromptOverrides`);插槽只能补充、不可移除契约/时间锚点。
- **importance 衰减 ✅**:`enable_importance_decay` 开启后,长期记忆可见窗口按"随时间衰减的重要度"排序(久未强化的记忆淡出),衰减对**全部**候选生效、不静默截断。
- **raw token 压缩 ✅**:默认仍按条数压缩;长文本/金融/研报场景可显式开启 `raw_compaction_policy="token"`,并注入 `TokenCounter`。
- **星期感知时间锚点 ✅**:raw、摘要、语义与时间线渲染会由 `timestamp + timezone` 自动派生 `周一..周日`,
  让“上周二/下周三”这类相对表达在压缩与检索回填时有明确参照。
- **内存索引加速 ✅**:默认不强制向量数据库;`InMemoryVectorIndex` 会先按 namespace/time/exclude 与可前置 metadata 过滤候选,
  再计算分数。安装 `memcore[speed]` 后向量以 float32 存储,语义 cosine 自动走可选 NumPy 批量计算。
- **模型协作提示词 ✅**:标准输出契约与压缩链提示词已补充时间锚点、工具选择、群聊归因、metadata 标注规则。
  接入方提示词指南见 `docs/model_prompt_playbook_v1.md`。
- **工具轨迹类别 ✅**:`record_tool_exchange(...)` 可把工具调用/结果以 `assistant.tool_call ...` +
  `tool.<name> ...` 的线性事件块追加进 raw。
  默认参与 count-based raw 压缩触发并进入正常摘要生命周期;普通检索仍默认排除,显式 `categories=["tool_trace"]` 时可检索工具轨迹。
- **材料轨迹类别 ✅**:`record_material_reference(...)` / `record_material_cleanup(...)` 可把图片、文件、解析物状态以
  `user.attachment ...` / `system.material_cleanup ...` 事件块追加进 raw。事件只保存 file_id、文件名、类型和状态;
  原始文件与 OCR/视觉描述/文档 chunks 由宿主 file_store/derived_store 管理。默认不计入 count-based raw 压缩触发数量,
  普通检索也默认排除;显式 `categories=["material_trace"]` 时可检索材料锚点。
- 可配置:`visible_memory_scope`(conversation/user)、`enable_verifier`、`enable_flavor`、`enable_importance_decay`、`raw_compaction_policy`。
- 压缩重试:`llm_max_retries` 会传给注入的 `LLMClient`;最终仍失败时压缩层不标记已完成,下一轮继续重试。
- **Chat Output Adapter 设计草案**:标准 JSON 输出契约、`speech` 流式解析、普通文本尽力分段、raw metadata 回写流程见 `docs/chat_output_adapter_v1.md`;工具调用阶段不套该 JSON,只在最终回复阶段输出 memcore JSON。

核心 + 评测台 + 时间线 + embedding 三路 + outbox 自愈 + 打包(专有授权)+ importance 衰减 均已完成。
可选后续(按需):陪伴 flavor、大语料 BM25 可扩展后端 —— 见设计文档 §15。

可运行的最小接入样板见 `examples/minimal_chat_integration.py`。它演示一轮聊天里
`record_user_turn` → 可见三层 → `retrieve_for_turn` / `read_timeline` 工具 → final JSON 解析 →
metadata 回写 → `record_assistant_turn` → 后台压缩的完整闭环。

原生 tool calling 接入可用 `build_native_memory_tool_specs(...)` 生成工具 schema,再用
`dispatch_native_memory_tool(...)` 分发 `retrieve_for_turn` / `read_timeline` / `load_material`。
`load_material` 只调用宿主传入的 `material_loader` 回调,用于读取 file_store/derived_store 中的原图、
OCR、视觉描述、文档 chunks 或当前清理状态;memcore 不保存文件本体。
非多模态接入不要让最终聊天模型和视觉/OCR 解析赛跑:要么先等宿主 derived_store 写入同一
`file_id` 的摘要/OCR,要么让 `load_material` 返回 pending/unavailable,避免模型只凭附件锚点或旧结果猜图。

设计亮点说明见 `docs/design_highlights_v1.md`;接入聊天模型时建议先读 `docs/model_prompt_playbook_v1.md`。
如果让 AI 编码助手接入本库,请先把根目录 `AGENTS.md` 交给它读;独立接入流程见
`docs/usage_flow_v1.md`。
模型服务前缀缓存友好的 prompt 拼接顺序见 `docs/model_prompt_playbook_v1.md` 的“缓存友好 Prompt 布局”。

## 端到端用法

```python
mem = MemorySystem(llm=MyLLMClient(), namespace=Namespace(user_id="u1", conversation_id="c1"),
                   timezone="Asia/Shanghai", embedding="BAAI/bge-m3")
cur = mem.record_user_turn("我之前说过爱喝什么")
ctx = mem.build_prompt_context(current=cur)   # 可见三层;是否检索由聊天模型自行调用 retrieve/read_timeline
ctx_text = mem.render_prompt_context(ctx)     # 推荐文本渲染:近期 raw 按日期分组,自带星期/时间段
# ...用 ctx + memcore 的原生记忆工具拼你自己的最终聊天 prompt、调你自己的聊天模型...
# 当前轮工具包装推荐用 retrieve_for_turn(current=cur, ...),避免把 prompt 已可见三层重复检索回来。
mem.record_tool_exchange(
    tool_name="web_search",
    tool_call_id="call_001",
    tool_input={"query": "北京天气"},
    result="北京今天 25°C,晴天",
)
mem.record_material_reference(
    file_id="file_img_001",
    kind="image",
    actor=actor_or_none,  # 群聊/多人上传时传 Actor,保留上传者归因
    filename="photo.jpg",
    mime_type="image/jpeg",
    file_status="ready",
    derived_status="ocr_ready",  # 多模态可直接看当前图片;非多模态可读取宿主保存的 OCR/描述
)
mem.record_assistant_turn(reply, in_reply_to=cur)
future = mem.compact_due_background()          # 聊天链路推荐后台沉淀,不阻塞用户可见回复
# 可忽略 future 做 fire-and-forget;测试/脚本可 future.result() 读取压缩统计
```

原生工具循环示意:

```python
tools = build_native_memory_tool_specs(categories=config.categories)

tool_payload = dispatch_native_memory_tool(
    tool_name=tool_call.name,
    arguments=tool_call.arguments,
    mem=mem,
    current=cur,
    material_loader=load_material_from_host_store,  # 宿主实现
)

# 把 tool_payload 作为 provider 原生 tool_result 回给聊天模型。
# 如需跨轮追问该工具结果,再用 record_tool_exchange(...) 写入 tool_trace。
```

进程重启、换一个新的内存索引实例,或升级索引 metadata 字段后,可以从 SQLite 真相源补建/热加载索引:

```python
stats = mem.reindex_all()  # 默认 upsert 当前 tenant/user/domain 下全部会话的 raw/summary/semantic
# stats: {"scanned": 42, "reindexed": 42, "failed": 0}
```

`reindex_all()` 不会清空已有 index 里的陈旧条目;它适合空内存索引冷启动/补 upsert。若外部向量库已污染,
应先用后端管理工具清理对应集合或新建空 index。

`compact_due_sync()` 是同步确定性入口,适合单测、CLI、管理脚本或进程退出前 flush。在线聊天产品默认应使用
`compact_due_background()`。

raw token 压缩只影响 raw → episodic 的触发/批次选择,不会改变检索条目结构。开启时必须提供与模型 tokenizer 对齐的
`TokenCounter`;memcore 不会静默用字符估算冒充 token:

```python
class MyTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return count_with_your_model_tokenizer(text)

cfg = MemoryConfig(raw_compaction_policy="token", raw_token_trigger=12000, raw_token_batch_ratio=0.67)
mem = MemorySystem(..., config=cfg, token_counter=MyTokenCounter())
```

## 公开边界:本库提供什么 / 接入方自备什么

memcore 是**纯机制**:它不含任何具体人格、领域调教或模型权重。这条边界让它可以放心交付/授权。

| memcore 提供(机制) | 接入方自备(你的资产) |
|---|---|
| 三层记忆、压缩、强化、时间锚点 | 具体**人格文本**(经 `persona_text` / `PromptOverrides` 运行时注入) |
| 混合检索、verifier、放宽、核心读工具与原生工具分发辅助 | 你的**聊天模型**(`LLMClient` 适配器) |
| 焊死提示词骨架 + 校验插槽 | **领域词表**(`categories`)与**调参**(窗口/阈值) |
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
默认实现 `SQLiteMemoryStore`/`InMemoryVectorIndex`/`HashedEmbeddingProvider`/`HTTPEmbeddingProvider`、
`verify_embedding`、`build_native_memory_tool_specs` / `dispatch_native_memory_tool`、契约
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
