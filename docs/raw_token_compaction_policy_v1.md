# Raw Token Compaction Policy v1

> 状态:已实现。
> 目标:只把 raw -> episodic 的触发/批次选择从固定条数扩展为可选 token policy;episodic -> semantic 和 semantic 强化合并继续按条目。

## 背景

当前 raw 压缩由 `MemoryConfig.raw_trigger_count` 和 `MemoryConfig.summary_batch_size` 控制:

```text
未摘要 raw 条数 >= raw_trigger_count
-> 压缩最老 summary_batch_size 条
-> 剩余 raw 继续作为近期可见层
```

这个策略简单、稳定、可追溯,但对超长输入不敏感。用户可能一条消息贴入研报、公告、长文或大量金融数据;按条数看只是一条,按模型上下文看却可能已经很重。

token policy 的目标不是推翻按条目架构,而是把"什么时候压缩、压缩多少条"改成 token 预算驱动:

- 存储仍按完整 raw message 条目保存。
- 检索仍按 `source_id` / `summary_id` / `semantic_id` 条目索引与回取。
- 摘要仍记录完整 `source_ids`,不切断单条消息。
- 只计算 raw `content`,不计算时间、role、系统提示词、最终 prompt 模板等额外开销。

## 设计结论

### 只改 raw -> episodic

raw 层面对真实用户输入,长度不可控,适合 token policy。

episodic summary 是本系统自己生成的结构化摘要,长度应由摘要提示词和 JSON 契约控制;semantic 层又依赖主题重叠与强化合并。为了保持长期记忆清晰可追溯:

- `episodic -> semantic` 继续按 `episodic_compact_trigger_count` / `episodic_compact_batch_size`。
- semantic 长期记忆不做 token 压缩,只做强化合并、重要度/时间排序、检索召回。
- 暂不引入 episodic token 安全阀,避免复杂度扩散。

### 压缩边界落在 assistant 回复之后

token policy 不应在用户输入刚入库后立即压缩。推荐生命周期仍是:

```text
record_user_turn
-> build_prompt_context
-> 聊天模型回复
-> record_assistant_turn
-> compact_due_background / compact_due_sync
```

由于 `compact_due_*` 在 assistant 回复落库后调用,token policy 可以天然以完整对话轮作为压缩边界:

- 如果用户输入使 raw token 超过阈值,本轮先不压;等 assistant 回复落库后,把 assistant 回复及之前选中的条目一起作为可压缩批次。
- 如果用户输入未超过 cut ratio,但 assistant 回复后达到 cut ratio 或 trigger,则 cutpoint 可以落到这条 assistant 回复。
- 压缩批次尽量以 assistant 消息结束,避免摘要只看到问题没看到回答。

## 配置

`MemoryConfig` 提供以下可选策略字段:

```python
raw_compaction_policy: str = "count"  # "count" | "token"

# 仅 raw_compaction_policy == "token" 生效
raw_token_trigger: int = 12000
raw_token_batch_ratio: float = 0.67
raw_token_min_remainder_messages: int = 1
raw_token_boundary_role: str = "assistant"
```

含义:

- `raw_compaction_policy="count"`:保持当前行为,完全兼容现有测试与默认配置。
- `raw_token_trigger`:未摘要 raw 的 `content` token 总量达到该值后触发 raw 摘要。
- `raw_token_batch_ratio`:触发后计划压缩的 token 比例。按当前默认 count 策略类比,`20 / 30 = 0.666...`,所以 token 默认建议 `0.67`。
- `raw_token_min_remainder_messages`:压缩后至少保留多少条最近 raw,避免近期上下文被清空。
- `raw_token_boundary_role="assistant"`:cutpoint 向后对齐到 assistant 消息,形成完整问答边界。

校验规则:

- `raw_compaction_policy` 只能是 `"count"` 或 `"token"`。
- token policy 下 `raw_token_trigger > 0`。
- `0 < raw_token_batch_ratio < 1`。
- `raw_token_min_remainder_messages >= 1`。
- token policy 必须注入 `TokenCounter`,否则构造时报错,不静默退回字符估算。

## TokenCounter 接口

token 与模型 tokenizer 绑定,memcore 不应假装有通用精确算法。设计为依赖注入:

```python
class TokenCounter(ABC):
    @abstractmethod
    def count_text(self, text: str) -> int:
        raise NotImplementedError
```

`MemorySystem` 参数:

```python
MemorySystem(..., token_counter: TokenCounter | None = None)
```

`Compaction` 持有 `token_counter`。当 `raw_compaction_policy == "token"` 且未提供 counter 时,构造阶段报 `ConfigError` 或 `ValueError`。

可选默认适配器可后续再做:

- `TiktokenTokenCounter`:OpenAI/兼容模型。
- `HuggingFaceTokenCounter`:开源模型 tokenizer。
- `EstimatedTokenCounter`:只可作为显式 demo/测试估算器,名字必须带 estimated,避免假装精确。

本策略只调用:

```python
token_counter.count_text(message["content"])
```

不计算:

- role/name/time 字段。
- summary prompt 模板。
- system/developer/persona 提示词。
- 检索片段、工具 schema、最终聊天 prompt。

这些属于宿主最终 prompt 预算,不是 raw 沉淀策略的职责。

## 批次选择算法

输入:

- `msgs = store.get_unsummarized_messages(namespace=namespace)`。
- 每条消息的 `content_tokens`。
- `total_tokens = sum(content_tokens)`。

触发:

```text
total_tokens >= raw_token_trigger
```

未触发则不压缩。

选择批次:

```text
target_tokens = raw_token_trigger * raw_token_batch_ratio

从最老 raw 开始累计 content_tokens。
当累计达到或超过 target_tokens 时,得到初始 cut_index。
如果 cut_index 后面还有 assistant 消息,且不会导致剩余条数小于 raw_token_min_remainder_messages,
则 cut_index 向后移动到最近一个 assistant 消息。
最终 batch = msgs[: cut_index + 1]。
```

保留尾巴:

```text
len(msgs) - len(batch) >= raw_token_min_remainder_messages
```

如果对齐 assistant 后会吃掉全部 raw,则优先回退到能保留最小尾巴的位置。若没有任何合法尾巴可保留,
但当前未摘要 raw 已经以 assistant 结尾形成完整对话块,则允许压缩整个块。这样避免"第一轮就是超长 user + assistant"
在达到 token trigger 后仍永久暴露为可见 raw。`raw_token_min_remainder_messages` 是有可保留尾巴时的下限,不是阻止完整
assistant 边界落库的硬门槛。

## 单条超长输入

如果单条用户输入已经超过 `raw_token_trigger`,token policy 仍不在 `record_user_turn` 时立即压缩。原因:

- 当前轮聊天模型可能还需要完整用户输入。
- 摘要边界应包含 assistant 回复,否则阶段摘要容易只有问题没有回答。
- 记忆系统不应替宿主处理"当前 prompt 已爆上下文"的问题。

推荐行为:

```text
用户超长输入落 raw
-> 当前轮由宿主决定如何喂模型(可能走文档摘要/RAG/文件工具)
-> assistant 回复落 raw
-> compact_due_* 再按 token policy 压缩完整问答边界
```

如果 `user + assistant` 两条合起来已经超过 trigger,且 assistant 回复已经落库,则压缩这两条完整 raw。v1 不新增 result
状态;压缩成功仍计入 `summaries_created`,摘要通过 `source_ids` 保留完整来源。

## 与现有实现的接入点

### `memcore/config.py`

已新增配置字段与校验。

不要破坏现有 count policy 的差值约束:

```python
summary_batch_size < raw_trigger_count
```

token policy 另有自己的 ratio 约束,不是直接复用 count 差值。

### `memcore/memory_system.py`

已新增构造参数:

```python
token_counter: TokenCounter | None = None
```

构造 `Compaction` 时传入。若 token policy 开启但未传 counter,构造即失败。

### `memcore/compaction.py`

`_summarize_raw()` 已从固定 count 循环改为策略选择。一次
`Compaction.run_due()` 只推进一个可用 raw batch；如果仍有更多欠账,
由下一次后台调度继续推进。这样可以限制单次维护的模型调用和执行时间,
但不改变阈值、批次选择、失败重试或 raw→summary 的数据语义:

```python
msgs = self.store.get_unsummarized_messages(namespace=namespace)
batch = self._select_raw_summary_batch(msgs)
if batch:
    ...  # 创建一个 summary 并标记这一批 source_ids
```

已新增:

```python
def _select_raw_summary_batch(self, msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if self.config.raw_compaction_policy == "count":
        ...
    if self.config.raw_compaction_policy == "token":
        ...
```

count 分支保持现有行为:

```python
return msgs[: cfg.summary_batch_size] if len(msgs) >= cfg.raw_trigger_count else []
```

token 分支执行上文批次选择算法。

阶段摘要到长期语义记忆同样采用单批推进:一次 `run_due()` 最多处理一个
`episodic_compact_batch_size` 批次(包括一次必要的 semantic reinforcement),
后续批次由下一次调度处理。压缩欠账因此是渐进清理,不会在一次请求内连续
调用模型直到 namespace 被清空。

### 新模块

已新增:

```text
memcore/token_counter.py
```

v1 只实现 `TokenCounter` 接口;生产 tokenizer 适配器后续可按需增加。

## 测试清单

### 配置测试

- 默认 `raw_compaction_policy == "count"` 保持现有行为。
- 非法 policy 报错。
- `raw_token_trigger <= 0` 报错。
- `raw_token_batch_ratio <= 0` 或 `>= 1` 报错。
- token policy 未注入 `TokenCounter` 构造失败。

### raw token policy 测试

- 未达到 `raw_token_trigger` 不压缩。
- 达到 trigger 后,按 `raw_token_batch_ratio` 选择完整消息批次。
- cutpoint 向后对齐到 assistant 消息。
- 用户输入导致超过 target 时,包含下一条 assistant 回复一起压缩。
- assistant 回复导致超过 target 时,cutpoint 可落在该 assistant 回复。
- 有可保留尾巴时,压缩后至少保留 `raw_token_min_remainder_messages` 条 raw。
- 单条超长 user + assistant 后仍按完整条目压缩,不切内容。
- 摘要 `source_ids` 等于被压缩的完整 raw source_id 列表。
- 失败重试仍保持 raw 未标记 summarized,与现有 `test_summary_failure_keeps_raw_for_retry` 一致。

### 不改的测试

- episodic -> semantic 仍按条目触发。
- semantic reinforcement 仍按条目和重叠分。
- retrieve/read_timeline/visible context 不受 token policy 影响。

## 非目标

- 不做最终聊天 prompt 的精确 token 预算。
- 不按字符切消息。
- 不切分单条 raw content。
- 不把 token policy 扩散到 episodic -> semantic。
- 不引入 hidden router 或额外 LLM 判断是否压缩。
- 不静默使用 hashed/estimated/token fallback 冒充生产精确 token。

## 推荐默认

```python
MemoryConfig(
    raw_compaction_policy="count",  # 默认保守兼容
)
```

长文本/金融/研报场景可显式开启:

```python
MemoryConfig(
    raw_compaction_policy="token",
    raw_token_trigger=12000,
    raw_token_batch_ratio=0.67,
    raw_token_min_remainder_messages=1,
)
```

接入方必须同时提供与聊天模型或摘要模型口径一致的 `TokenCounter`。
