# Operation projection settlement API v1

本文是 `compact_after_terminal` 的权威接入文档。它描述的是 **provider-visible
历史投影** 如何在一个 turn 完成后变紧凑，而不是删除或重写 SQLite 中的工具
调用/结果真相。

适用场景：宿主会把工具调用和完整结果写进 MemCore，希望当前工具循环保持完整，
但不希望几千到几万字符的旧结果在后续每次模型请求中重复出现。

## 1. Public surface

公开入口只有 `MemorySystem`、`MemoryConfig` 和稳定策略枚举；不要直接调用 store
里的 settlement 写方法。

```python
from memcore import MemoryConfig, OperationProjectionPolicy

config = MemoryConfig(
    operation_projection_policy=(
        OperationProjectionPolicy.COMPACT_AFTER_TERMINAL.value
    ),
)
```

`operation_projection_policy` 有两个稳定 wire value：

| Value | Behavior |
| --- | --- |
| `full_until_raw_compaction` | 默认值。完整 action/observation 一直进入 provider 历史，直到统一 raw compaction 接替该 turn；不创建 settlement。 |
| `compact_after_terminal` | turn 成功完成后，把足够大的 observation 正文替换成可回读紧凑卡片；action、已有 final 和 SQLite 原文不变。 |

这些枚举值会持久化到 turn，并参与 settled hash。已发布值不得改名、删除、复用或
改变语义；schema 升级只允许新增值。

策略与两个收益门槛都在 `begin_turn()` 时冻结。运行中改变 `MemoryConfig` 不会改变已经开始的 turn：

```python
handle = mem.begin_turn(stimuli=[...])
assert handle.operation_projection_policy == "compact_after_terminal"
```

旧数据库打开时会自动迁移 schema。迁移前已经存在的 turn 按
`full_until_raw_compaction` 处理，不会在启动时批量重写历史。

## 2. Complete lifecycle

settlement 只在成功的 `complete_turn()` 之后发生，包括使用 `append_final=False`
完成工具工作且没有新增回复的轮次。当前开放 turn 的每轮工具调用仍
完整可见，因此多轮工具链不会因为压缩而丢失中间证据。

```python
handle = mem.begin_turn(stimuli=[user_entry])

exchange = mem.record_tool_exchange(
    turn_id=handle.turn_id,
    tool_name="web_search",
    tool_call_id="call_001",
    tool_input={"query": "DeepSeek cache rules"},
    result=full_tool_result,
    source="web",
)
result_source_id = exchange["tool_result"]["source_id"]

# 本轮下一次模型请求使用这份完整历史；新增动作/结果后再次构建。
history = mem.build_context_projection(provider_profile="openai_chat")
# ...宿主将 history.payloads 发给模型，并取得 raw_model_output / assistant_speech...

# 本例的 final 是文本型 Chat Completion；保留实际返回的 content。
# 若实际消息还含 provider 扩展字段，传入宿主保存的完整消息对象。
final_message = {"role": "assistant", "content": raw_model_output}
completed = mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text=assistant_speech,
    provider_output_raw=raw_model_output,
    provider_profile="openai_chat",  # pass the profile actually used by this turn
    provider_projection=final_message,
)
assert completed.completed
```

provider-ready profile 包括 `openai_chat`、`openai_responses`、`deepseek_chat` 和
`anthropic_messages`；`canonical_user_assistant` 用于中性存储/回放。
支持范围与真实传输验收见 [provider 支持表](provider_support_matrix_v1.md)。
settlement 按 `(turn, provider_profile)` 冻结；动态模型/视觉
路由宿主应传本轮真正生成 final 的 profile，而不是只依赖全局默认值。
带新增 final 时，显式 `provider_profile` 必须同时提供 `provider_projection`，
只传 profile 会被拒绝。MemCore JSON 的投影保留原始 JSON content，
`semantic_text` 则保存解析后的 speech；其它 provider 应保留自己的实际消息形状。
提交 final 前，前面的可见 entry 必须已通过同 profile 的 `build_context_projection()`、
`build_context_surface()` 或真实请求冻结流程进入投影账本；直接跳到 final 会违反
追加顺序。应在每次模型请求前完成这一构建，而不是等到回复结束后补造历史。

协议中立宿主应优先使用 `append_action()` / `append_observation()`。如果实际发给
provider 的 action/result 形状不同于标准 adapter，在每次真实请求时使用
`record_request_projection()` 冻结真实 wire；settlement 会从该权威投影派生，
不会重新猜测 action 参数或丢掉 provider 扩展字段。完整例子见
[`operation_timeline_v1.md`](operation_timeline_v1.md)。

并行工具仍以 `correlation_id` 配对。存在未配对 action 时 `complete_turn()` 本来就
不能成功，因此也不会提前 settlement 半个工具批次。

## 3. What is compacted

settlement 只考虑 prompt-visible observation 的文本正文。以下为默认门槛：

- 正文 UTF-8 长度小于 256 bytes：`inline_full`；
- 卡片相对正文节省的字节比例不超过 50%：`inline_full`（no-expansion）；
- 满足收益门槛：`compact_reloadable`；
- 显式空正文保持原形状：`explicit_empty`。

门槛可以由宿主配置，例如仅对至少 32 KiB 的正文尝试卡片化：

```python
config = MemoryConfig(
    operation_projection_policy="compact_after_terminal",
    operation_settlement_min_utf8_bytes=32_768,
    operation_settlement_min_saved_ratio=0.5,
)
```

字节门槛必须为正整数；收益比例必须为 `0 < ratio < 1` 的有限数，当前分类器
使用的有效收益比例最高为 `0.99`。正文达到字节门槛后仍需通过收益检查，恰好
节省有效比例时仍保留全文。这些值与策略一起冻结；修改配置只影响新 turn。
UTF-8 字节收益不等于 provider token 或总费用收益。

一个典型的 settled observation：

```text
[compact_reloadable]
time: 2026-08-17 00:12
tool: web_search
call_id: call_001
status: success
source_id: <raw-tool-result-source-id>
stored_chars: 53870
stored_utf8_bytes: 64102
result_hash: sha256:<hash>
reload: open_memory(memory_id="<raw-tool-result-source-id>", view="content", detail="full")
```

`time` 行是通用可读时间（从 observation 既有 timestamp 与宿主时区派生，仅
新增字段；不复制工具入参，不要求生产者新增 summary/anchor）。未配置时区时
渲染结果与历史逐字节一致（旧卡仍可正常打开，新完成回合使用新格式）。

以下内容不会被该策略压缩：

- 当前仍开放的 turn；
- user stimulus 和 assistant final；
- action 的工具名、调用 ID 和参数；
- SQLite 中的 observation payload/semantic text；
- 不支持安全文本替换的未知 provider 或复杂非文本结果。

最后一种情况会结构化降级为 `full_fallback`，而不是猜一种近似 wire。

## 4. Settlement states and persistence

每个 `(turn, provider_profile)` 最多冻结一份 settlement：

| `settlement_status` | Meaning | History used on later requests |
| --- | --- | --- |
| `settled` | 至少一个长 observation 被卡片替换 | settled provider payloads |
| `settled_noop` | 没有可获益的替换 | 原 full payloads；不额外保存重复 settled rows |
| `full_fallback` | 生成或原子保存 settlement 失败 | 原 full payloads |

settlement 行和全部 settled provider payload 在同一个 SQLite 事务发布。重启和
`complete_turn()` 幂等重试采用 first-write-wins，不会生成第二套 hash。损坏、缺行、
projection index 不连续、profile/hash 不一致会返回结构化 `SchemaError`，不会把
半套 settled 历史当成权威输入。

assistant final 已经提交后，settlement 失败不会撤回或替换用户已经看到的回复。

## 5. Building provider history

宿主继续使用原有 API，不需要自己读取 settlement 表：

```python
projection = mem.build_context_projection(provider_profile="openai_chat")
provider_messages = list(projection.payloads)

if projection.has_compact_history:
    # At least one visible historical message came from a settled ledger.
    ...
```

`build_context_projection()`、raw compaction 的 token planning 和重启恢复共用同一份
settled ledger，因此“实际发给 provider 的历史”和“用于判断 raw compaction 的历史”
不会采用两种尺寸口径。

`has_compact_history` 是宿主提示词/观测信号，不是让宿主重新压缩 payload 的指令。

## 6. Model prompt requirement

如果启用 `compact_after_terminal`，必须在稳定 system/developer 规则中告诉最终聊天
模型旧结果可以回读。推荐文案：

```text
部分已完成轮次的历史工具结果会显示为紧凑回执，而不是完整正文。
这不代表结果丢失。回执保留原工具名、调用参数、状态与 source_id。
若当前问题依赖其中的具体内容，可以按 source_id 单个或批量打开完整结果；
若旧结果不足，可以继续调用相应工具。当前开放轮次中的工具结果始终完整可见；
不要在已有信息足够时机械回读。
```

为了 provider 前缀缓存稳定，compact 策略开启期间应始终注入完全相同的文案，不要等
第一条卡片出现后才临时修改 system prompt。如果后来切回 full 策略，但当前可见历史
仍有 settled turn，可根据 `projection.has_compact_history` 暂时保留这条规则。

该提示只说明能力和边界，不限制模型自由：final/卡片足够时直接回答；需要具体数字、
原文或旧工具正文时才回读。

## 7. Reload API

单条卡片的 `source_id` 可以直接作为 `open_memory.memory_id`：

```python
full = mem.open_memory(
    memory_id=result_source_id,
    view="content",
    detail="full",
)
```

批量恢复几个结果：

```python
batch = mem.open_memory(
    memory_ids=[source_id_a, source_id_b],
    view="content",
    detail="full",
)
```

批量契约只适用于 `card` / `content`。`view="sources"` 每棵 lineage 有自己的 cursor，
因此 v1 只接受单个 `memory_id`，也不能把 cursor 与其它 selector 混用。

给模型暴露工具时不需要增加 settlement 专用工具：

```python
tools = build_native_memory_tool_specs(tool_format="openai")
result = dispatch_native_memory_tool(
    "open_memory",
    {"memory_id": result_source_id, "view": "content", "detail": "full"},
    mem=mem,
    current=current,
)
```

回读本身也是普通工具调用：完整结果在该开放 turn 中可见；assistant final 生成后，
这次回读 observation 也可以按相同规则 settlement，形成闭环。

## 8. Metrics

`settlement_metrics()` 返回当前 conversation namespace 的安全指标，不返回正文、原始
turn ID、payload、路径或密钥：

```python
for row in mem.settlement_metrics():
    print(row["settlement_status"], row["saved_ratio"])
```

每行字段：

```text
operation_projection_policy
settlement_status
turn_id_hash
provider_profile
full_projected_tokens
settled_projected_tokens
saved_projected_tokens
saved_ratio
first_changed_projection_index
full_projection_hash
settled_projection_hash
token_count_quality
fallback_reason
settlement_min_utf8_bytes
settlement_min_saved_ratio
settlement_config_hash
```

`token_count_quality="exact"` 只表示宿主注入的 `TokenCounter` 声明自己使用精确
tokenizer；没有 counter 时为 `estimated`。不要把 UTF-8 bytes 冒充 provider billing
tokens，也不要跨不同 tokenizer 直接比较绝对 token 数。

三个 `settlement_*` 配置字段记录该轮冻结的实际门槛与配置 hash，可用于区分
不同配置的观测结果；重启时不会用当前配置覆盖历史值。

## 9. Relationship to raw compaction

terminal settlement 与 raw/episodic/semantic compaction 是两个独立阶段：

```text
open turn
  -> full action/result loop
turn completed (with or without a new assistant final)
  -> optional settled provider projection
later raw token pressure
  -> episodic summary + operation digest/retention anchors
```

settlement 不改变检索准入、source lineage、raw truth、summary 生成或 retention anchor。
`retention_anchor` 仍只能保存小型资源 ID/version/hash/cursor，不能因为 provider 历史已
变紧凑就把完整结果复制进 anchor。

## 10. Rollout checklist

1. 保持默认 `full_until_raw_compaction` 完成基线回放。
2. 为一个 namespace 启用 `compact_after_terminal`。
3. 确认当前工具循环每一轮仍收到完整 observation。
4. 确认 final 后的下一次 `build_context_projection()` 出现卡片且 action/final 不变。
5. 用卡片 `source_id` 单条和批量 `open_memory(content)` 回读完整正文。
6. 检查 `settlement_metrics()` 的 `settled/settled_noop/full_fallback` 与 token quality。
7. 观察 provider cache hit/miss；MemCore 只保证冻结后的字节稳定，不承诺供应商缓存命中。
8. 保留回滚开关：切回 `full_until_raw_compaction` 只影响新 turn，已冻结 settlement
   仍按原策略稳定读取，直到离开可见历史或进入后续 raw compaction。
