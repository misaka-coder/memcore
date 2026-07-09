# Trace Context Compaction Execution V1

本文档用于落地 tool/material 事件流与 memcore raw 压缩的交互规则。目标不是把附件本体或 provider 原生消息协议塞进 memcore,而是让模型在跨轮上下文里持续看见清晰的事件线索,并且在 raw 被压缩后仍能接着调用工具恢复上下文。

## 当前实现状态

截至本版文档,memcore 已落地:

- `tool_trace` / `material_trace` 默认 category、默认 count 压缩触发排除、默认普通检索排除。
- `record_tool_exchange(...)`、`record_material_reference(...)`、`record_material_cleanup(...)` 三个事件写入入口。
- 可见 raw、raw snippet、summary transcript 均使用 trace-aware 渲染,工具/材料事件不伪装成普通用户消息。
- count policy 已改为“触发排除 trace、summary batch 包含跨度内 trace”。
- raw -> summary 保存前会确定性合并跨度内 trace category/keywords/source scopes。
- 相关单测覆盖默认配置、渲染、count 压缩跨度、summary metadata 继承、默认检索排除与显式检索。

## 当前代码勘探结论

### Raw 压缩链路

- 入口:`MemorySystem.compact_due_sync/background()` -> `Compaction.run_due()`。
- raw 到 summary 的主逻辑在 `memcore/compaction.py`:
  - `_summarize_raw()` 读取 `store.get_unsummarized_messages(...)`。
  - `_select_raw_summary_batch(...)` 决定是否压缩、压缩哪批 raw。
  - `_render_transcript(batch)` 把 batch 渲染给 summary LLM。
  - summary 成功后写 `summaries`,再 `mark_messages_summarized(source_ids, summary_id)`。
- 旧 count policy 的行为:
  - `raw_compaction_excluded_categories` 已用于触发计数排除。
  - 但 batch 当前只取 eligible 消息,也就是说被排除的 trace 不进入 summary batch。
- 新 count policy 的行为:
  - `raw_compaction_excluded_categories` 只影响触发计数和普通消息边界。
  - summary batch 会包含两个普通消息边界之间的完整 raw 跨度,夹在中间的 trace 会一起被压缩。
- 当前 token policy 的行为:
  - 按所有未摘要 raw 的 content token 总量触发。
  - 批次边界要求最后一条 raw 的 `role == "assistant"`。
  - 中间的 tool/material trace 会参与 token 压力,这是容量兜底需要的。

### 检索链路

- `memcore/retrieval.py` 已支持 `retrieval_default_excluded_categories`。
- 默认检索会对配置里的 excluded categories 下推 `category_filter_key(category): {"$ne": True}`。
- 如果请求显式传入某个默认排除 category,例如 `categories=["tool_trace"]`,默认排除会关闭,可以检索该类轨迹。
- 这套机制适合直接复用到 `material_trace`。

### 渲染链路

- `memcore/rendering.py` 负责 raw/summary/semantic/timeline 的文本投影。
- tool trace 当前已经走线性事件投影:

```text
[20:31 | 晚上] assistant.tool_call web_search call_001
input:
{"query": "北京天气"}

[20:31 | 晚上] tool.web_search call_001
source: web_search
output:
北京今天多云,气温 27~34°C。
```

- 可见 raw 和 raw snippet 已对 `tool_trace` 做特殊渲染:事件头独立一行,正文接在下面。
- `_render_transcript()` 已复用 trace-aware raw 渲染,summary prompt 中的工具/材料事件不会退回成普通聊天行。

### 已有未提交变更状态

- `tool_trace` 已经作为默认 category、默认压缩触发排除、默认检索排除存在。
- `material_trace` 已加入 `DEFAULT_CATEGORIES` 和默认排除配置,并已有 facade、渲染、测试和文档闭环。

## 总体设计

### 核心规则

1. **触发压缩时排除 trace**

   count policy 下,`tool_trace` / `material_trace` 不计入 `raw_trigger_count`。普通对话达到触发阈值才触发压缩。

2. **真正压缩时包含跨度内 trace**

   触发和 batch 边界按普通消息决定,但 summary batch 应取从第 1 条 eligible 到第 N 条 eligible 之间的完整 raw 跨度。中间夹着的工具调用、工具结果、附件上传、材料清理事件都进入 transcript。

3. **压缩后保留可接续锚点**

   summary 必须能让模型继续知道该调用什么工具。被压缩的 trace 里出现的下列锚点要进入 summary 的事实或 metadata:

   - `tool_call_id`
   - `tool_name`
   - `file_id`
   - `filename`
   - `kind`
   - `mime`
   - `file_status`
   - `derived_status`
   - `cleanup reason`
   - `source`

4. **文件本体和派生结果不由 memcore 保命**

   memcore 记录事件和锚点;宿主管 `file_store` 和 `derived_store`:

```text
file_store: 原始图片/PDF/文件,容量优先清理
derived_store: OCR/vision 描述/chunks/embedding/页摘要,可比原文件留久
memcore raw/summary: file_id、文件名、状态、清理事件、工具调用/结果投影
```

5. **工具读取永远返回当前 status**

   历史 raw 或 summary 里有 `file_id` 不代表原文件还在。工具必须查宿主 material index:

```text
ready: 原文件和派生结果可用
original_expired: 原文件已清,但 derived_store 仍可用
derived_ready: 可返回旧 OCR/描述/chunks,但必须标明不是重新看原图
expired: 原文件和派生结果都不可用,只能返回上传事件元数据
```

## 事件格式

### 工具事件

继续使用现有 `record_tool_exchange(...)`:

```text
[20:31 | 晚上] assistant.tool_call web_search call_001
input:
{"query": "北京天气"}

[20:31 | 晚上] tool.web_search call_001
source: web_search
output:
北京今天多云,气温 27~34°C。
```

metadata:

```json
{
  "categories": ["tool_trace"],
  "subject_scopes": ["assistant"],
  "keywords": ["web_search", "北京天气"],
  "importance": 0.2,
  "confidence": 1.0
}
```

### 材料引用事件

新增 facade 推荐形态:

```python
mem.record_material_reference(
    file_id="file_img_001",
    kind="image",
    filename="photo.jpg",
    mime_type="image/jpeg",
    file_status="ready",
    derived_status="ready",
    timestamp=now_ts,
    keywords=["photo.jpg", "物理题"],
)
```

渲染:

```text
[20:10 | 晚上] user.attachment image file_img_001
source: attachment
filename: photo.jpg
mime: image/jpeg
file_status: ready
derived_status: ready
```

metadata:

```json
{
  "categories": ["material_trace"],
  "subject_scopes": ["user"],
  "keywords": ["file_img_001", "photo.jpg", "image", "物理题"],
  "importance": 0.25,
  "confidence": 1.0
}
```

### 材料清理事件

容量优先清理由宿主执行,但清理结果要追加事件:

```python
mem.record_material_cleanup(
    file_id="file_img_001",
    kind="image",
    filename="photo.jpg",
    file_status="original_expired",
    derived_status="ready",
    reason="local_storage_quota",
    timestamp=cleanup_ts,
)
```

渲染:

```text
[23:50 | 晚上] system.material_cleanup image file_img_001
source: attachment_cleanup
filename: photo.jpg
file_status: original_expired
derived_status: ready
reason: local_storage_quota
```

如果 derived_store 也被清掉:

```text
file_status: expired
derived_status: expired
```

### 派生结果事件

OCR、视觉描述、文件解析、文件检索结果不放进 material reference,而是继续用 `record_tool_exchange(...)`:

```text
[20:11 | 晚上] assistant.tool_call vision_describe call_003
input:
{"file_id": "file_img_001"}

[20:11 | 晚上] tool.vision_describe call_003
source: attachment.image
output:
图片是一道物理题,左侧有竖直导线,右侧有矩形线圈...
```

这条事件属于 `tool_trace`,也可以由宿主把关键词带上 `file_img_001/photo.jpg`。真正完整描述仍然应在 derived_store。

## 压缩算法变更

### Count policy:触发排除,跨度包含

当前逻辑:

```python
eligible = [m for m in msgs if not excluded(m)]
return eligible[:summary_batch_size] if len(eligible) >= raw_trigger_count else []
```

目标逻辑:

```python
eligible_indexes = [i for i, m in enumerate(msgs) if not excluded(m)]
if len(eligible_indexes) < raw_trigger_count:
    return []

start = eligible_indexes[0]
end = eligible_indexes[summary_batch_size - 1]
return msgs[start : end + 1]
```

含义:

- 触发数只数普通消息。
- 压缩 batch 覆盖最老 `summary_batch_size` 条普通消息。
- 跨度内夹着的 `tool_trace/material_trace` 一起进入 summary。
- 跨度前只有 trace、还没有普通消息时,这些 trace 暂不压缩,等待后续普通消息形成可叙述上下文。

### Token policy:容量兜底

token policy 维持“所有 raw 都计入 token 压力”。原因:

- 工具结果、OCR、文件片段可能很长,不应绕过容量控制。
- `raw_token_boundary_role` 目前只接受 `assistant`,所以中间 trace 可以被压缩,但最后一条如果是 trace 不会立刻压缩,避免半轮工具事件被截断。

后续如果工具事件长期堆积且没有 assistant 结尾,可再引入完整 turn 边界,本轮不做。

## Summary 生成要求

### Transcript 渲染

summary prompt 必须保留 trace 结构。建议新增或复用公共渲染函数,不要在 `_render_transcript()` 里手写 `speaker: content`。

目标 transcript:

```text
[2026-07-09 周四 20:10 | 晚上] user: 伙伴,这题怎么做
[2026-07-09 周四 20:10 | 晚上] user.attachment image file_img_001
source: attachment
filename: photo.jpg
mime: image/jpeg
file_status: ready
derived_status: ready
[2026-07-09 周四 20:11 | 晚上] assistant.tool_call vision_describe call_003
input:
{"file_id": "file_img_001"}
[2026-07-09 周四 20:11 | 晚上] tool.vision_describe call_003
source: attachment.image
output:
图片是一道物理题...
```

### 提示词补强

`MULTI_ACTOR_MEMORY_RULES` 或新规则中补充:

- `tool.*` 不是用户原话。
- `user.attachment` 是用户上传材料事件,不是材料正文。
- `system.material_cleanup` 是本地存储状态变化。
- 摘要中必须保留 file_id/tool_call_id 等可接续锚点。
- 不要把工具结果、OCR、旧派生描述写成用户亲口说过的事实。

### Metadata 确定性合并

不要完全依赖 summary LLM 记得打标签。保存 summary 前确定性合并 batch 里的 trace metadata:

```python
trace_categories = union(batch.memory_metadata.categories & {"tool_trace", "material_trace"})
trace_keywords = union(batch.memory_metadata.keywords)[:4]
summary_metadata.categories |= trace_categories
summary_metadata.keywords = merge(summary_keywords, trace_keywords)[:4]
summary_metadata.confidence = max(summary_confidence, 0.8 if trace_categories else summary_confidence)
```

注意:

- 如果 batch 里有普通长期事实和 trace,summary 可以同时带普通 category 与 trace category。
- `material_trace` 默认检索排除,所以普通检索不会被材料事件污染。
- 用户显式问“之前那张图/那个文件”时,模型应传 `categories=["material_trace"]` 或调用 timeline。

## 压缩后如何接续

### 原文件仍在

1. 模型从 raw 或 summary 看见 `file_id=file_img_001`。
2. 调 `describe_image(file_img_001)` 或 `load_attachment(file_img_001)`。
3. 工具从 file_store/derived_store 返回最新可用内容。

### 原文件已清,derived_store 仍在

工具返回:

```text
status: original_expired
derived_status: ready
source: derived_store
output:
以下是历史 OCR/视觉描述,不是重新读取原图...
```

模型可以基于旧解析回答,但必须避免说“我刚看了原图”。

### 原文件和派生结果都已清

工具返回:

```text
status: expired
derived_status: expired
message: 原始文件和解析结果已按本地容量策略清理。
available_metadata:
file_id: file_img_001
filename: photo.jpg
uploaded_at: ...
```

模型应说明无法再查看内容,只能根据历史对话/摘要里的残留线索回答。

## 实施步骤

1. **默认枚举和排除项**
   - `DEFAULT_CATEGORIES` 加 `material_trace`。
   - `raw_compaction_excluded_categories` 默认包含 `tool_trace/material_trace`。
   - `retrieval_default_excluded_categories` 默认包含 `tool_trace/material_trace`。

2. **材料事件 facade**
   - 新增 `MemorySystem.record_material_reference(...)`。
   - 新增 `MemorySystem.record_material_cleanup(...)`。
   - 只写 lightweight event,不接收二进制和长解析正文。

3. **材料事件渲染**
   - `role` 采用:
     - `user.attachment <kind> <file_id>`
     - `system.material_cleanup <kind> <file_id>`
   - content 采用 `source/filename/mime/file_status/derived_status/reason` 多行块。
   - `material_trace` 与 `tool_trace` 一样使用独立事件头,不渲染成 `speaker: content`。

4. **Count batch 选择改为跨度包含**
   - 触发和 eligible 边界排除 trace。
   - 返回 batch 时包含两个 eligible 边界之间的所有消息。

5. **Summary transcript 复用 trace-aware 渲染**
   - 避免工具/材料事件在压缩提示词里退化成普通聊天行。

6. **Summary metadata 确定性合并**
   - 从 batch 原 metadata 继承 `tool_trace/material_trace`。
   - 合并 file/tool 关键词。

7. **Prompt 文档补充**
   - 更新 `docs/model_prompt_playbook_v1.md`。
   - 更新 `AGENTS.md` / `README.md` / `docs/usage_flow_v1.md`。

8. **测试**
   - 默认 config 含 `material_trace`。
   - `material_trace` 默认不计入 count 触发。
   - count batch 包含跨度内 tool/material trace。
   - summary source_ids 包含跨度内 trace。
   - summary metadata 继承 trace category/关键词。
   - 默认 retrieve 不搜 material trace。
   - 显式 `categories=["material_trace"]` 可搜 material trace 或其 summary。
   - `record_material_cleanup` 渲染 status 正确。

## 不在本轮做

- 不实现 file_store / derived_store。
- 不实现 OCR、vision、PDF chunk、embedding。
- 不实现真实容量清理器。
- 不把 memcore prompt 输出改成 provider message array。
- 不保证原文件永久可用。

这些属于宿主 Akane 的 attachment/file 子系统。memcore 只提供事件流、压缩接续和检索锚点。
