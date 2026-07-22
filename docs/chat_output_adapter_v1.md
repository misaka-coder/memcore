# Chat Output Adapter V1

## 目标

`memcore` 的核心职责是记忆:写入、压缩、检索、时间锚点、隔离与 metadata 契约。

真实聊天产品还需要一层对接能力:模型输出可能是标准 JSON、普通文本、或带少量自定义字段的 JSON。`Chat Output Adapter` 负责把这些输出安全转换成可消费的结果,并尽量在流式阶段提前吐出可展示/可播放的 `speech`。

这层是可选接入层,不是记忆压缩核心。

## 非目标

- 不强迫所有接入方使用 JSON。
- 不替接入方解析完全自定义的业务 JSON。
- 不把解析失败的整段 JSON 当成原始对话写入记忆。
- 不替客户端决定气泡、TTS、QQ、Web 等具体投递策略。
- 不额外调用轻量 LLM 给每轮 raw turn 打标签;优先使用聊天模型本人输出的 `memory_metadata`。

## 输出模式

### plain

模型输出普通文本。

adapter 尽力把全文视为 `speech`,并按句末标点/换行生成可选分段事件。`memory_metadata` 为空。

```json
{
  "status": "plain_text",
  "speech": "普通回复文本。",
  "memory_metadata": {}
}
```

### memcore_json

接入方启用 memcore 标准输出契约。模型必须输出一个 JSON 对象,最小字段为:

```json
{
  "speech": "给用户看的回复",
  "memory_metadata": {
    "keywords": [],
    "subject_scopes": [],
    "categories": [],
    "mood_tags": [],
    "importance": 0.0,
    "confidence": 0.0
  }
}
```

规则:

- `speech` 必填,是最终给用户看的文本。
- `memory_metadata` 可缺省;缺省时归一化为空 metadata。
- 不内置 `speech_segments` 字段。分段由 adapter 从 `speech` 解析。
- 可允许 presentation 字段,如 `emotion`、`reply_medium`;这些字段不进入记忆核心契约。
- 未知字段保留在 `extra` 或忽略,不直接写入 raw memory。

### custom_json

接入方使用自己的 JSON 模板。

adapter 只做保守解析:

- 如果存在 `speech`,提取为回复文本。
- 如果存在 `memory_metadata`,按 memcore 契约归一化。
- 如果字段名不匹配,返回结构化降级,由接入方自己处理。

不提供猜测式深度解析,避免误把业务 JSON 存成自然语言记忆。

## Prompt 片段

当启用 `memcore_json` 时,memcore 可提供一段可拼接到聊天模型 system prompt 的契约:

```text
请只输出一个合法 JSON 对象,不要输出代码块或解释。
字段固定为 speech, memory_metadata。
speech 是给用户看的最终回复,必须是字符串。
memory_metadata 用于本轮用户原始消息的记忆检索标注,字段为 keywords, subject_scopes, categories, mood_tags, importance, confidence。
keywords 最多 4 个可复用检索标签,按用户未来正常聊天里可能命中的问法选词;例如可乐可补饮料/偏好,但不要机械补太宽泛的上位词。不要写整句或短句。
subject_scopes 标注本轮原始消息涉及的事实主体,只能从 user/assistant/other 中选择;群聊中不要把别人的事实归到 user。
categories 必须从当前配置枚举中选择;mood_tags 只在启用情感温度时填写。
importance/confidence 必须是 0.0 到 1.0 的数字。
不要把 memory_metadata 当作给用户看的内容。
工具调用阶段不适用本 JSON 契约;如果需要调用工具,请正常使用宿主项目的工具调用机制。
只有在所有工具调用完成、准备给用户最终回复时,才按本契约只输出一个合法 JSON 对象。
把可用工具当作你的能力和结构化信息通道,不是摆设。凡是答案依赖当前 prompt 没有明确给出的旧记忆、精确时间线、人物归因、偏好、关系、承诺或平台事件时,请主动调用合适工具;一次结果不够时可以继续补查。
如果用户提到“昨天/上周/上周二/最近”等相对时间,请结合 prompt 中的日期与星期锚点理解;需要精确日期范围时优先调用 read_timeline,需要模糊事实时调用 retrieve。
如果用户追问图片、附件或 PDF 内容,先找到 material_trace 的 file_id,再调用宿主原生 load_material 工具读取当前可用内容或清理状态。
多方/群聊场景请保留谁说的、谁的偏好、谁的计划。若有 actor/昵称/稳定ID 信息,稳定ID 相同才视为同一人。回答“谁说/谁戳/谁答应/谁负责”时必须依据可见原文或工具结果,没有明确记录就不要猜。
若当前可见记忆没有明确证据,工具仍无证据时说没有看到明确记录,不要为了显得记得而编造。
```

如果启用分段回复,追加:

```text
请把 speech 写成自然短句。每个完整句子请用 。！？.!? 或换行结尾,方便系统按句分段展示或播放。
不要为了分段把同一句话硬拆碎。
```

### 工具调用边界

Chat Output Adapter 只处理模型的最终回复输出,不接管宿主项目的工具调用循环。

推荐接入方式:

1. 需要工具时,模型照常走宿主项目的原生 tool calling / tool result 流程。
2. 工具结果回到模型后,由模型生成最终给用户看的回复。
3. 只有这一步最终回复使用 `memcore_json` 契约。
4. `speech` 用于展示/播放;`memory_metadata` 回写到本轮用户 raw turn。

不要把工具调用包装进 `speech` 或 `memory_metadata`,也不要要求中间工具调用步骤输出 memcore JSON。
尽量保留工具调用的结构化边界,例如 tool id、tool name、arguments、result。不要把工具结果伪装成普通用户文本;legacy 文本 followup 只应作为兼容方案。

## 流式事件

标准 JSON 模式下,adapter 在流式 token 中提前捕捉 `speech` 字段的字符串内容。

事件建议:

```json
{"type":"speech_chunk","text":"增量文本"}
{"type":"speech_segment","index":0,"text":"完整一句话。"}
{"type":"metadata_ready","memory_metadata":{}}
{"type":"final","payload":{}}
```

普通文本模式下,也可以直接对原始 token 流做 `speech_chunk` 和 `speech_segment`。

## 分段规则

分段器只负责给客户端一个可用建议,不负责最终投递。

基础规则:

- 中文句末:`。！？`
- 英文句末:`.!?`
- 换行是强分段边界。
- 连续句末标点归为同一句结尾,例如 `哈啊？！`、`Really?!`、`!!!`。
- 句末后的右引号/右括号跟随当前句,例如 `“好吧。”`。
- 太短片段可并入前一句或等待更多内容,避免 `！` 自成气泡。
- 超长片段可在逗号、分号、空格附近软切,但不得破坏 JSON 字符串解析。

英文句号需要保守:

- 不应在小数中切分:`3.5%`
- 不应在常见缩写中切分:`e.g.`、`i.e.`、`U.S.`
- 不应在域名/文件名中切分:`example.com`、`report.pdf`
- 流式阶段遇到英文 `.` 时,最好等到下一个字符是空格、换行、引号、右括号或字符串结束后再确认边界。

## 解析结果

建议统一返回:

```python
ChatOutputParseResult(
    status="parsed" | "plain_text" | "output_unparsed" | "invalid_contract",
    speech="",
    memory_metadata={},
    presentation={},
    extra={},
    reason="",
)
```

状态含义:

- `parsed`:标准 JSON 或可识别 JSON 解析成功。
- `plain_text`:非 JSON 文本,已作为普通回复处理。
- `output_unparsed`:看起来像 JSON,但无法解析或字段不匹配。
- `invalid_contract`:启用了 `memcore_json`,但缺少必填 `speech` 或类型错误。

## 记忆写入流程

推荐流程:

1. `record_user_turn(user_text)` 先安全落 raw。
2. 接入方调用聊天模型,可拼接 memcore 的输出契约 prompt。
3. adapter 解析模型输出,得到 `speech` 与 `memory_metadata`。
4. 用 `update_turn_metadata(source_id, memory_metadata)` 回写用户 raw metadata 并重建 raw 索引。
5. 用 `record_assistant_turn(speech, memory_metadata=assistant_timeline_metadata)` 记录助手可见回复。

这样能保留 Akane 的优势:聊天模型本人决定这一轮该怎么记,memcore 负责归一化、校验、回写和索引。

## 失败纪律

- JSON 解析失败时,不把整段 JSON 存进助手自然语言记忆。
- `memory_metadata` 缺失时,raw 照常入库,metadata 为空。
- metadata 非法枚举被丢弃,数值 clamp 到 0..1。
- 开启 `memcore_json` 后缺少 `speech`,返回 `invalid_contract`。
- 流式分段只是体验优化;最终入库以完整 `speech` 和归一化 metadata 为准。

## 实现切片

1. `chat_output.schema`:配置、结果对象、状态枚举。
2. `chat_output.prompts`:标准 JSON 输出契约 prompt。
3. `chat_output.parser`:JSON/plain text 解析 + metadata 归一化。
4. `chat_output.segmenter`:中英文句末标点分段。
5. `chat_output.streaming`:流式 `speech` 捕捉与 segment 事件。
6. `MemorySystem.update_turn_metadata`:回写 raw metadata 并重建 raw 索引。
7. 测试:标准 JSON、plain text、invalid JSON、缺 speech、连续标点、小数/缩写/域名、metadata 枚举清洗、回写索引。

## 当前代码地图

这部分记录当前实现的真实接入位置,用于让 metadata 回写与后续 raw/summary 压缩
保持同一 source lineage,不是未来占位设计。

### raw 写入

`MemorySystem.record_user_turn()` / `record_assistant_turn()` 都走 `MemorySystem._record()`。

当前 `_record()` 行为:

- 生成时间锚点:`timestamp_to_date_label()`、`infer_time_of_day()`。
- 调 `coerce_memory_metadata(..., categories=self.config.categories, enable_flavor=self.config.enable_flavor)` 清洗传入 metadata。
- 调 `store.add_message(...)` 写 SQLite,初始 `index_status=pending`。
- 调 `index.upsert([build_raw_entry(rec)])`。
- upsert 成功后 `store.set_index_status(source_id, "indexed")`,并同步修改返回值 `rec["index_status"]="indexed"`。
- upsert 失败时不抛错,返回 `index_status="pending"` 交给 `reindex_pending()`。

这说明 Chat Output Adapter 不需要改变第一步 raw 落库。推荐流程仍然是先 `record_user_turn(user_text)`,等聊天模型输出后再回写 metadata。

### raw metadata 入索引

`memcore/index/entry_builder.py` 已经支持 raw metadata:

- `build_raw_entry(record)` 的向量文本是 `record["content"]`。
- `_metadata_tags(record)` 会把 `memory_metadata` 转成:
  - `memory_keywords_text`
  - `memory_subject_scopes_text`
  - `memory_categories_text`
  - `memory_mood_tags_text`
  - `memory_importance`
- `InMemoryVectorIndex` / `ChromaVectorIndex` 的关键词侧会把这些 tag 文本拼进 BM25 文本。

所以回写 raw metadata 后,只要重新 `build_raw_entry(record)` 并 upsert,检索就能吃到新标签。

### metadata 契约

`memcore/schema.py` 已有稳定契约:

- `MemoryMetadata`: `keywords`, `subject_scopes`, `categories`, `mood_tags`, `importance`, `confidence`。
- `coerce_memory_metadata()`:
  - keywords 去重截断到 4。
  - subject_scopes 只收 `user/assistant/other`。
  - categories 只收 `MemoryConfig.categories`。
  - `enable_flavor=False` 时清空 `mood_tags`。
  - importance/confidence clamp 到 0..1。

Chat Output Parser 应复用这个函数,不要再写第二套校验。

### store 回写接口

`MemoryStore` 已新增更新 raw `memory_metadata` 的接口。

```python
def update_message_memory_metadata(
    self,
    *,
    namespace: Namespace,
    source_id: str,
    memory_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """更新 raw message 的 memory_metadata,返回更新后的 raw record。找不到或不是 raw 时返回 None。"""
```

`SQLiteMemoryStore` 实现:

- 只更新 `messages.memory_metadata_json`,不要更新 summaries / semantic_summaries。
- 必须用当前 `namespace` 校验 owner;同 `source_id` 跨 user / conversation / actor 回写应抛 `NamespaceError`,不能污染别人的 raw metadata。
- 更新 metadata 时同步把该 raw message 的 `index_status` 置为 `pending`,因为旧 raw index 已过期。
- 在同一个 `_lock` + SQLite transaction 中完成。
- 更新后重新 SELECT 该 message 并 `_row_to_record(..., "messages")` 返回。
- 找不到时返回 None,不要假成功。

后续如果第三方实现了 `MemoryStore`,也必须显式实现这个方法;这是接口层清晰边界。

### MemorySystem 回写门面

已新增:

```python
def update_turn_metadata(self, source_id: str, memory_metadata: dict[str, Any]) -> dict[str, Any]:
    ...
```

建议返回结构:

```python
{
    "ok": True,
    "status": "updated" | "not_found" | "pending",
    "source_id": "...",
    "memory_metadata": {},
    "index_status": "indexed" | "pending",
    "reason": ""
}
```

行为:

- 先用 `coerce_memory_metadata(...).to_dict()` 清洗。
- 调 `store.update_message_memory_metadata(...)`。
- 如果返回 None,返回 `ok=False,status="not_found"`。
- 如果更新成功,用 `build_raw_entry(updated_record)` 重新 upsert。
- upsert 成功: `set_index_status(source_id, "indexed")`,返回 `index_status="indexed"`。
- upsert 失败: `set_index_status(source_id, "pending")`,返回 `ok=False,status="pending"` 和 reason。
- 不要修改 `content`、`timestamp`、`date_label`、`seq_no`、`namespace`。

这个方法解决 Akane 的关键行为:聊天模型 final JSON 里的 `memory_metadata` 可回写到本轮用户 raw message。

### 公共导出

新增包建议:

```text
memcore/chat_output/
  __init__.py
  schema.py
  prompts.py
  parser.py
  segmenter.py
  streaming.py
```

`memcore/__init__.py` 可导出:

- `ChatOutputConfig`
- `ChatOutputParseResult`
- `ChatOutputMode`
- `parse_chat_output`
- `segment_speech`
- `StreamingSpeechParser`
- `build_chat_output_contract_prompt`

保持导出名少而稳定,内部 helper 不导出。

## 分步实现细案

### Step 1: segmenter

先做纯函数,不碰 JSON、不碰 store。

建议 API:

```python
def segment_speech(
    text: Any,
    *,
    min_chars: int = 2,
    max_chars: int = 180,
    max_segments: int | None = None,
) -> list[str]:
    ...
```

测试重点:

- `哈啊？！真的吗。` -> `["哈啊？！", "真的吗。"]`
- `Really?! I see.` -> `["Really?!", "I see."]`
- `收益率是 3.5%。` 不在 `3.` 处切。
- `U.S. market is open. OK.` 不在 `U.S.` 内部切。
- `example.com is down. Fixed.` 不在域名内部切。
- 换行分段。
- 太短孤立标点不自成段。

### Step 2: parser

建议 API:

```python
def parse_chat_output(
    output: Any,
    *,
    mode: ChatOutputMode = "auto",
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    enable_flavor: bool = False,
) -> ChatOutputParseResult:
    ...
```

模式建议:

- `"plain"`:强制把输入当普通文本。
- `"memcore_json"`:必须 JSON object + `speech` string。
- `"custom_json"`:保守抽取 `speech` / `memory_metadata`。
- `"auto"`:能 JSON object 就按 custom_json;否则 plain_text。

解析策略:

- `dict` 输入直接用。
- `str` 输入先 strip;如果看起来是 JSON object 再 `json.loads`。
- 非 JSON 文本返回 `plain_text`。
- JSON 解析失败:
  - `memcore_json` -> `output_unparsed` 或 `invalid_contract`,不回退成 speech。
  - `auto/custom_json` -> `output_unparsed`,由接入方决定;不要把 JSON 字符串当 speech。
- `memory_metadata` 一律过 `coerce_memory_metadata()`。
- `presentation` 暂收 `emotion`, `reply_medium` 等不入库字段。
- `extra` 可收未知字段,但文档强调不要写入 raw memory。

### Step 3: prompts

提供:

```python
def build_chat_output_contract_prompt(
    *,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    enable_flavor: bool = False,
    enable_sentence_segments: bool = False,
) -> str:
    ...
```

要求:

- 写清 `speech` 必填。
- 写清 `memory_metadata` 不给用户看。
- categories 从当前配置枚举选择。
- `enable_flavor=False` 时不提 `mood_tags` 或说明必须为空;为减少混乱,建议不提 mood。
- `enable_sentence_segments=True` 时追加句末标点规则。

### Step 4: streaming

已新增 `memcore.chat_output.streaming.StreamingSpeechParser`,并保持和 parser/segmenter 解耦。

建议类:

```python
class StreamingSpeechParser:
    def feed(self, chunk: Any) -> list[dict[str, Any]]: ...
    def finish(self) -> list[dict[str, Any]]: ...
```

职责:

- `memcore_json` 模式下只捕捉顶层 `"speech"` 字符串。
- 普通文本模式下直接把 chunk 当 speech。
- 发 `speech_chunk`。
- 根据 `segment_speech` 的增量逻辑发 `speech_segment`。
- 连续标点簇一起发,避免 `哈啊？！` 拆成 `哈啊？` 和 `！`。
- `finish()` 会返回最终 `parse_chat_output()` 结果;最终结果仍是入库依据。

实现时注意 JSON string escape,不要把 `\"` 提前当字符串结束。

### Step 5: MemorySystem.update_turn_metadata

已新增 `MemorySystem.update_turn_metadata()`,用于把聊天模型 final JSON 里的 `memory_metadata`
回写到本轮用户 raw message,并重建 raw index。

已同时补:

- `MemoryStore.update_message_memory_metadata` 抽象方法。
- `SQLiteMemoryStore.update_message_memory_metadata` 实现。
- `MemorySystem.update_turn_metadata` 门面。

测试重点:

- 回写后 `store.get_record_by_source_id(source_id)["memory_metadata"]` 更新。
- 回写后 raw index 重新 upsert,关键词检索能命中新 metadata。
- 非法 metadata 被清洗。
- index upsert 失败时返回 `pending`,store 里 `index_status=pending`。
- 不存在 source_id 返回 `not_found`。

## 不要做的事

- 不要给每轮 raw turn 默认再调一个 LLM 标注器。
- 不要把 `speech_segments` 加回标准 JSON 字段。
- 不要把解析失败的 JSON 存成助手 reply。
- 不要为了 custom_json 猜测业务字段含义。
- 不要让 Chat Output Adapter 直接调用聊天模型;它只提供 prompt、解析与流式辅助。
