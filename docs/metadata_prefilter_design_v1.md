# Metadata Prefilter Design v1

> 历史设计文档：下文记录 schema 2 的旧 `categories/subject_scopes` 方案，
> 不是当前接入 API。schema 3 的唯一权威见
> [`memory_metadata_raw_retrieval_design_v1.md`](memory_metadata_raw_retrieval_design_v1.md)。
> 新代码不得照此文档恢复旧字段或多级自动放宽。

> 状态：本文保留 metadata flag/recursive where 的 V1 设计依据；当前运行权威已经是
> Unified Timeline Retrieval V2，详见 `unified_timeline_v2_design_v1.md` §20。
> V2 中 Namespace、conversation、time、visibility、annotation、kind、source layer、
> lineage 与 index generation 均是不可放宽的 hard filter；只有
> importance/categories/subject scopes 可有界放宽。旧 trace category 不再负责打开
> 工具、事件或材料候选。下文涉及 V1 改造过程的措辞属于历史实现记录，不是待办。

## 背景

当前 memcore 的检索链路已经把 `namespace / time_hint / exclude_source_ids / source_layers / importance_min` 下推到了 `VectorIndex.where`。其中:

- `ReadPipeline._retrieve_and_verify()` 每个放宽阶段都会构造 `where` 后调用 `index.semantic_search()` 和 `index.keyword_search()`。
- `InMemoryVectorIndex` 会先用 `_candidate_entries()` 按 `where` 和 `exclude_source_ids` 过滤候选,再做 cosine / BM25。
- `ChromaVectorIndex` 会把 `where` 翻译成 Chroma where,并把 `exclude_source_ids` 合入 `source_id: {"$nin": ...}`。

设计前的缺口是 `categories` 和 `subject_scopes`。它们曾经只以 `memory_categories_text` / `memory_subject_scopes_text` 的空格拼接文本写进 metadata:

- 优点:BM25 可以把这些标签当作关键词增强召回。
- 缺点:不适合做稳定的 OR 前置过滤,容易退化成 fused hits 后筛。

目标是把 `categories` 和 `subject_scopes` 也升级为真正的前置候选裁剪条件:模型在工具调用里传入结构化过滤参数时,相似度计算只发生在满足结构化条件的记忆集合上。

## 目标语义

过滤语义必须固定为:

```text
不同维度之间: AND 取交集
同一维度多个值: OR 取并集
```

示例:

```python
retrieve(
    "风险偏好",
    source_layers=["raw", "semantic_summary"],
    categories=["preference", "plan_goal"],
    subject_scopes=["user"],
    importance_min=0.6,
)
```

含义:

```text
(entry_type in raw/semantic_summary)
AND (category has preference OR plan_goal)
AND (subject_scope has user)
AND (memory_importance >= 0.6)
```

只有满足该交集的候选才进入语义相似度计算和 BM25 计算。不能先全量检索再后置匹配这些字段。

## 保留和新增的 index metadata

继续保留文本标签字段,用于 BM25 标签增强和兼容已有行为:

```text
memory_keywords_text
memory_subject_scopes_text
memory_categories_text
memory_mood_tags_text
semantic_tags_text
```

新增前置过滤专用布尔字段:

```text
memory_category__<safe_key> = true
memory_scope__user = true
memory_scope__assistant = true
memory_scope__other = true
```

`<safe_key>` 必须由配置枚举值稳定生成,不能直接信任任意字符串。建议实现一个小 helper:

```python
def metadata_filter_key(prefix: str, value: str) -> str:
    # ASCII 字母/数字/下划线可直接使用;其他值用 hash 后缀稳定映射。
```

默认 categories 都是 `preference / plan_goal / project_work` 这类安全键,金融领域也建议使用 `risk_profile / investment_goal / asset_preference` 这种枚举名。若用户配置中文枚举,helper 仍必须产出稳定字段名。

新增字段只存在于 index metadata,不需要改 SQLite 表结构。SQLite 仍只存规范化后的 `memory_metadata_json`。

## where 方言扩展

当前 memcore where 是扁平 dict,隐式 AND,支持字段操作符:

```python
{"memory_importance": {"$gte": 0.6}, "entry_type": {"$in": ["raw"]}}
```

为了表达 category/scope 的同维度 OR,需要把内部 where 方言扩成递归逻辑:

```python
{
    "tenant_id": "default",
    "user_id": "u1",
    "domain_id": "",
    "$and": [
        {"entry_type": {"$in": ["raw", "semantic_summary"]}},
        {"memory_importance": {"$gte": 0.6}},
        {
            "$or": [
                {"memory_category__preference": True},
                {"memory_category__plan_goal": True},
            ]
        },
        {"memory_scope__user": True},
    ],
}
```

实现要求:

- 顶层普通字段和 `$and/$or` 同时出现时,整体仍是 AND。
- `$and` / `$or` 支持递归子 where。
- 字段操作符继续支持 `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$ne`；精确时间范围使用 `$gte` 起点包含与 `$lt` 终点不包含。
- `InMemoryVectorIndex._match_where()` 改成递归匹配。
- `ChromaVectorIndex._to_chroma_where()` 改成递归翻译,并继续处理 Chroma 对多条件 `$and` 和单字段多操作符的限制。

## 当前前置放宽策略

放宽必须发生在前置 where 构造阶段。每一级都重新生成 index where,先过滤候选,
再做语义/BM25。当前 V2 阶段顺序是:

当前阶段顺序:

```text
stage 1 strict:
  hard filters + source_layers + importance_min + categories + subject_scopes

stage 2 drop importance:
  hard filters + source_layers + categories + subject_scopes

stage 3 drop categories:
  hard filters + source_layers + subject_scopes

stage 4 drop subject_scopes:
  hard filters + source_layers
```

hard filters 永不放宽:

```text
tenant_id / user_id / domain_id / conversation scope
time_hint(date_label/time_of_day/start_ts/end_ts)
retrieval visibility / annotation / kind / trust
source_layers / exclude_source_ids / lineage closure
index schema generation
```

说明:

- `importance_min` 最容易因模型估计过高导致搜空,所以优先放宽。
- `categories` 可能比 `subject_scopes` 更领域化,模型可能猜错领域类目,所以在 scope 前放宽。
- `source_layers` 表达调用方明确要求的记忆层,属于 `HardFilterPlan`,不会因候选不足而放宽。
- 如果某阶段候选数达到 `relaxation_stop_candidate_count`,停止继续放宽。

## ReadPipeline V1 实现记录

V1 的 `ReadPipeline._stage_index_where(stage)` 在改造前只下推:

```python
entry_type
memory_importance
```

V1 当时的改造目标如下；V2 在此基础上把 hard/semantic plan 正式拆开:

```python
where = self._build_where(namespace, time_hint)
where = merge_where(where, self._stage_index_where(stage))
```

其中 `_stage_index_where(stage)` 应输出:

- `entry_type: {"$in": source_layers}`
- `memory_importance: {"$gte": importance_min}`
- category OR 子句
- subject_scope OR 子句

后置精筛必须删除:

- `ReadPipeline` 不再在 fused hits 后按 category/scope/importance/source_layers 再筛一遍。
- `VectorIndex.where` 是强制契约。内置 InMemory / Chroma 必须在相似度/BM25 计算前执行 where。
- 第三方 index 后端如果忽略 where,属于后端实现错误,不由读侧用后置过滤兜底。

## Index entry builder V1 实现记录

`memcore/index/entry_builder.py` 的 `_metadata_tags(record)` 增加布尔字段:

```python
def _metadata_filter_flags(meta: dict[str, Any]) -> dict[str, bool]:
    flags = {}
    for category in meta.get("categories") or []:
        flags[metadata_filter_key("memory_category", category)] = True
    for scope in meta.get("subject_scopes") or []:
        flags[metadata_filter_key("memory_scope", scope)] = True
    return flags
```

然后并入 raw / summary / semantic 三种 entry metadata。

注意:

- `update_turn_metadata()` 已经会重建 raw index,所以 raw metadata 更新后新布尔字段会同步。
- 摘要和长期语义由压缩链路产出,新 entry builder 生效后自然带新字段。
- 老索引里的旧 entry 没有这些布尔字段,需要 reindex 后才能享受 category/scope 前置过滤。

## Reindex / 兼容策略

新增 index metadata 字段后,关系库不需要迁移,但向量索引需要补 upsert 到新字段版本。

已提供 `MemorySystem.reindex_all()` 作为升级/冷启动维护入口:

```python
stats = mem.reindex_all()
```

行为:

- 从 `SQLiteMemoryStore` 读取 raw / summary / semantic 三层记录并重新 upsert 到当前 index。
- 默认按 hard namespace(`tenant_id / user_id / domain_id`)补 upsert 全部会话,保证换会话后长期记忆仍可被 `retrieve` 搜到。
- 如只想热当前会话,传 `current_conversation_only=True`。
- `limit` 是一次性安全上限,不是分页 cursor;若未来要做超大库分批重建,需要新增 cursor/批处理接口。
- 成功的记录标为 `indexed`;失败的记录标回 `pending`,后续可交给 `reindex_pending()` 重试。
- 不会清空当前 index 里的陈旧条目;适合空内存索引冷启动、索引字段升级补 upsert。外部向量库若已污染,应先用后端管理工具清理集合或新建空 index。

`reindex_pending()` 仍只处理 `index_status='pending'` 的 outbox 自愈场景;升级索引 metadata 字段或替换空内存索引时,应使用 `reindex_all()`。

## Chroma 设计注意点

Chroma metadata 只保存 str/int/float/bool 等标量,所以布尔字段适合它。

需要扩展 `_to_chroma_where()`:

- 输入普通扁平 where 时,保持现有行为。
- 输入 `$and` / `$or` 时递归翻译。
- 多个顶层字段 + 逻辑子句时,统一包装成 `$and`。
- 单字段多个操作符仍拆开,例如 `{"timestamp": {"$gte": 1, "$lte": 2}}` 翻成两个子句。

必须补纯逻辑测试,不依赖安装 chromadb。

## InMemory 设计注意点

`InMemoryVectorIndex` 已经在相似度计算前调用 `_candidate_entries()`。改造后只要 `_match_where()` 支持递归 `$and/$or`,就能保证:

```text
先按 metadata where 取候选
再进入 numpy cosine / Python cosine / BM25
```

需要补一个 spy 或计数测试证明:

- 不满足 category/scope/importance 的 entry 没有进入 semantic cosine 计算。
- 不满足 category/scope/importance 的 entry 没有进入 BM25 doc_terms 统计。

这比只断言最终结果不包含无关记忆更重要,因为本设计的核心价值是前置裁剪计算量。

## 测试清单

必须新增或更新:

- `entry_builder` 为 raw / summary / semantic 写入 category/scope 布尔字段。
- `metadata_filter_key()` 对普通英文枚举稳定可读,对中文/特殊字符稳定 hash,无碰撞风险测试。
- `ReadPipeline._stage_index_where()` 对 `categories=["preference","plan_goal"]` 生成 OR where。
- `ReadPipeline._stage_index_where()` 对 `subject_scopes=["user","assistant"]` 生成 OR where。
- 跨维度是 AND:category 命中但 scope 不命中时不进候选。
- 同维度是 OR:category 命中任意一个即可进候选。
- `importance_min` 是前置 `$gte`,只让大于等于阈值的候选参与计算。
- 放宽阶段每一级都重新调用 index,且 where 逐级减少约束。
- Chroma `_to_chroma_where()` 支持 `$and/$or` 递归翻译。
- InMemory semantic/BM25 在候选阶段排除不匹配 metadata,不是最终结果后筛。
- `retrieve_for_turn()` 的可见三层 `exclude_source_ids` 及其上下游 lineage closure 合入前置过滤,不允许 raw/derived 互相旁路重复返回。

## 非目标

本设计不改变:

- `memory_metadata` 的 JSON 契约。
- SQLite 存储结构。
- BM25 使用 metadata text tag 增强召回的行为。
- verifier 行为。
- 可见三层和工具检索的职责划分。

本设计也不引入 router。工具是否调用仍由聊天模型判断;memcore 只负责当工具带着结构化 metadata 参数进来时,把这些参数用于真正的前置候选裁剪。

## 验收标准

完成后,下面这句话必须为真:

```text
当 retrieve 传入 source_layers/categories/subject_scopes/importance_min 时,
语义相似度和 BM25 只在满足当前放宽阶段 where 的候选集合上计算。
```

如果 strict 阶段搜不到足够候选,系统可以逐级放宽,但每一级仍然先过滤再计算,不能回到全量检索后筛。
