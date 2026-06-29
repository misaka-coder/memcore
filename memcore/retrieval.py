"""读侧:可见三层 + 显式混合检索工具(语义+关键词+RRF+五级放宽)→ verifier。

聊天模型自己决定是否调用 retrieve/read_timeline。memcore 不再内置前置 router,
避免每轮额外 LLM 判定带来的成本、延迟和误判。
精度过滤可逐级放宽(软过滤),硬隔离(namespace)和时间走 index where(不放宽)。
"""

from __future__ import annotations

import json
from typing import Any

from .config import MemoryConfig
from .index.base import VectorIndex
from .index.metadata_filters import category_filter_key, subject_scope_filter_key
from .index.rrf import fuse_with_rrf
from .llm.base import LLMClient, LLMRequest, ResponseFormat, TaskType
from .namespace import Namespace
from .prompts import build_verifier_prompts
from .rendering import render_raw_snippet, render_semantic_snippet, render_summary_snippet
from .store.base import MemoryStore
from .text_utils import normalize_text

_QUESTION_MARKERS = ("?", "?", "吗", "什么", "怎么", "为什么", "谁", "哪", "几", "多少")


def parse_ndjson(text: Any) -> list[dict[str, Any]]:
    """解析 NDJSON;若已是事件列表(stub 直接给)也接受。"""
    if isinstance(text, list):
        return [e for e in text if isinstance(e, dict)]
    events: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _filter_values(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _flag_or_clause(keys: Any) -> dict[str, Any]:
    unique: list[str] = []
    seen: set[str] = set()
    for key in keys:
        text = str(key or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    if not unique:
        return {}
    if len(unique) == 1:
        return {unique[0]: True}
    return {"$or": [{key: True} for key in unique]}


class ReadPipeline:
    def __init__(
        self,
        *,
        store: MemoryStore,
        index: VectorIndex,
        llm: LLMClient,
        config: MemoryConfig,
        timezone: str,
    ) -> None:
        self.store = store
        self.index = index
        self.llm = llm
        self.config = config
        self.timezone = timezone

    # --- 对外:拼最终上下文 ---

    def build_context(self, *, namespace: Namespace, now_ts: int) -> dict[str, Any]:
        cross = self.config.visible_memory_scope == "user"
        raw = self.store.get_unsummarized_messages(namespace=namespace)
        episodic = self.store.get_visible_episodic_summaries(
            namespace=namespace, limit=self.config.episodic_visible_max, cross_conversation=cross
        )
        semantic = self._visible_semantic(namespace=namespace, cross=cross, now_ts=now_ts)

        return {
            "raw": raw,
            "episodic": episodic,
            "semantic": semantic,
        }

    def visible_source_ids(self, *, namespace: Namespace, now_ts: int) -> set[str]:
        """返回当前 prompt 可见三层的索引 ID,供工具检索在候选阶段去重。"""
        context = self.build_context(namespace=namespace, now_ts=now_ts)
        ids: set[str] = set()
        ids.update(str(row.get("source_id")) for row in context["raw"] if row.get("source_id"))
        ids.update(str(row.get("summary_id")) for row in context["episodic"] if row.get("summary_id"))
        ids.update(str(row.get("semantic_id")) for row in context["semantic"] if row.get("semantic_id"))
        return ids

    def _visible_semantic(self, *, namespace: Namespace, cross: bool, now_ts: int) -> list[dict[str, Any]]:
        limit = self.config.semantic_visible_limit
        if not self.config.enable_importance_decay:
            return self.store.get_recent_semantic_summaries(namespace=namespace, limit=limit, cross_conversation=cross)
        # 衰减模式:对**全部**长期记忆按"随时间衰减的重要度"重排后取前 N。
        # 不按 recency 预截断候选——否则旧但衰减后仍高价值的记忆会被静默排除。
        from .decay import decayed_importance

        pool = self.store.get_recent_semantic_summaries(namespace=namespace, limit=None, cross_conversation=cross)
        hl = self.config.importance_half_life_days

        def score(record: dict[str, Any]) -> float:
            anchor = int(record.get("last_reinforced_ts") or record.get("timestamp") or 0)
            age = max(0, int(now_ts) - anchor)
            return decayed_importance(record.get("importance", 0.0), age_seconds=age, half_life_days=hl)

        return sorted(pool, key=score, reverse=True)[:limit]

    def retrieve(
        self,
        *,
        namespace: Namespace,
        query: str,
        keywords: list[str] | None = None,
        time_hint: dict[str, Any] | None = None,
        source_layers: list[str] | None = None,
        subject_scopes: list[str] | None = None,
        categories: list[str] | None = None,
        importance_min: float | None = None,
        exclude_source_ids: list[str] | None = None,
    ) -> list[str]:
        """显式检索(工具式),带精度过滤参数。"""
        return self._retrieve_and_verify(
            namespace=namespace,
            query=query,
            keywords=keywords or [],
            time_hint=time_hint,
            source_layers=source_layers,
            subject_scopes=subject_scopes,
            categories=categories,
            importance_min=importance_min,
            exclude_source_ids=set(exclude_source_ids or []),
        )

    # --- 检索 + 校验 ---

    def _retrieve_and_verify(
        self,
        *,
        namespace: Namespace,
        query: str,
        keywords: list[str],
        time_hint: dict[str, Any] | None = None,
        source_layers: list[str] | None = None,
        subject_scopes: list[str] | None = None,
        categories: list[str] | None = None,
        importance_min: float | None = None,
        exclude_source_ids: set[str],
    ) -> list[str]:
        pool = max(10, self.config.retrieval_limit * 10)
        excluded = sorted(exclude_source_ids)
        requested = {
            "source_layers": source_layers or [],
            "subject_scopes": subject_scopes or [],
            "categories": categories or [],
            "importance_min": importance_min,
        }
        selected: list[dict[str, Any]] = []
        search_cache: dict[str, list[dict[str, Any]]] = {}
        for stage in self._build_stages(requested):
            where = self._build_where(namespace, time_hint)
            where.update(self._stage_index_where(stage))
            cache_key = repr((where, excluded))
            if cache_key not in search_cache:
                semantic_hits = self.index.semantic_search(
                    query_text=query, where=where, n_results=pool, exclude_source_ids=excluded
                )
                keyword_hits = self.index.keyword_search(
                    query_text=query, keywords=keywords, where=where, n_results=pool, exclude_source_ids=excluded
                )
                search_cache[cache_key] = fuse_with_rrf(semantic_hits, keyword_hits)
            selected = search_cache[cache_key]
            if len(selected) >= self.config.relaxation_stop_candidate_count:
                break
        selected = selected[: self.config.retrieval_limit]
        snippets = self._build_snippets(selected, exclude_context_source_ids=exclude_source_ids)
        return self._verify(query=query, snippets=snippets)

    def _build_where(self, namespace: Namespace, time_hint: dict[str, Any] | None) -> dict[str, Any]:
        tenant, user, domain = namespace.hard_key()
        where: dict[str, Any] = {"tenant_id": tenant, "user_id": user, "domain_id": domain}
        hint = time_hint if isinstance(time_hint, dict) else {}
        if hint.get("date_label"):
            where["date_label"] = str(hint["date_label"])
        if hint.get("time_of_day"):
            where["time_of_day"] = str(hint["time_of_day"])
        ts_range: dict[str, Any] = {}
        if hint.get("start_ts") is not None:
            ts_range["$gte"] = int(hint["start_ts"])
        if hint.get("end_ts") is not None:
            ts_range["$lte"] = int(hint["end_ts"])
        if ts_range:
            where["timestamp"] = ts_range
        return where

    # --- 五级精度放宽(软过滤) ---

    @staticmethod
    def _stage_index_where(stage: dict[str, Any]) -> dict[str, Any]:
        """把可安全前置的 metadata 过滤下推到 VectorIndex。

        不同维度之间是 AND;同一维度多值是 OR。VectorIndex 必须在相似度/BM25 计算前执行 where,
        读侧不再做后置精筛,避免把后端契约做成灰色地带。
        """
        where: dict[str, Any] = {}
        clauses: list[dict[str, Any]] = []
        source_layers = [str(x) for x in (stage.get("source_layers") or []) if str(x).strip()]
        if source_layers:
            where["entry_type"] = {"$in": source_layers}
        if stage.get("importance_min") is not None:
            where["memory_importance"] = {"$gte": float(stage["importance_min"])}
        category_clause = _flag_or_clause(category_filter_key(x) for x in _filter_values(stage.get("categories")))
        if category_clause:
            clauses.append(category_clause)
        scope_clause = _flag_or_clause(subject_scope_filter_key(x) for x in _filter_values(stage.get("subject_scopes")))
        if scope_clause:
            clauses.append(scope_clause)
        if clauses:
            where["$and"] = clauses
        return where

    @staticmethod
    def _build_stages(requested: dict[str, Any]) -> list[dict[str, Any]]:
        full = {
            "source_layers": list(requested.get("source_layers") or []),
            "subject_scopes": list(requested.get("subject_scopes") or []),
            "categories": list(requested.get("categories") or []),
            "importance_min": requested.get("importance_min"),
        }
        stages = [dict(full)]  # strict
        if full["importance_min"] is not None:
            full = {**full, "importance_min": None}
            stages.append(dict(full))
        if full["categories"]:
            full = {**full, "categories": []}
            stages.append(dict(full))
        if full["subject_scopes"]:
            full = {**full, "subject_scopes": []}
            stages.append(dict(full))
        if full["source_layers"]:
            full = {**full, "source_layers": []}
            stages.append(dict(full))
        return stages

    # --- 片段构建(回关系库取原文 + raw 上下文扩窗) ---

    def _build_snippets(self, hits: list[dict[str, Any]], *, exclude_context_source_ids: set[str]) -> list[str]:
        snippets: list[str] = []
        seen: set[str] = set()
        flavor = self.config.enable_flavor
        for hit in hits:
            source_id = str(hit.get("source_id"))
            record = self.store.get_record_by_source_id(source_id)
            if not record:
                continue
            entry_type = record.get("entry_type")
            if entry_type == "summary":
                snippet, key = (
                    render_summary_snippet(record, tz=self.timezone, enable_flavor=flavor),
                    f"summary::{source_id}",
                )
            elif entry_type == "semantic_summary":
                snippet, key = (
                    render_semantic_snippet(record, tz=self.timezone, enable_flavor=flavor),
                    f"semantic::{source_id}",
                )
            else:
                window = 2 if self._is_question_like(record.get("content", "")) else 1
                ns = Namespace(
                    user_id=str(record.get("user_id") or ""),
                    tenant_id=str(record.get("tenant_id") or ""),
                    domain_id=str(record.get("domain_id") or ""),
                    conversation_id=str(record.get("conversation_id") or ""),
                )
                rows = self.store.get_context_slice(namespace=ns, seq_no=int(record.get("seq_no") or 0), window=window)
                rows = [row for row in rows if str(row.get("source_id")) not in exclude_context_source_ids]
                if not rows:
                    continue
                snippet = render_raw_snippet(rows, tz=self.timezone)
                key = f"raw::{record.get('conversation_id')}::{record.get('seq_no')}"
            if snippet and key not in seen:
                seen.add(key)
                snippets.append(snippet)
        return snippets

    @staticmethod
    def _is_question_like(text: Any) -> bool:
        normalized = normalize_text(text)
        return any(marker in normalized for marker in _QUESTION_MARKERS)

    # --- verifier ---

    def _verify(self, *, query: str, snippets: list[str]) -> list[str]:
        if not snippets:
            return []
        if not self.config.enable_verifier:
            return snippets
        numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets))
        res = self.llm.call(
            LLMRequest(
                task_type=TaskType.VERIFIER,
                **dict(
                    zip(("system_prompt", "user_prompt"), build_verifier_prompts(query=query, snippets_text=numbered))
                ),
                response_format=ResponseFormat.NDJSON,
                max_retries=self.config.llm_max_retries,
                fallback={},
            )
        )
        if not res.ok:
            return snippets  # verifier 不可用时不清零,保留召回(降级)
        events = parse_ndjson(res.data)
        decision = next((e for e in events if e.get("type") == "decision"), None)
        if decision is None:
            return snippets
        if str(decision.get("match_result")) != "match":
            return []
        selection = next((e for e in events if e.get("type") == "selection"), {})
        indexes = [int(i) for i in (selection.get("selected_indexes") or []) if isinstance(i, int) or str(i).isdigit()]
        chosen = [snippets[i - 1] for i in indexes if 1 <= i <= len(snippets)]
        return chosen or snippets  # match 但没给编号 → 保守保留全部
