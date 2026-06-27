"""读侧:router → 混合检索(语义+关键词+RRF+五级放宽)→ verifier → 拼可见三层 + 检索片段。

设计见 §6。两道门:router 决定要不要检索(省开销),verifier 决定片段够不够(防乱编)。
精度过滤可逐级放宽(软过滤),硬隔离(namespace)和时间走 index where(不放宽)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import MemoryConfig
from .index.base import VectorIndex
from .index.rrf import fuse_with_rrf
from .llm.base import LLMClient, LLMRequest, ResponseFormat, TaskType
from .namespace import Namespace
from .prompts import build_router_prompts, build_verifier_prompts
from .rendering import render_raw_snippet, render_semantic_snippet, render_summary_snippet
from .store.base import MemoryStore
from .text_utils import normalize_text

_PAST_MARKERS = ("记得", "之前", "上次", "上回", "以前", "曾经", "约定", "答应", "叫什么", "来着", "还记得")
_QUESTION_MARKERS = ("?", "?", "吗", "什么", "怎么", "为什么", "谁", "哪", "几", "多少")


@dataclass
class RouterDecision:
    need_retrieval: bool
    query: str = ""
    keywords: list[str] = field(default_factory=list)
    time_hint: dict[str, Any] | None = None
    degraded: bool = False  # LLM 失败时走启发式兜底


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

    def build_context(
        self, *, namespace: Namespace, current_message: str, now_ts: int, exclude_source_ids: list[str] | None = None
    ) -> dict[str, Any]:
        cross = self.config.visible_memory_scope == "user"
        raw = self.store.get_unsummarized_messages(namespace=namespace)
        episodic = self.store.get_visible_episodic_summaries(
            namespace=namespace, limit=self.config.episodic_visible_max, cross_conversation=cross
        )
        semantic = self.store.get_recent_semantic_summaries(
            namespace=namespace, limit=self.config.semantic_visible_limit, cross_conversation=cross
        )

        visible_ids = set(exclude_source_ids or [])
        visible_ids |= {str(r.get("source_id")) for r in raw}
        visible_ids |= {str(s.get("summary_id")) for s in episodic}
        visible_ids |= {str(s.get("semantic_id")) for s in semantic}

        router = self._route(namespace=namespace, current_message=current_message, visible=(raw, episodic, semantic))
        retrieved: list[str] = []
        if router.need_retrieval:
            retrieved = self._retrieve_and_verify(
                namespace=namespace,
                query=router.query or current_message,
                keywords=router.keywords,
                time_hint=router.time_hint,
                exclude_source_ids=visible_ids,
            )
        return {
            "raw": raw,
            "episodic": episodic,
            "semantic": semantic,
            "retrieved_snippets": retrieved,
            "router": router,
        }

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

    # --- router ---

    def _route(self, *, namespace: Namespace, current_message: str, visible: tuple) -> RouterDecision:
        if not self.config.enable_pre_retrieval:
            return RouterDecision(need_retrieval=False)
        recent_context = self._render_recent_context(visible)
        res = self.llm.call(
            LLMRequest(
                task_type=TaskType.ROUTER,
                **dict(
                    zip(
                        ("system_prompt", "user_prompt"),
                        build_router_prompts(recent_context=recent_context, current_message=current_message),
                    )
                ),
                response_format=ResponseFormat.NDJSON,
                max_retries=self.config.llm_max_retries,
                fallback={},
            )
        )
        if not res.ok:
            return self._heuristic_route(current_message)
        events = parse_ndjson(res.data)
        decision = next((e for e in events if e.get("type") == "decision"), None)
        if decision is None:
            return self._heuristic_route(current_message)
        if not bool(decision.get("need_retrieval")):
            return RouterDecision(need_retrieval=False)
        query_ev = next((e for e in events if e.get("type") == "query"), {})
        return RouterDecision(
            need_retrieval=True,
            query=str(query_ev.get("rewritten_query") or "").strip(),
            keywords=[str(k).strip() for k in (query_ev.get("keywords") or []) if str(k).strip()],
            time_hint=query_ev.get("time_hint") if isinstance(query_ev.get("time_hint"), dict) else None,
        )

    @staticmethod
    def _heuristic_route(message: str) -> RouterDecision:
        text = normalize_text(message)
        need = any(marker in text for marker in _PAST_MARKERS)
        return RouterDecision(need_retrieval=need, query=text if need else "", degraded=True)

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
        where = self._build_where(namespace, time_hint)
        pool = max(10, self.config.retrieval_limit * 10)
        semantic_hits = self.index.semantic_search(query_text=query, where=where, n_results=pool)
        keyword_hits = self.index.keyword_search(query_text=query, keywords=keywords, where=where, n_results=pool)
        fused = fuse_with_rrf(semantic_hits, keyword_hits)
        fused = [h for h in fused if str(h.get("source_id")) not in exclude_source_ids]

        requested = {
            "source_layers": source_layers or [],
            "subject_scopes": subject_scopes or [],
            "categories": categories or [],
            "importance_min": importance_min,
        }
        selected = self._apply_precision_relaxation(fused, requested)[: self.config.retrieval_limit]
        snippets = self._build_snippets(selected, exclude_source_ids=exclude_source_ids)
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

    def _apply_precision_relaxation(
        self, hits: list[dict[str, Any]], requested: dict[str, Any]
    ) -> list[dict[str, Any]]:
        stages = self._build_stages(requested)
        stop = self.config.relaxation_stop_candidate_count
        selected = [h for h in hits if self._match_filters(h, stages[-1])]  # 默认最宽
        for stage in stages:
            matching = [h for h in hits if self._match_filters(h, stage)]
            selected = matching
            if len(matching) >= stop:
                break
        return selected

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

    @staticmethod
    def _match_filters(hit: dict[str, Any], stage: dict[str, Any]) -> bool:
        meta = hit.get("metadata") or {}
        if stage.get("source_layers") and str(meta.get("entry_type", "")) not in stage["source_layers"]:
            return False
        if stage.get("importance_min") is not None and float(meta.get("memory_importance") or 0.0) < float(
            stage["importance_min"]
        ):
            return False
        for key, meta_key in (
            ("subject_scopes", "memory_subject_scopes_text"),
            ("categories", "memory_categories_text"),
        ):
            wanted = stage.get(key)
            if wanted:
                have = set(str(meta.get(meta_key, "")).split())
                if not (set(wanted) & have):  # OR 匹配:有交集才过
                    return False
        return True

    # --- 片段构建(回关系库取原文 + raw 上下文扩窗) ---

    def _build_snippets(self, hits: list[dict[str, Any]], *, exclude_source_ids: set[str]) -> list[str]:
        snippets: list[str] = []
        seen: set[str] = set()
        flavor = self.config.enable_flavor
        for hit in hits:
            source_id = str(hit.get("source_id"))
            if source_id in exclude_source_ids:
                continue
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

    # --- 渲染可见上下文摘要(给 router 看) ---

    @staticmethod
    def _render_recent_context(visible: tuple) -> str:
        raw, episodic, semantic = visible
        lines: list[str] = []
        for r in raw[-6:]:
            lines.append(f"{r.get('role', '')}: {normalize_text(r.get('content'))}")
        for s in episodic[:3]:
            lines.append(f"[摘要] {normalize_text(s.get('diary_summary'))}")
        for s in semantic[:2]:
            lines.append(f"[长期] {normalize_text(s.get('semantic_summary'))}")
        return "\n".join(lines)
