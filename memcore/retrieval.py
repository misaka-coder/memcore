"""Retrieval V2: immutable hard admission, bounded semantic relaxation, structured results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .config import MemoryConfig
from .index.base import VectorIndex
from .index.metadata_filters import (
    INDEX_SCHEMA_KEY,
    INDEX_SCHEMA_VERSION,
    about_role_filter_key,
    entity_filter_key,
    facet_filter_key,
    kind_filter_key,
    normalize_kind_pattern,
)
from .index.rrf import fuse_with_rrf
from .namespace import Namespace
from .rendering import render_raw_snippet, render_semantic_snippet, render_summary_snippet
from .store.base import LineageClosure, MemoryStore
from .schema import ABOUT_ROLES, MEMORY_FACETS
from .timeline import TimelineEntry, TurnRole
from .time_anchor import normalize_retrieval_time_hint
from .token_counter import TokenCounter

_ACCEPTED_DEFAULT_STATUSES = (
    "accepted_host",
    "accepted_model",
    "derived",
    "derived_turn_final",
)


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    entity_anchors: tuple[str, ...] = ()
    topic_terms: tuple[str, ...] = ()
    source_layers: tuple[str, ...] = ()
    memory_facets: tuple[str, ...] = ()
    about_roles: tuple[str, ...] = ()
    time_hint: Mapping[str, Any] = field(default_factory=dict)
    kind_patterns: tuple[str, ...] = ()
    include_explicit: bool = False
    cross_conversation: bool = False
    exclude_source_ids: tuple[str, ...] = ()
    max_matches: int = 0
    result_token_budget: int = 0


@dataclass(frozen=True)
class HardFilterPlan:
    namespace_key: tuple[str, str, str, str]
    cross_conversation: bool
    include_explicit: bool
    kind_patterns: tuple[str, ...]
    source_layers: tuple[str, ...]
    time_hint: Mapping[str, Any]
    exclude_source_ids: tuple[str, ...]
    index_where: Mapping[str, Any]


@dataclass(frozen=True)
class SemanticFilterPlan:
    memory_facets: tuple[str, ...] = ()
    about_roles: tuple[str, ...] = ()
    entity_anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationExpansionPlan:
    result_token_budget: int = 0


@dataclass(frozen=True)
class RetrievalQueryPlan:
    request: RetrievalRequest
    hard: HardFilterPlan
    semantic: SemanticFilterPlan
    relation: RelationExpansionPlan


@dataclass(frozen=True)
class RetrievalMatch:
    source_ids: tuple[str, ...]
    source_id: str
    turn_id: str
    correlation_id: str
    kind: str
    layer: str
    timestamp: int
    semantic_score: float
    bm25_score: float
    fused_score: float
    semantic_text: str
    rendered_text: str
    lineage: tuple[str, ...] = ()
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ids": list(self.source_ids),
            "source_id": self.source_id,
            "turn_id": self.turn_id,
            "correlation_id": self.correlation_id,
            "kind": self.kind,
            "layer": self.layer,
            "timestamp": self.timestamp,
            "scores": {
                "semantic": self.semantic_score,
                "bm25": self.bm25_score,
                "fused": self.fused_score,
            },
            "semantic_text": self.semantic_text,
            "rendered_text": self.rendered_text,
            "lineage": list(self.lineage),
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    matches: tuple[RetrievalMatch, ...] = ()
    effective_filters: Mapping[str, Any] = field(default_factory=dict)
    relaxation_steps: tuple[str, ...] = ()
    candidate_counts: Mapping[str, int] = field(default_factory=dict)
    entity_filter_relaxed: bool = False
    rejected_counts: Mapping[str, int] = field(default_factory=dict)
    token_usage: int = 0
    truncated: bool = False
    omitted_match_count: int = 0
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.status == "found"

    @property
    def rendered_texts(self) -> tuple[str, ...]:
        return tuple(match.rendered_text for match in self.matches if match.rendered_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "matches": [match.to_dict() for match in self.matches],
            "effective_filters": dict(self.effective_filters),
            "relaxation_steps": list(self.relaxation_steps),
            "candidate_counts": dict(self.candidate_counts),
            "entity_filter_relaxed": self.entity_filter_relaxed,
            "rejected_counts": dict(self.rejected_counts),
            "token_usage": self.token_usage,
            "truncated": self.truncated,
            "omitted_match_count": self.omitted_match_count,
            "reason": self.reason,
        }


def _filter_values(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("filter_values_must_be_array")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _flag_or_clause(keys: tuple[str, ...]) -> dict[str, Any]:
    if not keys:
        return {}
    if len(keys) == 1:
        return {keys[0]: True}
    return {"$or": [{key: True} for key in keys]}


class ReadPipeline:
    def __init__(
        self,
        *,
        store: MemoryStore,
        index: VectorIndex,
        config: MemoryConfig,
        timezone: str,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.config = config
        self.timezone = timezone
        self.token_counter = token_counter

    # --- visible prompt context ---

    def build_context(self, *, namespace: Namespace, now_ts: int) -> dict[str, Any]:
        cross = self.config.visible_memory_scope == "user"
        raw = self.store.get_unsummarized_messages(namespace=namespace)
        episodic = self.store.get_visible_episodic_summaries(
            namespace=namespace,
            limit=self.config.episodic_visible_max,
            cross_conversation=cross,
        )
        semantic = self._visible_semantic(namespace=namespace, cross=cross, now_ts=now_ts)
        return {"raw": raw, "episodic": episodic, "semantic": semantic}

    def visible_source_ids(self, *, namespace: Namespace, now_ts: int) -> set[str]:
        context = self.build_context(namespace=namespace, now_ts=now_ts)
        ids: set[str] = set()
        ids.update(str(row.get("source_id")) for row in context["raw"] if row.get("source_id"))
        ids.update(str(row.get("summary_id")) for row in context["episodic"] if row.get("summary_id"))
        ids.update(str(row.get("semantic_id")) for row in context["semantic"] if row.get("semantic_id"))
        return ids

    def visible_lineage_source_ids(self, *, namespace: Namespace, now_ts: int) -> set[str]:
        """Exclude visible entries plus every connected derived/raw lineage record."""
        visible = self.visible_source_ids(namespace=namespace, now_ts=now_ts)
        if not visible:
            return set()
        closure = self.store.resolve_lineage_source_ids(
            namespace=namespace,
            source_ids=tuple(sorted(visible)),
            cross_conversation=self.config.visible_memory_scope == "user",
        )
        if closure.status == "resolved":
            return set(closure.all_ids)
        expanded = set(visible)
        for source_id in sorted(visible):
            item = self.store.resolve_lineage_source_ids(
                namespace=namespace,
                source_ids=(source_id,),
                cross_conversation=self.config.visible_memory_scope == "user",
            )
            if item.status == "resolved":
                expanded.update(item.all_ids)
        return expanded

    def _visible_semantic(self, *, namespace: Namespace, cross: bool, now_ts: int) -> list[dict[str, Any]]:
        limit = self.config.semantic_visible_limit
        if not self.config.enable_importance_decay:
            return self.store.get_recent_semantic_summaries(
                namespace=namespace,
                limit=limit,
                cross_conversation=cross,
            )
        from .decay import decayed_importance

        pool = self.store.get_recent_semantic_summaries(
            namespace=namespace,
            limit=None,
            cross_conversation=cross,
        )
        half_life = self.config.importance_half_life_days

        def score(record: dict[str, Any]) -> float:
            anchor = int(record.get("last_reinforced_ts") or record.get("timestamp") or 0)
            return decayed_importance(
                record.get("importance", 0.0),
                age_seconds=max(0, int(now_ts) - anchor),
                half_life_days=half_life,
            )

        return sorted(pool, key=score, reverse=True)[:limit]

    # --- public retrieval ---

    def retrieve_result(self, *, namespace: Namespace, request: RetrievalRequest) -> RetrievalResult:
        try:
            plan = self._compile_plan(namespace=namespace, request=request)
        except (TypeError, ValueError) as exc:
            return RetrievalResult(status="invalid", reason=str(exc) or "invalid_filter")
        try:
            return self._execute_plan(plan)
        except NotImplementedError:
            return RetrievalResult(status="unavailable", reason="retrieval_store_unsupported")
        except RuntimeError:
            return RetrievalResult(status="unavailable", reason="index_unavailable")
        except Exception:
            return RetrievalResult(status="failed", reason="internal_error")

    def retrieve(
        self,
        *,
        namespace: Namespace,
        query: str,
        entity_anchors: list[str] | None = None,
        topic_terms: list[str] | None = None,
        time_hint: dict[str, Any] | None = None,
        source_layers: list[str] | None = None,
        memory_facets: list[str] | None = None,
        about_roles: list[str] | None = None,
        exclude_source_ids: list[str] | None = None,
        kind_patterns: list[str] | None = None,
        include_explicit: bool = False,
        cross_conversation: bool = True,
        result_token_budget: int = 0,
    ) -> list[str]:
        """Thin text adapter over Retrieval V2; no independent legacy search path remains."""
        result = self.retrieve_result(
            namespace=namespace,
            request=RetrievalRequest(
                query=query,
                entity_anchors=tuple(entity_anchors or ()),
                topic_terms=tuple(topic_terms or ()),
                source_layers=tuple(source_layers or ()),
                memory_facets=tuple(memory_facets or ()),
                about_roles=tuple(about_roles or ()),
                time_hint=dict(time_hint or {}),
                kind_patterns=tuple(kind_patterns or ()),
                include_explicit=bool(include_explicit),
                cross_conversation=bool(cross_conversation),
                exclude_source_ids=tuple(exclude_source_ids or ()),
                result_token_budget=result_token_budget,
            ),
        )
        return list(result.rendered_texts)

    # --- planning ---

    def _compile_plan(self, *, namespace: Namespace, request: RetrievalRequest) -> RetrievalQueryPlan:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request_must_be_retrieval_request")
        query = str(request.query or "").strip()
        if not query:
            raise ValueError("query_required")
        entity_anchors = _filter_values(request.entity_anchors)
        topic_terms = _filter_values(request.topic_terms)
        source_layers = _filter_values(request.source_layers)
        if any(item not in {"raw", "semantic_summary", "summary"} for item in source_layers):
            raise ValueError("invalid_source_layer")
        memory_facets = _filter_values(request.memory_facets)
        if any(item not in MEMORY_FACETS for item in memory_facets):
            raise ValueError("invalid_memory_facet")
        about_roles = _filter_values(request.about_roles)
        if any(item not in ABOUT_ROLES for item in about_roles):
            raise ValueError("invalid_about_role")
        kind_patterns = _filter_values(request.kind_patterns)
        normalized_patterns: list[str] = []
        for pattern in kind_patterns:
            kind, is_prefix = normalize_kind_pattern(pattern)
            normalized_patterns.append(f"{kind}.*" if is_prefix else kind)
        if request.include_explicit and not normalized_patterns:
            raise ValueError("explicit_kind_patterns_required")
        time_hint = self._normalize_time_hint(request.time_hint)
        excluded = _filter_values(request.exclude_source_ids)
        max_matches = int(request.max_matches or self.config.retrieval_limit)
        if max_matches < 1:
            raise ValueError("invalid_max_matches")
        max_matches = min(max_matches, self.config.retrieval_limit)
        try:
            result_token_budget = int(request.result_token_budget or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_result_token_budget") from exc
        if result_token_budget < 0:
            raise ValueError("invalid_result_token_budget")
        normalized_request = RetrievalRequest(
            query=query,
            entity_anchors=entity_anchors,
            topic_terms=topic_terms,
            source_layers=source_layers,
            memory_facets=memory_facets,
            about_roles=about_roles,
            time_hint=time_hint,
            kind_patterns=tuple(normalized_patterns),
            include_explicit=bool(request.include_explicit),
            cross_conversation=bool(request.cross_conversation),
            exclude_source_ids=excluded,
            max_matches=max_matches,
            result_token_budget=result_token_budget,
        )
        hard_where = self._build_hard_where(namespace=namespace, request=normalized_request)
        hard = HardFilterPlan(
            namespace_key=(
                namespace.tenant_id or "",
                namespace.user_id,
                namespace.domain_id or "",
                namespace.conversation_id or "",
            ),
            cross_conversation=normalized_request.cross_conversation,
            include_explicit=normalized_request.include_explicit,
            kind_patterns=normalized_request.kind_patterns,
            source_layers=source_layers,
            time_hint=time_hint,
            exclude_source_ids=excluded,
            index_where=hard_where,
        )
        semantic = SemanticFilterPlan(
            memory_facets=memory_facets,
            about_roles=about_roles,
            entity_anchors=entity_anchors,
        )
        return RetrievalQueryPlan(
            request=normalized_request,
            hard=hard,
            semantic=semantic,
            relation=RelationExpansionPlan(result_token_budget=result_token_budget),
        )

    def _normalize_time_hint(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return normalize_retrieval_time_hint(timezone=self.timezone, value=value)

    def _build_hard_where(self, *, namespace: Namespace, request: RetrievalRequest) -> dict[str, Any]:
        tenant, user, domain = namespace.hard_key()
        where: dict[str, Any] = {
            "tenant_id": tenant,
            "user_id": user,
            "domain_id": domain,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "index_schema_key": INDEX_SCHEMA_KEY,
            "is_legacy_migration": False,
            "trust": "untrusted_data",
            "lineage_status": {"$in": ["raw", "valid"]},
        }
        if not request.cross_conversation:
            where["conversation_id"] = namespace.conversation_id or ""
        hint = request.time_hint
        periods = list(hint.get("time_periods") or [])
        if periods:
            where["time_of_day"] = periods[0] if len(periods) == 1 else {"$in": periods}
        timestamp: dict[str, int] = {}
        if hint.get("start_ts") is not None:
            timestamp["$gte"] = int(hint["start_ts"])
        if hint.get("end_ts") is not None:
            timestamp["$lt"] = int(hint["end_ts"])
        if timestamp:
            where["timestamp"] = timestamp

        kind_clause = self._kind_clause(request.kind_patterns)
        default_admission: dict[str, Any] = {
            "$and": [
                {"retrieval_visibility": "default"},
                {
                    "$or": [
                        {"annotation_status": {"$in": list(_ACCEPTED_DEFAULT_STATUSES)}},
                        {"retrieval_policy": "always"},
                    ]
                },
                {"is_trace_kind": False},
            ]
        }
        admission: dict[str, Any] = default_admission
        if request.include_explicit:
            explicit_admission = {
                "$and": [
                    kind_clause,
                    {"annotation_status": {"$ne": "accepted_legacy"}},
                    {
                        "$or": [
                            {"retrieval_visibility": "explicit"},
                            {"is_trace_kind": True},
                        ]
                    },
                ]
            }
            admission = {"$or": [default_admission, explicit_admission]}
        clauses = [admission]
        if kind_clause:
            clauses.append(kind_clause)
        where["$and"] = clauses
        return where

    @staticmethod
    def _kind_clause(patterns: tuple[str, ...]) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = []
        for pattern in patterns:
            kind, is_prefix = normalize_kind_pattern(pattern)
            clauses.append({kind_filter_key(kind): True} if is_prefix else {"kind_exact": kind})
        if not clauses:
            return {}
        return clauses[0] if len(clauses) == 1 else {"$or": clauses}

    @staticmethod
    def _semantic_where(stage: SemanticFilterPlan, *, include_entities: bool = True) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = []
        facet_clause = _flag_or_clause(tuple(facet_filter_key(item) for item in stage.memory_facets))
        if facet_clause:
            clauses.append(facet_clause)
        role_clause = _flag_or_clause(tuple(about_role_filter_key(item) for item in stage.about_roles))
        if role_clause:
            clauses.append(role_clause)
        if include_entities:
            entity_clause = _flag_or_clause(tuple(entity_filter_key(item) for item in stage.entity_anchors))
            if entity_clause:
                clauses.append(entity_clause)
        return {"$and": clauses} if clauses else {}

    @staticmethod
    def _merge_where(hard: Mapping[str, Any], semantic: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(hard)
        clauses = list(merged.pop("$and", []))
        semantic_copy = dict(semantic)
        clauses.extend(semantic_copy.pop("$and", []))
        for key, value in semantic_copy.items():
            if key in merged:
                clauses.append({key: value})
            else:
                merged[key] = value
        if clauses:
            merged["$and"] = clauses
        return merged

    # --- execution ---

    def _execute_plan(self, plan: RetrievalQueryPlan) -> RetrievalResult:
        pool = max(10, plan.request.max_matches * 10)
        rejected = {
            "below_bm25_score": 0,
            "below_dense_score": 0,
            "below_fused_score": 0,
            "invalid_lineage": 0,
            "lineage_quarantine_failed": 0,
            "incomplete_relation": 0,
        }
        allowed_layers = set(plan.request.source_layers or ("raw", "summary", "semantic_summary"))
        raw_layers = ("raw",) if "raw" in allowed_layers else ()
        derived_layers = tuple(layer for layer in ("summary", "semantic_summary") if layer in allowed_layers)
        raw_matches, unsafe, raw_strict, raw_effective, raw_relaxed = self._search_pool(
            plan=plan,
            layers=raw_layers,
            pool=pool,
            rejected=rejected,
        )
        if unsafe:
            return RetrievalResult(status="unavailable", reason="index_filter_unsupported")
        derived_matches, unsafe, derived_strict, derived_effective, derived_relaxed = self._search_pool(
            plan=plan,
            layers=derived_layers,
            pool=pool,
            rejected=rejected,
        )
        if unsafe:
            return RetrievalResult(status="unavailable", reason="index_filter_unsupported")
        candidate_counts = {
            "raw_strict": raw_strict,
            "raw_effective": raw_effective,
            "derived_strict": derived_strict,
            "derived_effective": derived_effective,
        }
        relaxed_layers = [layer for layer, relaxed in (("raw", raw_relaxed), ("derived", derived_relaxed)) if relaxed]
        relaxation = tuple(f"drop_entity_requirement_after_zero_candidates:{layer}" for layer in relaxed_layers)
        filters = self._effective_filters(
            plan=plan,
            semantic=plan.semantic,
            entity_relaxed_layers=tuple(relaxed_layers),
        )
        selected_matches = list(raw_matches[: plan.request.max_matches])
        remaining = max(0, plan.request.max_matches - len(selected_matches))
        if remaining:
            selected_matches.extend(derived_matches[:remaining])
        if plan.relation.result_token_budget > 0 and self.token_counter is None:
            return RetrievalResult(
                status="unavailable",
                effective_filters=filters,
                relaxation_steps=relaxation,
                candidate_counts=candidate_counts,
                entity_filter_relaxed=bool(relaxed_layers),
                rejected_counts=rejected,
                reason="token_counter_required",
            )
        expanded, unsafe = self._expand_matches(plan=plan, matches=selected_matches, rejected=rejected)
        if unsafe:
            return RetrievalResult(
                status="unavailable",
                effective_filters=filters,
                relaxation_steps=relaxation,
                candidate_counts=candidate_counts,
                entity_filter_relaxed=bool(relaxed_layers),
                rejected_counts=rejected,
                reason="relation_store_unsupported",
            )
        budgeted, token_usage, truncated, omitted = self._apply_result_budget(
            expanded,
            token_budget=plan.relation.result_token_budget,
        )
        return RetrievalResult(
            status="found" if budgeted else "empty",
            matches=tuple(budgeted),
            effective_filters=filters,
            relaxation_steps=relaxation,
            candidate_counts=candidate_counts,
            entity_filter_relaxed=bool(relaxed_layers),
            rejected_counts=rejected,
            token_usage=token_usage,
            truncated=truncated,
            omitted_match_count=omitted,
            reason="" if budgeted else "no_match",
        )

    def _search_pool(
        self,
        *,
        plan: RetrievalQueryPlan,
        layers: tuple[str, ...],
        pool: int,
        rejected: dict[str, int],
    ) -> tuple[list[RetrievalMatch], bool, int, int, bool]:
        if not layers:
            return [], False, 0, 0, False
        layer_where: dict[str, Any] = {"entry_type": layers[0] if len(layers) == 1 else {"$in": list(layers)}}
        hard_with_layer = self._merge_where(plan.hard.index_where, layer_where)
        strict_where = self._merge_where(
            hard_with_layer,
            self._semantic_where(plan.semantic, include_entities=True),
        )
        excluded = list(plan.hard.exclude_source_ids)
        strict_count = self.index.count_candidates(where=strict_where, exclude_source_ids=excluded)
        attempted_relaxation = bool(plan.semantic.entity_anchors and strict_count == 0)
        effective_where = (
            self._merge_where(
                hard_with_layer,
                self._semantic_where(plan.semantic, include_entities=False),
            )
            if attempted_relaxation
            else strict_where
        )
        effective_count = (
            self.index.count_candidates(where=effective_where, exclude_source_ids=excluded)
            if attempted_relaxation
            else strict_count
        )
        # A pool with no candidates at all did not materially relax anything;
        # do not report a misleading entity-relaxation diagnostic for it.
        relaxed = attempted_relaxation and effective_count > 0
        if effective_count == 0:
            return [], False, strict_count, effective_count, relaxed
        semantic_hits = self.index.semantic_search(
            query_text=plan.request.query,
            where=effective_where,
            n_results=pool,
            exclude_source_ids=excluded,
        )
        keyword_hits = self.index.keyword_search(
            query_text=plan.request.query,
            entity_anchors=list(plan.request.entity_anchors),
            topic_terms=list(plan.request.topic_terms),
            where=effective_where,
            n_results=pool,
            exclude_source_ids=excluded,
        )
        # Hard filters are a security boundary, not a ranking preference. Check
        # every candidate returned by the backend before score thresholds can
        # hide evidence that the backend ignored namespace/layer/visibility.
        positive_candidates: list[dict[str, Any]] = []
        for hit, score_key in (
            *((item, "semantic_score") for item in semantic_hits),
            *((item, "tag_score") for item in keyword_hits),
        ):
            try:
                score = float(hit.get(score_key) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if math.isfinite(score) and score > 0.0:
                positive_candidates.append(hit)
        backend_candidates = list(
            {
                str(hit.get("source_id") or ""): hit for hit in positive_candidates if str(hit.get("source_id") or "")
            }.values()
        )
        _validated, unsafe = self._build_matches(plan=plan, hits=backend_candidates)
        if unsafe:
            return [], True, strict_count, effective_count, relaxed
        semantic_hits = self._score_filter(
            semantic_hits,
            key="semantic_score",
            minimum=float(self.config.retrieval_min_dense_score),
            rejected=rejected,
            rejected_key="below_dense_score",
        )
        keyword_hits = self._score_filter(
            keyword_hits,
            key="tag_score",
            minimum=float(self.config.retrieval_min_bm25_score),
            rejected=rejected,
            rejected_key="below_bm25_score",
        )
        fused = self._score_filter(
            fuse_with_rrf(semantic_hits, keyword_hits),
            key="rrf_score",
            minimum=float(self.config.retrieval_min_fused_score),
            rejected=rejected,
            rejected_key="below_fused_score",
        )
        matches, unsafe = self._build_matches(
            plan=plan,
            hits=fused[: plan.request.max_matches],
        )
        return matches, unsafe, strict_count, effective_count, relaxed

    @staticmethod
    def _score_filter(
        hits: list[dict[str, Any]],
        *,
        key: str,
        minimum: float,
        rejected: dict[str, int],
        rejected_key: str,
    ) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        threshold = max(0.0, float(minimum))
        for hit in hits:
            try:
                score = float(hit.get(key) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score) or score <= 0.0 or score < threshold:
                rejected[rejected_key] += 1
                continue
            accepted.append(hit)
        return accepted

    def _build_matches(
        self,
        *,
        plan: RetrievalQueryPlan,
        hits: list[dict[str, Any]],
    ) -> tuple[list[RetrievalMatch], bool]:
        matches: list[RetrievalMatch] = []
        for hit in hits:
            source_id = str(hit.get("source_id") or "")
            record = self.store.get_retrieval_record(
                namespace=Namespace(
                    tenant_id=plan.hard.namespace_key[0],
                    user_id=plan.hard.namespace_key[1],
                    domain_id=plan.hard.namespace_key[2],
                    conversation_id=plan.hard.namespace_key[3],
                ),
                source_id=source_id,
                cross_conversation=plan.hard.cross_conversation,
            )
            if record is None:
                return [], True
            if not self._record_satisfies_hard_plan(record=record, hard=plan.hard):
                return [], True
            match = self._record_match(record=record, hit=hit)
            if match.rendered_text:
                matches.append(match)
        return matches, False

    def _record_satisfies_hard_plan(self, *, record: dict[str, Any], hard: HardFilterPlan) -> bool:
        source_id = str(record.get("source_id") or record.get("summary_id") or record.get("semantic_id") or "")
        if not source_id or source_id in hard.exclude_source_ids:
            return False
        namespace_key = (
            str(record.get("tenant_id") or ""),
            str(record.get("user_id") or ""),
            str(record.get("domain_id") or ""),
            str(record.get("conversation_id") or ""),
        )
        if namespace_key[:3] != hard.namespace_key[:3]:
            return False
        if not hard.cross_conversation and namespace_key[3] != hard.namespace_key[3]:
            return False
        if str(record.get("index_status") or "") != "indexed":
            return False
        if int(record.get("index_schema_version") or 0) != INDEX_SCHEMA_VERSION:
            return False
        if str(record.get("index_key") or "") != INDEX_SCHEMA_KEY:
            return False
        if str(record.get("trust") or "untrusted_data") != "untrusted_data":
            return False
        trace_metadata = record.get("trace_metadata") if isinstance(record.get("trace_metadata"), dict) else {}
        if str(record.get("annotation_status") or "") == "accepted_legacy" or trace_metadata.get("legacy_categories"):
            return False
        kind = str(record.get("kind") or "legacy.unknown").lower()
        if hard.kind_patterns and not any(self._kind_matches(kind, pattern) for pattern in hard.kind_patterns):
            return False
        visibility = str(record.get("retrieval_visibility") or "explicit")
        annotation = str(record.get("annotation_status") or "unannotated")
        retrieval_policy = str(record.get("retrieval_policy") or "auto")
        if retrieval_policy == "never":
            return False
        is_trace = kind.split(".", 1)[0] in {"material", "tool"}
        default_allowed = (
            visibility == "default"
            and (annotation in _ACCEPTED_DEFAULT_STATUSES or retrieval_policy == "always")
            and not is_trace
        )
        explicit_allowed = (
            hard.include_explicit and annotation != "accepted_legacy" and (visibility == "explicit" or is_trace)
        )
        if not (default_allowed or explicit_allowed):
            return False
        if visibility == "never":
            return False
        layer = str(record.get("entry_type") or "raw")
        if hard.source_layers and layer not in hard.source_layers:
            return False
        if layer != "raw" and str(record.get("lineage_status") or "") != "valid":
            return False
        hint = hard.time_hint
        periods = set(hint.get("time_periods") or ())
        if periods and str(record.get("time_of_day") or "") not in periods:
            return False
        timestamp = int(record.get("timestamp") or 0)
        if hint.get("start_ts") is not None and timestamp < int(hint["start_ts"]):
            return False
        return not (hint.get("end_ts") is not None and timestamp >= int(hint["end_ts"]))

    @staticmethod
    def _kind_matches(kind: str, pattern: str) -> bool:
        normalized, is_prefix = normalize_kind_pattern(pattern)
        return kind == normalized or (is_prefix and kind.startswith(normalized + "."))

    def _record_match(self, *, record: dict[str, Any], hit: dict[str, Any]) -> RetrievalMatch:
        source_id = str(record.get("source_id") or record.get("summary_id") or record.get("semantic_id") or "")
        layer = str(record.get("entry_type") or "raw")
        if layer == "summary":
            rendered = render_summary_snippet(record, tz=self.timezone, enable_flavor=self.config.enable_flavor)
            semantic_text = str(record.get("diary_summary") or "")
            lineage = tuple(str(item) for item in (record.get("source_ids") or ()))
        elif layer == "semantic_summary":
            rendered = render_semantic_snippet(record, tz=self.timezone, enable_flavor=self.config.enable_flavor)
            semantic_text = str(record.get("semantic_summary") or "")
            lineage = tuple(str(item) for item in (record.get("source_summary_ids") or ()))
        else:
            rendered = render_raw_snippet([record], tz=self.timezone)
            semantic_text = str(record.get("semantic_text") or record.get("content") or "")
            lineage = ()
        return RetrievalMatch(
            source_ids=(source_id,),
            source_id=source_id,
            turn_id=str(record.get("turn_id") or ""),
            correlation_id=str(record.get("correlation_id") or ""),
            kind=str(record.get("kind") or "legacy.unknown"),
            layer=layer,
            timestamp=int(record.get("timestamp") or 0),
            semantic_score=float(hit.get("semantic_score") or 0.0),
            bm25_score=float(hit.get("tag_score") or 0.0),
            fused_score=float(hit.get("rrf_score") or 0.0),
            semantic_text=semantic_text,
            rendered_text=rendered,
            lineage=lineage,
        )

    def _expand_matches(
        self,
        *,
        plan: RetrievalQueryPlan,
        matches: list[RetrievalMatch],
        rejected: dict[str, int],
    ) -> tuple[list[RetrievalMatch], bool]:
        if not matches:
            return [], False
        records: dict[str, dict[str, Any]] = {}
        closures: dict[str, LineageClosure] = {}
        valid: list[RetrievalMatch] = []
        try:
            for match in matches:
                record = self.store.get_retrieval_record(
                    namespace=self._plan_namespace(plan),
                    source_id=match.source_id,
                    cross_conversation=plan.hard.cross_conversation,
                )
                if record is None or not self._record_satisfies_hard_plan(record=record, hard=plan.hard):
                    return [], True
                records[match.source_id] = record
                if match.layer in {"summary", "semantic_summary"}:
                    closure = self.store.resolve_lineage_source_ids(
                        namespace=self._record_namespace(record),
                        source_ids=(match.source_id,),
                        cross_conversation=False,
                        include_ancestors=False,
                    )
                    if closure.status != "resolved":
                        rejected["invalid_lineage"] += 1
                        if not self._invalidate_lineage(match.source_id):
                            rejected["lineage_quarantine_failed"] += 1
                        continue
                    closures[match.source_id] = closure
                valid.append(match)
        except NotImplementedError:
            return [], True

        raw_source_ids = {match.source_id for match in valid if match.layer == "raw"}
        expanded: list[RetrievalMatch] = []
        seen_groups: set[tuple[str, ...]] = set()
        for match in valid:
            record = records[match.source_id]
            if match.layer in {"summary", "semantic_summary"}:
                lineage = closures[match.source_id].descendant_ids
                if raw_source_ids.intersection(lineage):
                    continue
                candidate = replace(match, lineage=lineage)
            else:
                candidate = self._expand_raw_match(plan=plan, match=match, record=record)
                if candidate is None:
                    rejected["incomplete_relation"] += 1
                    continue
            group_key = candidate.source_ids
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            expanded.append(candidate)
        return expanded, False

    @staticmethod
    def _plan_namespace(plan: RetrievalQueryPlan) -> Namespace:
        return Namespace(
            tenant_id=plan.hard.namespace_key[0],
            user_id=plan.hard.namespace_key[1],
            domain_id=plan.hard.namespace_key[2],
            conversation_id=plan.hard.namespace_key[3],
        )

    @staticmethod
    def _record_namespace(record: Mapping[str, Any]) -> Namespace:
        return Namespace(
            tenant_id=str(record.get("tenant_id") or ""),
            user_id=str(record.get("user_id") or ""),
            domain_id=str(record.get("domain_id") or ""),
            conversation_id=str(record.get("conversation_id") or ""),
        )

    def _invalidate_lineage(self, source_id: str) -> bool:
        success = True
        try:
            self.index.delete([source_id])
        except Exception:
            success = False
        try:
            self.store.set_index_status(source_id, "invalid_lineage")
        except Exception:
            success = False
        return success

    def _expand_raw_match(
        self,
        *,
        plan: RetrievalQueryPlan,
        match: RetrievalMatch,
        record: dict[str, Any],
    ) -> RetrievalMatch | None:
        turn_id = str(record.get("turn_id") or "")
        if not turn_id or str(record.get("relation_status") or "") != "linked":
            return match
        namespace = self._record_namespace(record)
        entries = self.store.get_turn_entries(namespace=namespace, turn_id=turn_id)
        if not entries or any(not self._relation_entry_in_scope(entry, namespace) for entry in entries):
            return None

        root_kind = str(record.get("kind") or "").split(".", 1)[0]
        correlation_id = str(record.get("correlation_id") or "")
        if root_kind == "tool" and correlation_id:
            branch = self.store.get_correlation_entries(
                namespace=namespace,
                turn_id=turn_id,
                correlation_id=correlation_id,
            )
            if not self._complete_correlation_branch(branch):
                return None
            selected = branch
        else:
            turn = self.store.get_turn(namespace=namespace, turn_id=turn_id)
            if turn is None:
                return None
            target_ids = set(turn.annotation_target_ids)
            selected = [
                entry for entry in entries if entry.source_id in target_ids or entry.turn_role is TurnRole.FINAL
            ]
            if root_kind == "event" and not any(entry.source_id == match.source_id for entry in selected):
                selected.extend(entry for entry in entries if entry.source_id == match.source_id)
            if not selected:
                return None

        selected = sorted(selected, key=lambda item: item.seq_no)
        if any(
            not self._relation_entry_visible(entry, include_explicit=plan.hard.include_explicit) for entry in selected
        ):
            return None
        records = [entry.to_record() for entry in selected]
        source_ids = tuple(entry.source_id for entry in selected)
        return replace(
            match,
            source_ids=source_ids,
            turn_id=turn_id,
            correlation_id=correlation_id if root_kind == "tool" else "",
            semantic_text="\n".join(entry.semantic_text for entry in selected if entry.semantic_text),
            rendered_text=render_raw_snippet(records, tz=self.timezone),
            lineage=(),
        )

    @staticmethod
    def _relation_entry_in_scope(entry: TimelineEntry, namespace: Namespace) -> bool:
        return entry.namespace.hard_key() == namespace.hard_key() and (entry.namespace.conversation_id or "") == (
            namespace.conversation_id or ""
        )

    @staticmethod
    def _relation_entry_visible(entry: TimelineEntry, *, include_explicit: bool) -> bool:
        policy = entry.retrieval_policy.value
        visibility = entry.retrieval_visibility.value
        if policy == "never" or visibility == "never" or entry.trust.value != "untrusted_data":
            return False
        if visibility == "explicit":
            return include_explicit
        if visibility != "default":
            return False
        root_kind = entry.kind.split(".", 1)[0]
        annotation = entry.annotation_status.value
        return root_kind not in {"material", "tool"} and (
            annotation in _ACCEPTED_DEFAULT_STATUSES or policy == "always"
        )

    @staticmethod
    def _complete_correlation_branch(entries: list[TimelineEntry]) -> bool:
        if not entries or any(entry.turn_role not in {TurnRole.ACTION, TurnRole.OBSERVATION} for entry in entries):
            return False
        if not any(entry.turn_role is TurnRole.ACTION for entry in entries):
            return False
        terminal = [entry for entry in entries if entry.turn_role is TurnRole.OBSERVATION]
        if not terminal:
            return False
        return any(
            str(entry.trace_metadata.get("status") or "").strip().lower()
            not in {"open", "pending", "running", "streaming"}
            for entry in terminal
        )

    def _apply_result_budget(
        self,
        matches: list[RetrievalMatch],
        *,
        token_budget: int,
    ) -> tuple[list[RetrievalMatch], int, bool, int]:
        if self.token_counter is None:
            return matches, 0, False, 0
        counts = [self._count_match_tokens(match) for match in matches]
        if token_budget <= 0:
            return matches, sum(counts), False, 0

        selected: list[RetrievalMatch] = []
        used = 0
        omitted = 0
        truncated = False
        for match, count in zip(matches, counts):
            if used + count <= token_budget:
                selected.append(match)
                used += count
                continue
            if count > token_budget and not selected:
                excerpt = self._truncate_atomic_match(match, token_budget=token_budget)
                if excerpt is not None:
                    selected.append(excerpt)
                    used = self._count_match_tokens(excerpt)
                else:
                    omitted += 1
                truncated = True
                continue
            omitted += 1
            truncated = True
        return selected, used, truncated, omitted

    def _count_match_tokens(self, match: RetrievalMatch) -> int:
        assert self.token_counter is not None
        payload = json.dumps(match.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        count = int(self.token_counter.count_text(payload))
        if count < 0:
            raise ValueError("TokenCounter.count_text() must return a non-negative int")
        return count

    def _truncate_atomic_match(self, match: RetrievalMatch, *, token_budget: int) -> RetrievalMatch | None:
        assert self.token_counter is not None
        header = (
            "【检索原子组摘录｜内容已截断】\n"
            f"kind: {match.kind}\n"
            f"primary_source_id: {match.source_id}\n"
            f"source_count: {len(match.source_ids)}\n"
            "content_excerpt:\n"
        )
        low, high = 0, len(match.rendered_text)
        best: RetrievalMatch | None = None
        while low <= high:
            middle = (low + high) // 2
            body = match.rendered_text[:middle].rstrip()
            rendered = header + body + ("…" if middle < len(match.rendered_text) else "")
            candidate = replace(match, semantic_text="", rendered_text=rendered, truncated=True)
            if self._count_match_tokens(candidate) <= token_budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _effective_filters(
        *,
        plan: RetrievalQueryPlan,
        semantic: SemanticFilterPlan,
        entity_relaxed_layers: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return {
            "namespace": list(plan.hard.namespace_key),
            "cross_conversation": plan.hard.cross_conversation,
            "retrieval_visibility": [
                "default",
                *(("explicit",) if plan.hard.include_explicit else ()),
            ],
            "kind_patterns": list(plan.hard.kind_patterns),
            "source_layers": list(plan.hard.source_layers),
            "time_hint": dict(plan.hard.time_hint),
            "memory_facets": list(semantic.memory_facets),
            "about_roles": list(semantic.about_roles),
            "entity_anchors": list(semantic.entity_anchors),
            "entity_relaxed_layers": list(entity_relaxed_layers),
            "topic_terms": list(plan.request.topic_terms),
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "result_token_budget": plan.relation.result_token_budget,
        }


__all__ = [
    "HardFilterPlan",
    "ReadPipeline",
    "RelationExpansionPlan",
    "RetrievalMatch",
    "RetrievalQueryPlan",
    "RetrievalRequest",
    "RetrievalResult",
    "SemanticFilterPlan",
]
