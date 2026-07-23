"""InMemoryVectorIndex —— 纯 Python 默认/测试后端。

实现 VectorIndex 全部能力(语义 cosine + 关键词 BM25 + where 硬过滤),不依赖任何外部服务,
让检索主干(混合 + RRF + 放宽)在无 chroma/无网络下可单测。生产可换 ChromaVectorIndex 等。

关键词侧文本拼入实体、主题、facet、主体和 mood 标签，让结构化标注参与召回。
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any

from ..embedding.base import EmbeddingProvider
from ..text_utils import tokenize
from .base import VectorIndex

try:  # optional speed path; core memcore remains dependency-free
    import numpy as _np
except Exception:  # pragma: no cover - depends on optional runtime dependency
    _np = None


def _vector_len(vector: Any) -> int:
    try:
        return len(vector)
    except TypeError:
        return 0


def _compact_vector(vector: Any) -> Any:
    """Use float32 arrays when NumPy is available; keep pure-Python fallback dependency-free."""
    if _np is None:
        return vector
    try:
        array = _np.asarray(vector, dtype=_np.float32)
    except (TypeError, ValueError):
        return vector
    return array if array.ndim == 1 else vector


def _vector_norm(vector: Any) -> float:
    if _vector_len(vector) <= 0:
        return 0.0
    if _np is not None:
        try:
            return float(_np.linalg.norm(vector))
        except (TypeError, ValueError):
            pass
    try:
        return math.sqrt(sum(x * x for x in vector))
    except (TypeError, ValueError):
        return 0.0


def _cosine(a: Any, b: Any, *, a_norm: float | None = None, b_norm: float | None = None) -> float:
    if _vector_len(a) <= 0 or _vector_len(b) <= 0 or _vector_len(a) != _vector_len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = _vector_norm(a) if a_norm is None else float(a_norm)
    nb = _vector_norm(b) if b_norm is None else float(b_norm)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _match_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, cond in (where or {}).items():
        if key == "$and":
            if not isinstance(cond, list):
                return False
            children = cond
            if any(not isinstance(child, dict) or not _match_where(metadata, child) for child in children):
                return False
            continue
        if key == "$or":
            children = cond if isinstance(cond, list) else []
            if not children or not any(isinstance(child, dict) and _match_where(metadata, child) for child in children):
                return False
            continue
        if not _match_field(metadata.get(key), cond):
            return False
    return True


def _match_field(value: Any, cond: Any) -> bool:
    if not isinstance(cond, dict):
        return value == cond
    for op, expected in cond.items():
        if op == "$gte":
            if not _compare(value, expected, lambda left, right: left >= right):
                return False
        elif op == "$lte":
            if not _compare(value, expected, lambda left, right: left <= right):
                return False
        elif op == "$in":
            allowed = set(expected or [])
            if isinstance(value, (list, tuple, set)):
                if not any(item in allowed for item in value):
                    return False
            elif value not in allowed:
                return False
        elif op == "$nin":
            blocked = set(expected or [])
            if isinstance(value, (list, tuple, set)):
                if any(item in blocked for item in value):
                    return False
            elif value in blocked:
                return False
        elif op == "$ne":
            if value == expected:
                return False
        elif op == "$eq":
            if value != expected:
                return False
        else:
            return False
    return True


def _compare(value: Any, expected: Any, predicate) -> bool:
    if value is None:
        return False
    try:
        return bool(predicate(value, expected))
    except TypeError:
        return False


def _keyword_doc_text(document: str, metadata: dict[str, Any]) -> str:
    extra = " ".join(
        str(metadata.get(field_key, "") or "")
        for field_key in (
            "semantic_tags_text",
            "memory_entity_text",
            "memory_topic_text",
            "memory_facets_text",
            "memory_about_roles_text",
            "memory_mood_tags_text",
        )
    )
    return f"{document} {extra}".strip()


def _candidate_entries(
    entries: list[dict[str, Any]], *, where: dict[str, Any], excluded: set[str]
) -> list[dict[str, Any]]:
    return [entry for entry in entries if entry["source_id"] not in excluded and _match_where(entry["metadata"], where)]


class InMemoryVectorIndex(VectorIndex):
    def __init__(self, *, embedding: EmbeddingProvider) -> None:
        self.embedding = embedding
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()  # 后台压缩线程会写入,保证线程安全

    def upsert(self, entries: list[dict[str, Any]]) -> None:
        prepared = []
        for entry in entries:
            source_id = str(entry.get("source_id") or "").strip()
            if not source_id:
                continue
            text = str(entry.get("text") or "")
            metadata = dict(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {})
            metadata["source_id"] = source_id
            vector = _compact_vector(self.embedding.embed_text(text))
            # embed 在锁外算(可能较慢),只在写 dict 时加锁。
            prepared.append(
                (
                    source_id,
                    {
                        "source_id": source_id,
                        "document": text,
                        "metadata": dict(metadata),
                        "vector": vector,
                        "vector_norm": _vector_norm(vector),
                    },
                )
            )
        with self._lock:
            for source_id, record in prepared:
                self._entries[source_id] = record

    def semantic_search(
        self,
        *,
        query_text: str,
        where: dict[str, Any],
        n_results: int = 8,
        exclude_source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_vec = _compact_vector(self.embedding.embed_text(str(query_text or "")))
        query_norm = _vector_norm(query_vec)
        excluded = {str(source_id) for source_id in (exclude_source_ids or [])}
        with self._lock:
            snapshot = list(self._entries.values())
        candidates = _candidate_entries(snapshot, where=where, excluded=excluded)
        numpy_hits = self._semantic_hits_numpy(query_vec, candidates)
        if numpy_hits is not None:
            numpy_hits.sort(key=lambda item: item["semantic_score"], reverse=True)
            return numpy_hits[: max(1, int(n_results))]
        hits: list[dict[str, Any]] = []
        for entry in candidates:
            hits.append(
                {
                    "source_id": entry["source_id"],
                    "document": entry["document"],
                    "metadata": entry["metadata"],
                    "semantic_score": max(
                        0.0,
                        _cosine(query_vec, entry["vector"], a_norm=query_norm, b_norm=entry.get("vector_norm")),
                    ),
                }
            )
        hits.sort(key=lambda item: item["semantic_score"], reverse=True)
        return hits[: max(1, int(n_results))]

    @staticmethod
    def _semantic_hits_numpy(query_vec: Any, candidates: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if _np is None or _vector_len(query_vec) <= 0 or not candidates:
            return None
        try:
            query = _np.asarray(query_vec, dtype=_np.float32)
            if query.ndim != 1 or query.size == 0:
                return None
            vectors = [entry["vector"] for entry in candidates]
            if any(_vector_len(vector) != query.size for vector in vectors):
                return None
            matrix = _np.asarray(vectors, dtype=_np.float32)
            if matrix.ndim != 2:
                return None
            query_norm = float(_np.linalg.norm(query))
            norm_values: list[float] = []
            for entry in candidates:
                norm = entry.get("vector_norm")
                norm_values.append(float(_vector_norm(entry["vector"]) if norm is None else norm))
            vector_norms = _np.asarray(norm_values, dtype=_np.float32)
            denom = vector_norms * query_norm
            dots = matrix @ query
            scores = _np.divide(dots, denom, out=_np.zeros_like(dots, dtype=_np.float32), where=denom != 0)
            scores = _np.maximum(scores, 0.0)
        except (TypeError, ValueError):
            return None
        return [
            {
                "source_id": entry["source_id"],
                "document": entry["document"],
                "metadata": entry["metadata"],
                "semantic_score": float(score),
            }
            for entry, score in zip(candidates, scores.tolist())
        ]

    def keyword_search(
        self,
        *,
        query_text: str,
        entity_anchors: list[str],
        topic_terms: list[str],
        where: dict[str, Any],
        n_results: int = 8,
        exclude_source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        excluded = {str(source_id) for source_id in (exclude_source_ids or [])}
        with self._lock:
            snapshot = list(self._entries.values())
        candidates = _candidate_entries(snapshot, where=where, excluded=excluded)
        if not candidates:
            return []

        base_terms = tokenize(query_text)
        entity_terms = [
            term for entity in (entity_anchors or []) if str(entity).strip() for term in tokenize(str(entity))
        ]
        topic_query_terms = [
            term for topic in (topic_terms or []) if str(topic).strip() for term in tokenize(str(topic))
        ]
        query_terms = [*base_terms, *entity_terms, *entity_terms, *entity_terms, *topic_query_terms]
        if not query_terms:
            return []

        doc_terms: dict[str, list[str]] = {}
        doc_freq: Counter[str] = Counter()
        for entry in candidates:
            terms = tokenize(_keyword_doc_text(entry["document"], entry["metadata"]))
            doc_terms[entry["source_id"]] = terms
            for term in set(terms):
                doc_freq[term] += 1
        avgdl = sum(len(t) for t in doc_terms.values()) / max(1, len(doc_terms))

        hits: list[dict[str, Any]] = []
        for entry in candidates:
            terms = doc_terms[entry["source_id"]]
            score = self._bm25(query_terms, terms, len(terms), avgdl, len(candidates), doc_freq)
            if score <= 0:
                continue
            hits.append(
                {
                    "source_id": entry["source_id"],
                    "document": entry["document"],
                    "metadata": entry["metadata"],
                    "tag_score": float(score),
                }
            )
        hits.sort(key=lambda item: item["tag_score"], reverse=True)
        return hits[: max(1, int(n_results))]

    def count_candidates(
        self,
        *,
        where: dict[str, Any],
        exclude_source_ids: list[str] | None = None,
    ) -> int:
        excluded = {str(source_id) for source_id in (exclude_source_ids or [])}
        with self._lock:
            snapshot = list(self._entries.values())
        return len(_candidate_entries(snapshot, where=where, excluded=excluded))

    @staticmethod
    def _bm25(
        query_terms: list[str],
        doc_terms: list[str],
        doc_len: int,
        avgdl: float,
        doc_count: int,
        doc_freq: Counter[str],
    ) -> float:
        if not doc_terms:
            return 0.0
        tf = Counter(doc_terms)
        k1, b = 1.5, 0.75
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            df = max(1, doc_freq.get(term, 0))
            idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / max(1e-6, avgdl)))
            score += idf * (numerator / max(1e-6, denominator))
        return score

    def delete(self, source_ids: list[str]) -> None:
        with self._lock:
            for source_id in source_ids:
                self._entries.pop(str(source_id), None)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)
