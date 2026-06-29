"""InMemoryVectorIndex —— 纯 Python 默认/测试后端。

实现 VectorIndex 全部能力(语义 cosine + 关键词 BM25 + where 硬过滤),不依赖任何外部服务,
让检索主干(混合 + RRF + 放宽)在无 chroma/无网络下可单测。生产可换 ChromaVectorIndex 等。

关键词侧文本拼入多维标签(keywords/categories/subjects/moods),让标签双向赋能。
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


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _match_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, cond in (where or {}).items():
        value = metadata.get(key)
        if isinstance(cond, dict):
            if "$gte" in cond and not (value is not None and value >= cond["$gte"]):
                return False
            if "$lte" in cond and not (value is not None and value <= cond["$lte"]):
                return False
            if "$in" in cond and value not in set(cond["$in"] or []):
                return False
            if "$nin" in cond and value in set(cond["$nin"] or []):
                return False
            if "$ne" in cond and value == cond["$ne"]:
                return False
        elif value != cond:
            return False
    return True


def _keyword_doc_text(document: str, metadata: dict[str, Any]) -> str:
    extra = " ".join(
        str(metadata.get(field_key, "") or "")
        for field_key in (
            "semantic_tags_text",
            "memory_keywords_text",
            "memory_categories_text",
            "memory_subject_scopes_text",
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
            # embed 在锁外算(可能较慢),只在写 dict 时加锁。
            prepared.append(
                (
                    source_id,
                    {
                        "source_id": source_id,
                        "document": text,
                        "metadata": dict(metadata),
                        "vector": self.embedding.embed_text(text),
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
        query_vec = self.embedding.embed_text(str(query_text or ""))
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
                    "semantic_score": max(0.0, _cosine(query_vec, entry["vector"])),
                }
            )
        hits.sort(key=lambda item: item["semantic_score"], reverse=True)
        return hits[: max(1, int(n_results))]

    @staticmethod
    def _semantic_hits_numpy(query_vec: list[float], candidates: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if _np is None or not query_vec or not candidates:
            return None
        try:
            query = _np.asarray(query_vec, dtype=float)
            if query.ndim != 1 or query.size == 0:
                return None
            vectors = [entry["vector"] for entry in candidates]
            if any(len(vector) != query.size for vector in vectors):
                return None
            matrix = _np.asarray(vectors, dtype=float)
            if matrix.ndim != 2:
                return None
            query_norm = float(_np.linalg.norm(query))
            vector_norms = _np.linalg.norm(matrix, axis=1)
            denom = vector_norms * query_norm
            dots = matrix @ query
            scores = _np.divide(dots, denom, out=_np.zeros_like(dots, dtype=float), where=denom != 0)
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
        keywords: list[str],
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

        query_terms = [t for t in (keywords or []) if str(t).strip()] or tokenize(query_text)
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
