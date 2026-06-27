"""ChromaVectorIndex —— 基于 Chroma 的生产实现(可选依赖 extras: chroma)。

延迟导入 chromadb;关键词侧仍为内存 BM25(后续可换后端原生 FTS)。
本切片提供实现骨架,重负载路径不进单测(无 chroma 时不影响其余模块)。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..embedding.base import EmbeddingProvider
from ..text_utils import tokenize
from .base import VectorIndex
from .memory_index import _keyword_doc_text


def _to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """把 memcore 的 where(多顶层键隐式 AND、单字段多操作符)翻成 Chroma 严格语法。

    Chroma(0.5+)要求:多条件必须显式 $and;一个字段 dict 只能含一个操作符。
    例:{"u":"x","ts":{"$gte":a,"$lte":b}} → {"$and":[{"u":"x"},{"ts":{"$gte":a}},{"ts":{"$lte":b}}]}
    """
    if not where:
        return None
    clauses: list[dict[str, Any]] = []
    for key, cond in where.items():
        if isinstance(cond, dict):
            for op, val in cond.items():  # 每个操作符拆成独立子句
                clauses.append({key: {op: val}})
        else:
            clauses.append({key: cond})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorIndex(VectorIndex):
    def __init__(self, *, base_dir: str, embedding: EmbeddingProvider) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb not installed; `pip install memcore[chroma]`") from exc
        self.embedding = embedding
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(base))
        self._collection = self._client.get_or_create_collection(
            name=f"memcore_{embedding.collection_key()}",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, entries: list[dict[str, Any]]) -> None:
        ids, docs, metas = [], [], []
        for entry in entries:
            source_id = str(entry.get("source_id") or "").strip()
            if not source_id:
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            ids.append(source_id)
            docs.append(str(entry.get("text") or ""))
            metas.append({k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))})
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=self.embedding.embed_texts(docs),
            metadatas=metas,
        )

    def semantic_search(self, *, query_text: str, where: dict[str, Any], n_results: int = 8) -> list[dict[str, Any]]:
        result = self._collection.query(
            query_embeddings=self.embedding.embed_texts([str(query_text or "")]),
            n_results=max(1, int(n_results)),
            where=_to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        for idx, source_id in enumerate(result.get("ids", [[]])[0]):
            metadata = (result.get("metadatas", [[]])[0] or [{}])[idx] or {}
            document = (result.get("documents", [[]])[0] or [""])[idx] or ""
            distance = float((result.get("distances", [[]])[0] or [1.0])[idx] or 1.0)
            hits.append(
                {
                    "source_id": source_id,
                    "document": document,
                    "metadata": metadata,
                    "semantic_score": max(0.0, 1.0 - distance),
                }
            )
        return hits

    def keyword_search(
        self,
        *,
        query_text: str,
        keywords: list[str],
        where: dict[str, Any],
        n_results: int = 8,
    ) -> list[dict[str, Any]]:
        import math

        got = self._collection.get(where=_to_chroma_where(where), include=["documents", "metadatas"])
        ids = got.get("ids") or []
        documents = got.get("documents") or []
        metadatas = got.get("metadatas") or []
        if not ids:
            return []
        query_terms = [t for t in (keywords or []) if str(t).strip()] or tokenize(query_text)
        if not query_terms:
            return []

        doc_terms: dict[str, list[str]] = {}
        doc_freq: Counter[str] = Counter()
        for source_id, document, metadata in zip(ids, documents, metadatas):
            terms = tokenize(_keyword_doc_text(str(document or ""), metadata or {}))
            doc_terms[source_id] = terms
            for term in set(terms):
                doc_freq[term] += 1
        avgdl = sum(len(t) for t in doc_terms.values()) / max(1, len(doc_terms))

        hits: list[dict[str, Any]] = []
        for source_id, document, metadata in zip(ids, documents, metadatas):
            terms = doc_terms[source_id]
            tf = Counter(terms)
            k1, b = 1.5, 0.75
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                df = max(1, doc_freq.get(term, 0))
                idf = math.log(1 + ((len(ids) - df + 0.5) / (df + 0.5)))
                score += idf * (freq * (k1 + 1) / max(1e-6, freq + k1 * (1 - b + b * (len(terms) / max(1e-6, avgdl)))))
            if score > 0:
                hits.append(
                    {
                        "source_id": source_id,
                        "document": document or "",
                        "metadata": metadata or {},
                        "tag_score": float(score),
                    }
                )
        hits.sort(key=lambda item: item["tag_score"], reverse=True)
        return hits[: max(1, int(n_results))]

    def delete(self, source_ids: list[str]) -> None:
        if source_ids:
            self._collection.delete(ids=[str(s) for s in source_ids])

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0
