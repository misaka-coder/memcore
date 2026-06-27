"""RRF(Reciprocal Rank Fusion):融合语义与关键词两路命中。

照搬旧 vector_store.fuse_with_rrf 行为:双命中优先,再按语义分、标签分排序。
"""

from __future__ import annotations

from typing import Any


def fuse_with_rrf(
    semantic_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
    *,
    k: int = 60,
) -> list[dict[str, Any]]:
    semantic_rank = {hit["source_id"]: idx + 1 for idx, hit in enumerate(semantic_hits)}
    keyword_rank = {hit["source_id"]: idx + 1 for idx, hit in enumerate(keyword_hits)}
    semantic_map = {hit["source_id"]: hit for hit in semantic_hits}
    keyword_map = {hit["source_id"]: hit for hit in keyword_hits}

    all_ids: list[str] = []
    for bucket in (semantic_hits, keyword_hits):
        for hit in bucket:
            if hit["source_id"] not in all_ids:
                all_ids.append(hit["source_id"])

    fused: list[dict[str, Any]] = []
    for source_id in all_ids:
        rrf_score = 0.0
        if source_id in semantic_rank:
            rrf_score += 1.0 / (k + semantic_rank[source_id])
        if source_id in keyword_rank:
            rrf_score += 1.0 / (k + keyword_rank[source_id])
        sample = semantic_map.get(source_id) or keyword_map.get(source_id) or {}
        fused.append(
            {
                "source_id": source_id,
                "dual_hit": source_id in semantic_rank and source_id in keyword_rank,
                "entry_type": str((sample.get("metadata") or {}).get("entry_type", "")),
                "rrf_score": float(rrf_score),
                "semantic_score": float((semantic_map.get(source_id) or {}).get("semantic_score", 0.0)),
                "tag_score": float((keyword_map.get(source_id) or {}).get("tag_score", 0.0)),
                "metadata": sample.get("metadata") or {},
                "document": sample.get("document", "") or "",
            }
        )

    fused.sort(key=lambda item: (-item["rrf_score"], -item["semantic_score"], -item["tag_score"]))
    return fused
