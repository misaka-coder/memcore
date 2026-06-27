"""语义自检 —— 把"语义到底是真的还是塌的"摊在明面上(见设计文档 §13)。

测一对近义词应明显比一对无关词更相似。hashed / 弱模型会过不了,从而响亮暴露降级,
而不是上线后才发现召回烂。
"""

from __future__ import annotations

import math
from typing import Any

from .base import EmbeddingProvider


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def verify_embedding(
    provider: EmbeddingProvider,
    *,
    similar: tuple[str, str] = ("可乐", "饮料"),
    dissimilar: tuple[str, str] = ("可乐", "股票"),
    margin: float = 0.05,
) -> dict[str, Any]:
    """近义对相似度应显著高于无关对。返回结构化报告;ok=False 表示语义可能已降级。"""
    anchor, near = similar
    _, far = dissimilar
    vectors = provider.embed_texts([anchor, near, far])
    sim = _cosine(vectors[0], vectors[1])
    dis = _cosine(vectors[0], vectors[2])
    gap = sim - dis
    ok = sim > 0.0 and gap > margin
    return {
        "ok": ok,
        "provider": provider.name,
        "similar_pair": list(similar),
        "dissimilar_pair": list(dissimilar),
        "similar_score": round(sim, 4),
        "dissimilar_score": round(dis, 4),
        "gap": round(gap, 4),
        "margin": margin,
        "reason": ""
        if ok
        else "semantic separation too weak — likely hashed/degraded or a weak model; do not rely on semantic recall",
    }
