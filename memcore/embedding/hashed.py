"""HashedEmbeddingProvider —— 无语义的占位实现。

⚠️ 仅用于单元测试或调用方**显式** allow_degraded;生产绝不静默回退到它(见 §13.1)。
sha256 把 token 散列进固定维度,只能匹配字面相同的词,**没有语义**(可乐≠饮料)。
"""

from __future__ import annotations

import hashlib
import math

from ..text_utils import tokenize
from .base import EmbeddingProvider


class HashedEmbeddingProvider(EmbeddingProvider):
    name = "hashed"
    version = "v1"

    def __init__(self, *, dimension: int = 128) -> None:
        self._dimension = max(1, int(dimension))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (len(token) / 10.0)
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector
