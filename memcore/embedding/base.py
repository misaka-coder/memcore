"""EmbeddingProvider 接口 —— 把文本变向量(引擎可换)。

后续切片提供:真实语义(HuggingFace / BGE-M3)+ hashed(仅测试 / 显式 degraded)。

⚠️ hashed 政策(设计文档 §13.1):hashed 无语义,**仅**用于单元测试或调用方显式 allow_degraded;
生产默认真实模型加载失败 = 报错 / 上报状态,绝不静默退到 hashed。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class EmbeddingProvider(ABC):
    name: str = "base"
    version: str = "v1"

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def collection_key(self) -> str:
        """集合命名键:换模型/维度 = 换集合,旧向量需重建索引。"""
        return f"{self.name}_{self.version}_{self.dimension}".lower()
