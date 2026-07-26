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

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query.

        Symmetric providers may keep the default. Asymmetric providers can
        override this method without leaking provider task names into memcore.
        """
        if type(self).embed_queries is not EmbeddingProvider.embed_queries:
            vectors = self.embed_queries([text])
            if len(vectors) != 1:
                raise RuntimeError(f"embedding provider returned {len(vectors)} query vectors for 1 input")
            return vectors[0]
        return self.embed_text(text)

    def embed_queries(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed retrieval queries as a batch.

        A provider that only overrides ``embed_query`` remains correct. A
        symmetric provider keeps its existing optimized ``embed_texts`` path.
        """
        items = [str(text or "") for text in texts]
        if type(self).embed_query is EmbeddingProvider.embed_query:
            return self.embed_texts(items)
        return [self.embed_query(text) for text in items]

    def embed_document(self, text: str) -> list[float]:
        """Embed one index document/passage."""
        if type(self).embed_documents is not EmbeddingProvider.embed_documents:
            vectors = self.embed_documents([text])
            if len(vectors) != 1:
                raise RuntimeError(f"embedding provider returned {len(vectors)} document vectors for 1 input")
            return vectors[0]
        return self.embed_text(text)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed index documents/passages as a batch.

        A provider that only overrides ``embed_document`` remains correct. A
        symmetric provider keeps its existing optimized ``embed_texts`` path.
        """
        items = [str(text or "") for text in texts]
        if type(self).embed_document is EmbeddingProvider.embed_document:
            return self.embed_texts(items)
        return [self.embed_document(text) for text in items]

    def collection_key(self) -> str:
        """集合命名键:换模型/维度 = 换集合,旧向量需重建索引。"""
        return f"{self.name}_{self.version}_{self.dimension}".lower()
