"""VectorIndex 接口 —— 相似侧(语义 + 关键词,引擎可换)。

默认实现 ChromaVectorIndex 在后续切片提供。关键词检索(BM25/FTS)也进此接口,
让 Postgres 全文索引 / Qdrant 原生检索能各自实现,顺带解决全量扫描的扩展性问题。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, entries: list[dict[str, Any]]) -> None:
        """写入/更新条目(每条含 source_id / text / metadata)。"""
        raise NotImplementedError

    @abstractmethod
    def semantic_search(
        self,
        *,
        query_text: str,
        where: dict[str, Any],
        n_results: int = 8,
        exclude_source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """向量相似检索;where 为硬过滤(隔离 + 时间),exclude 为候选前排除。返回含 source_id / semantic_score。"""
        raise NotImplementedError

    @abstractmethod
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
        """关键词/BM25 检索(文本含多维标签),exclude 为候选前排除;返回含 source_id / tag_score。"""
        raise NotImplementedError

    def count_candidates(
        self,
        *,
        where: dict[str, Any],
        exclude_source_ids: list[str] | None = None,
    ) -> int:
        """Count the exact pre-scoring candidate set selected by ``where``."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, source_ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
