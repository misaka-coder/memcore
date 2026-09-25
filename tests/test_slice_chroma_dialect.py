"""Chroma where 方言适配:翻译器纯逻辑(始终跑)+ 真 Chroma 集成(装了 chroma 才跑)。

修复点:memcore 的 where(多顶层键隐式 AND、单字段双操作符)直接喂 Chroma(0.5+)会被严格校验器拒。
翻译器把它转成 {"$and":[...]}、并把 $gte+$lte 拆成两个子句。
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest

from memcore.embedding.base import EmbeddingProvider
from memcore.index.chroma_index import ChromaVectorIndex
from memcore.index.chroma_index import _to_chroma_where, _with_source_excludes

_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None


class _RoleAwareEmbedding(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0]

    def embed_documents(self, texts) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


class _FakeCollection:
    def __init__(self) -> None:
        self.upsert_payload = {}
        self.query_payload = {}

    def upsert(self, **kwargs) -> None:
        self.upsert_payload = kwargs

    def query(self, **kwargs):
        self.query_payload = kwargs
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


class WhereTranslation(unittest.TestCase):
    def test_empty_is_none(self) -> None:
        self.assertIsNone(_to_chroma_where(None))
        self.assertIsNone(_to_chroma_where({}))

    def test_single_scalar_no_and(self) -> None:
        self.assertEqual(_to_chroma_where({"user_id": "u1"}), {"user_id": "u1"})

    def test_multi_keys_wrapped_in_and(self) -> None:
        out = _to_chroma_where({"tenant_id": "t", "user_id": "u1", "domain_id": "d"})
        self.assertIn("$and", out)
        self.assertEqual(out["$and"], [{"tenant_id": "t"}, {"user_id": "u1"}, {"domain_id": "d"}])

    def test_range_double_operator_split(self) -> None:
        # 单字段 $gte+$lte 必须拆成两个子句(Chroma 一个字段只能一个操作符)。
        out = _to_chroma_where({"timestamp": {"$gte": 50, "$lte": 150}})
        self.assertEqual(out, {"$and": [{"timestamp": {"$gte": 50}}, {"timestamp": {"$lte": 150}}]})

    def test_start_inclusive_end_exclusive_range_is_preserved(self) -> None:
        out = _to_chroma_where({"timestamp": {"$gte": 50, "$lt": 150}})
        self.assertEqual(out, {"$and": [{"timestamp": {"$gte": 50}}, {"timestamp": {"$lt": 150}}]})

    def test_mixed_scalar_and_range(self) -> None:
        out = _to_chroma_where({"user_id": "u1", "timestamp": {"$gte": 50, "$lte": 150}})
        self.assertEqual(
            out,
            {"$and": [{"user_id": "u1"}, {"timestamp": {"$gte": 50}}, {"timestamp": {"$lte": 150}}]},
        )

    def test_chroma_uses_document_and_query_embedding_roles(self) -> None:
        index = ChromaVectorIndex.__new__(ChromaVectorIndex)
        index.embedding = _RoleAwareEmbedding()
        index._collection = _FakeCollection()

        index.upsert([{"source_id": "m1", "text": "历史文档", "metadata": {"user_id": "u1"}}])
        index.semantic_search(query_text="查询问题", where={"user_id": "u1"})

        self.assertEqual(index._collection.upsert_payload["embeddings"], [[1.0, 0.0]])
        self.assertEqual(index._collection.query_payload["query_embeddings"], [[0.0, 1.0]])

    def test_source_exclude_nin_kept_as_operator_clause(self) -> None:
        out = _to_chroma_where({"user_id": "u1", "source_id": {"$nin": ["m1", "m2"]}})
        self.assertEqual(out, {"$and": [{"user_id": "u1"}, {"source_id": {"$nin": ["m1", "m2"]}}]})

    def test_source_excludes_do_not_overwrite_lineage_include_scope(self) -> None:
        scoped = _with_source_excludes(
            {"user_id": "u1", "source_id": {"$in": ["episode", "raw"]}},
            ["episode", "visible"],
        )

        self.assertEqual(
            _to_chroma_where(scoped),
            {
                "$and": [
                    {"user_id": "u1"},
                    {"source_id": {"$in": ["episode", "raw"]}},
                    {"source_id": {"$nin": ["episode", "visible"]}},
                ]
            },
        )

    def test_or_clause_is_preserved(self) -> None:
        out = _to_chroma_where({"$or": [{"memory_facet__preference": True}, {"memory_facet__plan": True}]})
        self.assertEqual(
            out,
            {"$or": [{"memory_facet__preference": True}, {"memory_facet__plan": True}]},
        )

    def test_empty_or_is_not_silently_dropped(self) -> None:
        self.assertEqual(_to_chroma_where({"$or": []}), {"__memcore_never_match__": True})

    def test_mixed_scalar_and_logic_becomes_top_level_and(self) -> None:
        out = _to_chroma_where(
            {
                "user_id": "u1",
                "$and": [
                    {"$or": [{"memory_facet__preference": True}, {"memory_facet__plan": True}]},
                    {"memory_about_role__user": True},
                ],
            }
        )
        self.assertEqual(
            out,
            {
                "$and": [
                    {"user_id": "u1"},
                    {"$or": [{"memory_facet__preference": True}, {"memory_facet__plan": True}]},
                    {"memory_about_role__user": True},
                ]
            },
        )


@unittest.skipUnless(_HAS_CHROMA, "chromadb not installed (pip install memcore-kernel[chroma])")
class ChromaIntegration(unittest.TestCase):
    def setUp(self) -> None:
        from memcore import HashedEmbeddingProvider
        from memcore.index.chroma_index import ChromaVectorIndex

        self.tmp = tempfile.mkdtemp(prefix="memcore-chroma-")
        self.index = ChromaVectorIndex(base_dir=self.tmp, embedding=HashedEmbeddingProvider())
        self.index.upsert(
            [
                {
                    "source_id": "m1",
                    "text": "主人喜欢喝可乐",
                    "metadata": {
                        "tenant_id": "t",
                        "user_id": "u1",
                        "domain_id": "d",
                        "timestamp": 100,
                        "memory_entity_text": "可乐 饮料",
                        "entry_type": "raw",
                    },
                },
                {
                    "source_id": "m2",
                    "text": "另一个用户的私密记录",
                    "metadata": {
                        "tenant_id": "t",
                        "user_id": "u2",
                        "domain_id": "d",
                        "timestamp": 100,
                        "entry_type": "raw",
                    },
                },
            ]
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_multi_key_and_range_filter_does_not_crash_and_isolates(self) -> None:
        # 修复前:多顶层键 + $gte/$lte 同框 → Chroma 校验器直接抛错。
        where = {"tenant_id": "t", "user_id": "u1", "domain_id": "d", "timestamp": {"$gte": 50, "$lte": 150}}
        sem = self.index.semantic_search(query_text="可乐", where=where, n_results=8)
        ids = {h["source_id"] for h in sem}
        self.assertIn("m1", ids)
        self.assertNotIn("m2", ids)  # u2 硬隔离,不串给 u1

    def test_keyword_search_with_multi_key_where(self) -> None:
        where = {"tenant_id": "t", "user_id": "u1", "domain_id": "d"}
        hits = self.index.keyword_search(query_text="饮料", entity_anchors=["饮料"], topic_terms=[], where=where)
        self.assertEqual(hits[0]["source_id"], "m1")  # 标签命中,且只在 u1 范围

    def test_count_candidates_uses_same_filter_and_exclusions(self) -> None:
        where = {"tenant_id": "t", "user_id": "u1", "domain_id": "d"}

        self.assertEqual(self.index.count_candidates(where=where), 1)
        self.assertEqual(self.index.count_candidates(where=where, exclude_source_ids=["m1"]), 0)

    def test_lineage_include_and_visible_exclude_are_both_enforced(self) -> None:
        where = {
            "tenant_id": "t",
            "user_id": "u1",
            "domain_id": "d",
            "source_id": {"$in": ["m1", "m2"]},
        }

        self.assertEqual(self.index.count_candidates(where=where), 1)
        self.assertEqual(self.index.count_candidates(where=where, exclude_source_ids=["m1"]), 0)


if __name__ == "__main__":
    unittest.main()
