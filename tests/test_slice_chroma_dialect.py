"""Chroma where 方言适配:翻译器纯逻辑(始终跑)+ 真 Chroma 集成(装了 chroma 才跑)。

修复点:memcore 的 where(多顶层键隐式 AND、单字段双操作符)直接喂 Chroma(0.5+)会被严格校验器拒。
翻译器把它转成 {"$and":[...]}、并把 $gte+$lte 拆成两个子句。
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest

from memcore.index.chroma_index import _to_chroma_where

_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None


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

    def test_mixed_scalar_and_range(self) -> None:
        out = _to_chroma_where({"user_id": "u1", "timestamp": {"$gte": 50, "$lte": 150}})
        self.assertEqual(
            out,
            {"$and": [{"user_id": "u1"}, {"timestamp": {"$gte": 50}}, {"timestamp": {"$lte": 150}}]},
        )


@unittest.skipUnless(_HAS_CHROMA, "chromadb not installed (pip install memcore[chroma])")
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
                        "memory_keywords_text": "可乐 饮料",
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
        hits = self.index.keyword_search(query_text="饮料", keywords=["饮料"], where=where)
        self.assertEqual(hits[0]["source_id"], "m1")  # 标签命中,且只在 u1 范围


if __name__ == "__main__":
    unittest.main()
