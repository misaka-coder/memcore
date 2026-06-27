"""HTTP embedding 适配器(OpenAI 兼容)+ 语义自检 verify_embedding。

不打真实网络:HTTPEmbeddingProvider 子类覆盖 _call_api 注入确定性向量。
"""

from __future__ import annotations

import unittest

from memcore import (
    HashedEmbeddingProvider,
    HTTPEmbeddingProvider,
    InMemoryVectorIndex,
    MemorySystem,
    Namespace,
    verify_embedding,
)
from memcore.embedding.base import EmbeddingProvider
from memcore.llm.base import LLMClient, LLMRequest, LLMResult


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


class _FakeHTTP(HTTPEmbeddingProvider):
    """用确定性向量替代真实 API:近义词更近,无关词远。"""

    _TABLE = {
        "可乐": [1.0, 0.0, 0.0],
        "饮料": [0.95, 0.05, 0.0],
        "股票": [0.0, 0.0, 1.0],
    }

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        return [self._TABLE.get(t, [0.0, 0.0, 0.0]) for t in texts]


class _GoodSemantic(EmbeddingProvider):
    name = "fake-good"
    version = "v1"
    _TABLE = {"可乐": [1.0, 0.0, 0.0], "饮料": [0.9, 0.1, 0.0], "股票": [0.0, 0.0, 1.0]}

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        return self._TABLE.get(text, [0.0, 0.0, 0.0])


class HTTPProvider(unittest.TestCase):
    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            HTTPEmbeddingProvider(base_url="", api_key="k", model="m", dimension=3)
        with self.assertRaises(ValueError):
            HTTPEmbeddingProvider(base_url="http://x/v1", api_key="k", model="m", dimension=0)

    def test_embed_and_dimension(self) -> None:
        p = _FakeHTTP(base_url="http://x/v1", api_key="k", model="m", dimension=3)
        self.assertEqual(p.dimension, 3)
        self.assertEqual(p.embed_text("可乐"), [1.0, 0.0, 0.0])
        self.assertEqual(len(p.embed_texts(["可乐", "股票"])), 2)
        self.assertTrue(p.name.startswith("http:"))

    def test_dimension_mismatch_raises(self) -> None:
        p = _FakeHTTP(base_url="http://x/v1", api_key="k", model="m", dimension=5)  # 真实返回 3 维
        with self.assertRaises(RuntimeError):
            p.embed_text("可乐")

    def test_integrates_with_index(self) -> None:
        p = _FakeHTTP(base_url="http://x/v1", api_key="k", model="m", dimension=3)
        idx = InMemoryVectorIndex(embedding=p)
        idx.upsert([{"source_id": "1", "text": "可乐", "metadata": {"user_id": "u1"}}])
        hits = idx.semantic_search(query_text="饮料", where={"user_id": "u1"})
        self.assertEqual(hits[0]["source_id"], "1")  # 饮料≈可乐,能召回


class SemanticVerify(unittest.TestCase):
    def test_good_provider_passes(self) -> None:
        report = verify_embedding(_GoodSemantic())
        self.assertTrue(report["ok"])
        self.assertGreater(report["similar_score"], report["dissimilar_score"])

    def test_hashed_flagged_as_degraded(self) -> None:
        report = verify_embedding(HashedEmbeddingProvider())
        self.assertFalse(report["ok"])  # 无语义:近义词拉不开
        self.assertIn("degraded", report["reason"])

    def test_facade_verify_embedding(self) -> None:
        mem = MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1"),
            timezone="UTC",
            embedding=_GoodSemantic(),
        )
        self.assertTrue(mem.verify_embedding()["ok"])


if __name__ == "__main__":
    unittest.main()
