"""HTTP embedding 适配器(OpenAI 兼容)+ 语义自检 verify_embedding。

不打真实网络:HTTPEmbeddingProvider 子类覆盖 _call_api 注入确定性向量。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch
import urllib.error

from memcore import (
    HashedEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    HTTPEmbeddingProvider,
    InMemoryVectorIndex,
    MemorySystem,
    Namespace,
    RoleAwareHTTPEmbeddingProvider,
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


class _FakeRoleHTTP(RoleAwareHTTPEmbeddingProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.payloads: list[dict[str, object]] = []

    def _request_api(self, payload) -> list[list[float]]:
        self.payloads.append(dict(payload))
        task = str(payload.get("task") or "")
        vector = [1.0, 0.0, 0.0] if task.endswith("passage") else [0.0, 1.0, 0.0]
        return [list(vector) for _ in payload["input"]]


class _GoodSemantic(EmbeddingProvider):
    name = "fake-good"
    version = "v1"
    _TABLE = {"可乐": [1.0, 0.0, 0.0], "饮料": [0.9, 0.1, 0.0], "股票": [0.0, 0.0, 1.0]}

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        return self._TABLE.get(text, [0.0, 0.0, 0.0])


class _AsymmetricSemantic(EmbeddingProvider):
    name = "fake-asymmetric"
    version = "v1"

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.generic_calls: list[str] = []

    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        self.generic_calls.append(text)
        return [0.0, 1.0]

    def embed_documents(self, texts) -> list[list[float]]:
        items = [str(text or "") for text in texts]
        self.document_batches.append(items)
        return [[1.0, 0.0] for _ in items]

    def embed_document(self, text: str) -> list[float]:
        self.documents.append(text)
        return [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0, 1.0] if text == "无关" else [1.0, 0.0]

    def embed_queries(self, texts) -> list[list[float]]:
        return [self.embed_query(str(text or "")) for text in texts]


class _SingleRoleOnlySemantic(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_document(self, text: str) -> list[float]:
        return [0.0, 1.0]


class _BatchRoleOnlySemantic(EmbeddingProvider):
    def __init__(self) -> None:
        self.query_batches: list[list[str]] = []
        self.document_batches: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0]

    def embed_queries(self, texts) -> list[list[float]]:
        items = [str(text or "") for text in texts]
        self.query_batches.append(items)
        return [[1.0, 0.0] for _ in items]

    def embed_documents(self, texts) -> list[list[float]]:
        items = [str(text or "") for text in texts]
        self.document_batches.append(items)
        return [[0.0, 1.0] for _ in items]


class HTTPProvider(unittest.TestCase):
    def test_huggingface_provider_is_public_without_loading_optional_dependency(self) -> None:
        self.assertTrue(issubclass(HuggingFaceEmbeddingProvider, EmbeddingProvider))

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

    def test_role_aware_http_keeps_vendor_task_mapping_outside_index(self) -> None:
        provider = _FakeRoleHTTP(
            base_url="https://example.test/v1",
            api_key="secret",
            model="asymmetric",
            dimension=3,
            name="host-provider",
            common_body={"normalized": True},
            query_body={"task": "vendor.query"},
            document_body={"task": "vendor.passage"},
        )

        documents = provider.embed_documents(["历史一", "历史二"])
        query = provider.embed_query("问题")

        self.assertEqual(documents, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(query, [0.0, 1.0, 0.0])
        self.assertEqual(
            provider.payloads,
            [
                {
                    "input": ["历史一", "历史二"],
                    "model": "asymmetric",
                    "normalized": True,
                    "task": "vendor.passage",
                },
                {
                    "input": ["问题"],
                    "model": "asymmetric",
                    "normalized": True,
                    "task": "vendor.query",
                },
            ],
        )
        self.assertIn("http-role-v1-", provider.version)

    def test_role_aware_http_identity_changes_when_vector_space_changes(self) -> None:
        common = {
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "model": "asymmetric",
            "dimension": 3,
        }
        first = RoleAwareHTTPEmbeddingProvider(
            **common,
            query_body={"task": "vendor.query"},
            document_body={"task": "vendor.passage"},
        )
        second = RoleAwareHTTPEmbeddingProvider(
            **common,
            query_body={"task": "vendor.query.v2"},
            document_body={"task": "vendor.passage"},
        )

        self.assertNotEqual(first.collection_key(), second.collection_key())

    def test_role_aware_http_rejects_protected_body_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected fields"):
            RoleAwareHTTPEmbeddingProvider(
                base_url="https://example.test/v1",
                api_key="secret",
                model="asymmetric",
                dimension=3,
                query_body={"model": "other"},
            )

    def test_http_provider_returns_structured_status_without_leaking_response_body(self) -> None:
        provider = HTTPEmbeddingProvider(
            base_url="https://example.test/v1",
            api_key="secret",
            model="model",
            dimension=3,
        )
        error = urllib.error.HTTPError(
            url="https://example.test/v1/embeddings",
            code=401,
            msg="unauthorized",
            hdrs=None,
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "^embedding_http_error:401$"):
                provider.embed_text("测试")

    def test_http_provider_restores_response_order_from_indices(self) -> None:
        provider = HTTPEmbeddingProvider(
            base_url="https://example.test/v1",
            api_key="secret",
            model="model",
            dimension=2,
        )

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "data": [
                            {"index": 1, "embedding": [0.0, 1.0]},
                            {"index": 0, "embedding": [1.0, 0.0]},
                        ]
                    }
                ).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=_Response()):
            self.assertEqual(provider.embed_texts(["一", "二"]), [[1.0, 0.0], [0.0, 1.0]])

    def test_integrates_with_index(self) -> None:
        p = _FakeHTTP(base_url="http://x/v1", api_key="k", model="m", dimension=3)
        idx = InMemoryVectorIndex(embedding=p)
        idx.upsert([{"source_id": "1", "text": "可乐", "metadata": {"user_id": "u1"}}])
        hits = idx.semantic_search(query_text="饮料", where={"user_id": "u1"})
        self.assertEqual(hits[0]["source_id"], "1")  # 饮料≈可乐,能召回

    def test_index_uses_document_batch_and_query_roles(self) -> None:
        provider = _AsymmetricSemantic()
        idx = InMemoryVectorIndex(embedding=provider)

        idx.upsert(
            [
                {"source_id": "1", "text": "第一条记忆", "metadata": {"user_id": "u1"}},
                {"source_id": "2", "text": "第二条记忆", "metadata": {"user_id": "u1"}},
            ]
        )
        hits = idx.semantic_search(query_text="模糊查询", where={"user_id": "u1"})

        self.assertEqual(provider.document_batches, [["第一条记忆", "第二条记忆"]])
        self.assertEqual(provider.queries, ["模糊查询"])
        self.assertEqual(provider.generic_calls, [])
        self.assertEqual({hit["source_id"] for hit in hits}, {"1", "2"})

    def test_legacy_symmetric_provider_keeps_working(self) -> None:
        provider = _GoodSemantic()
        idx = InMemoryVectorIndex(embedding=provider)
        idx.upsert([{"source_id": "1", "text": "可乐", "metadata": {"user_id": "u1"}}])

        hits = idx.semantic_search(query_text="饮料", where={"user_id": "u1"})

        self.assertEqual(hits[0]["source_id"], "1")

    def test_single_role_overrides_are_respected_by_batch_defaults(self) -> None:
        provider = _SingleRoleOnlySemantic()

        self.assertEqual(provider.embed_queries(["q1", "q2"]), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(provider.embed_documents(["d1", "d2"]), [[0.0, 1.0], [0.0, 1.0]])

    def test_batch_role_overrides_are_respected_by_single_defaults(self) -> None:
        provider = _BatchRoleOnlySemantic()

        self.assertEqual(provider.embed_query("q1"), [1.0, 0.0])
        self.assertEqual(provider.embed_document("d1"), [0.0, 1.0])
        self.assertEqual(provider.query_batches, [["q1"]])
        self.assertEqual(provider.document_batches, [["d1"]])


class SemanticVerify(unittest.TestCase):
    def test_good_provider_passes(self) -> None:
        report = verify_embedding(_GoodSemantic())
        self.assertTrue(report["ok"])
        self.assertGreater(report["similar_score"], report["dissimilar_score"])

    def test_hashed_flagged_as_degraded(self) -> None:
        report = verify_embedding(HashedEmbeddingProvider())
        self.assertFalse(report["ok"])  # 无语义:近义词拉不开
        self.assertIn("degraded", report["reason"])

    def test_verify_exercises_retrieval_roles(self) -> None:
        provider = _AsymmetricSemantic()

        report = verify_embedding(
            provider,
            similar=("锚点", "近义"),
            dissimilar=("锚点", "无关"),
        )

        self.assertTrue(report["ok"])
        self.assertEqual(provider.document_batches, [["锚点"]])
        self.assertEqual(provider.queries, ["近义", "无关"])
        self.assertEqual(provider.generic_calls, [])

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
