"""Production embedding adapter failure, identity, and health-probe boundaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from memcore import HashedEmbeddingProvider

from examples.research_pilot.production_embedding import (
    PROBE_TEXTS,
    ProductionEmbeddingError,
    create_production_embedding,
    verify_production_embedding,
)


class ProductionEmbeddingTests(unittest.TestCase):
    @staticmethod
    def delegate() -> Mock:
        provider = Mock()
        provider.dimension = 2
        values = {PROBE_TEXTS[0]: [1.0, 0.0], PROBE_TEXTS[1]: [0.8, 0.6], PROBE_TEXTS[2]: [0.0, 1.0]}
        provider.embed_texts.side_effect = lambda texts: [values[text] for text in texts]
        return provider

    def test_factory_is_local_only_and_local_path_never_becomes_public_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "examples.research_pilot.production_embedding.HuggingFaceEmbeddingProvider",
                return_value=self.delegate(),
            ) as loader:
                provider = create_production_embedding(local_model_path=directory, device="cuda")
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertEqual(loader.call_args.kwargs["model_name"], directory)
            self.assertEqual(provider.name, "BAAI/bge-m3")
            self.assertNotIn(directory, provider.collection_key())
            report = verify_production_embedding(provider)
            self.assertEqual(report["status"], "passed")
            self.assertNotIn(directory, json.dumps(report))
            self.assertEqual(report["dimension"], 2)
            self.assertEqual(report["semantic_gap"], 0.8)
            self.assertEqual(report["paid_embedding_api_calls"], 0)
            self.assertFalse(report["hashed_fallback"])

    def test_model_load_failure_has_fixed_error_and_no_fallback(self) -> None:
        with patch(
            "examples.research_pilot.production_embedding.HuggingFaceEmbeddingProvider",
            side_effect=RuntimeError("private local directory or provider details"),
        ) as loader:
            with self.assertRaises(ProductionEmbeddingError) as caught:
                create_production_embedding()
        self.assertEqual(caught.exception.code, "production_embedding_load_failed")
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(loader.call_count, 1)

    def test_unavailable_directory_and_path_like_public_id_fail_before_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("examples.research_pilot.production_embedding.HuggingFaceEmbeddingProvider") as loader:
                with self.assertRaisesRegex(ProductionEmbeddingError, "local_embedding_model_missing"):
                    create_production_embedding(local_model_path=Path(directory) / "missing")
                with self.assertRaisesRegex(ProductionEmbeddingError, "invalid_public_embedding_model_id"):
                    create_production_embedding(model_id=directory)
                loader.assert_not_called()

    def test_missing_corrupt_or_non_normalized_vectors_fail_inference(self) -> None:
        cases = [
            ([], "embedding_batch_size_mismatch"),
            ([[1.0]], "embedding_dimension_mismatch"),
            ([[float("nan"), 0]], "embedding_nonfinite_vector"),
            ([[float("inf"), 0]], "embedding_nonfinite_vector"),
            ([[0, 0]], "embedding_zero_or_invalid_norm"),
            ([[2.0, 0]], "embedding_not_normalized"),
        ]
        for vectors, code in cases:
            with self.subTest(code=code):
                delegate = self.delegate()
                delegate.embed_texts.side_effect = None
                delegate.embed_texts.return_value = vectors
                with patch(
                    "examples.research_pilot.production_embedding.HuggingFaceEmbeddingProvider", return_value=delegate
                ):
                    provider = create_production_embedding()
                with self.assertRaisesRegex(ProductionEmbeddingError, code):
                    provider.embed_query("synthetic text")

    def test_collapsed_vectors_fail_semantic_health_and_hashed_is_rejected(self) -> None:
        delegate = self.delegate()
        delegate.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        with patch("examples.research_pilot.production_embedding.HuggingFaceEmbeddingProvider", return_value=delegate):
            provider = create_production_embedding()
        report = verify_production_embedding(provider)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason"], "semantic_health_check_failed")
        with self.assertRaisesRegex(ProductionEmbeddingError, "production_embedding_provider_required"):
            verify_production_embedding(HashedEmbeddingProvider())

    def test_inference_error_is_sanitized_and_empty_batch_does_not_call_model(self) -> None:
        delegate = self.delegate()
        delegate.embed_texts.side_effect = RuntimeError("sensitive inference detail")
        with patch("examples.research_pilot.production_embedding.HuggingFaceEmbeddingProvider", return_value=delegate):
            provider = create_production_embedding()
        self.assertEqual(provider.embed_texts([]), [])
        delegate.embed_texts.assert_not_called()
        with self.assertRaisesRegex(ProductionEmbeddingError, "production_embedding_inference_failed"):
            provider.embed_document("synthetic text")


if __name__ == "__main__":
    unittest.main()
