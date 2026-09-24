"""Local-only production embeddings for the isolated research pilot.

The caller selects an already installed sentence-transformers runtime and cached
model. This adapter never downloads, calls a paid embedding API, or substitutes
hashed vectors. Local model/cache paths stay inside the loader; public metadata
uses the caller's explicit repository model ID.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from memcore import EmbeddingProvider, HuggingFaceEmbeddingProvider

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")
PROBE_TEXTS = (
    "我喜欢喝不加糖的咖啡。",
    "我的咖啡要无糖。",
    "今晚可以观测流星雨。",
)


class ProductionEmbeddingError(RuntimeError):
    """A fixed, path-free failure code suitable for experiment reports."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _validated_vectors(vectors: Any, *, count: int, dimension: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ProductionEmbeddingError("embedding_batch_size_mismatch")
    result: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ProductionEmbeddingError("embedding_dimension_mismatch")
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in vector):
            raise ProductionEmbeddingError("embedding_nonfinite_vector")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ProductionEmbeddingError("embedding_zero_or_invalid_norm")
        if abs(norm - 1.0) > 0.001:
            raise ProductionEmbeddingError("embedding_not_normalized")
        result.append([float(value) for value in vector])
    return result


class ProductionEmbedding(EmbeddingProvider):
    """A validated semantic provider with path-free public identity."""

    version = "st-local-v1"

    def __init__(self, delegate: HuggingFaceEmbeddingProvider, *, model_id: str, device: str):
        self.name = model_id
        self.device = device
        self._delegate = delegate
        self._dimension = delegate.dimension
        if type(self._dimension) is not int or self._dimension <= 0:
            raise ProductionEmbeddingError("embedding_invalid_dimension")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        if any(not isinstance(text, str) for text in items):
            raise ProductionEmbeddingError("embedding_text_must_be_string")
        try:
            vectors = self._delegate.embed_texts(items)
        except Exception:
            raise ProductionEmbeddingError("production_embedding_inference_failed") from None
        return _validated_vectors(vectors, count=len(items), dimension=self.dimension)


def create_production_embedding(
    *,
    model_id: str = "BAAI/bge-m3",
    local_model_path: str | Path | None = None,
    device: str = "cpu",
    cache_folder: str | Path | None = None,
) -> ProductionEmbedding:
    """Load only existing local weights; errors never trigger another provider.

    A local directory may be passed separately from its public repository ID.
    When omitted, sentence-transformers may use only the existing local cache
    for model_id. Model installation and downloads are outside this factory.
    """
    if not isinstance(model_id, str) or _MODEL_ID.fullmatch(model_id) is None:
        raise ProductionEmbeddingError("invalid_public_embedding_model_id")
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise ProductionEmbeddingError("invalid_embedding_device")
    if local_model_path is not None and not Path(local_model_path).is_dir():
        raise ProductionEmbeddingError("local_embedding_model_missing")
    try:
        delegate = HuggingFaceEmbeddingProvider(
            model_name=str(local_model_path) if local_model_path is not None else model_id,
            device=device,
            local_files_only=True,
            cache_folder=str(cache_folder) if cache_folder is not None else None,
        )
    except Exception:
        raise ProductionEmbeddingError("production_embedding_load_failed") from None
    return ProductionEmbedding(delegate, model_id=model_id, device=device)


def _cosine(first: list[float], second: list[float]) -> float:
    return sum(a * b for a, b in zip(first, second)) / (
        math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    )


def verify_production_embedding(provider: ProductionEmbedding) -> dict[str, Any]:
    """Check fixed synthetic evidence, finite normalized vectors, and semantics.

    This is a small health probe, not a retrieval benchmark or a quality claim
    about the later research scenarios. No user messages or personal data enter
    the model. Failure is explicit and must stop the paid experiment.
    """
    if not isinstance(provider, ProductionEmbedding):
        raise ProductionEmbeddingError("production_embedding_provider_required")
    vectors = provider.embed_documents(PROBE_TEXTS)
    query = provider.embed_query(PROBE_TEXTS[0])
    similar = _cosine(vectors[0], vectors[1])
    unrelated = _cosine(vectors[0], vectors[2])
    repeat = _cosine(vectors[0], query)
    gap = similar - unrelated
    passed = similar > 0 and gap > 0.05 and repeat > 0.999
    vector_fingerprint = hashlib.sha256(
        json.dumps(vectors, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "format": "memcore_pilot_production_embedding_preflight_v1",
        "status": "passed" if passed else "failed",
        "reason": None if passed else "semantic_health_check_failed",
        "model_id": provider.name,
        "provider": "HuggingFaceEmbeddingProvider",
        "adapter_version": provider.version,
        "device": provider.device,
        "local_files_only": True,
        "hashed_fallback": False,
        "paid_embedding_api_calls": 0,
        "downloaded_model": False,
        "dimension": provider.dimension,
        "finite_normalized_vectors": True,
        "synthetic_texts": list(PROBE_TEXTS),
        "similar_score": round(similar, 6),
        "unrelated_score": round(unrelated, 6),
        "semantic_gap": round(gap, 6),
        "required_gap": 0.05,
        "repeat_score": round(repeat, 6),
        "vector_fingerprint": vector_fingerprint,
        "probe_scope": "small_synthetic_health_check_not_retrieval_benchmark",
    }
