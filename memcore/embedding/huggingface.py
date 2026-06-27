"""HuggingFaceEmbeddingProvider —— 真实语义(sentence-transformers / BGE-M3)。

可选依赖(extras: huggingface)。延迟导入,不装不影响其余模块。
加载失败**抛异常**,由上层决定是否显式降级,绝不在此静默退到 hashed(见 §13.1)。
"""

from __future__ import annotations

from typing import Iterable

from .base import EmbeddingProvider

DEFAULT_MODEL = "BAAI/bge-m3"


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        local_files_only: bool = True,
        cache_folder: str | None = None,
    ) -> None:
        self.model_name = str(model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.name = self.model_name
        self.version = "st-v1"
        self._model = self._load(device=device, local_files_only=local_files_only, cache_folder=cache_folder)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def _load(self, *, device: str | None, local_files_only: bool, cache_folder: str | None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # 缺可选依赖:明确报错,不静默
            raise RuntimeError(
                "sentence-transformers not installed; `pip install memcore[huggingface]` or run degraded explicitly"
            ) from exc
        return SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [str(t or "") for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]
