"""HTTPEmbeddingProvider —— 调用 OpenAI 兼容的 /embeddings 接口拿真语义向量。

给"跑不动本地模型、又想要真语义"的接入方兜底:没有 torch、没有 GB 下载,填 base_url + api_key + model 即可。
纯标准库(urllib),零新增依赖。网络调用集中在 _call_api,便于测试时替换。

不捆绑任何模型权重:库只提供"怎么接",模型由接入方选择(本地 / API / 自有)。
"""

from __future__ import annotations

import json
import urllib.request
from typing import Iterable

from .base import EmbeddingProvider


class HTTPEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 30.0,
        name: str | None = None,
    ) -> None:
        if not str(base_url or "").strip():
            raise ValueError("base_url is required (e.g. 'https://api.openai.com/v1')")
        if int(dimension) <= 0:
            raise ValueError("dimension must be a positive int (your embedding model's output size)")
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model)
        self._dimension = int(dimension)
        self.timeout = float(timeout)
        self.name = name or f"http:{self.model}"
        self.version = "http-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(t or "") for t in texts]
        if not items:
            return []
        vectors = self._call_api(items)
        if len(vectors) != len(items):
            raise RuntimeError(f"embedding API returned {len(vectors)} vectors for {len(items)} inputs")
        for vector in vectors:
            if len(vector) != self._dimension:
                raise RuntimeError(
                    f"embedding dimension mismatch: got {len(vector)}, expected {self._dimension} "
                    "(check the `dimension` you configured matches the model)"
                )
        return vectors

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """POST {base_url}/embeddings(OpenAI 兼容)。测试时可覆盖此方法,避免真实网络。"""
        body = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310 (受控 URL)
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        return [list(item.get("embedding") or []) for item in data]
