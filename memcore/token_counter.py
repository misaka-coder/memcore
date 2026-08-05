"""TokenCounter interface and the explicit coarse fallback estimator."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TokenCounter(ABC):
    @property
    def quality(self) -> str:
        """Override with ``estimated`` when the implementation is not an exact tokenizer."""
        return "exact"

    @abstractmethod
    def count_text(self, text: str) -> int:
        """返回 text 在接入方模型 tokenizer 下的 token 数。"""
        raise NotImplementedError


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens from UTF-8 bytes when no provider tokenizer is available.

    Callers must expose the result as ``quality=estimated``.  The estimate is
    intentionally simple and is not presented as provider billing usage.
    """

    return max(1, (len(str(text or "").encode("utf-8")) + 3) // 4)


__all__ = ["TokenCounter", "estimate_text_tokens"]
