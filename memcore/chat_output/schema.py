"""Schema objects for the optional chat-output adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChatOutputMode = Literal["auto", "plain", "memcore_json", "custom_json"]
ChatOutputStatus = Literal["parsed", "plain_text", "output_unparsed", "invalid_contract"]


@dataclass(frozen=True)
class ChatOutputConfig:
    mode: ChatOutputMode = "auto"
    enable_sentence_segments: bool = True
    min_segment_chars: int = 2
    max_segment_chars: int = 180
    max_segments: int | None = None


@dataclass(frozen=True)
class ChatOutputParseResult:
    status: ChatOutputStatus
    speech: str = ""
    memory_metadata: dict[str, Any] = field(default_factory=dict)
    presentation: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    segments: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("parsed", "plain_text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "speech": self.speech,
            "memory_metadata": dict(self.memory_metadata),
            "presentation": dict(self.presentation),
            "extra": dict(self.extra),
            "segments": list(self.segments),
            "reason": self.reason,
        }
