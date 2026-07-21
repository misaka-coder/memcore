"""Chat output adapter: optional parsing layer for model replies."""

from __future__ import annotations

from .parser import parse_chat_output
from .prompts import build_chat_output_contract_prompt
from .schema import ChatOutputConfig, ChatOutputMode, ChatOutputParseResult, ChatOutputStatus, MemoryMetadataStatus
from .segmenter import segment_speech
from .streaming import StreamingSpeechParser

__all__ = [
    "ChatOutputConfig",
    "ChatOutputMode",
    "ChatOutputParseResult",
    "ChatOutputStatus",
    "MemoryMetadataStatus",
    "StreamingSpeechParser",
    "build_chat_output_contract_prompt",
    "parse_chat_output",
    "segment_speech",
]
