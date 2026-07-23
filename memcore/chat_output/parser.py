"""Parse model chat output into speech + memory metadata."""

from __future__ import annotations

import json
from typing import Any

from ..schema import coerce_memory_metadata
from .schema import ChatOutputMode, ChatOutputParseResult
from .segmenter import segment_speech

_PRESENTATION_KEYS = ("emotion", "reply_medium")
_CORE_KEYS = {"speech", "memory_metadata"}


def parse_chat_output(
    output: Any,
    *,
    mode: ChatOutputMode = "auto",
    enable_flavor: bool = False,
    enable_sentence_segments: bool = True,
    min_segment_chars: int = 2,
    max_segment_chars: int = 180,
    max_segments: int | None = None,
) -> ChatOutputParseResult:
    mode = _normalize_mode(mode)
    if mode == "plain":
        return _plain_result(
            output,
            enable_sentence_segments=enable_sentence_segments,
            min_segment_chars=min_segment_chars,
            max_segment_chars=max_segment_chars,
            max_segments=max_segments,
        )

    parsed, reason = _coerce_json_object(output)
    if parsed is None:
        if mode == "auto" and reason == "not_json":
            return _plain_result(
                output,
                enable_sentence_segments=enable_sentence_segments,
                min_segment_chars=min_segment_chars,
                max_segment_chars=max_segment_chars,
                max_segments=max_segments,
            )
        return ChatOutputParseResult(status="output_unparsed", reason=reason)

    speech_value = parsed.get("speech")
    if not isinstance(speech_value, str):
        status = "invalid_contract" if mode == "memcore_json" else "output_unparsed"
        return ChatOutputParseResult(status=status, extra=_extra_fields(parsed), reason="speech_required")

    speech = speech_value.strip()
    metadata_present = "memory_metadata" in parsed
    metadata_value = parsed.get("memory_metadata")
    metadata_status = "accepted" if isinstance(metadata_value, dict) else "invalid" if metadata_present else "missing"
    metadata = coerce_memory_metadata(
        metadata_value if isinstance(metadata_value, dict) else None,
        enable_flavor=enable_flavor,
    ).to_dict()
    segments = (
        segment_speech(
            speech,
            min_chars=min_segment_chars,
            max_chars=max_segment_chars,
            max_segments=max_segments,
        )
        if enable_sentence_segments
        else []
    )
    return ChatOutputParseResult(
        status="parsed",
        speech=speech,
        memory_metadata=metadata,
        metadata_status=metadata_status,
        metadata_present=metadata_present,
        presentation=_presentation_fields(parsed),
        extra=_extra_fields(parsed),
        segments=segments,
    )


def _normalize_mode(mode: Any) -> ChatOutputMode:
    value = str(mode or "auto").strip()
    return value if value in ("auto", "plain", "memcore_json", "custom_json") else "auto"  # type: ignore[return-value]


def _plain_result(
    output: Any,
    *,
    enable_sentence_segments: bool,
    min_segment_chars: int,
    max_segment_chars: int,
    max_segments: int | None,
) -> ChatOutputParseResult:
    speech = str(output or "").strip()
    segments = (
        segment_speech(
            speech,
            min_chars=min_segment_chars,
            max_chars=max_segment_chars,
            max_segments=max_segments,
        )
        if enable_sentence_segments
        else []
    )
    return ChatOutputParseResult(
        status="plain_text",
        speech=speech,
        memory_metadata={},
        metadata_status="plain",
        metadata_present=False,
        segments=segments,
    )


def _coerce_json_object(output: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(output, dict):
        return dict(output), ""
    if not isinstance(output, str):
        return None, "not_json"
    text = output.strip()
    if not text:
        return None, "not_json"
    if not text.startswith(("{", "[")):
        return None, "not_json"
    if text.startswith("{") and not text.endswith("}"):
        return None, "invalid_json"
    if text.startswith("[") and not text.endswith("]"):
        return None, "invalid_json"
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, ""


def _presentation_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in _PRESENTATION_KEYS if key in payload}


def _extra_fields(payload: dict[str, Any]) -> dict[str, Any]:
    reserved = _CORE_KEYS | set(_PRESENTATION_KEYS)
    return {key: value for key, value in payload.items() if key not in reserved}
