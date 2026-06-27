"""Streaming speech extraction for optional chat-output contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..schema import DEFAULT_CATEGORIES
from .parser import parse_chat_output
from .schema import ChatOutputMode
from .segmenter import (
    _append_raw,
    _consume_sentence_tail,
    _is_sentence_end,
    _merge_short_segments,
    _soft_cut,
)


class StreamingSpeechParser:
    """Incrementally emit display speech from plain text or memcore JSON output.

    The final parse result remains authoritative. Streaming events are delivery
    hints for UI/TTS latency and must not be written as memory by themselves.
    """

    def __init__(
        self,
        *,
        mode: ChatOutputMode = "auto",
        categories: Iterable[str] = DEFAULT_CATEGORIES,
        enable_flavor: bool = False,
        enable_sentence_segments: bool = True,
        min_segment_chars: int = 2,
        max_segment_chars: int = 180,
        max_segments: int | None = None,
    ) -> None:
        self.mode = _normalize_mode(mode)
        self.categories = tuple(categories)
        self.enable_flavor = bool(enable_flavor)
        self.enable_sentence_segments = bool(enable_sentence_segments)
        self.min_segment_chars = max(1, int(min_segment_chars))
        self.max_segment_chars = max(self.min_segment_chars, int(max_segment_chars))
        self.max_segments = max_segments

        self._raw_parts: list[str] = []
        self._plain_resolved = self.mode == "plain"
        self._json_speech = _TopLevelSpeechScanner()
        self._segment_pending = ""
        self._segment_index = 0
        self._finished = False

    def feed(self, chunk: Any) -> list[dict[str, Any]]:
        """Consume one model stream chunk and return zero or more events."""

        self._assert_not_finished()
        text = str(chunk or "")
        if not text:
            return []
        self._raw_parts.append(text)

        if self._plain_resolved:
            return self._emit_speech_delta(text)

        if self.mode == "auto" and self._auto_stream_shape() == "plain":
            self._plain_resolved = True
            return self._emit_speech_delta("".join(self._raw_parts))

        events: list[dict[str, Any]] = []
        for speech_delta in self._json_speech.feed(text):
            events.extend(self._emit_speech_delta(speech_delta))
        return events

    def finish(self) -> list[dict[str, Any]]:
        """Finish the stream and return remaining segment/final events."""

        self._assert_not_finished()
        self._finished = True

        events = self._flush_segments()
        raw_output = "".join(self._raw_parts)
        final_mode: ChatOutputMode = "plain" if self._plain_resolved else self.mode
        result = parse_chat_output(
            raw_output,
            mode=final_mode,
            categories=self.categories,
            enable_flavor=self.enable_flavor,
            enable_sentence_segments=self.enable_sentence_segments,
            min_segment_chars=self.min_segment_chars,
            max_segment_chars=self.max_segment_chars,
            max_segments=self.max_segments,
        )
        if result.status == "parsed":
            events.append({"type": "metadata_ready", "memory_metadata": dict(result.memory_metadata)})
        events.append({"type": "final", "payload": result.to_dict()})
        return events

    def _assert_not_finished(self) -> None:
        if self._finished:
            raise ValueError("stream already finished")

    def _auto_stream_shape(self) -> str:
        raw = "".join(self._raw_parts).lstrip()
        if not raw:
            return "undecided"
        return "json" if raw[0] in "{[" else "plain"

    def _emit_speech_delta(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [{"type": "speech_chunk", "text": text}]
        if self.enable_sentence_segments:
            self._segment_pending += text
            events.extend(self._pop_available_segments(flush=False))
        return events

    def _flush_segments(self) -> list[dict[str, Any]]:
        if not self.enable_sentence_segments:
            self._segment_pending = ""
            return []
        return self._pop_available_segments(flush=True)

    def _pop_available_segments(self, *, flush: bool) -> list[dict[str, Any]]:
        if self.max_segments is not None and self._segment_index >= max(0, int(self.max_segments)):
            self._segment_pending = "" if flush else self._segment_pending
            return []

        segments, remainder = _split_stream_segments(
            self._segment_pending,
            min_chars=self.min_segment_chars,
            max_chars=self.max_segment_chars,
            flush=flush,
        )
        self._segment_pending = remainder

        events: list[dict[str, Any]] = []
        for segment in segments:
            if self.max_segments is not None and self._segment_index >= max(0, int(self.max_segments)):
                break
            events.append({"type": "speech_segment", "index": self._segment_index, "text": segment})
            self._segment_index += 1
        return events


class _TopLevelSpeechScanner:
    """Find and decode the top-level JSON string value for key ``speech``."""

    def __init__(self) -> None:
        self._buffer = ""
        self._pos = 0
        self._depth = 0
        self._state = "seek_key"
        self._key_chars: list[str] = []
        self._last_key = ""

    def feed(self, text: str) -> list[str]:
        if self._state == "done":
            return []
        self._buffer += text
        out: list[str] = []

        while self._pos < len(self._buffer) and self._state != "done":
            if self._state == "seek_key":
                self._scan_for_key_start()
            elif self._state == "read_key":
                token = _read_json_string_token(self._buffer, self._pos)
                if token.need_more:
                    break
                self._pos = token.next_pos
                if token.closed:
                    self._last_key = "".join(self._key_chars)
                    self._key_chars = []
                    self._state = "await_colon"
                else:
                    self._key_chars.append(token.text)
            elif self._state == "await_colon":
                self._await_colon()
            elif self._state == "await_speech_value":
                self._await_speech_value()
            elif self._state == "skip_string":
                token = _read_json_string_token(self._buffer, self._pos)
                if token.need_more:
                    break
                self._pos = token.next_pos
                if token.closed:
                    self._state = "seek_key"
            elif self._state == "read_speech":
                token = _read_json_string_token(self._buffer, self._pos)
                if token.need_more:
                    break
                self._pos = token.next_pos
                if token.closed:
                    self._state = "done"
                else:
                    out.append(token.text)
        return ["".join(out)] if out else []

    def _scan_for_key_start(self) -> None:
        char = self._buffer[self._pos]
        if char == '"':
            if self._depth == 1:
                self._state = "read_key"
                self._key_chars = []
            else:
                self._state = "skip_string"
            self._pos += 1
            return
        if char in "{[":
            self._depth += 1
        elif char in "}]":
            self._depth = max(0, self._depth - 1)
        self._pos += 1

    def _await_colon(self) -> None:
        char = self._buffer[self._pos]
        if char.isspace():
            self._pos += 1
            return
        if char == ":":
            self._state = "await_speech_value" if self._last_key == "speech" else "seek_key"
            self._pos += 1
            return
        self._state = "seek_key"

    def _await_speech_value(self) -> None:
        char = self._buffer[self._pos]
        if char.isspace():
            self._pos += 1
            return
        if char == '"':
            self._state = "read_speech"
            self._pos += 1
            return
        self._state = "done"


class _StringToken:
    def __init__(self, *, text: str = "", next_pos: int = 0, closed: bool = False, need_more: bool = False) -> None:
        self.text = text
        self.next_pos = next_pos
        self.closed = closed
        self.need_more = need_more


def _read_json_string_token(buffer: str, pos: int) -> _StringToken:
    if pos >= len(buffer):
        return _StringToken(next_pos=pos, need_more=True)
    char = buffer[pos]
    if char == '"':
        return _StringToken(next_pos=pos + 1, closed=True)
    if char != "\\":
        return _StringToken(text=char, next_pos=pos + 1)
    decoded, next_pos, need_more = _decode_escape_at(buffer, pos)
    if need_more:
        return _StringToken(next_pos=pos, need_more=True)
    return _StringToken(text=decoded, next_pos=next_pos)


def _decode_escape_at(buffer: str, pos: int) -> tuple[str, int, bool]:
    if pos + 1 >= len(buffer):
        return "", pos, True
    esc = buffer[pos + 1]
    mapping = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    if esc != "u":
        return mapping.get(esc, esc), pos + 2, False

    if pos + 6 > len(buffer):
        return "", pos, True
    digits = buffer[pos + 2 : pos + 6]
    try:
        codepoint = int(digits, 16)
    except ValueError:
        return "u", pos + 2, False

    next_pos = pos + 6
    if 0xD800 <= codepoint <= 0xDBFF and buffer[next_pos : next_pos + 2] == "\\u":
        if next_pos + 6 > len(buffer):
            return "", pos, True
        try:
            low = int(buffer[next_pos + 2 : next_pos + 6], 16)
        except ValueError:
            low = -1
        if 0xDC00 <= low <= 0xDFFF:
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            next_pos += 6
    return chr(codepoint), next_pos, False


def _split_stream_segments(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    flush: bool,
) -> tuple[list[str], str]:
    raw: list[str] = []
    start = 0
    cut_end = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            _append_raw(raw, text[start:i])
            start = i + 1
            cut_end = start
            i = start
            continue
        if _is_sentence_end(text, i):
            end = _consume_sentence_tail(text, i)
            if end >= len(text) and not flush:
                break
            _append_raw(raw, text[start:end])
            start = end
            cut_end = start
            i = start
            continue
        if i - start + 1 >= max_chars:
            cut = _soft_cut(text, start, i + 1, min_chars=min_chars)
            if cut > start:
                _append_raw(raw, text[start:cut])
                start = cut
                cut_end = start
                i = start
                continue
        i += 1

    if flush:
        _append_raw(raw, text[start:])
        cut_end = len(text)

    return _merge_short_segments(raw, min_chars=min_chars), text[cut_end:]


def _normalize_mode(mode: Any) -> ChatOutputMode:
    value = str(mode or "auto").strip()
    return value if value in ("auto", "plain", "memcore_json", "custom_json") else "auto"  # type: ignore[return-value]
