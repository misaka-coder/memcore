"""Sentence segmentation for chat speech.

This is a delivery hint only: clients decide whether to show/send/play each segment.
"""

from __future__ import annotations

import re
from typing import Any

_END_PUNCT = set("。！？!?")
_OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "（": "）",
    "【": "】",
    "《": "》",
    "「": "」",
    "『": "』",
    "“": "”",
    "‘": "’",
    "〈": "〉",
    "〔": "〕",
    "〖": "〗",
    "〘": "〙",
    "〚": "〛",
    "［": "］",
    "｛": "｝",
    "｟": "｠",
    "«": "»",
    "‹": "›",
    "<": ">",
}
_CLOSERS = set(_OPEN_TO_CLOSE.values()) | set("\"'`")
_ABBREVIATIONS = {
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "corp",
    "no",
    "st",
}


def segment_speech(
    text: Any,
    *,
    min_chars: int = 2,
    max_chars: int = 180,
    max_segments: int | None = None,
) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    min_chars = max(1, int(min_chars))
    max_chars = max(min_chars, int(max_chars))
    protected = _protected_span_boundaries(normalized)

    raw: list[str] = []
    start = 0
    i = 0
    while i < len(normalized):
        char = normalized[i]
        if char == "\n" and not protected[i]:
            _append_raw(raw, normalized[start:i])
            start = i + 1
            i += 1
            continue
        if _is_sentence_end(normalized, i, protected=protected):
            end = _consume_sentence_tail(normalized, i, protected=protected)
            _append_raw(raw, normalized[start:end])
            start = end
            i = end
            continue
        if i - start + 1 >= max_chars:
            cut = _soft_cut(normalized, start, i + 1, min_chars=min_chars, protected=protected)
            if cut > start:
                _append_raw(raw, normalized[start:cut])
                start = cut
                i = cut
                continue
        i += 1
    _append_raw(raw, normalized[start:])

    segments = _merge_short_segments(raw, min_chars=min_chars)
    if max_segments is not None:
        return segments[: max(0, int(max_segments))]
    return segments


def _append_raw(items: list[str], value: str) -> None:
    text = _clean_segment(value)
    if text:
        items.append(text)


def _clean_segment(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _is_sentence_end(text: str, index: int, *, protected: list[bool] | None = None) -> bool:
    char = text[index]
    if (protected if protected is not None else _protected_span_boundaries(text))[index]:
        return False
    if char in _END_PUNCT:
        return True
    if char == "…":
        return index + 1 < len(text) and text[index + 1] == "…"
    if char != ".":
        return False
    return _is_period_sentence_end(text, index)


def _is_period_sentence_end(text: str, index: int) -> bool:
    prev_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    if prev_char.isdigit() and next_char.isdigit():
        return False
    if prev_char.isalnum() and next_char.isalnum():
        return False
    if _is_numbered_list_marker(text, index):
        return False
    token = _token_ending_at(text, index).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return False
    return True


def _protected_span_boundaries(text: str) -> list[bool]:
    """Scan once per pending text, preserving nested title/quote/bracket state.

    Inner punctuation in values such as ``《孤独摇滚！》`` or ``“好吧。”`` is
    not an outer delivery boundary.  This is intentionally syntax-oriented;
    it does not try to infer sentence semantics from the surrounding words.
    """

    stack: list[str] = []
    escaped = False
    boundaries: list[bool] = []
    code_end = 0
    for pos, char in enumerate(text):
        boundaries.append(bool(stack))
        if pos < code_end:
            continue
        previous = text[pos - 1] if pos else ""
        following = text[pos + 1] if pos + 1 < len(text) else ""
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "`":
            end = pos + 1
            while end < len(text) and text[end] == "`":
                end += 1
            delimiter = text[pos:end]
            code_end = end
            if stack and stack[-1] == delimiter:
                stack.pop()
            elif not stack or not stack[-1].startswith("`"):
                stack.append(delimiter)
            continue
        if stack and stack[-1].startswith("`"):
            continue
        # Apostrophes inside words (including curly contractions) are not quotes.
        if char in "'’" and previous.isascii() and previous.isalnum() and following.isascii() and following.isalnum():
            continue
        if stack and char == stack[-1]:
            stack.pop()
            continue
        if char in "\"'":
            # Possessives and inch marks outside an open quote.
            if previous.isascii() and previous.isalnum():
                continue
            stack.append(char)
            continue
        if char == "<" and (not following or following.isspace() or following == "=" or previous.isalnum()):
            continue
        if char in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[char])
    boundaries.append(bool(stack))
    return boundaries


def _is_numbered_list_marker(text: str, index: int) -> bool:
    """Keep a line-leading marker such as ``1. item`` with its item text."""

    digit_start = index
    while digit_start > 0 and text[digit_start - 1].isdigit():
        digit_start -= 1
    if digit_start == index:
        return False

    line_start = text.rfind("\n", 0, digit_start) + 1
    line_prefix = text[line_start:digit_start]
    if line_prefix.strip():
        boundary_index = digit_start - 1
        while boundary_index >= line_start and text[boundary_index].isspace():
            boundary_index -= 1
        while boundary_index >= line_start and text[boundary_index] in _CLOSERS:
            boundary_index -= 1
        if boundary_index >= line_start and text[boundary_index] not in _END_PUNCT and text[boundary_index] != "…":
            return False

    next_index = index + 1
    while next_index < len(text) and text[next_index] in " \t":
        next_index += 1
    # A provider may stream ``1.``, its following space, and the item text in
    # three different deltas.  Keep a syntactically valid line-leading marker
    # pending while its body is still unknown.  Static/final segmentation will
    # still flush a genuinely standalone ``1.`` unchanged.
    return next_index >= len(text) or text[next_index] != "\n"


def _token_ending_at(text: str, index: int) -> str:
    start = index
    while start > 0 and not text[start - 1].isspace():
        if text[start - 1] in "([{（【《「『\"'“‘":
            break
        start -= 1
    return text[start : index + 1]


def _consume_sentence_tail(text: str, index: int, *, protected: list[bool] | None = None) -> int:
    protected = protected if protected is not None else _protected_span_boundaries(text)
    end = index + 1
    while end < len(text) and (text[end] in _END_PUNCT or text[end] in ".…" or text[end] in _CLOSERS):
        if text[end] in _CLOSERS and protected[end + 1]:
            break
        if text[end] == "." and not _is_period_sentence_end(text, end):
            break
        end += 1
    return end


def _soft_cut(text: str, start: int, end: int, *, min_chars: int, protected: list[bool] | None = None) -> int:
    protected = protected if protected is not None else _protected_span_boundaries(text)
    window = text[start:end]
    candidates = [match.end() for match in re.finditer(r"[，,；;、\s]", window)]
    for offset in reversed(candidates):
        if offset >= min_chars and not protected[start + offset - 1]:
            return start + offset
    # Length is a soft delivery target, never a reason to split a word or pair.
    return start


def _merge_short_segments(raw: list[str], *, min_chars: int) -> list[str]:
    out: list[str] = []
    pending = ""
    for item in raw:
        segment = f"{pending}{item}" if pending else item
        pending = ""
        if len(segment) < min_chars:
            if out:
                out[-1] = f"{out[-1]}{segment}"
            else:
                pending = segment
            continue
        out.append(segment)
    if pending:
        if out:
            out[-1] = f"{out[-1]}{pending}"
        else:
            out.append(pending)
    return out
