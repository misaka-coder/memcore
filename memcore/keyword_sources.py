"""Reloadable literal keyword evidence, independent from fuzzy retrieval."""

from __future__ import annotations

import json
import base64
import hashlib
import zlib
import unicodedata
from dataclasses import replace
from typing import Any

from .catalog_keywords import normalize_catalog_terms
from .memory_navigation import decode_memory_cursor, encode_memory_cursor
from .memory_sources import _with_source_ids
from .projection import canonical_json_bytes, stable_projection_hash
from .rendering import render_speaker_label
from .timeline_read import build_timeline_read_units, timestamp_iso
from .token_counter import estimate_text_tokens

MODE = "keyword_sources"


def keyword_preview(source: dict[str, Any], words: list[str], timezone: str) -> dict[str, Any]:
    """Take an original-text window near a hit; never render a second full body."""
    text = str(source["text"])
    normalized = normalize_catalog_terms([text])[0].casefold()
    position = min((normalized.find(word.casefold()) for word in words if word.casefold() in normalized), default=0)
    offsets: list[int] = []
    last_space = True
    for i, char in enumerate(text):
        for folded in unicodedata.normalize("NFKC", char).casefold():
            if folded.isspace():
                if not last_space:
                    offsets.append(i)
                last_space = True
            else:
                offsets.append(i)
                last_space = False
    anchor = offsets[min(position, len(offsets) - 1)] if offsets else 0
    start = max(0, anchor - 20)
    end = min(len(text), start + 80)
    return {
        "source_id": source["source_id"],
        "timestamp": timestamp_iso(source["timestamp"], timezone=timezone),
        "actor_id": source["actor_id"],
        "actor_display_name": source["actor_display_name"],
        "text": ("…" if start else "") + text[start:end] + ("…" if end < len(text) else ""),
    }


def build_match_ref(mem: Any, card_id: str, match: dict[str, Any], selector: dict[str, Any]) -> str:
    return _compress_ref(
        encode_memory_cursor(
            namespace=mem.namespace,
            mode=MODE,
            selector={
                "memory_id": card_id,
                "keywords": selector["keywords"],
                "keyword_match": selector["keyword_match"],
                "cross_conversation": bool(selector["cross_conversation"]),
                "source_fingerprint": match["source_fingerprint"],
                "catalog_selector": _public_selector(selector, mem.timezone),
            },
            page_size=50,
            last_key=(-1,),
        )
    )


def open_keyword_sources(
    mem: Any,
    decoded: dict[str, Any],
    *,
    page_token_budget: int = 0,
    page_byte_budget: int = 0,
) -> dict[str, Any]:
    """Recompute exact membership, check the fingerprint, then project this page."""
    selector = dict(decoded["selector"])
    memory_id = str(selector.get("memory_id") or "")
    try:
        words = normalize_catalog_terms(selector["keywords"], strict=True)
        cross = selector["cross_conversation"]
        if not words or type(cross) is not bool or selector["keyword_match"] not in {"any", "all"}:
            raise ValueError("invalid_keyword_source_selector")
        last_key = decoded["last_key"]
        if len(last_key) != 1 or type(last_key[0]) is not int or last_key[0] < -1:
            raise ValueError("invalid_cursor")
        matches = mem.store.get_catalog_keyword_matches(
            namespace=mem.namespace,
            memory_ids=(memory_id,),
            keywords=words,
            cross_conversation=cross,
        )
        match = matches.get(memory_id)
        if match is None or match["source_fingerprint"] != selector.get("source_fingerprint"):
            return {
                "status": "invalid_filter",
                "reason": "source_changed_restart_required",
                "view": "sources",
                "memory_id": memory_id,
                "result": None,
                "suggested_next_actions": [
                    {"tool": "browse_memory", "arguments": selector.get("catalog_selector", {})}
                ],
            }
    except NotImplementedError:
        return {
            "status": "unavailable",
            "reason": "keyword_sources_store_unsupported",
            "view": "sources",
            "result": None,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "invalid_filter", "reason": str(exc), "view": "sources", "result": None}

    # Trusted direct reads can remain unlimited; native dispatch supplies its
    # existing finite maxima, also capping a trusted diagnostic cursor.
    for name, maximum in (("page_token_budget", page_token_budget), ("page_byte_budget", page_byte_budget)):
        saved = int(selector.get(name) or 0)
        selector[name] = min(saved, maximum) if saved and maximum else saved or maximum
    token_budget = selector["page_token_budget"]
    byte_budget = selector["page_byte_budget"]
    all_ids = match["matched_source_ids"]
    offset = last_key[0] + 1
    if offset > len(all_ids):
        return {"status": "invalid_filter", "reason": "invalid_cursor", "view": "sources", "result": None}
    candidate_ids = all_ids[offset : offset + int(decoded["page_size"])]
    entries = mem.store.get_entries_by_source_ids(
        namespace=mem.namespace,
        source_ids=tuple(candidate_ids),
        cross_conversation=cross,
    )
    by_id = {entry.source_id: entry for entry in entries}
    if len(by_id) != len(candidate_ids) or any(
        entry.row_version != match["matched_source_versions"].get(sid) for sid, entry in by_id.items()
    ):
        return {
            "status": "invalid_filter",
            "reason": "source_changed_restart_required",
            "view": "sources",
            "result": None,
        }
    count = mem.token_counter.count_text if mem.token_counter is not None else estimate_text_tokens
    selected: list[dict[str, Any]] = []
    oversized = False

    def compose(messages: list[dict[str, Any]]) -> dict[str, Any]:
        returned = len(messages)
        remaining = len(all_ids) - offset - returned
        cursor = (
            _compress_ref(
                encode_memory_cursor(
                    namespace=mem.namespace,
                    mode=MODE,
                    selector=selector,
                    page_size=decoded["page_size"],
                    last_key=(offset + returned - 1,),
                )
            )
            if remaining and returned
            else ""
        )
        return {
            "status": "partial" if remaining or match["missing_source_count"] else "found",
            "reason": "missing_sources" if match["missing_source_count"] else "",
            "memory_id": memory_id,
            "view": "sources",
            "projection": "conversation",
            "source_mode": "keyword_filtered",
            "keywords": words,
            "keyword_match": selector["keyword_match"],
            "cross_conversation": cross,
            "matched_source_count": len(all_ids),
            "returned_source_count": returned,
            "remaining_source_count": remaining,
            "missing_source_count": match["missing_source_count"],
            "page_complete": remaining == 0,
            "next_cursor": cursor,
            "result": {"messages": messages},
            "text": "\n\n".join(
                f"{timestamp_iso(row['timestamp'], timezone=mem.timezone)} {render_speaker_label(row)}\n{row['content']}"
                for row in messages
            ),
        }

    for sid in candidate_ids:
        # Raw utterances, including assistant speech, without protocol/debug
        # payloads. Attribution remains the actor recorded at that time.
        entry = replace(by_id[sid], payload={}, trace_metadata={}, target_actor=None)
        unit = build_timeline_read_units(
            [entry],
            projection="conversation",
            renderer_registry=mem.renderer_registry,
            timezone=mem.timezone,
        )[0]
        projected = dict(unit.projected_messages[0])
        projected["content"] = entry.semantic_text or entry.content
        message = _with_source_ids([projected])[0]
        candidate = compose([*selected, message])
        projection = {key: value for key, value in candidate.items() if key != "result"}
        encoded = json.dumps(projection, ensure_ascii=False, sort_keys=True)
        too_many_bytes = byte_budget and len(encoded.encode("utf-8")) > byte_budget
        too_many_tokens = token_budget and count(encoded) > token_budget
        if (too_many_bytes or too_many_tokens) and selected:
            break
        if too_many_bytes:
            return {
                "status": "unavailable",
                "reason": "source_group_exceeds_byte_budget",
                "view": "sources",
                "memory_id": memory_id,
                "matched_source_count": len(all_ids),
                "remaining_source_count": len(all_ids) - offset,
                "suggested_next_actions": [
                    {
                        "tool": "open_memory",
                        "arguments": {
                            "memory_id": sid,
                            "view": "content",
                            **({"cross_conversation": True} if cross else {}),
                        },
                    }
                ],
                "result": None,
                "text": "",
            }
        selected.append(message)
        if too_many_tokens:
            oversized = True
            break
    result = compose(selected)
    result["coverage"] = (
        {
            "oversized_unit": True,
            "page_token_budget": token_budget,
            "token_count_quality": mem.token_counter.quality if mem.token_counter is not None else "estimated",
        }
        if oversized
        else {}
    )
    return result


def _compress_ref(token: str) -> str:
    # Compress canonical JSON directly instead of a base64 encoding of JSON.
    # The selector hash is derivable; restore it before ordinary cursor validation.
    _, encoded, signature = token.split(":")
    body = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    body.pop("selector_fingerprint")
    blob = zlib.compress(canonical_json_bytes(body), 9)
    encoded = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")
    return f"kmr2:{encoded}:{signature}"


def _public_selector(selector: dict[str, Any], timezone: str) -> dict[str, Any]:
    result = {
        key: selector[key]
        for key in ("keywords", "keyword_match", "node_types", "participants", "participant_match")
        if selector.get(key)
    }
    if selector.get("start_ts") is not None and selector.get("end_ts") is not None:
        result["time_range"] = {
            "start_at": timestamp_iso(selector["start_ts"], timezone=timezone),
            "end_at": timestamp_iso(selector["end_ts"], timezone=timezone),
        }
    if selector.get("cross_conversation"):
        result["cross_conversation"] = True
    return result


def decode_keyword_ref(mem: Any, token: str) -> dict[str, Any]:
    if token.startswith(("kmr1:", "kmr2:")):
        try:
            prefix, encoded, signature = token.split(":")
            blob = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if prefix == "kmr1" and hashlib.sha256(blob).hexdigest()[:16] != signature:
                raise ValueError
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(blob, 65537)
            if len(decoded) > 65536 or not decoder.eof or decoder.unused_data:
                raise ValueError
            if prefix == "kmr2":
                body = json.loads(decoded)
                body["selector_fingerprint"] = stable_projection_hash(body["selector"])
                encoded = base64.urlsafe_b64encode(canonical_json_bytes(body)).decode("ascii").rstrip("=")
                token = f"memory-v1:{encoded}:{signature}"
            else:
                token = decoded.decode("ascii")
        except (KeyError, TypeError, ValueError, UnicodeError, zlib.error) as exc:
            raise ValueError("invalid_keyword_ref") from exc
    return decode_memory_cursor(token, namespace=mem.namespace, expected_mode=MODE)
