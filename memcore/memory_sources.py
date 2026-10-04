"""Exact source-tree reads, optional speaker excerpts, and shared batch pages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .errors import NamespaceError, SchemaError
from .kind_contract import USER_CHAT_KINDS
from .memory_catalog import build_memory_card
from .memory_navigation import (
    MEMORY_DETAILS,
    decode_memory_cursor,
    encode_memory_cursor,
    normalize_page_size,
)
from .namespace import Namespace
from .projection import RendererRegistry, canonical_json_bytes
from .rendering import render_timeline
from .store.base import MemoryStore
from .timeline import TimelineEntry
from .timeline_read import build_timeline_read_units, group_timeline_entries, normalize_timeline_projection
from .token_counter import TokenCounter, estimate_text_tokens


def normalize_speaker_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("speaker_ids_must_be_nonempty_array")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("speaker_ids_must_contain_strings")
        if not item.strip() or len(item) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in item):
            raise ValueError("invalid_speaker_id")
        if item != item.strip():
            raise ValueError("invalid_speaker_id")
        if item not in ids:
            ids.append(item)
    return ids


@dataclass
class _SourceGroup:
    unit_id: str
    key: tuple[Any, ...]
    messages: list[dict[str, Any]] = field(default_factory=list)
    rendered_messages: list[dict[str, Any]] = field(default_factory=list)
    card: dict[str, Any] | None = None
    compacted: int = 0
    raw_entries: tuple[TimelineEntry, ...] = ()
    projected: bool = False


@dataclass
class _SourceTree:
    memory_id: str
    node_type: str = ""
    header: dict[str, Any] = field(default_factory=dict)
    source_count: int = 0
    missing: list[str] = field(default_factory=list)
    unknown_speakers: int = 0
    fingerprint: str = ""
    groups: list[_SourceGroup] = field(default_factory=list)
    available: list[_SourceGroup] = field(default_factory=list)
    selected: list[_SourceGroup] = field(default_factory=list)
    error: str = ""
    error_status: str = "unavailable"
    error_actions: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    render_cache: dict[int, tuple[str, tuple[str, ...]]] = field(default_factory=dict)


def _invalid(reason: str) -> dict[str, Any]:
    return {"status": "invalid_filter", "reason": reason, "memory_id": "", "view": "", "result": None}


def _scoped_records(store: MemoryStore, namespace: Namespace, ids: list[str], cross: bool) -> dict[str, Any]:
    try:
        return store.get_retrieval_records(namespace=namespace, source_ids=tuple(ids), cross_conversation=cross)
    except (SchemaError, NamespaceError, NotImplementedError):
        # An ambiguous or unavailable sibling must not discard other trees.
        records: dict[str, Any] = {}
        for sid in ids:
            try:
                records[sid] = store.get_retrieval_record(namespace=namespace, source_id=sid, cross_conversation=cross)
            except (SchemaError, NamespaceError, NotImplementedError) as exc:
                records[sid] = exc
        return records


def _entry_header(entry: TimelineEntry) -> dict[str, Any]:
    actor = entry.namespace.actor
    return {
        "source_id": entry.source_id,
        "kind": entry.kind,
        "origin": str(entry.origin),
        "turn_role": str(entry.turn_role or ""),
        "actor_id": actor.stable_id if actor else "",
        "actor_display_name": actor.display_name if actor else "",
        "timestamp": entry.timestamp,
        "seq_no": entry.seq_no,
        "turn_id": entry.turn_id,
        "correlation_id": entry.correlation_id,
        "reply_to_source_id": entry.reply_to_source_id,
        "conversation_id": entry.namespace.conversation_id,
        "tenant_id": entry.namespace.tenant_id,
        "user_id": entry.namespace.user_id,
        "domain_id": entry.namespace.domain_id,
    }


def _entry_in_scope(entry: TimelineEntry, namespace: Namespace, cross: bool) -> bool:
    return entry.namespace.hard_key() == namespace.hard_key() and (
        cross or entry.namespace.conversation_id == namespace.conversation_id
    )


def _user_utterance(header: Mapping[str, Any]) -> bool:
    return header.get("kind") in USER_CHAT_KINDS and header.get("origin") == "user"


def _source_header(card: Mapping[str, Any]) -> dict[str, Any]:
    return {key: card[key] for key in ("memory_title", "period_start_at", "period_end_at") if card.get(key)}


def _body_fingerprint_fields(entry: TimelineEntry, *, speakers: bool) -> list[Any]:
    if speakers:
        return [entry.source_id, entry.semantic_text]
    target = entry.target_actor
    return [
        entry.source_id,
        entry.semantic_text,
        entry.payload,
        entry.trace_metadata,
        entry.memory_metadata,
        entry.content,
        entry.renderer_id,
        entry.renderer_version,
        target.stable_id if target else "",
        target.display_name if target else "",
    ]


def open_memory_sources(
    *,
    store: MemoryStore,
    namespace: Namespace,
    renderer_registry: RendererRegistry,
    timezone: str,
    token_counter: TokenCounter | None,
    enrich_records: Callable[..., list[dict[str, Any]]],
    memory_id: str = "",
    memory_ids: Any = None,
    view: str = "sources",
    detail: str = "full",
    projection: str = "conversation",
    cross_conversation: bool = False,
    page_size: int = 50,
    speaker_ids: Any = None,
    page_token_budget: int = 0,
    page_byte_budget: int = 0,
    cursor: str = "",
) -> dict[str, Any]:
    """Read only selected lineage; budget zero is trusted unlimited diagnostics."""
    after_key = None
    expected_fingerprint = ""
    batch = memory_ids is not None
    try:
        if isinstance(page_token_budget, bool) or not isinstance(page_token_budget, int) or page_token_budget < 0:
            raise ValueError("page_token_budget_must_be_nonnegative_integer")
        if isinstance(page_byte_budget, bool) or not isinstance(page_byte_budget, int) or page_byte_budget < 0:
            raise ValueError("page_byte_budget_must_be_nonnegative_integer")
        if cursor:
            if (
                memory_id
                or memory_ids is not None
                or speaker_ids is not None
                or (
                    view != "card"
                    or detail != "full"
                    or projection != "conversation"
                    or cross_conversation
                    or page_size != 50
                )
            ):
                raise ValueError("cursor_options_are_embedded")
            decoded = decode_memory_cursor(cursor, namespace=namespace, expected_mode="open_sources")
            selector = decoded["selector"]
            memory_id = str(selector["memory_id"])
            detail = str(selector["detail"])
            projection = str(selector["projection"])
            cross_conversation = bool(selector["cross_conversation"])
            speaker_ids = selector.get("speaker_ids") or None
            expected_fingerprint = str(selector.get("source_fingerprint") or "")
            if not expected_fingerprint:
                failure = _invalid("source_cursor_upgrade_restart_required")
                failure["memory_id"] = memory_id
                failure["view"] = "sources"
                failure["suggested_next_actions"] = [
                    _restart_call(
                        _SourceTree(memory_id), detail, projection, [], int(decoded["page_size"]), cross_conversation
                    )
                ]
                return failure
            frozen_budget = int(selector.get("page_token_budget") or 0)
            frozen_bytes = int(selector.get("page_byte_budget") or 0)
            if frozen_budget < 0 or frozen_bytes < 0:
                raise ValueError("invalid_cursor")
            # The host may impose a smaller current limit, never lift a frozen one.
            page_token_budget = (
                min(frozen_budget, page_token_budget)
                if frozen_budget and page_token_budget
                else (frozen_budget or page_token_budget)
            )
            page_byte_budget = (
                min(frozen_bytes, page_byte_budget)
                if frozen_bytes and page_byte_budget
                else (frozen_bytes or page_byte_budget)
            )
            page_size = int(decoded["page_size"])
            after_key = tuple(decoded["last_key"])
            view = "sources"
            batch = False
        if view != "sources":
            raise ValueError(
                "speaker_filter_requires_sources_view" if speaker_ids is not None else "invalid_memory_view"
            )
        if detail not in MEMORY_DETAILS:
            raise ValueError("invalid_memory_detail")
        projection = normalize_timeline_projection(projection)
        page_size = normalize_page_size(page_size)
        speakers = normalize_speaker_ids(speaker_ids)
        if speakers and projection != "conversation":
            raise ValueError("speaker_filter_requires_conversation_projection")
        if batch:
            if memory_id or not isinstance(memory_ids, (list, tuple)) or not memory_ids:
                raise ValueError("memory_id_memory_ids_are_mutually_exclusive" if memory_id else "memory_ids_required")
            if any(not isinstance(sid, str) or not sid.strip() for sid in memory_ids):
                raise ValueError("memory_ids_must_contain_nonempty_strings")
            requested = list(dict.fromkeys(sid.strip() for sid in memory_ids))
        else:
            if not isinstance(memory_id, str) or not memory_id.strip():
                raise ValueError("memory_id_required")
            requested = [memory_id.strip()]
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid(str(exc) or "invalid_cursor")

    roots = _scoped_records(store, namespace, requested, cross_conversation)
    missing_roots = tuple(sid for sid in requested if roots.get(sid) is None)
    try:
        replacement_map = (
            store.get_memory_replacement_map(
                namespace=namespace, memory_ids=missing_roots, cross_conversation=cross_conversation
            )
            if missing_roots
            else {}
        )
    except NotImplementedError:
        replacement_map = {}
    trees: list[_SourceTree] = []
    children: dict[str, list[str]] = {}
    raw_ids: list[str] = []
    episode_ids: list[str] = []
    for sid in requested:
        tree = _SourceTree(sid)
        trees.append(tree)
        root = roots.get(sid)
        if isinstance(root, Exception):
            tree.error = str(root) or "ambiguous_memory_id"
            continue
        if root is None:
            tree.error, tree.error_status = "memory_not_found_or_out_of_scope", "empty"
            replacements = replacement_map.get(sid)
            if replacements is not None:
                tree.error, tree.error_status = "semantic_history_repaired", "replaced"
                tree.error_actions = [
                    {
                        "tool": "open_memory",
                        "arguments": {
                            "memory_id": replacement,
                            "view": "sources",
                            "projection": projection,
                            **({"cross_conversation": True} if cross_conversation else {}),
                        },
                    }
                    for replacement in replacements
                ]
            continue
        card = build_memory_card(root, timezone=timezone)
        tree.node_type = str(card["node_type"])
        tree.header = _source_header(card)
        if speakers and tree.node_type != "episodic":
            tree.error = "speaker_filter_requires_episodic_sources"
            continue
        if tree.node_type == "raw":
            tree.error, tree.error_status = "raw_node_has_no_sources", "empty"
            continue
        ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in (root.get("source_ids") if tree.node_type == "episodic" else root.get("source_summary_ids"))
                or []
                if str(item or "").strip()
            )
        )
        children[sid] = ids
        tree.source_count = len(ids)
        (raw_ids if tree.node_type == "episodic" else episode_ids).extend(ids)

    raw_ids = list(dict.fromkeys(raw_ids))
    headers: dict[str, dict[str, Any]] = {}
    entries: list[TimelineEntry] = []
    raw_error = ""
    try:
        if speakers and raw_ids:
            try:
                headers = store.get_source_entry_headers(
                    namespace=namespace, source_ids=tuple(raw_ids), cross_conversation=cross_conversation
                )
            except NotImplementedError:
                entries = store.get_entries_by_source_ids(
                    namespace=namespace, source_ids=tuple(raw_ids), cross_conversation=cross_conversation
                )
                headers = {entry.source_id: _entry_header(entry) for entry in entries}
            else:
                selected_ids = tuple(
                    sid
                    for sid in raw_ids
                    if _user_utterance(headers.get(sid, {})) and headers[sid].get("actor_id") in speakers
                )
                entries = (
                    store.get_entries_by_source_ids(
                        namespace=namespace, source_ids=selected_ids, cross_conversation=cross_conversation
                    )
                    if selected_ids
                    else []
                )
        elif raw_ids:
            entries = store.get_entries_by_source_ids(
                namespace=namespace, source_ids=tuple(raw_ids), cross_conversation=cross_conversation
            )
            headers = {entry.source_id: _entry_header(entry) for entry in entries}
    except (NotImplementedError, NamespaceError, SchemaError) as exc:
        raw_error = str(exc) or "memory_source_store_unsupported"
    by_id = {entry.source_id: entry for entry in entries if _entry_in_scope(entry, namespace, cross_conversation)}
    if speakers:
        by_id = {
            sid: entry
            for sid, entry in by_id.items()
            if _user_utterance(_entry_header(entry))
            and entry.namespace.actor
            and entry.namespace.actor.stable_id in speakers
        }

    child_records = _scoped_records(store, namespace, list(dict.fromkeys(episode_ids)), cross_conversation)
    valid_children = [
        record
        for record in child_records.values()
        if isinstance(record, dict) and record.get("entry_type") == "summary"
    ]
    enriched = enrich_records(valid_children, cross_conversation=cross_conversation) if valid_children else []
    child_cards = {str(record["summary_id"]): build_memory_card(record, timezone=timezone) for record in enriched}
    for tree in trees:
        if tree.error:
            if expected_fingerprint:
                failure = _invalid("source_changed_restart_required")
                failure["memory_id"] = tree.memory_id
                failure["view"] = "sources"
                failure["suggested_next_actions"] = [
                    _restart_call(tree, detail, projection, speakers, page_size, cross_conversation)
                ]
                return failure
            continue
        ids = children[tree.memory_id]
        if tree.node_type == "episodic":
            if raw_error:
                tree.error = raw_error
                continue
            tree.missing = [
                sid
                for sid in ids
                if sid not in headers
                or (not speakers and sid not in by_id)
                or (_user_utterance(headers[sid]) and headers[sid].get("actor_id") in speakers and sid not in by_id)
            ]
            tree.unknown_speakers = (
                sum(
                    _user_utterance(headers.get(sid, {})) and not headers[sid].get("actor_id")
                    for sid in ids
                    if sid in headers
                )
                if speakers
                else 0
            )
            local_entries = [by_id[sid] for sid in ids if sid in by_id]
            if projection == "tools":
                # Explicit tools views omit non-tool groups. Preserve that
                # established projection contract through its shared builder.
                units = build_timeline_read_units(
                    local_entries, projection=projection, renderer_registry=renderer_registry, timezone=timezone
                )
                tree.groups = [
                    _SourceGroup(
                        unit.unit_id,
                        unit.sort_key,
                        list(unit.projected_messages),
                        _with_source_ids(list(unit.projected_messages)),
                        compacted=unit.compacted_entry_count,
                        projected=True,
                    )
                    for unit in units
                ]
            else:
                tree.groups = [
                    _SourceGroup(unit_id, key, raw_entries=ordered)
                    for unit_id, key, ordered in group_timeline_entries(local_entries)
                ]
            # Freeze the exact source data without rendering every unread group
            # or serializing full provider records to compute the checksum.
            evidence = {
                "headers": [[sid, headers.get(sid)] for sid in ids],
                "bodies": [_body_fingerprint_fields(entry, speakers=bool(speakers)) for entry in local_entries],
            }
        else:
            tree.missing = [sid for sid in ids if sid not in child_cards]
            cards = sorted(
                (child_cards[sid] for sid in ids if sid in child_cards),
                key=lambda card: (card["period_start_ts"], card["period_end_ts"], card["memory_id"]),
            )
            tree.groups = [
                _SourceGroup(
                    card["memory_id"], (card["period_start_ts"], card["period_end_ts"], card["memory_id"]), card=card
                )
                for card in cards
            ]
            evidence = [[sid, child_cards.get(sid)] for sid in ids]
        # Headers/source fields/cards already contain JSON primitives. Avoid
        # recursively normalizing this whole tree a second time on every read.
        fingerprint_data = {"header": tree.header, "children": evidence}
        try:
            encoded_evidence = json.dumps(
                fingerprint_data,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            # Legacy stores may contain nonfinite/custom extension values.
            # Preserve the shared safe normalization only for those rare rows.
            encoded_evidence = canonical_json_bytes(fingerprint_data)
        tree.fingerprint = hashlib.sha256(encoded_evidence).hexdigest()
        if expected_fingerprint and tree.fingerprint != expected_fingerprint:
            failure = _invalid("source_changed_restart_required")
            failure["memory_id"] = tree.memory_id
            failure["view"] = "sources"
            failure["suggested_next_actions"] = [
                _restart_call(tree, detail, projection, speakers, page_size, cross_conversation)
            ]
            return failure
        try:
            if after_key is not None:
                types = (int, str, int, str) if tree.node_type == "episodic" else (int, int, str)
                if len(after_key) != len(types) or any(
                    type(value) is not expected for value, expected in zip(after_key, types, strict=True)
                ):
                    raise TypeError
            tree.available = [group for group in tree.groups if after_key is None or group.key > after_key]
        except TypeError:
            return _invalid("invalid_cursor")

    count = token_counter.count_text if token_counter is not None else estimate_text_tokens
    quality = token_counter.quality if token_counter is not None else "estimated"
    projected_groups: dict[tuple[str, ...], tuple[list[dict[str, Any]], list[dict[str, Any]], int]] = {}

    def prepare_group(group: _SourceGroup) -> None:
        if group.projected or group.card is not None:
            return
        key = tuple(entry.source_id for entry in group.raw_entries)
        cached = projected_groups.get(key)
        if cached is None:
            entries = group.raw_entries
            if speakers:
                # Excerpts contain the recorded utterance, not host payloads,
                # trace state or a receiver's identity.
                entries = tuple(
                    replace(
                        entry,
                        payload={"channel": entry.payload["channel"]} if entry.payload.get("channel") else {},
                        trace_metadata={},
                        target_actor=None,
                    )
                    for entry in entries
                )
            unit = build_timeline_read_units(
                entries, projection=projection, renderer_registry=renderer_registry, timezone=timezone
            )[0]
            messages = list(unit.projected_messages)
            cached = (messages, _with_source_ids(messages), unit.compacted_entry_count)
            projected_groups[key] = cached
        group.messages, group.rendered_messages, group.compacted = cached
        group.projected = True

    oversized = False
    accepted = 0
    if not page_token_budget and not page_byte_budget:
        for tree in trees:
            if not tree.error:
                tree.selected = tree.available[:page_size]
                for group in tree.selected:
                    prepare_group(group)
    while page_token_budget or page_byte_budget:
        progressed = False
        for tree in trees:
            if tree.error or tree.blocked or len(tree.selected) >= min(page_size, len(tree.available)):
                continue
            group = tree.available[len(tree.selected)]
            prepare_group(group)
            tree.selected.append(group)
            candidate = _compose(
                trees,
                batch,
                detail,
                projection,
                cross_conversation,
                speakers,
                page_size,
                page_token_budget,
                timezone,
                namespace,
                page_byte_budget,
            )
            encoded = json.dumps(project_source_result(candidate), ensure_ascii=False, sort_keys=True)
            if page_byte_budget and len(encoded.encode("utf-8")) > page_byte_budget:
                rejected = tree.selected.pop()
                solo = _compose(
                    [replace(tree, selected=[rejected])],
                    False,
                    detail,
                    projection,
                    cross_conversation,
                    speakers,
                    page_size,
                    page_token_budget,
                    timezone,
                    namespace,
                    page_byte_budget,
                )
                solo_size = len(json.dumps(project_source_result(solo), ensure_ascii=False).encode("utf-8"))
                if not tree.selected and solo_size > page_byte_budget:
                    tree.error = "source_unit_exceeds_byte_budget"
                    if rejected.messages:
                        tree.error_actions = [
                            {"memory_id": rejected.messages[0]["source_id"], "view": "content", "detail": "compact"}
                        ]
                    elif rejected.card:
                        tree.error_actions = [{"memory_id": rejected.card["memory_id"], "view": "content"}]
                    if cross_conversation:
                        for action in tree.error_actions:
                            action["cross_conversation"] = True
                else:
                    tree.blocked = True
                continue
            tokens = count(encoded)
            if page_token_budget and tokens > page_token_budget:
                if accepted:
                    tree.selected.pop()
                    tree.blocked = True
                    continue
                oversized = True
                accepted += 1
                break
            accepted += 1
            progressed = True
        if oversized or not progressed:
            break
    result = _compose(
        trees,
        batch,
        detail,
        projection,
        cross_conversation,
        speakers,
        page_size,
        page_token_budget,
        timezone,
        namespace,
        page_byte_budget,
    )
    result["coverage"] = {
        "page_token_budget": page_token_budget,
        "page_byte_budget": page_byte_budget,
        "token_count_quality": quality,
        "oversized_unit": oversized,
    }
    result["coverage"]["returned_projected_token_count"] = count(
        json.dumps(project_source_result(result), ensure_ascii=False, sort_keys=True)
    )
    if (
        page_byte_budget
        and len(json.dumps(project_source_result(result), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        > page_byte_budget
    ):
        return {
            "status": "unavailable",
            "reason": "source_page_metadata_exceeds_byte_budget",
            "view": "sources",
            "result": None,
            "text": "",
        }
    return result


def _restart_call(
    tree: _SourceTree, detail: str, projection: str, speakers: list[str], page_size: int, cross: bool = False
) -> dict[str, Any]:
    call: dict[str, Any] = {"memory_id": tree.memory_id, "view": "sources"}
    if detail != "full":
        call["detail"] = detail
    if projection != "conversation":
        call["projection"] = projection
    if speakers:
        call["speaker_ids"] = speakers
    if page_size != 50:
        call["page_size"] = page_size
    if cross:
        call["cross_conversation"] = True
    return call


def _item(
    tree: _SourceTree,
    detail: str,
    projection: str,
    cross: bool,
    speakers: list[str],
    page_size: int,
    budget: int,
    timezone: str,
    namespace: Namespace,
    byte_budget: int,
    render_text: bool = True,
) -> dict[str, Any]:
    base = {
        "memory_id": tree.memory_id,
        "node_type": tree.node_type,
        "view": "sources",
        "detail": detail,
        "projection": projection,
        "cross_conversation": bool(cross),
    }
    if tree.error:
        return {
            **base,
            "status": tree.error_status,
            "reason": tree.error,
            "result": None,
            "text": "",
            **({"suggested_next_actions": tree.error_actions} if tree.error_actions else {}),
        }
    remaining = len(tree.available) - len(tree.selected)
    next_cursor = ""
    if remaining and tree.selected:
        next_cursor = encode_memory_cursor(
            namespace=namespace,
            mode="open_sources",
            page_size=page_size,
            selector={
                "memory_id": tree.memory_id,
                "view": "sources",
                "detail": detail,
                "projection": projection,
                "cross_conversation": bool(cross),
                "speaker_ids": speakers,
                "page_token_budget": budget,
                "page_byte_budget": byte_budget,
                "source_fingerprint": tree.fingerprint,
            },
            last_key=tree.selected[-1].key,
        )
    payload: dict[str, Any] = {
        **tree.header,
        "source_node_type": "raw" if tree.node_type == "episodic" else "episodic",
        "source_count": tree.source_count,
        "missing_source_ids": tree.missing,
        "page_complete": not remaining,
        "next_cursor": next_cursor,
    }
    if tree.node_type == "episodic":
        payload.update(
            logical_unit_count=len(tree.groups),
            returned_logical_unit_count=len(tree.selected),
            remaining_logical_unit_count=remaining,
            compacted_entry_count=sum(group.compacted for group in tree.selected),
            source_units=[{"unit_id": group.unit_id, "entries": group.messages} for group in tree.selected],
        )
        if tree.unknown_speakers:
            payload["unknown_speaker_count"] = tree.unknown_speakers
    else:
        payload.update(
            returned_source_count=len(tree.selected),
            remaining_source_count=remaining,
            sources=[group.card for group in tree.selected],
        )
    if remaining and not tree.selected:
        payload["suggested_next_actions"] = [_restart_call(tree, detail, projection, speakers, page_size, cross)]
    reason = (
        "source_lineage_incomplete"
        if tree.missing
        else ("speaker_identity_incomplete" if tree.unknown_speakers else "")
    )
    status = "partial" if reason else "ok"
    if not tree.groups and not reason:
        status, reason = "empty", "no_matching_speaker_utterances" if speakers else "no_source_evidence"
    elif remaining and not tree.selected:
        status, reason = "partial", "source_page_budget_exhausted"
    messages = [row for group in tree.selected for row in group.rendered_messages]
    return {
        **base,
        "status": status,
        "reason": reason,
        "result": payload,
        "text": render_timeline(messages, tz=timezone, _message_cache=tree.render_cache)
        if messages and render_text
        else "",
    }


def _compose(
    trees: list[_SourceTree],
    batch: bool,
    detail: str,
    projection: str,
    cross: bool,
    speakers: list[str],
    page_size: int,
    budget: int,
    timezone: str,
    namespace: Namespace,
    byte_budget: int,
) -> dict[str, Any]:
    items = [
        _item(
            tree,
            detail,
            projection,
            cross,
            speakers,
            page_size,
            budget,
            timezone,
            namespace,
            byte_budget,
            render_text=not batch,
        )
        for tree in trees
    ]
    mode = "speaker_filtered" if speakers else "complete_sources"
    if not batch:
        return {**items[0], "read_mode": mode, **({"speaker_ids": speakers} if speakers else {})}
    seen: set[str] = set()
    blocks: list[str] = []
    for tree, item in zip(trees, items, strict=True):
        messages: list[dict[str, Any]] = []
        shared: list[str] = []
        for group in tree.selected:
            for row in group.rendered_messages:
                sid = str(row["source_id"])
                if sid in seen:
                    shared.append(sid)
                else:
                    seen.add(sid)
                    messages.append(row)
        body = render_timeline(messages, tz=timezone, _message_cache=tree.render_cache) if messages else ""
        if shared:
            body += (
                "\n[shared source_ids: "
                + ", ".join(dict.fromkeys(shared))
                + "; original text appears earlier in this page]"
            )
        blocks.append(
            f"[memory_id={tree.memory_id}]\n{body or item['reason'] or 'child episode cards are listed in sources'}"
        )
    complete = all((item.get("result") or {}).get("page_complete", True) for item in items)
    errors = [item for item in items if item["status"] not in {"ok", "empty"} or item.get("result") is None]
    status = "partial" if errors or not complete else "ok"
    if not any(tree.selected for tree in trees) and not errors and complete:
        status = "empty"
    return {
        "status": status,
        "reason": "some_source_nodes_incomplete" if status == "partial" else "",
        "memory_id": "",
        "memory_ids": [tree.memory_id for tree in trees],
        "view": "sources",
        "detail": detail,
        "projection": projection,
        "cross_conversation": bool(cross),
        "read_mode": mode,
        **({"speaker_ids": speakers} if speakers else {}),
        "result": {
            "requested_count": len(items),
            "opened_count": sum(item.get("result") is not None for item in items),
            "failed_count": sum(item.get("result") is None for item in items),
            "items": items,
            "page_complete": complete,
        },
        "text": "\n\n".join(blocks),
    }


def _with_source_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in messages:
        # The shared human speaker label escapes structural punctuation. Keep
        # platform IDs exact and copyable in this source anchor, once per role.
        anchor = {"source_id": row["source_id"]}
        rendered = dict(row)
        for key in ("actor_id", "target_actor_id"):
            stable_id = str(row.get(key) or "")
            if stable_id:
                anchor[key] = stable_id
                rendered[key] = ""
                name_key = key.removesuffix("_id") + "_display_name"
                if row.get(name_key) == stable_id:
                    rendered[name_key] = ""
        labels = "; ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in anchor.items())
        rendered["content"] = f"[{labels}]\n{row['content']}"
        rows.append(rendered)
    return rows


def project_source_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """One evidence body plus the bounded information needed to navigate it."""
    model = {
        key: result[key]
        for key in (
            "status",
            "reason",
            "memory_id",
            "memory_ids",
            "node_type",
            "view",
            "projection",
            "read_mode",
            "speaker_ids",
            "text",
            "suggested_next_actions",
        )
        if result.get(key)
    }
    payload = result.get("result")
    if isinstance(payload, Mapping):
        items = payload.get("items")
        if isinstance(items, list):
            projected_items = []
            for item in items:
                projected = project_source_result(item)
                for key in ("text", "view", "projection"):
                    projected.pop(key, None)
                projected_items.append(projected)
            model["result"] = {"items": projected_items, "page_complete": payload["page_complete"]}
        else:
            compact = {
                key: payload[key]
                for key in (
                    "memory_title",
                    "period_start_at",
                    "period_end_at",
                    "source_node_type",
                    "source_count",
                    "returned_logical_unit_count",
                    "remaining_logical_unit_count",
                    "returned_source_count",
                    "remaining_source_count",
                    "page_complete",
                    "next_cursor",
                    "unknown_speaker_count",
                    "suggested_next_actions",
                    "sources",
                )
                if key in payload and (payload[key] or key == "page_complete" or key.startswith("returned_"))
            }
            if payload.get("missing_source_ids"):
                compact["missing_source_count"] = len(payload["missing_source_ids"])
            if "source_units" in payload:
                compact["returned_logical_unit_ids"] = [unit["unit_id"] for unit in payload["source_units"]]
            model["result"] = compact
    if (result.get("coverage") or {}).get("oversized_unit"):
        model["coverage"] = {
            key: result["coverage"][key] for key in ("oversized_unit", "page_token_budget", "token_count_quality")
        }
    if result.get("view") == "sources" and result.get("text"):
        model["result_projection"] = (
            "batch_rendered_text_with_navigation_metadata"
            if isinstance(payload, Mapping) and isinstance(payload.get("items"), list)
            else "rendered_text_with_navigation_metadata"
        )
    return model
