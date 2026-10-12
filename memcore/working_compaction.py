"""Summarize a paired old tool segment before publishing its compact overlay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from .chat_compaction import paired_tool_boundary
from .llm.base import TaskType
from .projection import ProjectionMessage, ProjectionStatus, canonical_json_bytes, stable_projection_hash
from .settlement import build_settlement_plan
from .timeline import TurnRole


WORKING_CONTRACT = (
    'Return JSON {"summary":"..."}. Preserve completed changes, evidence/results, '
    "decisions, constraints and remaining work as of the selected range. Keep "
    "failure, cancellation and completion distinct; do not invent success. "
    "Keep useful task/artifact IDs verbatim. Be concise but sufficient to continue."
)


def apply_working_segments(
    messages: list[ProjectionMessage], segments: list[dict[str, Any]]
) -> list[ProjectionMessage]:
    result = list(messages)
    for segment in segments:
        ids = set(segment["source_ids"])
        indexes = [i for i, message in enumerate(result) if ids.intersection(message.source_ids)]
        if not indexes:
            continue
        covered = {sid for i in indexes for sid in result[i].source_ids}
        if not ids <= covered or any(not set(result[i].source_ids) <= ids for i in indexes):
            continue
        start, end = min(indexes), max(indexes)
        # Scope/version checks happen atomically at publication. Never consume
        # a newer unrelated message accidentally on a later read.
        if indexes != list(range(start, end + 1)):
            continue
        anchor = result[start]
        derived = [
            replace(
                anchor,
                turn_id="working." + segment["segment_id"],
                projection_index=i,
                source_ids=tuple(item["source_ids"]),
                payload=dict(item["payload"]),
                payload_hash=stable_projection_hash(item["payload"]),
                projection_status=ProjectionStatus.SETTLED,
            )
            for i, item in enumerate(segment["messages"])
        ]
        result[start : end + 1] = derived
    return result


def compact_working_turn(
    compaction: Any, namespace: Any, bundles: list[Any], result: dict[str, Any], *, provider_profile: str
) -> None:
    getter = getattr(compaction.store, "get_working_segments", None)
    commit = getattr(compaction.store, "commit_working_segment", None)
    if not callable(getter) or not callable(commit):
        result["working_status"] = "unavailable"
        result["reason"] = "store_working_compaction_unsupported"
        return
    try:
        segments = getter(namespace=namespace, provider_profile=provider_profile)
    except NotImplementedError:
        result["working_status"] = "unavailable"
        result["reason"] = "store_working_compaction_unsupported"
        return
    with compaction._context_lock:
        context = compaction._chat_context
    trigger = compaction.config.working_token_trigger
    if context is not None and context.working_token_limit:
        trigger = min(trigger, context.working_token_limit)
    for bundle in bundles:
        if bundle.status.terminal:
            continue
        full = compaction.projection_ledger.freeze_turn_entries(
            namespace=namespace,
            turn_id=bundle.turn_id,
            entries=bundle.entries,
            provider_profile=provider_profile,
        )
        effective = apply_working_segments(list(full), segments)
        effective_tokens = sum(compaction._count_projection_tokens(message) for message in effective)
        if effective_tokens < trigger:
            continue
        folded = {sid for segment in segments for sid in segment["source_ids"]}
        entries_by_id = {entry.source_id: entry for entry in bundle.entries}
        candidate: list[ProjectionMessage] = []
        candidate_tokens = 0
        pending_calls: set[str] = set()
        chosen: list[ProjectionMessage] = []
        target = max(1, int(effective_tokens * compaction.config.raw_token_batch_ratio))
        for message in full:
            if folded.intersection(message.source_ids):
                continue
            entries = [entries_by_id[sid] for sid in message.source_ids if sid in entries_by_id]
            if not entries or any(entry.turn_role is TurnRole.STIMULUS for entry in entries):
                if candidate:
                    break
                continue
            candidate.append(message)
            for entry in entries:
                if entry.turn_role is TurnRole.ACTION:
                    pending_calls.add(entry.correlation_id)
                elif entry.turn_role is TurnRole.OBSERVATION:
                    pending_calls.discard(entry.correlation_id)
            candidate_tokens += compaction._count_projection_tokens(message)
            if not pending_calls and paired_tool_boundary([item.payload for item in candidate]):
                chosen = list(candidate)
                if candidate_tokens >= target:
                    break
        if not chosen or not any(
            entries_by_id[sid].turn_role is TurnRole.ACTION for m in chosen for sid in m.source_ids
        ):
            result["working_status"] = "no_paired_segment"
            continue
        prefix = compaction._select_chat_prefix(chosen, "working")
        # Working summaries must use the captured effective request. A stale
        # or absent capture waits for the next boundary instead of re-rendering.
        if prefix is None:
            result["working_status"] = "waiting_for_request_snapshot"
            continue
        turn = compaction.store.get_turn(namespace=namespace, turn_id=bundle.turn_id)
        if turn is None:
            result["working_status"] = "stale_batch"
            result["status"] = "stale_batch"
            result["reason"] = "working_source_changed"
            return
        call = compaction._call_json(TaskType.SUMMARY, WORKING_CONTRACT, "", fallback={}, chat_prefix=prefix)
        if compaction._stop_if_shutdown(result):
            return
        text = str(call.data.get("summary") or "").strip() if call.ok else ""
        if not text:
            result["status"] = "failed"
            result["reason"] = "working_summary_retry_pending"
            return
        selected_text = canonical_json_bytes([message.payload for message in chosen]).decode("utf-8")
        pending = tuple(task_id for task_id in prefix.snapshot.pending_task_ids if task_id in selected_text)
        if pending:
            text += "\npending_tasks_as_of_segment: " + ", ".join(pending)
        ids = {sid for message in chosen for sid in message.source_ids}
        entries = sorted((entries_by_id[sid] for sid in ids), key=lambda entry: entry.seq_no)
        policy = turn.operation_projection_policy
        cards = []
        if policy != "full_until_raw_compaction":
            plan = build_settlement_plan(
                compaction.projection_adapter,
                entries,
                provider_profile=provider_profile,
                authoritative_messages=chosen,
                timezone=compaction.timezone,
                compact_arguments=policy == "compact_arguments_after_terminal",
                settlement_min_utf8_bytes=turn.operation_settlement_min_utf8_bytes,
                settlement_min_saved_ratio=turn.operation_settlement_min_saved_ratio,
                settlement_argument_min_utf8_bytes=turn.operation_argument_min_utf8_bytes,
                settlement_argument_preview_chars=turn.operation_argument_preview_chars,
            )
            cards = [{"payload": dict(m.payload), "source_ids": list(m.source_ids)} for m in plan.messages]
        hashes: dict[str, list[str]] = defaultdict(list)
        for message in chosen:
            for sid in message.source_ids:
                hashes[sid].append(message.payload_hash)
        # A source-addressable overlay, not a user-facing answer. It lives
        # after exactly A; new B remains in its original order behind it.
        summary = {
            "payload": {"role": "user", "content": "[working summary; as of preceding segment]\n" + text},
            "source_ids": [entry.source_id for entry in entries],
        }
        derived = [*cards, summary]
        before_size = sum(compaction._count_projection_tokens(message) for message in chosen)
        after_size = sum(
            compaction._count_text_tokens(canonical_json_bytes(item["payload"]).decode("utf-8")) + 4 for item in derived
        )
        if after_size >= before_size:
            result["working_status"] = "no_savings"
            result["status"] = "blocked_by_open_turn"
            result["reason"] = "working_projection_no_savings"
            return
        try:
            published = commit(
                namespace=namespace,
                provider_profile=provider_profile,
                turn_id=bundle.turn_id,
                source_versions=tuple((entry.source_id, entry.row_version) for entry in entries),
                projection_hashes=tuple((entry.source_id, tuple(hashes[entry.source_id])) for entry in entries),
                messages=derived,
                policy=policy,
            )
        except NotImplementedError:
            result["working_status"] = "unavailable"
            result["reason"] = "store_working_compaction_unsupported"
            return
        result["working_status"] = published["status"]
        if published["status"] in {"committed", "already_committed"}:
            result["status"] = "compacted"
            result["working_source_count"] = len(entries)
            result["working_summary_created"] = 1
            after = apply_working_segments(effective, getter(namespace=namespace, provider_profile=provider_profile))
            saved = effective_tokens - sum(compaction._count_projection_tokens(message) for message in after)
            result["after_raw_projected_tokens"] = max(0, result["before_raw_projected_tokens"] - saved)
            result["after_projected_tokens"] = max(0, result["before_projected_tokens"] - saved)
        else:
            result["status"] = published["status"]
            result["reason"] = published.get("reason", "")
        return
