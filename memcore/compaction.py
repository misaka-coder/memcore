"""写侧:三层压缩(raw→摘要→语义)+ 强化合并。注入 LLMClient,按 namespace 加锁同步执行。

差值关系(批量 < 触发数)由 MemoryConfig 保证。强化合并按主题重叠驱动(非位置淘汰):
新长期记忆形成时,回看最近 N 条,与重叠最高且 >= 阈值者融合,否则新建。

并发:压缩核心每 namespace 串行锁,压缩中不与自身重入(代际安全);MemorySystem 提供后台提交入口。
向量索引用 outbox:先写库(pending),upsert 成功后 set_index_status('indexed')。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

from .compaction_v2 import (
    CompactionResult,
    CompactionSnapshot,
    SemanticCommitInput,
    SemanticSnapshot,
    SummaryRecordInput,
    TurnBundle,
)
from .config import MemoryConfig
from .errors import SchemaError, StaleSnapshotError
from .index.base import VectorIndex
from .index.entry_builder import build_semantic_entry, build_summary_entry
from .index.metadata_filters import INDEX_SCHEMA_KEY, INDEX_SCHEMA_VERSION
from .llm.base import LLMClient, LLMRequest, ResponseFormat, TaskType
from .memory_catalog import (
    CATALOG_SCHEMA_VERSION,
    collect_participant_refs,
    count_logical_turns,
    normalize_catalog_fields,
)
from .namespace import Namespace
from .prompts import PromptOverrides, build_reinforcement_prompts, build_semantic_prompts, build_summary_prompts
from .projection import (
    ProjectionAdapter,
    ProjectionLedger,
    ProjectionMessage,
    canonical_json_bytes,
    default_renderer_registry,
    sanitize_projection_payload,
    stable_projection_hash,
)
from .rendering import render_raw_snippet
from .schema import RETRIEVAL_PRIORITIES, coerce_memory_metadata
from .store.base import MemoryStore
from .runtime import MemCoreRuntime
from .text_utils import normalize_text
from .time_anchor import (
    TIME_PERIOD_LABELS,
    format_time_range_label,
    infer_time_of_day,
    timestamp_to_date_label,
)
from .timeline import (
    MAX_OPERATION_RETENTION_ANCHOR_BYTES,
    OPERATION_RETENTION_ANCHOR_KEY,
    OPERATION_RETENTION_ANCHOR_STATUS_KEY,
    TimelineEntry,
    TurnRole,
)
from .token_counter import TokenCounter, estimate_text_tokens


@dataclass
class JsonCallResult:
    ok: bool
    data: dict[str, Any]


_MAX_OPERATION_RETENTION_TOTAL_BYTES = 16_384


class Compaction:
    def __init__(
        self,
        *,
        store: MemoryStore,
        index: VectorIndex,
        llm: LLMClient,
        config: MemoryConfig,
        timezone: str,
        overrides: PromptOverrides | None = None,
        token_counter: TokenCounter | None = None,
        projection_adapter: ProjectionAdapter | None = None,
        runtime: MemCoreRuntime | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.llm = llm
        self.config = config
        self.timezone = timezone
        self.overrides = overrides or PromptOverrides()
        if token_counter is not None and not isinstance(token_counter, TokenCounter):
            raise TypeError("token_counter must be a TokenCounter instance or None")
        self.token_counter = token_counter
        self.projection_adapter = projection_adapter or ProjectionAdapter(
            renderer_registry=default_renderer_registry(),
            timezone=timezone,
        )
        self.projection_ledger = ProjectionLedger(
            store=self.store,
            adapter=self.projection_adapter,
            enable_flavor=self.config.enable_flavor,
        )
        self.runtime = runtime or MemCoreRuntime()
        self._shutdown_requested = threading.Event()

    def request_shutdown(self) -> None:
        """Cooperatively stop this MemorySystem's compaction work.

        Provider calls cannot be interrupted portably once they are on the
        wire, so callers still bound transport timeouts.  This signal prevents
        retries and, critically, prevents a late response from committing new
        memory after host shutdown has started.
        """

        self._shutdown_requested.set()

    def _stop_if_shutdown(self, result: dict[str, Any]) -> bool:
        if not self._shutdown_requested.is_set():
            return False
        result["status"] = "cancelled"
        result["reason"] = "shutdown_requested"
        return True

    def run_due(self, *, namespace: Namespace, provider_profile: str = "") -> dict[str, Any]:
        profile = str(provider_profile or self.config.projection_profile).strip().lower()
        result = CompactionResult().to_dict()
        result["provider_profile"] = profile
        if self._stop_if_shutdown(result):
            return result
        lock = self.runtime.lock_registry.lock_for(
            store_identity=self.store.runtime_identity(),
            namespace=namespace,
        )
        if not lock.acquire(blocking=False):
            result["status"] = "busy"
            return result
        try:
            for snapshot_attempt in range(2):
                result = CompactionResult().to_dict()
                result["provider_profile"] = profile
                try:
                    if self._stop_if_shutdown(result):
                        return result
                    try:
                        bundles = self.store.list_compaction_bundles(namespace=namespace)
                    except NotImplementedError:
                        bundles = []
                    if bundles:
                        self._summarize_timeline(namespace, bundles, result, provider_profile=profile)
                    if self._stop_if_shutdown(result):
                        return result
                    self._semanticize_episodic(namespace, result)
                    return result
                except StaleSnapshotError:
                    if snapshot_attempt == 0:
                        continue
                    result["status"] = "stale_projection"
                    result["reason"] = "projection_generation_changed"
                    return result
            return result  # pragma: no cover - bounded loop always returns
        except SchemaError as exc:
            result = CompactionResult().to_dict()
            result["provider_profile"] = profile
            result["status"] = "failed"
            result["reason"] = str(exc)
            return result
        finally:
            lock.release()

    # --- raw → 阶段摘要 ---

    def _summarize_timeline(
        self,
        namespace: Namespace,
        bundles: list[TurnBundle],
        result: dict[str, Any],
        *,
        provider_profile: str,
    ) -> None:
        if not bundles:
            return
        profile = provider_profile
        generation, _ = self.store.get_conversation_generations(namespace=namespace)
        result["compaction_generation"] = generation
        projections: dict[str, list[ProjectionMessage]] = {}
        bundle_tokens: dict[str, int] = {}
        token_quality = self.token_counter.quality if self.token_counter is not None else "estimated"
        for bundle in bundles:
            frozen = self._freeze_bundle_projection(namespace, bundle, profile)
            projections[bundle.turn_id] = frozen
            bundle_tokens[bundle.turn_id] = sum(self._count_projection_tokens(item) for item in frozen)

        raw_projected_tokens = sum(bundle_tokens.values())
        memory_tokens = self._visible_memory_projection_tokens(namespace, profile)
        before_tokens = memory_tokens + raw_projected_tokens
        result["before_projected_tokens"] = before_tokens
        result["before_raw_projected_tokens"] = raw_projected_tokens
        result["token_count_quality"] = token_quality
        components = self._bundle_components(bundles)
        due, removal_target = self._compaction_due_target(
            bundles=bundles,
            raw_projected_tokens=raw_projected_tokens,
        )
        result["planned_source_tokens"] = removal_target if due else 0
        if not due:
            result["status"] = "not_due"
            result["after_projected_tokens"] = before_tokens
            result["after_raw_projected_tokens"] = raw_projected_tokens
            return

        selected: list[TurnBundle] = []
        selected_tokens = 0
        total_bundle_count = len(bundles)
        blocked_reason = ""
        for component in components:
            if any(not bundle.status.terminal for bundle in component):
                blocked_reason = "open_turn_in_prefix"
                break
            if total_bundle_count - len(selected) - len(component) < self.config.compaction_min_recent_turns:
                blocked_reason = "recent_turn_window"
                break
            selected.extend(component)
            selected_tokens += sum(bundle_tokens[bundle.turn_id] for bundle in component)
            if selected_tokens >= removal_target:
                break

        # V1's oversized-first-turn escape hatch, upgraded to V2 terminal
        # components.  A complete history with no legal recent tail must not
        # remain permanently above the token trigger merely because
        # compaction_min_recent_turns is non-zero.
        if (
            not selected
            and blocked_reason == "recent_turn_window"
            and all(bundle.status.terminal for bundle in bundles)
        ):
            selected = list(bundles)
            selected_tokens = raw_projected_tokens
            blocked_reason = ""

        if not selected:
            result["status"] = "blocked_by_open_turn" if blocked_reason == "open_turn_in_prefix" else "not_due"
            result["reason"] = blocked_reason
            result["after_projected_tokens"] = before_tokens
            result["after_raw_projected_tokens"] = raw_projected_tokens
            return

        selected_ids = {bundle.turn_id for bundle in selected}
        ordered_entries = sorted(
            (entry for bundle in selected for entry in bundle.entries),
            key=lambda entry: entry.seq_no,
        )
        source_ids = tuple(entry.source_id for entry in ordered_entries)
        result["selected_projected_tokens"] = selected_tokens
        projection_hashes_by_source: dict[str, list[str]] = {}
        for turn_id in selected_ids:
            for message in projections[turn_id]:
                for source_id in message.source_ids:
                    projection_hashes_by_source.setdefault(source_id, []).append(message.payload_hash)
        snapshot = CompactionSnapshot(
            namespace_key=(
                namespace.tenant_id or "",
                namespace.user_id,
                namespace.domain_id or "",
                namespace.conversation_id or "",
            ),
            provider_profile=profile,
            compaction_generation=generation,
            bundles=tuple(selected),
            ordered_source_ids=source_ids,
            message_row_versions=tuple((entry.source_id, entry.row_version) for entry in ordered_entries),
            turn_row_versions=tuple(
                (bundle.turn_id, bundle.turn_row_version) for bundle in selected if not bundle.legacy
            ),
            projection_hashes=tuple(
                (source_id, tuple(projection_hashes_by_source.get(source_id, ()))) for source_id in source_ids
            ),
            before_projected_tokens=before_tokens,
            selected_projected_tokens=selected_tokens,
            token_count_quality=token_quality,
        )

        episode_entries = [entry for entry in ordered_entries if self._is_episode_entry(entry)]
        operation_entries = [entry for entry in ordered_entries if entry not in episode_entries]
        summary_inputs: list[SummaryRecordInput] = []
        if episode_entries:
            call = self._call_json(
                TaskType.SUMMARY,
                *build_summary_prompts(
                    transcript=self._render_transcript([entry.to_record() for entry in episode_entries]),
                    batch_size=len(episode_entries),
                    overrides=self.overrides,
                    enable_flavor=self.config.enable_flavor,
                    reference_summary_text=self._render_reference_summaries(namespace),
                ),
                fallback={
                    "diary_summary": "",
                    "importance": 0.3,
                    "key_events": [],
                    "core_facts": [],
                    "memory_title": "",
                    "catalog_hint": "",
                    "topic_headings": [],
                },
            )
            if self._stop_if_shutdown(result):
                return
            if not call.ok or not _has_summary_content(call.data):
                result["status"] = "failed"
                result["reason"] = "summary_retry_pending"
                result["summary_retry_pending"] += 1
                result["after_projected_tokens"] = before_tokens
                result["after_raw_projected_tokens"] = raw_projected_tokens
                return
            summary_inputs.append(
                self._episode_summary_input(
                    namespace=namespace,
                    entries=episode_entries,
                    payload=call.data,
                )
            )
        if operation_entries:
            summary_inputs.append(self._operation_summary_input(namespace=namespace, entries=operation_entries))

        if self._stop_if_shutdown(result):
            return
        committed = self.store.commit_summary_batch(
            namespace=namespace,
            snapshot=snapshot,
            records=summary_inputs,
        )
        if not committed.committed:
            result["status"] = committed.status
            result["reason"] = committed.reason
            result["after_projected_tokens"] = before_tokens
            result["after_raw_projected_tokens"] = raw_projected_tokens
            return
        for saved in committed.summaries:
            self._record_index_result(
                result,
                self._index(build_summary_entry(saved), str(saved["summary_id"])),
            )
        summary_messages: list[ProjectionMessage] = []
        for saved in committed.summaries:
            if str(saved.get("kind") or "") != "memory.episode_summary":
                continue
            summary_messages.extend(
                self.projection_ledger.freeze_memory_record(
                    namespace=namespace,
                    record=saved,
                    id_key="summary_id",
                    turn_prefix="summary",
                    provider_profile=profile,
                )
            )
        summary_tokens = sum(self._count_projection_tokens(message) for message in summary_messages)
        result["status"] = "compacted"
        result["source_turn_count"] = len(selected)
        result["source_entry_count"] = len(source_ids)
        result["summary_source_ids"] = list(source_ids)
        result["compaction_generation"] = committed.compaction_generation
        result["summaries_created"] += len(committed.summaries)
        result["after_projected_tokens"] = max(0, before_tokens - selected_tokens) + summary_tokens
        result["after_raw_projected_tokens"] = max(0, raw_projected_tokens - selected_tokens)

    @staticmethod
    def _is_episode_entry(entry: TimelineEntry) -> bool:
        return entry.turn_role not in {TurnRole.ACTION, TurnRole.OBSERVATION} and not entry.kind.startswith("material.")

    def _freeze_bundle_projection(
        self,
        namespace: Namespace,
        bundle: TurnBundle,
        provider_profile: str,
    ) -> list[ProjectionMessage]:
        from .settlement import load_settled_projection

        settled = load_settled_projection(
            self.store,
            namespace=namespace,
            turn_id=bundle.turn_id,
            provider_profile=provider_profile,
        )
        if settled is not None:
            return settled
        return self.projection_ledger.freeze_turn_entries(
            namespace=namespace,
            turn_id=bundle.turn_id,
            entries=bundle.entries,
            provider_profile=provider_profile,
        )

    def _visible_memory_projection_tokens(self, namespace: Namespace, provider_profile: str) -> int:
        episodic = self.store.get_visible_episodic_summaries(
            namespace=namespace,
            limit=self.config.episodic_visible_max,
            cross_conversation=False,
        )
        semantic = self.store.get_recent_semantic_summaries(
            namespace=namespace,
            limit=self.config.semantic_visible_limit,
            cross_conversation=False,
        )
        messages: list[ProjectionMessage] = []
        for record, id_key, prefix in (
            *((item, "semantic_id", "semantic") for item in reversed(semantic)),
            *(
                (item, "summary_id", "summary")
                for item in reversed(episodic)
                if str(item.get("kind") or "") == "memory.episode_summary"
            ),
        ):
            messages.extend(
                self.projection_ledger.freeze_memory_record(
                    namespace=namespace,
                    record=record,
                    id_key=id_key,
                    turn_prefix=prefix,
                    provider_profile=provider_profile,
                )
            )
        return sum(self._count_projection_tokens(message) for message in messages)

    @staticmethod
    def _bundle_components(bundles: list[TurnBundle]) -> list[list[TurnBundle]]:
        components: list[list[TurnBundle]] = []
        current_last = -1
        for bundle in sorted(bundles, key=lambda item: (item.first_seq_no, item.last_seq_no)):
            if not components or bundle.first_seq_no > current_last:
                components.append([bundle])
                current_last = bundle.last_seq_no
                continue
            components[-1].append(bundle)
            current_last = max(current_last, bundle.last_seq_no)
        return components

    def _compaction_due_target(
        self,
        *,
        bundles: list[TurnBundle],
        raw_projected_tokens: int,
    ) -> tuple[bool, int]:
        trigger = self.config.raw_token_trigger
        planned = max(1, math.ceil(trigger * float(self.config.raw_token_batch_ratio)))
        return (raw_projected_tokens >= trigger, planned)

    def _count_projection_tokens(self, message: ProjectionMessage) -> int:
        text = canonical_json_bytes(message.payload).decode("utf-8")
        return self._count_text_tokens(text) + 4

    def _count_text_tokens(self, text: str) -> int:
        if self.token_counter is None:
            return estimate_text_tokens(text)
        try:
            count = int(self.token_counter.count_text(text))
        except (TypeError, ValueError) as exc:
            raise ValueError("TokenCounter.count_text() must return a non-negative int") from exc
        if count < 0:
            raise ValueError("TokenCounter.count_text() must return a non-negative int")
        return count

    def _episode_summary_input(
        self,
        *,
        namespace: Namespace,
        entries: list[TimelineEntry],
        payload: dict[str, Any],
    ) -> SummaryRecordInput:
        source_ids = tuple(entry.source_id for entry in entries)
        summary_id = self._stable_summary_id(namespace, source_ids, profile="episode")
        timestamps = [int(entry.timestamp) for entry in entries]
        metadata = coerce_memory_metadata(
            payload.get("memory_metadata"),
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        accepted_metadata = [entry.memory_metadata for entry in entries if entry.annotation_status.accepted]
        for key in ("memory_facets", "about_roles", "entity_anchors", "topic_terms", "mood_tags"):
            if not metadata.get(key):
                metadata[key] = _merge_unique(*(item.get(key) for item in accepted_metadata))
        raw_model_metadata = payload.get("memory_metadata")
        if (
            not isinstance(raw_model_metadata, dict)
            or raw_model_metadata.get("retrieval_priority") not in RETRIEVAL_PRIORITIES
        ):
            metadata["retrieval_priority"] = _highest_priority(
                *(item.get("retrieval_priority") for item in accepted_metadata)
            )
        catalog = normalize_catalog_fields(payload)
        record = {
            "summary_id": summary_id,
            "kind": "memory.episode_summary",
            "timestamp": max(timestamps),
            "period_start_ts": min(timestamps),
            "period_end_ts": max(timestamps),
            "date_label": timestamp_to_date_label(max(timestamps), self.timezone),
            "time_of_day": infer_time_of_day(max(timestamps), self.timezone),
            "period_label": str(payload.get("period_label") or ""),
            "event_type": str(payload.get("event_type") or ""),
            "importance": _clamp01(payload.get("importance")),
            "diary_summary": str(payload.get("diary_summary") or ""),
            "key_events": _str_list(payload.get("key_events")),
            "core_facts": _str_list(payload.get("core_facts")),
            **catalog,
            "participant_refs": collect_participant_refs(entries),
            "source_turn_count": count_logical_turns(entries),
            "source_entry_count": len(entries),
            "memory_metadata": metadata,
            "semantic_tags": _merge_unique(metadata.get("entity_anchors"), metadata.get("topic_terms")),
            # An episode is the normal model-readable compression of a prompt-
            # visible conversation segment.  Source metadata/annotation status
            # enriches it but never controls whether the episode remains
            # visible or may enter the 10 -> oldest 5 semantic roll-up.
            "retrieval_visibility": "default",
            "semanticize": True,
            "compaction_schema_version": self.config.compaction_schema_version,
        }
        return SummaryRecordInput(
            summary_id=summary_id,
            summary_profile=f"{self.config.summary_profile}:episode",
            source_ids=source_ids,
            record=record,
        )

    def _operation_summary_input(
        self,
        *,
        namespace: Namespace,
        entries: list[TimelineEntry],
    ) -> SummaryRecordInput:
        source_ids = tuple(entry.source_id for entry in entries)
        summary_id = self._stable_summary_id(namespace, source_ids, profile="operation")
        timestamps = [int(entry.timestamp) for entry in entries]
        facts: list[str] = []
        retained_anchors: list[dict[str, Any]] = []
        retained_bytes = 0
        for entry in entries:
            tool_name = str(entry.trace_metadata.get("tool_name") or entry.kind)
            status = str(entry.trace_metadata.get("status") or "")
            fact = f"{tool_name}" + (f" status={status}" if status else "")
            if entry.correlation_id:
                fact += f" correlation={entry.correlation_id}"
            retained = self._operation_retention_record(entry)
            if retained is not None:
                encoded = canonical_json_bytes(retained)
                if retained_bytes + len(encoded) <= _MAX_OPERATION_RETENTION_TOTAL_BYTES:
                    retained_anchors.append(retained)
                    retained_bytes += len(encoded)
                    fact += f" retention_anchor={canonical_json_bytes(retained['anchor']).decode('utf-8')}"
                else:
                    omitted = {
                        "source_id": entry.source_id,
                        "kind": entry.kind,
                        "correlation_id": entry.correlation_id,
                        "anchor": {
                            "status": "omitted_total_budget",
                            "sha256": stable_projection_hash(retained["anchor"]),
                            "bytes": len(canonical_json_bytes(retained["anchor"])),
                        },
                    }
                    retained_anchors.append(omitted)
                    fact += f" retention_anchor={canonical_json_bytes(omitted['anchor']).decode('utf-8')}"
            facts.append(fact)
        summary_trace: dict[str, Any] = {"source_kinds": sorted({entry.kind for entry in entries})}
        if retained_anchors:
            summary_trace["retention_anchors"] = retained_anchors
        record = {
            "summary_id": summary_id,
            "kind": "memory.operation_digest",
            "timestamp": max(timestamps),
            "period_start_ts": min(timestamps),
            "period_end_ts": max(timestamps),
            "date_label": timestamp_to_date_label(max(timestamps), self.timezone),
            "time_of_day": infer_time_of_day(max(timestamps), self.timezone),
            "importance": 0.2,
            "diary_summary": "；".join(facts),
            "core_facts": facts,
            "participant_refs": collect_participant_refs(entries),
            "source_turn_count": count_logical_turns(entries),
            "source_entry_count": len(entries),
            "memory_metadata": {},
            "trace_metadata": summary_trace,
            "retrieval_visibility": "explicit",
            "semanticize": False,
            "compaction_schema_version": self.config.compaction_schema_version,
        }
        return SummaryRecordInput(
            summary_id=summary_id,
            summary_profile=f"{self.config.summary_profile}:operation",
            source_ids=source_ids,
            record=record,
        )

    @staticmethod
    def _operation_retention_record(entry: TimelineEntry) -> dict[str, Any] | None:
        raw = entry.trace_metadata.get(OPERATION_RETENTION_ANCHOR_KEY)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            anchor: dict[str, Any] = {"status": "invalid", "reason": "anchor_must_be_object"}
            sanitization_status = "skipped_unsafe"
        else:
            safe_payload, projection_status = sanitize_projection_payload({"role": "user", "content": raw})
            clean = safe_payload.get("content")
            anchor = dict(clean) if isinstance(clean, dict) else {"value": clean}
            sanitization_status = str(
                entry.trace_metadata.get(OPERATION_RETENTION_ANCHOR_STATUS_KEY) or projection_status.value
            )
            encoded = canonical_json_bytes(anchor)
            if len(encoded) > MAX_OPERATION_RETENTION_ANCHOR_BYTES:
                anchor = {
                    "status": "omitted_entry_budget",
                    "sha256": stable_projection_hash(anchor),
                    "bytes": len(encoded),
                }
        record = {
            "source_id": entry.source_id,
            "kind": entry.kind,
            "correlation_id": entry.correlation_id,
            "anchor": anchor,
        }
        if sanitization_status != "complete":
            record["sanitization_status"] = sanitization_status
        return record

    def _stable_summary_id(
        self,
        namespace: Namespace,
        source_ids: tuple[str, ...],
        *,
        profile: str,
    ) -> str:
        return stable_projection_hash(
            {
                "namespace": [
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                ],
                "source_ids": list(source_ids),
                "compaction_schema_version": self.config.compaction_schema_version,
                "summary_profile": f"{self.config.summary_profile}:{profile}",
            }
        )

    # --- 阶段摘要 → 长期语义记忆(含强化合并) ---

    def _semanticize_episodic(self, namespace: Namespace, result: dict[str, Any]) -> None:
        if self._stop_if_shutdown(result):
            return
        cfg = self.config
        eps = self.store.get_uncompacted_episodic_summaries(namespace=namespace)
        if len(eps) < cfg.episodic_compact_trigger_count:
            return
        batch = eps[: cfg.episodic_compact_batch_size]
        generation, _ = self.store.get_conversation_generations(namespace=namespace)
        call = self._call_json(
            TaskType.SEMANTIC,
            *build_semantic_prompts(
                source_text=self._render_episodes(batch),
                overrides=self.overrides,
                enable_flavor=self.config.enable_flavor,
            ),
            fallback={
                "semantic_summary": "",
                "importance": 0.4,
                "stable_facts": [],
                "memory_title": "",
                "catalog_hint": "",
                "topic_headings": [],
            },
        )
        if self._stop_if_shutdown(result):
            return
        if not call.ok or not _has_semantic_content(call.data):
            result["semantic_retry_pending"] += 1
            return
        payload = call.data
        start_ts = min(int(s.get("period_start_ts") or s.get("timestamp") or 0) for s in batch)
        end_ts = max(int(s.get("period_end_ts") or s.get("timestamp") or 0) for s in batch)
        semantic_metadata = coerce_memory_metadata(
            payload.get("memory_metadata"), enable_flavor=cfg.enable_flavor
        ).to_dict()
        source_metadata = [
            summary.get("memory_metadata") for summary in batch if isinstance(summary.get("memory_metadata"), dict)
        ]
        for key in ("memory_facets", "about_roles", "entity_anchors", "topic_terms", "mood_tags"):
            if not semantic_metadata.get(key):
                semantic_metadata[key] = _merge_unique(*(item.get(key) for item in source_metadata))
        raw_model_metadata = payload.get("memory_metadata")
        if (
            not isinstance(raw_model_metadata, dict)
            or raw_model_metadata.get("retrieval_priority") not in RETRIEVAL_PRIORITIES
        ):
            semantic_metadata["retrieval_priority"] = _highest_priority(
                *(item.get("retrieval_priority") for item in source_metadata)
            )
        incoming = {
            "timestamp": end_ts,
            "period_start_ts": start_ts,
            "period_end_ts": end_ts,
            "date_label": timestamp_to_date_label(end_ts, self.timezone),
            "time_of_day": infer_time_of_day(end_ts, self.timezone),
            "importance": _clamp01(payload.get("importance")),
            "semantic_summary": str(payload.get("semantic_summary") or ""),
            "stable_facts": _str_list(payload.get("stable_facts")),
            "recurring_topics": _str_list(payload.get("recurring_topics")),
            "important_people": _str_list(payload.get("important_people")),
            "open_loops": _str_list(payload.get("open_loops")),
            **normalize_catalog_fields(payload),
            "memory_metadata": semantic_metadata,
            "source_summary_ids": [str(s["summary_id"]) for s in batch],
        }
        incoming["semantic_tags"] = _merge_unique(
            incoming["memory_metadata"].get("entity_anchors"),
            incoming["memory_metadata"].get("topic_terms"),
        )

        target = self._find_reinforcement_target(namespace, incoming)
        if target is not None:
            record = self._merge_reinforcement(target, incoming)
            if self._stop_if_shutdown(result):
                return
            reinforcement_target_id = str(target["semantic_id"])
            reinforcement_target_row_version = int(target.get("row_version") or 0)
        else:
            record = {
                "semantic_id": self._stable_semantic_id(
                    namespace,
                    tuple(str(summary["summary_id"]) for summary in batch),
                ),
                "reinforcement_count": 1,
                "last_reinforced_ts": end_ts,
                **incoming,
            }
            reinforcement_target_id = ""
            reinforcement_target_row_version = 0

        snapshot = SemanticSnapshot(
            namespace_key=(
                namespace.tenant_id or "",
                namespace.user_id,
                namespace.domain_id or "",
                namespace.conversation_id or "",
            ),
            compaction_generation=generation,
            summary_ids=tuple(str(summary["summary_id"]) for summary in batch),
            summary_row_versions=tuple(
                (str(summary["summary_id"]), int(summary.get("row_version") or 0)) for summary in batch
            ),
            reinforcement_target_id=reinforcement_target_id,
            reinforcement_target_row_version=reinforcement_target_row_version,
        )
        if self._stop_if_shutdown(result):
            return
        committed = self.store.commit_semantic_batch(
            namespace=namespace,
            commit=SemanticCommitInput(snapshot=snapshot, semantic_record=record),
        )
        if not committed.committed or committed.semantic_record is None:
            result["status"] = committed.status
            result["reason"] = committed.reason
            return
        saved = committed.semantic_record
        result["compaction_generation"] = committed.compaction_generation
        if target is not None:
            result["reinforced"] += 1
        else:
            result["semantic_created"] += 1
        self._record_index_result(
            result,
            self._index(build_semantic_entry(saved), saved["semantic_id"]),
        )

    def _stable_semantic_id(self, namespace: Namespace, summary_ids: tuple[str, ...]) -> str:
        return stable_projection_hash(
            {
                "namespace": [
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                ],
                "summary_ids": list(summary_ids),
                "semantic_schema_version": self.config.compaction_schema_version,
                "summary_profile": f"{self.config.summary_profile}:semantic",
            }
        )

    def _find_reinforcement_target(self, namespace: Namespace, incoming: dict[str, Any]) -> dict[str, Any] | None:
        cfg = self.config
        candidates = self.store.get_recent_semantic_summaries(
            namespace=namespace, limit=cfg.semantic_reinforcement_lookback
        )
        best, best_score = None, 0
        for cand in candidates:
            score = _overlap_score(cand, incoming)
            if score >= cfg.semantic_reinforcement_min_overlap and score > best_score:
                best, best_score = cand, score
        return best

    def _merge_reinforcement(self, target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        existing_count = max(1, int(target.get("reinforcement_count") or 1))
        blended = (float(target.get("importance") or 0.5) * existing_count + float(incoming["importance"])) / (
            existing_count + 1
        )
        fallback = {
            "semantic_summary": target.get("semantic_summary") or incoming["semantic_summary"],
            "importance": max(float(target.get("importance") or 0.0), float(incoming["importance"]), blended),
            "stable_facts": _merge_unique(target.get("stable_facts"), incoming["stable_facts"]),
            "recurring_topics": _merge_unique(target.get("recurring_topics"), incoming["recurring_topics"]),
            "important_people": _merge_unique(target.get("important_people"), incoming["important_people"]),
            "open_loops": _merge_unique(target.get("open_loops"), incoming["open_loops"]),
            "memory_title": target.get("memory_title") or incoming.get("memory_title") or "",
            "catalog_hint": target.get("catalog_hint") or incoming.get("catalog_hint") or "",
            "topic_headings": _merge_unique(target.get("topic_headings"), incoming.get("topic_headings")),
        }
        call = self._call_json(
            TaskType.REINFORCEMENT,
            *build_reinforcement_prompts(
                existing_text=_render_semantic_text(target, tz=self.timezone),
                incoming_text=_render_semantic_text(incoming, tz=self.timezone),
                overrides=self.overrides,
                enable_flavor=self.config.enable_flavor,
            ),
            fallback=fallback,
        )
        payload = call.data
        catalog = normalize_catalog_fields(payload)
        if not catalog["memory_title"]:
            catalog["memory_title"] = str(fallback["memory_title"] or "")
        if not catalog["catalog_hint"]:
            catalog["catalog_hint"] = str(fallback["catalog_hint"] or "")
        if not catalog["topic_headings"]:
            catalog["topic_headings"] = list(fallback["topic_headings"])
        catalog["catalog_schema_version"] = (
            CATALOG_SCHEMA_VERSION if catalog["memory_title"] and catalog["catalog_hint"] else 0
        )
        return {
            "semantic_id": target["semantic_id"],  # 同 id 覆盖 = 强化
            "timestamp": incoming["timestamp"],
            "period_start_ts": min(
                int(target.get("period_start_ts") or 0) or incoming["period_start_ts"], incoming["period_start_ts"]
            ),
            "period_end_ts": max(int(target.get("period_end_ts") or 0), incoming["period_end_ts"]),
            "date_label": incoming["date_label"],
            "time_of_day": incoming["time_of_day"],
            "importance": _clamp01(payload.get("importance", fallback["importance"])),
            "semantic_summary": str(payload.get("semantic_summary") or fallback["semantic_summary"]),
            "stable_facts": _str_list(payload.get("stable_facts")) or fallback["stable_facts"],
            "recurring_topics": _str_list(payload.get("recurring_topics")) or fallback["recurring_topics"],
            "important_people": _str_list(payload.get("important_people")) or fallback["important_people"],
            "open_loops": _str_list(payload.get("open_loops")) or fallback["open_loops"],
            **catalog,
            "memory_metadata": _merge_memory_metadata(
                target.get("memory_metadata"),
                incoming["memory_metadata"],
                enable_flavor=self.config.enable_flavor,
            ),
            "semantic_tags": _merge_unique(target.get("semantic_tags"), incoming.get("semantic_tags")),
            "source_summary_ids": _merge_unique(target.get("source_summary_ids"), incoming["source_summary_ids"]),
            "reinforcement_count": int(target.get("reinforcement_count") or 1) + 1,
            "last_reinforced_ts": incoming["timestamp"],
        }

    # --- 工具 ---

    def _call_json(
        self, task_type: TaskType, system_prompt: str, user_prompt: str, *, fallback: dict
    ) -> JsonCallResult:
        res = self.llm.call(
            LLMRequest(
                task_type=task_type,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=ResponseFormat.JSON,
                timeout_s=self.config.llm_timeout_s,
                max_retries=self.config.llm_max_retries,
                fallback=fallback,
            )
        )
        if res.ok and isinstance(res.data, dict):
            return JsonCallResult(ok=True, data=res.data)
        return JsonCallResult(ok=False, data=fallback)

    def _index(self, entry: dict[str, Any], source_id: str) -> bool:
        # outbox:库已 pending,upsert 成功才置 indexed;失败保持 pending,留给 reindex_pending 自愈。
        try:
            self.index.upsert([entry])
            self.store.set_index_state(
                source_id,
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )
        except Exception:
            return False
        return True

    @staticmethod
    def _record_index_result(result: dict[str, Any], indexed: bool) -> None:
        if not indexed:
            result["index_status"] = "pending"
        elif result.get("index_status") != "pending":
            result["index_status"] = "indexed"

    def _render_reference_summaries(self, namespace: Namespace) -> str:
        """取本会话最近的既有阶段摘要作参考,帮新摘要与旧摘要保持一致、避免冲突。"""
        existing = self.store.get_visible_episodic_summaries(
            namespace=namespace, limit=self.config.episodic_visible_max
        )
        lines = []
        for s in existing:
            diary = normalize_text(s.get("diary_summary"))
            facts = "; ".join(str(f) for f in (s.get("core_facts") or []))
            label = str(s.get("date_label") or "")
            lines.append(f"- [{label}] {diary}" + (f"(事实:{facts})" if facts else ""))
        return "\n".join(lines)

    def _render_transcript(self, batch: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for record in batch:
            entry = TimelineEntry.from_record(record)
            chat_text = self.projection_adapter.render_chat_entry(entry, include_weekday=True)
            if chat_text is not None:
                rendered.append(chat_text)
                continue
            if entry.kind.startswith("message.") or entry.turn_role is TurnRole.FINAL:
                rendered.append(render_raw_snippet([record], tz=self.timezone))
                continue
            # Typed event/skill/intermediate entries may keep their structured
            # facts in payload.  Summarize the same deterministic renderer view
            # used by provider projection instead of silently dropping payload.
            rendered.append(self.projection_adapter.renderer_registry.render(entry, timezone=self.timezone).text)
        return "\n".join(item for item in rendered if item)

    def _render_episodes(self, batch: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for s in batch:
            time_label = self._summary_time_label(s)
            period = TIME_PERIOD_LABELS.get(str(s.get("time_of_day") or ""), "")
            period_label = normalize_text(s.get("period_label"))
            event_type = normalize_text(s.get("event_type"))
            head = " | ".join(
                part
                for part in (
                    time_label,
                    period,
                    f"阶段:{period_label}" if period_label else "",
                    f"类型:{event_type}" if event_type else "",
                )
                if part
            )
            lines.append(f"- [{head}]" if head else "- 阶段摘要")
            memory_title = normalize_text(s.get("memory_title"))
            catalog_hint = normalize_text(s.get("catalog_hint"))
            topic_headings = _str_list(s.get("topic_headings"))
            if memory_title:
                lines.append(f"  记忆标题: {memory_title}")
            if catalog_hint:
                lines.append(f"  目录提示: {catalog_hint}")
            if topic_headings:
                lines.append("  主题: " + "; ".join(topic_headings))
            for key, label in (
                ("diary_summary", "阶段回忆"),
                ("key_events", "关键事件"),
                ("core_facts", "核心事实"),
            ):
                values = (
                    [normalize_text(s.get(key))]
                    if key == "diary_summary"
                    else [normalize_text(item) for item in (s.get(key) or [])]
                )
                rendered = "; ".join(item for item in values if item)
                if rendered:
                    lines.append(f"  {label}: {rendered}")
            metadata = _source_memory_metadata(s)
            if metadata:
                lines.append("  memory_metadata: " + canonical_json_bytes(metadata).decode("utf-8"))
        return "\n".join(lines)

    def _summary_time_label(self, summary: dict[str, Any]) -> str:
        start = _positive_ts(summary.get("period_start_ts"))
        end = _positive_ts(summary.get("period_end_ts")) or _positive_ts(summary.get("timestamp"))
        if start is None and end is None:
            ts = _positive_ts(summary.get("timestamp"))
            start, end = ts, ts
        return format_time_range_label(start_ts=start, end_ts=end, tz=self.timezone)


def _clamp01(value: Any) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, n))


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out, seen = [], set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _positive_ts(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _has_summary_content(payload: dict[str, Any]) -> bool:
    return bool(
        normalize_text(payload.get("diary_summary"))
        or _str_list(payload.get("key_events"))
        or _str_list(payload.get("core_facts"))
    )


def _has_semantic_content(payload: dict[str, Any]) -> bool:
    return bool(
        normalize_text(payload.get("semantic_summary"))
        or _str_list(payload.get("stable_facts"))
        or _str_list(payload.get("recurring_topics"))
        or _str_list(payload.get("important_people"))
        or _str_list(payload.get("open_loops"))
    )


def _merge_unique(*lists: Any, limit: int | None = None) -> list[str]:
    out, seen = [], set()
    for lst in lists:
        for item in lst or []:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def _highest_priority(*values: Any) -> str:
    ranking = {value: index for index, value in enumerate(RETRIEVAL_PRIORITIES)}
    normalized = [str(value or "").strip().lower() for value in values]
    valid = [value for value in normalized if value in ranking]
    return max(valid, key=ranking.__getitem__) if valid else "normal"


def _merge_memory_metadata(*values: Any, enable_flavor: bool) -> dict[str, Any]:
    metadata_values = [value for value in values if isinstance(value, dict)]
    merged = {
        "turn_intent": next(
            (str(value.get("turn_intent") or "") for value in reversed(metadata_values) if value.get("turn_intent")),
            "",
        ),
        "memory_facets": _merge_unique(*(value.get("memory_facets") for value in metadata_values)),
        "about_roles": _merge_unique(*(value.get("about_roles") for value in metadata_values)),
        "entity_anchors": _merge_unique(*(value.get("entity_anchors") for value in metadata_values)),
        "topic_terms": _merge_unique(*(value.get("topic_terms") for value in metadata_values)),
        "retrieval_priority": _highest_priority(*(value.get("retrieval_priority") for value in metadata_values)),
        "mood_tags": _merge_unique(*(value.get("mood_tags") for value in metadata_values)),
    }
    return coerce_memory_metadata(merged, enable_flavor=enable_flavor).to_dict()


def _overlap_score(candidate: dict[str, Any], incoming: dict[str, Any]) -> int:
    def norm_set(record: dict[str, Any], key: str) -> set[str]:
        return {normalize_text(x) for x in (record.get(key) or []) if normalize_text(x)}

    candidate_metadata = candidate.get("memory_metadata") if isinstance(candidate.get("memory_metadata"), dict) else {}
    incoming_metadata = incoming.get("memory_metadata") if isinstance(incoming.get("memory_metadata"), dict) else {}
    candidate_entities = {
        normalize_text(item).casefold()
        for item in candidate_metadata.get("entity_anchors") or []
        if normalize_text(item)
    }
    incoming_entities = {
        normalize_text(item).casefold()
        for item in incoming_metadata.get("entity_anchors") or []
        if normalize_text(item)
    }
    shared_entities = candidate_entities & incoming_entities
    if not shared_entities:
        return 0
    candidate_facets = set(candidate_metadata.get("memory_facets") or [])
    incoming_facets = set(incoming_metadata.get("memory_facets") or [])
    if candidate_facets and incoming_facets and not (candidate_facets & incoming_facets):
        return 0
    candidate_topics = {
        normalize_text(item).casefold() for item in candidate_metadata.get("topic_terms") or [] if normalize_text(item)
    }
    incoming_topics = {
        normalize_text(item).casefold() for item in incoming_metadata.get("topic_terms") or [] if normalize_text(item)
    }
    proposition_overlap = len(candidate_topics & incoming_topics)
    proposition_overlap += len(norm_set(candidate, "stable_facts") & norm_set(incoming, "stable_facts"))
    proposition_overlap += len(norm_set(candidate, "recurring_topics") & norm_set(incoming, "recurring_topics"))
    if proposition_overlap <= 0:
        return 0
    return len(shared_entities) + proposition_overlap


def _source_memory_metadata(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("memory_metadata")
    if not isinstance(raw, dict):
        return {}
    metadata: dict[str, Any] = {}
    turn_intent = normalize_text(raw.get("turn_intent"))
    if turn_intent:
        metadata["turn_intent"] = turn_intent
    for key in (
        "memory_facets",
        "about_roles",
        "entity_anchors",
        "topic_terms",
        "mood_tags",
    ):
        values = _str_list(raw.get(key))
        if values:
            metadata[key] = values
    priority = normalize_text(raw.get("retrieval_priority"))
    if priority in RETRIEVAL_PRIORITIES:
        metadata["retrieval_priority"] = priority
    return metadata


def _render_semantic_text(record: dict[str, Any], *, tz: str) -> str:
    start = _positive_ts(record.get("period_start_ts"))
    end = _positive_ts(record.get("period_end_ts")) or _positive_ts(record.get("timestamp"))
    time_label = format_time_range_label(start_ts=start, end_ts=end, tz=tz)
    lines = [f"时间范围: {time_label}" if time_label else "时间范围: 未提供"]
    for key, label in (
        ("semantic_summary", "长期印象"),
        ("stable_facts", "稳定事实"),
        ("recurring_topics", "反复话题"),
        ("important_people", "重要人物"),
        ("open_loops", "待续线索"),
    ):
        values = (
            [normalize_text(record.get(key))]
            if key == "semantic_summary"
            else [normalize_text(item) for item in (record.get(key) or [])]
        )
        rendered = "; ".join(item for item in values if item)
        if rendered:
            lines.append(f"{label}: {rendered}")
    metadata = _source_memory_metadata(record)
    if metadata:
        lines.append("memory_metadata: " + canonical_json_bytes(metadata).decode("utf-8"))
    return "\n".join(lines)
