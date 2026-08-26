"""MemorySystem —— 统一时间线、投影、检索和压缩的对外门面。

新宿主的写侧生命周期是 ``begin_turn → append_* → complete_turn``。少量
``record_*`` 便捷方法只负责把独立记录投影为 typed standalone entry，不再直写
旧 flat-message 路径。
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from concurrent.futures import Future
from dataclasses import replace
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from .compaction import Compaction
from .config import MemoryConfig, OperationProjectionPolicy
from .context_contract import CONTEXT_SURFACE_VERSION, ContextDiagnostic, ContextSurface
from .embedding.base import EmbeddingProvider
from .errors import NamespaceError, SchemaError
from .index.base import VectorIndex
from .index.entry_builder import build_raw_entry, build_semantic_entry, build_summary_entry
from .index.memory_index import InMemoryVectorIndex
from .index.metadata_filters import INDEX_SCHEMA_KEY, INDEX_SCHEMA_VERSION
from .llm.base import LLMClient
from .namespace import Actor, Namespace
from .prompts import PromptOverrides
from .projection import (
    ContextProjection,
    ProjectionAdapter,
    ProjectionLedger,
    ProjectionMessage,
    ProjectionMessageInput,
    ProjectionStatus,
    RendererRegistry,
    RequestProjectionResult,
    build_entry_projection_hashes,
    build_projection_audit_input,
    canonical_json_bytes,
    default_renderer_registry,
    merge_projection_status,
    normalize_provider_profile,
    provider_safe_projection_messages,
    sanitize_projection_payload,
    stable_projection_hash,
)
from .retrieval import ReadPipeline, RetrievalRequest, RetrievalResult
from .runtime import MemCoreRuntime
from .schema import coerce_memory_metadata
from .store.base import MemoryStore
from .store.sqlite_store import SQLiteMemoryStore
from .time_anchor import infer_time_of_day, timestamp_to_date_label
from .timeline import (
    AnnotationStatus,
    CompletionCommitResult,
    EntryOrigin,
    EntryTrust,
    MemoryAnnotation,
    RetrievalPolicy,
    RetrievalVisibility,
    TimelineEntry,
    TimelineEntryInput,
    TurnAbortResult,
    TurnCompletion,
    TurnHandle,
    TurnProjectionSettlement,
    TurnStatus,
    build_action_entry,
    build_observation_entry,
)
from .token_counter import TokenCounter

_TOOL_EVENT_PART = re.compile(r"[^A-Za-z0-9_.-]+")


class MemorySystem:
    def __init__(
        self,
        *,
        llm: LLMClient,
        namespace: Namespace,
        timezone: str,
        storage_dir: str | None = None,
        config: MemoryConfig | None = None,
        store: MemoryStore | None = None,
        index: VectorIndex | None = None,
        embedding: EmbeddingProvider | str | None = None,
        token_counter: TokenCounter | None = None,
        enable_flavor: bool | None = None,
        persona_text: str = "",
        prompt_overrides: PromptOverrides | None = None,
        renderer_registry: RendererRegistry | None = None,
        runtime: MemCoreRuntime | None = None,
    ) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError("llm must be an LLMClient instance (inject your model adapter)")
        if not isinstance(namespace, Namespace):
            raise TypeError("namespace must be a Namespace instance")
        if not str(timezone or "").strip():
            # 时间锚点是命根,时区缺失会让相对时间换算全错(见 §5)。
            raise ValueError("timezone is required (e.g. 'Asia/Shanghai'); time anchoring depends on it")
        tz_name = str(timezone).strip()
        try:  # 构造时就验合法 IANA 时区,别拖到渲染才炸
            ZoneInfo(tz_name)
        except Exception as exc:
            raise ValueError(f"invalid timezone {tz_name!r}; must be a valid IANA name like 'Asia/Shanghai'") from exc

        self.llm = llm
        self.namespace = namespace
        self.timezone = tz_name
        self.storage_dir = storage_dir
        self.config = config or MemoryConfig()
        if enable_flavor is not None:  # 显式传入覆盖 config 默认
            self.config.enable_flavor = bool(enable_flavor)
        if token_counter is not None and not isinstance(token_counter, TokenCounter):
            raise TypeError("token_counter must be a TokenCounter instance or None")
        self.persona_text = str(persona_text or "")
        # 提示词治理:persona_text 便捷参数填进 overrides 的对应插槽(显式 overrides 优先)。
        self.prompt_overrides = self._resolve_overrides(prompt_overrides, self.persona_text)
        if renderer_registry is not None and not isinstance(renderer_registry, RendererRegistry):
            raise TypeError("renderer_registry must be a RendererRegistry or None")
        self.renderer_registry = renderer_registry or default_renderer_registry()
        self._projection = ProjectionAdapter(
            renderer_registry=self.renderer_registry,
            timezone=self.timezone,
        )
        if runtime is not None and not isinstance(runtime, MemCoreRuntime):
            raise TypeError("runtime must be a MemCoreRuntime or None")
        self.runtime = runtime or MemCoreRuntime()
        self._owns_runtime = runtime is None

        # 依赖装配:缺省自带 SQLite + 内存索引;embedding 必须显式(生产不静默退 hashed,见 §13.1)。
        self.embedding = self._resolve_embedding(embedding)
        self.token_counter = token_counter
        self.store: MemoryStore = store or SQLiteMemoryStore(storage_dir or ":memory:")
        self.index: VectorIndex = index or InMemoryVectorIndex(embedding=self.embedding)
        self._projection_ledger = ProjectionLedger(
            store=self.store,
            adapter=self._projection,
            enable_flavor=self.config.enable_flavor,
        )
        self._compaction = Compaction(
            store=self.store,
            index=self.index,
            llm=self.llm,
            config=self.config,
            timezone=self.timezone,
            overrides=self.prompt_overrides,
            token_counter=self.token_counter,
            projection_adapter=self._projection,
            runtime=self.runtime,
        )
        self._read = ReadPipeline(
            store=self.store,
            index=self.index,
            config=self.config,
            timezone=self.timezone,
            token_counter=self.token_counter,
        )

    @staticmethod
    def _resolve_overrides(prompt_overrides: PromptOverrides | None, persona_text: str) -> PromptOverrides:
        import dataclasses

        if prompt_overrides is not None and not isinstance(prompt_overrides, PromptOverrides):
            # 坏配置在构造时就拒绝,别拖到压缩时 AttributeError 或被空 dict 静默忽略。
            raise TypeError(
                f"prompt_overrides must be a PromptOverrides or None, got {type(prompt_overrides).__name__}"
            )
        ov = prompt_overrides or PromptOverrides()
        if persona_text and not ov.persona_text:  # 便捷参数只在 overrides 没填 persona 时生效
            ov = dataclasses.replace(ov, persona_text=persona_text)
        return ov

    @staticmethod
    def _resolve_embedding(embedding: EmbeddingProvider | str | None) -> EmbeddingProvider:
        if isinstance(embedding, EmbeddingProvider):
            return embedding
        if isinstance(embedding, str) and embedding.strip():
            from .embedding.huggingface import HuggingFaceEmbeddingProvider  # 延迟导入重依赖

            return HuggingFaceEmbeddingProvider(model_name=embedding.strip())
        raise ValueError(
            "embedding is required: pass an EmbeddingProvider or a model name string. "
            "Do not rely on a silent hashed fallback in production (see design §13.1)."
        )

    # --- 写侧生命周期 ---

    def begin_turn(
        self,
        *,
        stimuli: list[TimelineEntryInput],
        annotation_target_ids: list[str] | None = None,
        turn_id: str = "",
        opened_at: int | None = None,
    ) -> TurnHandle:
        """Atomically open a V2 turn and persist all stimulus entries."""

        if not stimuli:
            raise SchemaError("turn_stimulus_required")
        prepared: list[TimelineEntryInput] = []
        now = int(opened_at or time.time())
        for entry in stimuli:
            if not isinstance(entry, TimelineEntryInput):
                raise TypeError("stimuli must contain TimelineEntryInput values")
            entry = self.renderer_registry.bind_input(entry)
            timestamp = int(entry.timestamp or now)
            prepared.append(
                replace(
                    entry,
                    source_id=entry.source_id or uuid.uuid4().hex,
                    timestamp=timestamp,
                    date_label=entry.date_label or timestamp_to_date_label(timestamp, self.timezone),
                    time_of_day=entry.time_of_day or infer_time_of_day(timestamp, self.timezone),
                )
            )
        if annotation_target_ids is None:
            if len(prepared) != 1:
                raise SchemaError("multiple_stimuli_require_explicit_annotation_targets")
            targets = [prepared[0].source_id]
        else:
            targets = [str(item or "").strip() for item in annotation_target_ids]
        try:
            handle = self.store.begin_turn(
                namespace=self.namespace,
                stimulus_entries=prepared,
                annotation_target_ids=targets,
                turn_id=turn_id,
                opened_at=now,
                operation_projection_policy=self.config.operation_projection_policy,
                operation_settlement_min_utf8_bytes=self.config.operation_settlement_min_utf8_bytes,
                operation_settlement_min_saved_ratio=self.config.operation_settlement_min_saved_ratio,
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return self._refresh_turn_handle_after_index(handle)

    def begin_turn_from_existing_sources(
        self,
        *,
        stimulus_source_ids: list[str],
        annotation_target_ids: list[str] | None = None,
        turn_id: str = "",
        opened_at: int | None = None,
    ) -> TurnHandle:
        """Open a model-response turn around existing standalone facts/events.

        This preserves the original sequence, timestamp, actor, content and
        source IDs.  It is intended for delayed/ambient response decisions
        where the real stimulus was already appended to the timeline and must
        not be copied into a synthetic second message.
        """

        source_ids = [str(item or "").strip() for item in stimulus_source_ids]
        if not source_ids or any(not item for item in source_ids):
            raise SchemaError("turn_stimulus_source_id_required")
        if len(set(source_ids)) != len(source_ids):
            raise SchemaError("turn_duplicate_stimulus_source_id")
        targets = (
            source_ids
            if annotation_target_ids is None
            else [str(item or "").strip() for item in annotation_target_ids]
        )
        now = int(opened_at or time.time())
        try:
            handle = self.store.begin_turn_from_existing_sources(
                namespace=self.namespace,
                stimulus_source_ids=source_ids,
                annotation_target_ids=targets,
                turn_id=turn_id,
                opened_at=now,
                operation_projection_policy=self.config.operation_projection_policy,
                operation_settlement_min_utf8_bytes=self.config.operation_settlement_min_utf8_bytes,
                operation_settlement_min_saved_ratio=self.config.operation_settlement_min_saved_ratio,
            )
        except NotImplementedError as exc:
            raise SchemaError("store_existing_stimulus_turn_unsupported") from exc
        return self._refresh_turn_handle_after_index(handle)

    def append_entry(self, entry: TimelineEntryInput, *, turn_id: str = "") -> TimelineEntry:
        """Append an intermediate/action/observation to an open V2 turn."""

        if not isinstance(entry, TimelineEntryInput):
            raise TypeError("entry must be a TimelineEntryInput")
        entry = self.renderer_registry.bind_input(entry)
        resolved_turn_id = str(turn_id or entry.turn_id or "").strip()
        if not resolved_turn_id:
            raise SchemaError("timeline_entry_turn_id_required")
        timestamp = int(entry.timestamp or time.time())
        prepared = replace(
            entry,
            turn_id=resolved_turn_id,
            timestamp=timestamp,
            date_label=entry.date_label or timestamp_to_date_label(timestamp, self.timezone),
            time_of_day=entry.time_of_day or infer_time_of_day(timestamp, self.timezone),
        )
        try:
            stored = self.store.append_entry(namespace=self.namespace, entry=prepared)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return self._index_timeline_entry(stored)

    def append_action(
        self,
        *,
        turn_id: str,
        kind: str,
        correlation_id: str,
        semantic_text: str = "",
        payload: dict[str, Any] | None = None,
        source_id: str = "",
        timestamp: int = 0,
        trace_metadata: dict[str, Any] | None = None,
        retention_anchor: dict[str, Any] | None = None,
        prompt_visible: bool = True,
        trust: EntryTrust | str = EntryTrust.UNTRUSTED_DATA,
    ) -> TimelineEntry:
        """Append a host-neutral model action without prescribing its wire protocol."""

        return self.append_entry(
            build_action_entry(
                kind=kind,
                correlation_id=correlation_id,
                semantic_text=semantic_text,
                payload=payload,
                source_id=source_id,
                timestamp=timestamp,
                trace_metadata=trace_metadata,
                retention_anchor=retention_anchor,
                prompt_visible=prompt_visible,
                trust=trust,
            ),
            turn_id=turn_id,
        )

    def append_observation(
        self,
        *,
        turn_id: str,
        kind: str,
        correlation_id: str,
        semantic_text: str = "",
        payload: dict[str, Any] | None = None,
        source_id: str = "",
        timestamp: int = 0,
        status: str = "",
        trace_metadata: dict[str, Any] | None = None,
        retention_anchor: dict[str, Any] | None = None,
        prompt_visible: bool = True,
        trust: EntryTrust | str = EntryTrust.UNTRUSTED_DATA,
    ) -> TimelineEntry:
        """Append a host/environment result linked to a prior action."""

        return self.append_entry(
            build_observation_entry(
                kind=kind,
                correlation_id=correlation_id,
                semantic_text=semantic_text,
                payload=payload,
                source_id=source_id,
                timestamp=timestamp,
                status=status,
                trace_metadata=trace_metadata,
                retention_anchor=retention_anchor,
                prompt_visible=prompt_visible,
                trust=trust,
            ),
            turn_id=turn_id,
        )

    def append_standalone_entry(self, entry: TimelineEntryInput) -> TimelineEntry:
        """Persist a typed entry that does not open or complete a model-response turn."""

        if not isinstance(entry, TimelineEntryInput):
            raise TypeError("entry must be a TimelineEntryInput")
        entry = self.renderer_registry.bind_input(entry)
        if entry.turn_id or entry.turn_role is not None:
            raise SchemaError("standalone_entry_must_not_have_turn")
        timestamp = int(entry.timestamp or time.time())
        prepared = replace(
            entry,
            source_id=entry.source_id or uuid.uuid4().hex,
            timestamp=timestamp,
            date_label=entry.date_label or timestamp_to_date_label(timestamp, self.timezone),
            time_of_day=entry.time_of_day or infer_time_of_day(timestamp, self.timezone),
        )
        try:
            stored = self.store.append_standalone_entry(namespace=self.namespace, entry=prepared)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return self._index_timeline_entry(stored)

    def complete_turn(
        self,
        *,
        turn_id: str,
        semantic_text: str,
        provider_output_raw: str,
        memory_annotation: dict[str, Any] | None = None,
        annotation_status: AnnotationStatus | str = AnnotationStatus.MISSING,
        annotations: list[MemoryAnnotation] | None = None,
        timestamp: int | None = None,
        source_id: str = "",
        close_reason: str = "completed",
        kind: str = "message.assistant",
        payload: dict[str, Any] | None = None,
        trace_metadata: dict[str, Any] | None = None,
        provider_profile: str = "",
        provider_projection: ProjectionMessageInput | dict[str, Any] | None = None,
    ) -> CompletionCommitResult:
        """Atomically commit final speech, target annotations, visibility, and turn close."""

        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            raise SchemaError("turn_completion_invalid_turn_id")
        try:
            handle = self.store.get_turn(namespace=self.namespace, turn_id=normalized_turn_id)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        if handle is None:
            return CompletionCommitResult(status="not_found", turn_id=normalized_turn_id, reason="turn_not_found")

        if handle.status is not TurnStatus.OPEN:
            resolved_annotations = ()
        elif annotations is None:
            if len(handle.annotation_target_ids) > 1:
                raise SchemaError("multiple_annotation_targets_require_explicit_annotations")
            resolved_annotations: tuple[MemoryAnnotation, ...]
            if handle.annotation_target_ids:
                status = self._coerce_completion_annotation_status(annotation_status)
                metadata = coerce_memory_metadata(
                    memory_annotation,
                    enable_flavor=self.config.enable_flavor,
                ).to_dict()
                resolved_annotations = (
                    MemoryAnnotation(
                        target_source_id=handle.annotation_target_ids[0],
                        status=status,
                        memory_metadata=metadata,
                        source="host" if status is AnnotationStatus.ACCEPTED_HOST else "model",
                    ),
                )
            else:
                resolved_annotations = ()
        else:
            resolved_annotations = tuple(self._coerce_memory_annotation(annotation) for annotation in annotations)

        completed_at = int(timestamp or time.time())
        resolved_source_id = str(source_id or "").strip()
        resolved_final_projection: ProjectionMessageInput | None = None
        if handle.status is TurnStatus.OPEN:
            resolved_source_id = resolved_source_id or uuid.uuid4().hex
            if isinstance(provider_projection, ProjectionMessageInput):
                if (
                    provider_profile
                    and normalize_provider_profile(provider_profile) != provider_projection.provider_profile
                ):
                    raise SchemaError("turn_completion_projection_profile_mismatch")
                resolved_final_projection = provider_projection
            elif isinstance(provider_projection, dict):
                if not provider_profile:
                    raise SchemaError("turn_completion_projection_profile_required")
                resolved_final_projection = ProjectionMessageInput(
                    provider_profile=provider_profile,
                    payload=provider_projection,
                    source_ids=(resolved_source_id,),
                )
            elif provider_projection is not None:
                raise TypeError("provider_projection must be a ProjectionMessageInput, dict, or None")
            elif provider_profile:
                raise SchemaError("turn_completion_projection_payload_required")
            if resolved_final_projection is not None and not resolved_final_projection.source_ids:
                resolved_final_projection = replace(
                    resolved_final_projection,
                    source_ids=(resolved_source_id,),
                )
        completion = TurnCompletion(
            turn_id=normalized_turn_id,
            semantic_text=semantic_text,
            provider_output_raw=provider_output_raw,
            annotations=resolved_annotations,
            timestamp=completed_at,
            source_id=resolved_source_id,
            close_reason=close_reason,
            kind=kind,
            payload=dict(payload or {}),
            trace_metadata=dict(trace_metadata or {}),
            final_projection=resolved_final_projection,
            date_label=timestamp_to_date_label(completed_at, self.timezone),
            time_of_day=infer_time_of_day(completed_at, self.timezone),
        )
        try:
            result = self.store.commit_turn_completion(namespace=self.namespace, completion=completion)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        if not result.completed:
            return result
        refreshed = self._refresh_completion_after_index(result)
        self._settle_turn_after_completion(
            turn_id=normalized_turn_id,
            provider_profile=provider_profile,
            terminal_source_id=str(getattr(refreshed.final_entry, "source_id", "") or resolved_source_id),
            settled_at=completed_at,
        )
        return refreshed

    def abort_turn(self, turn_id: str, *, reason: str, closed_at: int | None = None) -> TurnAbortResult:
        try:
            return self.store.abort_turn(
                namespace=self.namespace,
                turn_id=turn_id,
                reason=reason,
                closed_at=int(closed_at or time.time()),
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

    def _settle_turn_after_completion(
        self,
        *,
        turn_id: str,
        provider_profile: str,
        terminal_source_id: str,
        settled_at: int,
    ) -> "TurnProjectionSettlement | None":
        """complete_turn 成功后按冻结策略建立终局紧凑投影(文档 §5/§12)。

        - full_until_raw_compaction: 不建立任何 settlement(默认零行为差异);
        - compact_after_terminal: 确定性生成 settled/settled_noop, 失败时结构化降级
          full_fallback。settlement 失败绝不影响已经提交的 final 回复。
        幂等: upsert first-write-wins, 重试 complete_turn 不会生成不同 hash。
        """
        try:
            frozen_turn = self.store.get_turn(namespace=self.namespace, turn_id=turn_id)
        except NotImplementedError:
            return None
        if frozen_turn is None:
            return None
        frozen_policy = str(frozen_turn.operation_projection_policy or "full_until_raw_compaction")
        if frozen_policy != OperationProjectionPolicy.COMPACT_AFTER_TERMINAL.value:
            return None
        from functools import partial

        from .settlement import (
            SETTLED_PROJECTION_SCHEMA_VERSION,
            build_settlement_plan,
            classify_observation,
            full_fallback_plan,
            settlement_config_hash,
        )

        profile = normalize_provider_profile(provider_profile or self.config.projection_profile)
        frozen_min = int(frozen_turn.operation_settlement_min_utf8_bytes or 256)
        frozen_ratio = float(frozen_turn.operation_settlement_min_saved_ratio or 0.5)
        config_hash = settlement_config_hash(
            policy=frozen_policy,
            min_utf8_bytes=frozen_min,
            saved_ratio=frozen_ratio,
        )
        try:
            entries = self.store.get_turn_entries(namespace=self.namespace, turn_id=turn_id)
            authoritative = self._projection_ledger.freeze_turn_entries(
                namespace=self.namespace,
                turn_id=turn_id,
                entries=entries,
                provider_profile=profile,
            )
            decider = partial(
                classify_observation,
                min_inline_bytes=frozen_min,
                required_savings_ratio=frozen_ratio,
                timezone=self.timezone,
            )
            plan = build_settlement_plan(
                self._projection,
                entries,
                provider_profile=profile,
                authoritative_messages=authoritative,
                count_text=self.token_counter.count_text if self.token_counter is not None else None,
                token_count_quality=self.token_counter.quality if self.token_counter is not None else "estimated",
                observation_decider=decider,
                settlement_min_utf8_bytes=frozen_min,
                settlement_min_saved_ratio=frozen_ratio,
                settlement_config_hash=config_hash,
            )
        except Exception as exc:  # noqa: BLE001
            detail = str(exc) if isinstance(exc, SchemaError) else ""
            reason = f"{type(exc).__name__}:{detail}" if detail else type(exc).__name__
            plan = full_fallback_plan(turn_id=turn_id, provider_profile=profile, reason=reason)

        settlement = TurnProjectionSettlement(
            turn_id=turn_id,
            policy=frozen_policy,
            settlement_status=plan.settlement_status,
            settlement_schema_version=SETTLED_PROJECTION_SCHEMA_VERSION,
            provider_profile=profile,
            terminal_source_id=terminal_source_id,
            full_projection_hash=plan.full_projection_hash,
            settled_projection_hash=plan.settled_projection_hash,
            first_changed_projection_index=plan.first_changed_projection_index,
            full_projected_tokens=plan.full_projected_tokens,
            settled_projected_tokens=plan.settled_projected_tokens,
            token_count_quality=plan.token_count_quality,
            reason=plan.reason,
            settled_at=int(settled_at),
            settlement_min_utf8_bytes=frozen_min,
            settlement_min_saved_ratio=frozen_ratio,
            settlement_config_hash=config_hash,
        )
        try:
            self.store.commit_turn_projection_settlement(
                namespace=self.namespace,
                settlement=settlement,
                projections=list(plan.messages) if plan.settlement_status == "settled" else [],
            )
        except Exception as exc:  # noqa: BLE001
            fallback = TurnProjectionSettlement(
                turn_id=turn_id,
                policy=frozen_policy,
                settlement_status="full_fallback",
                settlement_schema_version=SETTLED_PROJECTION_SCHEMA_VERSION,
                provider_profile=profile,
                terminal_source_id=terminal_source_id,
                reason=f"settlement_persist_failed:{type(exc).__name__}",
                settled_at=int(settled_at),
                settlement_min_utf8_bytes=frozen_min,
                settlement_min_saved_ratio=frozen_ratio,
                settlement_config_hash=config_hash,
            )
            try:
                self.store.commit_turn_projection_settlement(
                    namespace=self.namespace,
                    settlement=fallback,
                    projections=[],
                )
            except Exception:  # noqa: BLE001
                return None
            return fallback
        return settlement

    def recover_stale_open_turns(
        self,
        *,
        max_age_seconds: int,
        now: int | None = None,
        reason: str = "stale_open_turn_recovered",
    ) -> tuple[TurnAbortResult, ...]:
        """Abort abandoned turns older than a host-selected safety window.

        Hosts should still abort a turn immediately when their model or
        delivery path fails. This recovery boundary handles process crashes
        and forced shutdowns that prevent that normal ``finally`` path from
        running. Recovery is conversation-scoped and never crosses the
        current namespace.
        """

        max_age = int(max_age_seconds or 0)
        if max_age <= 0:
            raise ValueError("max_age_seconds must be positive")
        recovered_at = int(now or time.time())
        try:
            return self.store.abort_stale_open_turns(
                namespace=self.namespace,
                opened_before=max(0, recovered_at - max_age),
                reason=str(reason or "stale_open_turn_recovered"),
                closed_at=recovered_at,
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

    def build_context_projection(self, *, provider_profile: str) -> ContextProjection:
        """Build and freeze this conversation's provider-visible append-only history."""

        projection, _entries = self._build_context_projection_with_entries(provider_profile=provider_profile)
        return projection

    def _build_context_projection_with_entries(
        self,
        *,
        provider_profile: str,
    ) -> tuple[ContextProjection, list[TimelineEntry]]:
        """Build one projection and return the exact entries used by it.

        ``ContextSurface`` needs both products.  Returning the owned snapshot
        avoids re-reading and re-decoding every visible timeline row after the
        projection has already been built.
        """

        profile = normalize_provider_profile(provider_profile)
        try:
            episodic = self.store.get_visible_episodic_summaries(
                namespace=self.namespace,
                limit=self.config.episodic_visible_max,
                cross_conversation=False,
            )
            semantic = self.store.get_recent_semantic_summaries(
                namespace=self.namespace,
                limit=self.config.semantic_visible_limit,
                cross_conversation=False,
            )
            entries = self.store.list_prompt_visible_entries(namespace=self.namespace)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

        grouped: dict[str, list[TimelineEntry]] = {}
        for entry in entries:
            group_id = entry.turn_id or f"legacy.{stable_projection_hash({'source_id': entry.source_id})}"
            grouped.setdefault(group_id, []).append(entry)

        memory_records: list[tuple[dict[str, Any], str, str, str]] = []
        for record, id_key, prefix in (
            *((item, "semantic_id", "semantic") for item in reversed(semantic)),
            *(
                (item, "summary_id", "summary")
                for item in reversed(episodic)
                if str(item.get("retrieval_visibility") or "default") == "default"
            ),
        ):
            source_id = str(record.get(id_key) or "").strip()
            turn_id = self._projection_ledger.memory_turn_id(
                source_id=source_id,
                id_key=id_key,
                turn_prefix=prefix,
            )
            memory_records.append((record, id_key, prefix, turn_id))

        projection_rows = None
        batch_reader = getattr(self.store, "get_context_projection_rows", None)
        if callable(batch_reader):
            try:
                projection_rows = batch_reader(
                    namespace=self.namespace,
                    turn_ids=tuple((*[item[3] for item in memory_records], *grouped.keys())),
                    provider_profile=profile,
                )
            except NotImplementedError:
                projection_rows = None

        messages: list[ProjectionMessage] = []
        for record, id_key, prefix, turn_id in memory_records:
            messages.extend(
                self._freeze_memory_record_projection(
                    record=record,
                    id_key=id_key,
                    turn_prefix=prefix,
                    provider_profile=profile,
                    saved_projections=(
                        projection_rows.projections_by_turn.get(turn_id, ())
                        if projection_rows is not None
                        else None
                    ),
                )
            )
        for turn_id, turn_entries in grouped.items():
            messages.extend(
                self._freeze_turn_projection(
                    turn_id=turn_id,
                    entries=turn_entries,
                    provider_profile=profile,
                    saved_projections=(
                        projection_rows.projections_by_turn.get(turn_id, ())
                        if projection_rows is not None
                        else None
                    ),
                    settlement=(
                        projection_rows.settlements_by_turn.get(turn_id)
                        if projection_rows is not None
                        else None
                    ),
                    settled_rows=(
                        projection_rows.settled_rows_by_turn.get(turn_id, ())
                        if projection_rows is not None
                        else None
                    ),
                    rows_preloaded=projection_rows is not None,
                )
            )

        try:
            compaction_generation, projection_generation = self.store.get_conversation_generations(
                namespace=self.namespace
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        messages = list(
            provider_safe_projection_messages(
                messages,
                provider_profile=profile,
            )
        )
        projection = ContextProjection(
            provider_profile=profile,
            messages=tuple(messages),
            projection_version=max((message.projection_version for message in messages), default=1),
            stable_prefix_hash=stable_projection_hash([message.payload for message in messages]),
            entry_projection_hashes=build_entry_projection_hashes(messages),
            compaction_generation=compaction_generation,
            projection_generation=projection_generation,
            has_compact_history=any(message.projection_status is ProjectionStatus.SETTLED for message in messages),
        )
        return projection, entries

    def build_open_turn_projection(
        self,
        *,
        turn_id: str,
        provider_profile: str,
    ) -> ContextProjection:
        """Build the provider projection for one owned open turn only.

        This focused facade is for hosts that need to refresh an active native
        tool loop after appending a batch. It does not include compact memory
        records or any other turn, and it never replaces
        :meth:`build_context_projection` as the authoritative full-history
        request builder.
        """

        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            raise SchemaError("open_turn_projection_turn_id_required")
        profile = normalize_provider_profile(provider_profile)
        try:
            handle = self.store.get_turn(
                namespace=self.namespace,
                turn_id=normalized_turn_id,
            )
            if handle is None:
                raise SchemaError("open_turn_projection_turn_not_found")
            if handle.status != TurnStatus.OPEN:
                raise SchemaError("open_turn_projection_turn_not_open")
            entries = [
                entry
                for entry in self.store.get_turn_entries(
                    namespace=self.namespace,
                    turn_id=normalized_turn_id,
                )
                if entry.prompt_visible
            ]
            compaction_generation, projection_generation = self.store.get_conversation_generations(
                namespace=self.namespace
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

        messages = list(
            provider_safe_projection_messages(
                self._freeze_turn_projection(
                    turn_id=normalized_turn_id,
                    entries=entries,
                    provider_profile=profile,
                ),
                provider_profile=profile,
            )
        )
        return ContextProjection(
            provider_profile=profile,
            messages=tuple(messages),
            projection_version=max((message.projection_version for message in messages), default=1),
            stable_prefix_hash=stable_projection_hash([message.payload for message in messages]),
            entry_projection_hashes=build_entry_projection_hashes(messages),
            compaction_generation=compaction_generation,
            projection_generation=projection_generation,
            has_compact_history=False,
        )

    def build_context_surface(
        self,
        *,
        session_id: str | None = None,
        provider_profile: str,
        current_source_id: str | None = None,
        active_turn_messages: Sequence[Mapping[str, Any]] = (),
    ) -> ContextSurface:
        """Build the public provider-ready surface for one host request.

        ``session_id`` is accepted as a stable integration label for wrappers;
        namespace/conversation isolation remains owned by this MemorySystem.
        It is intentionally not included in model-visible messages.
        """

        del session_id
        projection, visible_entries = self._build_context_projection_with_entries(provider_profile=provider_profile)
        current_id = str(current_source_id or "").strip()
        entries_by_source_id = {entry.source_id: entry for entry in visible_entries}

        def _surface_payload(message: ProjectionMessage) -> dict[str, Any]:
            if message.projection_status in {ProjectionStatus.REQUEST_FROZEN, ProjectionStatus.SETTLED}:
                return dict(message.payload)
            if len(message.source_ids) != 1:
                return dict(message.payload)
            entry = entries_by_source_id.get(message.source_ids[0])
            if entry is None:
                return dict(message.payload)
            rendered = self._projection.context_surface_payload(entry, provider_profile=projection.provider_profile)
            return rendered if rendered is not None else dict(message.payload)

        history: list[Mapping[str, Any]] = []
        history_source_ids: list[tuple[str, ...]] = []
        history_projection_metadata: list[Mapping[str, Any]] = []
        current: Mapping[str, Any] | None = None
        current_source_ids: tuple[str, ...] = ()
        current_projection_metadata: Mapping[str, Any] | None = None
        current_turn_id = ""
        active: list[Mapping[str, Any]] = []
        active_source_ids: list[tuple[str, ...]] = []
        active_projection_metadata: list[Mapping[str, Any]] = []
        diagnostics: list[ContextDiagnostic] = []
        current_index: int | None = None
        if current_id:
            for index, message in enumerate(projection.messages):
                if current_id in message.source_ids:
                    if current_index is None:
                        current_index = index
                    else:
                        diagnostics.append(ContextDiagnostic("degraded", "current_message_multiple_projection_records"))

        def _projection_metadata(message: ProjectionMessage) -> dict[str, Any]:
            return {
                "turn_id": str(message.turn_id or ""),
                "source_ids": list(message.source_ids),
                "projection_index": int(message.projection_index),
                "projection_status": message.projection_status.value,
                "projection_version": int(message.projection_version),
            }

        for index, message in enumerate(projection.messages):
            payload = _surface_payload(message)
            if current_index is None:
                history.append(payload)
                history_source_ids.append(tuple(message.source_ids))
                history_projection_metadata.append(_projection_metadata(message))
            elif index < current_index:
                history.append(payload)
                history_source_ids.append(tuple(message.source_ids))
                history_projection_metadata.append(_projection_metadata(message))
            elif index == current_index:
                current = payload
                current_source_ids = tuple(message.source_ids)
                current_projection_metadata = _projection_metadata(message)
                current_turn_id = str(message.turn_id or "")
            else:
                active.append(payload)
                active_source_ids.append(tuple(message.source_ids))
                active_projection_metadata.append(_projection_metadata(message))
        if current_id and current is None:
            diagnostics.append(ContextDiagnostic("degraded", "current_message_source_not_found"))

        for item in active_turn_messages:
            if not isinstance(item, Mapping):
                diagnostics.append(ContextDiagnostic("rejected", "active_turn_message_not_object"))
                continue
            active.append(dict(item))
            active_source_ids.append(())
            active_projection_metadata.append(
                {
                    "turn_id": current_turn_id,
                    "source_ids": [],
                    "projection_index": -1,
                    "projection_status": "complete",
                    "projection_version": int(projection.projection_version),
                }
            )

        surface_payload = {
            "version": CONTEXT_SURFACE_VERSION,
            "provider_profile": projection.provider_profile,
            "history_messages": history,
            "current_message": current,
            "active_turn_messages": active,
        }
        return ContextSurface(
            version=CONTEXT_SURFACE_VERSION,
            provider_profile=projection.provider_profile,
            history_messages=tuple(history),
            current_message=current,
            active_turn_messages=tuple(active),
            projection_hash=stable_projection_hash(surface_payload),
            projection_generation=projection.projection_generation,
            message_source_ids=tuple(
                (
                    *history_source_ids,
                    *((current_source_ids,) if current is not None else ()),
                    *active_source_ids,
                )
            ),
            message_projection_metadata=tuple(
                (
                    *history_projection_metadata,
                    *((current_projection_metadata,) if current_projection_metadata is not None else ()),
                    *active_projection_metadata,
                )
            ),
            has_compact_history=projection.has_compact_history,
            current_turn_id=current_turn_id,
            projection_version=projection.projection_version,
            compaction_generation=projection.compaction_generation,
            diagnostics=tuple(diagnostics),
        )

    def settlement_metrics(self) -> list[dict[str, Any]]:
        """安全审计指标(文档 §13): 只含指标, 不含任何 prompt 正文/payload。

        每个 closed turn 一行 settlement 指标; saved/ratio 由账本字段确定性计算。
        turn_id 以 sha256 前 16 位 hash 暴露, 不泄露原始 ID 语义。
        """
        settlements = self.store.list_turn_projection_settlements(namespace=self.namespace)
        metrics: list[dict[str, Any]] = []
        for settlement in settlements:
            full_tokens = int(settlement.get("full_projected_tokens") or 0)
            settled_tokens = int(settlement.get("settled_projected_tokens") or 0)
            saved = max(0, full_tokens - settled_tokens)
            metrics.append(
                {
                    "operation_projection_policy": str(settlement.get("policy") or ""),
                    "settlement_status": str(settlement.get("settlement_status") or ""),
                    "turn_id_hash": hashlib.sha256(
                        str(settlement.get("turn_id") or "").encode("utf-8", errors="ignore")
                    ).hexdigest()[:16],
                    "provider_profile": str(settlement.get("provider_profile") or ""),
                    "full_projected_tokens": full_tokens,
                    "settled_projected_tokens": settled_tokens,
                    "saved_projected_tokens": saved,
                    "saved_ratio": round(saved / full_tokens, 4) if full_tokens else 0.0,
                    "first_changed_projection_index": int(
                        -1
                        if settlement.get("first_changed_projection_index") is None
                        else settlement.get("first_changed_projection_index")
                    ),
                    "full_projection_hash": str(settlement.get("full_projection_hash") or ""),
                    "settled_projection_hash": str(settlement.get("settled_projection_hash") or ""),
                    "token_count_quality": str(settlement.get("token_count_quality") or ""),
                    "fallback_reason": str(settlement.get("reason") or ""),
                    "settlement_min_utf8_bytes": int(settlement.get("settlement_min_utf8_bytes") or 256),
                    "settlement_min_saved_ratio": float(settlement.get("settlement_min_saved_ratio") or 0.5),
                    "settlement_config_hash": str(settlement.get("settlement_config_hash") or ""),
                }
            )
        return metrics

    def migrate_legacy_path_projections(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Explicitly re-project legacy path-damage rows for this namespace.

        Runs at a controlled pre-traffic maintenance point (idempotent).
        Returns a structured report; payload text is never included.
        Unaffected frozen projections keep their exact bytes.  Stale
        settlements are rebuilt through the regular settlement planner.
        """

        from .projection_migration import migrate_legacy_path_projections

        count_text = getattr(self.token_counter, "count_text", None) if self.token_counter is not None else None
        return migrate_legacy_path_projections(
            store=self.store,
            adapter=self._projection_ledger.adapter,
            namespace=self.namespace,
            dry_run=bool(dry_run),
            count_text=count_text if callable(count_text) else None,
        )

    def _freeze_turn_projection(
        self,
        *,
        turn_id: str,
        entries: list[TimelineEntry],
        provider_profile: str,
        saved_projections: Sequence[ProjectionMessage] | None = None,
        settlement: Mapping[str, Any] | None = None,
        settled_rows: Sequence[Mapping[str, Any]] | None = None,
        rows_preloaded: bool = False,
    ) -> list[ProjectionMessage]:
        from .settlement import load_settled_projection, settled_rows_to_messages

        if rows_preloaded:
            settled = None
            if settlement and str(settlement.get("settlement_status") or "") == "settled":
                if not settled_rows:
                    raise SchemaError("settled_projection_rows_missing")
                settled = settled_rows_to_messages(settlement, settled_rows)
        else:
            settled = load_settled_projection(
                self.store,
                namespace=self.namespace,
                turn_id=turn_id,
                provider_profile=provider_profile,
            )
        if settled is not None:
            return settled
        try:
            return self._projection_ledger.freeze_turn_entries(
                namespace=self.namespace,
                turn_id=turn_id,
                entries=entries,
                provider_profile=provider_profile,
                saved_projections=saved_projections,
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

    def _freeze_memory_record_projection(
        self,
        *,
        record: dict[str, Any],
        id_key: str,
        turn_prefix: str,
        provider_profile: str,
        saved_projections: Sequence[ProjectionMessage] | None = None,
    ) -> list[ProjectionMessage]:
        try:
            return self._projection_ledger.freeze_memory_record(
                namespace=self.namespace,
                record=record,
                id_key=id_key,
                turn_prefix=turn_prefix,
                provider_profile=provider_profile,
                saved_projections=saved_projections,
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

    def record_request_projection(
        self,
        *,
        turn_id: str,
        provider_profile: str,
        turn_messages: list[ProjectionMessageInput],
        history_messages: list[dict[str, Any]],
        history_message_indexes: Sequence[int] | None = None,
        audit_history_messages: list[dict[str, Any]] | None = None,
        attempt: int = 0,
        model_route: Any = "",
        system_prefix: Any = "",
        tool_schema: Any = (),
        created_at: int | None = None,
    ) -> RequestProjectionResult:
        """Atomically freeze current-turn messages and hash the actual request prefix."""

        profile = normalize_provider_profile(provider_profile)
        resolved_attempt = int(attempt or 0)
        if resolved_attempt < 1:
            try:
                existing_audits = self.store.list_projection_audits(
                    namespace=self.namespace,
                    turn_id=turn_id,
                    provider_profile=profile,
                )
            except NotImplementedError as exc:
                raise SchemaError("store_timeline_v2_unsupported") from exc
            resolved_attempt = max((audit.attempt for audit in existing_audits), default=0) + 1
        if not turn_messages:
            raise SchemaError("projection_turn_messages_required")
        prepared: list[ProjectionMessageInput] = []
        for index, message in enumerate(turn_messages):
            if not isinstance(message, ProjectionMessageInput):
                raise TypeError("turn_messages must contain ProjectionMessageInput values")
            if message.provider_profile != profile:
                raise SchemaError("projection_audit_profile_mismatch")
            if not message.source_ids:
                raise SchemaError("projection_source_ids_required")
            prepared.append(
                replace(
                    message,
                    projection_index=index if message.projection_index < 0 else message.projection_index,
                    projection_status=merge_projection_status(
                        ProjectionStatus.REQUEST_FROZEN,
                        message.projection_status,
                    ),
                )
            )

        if history_message_indexes is None:
            if len(history_messages) < len(prepared):
                raise SchemaError("projection_actual_history_missing_turn_suffix")
            actual_messages = history_messages[-len(prepared) :] if prepared else []
        else:
            indexes = tuple(int(index) for index in history_message_indexes)
            if len(indexes) != len(prepared) or len(set(indexes)) != len(indexes):
                raise SchemaError("projection_actual_history_indexes_invalid")
            if any(index < 0 or index >= len(history_messages) for index in indexes):
                raise SchemaError("projection_actual_history_indexes_invalid")
            if tuple(sorted(indexes)) != indexes:
                raise SchemaError("projection_actual_history_indexes_invalid")
            actual_messages = [history_messages[index] for index in indexes]
        for actual, declared in zip(actual_messages, prepared):
            safe_actual, _ = sanitize_projection_payload(actual, provider_profile=profile)
            if canonical_json_bytes(safe_actual) != canonical_json_bytes(declared.payload):
                raise SchemaError("projection_actual_history_mismatch")
        media_omitted = any(
            message.projection_status is ProjectionStatus.MEDIA_OMITTED
            or b"media omitted from persistent history" in canonical_json_bytes(message.payload)
            for message in prepared
        )
        audit = build_projection_audit_input(
            turn_id=turn_id,
            attempt=resolved_attempt,
            provider_profile=profile,
            model_route=model_route,
            system_prefix=system_prefix,
            tool_schema=tool_schema,
            history_messages=(audit_history_messages if audit_history_messages is not None else history_messages),
            media_omitted=media_omitted,
            created_at=int(created_at or time.time()),
        )
        try:
            return self.store.commit_request_projection(
                namespace=self.namespace,
                turn_id=turn_id,
                projections=prepared,
                audit=audit,
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc

    def _refresh_turn_handle_after_index(self, handle: TurnHandle) -> TurnHandle:
        for entry in handle.stimuli:
            self._index_timeline_entry(entry)
        try:
            refreshed = self.store.get_turn(namespace=self.namespace, turn_id=handle.turn_id)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return refreshed or handle

    def _refresh_completion_after_index(self, result: CompletionCommitResult) -> CompletionCommitResult:
        for entry in (*result.updated_targets, *((result.final_entry,) if result.final_entry else ())):
            self._index_timeline_entry(entry)
        try:
            final_entry = (
                self.store.get_entry(namespace=self.namespace, source_id=result.final_entry.source_id)
                if result.final_entry
                else None
            )
            targets = tuple(
                entry
                for entry in (
                    self.store.get_entry(namespace=self.namespace, source_id=item.source_id)
                    for item in result.updated_targets
                )
                if entry is not None
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return replace(result, final_entry=final_entry, updated_targets=targets)

    def _index_timeline_entry(self, entry: TimelineEntry) -> TimelineEntry:
        try:
            self._reindex_record(entry.to_record())
        except Exception:
            self.store.set_index_status(entry.source_id, "pending")
        try:
            refreshed = self.store.get_entry(namespace=self.namespace, source_id=entry.source_id)
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return refreshed or entry

    def _coerce_memory_annotation(self, annotation: MemoryAnnotation) -> MemoryAnnotation:
        if not isinstance(annotation, MemoryAnnotation):
            raise TypeError("annotations must contain MemoryAnnotation values")
        metadata = coerce_memory_metadata(
            annotation.memory_metadata,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        return replace(annotation, memory_metadata=metadata)

    @staticmethod
    def _coerce_completion_annotation_status(value: AnnotationStatus | str) -> AnnotationStatus:
        if isinstance(value, AnnotationStatus):
            return value
        normalized = str(value or "").strip().lower()
        if normalized == "accepted":
            return AnnotationStatus.ACCEPTED_MODEL
        try:
            return AnnotationStatus(normalized)
        except ValueError as exc:
            raise SchemaError("memory_annotation_invalid_status") from exc

    def record_user_turn(self, content: str, *, actor: Actor | None = None, **fields: Any) -> dict[str, Any]:
        """Append one independent user message through the V2 typed writer.

        A request that expects a model response should use :meth:`begin_turn` so
        the stimulus, intermediate operations and final response share lineage.
        """

        return self._append_standalone_message(
            kind="message.user",
            origin=EntryOrigin.USER,
            compatibility_role="user",
            content=content,
            actor=actor,
            **fields,
        )

    def record_assistant_turn(
        self, reply: str, *, in_reply_to: dict[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Append one independent assistant message through the V2 typed writer."""

        reply_to_source_id = str((in_reply_to or {}).get("source_id") or "")
        return self._append_standalone_message(
            kind="message.assistant",
            origin=EntryOrigin.ASSISTANT,
            compatibility_role="assistant",
            content=reply,
            actor=None,
            reply_to_source_id=reply_to_source_id,
            **fields,
        )

    def record_external_event(
        self,
        *,
        event_type: str,
        fields: dict[str, Any] | None = None,
        source: str = "",
        timestamp: int | None = None,
        source_id: str | None = None,
        entity_anchors: list[str] | None = None,
        topic_terms: list[str] | None = None,
        retrieval_priority: str = "normal",
    ) -> dict[str, Any]:
        """Append one independent typed external event.

        If the event starts a model-response turn, pass an equivalent
        ``TimelineEntryInput`` to :meth:`begin_turn` instead so its final reply
        receives explicit turn lineage and memory annotation.
        """

        from .rendering import render_external_event_text

        label = self._tool_event_part(event_type, fallback="external")
        metadata = {
            "memory_facets": ["event"],
            "about_roles": ["external"],
            "entity_anchors": list(entity_anchors or []),
            "topic_terms": [label, *(topic_terms or [])],
            "retrieval_priority": retrieval_priority,
        }
        metadata = coerce_memory_metadata(
            metadata,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        event_fields = dict(fields or {})
        semantic_text = render_external_event_text(
            event_type=label,
            fields=event_fields,
            source=source,
        )
        payload = {"source": str(source or "").strip(), **event_fields}
        entry = self.append_standalone_entry(
            TimelineEntryInput(
                source_id=str(source_id or "").strip(),
                kind=f"event.{label}",
                origin=EntryOrigin.ENVIRONMENT,
                turn_role=None,
                semantic_text=semantic_text,
                payload=payload,
                timestamp=int(timestamp or 0),
                memory_metadata=metadata,
                annotation_status=AnnotationStatus.UNANNOTATED,
                retrieval_policy=RetrievalPolicy.EXPLICIT,
                retrieval_visibility=RetrievalVisibility.EXPLICIT,
                semanticize=False,
                compatibility_role=f"event.{label}",
            )
        )
        return entry.to_record()

    def record_tool_exchange(
        self,
        *,
        turn_id: str,
        tool_name: str,
        result: Any,
        tool_input: Any = None,
        tool_call_id: str = "",
        source: str = "",
        timestamp: int | None = None,
        source_id_prefix: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Append one correlated action/observation pair to an open V2 turn.

        Operation retrieval and compaction are governed by typed roles and
        correlation lineage, not by semantic metadata categories.
        """
        from .rendering import render_tool_result_text, render_tool_use_text

        resolved_turn_id = str(turn_id or "").strip()
        if not resolved_turn_id:
            raise SchemaError("tool_exchange_turn_id_required")
        tool = str(tool_name or "").strip()
        prefix = str(source_id_prefix or "").strip()
        call_id = self._tool_event_part(tool_call_id or prefix or f"call_{uuid.uuid4().hex[:8]}", fallback="call")
        tool_label = self._tool_event_part(tool, fallback="tool")
        result_source = str(source or "").strip() or self._default_tool_source(tool)
        ts = int(timestamp or time.time())
        tool_use = self.append_action(
            turn_id=resolved_turn_id,
            kind=f"tool.{tool_label}.call",
            correlation_id=call_id,
            semantic_text=render_tool_use_text(tool_input=tool_input),
            payload={"input": tool_input if tool_input is not None else {}},
            source_id=f"{prefix}:tool_use" if prefix else "",
            timestamp=ts,
            trace_metadata={"tool_name": tool_label, "status": "running"},
        )
        tool_result = self.append_observation(
            turn_id=resolved_turn_id,
            kind=f"tool.{tool_label}.result",
            correlation_id=call_id,
            semantic_text=render_tool_result_text(result=result, source=result_source),
            payload={"source": result_source, "output": result},
            source_id=f"{prefix}:tool_result" if prefix else "",
            timestamp=ts + 1,
            status="success",
            trace_metadata={"tool_name": tool_label},
        )
        return {"tool_use": tool_use.to_record(), "tool_result": tool_result.to_record()}

    def record_material_reference(
        self,
        *,
        file_id: str,
        kind: str,
        actor: Actor | None = None,
        filename: str = "",
        mime_type: str = "",
        file_status: str = "ready",
        derived_status: str = "",
        timestamp: int | None = None,
        source_id: str | None = None,
        entity_anchors: list[str] | None = None,
        topic_terms: list[str] | None = None,
        retrieval_priority: str = "normal",
    ) -> dict[str, Any]:
        """追加附件/文件引用事件:只记录材料锚点,不把文件本体写进 memory。"""
        from .rendering import render_material_reference_text

        file_key = self._tool_event_part(file_id, fallback="file")
        kind_label = self._tool_event_part(kind, fallback="material")
        metadata = {
            "memory_facets": ["event", "state"],
            "about_roles": ["external"],
            "entity_anchors": [file_id, filename, *(entity_anchors or [])],
            "topic_terms": [kind, file_status, *(topic_terms or [])],
            "retrieval_priority": retrieval_priority,
        }
        fields: dict[str, Any] = {
            "annotation_status": "unannotated",
            "memory_metadata": metadata,
            "retrieval_policy": "explicit",
            "retrieval_visibility": "explicit",
        }
        if timestamp is not None:
            fields["timestamp"] = timestamp
        if source_id:
            fields["source_id"] = source_id
        return self._append_standalone_message(
            kind=f"material.{kind_label}.reference",
            origin=EntryOrigin.USER,
            compatibility_role=f"user.attachment {kind_label} {file_key}",
            content=render_material_reference_text(
                file_id=file_id,
                kind=kind,
                filename=filename,
                mime_type=mime_type,
                file_status=file_status,
                derived_status=derived_status,
            ),
            actor=actor,
            **fields,
        )

    def record_material_cleanup(
        self,
        *,
        file_id: str,
        kind: str = "",
        filename: str = "",
        file_status: str = "deleted",
        derived_status: str = "",
        reason: str = "",
        timestamp: int | None = None,
        source_id: str | None = None,
        entity_anchors: list[str] | None = None,
        topic_terms: list[str] | None = None,
        retrieval_priority: str = "normal",
    ) -> dict[str, Any]:
        """追加材料清理事件,让模型知道旧文件/解析物可能已经不可再读。"""
        from .rendering import render_material_cleanup_text

        file_key = self._tool_event_part(file_id, fallback="file")
        kind_label = self._tool_event_part(kind, fallback="material")
        metadata = {
            "memory_facets": ["event", "state"],
            "about_roles": ["external"],
            "entity_anchors": [file_id, filename, *(entity_anchors or [])],
            "topic_terms": [kind, file_status, reason, *(topic_terms or [])],
            "retrieval_priority": retrieval_priority,
        }
        fields: dict[str, Any] = {
            "annotation_status": "unannotated",
            "memory_metadata": metadata,
            "retrieval_policy": "explicit",
            "retrieval_visibility": "explicit",
        }
        if timestamp is not None:
            fields["timestamp"] = timestamp
        if source_id:
            fields["source_id"] = source_id
        return self._append_standalone_message(
            kind=f"material.{kind_label}.cleanup",
            origin=EntryOrigin.ENVIRONMENT,
            compatibility_role=f"system.material_cleanup {kind_label} {file_key}",
            content=render_material_cleanup_text(
                file_id=file_id,
                kind=kind,
                filename=filename,
                file_status=file_status,
                derived_status=derived_status,
                reason=reason,
            ),
            actor=None,
            **fields,
        )

    @staticmethod
    def _tool_event_part(value: str, *, fallback: str) -> str:
        text = _TOOL_EVENT_PART.sub("_", str(value or "").strip()).strip("._-")
        return (text[:64] or fallback).rstrip("._-") or fallback

    @staticmethod
    def _default_tool_source(tool_name: str) -> str:
        tool = str(tool_name or "").strip()
        normalized = tool.lower()
        if normalized in {"retrieve", "retrieve_for_turn"}:
            return "long_term_memory"
        if normalized == "read_timeline":
            return "timeline"
        return tool or "tool"

    def _append_standalone_message(
        self,
        *,
        kind: str,
        origin: EntryOrigin,
        compatibility_role: str,
        content: str,
        actor: Actor | None,
        reply_to_source_id: str = "",
        **fields: Any,
    ) -> dict[str, Any]:
        """Thin convenience adapter over the single typed standalone writer."""

        ts = int(fields.pop("timestamp", None) or time.time())
        index_in_vector = bool(fields.pop("index_in_vector", True))
        source_id = str(fields.pop("source_id", None) or uuid.uuid4().hex)
        metadata = coerce_memory_metadata(
            fields.pop("memory_metadata", None),
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        if compatibility_role in {"assistant", "user"}:
            fields.setdefault("annotation_status", AnnotationStatus.ACCEPTED_HOST)
            fields.setdefault("annotation_source", "host_adapter")
            fields.setdefault("retrieval_policy", RetrievalPolicy.ALWAYS)
            fields.setdefault("retrieval_visibility", RetrievalVisibility.DEFAULT)
        allowed = {
            "annotation_status",
            "annotation_source",
            "retrieval_policy",
            "retrieval_visibility",
            "semanticize",
            "prompt_visible",
            "trust",
            "trace_metadata",
            "payload",
        }
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise TypeError(f"unsupported standalone entry fields: {', '.join(unknown)}")
        entry = self.append_standalone_entry(
            TimelineEntryInput(
                source_id=source_id,
                kind=kind,
                origin=origin,
                turn_role=None,
                semantic_text=content,
                payload=fields.pop("payload", {"text": content}),
                timestamp=ts,
                actor=actor,
                reply_to_source_id=reply_to_source_id,
                trace_metadata=fields.pop("trace_metadata", {}),
                memory_metadata=metadata,
                annotation_status=fields.pop("annotation_status", AnnotationStatus.UNANNOTATED),
                annotation_source=str(fields.pop("annotation_source", "")),
                retrieval_policy=fields.pop("retrieval_policy", RetrievalPolicy.AUTO),
                retrieval_visibility=fields.pop("retrieval_visibility", RetrievalVisibility.EXPLICIT),
                semanticize=bool(fields.pop("semanticize", True)),
                prompt_visible=bool(fields.pop("prompt_visible", True)),
                trust=fields.pop("trust", EntryTrust.UNTRUSTED_DATA),
                compatibility_role=compatibility_role,
            )
        )
        rec = entry.to_record()
        if not index_in_vector:
            self.index.delete([source_id])
            self.store.set_index_status(source_id, "skipped")
            rec["index_status"] = "skipped"
        return rec

    def _with_actor(self, actor: Actor) -> Namespace:
        ns = self.namespace
        return Namespace(
            user_id=ns.user_id,
            tenant_id=ns.tenant_id,
            domain_id=ns.domain_id,
            conversation_id=ns.conversation_id,
            actor=actor,
        )

    def compact_due_sync(self, *, provider_profile: str = "") -> dict[str, Any]:
        """同步压缩(阻塞直到完成)。

        仅建议用于单测、CLI、管理脚本或进程退出前的确定性 flush。聊天请求链路应使用
        compact_due_background(),避免压缩 LLM 调用阻塞用户可见回复。显式传入
        provider_profile 时，压缩预算按该次真实 provider 投影计算；空值继续使用配置默认值。
        """
        return self._compaction.run_due(
            namespace=self.namespace,
            provider_profile=provider_profile,
        )

    def compact_due_background(self, *, provider_profile: str = "") -> Future:
        """异步压缩:提交到单 worker 后台线程,立刻返回 Future,不阻塞聊天链路。

        单 worker = 所有压缩串行,不会压垮 LLM/库;同 namespace 还有 Compaction 内部锁兜底。
        Future.result() 可取压缩统计;聊天产品里通常 fire-and-forget。provider_profile 语义与
        compact_due_sync() 相同。用完调 close() 收线程。
        """
        return self.runtime.submit_compaction(
            self._compaction.run_due,
            namespace=self.namespace,
            provider_profile=provider_profile,
        )

    def close(self, *, wait: bool = True) -> None:
        """收掉后台压缩线程。store/index 生命周期由调用方自理。"""
        if self._owns_runtime:
            self.runtime.close(wait=wait)

    def acquaintance_note(self, *, now_ts: int | None = None, cross_conversation: bool = True) -> str:
        """(opt-in,陪伴向)相处时间感:"第一次聊天是哪天、到今天认识第几天"。

        无历史返回 ""。其它领域(金融/客服)不调用即可,不强加。
        """
        first = self.store.get_first_message_timestamp(namespace=self.namespace, cross_conversation=cross_conversation)
        if not first:
            return ""
        tz = ZoneInfo(self.timezone)
        from datetime import datetime

        now = int(now_ts or time.time())
        first_date = datetime.fromtimestamp(first, tz).date()
        now_date = datetime.fromtimestamp(now, tz).date()
        days = max(1, (now_date - first_date).days + 1)
        return (
            f"按本地留下的记录,你和该用户第一次留下对话是在 {first_date.isoformat()};到今天是认识的第 {days} 天。"
            "这是一条相处时间线索,只在谈到初识/陪伴/纪念/久未联系时自然带入,不必每轮报数,"
            "也不要把『首次留下记录』夸张成你能证明的现实起点。"
        )

    def reindex_pending(self, *, limit: int = 100, batch_size: int = 64) -> dict[str, int]:
        """outbox 自愈:把 index_status=pending 的记录补做向量 upsert。可定期/启动时调用。"""
        pending = self.store.list_pending_index(limit=limit)
        repaired = failed = 0
        for batch in self._record_batches(pending, batch_size=batch_size):
            batch_repaired, batch_failed = self._reindex_batch(batch)
            repaired += batch_repaired
            failed += batch_failed
        return {"scanned": len(pending), "repaired": repaired, "failed": failed}

    def reindex_all(
        self,
        *,
        namespace: Namespace | None = None,
        limit: int | None = None,
        current_conversation_only: bool = False,
        batch_size: int = 64,
    ) -> dict[str, int]:
        """从 SQLite 真相源补建/热加载三层向量索引。

        默认按硬隔离边界(tenant/user/domain)upsert 全部会话,这样换会话后的长期记忆仍可被 retrieve 搜到。
        current_conversation_only=True 时只热当前 conversation。limit 是一次性安全上限,不是分页 cursor。
        该方法不会清空 index 里的旧条目;已有污染/陈旧向量库应由接入方先创建空 index 或使用后端管理工具清理。
        失败的记录会标回 pending,可交给 reindex_pending 重试。
        """
        target = namespace or self.namespace
        records = self.store.list_index_records(
            namespace=target,
            limit=limit,
            with_conversation=current_conversation_only,
        )
        reindexed = failed = 0
        for batch in self._record_batches(records, batch_size=batch_size):
            batch_reindexed, batch_failed = self._reindex_batch(batch)
            reindexed += batch_reindexed
            failed += batch_failed
        return {"scanned": len(records), "reindexed": reindexed, "failed": failed}

    @staticmethod
    def _record_batches(records: list[dict[str, Any]], *, batch_size: int):
        size = max(1, int(batch_size or 1))
        for start in range(0, len(records), size):
            yield records[start : start + size]

    def _reindex_batch(self, records: list[dict[str, Any]]) -> tuple[int, int]:
        indexable: list[tuple[dict[str, Any], dict[str, Any]]] = []
        completed = 0
        for record in records:
            entry = self._build_index_entry(record)
            source_id = str(entry.get("source_id") or "").strip()
            if not source_id:
                continue
            if str(record.get("retrieval_visibility") or "") == RetrievalVisibility.NEVER.value:
                self.index.delete([source_id])
                self.store.set_index_status(source_id, "skipped")
                completed += 1
                continue
            indexable.append((record, entry))
        if not indexable:
            return completed, 0
        try:
            self.index.upsert([entry for _, entry in indexable])
        except Exception:
            if len(indexable) > 1:
                midpoint = len(indexable) // 2
                left_completed, left_failed = self._reindex_batch([record for record, _ in indexable[:midpoint]])
                right_completed, right_failed = self._reindex_batch([record for record, _ in indexable[midpoint:]])
                return completed + left_completed + right_completed, left_failed + right_failed
            for record, entry in indexable:
                source_id = str(entry.get("source_id") or self._record_index_id(record)).strip()
                if source_id:
                    self.store.set_index_status(source_id, "pending")
            return completed, len(indexable)
        for _, entry in indexable:
            self.store.set_index_state(
                str(entry["source_id"]),
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )
        return completed + len(indexable), 0

    @staticmethod
    def _record_index_id(record: dict[str, Any]) -> str:
        entry_type = str(record.get("entry_type") or "")
        if entry_type == "summary":
            return str(record.get("summary_id") or "").strip()
        if entry_type == "semantic_summary":
            return str(record.get("semantic_id") or "").strip()
        return str(record.get("source_id") or "").strip()

    @staticmethod
    def _build_index_entry(record: dict[str, Any]) -> dict[str, Any]:
        builders = {
            "raw": build_raw_entry,
            "summary": build_summary_entry,
            "semantic_summary": build_semantic_entry,
        }
        return builders.get(str(record.get("entry_type")), build_raw_entry)(record)

    def _reindex_record(self, record: dict[str, Any]) -> None:
        entry = self._build_index_entry(record)
        if str(record.get("retrieval_visibility") or "") == RetrievalVisibility.NEVER.value:
            self.index.delete([entry["source_id"]])
            self.store.set_index_status(entry["source_id"], "skipped")
            return
        self.index.upsert([entry])
        self.store.set_index_state(
            entry["source_id"],
            "indexed",
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_key=INDEX_SCHEMA_KEY,
        )

    def stage_turn_metadata(
        self,
        source_id: str,
        memory_metadata: dict[str, Any] | None,
        *,
        actor: Actor | None = None,
    ) -> dict[str, Any]:
        """Stage final-model metadata on an open stimulus without admitting it to retrieval."""

        sid = str(source_id or "").strip()
        metadata = coerce_memory_metadata(
            memory_metadata,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        namespace = self.namespace if actor is None else self._with_actor(actor)
        try:
            record = self.store.stage_message_memory_metadata(
                namespace=namespace,
                source_id=sid,
                memory_metadata=metadata,
            )
        except NamespaceError as exc:
            return {
                "ok": False,
                "status": "forbidden",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "",
                "reason": str(exc) or "namespace_mismatch",
            }
        except (NotImplementedError, SchemaError) as exc:
            return {
                "ok": False,
                "status": "invalid",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "",
                "reason": str(exc) or "staged_annotation_invalid",
            }
        if record is None:
            return {
                "ok": False,
                "status": "not_found",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "",
                "reason": "source_id_not_found",
            }
        try:
            self.index.delete([sid])
        except Exception as exc:
            return {
                "ok": False,
                "status": "pending",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "pending",
                "reason": str(exc) or "stale_index_delete_failed",
            }
        return {
            "ok": True,
            "status": "staged",
            "source_id": sid,
            "memory_metadata": metadata,
            "index_status": "pending",
            "reason": "",
        }

    def update_turn_metadata(
        self,
        source_id: str,
        memory_metadata: dict[str, Any] | None,
        *,
        actor: Actor | None = None,
    ) -> dict[str, Any]:
        """回写 raw turn 的 memory_metadata,并重建 raw 向量索引。

        用于 Chat Output Adapter:先安全记录用户原文,等聊天模型 final JSON 出来后,
        再把模型本人给出的 memory_metadata 回写到该 raw message。群聊 user raw
        若使用 Actor 写入,回写时必须传入同一个稳定 Actor,避免跨发言人覆盖。
        """
        sid = str(source_id or "").strip()
        if not sid:
            return {
                "ok": False,
                "status": "not_found",
                "source_id": "",
                "memory_metadata": {},
                "index_status": "",
                "reason": "source_id_required",
            }
        metadata = coerce_memory_metadata(
            memory_metadata,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        owner_namespace = self.namespace if actor is None else self._with_actor(actor)
        previous = self.store.get_record_by_source_id(sid)
        preserve_index_opt_out = bool(previous and str(previous.get("index_status") or "") == "skipped")
        rec = self.store.update_message_memory_metadata(
            namespace=owner_namespace, source_id=sid, memory_metadata=metadata
        )
        if rec is None:
            return {
                "ok": False,
                "status": "not_found",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "",
                "reason": "source_id_not_found_or_not_raw",
            }
        if preserve_index_opt_out:
            # Metadata updates must preserve a prior retrieval opt-out.  The
            # store marks every metadata update pending while the raw index is
            # rebuilt; restore the durable opt-out without calling upsert.
            self.store.set_index_status(sid, "skipped")
            rec["index_status"] = "skipped"
            return {
                "ok": True,
                "status": "updated",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "skipped",
                "reason": "index_in_vector_disabled",
            }
        try:
            self.index.upsert([build_raw_entry(rec)])
            self.store.set_index_state(
                sid,
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )
            rec["index_status"] = "indexed"
            return {
                "ok": True,
                "status": "updated",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "indexed",
                "reason": "",
            }
        except Exception as exc:
            self.store.set_index_status(sid, "pending")
            return {
                "ok": False,
                "status": "pending",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "pending",
                "reason": str(exc) or exc.__class__.__name__,
            }

    def embedding_status(self) -> dict[str, Any]:
        return {
            "provider": self.embedding.name,
            "version": self.embedding.version,
            "dimension": self.embedding.dimension,
            "degraded": self.embedding.name == "hashed",  # hashed = 无语义,只应出现在测试/显式降级
        }

    def verify_embedding(self, **kwargs: Any) -> dict[str, Any]:
        """语义自检:确认当前 embedding 真有语义(近义词更近)。上线前调用,ok=False 说明已降级。"""
        from .embedding.verify import verify_embedding as _verify

        return _verify(self.embedding, **kwargs)

    # --- 读侧生命周期 ---

    def build_prompt_context(self, *, current: dict[str, Any]) -> dict[str, Any]:
        """拼"可见三层",供使用方拼最终聊天 prompt。

        当前轮是否需要检索由聊天模型通过 retrieve/read_timeline 工具自行决定;
        memcore 不做前置 router,避免每轮额外 LLM 判定带来的成本和误判。
        """
        now_ts = int(current.get("timestamp") or 0)
        return self._read.build_context(namespace=self.namespace, now_ts=now_ts)

    def render_prompt_context(self, context: dict[str, Any]) -> str:
        """把 build_prompt_context 的结构化三层渲染成推荐 prompt 文本。"""
        from .rendering import render_prompt_context

        return render_prompt_context(context, tz=self.timezone, enable_flavor=self.config.enable_flavor)

    def retrieve(self, query: str, **filters: Any) -> list[str]:
        """Thin text adapter over Retrieval V2 for the current model-tool migration window."""
        return self._read.retrieve(namespace=self.namespace, query=query, **filters)

    def retrieve_structured(
        self,
        query: str,
        *,
        entity_anchors: list[str] | None = None,
        topic_terms: list[str] | None = None,
        source_layers: list[str] | None = None,
        memory_facets: list[str] | None = None,
        about_roles: list[str] | None = None,
        time_hint: dict[str, Any] | None = None,
        kind_patterns: list[str] | None = None,
        include_explicit: bool = False,
        cross_conversation: bool = False,
        exclude_source_ids: list[str] | None = None,
        within_memory_id: str = "",
        max_matches: int = 0,
        result_token_budget: int = 0,
    ) -> RetrievalResult:
        """Return typed found/empty/invalid/unavailable/failed retrieval state."""
        return self._read.retrieve_result(
            namespace=self.namespace,
            request=RetrievalRequest(
                query=query,
                entity_anchors=tuple(entity_anchors or ()),
                topic_terms=tuple(topic_terms or ()),
                source_layers=tuple(source_layers or ()),
                memory_facets=tuple(memory_facets or ()),
                about_roles=tuple(about_roles or ()),
                time_hint=dict(time_hint or {}),
                kind_patterns=tuple(kind_patterns or ()),
                include_explicit=include_explicit,
                cross_conversation=cross_conversation,
                exclude_source_ids=tuple(exclude_source_ids or ()),
                within_memory_id=within_memory_id,
                max_matches=max_matches,
                result_token_budget=result_token_budget,
            ),
        )

    def retrieve_for_turn(self, *, current: dict[str, Any], query: str, **filters: Any) -> list[str]:
        """当前聊天轮的安全检索工具:默认排除本轮 prompt 已可见记忆。

        生命周期是先 record_user_turn(current) 再让聊天模型决定是否调用检索工具。此时当前消息已经入索引,
        且 build_prompt_context 会把当前未摘要 raw/可见摘要/可见语义放进 prompt。宿主工具包装层应优先调用
        这个方法,避免工具检索把已可见内容重复搜回来。
        """
        filters.setdefault("cross_conversation", True)
        result = self.retrieve_for_turn_structured(
            current=current,
            query=query,
            **filters,
        )
        return list(result.rendered_texts)

    def retrieve_for_turn_structured(
        self,
        *,
        current: dict[str, Any],
        query: str,
        exclude_source_ids: list[str] | None = None,
        **filters: Any,
    ) -> RetrievalResult:
        """Structured retrieval with every prompt-visible source excluded before scoring."""
        filters.setdefault("cross_conversation", True)
        exclude_ids = {str(item).strip() for item in (exclude_source_ids or ()) if str(item or "").strip()}
        now_ts = int((current or {}).get("timestamp") or 0)
        try:
            exclude_ids |= self._read.visible_lineage_source_ids(namespace=self.namespace, now_ts=now_ts)
        except NotImplementedError:
            return RetrievalResult(status="unavailable", reason="lineage_store_unsupported")
        except Exception:
            return RetrievalResult(status="failed", reason="lineage_resolution_failed")
        current_id = str((current or {}).get("source_id") or "").strip()
        if current_id:
            exclude_ids.add(current_id)
        return self.retrieve_structured(
            query,
            exclude_source_ids=sorted(exclude_ids),
            **filters,
        )

    def read_timeline(
        self,
        *,
        time_range: dict[str, Any] | None = None,
        date_from: str = "",
        date_to: str = "",
        time_periods: list[str] | None = None,
        anchor_source_id: str = "",
        before_turns: int = 0,
        after_turns: int = 0,
        cross_conversation: bool = False,
        projection: str = "conversation",
        page_token_budget: int = 0,
        cursor: str = "",
    ) -> dict[str, Any]:
        """时间线工具:精确读取、按目的投影，并在显式预算下按完整逻辑单元分页。

        ``time_range`` 接受 start_at/end_at ISO 或本地时间字符串。旧日期字段归一到
        同一 timestamp 查询路径。时间、anchor、cursor 模式互斥；非法参数绝不静默放宽。
        """
        from .rendering import render_timeline
        from .time_anchor import normalize_timeline_time_selector
        from .timeline_read import (
            build_timeline_read_units,
            decode_timeline_cursor,
            normalize_timeline_projection,
            paginate_timeline_units,
            timestamp_iso,
        )

        def _invalid(reason: str) -> dict[str, Any]:
            return {"status": "invalid_filter", "reason": reason, "messages": [], "message_count": 0, "text": ""}

        start = str(date_from or "").strip()
        anchor_id = str(anchor_source_id or "").strip()
        cursor_token = str(cursor or "").strip()
        has_exact_mode = time_range is not None
        has_legacy_mode = bool(start or str(date_to or "").strip() or list(time_periods or []))
        selector_count = int(bool(anchor_id)) + int(has_exact_mode) + int(has_legacy_mode) + int(bool(cursor_token))
        if selector_count > 1:
            return _invalid("timeline_modes_are_mutually_exclusive")
        if selector_count == 0:
            return _invalid("timeline_selector_required")

        after_key = None
        selector_payload: dict[str, Any]
        result_metadata: dict[str, Any]
        raw_messages: list[dict[str, Any]]
        if cursor_token:
            if (
                bool(before_turns)
                or bool(after_turns)
                or bool(cross_conversation)
                or bool(page_token_budget)
                or str(projection or "conversation").strip().lower() != "conversation"
            ):
                return _invalid("cursor_options_are_embedded")
            try:
                decoded = decode_timeline_cursor(cursor_token, namespace=self.namespace)
            except ValueError as exc:
                return _invalid(str(exc) or "invalid_cursor")
            selector_payload = dict(decoded["selector"])
            projection = str(decoded["projection"])
            page_token_budget = int(decoded["page_token_budget"])
            after_key = decoded["last_unit_key"]
            selector_mode = str(selector_payload.get("mode") or "")
            if selector_mode == "time":
                try:
                    start_ts = int(selector_payload["start_ts"])
                    end_ts = int(selector_payload["end_ts"])
                    periods = [str(item) for item in list(selector_payload.get("time_periods") or [])]
                    cross = bool(selector_payload.get("cross_conversation"))
                except (KeyError, TypeError, ValueError):
                    return _invalid("invalid_cursor")
                raw_messages = self.store.get_messages_by_time_range(
                    namespace=self.namespace,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    time_periods=periods,
                    cross_conversation=cross,
                )
                result_metadata = {
                    "selector_mode": str(selector_payload.get("selector_mode") or "time_range"),
                    "time_range": {
                        "start_at": str(selector_payload.get("start_at") or ""),
                        "end_at": str(selector_payload.get("end_at") or ""),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    },
                    "time_periods": periods,
                }
                if result_metadata["selector_mode"] == "date":
                    result_metadata["date_from"] = str(selector_payload.get("date_from") or "")
                    result_metadata["date_to"] = str(selector_payload.get("date_to") or "")
            elif selector_mode == "anchor":
                anchor_id = str(selector_payload.get("anchor_source_id") or "")
                try:
                    before = int(selector_payload.get("before_turns") or 0)
                    after = int(selector_payload.get("after_turns") or 0)
                    window = self.store.get_raw_turn_window(
                        namespace=self.namespace,
                        anchor_source_id=anchor_id,
                        before_turns=before,
                        after_turns=after,
                    )
                except (NotImplementedError, TypeError, ValueError):
                    return _invalid("invalid_cursor")
                raw_messages = [entry.to_record() for entry in window.entries]
                result_metadata = {
                    "anchor_source_id": anchor_id,
                    "before_turns": before,
                    "after_turns": after,
                }
            else:
                return _invalid("invalid_cursor")
        else:
            try:
                resolved_projection = normalize_timeline_projection(projection)
                if isinstance(page_token_budget, bool):
                    raise ValueError("page_token_budget_must_be_non_negative_integer")
                resolved_budget = int(page_token_budget or 0)
                if resolved_budget < 0:
                    raise ValueError("page_token_budget_must_be_non_negative_integer")
            except (TypeError, ValueError) as exc:
                return _invalid(str(exc) or "invalid_timeline_options")
            projection = resolved_projection
            page_token_budget = resolved_budget
            if anchor_id:
                if cross_conversation:
                    return _invalid("raw_anchor_requires_current_conversation")
                try:
                    before = int(before_turns)
                    after = int(after_turns)
                except (TypeError, ValueError):
                    return _invalid("turn_window_must_be_non_negative_integer")
                if before < 0 or after < 0:
                    return _invalid("turn_window_must_be_non_negative_integer")
                try:
                    record = self.store.get_retrieval_record(
                        namespace=self.namespace,
                        source_id=anchor_id,
                        cross_conversation=False,
                    )
                except SchemaError as exc:
                    return _invalid(str(exc) or "ambiguous_memory_id")
                if record is not None and str(record.get("entry_type") or "raw") != "raw":
                    return _invalid("raw_anchor_required")
                try:
                    window = self.store.get_raw_turn_window(
                        namespace=self.namespace,
                        anchor_source_id=anchor_id,
                        before_turns=before,
                        after_turns=after,
                    )
                except NotImplementedError:
                    return {
                        "status": "unavailable",
                        "reason": "raw_turn_window_store_unsupported",
                        "messages": [],
                        "message_count": 0,
                        "text": "",
                    }
                raw_messages = [entry.to_record() for entry in window.entries]
                selector_payload = {
                    "mode": "anchor",
                    "anchor_source_id": anchor_id,
                    "before_turns": before,
                    "after_turns": after,
                }
                result_metadata = {
                    "reason": window.reason,
                    "anchor_source_id": anchor_id,
                    "before_turns": before,
                    "after_turns": after,
                }
            else:
                try:
                    selector = normalize_timeline_time_selector(
                        timezone=self.timezone,
                        time_range=time_range,
                        date_from=start,
                        date_to=str(date_to or "").strip(),
                        time_periods=time_periods,
                    )
                except (TypeError, ValueError) as exc:
                    return _invalid(str(exc) or "invalid_time_selector")
                periods = list(selector.time_periods)
                raw_messages = self.store.get_messages_by_time_range(
                    namespace=self.namespace,
                    start_ts=selector.start_ts,
                    end_ts=selector.end_ts,
                    time_periods=periods,
                    cross_conversation=cross_conversation,
                )
                selector_payload = {
                    "mode": "time",
                    "selector_mode": selector.mode,
                    "start_ts": selector.start_ts,
                    "end_ts": selector.end_ts,
                    "start_at": selector.start_at,
                    "end_at": selector.end_at,
                    "time_periods": periods,
                    "cross_conversation": bool(cross_conversation),
                    "date_from": start if selector.mode == "date" else "",
                    "date_to": (str(date_to or "").strip() or start) if selector.mode == "date" else "",
                }
                result_metadata = {
                    "selector_mode": selector.mode,
                    "time_range": selector.to_dict(),
                    "time_periods": periods,
                }
                if selector.mode == "date":
                    result_metadata["date_from"] = start
                    result_metadata["date_to"] = str(date_to or "").strip() or start

        try:
            resolved_projection = normalize_timeline_projection(projection)
            entries = [TimelineEntry.from_record(record) for record in raw_messages]
            units = build_timeline_read_units(
                entries,
                projection=resolved_projection,
                renderer_registry=self.renderer_registry,
                timezone=self.timezone,
            )
            count_text = self.token_counter.count_text if self.token_counter is not None else None
            page = paginate_timeline_units(
                units,
                page_token_budget=int(page_token_budget or 0),
                count_text=count_text,
                cursor_selector=selector_payload,
                projection=resolved_projection,
                namespace=self.namespace,
                after_key=after_key,
                timezone=self.timezone,
            )
        except (TypeError, ValueError) as exc:
            return _invalid(str(exc) or "invalid_timeline_options")

        messages = list(page.messages)
        text = render_timeline(messages, tz=self.timezone) if messages else ""
        timestamps = [int(message.get("timestamp") or 0) for message in messages if int(message.get("timestamp") or 0)]
        token_count_quality = self.token_counter.quality if self.token_counter is not None else "estimated"
        coverage = {
            "complete": page.complete,
            "coverage_complete": page.complete,
            "requested_start": str(selector_payload.get("start_at") or ""),
            "requested_end": str(selector_payload.get("end_at") or ""),
            "returned_start": timestamp_iso(min(timestamps), timezone=self.timezone) if timestamps else "",
            "returned_end": timestamp_iso(max(timestamps), timezone=self.timezone) if timestamps else "",
            "logical_unit_count": page.logical_unit_count,
            "total_logical_unit_count": page.total_logical_unit_count,
            "selected_logical_unit_count": page.total_logical_unit_count,
            "returned_logical_unit_count": page.logical_unit_count,
            "returned_logical_unit_ids": list(page.logical_unit_ids),
            "remaining_logical_unit_count": page.remaining_logical_unit_count,
            "entry_count": page.entry_count,
            "total_entry_count": page.total_entry_count,
            "selected_entry_count": page.total_entry_count,
            "returned_entry_count": page.entry_count,
            "remaining_entry_count": page.remaining_entry_count,
            "compacted_entry_count": page.compacted_entry_count,
            "next_cursor": page.next_cursor,
            "oversized_unit": page.oversized_unit,
            "page_token_budget": int(page_token_budget or 0),
            "projected_token_count": page.returned_projected_token_count,
            "selected_projected_token_count": page.selected_projected_token_count,
            "returned_projected_token_count": page.returned_projected_token_count,
            "remaining_projected_token_count": page.remaining_projected_token_count,
            "token_count_quality": token_count_quality,
        }
        selector_reason = str(result_metadata.pop("reason", "") or "")
        status = "partial" if messages and not page.complete else ("ok" if messages else "empty")
        reason = (
            "page_boundary"
            if status == "partial"
            else (selector_reason or ("" if status == "ok" else "no_timeline_entries"))
        )
        suggested_next_actions = ["continue_page", "browse_memory_for_overview"] if status == "partial" else []
        return {
            "status": status,
            "reason": reason,
            **result_metadata,
            "projection": resolved_projection,
            "coverage": coverage,
            "selected_logical_unit_count": page.total_logical_unit_count,
            "selected_entry_count": page.total_entry_count,
            "selected_projected_token_count": page.selected_projected_token_count,
            "returned_logical_unit_count": page.logical_unit_count,
            "returned_entry_count": page.entry_count,
            "returned_projected_token_count": page.returned_projected_token_count,
            "coverage_complete": page.complete,
            "next_cursor": page.next_cursor,
            "suggested_next_actions": suggested_next_actions,
            "message_count": len(messages),
            "messages": messages,
            "text": text,
        }

    def browse_memory(
        self,
        *,
        time_range: dict[str, Any] | None = None,
        date_from: str = "",
        date_to: str = "",
        node_types: list[str] | None = None,
        cross_conversation: bool = False,
        page_size: int = 50,
        cursor: str = "",
    ) -> dict[str, Any]:
        """Browse compact memory cards for a deterministic time range.

        This is a SQLite catalog read, not semantic Top-K.  Continuation cursors
        freeze the selector and return complete cards without hidden truncation.
        """

        from .memory_catalog import build_memory_card
        from .memory_navigation import (
            MEMORY_NODE_TYPES,
            decode_memory_cursor,
            merge_intervals,
            normalize_page_size,
            paginate_memory_items,
        )
        from .time_anchor import normalize_timeline_time_selector
        from .timeline_read import timestamp_iso

        def _invalid(reason: str) -> dict[str, Any]:
            return {
                "status": "invalid_filter",
                "reason": reason,
                "cards": [],
                "matched_card_count": 0,
                "returned_card_count": 0,
                "next_cursor": "",
            }

        cursor_token = str(cursor or "").strip()
        if cursor_token:
            if (
                time_range is not None
                or str(date_from or "").strip()
                or str(date_to or "").strip()
                or node_types is not None
                or bool(cross_conversation)
                or page_size != 50
            ):
                return _invalid("cursor_options_are_embedded")
            try:
                decoded = decode_memory_cursor(
                    cursor_token,
                    namespace=self.namespace,
                    expected_mode="browse",
                )
                selector_payload = dict(decoded["selector"])
                page_size = int(decoded["page_size"])
                after_key = tuple(decoded["last_key"])
                start_ts = int(selector_payload["start_ts"])
                end_ts = int(selector_payload["end_ts"])
                resolved_node_types = [str(item) for item in list(selector_payload["node_types"])]
                cross_conversation = bool(selector_payload["cross_conversation"])
            except (KeyError, TypeError, ValueError) as exc:
                return _invalid(str(exc) or "invalid_cursor")
        else:
            try:
                page_size = normalize_page_size(page_size)
                requested = list(node_types or ["episodic"])
                resolved_node_types = list(dict.fromkeys(str(item or "").strip().lower() for item in requested))
                if not resolved_node_types or any(item not in MEMORY_NODE_TYPES for item in resolved_node_types):
                    raise ValueError("invalid_memory_node_types")
                selector = normalize_timeline_time_selector(
                    timezone=self.timezone,
                    time_range=time_range,
                    date_from=str(date_from or "").strip(),
                    date_to=str(date_to or "").strip(),
                )
            except (TypeError, ValueError) as exc:
                return _invalid(str(exc) or "invalid_memory_browse_options")
            start_ts = selector.start_ts
            end_ts = selector.end_ts
            after_key = None
            selector_payload = {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "start_at": selector.start_at,
                "end_at": selector.end_at,
                "node_types": resolved_node_types,
                "cross_conversation": bool(cross_conversation),
            }

        try:
            records: list[dict[str, Any]] = []
            if "episodic" in resolved_node_types:
                records.extend(
                    self.store.get_episodic_summaries_by_time_range(
                        namespace=self.namespace,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        cross_conversation=cross_conversation,
                    )
                )
            if "semantic" in resolved_node_types:
                records.extend(
                    self.store.get_semantic_summaries_by_time_range(
                        namespace=self.namespace,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        cross_conversation=cross_conversation,
                    )
                )
            cards = [build_memory_card(record, timezone=self.timezone) for record in records]
            cards.sort(
                key=lambda card: (
                    int(card.get("period_start_ts") or 0),
                    int(card.get("period_end_ts") or 0),
                    str(card.get("node_type") or ""),
                    str(card.get("memory_id") or ""),
                )
            )
            page = paginate_memory_items(
                cards,
                item_keys=[
                    (
                        int(card.get("period_start_ts") or 0),
                        int(card.get("period_end_ts") or 0),
                        str(card.get("node_type") or ""),
                        str(card.get("memory_id") or ""),
                    )
                    for card in cards
                ],
                namespace=self.namespace,
                mode="browse",
                selector=selector_payload,
                page_size=page_size,
                after_key=after_key,
            )
            raw_coverage = self.store.get_catalog_raw_coverage(
                namespace=self.namespace,
                start_ts=start_ts,
                end_ts=end_ts,
                cross_conversation=cross_conversation,
            )
        except NotImplementedError:
            return {
                "status": "unavailable",
                "reason": "memory_catalog_store_unsupported",
                "cards": [],
                "matched_card_count": 0,
                "returned_card_count": 0,
                "next_cursor": "",
            }
        except (TypeError, ValueError) as exc:
            return _invalid(str(exc) or "invalid_memory_browse_options")

        covered_intervals = merge_intervals(
            [
                {
                    "start_ts": max(start_ts, int(card.get("period_start_ts") or 0)),
                    "end_ts": min(
                        end_ts,
                        max(
                            int(card.get("period_start_ts") or 0) + 1,
                            int(card.get("period_end_ts") or 0) + 1,
                        ),
                    ),
                }
                for card in cards
            ]
        )
        for interval in covered_intervals:
            interval["start_at"] = timestamp_iso(interval["start_ts"], timezone=self.timezone)
            interval["end_at"] = timestamp_iso(interval["end_ts"], timezone=self.timezone)
        coverage = {
            "requested_range": {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "start_at": str(selector_payload.get("start_at") or ""),
                "end_at": str(selector_payload.get("end_at") or ""),
            },
            "covered_intervals": covered_intervals,
            "live_raw_intervals": list(raw_coverage.get("live_raw_intervals") or []),
            "gap_intervals": list(raw_coverage.get("gap_intervals") or []),
            "total_source_count": int(raw_coverage.get("total_source_count") or 0),
            "covered_source_count": int(raw_coverage.get("covered_source_count") or 0),
            "live_source_count": int(raw_coverage.get("live_source_count") or 0),
            "gap_source_count": int(raw_coverage.get("gap_source_count") or 0),
            "complete": int(raw_coverage.get("gap_source_count") or 0) == 0,
            "meaning": "all stored raw sources in the requested range are accounted for; quiet wall-clock gaps are not missing memory",
        }
        return {
            "status": "ok" if cards or coverage["total_source_count"] else "empty",
            "reason": "" if cards else "no_catalog_cards_in_range",
            "node_types": resolved_node_types,
            "cross_conversation": bool(cross_conversation),
            "coverage": coverage,
            "matched_card_count": len(cards),
            "returned_card_count": page.returned_count,
            "remaining_card_count": page.remaining_count,
            "cards": list(page.items),
            "next_cursor": page.next_cursor,
            "page_complete": page.complete,
            "page_size": page_size,
            "suggested_next_actions": [
                "open_memory(view=content) to read a selected summary",
                "open_memory(view=sources) to inspect its exact child evidence",
                "call browse_memory again with only next_cursor when page_complete is false",
            ],
        }

    def open_memory(
        self,
        *,
        memory_id: str = "",
        memory_ids: list[str] | tuple[str, ...] | None = None,
        view: str = "card",
        detail: str = "full",
        projection: str = "conversation",
        cross_conversation: bool = False,
        page_size: int = 50,
        cursor: str = "",
    ) -> dict[str, Any]:
        """Open one or several raw, episodic, or semantic nodes through one facade.

        ``detail`` controls one raw content node. Episodic raw ``sources`` use
        ``projection``: conversation keeps dialogue/events full and compacts
        operation/material bodies; full/tools must be explicitly requested.

        ``memory_ids`` is the batch form for ``card`` and ``content``. Results
        preserve request order and report status/reason independently for each
        node. Batch ``sources`` is deliberately rejected because each source
        tree owns an independent cursor.
        """

        from .memory_catalog import build_memory_card
        from .memory_navigation import (
            MEMORY_DETAILS,
            MEMORY_VIEWS,
            decode_memory_cursor,
            normalize_page_size,
            paginate_memory_items,
        )
        from .rendering import render_semantic_snippet, render_summary_snippet, render_timeline
        from .timeline_read import (
            build_timeline_read_units,
            normalize_timeline_projection,
            project_timeline_entry,
        )

        def _invalid(reason: str) -> dict[str, Any]:
            return {"status": "invalid_filter", "reason": reason, "memory_id": "", "view": "", "result": None}

        cursor_token = str(cursor or "").strip()
        if memory_ids is not None:
            if isinstance(memory_ids, (str, bytes, bytearray)):
                return _invalid("memory_ids_must_be_array")
            requested_ids = [str(item or "").strip() for item in memory_ids]
            if any(not item for item in requested_ids):
                return _invalid("memory_ids_must_not_contain_empty_values")
            if str(memory_id or "").strip() or cursor_token:
                return _invalid("memory_id_memory_ids_and_cursor_are_mutually_exclusive")
            batch_view = str(view or "card").strip().lower()
            if not requested_ids:
                return _invalid("memory_ids_required")
            if batch_view == "sources":
                return _invalid("batch_sources_not_supported")
            if batch_view not in {"card", "content"}:
                return _invalid("invalid_memory_view")

            opened = [
                self.open_memory(
                    memory_id=item,
                    view=batch_view,
                    detail=detail,
                    projection=projection,
                    cross_conversation=cross_conversation,
                )
                for item in requested_ids
            ]
            opened_count = sum(str(item.get("status") or "") == "ok" for item in opened)
            failed_count = len(opened) - opened_count
            if opened_count == len(opened):
                status = "ok"
                reason = ""
            elif opened_count:
                status = "partial"
                reason = "some_memory_nodes_unavailable"
            else:
                status = "empty"
                reason = "no_memory_nodes_opened"
            text_parts: list[str] = []
            for item in opened:
                item_id = str(item.get("memory_id") or "")
                item_status = str(item.get("status") or "failed")
                item_reason = str(item.get("reason") or "")
                header = f"[memory_id={item_id} status={item_status}]"
                body = str(item.get("text") or "").strip()
                text_parts.append("\n".join(part for part in (header, body or f"reason={item_reason}") if part))
            return {
                "status": status,
                "reason": reason,
                "memory_id": "",
                "memory_ids": requested_ids,
                "view": batch_view,
                "detail": str(detail or "full").strip().lower(),
                "projection": str(projection or "conversation").strip().lower(),
                "cross_conversation": bool(cross_conversation),
                "result": {
                    "requested_count": len(requested_ids),
                    "opened_count": opened_count,
                    "failed_count": failed_count,
                    "items": opened,
                },
                "text": "\n\n".join(text_parts),
            }
        if cursor_token:
            if (
                str(memory_id or "").strip()
                or str(view or "card").strip().lower() != "card"
                or str(detail or "full").strip().lower() != "full"
                or str(projection or "conversation").strip().lower() != "conversation"
                or bool(cross_conversation)
                or page_size != 50
            ):
                return _invalid("cursor_options_are_embedded")
            try:
                decoded = decode_memory_cursor(
                    cursor_token,
                    namespace=self.namespace,
                    expected_mode="open_sources",
                )
                selector = dict(decoded["selector"])
                memory_id = str(selector["memory_id"])
                view = str(selector["view"])
                detail = str(selector["detail"])
                projection = str(selector.get("projection") or "conversation")
                cross_conversation = bool(selector["cross_conversation"])
                page_size = int(decoded["page_size"])
                after_key = tuple(decoded["last_key"])
            except (KeyError, TypeError, ValueError) as exc:
                return _invalid(str(exc) or "invalid_cursor")
        else:
            memory_id = str(memory_id or "").strip()
            view = str(view or "card").strip().lower()
            detail = str(detail or "full").strip().lower()
            try:
                projection = normalize_timeline_projection(projection)
            except ValueError:
                return _invalid("invalid_memory_projection")
            try:
                page_size = normalize_page_size(page_size)
            except ValueError as exc:
                return _invalid(str(exc))
            after_key = None
            if not memory_id:
                return _invalid("memory_id_required")
            if view not in MEMORY_VIEWS:
                return _invalid("invalid_memory_view")
            if detail not in MEMORY_DETAILS:
                return _invalid("invalid_memory_detail")

        try:
            record = self.store.get_retrieval_record(
                namespace=self.namespace,
                source_id=memory_id,
                cross_conversation=cross_conversation,
            )
        except NamespaceError:
            record = None
        except SchemaError as exc:
            return {
                "status": "unavailable",
                "reason": str(exc) or "ambiguous_memory_id",
                "memory_id": memory_id,
                "view": view,
                "result": None,
            }
        if record is None:
            return {
                "status": "empty",
                "reason": "memory_not_found_or_out_of_scope",
                "memory_id": memory_id,
                "view": view,
                "result": None,
            }

        card = build_memory_card(record, timezone=self.timezone)
        node_type = str(card["node_type"])
        base = {
            "memory_id": memory_id,
            "node_type": node_type,
            "view": view,
            "detail": detail,
            "projection": projection,
            "cross_conversation": bool(cross_conversation),
        }
        if view == "card":
            return {"status": "ok", "reason": "", **base, "result": card, "text": ""}

        if view == "content":
            if node_type == "raw":
                entry = TimelineEntry.from_record(record)
                projected, _ = project_timeline_entry(
                    entry,
                    projection="full",
                    renderer_registry=self.renderer_registry,
                    timezone=self.timezone,
                    detail_override=detail,
                )
                return {
                    "status": "ok",
                    "reason": "",
                    **base,
                    "result": projected,
                    "text": str((projected or {}).get("content") or ""),
                }
            if node_type == "episodic":
                content = {
                    "card": card,
                    "diary_summary": str(record.get("diary_summary") or ""),
                    "key_events": list(record.get("key_events") or []),
                    "core_facts": list(record.get("core_facts") or []),
                    "period_label": str(record.get("period_label") or ""),
                    "event_type": str(record.get("event_type") or ""),
                }
                text = render_summary_snippet(record, tz=self.timezone, enable_flavor=self.config.enable_flavor)
            else:
                content = {
                    "card": card,
                    "semantic_summary": str(record.get("semantic_summary") or ""),
                    "stable_facts": list(record.get("stable_facts") or []),
                    "recurring_topics": list(record.get("recurring_topics") or []),
                    "important_people": list(record.get("important_people") or []),
                    "open_loops": list(record.get("open_loops") or []),
                }
                text = render_semantic_snippet(record, tz=self.timezone, enable_flavor=self.config.enable_flavor)
            return {"status": "ok", "reason": "", **base, "result": content, "text": text}

        selector = {
            "memory_id": memory_id,
            "view": "sources",
            "detail": detail,
            "projection": projection,
            "cross_conversation": bool(cross_conversation),
        }
        if node_type == "raw":
            return {
                "status": "empty",
                "reason": "raw_node_has_no_sources",
                **base,
                "result": {"source_count": 0, "sources": []},
                "text": "",
            }
        child_ids = [
            str(item or "").strip()
            for item in (record.get("source_ids") if node_type == "episodic" else record.get("source_summary_ids"))
            if str(item or "").strip()
        ]
        if node_type == "semantic":
            children = [
                self.store.get_retrieval_record(
                    namespace=self.namespace,
                    source_id=source_id,
                    cross_conversation=cross_conversation,
                )
                for source_id in child_ids
            ]
            valid_records = [
                child for child in children if child is not None and str(child.get("entry_type") or "") == "summary"
            ]
            found_ids = {str(child.get("summary_id") or "") for child in valid_records}
            missing = [source_id for source_id in child_ids if source_id not in found_ids]
            cards = [build_memory_card(child, timezone=self.timezone) for child in valid_records]
            cards.sort(
                key=lambda child: (
                    int(child.get("period_start_ts") or 0),
                    int(child.get("period_end_ts") or 0),
                    str(child.get("memory_id") or ""),
                )
            )
            page = paginate_memory_items(
                cards,
                item_keys=[
                    (
                        int(child.get("period_start_ts") or 0),
                        int(child.get("period_end_ts") or 0),
                        str(child.get("memory_id") or ""),
                    )
                    for child in cards
                ],
                namespace=self.namespace,
                mode="open_sources",
                selector=selector,
                page_size=page_size,
                after_key=after_key,
            )
            result = {
                "source_node_type": "episodic",
                "source_count": len(child_ids),
                "returned_source_count": page.returned_count,
                "missing_source_ids": missing,
                "sources": list(page.items),
                "next_cursor": page.next_cursor,
                "page_complete": page.complete,
            }
            return {
                "status": "partial" if missing else "ok",
                "reason": "source_lineage_incomplete" if missing else "",
                **base,
                "result": result,
                "text": "",
            }

        try:
            entries = self.store.get_entries_by_source_ids(
                namespace=self.namespace,
                source_ids=tuple(child_ids),
                cross_conversation=cross_conversation,
            )
        except NotImplementedError:
            return {
                "status": "unavailable",
                "reason": "memory_source_store_unsupported",
                **base,
                "result": None,
                "text": "",
            }
        found_ids = {entry.source_id for entry in entries}
        missing = [source_id for source_id in child_ids if source_id not in found_ids]
        units = build_timeline_read_units(
            entries,
            projection=projection,
            renderer_registry=self.renderer_registry,
            timezone=self.timezone,
        )
        page = paginate_memory_items(
            units,
            item_keys=[unit.sort_key for unit in units],
            namespace=self.namespace,
            mode="open_sources",
            selector=selector,
            page_size=page_size,
            after_key=after_key,
        )
        source_units = [{"unit_id": unit.unit_id, "entries": list(unit.projected_messages)} for unit in page.items]
        messages = [message for unit in page.items for message in unit.projected_messages]
        result = {
            "source_node_type": "raw",
            "source_count": len(child_ids),
            "logical_unit_count": len(units),
            "returned_logical_unit_count": page.returned_count,
            "compacted_entry_count": sum(unit.compacted_entry_count for unit in page.items),
            "missing_source_ids": missing,
            "source_units": source_units,
            "next_cursor": page.next_cursor,
            "page_complete": page.complete,
        }
        return {
            "status": "partial" if missing else "ok",
            "reason": "source_lineage_incomplete" if missing else "",
            **base,
            "result": result,
            "text": render_timeline(messages, tz=self.timezone) if messages else "",
        }

    def read_entry(self, *, source_id: str, detail: str = "full") -> dict[str, Any]:
        """Compatibility adapter for the raw-only ``open_memory(content)`` view."""

        sid = str(source_id or "").strip()
        resolved_detail = str(detail or "full").strip().lower()
        opened = self.open_memory(
            memory_id=sid,
            view="content",
            detail=resolved_detail,
            cross_conversation=False,
        )
        if opened.get("status") == "invalid_filter":
            reason = str(opened.get("reason") or "invalid_filter")
            if reason == "memory_id_required":
                reason = "source_id_required"
            elif reason == "invalid_memory_detail":
                reason = "invalid_entry_detail"
            return {"status": "invalid_filter", "reason": reason, "entry": None, "text": ""}
        if opened.get("status") != "ok":
            return {
                "status": "empty",
                "reason": "entry_not_found_or_out_of_scope",
                "source_id": sid,
                "detail": resolved_detail,
                "entry": None,
                "text": "",
            }
        if opened.get("node_type") != "raw":
            return {
                "status": "empty",
                "reason": "raw_entry_required",
                "source_id": sid,
                "detail": resolved_detail,
                "entry": None,
                "text": "",
            }
        return {
            "status": "ok",
            "reason": "",
            "source_id": sid,
            "detail": resolved_detail,
            "entry": opened.get("result"),
            "text": str(opened.get("text") or ""),
        }

    def forget_namespace(self, namespace: Namespace | None = None) -> dict[str, Any]:
        """定向遗忘:删除 store 记录,并尽力同步清 VectorIndex;失败时结构化报告 partial。"""
        target = namespace or self.namespace
        deleted_ids = self.store.delete_namespace(namespace=target)
        try:
            self.index.delete(deleted_ids)
        except Exception as exc:
            return {
                "ok": False,
                "status": "partial",
                "deleted_count": len(deleted_ids),
                "source_ids": deleted_ids,
                "index_deleted": False,
                "reason": str(exc) or exc.__class__.__name__,
            }
        return {
            "ok": True,
            "status": "deleted",
            "deleted_count": len(deleted_ids),
            "source_ids": deleted_ids,
            "index_deleted": True,
        }
