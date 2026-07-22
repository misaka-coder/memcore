"""MemorySystem —— 对外唯一门面。

写侧(切片 4)已接通:record_user_turn / record_assistant_turn / compact_due_background / compact_due_sync / embedding_status。
读侧(切片 5)已接通:build_prompt_context / retrieve。遗忘会同步清 store 与 VectorIndex。

生命周期(见设计文档 §3.3),使用方按顺序接:
    record_user_turn → build_prompt_context → record_assistant_turn → compact_due_background
"""

from __future__ import annotations

import re
import time
import uuid
from concurrent.futures import Future
from dataclasses import replace
from typing import Any
from zoneinfo import ZoneInfo

from .compaction import Compaction
from .config import MemoryConfig
from .embedding.base import EmbeddingProvider
from .errors import ConfigError, NamespaceError, SchemaError
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
    sanitize_projection_payload,
    stable_projection_hash,
)
from .retrieval import ReadPipeline, RetrievalRequest, RetrievalResult
from .runtime import MemCoreRuntime
from .schema import TRACE_CATEGORIES, coerce_memory_metadata
from .store.base import MemoryStore
from .store.sqlite_store import SQLiteMemoryStore
from .time_anchor import infer_time_of_day, timestamp_to_date_label
from .timeline import (
    AnnotationStatus,
    CompletionCommitResult,
    EntryTrust,
    MemoryAnnotation,
    RetrievalVisibility,
    TimelineEntry,
    TimelineEntryInput,
    TurnAbortResult,
    TurnCompletion,
    TurnHandle,
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
        if self.config.raw_compaction_policy == "token" and token_counter is None:
            raise ConfigError("token_counter is required when raw_compaction_policy='token'")
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
            llm=self.llm,
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
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
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
                    categories=self.config.categories,
                    enable_flavor=self.config.enable_flavor,
                ).to_dict()
                metadata["categories"] = [item for item in metadata["categories"] if item not in TRACE_CATEGORIES]
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
        return self._refresh_completion_after_index(result)

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

    def build_context_projection(self, *, provider_profile: str) -> ContextProjection:
        """Build and freeze this conversation's provider-visible append-only history."""

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

        messages: list[ProjectionMessage] = []
        for record, id_key, prefix in (
            *((item, "semantic_id", "semantic") for item in reversed(semantic)),
            *(
                (item, "summary_id", "summary")
                for item in reversed(episodic)
                if str(item.get("retrieval_visibility") or "default") == "default"
            ),
        ):
            messages.extend(
                self._freeze_memory_record_projection(
                    record=record,
                    id_key=id_key,
                    turn_prefix=prefix,
                    provider_profile=profile,
                )
            )
        for turn_id, turn_entries in grouped.items():
            messages.extend(
                self._freeze_turn_projection(
                    turn_id=turn_id,
                    entries=turn_entries,
                    provider_profile=profile,
                )
            )

        try:
            compaction_generation, projection_generation = self.store.get_conversation_generations(
                namespace=self.namespace
            )
        except NotImplementedError as exc:
            raise SchemaError("store_timeline_v2_unsupported") from exc
        return ContextProjection(
            provider_profile=profile,
            messages=tuple(messages),
            projection_version=max((message.projection_version for message in messages), default=1),
            stable_prefix_hash=stable_projection_hash([message.payload for message in messages]),
            entry_projection_hashes=build_entry_projection_hashes(messages),
            compaction_generation=compaction_generation,
            projection_generation=projection_generation,
        )

    def _freeze_turn_projection(
        self,
        *,
        turn_id: str,
        entries: list[TimelineEntry],
        provider_profile: str,
    ) -> list[ProjectionMessage]:
        try:
            return self._projection_ledger.freeze_turn_entries(
                namespace=self.namespace,
                turn_id=turn_id,
                entries=entries,
                provider_profile=provider_profile,
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
    ) -> list[ProjectionMessage]:
        try:
            return self._projection_ledger.freeze_memory_record(
                namespace=self.namespace,
                record=record,
                id_key=id_key,
                turn_prefix=turn_prefix,
                provider_profile=provider_profile,
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

        if len(history_messages) < len(prepared):
            raise SchemaError("projection_actual_history_missing_turn_suffix")
        actual_tail = history_messages[-len(prepared) :] if prepared else []
        for actual, declared in zip(actual_tail, prepared):
            safe_actual, _ = sanitize_projection_payload(actual)
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
            categories=self.config.categories,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        metadata["categories"] = [item for item in metadata["categories"] if item not in TRACE_CATEGORIES]
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
        return self._record(role="user", content=content, actor=actor, **fields)

    def record_assistant_turn(
        self, reply: str, *, in_reply_to: dict[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        return self._record(role="assistant", content=reply, actor=None, **fields)

    def record_external_event(
        self,
        *,
        event_type: str,
        fields: dict[str, Any] | None = None,
        source: str = "",
        timestamp: int | None = None,
        source_id: str | None = None,
        keywords: list[str] | None = None,
        importance: float = 0.4,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Append one typed external event to the same linear raw timeline."""

        from .rendering import render_external_event_text

        label = self._tool_event_part(event_type, fallback="external")
        tags = [label, *[str(item or "").strip() for item in (keywords or [])]]
        metadata = {
            "categories": ["event_trace"],
            "keywords": [item for item in tags if item][:4],
            "subject_scopes": ["other"],
            "importance": importance,
            "confidence": confidence,
        }
        record_fields: dict[str, Any] = {
            "annotation_status": "unannotated",
            "memory_metadata": metadata,
            "retrieval_policy": "explicit",
            "retrieval_visibility": "explicit",
        }
        if timestamp is not None:
            record_fields["timestamp"] = timestamp
        if str(source_id or "").strip():
            record_fields["source_id"] = str(source_id).strip()
        return self._record(
            role=f"event.{label}",
            content=render_external_event_text(
                event_type=label,
                fields=dict(fields or {}),
                source=source,
            ),
            actor=None,
            **record_fields,
        )

    def record_tool_exchange(
        self,
        *,
        tool_name: str,
        result: Any,
        tool_input: Any = None,
        tool_call_id: str = "",
        source: str = "",
        timestamp: int | None = None,
        source_id_prefix: str | None = None,
        keywords: list[str] | None = None,
        importance: float = 0.2,
        confidence: float = 1.0,
    ) -> dict[str, dict[str, Any]]:
        """按线性消息序列追加一次工具调用:assistant.tool_call + tool.<name>。"""
        from .rendering import render_tool_result_text, render_tool_use_text

        tool = str(tool_name or "").strip()
        prefix = str(source_id_prefix or "").strip()
        call_id = self._tool_event_part(tool_call_id or prefix or f"call_{uuid.uuid4().hex[:8]}", fallback="call")
        tool_label = self._tool_event_part(tool, fallback="tool")
        result_source = str(source or "").strip() or self._default_tool_source(tool)
        tags = [tool, *[str(item or "").strip() for item in (keywords or [])]]
        tags = [item for item in tags if item]
        metadata = {
            "categories": ["tool_trace"],
            "keywords": tags[:4],
            "subject_scopes": ["assistant"],
            "importance": importance,
            "confidence": confidence,
        }
        ts = timestamp
        trace_fields = {
            "annotation_status": "unannotated",
            "memory_metadata": metadata,
            "retrieval_policy": "explicit",
            "retrieval_visibility": "explicit",
        }
        use_fields: dict[str, Any] = dict(trace_fields)
        result_fields: dict[str, Any] = dict(trace_fields)
        if ts is not None:
            use_fields["timestamp"] = ts
            result_fields["timestamp"] = ts + 1
        if prefix:
            use_fields["source_id"] = f"{prefix}:tool_use"
            result_fields["source_id"] = f"{prefix}:tool_result"

        tool_use = self._record(
            role=f"assistant.tool_call {tool_label} {call_id}",
            content=render_tool_use_text(tool_input=tool_input),
            actor=None,
            **use_fields,
        )
        tool_result = self._record(
            role=f"tool.{tool_label} {call_id}",
            content=render_tool_result_text(result=result, source=result_source),
            actor=None,
            **result_fields,
        )
        return {"tool_use": tool_use, "tool_result": tool_result}

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
        keywords: list[str] | None = None,
        importance: float = 0.25,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """追加附件/文件引用事件:只记录材料锚点,不把文件本体写进 memory。"""
        from .rendering import render_material_reference_text

        file_key = self._tool_event_part(file_id, fallback="file")
        kind_label = self._tool_event_part(kind, fallback="material")
        tags = self._material_keywords(
            file_id=file_id,
            kind=kind,
            filename=filename,
            extra=keywords,
        )
        metadata = {
            "categories": ["material_trace"],
            "keywords": tags,
            "subject_scopes": ["user"],
            "importance": importance,
            "confidence": confidence,
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
        return self._record(
            role=f"user.attachment {kind_label} {file_key}",
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
        keywords: list[str] | None = None,
        importance: float = 0.2,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """追加材料清理事件,让模型知道旧文件/解析物可能已经不可再读。"""
        from .rendering import render_material_cleanup_text

        file_key = self._tool_event_part(file_id, fallback="file")
        kind_label = self._tool_event_part(kind, fallback="material")
        tags = self._material_keywords(
            file_id=file_id,
            kind=kind,
            filename=filename,
            extra=keywords,
        )
        metadata = {
            "categories": ["material_trace"],
            "keywords": tags,
            "subject_scopes": ["other"],
            "importance": importance,
            "confidence": confidence,
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
        return self._record(
            role=f"system.material_cleanup {kind_label} {file_key}",
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
    def _material_keywords(*, file_id: str, kind: str, filename: str, extra: list[str] | None = None) -> list[str]:
        tags = [file_id, filename, kind, *[str(item or "").strip() for item in (extra or [])]]
        out: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            text = str(tag or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= 4:
                break
        return out

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

    def _record(self, *, role: str, content: str, actor: Actor | None, **fields: Any) -> dict[str, Any]:
        ts = int(fields.pop("timestamp", None) or time.time())
        # ``index_in_vector`` is a retrieval-routing decision, not a write
        # decision.  Even when a caller opts this turn out of vector search,
        # the SQLite raw record remains the truth source and must be kept.
        index_in_vector = bool(fields.pop("index_in_vector", True))
        source_id = fields.pop("source_id", None) or uuid.uuid4().hex
        metadata = coerce_memory_metadata(
            fields.pop("memory_metadata", None),
            categories=self.config.categories,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        if role in {"assistant", "user"}:
            fields.setdefault("annotation_status", "accepted_host")
            fields.setdefault("annotation_source", "host_adapter")
            fields.setdefault("retrieval_policy", "always")
            fields.setdefault("retrieval_visibility", "default")
        ns = self.namespace if actor is None else self._with_actor(actor)
        rec = self.store.add_message(
            namespace=ns,
            role=role,
            content=content,
            timestamp=ts,
            source_id=source_id,
            date_label=timestamp_to_date_label(ts, self.timezone),
            time_of_day=infer_time_of_day(ts, self.timezone),
            memory_metadata=metadata,
            **fields,
        )
        if not index_in_vector:
            # Keep the opt-out durable and out of the repair outbox.  This is
            # intentionally distinct from ``pending``: no vector upsert is
            # expected for this record.
            self.store.set_index_status(rec["source_id"], "skipped")
            rec["index_status"] = "skipped"
            return rec
        # outbox:写库已成功(pending);向量 upsert 失败就留 pending,交给 reindex_pending 自愈,不阻断记录。
        try:
            self.index.upsert([build_raw_entry(rec)])
            self.store.set_index_state(
                rec["source_id"],
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )
            rec["index_status"] = "indexed"  # 返回值与 store 同步,别让调用方误判
            rec["index_schema_version"] = INDEX_SCHEMA_VERSION
            rec["index_key"] = INDEX_SCHEMA_KEY
        except Exception:
            rec["index_status"] = "pending"
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

    def reindex_pending(self, *, limit: int = 100) -> dict[str, int]:
        """outbox 自愈:把 index_status=pending 的记录补做向量 upsert。可定期/启动时调用。"""
        pending = self.store.list_pending_index(limit=limit)
        repaired = failed = 0
        for rec in pending:
            try:
                self._reindex_record(rec)
                repaired += 1
            except Exception:
                failed += 1
        return {"scanned": len(pending), "repaired": repaired, "failed": failed}

    def reindex_all(
        self,
        *,
        namespace: Namespace | None = None,
        limit: int | None = None,
        current_conversation_only: bool = False,
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
        for rec in records:
            try:
                self._reindex_record(rec)
                reindexed += 1
            except Exception:
                source_id = self._record_index_id(rec)
                if source_id:
                    self.store.set_index_status(source_id, "pending")
                failed += 1
        return {"scanned": len(records), "reindexed": reindexed, "failed": failed}

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
            categories=self.config.categories,
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
            categories=self.config.categories,
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
        keywords: list[str] | None = None,
        source_layers: list[str] | None = None,
        categories: list[str] | None = None,
        subject_scopes: list[str] | None = None,
        importance_min: float | None = None,
        time_hint: dict[str, Any] | None = None,
        kind_patterns: list[str] | None = None,
        include_explicit: bool = False,
        cross_conversation: bool = False,
        exclude_source_ids: list[str] | None = None,
        max_matches: int = 0,
        result_token_budget: int = 0,
    ) -> RetrievalResult:
        """Return typed found/empty/invalid/unavailable/failed retrieval state."""
        return self._read.retrieve_result(
            namespace=self.namespace,
            request=RetrievalRequest(
                query=query,
                keywords=tuple(keywords or ()),
                source_layers=tuple(source_layers or ()),
                categories=tuple(categories or ()),
                subject_scopes=tuple(subject_scopes or ()),
                importance_min=importance_min,
                time_hint=dict(time_hint or {}),
                kind_patterns=tuple(kind_patterns or ()),
                include_explicit=include_explicit,
                cross_conversation=cross_conversation,
                exclude_source_ids=tuple(exclude_source_ids or ()),
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
        date_from: str,
        date_to: str = "",
        time_periods: list[str] | None = None,
        cross_conversation: bool = False,
    ) -> dict[str, Any]:
        """时间线工具:按日期(范围)精确读原始对话,不走向量。与 retrieve 互补。

        date_to 省略时等于 date_from(单日)。time_periods 接受上午/下午/晚上/凌晨等别名。
        这是精确读工具:日期/时间段非法时**结构化报 invalid_filter**,绝不静默放宽成"查整天"。
        """
        from datetime import date

        from .rendering import render_timeline
        from .time_anchor import normalize_time_periods

        def _invalid(reason: str) -> dict[str, Any]:
            return {"status": "invalid_filter", "reason": reason, "messages": [], "message_count": 0, "text": ""}

        def _is_iso_date(value: str) -> bool:
            try:
                date.fromisoformat(value)
                return True
            except ValueError:
                return False

        start = str(date_from or "").strip()
        end = str(date_to or "").strip() or start
        if not _is_iso_date(start) or not _is_iso_date(end):
            return _invalid("date_must_be_YYYY-MM-DD")
        if start > end:
            return _invalid("date_from_after_date_to")

        raw_periods = [str(p).strip() for p in (time_periods or []) if str(p or "").strip()]
        unknown = [p for p in raw_periods if not normalize_time_periods([p])]
        if unknown:
            return _invalid(f"unknown_time_periods:{unknown}")
        periods = normalize_time_periods(raw_periods)
        messages = self.store.get_messages_by_date_range(
            namespace=self.namespace,
            date_from=start,
            date_to=end,
            time_periods=periods,
            cross_conversation=cross_conversation,
        )
        return {
            "status": "ok" if messages else "empty",
            "date_from": start,
            "date_to": end,
            "time_periods": periods,
            "message_count": len(messages),
            "messages": messages,
            "text": render_timeline(messages, tz=self.timezone),
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
