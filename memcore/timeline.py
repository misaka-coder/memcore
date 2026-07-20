"""Typed contracts for the Unified Timeline V2 turn lifecycle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from .errors import SchemaError
from .namespace import Actor, Namespace

if TYPE_CHECKING:
    from .projection import ProjectionMessage, ProjectionMessageInput

_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")
_ID_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EntryOrigin(_TextEnum):
    USER = "user"
    ASSISTANT = "assistant"
    ENVIRONMENT = "environment"


class TurnRole(_TextEnum):
    STIMULUS = "stimulus"
    INTERMEDIATE = "intermediate"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"


class RetrievalPolicy(_TextEnum):
    AUTO = "auto"
    ALWAYS = "always"
    EXPLICIT = "explicit"
    NEVER = "never"


class RetrievalVisibility(_TextEnum):
    DEFAULT = "default"
    EXPLICIT = "explicit"
    NEVER = "never"


class EntryTrust(_TextEnum):
    UNTRUSTED_DATA = "untrusted_data"
    TRUSTED_INSTRUCTION = "trusted_instruction"


class TurnStatus(_TextEnum):
    OPEN = "open"
    CLOSED = "closed"
    ABORTED = "aborted"


class AnnotationStatus(_TextEnum):
    UNANNOTATED = "unannotated"
    ACCEPTED_MODEL = "accepted_model"
    ACCEPTED_HOST = "accepted_host"
    ACCEPTED_LEGACY = "accepted_legacy"
    DERIVED_TURN_FINAL = "derived_turn_final"
    DERIVED = "derived"
    MISSING = "missing"
    INVALID = "invalid"
    PLAIN = "plain"
    FALLBACK = "fallback"
    REJECTED = "rejected"

    @property
    def accepted(self) -> bool:
        return self in {
            AnnotationStatus.ACCEPTED_MODEL,
            AnnotationStatus.ACCEPTED_HOST,
            AnnotationStatus.ACCEPTED_LEGACY,
        }


@dataclass(frozen=True)
class TimelineEntryInput:
    kind: str
    origin: EntryOrigin | str
    turn_role: TurnRole | str
    semantic_text: str
    timestamp: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_id: str = ""
    turn_id: str = ""
    reply_to_source_id: str = ""
    correlation_id: str = ""
    actor: Actor | None = None
    target_actor: Actor | None = None
    trace_metadata: Mapping[str, Any] = field(default_factory=dict)
    memory_metadata: Mapping[str, Any] = field(default_factory=dict)
    annotation_status: AnnotationStatus | str = AnnotationStatus.UNANNOTATED
    annotation_source: str = ""
    retrieval_policy: RetrievalPolicy | str = RetrievalPolicy.AUTO
    retrieval_visibility: RetrievalVisibility | str = RetrievalVisibility.EXPLICIT
    semanticize: bool = True
    prompt_visible: bool = True
    trust: EntryTrust | str = EntryTrust.UNTRUSTED_DATA
    renderer_id: str = "canonical"
    renderer_version: int = 1
    date_label: str = ""
    time_of_day: str = ""
    compatibility_role: str = ""

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip()
        if len(kind) > 160 or not _KIND_PATTERN.fullmatch(kind):
            raise SchemaError("timeline_entry_invalid_kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "origin", _coerce_enum(EntryOrigin, self.origin, "timeline_entry_invalid_origin"))
        object.__setattr__(
            self,
            "turn_role",
            _coerce_enum(TurnRole, self.turn_role, "timeline_entry_invalid_turn_role"),
        )
        object.__setattr__(
            self,
            "annotation_status",
            _coerce_enum(AnnotationStatus, self.annotation_status, "timeline_entry_invalid_annotation_status"),
        )
        object.__setattr__(
            self,
            "retrieval_policy",
            _coerce_enum(RetrievalPolicy, self.retrieval_policy, "timeline_entry_invalid_retrieval_policy"),
        )
        object.__setattr__(
            self,
            "retrieval_visibility",
            _coerce_enum(
                RetrievalVisibility,
                self.retrieval_visibility,
                "timeline_entry_invalid_retrieval_visibility",
            ),
        )
        object.__setattr__(self, "trust", _coerce_enum(EntryTrust, self.trust, "timeline_entry_invalid_trust"))
        object.__setattr__(self, "semantic_text", str(self.semantic_text or ""))
        object.__setattr__(self, "timestamp", max(0, int(self.timestamp or 0)))
        object.__setattr__(self, "payload", _json_object(self.payload, "timeline_entry_invalid_payload"))
        object.__setattr__(
            self,
            "trace_metadata",
            _json_object(self.trace_metadata, "timeline_entry_invalid_trace_metadata"),
        )
        object.__setattr__(
            self,
            "memory_metadata",
            _json_object(self.memory_metadata, "timeline_entry_invalid_memory_metadata"),
        )
        for name in ("source_id", "turn_id", "reply_to_source_id", "correlation_id"):
            value = str(getattr(self, name) or "").strip()
            if value and not _ID_PATTERN.fullmatch(value):
                raise SchemaError(f"timeline_entry_invalid_{name}")
            object.__setattr__(self, name, value)
        renderer_id = str(self.renderer_id or "").strip()
        if not renderer_id or len(renderer_id) > 120:
            raise SchemaError("timeline_entry_invalid_renderer_id")
        object.__setattr__(self, "renderer_id", renderer_id)
        if int(self.renderer_version) < 1:
            raise SchemaError("timeline_entry_invalid_renderer_version")
        object.__setattr__(self, "renderer_version", int(self.renderer_version))
        if self.turn_role in {TurnRole.ACTION, TurnRole.OBSERVATION} and not self.correlation_id:
            raise SchemaError("timeline_entry_correlation_id_required")

    def with_runtime_fields(
        self,
        *,
        turn_id: str,
        timestamp: int,
        date_label: str,
        time_of_day: str,
    ) -> "TimelineEntryInput":
        from dataclasses import replace

        return replace(
            self,
            turn_id=turn_id,
            timestamp=timestamp,
            date_label=date_label,
            time_of_day=time_of_day,
        )


@dataclass(frozen=True)
class TimelineEntry:
    source_id: str
    namespace: Namespace
    seq_no: int
    kind: str
    origin: EntryOrigin
    turn_role: TurnRole | None
    semantic_text: str
    timestamp: int
    payload: dict[str, Any]
    turn_id: str = ""
    reply_to_source_id: str = ""
    correlation_id: str = ""
    relation_status: str = ""
    target_actor: Actor | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    memory_metadata: dict[str, Any] = field(default_factory=dict)
    annotation_status: AnnotationStatus = AnnotationStatus.UNANNOTATED
    annotation_source: str = ""
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.AUTO
    retrieval_visibility: RetrievalVisibility = RetrievalVisibility.EXPLICIT
    semanticize: bool = True
    prompt_visible: bool = True
    trust: EntryTrust = EntryTrust.UNTRUSTED_DATA
    renderer_id: str = "canonical"
    renderer_version: int = 1
    row_version: int = 1
    index_status: str = "pending"
    compatibility_role: str = ""
    content: str = ""
    date_label: str = ""
    time_of_day: str = ""

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TimelineEntry":
        actor_id = str(record.get("actor_id") or "")
        target_actor_id = str(record.get("target_actor_id") or "")
        raw_turn_role = str(record.get("turn_role") or "")
        return cls(
            source_id=str(record.get("source_id") or ""),
            namespace=Namespace(
                tenant_id=str(record.get("tenant_id") or ""),
                user_id=str(record.get("user_id") or ""),
                domain_id=str(record.get("domain_id") or ""),
                conversation_id=str(record.get("conversation_id") or ""),
                actor=(
                    Actor(stable_id=actor_id, display_name=str(record.get("actor_display_name") or ""))
                    if actor_id
                    else None
                ),
            ),
            seq_no=int(record.get("seq_no") or 0),
            kind=str(record.get("kind") or "legacy.unknown"),
            origin=_coerce_enum(EntryOrigin, record.get("origin") or "environment", "timeline_entry_invalid_origin"),
            turn_role=(
                _coerce_enum(TurnRole, raw_turn_role, "timeline_entry_invalid_turn_role") if raw_turn_role else None
            ),
            semantic_text=str(record.get("semantic_text") or record.get("content") or ""),
            timestamp=int(record.get("timestamp") or 0),
            payload=dict(record.get("payload") or {}),
            turn_id=str(record.get("turn_id") or ""),
            reply_to_source_id=str(record.get("reply_to_source_id") or ""),
            correlation_id=str(record.get("correlation_id") or ""),
            relation_status=str(record.get("relation_status") or ""),
            target_actor=(
                Actor(
                    stable_id=target_actor_id,
                    display_name=str(record.get("target_actor_display_name") or ""),
                )
                if target_actor_id
                else None
            ),
            trace_metadata=dict(record.get("trace_metadata") or {}),
            memory_metadata=dict(record.get("memory_metadata") or {}),
            annotation_status=_coerce_enum(
                AnnotationStatus,
                record.get("annotation_status") or "unannotated",
                "timeline_entry_invalid_annotation_status",
            ),
            annotation_source=str(record.get("annotation_source") or ""),
            retrieval_policy=_coerce_enum(
                RetrievalPolicy,
                record.get("retrieval_policy") or "auto",
                "timeline_entry_invalid_retrieval_policy",
            ),
            retrieval_visibility=_coerce_enum(
                RetrievalVisibility,
                record.get("retrieval_visibility") or "explicit",
                "timeline_entry_invalid_retrieval_visibility",
            ),
            semanticize=bool(record.get("semanticize", 1)),
            prompt_visible=bool(record.get("prompt_visible", 1)),
            trust=_coerce_enum(
                EntryTrust,
                record.get("trust") or "untrusted_data",
                "timeline_entry_invalid_trust",
            ),
            renderer_id=str(record.get("renderer_id") or "canonical"),
            renderer_version=int(record.get("renderer_version") or 1),
            row_version=int(record.get("row_version") or 1),
            index_status=str(record.get("index_status") or "pending"),
            compatibility_role=str(record.get("role") or ""),
            content=str(record.get("content") or ""),
            date_label=str(record.get("date_label") or ""),
            time_of_day=str(record.get("time_of_day") or ""),
        )

    def to_record(self) -> dict[str, Any]:
        actor = self.namespace.actor
        return {
            "source_id": self.source_id,
            "tenant_id": self.namespace.tenant_id,
            "user_id": self.namespace.user_id,
            "domain_id": self.namespace.domain_id,
            "conversation_id": self.namespace.conversation_id,
            "actor_id": actor.stable_id if actor else "",
            "actor_display_name": actor.display_name if actor else "",
            "seq_no": self.seq_no,
            "kind": self.kind,
            "origin": self.origin.value,
            "turn_role": self.turn_role.value if self.turn_role else "",
            "semantic_text": self.semantic_text,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "turn_id": self.turn_id,
            "reply_to_source_id": self.reply_to_source_id,
            "correlation_id": self.correlation_id,
            "relation_status": self.relation_status,
            "target_actor_id": self.target_actor.stable_id if self.target_actor else "",
            "target_actor_display_name": self.target_actor.display_name if self.target_actor else "",
            "trace_metadata": dict(self.trace_metadata),
            "memory_metadata": dict(self.memory_metadata),
            "annotation_status": self.annotation_status.value,
            "annotation_source": self.annotation_source,
            "retrieval_policy": self.retrieval_policy.value,
            "retrieval_visibility": self.retrieval_visibility.value,
            "semanticize": int(self.semanticize),
            "prompt_visible": int(self.prompt_visible),
            "trust": self.trust.value,
            "renderer_id": self.renderer_id,
            "renderer_version": self.renderer_version,
            "row_version": self.row_version,
            "index_status": self.index_status,
            "role": self.compatibility_role,
            "content": self.content or self.semantic_text,
            "date_label": self.date_label,
            "time_of_day": self.time_of_day,
            "entry_type": "raw",
        }


@dataclass(frozen=True)
class MemoryAnnotation:
    target_source_id: str
    status: AnnotationStatus | str
    memory_metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "model"

    def __post_init__(self) -> None:
        target = str(self.target_source_id or "").strip()
        if not target or not _ID_PATTERN.fullmatch(target):
            raise SchemaError("memory_annotation_invalid_target_source_id")
        object.__setattr__(self, "target_source_id", target)
        status = _coerce_enum(AnnotationStatus, self.status, "memory_annotation_invalid_status")
        if status in {AnnotationStatus.DERIVED, AnnotationStatus.DERIVED_TURN_FINAL, AnnotationStatus.UNANNOTATED}:
            raise SchemaError("memory_annotation_status_not_terminal_input")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "memory_metadata",
            _json_object(self.memory_metadata, "memory_annotation_invalid_metadata"),
        )
        object.__setattr__(self, "source", str(self.source or "").strip()[:80])


@dataclass(frozen=True)
class TurnHandle:
    turn_id: str
    namespace: Namespace
    status: TurnStatus
    stimuli: tuple[TimelineEntry, ...]
    annotation_target_ids: tuple[str, ...]
    opened_at: int


@dataclass(frozen=True)
class TurnCompletion:
    turn_id: str
    semantic_text: str
    provider_output_raw: str
    annotations: tuple[MemoryAnnotation, ...]
    timestamp: int
    source_id: str = ""
    close_reason: str = "completed"
    kind: str = "message.assistant"
    payload: Mapping[str, Any] = field(default_factory=dict)
    trace_metadata: Mapping[str, Any] = field(default_factory=dict)
    final_projection: ProjectionMessageInput | None = None
    date_label: str = ""
    time_of_day: str = ""

    def __post_init__(self) -> None:
        turn_id = str(self.turn_id or "").strip()
        if not turn_id or not _ID_PATTERN.fullmatch(turn_id):
            raise SchemaError("turn_completion_invalid_turn_id")
        object.__setattr__(self, "turn_id", turn_id)
        source_id = str(self.source_id or "").strip()
        if source_id and not _ID_PATTERN.fullmatch(source_id):
            raise SchemaError("turn_completion_invalid_source_id")
        object.__setattr__(self, "source_id", source_id)
        semantic_text = str(self.semantic_text or "")
        provider_output_raw = str(self.provider_output_raw or "")
        if not semantic_text.strip() and not provider_output_raw.strip():
            raise SchemaError("turn_completion_empty_final")
        object.__setattr__(self, "semantic_text", semantic_text)
        object.__setattr__(self, "provider_output_raw", provider_output_raw)
        annotations = tuple(self.annotations or ())
        if any(not isinstance(item, MemoryAnnotation) for item in annotations):
            raise TypeError("TurnCompletion.annotations must contain MemoryAnnotation values")
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "timestamp", max(0, int(self.timestamp or 0)))
        object.__setattr__(self, "kind", _validated_kind(self.kind))
        object.__setattr__(self, "payload", _json_object(self.payload, "turn_completion_invalid_payload"))
        object.__setattr__(
            self,
            "trace_metadata",
            _json_object(self.trace_metadata, "turn_completion_invalid_trace_metadata"),
        )
        if self.final_projection is not None:
            from .projection import ProjectionMessageInput

            if not isinstance(self.final_projection, ProjectionMessageInput):
                raise TypeError("final_projection must be a ProjectionMessageInput or None")
            if str(self.final_projection.payload.get("role") or "").strip().lower() != "assistant":
                raise SchemaError("turn_completion_final_projection_must_be_assistant")
        object.__setattr__(self, "close_reason", str(self.close_reason or "completed").strip()[:160])


@dataclass(frozen=True)
class CompletionCommitResult:
    status: str
    turn_id: str
    final_entry: TimelineEntry | None = None
    updated_targets: tuple[TimelineEntry, ...] = ()
    pending_correlations: tuple[str, ...] = ()
    final_projection: ProjectionMessage | None = None
    reason: str = ""

    @property
    def completed(self) -> bool:
        return self.status in {"completed", "already_completed"}


@dataclass(frozen=True)
class TurnAbortResult:
    status: str
    turn_id: str
    reason: str = ""


def resolve_retrieval_visibility(
    policy: RetrievalPolicy | str,
    annotation_status: AnnotationStatus | str,
) -> RetrievalVisibility:
    resolved_policy = _coerce_enum(RetrievalPolicy, policy, "timeline_entry_invalid_retrieval_policy")
    resolved_status = _coerce_enum(
        AnnotationStatus,
        annotation_status,
        "timeline_entry_invalid_annotation_status",
    )
    if resolved_policy is RetrievalPolicy.ALWAYS:
        return RetrievalVisibility.DEFAULT
    if resolved_policy is RetrievalPolicy.EXPLICIT:
        return RetrievalVisibility.EXPLICIT
    if resolved_policy is RetrievalPolicy.NEVER:
        return RetrievalVisibility.NEVER
    return RetrievalVisibility.DEFAULT if resolved_status.accepted else RetrievalVisibility.EXPLICIT


def _validated_kind(value: Any) -> str:
    kind = str(value or "").strip()
    if len(kind) > 160 or not _KIND_PATTERN.fullmatch(kind):
        raise SchemaError("timeline_entry_invalid_kind")
    return kind


def _coerce_enum(enum_type: type[_TextEnum], value: Any, reason: str) -> Any:
    try:
        return enum_type(str(value.value if isinstance(value, Enum) else value))
    except (TypeError, ValueError) as exc:
        raise SchemaError(reason) from exc


def _json_object(value: Any, reason: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(reason)
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError(reason) from exc
    return result


__all__ = [
    "AnnotationStatus",
    "CompletionCommitResult",
    "EntryOrigin",
    "EntryTrust",
    "MemoryAnnotation",
    "RetrievalPolicy",
    "RetrievalVisibility",
    "TimelineEntry",
    "TimelineEntryInput",
    "TurnAbortResult",
    "TurnCompletion",
    "TurnHandle",
    "TurnRole",
    "TurnStatus",
    "resolve_retrieval_visibility",
]
