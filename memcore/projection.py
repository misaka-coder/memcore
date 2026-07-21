"""Deterministic Timeline V2 rendering, provider projection, and cache hashes.

SQLite timeline entries remain the truth source.  This module only turns those
entries into safe provider-visible messages and typed ledger records; it never
sends a model request.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import SchemaError
from .namespace import Namespace
from .rendering import render_semantic_snippet, render_summary_snippet
from .text_utils import normalize_text
from .time_anchor import TIME_PERIOD_LABELS, timestamp_to_datetime_weekday_label
from .timeline import TimelineEntry, TimelineEntryInput, TurnRole

PROJECTION_VERSION = 1
CANONICAL_PROFILE = "canonical_user_assistant"
OPENAI_PROFILE = "openai_chat"
ANTHROPIC_PROFILE = "anthropic_messages"
STANDARD_PROJECTION_PROFILES = frozenset({CANONICAL_PROFILE, OPENAI_PROFILE, ANTHROPIC_PROFILE})

_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")
_KIND_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*$")
_PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")
_ID_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET_FIELD = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|password|passwd|secret)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|\bsk-[A-Za-z0-9_-]{12,}|\b(?:api[_-]?key|token|secret)\s*[:=]\s*\S{8,})",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?:^|[\s'\"])[A-Za-z]:[\\/][^\r\n]+")
_UNC_PATH = re.compile(r"(?:^|[\s'\"])[\\/]{2}[^\s\\/]+[\\/][^\r\n]+")
_POSIX_PRIVATE_PATH = re.compile(r"(?:^|[\s'\"])/(?:home|Users|root|opt|var|tmp|etc)/[^\r\n]+")
_LOCAL_PATH_FIELD = re.compile(
    r"(?:^|[_-])(?:absolute[_-]?path|cached[_-]?path|storage[_-]?relpath|local[_-]?path|file[_-]?path|path)(?:$|[_-])",
    re.IGNORECASE,
)
_MEDIA_TYPES = frozenset(
    {
        "audio",
        "document",
        "file",
        "image",
        "image_url",
        "input_audio",
        "input_file",
        "input_image",
        "video",
    }
)
_MEDIA_MARKER = "[media omitted from persistent history]"
_SECRET_MARKER = "[secret omitted from persistent history]"
_PATH_MARKER = "[local path omitted from persistent history]"


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ProjectionStatus(_TextEnum):
    COMPLETE = "complete"
    CANONICAL_FALLBACK = "canonical_fallback"
    REQUEST_FROZEN = "request_frozen"
    MEDIA_OMITTED = "media_omitted"
    SKIPPED_UNSAFE = "skipped_unsafe"


_STATUS_PRIORITY = {
    ProjectionStatus.COMPLETE: 0,
    ProjectionStatus.CANONICAL_FALLBACK: 1,
    ProjectionStatus.MEDIA_OMITTED: 2,
    ProjectionStatus.SKIPPED_UNSAFE: 3,
    # Request freezing is a lifecycle boundary: once a safe payload has crossed
    # it, later attempts must not replace that projection.  Keep it above the
    # rendering/safety statuses; media omission is still carried by the sanitized
    # payload marker and the request audit's ``media_omitted`` flag.
    ProjectionStatus.REQUEST_FROZEN: 4,
}


def _coerce_status(value: ProjectionStatus | str) -> ProjectionStatus:
    try:
        return value if isinstance(value, ProjectionStatus) else ProjectionStatus(str(value or ""))
    except ValueError as exc:
        raise SchemaError("projection_invalid_status") from exc


def merge_projection_status(*values: ProjectionStatus | str) -> ProjectionStatus:
    statuses = tuple(_coerce_status(value) for value in values)
    return max(statuses or (ProjectionStatus.COMPLETE,), key=_STATUS_PRIORITY.__getitem__)


def _validated_profile(value: Any) -> str:
    profile = str(value or "").strip().lower()
    if not _PROFILE_PATTERN.fullmatch(profile):
        raise SchemaError("projection_invalid_provider_profile")
    return profile


def normalize_provider_profile(value: Any) -> str:
    return _validated_profile(value)


def _validated_source_ids(values: Iterable[Any]) -> tuple[str, ...]:
    source_ids = tuple(str(value or "").strip() for value in values)
    if any(not source_id or not _ID_PATTERN.fullmatch(source_id) for source_id in source_ids):
        raise SchemaError("projection_invalid_source_id")
    if len(set(source_ids)) != len(source_ids):
        raise SchemaError("projection_duplicate_source_id")
    return source_ids


def _json_ready(value: Any) -> Any:
    """Normalize arbitrary request fragments for hashing without retaining raw bytes."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bytes):
        return {"binary_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, bytearray):
        data = bytes(value)
        return {"binary_sha256": hashlib.sha256(data).hexdigest(), "length": len(data)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return str(value)
        return value
    return {"unsupported_type": type(value).__name__}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_projection_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _looks_like_local_path(value: str) -> bool:
    text = str(value or "")
    return bool(
        text.lower().startswith("file://")
        or _WINDOWS_ABSOLUTE_PATH.search(text)
        or _UNC_PATH.search(text)
        or _POSIX_PRIVATE_PATH.search(text)
    )


def _media_marker_block() -> dict[str, str]:
    return {"type": "text", "text": _MEDIA_MARKER}


def _sanitize_value(value: Any, *, key: str = "") -> tuple[Any, ProjectionStatus]:
    normalized_key = str(key or "")
    if _SECRET_FIELD.search(normalized_key):
        return _SECRET_MARKER, ProjectionStatus.SKIPPED_UNSAFE
    if _LOCAL_PATH_FIELD.search(normalized_key):
        return _PATH_MARKER, ProjectionStatus.SKIPPED_UNSAFE
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _MEDIA_MARKER, ProjectionStatus.MEDIA_OMITTED
    if isinstance(value, Mapping):
        raw_type = str(value.get("type") or "").strip().lower()
        if raw_type in _MEDIA_TYPES or "image_url" in value:
            return _media_marker_block(), ProjectionStatus.MEDIA_OMITTED
        result: dict[str, Any] = {}
        status = ProjectionStatus.COMPLETE
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            child_key = str(raw_key)
            clean, child_status = _sanitize_value(item, key=child_key)
            result[child_key] = clean
            status = merge_projection_status(status, child_status)
        return result, status
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        status = ProjectionStatus.COMPLETE
        for item in value:
            clean, child_status = _sanitize_value(item)
            items.append(clean)
            status = merge_projection_status(status, child_status)
        return items, status
    if isinstance(value, str):
        if value.lower().startswith("data:") and ";base64," in value[:160].lower():
            return _MEDIA_MARKER, ProjectionStatus.MEDIA_OMITTED
        if _SECRET_VALUE.search(value):
            return _SECRET_MARKER, ProjectionStatus.SKIPPED_UNSAFE
        if _looks_like_local_path(value):
            return _PATH_MARKER, ProjectionStatus.SKIPPED_UNSAFE
        return _CONTROL_CHARS.sub(" ", value), ProjectionStatus.COMPLETE
    if value is None or isinstance(value, (bool, int, float)):
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return str(value), ProjectionStatus.SKIPPED_UNSAFE
        return value, ProjectionStatus.COMPLETE
    return f"[unsupported {type(value).__name__} omitted]", ProjectionStatus.SKIPPED_UNSAFE


def sanitize_projection_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ProjectionStatus]:
    if not isinstance(payload, Mapping):
        raise SchemaError("projection_payload_must_be_object")
    role = str(payload.get("role") or "").strip().lower()
    if role in {"system", "developer"}:
        raise SchemaError("projection_system_message_not_persistable")
    if not role:
        raise SchemaError("projection_message_role_required")
    sanitized, status = _sanitize_value(dict(payload))
    if not isinstance(sanitized, dict):
        raise SchemaError("projection_payload_must_be_object")
    return sanitized, status


@dataclass(frozen=True)
class ProjectionMessageInput:
    provider_profile: str
    payload: Mapping[str, Any]
    source_ids: tuple[str, ...] = ()
    projection_index: int = -1
    projection_status: ProjectionStatus | str = ProjectionStatus.COMPLETE
    projection_version: int = PROJECTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_profile", _validated_profile(self.provider_profile))
        object.__setattr__(self, "source_ids", _validated_source_ids(self.source_ids))
        index = int(self.projection_index)
        if index < -1:
            raise SchemaError("projection_invalid_index")
        object.__setattr__(self, "projection_index", index)
        version = int(self.projection_version)
        if version < 1:
            raise SchemaError("projection_invalid_version")
        object.__setattr__(self, "projection_version", version)
        payload, safety_status = sanitize_projection_payload(self.payload)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "projection_status",
            merge_projection_status(self.projection_status, safety_status),
        )


@dataclass(frozen=True)
class ProjectionMessage:
    projection_id: str
    namespace_key: tuple[str, str, str, str]
    turn_id: str
    projection_index: int
    provider_profile: str
    payload: dict[str, Any]
    source_ids: tuple[str, ...]
    payload_hash: str
    projection_status: ProjectionStatus
    projection_version: int
    created_at: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProjectionMessage":
        payload = dict(record.get("payload") or {})
        payload_hash = str(record.get("payload_hash") or "")
        if payload_hash != stable_projection_hash(payload):
            raise SchemaError("projection_payload_hash_mismatch")
        return cls(
            projection_id=str(record.get("projection_id") or ""),
            namespace_key=(
                str(record.get("tenant_id") or ""),
                str(record.get("user_id") or ""),
                str(record.get("domain_id") or ""),
                str(record.get("conversation_id") or ""),
            ),
            turn_id=str(record.get("turn_id") or ""),
            projection_index=int(record.get("projection_index") or 0),
            provider_profile=_validated_profile(record.get("provider_profile")),
            payload=payload,
            source_ids=_validated_source_ids(record.get("source_ids") or ()),
            payload_hash=payload_hash,
            projection_status=_coerce_status(record.get("projection_status") or "complete"),
            projection_version=int(record.get("projection_version") or 1),
            created_at=int(record.get("created_at") or 0),
        )


@dataclass(frozen=True)
class ProjectionAuditInput:
    turn_id: str
    attempt: int
    provider_profile: str
    model_route_hash: str
    system_prefix_hash: str
    tool_schema_hash: str
    history_hash: str
    full_prefix_hash: str
    projection_version: int = PROJECTION_VERSION
    media_omitted: bool = False
    created_at: int = 0

    def __post_init__(self) -> None:
        turn_id = str(self.turn_id or "").strip()
        if not turn_id or not _ID_PATTERN.fullmatch(turn_id):
            raise SchemaError("projection_audit_invalid_turn_id")
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "provider_profile", _validated_profile(self.provider_profile))
        if int(self.attempt) < 1:
            raise SchemaError("projection_audit_invalid_attempt")
        object.__setattr__(self, "attempt", int(self.attempt))
        for field_name in (
            "model_route_hash",
            "system_prefix_hash",
            "tool_schema_hash",
            "history_hash",
            "full_prefix_hash",
        ):
            value = str(getattr(self, field_name) or "")
            if not re.fullmatch(r"[a-f0-9]{64}", value):
                raise SchemaError(f"projection_audit_invalid_{field_name}")
            object.__setattr__(self, field_name, value)
        if int(self.projection_version) < 1:
            raise SchemaError("projection_invalid_version")
        object.__setattr__(self, "projection_version", int(self.projection_version))
        object.__setattr__(self, "created_at", max(0, int(self.created_at or 0)))


@dataclass(frozen=True)
class ProjectionAudit:
    namespace_key: tuple[str, str, str, str]
    turn_id: str
    attempt: int
    provider_profile: str
    model_route_hash: str
    system_prefix_hash: str
    tool_schema_hash: str
    history_hash: str
    full_prefix_hash: str
    projection_version: int
    media_omitted: bool
    created_at: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProjectionAudit":
        return cls(
            namespace_key=(
                str(record.get("tenant_id") or ""),
                str(record.get("user_id") or ""),
                str(record.get("domain_id") or ""),
                str(record.get("conversation_id") or ""),
            ),
            turn_id=str(record.get("turn_id") or ""),
            attempt=int(record.get("attempt") or 0),
            provider_profile=_validated_profile(record.get("provider_profile")),
            model_route_hash=str(record.get("model_route_hash") or ""),
            system_prefix_hash=str(record.get("system_prefix_hash") or ""),
            tool_schema_hash=str(record.get("tool_schema_hash") or ""),
            history_hash=str(record.get("history_hash") or ""),
            full_prefix_hash=str(record.get("full_prefix_hash") or ""),
            projection_version=int(record.get("projection_version") or 1),
            media_omitted=bool(record.get("media_omitted") or 0),
            created_at=int(record.get("created_at") or 0),
        )


@dataclass(frozen=True)
class RequestProjectionResult:
    projections: tuple[ProjectionMessage, ...]
    audit: ProjectionAudit


@dataclass(frozen=True)
class EntryProjectionHash:
    source_id: str
    payload_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ContextProjection:
    provider_profile: str
    messages: tuple[ProjectionMessage, ...]
    projection_version: int
    stable_prefix_hash: str
    entry_projection_hashes: tuple[EntryProjectionHash, ...]
    compaction_generation: int
    projection_generation: int

    @property
    def payloads(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(message.payload) for message in self.messages)


@dataclass(frozen=True)
class RendererResult:
    text: str
    renderer_id: str
    renderer_version: int
    projection_status: ProjectionStatus = ProjectionStatus.COMPLETE


Renderer = Callable[[TimelineEntry, str], str]


@dataclass(frozen=True)
class _RendererSpec:
    renderer_id: str
    version: int
    renderer: Renderer


class RendererRegistry:
    """Versioned exact/prefix registry with canonical fallback for unknown kinds."""

    def __init__(self) -> None:
        self._exact: dict[str, _RendererSpec] = {}
        self._prefix: dict[str, _RendererSpec] = {}
        self._by_identity: dict[tuple[str, int], _RendererSpec] = {
            ("canonical", 1): _RendererSpec("canonical", 1, _render_canonical_entry)
        }

    def register_exact(
        self,
        kind: str,
        *,
        renderer_id: str,
        version: int,
        renderer: Renderer,
    ) -> None:
        normalized = str(kind or "").strip()
        if not _KIND_PATTERN.fullmatch(normalized):
            raise SchemaError("renderer_invalid_exact_kind")
        self._register(self._exact, normalized, renderer_id=renderer_id, version=version, renderer=renderer)

    def register_prefix(
        self,
        kind_prefix: str,
        *,
        renderer_id: str,
        version: int,
        renderer: Renderer,
    ) -> None:
        normalized = str(kind_prefix or "").strip().rstrip(".*")
        if not _KIND_PREFIX_PATTERN.fullmatch(normalized):
            raise SchemaError("renderer_invalid_kind_prefix")
        self._register(self._prefix, normalized, renderer_id=renderer_id, version=version, renderer=renderer)

    def _register(
        self,
        target: dict[str, _RendererSpec],
        selector: str,
        *,
        renderer_id: str,
        version: int,
        renderer: Renderer,
    ) -> None:
        normalized_id = str(renderer_id or "").strip()
        if not normalized_id or len(normalized_id) > 120 or not callable(renderer):
            raise SchemaError("renderer_invalid_registration")
        resolved_version = int(version)
        if resolved_version < 1:
            raise SchemaError("renderer_invalid_version")
        spec = _RendererSpec(normalized_id, resolved_version, renderer)
        identity = (normalized_id, resolved_version)
        existing = self._by_identity.get(identity)
        if existing is not None and existing.renderer is not renderer:
            raise SchemaError("renderer_identity_conflict")
        self._by_identity[identity] = spec
        target[selector] = spec

    def select(self, kind: str) -> tuple[str, int]:
        normalized = str(kind or "").strip()
        exact = self._exact.get(normalized)
        if exact is not None:
            return exact.renderer_id, exact.version
        matches = [
            (prefix, spec)
            for prefix, spec in self._prefix.items()
            if normalized == prefix or normalized.startswith(prefix + ".")
        ]
        if matches:
            _, spec = max(matches, key=lambda item: (item[0].count("."), len(item[0])))
            return spec.renderer_id, spec.version
        return "canonical", 1

    def bind_input(self, entry: TimelineEntryInput) -> TimelineEntryInput:
        if entry.renderer_id != "canonical" or entry.renderer_version != 1:
            return entry
        renderer_id, version = self.select(entry.kind)
        return replace(entry, renderer_id=renderer_id, renderer_version=version)

    def render(self, entry: TimelineEntry, *, timezone: str) -> RendererResult:
        spec = self._by_identity.get((entry.renderer_id, entry.renderer_version))
        status = ProjectionStatus.COMPLETE
        if spec is None:
            spec = self._by_identity[("canonical", 1)]
            status = ProjectionStatus.CANONICAL_FALLBACK
        text = str(spec.renderer(entry, timezone) or "")
        clean, safety_status = _sanitize_value(text)
        return RendererResult(
            text=str(clean),
            renderer_id=spec.renderer_id,
            renderer_version=spec.version,
            projection_status=merge_projection_status(status, safety_status),
        )


def default_renderer_registry() -> RendererRegistry:
    registry = RendererRegistry()
    registry.register_prefix(
        "event.finance",
        renderer_id="event.finance",
        version=1,
        renderer=_render_finance_event,
    )
    return registry


def _display_scalar(value: Any) -> str:
    text = normalize_text(value)
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def _display_multiline(value: Any) -> str:
    text = normalize_text(value).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub(" ", text)


def _entry_header(entry: TimelineEntry, timezone: str) -> str:
    stamp = timestamp_to_datetime_weekday_label(entry.timestamp, timezone) if entry.timestamp > 0 else ""
    period = TIME_PERIOD_LABELS.get(str(entry.time_of_day or ""), "")
    anchor = " | ".join(part for part in (stamp, period) if part)
    return f"[{anchor}] {entry.kind}" if anchor else entry.kind


def _actor_line(label: str, actor: Any) -> str:
    if actor is None:
        return ""
    stable_id = _display_scalar(getattr(actor, "stable_id", ""))
    display_name = _display_scalar(getattr(actor, "display_name", ""))
    if display_name and stable_id and display_name != stable_id:
        return f"{label}: {display_name} (id={stable_id})"
    value = display_name or stable_id
    return f"{label}: {value}" if value else ""


def _render_canonical_entry(entry: TimelineEntry, timezone: str) -> str:
    lines = [_entry_header(entry, timezone)]
    for actor_line in (
        _actor_line("actor", entry.namespace.actor),
        _actor_line("target_actor", entry.target_actor),
    ):
        if actor_line:
            lines.append(actor_line)
    if entry.correlation_id:
        lines.append(f"correlation_id: {_display_scalar(entry.correlation_id)}")
    if entry.trust.value != "untrusted_data":
        lines.append(f"trust: {entry.trust.value}")
    semantic_text = _display_multiline(entry.semantic_text)
    if semantic_text:
        lines.extend(("content:", semantic_text))
    payload = dict(entry.payload)
    if str(payload.get("text") or "") == entry.semantic_text:
        payload.pop("text", None)
    if payload:
        safe_payload, _ = _sanitize_value(payload)
        lines.extend(("data:", json.dumps(safe_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))))
    return "\n".join(lines)


_FINANCE_ORDER = ("source", "published_at", "title", "summary", "url")


def _render_finance_event(entry: TimelineEntry, timezone: str) -> str:
    lines = [_entry_header(entry, timezone)]
    actor_line = _actor_line("actor", entry.namespace.actor)
    if actor_line:
        lines.append(actor_line)
    payload = dict(entry.payload)
    emitted: set[str] = set()
    for key in _FINANCE_ORDER:
        value = _display_scalar(payload.get(key))
        if value:
            lines.append(f"{key}: {value}")
            emitted.add(key)
    for key in sorted(str(item) for item in payload if str(item) not in emitted and str(item) != "text"):
        value = _display_scalar(payload.get(key))
        if value:
            lines.append(f"{key}: {value}")
    if entry.semantic_text and not any(key in emitted for key in ("summary", "title")):
        lines.append(f"summary: {_display_scalar(entry.semantic_text)}")
    return "\n".join(lines)


class ProjectionAdapter:
    """Project ordered entries into one of the standard provider cache families."""

    def __init__(self, *, renderer_registry: RendererRegistry, timezone: str) -> None:
        if not isinstance(renderer_registry, RendererRegistry):
            raise TypeError("renderer_registry must be a RendererRegistry")
        self.renderer_registry = renderer_registry
        self.timezone = str(timezone or "").strip()

    def project_entries(
        self,
        entries: Sequence[TimelineEntry],
        *,
        provider_profile: str,
        start_index: int = 0,
    ) -> tuple[ProjectionMessageInput, ...]:
        profile = _validated_profile(provider_profile)
        if profile not in STANDARD_PROJECTION_PROFILES:
            raise SchemaError("projection_profile_unsupported")
        visible = [entry for entry in entries if entry.prompt_visible]
        if profile == OPENAI_PROFILE:
            raw = self._project_openai(visible)
        elif profile == ANTHROPIC_PROFILE:
            raw = self._project_anthropic(visible)
        else:
            raw = self._project_canonical(visible)
        return tuple(replace(message, projection_index=start_index + index) for index, message in enumerate(raw))

    def project_memory_record(
        self,
        record: Mapping[str, Any],
        *,
        provider_profile: str,
        source_id: str,
        projection_index: int = 0,
        enable_flavor: bool = False,
    ) -> ProjectionMessageInput:
        profile = _validated_profile(provider_profile)
        if profile not in STANDARD_PROJECTION_PROFILES:
            raise SchemaError("projection_profile_unsupported")
        kind = str(record.get("kind") or "")
        if kind == "memory.semantic_summary":
            text = render_semantic_snippet(dict(record), tz=self.timezone, enable_flavor=enable_flavor)
        elif kind == "memory.operation_digest":
            text = self._render_operation_digest(record)
        else:
            text = render_summary_snippet(dict(record), tz=self.timezone, enable_flavor=enable_flavor)
        if profile == ANTHROPIC_PROFILE:
            payload: dict[str, Any] = {"role": "user", "content": [{"type": "text", "text": text}]}
        else:
            payload = {"role": "user", "content": text}
        return ProjectionMessageInput(
            provider_profile=profile,
            payload=payload,
            source_ids=(source_id,),
            projection_index=projection_index,
            projection_status=ProjectionStatus.CANONICAL_FALLBACK,
        )

    @staticmethod
    def _render_operation_digest(record: Mapping[str, Any]) -> str:
        lines = ["【运行摘要】", str(record.get("diary_summary") or "").strip()]
        if record.get("core_facts"):
            lines.append("运行事实:" + ";".join(str(item) for item in record.get("core_facts") or ()))
        return "\n".join(line for line in lines if line)

    def _render(self, entry: TimelineEntry) -> RendererResult:
        return self.renderer_registry.render(entry, timezone=self.timezone)

    @staticmethod
    def _assistant_final_text(entry: TimelineEntry) -> str:
        raw = entry.payload.get("provider_output_raw") if isinstance(entry.payload, dict) else ""
        return str(raw if raw is not None and str(raw) else entry.semantic_text)

    @staticmethod
    def _assistant_intermediate_text(entry: TimelineEntry) -> str:
        return str(entry.semantic_text or "")

    @staticmethod
    def _tool_name(entry: TimelineEntry) -> str:
        trace_name = str(entry.trace_metadata.get("tool_name") or "").strip()
        if trace_name:
            return re.sub(r"[^A-Za-z0-9_-]+", "_", trace_name)[:64] or "tool"
        name = entry.kind.removeprefix("tool.")
        for suffix in (".call", ".result"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:64] or "tool"

    @staticmethod
    def _tool_input(entry: TimelineEntry) -> Any:
        payload = dict(entry.payload)
        if "input" in payload:
            return payload["input"]
        if "arguments" in payload:
            return payload["arguments"]
        return payload

    def _tool_result_content(self, entry: TimelineEntry) -> str:
        payload = dict(entry.payload)
        output = payload.get("output")
        if isinstance(output, str):
            return output
        return self._render(entry).text

    @staticmethod
    def _tool_result_is_error(entry: TimelineEntry) -> bool:
        return str(entry.trace_metadata.get("status") or "").strip().lower() in {
            "cancelled",
            "canceled",
            "denied",
            "error",
            "failed",
            "failure",
            "rejected",
        }

    def _project_canonical(self, entries: Sequence[TimelineEntry]) -> list[ProjectionMessageInput]:
        messages: list[ProjectionMessageInput] = []
        for entry in entries:
            rendered = self._render(entry)
            role = "assistant" if entry.origin.value == "assistant" else "user"
            content = self._assistant_final_text(entry) if entry.turn_role is TurnRole.FINAL else rendered.text
            messages.append(
                ProjectionMessageInput(
                    provider_profile=CANONICAL_PROFILE,
                    payload={"role": role, "content": content},
                    source_ids=(entry.source_id,),
                    projection_status=merge_projection_status(
                        ProjectionStatus.CANONICAL_FALLBACK,
                        rendered.projection_status,
                    ),
                )
            )
        return messages

    def _project_openai(self, entries: Sequence[TimelineEntry]) -> list[ProjectionMessageInput]:
        messages: list[ProjectionMessageInput] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            intermediate: TimelineEntry | None = None
            if (
                entry.turn_role is TurnRole.INTERMEDIATE
                and entry.origin.value == "assistant"
                and index + 1 < len(entries)
                and entries[index + 1].turn_role is TurnRole.ACTION
            ):
                intermediate = entry
                index += 1
                entry = entries[index]
            if entry.turn_role is TurnRole.ACTION:
                batch: list[TimelineEntry] = []
                while index < len(entries) and entries[index].turn_role is TurnRole.ACTION:
                    batch.append(entries[index])
                    index += 1
                tool_calls = [
                    {
                        "id": item.correlation_id,
                        "type": "function",
                        "function": {
                            "name": self._tool_name(item),
                            "arguments": json.dumps(
                                self._tool_input(item),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for item in batch
                ]
                messages.append(
                    ProjectionMessageInput(
                        provider_profile=OPENAI_PROFILE,
                        payload={
                            "role": "assistant",
                            "content": self._assistant_intermediate_text(intermediate)
                            if intermediate is not None
                            else None,
                            "tool_calls": tool_calls,
                        },
                        source_ids=(
                            *((intermediate.source_id,) if intermediate is not None else ()),
                            *(item.source_id for item in batch),
                        ),
                        projection_status=ProjectionStatus.CANONICAL_FALLBACK,
                    )
                )
                continue
            rendered = self._render(entry)
            if entry.turn_role is TurnRole.OBSERVATION:
                payload = {
                    "role": "tool",
                    "tool_call_id": entry.correlation_id,
                    "content": self._tool_result_content(entry),
                }
            elif entry.origin.value == "assistant":
                payload = {
                    "role": "assistant",
                    "content": self._assistant_final_text(entry)
                    if entry.turn_role is TurnRole.FINAL
                    else rendered.text,
                }
            else:
                payload = {"role": "user", "content": rendered.text}
            messages.append(
                ProjectionMessageInput(
                    provider_profile=OPENAI_PROFILE,
                    payload=payload,
                    source_ids=(entry.source_id,),
                    projection_status=merge_projection_status(
                        ProjectionStatus.CANONICAL_FALLBACK,
                        rendered.projection_status,
                    ),
                )
            )
            index += 1
        return messages

    def _project_anthropic(self, entries: Sequence[TimelineEntry]) -> list[ProjectionMessageInput]:
        messages: list[ProjectionMessageInput] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            intermediate: TimelineEntry | None = None
            if (
                entry.turn_role is TurnRole.INTERMEDIATE
                and entry.origin.value == "assistant"
                and index + 1 < len(entries)
                and entries[index + 1].turn_role is TurnRole.ACTION
            ):
                intermediate = entry
                index += 1
                entry = entries[index]
            if entry.turn_role is TurnRole.ACTION:
                batch: list[TimelineEntry] = []
                while index < len(entries) and entries[index].turn_role is TurnRole.ACTION:
                    batch.append(entries[index])
                    index += 1
                content = (
                    [{"type": "text", "text": self._assistant_intermediate_text(intermediate)}] if intermediate else []
                ) + [
                    {
                        "type": "tool_use",
                        "id": item.correlation_id,
                        "name": self._tool_name(item),
                        "input": self._tool_input(item),
                    }
                    for item in batch
                ]
                messages.append(
                    ProjectionMessageInput(
                        provider_profile=ANTHROPIC_PROFILE,
                        payload={"role": "assistant", "content": content},
                        source_ids=(
                            *((intermediate.source_id,) if intermediate is not None else ()),
                            *(item.source_id for item in batch),
                        ),
                        projection_status=ProjectionStatus.CANONICAL_FALLBACK,
                    )
                )
                continue
            if entry.turn_role is TurnRole.OBSERVATION:
                batch = []
                while index < len(entries) and entries[index].turn_role is TurnRole.OBSERVATION:
                    batch.append(entries[index])
                    index += 1
                content = []
                for item in batch:
                    result_block = {
                        "type": "tool_result",
                        "tool_use_id": item.correlation_id,
                        "content": self._tool_result_content(item),
                    }
                    if self._tool_result_is_error(item):
                        result_block["is_error"] = True
                    content.append(result_block)
                messages.append(
                    ProjectionMessageInput(
                        provider_profile=ANTHROPIC_PROFILE,
                        payload={"role": "user", "content": content},
                        source_ids=tuple(item.source_id for item in batch),
                        projection_status=ProjectionStatus.CANONICAL_FALLBACK,
                    )
                )
                continue
            rendered = self._render(entry)
            text = self._assistant_final_text(entry) if entry.turn_role is TurnRole.FINAL else rendered.text
            role = "assistant" if entry.origin.value == "assistant" else "user"
            messages.append(
                ProjectionMessageInput(
                    provider_profile=ANTHROPIC_PROFILE,
                    payload={"role": role, "content": [{"type": "text", "text": text}]},
                    source_ids=(entry.source_id,),
                    projection_status=merge_projection_status(
                        ProjectionStatus.CANONICAL_FALLBACK,
                        rendered.projection_status,
                    ),
                )
            )
            index += 1
        return messages


class ProjectionLedger:
    """Single orchestration path for freezing raw turns and derived memory records."""

    def __init__(self, *, store: Any, adapter: ProjectionAdapter, enable_flavor: bool = False) -> None:
        self.store = store
        self.adapter = adapter
        self.enable_flavor = bool(enable_flavor)

    def freeze_turn_entries(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        entries: Sequence[TimelineEntry],
        provider_profile: str,
    ) -> list[ProjectionMessage]:
        profile = _validated_profile(provider_profile)
        saved = self.store.get_turn_projections(
            namespace=namespace,
            turn_id=turn_id,
            provider_profile=profile,
        )
        visible_entries = [entry for entry in entries if entry.prompt_visible]
        covered = tuple(source_id for message in saved for source_id in message.source_ids)
        expected = tuple(entry.source_id for entry in visible_entries)
        if covered != expected[: len(covered)]:
            raise SchemaError("projection_history_not_append_only")
        remaining = visible_entries[len(covered) :]
        if remaining:
            start_index = max((message.projection_index for message in saved), default=-1) + 1
            generated = list(
                self.adapter.project_entries(
                    remaining,
                    provider_profile=profile,
                    start_index=start_index,
                )
            )
            self.store.save_turn_projections(
                namespace=namespace,
                turn_id=turn_id,
                projections=generated,
            )
            saved = self.store.get_turn_projections(
                namespace=namespace,
                turn_id=turn_id,
                provider_profile=profile,
            )
        return saved

    def freeze_memory_record(
        self,
        *,
        namespace: Namespace,
        record: Mapping[str, Any],
        id_key: str,
        turn_prefix: str,
        provider_profile: str,
    ) -> list[ProjectionMessage]:
        profile = _validated_profile(provider_profile)
        source_id = str(record.get(id_key) or "").strip()
        if not source_id:
            raise SchemaError("projection_memory_source_id_required")
        turn_id = self.memory_turn_id(source_id=source_id, id_key=id_key, turn_prefix=turn_prefix)
        saved = self.store.get_turn_projections(
            namespace=namespace,
            turn_id=turn_id,
            provider_profile=profile,
        )
        if saved:
            return saved
        generated = self.adapter.project_memory_record(
            record,
            provider_profile=profile,
            source_id=source_id,
            enable_flavor=self.enable_flavor,
        )
        self.store.save_turn_projections(
            namespace=namespace,
            turn_id=turn_id,
            projections=[generated],
        )
        return self.store.get_turn_projections(
            namespace=namespace,
            turn_id=turn_id,
            provider_profile=profile,
        )

    @staticmethod
    def memory_turn_id(*, source_id: str, id_key: str, turn_prefix: str) -> str:
        return f"{turn_prefix}.{stable_projection_hash({id_key: source_id})}"


def build_projection_audit_input(
    *,
    turn_id: str,
    attempt: int,
    provider_profile: str,
    model_route: Any,
    system_prefix: Any,
    tool_schema: Any,
    history_messages: Sequence[Mapping[str, Any]],
    projection_version: int = PROJECTION_VERSION,
    media_omitted: bool = False,
    created_at: int = 0,
) -> ProjectionAuditInput:
    profile = _validated_profile(provider_profile)
    history = [dict(message) for message in history_messages]
    return ProjectionAuditInput(
        turn_id=turn_id,
        attempt=attempt,
        provider_profile=profile,
        model_route_hash=stable_projection_hash(model_route),
        system_prefix_hash=stable_projection_hash(system_prefix),
        tool_schema_hash=stable_projection_hash(tool_schema),
        history_hash=stable_projection_hash(history),
        full_prefix_hash=stable_projection_hash(
            {
                "model_route": model_route,
                "system_prefix": system_prefix,
                "tool_schema": tool_schema,
                "history": history,
            }
        ),
        projection_version=projection_version,
        media_omitted=media_omitted,
        created_at=created_at,
    )


def is_strict_message_prefix(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
) -> bool:
    if len(previous) > len(current):
        return False
    return all(canonical_json_bytes(left) == canonical_json_bytes(right) for left, right in zip(previous, current))


def build_entry_projection_hashes(messages: Sequence[ProjectionMessage]) -> tuple[EntryProjectionHash, ...]:
    hashes: dict[str, list[str]] = {}
    for message in messages:
        for source_id in message.source_ids:
            hashes.setdefault(source_id, []).append(message.payload_hash)
    return tuple(
        EntryProjectionHash(source_id=source_id, payload_hashes=tuple(payload_hashes))
        for source_id, payload_hashes in hashes.items()
    )


__all__ = [
    "ANTHROPIC_PROFILE",
    "CANONICAL_PROFILE",
    "ContextProjection",
    "EntryProjectionHash",
    "OPENAI_PROFILE",
    "PROJECTION_VERSION",
    "ProjectionAdapter",
    "ProjectionAudit",
    "ProjectionAuditInput",
    "ProjectionMessage",
    "ProjectionMessageInput",
    "ProjectionLedger",
    "ProjectionStatus",
    "RendererRegistry",
    "RequestProjectionResult",
    "STANDARD_PROJECTION_PROFILES",
    "build_entry_projection_hashes",
    "build_projection_audit_input",
    "canonical_json_bytes",
    "default_renderer_registry",
    "is_strict_message_prefix",
    "merge_projection_status",
    "normalize_provider_profile",
    "sanitize_projection_payload",
    "stable_projection_hash",
]
