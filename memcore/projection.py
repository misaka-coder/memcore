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

from .errors import SchemaError, StaleSnapshotError
from .kind_contract import is_valid_kind, is_valid_kind_prefix, normalize_kind
from .namespace import Namespace
from .rendering import render_semantic_snippet, render_summary_snippet
from .text_utils import normalize_text
from .time_anchor import (
    TIME_PERIOD_LABELS,
    timestamp_to_datetime_label,
    timestamp_to_datetime_weekday_label,
    timestamp_to_weekday_label,
)
from .timeline import TimelineEntry, TimelineEntryInput, TurnRole

PROJECTION_VERSION = 5
CANONICAL_PROFILE = "canonical_user_assistant"
OPENAI_PROFILE = "openai_chat"
ANTHROPIC_PROFILE = "anthropic_messages"
DEEPSEEK_PROFILE = "deepseek_chat"
OPENAI_RESPONSES_PROFILE = "openai_responses"
STANDARD_PROJECTION_PROFILES = frozenset(
    {CANONICAL_PROFILE, OPENAI_PROFILE, ANTHROPIC_PROFILE, DEEPSEEK_PROFILE, OPENAI_RESPONSES_PROFILE}
)

_PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")
_ID_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET_FIELD = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|auth[_-]?key|pt[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|secret)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_NAME = (
    r"(?:api[_-]?key|authorization|auth[_-]?key|pt[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|secret|token)"
)
_BEARER_SECRET = re.compile(r"(?P<prefix>\bBearer\s+)(?P<value>[A-Za-z0-9._~+/=-]{12,})", re.IGNORECASE)
_OPENAI_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}", re.IGNORECASE)
_JWT_SECRET = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_TEXT_SECRET_PREFIX = rf"(?<![A-Za-z0-9_])(?:['\"])?{_SECRET_NAME}(?:['\"])?\s*[:=]\s*"
_QUOTED_SECRET_ASSIGNMENT = re.compile(
    rf"(?P<prefix>{_TEXT_SECRET_PREFIX})(?P<quote>['\"])(?P<value>[^'\"\r\n]{{8,}})(?P=quote)",
    re.IGNORECASE,
)
_UNQUOTED_SECRET_ASSIGNMENT = re.compile(
    rf"(?P<prefix>{_TEXT_SECRET_PREFIX})(?P<value>[A-Za-z0-9_~+/=-]{{16,}})(?=$|[\s,;}}\]])",
    re.IGNORECASE,
)
# Field-level hiding is reserved for explicit host-internal structure fields.
# Executable paths (tool arguments, command output, operation evidence) are
# model-visible evidence and must never be rewritten by the projection layer.
_LOCAL_PATH_FIELD = re.compile(
    r"(?:^|[_-])(?:absolute[_-]?path|backing[_-]?path|cache[_-]?path|cached[_-]?path|"
    r"database[_-]?path|db[_-]?path|local[_-]?path|log[_-]?path|run[_-]?log[_-]?path|"
    r"storage[_-]?path|storage[_-]?relpath)(?:$|[_-])",
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
_SECRET_MARKER = "[secret value omitted]"
_PATH_MARKER = "[local path omitted from persistent history]"
# Marker emitted by projection versions < 3 when executable path evidence was
# replaced.  Used by the explicit legacy-projection migration to find affected
# frozen rows; never used to rewrite live evidence.
LEGACY_PATH_OMISSION_MARKER = "[local path omitted from persistent history]"


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ProjectionStatus(_TextEnum):
    COMPLETE = "complete"
    CANONICAL_FALLBACK = "canonical_fallback"
    REQUEST_FROZEN = "request_frozen"
    MEDIA_OMITTED = "media_omitted"
    SKIPPED_UNSAFE = "skipped_unsafe"
    SETTLED = "settled"


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
    # Settled history projection is a frozen, deterministic compact record of a
    # closed turn (终局紧凑投影账本)。它比 REQUEST_FROZEN 更接近历史终态: 一旦建立,
    # request builder 只从 settled 账本读取, 不再重新投影该 turn。
    ProjectionStatus.SETTLED: 5,
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


def _media_marker_block() -> dict[str, str]:
    return {"type": "text", "text": _MEDIA_MARKER}


def _sanitize_text(value: str) -> tuple[str, ProjectionStatus]:
    """Hide credential values without destroying the surrounding evidence.

    Provider-visible tool results often contain source code that manipulates a
    variable named ``token`` or ``secret``.  A field name is not itself a
    credential, so whole-string omission makes ordinary auth/config work
    impossible.  Only high-confidence literal spans are replaced here; typed
    mappings are still protected by ``_SECRET_FIELD`` below.
    """

    text = str(value)
    redacted = False

    def replace_group(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return f"{match.group('prefix')}{_SECRET_MARKER}"

    def replace_match(_match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return _SECRET_MARKER

    def replace_quoted(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        quote = match.group("quote")
        return f"{match.group('prefix')}{quote}{_SECRET_MARKER}{quote}"

    text = _BEARER_SECRET.sub(replace_group, text)
    text = _OPENAI_SECRET.sub(replace_match, text)
    text = _JWT_SECRET.sub(replace_match, text)
    text = _QUOTED_SECRET_ASSIGNMENT.sub(replace_quoted, text)
    text = _UNQUOTED_SECRET_ASSIGNMENT.sub(replace_group, text)
    text = _CONTROL_CHARS.sub(" ", text)
    return text, ProjectionStatus.SKIPPED_UNSAFE if redacted else ProjectionStatus.COMPLETE


def _sanitize_value(
    value: Any,
    *,
    key: str = "",
    preserve_path_fields: bool = False,
) -> tuple[Any, ProjectionStatus]:
    normalized_key = str(key or "")
    if _SECRET_FIELD.search(normalized_key):
        return _SECRET_MARKER, ProjectionStatus.SKIPPED_UNSAFE
    if not preserve_path_fields and _LOCAL_PATH_FIELD.search(normalized_key):
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
            child_preserves_paths = preserve_path_fields or (raw_type == "tool_use" and child_key == "input")
            clean, child_status = _sanitize_value(
                item,
                key=child_key,
                preserve_path_fields=child_preserves_paths,
            )
            result[child_key] = clean
            status = merge_projection_status(status, child_status)
        return result, status
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        status = ProjectionStatus.COMPLETE
        for item in value:
            clean, child_status = _sanitize_value(
                item,
                preserve_path_fields=preserve_path_fields,
            )
            items.append(clean)
            status = merge_projection_status(status, child_status)
        return items, status
    if isinstance(value, str):
        if value.lower().startswith("data:") and ";base64," in value[:160].lower():
            return _MEDIA_MARKER, ProjectionStatus.MEDIA_OMITTED
        # OpenAI-compatible tool arguments are serialized JSON.  Preserve the
        # provider wire byte-for-byte when clean, but redact typed secret fields
        # structurally so the result remains valid JSON instead of becoming an
        # unusable marker string.
        if normalized_key == "arguments":
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, Mapping):
                clean, status = _sanitize_value(parsed, preserve_path_fields=True)
                if status is ProjectionStatus.COMPLETE:
                    return _CONTROL_CHARS.sub(" ", value), status
                return json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")), status
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return str(value), ProjectionStatus.SKIPPED_UNSAFE
        return value, ProjectionStatus.COMPLETE
    return f"[unsupported {type(value).__name__} omitted]", ProjectionStatus.SKIPPED_UNSAFE


def sanitize_projection_payload(
    payload: Mapping[str, Any],
    *,
    provider_profile: str = "",
) -> tuple[dict[str, Any], ProjectionStatus]:
    if not isinstance(payload, Mapping):
        raise SchemaError("projection_payload_must_be_object")
    role = str(payload.get("role") or "").strip().lower()
    if role in {"system", "developer"}:
        raise SchemaError("projection_system_message_not_persistable")
    item_type = str(payload.get("type") or "").strip().lower()
    responses_item = provider_profile == OPENAI_RESPONSES_PROFILE and item_type in {
        "function_call",
        "function_call_output",
    }
    if not role and not responses_item:
        raise SchemaError("projection_message_role_required")
    sanitized, status = _sanitize_value(dict(payload))
    if not isinstance(sanitized, dict):
        raise SchemaError("projection_payload_must_be_object")
    return sanitized, status


def sanitize_timeline_value(value: Any) -> tuple[Any, ProjectionStatus]:
    """Sanitize model-visible timeline evidence without inventing a provider message role."""

    return _sanitize_value(value)


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
        payload, safety_status = sanitize_projection_payload(
            self.payload,
            provider_profile=self.provider_profile,
        )
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


def _openai_tool_arguments_are_provider_safe(value: Any) -> bool:
    """Return whether a frozen OpenAI tool call still has an object-shaped input.

    Older MemCore builds could replace the whole ``function.arguments`` string
    with a privacy marker.  That retained the fact that a tool was called, but
    the resulting value was neither JSON nor accepted by stricter compatible
    gateways.  New writes preserve the JSON shape; this guard only protects
    reads of already-frozen legacy rows.
    """

    if not isinstance(value, str):
        return False
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(parsed, Mapping)


def _legacy_tool_call_card(message: ProjectionMessage, calls: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "[historical tool call retained as canonical trace]",
        "arguments_status: unavailable_after_safety_projection",
    ]
    content = message.payload.get("content")
    if isinstance(content, str) and content.strip():
        lines.append(f"assistant_preface: {content.strip()}")
    for call in calls:
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        name = str(function.get("name") or "tool").strip() or "tool"
        call_id = str(call.get("id") or "unknown").strip() or "unknown"
        lines.append(f"tool: {name}; call_id: {call_id}")
    if message.source_ids:
        lines.append("source_ids: " + ", ".join(message.source_ids))
    lines.append("The original result remains available in the following canonical trace.")
    return "\n".join(lines)


def _legacy_tool_result_card(message: ProjectionMessage) -> str:
    call_id = str(message.payload.get("tool_call_id") or "unknown").strip() or "unknown"
    content = message.payload.get("content")
    if isinstance(content, str):
        rendered = content
    else:
        rendered = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    lines = [
        "[historical tool result retained as canonical trace]",
        f"call_id: {call_id}",
    ]
    if message.source_ids:
        lines.append("source_ids: " + ", ".join(message.source_ids))
    lines.extend(("result:", rendered))
    return "\n".join(lines)


def provider_safe_projection_messages(
    messages: Sequence[ProjectionMessage],
    *,
    provider_profile: str,
) -> tuple[ProjectionMessage, ...]:
    """Downgrade malformed frozen native calls without mutating source truth.

    The projection ledger remains the immutable record.  This read boundary
    changes only the provider-visible representation of legacy native calls
    whose argument JSON can no longer be reconstructed.  Their call/result
    lineage and full result text stay visible, while gateways no longer parse
    an invalid native ``tool_calls`` envelope.

    Valid projections are returned byte-for-byte unchanged.
    """

    profile = _validated_profile(provider_profile)
    if profile not in {OPENAI_PROFILE, DEEPSEEK_PROFILE}:
        return tuple(messages)

    degraded_call_ids: set[str] = set()
    degraded_action_indexes: set[int] = set()
    action_calls: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for index, message in enumerate(messages):
        payload = message.payload
        if str(payload.get("role") or "").strip().lower() != "assistant":
            continue
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            continue
        calls = tuple(item for item in raw_calls if isinstance(item, Mapping))
        malformed = len(calls) != len(raw_calls) or any(
            not _openai_tool_arguments_are_provider_safe(
                (call.get("function") if isinstance(call.get("function"), Mapping) else {}).get("arguments")
            )
            for call in calls
        )
        if not malformed:
            continue
        degraded_action_indexes.add(index)
        action_calls[index] = calls
        degraded_call_ids.update(str(call.get("id") or "").strip() for call in calls)

    if not degraded_action_indexes:
        return tuple(messages)

    repaired: list[ProjectionMessage] = []
    for index, message in enumerate(messages):
        payload = message.payload
        replacement_payload: dict[str, Any] | None = None
        if index in degraded_action_indexes:
            replacement_payload = {
                "role": "assistant",
                "content": _legacy_tool_call_card(message, action_calls.get(index, ())),
            }
        elif (
            str(payload.get("role") or "").strip().lower() == "tool"
            and str(payload.get("tool_call_id") or "").strip() in degraded_call_ids
        ):
            replacement_payload = {
                "role": "user",
                "content": _legacy_tool_result_card(message),
            }
        if replacement_payload is None:
            repaired.append(message)
            continue
        safe_payload, safety_status = sanitize_projection_payload(replacement_payload)
        repaired.append(
            replace(
                message,
                payload=safe_payload,
                payload_hash=stable_projection_hash(safe_payload),
                projection_status=merge_projection_status(
                    message.projection_status,
                    ProjectionStatus.CANONICAL_FALLBACK,
                    safety_status,
                ),
            )
        )
    return tuple(repaired)


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
    has_compact_history: bool = False

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
    compact_renderer: Renderer | None = None


class RendererRegistry:
    """Versioned exact/prefix registry with canonical fallback for unknown kinds."""

    def __init__(self) -> None:
        self._exact: dict[str, _RendererSpec] = {}
        self._prefix: dict[str, _RendererSpec] = {}
        self._by_identity: dict[tuple[str, int], _RendererSpec] = {
            ("canonical", 1): _RendererSpec(
                "canonical",
                1,
                _render_canonical_entry,
                _render_compact_entry,
            )
        }

    def register_exact(
        self,
        kind: str,
        *,
        renderer_id: str,
        version: int,
        renderer: Renderer,
        compact_renderer: Renderer | None = None,
    ) -> None:
        normalized = normalize_kind(kind)
        if not is_valid_kind(normalized):
            raise SchemaError("renderer_invalid_exact_kind")
        self._register(
            self._exact,
            normalized,
            renderer_id=renderer_id,
            version=version,
            renderer=renderer,
            compact_renderer=compact_renderer,
        )

    def register_prefix(
        self,
        kind_prefix: str,
        *,
        renderer_id: str,
        version: int,
        renderer: Renderer,
        compact_renderer: Renderer | None = None,
    ) -> None:
        normalized = normalize_kind(str(kind_prefix or "").rstrip(".*"))
        if not is_valid_kind_prefix(normalized):
            raise SchemaError("renderer_invalid_kind_prefix")
        self._register(
            self._prefix,
            normalized,
            renderer_id=renderer_id,
            version=version,
            renderer=renderer,
            compact_renderer=compact_renderer,
        )

    def _register(
        self,
        target: dict[str, _RendererSpec],
        selector: str,
        *,
        renderer_id: str,
        version: int,
        renderer: Renderer,
        compact_renderer: Renderer | None,
    ) -> None:
        normalized_id = str(renderer_id or "").strip()
        if (
            not normalized_id
            or len(normalized_id) > 120
            or not callable(renderer)
            or (compact_renderer is not None and not callable(compact_renderer))
        ):
            raise SchemaError("renderer_invalid_registration")
        resolved_version = int(version)
        if resolved_version < 1:
            raise SchemaError("renderer_invalid_version")
        spec = _RendererSpec(normalized_id, resolved_version, renderer, compact_renderer)
        identity = (normalized_id, resolved_version)
        existing = self._by_identity.get(identity)
        if existing is not None and (
            existing.renderer is not renderer or existing.compact_renderer is not compact_renderer
        ):
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
        return self.render_detail(entry, timezone=timezone, detail="full")

    def render_detail(self, entry: TimelineEntry, *, timezone: str, detail: str = "full") -> RendererResult:
        """Render one entry for provider history (full) or explicit evidence reads (compact)."""

        resolved_detail = str(detail or "full").strip().lower()
        if resolved_detail not in {"full", "compact"}:
            raise SchemaError("renderer_invalid_detail")
        spec = self._by_identity.get((entry.renderer_id, entry.renderer_version))
        status = ProjectionStatus.COMPLETE
        if spec is None:
            spec = self._by_identity[("canonical", 1)]
            status = ProjectionStatus.CANONICAL_FALLBACK
        renderer = spec.renderer if resolved_detail == "full" else spec.compact_renderer or _render_compact_entry
        text = str(renderer(entry, timezone) or "")
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
        "event",
        renderer_id="event.structured",
        version=1,
        renderer=_render_structured_event,
    )
    registry.register_prefix(
        "event.finance",
        renderer_id="event.finance",
        version=1,
        renderer=_render_finance_event,
    )
    registry.register_prefix(
        "material",
        renderer_id="material.structured",
        version=1,
        renderer=_render_structured_material,
    )
    return registry


def _display_scalar(value: Any) -> str:
    text = normalize_text(value)
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def _display_multiline(value: Any) -> str:
    text = normalize_text(value).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub(" ", text)


_DISPLAY_FIELD_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


def _display_field_value(value: Any) -> str:
    if isinstance(value, str):
        return _display_scalar(value)
    if isinstance(value, (Mapping, list, tuple, bool, int, float)) or value is None:
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _display_scalar(value)


def _parse_structured_semantic_fields(value: str) -> tuple[dict[str, str], str]:
    text = _display_multiline(value).strip()
    if not text:
        return {}, ""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, separator, raw_value = stripped.partition(":")
        key = key.strip()
        field_value = _display_scalar(raw_value)
        if not separator or not field_value or _DISPLAY_FIELD_KEY.fullmatch(key) is None or key in fields:
            return {}, text
        fields[key] = field_value
    return (fields, "") if fields else ({}, text)


def _render_structured_entry(
    entry: TimelineEntry,
    timezone: str,
    *,
    field_order: Sequence[str],
    field_aliases: Mapping[str, str] | None = None,
) -> str:
    """Render open event/material payloads once while preserving free-form meaning."""

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

    semantic_fields, free_text = _parse_structured_semantic_fields(entry.semantic_text)
    aliases = dict(field_aliases or {})
    merged_fields: dict[str, str] = {aliases.get(key, key): value for key, value in semantic_fields.items()}
    safe_payload, _ = _sanitize_value(dict(entry.payload))
    for raw_key, raw_value in dict(safe_payload or {}).items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key == "text" and _display_multiline(raw_value).strip() == _display_multiline(entry.semantic_text).strip():
            continue
        rendered = _display_field_value(raw_value)
        if rendered:
            merged_fields[key] = rendered

    preferred = {name: index for index, name in enumerate(field_order)}
    for key in sorted(merged_fields, key=lambda item: (preferred.get(item, len(preferred)), item)):
        lines.append(f"{key}: {merged_fields[key]}")
    if free_text:
        lines.extend(("content:", free_text))
    return "\n".join(lines)


def _entry_header(entry: TimelineEntry, timezone: str) -> str:
    stamp = timestamp_to_datetime_weekday_label(entry.timestamp, timezone) if entry.timestamp > 0 else ""
    period = TIME_PERIOD_LABELS.get(str(entry.time_of_day or ""), "")
    anchor = " | ".join(part for part in (stamp, period) if part)
    return f"[{anchor}] {entry.kind}" if anchor else entry.kind


def _chat_actor_value(*, stable_id: Any = "", display_name: Any = "") -> str:
    stable = _display_scalar(stable_id)
    display = _display_scalar(display_name)
    if display and stable and display != stable:
        return f"{display} (id={stable})"
    return display or stable


def _append_chat_body(lines: list[str], *, label: str, value: Any, indent: str = "") -> None:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)
    if indent and text and "\n" not in text:
        lines.append(f"{indent}{label}: {text}")
        return
    lines.append(f"{indent}{label}:")
    if text:
        body_indent = f"{indent}  " if indent else ""
        lines.extend(f"{body_indent}{line}" for line in text.splitlines())


def _render_chat_message(entry: TimelineEntry, timezone: str) -> str:
    """Render chat history as compact facts, never as an output-protocol replay."""

    payload = dict(entry.payload)
    lines: list[str] = []
    if entry.timestamp > 0:
        lines.append(
            f"time: {timestamp_to_datetime_label(entry.timestamp, timezone)} "
            f"{timestamp_to_weekday_label(entry.timestamp, timezone)}"
        )

    actor = entry.namespace.actor
    if actor is not None:
        value = _chat_actor_value(
            stable_id=getattr(actor, "stable_id", ""),
            display_name=getattr(actor, "display_name", ""),
        )
        if value:
            lines.append(f"actor: {value}")
    if entry.target_actor is not None:
        value = _chat_actor_value(
            stable_id=getattr(entry.target_actor, "stable_id", ""),
            display_name=getattr(entry.target_actor, "display_name", ""),
        )
        if value:
            lines.append(f"target: {value}")

    mentions: list[str] = []
    for mention in payload.get("mentioned_actors") or ():
        if not isinstance(mention, Mapping):
            continue
        value = _chat_actor_value(
            stable_id=mention.get("actor_id"),
            display_name=mention.get("display_name"),
        )
        if value and value not in mentions:
            mentions.append(value)
    if mentions:
        lines.append("mentions: " + json.dumps(mentions, ensure_ascii=False, separators=(",", ":")))

    reply = payload.get("reply_reference")
    if isinstance(reply, Mapping) and any(
        str(reply.get(key) or "").strip() for key in ("actor_id", "actor_display_name", "message_id", "excerpt")
    ):
        lines.append("reply_to:")
        value = _chat_actor_value(
            stable_id=reply.get("actor_id"),
            display_name=reply.get("actor_display_name"),
        )
        if value:
            lines.append(f"  actor: {value}")
        reply_timestamp = reply.get("timestamp")
        try:
            resolved_reply_timestamp = int(float(reply_timestamp or 0))
        except (TypeError, ValueError):
            resolved_reply_timestamp = 0
        if resolved_reply_timestamp > 0:
            lines.append(f"  time: {timestamp_to_datetime_label(resolved_reply_timestamp, timezone)}")
        for key in ("message_id", "conversation_kind", "conversation_id", "attachment_count"):
            value = _display_scalar(reply.get(key))
            if value:
                lines.append(f"  {key}: {value}")
        if str(reply.get("excerpt") or ""):
            _append_chat_body(lines, label="text", value=reply.get("excerpt"), indent="  ")

    forwards = payload.get("forward_references")
    if isinstance(forwards, (list, tuple)) and forwards:
        safe_forwards, _ = _sanitize_value(list(forwards), key="forward_references")
        lines.append(
            "forwards: " + json.dumps(safe_forwards, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    normalized_kind = str(entry.kind or "").strip().lower()
    is_assistant = entry.origin.value == "assistant"
    is_voice = normalized_kind in {"message.user.voice", "message.assistant.voice"}
    if normalized_kind == "message.user.observed":
        lines.append("mode: observed")
    if is_voice:
        lines.append("medium: voice")
    if is_assistant:
        emotion = _display_scalar(payload.get("emotion"))
        if emotion:
            lines.append(f"emotion: {emotion}")
    if is_voice:
        delivery = _display_scalar(payload.get("delivery_status") or payload.get("delivery"))
        if delivery and delivery not in {"delivered", "success", "completed"}:
            lines.append(f"delivery: {delivery}")
        for key in ("delivered_units", "interrupted_units"):
            value = payload.get(key)
            if isinstance(value, (list, tuple)) and value:
                safe_value, _ = _sanitize_value(list(value), key=key)
                lines.append(
                    f"{key}: " + json.dumps(safe_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )

    _append_chat_body(
        lines,
        label="speech" if is_assistant else "text",
        value=payload.get("text") or entry.semantic_text or "",
    )
    return "\n".join(lines)


def _is_chat_message_entry(entry: TimelineEntry) -> bool:
    return str(entry.kind or "") in {
        "message.user",
        "message.user.observed",
        "message.user.voice",
        "message.assistant",
        "message.assistant.voice",
    }


def _tool_arguments_json(value: Any) -> str:
    """Keep host-supplied JSON argument bytes intact when already serialized."""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, Mapping):
            return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _render_compact_entry(entry: TimelineEntry, timezone: str) -> str:
    """Compact, reloadable evidence for operations/materials without copying large payloads."""

    lines = [_entry_header(entry, timezone), f"source_id: {_display_scalar(entry.source_id)}"]
    for actor_line in (
        _actor_line("actor", entry.namespace.actor),
        _actor_line("target_actor", entry.target_actor),
    ):
        if actor_line:
            lines.append(actor_line)
    if entry.correlation_id:
        lines.append(f"correlation_id: {_display_scalar(entry.correlation_id)}")
    if entry.relation_status:
        lines.append(f"relation_status: {_display_scalar(entry.relation_status)}")
    status = _display_scalar(entry.trace_metadata.get("status") or entry.payload.get("status"))
    if status:
        lines.append(f"status: {status}")

    retained = entry.trace_metadata.get("retention_anchor")
    if isinstance(retained, Mapping) and retained:
        safe_anchor, _ = _sanitize_value(dict(retained))
        lines.append("anchor: " + json.dumps(safe_anchor, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    for key in ("file_id", "filename", "kind", "file_status", "derived_status"):
        value = _display_scalar(entry.payload.get(key))
        if value:
            lines.append(f"{key}: {value}")

    payload_chars = len(entry.semantic_text) + len(
        json.dumps(_json_ready(entry.payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    lines.append(f"stored_chars: {payload_chars}")
    memory_id = json.dumps(_display_scalar(entry.source_id), ensure_ascii=False)
    lines.append(
        "expand: 完整内容已保留；需要该条正文时调用 "
        f'open_memory(memory_id={memory_id}, view="content", detail="full")。'
    )
    return "\n".join(lines)


_FINANCE_ORDER = ("source", "published_at", "title", "summary", "url")
_MATERIAL_ORDER = (
    "source",
    "file_id",
    "kind",
    "filename",
    "mime",
    "file_status",
    "status",
    "derived_status",
    "reason",
)


def _render_structured_event(entry: TimelineEntry, timezone: str) -> str:
    return _render_structured_entry(entry, timezone, field_order=_FINANCE_ORDER)


def _render_structured_material(entry: TimelineEntry, timezone: str) -> str:
    return _render_structured_entry(
        entry,
        timezone,
        field_order=_MATERIAL_ORDER,
        field_aliases={"mime_type": "mime"},
    )


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
        observation_decider: Callable[[TimelineEntry, str], tuple[str, str]] | None = None,
    ) -> tuple[ProjectionMessageInput, ...]:
        profile = _validated_profile(provider_profile)
        if profile not in STANDARD_PROJECTION_PROFILES:
            raise SchemaError("projection_profile_unsupported")
        visible = [entry for entry in entries if entry.prompt_visible]
        if profile in {OPENAI_PROFILE, DEEPSEEK_PROFILE}:
            raw = self._project_openai(
                visible,
                provider_profile=profile,
                observation_decider=observation_decider,
            )
        elif profile == OPENAI_RESPONSES_PROFILE:
            raw = self._project_openai_responses(visible, observation_decider=observation_decider)
        elif profile == ANTHROPIC_PROFILE:
            raw = self._project_anthropic(visible, observation_decider=observation_decider)
        else:
            raw = self._project_canonical(visible, observation_decider=observation_decider)
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

    def render_chat_entry(self, entry: TimelineEntry) -> str | None:
        """Return the shared semantic chat wire, or ``None`` for non-chat kinds."""

        if not _is_chat_message_entry(entry):
            return None
        return _render_chat_message(entry, self.timezone)

    def context_surface_payload(self, entry: TimelineEntry, *, provider_profile: str) -> dict[str, Any] | None:
        """Render an unfrozen chat entry without exposing host/output protocol fields."""

        text = self.render_chat_entry(entry)
        if text is None:
            return None
        role = "assistant" if entry.origin.value == "assistant" else "user"
        original = dict(entry.payload)
        for key in tuple(original):
            if key not in {"content", "type"}:
                original.pop(key, None)
        content = original.get("content")
        if isinstance(content, list):
            blocks = [dict(block) for block in content if isinstance(block, Mapping)]
            text_types = {"text", "input_text"}
            first_text = next((block for block in blocks if str(block.get("type") or "") in text_types), None)
            if first_text is not None:
                first_text["text"] = text
            else:
                block_type = "input_text" if provider_profile == OPENAI_RESPONSES_PROFILE else "text"
                blocks.insert(0, {"type": block_type, "text": text})
            original["content"] = blocks
        else:
            original["content"] = text
        if provider_profile == OPENAI_RESPONSES_PROFILE:
            original.setdefault("type", "message")
            original["role"] = role
            return original
        original["role"] = role
        if provider_profile == ANTHROPIC_PROFILE and not isinstance(original.get("content"), list):
            original["content"] = [{"type": "text", "text": text}]
        return original

    def _assistant_final_text(self, entry: TimelineEntry) -> str:
        if _is_chat_message_entry(entry):
            return _render_chat_message(entry, self.timezone)
        host_state, _ = _sanitize_value(
            {
                "kind": entry.kind,
                "data": {key: value for key, value in dict(entry.payload).items() if key != "provider_output_raw"},
            }
        )
        return json.dumps(
            {
                "speech": str(entry.semantic_text or ""),
                "host_state": host_state,
            },
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )

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
        if "output" in payload:
            output, _ = _sanitize_value(payload.get("output"), key="output")
            if isinstance(output, str):
                return output
            return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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

    def _project_canonical(
        self,
        entries: Sequence[TimelineEntry],
        *,
        observation_decider: Callable[[TimelineEntry, str], tuple[str, str]] | None = None,
    ) -> list[ProjectionMessageInput]:
        messages: list[ProjectionMessageInput] = []
        for entry in entries:
            rendered = self._render(entry)
            role = "assistant" if entry.origin.value == "assistant" else "user"
            content = (
                _render_chat_message(entry, self.timezone)
                if _is_chat_message_entry(entry)
                else self._assistant_final_text(entry)
                if entry.turn_role is TurnRole.FINAL
                else rendered.text
            )
            if entry.turn_role is TurnRole.OBSERVATION and observation_decider is not None:
                _, content = observation_decider(entry, content)
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

    def _project_openai(
        self,
        entries: Sequence[TimelineEntry],
        *,
        provider_profile: str = OPENAI_PROFILE,
        observation_decider: Callable[[TimelineEntry, str], tuple[str, str]] | None = None,
    ) -> list[ProjectionMessageInput]:
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
                            "arguments": _tool_arguments_json(self._tool_input(item)),
                        },
                    }
                    for item in batch
                ]
                messages.append(
                    ProjectionMessageInput(
                        provider_profile=provider_profile,
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
                content = self._tool_result_content(entry)
                if observation_decider is not None:
                    _, content = observation_decider(entry, content)
                payload = {
                    "role": "tool",
                    "tool_call_id": entry.correlation_id,
                    "content": content,
                }
            elif entry.origin.value == "assistant":
                payload = {
                    "role": "assistant",
                    "content": (
                        _render_chat_message(entry, self.timezone)
                        if _is_chat_message_entry(entry)
                        else self._assistant_final_text(entry)
                        if entry.turn_role is TurnRole.FINAL
                        else rendered.text
                    ),
                }
            else:
                payload = {
                    "role": "user",
                    "content": (
                        _render_chat_message(entry, self.timezone) if _is_chat_message_entry(entry) else rendered.text
                    ),
                }
            messages.append(
                ProjectionMessageInput(
                    provider_profile=provider_profile,
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

    def _project_openai_responses(
        self,
        entries: Sequence[TimelineEntry],
        *,
        observation_decider: Callable[[TimelineEntry, str], tuple[str, str]] | None = None,
    ) -> list[ProjectionMessageInput]:
        """Project the lossless timeline into OpenAI Responses input items."""

        messages: list[ProjectionMessageInput] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            if entry.turn_role is TurnRole.ACTION:
                batch: list[TimelineEntry] = []
                while index < len(entries) and entries[index].turn_role is TurnRole.ACTION:
                    batch.append(entries[index])
                    index += 1
                for item in batch:
                    messages.append(
                        ProjectionMessageInput(
                            provider_profile=OPENAI_RESPONSES_PROFILE,
                            payload={
                                "type": "function_call",
                                "call_id": item.correlation_id,
                                "name": self._tool_name(item),
                                "arguments": _tool_arguments_json(self._tool_input(item)),
                            },
                            source_ids=(item.source_id,),
                            projection_status=ProjectionStatus.CANONICAL_FALLBACK,
                        )
                    )
                continue
            rendered = self._render(entry)
            if entry.turn_role is TurnRole.OBSERVATION:
                content = self._tool_result_content(entry)
                if observation_decider is not None:
                    _, content = observation_decider(entry, content)
                payload = {
                    "type": "function_call_output",
                    "call_id": entry.correlation_id,
                    "output": content,
                }
            else:
                role = "assistant" if entry.origin.value == "assistant" else "user"
                content = (
                    _render_chat_message(entry, self.timezone)
                    if _is_chat_message_entry(entry)
                    else self._assistant_final_text(entry)
                    if entry.turn_role is TurnRole.FINAL
                    else rendered.text
                )
                payload = {"type": "message", "role": role, "content": content}
            messages.append(
                ProjectionMessageInput(
                    provider_profile=OPENAI_RESPONSES_PROFILE,
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

    def _project_anthropic(
        self,
        entries: Sequence[TimelineEntry],
        *,
        observation_decider: Callable[[TimelineEntry, str], tuple[str, str]] | None = None,
    ) -> list[ProjectionMessageInput]:
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
                    result_content = self._tool_result_content(item)
                    if observation_decider is not None:
                        _, result_content = observation_decider(item, result_content)
                    result_block = {
                        "type": "tool_result",
                        "tool_use_id": item.correlation_id,
                        "content": result_content,
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
            text = (
                _render_chat_message(entry, self.timezone)
                if _is_chat_message_entry(entry)
                else self._assistant_final_text(entry)
                if entry.turn_role is TurnRole.FINAL
                else rendered.text
            )
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
        saved_projections: Sequence[ProjectionMessage] | None = None,
    ) -> list[ProjectionMessage]:
        profile = _validated_profile(provider_profile)
        saved = (
            list(saved_projections)
            if saved_projections is not None
            else self.store.get_turn_projections(
                namespace=namespace,
                turn_id=turn_id,
                provider_profile=profile,
            )
        )
        visible_entries = [entry for entry in entries if entry.prompt_visible]
        covered = tuple(source_id for message in saved for source_id in message.source_ids)
        expected = tuple(entry.source_id for entry in visible_entries)
        if covered != expected[: len(covered)]:
            if len(covered) > len(expected) and expected == covered[: len(expected)]:
                raise StaleSnapshotError("projection_entries_stale")
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
            appended = self.store.save_turn_projections(
                namespace=namespace,
                turn_id=turn_id,
                projections=generated,
            )
            saved.extend(appended)
        return saved

    def freeze_memory_record(
        self,
        *,
        namespace: Namespace,
        record: Mapping[str, Any],
        id_key: str,
        turn_prefix: str,
        provider_profile: str,
        saved_projections: Sequence[ProjectionMessage] | None = None,
    ) -> list[ProjectionMessage]:
        profile = _validated_profile(provider_profile)
        source_id = str(record.get(id_key) or "").strip()
        if not source_id:
            raise SchemaError("projection_memory_source_id_required")
        turn_id = self.memory_turn_id(source_id=source_id, id_key=id_key, turn_prefix=turn_prefix)
        saved = (
            list(saved_projections)
            if saved_projections is not None
            else self.store.get_turn_projections(
                namespace=namespace,
                turn_id=turn_id,
                provider_profile=profile,
            )
        )
        if saved:
            return saved
        generated = self.adapter.project_memory_record(
            record,
            provider_profile=profile,
            source_id=source_id,
            enable_flavor=self.enable_flavor,
        )
        stored = self.store.save_turn_projections(
            namespace=namespace,
            turn_id=turn_id,
            projections=[generated],
        )
        return list(stored)

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
    "DEEPSEEK_PROFILE",
    "EntryProjectionHash",
    "OPENAI_PROFILE",
    "OPENAI_RESPONSES_PROFILE",
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
    "sanitize_timeline_value",
    "stable_projection_hash",
]
