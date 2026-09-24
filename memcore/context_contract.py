"""Stable public context-surface contract.

The contract deliberately contains provider payloads rather than SQLite records.
Hosts can replace their model-visible history with this value without learning
MemCore's timeline, settlement, or projection ledger internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CONTEXT_SURFACE_VERSION = "context_surface_v1"
CONTEXT_SURFACE_MESSAGE_METADATA_VERSION = "context_surface_message_metadata_v1"


@dataclass(frozen=True)
class ContextDiagnostic:
    """A stable, non-sensitive diagnostic attached to a surface build."""

    status: str
    reason: str


@dataclass(frozen=True)
class ContextSurface:
    """Provider-ready history split at the host/context boundary.

    ``history_messages`` contains closed history owned by MemCore.
    ``current_message`` is the current stimulus and appears at most once.
    ``active_turn_messages`` contains the still-open provider-native tool round
    and is intentionally preserved verbatim by the host adapter.
    """

    version: str
    provider_profile: str
    history_messages: tuple[Mapping[str, Any], ...]
    current_message: Mapping[str, Any] | None
    active_turn_messages: tuple[Mapping[str, Any], ...]
    projection_hash: str
    projection_generation: int
    message_source_ids: tuple[tuple[str, ...], ...] = ()
    message_projection_metadata: tuple[Mapping[str, Any], ...] = ()
    has_compact_history: bool = False
    current_turn_id: str = ""
    projection_version: int = 1
    compaction_generation: int = 0
    diagnostics: tuple[ContextDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.version != CONTEXT_SURFACE_VERSION:
            raise ValueError("unsupported_context_surface_version")
        if not self.provider_profile:
            raise ValueError("context_surface_provider_profile_required")
        if not self.projection_hash:
            raise ValueError("context_surface_projection_hash_required")
        if int(self.projection_generation) < 0:
            raise ValueError("context_surface_projection_generation_invalid")
        if int(self.projection_version) < 1 or int(self.compaction_generation) < 0:
            raise ValueError("context_surface_generation_invalid")
        if self.message_source_ids and len(self.message_source_ids) != len(self.messages):
            raise ValueError("context_surface_source_ids_length_mismatch")
        if self.message_projection_metadata and len(self.message_projection_metadata) != len(self.messages):
            raise ValueError("context_surface_projection_metadata_length_mismatch")

    @property
    def messages(self) -> tuple[Mapping[str, Any], ...]:
        """The exact ordered message sequence for a provider request."""

        return (
            *self.history_messages,
            *((self.current_message,) if self.current_message is not None else ()),
            *self.active_turn_messages,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready view without exposing internal records."""

        return {
            "version": self.version,
            "provider_profile": self.provider_profile,
            "history_messages": [dict(item) for item in self.history_messages],
            "current_message": dict(self.current_message) if self.current_message is not None else None,
            "active_turn_messages": [dict(item) for item in self.active_turn_messages],
            "projection_hash": self.projection_hash,
            "projection_generation": self.projection_generation,
            "message_source_ids": [list(item) for item in self.message_source_ids],
            "message_projection_metadata": [dict(item) for item in self.message_projection_metadata],
            "has_compact_history": self.has_compact_history,
            "current_turn_id": self.current_turn_id,
            "projection_version": self.projection_version,
            "compaction_generation": self.compaction_generation,
            "diagnostics": [{"status": item.status, "reason": item.reason} for item in self.diagnostics],
        }


@dataclass(frozen=True)
class RequestBindingMessage:
    """One provider-visible message with its immutable source ownership."""

    payload: Mapping[str, Any]
    source_ids: tuple[str, ...]
    source_turn_id: str
    projection_index: int
    projection_status: str
    projection_version: int
    request_index: int


@dataclass(frozen=True)
class RequestBindingGroup:
    """Ordered messages owned by one real source turn."""

    turn_id: str
    relation: str
    messages: tuple[RequestBindingMessage, ...]

    @property
    def request_indexes(self) -> tuple[int, ...]:
        return tuple(message.request_index for message in self.messages)


@dataclass(frozen=True)
class RequestBindingResult:
    """Turn-aware binding for the provider-visible open request suffix."""

    status: str
    reason: str
    messages: tuple[RequestBindingMessage, ...]
    groups: tuple[RequestBindingGroup, ...]
    active_group: RequestBindingGroup | None

    @property
    def ok(self) -> bool:
        return self.status == "bound"


def bind_request_projection_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    active_turn_id: str,
) -> RequestBindingResult:
    """Bind ordered projection descriptors without changing source ownership.

    Hosts use this helper at the provider boundary.  A message attributed to a
    different source turn remains visible in its original request position but
    is never frozen under the active turn.  Missing ownership is rejected
    because guessing would corrupt the projection ledger.
    """

    active_id = str(active_turn_id or "").strip()
    if not active_id:
        return RequestBindingResult("rejected", "active_turn_id_required", (), (), None)

    prepared: list[RequestBindingMessage] = []
    grouped: dict[str, list[RequestBindingMessage]] = {}
    group_order: list[str] = []
    for request_index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            return RequestBindingResult("rejected", "request_binding_message_not_object", (), (), None)
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            return RequestBindingResult("rejected", "request_binding_payload_required", (), (), None)
        source_ids = tuple(
            str(item or "").strip() for item in list(raw.get("source_ids") or ()) if str(item or "").strip()
        )
        if not source_ids:
            return RequestBindingResult("rejected", "request_binding_source_ids_required", (), (), None)
        source_turn_id = str(raw.get("turn_id") or raw.get("source_turn_id") or "").strip()
        if not source_turn_id:
            return RequestBindingResult("rejected", "request_binding_source_turn_ambiguous", (), (), None)
        try:
            projection_index = int(raw.get("projection_index", -1))
            projection_version = int(raw.get("projection_version") or 0)
        except (TypeError, ValueError):
            return RequestBindingResult("rejected", "request_binding_projection_metadata_invalid", (), (), None)
        if projection_index < 0 or projection_version < 1:
            return RequestBindingResult("rejected", "request_binding_projection_metadata_invalid", (), (), None)
        message = RequestBindingMessage(
            payload=dict(payload),
            source_ids=source_ids,
            source_turn_id=source_turn_id,
            projection_index=projection_index,
            projection_status=str(raw.get("projection_status") or "complete"),
            projection_version=projection_version,
            request_index=request_index,
        )
        prepared.append(message)
        if source_turn_id not in grouped:
            grouped[source_turn_id] = []
            group_order.append(source_turn_id)
        grouped[source_turn_id].append(message)

    groups = tuple(
        RequestBindingGroup(
            turn_id=turn_id,
            relation="active" if turn_id == active_id else "standalone",
            messages=tuple(grouped[turn_id]),
        )
        for turn_id in group_order
    )
    active_group = next((group for group in groups if group.relation == "active"), None)
    if active_group is None:
        return RequestBindingResult(
            "rejected",
            "request_binding_active_turn_missing",
            tuple(prepared),
            groups,
            None,
        )
    return RequestBindingResult("bound", "", tuple(prepared), groups, active_group)


__all__ = [
    "CONTEXT_SURFACE_MESSAGE_METADATA_VERSION",
    "CONTEXT_SURFACE_VERSION",
    "ContextDiagnostic",
    "ContextSurface",
    "RequestBindingGroup",
    "RequestBindingMessage",
    "RequestBindingResult",
    "bind_request_projection_messages",
]
