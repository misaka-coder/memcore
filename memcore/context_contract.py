"""Stable public context-surface contract.

The contract deliberately contains provider payloads rather than SQLite records.
Hosts can replace their model-visible history with this value without learning
MemCore's timeline, settlement, or projection ledger internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


__all__ = [
    "CONTEXT_SURFACE_MESSAGE_METADATA_VERSION",
    "CONTEXT_SURFACE_VERSION",
    "ContextDiagnostic",
    "ContextSurface",
]
