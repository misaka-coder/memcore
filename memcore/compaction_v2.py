"""Typed snapshots and atomic commit results for Timeline V2 compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .timeline import TimelineEntry, TurnStatus


@dataclass(frozen=True)
class TurnBundle:
    turn_id: str
    status: TurnStatus
    entries: tuple[TimelineEntry, ...]
    turn_row_version: int
    first_seq_no: int
    last_seq_no: int
    legacy: bool = False

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(entry.source_id for entry in self.entries)


@dataclass(frozen=True)
class CompactionSnapshot:
    namespace_key: tuple[str, str, str, str]
    provider_profile: str
    compaction_generation: int
    bundles: tuple[TurnBundle, ...]
    ordered_source_ids: tuple[str, ...]
    message_row_versions: tuple[tuple[str, int], ...]
    turn_row_versions: tuple[tuple[str, int], ...]
    projection_hashes: tuple[tuple[str, tuple[str, ...]], ...]
    before_projected_tokens: int
    selected_projected_tokens: int
    token_count_quality: str


@dataclass(frozen=True)
class SummaryRecordInput:
    summary_id: str
    summary_profile: str
    source_ids: tuple[str, ...]
    record: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummaryBatchCommitResult:
    status: str
    summaries: tuple[dict[str, Any], ...] = ()
    compaction_generation: int = 0
    reason: str = ""

    @property
    def committed(self) -> bool:
        return self.status in {"committed", "already_committed"}


@dataclass(frozen=True)
class SemanticSnapshot:
    namespace_key: tuple[str, str, str, str]
    compaction_generation: int
    summary_ids: tuple[str, ...]
    summary_row_versions: tuple[tuple[str, int], ...]
    reinforcement_target_id: str = ""
    reinforcement_target_row_version: int = 0


@dataclass(frozen=True)
class SemanticCommitInput:
    snapshot: SemanticSnapshot
    semantic_record: Mapping[str, Any]


@dataclass(frozen=True)
class SemanticBatchCommitResult:
    status: str
    semantic_record: dict[str, Any] | None = None
    compaction_generation: int = 0
    reason: str = ""

    @property
    def committed(self) -> bool:
        return self.status in {"committed", "already_committed"}


@dataclass
class CompactionResult:
    status: str = "not_due"
    provider_profile: str = ""
    source_turn_count: int = 0
    source_entry_count: int = 0
    before_projected_tokens: int = 0
    after_projected_tokens: int = 0
    selected_projected_tokens: int = 0
    source_token_limit: int = 0
    selected_episode_entry_count: int = 0
    source_entry_limit: int = 0
    token_count_quality: str = "not_counted"
    summary_source_ids: list[str] = field(default_factory=list)
    compaction_generation: int = 0
    index_status: str = "not_applicable"
    summaries_created: int = 0
    semantic_created: int = 0
    reinforced: int = 0
    summary_retry_pending: int = 0
    semantic_retry_pending: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider_profile": self.provider_profile,
            "source_turn_count": self.source_turn_count,
            "source_entry_count": self.source_entry_count,
            "before_projected_tokens": self.before_projected_tokens,
            "after_projected_tokens": self.after_projected_tokens,
            "selected_projected_tokens": self.selected_projected_tokens,
            "source_token_limit": self.source_token_limit,
            "selected_episode_entry_count": self.selected_episode_entry_count,
            "source_entry_limit": self.source_entry_limit,
            "token_count_quality": self.token_count_quality,
            "summary_source_ids": list(self.summary_source_ids),
            "compaction_generation": self.compaction_generation,
            "index_status": self.index_status,
            "summaries_created": self.summaries_created,
            "semantic_created": self.semantic_created,
            "reinforced": self.reinforced,
            "summary_retry_pending": self.summary_retry_pending,
            "semantic_retry_pending": self.semantic_retry_pending,
            "reason": self.reason,
        }


__all__ = [
    "CompactionResult",
    "CompactionSnapshot",
    "SemanticBatchCommitResult",
    "SemanticCommitInput",
    "SemanticSnapshot",
    "SummaryBatchCommitResult",
    "SummaryRecordInput",
    "TurnBundle",
]
