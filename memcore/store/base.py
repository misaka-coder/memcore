"""MemoryStore 接口 —— 记录侧(关系型真相源)。

向量库只做语义索引;真相、原文、时间、字段全在这里。检索拿到 source_id 后回这里取原文、做上下文扩窗。
默认实现 SQLiteMemoryStore 在后续切片提供;此处只钉接口。

写入一致性见设计文档 §B:用 outbox 模式(index_status: pending → indexed + 修复扫描),
不做 SQLite/向量跨库事务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..timeline import TurnProjectionSettlement

from ..compaction_v2 import (
    CompactionSnapshot,
    SemanticBatchCommitResult,
    SemanticCommitInput,
    SummaryBatchCommitResult,
    SummaryRecordInput,
    TurnBundle,
)
from ..namespace import Namespace
from ..projection import (
    ProjectionAudit,
    ProjectionAuditInput,
    ProjectionMessage,
    ProjectionMessageInput,
    RequestProjectionResult,
)
from ..timeline import (
    CompletionCommitResult,
    TimelineEntry,
    TimelineEntryInput,
    TurnAbortResult,
    TurnCompletion,
    TurnHandle,
)


@dataclass(frozen=True)
class LineageClosure:
    """Namespace-scoped lineage graph around one or more raw/derived sources."""

    status: str
    requested_ids: tuple[str, ...]
    descendant_ids: tuple[str, ...] = ()
    ancestor_ids: tuple[str, ...] = ()
    reason: str = ""

    @property
    def all_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.requested_ids, *self.descendant_ids, *self.ancestor_ids)))


@dataclass(frozen=True)
class RawTurnWindow:
    """Namespace-safe raw entries around one complete turn/standalone anchor."""

    status: str
    entries: tuple[TimelineEntry, ...] = ()
    anchor_source_id: str = ""
    before_turns: int = 0
    after_turns: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ContextProjectionRows:
    """One namespace-scoped batch of rows needed to rebuild provider history.

    Stores may leave this facade unsupported; ``MemorySystem`` keeps the
    existing per-turn read path as a compatibility fallback.  Implementations
    must preserve the same row ordering as the corresponding scalar methods.
    """

    projections_by_turn: dict[str, tuple[ProjectionMessage, ...]]
    settlements_by_turn: dict[str, dict[str, Any]]
    settled_rows_by_turn: dict[str, tuple[dict[str, Any], ...]]


class MemoryStore(ABC):
    def runtime_identity(self) -> str:
        """Safe process-local identity; must never expose a database path."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{id(self)}"

    # --- 写 ---
    def begin_turn(
        self,
        *,
        namespace: Namespace,
        stimulus_entries: list[TimelineEntryInput],
        annotation_target_ids: list[str],
        turn_id: str = "",
        opened_at: int = 0,
        operation_projection_policy: str = "full_until_raw_compaction",
        operation_settlement_min_utf8_bytes: int = 256,
        operation_settlement_min_saved_ratio: float = 0.5,
    ) -> TurnHandle:
        raise NotImplementedError

    def begin_turn_from_existing_sources(
        self,
        *,
        namespace: Namespace,
        stimulus_source_ids: list[str],
        annotation_target_ids: list[str],
        turn_id: str = "",
        opened_at: int = 0,
        operation_projection_policy: str = "full_until_raw_compaction",
        operation_settlement_min_utf8_bytes: int = 256,
        operation_settlement_min_saved_ratio: float = 0.5,
    ) -> TurnHandle:
        """Open a turn by linking existing standalone timeline entries.

        Stores that do not implement this optional Timeline V2 primitive fail
        explicitly instead of duplicating or fabricating stimulus entries.
        """

        raise NotImplementedError

    def append_entry(self, *, namespace: Namespace, entry: TimelineEntryInput) -> TimelineEntry:
        raise NotImplementedError

    def append_standalone_entry(self, *, namespace: Namespace, entry: TimelineEntryInput) -> TimelineEntry:
        """Persist a typed fact/event that does not trigger a model-response turn."""
        raise NotImplementedError

    def commit_turn_completion(self, *, namespace: Namespace, completion: TurnCompletion) -> CompletionCommitResult:
        raise NotImplementedError

    def abort_turn(self, *, namespace: Namespace, turn_id: str, reason: str, closed_at: int) -> TurnAbortResult:
        raise NotImplementedError

    def abort_stale_open_turns(
        self,
        *,
        namespace: Namespace,
        opened_before: int,
        reason: str,
        closed_at: int,
    ) -> tuple[TurnAbortResult, ...]:
        """Atomically abort stale open turns inside one conversation namespace."""
        raise NotImplementedError

    def save_turn_projections(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        projections: list[ProjectionMessageInput],
    ) -> tuple[ProjectionMessage, ...]:
        raise NotImplementedError

    def commit_request_projection(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        projections: list[ProjectionMessageInput],
        audit: ProjectionAuditInput,
    ) -> RequestProjectionResult:
        raise NotImplementedError

    def list_compaction_bundles(self, *, namespace: Namespace) -> list[TurnBundle]:
        raise NotImplementedError

    def commit_summary_batch(
        self,
        *,
        namespace: Namespace,
        snapshot: CompactionSnapshot,
        records: list[SummaryRecordInput],
    ) -> SummaryBatchCommitResult:
        raise NotImplementedError

    def commit_semantic_batch(
        self,
        *,
        namespace: Namespace,
        commit: SemanticCommitInput,
    ) -> SemanticBatchCommitResult:
        raise NotImplementedError

    @abstractmethod
    def add_message(
        self, *, namespace: Namespace, role: str, content: str, timestamp: int, **fields: Any
    ) -> dict[str, Any]:
        """落一条 raw 消息(幂等 source_id;初始 index_status=pending)。返回入库记录。"""
        raise NotImplementedError

    @abstractmethod
    def add_summary(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def add_semantic_summary(self, *, namespace: Namespace, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def mark_messages_summarized(self, source_ids: list[str], summary_id: str) -> None:
        """raw → 摘要后,把这些原始消息标记已摘要,移出未摘要窗口。"""
        raise NotImplementedError

    @abstractmethod
    def mark_summaries_semanticized(self, summary_ids: list[str], semantic_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_index_status(self, source_id: str, status: str) -> None:
        """设置索引状态:pending / indexed / skipped(明确不进入向量索引)。"""
        raise NotImplementedError

    def set_index_state(
        self,
        source_id: str,
        status: str,
        *,
        index_schema_version: int = 0,
        index_key: str = "",
    ) -> None:
        """Persist index status plus the schema generation that produced the entry."""
        self.set_index_status(source_id, status)

    @abstractmethod
    def update_message_memory_metadata(
        self, *, namespace: Namespace, source_id: str, memory_metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        """更新 raw message 的 memory_metadata,并置回 pending 等待 raw index 重建。找不到返回 None。"""
        raise NotImplementedError

    def stage_message_memory_metadata(
        self, *, namespace: Namespace, source_id: str, memory_metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Stage metadata on an open stimulus without granting retrieval admission."""
        raise NotImplementedError

    # --- 读(可见三层窗口 + 检索回取) ---
    @abstractmethod
    def get_record_by_source_id(self, source_id: str) -> dict[str, Any] | None:
        """Legacy global lookup. New Timeline V2 runtime paths must use get_entry()."""
        raise NotImplementedError

    def get_entry(self, *, namespace: Namespace, source_id: str) -> TimelineEntry | None:
        """Namespace-safe lookup for one raw TimelineEntry.

        Kept non-abstract during the V2 migration window so existing third-party
        V1 stores still construct; V2 runtime activation requires an override.
        """
        raise NotImplementedError

    def get_retrieval_record(
        self,
        *,
        namespace: Namespace,
        source_id: str,
        cross_conversation: bool = False,
    ) -> dict[str, Any] | None:
        """Namespace-safe lookup for a raw/summary/semantic retrieval candidate."""
        raise NotImplementedError

    def get_entries_by_source_ids(
        self,
        *,
        namespace: Namespace,
        source_ids: tuple[str, ...],
        cross_conversation: bool = False,
    ) -> list[TimelineEntry]:
        """Read exact raw lineage children in requested-id order within hard scope."""
        raise NotImplementedError

    def resolve_lineage_source_ids(
        self,
        *,
        namespace: Namespace,
        source_ids: tuple[str, ...],
        cross_conversation: bool = False,
        include_ancestors: bool = True,
    ) -> LineageClosure:
        """Resolve descendants and ancestors without leaving the authorized Namespace."""
        raise NotImplementedError

    def get_turn(self, *, namespace: Namespace, turn_id: str) -> TurnHandle | None:
        """Namespace-safe turn lookup for lifecycle resume/idempotency."""
        raise NotImplementedError

    def upsert_turn_projection_settlement(
        self,
        *,
        namespace: Namespace,
        settlement: "TurnProjectionSettlement",
    ) -> bool:
        """Idempotently persist one frozen turn settlement (first write wins)."""
        raise NotImplementedError

    def commit_turn_projection_settlement(
        self,
        *,
        namespace: Namespace,
        settlement: "TurnProjectionSettlement",
        projections: list[ProjectionMessageInput],
    ) -> bool:
        """Atomically publish one settlement and its complete provider projection set."""
        raise NotImplementedError

    def get_turn_projection_settlement(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        provider_profile: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def list_settled_projections(self, *, settlement_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def list_turn_projection_settlements(self, *, namespace: Namespace) -> list[dict[str, Any]]:
        """List namespace-scoped settlement rows (no prompt payloads) for observability."""
        raise NotImplementedError

    def upsert_settled_prompt_projection(
        self,
        *,
        settlement_id: str,
        projection_index: int,
        provider_profile: str,
        message: Any,
        projection_status: str = "settled",
    ) -> None:
        """Idempotently persist one frozen settled provider message (first write wins)."""
        raise NotImplementedError

    def get_turn_entries(self, *, namespace: Namespace, turn_id: str) -> list[TimelineEntry]:
        """Read a complete turn only after validating namespace + conversation ownership."""
        raise NotImplementedError

    def get_correlation_entries(
        self, *, namespace: Namespace, turn_id: str, correlation_id: str
    ) -> list[TimelineEntry]:
        """Read one action/observation branch under an owned turn."""
        raise NotImplementedError

    def list_prompt_visible_entries(self, *, namespace: Namespace) -> list[TimelineEntry]:
        """Read the current conversation's unsummarized provider-visible timeline."""
        raise NotImplementedError

    def get_turn_projections(
        self,
        *,
        namespace: Namespace,
        turn_id: str,
        provider_profile: str,
    ) -> list[ProjectionMessage]:
        raise NotImplementedError

    def get_context_projection_rows(
        self,
        *,
        namespace: Namespace,
        turn_ids: tuple[str, ...],
        provider_profile: str,
    ) -> ContextProjectionRows:
        """Batch-read frozen/full and settled projection rows for visible turns."""

        raise NotImplementedError

    def migrate_legacy_path_projections(
        self,
        *,
        namespace: Namespace,
        target_version: int,
        projection_builder: Callable,
        settlement_builder: Callable | None = None,
        markers: tuple = (),
        dry_run: bool = False,
    ) -> dict:
        """Re-project legacy path-damage rows and rebuild stale settlements."""
        raise NotImplementedError

    def list_projection_namespaces(self) -> list:
        """Distinct (tenant, user, domain, conversation) scopes with frozen projections."""
        raise NotImplementedError

    def list_projection_audits(
        self,
        *,
        namespace: Namespace,
        turn_id: str = "",
        provider_profile: str = "",
    ) -> list[ProjectionAudit]:
        raise NotImplementedError

    def get_conversation_generations(self, *, namespace: Namespace) -> tuple[int, int]:
        """Return (compaction_generation, projection_generation)."""
        raise NotImplementedError

    @abstractmethod
    def get_context_slice(self, *, namespace: Namespace, seq_no: int, window: int) -> list[dict[str, Any]]:
        """raw 上下文扩窗:取目标消息前后 window 条邻居。"""
        raise NotImplementedError

    @abstractmethod
    def get_raw_turn_window(
        self,
        *,
        namespace: Namespace,
        anchor_source_id: str,
        before_turns: int,
        after_turns: int,
    ) -> RawTurnWindow:
        """Read complete raw turn groups around an owned source id."""
        raise NotImplementedError

    @abstractmethod
    def get_unsummarized_messages(self, *, namespace: Namespace) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_visible_episodic_summaries(
        self, *, namespace: Namespace, limit: int, cross_conversation: bool = False
    ) -> list[dict[str, Any]]:
        """cross_conversation=False(默认):只看当前会话;True:跨会话看同一用户(visible_memory_scope='user')。"""
        raise NotImplementedError

    def get_episodic_summaries_by_time_range(
        self,
        *,
        namespace: Namespace,
        start_ts: int,
        end_ts: int,
        cross_conversation: bool = False,
    ) -> list[dict[str, Any]]:
        """Return every overlapping episodic summary in deterministic time order."""
        raise NotImplementedError

    def get_semantic_summaries_by_time_range(
        self,
        *,
        namespace: Namespace,
        start_ts: int,
        end_ts: int,
        cross_conversation: bool = False,
    ) -> list[dict[str, Any]]:
        """Return every overlapping semantic summary in deterministic time order."""
        raise NotImplementedError

    def get_catalog_raw_coverage(
        self,
        *,
        namespace: Namespace,
        start_ts: int,
        end_ts: int,
        cross_conversation: bool = False,
    ) -> dict[str, Any]:
        """Return compact accounting for covered, live, and broken raw lineage."""
        raise NotImplementedError

    @abstractmethod
    def get_recent_semantic_summaries(
        self, *, namespace: Namespace, limit: int | None = None, cross_conversation: bool = False
    ) -> list[dict[str, Any]]:
        """limit=None 取全部(供衰减排序对完整候选集生效,不按 recency 预截断)。"""
        raise NotImplementedError

    @abstractmethod
    def get_uncompacted_episodic_summaries(
        self, *, namespace: Namespace, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """语义压缩用:当前会话内未语义化的阶段摘要,最老在前(与可见层作用域无关,压缩始终按会话)。"""
        raise NotImplementedError

    @abstractmethod
    def get_first_message_timestamp(self, *, namespace: Namespace, cross_conversation: bool = True) -> int | None:
        """该 namespace 下最早一条消息的时间戳(供"认识第 N 天"等相处时间感)。无记录返回 None。"""
        raise NotImplementedError

    @abstractmethod
    def get_messages_by_time_range(
        self,
        *,
        namespace: Namespace,
        start_ts: int | None = None,
        end_ts: int | None = None,
        time_periods: list[str] | None = None,
        cross_conversation: bool = False,
    ) -> list[dict[str, Any]]:
        """按 UTC timestamp 范围精确读原始时间线(不走向量),最早在前。

        ``start_ts`` 包含、``end_ts`` 不包含。cross_conversation=False(默认):
        仅当前会话;True:该用户全部会话。time_periods 应已归一化。
        """
        raise NotImplementedError

    @abstractmethod
    def list_pending_index(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """修复扫描:列出 index_status=pending 的记录,供补做向量 upsert。"""
        raise NotImplementedError

    @abstractmethod
    def list_index_records(
        self, *, namespace: Namespace, limit: int | None = None, with_conversation: bool = False
    ) -> list[dict[str, Any]]:
        """列出可补建/热加载向量索引的三层记录。

        默认按硬隔离边界(tenant/user/domain)列出全部会话;with_conversation=True 时只列当前会话。
        limit 是全局安全上限,不是分页 cursor;批量重建需要另行设计 cursor。
        """
        raise NotImplementedError

    # --- 遗忘 / 合规 ---
    @abstractmethod
    def delete_namespace(self, *, namespace: Namespace) -> list[str]:
        """定向遗忘:删除某硬隔离边界下的全部记忆。返回被删的 source_id 列表(供同步清 VectorIndex)。"""
        raise NotImplementedError
