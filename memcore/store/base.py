"""MemoryStore 接口 —— 记录侧(关系型真相源)。

向量库只做语义索引;真相、原文、时间、字段全在这里。检索拿到 source_id 后回这里取原文、做上下文扩窗。
默认实现 SQLiteMemoryStore 在后续切片提供;此处只钉接口。

写入一致性见设计文档 §B:用 outbox 模式(index_status: pending → indexed + 修复扫描),
不做 SQLite/向量跨库事务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..namespace import Namespace


class MemoryStore(ABC):
    # --- 写 ---
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
        """outbox 状态机:pending / indexed(向量 upsert 成功后置 indexed)。"""
        raise NotImplementedError

    # --- 读(可见三层窗口 + 检索回取) ---
    @abstractmethod
    def get_record_by_source_id(self, source_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def get_context_slice(self, *, namespace: Namespace, seq_no: int, window: int) -> list[dict[str, Any]]:
        """raw 上下文扩窗:取目标消息前后 window 条邻居。"""
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
    def get_messages_by_date_range(
        self,
        *,
        namespace: Namespace,
        date_from: str = "",
        date_to: str = "",
        time_periods: list[str] | None = None,
        cross_conversation: bool = False,
    ) -> list[dict[str, Any]]:
        """按时间精确读原始对话(时间线工具,不走向量)。date_label 用 YYYY-MM-DD,最早在前。

        cross_conversation=False(默认):仅当前会话;True:该用户全部会话。time_periods 应已归一化。
        """
        raise NotImplementedError

    @abstractmethod
    def list_pending_index(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """修复扫描:列出 index_status=pending 的记录,供补做向量 upsert。"""
        raise NotImplementedError

    # --- 遗忘 / 合规 ---
    @abstractmethod
    def delete_namespace(self, *, namespace: Namespace) -> list[str]:
        """定向遗忘:删除某硬隔离边界下的全部记忆。返回被删的 source_id 列表(供同步清 VectorIndex)。"""
        raise NotImplementedError
