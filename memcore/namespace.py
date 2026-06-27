"""命名空间五层模型 + Actor(稳定 ID / 显示名分离)。

见设计文档 §10。

- **硬隔离键**:tenant_id / user_id / domain_id —— 检索 where 强制过滤,跨它检索不到
  ("A 绝不会搜到 B 的私密记忆"的保证)。
- **会话键**:conversation_id —— 决定"可见三层窗口"取哪一段;检索仍可跨会话捞同一 user 历史。
- **软标签**:actor —— 群聊里"谁说的",做软区分,不做硬隔离。

⚠️ Actor 必须用平台**稳定 ID**(QQ uin / uid,永不变),群昵称是**可变显示名**,二者分字段,绝不合一。
否则发言人改名后记忆会对不上、甚至串到别人头上。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import NamespaceError


@dataclass(frozen=True)
class Actor:
    """群聊/多方场景里的发言人。stable_id 焊死,display_name 可变。"""

    stable_id: str
    display_name: str = ""

    def __post_init__(self) -> None:
        if not str(self.stable_id or "").strip():
            raise NamespaceError("Actor.stable_id is required and must be a stable platform id, not a nickname")

    def with_display_name(self, name: str) -> "Actor":
        """改名只换显示名,stable_id 不变(归并/隔离始终认 stable_id)。"""
        return Actor(stable_id=self.stable_id, display_name=str(name or "").strip())


@dataclass(frozen=True)
class Namespace:
    """记忆的隔离与作用域坐标。

    user_id 必填;tenant_id / domain_id 多租户/多领域时填;conversation_id 管上下文窗口。
    actor 仅群聊等多方场景需要。
    """

    user_id: str
    tenant_id: str = ""
    domain_id: str = ""
    conversation_id: str = ""
    actor: Actor | None = None

    def __post_init__(self) -> None:
        if not str(self.user_id or "").strip():
            raise NamespaceError("Namespace.user_id is required (hard-isolation key)")

    def hard_key(self) -> tuple[str, str, str]:
        """硬隔离键:跨此键的记忆互不可见。检索 where 必须按它过滤。"""
        return (str(self.tenant_id or ""), str(self.user_id), str(self.domain_id or ""))

    def actor_id(self) -> str:
        """软标签:发言人稳定 ID(无 actor 时为空)。"""
        return self.actor.stable_id if self.actor is not None else ""
