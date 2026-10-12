"""Shared process runtime for compaction/index jobs and conversation locks."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterator

from .namespace import Namespace


class ConversationLockRegistry:
    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._locks: dict[tuple[str, str, str, str, str], threading.RLock] = {}

    def lock_for(self, *, store_identity: str, namespace: Namespace) -> threading.RLock:
        key = (
            str(store_identity or ""),
            namespace.tenant_id or "",
            namespace.user_id,
            namespace.domain_id or "",
            namespace.conversation_id or "",
        )
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock


class MemCoreRuntime:
    """Executors are shared by hosts; only the creator/owner should close them."""

    def __init__(self, *, compaction_workers: int = 2, index_workers: int = 1) -> None:
        if int(compaction_workers) < 1 or int(index_workers) < 1:
            raise ValueError("runtime worker counts must be positive")
        self.lock_registry = ConversationLockRegistry()
        self._compaction_executor = ThreadPoolExecutor(
            max_workers=int(compaction_workers),
            thread_name_prefix="memcore-compact",
        )
        self._index_executor = ThreadPoolExecutor(
            max_workers=int(index_workers),
            thread_name_prefix="memcore-index",
        )
        self._closed = False
        self._guard = threading.RLock()
        self._completing_turns: dict[tuple[str, str, str, str, str], dict[str, int]] = {}

    @contextmanager
    def completing_turn(self, *, store_identity: str, namespace: Namespace, turn_id: str) -> Iterator[None]:
        """Exclude a turn from ordinary planning until its completion/settlement finishes.

        This is a process-local write boundary, not a queue or a conversation
        lock. Other completed history and active tool work remain available.
        """
        key = (store_identity, *namespace.hard_key(), namespace.conversation_id or "")
        with self._guard:
            active = self._completing_turns.setdefault(key, {})
            active[turn_id] = active.get(turn_id, 0) + 1
        try:
            yield
        finally:
            with self._guard:
                active[turn_id] -= 1
                if not active[turn_id]:
                    active.pop(turn_id)
                if not active:
                    self._completing_turns.pop(key)

    def completing_turn_ids(self, *, store_identity: str, namespace: Namespace) -> frozenset[str]:
        key = (store_identity, *namespace.hard_key(), namespace.conversation_id or "")
        with self._guard:
            return frozenset(self._completing_turns.get(key, ()))

    def submit_compaction(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        with self._guard:
            if self._closed:
                raise RuntimeError("memcore_runtime_closed")
            return self._compaction_executor.submit(fn, *args, **kwargs)

    def submit_index_repair(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        with self._guard:
            if self._closed:
                raise RuntimeError("memcore_runtime_closed")
            return self._index_executor.submit(fn, *args, **kwargs)

    def close(self, *, wait: bool = True) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
        self._compaction_executor.shutdown(wait=wait, cancel_futures=True)
        self._index_executor.shutdown(wait=wait, cancel_futures=True)


__all__ = ["ConversationLockRegistry", "MemCoreRuntime"]
