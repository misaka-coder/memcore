"""Shared process runtime for compaction/index jobs and conversation locks."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

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
        self._compaction_executor.shutdown(wait=wait)
        self._index_executor.shutdown(wait=wait)


__all__ = ["ConversationLockRegistry", "MemCoreRuntime"]
