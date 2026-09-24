"""Observe public MemCore calls without replacing host execution or responses."""

from __future__ import annotations

import copy
import threading
import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any, Iterator
from unittest.mock import patch

from .akane_transport import AkaneBudgetedTransport, AkaneTransportInterrupted
from .runner import digest


class LongRunTransport(AkaneBudgetedTransport):
    """Serialize physical requests and label background work with its owning run.

    Non-streaming is an explicit experiment condition. The lock lasts through
    usage settlement, so a background summary cannot collide with the ledger's
    single outstanding reservation. No provider request body is modified.
    """

    def __init__(self, *args: Any, run_id: str, **kwargs: Any) -> None:
        super().__init__(
            *args,
            input_token_budget=65536,
            provider_default_output_roles=("memcore_summary", "aux"),
            require_non_streaming=True,
            **kwargs,
        )
        self.run_id = run_id
        self.current_step = "startup"
        self._dispatch_lock = threading.RLock()

    def _send(self, original: Any, request: Any, args: tuple, kwargs: dict, role: str) -> Any:
        with self._dispatch_lock:
            if self._scope.get() == ("host", "background"):
                with self.scope(self.run_id, self.current_step):
                    return super()._send(original, request, args, kwargs, role)
            return super()._send(original, request, args, kwargs, role)


class PublicMemoryObserver:
    """Return the original values/Futures and retain only observation evidence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_indexes = 0
        self._futures: list[Any] = []
        self.compactions: list[dict[str, Any]] = []
        self.index_runs: list[dict[str, Any]] = []
        self.completions: list[dict[str, Any]] = []
        self.entries: list[dict[str, Any]] = []
        self.system: Any = None
        self._active = False

    @staticmethod
    def _entry(entry: Any) -> dict[str, Any]:
        return {
            "source_id": entry.source_id,
            "turn_id": entry.turn_id,
            "kind": entry.kind,
            "turn_role": entry.turn_role.value,
            "correlation_id": entry.correlation_id,
            "semantic_text_hash": digest(entry.semantic_text),
        }

    @contextmanager
    def installed(self) -> Iterator[PublicMemoryObserver]:
        from memcore import MemorySystem

        if self._active:
            raise RuntimeError("nested_public_memory_observer")
        self._active = True
        original_background = MemorySystem.compact_due_background
        original_index = MemorySystem.reindex_all
        original_complete = MemorySystem.complete_turn
        original_append = MemorySystem.append_entry

        @wraps(original_background)
        def background(system: Any, *args: Any, **kwargs: Any) -> Any:
            row: dict[str, Any] = {"status": "running", "provider_profile": kwargs.get("provider_profile", "")}
            future = original_background(system, *args, **kwargs)
            with self._lock:
                self._futures.append(future)
                self.compactions.append(row)

            def complete(done: Any) -> None:
                try:
                    stats = done.result()
                    safe = copy.deepcopy(stats)
                    error = None
                except BaseException as exc:
                    safe, error = {}, type(exc).__name__
                with self._lock:
                    row.update(status="finished" if error is None else "failed", stats=safe, error=error)

            future.add_done_callback(complete)
            return future

        @wraps(original_index)
        def reindex(system: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            row: dict[str, Any] = {"status": "running", "batch_size": kwargs.get("batch_size", 64)}
            with self._lock:
                self._active_indexes += 1
                self.index_runs.append(row)
            try:
                result = original_index(system, *args, **kwargs)
                row.update(status="finished", result=copy.deepcopy(result))
                return result
            except BaseException as exc:
                row.update(status="failed", error=type(exc).__name__)
                raise
            finally:
                with self._lock:
                    row["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
                    self._active_indexes -= 1

        @wraps(original_complete)
        def completion(system: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_complete(system, *args, **kwargs)
            self.system = system
            row = {
                "completed": result.completed,
                "status": result.status,
                "turn_id": result.turn_id,
                "final_entry": self._entry(result.final_entry) if result.final_entry else None,
                "submitted_speech": str(kwargs.get("semantic_text", "")),
                "final_projection": dict(result.final_projection.payload) if result.final_projection else None,
            }
            with self._lock:
                self.completions.append(row)
            return result

        @wraps(original_append)
        def append(system: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_append(system, *args, **kwargs)
            record = self._entry(result)
            if record["kind"] == "tool.lookup_fixture.result":
                opened = system.open_memory(memory_id=result.source_id, view="content")
                if opened.get("status") != "ok" or not isinstance(opened.get("result", {}).get("content"), str):
                    raise AkaneTransportInterrupted("initial_public_fixture_content_read_failed")
                record["initial_public_content_hash"] = digest(opened["result"]["content"])
            with self._lock:
                self.entries.append(record)
            return result

        try:
            with ExitStack() as stack:
                for name, wrapper in (
                    ("compact_due_background", background),
                    ("reindex_all", reindex),
                    ("complete_turn", completion),
                    ("append_entry", append),
                ):
                    stack.enter_context(patch.object(MemorySystem, name, wrapper))
                yield self
        finally:
            self._active = False

    def wait_idle(self, *, timeout: float = 120.0) -> dict[str, int]:
        """Wait between synthetic user turns; do not block Engine's speech return.

        Observe public Futures and index-call completion. A short quiet period
        also permits Akane's normal Future callbacks to submit coalesced work.
        """
        deadline, quiet_since = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            with self._lock:
                idle = not self._active_indexes and all(future.done() for future in self._futures)
                idle = idle and all(row["status"] != "running" for row in self.compactions)
                signature = (len(self._futures), len(self.index_runs))
            if idle:
                if quiet_since is None or quiet_since[1] != signature:
                    quiet_since = (time.monotonic(), signature)
                elif time.monotonic() - quiet_since[0] >= 0.1:
                    return {"observed_compaction_jobs": signature[0], "observed_index_jobs": signature[1]}
            else:
                quiet_since = None
            time.sleep(0.02)
        raise AkaneTransportInterrupted("long_experiment_background_wait_timeout")

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(
                {"compactions": self.compactions, "index_runs": self.index_runs, "completions": self.completions}
            )
