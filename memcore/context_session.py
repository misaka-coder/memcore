"""Small Session-protocol decorator for MemCore-owned context history.

The decorator deliberately speaks the four operations used by the OpenAI
Agents ``Session`` protocol without importing that optional package.  A host
can attach a ``MemorySystem`` as ``memory_system``/``memcore`` on its existing
session, or pass it through the required ``memory=`` argument.  The original
session remains the recovery store: a projection failure never turns history
into an empty list.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .context_contract import ContextDiagnostic
from .memory_system import MemorySystem
from .projection import ProjectionMessageInput
from .timeline import EntryOrigin, TimelineEntryInput, TurnRole


def _stable_item_id(session_id: str, item: Mapping[str, Any], index: int) -> str:
    material = json.dumps(dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{session_id}\x00{index}\x00{material}".encode("utf-8")).hexdigest()[:24]
    return f"session-{digest}"


def _item_fingerprint(item: Mapping[str, Any]) -> str:
    material = json.dumps(dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _item_timestamp(item: Mapping[str, Any]) -> int:
    """Use host time evidence when available; otherwise use import time."""

    for name in ("timestamp", "created_at", "createdAt"):
        value = item.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            return int(numeric / 1000 if numeric > 10_000_000_000 else numeric)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            try:
                numeric = float(text)
                return int(numeric / 1000 if numeric > 10_000_000_000 else numeric)
            except ValueError:
                try:
                    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    continue
    return int(time.time())


async def _call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    value = method(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


def _role(item: Mapping[str, Any]) -> str:
    return str(item.get("role") or "").strip().lower()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if str(block.get("type") or "") in {"text", "input_text"}:
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _tool_calls(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = item.get("tool_calls")
    if isinstance(value, list):
        return [call for call in value if isinstance(call, Mapping)]
    if str(item.get("type") or "") == "function_call":
        return [item]
    content = item.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, Mapping) and block.get("type") == "tool_use"]
    return []


def _tool_result(item: Mapping[str, Any]) -> tuple[str, str] | None:
    role = _role(item)
    item_type = str(item.get("type") or "")
    if role == "tool":
        return str(item.get("tool_call_id") or ""), _text(item.get("content"))
    if item_type == "function_call_output":
        return str(item.get("call_id") or ""), _text(item.get("output"))
    content = item.get("content")
    if role == "user" and isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_result":
                return str(block.get("tool_use_id") or ""), _text(block.get("content"))
    return None


def _call_parts(call: Mapping[str, Any]) -> tuple[str, str, str]:
    item_type = str(call.get("type") or "")
    if item_type == "function_call":
        return (
            str(call.get("call_id") or call.get("id") or ""),
            str(call.get("name") or ""),
            str(call.get("arguments") or "{}"),
        )
    function = call.get("function")
    if isinstance(function, Mapping):
        return (
            str(call.get("id") or ""),
            str(function.get("name") or ""),
            str(function.get("arguments") or "{}"),
        )
    return (
        str(call.get("id") or ""),
        str(call.get("name") or ""),
        json.dumps(
            call.get("input") if call.get("input") is not None else {}, ensure_ascii=False, separators=(",", ":")
        ),
    )


@dataclass(frozen=True)
class ContextSessionStatus:
    mode: str
    diagnostics: tuple[ContextDiagnostic, ...] = ()


class MemCoreContextSession:
    """An OpenAI-Agents-style session decorator with MemCore as context owner.

    ``get_items`` returns the MemCore provider surface when a MemorySystem is
    bound. ``add_items`` mirrors the normal runner persistence calls into one
    hidden MemCore turn lifecycle, including native tool calls/results.  The
    wrapped session is always updated first and is used as the complete-history
    fallback when MemCore is unavailable.
    """

    def __init__(
        self,
        existing_session: Any,
        *,
        memory: MemorySystem | None = None,
        provider_profile: str = "openai_responses",
        timezone: str = "Asia/Shanghai",
        session_id: str = "",
    ) -> None:
        if existing_session is None:
            raise TypeError("existing_session is required")
        try:
            ZoneInfo(str(timezone or "").strip())
        except Exception as exc:
            raise ValueError("invalid timezone") from exc
        resolved_memory = memory or self._discover_memory(existing_session)
        if not isinstance(resolved_memory, MemorySystem):
            raise TypeError("memory must be a MemorySystem; authoritative wrapping cannot be storage-only")
        self.underlying_session = existing_session
        self.memory = resolved_memory
        self.provider_profile = str(provider_profile or "").strip()
        if not self.provider_profile:
            raise ValueError("provider_profile is required")
        self.timezone = str(timezone).strip()
        self.session_id = str(session_id or getattr(existing_session, "session_id", "") or "session").strip()
        memory_conversation_id = str(resolved_memory.namespace.conversation_id or "").strip()
        if memory_conversation_id != self.session_id:
            raise ValueError("memory namespace conversation_id must match the wrapped session_id")
        self.session_settings = getattr(existing_session, "session_settings", None)
        self._active_turn_id = ""
        self._current_source_id = ""
        self._item_index = 0
        self._diagnostics: list[ContextDiagnostic] = []
        self._wire: list[dict[str, Any]] = []
        self._prepared_item_fingerprints: list[str] = []
        self._history_prefix: list[dict[str, Any]] = []
        self._bootstrapped = False
        self._native_fallback_reason = ""

    @classmethod
    def wrap(
        cls,
        existing_session: Any,
        *,
        memory: MemorySystem | None = None,
        provider_profile: str = "openai_responses",
        timezone: str = "Asia/Shanghai",
        session_id: str = "",
    ) -> "MemCoreContextSession":
        """Wrap a host session; a bound or explicit MemorySystem is required."""

        return cls(
            existing_session,
            memory=memory,
            provider_profile=provider_profile,
            timezone=timezone,
            session_id=session_id,
        )

    @staticmethod
    def _discover_memory(session: Any) -> MemorySystem | None:
        for name in ("memory_system", "memcore", "memory"):
            candidate = getattr(session, name, None)
            if isinstance(candidate, MemorySystem):
                return candidate
        return None

    @property
    def status(self) -> ContextSessionStatus:
        return ContextSessionStatus(
            mode="degraded" if self._native_fallback_reason else "authoritative",
            diagnostics=tuple(self._diagnostics),
        )

    async def input_callback(
        self,
        history_items: list[Mapping[str, Any]],
        new_items: list[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """OpenAI Agents ``session_input_callback`` with MemCore as request authority.

        The Agents runner reads session history before it exposes the current
        input.  This callback is therefore the required request-boundary hook:
        it records the current input, builds the authoritative surface, and
        returns the exact sequence used for the first provider request.
        """

        del history_items
        base_history = [dict(item) for item in await self.get_items() if isinstance(item, Mapping)]
        normalized = [dict(item) for item in new_items if isinstance(item, Mapping)]
        if not normalized:
            return base_history
        if self._native_fallback_reason:
            return [*base_history, *normalized]
        try:
            if self._active_turn_id:
                self.memory.abort_turn(self._active_turn_id, reason="new_request_before_previous_turn_closed")
                self._reset_active_turn()
            for item in normalized:
                self._ingest(item)
                self._prepared_item_fingerprints.append(_item_fingerprint(item))
            surface = self.memory.build_context_surface(
                session_id=self.session_id,
                provider_profile=self.provider_profile,
                current_source_id=self._current_source_id or None,
            )
            if surface.diagnostics:
                self._diagnostics.extend(surface.diagnostics)
            self._history_prefix = [dict(item) for item in surface.history_messages]
            if surface.current_message is not None and self._wire:
                self._wire[0]["payload"] = dict(surface.current_message)
            self._freeze_request_wire(history_messages=[dict(item) for item in surface.messages])
            return [dict(item) for item in surface.messages]
        except Exception:
            self._degrade_to_native("context_surface_unavailable", clear_namespace=False)
            return [*base_history, *normalized]

    async def get_items(self, limit: int | None = None) -> list[Mapping[str, Any]]:
        full_fallback = list(await _call(self.underlying_session.get_items))
        fallback = full_fallback[-limit:] if limit is not None else full_fallback
        try:
            if self._native_fallback_reason:
                self._rebuild_from_host(full_fallback)
            elif not self._bootstrapped:
                initial_surface = self.memory.build_context_surface(
                    session_id=self.session_id,
                    provider_profile=self.provider_profile,
                )
                if not initial_surface.messages and full_fallback:
                    self._rebuild_from_host(full_fallback)
                else:
                    self._bootstrapped = True
            surface = self.memory.build_context_surface(
                session_id=self.session_id,
                provider_profile=self.provider_profile,
                current_source_id=self._current_source_id or None,
            )
            if surface.diagnostics:
                self._diagnostics.extend(surface.diagnostics)
            items = [dict(item) for item in surface.messages]
            return items[-limit:] if limit is not None else items
        except Exception:
            self._degrade_to_native(
                self._native_fallback_reason or "context_surface_unavailable",
                clear_namespace=False,
            )
            return fallback

    async def add_items(self, items: list[Mapping[str, Any]]) -> None:
        normalized = [dict(item) for item in items if isinstance(item, Mapping)]
        await _call(self.underlying_session.add_items, normalized)
        if self._native_fallback_reason:
            return
        for item in normalized:
            fingerprint = _item_fingerprint(item)
            if self._prepared_item_fingerprints and self._prepared_item_fingerprints[0] == fingerprint:
                self._prepared_item_fingerprints.pop(0)
                continue
            try:
                self._ingest(item)
            except Exception:
                self._degrade_to_native("session_item_ingest_failed", clear_namespace=False)
                break

    async def pop_item(self) -> Mapping[str, Any] | None:
        item = await _call(self.underlying_session.pop_item)
        if item is None:
            return None
        remaining = list(await _call(self.underlying_session.get_items))
        try:
            self._rebuild_from_host(remaining)
            self._diagnostics.append(ContextDiagnostic("ok", "session_pop_rebuilt_memcore"))
        except Exception:
            self._degrade_to_native("session_pop_rebuild_failed")
        return item

    async def clear_session(self) -> None:
        await _call(self.underlying_session.clear_session)
        self.memory.forget_namespace()
        self._reset_active_turn()
        self._prepared_item_fingerprints = []
        self._item_index = 0
        self._bootstrapped = True
        self._native_fallback_reason = ""

    def _reset_active_turn(self) -> None:
        self._active_turn_id = ""
        self._current_source_id = ""
        self._wire = []
        self._history_prefix = []

    def _degrade_to_native(self, reason: str, *, clear_namespace: bool = True) -> None:
        if self._active_turn_id:
            try:
                self.memory.abort_turn(self._active_turn_id, reason=reason)
            except Exception:
                pass
        self._reset_active_turn()
        self._prepared_item_fingerprints = []
        if clear_namespace:
            try:
                self.memory.forget_namespace()
            except Exception:
                pass
            self._item_index = 0
            self._bootstrapped = False
        self._native_fallback_reason = str(reason or "native_history_fallback")
        self._diagnostics.append(ContextDiagnostic("degraded", self._native_fallback_reason))

    def _rebuild_from_host(self, items: list[Any]) -> None:
        self.memory.forget_namespace()
        self._reset_active_turn()
        self._item_index = 0
        self._prepared_item_fingerprints = []
        self._bootstrapped = False
        self._native_fallback_reason = ""
        try:
            for item in items:
                if not isinstance(item, Mapping):
                    raise TypeError("session_item_not_object")
                self._ingest(dict(item))
        except Exception:
            try:
                self.memory.forget_namespace()
            finally:
                self._reset_active_turn()
                self._item_index = 0
                self._native_fallback_reason = "session_history_rebuild_failed"
            raise
        self._bootstrapped = True

    def _ingest(self, item: Mapping[str, Any]) -> None:
        item_id = _stable_item_id(self.session_id, item, self._item_index)
        self._item_index += 1
        role = _role(item)
        item_type = str(item.get("type") or "")
        if role == "user" and _tool_result(item) is None:
            if self._active_turn_id:
                self.memory.abort_turn(self._active_turn_id, reason="new_user_before_previous_turn_closed")
            text = _text(item.get("content"))
            handle = self.memory.begin_turn(
                stimuli=[
                    TimelineEntryInput(
                        source_id=item_id,
                        kind="message.user",
                        origin=EntryOrigin.USER,
                        turn_role=TurnRole.STIMULUS,
                        semantic_text=text,
                        payload=dict(item),
                        timestamp=_item_timestamp(item),
                        compatibility_role="user",
                    )
                ],
                turn_id=f"{self.session_id}::{item_id}",
            )
            self._active_turn_id = handle.turn_id
            self._current_source_id = item_id
            self._wire = [{"payload": dict(item), "source_ids": [item_id]}]
            return
        result = _tool_result(item)
        if result is not None and result[0]:
            if not self._active_turn_id:
                return
            call_id, output = result
            self.memory.append_observation(
                turn_id=self._active_turn_id,
                kind="tool.result",
                correlation_id=call_id,
                semantic_text=output,
                payload={"output": output, "status": "error" if item.get("is_error") else "success"},
                source_id=item_id,
                status="error" if item.get("is_error") else "success",
                trace_metadata={"tool_name": "tool", "status": "error" if item.get("is_error") else "success"},
            )
            self._wire.append({"payload": dict(item), "source_ids": [item_id]})
            return
        calls = _tool_calls(item)
        if calls and (role == "assistant" or item_type == "function_call"):
            if not self._active_turn_id:
                return
            for call in calls:
                call_id, name, arguments = _call_parts(call)
                if not call_id:
                    continue
                self.memory.append_action(
                    turn_id=self._active_turn_id,
                    kind="tool.call",
                    correlation_id=call_id,
                    semantic_text=arguments,
                    payload={"name": name, "arguments": arguments},
                    source_id=f"{item_id}-{call_id}",
                    trace_metadata={"tool_name": name},
                )
            self._wire.append(
                {
                    "payload": dict(item),
                    "source_ids": [f"{item_id}-{_call_parts(call)[0]}" for call in calls if _call_parts(call)[0]],
                }
            )
            return
        if role == "assistant" or (item_type == "message" and role == "assistant"):
            if not self._active_turn_id:
                return
            self._freeze_request_wire()
            self.memory.complete_turn(
                turn_id=self._active_turn_id,
                semantic_text=_text(item.get("content")),
                provider_output_raw=_text(item.get("content")),
                source_id=item_id,
                provider_profile=self.provider_profile,
                provider_projection=dict(item),
            )
            self._reset_active_turn()
            return
        raise ValueError("unsupported_or_orphan_session_item")

    def _freeze_request_wire(self, *, history_messages: list[dict[str, Any]] | None = None) -> None:
        """Freeze the provider-native open round before the next model call."""

        if not self._active_turn_id or not self._wire:
            return
        wire = [
            {
                "payload": dict(item.get("payload") or {}),
                "source_ids": tuple(str(source_id) for source_id in item.get("source_ids") or ()),
            }
            for item in self._wire
        ]
        try:
            surface = self.memory.build_context_surface(
                session_id=self.session_id,
                provider_profile=self.provider_profile,
                current_source_id=self._current_source_id or None,
            )
            if surface.current_message is not None and wire:
                wire[0]["payload"] = dict(surface.current_message)
            prepared = [
                ProjectionMessageInput(
                    provider_profile=self.provider_profile,
                    payload=dict(item["payload"]),
                    source_ids=tuple(item["source_ids"]),
                    projection_index=index,
                )
                for index, item in enumerate(wire)
            ]
            self.memory.record_request_projection(
                turn_id=self._active_turn_id,
                provider_profile=self.provider_profile,
                turn_messages=prepared,
                history_messages=(
                    [dict(item) for item in history_messages]
                    if history_messages is not None
                    else [*self._history_prefix, *(dict(item["payload"]) for item in wire)]
                ),
            )
        except Exception:
            self._diagnostics.append(ContextDiagnostic("degraded", "request_wire_freeze_failed"))


__all__ = ["ContextSessionStatus", "MemCoreContextSession"]
