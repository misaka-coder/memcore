"""Optional, host-owned actual request prefix for internal memory calls.

Only the host can interpret ``transport``. It stays in memory and is never
serialized to the database, a prompt or an audit. Source positions describe
the effective request, not a second rendering of its transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .projection import ProjectionMessage, sanitize_projection_payload, stable_projection_hash


@dataclass(frozen=True)
class ChatContextSnapshot:
    provider_profile: str
    messages: tuple[dict[str, Any], ...] = field(repr=False)
    message_source_ids: tuple[tuple[str, ...], ...]
    transport: Any = field(repr=False, compare=False)
    pending_task_ids: tuple[str, ...] = ()
    working_token_limit: int = 0
    terminal_context_refresh: (
        Callable[["ChatContextSnapshot", tuple[ProjectionMessage, ...]], "ChatContextSnapshot | None"] | None
    ) = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.messages) != len(self.message_source_ids):
            raise ValueError("chat_context_source_positions_mismatch")

    def select(self, projections: list[ProjectionMessage], *, purpose: str) -> "ChatCompactionPrefix | None":
        """Use only an unchanged, fully covered range in this actual request."""
        required = {sid for message in projections for sid in message.source_ids}
        if not required or any(message.provider_profile != self.provider_profile for message in projections):
            return None
        expected = {(tuple(message.source_ids), stable_projection_hash(message.payload)) for message in projections}
        matched = []
        covered: set[str] = set()
        for index, (payload, ids) in enumerate(zip(self.messages, self.message_source_ids)):
            if not required.intersection(ids):
                continue
            persisted_payload, _ = sanitize_projection_payload(payload)
            if not set(ids) <= required or (ids, stable_projection_hash(persisted_payload)) not in expected:
                return None
            covered.update(ids)
            matched.append(index)
        if covered != required:
            return None
        first, last = min(matched), max(matched)
        # Unmapped stable/dynamic context can remain background; another raw
        # source inside the selected interval cannot be silently summarized.
        if any(ids and not set(ids) <= required for ids in self.message_source_ids[first : last + 1]):
            return None
        if not paired_tool_boundary(self.messages[: last + 1]):
            return None
        return ChatCompactionPrefix(self, first, last + 1, purpose)


@dataclass(frozen=True)
class ChatCompactionPrefix:
    snapshot: ChatContextSnapshot
    source_start: int
    prefix_end: int
    purpose: str

    def instruction(self, contract: str) -> str:
        first = self.snapshot.messages[self.source_start]
        calls = first.get("tool_calls") or []
        blocks = first.get("content")
        if isinstance(blocks, list):
            calls = [
                *calls,
                *[block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"],
            ]
        if first.get("type") == "function_call":
            calls = [{"id": first.get("call_id")}]
        cue = ""
        if calls:
            cue = " Start at tool call ID(s): " + ", ".join(str(call.get("id")) for call in calls) + "."
        elif isinstance(blocks, str) and blocks:
            import json

            cue = " The first selected message begins " + json.dumps(blocks[:120], ensure_ascii=False) + "."
        return (
            "Internal memory maintenance. Do not reply to the user or call tools. "
            f"Summarize only request messages {self.source_start + 1} through {self.prefix_end} "
            "(1-based, including the system message). Earlier messages are background, not new facts. "
            + cue
            + " The selected range ends immediately before this instruction. "
            + "Keep speakers, channels, dates, outcomes and unresolved work distinct. "
            "Describe pending work as of this range; later events may supersede it. "
            "Return only the following memory JSON contract; ignore the chat presentation contract.\n" + contract
        )


def paired_tool_boundary(messages: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> bool:
    """A native call batch can be cut only after every result has arrived."""
    pending: set[str] = set()
    for message in messages:
        item_type = str(message.get("type") or "")
        if item_type == "function_call":
            pending.add(str(message.get("call_id") or ""))
            continue
        if item_type == "function_call_output":
            call_id = str(message.get("call_id") or "")
            if call_id not in pending:
                return False
            pending.remove(call_id)
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            calls = list(message.get("tool_calls") or [])
            content = message.get("content")
            if isinstance(content, list):
                calls += [item for item in content if isinstance(item, dict) and item.get("type") == "tool_use"]
            if calls and pending:
                return False
            pending.update(str(call.get("id") or "") for call in calls)
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending:
                return False
            pending.remove(call_id)
        elif role == "user" and isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    call_id = str(block.get("tool_use_id") or "")
                    if call_id not in pending:
                        return False
                    pending.remove(call_id)
    return not pending
