# Context Surface Contract V1

`context_surface_v1` is the public boundary between a host agent loop and
MemCore. It is provider-ready data, not a database record and not a settlement
planner input.

## Public Build

```python
surface = memory.build_context_surface(
    session_id=session_id,
    provider_profile="deepseek_chat",
    current_source_id=current_source_id,
    active_turn_messages=active_turn_messages,
)
```

The host supplies a current source ID and any provider-native messages from the
still-open tool round. It does not supply a projection generation, timestamp
prefix, event renderer, or card decision.

## Stable Shape

```json
{
  "version": "context_surface_v1",
  "provider_profile": "deepseek_chat",
  "history_messages": [],
  "current_message": {},
  "active_turn_messages": [],
  "projection_hash": "sha256 hex",
  "projection_generation": 0,
  "projection_version": 1,
  "compaction_generation": 0,
  "message_source_ids": [],
  "message_projection_metadata": [],
  "has_compact_history": false,
  "current_turn_id": "",
  "diagnostics": []
}
```

`history_messages` contains closed MemCore history. `current_message` is the
single current stimulus and is omitted from history. `active_turn_messages`
contains the open provider-native tool round. `surface.messages` is the exact
ordered provider sequence: history, current message, then active round.
`message_source_ids` is aligned one-for-one with `surface.messages`; it lets a
host retain attachment/provenance links without reading the private projection
ledger.

`message_projection_metadata` is also aligned one-for-one with
`surface.messages`. Each item carries only the stable request-freeze identity:
`turn_id`, `source_ids`, `projection_index`, `projection_status`, and
`projection_version`. A host that records the actual provider request must pass
these values back with the provider-visible payload. It must not invent a
default projection version or derive an index from the whole conversation.
Support is advertised by
`CONTEXT_SURFACE_MESSAGE_METADATA_VERSION=context_surface_message_metadata_v1`.

The host owns persona/system/developer content and may prepend its own stable
system prompt. MemCore owns the message sequence inside this contract.

## Rendering Rules

Ordinary timeline messages are rendered as:

```text
[2026-08-16 14:24] User: 正文
```

Structured entries remain their kind:

```text
[2026-08-16 周日 14:24 | 下午] event.finance
source: 东方财富
title: ...
summary: ...
```

They are never rewritten as `context.inject`. Provider-native frozen messages
are replayed without a second time wrapper. The current message appears once.

Tool calls preserve the arguments JSON string and call ID. Tool results preserve
the result body, empty body, and failure status. An open tool round is never
settled before the host closes it. A completed operation may become a
`[compact_reloadable]` card according to MemCore's frozen no-expansion policy;
the card's `open_memory(...)` arguments are accepted directly by the official
tool schema and return the stored bytes without re-running the tool.

## Profiles

| Profile | Wire family |
| --- | --- |
| `openai_chat` | `assistant.tool_calls` and `role=tool` |
| `openai_responses` | `function_call` and `function_call_output` |
| `anthropic_messages` | `tool_use` and `tool_result` blocks |
| `deepseek_chat` | OpenAI-compatible chat tool calls/results |

MemCore maintains provider-ready context shapes and provider-neutral
`NormalizedContextEvent` adapters. The host still owns its SDK serializer and
must prove the final transport boundary with
`validate_provider_wire_capture(surface, captured_context_messages)`.

## Diagnostics And Fallback

Diagnostics contain stable status/reason codes only. They must not contain API
keys, local database/cache paths, runtime log paths, or full private storage
records. Projection/settlement failures are structured and leave the host's
full history usable; they do not produce an empty history or a fake card.
