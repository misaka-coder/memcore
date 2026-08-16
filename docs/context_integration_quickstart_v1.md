# MemCore Context Quickstart V1

MemCore can decorate any session implementing the OpenAI Agents session shape:
`get_items`, `add_items`, `pop_item`, and `clear_session`. The host keeps its
existing session as the recovery store and passes its configured
`MemorySystem`. OpenAI Agents exposes the current input only through
`session_input_callback`, so that hook is part of the required integration.

```python
from agents import RunConfig
from memcore import MemCoreContextSession

session = MemCoreContextSession.wrap(existing_session, memory=memory_system)
run_config = RunConfig(session_input_callback=session.input_callback)
result = await Runner.run(agent, request, session=session, run_config=run_config)
```

The wrapper owns timeline entry creation, ordinary-message timestamps,
provider-native tool call/result preservation, projection replacement, and
structured degradation. A projection failure returns the original session
history; it never returns an empty history or claims a fake settlement.
When a populated host Session supplies `timestamp`, `created_at`, or
`createdAt`, the wrapper preserves that time evidence. Items without host time
metadata are stamped at import time rather than assigned a fabricated original
time.

The wrapper refuses construction when no `MemorySystem` is supplied or bound.
Its `namespace.conversation_id` must equal the wrapped Session ID; passing a
shared or differently scoped MemorySystem is rejected before any history read.
It never silently changes to storage-only mode. Calling `Runner.run` without
`session.input_callback` is plain session persistence, not authoritative
MemCore context ownership.

If an existing Session contains an item shape that MemCore cannot reproduce
byte-safely, `status.mode` becomes `degraded` and the wrapper serves the full
native Session history. It retries a full rebuild on a later read; it does not
drop only the unknown item or return a partially imported history.

## Harness

```ts
await ctx.plugin(MemCorePlugin, { mode: "authoritative" })
```

The plugin uses `session/event`, `agent/pre-step`, and the tool registry. The
bridge owns MemCore calls; the plugin does not format time, decide settlement,
or serialize provider tools.

## Capability levels

- `storage-only`: durable host history, no MemCore context authority.
- `memory-retrieval`: explicit retrieval and read tools, host owns history.
- `authoritative-context`: MemCore owns the model-visible sequence and passes
  the provider conformance kit.

The final level requires a request captured at the host transport boundary and
`validate_provider_wire_capture(...)` to pass. `validate_context_adapter(...)`
checks provider-shaped inputs only; it is not transport proof. System/persona prompts, provider
credentials, actual tool execution, and UI remain outside MemCore.
