# Host-neutral action/result timeline

MemCore can keep model actions and host/environment results in the same append-only
timeline as messages and events. This is a mechanism, not a required tool system.

The host remains free to:

- keep a small stable tool schema in the system prompt;
- expose every stable tool directly;
- let the model load a catalog or Skill on demand;
- use provider-native tool calls, JSON, XML, tags, or another protocol;
- record all, some, or none of those intermediate exchanges.

MemCore does not parse arbitrary model syntax, choose permissions, register tools,
or execute capabilities. It stores what the host has already recognized as having
happened, preserves request/result relations, and projects that history back to the
model deterministically.

## Small public API

An open turn can append a model action and one or more environment observations:

```python
handle = mem.begin_turn(stimuli=[current_message], turn_id="turn-1")

mem.append_action(
    turn_id=handle.turn_id,
    kind="operation.catalog.request",
    correlation_id="catalog-1",
    semantic_text='{"action":"list_capabilities"}',
    payload={"action": "list_capabilities"},
)

mem.append_observation(
    turn_id=handle.turn_id,
    kind="operation.catalog.response",
    correlation_id="catalog-1",
    semantic_text="search and weather are available",
    payload={"items": ["search", "weather"]},
    status="success",
)
```

These are thin wrappers around `append_entry(TimelineEntryInput(...))`. Advanced
hosts can continue using the typed entry API directly. The wrappers do not require
the `operation.*` prefix; any valid namespaced `kind` is allowed.

`correlation_id` is the relation key. Several actions can be appended in one model
round, and observations may arrive out of order. A terminal observation closes its
own branch; statuses `open`, `pending`, `running`, and `streaming` keep that branch
open. `complete_turn()` returns `pending_actions` rather than pretending unfinished
branches succeeded.

## Native and non-native model protocols

For standard OpenAI/Anthropic projection profiles, an `action`/`observation` pair
has a deterministic provider-native tool-call/tool-result fallback.

If the host actually sent a different representation, such as JSON or XML-like
tags, pass those exact provider messages to `record_request_projection()`. The
projection ledger freezes the real representation instead of silently rewriting it
into the fallback:

```python
actual_history = [
    {"role": "user", "content": "列出能力"},
    {"role": "assistant", "content": '<action name="list" id="catalog-1" />'},
    {"role": "user", "content": '<result id="catalog-1">search,weather</result>'},
]

mem.record_request_projection(
    turn_id=handle.turn_id,
    provider_profile="openai_chat",
    turn_messages=[
        ProjectionMessageInput(
            provider_profile="openai_chat",
            payload=payload,
            source_ids=(source_id,),
            projection_index=index,
        )
        for index, (payload, source_id) in enumerate(
            zip(actual_history, ("user-1", "action-1", "observation-1"))
        )
    ],
    history_messages=actual_history,
)
```

See [`../examples/non_native_operation_timeline.py`](../examples/non_native_operation_timeline.py)
for a runnable end-to-end example.

## Three independent decisions

Recording an entry does not automatically make it ordinary long-term memory:

- `prompt_visible`: whether unsummarized raw appears in subsequent model context;
- `retrieval_visibility`: normal, explicit-only, or never retrievable;
- `semanticize`: whether it may become long-term semantic memory.

The action/result helpers use conservative defaults: visible in recent raw,
explicit-only retrieval, and no semanticization. Hosts can use
`TimelineEntryInput` directly when they need another policy. Unknown kinds still
receive a safe canonical renderer; no business-type enum change is required.

## Optional retention anchors

Operation payloads are usually transient. Raw compaction therefore keeps a small
operation digest instead of copying complete tool inputs/results indefinitely.

When a later model turn may need to identify or reload something, an entry can opt
in to a small structured `retention_anchor`:

```python
mem.append_observation(
    turn_id=handle.turn_id,
    kind="operation.catalog.response",
    correlation_id="catalog-1",
    semantic_text="catalog loaded",
    payload={"items": [...]},
    status="success",
    retention_anchor={
        "catalog_ref": "catalog:v3",
        "schema_hash": "sha256:abc",
        "result_ref": "result:42",
    },
)
```

The anchor is deliberately not a second result store:

- keys are host-defined; MemCore does not require capability-specific fields;
- each anchor is a JSON object capped at 4096 encoded bytes;
- API keys, secret-looking values, local absolute paths, binary/media content, and
  unsafe values are replaced before persistence;
- one operation digest retains at most 16384 encoded bytes of full anchors;
  overflow is represented by a structured status, hash, and byte count;
- entries without an anchor preserve the old compact behavior exactly.

After compaction, retained anchors live in the explicit
`memory.operation_digest` summary and remain outside ordinary conversational
retrieval/semantic memory. A host that wants the model to inspect old operation
anchors can authorize an explicit retrieval for `memory.operation_digest`; a host
that can cheaply reload the catalog may simply execute the loader again.

Do not put full files, large tool outputs, credentials, or local paths in an
anchor. Keep those in host storage and retain a stable resource ID, version, hash,
or other small lookup key.

## Cache behavior

Timeline truth is append-only. With the default
`operation_projection_policy="full_until_raw_compaction"`, once an actual request
projection has been frozen, later turns reuse it byte-for-byte until raw compaction
or migration replaces that history segment. Adding a new action/result therefore
appends a suffix; it does not require changing the stable system prompt or all
previously projected messages.

The optional `compact_after_terminal` policy introduces one controlled projection
transition after the assistant final commits: large observation bodies become
deterministic reloadable cards while action parameters, final speech, provider
extensions, and SQLite truth remain unchanged. The settled payload set is then
frozen and reused on restart. The current open tool loop is never settled.

Hosts enabling this policy must add one stable model rule explaining that a compact
`source_id` can be reopened with `open_memory(view="content")`; do not inject that
rule dynamically only after the first compact result appears. Exact configuration,
status, metrics, batch readback, migration, and rollback contracts are authoritative
in [`operation_projection_settlement_v1.md`](operation_projection_settlement_v1.md).

This makes dynamic catalogs and loaded instructions possible without requiring
them. A host with a small stable tool set can still keep all tool descriptions in
the system prompt, which is often the simplest design.
