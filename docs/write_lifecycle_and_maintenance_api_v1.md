# Write Lifecycle and Maintenance API V1

This document is the authoritative public reference for writing turns, events,
operations, metadata, and maintenance state through `MemorySystem`. Normal host
integrations must not write directly to `MemoryStore` tables.

## Successful completion without a new response

After actions finish, a host can end a turn without asking the model for an
additional response:

```python
completed = mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text="",
    provider_output_raw="",
    append_final=False,
    provider_profile="openai_chat",
)
```

This atomically commits target annotations and closes the turn. Pending actions
still block completion. No assistant entry or placeholder projection is created;
`final_entry` is `None`. Tool pairs stay readable, and terminal settlement and raw
compaction use the normal lifecycle. Retrying a closed turn is idempotent.
Nonempty response text or a provider projection conflicts with `append_final=False`.
The default remains `True`; a real model-authored silent envelope is still real
output and should be saved normally.

## Failure model

MemCore uses two failure forms intentionally:

- malformed configuration, contract objects, namespace ownership, or illegal
  lifecycle transitions raise a typed exception such as `ConfigError`,
  `SchemaError`, or `NamespaceError`;
- expected domain outcomes and partial side effects return structured
  `status/reason` values.

Do not turn either form into a broad silent success. In particular, `pending`
and `partial` mean durable truth may already have changed and follow-up repair
is required.

## Normal model-response turn

In the example below, `actual_final_message` is the sanitized, provider-shaped
assistant message preserved from the host's final response, including any
provider extension fields needed for later replay. For a text-only OpenAI Chat
Completion it can be `{"role": "assistant", "content": raw_model_output}`;
when using MemCore JSON, keep the original JSON content in this projection
and store `parsed.speech` separately as semantic text.

```python
from memcore import EntryOrigin, TimelineEntryInput, TurnRole

handle = mem.begin_turn(
    stimuli=[TimelineEntryInput(
        kind="message.user",
        origin=EntryOrigin.USER,
        turn_role=TurnRole.STIMULUS,
        semantic_text=user_text,
        payload={"text": user_text},
        actor=actor_or_none,
        timestamp=now_ts,
    )]
)
current = handle.stimuli[0].to_record()

try:
    # Build history before each model request; refresh after appending tool results.
    history = mem.build_context_projection(provider_profile=actual_provider_profile)
    # Send history.payloads with the host prompt, run any tool rounds, then parse
    # the final response and preserve actual_final_message from that response.
    completed = mem.complete_turn(
        turn_id=handle.turn_id,
        semantic_text=parsed.speech,
        provider_output_raw=raw_model_output,
        memory_annotation=parsed.memory_metadata,
        annotation_status="accepted_model",
        provider_profile=actual_provider_profile,
        provider_projection=actual_final_message,
    )
    if not completed.completed:
        handle_completion_state(completed)
finally:
    if turn_failed_before_completion:
        mem.abort_turn(handle.turn_id, reason="model_or_delivery_failed")
```

Use `begin_turn()` whenever a stimulus expects a model response. The convenience
methods `record_user_turn()` and `record_assistant_turn()` create standalone
entries; they do not open a response turn and cannot own later tool calls or a
`complete_turn()` final.

## `TimelineEntryInput`

Required constructor fields are:

| Field | Contract |
| --- | --- |
| `kind` | Lowercase namespaced identifier such as `message.user`, `event.qq.poke`, or `skill.catalog.result`. It must contain at least one dot and be at most 160 characters. |
| `origin` | `user`, `assistant`, or `environment`. |
| `turn_role` | `stimulus`, `intermediate`, `action`, `observation`, `final`, or `None` for standalone entries. |
| `semantic_text` | Model-readable natural text. It may be empty only when the typed payload still makes the entry meaningful. |

Important optional fields:

- `payload`, `trace_metadata`, and `memory_metadata` must be JSON-serializable
  objects with finite values;
- IDs are trimmed strings of at most 256 characters and may not contain control
  characters;
- action/observation entries require a non-empty `correlation_id`;
- `actor` and `target_actor` use stable platform IDs for attribution;
- `retrieval_policy` accepts `auto / always / explicit / never`;
- `retrieval_visibility` accepts `default / explicit / never`;
- `annotation_status` accepts the public `AnnotationStatus` values;
- `auto` resolves ordinary conversational records to `default` independently of
  optional metadata/annotation completeness. Use `explicit` or `never` only as
  an intentional caller policy for typed traces or product authorization, not
  as a fallback for missing model metadata;
- `trust` accepts `untrusted_data / trusted_instruction`. Historical memory and
  tool output should normally remain `untrusted_data`;
- `semanticize=False` prevents the entry from becoming semantic-memory input;
- `prompt_visible=False` keeps it out of normal provider history;
- hosts should normally leave renderer and derived time-label fields at their
  defaults and let `MemorySystem` bind/fill them.

## `begin_turn`

```python
handle = mem.begin_turn(
    stimuli=[...],
    annotation_target_ids=None,
    turn_id="",
    opened_at=None,
)
```

- At least one stimulus is required.
- With exactly one stimulus, omitted `annotation_target_ids` defaults to that
  stimulus source ID.
- Multiple stimuli require explicit annotation targets.
- Every annotation target must be one of this turn's stimulus source IDs and
  targets may not be duplicated.
- A caller-supplied `turn_id` is idempotent only when the original stimulus IDs
  and annotation targets are identical; a different retry raises
  `turn_idempotency_conflict`.
- The current `operation_projection_policy` is frozen into the returned
  `TurnHandle` together with `operation_settlement_min_utf8_bytes` and
  `operation_settlement_min_saved_ratio`; changing configuration later does
  not rewrite an open turn.

The returned `TurnHandle` contains `turn_id`, `namespace`, `status`, persisted
`stimuli`, `annotation_target_ids`, `opened_at`, and the frozen projection
policy.

## Intermediate entries and operations

`append_entry(entry, turn_id=...)` appends only to an existing open turn. It
accepts `intermediate`, `action`, and `observation`; stimulus and final entries
must use `begin_turn()` and `complete_turn()`.

For operations, prefer the typed helpers:

```python
action = mem.append_action(
    turn_id=handle.turn_id,
    kind="tool.web_search.call",
    correlation_id=tool_call_id,
    semantic_text="web_search requested",
    payload={"input": sanitized_arguments},
)

observation = mem.append_observation(
    turn_id=handle.turn_id,
    kind="tool.web_search.result",
    correlation_id=tool_call_id,
    semantic_text=model_visible_result,
    payload={"output": model_visible_result},
    status="success",
    retention_anchor=small_reload_receipt,
)
```

Within one turn, an action correlation ID is unique. An observation is rejected
until its matching action exists. Results may arrive out of action order, but
each must use the correct correlation ID. `record_tool_exchange(...)` is a thin
convenience wrapper that writes one action and one successful observation; use
the primitives when calls are parallel, delayed, failed, streamed, or need
different statuses.

`retention_anchor` is optional navigation metadata. It must be a JSON object and
is sanitized before projection. Anchors above 4096 UTF-8 bytes are preserved
but marked with `retention_anchor_status="oversized"` and
`retention_anchor_bytes`; they no longer abort the whole write. Keep credentials,
absolute paths, binary data, and complete tool results out of anchors.

## `complete_turn`

```python
result = mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text=clean_user_visible_speech,
    provider_output_raw=raw_provider_output,
    memory_annotation=metadata_for_the_single_target,
    annotation_status="accepted_model",
    annotations=None,
    timestamp=None,
    source_id="",
    close_reason="completed",
    kind="message.assistant",
    payload=None,
    trace_metadata=None,
    provider_profile="",
    provider_projection=None,
    append_final=True,
)
```

For one annotation target, the convenience `memory_annotation` and
`annotation_status` arguments are sufficient. Multiple annotation targets
require an explicit `list[MemoryAnnotation]` covering every target exactly once.
`MemoryAnnotation.status` may be a terminal input status such as
`accepted_model`, `accepted_host`, `missing`, `invalid`, `plain`, `fallback`, or
`rejected`; derived/internal statuses are rejected as host input.

With the default `append_final=True`, `semantic_text` and
`provider_output_raw` cannot both be empty. When completing an open turn,
an explicit `provider_profile` requires `provider_projection`; passing the
profile alone raises `SchemaError("turn_completion_projection_payload_required")`.
A raw projection dict requires `provider_profile`; a supplied
`ProjectionMessageInput` must represent an assistant message, and its profile
must match any explicit `provider_profile`. Preserve the actual final message
rather than rebuilding it from the parsed user-visible speech.

All preceding prompt-visible entries must already have projections for that
profile before appending an explicit final projection. Build them with
`build_context_projection()` or `build_context_surface()` as part of each
model request, or freeze a custom wire with `record_request_projection()`.
Skipping earlier entries can raise `SchemaError("projection_source_not_next_append")`;
do not reconstruct a fictitious request after the response to bypass it.

Omitting both projection arguments uses the standard final renderer. For
`append_final=False`, both text arguments must be empty and
`provider_projection` must be omitted; `provider_profile` may still select
terminal settlement for the completed tool history. No final entry is created,
and pending actions still prevent completion.

`CompletionCommitResult.status` values:

| Status | Meaning |
| --- | --- |
| `completed` | Annotations, visibility, and turn close committed, with a final entry only when `append_final=True`. |
| `already_completed` | Idempotent retry found an already closed turn. `completed` property is still true. |
| `not_found` | The turn does not exist in this namespace. |
| `pending_actions` | One or more action correlations have no observation; inspect `pending_correlations` and do not fake a final. |
| `invalid` | Annotation targets are duplicate, missing, mismatched, or not owned stimuli. |
| `conflict` | The turn was aborted or the chosen final source ID already exists. |

On `completed`, vector-index failure does not roll back SQLite truth. Affected
entries remain `pending` for `reindex_pending()` repair.

## Abort and crash recovery

```python
abort = mem.abort_turn(
    handle.turn_id,
    reason="provider_failed",
    closed_at=None,
)
```

`TurnAbortResult.status` is `aborted`, `already_aborted`, `not_found`, or
`conflict` (`turn_already_closed`). Abort is the normal immediate failure path.

On startup, hosts may additionally recover turns abandoned by process crashes:

```python
recovered = mem.recover_stale_open_turns(
    max_age_seconds=1800,
    now=None,
    reason="stale_open_turn_recovered",
)
```

`max_age_seconds` must be positive. Recovery returns a tuple of
`TurnAbortResult` values and only aborts old open turns inside the current
conversation namespace. It does not delete entries and must not replace the
normal `finally` abort path.

## Standalone entries and convenience writers

Use `append_standalone_entry(TimelineEntryInput(...))` for a fact/event that
does not trigger a model response. A standalone entry must have no `turn_id`,
`turn_role`, `reply_to_source_id`, or `correlation_id`. A repeated `source_id` is
idempotent only when the stored typed content and policies match; otherwise it
raises a conflict.

Public convenience writers:

- `record_user_turn(...)` and `record_assistant_turn(...)`: standalone message
  adapters only;
- `record_external_event(...)`: standalone `event.*` entry with explicit
  retrieval visibility;
- `record_material_reference(...)`: standalone `material.*.reference` anchor;
- `record_material_cleanup(...)`: standalone cleanup/tombstone event;
- `acquaintance_note(now_ts=None, cross_conversation=True)`: optional companion
  prompt note. It returns `""` with no history and describes only the first
  locally recorded date, not a provable real-world first meeting.

If an event or material triggers a model response, construct it as a stimulus
for `begin_turn()` instead of first recording it standalone.

Do not use `record_assistant_turn(in_reply_to=...)` to create turn lineage. A
standalone entry cannot carry `reply_to_source_id`; use `complete_turn()` for a
real response relationship. In the current V1 facade, a non-empty
`in_reply_to.source_id` is rejected by the standalone invariant.

Material writers store only IDs, names, types, statuses, attribution, and small
metadata. The original file, OCR, image description, and document chunks remain
in host-controlled storage.

## Metadata staging and repair

`stage_turn_metadata(source_id, memory_metadata, actor=None)` stores metadata on
an open stimulus without admitting it to retrieval. It is useful when a host
must persist final-model metadata before the final transaction completes.

Statuses:

- `staged`: metadata stored; index remains pending/not admitted;
- `forbidden`: namespace or Actor ownership mismatch;
- `invalid`: store lacks the staging contract or the target state is invalid;
- `not_found`: source does not exist;
- `pending`: metadata was staged but stale vector deletion failed.

`update_turn_metadata(source_id, memory_metadata, actor=None)` updates an
existing raw message and rebuilds its vector. Group-chat callers must pass the
same stable Actor used for the original entry. Results are `updated`,
`not_found`, or `pending`. `pending` means SQLite metadata changed but vector
repair remains. A record previously opted out of vectors remains `skipped` and
is not silently re-admitted.

New Timeline V2 chat integrations should normally let `complete_turn()` commit
annotations atomically rather than calling these methods as a parallel write
path.

## Projection lifecycle

For a complete provider-ready request, use
`build_context_surface(provider_profile=..., current_source_id=...)`. It
separates history, the current stimulus, and the active tool round;
`surface.messages` returns them in request order with aligned source and
projection metadata. Preserve that sequence rather than appending a second
copy of the current message or rendered raw history. The full contract is in
[`context_surface_contract_v1.md`](context_surface_contract_v1.md); existing
Session integrations can use
[`MemCoreContextSession`](context_integration_quickstart_v1.md).

`build_context_projection(provider_profile=...)` returns the deterministic
provider-visible message projection for the current conversation. The result
includes message payloads, stable prefix hash, per-entry hashes, projection and
compaction generations, and `has_compact_history`.

`build_open_turn_projection(turn_id=..., provider_profile=...)` is the focused
host adapter for refreshing one active native tool loop after appending a tool
batch. It accepts only a turn owned by the current conversation namespace whose
status is still `open`; it returns that turn's prompt-visible provider messages
without scanning compact summaries or other turns. It is an optimization of
active-turn reconstruction, not a second history surface: every complete model
request must still use `build_context_projection()` or `build_context_surface()`.

When the host's actual wire messages differ from MemCore's standard adapter,
call `record_request_projection(...)` before sending the request. The declared
turn suffix must byte-canonically match the actual history suffix; missing
source IDs, profile mismatch, or payload mismatch raises `SchemaError` rather
than freezing false history. See `operation_timeline_v1.md` for the complete
non-native adapter example.

## Compaction and runtime lifecycle

```python
future = mem.compact_due_background(provider_profile=actual_profile)
# or, for tests/CLI/shutdown:
stats = mem.compact_due_sync(provider_profile=actual_profile)
```

Background compaction returns a `Future`; `future.result()` yields the same
statistics shape as the synchronous call. Live chat should use the background
path so summary-model latency does not block the visible reply. One call advances
at most one raw generation and one semantic batch. Important result states and
fields are documented in `raw_token_compaction_policy_v1.md`.

`MemorySystem.close(wait=True)` closes only an internally created runtime. A
host that injected a shared `MemCoreRuntime` must close that runtime itself.
Store/index lifecycle remains host-owned.

## Index maintenance

```python
pending = mem.reindex_pending(limit=100, batch_size=64)
all_stats = mem.reindex_all(
    namespace=None,
    limit=None,
    current_conversation_only=False,
    batch_size=64,
)
```

- `reindex_pending()` repairs the pending outbox and returns
  `scanned/repaired/failed`.
- `reindex_all()` reloads raw, episodic, and semantic records from SQLite and
  returns `scanned/reindexed/failed`.
- `reindex_all()` defaults to every conversation inside the target hard
  namespace. Set `current_conversation_only=True` to narrow it.
- `limit` is a one-shot safety cap, not a cursor.
- Batch upsert failure is isolated down to individual entries; failures remain
  pending.
- Neither method clears stale entries from an external vector index.

## Destructive namespace deletion

> **Danger:** `forget_namespace()` permanently deletes truth records before it
> attempts vector-index cleanup. Resolve and verify the exact hard namespace in
> trusted host code; never expose this method directly as a model tool.

```python
result = mem.forget_namespace(namespace=exact_target_namespace)
```

The default target is `mem.namespace`. Deletion is scoped by
`tenant_id / user_id / domain_id` and removes **all conversations** inside that
hard namespace; `conversation_id` and `Actor` do not narrow it. There is no
single-conversation delete facade in V1.

Successful result:

```python
{
    "ok": True,
    "status": "deleted",
    "deleted_count": 42,
    "source_ids": [...],
    "index_deleted": True,
}
```

If index cleanup fails, the method returns `ok=False`, `status="partial"`,
`index_deleted=False`, and a reason. **The SQLite deletion has already happened
and is not rolled back.** Operations must treat this as a vector-backend cleanup
incident, not retry the database deletion blindly.

## Public status versus exceptions

Catch `MemcoreError` subclasses at the host boundary for configuration/schema/
namespace/prompt failures, while still inspecting returned status objects:

```python
from memcore import ConfigError, MemcoreError, NamespaceError, PromptError, SchemaError
```

- `ConfigError`: invalid configuration invariant;
- `SchemaError`: malformed typed entry, illegal turn transition, projection
  mismatch, or unsupported Timeline V2 store capability;
- `NamespaceError`: missing hard identity or ownership mismatch;
- `PromptError`: invalid/oversized additive prompt slot.

Type mismatches may raise `TypeError`; invalid timezones and positive-range
requirements may raise `ValueError`. These are programmer/configuration errors,
not empty-memory results.
