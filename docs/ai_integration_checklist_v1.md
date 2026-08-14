# AI Integration Checklist V1

This is the shortest authoritative path for an AI coding agent integrating
MemCore into a new host project. It lists the rules most likely to cause a
startup rejection, `SchemaError`, broken lineage, lost cache reuse, or memory
leakage when omitted.

Use only public imports from `memcore` and the `MemorySystem` facade. Do not
reimplement its schemas, write SQLite tables directly, or copy private
rendering/retrieval logic into the host.

## Prompt to give an AI coding agent

```text
Read the repository-root AGENTS.md first. Then read, in order:
1. docs/ai_integration_checklist_v1.md
2. docs/configuration_api_v1.md
3. docs/write_lifecycle_and_maintenance_api_v1.md
4. docs/memory_read_api_v1.md
5. docs/operation_projection_settlement_v1.md
6. docs/model_prompt_playbook_v1.md
7. examples/minimal_chat_integration.py

Integrate through the public MemorySystem facade only. Preserve the host's
actual provider request/tool-result wire, handle every structured status, and
run the focused acceptance checklist before claiming completion.
```

## Decide the integration path first

### Standard provider-native tools

Use `build_native_memory_tool_specs(...)` and
`dispatch_native_memory_tool(...)`. Persist the exact model-visible result as
an observation in the same open turn. Use the provider profile matching the
actual wire:

- `canonical_user_assistant`;
- `openai_chat`;
- `anthropic_messages`.

### Custom JSON, XML, tags, or nonstandard tool history

Use `append_action(...)` / `append_observation(...)` for typed timeline truth,
then call `record_request_projection(...)` with the exact messages that will be
sent. Do not let MemCore infer a different provider wire from the internal
entries. See `examples/non_native_operation_timeline.py`.

## Construction rules

| Rule | Correct behavior | Failure when omitted |
| --- | --- | --- |
| Inject `LLMClient` | Return `LLMResult`; never raw text/dict or a bare provider exception. | Construction/type failure or compaction cannot distinguish retry/fallback. |
| Inject an embedding | Use a real local/API `EmbeddingProvider` in production. | `MemorySystem` rejects missing embedding; hashed vectors have no real semantics. |
| Pass `timezone` | Use a valid IANA name such as `Asia/Shanghai`. | Construction raises `ValueError`; relative dates cannot be anchored safely. |
| Pass `Namespace.user_id` | Map tenant/user/domain to hard product ownership. | Missing user raises `NamespaceError`; wrong mapping can mix or hide memory. |
| Treat Actor as soft | `Actor(stable_id=platform_id, display_name=nickname)` for group attribution only. | Actor cannot isolate tenants/users; nickname IDs break attribution after rename. |
| Share runtime deliberately | Multiple systems in one process may share one `MemCoreRuntime`; its creator closes it. | Unnecessary executor sets or premature shared-runtime shutdown. |

Never place API keys, bearer tokens, passwords, local absolute paths, cached
paths, storage-relative paths, database paths, or binary/base64 media in stored
payloads, prompt projections, receipts, logs, or snapshots.

## Typed `kind` rules

Every timeline `kind` is lowercase, namespaced, and at most 160 characters. It
must match this shape:

```text
^[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)+$
```

Valid examples:

```text
message.user
message.assistant
event.qq.poke
tool.web_search.call
tool.web_search.result
material.image.reference
skill.catalog.request
skill.catalog.response
```

Invalid examples include `tool`, `Tool.Search`, `web search`, and names with no
dot. The core supports any valid namespaced kind; the canonical convention for
a provider tool is:

```text
tool.<lowercase_tool_name>.call
tool.<lowercase_tool_name>.result
```

`record_tool_exchange(tool_name=...)` constructs that convention but does not
turn an uppercase host name into a valid lowercase kind. Normalize host tool
names before passing them. Use other prefixes such as `skill.*` or
`operation.*` only when they truthfully describe a different host protocol.

Timeline relation IDs (`source_id`, `turn_id`, `reply_to_source_id`, and
`correlation_id`) are at most 256 characters and must not contain ASCII control
characters. Values used as actions/observations must also obey the non-empty
correlation rules below. `payload`, `trace_metadata`, `memory_metadata`, and a
`retention_anchor` are JSON objects; every nested value that reaches SQLite must
be JSON-serializable. Do not pass provider SDK objects, exceptions, streams, or
raw bytes as stored output—convert them to the exact safe text/JSON body shown
to the model first.

Normal retrieval excludes tool/material/operation traces. Reading them requires
all three conditions:

1. model request uses `include_explicit=true`;
2. request provides an exact kind or tail-prefix pattern such as `tool.web_search.*`;
3. host `ToolDispatchPolicy` authorizes that prefix.

Do not broaden kind authorization merely to turn an empty result into success.

## Required turn lifecycle

```text
begin_turn(stimulus)
  -> build/render visible memory
  -> zero or more model/tool rounds
  -> append action/observation pairs
  -> one final model output
  -> complete_turn
  -> compact_due_background
```

Hard rules:

- A request expecting a model response must start with `begin_turn()`.
- `record_user_turn()` / `record_assistant_turn()` are standalone adapters; they
  do not open a turn.
- Multiple stimuli require explicit, unique `annotation_target_ids`, all drawn
  from those stimuli.
- Append only `intermediate/action/observation` to an open turn. Stimulus and
  final use `begin_turn()` and `complete_turn()`.
- Every action/observation has a non-empty `correlation_id`.
- One action correlation is unique within a turn.
- An observation is written only after its matching action exists. Parallel
  results may finish out of order but must retain their own call IDs.
- `complete_turn()` must not run while an action is pending. Handle
  `pending_actions` and inspect `pending_correlations`.
- On model, tool-loop, or delivery failure call `abort_turn()` in `finally`.
- On startup, `recover_stale_open_turns(...)` is a crash-recovery supplement,
  not a substitute for immediate abort.

## Tool-result context custody

The observation body must be the same sanitized body the model actually saw.
Do not store a shorter shadow result while sending a larger provider result, and
do not serialize the same evidence twice into the provider request.

Persist:

- action name/arguments in the action entry;
- result body/status in the observation entry;
- the same `turn_id` and `correlation_id` across the pair;
- an optional small `retention_anchor` containing reload IDs, cursor, coverage,
  and hashes.

A retention anchor never replaces the full result while the turn is open. Do
not put full search output, credentials, files, or large ID lineage into it.
Anchors over 4096 UTF-8 bytes remain stored but are marked `oversized`; the host
should monitor and simplify future anchors.

## Final chat output

The optional MemCore JSON contract applies only after all tools finish. Do not
ask intermediate tool rounds to emit final `speech/memory_metadata` JSON.

After parsing:

- user sees only clean `speech`;
- `complete_turn()` atomically commits final speech and target metadata;
- invalid/broken JSON does not close the turn or become natural-language
  memory;
- group metadata preserves the correct Actor subject;
- `memory_metadata` describes the selected stimulus/event, not the assistant
  reply.

## Time rules

- Stored `timestamp` values are Unix seconds. Let `MemorySystem` derive
  `date_label`, weekday, and time-of-day from `timestamp + timezone`.
- Exact ranges use local or ISO `start_at/end_at`; start is inclusive and end is
  exclusive.
- Do not mix exact, date-label/time-of-day, and legacy epoch selector modes.
- A cursor continuation sends only `cursor`; selector, scope, projection, and
  budget are already embedded.
- Use `read_timeline` for exact date/time/quotation questions and
  `retrieve_for_turn` for unknown-time fuzzy facts.

## Three-layer memory lifecycle

The host must do all of the following:

1. Build current visible memory with `build_prompt_context(current=...)` and
   `render_prompt_context(...)`.
2. Expose `retrieve_for_turn`, `browse_memory`, `open_memory`, and
   `read_timeline` to the final chat model.
3. Use `retrieve_for_turn(current=current_raw, ...)` in live turns so current and
   prompt-visible lineage are excluded safely.
4. Call `compact_due_background(provider_profile=actual_profile)` after a
   successful visible reply; use sync only for tests/CLI/shutdown.
5. On process restart, warm an in-memory index with `reindex_all()` and repair
   pending outbox rows with `reindex_pending()`.
6. Run `embedding_status()` and `verify_embedding()` before trusting semantic
   recall. `ok=False` is degradation, not an authoritative empty memory.

SQLite is truth; the vector index is acceleration. Index failure must leave
truth pending for repair rather than dropping the chat turn or summary.

## Provider-prefix cache rules

MemCore can make provider history deterministic; it cannot promise a provider
cache hit or TTL.

Keep this order:

```text
stable system/persona/safety
stable tool schemas
stable memory/output contracts
stable domain guidance
dynamic time and visible memory
dynamic current message
dynamic tool results/final tail
```

Hard cache rules:

- keep tool schema order, field order, text, whitespace, and capability set
  stable across requests;
- do not place request IDs, current timestamps, current memory, or per-turn
  capability probes in the stable prefix;
- use `build_context_projection(provider_profile=actual_profile)` for provider
  history;
- when the host's real wire differs, freeze it with
  `record_request_projection(...)` before sending;
- pass the actual provider profile to `complete_turn()` and compaction;
- changing persona, system prompt, tool schema, or provider wire is an expected
  cache-prefix invalidation, not a MemCore retrieval failure.

## Optional terminal tool-result settlement

Default `full_until_raw_compaction` preserves full action/result history until
normal raw compaction. It is the safest compatibility mode.

Use `compact_after_terminal` only when the host also:

- keeps the full result during every open tool round;
- exposes `open_memory(memory_id=source_id, view="content")`;
- injects the stable `[compact_reloadable]` readback instruction;
- builds later history through `build_context_projection()`;
- monitors `settlement_metrics()` and handles `full_fallback` honestly.

Settlement changes provider-visible closed-turn history only after a successful
final. It does not delete SQLite truth, action parameters, final replies, or
provider-specific extensions.

## Structured failures to handle

Do not collapse these states into empty text or generic success:

- completion: `completed / already_completed / not_found / pending_actions /
  invalid / conflict`;
- abort: `aborted / already_aborted / not_found / conflict`;
- metadata: `staged / updated / forbidden / invalid / not_found / pending`;
- reads: `found|ok / empty / partial / invalid|invalid_filter / unavailable /
  failed`;
- namespace deletion: `deleted / partial` where partial means SQLite truth is
  already deleted but vector cleanup failed;
- compaction: `not_due / compacted / blocked_by_open_turn / busy / failed /
  structured stale or retry-pending states`.

Catch `MemcoreError` subclasses for programmer/configuration/schema/namespace
violations, but still inspect every returned status object.

## Focused acceptance checklist

Before claiming integration complete, verify with real host request construction:

- [ ] Missing/invalid timezone fails startup clearly.
- [ ] Production never silently selects `HashedEmbeddingProvider`.
- [ ] `verify_embedding().ok` is checked and reported.
- [ ] One user stimulus opens a turn and produces a current raw `source_id`.
- [ ] A tool action/result pair is visible to the next model call with the same
      call ID and no duplicated evidence body.
- [ ] Parallel tool results do not cross correlation IDs.
- [ ] Pending actions prevent final completion.
- [ ] A successful final atomically stores speech and stimulus metadata.
- [ ] A failed final/tool/delivery path aborts the open turn.
- [ ] Visible memory appears on the next normal turn.
- [ ] `retrieve_for_turn` excludes current/prompt-visible lineage.
- [ ] `browse_memory -> open_memory(content) -> sources/raw` navigation works.
- [ ] Exact local-time `read_timeline` returns the intended start-inclusive,
      end-exclusive range.
- [ ] Background compaction creates episodic and, when due, semantic memory
      without blocking the visible reply.
- [ ] Restart plus `reindex_all()` restores searchable records.
- [ ] Stable-prefix request bytes remain unchanged across two equivalent text
      turns except for the appended dynamic tail.
- [ ] If settlement is enabled, the open turn stays full, the next turn sees a
      reloadable card, and `open_memory(content)` restores the original result.
- [ ] Cross-user/hard-namespace reads and writes are rejected.
- [ ] No secret, binary media, or host-internal path field (`cached_path`,
      `storage_relpath`, `database_path`, …) enters persisted projection data;
      executable path evidence stays byte-for-byte intact.

## Read next

- `configuration_api_v1.md`: every constructor/config/provider rule;
- `write_lifecycle_and_maintenance_api_v1.md`: exact write signatures and
  statuses;
- `memory_read_api_v1.md`: exact read/native-tool contracts;
- `operation_projection_settlement_v1.md`: terminal compact/readback contract;
- `model_prompt_playbook_v1.md`: final model tool-selection guidance.
