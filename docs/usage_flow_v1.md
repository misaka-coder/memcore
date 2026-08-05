# memcore usage flow v1

This document is for humans and AI coding agents integrating `memcore` into a
host chat application.

The goal is to use the public `MemorySystem` facade without reading private
implementation modules for normal integration work.

## Package Role

`memcore` is a reusable layered memory kernel. It owns:

- typed turn recording for messages, events, actions, observations, materials, and finals;
- visible prompt context construction;
- fuzzy memory retrieval and exact timeline reading;
- raw -> episodic -> semantic compaction;
- metadata parsing helpers for final chat output;
- namespace, actor, and time-anchor semantics;
- store/index lifecycle helpers.

It does not own:

- the final chat model client;
- the embedding model choice for production;
- product persona or prompt style;
- host UI, profile storage, routing, or authentication;
- API keys, `.env`, runtime logs, or model cache files.

## Read First

For normal host integration, read these files in order:

1. `examples/minimal_chat_integration.py` - runnable chat-loop wiring.
2. `README.md` - public API, lifecycle, and boundaries.
3. This file - concise integration flow.
4. `docs/model_prompt_playbook_v1.md` - prompt/tool instructions for the chat
   model.
5. `docs/chat_output_adapter_v1.md` - optional final JSON contract and
   streaming speech parsing.
6. `docs/memory_metadata_raw_retrieval_design_v1.md` - current metadata and raw-first retrieval semantics.
7. `docs/raw_token_compaction_policy_v1.md` - token/ratio raw compaction.

If you are changing `memcore` itself, also inspect nearby tests before editing.

## Normal Chat Turn

Use `MemorySystem` as the facade:

```python
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
cur = handle.stimuli[0].to_record()

ctx = mem.build_prompt_context(current=cur)
ctx_text = mem.render_prompt_context(ctx)

# Put ctx_text into the final chat model prompt.
# Expose wrappers around:
# - mem.retrieve_for_turn(current=cur, ...)
# - mem.read_timeline(...)

raw_model_output = call_chat_model(...)

parsed = parse_chat_output(
    raw_model_output,
    mode="memcore_json",
    enable_flavor=mem.config.enable_flavor,
)
if not parsed.ok:
    handle_model_output_error(parsed)
    return

completed = mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text=parsed.speech,
    provider_output_raw=raw_model_output,
    memory_annotation=parsed.memory_metadata,
    annotation_status="accepted_model",
    timestamp=now_ts2,
)
if not completed.completed:
    handle_memory_commit_error(completed)
    return

mem.compact_due_background()
```

`kind` 默认为 `message.assistant`。宿主若完成的是语音等 typed turn，可传
`kind="message.assistant.voice"` 以及对应的小型结构化 `payload`。MemCore 不解释
业务字段，但会在 provider projection 中把自然回复保留为顶层 `speech`，把非默认
final 的 kind 和 payload 收进 `host_state`；默认文本 final 的既有纯文本投影不变。
这个边界让模型既能看到打断/交付状态，又不会把内部状态对象误当成下一轮回复契约。

For tests, scripts, or deterministic shutdown, use `compact_due_sync()`.
For live chat, prefer `compact_due_background()` so summarization does not block
the visible reply.

Wrap the host model/delivery lifecycle in `try/finally` and call
`abort_turn()` whenever an opened turn cannot be completed. On startup or
before opening a new turn, hosts may additionally call
`recover_stale_open_turns(max_age_seconds=...)` with a product-selected timeout
to recover turns abandoned by a process crash or forced shutdown. Recovery is
atomic and conversation-scoped; it does not delete timeline entries or replace
the immediate abort path.

If the host dynamically selects a provider, pass the actual projection profile
used by the completed request, for example
`mem.compact_due_background(provider_profile="openai_chat")`. Omitting it keeps
the configured `MemoryConfig.projection_profile` fallback.

## Required Host Pieces

A production host must provide or choose:

- `LLMClient` for summary, semantic, and reinforcement calls. Retrieval remains deterministic and does not call an LLM verifier.
- `EmbeddingProvider` for semantic retrieval. Do not use
  `HashedEmbeddingProvider` in production.
- `Namespace` values for tenant/user/domain/conversation isolation.
- Optional `Actor` for group or multi-speaker messages.
- Optional exact `TokenCounter`; without one, compaction uses an explicitly
  labelled estimated count instead of disabling the memory lifecycle.
- Store/index lifecycle. SQLite is the source of truth; vector indexes are
  search acceleration.

`EmbeddingProvider` keeps `embed_text()` / `embed_texts()` as the symmetric
compatibility API and also exposes role-aware methods:

- `embed_query()` / `embed_queries()` for retrieval queries;
- `embed_document()` / `embed_documents()` for indexed memory passages.

Symmetric models such as BGE-M3 may use the inherited defaults. Providers with
query/passage task adapters should override the role-aware methods and keep
their provider-specific task names inside the adapter. MemCore indexes call the
document batch method during upsert/reindex and the query method during search,
so remote providers do not need one HTTP request per memory entry. If task
selection, output dimension, normalization, or model revision changes the
vector space, reflect that in the provider's `version`/`dimension`; the
resulting `collection_key()` must change so incompatible vectors are rebuilt
rather than mixed.

For OpenAI-compatible endpoints whose query and passage requests need different
fields, use `RoleAwareHTTPEmbeddingProvider` and pass those field mappings from
the host adapter. MemCore deliberately does not name or hard-code a vendor's
task strings. `reindex_all(..., batch_size=...)` and
`reindex_pending(..., batch_size=...)` preserve remote batch calls.

## Retrieval Tools To Expose

Expose memory tools to the final chat model:

- `retrieve_for_turn(current=cur, ...)` for fuzzy preferences, plans, people,
  topics, and long-term facts. When exact time is useful to narrow a fuzzy
  event search, pass local/ISO `time_hint={"start_at": ..., "end_at": ...}`;
  MemCore normalizes it with the same timezone rules as `read_timeline` and
  filters candidates before vector/BM25 ranking.
- `read_timeline(time_range={"start_at": ..., "end_at": ...}, projection="conversation")` for exact hour/minute questions without calculating epoch; legacy date fields remain available for whole-day/coarse-period reads, or use it for expanding a raw retrieval `source_id` into complete nearby turns. The default view keeps dialogue/events full and returns reloadable compact evidence for operations/materials.
- `browse_memory(date_from=..., date_to=...)` for broad multi-day overviews. It returns compact chronological cards plus stored-history coverage instead of loading the whole raw range. Continue an incomplete page with only `cursor`.
- `open_memory(memory_id=..., view="content")` opens one selected raw/episodic/semantic node; `view="sources"` follows exact lineage to child episode cards or complete raw logical units. Raw sources default to `projection="conversation"`, so dialogue/events remain complete while large operation/Skill/tool/material bodies become compact records containing type, call/result linkage, status, `source_id`, and small retained anchors. Use `projection="full"` or `"tools"`, or open one compact `source_id` as `content`, only when that body is actually needed. Use sources only when summary content is insufficient.
- Provider-native dispatch always applies `MemoryConfig.native_timeline_page_token_budget` as a finite maximum. Omitted/zero uses that maximum; a smaller model request is honored and a larger one is capped. If `status=partial`, inspect selected/returned token and logical-unit counts, then either continue with `read_timeline(cursor=next_cursor)` only or use `browse_memory` for an overview. One oversized turn is returned whole and marked explicitly. The native result keeps the readable rendered `text` plus navigation metadata and omits the duplicate structured `messages` body. Trusted host/diagnostic code may still call `MemorySystem.read_timeline(page_token_budget=0)` directly for an unlimited read and receives both messages and text.
- `read_entry(source_id=..., detail="full")` is a raw-only compatibility adapter. New integrations use `open_memory(view="content")`.

If the host supports images/files, also expose `load_material(file_id, kind?,
preferred_source?, purpose?)` as a provider-native tool backed by host
file/derived storage. Historical image/file follow-ups should find a
typed `material.*` anchor first, then call `load_material` for current original
content, OCR, image descriptions, document chunks, or an expired status.

Prefer `retrieve_for_turn` over direct `retrieve` during live turns because it
excludes memory already visible in the prompt and excludes the current user
message.

Do not turn invalid filters into broad successful searches. Return structured
tool errors according to host policy.

memcore provides optional native-tool helpers:

```python
tools = build_native_memory_tool_specs()
tool_result = dispatch_native_memory_tool(
    tool_name=tool_call.name,
    arguments=tool_call.arguments,
    mem=mem,
    current=cur,
    material_loader=load_material_from_host_store,
    policy=ToolDispatchPolicy(
        allow_explicit_trace=True,
        allowed_kind_prefixes=("material",),  # choose only product-approved prefixes
    ),
)
```

Send `tool_result` back through the model provider's native tool-result channel.
The dispatcher also returns `tool_result["receipt"]`, a deterministic compact
record containing selectors, returned IDs, coverage, cursor, request hash, and
sanitized result hash. If the product wants cross-turn recall, persist that
receipt as the observation payload; keep the full result only in the active
provider tool loop. Do not duplicate raw text, summary bodies, snippets,
credentials, files, or local paths into the timeline.

For `open_memory(content/sources)`, the direct trusted Python facade keeps both
structured records and rendered text for diagnostics. Native dispatch sends
only one rendered evidence body plus navigation metadata and logical-unit IDs,
so the same raw page is not serialized twice into the model context.

```python
mem.append_observation(
    turn_id=handle.turn_id,
    kind=f"operation.memory.{tool_call.name}.result",
    correlation_id=tool_call.id,
    payload=tool_result["receipt"],
    status=tool_result["receipt"]["status"],
)
```

`build_memory_operation_receipt(...)` is public for non-native adapters that
need the same receipt contract.

If the host app records tool calls or tool results into raw memory, prefer
`record_tool_exchange(...)`. It writes typed action/observation records. Their
identity and admission use `kind`, turn role, visibility, and correlation
lineage; they do not masquerade as a semantic category. These records contribute to the
normal token compaction lifecycle. Retrieval requires `include_explicit=true`,
an authorized precise `kind_patterns` value, and host policy approval.

For new Timeline V2 hosts, `append_action(...)` and
`append_observation(...)` are the protocol-neutral primitives. Their `kind` and
payload are host-defined, so they also cover model-emitted JSON/XML/tags, Skill
steps, catalog loading, or other request/result flows. MemCore does not parse or
execute those protocols. If the host did not use provider-native tool messages,
freeze the actual messages with `record_request_projection(...)`; see
`docs/operation_timeline_v1.md` and
`examples/non_native_operation_timeline.py`.

An operation may opt in to a small structured `retention_anchor` when later
turns need a resource ID, version, schema hash, or result reference after raw
compaction. Do not copy complete results, credentials, local paths, or files into
that anchor. Entries without an anchor keep the existing lossy operation-digest
behavior.

If the host app handles images or files, record only material references with
`record_material_reference(...)` and cleanup events with
`record_material_cleanup(...)`. These write typed `material.*` references and
cleanup events. Store original files and derived OCR, vision
descriptions, or document chunks in the host file/derived stores. Current-turn
multimodal models may receive the image through the provider request. For non-
multimodal models, do not race the final chat model against OCR/vision parsing:
wait for derived content, or return a structured pending/unavailable material
tool result and do not let the model describe the image from the anchor alone.
Historical follow-ups should find the material anchor in visible/timeline
context or through authorized explicit `material.*` retrieval, then load
available derived content via host tools.
In group or multi-speaker uploads, pass `actor=Actor(stable_id=..., display_name=...)`
to `record_material_reference(...)` so the attachment keeps uploader attribution.

## Memory Metadata Contract

Every chat annotation, stage summary, semantic summary, index entry, and
retrieval tool uses the same fields:

```json
{
  "turn_intent": "",
  "memory_facets": [],
  "about_roles": [],
  "entity_anchors": [],
  "topic_terms": [],
  "retrieval_priority": "normal",
  "mood_tags": []
}
```

- `memory_facets` describes which future question the content can answer.
- `about_roles` describes who or what the content is about, not who spoke.
- `entity_anchors` contains only exact names/aliases; `topic_terms` contains supporting actions or themes.
- An identity or answer currently being asked for is not an `entity_anchor`.
  Pass only known names plus known relationships/topics/time; never guess the
  missing answer to make a retrieval call.
- `turn_intent="memory_query"` marks the current turn as asking about memory; it is not a target history facet.
- Tool/event/material identity stays in typed timeline fields, not in metadata facets.

The old `keywords/categories/subject_scopes/importance/confidence` write API is
not a runtime compatibility path. SQLite schema migration converts old stored
JSON once and schedules every affected entry for reindexing.

## Prompt Composition

When building the final chat model prompt:

1. Keep stable system/developer instructions first.
2. Include the memcore tool-use instructions from
   `docs/model_prompt_playbook_v1.md`.
3. Include the optional memcore JSON contract only for the final user-facing
   reply, not for intermediate tool calls.
4. Put dynamic content after the stable prefix: current time, rendered visible
   memory, tool results, and current user message.
5. Preserve group-chat attribution when `Actor` is used.

Relative time words must be interpreted from visible date and weekday anchors.

## Namespace And Actor Rules

Hard isolation is:

```text
tenant_id / user_id / domain_id
```

`conversation_id` controls the visible context window. Retrieval may still
search across conversations under the same hard namespace when host policy
allows it.

`actor` is a soft speaker label. It is not a hard isolation key.

For group chat, use stable platform ids:

```python
Actor(stable_id="platform-stable-id", display_name="current nickname")
```

Do not use nicknames as stable ids.

## AI Agent Checklist

When an AI agent integrates `memcore`, follow this order:

1. Use `MemorySystem`; do not bypass it to write private internals.
2. Record the user turn before building prompt context.
3. Render visible memory and add it to the final chat model prompt.
4. Expose `retrieve_for_turn`, `browse_memory`, `open_memory`, and `read_timeline` as model tools.
5. Add the model prompt guidance and optional JSON contract.
6. Parse the final reply before storing assistant speech.
7. Commit parsed `memory_metadata` to the host-selected annotation target.
8. Record the assistant turn with clean speech, not broken JSON.
9. Run background compaction after the visible reply path.
10. Keep API keys, local paths, logs, databases, and cached model files out of
    prompts, docs, snapshots, and commits.

Normal integration should not require reading private memcore modules.

## Validation

When modifying `memcore`, run:

```bash
uv run --extra dev python -m unittest discover -s tests -v
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
git diff --check
uv run --extra dev python -m build
```
