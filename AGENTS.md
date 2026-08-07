# AGENTS.md — memcore AI Integration Guide

This file is for AI coding agents that need to integrate `memcore` into a host chat application.

Your goal is not to rewrite memcore. Your goal is to wire the host app to memcore's public API, give the chat model the right tools/prompts, and preserve memcore's safety boundaries.

## Read First

Read these files in order before coding:

1. `examples/minimal_chat_integration.py` — runnable minimal chat-loop wiring.
2. `README.md` — current public API, lifecycle, boundaries.
3. `docs/usage_flow_v1.md` — concise host and AI-agent integration flow.
4. `docs/memory_read_api_v1.md` — authoritative read/navigation API signatures and result contracts.
5. `docs/model_prompt_playbook_v1.md` — how to prompt the chat model so memory works well.
6. `docs/design_highlights_v1.md` — why the system is designed this way.
7. `docs/chat_output_adapter_v1.md` — optional final-output JSON contract and streaming speech parsing.
8. `docs/memory_metadata_raw_retrieval_design_v1.md` — current metadata and prefilter semantics.
9. `docs/raw_token_compaction_policy_v1.md` — the single token/ratio raw compaction policy.
10. `docs/operation_projection_settlement_v1.md` — optional final-after-tool
    settlement, reload API, prompt rule, metrics, and migration behavior.

If you are changing memcore itself, inspect nearby tests first and run the validation commands at the end of this file.

## Core Integration Shape

Use `MemorySystem` as the only facade. A normal chat turn should look like this:

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
# Expose retrieve_for_turn, browse_memory, open_memory, and read_timeline as model tools.
# If using memcore_json, append build_chat_output_contract_prompt(...).

result = call_chat_model(...)

# If memcore_json is enabled, parse result and use parsed speech/metadata.
parsed = parse_chat_output(
    result,
    mode="memcore_json",
    enable_flavor=mem.config.enable_flavor,
)
if not parsed.ok:
    handle_model_output_error(parsed)  # retry or surface a structured failure; do not store empty speech
    return

completed = mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text=parsed.speech,
    provider_output_raw=result,
    memory_annotation=parsed.memory_metadata,
    annotation_status="accepted_model",
    timestamp=now_ts2,
)
if not completed.completed:
    handle_memory_commit_error(completed)
    return

mem.compact_due_background()
```

For tests, scripts, or deterministic shutdown, `compact_due_sync()` is acceptable. In live chat, prefer `compact_due_background()` so summarization does not block the user-visible reply.

## Required Host Adapters

Implement or choose these pieces in the host project:

- `LLMClient`: adapter for summary / semantic / reinforcement calls. Retrieval must not add an LLM verifier pass.
- `EmbeddingProvider`: production embedding model. Do not use `HashedEmbeddingProvider` in production.
- `Namespace`: decide `tenant_id`, `user_id`, `domain_id`, `conversation_id`.
- Optional `Actor`: use for group/multi-speaker messages.
- Optional `TokenCounter`: inject an exact tokenizer when available. Without it,
  compaction remains available and reports `token_count_quality=estimated`.

Do not put API keys, local absolute paths, `.env`, runtime logs, databases, or cached model files into prompts, docs, snapshots, or commits.

## Namespace And Actor Rules

Hard isolation is:

```text
tenant_id / user_id / domain_id
```

`conversation_id` controls visible context windows. Retrieval may still search across conversations under the same hard namespace.

`actor` is a soft speaker label for group chats. It is not a hard isolation key.

For group chat:

```python
Actor(stable_id="platform-stable-id", display_name="current nickname")
```

Use a stable platform ID for `stable_id`, not a nickname. Nicknames can change.

If two people must be fully isolated, map them to different `user_id` values instead of relying on `actor`.

## Config Choices

Start conservative:

```python
cfg = MemoryConfig(
    visible_memory_scope="conversation",
    enable_flavor=False,
    retrieval_result_token_budget=0,
)
```

Use `visible_memory_scope="user"` only when the product wants cross-conversation continuity, such as companion apps.

`memory_facets` and `about_roles` are protocol constants rather than host-configurable
category lists. Domain-specific names belong in `entity_anchors`, supporting actions
and themes belong in `topic_terms`, and protocol identity belongs in typed `kind`
values. Do not restore the removed `categories/subject_scopes/importance` runtime
contract or add a second host-owned taxonomy beside the public schema.

Record tool calls/results with `append_action(...)` and `append_observation(...)`
inside the same open turn. `record_tool_exchange(turn_id=...)` is only a thin
convenience adapter. Operation lineage depends on typed roles and
`correlation_id`, not category strings. It participates in the one token
compaction lifecycle and remains explicit-only in normal retrieval.

For Timeline V2 or non-native model protocols, use `append_action(...)` and
`append_observation(...)` with an open namespaced `kind` and stable
`correlation_id`. The host may use provider-native tools, JSON, XML, tags, or
another protocol; MemCore does not parse or execute it. Freeze the actual
provider messages with `record_request_projection(...)` when they differ from
the standard adapter fallback. Use the optional `retention_anchor` only for a
small resource ID/version/hash/result reference that must survive operation
compaction, never for a full result, credential, local path, or file.

Record images/files as `material.*` intermediate entries in the current turn,
or use the material standalone convenience methods when no model response is
expected. Store only file/material anchors; originals and OCR/vision/document
chunks stay in host storage. Materials compact into the operation partition and
remain explicit-only in normal retrieval. Preserve group uploader attribution
with `Actor(stable_id=..., display_name=...)`.

## Retrieval Tools To Expose

Expose memory tools to the chat model:

### `retrieve_for_turn`

Use this for fuzzy memory search. Prefer this over `retrieve` during a live turn because it excludes the current visible raw/episodic/semantic memory and the current user message.

Recommended tool parameters:

- `query: str` — always retained in dense/BM25 query construction
- `entity_anchors: list[str]` — exact entities already known from the question/context
- `topic_terms: list[str]` — actions, relations, attributes, and supporting topics
- `source_layers: list[str]` — `raw`, `summary`, `semantic_summary`
- `memory_facets: list[str]` — fixed answer-type facets; omit rather than guess
- `about_roles: list[str]` — who/what the historical content is about, not who spoke
- `time_hint: dict` — known time evidence only
- `within_memory_id: str` — optional hard lineage boundary from browse/open
- `include_explicit: bool` plus `kind_patterns: list[str]` — only for a
  host-authorized explicit trace/event/material read

For exact fuzzy-retrieval bounds, use `time_hint={"start_at": ..., "end_at": ...}`
with local or ISO 8601 strings. MemCore applies `MemorySystem.timezone` when an
offset is omitted and hard-filters the index before dense/BM25 ranking. Legacy
`date_label/time_of_day/start_ts/end_ts` inputs remain aliases; do not mix time
selector modes.

Metadata filters are prefilters: memcore narrows candidates before vector/BM25 scoring.

### `read_timeline`

Use this for exact date/time questions.

Recommended tool parameters:

- `time_range: {start_at, end_at}` optional — exact start-inclusive/end-exclusive ISO 8601 or local date-time range
- `date_from: YYYY-MM-DD`
- `date_to: YYYY-MM-DD` optional
- `time_periods: list[str]` optional, such as morning/afternoon/night or localized aliases supported by the host
- `cross_conversation: bool` only if the product allows it
- `projection: conversation|full|tools` optional; conversation is the normal evidence view
- `page_token_budget: int` optional; native dispatch uses the configured finite maximum when omitted/zero and caps larger values; direct trusted Python calls may use zero for unlimited diagnostics
- `cursor: str` optional; when present, send the cursor alone because selector/view/budget are embedded

### `browse_memory`

Use this for a broad date span or when loading every raw message would be noisy.
It returns a deterministic chronological catalog of compact episodic/semantic
cards plus explicit stored-history coverage. Continue an incomplete page by
calling it again with only `cursor`.

Recommended tool parameters:

- `time_range: {start_at, end_at}` optional
- `date_from: YYYY-MM-DD` optional
- `date_to: YYYY-MM-DD` optional
- `node_types: list[str]` optional — `episodic`, `semantic`
- `cross_conversation: bool` only if the product allows it
- `page_size: int` optional
- `cursor: str` optional; when present, send the cursor alone

### `open_memory`

Use this only with IDs returned by retrieval or `browse_memory`. Pass one
`memory_id`, or pass `memory_ids` to open several `card`/`content` nodes in one
ordered batch. Batch results report status/reason per node. `sources` remains a
single-ID operation because every source tree owns an independent cursor.
`view="card"` repeats compact metadata, `view="content"` opens the selected
raw/summary body, and `view="sources"` follows exact lineage to child episode
cards or complete raw logical units. Raw sources default to
`projection="conversation"`: dialogue/events remain full while operation,
Skill, tool, and material bodies become reloadable compact evidence. Use
`projection="full"`/`"tools"`, or open one compact source ID with
`view="content"`, only when the full operation body is actually needed. Use
`sources` only when summary content is insufficient. Continue an incomplete
sources page with only `cursor`.

### `load_material` optional

If the host supports images/files, expose this as a provider-native tool backed by host file/derived storage. Use it after the model has found a material anchor in visible raw context, `read_timeline`, or an authorized explicit retrieval such as `retrieve_for_turn(include_explicit=True, kind_patterns=["material.*"])`.

Recommended tool parameters:

- `file_id: str`
- `kind: str` optional, such as image/pdf/file
- `preferred_source: str` optional — `auto`, `original`, `derived`
- `purpose: str` optional

Tell the chat model:

- Use `read_timeline` for "yesterday", "last Tuesday", "that night", and exact hour/minute ranges.
- If timeline status is `partial`, inspect selected/returned volume; either call `read_timeline(cursor=next_cursor)` without repeating the selector or use `browse_memory` for an overview.
- Use `open_memory(memory_id=..., view="content")` when a compact memory result is relevant but its body is insufficient. `read_entry` is a raw-only Python/dispatcher compatibility API, not a model-facing tool for new integrations.
- Use `retrieve_for_turn` for preferences, plans, long-term facts, people, topics, and fuzzy recall.
- Pass only already-known names in `entity_anchors`. A person or answer being
  asked for is unknown evidence, not an anchor; use known people, relations,
  topics, and time instead of guessing it.
- Use `load_material` for historical image/file/PDF content only after a visible or retrieved material anchor provides the `file_id`. If no anchor or retained derived content exists, say the evidence is unavailable instead of guessing.

For provider-native tool loops, prefer `build_native_memory_tool_specs(...)` and
`dispatch_native_memory_tool(...)` over legacy text wrappers. The dispatcher strictly rejects invalid filters instead of broadening them.
It also returns a compact `receipt`. Persist the complete result that the model
actually received as the observation in the same open turn, so later normal
turns can continue discussing it. The default `full_until_raw_compaction` policy
keeps the full provider projection until unified raw compaction. Hosts may opt in
to `compact_after_terminal`; it still stores and exposes the full result during
the open tool loop, then projects a reloadable `source_id` card after final. Add
the stable readback prompt rule and expose `open_memory(content)` as specified in
`docs/operation_projection_settlement_v1.md`. Store the receipt beside that body
as a small `retention_anchor` for IDs, coverage, cursor, and hashes; a receipt
never replaces the observation.
Do not serialize the same body into both `semantic_text` and a second rendered
structure in the provider result, and never persist credentials, local paths,
or binary file content.
When a product allows explicit operation/event/material retrieval, pass a
`ToolDispatchPolicy` with only the approved kind prefixes, for example
`ToolDispatchPolicy(allow_explicit_trace=True, allowed_kind_prefixes=("material",))`.
Do not enable broad prefixes merely to avoid an empty result.

## Prompt Requirements

When building the final chat model prompt, include:

- Rendered visible memory from `render_prompt_context`.
- Tool descriptions for `retrieve_for_turn`, `browse_memory`, `open_memory`, `read_timeline`, and optional `load_material`.
- Time-anchor instruction from `docs/model_prompt_playbook_v1.md`.
- Group-chat attribution instruction if `Actor` is used.
- Optional `build_chat_output_contract_prompt(...)` if using memcore JSON.

Cache-friendly ordering:

1. Put stable system/developer content first: persona, safety policy, tool rules, metadata rules, JSON output contract.
2. Keep that stable prefix byte-for-byte stable across turns: same order, whitespace, field names, and tool schema.
3. Put dynamic content after the stable prefix: current time, rendered visible memory, current user message, and tool results.
4. For repeated long documents, put the unchanged document before the variable question so provider prefix caches can reuse it.
5. Do not inject request ids, timestamps, rendered memory, or retrieved snippets into the stable prefix.

Important model guidance:

- Treat exposed tools as part of the model's working ability and structured information channel, not optional decoration. If an answer depends on facts not clearly visible in the current prompt, old memory, exact timelines, attribution, preferences, relationships, promises, or platform events, the model should proactively call the appropriate tool. Multi-step tool use is allowed when the first result is insufficient.
- Prefer native tool calling/tool result blocks when the host model provider supports them. Preserve tool id/name/arguments/result boundaries. Legacy text followup should be a compatibility fallback, not the primary path.
- Relative time words must be interpreted from the visible date/weekday anchors.
- In group chat, preserve who said what. Do not merge different speakers into "the user".
- For attribution questions such as who said, poked, promised, or owns a task, answer only from visible raw text or tool results. If evidence is missing, call `read_timeline`/`retrieve_for_turn` or say there is no clear record; do not guess a name.
- For memory questions about birthdays, preferences, relationships, past statements, promises, or old events, do not answer from persona confidence. If the answer is not clearly visible, call `retrieve_for_turn` or `read_timeline`; if still unsupported, say there is no clear record instead of inventing one.
- `memory_metadata` describes the host-selected annotation target, which may be a user message or an external event that triggered the reply; it never describes the assistant reply.
- `memory_metadata.entity_anchors` contains only exact known names/aliases;
  `topic_terms` contains reusable action, relation, attribute, or topic terms,
  not sentences. Do not put a guessed answer into either field.
- If unsure about metadata, leave the uncertain arrays empty. `retrieval_priority`
  is a small retrieval-order hint, not a confidence score; do not use it to
  disguise uncertain labels.
- Tool calls are not wrapped in memcore JSON. Only the final user-facing reply uses the JSON contract.

## Chat Output Adapter

If the host app can require final JSON output, use:

```python
from memcore import build_chat_output_contract_prompt, parse_chat_output

contract = build_chat_output_contract_prompt(
    enable_flavor=cfg.enable_flavor,
    enable_sentence_segments=True,
)
```

After the model returns:

```python
parsed = parse_chat_output(
    raw_model_output,
    mode="memcore_json",
    enable_flavor=cfg.enable_flavor,
)
if not parsed.ok:
    handle_model_output_error(parsed)  # retry or surface a structured failure
    return

completed = mem.complete_turn(
    turn_id=handle.turn_id,
    semantic_text=parsed.speech,
    provider_output_raw=raw_model_output,
    memory_annotation=parsed.memory_metadata,
    annotation_status="accepted_model",
)
if not completed.completed:
    handle_memory_commit_error(completed)
```

For streaming UI/TTS, use `StreamingSpeechParser` so `speech` can appear before `memory_metadata` is complete.

If the model fails the JSON contract, do not store the broken JSON as natural language memory. Surface a structured failure or retry according to host policy.

## Index And Startup

SQLite is the source of truth. The vector index is a searchable acceleration layer.

Default small/mid-size setup:

```python
store = SQLiteMemoryStore("path/to/memcore.sqlite3")
index = InMemoryVectorIndex(embedding=embedding)
mem = MemorySystem(..., store=store, index=index, embedding=embedding)
mem.reindex_all()
```

`reindex_all()` warm-loads records for the current hard namespace into the current index. It does not clear stale entries from an external vector database.

For large deployments, implement or configure another `VectorIndex` backend, such as Chroma. Keep the same public `MemorySystem` lifecycle.

## Do Not

- Do not bypass `MemorySystem` and write directly to private internals.
- Do not call `retrieve` directly in a live turn unless you manually exclude visible source IDs; use `retrieve_for_turn`.
- Do not use a router LLM to decide every turn unless the host explicitly wants that cost/latency. memcore is designed for chat-model-driven tool use.
- Do not silently fall back to fake embeddings in production.
- Do not ignore timezone. `timezone` is required because relative-time memory depends on it.
- Do not treat `actor` as hard isolation.
- Do not store generated summaries when the summary LLM failed.
- Do not turn invalid tool filters into broad successful searches. Return structured errors.

## Validation Commands

When modifying memcore itself, run:

```bash
uv run --extra dev python -m unittest discover -s tests -v
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
git diff --check
uv run --extra dev python -m build
```

When only integrating memcore into a host project, at minimum run the host app's chat smoke test:

- first message records raw memory;
- rendered prompt contains visible memory with time anchors;
- model can call `retrieve_for_turn`;
- model can browse compact history with `browse_memory`, open one selected `memory_id`, and batch-open several card/content IDs with `open_memory`;
- model can call `read_timeline`;
- assistant reply is recorded;
- background compaction does not block the visible reply;
- group chat preserves actor attribution if used.
