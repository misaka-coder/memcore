# AGENTS.md — memcore AI Integration Guide

This file is for AI coding agents that need to integrate `memcore` into a host chat application.

Your goal is not to rewrite memcore. Your goal is to wire the host app to memcore's public API, give the chat model the right tools/prompts, and preserve memcore's safety boundaries.

## Read First

Read these files in order before coding:

1. `examples/minimal_chat_integration.py` — runnable minimal chat-loop wiring.
2. `README.md` — current public API, lifecycle, boundaries.
3. `docs/usage_flow_v1.md` — concise host and AI-agent integration flow.
4. `docs/model_prompt_playbook_v1.md` — how to prompt the chat model so memory works well.
5. `docs/design_highlights_v1.md` — why the system is designed this way.
6. `docs/chat_output_adapter_v1.md` — optional final-output JSON contract and streaming speech parsing.
7. `docs/metadata_prefilter_design_v1.md` — metadata prefilter semantics for retrieval.
8. `docs/raw_token_compaction_policy_v1.md` — optional token-based raw compaction.

If you are changing memcore itself, inspect nearby tests first and run the validation commands at the end of this file.

## Core Integration Shape

Use `MemorySystem` as the only facade. A normal chat turn should look like this:

```python
cur = mem.record_user_turn(user_text, actor=actor_or_none, timestamp=now_ts)

ctx = mem.build_prompt_context(current=cur)
ctx_text = mem.render_prompt_context(ctx)

# Put ctx_text into the final chat model prompt.
# Expose mem.retrieve_for_turn(current=cur, ...) and mem.read_timeline(...) as model tools.
# If using memcore_json, append build_chat_output_contract_prompt(...).

result = call_chat_model(...)

# If memcore_json is enabled, parse result and use parsed speech/metadata.
parsed = parse_chat_output(
    result,
    mode="memcore_json",
    categories=mem.config.categories,
    enable_flavor=mem.config.enable_flavor,
)
if not parsed.ok:
    handle_model_output_error(parsed)  # retry or surface a structured failure; do not store empty speech
    return

mem.update_turn_metadata(cur["source_id"], parsed.memory_metadata)
mem.record_assistant_turn(parsed.speech, in_reply_to=cur, timestamp=now_ts2)

mem.compact_due_background()
```

For tests, scripts, or deterministic shutdown, `compact_due_sync()` is acceptable. In live chat, prefer `compact_due_background()` so summarization does not block the user-visible reply.

## Required Host Adapters

Implement or choose these pieces in the host project:

- `LLMClient`: adapter for summary / semantic / reinforcement / verifier calls.
- `EmbeddingProvider`: production embedding model. Do not use `HashedEmbeddingProvider` in production.
- `Namespace`: decide `tenant_id`, `user_id`, `domain_id`, `conversation_id`.
- Optional `Actor`: use for group/multi-speaker messages.
- Optional `TokenCounter`: required only when `raw_compaction_policy="token"`.

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
    enable_verifier=True,
    enable_flavor=False,
)
```

Use `visible_memory_scope="user"` only when the product wants cross-conversation continuity, such as companion apps.

For finance or other domains, customize `categories` with a fixed enum. Example:

```python
cfg = MemoryConfig(
    categories=(
        "risk_profile",
        "investment_goal",
        "asset_preference",
        "constraint",
        "plan_goal",
        "life_event",
        "memory_query",
    )
)
```

Never let the model invent category names. memcore will drop values outside the enum.
If the host records tool calls/results into raw memory, prefer `record_tool_exchange(...)` so the trace is stored as linear `assistant.tool_call <tool> <call_id>` and `tool.<tool> <call_id>` blocks with `categories=["tool_trace"]`. The default count policy includes `tool_trace` in raw compaction triggers, so tool-heavy timelines enter the normal summary lifecycle even without ordinary chat messages. Normal retrieval still excludes it unless `categories=["tool_trace"]` is explicitly requested. If you override `categories`, keep `tool_trace` in the enum if you need this behavior.

If the host records images/files into raw memory, prefer `record_material_reference(...)` and `record_material_cleanup(...)`. These store only file/material anchors as `user.attachment <kind> <file_id>` and `system.material_cleanup <kind> <file_id>` blocks with `categories=["material_trace"]`; original files and OCR/vision/document chunks stay in host storage. The default config excludes `material_trace` from count-based raw compaction triggers and from normal retrieval unless `categories=["material_trace"]` is explicitly requested. If you override `categories`, keep `material_trace` in the enum if you need this behavior.
For group or multi-speaker uploads, pass `actor=Actor(stable_id=..., display_name=...)` to `record_material_reference(...)` so the attachment keeps uploader attribution.

## Retrieval Tools To Expose

Expose memory tools to the chat model:

### `retrieve_for_turn`

Use this for fuzzy memory search. Prefer this over `retrieve` during a live turn because it excludes the current visible raw/episodic/semantic memory and the current user message.

Recommended tool parameters:

- `query: str`
- `keywords: list[str]`
- `source_layers: list[str]` — `raw`, `summary`, `semantic_summary`
- `categories: list[str]`
- `subject_scopes: list[str]` — `user`, `assistant`, `other`
- `importance_min: float`
- `time_hint: dict`

Metadata filters are prefilters: memcore narrows candidates before vector/BM25 scoring.

### `read_timeline`

Use this for exact date/time questions.

Recommended tool parameters:

- `date_from: YYYY-MM-DD`
- `date_to: YYYY-MM-DD` optional
- `time_periods: list[str]` optional, such as morning/afternoon/night or localized aliases supported by the host
- `cross_conversation: bool` only if the product allows it

### `load_material` optional

If the host supports images/files, expose this as a provider-native tool backed by host file/derived storage. Use it after the model has found a `material_trace` anchor in visible raw context, `read_timeline`, or `retrieve_for_turn(categories=["material_trace"])`.

Recommended tool parameters:

- `file_id: str`
- `kind: str` optional, such as image/pdf/file
- `preferred_source: str` optional — `auto`, `original`, `derived`
- `purpose: str` optional

Tell the chat model:

- Use `read_timeline` for "yesterday", "last Tuesday", "that night", exact date ranges.
- Use `retrieve_for_turn` for preferences, plans, long-term facts, people, topics, and fuzzy recall.
- Use `load_material` for historical image/file/PDF content only after a visible or retrieved material anchor provides the `file_id`. If no anchor or retained derived content exists, say the evidence is unavailable instead of guessing.

For provider-native tool loops, prefer `build_native_memory_tool_specs(...)` and
`dispatch_native_memory_tool(...)` over legacy text wrappers. The dispatcher strictly rejects invalid filters instead of broadening them.

## Prompt Requirements

When building the final chat model prompt, include:

- Rendered visible memory from `render_prompt_context`.
- Tool descriptions for `retrieve_for_turn`, `read_timeline`, and optional `load_material`.
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
- `memory_metadata` describes the current raw user message, not the assistant reply.
- `memory_metadata.keywords` should be reusable tags, not sentences. Choose terms likely to be used in a future natural chat query; add broader/field/intent tags only when they improve recall, such as `可乐 / 饮料 / 偏好`.
- If unsure about metadata, use empty arrays and lower `confidence`; do not invent tags.
- Tool calls are not wrapped in memcore JSON. Only the final user-facing reply uses the JSON contract.

## Chat Output Adapter

If the host app can require final JSON output, use:

```python
from memcore import build_chat_output_contract_prompt, parse_chat_output

contract = build_chat_output_contract_prompt(
    categories=cfg.categories,
    enable_flavor=cfg.enable_flavor,
    enable_sentence_segments=True,
)
```

After the model returns:

```python
parsed = parse_chat_output(raw_model_output, mode="memcore_json", categories=cfg.categories)
if not parsed.ok:
    handle_model_output_error(parsed)  # retry or surface a structured failure
    return

mem.update_turn_metadata(cur["source_id"], parsed.memory_metadata)
mem.record_assistant_turn(parsed.speech, in_reply_to=cur)
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
- model can call `read_timeline`;
- assistant reply is recorded;
- background compaction does not block the visible reply;
- group chat preserves actor attribution if used.
