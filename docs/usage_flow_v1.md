# memcore usage flow v1

This document is for humans and AI coding agents integrating `memcore` into a
host chat application.

The goal is to use the public `MemorySystem` facade without reading private
implementation modules for normal integration work.

## Package Role

`memcore` is a reusable layered memory kernel. It owns:

- raw user/assistant turn recording;
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
6. `docs/metadata_prefilter_design_v1.md` - retrieval filter semantics.
7. `docs/raw_token_compaction_policy_v1.md` - optional token-triggered raw
   compaction.

If you are changing `memcore` itself, also inspect nearby tests before editing.

## Normal Chat Turn

Use `MemorySystem` as the facade:

```python
cur = mem.record_user_turn(user_text, actor=actor_or_none, timestamp=now_ts)

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
    categories=mem.config.categories,
    enable_flavor=mem.config.enable_flavor,
)
if not parsed.ok:
    handle_model_output_error(parsed)
    return

mem.update_turn_metadata(cur["source_id"], parsed.memory_metadata)
mem.record_assistant_turn(parsed.speech, in_reply_to=cur, timestamp=now_ts2)

mem.compact_due_background()
```

For tests, scripts, or deterministic shutdown, use `compact_due_sync()`.
For live chat, prefer `compact_due_background()` so summarization does not block
the visible reply.

## Required Host Pieces

A production host must provide or choose:

- `LLMClient` for summary, semantic, reinforcement, and verifier calls.
- `EmbeddingProvider` for semantic retrieval. Do not use
  `HashedEmbeddingProvider` in production.
- `Namespace` values for tenant/user/domain/conversation isolation.
- Optional `Actor` for group or multi-speaker messages.
- Optional `TokenCounter` when `raw_compaction_policy="token"`.
- Store/index lifecycle. SQLite is the source of truth; vector indexes are
  search acceleration.

## Retrieval Tools To Expose

Expose two memory tools to the final chat model:

- `retrieve_for_turn(current=cur, ...)` for fuzzy preferences, plans, people,
  topics, and long-term facts.
- `read_timeline(...)` for exact date/time questions such as yesterday, last
  Tuesday, or a date range.

Prefer `retrieve_for_turn` over direct `retrieve` during live turns because it
excludes memory already visible in the prompt and excludes the current user
message.

Do not turn invalid filters into broad successful searches. Return structured
tool errors according to host policy.

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
4. Expose `retrieve_for_turn` and `read_timeline` as model tools.
5. Add the model prompt guidance and optional JSON contract.
6. Parse the final reply before storing assistant speech.
7. Update current user-turn metadata from parsed `memory_metadata`.
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
