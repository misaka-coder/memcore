# Trace context and compaction execution v1

This document records the current trace lifecycle after the Timeline V2
cutover. The former flat-message count compactor described by older revisions
has been deleted.

## One timeline, different policies

Messages, events, actions, observations, skills, materials, and final model
responses all live in the SQLite `messages` fact table as typed timeline
entries. Their storage is unified; their retrieval and compaction behavior is
not forced to be identical.

- normal annotated stimuli and their final responses are default retrieval
  candidates;
- actions, observations, and material references are explicit retrieval
  candidates;
- a typed event that receives a valid final memory annotation can participate
  in normal retrieval with its final response;
- a standalone unannotated event remains explicit;
- `semanticize=False` prevents operational rows from becoming long-term user
  facts.

These decisions use typed fields (`kind`, `turn_role`, `annotation_status`,
`retrieval_policy`, `retrieval_visibility`, `semanticize`) rather than a second
storage area or an ever-growing category exclusion tuple.

## Turn and relation structure

```text
turn
  stimulus(s)
  action A ── correlation_id A ── observation A
  action B ── correlation_id B ── observation B
  intermediate event/material/skill entries
  final response
```

Actions and observations may be interleaved or returned out of order. The
store validates relation ownership and computes component boundaries from
explicit lineage, not adjacency.

Use:

```python
handle = mem.begin_turn(stimuli=[...])
mem.append_action(turn_id=handle.turn_id, kind="tool.search.call", correlation_id="a", payload={...})
mem.append_observation(turn_id=handle.turn_id, kind="tool.search.result", correlation_id="a", payload={...})
mem.complete_turn(turn_id=handle.turn_id, ...)
```

`record_tool_exchange(turn_id=...)` is only a convenience adapter over the two
typed append calls. It cannot create unrelated flat tool rows.

## Compaction partitions

The token/ratio planner selects complete terminal components. Within the same
atomic generation:

- dialogue, annotated events, skills, and final responses become
  `memory.episode_summary`;
- action/observation entries and `material.*` entries become
  `memory.operation_digest`;
- source sets cannot overlap and must exactly partition the selected snapshot;
- optional sanitized `retention_anchor` values may survive in the operation
  digest; full tool payloads and sensitive paths do not.

Structured event/skill/intermediate payloads are rendered with the same
versioned renderer registry used by provider projection before summarization.
This prevents a short `semantic_text` from silently dropping payload facts.

## Retrieval

Default retrieval excludes explicit traces through indexed visibility and kind
filters before scoring. A host may authorize them with `include_explicit=True`
and concrete `kind_patterns`. Relation expansion then returns the linked
stimulus/final or action/observation group without treating nearby unrelated
rows as context.

The retrieval result size uses `retrieval_result_token_budget`; it is separate
from raw compaction thresholds and does not alter stored lineage.

## Failure behavior

- summary LLM failure: `summary_retry_pending`, no source marks committed;
- open prefix: `blocked_by_open_turn`;
- stale projection or row version: structured stale result, no duplicate
  summary;
- vector index failure: SQLite commit survives with pending outbox state;
- retry of an already committed generation: idempotent or structured stale,
  never a second lineage owner.
