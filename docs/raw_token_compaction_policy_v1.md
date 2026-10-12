# Raw token compaction policy v1

This document describes the current and only raw compaction policy.

## Policy

Since 0.1.7, ordinary raw compaction measures only terminal, unsummarized
history using its current provider-visible projection. An open turn's full
tool calls/results do not contribute to its trigger or retained-tail target.
Compaction becomes due when:

```text
terminal_raw_projected_tokens >= raw_token_trigger
```

It then plans to remove:

```text
ceil(raw_token_trigger * raw_token_batch_ratio)
```

tokens from the oldest prefix. Token count chooses the approximate amount;
complete terminal turn/relation components choose the actual cut point. A
component may exceed the planned amount. MemCore never cuts a user stimulus
away from its final response or splits correlated action/observation lineage.

In `chat_prefix` mode a large completed backlog removes at least the current
terminal raw volume minus the target retained tail:

```text
retained = raw_token_trigger - ceil(raw_token_trigger * raw_token_batch_ratio)
planned = max(ceil(raw_token_trigger * raw_token_batch_ratio),
              terminal_raw_projected_tokens - retained)
```

Open volume is excluded from both quantities. Complete the turn, apply its
configured terminal settlement, then call `compact_due_background()`; a full
history policy continues counting the full completed projection. Maintenance
may still run during later tool requests for independently due older history.

The recent tail is controlled by `compaction_min_recent_turns`. One oversized
terminal turn is still allowed to compact by itself, even if a new turn has
started, so a complete history cannot remain permanently over the trigger
merely because no completed recent tail exists. Only terminal turns count
toward that recent-tail protection. If independently due terminal history
lies behind an open oldest component, it returns `blocked_by_open_turn`
instead of guessing a boundary. Open-only pressure returns `not_due`.

`complete_turn()` has a short process-local completion boundary shared by
systems on the same store/conversation runtime. A turn already marked closed
but still applying its terminal settlement is not yet eligible for ordinary
triggering or selection. Other completed history may proceed; no database
flag, task queue, wait for a conversation to become idle, or provider lock is
added. The boundary clears on success and failure. Restart does not retain it.

Open tools still count toward the complete provider request budget and the
separate `working_token_trigger` mechanism. Those mechanisms do not borrow
the ordinary raw threshold. An older summary can commit while a later turn
is running and appear at its next request boundary; it does not consume the
later open turn or mutate an already sent request.

## Configuration

```python
MemoryConfig(
    raw_token_trigger=24_000,
    raw_token_batch_ratio=0.67,
    compaction_min_recent_turns=1,
    projection_profile="openai_chat",
)
```

- `raw_token_trigger` must be a positive integer.
- `raw_token_batch_ratio` must be greater than `0` and less than `1`.
- `compaction_min_recent_turns` must be a positive integer.
- There is no count policy, message batch size, assistant-role cut point, or
  category-exclusion selector in the raw compactor.

## Token counter quality

A host may inject a model-aligned `TokenCounter`:

```python
class ModelTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return tokenizer_count(text)
```

The default `TokenCounter.quality` is `exact`. An estimator must override it:

```python
class EstimatedTokenCounter(TokenCounter):
    @property
    def quality(self) -> str:
        return "estimated"
```

If no counter is provided, MemCore remains usable and applies its conservative
UTF-8 estimate. Compaction results always expose `token_count_quality`, so an
estimate is never presented as an exact provider tokenizer.

## Interaction with terminal operation settlement

`MemoryConfig.operation_projection_policy="compact_after_terminal"` may replace
large closed-turn observation bodies with deterministic reloadable cards before
raw compaction becomes due. The raw planner counts the same settled provider
projection that `build_context_projection()` will actually send, so it does not
keep charging the removed full body or maintain a second hidden token window.

This changes projected pressure, not raw truth or compaction boundaries. The later
raw compactor still selects complete terminal components, creates episodic and
operation-derived records, and preserves source lineage. Settlement metrics and raw
compaction result fields are separate observability surfaces. See
[`operation_projection_settlement_v1.md`](operation_projection_settlement_v1.md).

## Atomicity and lineage

For each selected generation MemCore:

1. freezes the selected provider projections and row versions;
2. partitions dialogue/event entries from operation/material entries;
3. creates an episode summary and/or operation digest;
4. commits summaries, source marks, lineage, and generation in one SQLite
   transaction;
5. leaves every source untouched on LLM failure, stale snapshot, or write
   failure.

Parallel tools are related by `turn_id` and `correlation_id`, not physical
adjacency. Old records without a `turn_id` are wrapped as closed standalone
components and pass through this same planner and atomic commit path.

## Result fields

Important fields returned by `compact_due_sync()` / the background future:

- `status`: `not_due`, `compacted`, `blocked_by_open_turn`, `failed`, `busy`, or
  `stale_projection` when two bounded reads both lose a race to a valid request
  append;
- `before_raw_projected_tokens` / `after_raw_projected_tokens`;
- `before_terminal_raw_projected_tokens` / `after_terminal_raw_projected_tokens`;
- `planned_source_tokens` / `selected_projected_tokens`;
- `source_turn_count` / `source_entry_count`;
- `compaction_generation`;
- `token_count_quality`.

The raw before/after diagnostics still include the visible open-turn projection;
the terminal raw before/after fields are the ordinary trigger/removal basis.
`before_projected_tokens` / `after_projected_tokens` also include visible memory.
These are counts of the captured maintenance snapshot, not a promise that no
new messages arrived while the provider was generating a summary. A completion
still settling may return `not_due` with `reason=terminal_settlement_in_progress`.

One call advances at most one raw generation and one semantic batch. Repeated
background scheduling drains a backlog without a single request loop calling
the summary model until the namespace is empty.

Compaction retries one stale provider-projection snapshot after reloading the
SQLite timeline. It never treats a longer request-frozen projection as evidence
that history was rewritten. If the projection changes again during that bounded
retry, the call returns `stale_projection/projection_generation_changed` and
leaves all raw records untouched for the next background pass.
