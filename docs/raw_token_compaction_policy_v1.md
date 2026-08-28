# Raw token compaction policy v1

This document describes the current and only raw compaction policy.

## Policy

MemCore measures the unsummarized raw timeline as the exact provider-visible
projection selected for the conversation. Compaction becomes due when:

```text
raw_projected_tokens >= raw_token_trigger
```

It then plans to remove:

```text
ceil(raw_token_trigger * raw_token_batch_ratio)
```

tokens from the oldest prefix. Token count chooses the approximate amount;
complete terminal turn/relation components choose the actual cut point. A
component may exceed the planned amount. MemCore never cuts a user stimulus
away from its final response or splits correlated action/observation lineage.

The recent tail is controlled by `compaction_min_recent_turns`. One oversized
terminal turn is still allowed to compact by itself, so a complete history
cannot remain permanently over the trigger merely because no second turn exists.
An open oldest component returns `blocked_by_open_turn` instead of guessing a
boundary.

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
- `planned_source_tokens` / `selected_projected_tokens`;
- `source_turn_count` / `source_entry_count`;
- `compaction_generation`;
- `token_count_quality`.

One call advances at most one raw generation and one semantic batch. Repeated
background scheduling drains a backlog without a single request loop calling
the summary model until the namespace is empty.

Compaction retries one stale provider-projection snapshot after reloading the
SQLite timeline. It never treats a longer request-frozen projection as evidence
that history was rewritten. If the projection changes again during that bounded
retry, the call returns `stale_projection/projection_generation_changed` and
leaves all raw records untouched for the next background pass.
