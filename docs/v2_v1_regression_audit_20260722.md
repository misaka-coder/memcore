# Timeline V2 / V1 regression audit — 2026-07-22

Status: completed repair pass.

## Decision

Timeline V2 is the sole runtime write/compaction authority. V1 is not retained
as a second rollback implementation; rollback is provided by Git and release
artifacts.

The shared SQLite `messages` table remains the fact source. Keeping the table
does not mean keeping the flat V1 compactor. Existing rows without `turn_id`
are read as closed standalone components and processed by the V2 planner.

## Removed runtime paths

- flat `_summarize_raw` compactor;
- count and token selector pair;
- role-based assistant cut point;
- raw category exclusion selector;
- non-atomic `add_summary + mark_messages_summarized` facade path;
- count-policy and reserved-budget fields that did not control V2 behavior;
- `MemorySystem._record → SQLiteMemoryStore.add_message` as a facade writer.

Removed public configuration fields:

```text
raw_trigger_count
summary_batch_size
raw_compaction_policy
raw_token_min_remainder_messages
raw_token_boundary_role
raw_compaction_excluded_categories
compaction_policy
reserved_current_turn_tokens
reserved_retrieval_tokens
retrieval_default_excluded_categories
```

`retrieval_result_token_budget` now names the retrieval output budget directly.

## Preserved and upgraded V1 properties

- token-triggered differential compaction;
- configurable compression ratio;
- recent raw tail;
- raw → episodic → semantic lifecycle;
- failed summaries remain retryable;
- SQLite truth survives vector-index failure.

V2 adds:

- complete turn/relation component cut points;
- parallel action/observation correlation;
- provider projection token accounting;
- atomic multi-partition summary commit;
- compaction generation and projection lineage;
- idempotent/stale retry detection;
- typed standalone history migration.

## Public writer status

The authoritative live-turn API is:

```text
begin_turn → append_entry / append_action / append_observation
           → complete_turn / abort_turn
```

Convenience `record_user_turn`, `record_assistant_turn`, external-event, and
material methods now call `append_standalone_entry`; they do not write through a
flat facade path. They are suitable only for records that intentionally do not
expect a model-response turn.

`record_tool_exchange` requires an open `turn_id` and delegates to typed action
and observation APIs.

## Verification

The repair pass covers:

- token ratio planning and recent-tail behavior;
- no partial turn/component selection;
- oversized single-turn escape hatch;
- open-turn blocking;
- parallel tool partitioning;
- atomic rollback when a summary write fails;
- stale/idempotent retry behavior;
- legacy standalone backlog migration;
- summary failure retry and outbox repair;
- structured event payload preservation in summary prompts;
- exact versus estimated token-count quality.

At completion of the code pass:

```text
python -m unittest discover -s tests
310 tests passed
```
