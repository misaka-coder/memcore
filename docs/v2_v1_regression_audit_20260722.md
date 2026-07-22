# MemCore V2 / V1 regression audit (2026-07-22)

## Conclusion

V2's timeline, projection, turn/relation, atomic commit and lineage model are
strict improvements over V1.  The regression was in raw compaction planning,
not in the V2 storage model: later relay-safety changes overlaid independent
source-token and episode-entry stops on top of the token difference target.

The repaired authority is:

```text
raw_token_trigger
  -> raw_token_batch_ratio
  -> oldest complete terminal turn/relation components
  -> one atomic summary batch + source lineage commit
```

`summary_batch_size` is count-policy-only.  It does not constrain V2
projected-token planning.

## Repaired regressions

1. Removed the independent `compaction_max_source_tokens` stop from V2 batch
   selection.
2. Removed the reuse of `summary_batch_size` as a projected-token episode-entry
   stop.
3. Reused V1's token trigger and batch ratio for V2, while counting the actual
   provider raw projection rather than plain message text.
4. Kept complete terminal turn/relation components as the only legal V2 cut
   boundary.
5. Restored V1's oversized-first-complete-turn escape hatch: when no recent
   tail can legally remain, a complete terminal history can be compacted as a
   whole instead of remaining permanently above the trigger.
6. Added separate raw before/after/planned/selected token observability so
   future cache regressions can be distinguished from total visible-memory
   size.

## V2 capabilities that must remain

- `closed` / `aborted` / `open` turn lifecycle;
- complete parallel action/observation correlation components;
- provider-specific immutable projection ledger and prefix audit;
- snapshot generation, row-version and projection-hash validation;
- atomic episode/operation source partition and lineage commit;
- operation digest separation from personal semantic facts;
- relation-aware retrieval expansion and visible-lineage duplicate exclusion;
- namespace, visibility, kind and index-generation hard filters.

Rolling these back to the flat V1 message path would reintroduce split tool
turns, non-atomic summary/source writes and provider-history reconstruction.

## Remaining cleanup candidates (not changed in this slice)

### 1. Two compatibility policy selectors

`raw_compaction_policy=count|token` still selects the legacy flat-message path,
while `compaction_policy=projected_tokens|count_compat` selects the V2 planner.
This is a documented migration seam, but it is a confusing public config
surface.  Once legacy Timeline rows no longer need the V1 path, collapse this
to one raw compaction policy and delete the non-atomic legacy writer rather
than keeping two authorities indefinitely.

### 2. Misnamed reservation fields

`reserved_retrieval_tokens` currently also acts as the retrieval result token
budget, while `reserved_current_turn_tokens` no longer participates in raw
compaction planning.  Their names no longer match their runtime authority.
A later isolated cleanup should introduce an explicit retrieval-result budget
and remove dead prompt-reservation state; it must not be mixed into this cache
repair.

### 3. Legacy non-atomic summary path

The flat V1 compatibility branch still performs `add_summary()` and
`mark_messages_summarized()` as separate calls.  V2 does not use it.  After
legacy migration is closed, delete this branch instead of upgrading and
maintaining a second compactor.

## Not regressions

- Estimated projected-token counting without a provider tokenizer is explicit
  (`token_count_quality=estimated`) and acceptable as planning approximation;
  it does not split stored entries.
- One raw generation per `compact_due()` remains bounded maintenance.  With
  ratio planning restored, that generation reaches the intended cut instead
  of producing a chain of tiny prefix rewrites.
- Episodic-to-semantic count compaction remains a separate layer and should not
  inherit raw token policy.
