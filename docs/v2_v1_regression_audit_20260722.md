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

### 4. Legacy count invariants still constrain V2 configuration

`MemoryConfig.validate()` still unconditionally requires
`summary_batch_size < raw_trigger_count`, even when the active V2 policy is
`compaction_policy="projected_tokens"` and neither count field participates in
planning.  The same compatibility surface also validates
`raw_compaction_policy` and can require a `TokenCounter` for the flat V1 path
while V2 itself explicitly supports an estimated token-count quality.

This is migration leakage, not a V2 safety boundary.  When the flat writer is
removed, remove these inactive constraints from the V2 construction path as
well.  Do not replace them with new projected-token restrictions.

### 5. Public configuration still contains fields with no V2 authority

`raw_compaction_excluded_categories` belongs to the flat raw-message
compactor.  `retrieval_default_excluded_categories` is retained in config, but
Retrieval V2 admission is governed by retrieval policy/visibility, annotation
status, kind and trust instead.  Keeping fields that appear configurable but
do not control the active path makes the package look more flexible than it
is and obscures which policy is authoritative.

The eventual cleanup should delete or explicitly migrate these fields; it
must not add another parallel interpretation of them.

### 6. Token budgeting needs one explicit host contract

V2 compaction deliberately labels its byte-based fallback as
`token_count_quality=estimated` when no provider tokenizer is injected.
Relation-aware retrieval, however, only applies the default result token
budget when a `TokenCounter` exists.  A host that omits the counter therefore
gets estimated compaction planning but unbudgeted retrieval relation groups.

This is not a reason to make MemCore silently pretend it has an exact
tokenizer.  The clean follow-up is one explicit host-provided counter (exact or
clearly labelled estimated) shared by compaction and retrieval.  Akane
currently does not inject one, so this remains a real integration gap.

## Retrieval behavior checked during this audit

- A V2 external event completed with accepted memory annotation can enter
  default retrieval together with its related final model reply.
- An unannotated standalone event remains explicit-only.
- Tool and material relations remain explicit-only by default.
- Relation expansion returns the annotation target and final reply as one
  group, so intermediate tool rounds do not become the owner of final memory
  metadata.

These are deliberate V2 improvements and should not be weakened while the
compatibility surface is removed.

## Not regressions

- Estimated projected-token counting without a provider tokenizer is explicit
  (`token_count_quality=estimated`) and acceptable as planning approximation;
  it does not split stored entries.
- One raw generation per `compact_due()` remains bounded maintenance.  With
  ratio planning restored, that generation reaches the intended cut instead
  of producing a chain of tiny prefix rewrites.
- Episodic-to-semantic count compaction remains a separate layer and should not
  inherit raw token policy.
