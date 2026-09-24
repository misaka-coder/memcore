# Chat authorship V6.2

This release builds on `0.1.0+chatv6.1`. Its distinct package version is
`0.1.0+chatv6.2`, so a `chatv6.1` or plain `0.1.0` wheel cannot be mistaken for
this release.

## Retrieval: keyword catalog search

`browse_memory` accepts `keywords` and `keyword_match` (`any` / `all`) as an
alternative entry point to date-only navigation. Matching is one-way against the
deduplicated union of summary and exact-source tags: a stored `文旅答辩项目`
matches the query term `文旅`, but a stored `文旅` does not match the longer query
`文旅答辩项目`. Exact and substring hits are returned together.

Cards now expose `matched_terms` and `keyword_hits` with one original stored term
per query. These are navigation clues, not evidence that two names share an event
or relationship, and a missing hit does not prove there is no history.

The catalog selector is skipped when keywords are supplied without date bounds.
`start_ts` / `end_ts` become `int | None` at the store contract and resolve to
`0` / `2**63 - 1` only inside SQLite. `CATALOG_SCHEMA_VERSION` stays `1`.

An incomplete page continues with `cursor` alone. If continuation returns
`catalog_changed_restart_required`, the original query must be repeated.

## Read path performance

- `MemoryStore.get_retrieval_records` batches by `source_id`: the base class
  keeps a compatibility loop, SQLite issues chunked reads of 400 inside a single
  lock.
- Lineage resolution uses a request-local BFS `prefetch_children` frontier
  instead of a per-node round trip.
- BM25 `keyword_tf` / `keyword_len` are computed at index upsert.
- `validate_only` skips rendering entirely when only a match count is needed.
- `render_timeline` and `paginate_timeline_units` memoize render rows through
  `_message_cache`.
- `get_raw_turn_window` selects only `source_id` / `turn_id`.
- Single and batch `open_memory` share `_open_memory_record`.

Measured on one fuzzy retrieve: 2,667 SELECTs before, 45 after. The benchmark is
`examples/benchmark_memory_reads.py`; the write-up is
[memory_read_performance_20260922.md](memory_read_performance_20260922.md).

## Research harness

The long-arc pilot harness and its recorded runs under `examples/research_pilot/`
and `docs/research/pilot_v1..v4/` are included. They document actual compaction,
history readback and restart recovery, and they also record delivery and
attribution failures. They are not a comprehensive reliability validation.

## Documentation

Contracts were repositioned and Akane integration notes added. Design story and
video storyboard material lives in
[design_story_and_video_v1.md](design_story_and_video_v1.md) and
[video_storyboard_v2.md](video_storyboard_v2.md).

## Validation

```bash
uv run --extra dev python -m unittest discover -s tests -v   # 778 tests, OK (skipped=9)
uv run --extra dev ruff check .                              # All checks passed
uv run --extra dev ruff format --check .                     # 144 files already formatted
git diff --check                                             # clean
uv run --extra dev python -m build                           # sdist + wheel
```

Back up databases before any migration. Projection migration does not rewrite raw
messages and does not run on the request hot path.
