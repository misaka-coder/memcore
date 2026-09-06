# Chat authorship V6 restoration

This release restores the previously implemented V6 chat projection on top of
the `42658a3` action-only completion contract. Its distinct package version is
`0.1.0+chatv6.1`, so an old `0.1.0` wheel cannot be mistaken for this release.

- Assistant final history preserves accepted provider output, without adding
  timestamps or role labels. Plain and JSON output both retain authorship.
- User history retains time, speaker and real reply/mention relationships.
- Summary input uses a separate semantic transcript, not the provider envelope.
- `migrate_chat_projections_v6` explicitly rebuilds derived legacy projections;
  it does not rewrite raw messages or run on the request hot path.
- `complete_turn(append_final=False)` remains available.

The source snapshot includes only the chat projection, related summary rendering,
migration implementation and their regression tests. Research harness changes
are not included. The original working directory remains unchanged.

Acceptance must check both the action-only contract and V6 authorship behavior
using the actual deployed interpreter. Checking a commit ID alone is insufficient.
Back up databases before migration. Model-authored erroneous history needs an
explicit, separately audited repair; projection migration does not rewrite it.
