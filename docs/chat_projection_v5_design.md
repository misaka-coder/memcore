# Chat Projection V5

Status: implemented locally; deployment migration remains explicit.
Date: 2026-08-30.

## Goal

Keep MemCore's complete raw evidence, timestamps, retrieval, compaction, and
provider-native tool traces while giving the chat model one concise,
deterministic conversation history format.

V5 separates two concerns that V2 coupled:

```text
provider_output_raw -> durable audit evidence
semantic projection -> model-visible conversation history
```

The raw final response remains stored byte-for-byte. It is no longer replayed
as the content of an ordinary assistant history message.

## Stable chat wire

The provider owns the outer `role` and `content` fields. MemCore renders the
string inside `content`.

Private user message:

```yaml
role: user
content: |
  time: 2026-08-30 20:56 周日
  text:
  小灵聪明
```

Assistant final:

```yaml
role: assistant
content: |
  time: 2026-08-30 20:56 周日
  emotion: 得意
  speech:
  那可不，刚那一圈工具全验通了，链路都顺。
```

`emotion` is optional. Hosts persist it only when it was actually parsed from
the model result; MemCore never guesses a value from speech.

Attributed group input adds only facts that exist:

```yaml
role: user
content: |
  time: 2026-08-30 20:56 周日
  actor: Olivia
  target: Akane
  mentions: [Akane]
  reply_to:
    actor: 千里朱音
    text: 小灵聪明
  text:
  那可不？
```

The quoted snapshot is a relation on the current message, not a second
chronological message. The original quoted entry remains independent in the
timeline.

Silently observed group speech adds `mode: observed`, so it cannot be confused
with a direct request even though both use the provider's `user` role.

The wire never adds `User:`, `Assistant:`, or the final response JSON's unused
fields. Provider roles already express speaker identity.

## Voice finals

Voice messages use the same chat envelope and add only model-relevant delivery
facts:

```yaml
role: assistant
content: |
  time: 2026-08-30 20:56 周日
  medium: voice
  delivery: interrupted
  delivered_units: [0]
  interrupted_units: [1]
  speech:
  好，我接着说。
```

Default successful delivery omits redundant status fields. Internal voice turn,
response, storage, and transport identifiers remain durable host data and do
not enter ordinary model-visible history.

## Unchanged typed surfaces

This change does not flatten all timeline kinds into chat text.

- Native tool calls and results retain provider-native call IDs, names,
  arguments, and result boundaries.
- Generic events retain time, kind, actor/target, correlation, trust, ordered
  structured fields, and semantic content.
- Materials retain their safe anchors and typed status.
- Compact operation cards retain time, source ID, status, stored size, and
  `open_memory` reload instructions.
- Explicit timeline and memory reads retain exact timestamps, attribution,
  relation fields, and source IDs.

Unknown safe event/material fields remain visible through the canonical
fallback. V5 does not introduce an allowlist that would reduce MemCore's open
kind extensibility.

## Storage boundary

An assistant final stores:

```text
provider_output_raw  exact model output for audit and diagnosis
semantic_text        parsed speech for search, compaction, and chat projection
payload.emotion      parsed optional emotion for chat projection
timestamp            authoritative time for ordering, filtering, and rendering
```

Memory annotations remain attached to their annotation target and are not
copied into assistant speech history.

## Versioned migration

V5 is an explicit projection-version upgrade. A request must never discover and
rewrite old frozen rows on its hot path.

The maintenance operation:

1. supports dry-run reporting without returning message bodies;
2. selects only affected chat/voice projection rows below V5;
3. rebuilds them from stored timeline entries for the same provider profile;
4. preserves namespace, turn ID, projection index, source IDs, and lineage;
5. rebuilds an affected settlement only when it embeds changed full rows;
6. increments the conversation projection generation once per changed
   namespace;
7. is idempotent;
8. records no raw message text in its report.

Legacy emotion recovery may parse an existing valid raw final JSON once during
maintenance. Missing, plain, or malformed raw output leaves emotion absent;
the migration never guesses.

The upgrade intentionally creates one cache boundary. After migration, repeated
projections for unchanged history must be byte-identical.

Akane exposes the maintenance entry point as:

```text
python scripts/migrate_chat_projections_v5.py --db <memcore_v01.db> --dry-run
python scripts/migrate_chat_projections_v5.py --db <memcore_v01.db> --apply
```

## Performance contract

The live request path must:

- use the existing batched entry/projection read;
- perform no migration work;
- perform no raw JSON parsing;
- add no LLM or embedding call;
- add no per-entry database query;
- render in linear time over the already selected visible entries;
- preserve background compaction so a visible reply never waits for summary
  generation.

Episode-summary input reuses the same semantic chat renderer for chat entries.
The weekday is a compact suffix inside `time` for both ordinary history and
summary input, so relative dates can be resolved without a second time field.
The summary model therefore receives the same time, actor, target, mention,
reply, forward, observed-mode, voice, emotion, and text/speech facts as the
chat model.  Tool actions/results remain outside episode prose and continue
through the operation-digest path; typed events and materials retain their
dedicated renderers.

Validation compares repeated projection bytes/hashes and measures projection
construction before and after V5 on a long mixed chat/tool fixture. The upgrade
is rejected if it introduces query-count growth proportional to the number of
messages beyond the existing batched reads.

## Acceptance

- Plain and JSON-producing models create the same assistant history wire.
- Historical assistant content contains no `Assistant:`, `tool_call: null`,
  empty `choices`, empty persona, status, state request, or raw memory metadata.
- Time remains visible and exact-time retrieval remains unchanged.
- Group actor, target, mention, and quoted text survive projection.
- Parsed emotion survives without replaying raw JSON.
- Voice interruption/delivery facts survive without exposing internal IDs.
- Native tool traces remain byte-equivalent.
- Event/material canonical fallback retains unknown safe fields.
- Compaction source lineage and memory IDs remain unchanged.
- Migration dry-run and repeated apply are deterministic and idempotent.
- The first request after migration may miss the old cache; subsequent
  unchanged prefixes are stable.
