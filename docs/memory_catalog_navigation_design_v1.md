# Memory Catalog & Evidence Navigation Design V1

> Status: implemented; Slice A-D in MemCore and Slice E in the Akane host
> Date: 2026-08-05  
> Scope: MemCore package; host products only provide identity, authorization, prompt assembly,
> and channel delivery

## 1. Objective

MemCore already preserves three authoritative memory layers:

```text
raw timeline -> episodic summary -> semantic summary
```

The missing part is not another memory layer. It is a compact, deterministic way for a model to
navigate those layers without first loading an entire raw date range or permanently injecting all
historical summaries into every prompt.

This design adds a **catalog projection** and a closed evidence-navigation protocol:

```text
browse compact cards
  -> open one node's content
    -> open its exact source evidence
      -> read complete raw turns when detail is required
```

The model remains responsible for deciding which path fits the user's question. MemCore provides
clear tool semantics, exact coverage, namespace-safe lineage, complete-unit pagination, and
structured feedback when a request is too large.

The intended user experience is:

- a precise question such as “What did we discuss from 11:00 to 12:00 that day?” reads raw turns
  directly;
- an overview such as “What did we discuss between July 22 and July 25?” first receives compact
  summary cards, then opens only relevant summaries;
- an uncertain question such as “Who came with me to breakfast?” performs semantic retrieval
  using only known clues and may narrow the search to a selected memory node;
- no result is silently truncated, no missing answer entity is guessed, and no verifier model adds
  latency after retrieval.

## 2. Non-goals

This design does not:

- create a fourth memory truth table;
- duplicate raw messages into a separate “catalog message” store;
- delete or rewrite old raw evidence;
- introduce an LLM router or an LLM retrieval verifier;
- route business meaning with keyword or regular-expression rules;
- force every time query through summaries;
- inject every historical card into every normal chat request;
- make QQ, desktop-pet, or provider-specific fields part of MemCore;
- split one existing compaction generation into LLM-invented raw subchapters in V1;
- make summaries a substitute for exact evidence when the user asks for exact wording or detail.

## 3. Current behavior and the concrete failure

### 3.1 Existing data is not being deleted

The `summaries` table already retains episodic summaries after semantic reinforcement. The normal
visible-summary query only returns rows with `is_semanticized = 0`, so an older episodic summary
stops being injected as full prompt content but remains in SQLite with exact `source_ids_json`.

Likewise, a semantic summary stores exact `source_summary_ids_json`. The store can already resolve:

```text
semantic_id -> summary_ids -> raw source_ids
```

Therefore, “keep an old summary as a compact title” is a new projection over existing truth, not a
new retention mechanism.

### 3.2 Flat raw range reads do not scale in active group chats

An actual group-chat request for a several-day overview selected thousands of timeline entries and
rendered hundreds of thousands of provider tokens. The provider was never called because the host
rejected the oversized prompt locally.

The harmful chain was:

```text
broad time request
  -> raw read with page_token_budget=0
  -> zero interpreted as unlimited
  -> host treated result as producer-bounded
  -> generic shaping was bypassed
  -> giant tool result entered the model-visible timeline
```

This is not primarily a semantic-retrieval failure. It is a prompt orchestration and result-envelope
failure: the system exposed the wrong representation at the wrong scale.

### 3.3 A fixed visible-summary window is not a historical index

Recent full summaries are useful as ordinary context. They are not enough to answer “what happened
during these older days” because older summaries disappear from the normal visible window even
though they are still stored.

Increasing the visible window indefinitely would merely create a slower version of the same problem.
The correct distinction is:

```text
recent summaries: full content may be prompt-visible
older summaries: catalog-visible and openable on demand
```

## 4. Design principles

### 4.1 SQLite and lineage remain authoritative

SQLite remains the truth source. The vector index remains an accelerator. Catalog cards never become
independent facts and can always be regenerated from a memory node and its lineage.

### 4.2 Representation changes, evidence does not

A memory node has three projections:

1. `card`: compact navigation metadata;
2. `content`: the node's full derived summary content;
3. `sources`: exact child evidence obtained through lineage.

The projections do not create new truth or change the source graph.

### 4.3 Model and system cooperate explicitly

The stable model prompt explains when each tool is useful. Runtime results state what was returned,
what remains, whether coverage is complete, and which next actions are valid. The model must not need
to infer a hidden truncation or reverse-engineer internal storage behavior.

### 4.4 Direct raw evidence remains first-class

Known time ranges and known raw anchors are allowed to read raw evidence directly. Summary navigation
is an additional route for breadth, not a mandatory detour for every question.

### 4.5 No guessed answer entity

If the user asks “who came with me,” the model knows the event but not the person's name. It should
search using known facts such as breakfast, location, date, relationship, or selected summary lineage.
An unknown answer value must not be fabricated as an `entity_anchor`.

### 4.6 All model-facing bulk results are bounded and lossless

Pagination is by complete cards or complete logical raw units. A page boundary may delay later evidence
but may not split a turn, omit data without a cursor, or claim complete coverage when only a prefix was
returned.

## 5. Unified memory-node model

### 5.1 Node types

V1 exposes three node types:

| Node type | Stable ID | Content | Direct children |
| --- | --- | --- | --- |
| `raw` | `source_id` | one projected timeline entry | none |
| `episodic` | `summary_id` | episode summary | raw entries grouped as complete turns |
| `semantic` | `semantic_id` | reinforced long-term summary | episodic summary cards |

The graph is:

```text
semantic node
  -> episodic nodes
      -> raw entries / complete turns
```

### 5.2 Generic node identity

Public navigation results use `memory_id` and `node_type`. Callers must not infer a table from an ID
prefix. Store lookup resolves the ID inside the current authorized namespace.

```json
{
  "memory_id": "...",
  "node_type": "episodic"
}
```

Duplicate IDs across the three tables are invalid storage state. Migration and write tests must reject
or diagnose them rather than selecting a table silently.

### 5.3 Card projection

An episodic card has this logical shape:

```json
{
  "memory_id": "...",
  "node_type": "episodic",
  "memory_title": "扬州行程、早茶账单和同行人员",
  "catalog_hint": "讨论当天早茶、同行人员、账单和后续安排",
  "topic_headings": ["早茶与账单", "同行人员", "后续安排"],
  "period_start_ts": 0,
  "period_end_ts": 0,
  "participant_refs": [],
  "source_turn_count": 0,
  "source_entry_count": 0,
  "importance": 0.0,
  "catalog_quality": "generated",
  "has_content": true,
  "has_sources": true
}
```

The card is intentionally small. It does not contain `diary_summary`, full key events, full facts, raw
messages, tool payloads, or material bodies.

### 5.4 Deterministic and model-generated fields

MemCore computes these fields from committed source data:

- `memory_id`, `node_type`;
- `period_start_ts`, `period_end_ts`;
- `participant_refs` using normalized actor identity already present in the generic timeline;
- `source_turn_count`, `source_entry_count`;
- `importance`;
- `has_content`, `has_sources`;
- lineage and namespace status.

The summary model generates:

- `memory_title`;
- `catalog_hint`;
- optional `topic_headings`.

The model must not generate IDs, counts, participants, timestamps, visibility, namespace, or lineage.

### 5.5 Title contract

`memory_title` is a short noun phrase that distinguishes the episode from adjacent episodes. It should
name concrete topics, events, places, or people only when supported by the source transcript.

`catalog_hint` is one concise sentence describing what questions the episode may answer. It is not a
second summary and must not introduce unsupported conclusions.

`topic_headings` is optional. It helps when one token-compaction batch contains several real topics,
especially in a group chat. It does not partition lineage in V1.

Recommended output constraints are semantic, not brittle character-count rejection:

- title: normally one short phrase;
- hint: normally one sentence;
- headings: zero or a small number of distinct phrases;
- empty fields are accepted only when the underlying episode truly has no conversational subject;
- invalid or missing catalog fields do not invalidate the whole summary transaction; deterministic
  fallback projection remains available and the result reports fallback quality.

### 5.6 No raw subchapter partition in V1

Current compaction passes a rendered transcript to the summary model without source IDs. Asking the
model to assign arbitrary raw IDs to several chapters would be unverifiable and could corrupt lineage.

V1 therefore keeps one exact source set per episodic summary. `topic_headings` only improves navigation.
Subchapter lineage may be evaluated later only if the model receives stable logical-unit IDs and the
system validates a complete, non-overlapping partition.

## 6. Storage and migration

### 6.1 Schema update

The next SQLite migration adds catalog fields to both derived-memory tables.

Episodic summaries:

```text
memory_title TEXT NOT NULL DEFAULT ''
catalog_hint TEXT NOT NULL DEFAULT ''
topic_headings_json TEXT NOT NULL DEFAULT '[]'
participant_refs_json TEXT NOT NULL DEFAULT '[]'
source_turn_count INTEGER NOT NULL DEFAULT 0
source_entry_count INTEGER NOT NULL DEFAULT 0
catalog_schema_version INTEGER NOT NULL DEFAULT 0
```

Semantic summaries use the same title/hint/headings/version fields. Their participant and source counts
are projected from descendant episodic nodes and need not duplicate raw-level actor rows unless profiling
shows the join to be expensive.

The migration also adds an overlap-oriented episodic index under the existing hard scope:

```text
(tenant_id, user_id, domain_id, conversation_id,
 period_start_ts, period_end_ts, timestamp, summary_id)
```

Implementations may use two indexes after query-plan inspection, but must not denormalize cards into a
new truth table merely for convenience.

### 6.2 New summary writes

Compaction freezes source projections as it does today. Before the atomic commit, MemCore:

1. computes participant and count fields from the frozen logical units;
2. asks the existing summary call for summary content plus title/hint/headings;
3. validates only the model-generated catalog fields;
4. commits the episode, exact source IDs, source marks, deterministic catalog data, and generation in
   the existing transaction;
5. leaves sources untouched on summary failure, stale snapshot, or write failure.

The catalog extension must not introduce a second summary-model call for normal new compaction.

### 6.3 Existing summaries

Migration must not synchronously re-summarize the database and must not block startup on model access.

Old nodes immediately receive a deterministic fallback card:

1. prefer the first non-empty existing `period_label` plus a concrete `key_event`;
2. otherwise derive a short display phrase from existing summary content;
3. otherwise use a localized time-range label;
4. mark `catalog_quality="fallback"` and `catalog_schema_version=0`.

Fallback generation is deterministic and contains no new factual claim. A bounded maintenance command
may later upgrade old cards by asking the configured summary model to title the already-existing summary
content. It must:

- operate in batches with a resumable cursor;
- update only catalog fields and row version;
- mark the derived index pending when catalog text is indexed;
- never rewrite summary content or lineage;
- return per-node structured failure status;
- not run automatically on every process startup.

### 6.4 Index behavior

Catalog browsing is a deterministic SQLite range operation, not vector Top-K. Semantic retrieval may
index title, hint, and headings as additional document text, but those fields never replace the original
query or exact metadata prefilters.

After migration, `reindex_all()` can rebuild derived entries with current catalog text. SQLite remains
usable if the vector index is unavailable.

## 7. Public navigation API

The long-term model-facing protocol has four complementary operations. Existing names are retained where
they are already clear; awkward compatibility paths are made thin adapters rather than permanent parallel
implementations.

### 7.1 `browse_memory`: deterministic catalog browse

Purpose: obtain compact cards for a time range or recent history without semantic Top-K.

Conceptual request:

```python
browse_memory(
    time_range={"start_at": "2026-07-22", "end_at": "2026-07-26"},
    node_types=["episodic"],
    cursor="",
)
```

Rules:

- at least one deterministic selector is required;
- V1 primarily browses episodic nodes; semantic nodes may be explicitly requested;
- range matching uses interval overlap, not the summary's display date alone;
- time semantics are half-open: `start <= event < end`;
- date-only end values follow the existing local-date normalization contract;
- hard namespace and conversation authorization are never relaxed;
- results are chronological by period start, timestamp, and stable ID;
- pagination uses complete cards and a signed/validated selector cursor;
- it returns all matching cards across pages, not the “best” cards;
- unsummarized live-tail intervals and genuine coverage gaps are reported separately.

Conceptual result:

```json
{
  "status": "ok",
  "requested_range": {"start_ts": 0, "end_ts": 0},
  "coverage": {
    "covered_intervals": [],
    "live_raw_intervals": [],
    "gap_intervals": [],
    "complete": true
  },
  "matched_card_count": 12,
  "returned_card_count": 12,
  "cards": [],
  "next_cursor": "",
  "suggested_next_actions": ["open_memory(content)", "open_memory(sources)"]
}
```

`coverage.complete` means the catalog result has accounted for the requested stored history. It does not
claim that every real-world event was recorded while the host was offline.

### 7.2 `retrieve_for_turn`: semantic find

Purpose: find memory from partial content, entity, relationship, or topic clues.

The existing raw-first retrieval contract remains. V1 adds an optional hard lineage selector:

```python
retrieve_for_turn(
    query="早茶 同行 账单",
    within_memory_id="episodic-id",
)
```

When `within_memory_id` is present:

- MemCore resolves the node inside the authorized namespace;
- the descendant closure is compiled into the pre-score hard candidate set;
- raw candidates outside the node cannot participate in dense or BM25 scoring;
- semantic/episodic descendants are included only when the requested source-layer policy allows them;
- missing, cyclic, or out-of-scope lineage returns structured `invalid_filter` or `unavailable`;
- the selector is never silently removed when candidates are empty.

This supports unknown-answer queries without inventing an entity. The model can browse a relevant period,
select one episode, then search its raw evidence using facts it actually knows.

No LLM verifier is added after retrieval. Relevance remains explainable through query, metadata, prefilter
diagnostics, dense/BM25 scores, and lineage scope.

### 7.3 `open_memory`: generic node expansion

Purpose: open any authorized memory node by stable ID.

```python
open_memory(
    memory_id="...",
    view="card",       # card | content | sources
    detail="full",     # full | compact for one raw content node
    projection="conversation",  # conversation | full | tools for raw sources
    cursor="",
)
```

Semantics:

| Node | `card` | `content` | `sources` |
| --- | --- | --- | --- |
| raw | compact raw identity | projected raw entry | empty / unsupported |
| episodic | episodic card | full episode summary | complete raw turns |
| semantic | semantic card | full semantic summary | episodic cards |

Source pagination is by complete child logical units. Opening episodic sources never splits a turn or
action/observation relation. Its default `conversation` projection keeps dialogue and events complete but
projects action/observation, Skill, tool, and material entries as reloadable compact evidence containing
their source ID, kind, correlation/status relation, and retained small anchors. Full payloads remain in
SQLite and are returned only when the model opens one raw source as `content` or explicitly selects
`projection=full/tools`. Opening semantic sources returns episode cards first; the model explicitly opens
the selected episode content or sources next. Native tool dispatch removes the duplicate structured raw
body when the same evidence is already present in rendered `text`; direct trusted Python calls retain both
views for diagnostics.

The current `read_entry(source_id, detail)` becomes a thin raw-only adapter to `open_memory` during one
documented migration window and is removed from model-visible native tools. It must not remain a second
independent implementation.

### 7.4 `read_timeline`: exact raw time and anchor read

Purpose: retrieve exact raw evidence by precise time selector or raw anchor.

Existing exact-time, date alias, and raw-anchor modes remain mutually exclusive and first-class. The
function continues to include summarized raw because it is an evidence reader.

For model-facing dispatch, an omitted page budget no longer creates an unbounded provider payload. The
dispatcher injects a finite, configurable raw-page budget and returns complete logical units plus a cursor.
Trusted host maintenance code may retain an explicit diagnostic-only unlimited path, but it must not be
exposed as the native tool default.

If the full selection exceeds the current page, the result includes:

```json
{
  "status": "partial",
  "reason": "page_boundary",
  "selected_turn_count": 1224,
  "selected_projected_tokens": 670000,
  "returned_turn_count": 18,
  "next_cursor": "...",
  "coverage_complete": false,
  "suggested_next_actions": ["continue_page", "browse_memory_for_overview"]
}
```

This is lossless pagination, not silent truncation. A single oversized turn remains whole and is reported
as `oversized_unit=true`.

`read_timeline` does not silently replace raw evidence with a summary. The model receives enough volume
feedback to choose whether to continue raw pages or switch to catalog navigation.

## 8. Adaptive use without a hidden router

No separate router model is needed. The stable prompt gives the chat model a small decision table:

| User need | First operation |
| --- | --- |
| exact known minute/hour/date slice, quotation, or nearby context | `read_timeline` |
| broad period overview or “what happened during these days” | `browse_memory` |
| vague person/topic/event/fact from an unknown time | `retrieve_for_turn` |
| inspect a returned memory ID | `open_memory` |
| search details inside one selected episode | `retrieve_for_turn(within_memory_id=...)` |

These are guidelines, not brittle routing conditions. Runtime still measures actual selection volume and
returns structured feedback. A broad but small private-chat range may fit in one raw page. A busy one-hour
group range may require pagination. No fixed “more than N hours means summary” rule is introduced.

The prompt must also say:

- do not call every memory tool for every question;
- do not fill unknown answer names into entity filters;
- a card is a navigation hint, not sufficient evidence for unsupported detail;
- if `coverage_complete=false`, either continue, choose another route, or disclose the incomplete scope;
- when a summary already answers the user's requested granularity, answer without needlessly opening raw;
- when exact wording, attribution, chronology, or contradiction matters, inspect raw evidence.

## 9. Prompt composition and provider-cache behavior

Normal request composition remains:

```text
stable system/persona/tool contract
  + stable visible semantic memory
  + recent visible episodic summaries
  + recent unsummarized raw timeline
  + current user/event/tool tail
```

Catalog history is not inserted in full on every request. `browse_memory`, `open_memory`, retrieval, and
timeline results are appended as dynamic tool results at the tail. This preserves a stable prefix and
keeps historical navigation demand-driven.

The catalog instruction and tool schemas belong in the stable prompt. Card contents and node results do
not. Updating a card does not rewrite prior provider projections already frozen in the projection ledger.

MemCore guarantees stable provider-visible projection and auditability; it does not claim a provider will
always cache the prefix.

## 10. Tool-result persistence

The model-visible memory read is itself an environment observation. The host stores that exact result once
beside its action in the same open turn. It remains available to later ordinary turns until the one raw token
compaction lifecycle processes that turn; there is no memory-tool-only turn overlay or fixed round TTL.

The host also stores a compact operation receipt as retention metadata:

```json
{
  "operation": "browse_memory",
  "selector": {"start_ts": 0, "end_ts": 0},
  "returned_memory_ids": [],
  "coverage": {},
  "next_cursor": "",
  "result_hash": "...",
  "status": "ok"
}
```

For `open_memory` and `read_timeline`, the receipt records returned node IDs or raw logical-unit IDs rather
than duplicating the rendered corpus inside the receipt. It is a reload/coverage anchor and never replaces
the complete observation body.

Receipts use the existing operation/action-observation timeline semantics and explicit retrieval policy.
They do not become ordinary conversational semantic candidates merely because a tool was called.

Provider-specific request projections may be frozen in the projection ledger. Future prompt assembly uses
the same stored observation body, preserving native tool-call/result boundaries; it must not render that
body a second time as `data.output`. After compaction, the operation digest may retain the receipt while the
dialogue episode summary records the user/final discussion with separate access and evidence timestamps.

## 11. Failure and status contract

All navigation APIs return `status`, `reason`, selector echo, and safe next actions. Minimum statuses are:

| Status | Meaning |
| --- | --- |
| `ok` | requested representation returned completely |
| `partial` | complete units returned, cursor required for remaining units |
| `empty` | valid selector, no authorized stored match |
| `invalid_filter` | selector or node type is invalid or conflicting |
| `unavailable` | lineage/index/tokenizer dependency cannot safely serve the request |
| `failed` | internal read failed without changing source truth |

Specific reasons include:

```text
memory_id_required
memory_node_not_found_or_out_of_scope
invalid_memory_view
timeline_selector_required
timeline_modes_are_mutually_exclusive
lineage_broken_or_cyclic
cursor_selector_mismatch
page_boundary
token_counter_required
```

Rules:

- `empty` must not be rewritten to a generic model failure;
- `partial` must include a cursor and `coverage_complete=false`;
- invalid cursor or selector mismatch never restarts from page one silently;
- no API returns `ok` with omitted units;
- a catalog fallback title is labeled with `catalog_quality`, not presented as model-generated quality;
- vector-index failure may degrade deterministic catalog/time reads to SQLite, but semantic retrieval must
  report its actual availability.

## 12. Group-chat and host boundaries

The catalog is generic and uses existing actor, target, timestamp, conversation, and relation facts. It
does not know QQ group IDs or nicknames.

Group chats make this feature especially valuable because passive ingestion can produce many raw turns
between two direct conversations. The package behavior remains:

- the host decides which conversations and passive events are admitted;
- MemCore isolates by namespace and conversation authorization;
- participant cards derive from generic actor identity;
- exact raw reads preserve who spoke, when, and the target/reply relationship when the host recorded it;
- summary titles may mention concrete people only when the source evidence supports them.

A separate host defect may exist when passively ingested quoted messages are recorded before quote
resolution. Fixing channel message anchors and reply edges belongs to the host. MemCore may expose generic
`reply_to/source` relations but must not add QQ-specific columns.

## 13. Implementation slices

Each slice must reduce or replace an authority; permanent old/new parallel paths are not acceptable.

Current implementation status:

```text
Slice A  implemented locally and covered by package regression
Slice B  implemented locally
Slice C  implemented locally
Slice D  implemented locally
Slice E  implemented in Akane host (timeline/browse/open/receipt complete)
```

### Slice A: catalog fields and summary generation

- add the schema migration and overlap index;
- extend summary/semantic result contracts with title, hint, and headings;
- compute participant/count fields from frozen source units;
- render deterministic fallback cards for old nodes;
- add store methods for deterministic card browse;
- do not add native tools yet.

Acceptance: new summaries commit catalog fields atomically, old summaries are immediately browsable through
fallback cards, and compaction failure semantics do not change.

### Slice B: `browse_memory` and `open_memory`

- add namespace-safe public facade methods;
- implement overlap selection, coverage, live-tail reporting, and complete-card cursors;
- make store-level generic ID lookup authoritative;
- implement card/content/sources views;
- turn `read_entry` into a thin adapter;
- add native specs and dispatcher support from the same schema source.

Acceptance: a multi-day group range returns compact cards rather than raw corpus, and any returned episode
can be opened to summary content and then exact complete-turn sources.

### Slice C: lineage-scoped semantic retrieval

- add `within_memory_id` to the retrieval request and native schema;
- compile descendant IDs into hard pre-score filters;
- preserve raw-first pools and existing entity-relaxation behavior;
- return explicit lineage diagnostics;
- do not add a verifier.

Acceptance: a vague query inside one selected episode can find unknown names from raw evidence without
guessing them as entity anchors or searching unrelated history.

### Slice D: bounded model-facing raw reads and compact receipts

- remove unlimited-zero semantics from model-facing native dispatch;
- keep complete-unit pagination and cursor validation;
- report total projected volume and navigation suggestions;
- persist one exact model-visible observation plus a compact retention receipt;
- update the prompt playbook and public capability docs.

Acceptance: the historical thousands-entry regression cannot generate an unbounded tool payload, while a
precise one-hour raw query still returns exact ordered evidence and can continue losslessly.

Implementation status: complete. Native dispatch injects and caps against
`MemoryConfig.native_timeline_page_token_budget`; direct Python calls retain an
explicit unlimited path. Timeline responses distinguish selected, returned,
and remaining logical-unit/entry/token volume, return `partial/page_boundary`
with navigation suggestions, preserve oversized turns whole, and use an
explicit `estimated` fallback when no tokenizer is injected. Native dispatch
also returns a deterministic compact receipt; host reintegration in Slice E is
responsible for persisting the exact model-visible body once and attaching the
receipt as retention metadata.

### Slice E: host reintegration

- replace host-authored memory tool schemas with package-generated specs;
- map host capability IDs to the package operations with thin adapters;
- ensure giant results are not mislabeled as producer-bounded unlimited content;
- add the stable model instruction once, outside dynamic per-turn blocks;
- preserve dynamic results as tail events for prefix caching;
- remove superseded host read-entry/result-shaping authority.

Acceptance: the chat model can browse, open, retrieve-within, and read exact raw from the real host, and
every partial/empty/failure status reaches the model without a generic silent fallback.

Implementation status: E1/E2 complete in Akane. The host projects package-owned
`read_timeline`, `browse_memory`, and `open_memory` specs, dispatches them through
`dispatch_native_memory_tool`, preserves package text/navigation metadata across
normal turns under the shared raw token lifecycle, and stores the package receipt
beside the complete result body. The obsolete model-visible `read_memory_entry` path and the host-owned
timeline filtering/rendering authority were removed. Akane's browse adapter only
selects the authorized namespace; MemCore remains the sole authority for cards,
coverage, paging, cursors, and receipts.

## 14. Test matrix

### 14.1 Storage and migration

- schema migration is idempotent;
- existing summaries keep content and lineage unchanged;
- fallback cards are deterministic across restarts;
- new catalog fields round-trip without JSON shape drift;
- participant/count fields match frozen complete source units;
- overlapping range index queries include boundary-crossing summaries and exclude end-boundary-only rows;
- namespace and conversation scope cannot be crossed by ID or cursor.

### 14.2 Summary generation

- one summary-model call returns content and catalog fields;
- missing title uses fallback without discarding a valid summary;
- title/hint/headings cannot override deterministic timestamps, IDs, actors, counts, or lineage;
- mixed-topic group batches retain one exact source set and may expose multiple headings;
- summary failure leaves all sources uncompacted.

### 14.3 Catalog browse

- broad date range returns every overlapping card across pages;
- card pagination never splits or drops a card;
- cursor freezes selector, projection version, scope, and stable ordering key;
- live unsummarized raw is reported distinctly from summary coverage;
- coverage gaps are visible rather than inferred as “nothing happened”;
- catalog browse succeeds from SQLite when vector search is unavailable.

### 14.4 Open and lineage

- generic lookup resolves raw/episodic/semantic within namespace;
- episodic content does not include raw corpus;
- episodic sources return complete ordered turns;
- semantic sources return exact source episode cards;
- broken or cyclic lineage is structured unavailable;
- raw-only compatibility adapter delegates to the generic implementation.

### 14.5 Retrieval

- `within_memory_id` filters candidates before dense/BM25 scoring;
- outside raw candidates never enter scoring;
- empty within-node results do not remove the hard lineage selector;
- unknown answer entity can be found using query/topic/time clues;
- exact entity zero-candidate relaxation remains visible and does not replace the original query;
- no LLM verifier call occurs.

### 14.6 Raw timeline and volume regression

- precise local hour range returns raw in exact time order;
- large range returns a finite complete-turn page, cursor, total counts, and incomplete coverage;
- one oversized turn remains whole and is flagged;
- omitted model-facing budget cannot mean unlimited;
- the real-world multi-day group fixture never creates a provider payload near the full raw corpus size;
- model can switch from returned volume feedback to catalog browse without losing authorization context.

### 14.7 Prompt, caching, and persistence

- stable memory navigation instruction and schemas do not vary per turn;
- dynamic card/content/source results append only after the stable prefix;
- tool observations retain the one model-visible body; receipts contain only IDs/coverage/hash and do not duplicate it;
- frozen historical provider projections are not silently rewritten when card fields are backfilled;
- result hashing is stable and excludes secrets/local paths.

## 15. Acceptance criteria

The design is complete only when all of these statements are true:

1. A model can ask what happened over several busy group-chat days without loading the full raw range.
2. Every catalog card can be opened to its full summary and exact source lineage.
3. A precise time question can still read raw directly without a mandatory summary detour.
4. A vague question can search within a chosen episode using known clues, without inventing the unknown
   answer entity.
5. Any incomplete result says exactly what was returned and how to continue.
6. Old summaries remain accessible after semantic reinforcement without permanently occupying normal prompt
   space.
7. No read path persists a second giant copy of content already stored as raw or summary truth.
8. No router LLM, verifier LLM, regex business rule, silent truncation, or fake success is introduced.
9. The host's stable prompt remains cache-friendly; retrieved content is a dynamic tail result.
10. The package remains provider-neutral and channel-neutral.

## 16. Documentation authority after implementation

After the slices land:

- this document defines catalog and node navigation;
- `memory_read_api_v1.md` defines the exact public signatures, defaults, result fields, and native projection behavior;
- `raw_token_compaction_policy_v1.md` remains the only raw compaction policy;
- `memory_metadata_raw_retrieval_design_v1.md` remains the metadata and raw-first semantic retrieval
  authority, updated only where `within_memory_id` extends it;
- `model_prompt_playbook_v1.md` becomes the model-facing tool-use authority;
- `public_capabilities_v1.md` describes the public package surface;
- historical V1/V2 design records must not be read as active alternative APIs.

Implementation must update those active documents in the same slice that changes public behavior. A future
implementation may change names, but it must preserve the four distinct user needs: browse breadth, semantic
find, node expansion, and exact raw evidence.
