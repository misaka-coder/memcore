# Captured chat-prefix compaction v1

Available in `memcore-kernel` 0.1.5. This is an optional request mode for the
existing memory lifecycle. It does not create a second memory database, model
tool, or user-facing conversation.

## Configuration

```python
config = MemoryConfig(
    compaction_request_mode="chat_prefix",  # default: independent
    working_token_trigger=200_000,          # SDK default: 0, disabled
    operation_projection_policy="compact_arguments_after_terminal",
)
```

Raw triggering and episodic/semantic windows keep their existing settings.
`working_token_trigger` is a soft threshold for an open turn's effective
projection. It must leave room for stable prompts, tools, retained history,
output and newly arriving results. The host still enforces the complete
provider request budget. A positive per-capture `working_token_limit` lowers
the effective working threshold without changing global configuration.

Turning the mode off prevents new working segments. Existing segments remain
readable. Terminal settlement policies and their frozen thresholds keep their
published meanings; working overlays are a separate, explicitly enabled
maintenance operation using those policies.

## Host capture and adapter

The host captures the finalized ordinary request before provider transport:
actual model and route, prompts, tools, reasoning, cache identity, and messages.
Do not rebuild an independent transcript and call it a cached chat request.

```python
from memcore import ChatContextSnapshot

mem.capture_chat_context(ChatContextSnapshot(
    provider_profile=actual_profile,
    messages=tuple(actual_public_messages),
    message_source_ids=tuple(source_ids_for_each_message),
    transport=frozen_host_request,
    pending_task_ids=tuple(current_pending_ids),
    working_token_limit=available_open_turn_budget,
))
future = mem.compact_due_background(provider_profile=actual_profile)
```

`messages` and source groups must have equal lengths. Unmapped stable context
uses an empty source tuple. Source IDs refer to the effective MemCore
projection, not a second copy of raw history. Record the actual request
projection first when the standard adapter differs from the wire. The host
must treat captured contents as immutable; dataclass freezing alone does not
freeze nested dictionaries.

`transport` is opaque to MemCore and remains in process memory. Freeze the
model/route and provider options with the request; do not resolve them again
after a concurrent settings change. Keep credentials, media, provider
signatures and complete wire snapshots out of persisted memory/audits.
Transient media can remain in this wire copy while eligibility compares its
usual sanitized persisted projection.

When `LLMRequest.chat_prefix` is present, the host adapter:

1. Copies the captured wire request through `prefix_end` (exclusive).
2. Appends `request.chat_prefix.instruction(request.system_prompt)` once.
3. Uses the captured model, route, tools, reasoning and cache identity; disables
   tool execution through supported provider controls.
4. Returns the existing `LLMResult` JSON result/failure contract. Invalid JSON,
   truncation and unexpected tool calls are failures, not executable actions.

The instruction identifies the selected 1-based message range and a short
start cue; earlier messages are background. It does not insert markers into
historical messages or repeat the selected transcript. Hosts with mandatory
chat output schemas should include a short stable exception for their trusted
internal memory task. Internal results bypass user delivery and turn-final
handlers, and must not consume background completion receipts.

## Prefix eligibility and fallback

MemCore requires full selected source coverage, matching effective projection
payloads/profile, no unrelated mapped source inside the selected interval,
and a legal native call/result boundary. OpenAI Chat, Responses function items
and Anthropic tool blocks are recognized; provider signing/serialization stays
the host adapter's responsibility.

For ordinary raw or semantic compaction, an absent/ineligible capture uses the
existing independent request. A failed eligible prefix request preserves the
sources; it does not silently switch routes and rerun on a different model.
Raw `summary_request_mode` identifies the chosen input mode; adapters can
identify semantic prefix calls through `LLMRequest.chat_prefix`.
Semantic compaction reuses a prefix only when the selected episode projections
are present in it; hidden/new episodes keep the independent path and lineage.

The last ordinary request precedes the final response and terminal tool
settlement. For this boundary, hosts may set the optional
`ChatContextSnapshot.terminal_context_refresh(snapshot, selected_projections)`
callback. It returns a new locally prepared `ChatContextSnapshot` or `None`.
MemCore invokes it only for an ineligible ordinary raw prefix while preparing
maintenance input; the host's normal background scheduling keeps this work
off the reply-delivery path. It does not refresh open-turn or semantic captures. The
selected projections are the current effective completed history, including
working notes and the final response. The host keeps the frozen route/options,
uses these provider-shaped payloads without duplicating the transcript, and
preserves unchanged native media/signatures only when still valid. There is
no extra provider call or ledger rewrite. Missing older capture coverage or
changed signed tool arguments can decline the refresh.

The returned snapshot undergoes the same source/hash/native-boundary checks.
A normal request captured concurrently takes precedence; a refreshed older
snapshot never overwrites it. Refresh failure retains the independent path.
That fallback also includes each existing derived working summary once at
its original source boundary, before later dialogue/final output, including
when the mode has been switched back to `independent`. It does not add full
operation bodies or treat derived notes as new timeline/source records.

Large completed latest turns can join the catch-up raw batch when retaining
them would leave the projection over the target. Selection still respects
complete terminal components and native batch boundaries.

## Working segments while a tool turn remains open

When raw compaction cannot consume an open turn, working maintenance can select
its old contiguous, paired, not-yet-folded segment A. It skips the stimulus and
never splits a native multi-call batch. Missing or stale request coverage
returns `waiting_for_request_snapshot`; it does not re-render A independently.

```text
captured request: background -> A
while summarizing: background -> A -> newly appended B
next request:     background -> cards(A) -> working summary(A) -> B
```

The internal working JSON is `{"summary":"..."}`. It preserves verified
changes, evidence, failures, decisions and pending work as of A. Pending IDs
are retained only when they appear in A; IDs introduced solely in B are not
retroactively inserted into A. The summary is one fixed background note after
the replaced segment, not a user instruction, task completion receipt, or
final assistant delivery.

Card policies reuse the normal result/argument settlement planner. With
`full_until_raw_compaction`, A goes directly to a summary-only overlay with no
cards. That overlay does not advertise `has_compact_history` as a card flag.
If cards plus summary would not reduce the segment, return `no_savings` and
keep its original projection.

Working publication is atomic, checks selected row versions, ownership,
unsummarized status, effective hashes and overlap, and increments projection
generation. Tail B appends do not invalidate A; changes/deletion/retirement of
A do. Original timeline and request ledger stay unchanged. `open_memory` and
timeline reads can still obtain the original evidence.

Hosts refresh through `build_context_surface`/`build_context_projection` at
the next request boundary. An in-flight request stays immutable. New actions,
metadata and the final response remain attached to the original open turn;
derived `working.*` projection messages must not be re-recorded as original
action/observation messages. Later raw compaction consumes the completed
turn's effective projection once and retires its working overlays.

The host should coalesce namespace maintenance. If the next request reaches
its physical limit, join the same pending maintenance result with normal
cancellation support, rather than start a competing synchronous summary.
Provider wait must not hold the database lock. Completion of maintenance
does not wake a task that has already delivered its result.

## States, persistence and rollback

`working_status` can be `committed`, `already_committed`, `no_paired_segment`,
`waiting_for_request_snapshot`, `no_savings`, `unavailable`, or `stale_batch`.
Summary failure reports `failed/working_summary_retry_pending` and publishes
nothing. Custom stores without working support report unavailable rather than
pretending success.

SQLite schema 11 stores only scoped source-addressable overlays. Opening a
schema-10 database adds the table/index without rewriting old history. Restart
reconstructs the fixed overlay order; unfinished network calls have no
persisted transport handle and retry from a later capture. No tools replay.
Namespace deletion also removes overlays.

For rollback, use this package with `compaction_request_mode="independent"`
and/or `working_token_trigger=0`. Do not downgrade a schema-11 database to an
older package that cannot read the overlays.

Exact prefix reuse is necessary, not sufficient, for provider cache hits.
Measure provider usage separately for ordinary chat, compaction rebuilds,
raw summaries, working summaries and semantic requests. Do not claim a cache
rate from local hashes or mix internal summary usage into chat charts.
