# Configuration API V1

This document is the authoritative configuration and dependency-construction
reference for normal MemCore host integrations. Use the public imports from
`memcore`; private modules are not part of the normal integration contract.

## Minimal construction

```python
from memcore import (
    InMemoryVectorIndex,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
)

embedding = build_your_production_embedding_provider()
store = SQLiteMemoryStore("path/to/memcore.sqlite3")
index = InMemoryVectorIndex(embedding=embedding)

mem = MemorySystem(
    llm=memory_llm,
    namespace=Namespace(user_id="user-1", conversation_id="chat-1"),
    timezone="Asia/Shanghai",
    config=MemoryConfig(),
    store=store,
    index=index,
    embedding=embedding,
)
```

`llm` must be an `LLMClient` instance. It is used only for summary, semantic,
and reinforcement compaction tasks; MemCore retrieval does not call a verifier
LLM. `timezone` is required and must be a valid IANA timezone name.

`embedding` is always required. MemCore never silently falls back to hashed
vectors. `store` defaults to `SQLiteMemoryStore(storage_dir or ":memory:")`, and
`index` defaults to `InMemoryVectorIndex(embedding=embedding)`. When a host
injects its own store or index, it owns that dependency's lifecycle.

`MemorySystem.close()` closes only a runtime created by that `MemorySystem`; it
does not close an injected store, index, embedding client, or shared runtime.

## Required `LLMClient` contract

The injected memory LLM must implement the public abstract interface rather
than a loose `call(system, user)` helper:

```python
from memcore import LLMClient, LLMRequest, LLMResult, TaskType

class MyMemoryLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        # request.task_type is SUMMARY, SEMANTIC, or REINFORCEMENT.
        data = call_structured_model(
            system=request.system_prompt,
            user=request.user_prompt,
            json_mode=request.response_format.value == "json",
            timeout=request.timeout_s,
        )
        return LLMResult(ok=True, data=data, attempts=1)
```

`LLMRequest` fields are `task_type`, `system_prompt`, `user_prompt`,
`response_format` (currently `json`), `timeout_s`, `max_retries`,
`temperature`, optional `fallback`, and provider-specific `extra`.
`TaskType` values are `summary`, `semantic`, and `reinforcement`.

The adapter must return `LLMResult`, not a raw string/dict. On a provider or
JSON failure return `LLMResult(ok=False, data=fallback, error=..., attempts=...)`
with a safe fallback supplied by MemCore. Do not raise a bare provider exception
through `call()` and do not return half-parsed JSON: compaction uses the
structured result to choose retry/deferred/fallback states without committing an
empty summary.

`LLMResult.degraded_to_fallback` should be true when the returned `data` is a
fallback rather than a successful model result. Preserve `latency_ms` and
`attempts` when the host can measure them; these fields are operational
telemetry, not memory content.

## `MemoryConfig`

Configuration constraints are checked during construction. Invalid constrained
values raise `ConfigError`; construction does not clamp them into valid values.

| Field | Default | Contract |
| --- | ---: | --- |
| `raw_token_trigger` | `12000` | Positive integer. Trigger threshold for unsummarized raw provider-projection tokens. |
| `raw_token_batch_ratio` | `0.67` | Finite number strictly between `0` and `1`; target ratio of oldest raw tokens selected after triggering. |
| `episodic_visible_max` | `8` | Positive integer; maximum visible episodic summaries. |
| `episodic_compact_trigger_count` | `10` | Positive integer; episodic count that makes semantic compaction due. |
| `episodic_compact_batch_size` | `5` | Positive integer and strictly smaller than `episodic_compact_trigger_count`. |
| `semantic_visible_limit` | `5` | Positive integer; maximum visible semantic memories. |
| `retrieval_result_token_budget` | `0` | Non-negative compatibility setting. Current Retrieval V2 facades take `result_token_budget` per call; native model schemas do not expose that control. `0` means no result-token cut. Do not expect a positive config value alone to cap native results. |
| `native_timeline_page_token_budget` | `12000` | Positive integer; finite maximum for model-facing native timeline pages. A trusted direct `read_timeline(page_token_budget=0)` call remains the explicit unlimited diagnostic path. |
| `projection_profile` | `canonical_user_assistant` | Non-empty profile name, normalized to lowercase. Use the actual provider profile at request/final/compaction boundaries when routing dynamically. |
| `compaction_min_recent_turns` | `1` | Positive integer; minimum recent terminal turns retained outside the selected raw batch. |
| `compaction_schema_version` | `2` | Positive integer; compaction contract generation. |
| `summary_profile` | `timeline_v2` | Non-empty string identifying the summary contract/profile. |
| `semantic_reinforcement_lookback` | `8` | Positive integer; number of recent semantic records considered for reinforcement. |
| `semantic_reinforcement_min_overlap` | `2` | Integer at least `1`; minimum compatible-overlap score before reinforcement. |
| `retrieval_limit` | `6` | Positive integer; default number of retrieval matches for trusted Python calls. It is not model-controlled in native tool schemas. |
| `relaxation_stop_candidate_count` | `12` | Positive integer; candidate count at which structured relaxation stops. |
| `retrieval_min_dense_score` | `0.0` | Non-negative finite number. |
| `retrieval_min_bm25_score` | `0.0` | Non-negative finite number. |
| `retrieval_min_fused_score` | `0.0` | Non-negative finite number. |
| `llm_timeout_s` | `30.0` | Positive finite number; timeout passed to each structured memory-model request. Host shutdown still requires cooperative cancellation. |
| `llm_max_retries` | `2` | Positive integer; retry allowance for structured memory-model calls. Failed calls still must not commit empty memories. |
| `visible_memory_scope` | `conversation` | Stable value `conversation` or `user`. Raw visibility remains conversation-local in both modes. |
| `enable_flavor` | `False` | Enables the optional mood/flavor metadata layer. |
| `enable_importance_decay` | `False` | When true, semantic visibility uses decayed importance rather than pure recency. |
| `importance_half_life_days` | `90.0` | Positive number; importance decay half-life in days. Validated even when decay is disabled. |
| `operation_projection_policy` | `full_until_raw_compaction` | Stable wire value described below. |
| `operation_settlement_min_utf8_bytes` | `256` | Positive integer; observation bodies below this UTF-8 byte threshold stay full. Frozen when the turn begins. |
| `operation_settlement_min_saved_ratio` | `0.5` | Finite number strictly between `0` and `1`; the card must save strictly more than the effective ratio to replace the body. The current classifier caps the effective ratio at `0.99` without changing the stored configuration. Frozen when the turn begins. |

`operation_projection_policy` accepts only:

- `full_until_raw_compaction`: keep full action/observation projection until the
  unified raw compactor processes the turn;
- `compact_after_terminal`: after successful turn completion, freeze a deterministic
  reloadable compact projection while retaining full SQLite truth.

These enum strings are persisted and enter settled hashes. Published values are
stable wire values: do not rename, delete, reuse, or change their meaning. A
future schema may add a new value but must keep historical values readable.

The policy and both settlement thresholds are frozen together by `begin_turn()`.
Changing configuration affects new turns only; existing turns and settled
history retain their recorded values. Byte savings describe the observation
projection, not provider token usage or total cost. See
[`operation_projection_settlement_v1.md`](operation_projection_settlement_v1.md)
for threshold examples, readback, and metrics, and
[`provider_support_matrix_v1.md`](provider_support_matrix_v1.md) for supported
projection profiles and transport validation.

## Namespace and actors

```python
from memcore import Actor, Namespace

namespace = Namespace(
    tenant_id="tenant-a",
    user_id="user-1",
    domain_id="companion",
    conversation_id="private-chat",
)

actor = Actor(stable_id="platform-user-id", display_name="Current nickname")
```

`user_id` is required. `tenant_id / user_id / domain_id` form the hard
isolation key. `conversation_id` selects the visible conversation window.
`Actor` is only a soft speaker-attribution label and must never be used as a
tenant or user isolation boundary. `Actor.stable_id` is required and must be a
stable platform identifier, not a nickname.

## Embedding providers

### Local Hugging Face

Install the optional dependency:

```text
pip install "memcore-kernel[huggingface]"
```

Then either pre-download the model and use the safe offline default:

```python
from memcore import HuggingFaceEmbeddingProvider

embedding = HuggingFaceEmbeddingProvider(
    model_name="BAAI/bge-m3",
    local_files_only=True,
    cache_folder="path/to/model-cache",
)
```

or explicitly allow the model library to download during a controlled setup
step with `local_files_only=False`. Passing a model-name string directly to
`MemorySystem(embedding="BAAI/bge-m3")` uses `local_files_only=True`; it only
works when the model is already available in the local cache.

### OpenAI-compatible HTTP embedding

```python
from memcore import HTTPEmbeddingProvider

embedding = HTTPEmbeddingProvider(
    base_url="https://embedding.example/v1",
    api_key=secret_from_host_config,
    model="bge-m3",
    dimension=1024,
    timeout=30.0,
    name="production-bge-m3",
)
```

`dimension` must match every returned vector. Transport, HTTP, invalid-response,
count, and dimension mismatches raise explicit runtime errors; they never cause
a hidden hashed fallback.

For endpoints that require different query and document task fields, use
`RoleAwareHTTPEmbeddingProvider(common_body=..., query_body=...,
document_body=...)`. `input` and `model` are protected fields and cannot be
overridden in those body mappings. Provider-specific task names remain in the
host adapter rather than MemCore core.

### Startup verification

```python
status = mem.embedding_status()
verification = mem.verify_embedding()

if status["degraded"] or not verification["ok"]:
    fail_startup_or_disable_semantic_retrieval(verification)
```

`embedding_status()` returns `provider`, `version`, `dimension`, and
`degraded`. `degraded=True` currently identifies the hashed test provider.
`verify_embedding(**kwargs)` runs the semantic self-check implemented by the
active provider verification helper; treat `ok=False` as semantic degradation,
not as a successful empty search.

Changing model revision, query/document task behavior, normalization, or output
dimension changes the vector space. Reflect that change in the provider
`name/version/dimension` identity and rebuild or switch the index collection;
never mix incompatible vectors.

### Custom embedding provider

An injected `EmbeddingProvider` must expose a positive `dimension`, implement
`embed_text(text)`, and return one vector per input from `embed_texts(texts)`.
Role-aware providers may override `embed_query/embed_queries` and
`embed_document/embed_documents`; the index uses document methods for upsert and
query methods for retrieval. Returned vector lengths must equal `dimension`.

`TokenCounter` is optional but, when supplied, must implement
`count_text(text) -> int`. Override `quality` with `estimated` for a coarse
counter; MemCore never presents that estimate as provider billing usage.

## Index choices and restart behavior

`InMemoryVectorIndex` provides cosine search, BM25 keyword search, metadata
prefiltering, and RRF without a separate vector service. It is process-local and
must be repopulated after restart:

```python
stats = mem.reindex_all(batch_size=64)
```

SQLite remains the truth source. `reindex_all()` upserts the current hard
namespace into the active index but does not clear stale entries from an
external index.

The repository contains an optional Chroma backend implementation, but it is
not currently exported from the stable top-level public API. Do not depend on a
private import path in third-party integrations until that backend is promoted
to the public contract.

## Shared runtime

For multiple `MemorySystem` instances in one process, inject one shared runtime
to avoid creating one executor set per instance:

```python
from memcore import MemCoreRuntime

runtime = MemCoreRuntime(compaction_workers=2, index_workers=1)
mem_a = MemorySystem(..., runtime=runtime)
mem_b = MemorySystem(..., runtime=runtime)

try:
    run_host()
finally:
    runtime.close(wait=True)
```

Worker counts must be positive. Only the component that created the shared
runtime should close it. Calling `MemorySystem.close()` on an instance with an
injected runtime intentionally does not close that shared runtime.

## Prompt overrides

`PromptOverrides` exposes only additive, validated slots:

- `persona_text`;
- `extra_summary_guidance`;
- `extra_semantic_guidance`;
- `extra_reinforcement_guidance`.

Every value must be a string of at most 4000 characters. Invalid types or
oversized slots raise `PromptError`. These slots cannot remove or replace the
welded JSON, metadata, time-anchor, attribution, or anti-fabrication rules.

## Configuration ownership and exposure boundaries

MemCore is a library, not a service. Nothing here is a runtime control surface,
and a host UI is not a configuration channel. Decide ownership by the question
*who holds the information needed to judge this value* — not by how important the
value feels.

| Surface | Owner | When it is fixed | May a UI change it |
| --- | --- | --- | --- |
| `llm`, `embedding`, `store`, `index`, `runtime`, `TokenCounter`, `material_loader` | Host | Startup, injected into the constructor | No |
| `Namespace(tenant_id/user_id/domain_id/conversation_id)`, `timezone` | Host | Startup | No |
| `PromptOverrides` (`persona_text`, extra guidance slots) | Host | Startup or session setup | No |
| Secrets (API keys, endpoint credentials) | Host | Startup, from host config | Never |
| `MemoryConfig` mechanism thresholds | Host deploy config | Startup; `begin_turn()` freezes the settlement trio per turn | No |
| Read parameters (`page_size`, `page_token_budget`, `time_range`, `projection`, `cross_conversation`, `view`, `cursor`) | Caller | Per call | Yes |

Three constraints follow from this table.

**Constructor-injected dependencies and identity are not knobs.** `llm`,
`embedding`, `store`, `index`, `namespace`, and `timezone` are bound for the
process lifetime. `timezone` anchors every relative-time resolution, and the
namespace is the hard isolation key: exposing either would let a caller move
between memory pools or shift displayed history away from the recorded truth.
The write path stores absolute times; presentation must not alter how they are
interpreted. A UI cannot and should not see which store, index, or embedding
backend is behind these interfaces.

**Mechanism thresholds are host deploy configuration, not end-user settings.**
`episodic_visible_max` and `episodic_compact_trigger_count` form the `[Min, Max]`
band that a prefix stays byte-stable inside; `raw_token_trigger` and
`raw_token_batch_ratio` set the raw watermark. These values are coupled — lowering
`episodic_compact_trigger_count` toward `episodic_visible_max` removes the
append-only headroom the band exists to provide. A UI slider here does not
tune the system, it disables the mechanism and then gets judged on the result.
`operation_projection_policy` is a binary architectural decision that shapes
how much long-lived context a product carries, not a preference. Keep all of
these in host configuration files, applied at startup.

**Read parameters are the only safe interactive surface.** `page_size`,
`page_token_budget`, `time_range`, `projection`, `cross_conversation`, `view`,
and `cursor` change what a single read returns. A wrong value narrows one page;
it cannot corrupt the truth source, break prefix stability, or leak across
namespaces. If a host builds an observation or debug UI, these are the controls
it may expose — and they must stay read-only. Do not let such a UI call
`begin_turn`, `complete_turn`, `append_action`, `append_observation`, or any
maintenance API to manufacture state for display, because a demo built on
synthetic writes is no longer evidence about the running system.

For observability, a UI reads what MemCore already exposes: the results of
`read_timeline`, `browse_memory`, `open_memory`, and `retrieve_for_turn`, plus
`MemorySystem.settlement_metrics()`, `embedding_status()`, and
`verify_embedding()`. Expose nothing beyond these without adding a read-only
snapshot to the host, not to the kernel. Values shown should remain traceable to
SQLite or to a documented metric; a UI must not invent its own indicators, since
headline numbers are only credible when each carries the same measurement basis
the API reports.

Never place credentials, connection strings, local absolute paths, or raw
database handles in any presentation layer, log line, or rendered output.

## Configuration exceptions

- `ConfigError`: invalid `MemoryConfig` value or invariant;
- `NamespaceError`: missing/invalid namespace or actor identity;
- `PromptError`: invalid prompt override slot;
- `TypeError` / `ValueError`: invalid injected dependency type, missing
  embedding, or invalid timezone.

Construct configuration and dependencies during host startup so these failures
surface before accepting chat traffic.
