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

## `MemoryConfig`

All fields are validated during construction. Invalid values raise
`ConfigError`; MemCore does not silently clamp them.

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
| `llm_max_retries` | `2` | Positive integer; retry allowance for structured memory-model calls. Failed calls still must not commit empty memories. |
| `visible_memory_scope` | `conversation` | Stable value `conversation` or `user`. Raw visibility remains conversation-local in both modes. |
| `enable_flavor` | `False` | Enables the optional mood/flavor metadata layer. |
| `enable_importance_decay` | `False` | When true, semantic visibility uses decayed importance rather than pure recency. |
| `importance_half_life_days` | `90.0` | Positive number; importance decay half-life in days. Validated even when decay is disabled. |
| `operation_projection_policy` | `full_until_raw_compaction` | Stable wire value described below. |

`operation_projection_policy` accepts only:

- `full_until_raw_compaction`: keep full action/observation projection until the
  unified raw compactor processes the turn;
- `compact_after_terminal`: after a successful final, freeze a deterministic
  reloadable compact projection while retaining full SQLite truth.

These enum strings are persisted and enter settled hashes. Published values are
stable wire values: do not rename, delete, reuse, or change their meaning. A
future schema may add a new value but must keep historical values readable.

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
pip install "memcore[huggingface]"
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

## Configuration exceptions

- `ConfigError`: invalid `MemoryConfig` value or invariant;
- `NamespaceError`: missing/invalid namespace or actor identity;
- `PromptError`: invalid prompt override slot;
- `TypeError` / `ValueError`: invalid injected dependency type, missing
  embedding, or invalid timezone.

Construct configuration and dependencies during host startup so these failures
surface before accepting chat traffic.
