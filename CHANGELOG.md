# Changelog

All notable public changes to MemCore are documented here.

## 0.1.0 - 2026-09-25

Initial public release.

### Highlights

- Unified causal timeline for messages, events, actions, observations, and final
  responses.
- Terminal Settlement for deterministic post-turn compaction of large tool
  observations into reloadable cards while preserving raw evidence.
- Raw → episodic → semantic memory lifecycle with complete-turn compaction
  boundaries.
- Hybrid vector/BM25 retrieval plus agentic navigation through
  `retrieve_for_turn`, `browse_memory`, `open_memory`, and
  `read_timeline`.
- Provider projection ledger and cache-stable context construction for OpenAI
  Chat/Responses, Anthropic Messages, DeepSeek Chat, and canonical projections.
- SQLite single source of truth with WAL for file-backed stores and outbox-style
  index repair.
- Apache-2.0 licensing.
