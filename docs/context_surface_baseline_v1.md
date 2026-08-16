# Context Surface Baseline V1

These fixtures are fixed-clock expected ContextSurface examples, not database
snapshots and not evidence of a real provider request. They define the
conversation segment a conforming host must send after adding its own system
prompt. Actual transport authority requires a host-side request capture checked
with `validate_provider_wire_capture(...)`. IDs, timestamps, tool arguments,
and result bodies are deterministic; credentials, cache paths, database paths,
and runtime logs are absent.

Fixtures:

- `tests/fixtures/context_surface_v1/akane_openai.json`
- `tests/fixtures/context_surface_v1/akane_anthropic.json`
- `tests/fixtures/context_surface_v1/harness_deepseek.json`

The MemCore-owned sequence is `history_messages`, `current_message`, then
`active_turn_messages`. The current user appears once. Ordinary messages use
`[YYYY-MM-DD HH:MM] User/Assistant: ...`; structured `event.*` entries keep
their event kind. Frozen native tool blocks keep their provider shape, call ID,
argument JSON, and result bytes.

The old Harness baseline is intentionally rejected: `[message_time:]` was a
private format, the current user was emitted twice, and structured events were
flattened into `context.inject`. These fixtures are the replacement golden
shape baseline, not transport proof.
