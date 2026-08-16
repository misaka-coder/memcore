# Provider Support Matrix V1

| Profile | Provider wire | Tool call | Tool result | Event kinds | Status |
| --- | --- | --- | --- | --- | --- |
| `openai_chat` | Chat Completions | `assistant.tool_calls` | `role=tool` | preserved | authoritative |
| `openai_responses` | Responses input | `function_call` | `function_call_output` | preserved | authoritative |
| `anthropic_messages` | Messages | `tool_use` | `tool_result` | preserved | authoritative |
| `deepseek_chat` | OpenAI-compatible chat | `assistant.tool_calls` | `role=tool` | preserved | authoritative |
| `canonical_user_assistant` | provider-neutral fallback | text trace | text trace | preserved | storage/replay |

Provider credentials, system/persona content, actual tool execution, and UI
remain host-owned. A host must not copy these serializers.
`validate_context_adapter(adapter)` proves only normalized/provider-shaped
conformance. A concrete integration is authoritative only after its final
transport capture also passes
`validate_provider_wire_capture(surface, captured_context_messages)`.
