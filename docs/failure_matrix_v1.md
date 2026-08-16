# Context Failure Matrix V1

| Failure | Model history | Agent loop | User-visible state |
| --- | --- | --- | --- |
| Projection failure | wrapped session returns original full history | continues | structured degraded diagnostic |
| Settlement failure | full history remains available | continues | no fake card |
| Bridge start/exit/timeout | Harness keeps native session history | continues | `bridge_*` reason |
| Stale open turn | MemCore recovers/aborts stale turn before next input | continues | no tool rerun |
| `open_memory` missing source | structured tool result | continues | exact recall unavailable |
| Serializer rejection | provider-neutral/canonical safe trace | continues | reason is explicit |
| Provider timeout/5xx | completed tools stay durable; open turn can abort | host retry policy | no duplicate tool execution |
| Duplicate hook | idempotent source/projection identity | continues | no duplicate message |
| Unknown/orphan host Session item | full native Session history | continues in `degraded` mode; later reads retry full rebuild | no partial import or empty history |

Internal database, cache, run-log paths and secrets never enter diagnostics,
snapshots, or stable model prompts. Executable paths in actual tool evidence
remain untouched when they are model-visible data.
