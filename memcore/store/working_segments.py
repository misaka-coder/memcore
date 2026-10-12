"""Atomic open-turn overlays. Original timeline and request ledger stay intact."""

from __future__ import annotations

import json
import time
from typing import Any

from ..errors import SchemaError
from ..namespace import Namespace
from ..projection import stable_projection_hash


class WorkingSegmentsStoreMixin:
    def get_working_segments(self, *, namespace: Namespace, provider_profile: str) -> list[dict[str, Any]]:
        scope, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM working_compaction_segments WHERE {scope} AND provider_profile = ? ORDER BY end_seq_no",
                [*params, provider_profile],
            ).fetchall()
        return [
            {
                **dict(row),
                "source_ids": json.loads(row["source_ids_json"]),
                "messages": json.loads(row["messages_json"]),
            }
            for row in rows
        ]

    def commit_working_segment(
        self,
        *,
        namespace: Namespace,
        provider_profile: str,
        turn_id: str,
        source_versions: tuple[tuple[str, int], ...],
        projection_hashes: tuple[tuple[str, tuple[str, ...]], ...],
        messages: list[dict[str, Any]],
        policy: str,
    ) -> dict[str, Any]:
        ids = tuple(sid for sid, _ in source_versions)
        if not ids or not messages:
            raise SchemaError("working_segment_sources_required")
        segment_id = stable_projection_hash({"turn": turn_id, "profile": provider_profile, "sources": ids})
        scope, params = self._scope_clause(namespace, with_conversation=True)
        with self._lock, self._immediate_transaction():
            existing = self._conn.execute(
                "SELECT segment_id FROM working_compaction_segments WHERE segment_id = ?", (segment_id,)
            ).fetchone()
            if existing is not None:
                return {"status": "already_committed", "segment_id": segment_id}
            marks = ",".join("?" for _ in ids)
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE {scope} AND source_id IN ({marks}) ORDER BY seq_no", [*params, *ids]
            ).fetchall()
            if tuple((str(row["source_id"]), int(row["row_version"])) for row in rows) != source_versions:
                return {"status": "stale_batch", "reason": "working_source_changed"}
            if any(row["turn_id"] != turn_id or row["is_summarized"] for row in rows):
                return {"status": "stale_batch", "reason": "working_source_retired"}
            current = self._projection_hashes_by_source_locked(
                namespace=namespace,
                provider_profile=provider_profile,
                source_ids=ids,
            )
            if tuple((sid, current.get(sid, ())) for sid in ids) != projection_hashes:
                return {"status": "stale_batch", "reason": "working_projection_changed"}
            prior = self.get_working_segments(namespace=namespace, provider_profile=provider_profile)
            if any(set(ids).intersection(item["source_ids"]) for item in prior):
                return {"status": "stale_batch", "reason": "working_source_overlap"}
            # New rows appended to this open turn do not invalidate A. Any
            # mutation of A itself still fails the source/hash checks above.
            self._conn.execute(
                """INSERT INTO working_compaction_segments(
                    segment_id, tenant_id, user_id, domain_id, conversation_id, turn_id,
                    provider_profile, end_seq_no, policy, source_ids_json, messages_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    segment_id,
                    namespace.tenant_id or "",
                    namespace.user_id,
                    namespace.domain_id or "",
                    namespace.conversation_id or "",
                    turn_id,
                    provider_profile,
                    max(int(row["seq_no"]) for row in rows),
                    policy,
                    json.dumps(ids),
                    json.dumps(messages, ensure_ascii=False),
                    int(time.time()),
                ],
            )
            self._increment_projection_generation_locked(namespace)
        return {"status": "committed", "segment_id": segment_id}
