"""Summary continuity uses one preceding chat episode, never operation traces."""

from __future__ import annotations

from memcore import Namespace
from tests.test_compaction_v2 import CompactionV2Base, _complete_simple_turn, _projected_config


class SummaryReferenceTests(CompactionV2Base):
    def test_reference_uses_source_order_and_omits_facts_and_other_scopes(self) -> None:
        mem, store, _ = self.make_mem(config=_projected_config())
        self.addCleanup(store.close)
        self.addCleanup(mem.close)
        for number in range(4):
            _complete_simple_turn(mem, number)
        records = [
            ("old", "older", ["user-0", "final-0"], 9000, "memory.episode_summary"),
            ("previous", "previous diary", ["user-1", "final-1"], 8000, "memory.episode_summary"),
            ("operation", "large operation trace", ["user-1"], 10000, "memory.operation_digest"),
            ("overlap", "overlapping diary", ["user-2"], 11000, "memory.episode_summary"),
            ("future", "future diary", ["user-3"], 12000, "memory.episode_summary"),
            ("broken", "broken diary", ["missing"], 13000, "memory.episode_summary"),
        ]
        for summary_id, diary, sources, timestamp, kind in records:
            store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": summary_id,
                    "diary_summary": diary,
                    "source_ids": sources,
                    "timestamp": timestamp,
                    "kind": kind,
                    "core_facts": ["do not duplicate facts"],
                },
            )
        store.add_summary(
            namespace=Namespace(user_id="user", conversation_id="other"),
            record={
                "summary_id": "foreign",
                "diary_summary": "foreign diary",
                "source_ids": ["user-1"],
            },
        )
        entries = store.get_turn_entries(namespace=mem.namespace, turn_id="turn-2")
        reference = mem._compaction._render_reference_summaries(mem.namespace, entries)
        self.assertEqual(reference, "previous diary")
        # Legacy period metadata is absent above; actual source lineage still works.
        store._conn.execute("UPDATE summaries SET period_end_ts=9999 WHERE summary_id='previous'")
        self.assertEqual(mem._compaction._render_reference_summaries(mem.namespace, entries), "older")

    def test_operation_digests_do_not_consume_chat_visible_limit(self) -> None:
        mem, store, _ = self.make_mem(config=_projected_config())
        self.addCleanup(store.close)
        self.addCleanup(mem.close)
        for index, kind in enumerate(("memory.episode_summary", "memory.operation_digest", "memory.operation_digest")):
            store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": str(index),
                    "kind": kind,
                    "diary_summary": str(index),
                    "timestamp": index,
                },
            )
        visible = store.get_visible_episodic_summaries(namespace=mem.namespace, limit=1)
        self.assertEqual([row["summary_id"] for row in visible], ["0"])
        self.assertEqual(mem._compaction._render_reference_summaries(mem.namespace, []), "")
