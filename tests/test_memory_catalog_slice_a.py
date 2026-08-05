"""Slice A: catalog fields, deterministic cards, migration-safe storage, and overlap reads."""

from __future__ import annotations

import unittest

from memcore import HashedEmbeddingProvider, InMemoryVectorIndex, MemoryConfig, Namespace, SQLiteMemoryStore
from memcore.compaction import Compaction
from memcore.index.entry_builder import build_semantic_entry
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType
from memcore.memory_catalog import CATALOG_SCHEMA_VERSION, build_memory_card
from memcore.namespace import Actor


class _CatalogLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type is TaskType.SUMMARY:
            return LLMResult(
                ok=True,
                data={
                    "diary_summary": "聊了扬州早茶、同行人员和账单。",
                    "period_label": "旅行讨论",
                    "event_type": "出行",
                    "importance": 0.8,
                    "key_events": ["确认早茶账单"],
                    "core_facts": ["同行人员在对话中得到确认"],
                    "memory_metadata": {
                        "memory_facets": ["event"],
                        "about_roles": ["user", "third_party"],
                        "entity_anchors": ["扬州"],
                        "topic_terms": ["早茶", "账单"],
                    },
                    "memory_title": "扬州早茶、同行人员和账单",
                    "catalog_hint": "可回答当天早茶、同行人员和账单相关问题。",
                    "topic_headings": ["早茶与账单", "同行人员"],
                },
                attempts=1,
            )
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


class CatalogProjectionTests(unittest.TestCase):
    def test_generated_and_fallback_cards_are_explicitly_distinguished(self) -> None:
        generated = build_memory_card(
            {
                "summary_id": "episode-generated",
                "memory_title": "扬州早茶与同行人员",
                "catalog_hint": "可回答早茶和同行人员问题。",
                "topic_headings": ["早茶", "同行人员", "早茶"],
                "catalog_schema_version": CATALOG_SCHEMA_VERSION,
                "period_start_ts": 100,
                "period_end_ts": 200,
                "participant_refs": [{"actor_id": "u1", "display_name": "旧名"}],
                "source_turn_count": 2,
                "source_entry_count": 4,
                "diary_summary": "完整摘要",
                "source_ids": ["a", "b", "c", "d"],
            }
        )
        fallback = build_memory_card(
            {
                "summary_id": "episode-old",
                "period_label": "夜间学习",
                "diary_summary": "复习了微积分。",
                "core_facts": ["用户复习高数"],
                "timestamp": 300,
                "source_ids": ["old-1"],
            }
        )

        self.assertEqual(generated["catalog_quality"], "generated")
        self.assertEqual(generated["topic_headings"], ["早茶", "同行人员"])
        self.assertEqual(generated["source_turn_count"], 2)
        self.assertEqual(fallback["catalog_quality"], "fallback")
        self.assertEqual(fallback["memory_title"], "夜间学习")
        self.assertEqual(fallback["catalog_hint"], "用户复习高数")
        self.assertEqual(fallback["source_entry_count"], 1)


class CatalogCompactionTests(unittest.TestCase):
    def test_new_summary_commits_catalog_and_source_metrics_atomically(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        index = InMemoryVectorIndex(embedding=embedding)
        conversation = Namespace(user_id="u1", conversation_id="group-1")
        speaker = Namespace(
            user_id="u1",
            conversation_id="group-1",
            actor=Actor(stable_id="member-1", display_name="群友甲"),
        )
        store.add_message(
            namespace=speaker,
            role="user",
            content="我们在扬州吃早茶时，是谁和我一起来的？",
            timestamp=100,
            source_id="catalog-source",
            target_actor_id="bot-1",
            target_actor_display_name="Akane",
        )
        compaction = Compaction(
            store=store,
            index=index,
            llm=_CatalogLLM(),
            config=MemoryConfig(
                raw_token_trigger=1,
                raw_token_batch_ratio=0.67,
                compaction_min_recent_turns=1,
                episodic_compact_trigger_count=99,
            ),
            timezone="Asia/Shanghai",
        )

        result = compaction.run_due(namespace=conversation)
        summaries = store.get_visible_episodic_summaries(namespace=conversation, limit=10)
        store.close()

        self.assertEqual(result["status"], "compacted")
        self.assertEqual(len(summaries), 1)
        saved = summaries[0]
        self.assertEqual(saved["memory_title"], "扬州早茶、同行人员和账单")
        self.assertEqual(saved["catalog_schema_version"], CATALOG_SCHEMA_VERSION)
        self.assertEqual(saved["topic_headings"], ["早茶与账单", "同行人员"])
        self.assertEqual(saved["source_turn_count"], 1)
        self.assertEqual(saved["source_entry_count"], 1)
        self.assertEqual(
            saved["participant_refs"],
            [
                {"actor_id": "member-1", "display_name": "群友甲"},
                {"actor_id": "bot-1", "display_name": "Akane"},
            ],
        )


class CatalogStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.c1 = Namespace(user_id="u1", tenant_id="t", domain_id="d", conversation_id="c1")
        self.c2 = Namespace(user_id="u1", tenant_id="t", domain_id="d", conversation_id="c2")

    def tearDown(self) -> None:
        self.store.close()

    def _add(self, namespace: Namespace, summary_id: str, **record: object) -> None:
        self.store.add_summary(
            namespace=namespace,
            record={
                "summary_id": summary_id,
                "kind": "memory.episode_summary",
                "diary_summary": summary_id,
                **record,
            },
        )

    def test_time_overlap_is_complete_ordered_and_includes_semanticized_episodes(self) -> None:
        self._add(self.c1, "crosses-start", period_start_ts=100, period_end_ts=200, timestamp=200)
        self._add(self.c1, "inside", period_start_ts=220, period_end_ts=260, timestamp=260)
        self._add(self.c1, "point", timestamp=300, is_semanticized=1, semantic_id="semantic-1")
        self._add(self.c1, "end-boundary", timestamp=400)
        self._add(self.c1, "explicit", timestamp=250, retrieval_visibility="explicit")
        self.store.add_summary(
            namespace=self.c1,
            record={
                "summary_id": "operation",
                "kind": "memory.operation_digest",
                "timestamp": 250,
                "diary_summary": "工具过程",
            },
        )
        self._add(self.c2, "other-conversation", timestamp=280)

        current = self.store.get_episodic_summaries_by_time_range(
            namespace=self.c1,
            start_ts=200,
            end_ts=400,
        )
        across = self.store.get_episodic_summaries_by_time_range(
            namespace=self.c1,
            start_ts=200,
            end_ts=400,
            cross_conversation=True,
        )

        self.assertEqual([item["summary_id"] for item in current], ["crosses-start", "inside", "point"])
        self.assertEqual(
            [item["summary_id"] for item in across],
            ["crosses-start", "inside", "other-conversation", "point"],
        )
        self.assertEqual(current[-1]["is_semanticized"], 1)

    def test_invalid_catalog_range_is_rejected_instead_of_broadened(self) -> None:
        with self.assertRaisesRegex(ValueError, "catalog_time_range_invalid"):
            self.store.get_episodic_summaries_by_time_range(
                namespace=self.c1,
                start_ts=300,
                end_ts=300,
            )

    def test_semantic_catalog_fields_round_trip_and_enrich_index_text(self) -> None:
        saved = self.store.add_semantic_summary(
            namespace=self.c1,
            record={
                "semantic_id": "semantic-catalog",
                "timestamp": 500,
                "semantic_summary": "长期讨论扬州出行。",
                "memory_title": "扬州出行主线",
                "catalog_hint": "可回答长期的扬州出行安排。",
                "topic_headings": ["行程", "同行人员"],
                "catalog_schema_version": CATALOG_SCHEMA_VERSION,
                "source_summary_ids": ["episode-a", "episode-b"],
            },
        )
        card = build_memory_card(saved)
        index_entry = build_semantic_entry(saved)

        self.assertEqual(saved["topic_headings"], ["行程", "同行人员"])
        self.assertEqual(card["node_type"], "semantic")
        self.assertEqual(card["source_entry_count"], 2)
        self.assertEqual(card["catalog_quality"], "generated")
        self.assertIn("扬州出行主线", index_entry["text"])


if __name__ == "__main__":
    unittest.main()
