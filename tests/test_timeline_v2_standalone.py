"""Unified Timeline V2 standalone entries and staged annotation tests."""

from __future__ import annotations

import unittest

from memcore import (
    AnnotationStatus,
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    RetrievalPolicy,
    RetrievalVisibility,
    SchemaError,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
)


class NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


class TimelineV2StandaloneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        embedding = HashedEmbeddingProvider()
        self.mem = MemorySystem(
            llm=NoopLLM(),
            namespace=Namespace(user_id="user", conversation_id="c1"),
            timezone="Asia/Shanghai",
            config=MemoryConfig(),
            store=self.store,
            index=InMemoryVectorIndex(embedding=embedding),
            embedding=embedding,
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()

    def test_standalone_entry_is_typed_idempotent_and_has_no_fake_turn(self) -> None:
        entry = TimelineEntryInput(
            source_id="standalone-event",
            kind="event.qq.notice",
            origin=EntryOrigin.ENVIRONMENT,
            turn_role=None,
            semantic_text="群公告更新",
            payload={"text": "群公告更新"},
            annotation_status=AnnotationStatus.ACCEPTED_HOST,
            retrieval_policy=RetrievalPolicy.ALWAYS,
            retrieval_visibility=RetrievalVisibility.DEFAULT,
            timestamp=1000,
        )

        first = self.mem.append_standalone_entry(entry)
        duplicate = self.mem.append_standalone_entry(entry)
        result = self.mem.retrieve_structured("群公告更新", topic_terms=["群公告更新"])

        self.assertEqual(first.source_id, "standalone-event")
        self.assertEqual(duplicate.source_id, first.source_id)
        self.assertEqual(first.turn_id, "")
        self.assertIsNone(first.turn_role)
        self.assertEqual(first.relation_status, "standalone")
        self.assertEqual(result.status, "found")
        self.assertEqual(result.matches[0].source_ids, ("standalone-event",))

        with self.assertRaises(SchemaError):
            self.mem.append_standalone_entry(
                TimelineEntryInput(
                    source_id="standalone-event",
                    kind="event.qq.notice",
                    origin=EntryOrigin.ENVIRONMENT,
                    turn_role=None,
                    semantic_text="冲突内容",
                    payload={"text": "冲突内容"},
                    timestamp=1000,
                )
            )

    def test_staged_metadata_stays_explicit_until_atomic_completion(self) -> None:
        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="staged-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="我喜欢滑雪",
                    timestamp=2000,
                )
            ],
            turn_id="staged-turn",
            opened_at=2000,
        )
        metadata = {
            "entity_anchors": ["滑雪"],
            "about_roles": ["user"],
            "memory_facets": ["preference"],
            "retrieval_priority": "high",
        }

        staged = self.mem.stage_turn_metadata("staged-user", metadata)
        before = self.store.get_entry(namespace=self.mem.namespace, source_id="staged-user")
        hidden = self.mem.retrieve_structured("滑雪", entity_anchors=["滑雪"])

        self.assertTrue(staged["ok"])
        self.assertEqual(staged["status"], "staged")
        self.assertEqual(before.annotation_status, AnnotationStatus.UNANNOTATED)
        self.assertEqual(before.retrieval_visibility, RetrievalVisibility.EXPLICIT)
        self.assertEqual(before.index_status, "pending")
        self.assertEqual(hidden.status, "empty")

        completed = self.mem.complete_turn(
            turn_id="staged-turn",
            semantic_text="记住了，你喜欢滑雪。",
            provider_output_raw='{"speech":"记住了，你喜欢滑雪。"}',
            memory_annotation=metadata,
            annotation_status="accepted",
            source_id="staged-final",
            timestamp=2001,
        )
        visible = self.mem.retrieve_structured("滑雪", entity_anchors=["滑雪"])

        self.assertTrue(completed.completed)
        self.assertEqual(visible.status, "found")
        self.assertEqual(visible.matches[0].source_ids, ("staged-user", "staged-final"))

    def test_staging_rejects_non_stimulus_or_closed_turn_without_fake_success(self) -> None:
        standalone = self.mem.append_standalone_entry(
            TimelineEntryInput(
                source_id="standalone-stage-target",
                kind="event.qq.notice",
                origin=EntryOrigin.ENVIRONMENT,
                turn_role=None,
                semantic_text="standalone",
                timestamp=3000,
            )
        )
        invalid = self.mem.stage_turn_metadata(standalone.source_id, {"topic_terms": ["x"]})
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["status"], "invalid")
        self.assertEqual(invalid["reason"], "staged_annotation_requires_turn_stimulus")

        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="closed-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="closed",
                    timestamp=3100,
                )
            ],
            turn_id="closed-turn",
            opened_at=3100,
        )
        self.mem.complete_turn(
            turn_id="closed-turn",
            semantic_text="done",
            provider_output_raw="done",
            annotation_status="missing",
            source_id="closed-final",
            timestamp=3101,
        )
        closed = self.mem.stage_turn_metadata("closed-user", {"topic_terms": ["x"]})
        self.assertFalse(closed["ok"])
        self.assertEqual(closed["status"], "invalid")
        self.assertEqual(closed["reason"], "staged_annotation_requires_open_turn")

    def test_staging_preserves_group_actor_ownership(self) -> None:
        actor = Actor(stable_id="qq-1", display_name="张三")
        self.mem.begin_turn(
            stimuli=[
                TimelineEntryInput(
                    source_id="actor-user",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="张三喜欢滑雪",
                    actor=actor,
                    timestamp=3200,
                )
            ],
            turn_id="actor-turn",
            opened_at=3200,
        )

        wrong = self.mem.stage_turn_metadata("actor-user", {"entity_anchors": ["滑雪"]})
        correct = self.mem.stage_turn_metadata("actor-user", {"entity_anchors": ["滑雪"]}, actor=actor)

        self.assertFalse(wrong["ok"])
        self.assertEqual(wrong["status"], "forbidden")
        self.assertTrue(correct["ok"])
        self.assertEqual(correct["status"], "staged")


if __name__ == "__main__":
    unittest.main()
