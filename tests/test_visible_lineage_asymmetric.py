"""Asymmetric visible-lineage exclusion acceptance tests.

A visible raw already showed its exact words, so its derived summary/semantic
is excluded from retrieval to avoid returning a duplicate copy. A visible
summary is a compression, so its raw sources stay retrievable for detail
questions; only the summary itself and the aggregations above it are excluded.

Regression under test: the old full-closure exclusion dropped every raw
descendant of a visible summary, so "already summarized away" raw evidence
could never be semantically re-found (the production 发烧 zero-hit case).
"""

from __future__ import annotations

import unittest

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
)
from memcore.index.entry_builder import build_semantic_entry, build_summary_entry
from memcore.index.metadata_filters import INDEX_SCHEMA_KEY, INDEX_SCHEMA_VERSION
from memcore.rendering import render_semantic_snippet, render_summary_snippet


class NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


def _memory(
    *, user: str = "user", conversation: str = "c1"
) -> tuple[MemorySystem, SQLiteMemoryStore, InMemoryVectorIndex]:
    store = SQLiteMemoryStore(":memory:")
    embedding = HashedEmbeddingProvider()
    index = InMemoryVectorIndex(embedding=embedding)
    mem = MemorySystem(
        llm=NoopLLM(),
        namespace=Namespace(user_id=user, conversation_id=conversation),
        timezone="Asia/Shanghai",
        config=MemoryConfig(),
        store=store,
        index=index,
        embedding=embedding,
    )
    return mem, store, index


def _index_summary(mem: MemorySystem, store: SQLiteMemoryStore, summary: dict[str, object]) -> None:
    mem.index.upsert([build_summary_entry(summary)])
    store.set_index_state(
        str(summary["summary_id"]),
        "indexed",
        index_schema_version=INDEX_SCHEMA_VERSION,
        index_key=INDEX_SCHEMA_KEY,
    )


def _index_semantic(mem: MemorySystem, store: SQLiteMemoryStore, semantic: dict[str, object]) -> None:
    mem.index.upsert([build_semantic_entry(semantic)])
    store.set_index_state(
        str(semantic["semantic_id"]),
        "indexed",
        index_schema_version=INDEX_SCHEMA_VERSION,
        index_key=INDEX_SCHEMA_KEY,
    )


class AsymmetricVisibleLineageTests(unittest.TestCase):
    def test_visible_summary_raw_descendant_is_re_retrievable(self) -> None:
        """Raw evidence summarized out of the visible raw window stays reachable."""
        mem, store, _index = _memory()
        try:
            raw = mem.record_user_turn(
                "8月5日晚上开始发烧，反复到38.9度",
                timestamp=1000,
                source_id="fever-raw",
                memory_metadata={"entity_anchors": ["发烧"], "memory_facets": ["event"]},
            )
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "fever-summary",
                    "timestamp": 2000,
                    "diary_summary": "8月2-6日发烧阶段摘要",
                    "source_ids": [raw["source_id"]],
                    "memory_metadata": {"entity_anchors": ["发烧"], "memory_facets": ["event"]},
                },
            )
            store.mark_messages_summarized([raw["source_id"]], summary["summary_id"])
            _index_summary(mem, store, summary)
            current = mem.record_user_turn("我记得发烧那几天很难受", timestamp=2100, source_id="cur")

            visible = store.get_unsummarized_messages(namespace=mem.namespace)
            self.assertNotIn(raw["source_id"], [row.get("source_id") for row in visible])

            result = mem.retrieve_for_turn_structured(
                current=current,
                query="发烧那几天发生了什么",
                entity_anchors=["发烧"],
            )

            self.assertEqual(result.status, "found")
            self.assertIn("fever-raw", [match.source_id for match in result.matches])
            self.assertNotIn("fever-summary", [match.source_id for match in result.matches])
        finally:
            mem.close()
            store.close()

    def test_visible_semantic_episodic_and_raw_descendants_are_re_retrievable(self) -> None:
        """Raw and episodic evidence under a visible semantic stay reachable."""
        mem, store, _index = _memory()
        try:
            raw = mem.record_user_turn(
                "半岛铁盒是我一直想听的歌",
                timestamp=1000,
                source_id="music-raw",
                memory_metadata={"entity_anchors": ["半岛铁盒"], "memory_facets": ["preference"]},
            )
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "music-summary",
                    "timestamp": 2000,
                    "diary_summary": "用户想听半岛铁盒",
                    "source_ids": [raw["source_id"]],
                    "memory_metadata": {"entity_anchors": ["半岛铁盒"], "memory_facets": ["preference"]},
                },
            )
            store.mark_messages_summarized([raw["source_id"]], summary["summary_id"])
            store.mark_summaries_semanticized([summary["summary_id"]], "music-semantic")
            semantic = store.add_semantic_summary(
                namespace=mem.namespace,
                record={
                    "semantic_id": "music-semantic",
                    "timestamp": 3000,
                    "last_reinforced_ts": 3000,
                    "semantic_summary": "稳定偏好：半岛铁盒",
                    "source_summary_ids": [summary["summary_id"]],
                    "memory_metadata": {"entity_anchors": ["半岛铁盒"], "memory_facets": ["preference"]},
                },
            )
            _index_summary(mem, store, summary)
            _index_semantic(mem, store, semantic)
            current = mem.record_user_turn("我最近想听的歌是哪首", timestamp=3100, source_id="cur")

            result = mem.retrieve_for_turn_structured(
                current=current,
                query="半岛铁盒",
                entity_anchors=["半岛铁盒"],
            )

            self.assertEqual(result.status, "found")
            source_ids = [match.source_id for match in result.matches]
            self.assertIn("music-raw", source_ids)
            self.assertNotIn("music-semantic", source_ids)
        finally:
            mem.close()
            store.close()

    def test_retrieve_within_visible_summary_id_still_reaches_raw_evidence(self) -> None:
        """within_memory_id on a visible summary must not empty the raw pool."""
        mem, store, _index = _memory()
        try:
            raw = mem.record_user_turn(
                "发烧到38.9度那次，我一晚上没睡好",
                timestamp=1000,
                source_id="fever-raw",
                memory_metadata={"entity_anchors": ["发烧"]},
            )
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "fever-summary",
                    "timestamp": 2000,
                    "diary_summary": "发烧阶段摘要",
                    "source_ids": [raw["source_id"]],
                    "memory_metadata": {"entity_anchors": ["发烧"]},
                },
            )
            store.mark_messages_summarized([raw["source_id"]], summary["summary_id"])
            _index_summary(mem, store, summary)
            current = mem.record_user_turn("你记不记得发烧那晚的细节", timestamp=2100, source_id="cur")

            result = mem.retrieve_for_turn_structured(
                current=current,
                query="发烧",
                entity_anchors=["发烧"],
                within_memory_id="fever-summary",
                source_layers=["raw"],
            )

            self.assertEqual(result.status, "found")
            self.assertIn("fever-raw", [match.source_id for match in result.matches])
        finally:
            mem.close()
            store.close()

    def test_visible_raw_still_excludes_its_derived_summary(self) -> None:
        """A visible raw never has its derived summary re-returned around it."""
        mem, store, _index = _memory()
        try:
            current = mem.record_user_turn("当前可见滑雪消息", timestamp=5000, source_id="visible-raw")
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "hidden-derived",
                    "timestamp": 5001,
                    "diary_summary": "不应重复返回的滑雪摘要",
                    "source_ids": ["visible-raw"],
                    "memory_metadata": {"entity_anchors": ["滑雪"]},
                },
            )
            _index_summary(mem, store, summary)

            result = mem.retrieve_for_turn_structured(
                current=current,
                query="滑雪",
                entity_anchors=["滑雪"],
            )

            source_ids = [match.source_id for match in result.matches]
            self.assertNotIn("visible-raw", source_ids)
            self.assertNotIn("hidden-derived", source_ids)
        finally:
            mem.close()
            store.close()

    def test_current_message_is_always_excluded(self) -> None:
        mem, store, _index = _memory()
        try:
            current = mem.record_user_turn("我之前说过我喜欢喝可乐吗", timestamp=6000, source_id="cur")
            result = mem.retrieve_for_turn_structured(
                current=current,
                query="可乐",
                entity_anchors=["可乐"],
            )
            self.assertNotIn("cur", [match.source_id for match in result.matches])
        finally:
            mem.close()
            store.close()

    def test_namespace_isolation_survives_asymmetric_lineage(self) -> None:
        """The asymmetric rule never widens the tenant/user/domain boundary."""
        mem, store, _index = _memory()
        other = Namespace(user_id="other-user", conversation_id="c1")
        try:
            other_store = store
            other_mem = MemorySystem(
                llm=NoopLLM(),
                namespace=other,
                timezone="Asia/Shanghai",
                config=MemoryConfig(),
                store=other_store,
                index=mem.index,
                embedding=mem.embedding,
            )
            try:
                foreign_raw = other_mem.record_user_turn(
                    "其他用户私密发烧记录",
                    timestamp=7000,
                    source_id="foreign-raw",
                    memory_metadata={"entity_anchors": ["发烧"]},
                )
            finally:
                other_mem.close()

            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "fever-summary",
                    "timestamp": 8000,
                    "diary_summary": "发烧阶段摘要",
                    "source_ids": [foreign_raw["source_id"]],
                    "memory_metadata": {"entity_anchors": ["发烧"]},
                },
            )
            store.mark_messages_summarized([foreign_raw["source_id"]], summary["summary_id"])
            _index_summary(mem, store, summary)
            current = mem.record_user_turn("我发烧过吗", timestamp=8100, source_id="cur")

            result = mem.retrieve_for_turn_structured(
                current=current,
                query="发烧",
                entity_anchors=["发烧"],
                cross_conversation=True,
            )

            self.assertNotIn("foreign-raw", [match.source_id for match in result.matches])
        finally:
            mem.close()
            store.close()

    def test_visible_summary_rendering_carries_openable_memory_id(self) -> None:
        """Prompt-visible cards expose a compact memory_id for explicit drill-down."""
        mem, store, _index = _memory()
        try:
            raw = mem.record_user_turn(
                "发烧记录",
                timestamp=1000,
                source_id="fever-raw",
                memory_metadata={"entity_anchors": ["发烧"]},
            )
            summary = store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "fever-summary",
                    "timestamp": 2000,
                    "diary_summary": "发烧阶段摘要",
                    "source_ids": [raw["source_id"]],
                    "memory_metadata": {"entity_anchors": ["发烧"]},
                },
            )
            store.mark_messages_summarized([raw["source_id"]], summary["summary_id"])
            semantic = store.add_semantic_summary(
                namespace=mem.namespace,
                record={
                    "semantic_id": "fever-semantic",
                    "timestamp": 3000,
                    "last_reinforced_ts": 3000,
                    "semantic_summary": "发烧长期记忆",
                    "source_summary_ids": [summary["summary_id"]],
                    "memory_metadata": {"entity_anchors": ["发烧"]},
                },
            )

            summary_text = render_summary_snippet(summary, tz="Asia/Shanghai")
            self.assertIn('open_memory(memory_id="fever-summary"', summary_text)
            self.assertIn('view="sources"', summary_text)

            semantic_text = render_semantic_snippet(semantic, tz="Asia/Shanghai")
            self.assertIn('open_memory(memory_id="fever-semantic"', semantic_text)
            self.assertIn('view="sources"', semantic_text)

            opened = mem.open_memory(memory_id="fever-summary", view="sources")
            self.assertEqual(opened["status"], "ok")
            self.assertIn("fever-raw", str(opened.get("result") or {}))
        finally:
            mem.close()
            store.close()


class LineageClosureDirectionTests(unittest.TestCase):
    def test_resolve_keeps_descendant_ids_separate_from_ancestor_ids(self) -> None:
        """The asymmetric exclusion relies on the closure splitting both directions."""
        mem, store, _index = _memory()
        try:
            raw = mem.record_user_turn("我欠阿姆的一笔钱还没还", timestamp=1000, source_id="debt-raw")
            store.add_summary(
                namespace=mem.namespace,
                record={
                    "summary_id": "debt-summary",
                    "timestamp": 2000,
                    "diary_summary": "欠款摘要",
                    "source_ids": [raw["source_id"]],
                    "memory_metadata": {"entity_anchors": ["阿姆"]},
                },
            )
            closure = store.resolve_lineage_source_ids(
                namespace=mem.namespace,
                source_ids=("debt-summary",),
                cross_conversation=True,
            )
            self.assertEqual(closure.status, "resolved")
            self.assertIn("debt-raw", closure.descendant_ids)
            self.assertNotIn("debt-summary", closure.descendant_ids)
            self.assertNotIn("debt-raw", closure.ancestor_ids)
        finally:
            mem.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
