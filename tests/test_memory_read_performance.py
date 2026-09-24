"""Read optimizations must preserve evidence, ranking, scope and pagination."""

from __future__ import annotations

from collections import Counter
import unittest
from unittest.mock import patch

from memcore import HashedEmbeddingProvider, InMemoryVectorIndex, MemorySystem, Namespace, SQLiteMemoryStore
from memcore.errors import SchemaError
from memcore.index.memory_index import _keyword_doc_text, _match_where
from memcore.rendering import render_timeline
from memcore.store.base import MemoryStore
from memcore.text_utils import tokenize
from tests.test_catalog_keywords import _SummaryLLM


class ReadPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.ns = Namespace(user_id="u", conversation_id="c")
        self.mem = MemorySystem(
            namespace=self.ns,
            timezone="Asia/Shanghai",
            llm=_SummaryLLM(),
            store=self.store,
            embedding=self.embedding,
            index=InMemoryVectorIndex(embedding=self.embedding),
        )
        self.addCleanup(self.store.close)
        self.addCleanup(self.mem.close)

    def raw(self, sid, *, ns=None, turn_id="", timestamp=100):
        return self.store.add_message(
            namespace=ns or self.ns,
            source_id=sid,
            role="user",
            content="合成证据文本 " + sid,
            timestamp=timestamp,
            turn_id=turn_id,
        )

    def episode(self, sid, sources=()):
        return self.store.add_summary(
            namespace=self.ns,
            record={"summary_id": sid, "timestamp": 100, "diary_summary": sid, "source_ids": list(sources)},
        )

    def traced(self, call):
        statements = []
        self.store._conn.set_trace_callback(statements.append)
        try:
            result = call()
        finally:
            self.store._conn.set_trace_callback(None)
        return result, [s for s in statements if s.lstrip().upper().startswith("SELECT")]

    def test_batch_reads_bound_queries_keep_scope_and_match_single_reads(self):
        ids = [f"r{i}" for i in range(405)]
        for sid in ids:
            self.raw(sid)
        self.episode("e", ids[:2])
        other = Namespace(user_id="other", conversation_id="c")
        self.raw("foreign", ns=other)
        requested = (*ids, "e", "foreign", "missing", ids[0])
        records, queries = self.traced(
            lambda: self.store.get_retrieval_records(namespace=self.ns, source_ids=requested)
        )
        self.assertEqual(len(queries), 6)
        self.assertEqual(set(records), {*ids, "e"})
        fallback = MemoryStore.get_retrieval_records(self.store, namespace=self.ns, source_ids=requested)
        self.assertEqual(records, fallback)
        self.store.add_semantic_summary(
            namespace=self.ns, record={"semantic_id": "e", "timestamp": 100, "semantic_summary": "collision"}
        )
        with self.assertRaisesRegex(SchemaError, "ambiguous_memory_id"):
            self.store.get_retrieval_records(namespace=self.ns, source_ids=("e",))

    def test_batch_open_preserves_order_duplicates_errors_and_single_content(self):
        self.raw("raw")
        self.episode("episode", ["raw"])
        requested = ["episode", "raw", "episode", "missing"]
        expected = [self.mem.open_memory(memory_id=sid, view="content") for sid in requested]
        result, queries = self.traced(lambda: self.mem.open_memory(memory_ids=requested, view="content"))
        self.assertEqual(len(queries), 3)
        self.assertEqual(result["result"]["items"], expected)
        self.assertEqual(result["status"], "partial")
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={"semantic_id": "episode", "timestamp": 100, "semantic_summary": "collision"},
        )
        result = self.mem.open_memory(memory_ids=["raw", "episode"], view="content")
        self.assertEqual([item["status"] for item in result["result"]["items"]], ["ok", "unavailable"])
        invalid = self.mem.open_memory(memory_ids=["raw"], detail="invalid")
        self.assertEqual(invalid["result"]["items"][0]["reason"], "invalid_memory_detail")

    def test_lineage_frontier_batches_preserve_dfs_order_and_fail_closed_after_edits(self):
        ids = [f"r{i}" for i in range(405)]
        for sid in ids:
            self.raw(sid)
        self.episode("e", ids)
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={"semantic_id": "s", "timestamp": 100, "semantic_summary": "s", "source_summary_ids": ["e"]},
        )
        closure, queries = self.traced(
            lambda: self.store.resolve_lineage_source_ids(namespace=self.ns, source_ids=("s",))
        )
        self.assertEqual(closure.descendant_ids, ("e", *ids))
        self.assertLessEqual(len(queries), 9)
        self.assertTrue(all("SELECT *" not in query for query in queries))
        with self.store._conn:
            self.store._conn.execute("DELETE FROM messages WHERE source_id = ?", (ids[-1],))
        self.assertEqual(
            self.store.resolve_lineage_source_ids(namespace=self.ns, source_ids=("s",)).reason,
            "lineage_source_missing_or_out_of_scope",
        )
        with self.store._conn:
            self.store._conn.execute("UPDATE summaries SET source_ids_json = '[\"s\"]' WHERE summary_id = 'e'")
        self.assertEqual(
            self.store.resolve_lineage_source_ids(namespace=self.ns, source_ids=("s",)).reason,
            "lineage_broken_or_cyclic",
        )

    def test_anchor_window_fetches_only_selected_bodies_and_keeps_interleaved_turns(self):
        self.raw("a1", turn_id="a")
        self.raw("b1", turn_id="b")
        self.raw("a2", turn_id="a")
        self.raw("standalone")
        self.raw("c1", turn_id="c")
        window, queries = self.traced(
            lambda: self.store.get_raw_turn_window(
                namespace=self.ns, anchor_source_id="b1", before_turns=1, after_turns=1
            )
        )
        self.assertEqual([entry.source_id for entry in window.entries], ["a1", "a2", "b1", "standalone"])
        full_reads = [q for q in queries if "SELECT * FROM messages" in q]
        self.assertEqual(len(full_reads), 1)
        self.assertIn("source_id IN", full_reads[0])
        self.assertNotIn("c1", full_reads[0])

    def test_timeline_render_reuse_preserves_exact_counts_and_continuations(self):
        for i in range(20):
            self.raw(f"r{i}", timestamp=1_700_000_000 + i * 4000)
        selector = {"date_from": "2023-11-15", "date_to": "2023-11-16", "page_token_budget": 180}
        cached = self.mem.read_timeline(**selector)

        # Reference path renders every probe afresh, with the same tokenizer.
        def uncached(messages, *, tz, **kwargs):
            return render_timeline(messages, tz=tz)

        with patch("memcore.timeline_read.render_timeline", side_effect=uncached):
            original = self.mem.read_timeline(**selector)
        self.assertEqual(cached, original)
        self.assertEqual(cached["status"], "partial")
        cursor = cached["next_cursor"]
        with patch("memcore.timeline_read.render_timeline", side_effect=uncached):
            continuation = self.mem.read_timeline(cursor=cursor)
        self.assertEqual(self.mem.read_timeline(cursor=cursor), continuation)


class KeywordIndexCacheTests(unittest.TestCase):
    def test_cached_term_counts_equal_fresh_scoped_bm25_after_upsert_and_delete(self):
        index = InMemoryVectorIndex(embedding=HashedEmbeddingProvider())
        entries = [
            {"source_id": "a", "text": "字幕 文旅 字幕", "metadata": {"scope": "ok", "memory_topic_text": "配音"}},
            {"source_id": "b", "text": "文旅答辩项目", "metadata": {"scope": "ok"}},
            {"source_id": "c", "text": "字幕 " * 100, "metadata": {"scope": "foreign"}},
        ]
        index.upsert(entries)

        def check():
            candidates = [e for e in entries if _match_where(e["metadata"], {"scope": "ok"})]
            terms = [tokenize(_keyword_doc_text(e["text"], e["metadata"])) for e in candidates]
            df = Counter(term for document in terms for term in set(document))
            avgdl = sum(map(len, terms)) / len(terms)
            query = tokenize("字幕 配音")
            expected = [
                (e["source_id"], index._bm25(query, t, len(t), avgdl, len(terms), df))
                for e, t in zip(candidates, terms)
            ]
            expected = sorted((item for item in expected if item[1] > 0), key=lambda item: -item[1])
            with patch("memcore.index.memory_index.tokenize", wraps=tokenize) as tokenizer:
                result = index.keyword_search(
                    query_text="字幕 配音", entity_anchors=[], topic_terms=[], where={"scope": "ok"}
                )
            self.assertEqual(tokenizer.call_count, 1)
            self.assertEqual([(hit["source_id"], hit["tag_score"]) for hit in result], expected)

        check()
        entries[0] = {"source_id": "a", "text": "已更改", "metadata": {"scope": "ok"}}
        index.upsert([entries[0]])
        check()
        index.delete(["a"])
        entries.pop(0)
        check()


if __name__ == "__main__":
    unittest.main()
