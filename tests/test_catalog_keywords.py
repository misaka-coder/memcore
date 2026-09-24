"""Keyword catalog navigation over current SQLite metadata and lineage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)
from memcore.catalog_keywords import normalize_catalog_terms
from memcore.memory_catalog import build_memory_card


class _SummaryLLM(LLMClient):
    def call(self, request):
        return LLMResult(
            ok=True,
            data={
                "diary_summary": "讨论配音与字幕。",
                "period_label": "视频制作",
                "importance": 0.5,
                "memory_title": "配音制作",
                "catalog_hint": "配音与字幕讨论",
                "memory_metadata": {"entity_anchors": ["GPT-SoVITS"], "topic_terms": ["字幕", "配音"]},
            },
        )


class CatalogKeywordTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.namespace = Namespace(tenant_id="t", user_id="u", domain_id="d", conversation_id="c1")
        self.embedding = HashedEmbeddingProvider()
        self.mem = self._system(self.namespace)
        self.addCleanup(self.store.close)
        self.addCleanup(self.mem.close)

    def _system(self, namespace, store=None, **config):
        return MemorySystem(
            namespace=namespace,
            timezone="Asia/Shanghai",
            llm=_SummaryLLM(),
            store=store or self.store,
            embedding=self.embedding,
            index=InMemoryVectorIndex(embedding=self.embedding),
            config=MemoryConfig(**config),
        )

    def _raw(self, sid, *, entities=(), topics=(), namespace=None, **fields):
        return self.store.add_message(
            namespace=namespace or self.namespace,
            role="user",
            content="原文正文不参与完整标签匹配",
            source_id=sid,
            timestamp=100,
            memory_metadata={"entity_anchors": list(entities), "topic_terms": list(topics)},
            **fields,
        )

    def _episode(self, sid, sources=(), *, entities=(), topics=(), namespace=None, timestamp=200, **fields):
        return self.store.add_summary(
            namespace=namespace or self.namespace,
            record={
                "summary_id": sid,
                "timestamp": timestamp,
                "diary_summary": "完整摘要仅在按需打开时读取。",
                "source_ids": list(sources),
                "memory_metadata": {"entity_anchors": list(entities), "topic_terms": list(topics)},
                **fields,
            },
        )

    def _ids(self, result):
        self.assertIn(result["status"], ("ok", "empty"), result)
        return [card["memory_id"] for card in result["cards"]]

    def test_source_and_summary_terms_union_deduplicate_without_expanding_display(self):
        self._raw("r", entities=["  ＧＰＴ-ＳｏＶＩＴＳ  ", "gpt-sovits"], topics=["音画同步", "字幕"])
        self._episode("e", ["r"], entities=["GPT-SoVITS"], topics=["字幕", "配音", "gpt-sovits"])
        pools = self.store.get_catalog_keyword_pools(namespace=self.namespace, memory_ids=("e",))
        self.assertEqual(pools["e"], ["GPT-SoVITS", "字幕", "配音", "音画同步"])
        with (
            patch.object(self.embedding, "embed_query", side_effect=AssertionError("embedding read")),
            patch.object(self.mem.llm, "call", side_effect=AssertionError("LLM read")),
        ):
            found = self.mem.browse_memory(keywords=["音画同步", "字幕", "字幕"], keyword_match="all")
        self.assertEqual(self._ids(found), ["e"])
        card = found["cards"][0]
        self.assertEqual(card["entity_anchors"], ["GPT-SoVITS"])
        self.assertEqual(card["topic_terms"], ["字幕", "配音"])
        self.assertEqual(card["matched_terms"], ["音画同步", "字幕"])
        self.assertIn("period_start_at", card)
        self.assertNotIn("source_ids", card)
        self.assertNotIn("diary_summary", card)
        self.assertEqual(found["coverage"]["requested_range"]["start_ts"], None)
        opened = self.mem.open_memory(memory_id="e", view="card")
        self.assertIn("配音", json.dumps(opened, ensure_ascii=False))

    def test_summary_tags_work_without_raw_annotations_or_surviving_sources(self):
        self._raw("plain", annotation_status="missing")
        self._episode("e", ["plain", "missing-source"], topics=["字幕"])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["字幕"])), ["e"])
        self._episode("untagged", ["plain"])
        result = self.mem.browse_memory(keywords=["不存在"])
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["reason"], "no_catalog_keyword_matches")
        self.assertGreater(result["coverage"]["total_source_count"], 0)
        self.assertFalse(result["coverage"]["keyword_filtered"])

    def test_normalized_terms_any_all_and_hard_time_filter(self):
        self._episode("a", entities=["GPT-SoVITS"], topics=["音画同步", "Voice   Model"])
        self._episode("b", topics=["字幕"], timestamp=300)
        self.assertEqual(
            self._ids(self.mem.browse_memory(keywords=["ＧＰＴ-ＳＯＶＩＴＳ", "voice model"], keyword_match="all")),
            ["a"],
        )
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["音画同步", "字幕"])), ["a", "b"])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["音画同步", "字幕"], keyword_match="all")), [])
        for word in ["音画", "同步", "GPT"]:
            self.assertEqual(self._ids(self.mem.browse_memory(keywords=[word])), ["a"])
        for word in ["GPTSoVITS", "原文正文"]:
            self.assertEqual(self._ids(self.mem.browse_memory(keywords=[word])), [])
        result = self.mem.browse_memory(
            keywords=["音画同步", "字幕"],
            time_range={"start_at": "1970-01-01T00:03:20Z", "end_at": "1970-01-01T00:05:00Z"},
        )
        self.assertEqual(self._ids(result), ["a"])
        self.assertEqual(len(result["coverage"]["covered_intervals"]), 1)

    def test_one_way_contains_combines_exact_and_partial_cards_with_stored_witnesses(self):
        self._episode("long", topics=["文旅答辩项目"], timestamp=200)
        self._episode("short", topics=["文旅"], timestamp=300)
        result = self.mem.browse_memory(keywords=["文旅"])
        self.assertEqual(self._ids(result), ["long", "short"])
        self.assertEqual(result["cards"][0]["keyword_hits"], [{"query": "文旅", "term": "文旅答辩项目"}])
        self.assertEqual(result["cards"][1]["keyword_hits"], [{"query": "文旅", "term": "文旅"}])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["文旅答辩项目"])), ["long"])
        both = self.mem.browse_memory(keywords=["文旅", "答辩"], keyword_match="all")
        self.assertEqual(self._ids(both), ["long"])
        self.assertEqual(both["cards"][0]["matched_terms"], ["文旅", "答辩"])
        self.assertEqual([hit["term"] for hit in both["cards"][0]["keyword_hits"]], ["文旅答辩项目"] * 2)

    def test_witness_prefers_exact_tag_and_partial_cursor_restarts_after_source_repair(self):
        self._raw("r", topics=["文旅答辩项目"])
        self._episode("a", ["r"], topics=["字幕"])
        self._episode("b", topics=["文旅答辩", "文旅"], timestamp=300)
        first = self.mem.browse_memory(keywords=["文旅"], page_size=1)
        second = self.mem.browse_memory(cursor=first["next_cursor"])
        self.assertEqual(second["cards"][0]["keyword_hits"], [{"query": "文旅", "term": "文旅"}])
        self.mem.update_turn_metadata("r", {"topic_terms": ["文旅展览项目"]})
        self.assertEqual(
            self.mem.browse_memory(cursor=first["next_cursor"])["reason"], "catalog_changed_restart_required"
        )

    def test_semantic_pool_contains_own_episode_and_raw_tags_only_from_exact_lineage(self):
        self._raw("r", topics=["原始词"])
        self._raw("unrelated", topics=["无关词"])
        self._episode("e", ["r"], topics=["摘要词"])
        self._episode("other", ["unrelated"])
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={
                "semantic_id": "s",
                "timestamp": 400,
                "semantic_summary": "长期正文",
                "source_summary_ids": ["e", "e", "missing"],
                "memory_metadata": {"topic_terms": ["长期词"]},
            },
        )
        result = self.mem.browse_memory(
            keywords=["原始词", "摘要词", "长期词"], keyword_match="all", node_types=["semantic"]
        )
        self.assertEqual(self._ids(result), ["s"])
        self.assertEqual(result["cards"][0]["topic_terms"], ["长期词"])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["无关词"], node_types=["semantic"])), [])

    def test_hard_namespace_and_conversation_scope_apply_to_roots_and_sources(self):
        namespaces = [
            Namespace(tenant_id="t", user_id="u", domain_id="d", conversation_id="c2"),
            Namespace(tenant_id="other", user_id="u", domain_id="d", conversation_id="c1"),
            Namespace(tenant_id="t", user_id="other", domain_id="d", conversation_id="c1"),
            Namespace(tenant_id="t", user_id="u", domain_id="other", conversation_id="c1"),
        ]
        for i, namespace in enumerate(namespaces):
            self._raw(f"r{i}", topics=[f"词{i}"], namespace=namespace)
            self._episode(f"e{i}", [f"r{i}"], topics=["共同词"], namespace=namespace)
        self._episode("local", ["r0", "r1", "r2", "r3"])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["共同词", "词0"])), [])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["共同词"], cross_conversation=True)), ["e0"])
        for word in ["词1", "词2", "词3"]:
            self.assertEqual(self._ids(self.mem.browse_memory(keywords=[word], cross_conversation=True)), [])

    def test_explicit_never_unaccepted_and_typed_operation_sources_do_not_supply_tags(self):
        fields = [
            {"retrieval_visibility": "explicit"},
            {"retrieval_visibility": "never"},
            {"annotation_status": "rejected"},
            {"annotation_status": "missing"},
            {"kind": "tool.search.result"},
            {"kind": "material.image.reference"},
            {"kind": "skill.catalog.response"},
            {"kind": "operation.memory.result"},
            {"kind": "custom.result", "turn_role": "observation"},
        ]
        for i, attrs in enumerate(fields):
            self._raw(str(i), topics=["隐藏词"], **attrs)
        self._episode("e", [str(i) for i in range(len(fields))])
        self._episode("digest", topics=["隐藏词"], kind="memory.operation_digest")
        self._episode("explicit-summary", topics=["隐藏词"], retrieval_visibility="explicit")
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["隐藏词"])), [])

    def test_keyword_cursor_freezes_selector_and_detects_changed_results(self):
        for i in range(3):
            self._episode(str(i), topics=["共同词"], timestamp=200 + i)
        first = self.mem.browse_memory(keywords=["共同词"], page_size=1, keyword_match="all")
        second = self.mem.browse_memory(cursor=first["next_cursor"])
        third = self.mem.browse_memory(cursor=second["next_cursor"])
        self.assertEqual(self._ids(first) + self._ids(second) + self._ids(third), ["0", "1", "2"])
        self.assertTrue(third["page_complete"])
        self.assertEqual(second["keywords"], ["共同词"])
        self.assertEqual(second["keyword_match"], "all")
        self.assertEqual(
            self.mem.browse_memory(cursor=first["next_cursor"], keywords=[])["reason"], "cursor_options_are_embedded"
        )
        other = self._system(Namespace(user_id="other"))
        self.addCleanup(other.close)
        self.assertEqual(other.browse_memory(cursor=first["next_cursor"])["reason"], "invalid_cursor_scope")
        self._episode("inserted", topics=["共同词"], timestamp=150)
        changed = self.mem.browse_memory(cursor=first["next_cursor"])
        self.assertEqual(changed["reason"], "catalog_changed_restart_required")

    def test_metadata_repair_and_source_delete_cannot_leave_cached_raw_terms(self):
        self._raw("r", topics=["旧词"])
        self._episode("e", ["r"], topics=["摘要词"])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["旧词"])), ["e"])
        self.mem.update_turn_metadata("r", {"topic_terms": ["新词"]})
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["旧词"])), [])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["新词"])), ["e"])
        with self.store._conn:
            self.store._conn.execute("DELETE FROM messages WHERE source_id = 'r'")
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["新词"])), [])
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["摘要词"])), ["e"])
        self.mem.forget_namespace()
        self.assertEqual(self._ids(self.mem.browse_memory(keywords=["摘要词"])), [])

    def test_invalid_keywords_never_broaden_a_dated_query(self):
        self._episode("e", topics=["词"])
        for kwargs in [
            {"keywords": "词"},
            {"keywords": ["词", 3]},
            {"keywords": [" "]},
            {"keywords": ["词"], "keyword_match": "fuzzy"},
            {"keyword_match": "all"},
        ]:
            with self.subTest(kwargs=kwargs):
                direct = self.mem.browse_memory(date_from="1970-01-01", **kwargs)
                native = dispatch_native_memory_tool(
                    "browse_memory", {"date_from": "1970-01-01", **kwargs}, mem=self.mem
                )
                self.assertEqual(direct["status"], "invalid_filter")
                self.assertFalse(native["ok"])
        self.assertEqual(self.mem.browse_memory()["status"], "invalid_filter")

    def test_unsupported_store_and_ambiguous_ids_are_unavailable(self):
        self._episode("e", topics=["词"])
        with patch.object(self.store, "get_catalog_keyword_pools", side_effect=NotImplementedError):
            result = self.mem.browse_memory(keywords=["词"])
        self.assertEqual(result["status"], "unavailable")
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={"semantic_id": "e", "timestamp": 200, "semantic_summary": "同 ID 长期记忆"},
        )
        result = self.mem.browse_memory(keywords=["词"], node_types=["episodic", "semantic"])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "ambiguous_memory_id")

    def test_coverage_is_not_reduced_to_keyword_hits_and_bad_metadata_does_not_hide_good_sources(self):
        self._raw("good", topics=["好词"])
        self._raw("bad", topics=["坏词"])
        self._episode("e", ["good", "bad"])
        self._episode("other", topics=["其他词"], timestamp=400)
        with self.store._conn:
            self.store._conn.execute("UPDATE messages SET memory_metadata_json = 'broken' WHERE source_id = 'bad'")
        result = self.mem.browse_memory(keywords=["好词"])
        self.assertEqual(self._ids(result), ["e"])
        self.assertEqual(len(result["coverage"]["covered_intervals"]), 2)
        self.assertEqual(result["coverage"]["total_source_count"], 2)

    def test_native_result_receipt_and_cursor_keep_keyword_contract(self):
        self._episode("e1", topics=["字幕"])
        self._episode("e2", topics=["字幕"], timestamp=300)
        first = dispatch_native_memory_tool("browse_memory", {"keywords": ["字幕"], "page_size": 1}, mem=self.mem)
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["receipt"]["selector"]["keywords"], ["字幕"])
        self.assertEqual(first["result"]["cards"][0]["matched_terms"], ["字幕"])
        cursor = first["result"]["next_cursor"]
        continued = dispatch_native_memory_tool(
            "browse_memory", {"cursor": cursor, "keywords": None, "keyword_match": None}, mem=self.mem
        )
        self.assertEqual(continued["receipt"]["selector"]["keywords"], ["字幕"])
        rejected = dispatch_native_memory_tool(
            "browse_memory", {"cursor": cursor, "keyword_match": "any"}, mem=self.mem
        )
        self.assertFalse(rejected["ok"])
        spec = next(s for s in build_native_memory_tool_specs(tool_format="plain") if s["name"] == "browse_memory")
        self.assertIn("keyword_match", spec["parameters"]["properties"])
        self.assertIn("stored tags must contain", spec["parameters"]["properties"]["keywords"]["description"])

    def test_real_compaction_retains_raw_only_and_generated_summary_tags(self):
        self._raw("annotated", topics=["音画同步"])
        self._raw("unannotated", annotation_status="missing")
        compactor = self._system(self.namespace, raw_token_trigger=1, episodic_compact_trigger_count=99)
        self.addCleanup(compactor.close)
        result = compactor.compact_due_sync()
        self.assertEqual(result["status"], "compacted")
        found = self.mem.browse_memory(keywords=["音画同步", "字幕"], keyword_match="all")
        self.assertEqual(found["matched_card_count"], 1, found)
        self.assertEqual(found["cards"][0]["topic_terms"], ["字幕", "配音"])

    def test_restart_old_catalog_without_generated_fields_needs_no_backfill(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memory.sqlite3")
            store = SQLiteMemoryStore(path)
            store.add_summary(
                namespace=self.namespace,
                record={
                    "summary_id": "old",
                    "timestamp": 100,
                    "diary_summary": "旧摘要",
                    "memory_metadata": {"topic_terms": ["历史词"]},
                },
            )
            store.close()
            reopened = SQLiteMemoryStore(path)
            mem = self._system(self.namespace, store=reopened)
            try:
                found = mem.browse_memory(keywords=["历史词"])
                self.assertEqual(self._ids(found), ["old"])
                self.assertEqual(found["cards"][0]["catalog_quality"], "fallback")
            finally:
                mem.close()
                reopened.close()

    def test_large_lineage_batches_metadata_columns_without_loading_raw_bodies(self):
        ids = [f"r{i}" for i in range(405)]
        for sid in ids:
            self._raw(sid, topics=["共同词", sid])
        self._episode("e", ids)
        queries = []
        self.store._conn.set_trace_callback(queries.append)
        try:
            found = self.mem.browse_memory(keywords=[ids[-1]])
        finally:
            self.store._conn.set_trace_callback(None)
        self.assertEqual(self._ids(found), ["e"])
        raw_reads = [q for q in queries if "SELECT source_id, memory_metadata_json" in q]
        self.assertEqual(len(raw_reads), 2)
        self.assertTrue(all("content" not in q and "payload" not in q for q in raw_reads))

    def test_display_and_pool_normalization_preserve_distinct_names_and_punctuation(self):
        self.assertEqual(
            normalize_catalog_terms([" ＡＢＣ ", "abc", "C++", "C#", "A  B", "a\tb"]), ["ABC", "C++", "C#", "A B"]
        )
        card = build_memory_card(
            {
                "summary_id": "e",
                "memory_metadata": {"entity_anchors": ["ＡＢＣ", "abc"], "topic_terms": ["ABC", "字幕", " 字幕 "]},
            }
        )
        self.assertEqual(card["entity_anchors"], ["ABC"])
        self.assertEqual(card["topic_terms"], ["字幕"])


if __name__ == "__main__":
    unittest.main()
