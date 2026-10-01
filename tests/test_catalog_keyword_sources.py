"""Literal raw evidence joins tags without changing fuzzy retrieval admission."""

import json
import unittest
from dataclasses import replace

from memcore import Actor, Namespace, dispatch_native_memory_tool
from tests import test_catalog_keywords as catalog_fixture


class CatalogRawEvidenceTests(unittest.TestCase):
    setUp = catalog_fixture.CatalogKeywordTests.setUp
    _system = catalog_fixture.CatalogKeywordTests._system
    _episode = catalog_fixture.CatalogKeywordTests._episode

    def utterance(self, sid, text, **fields):
        return self.store.add_message(
            namespace=replace(self.namespace, actor=Actor(stable_id="qq:1", display_name="小林")),
            role="user",
            source_id=sid,
            content=text,
            timestamp=100,
            **fields,
        )

    def test_tag_raw_union_all_unique_counts_and_complete_native_read(self):
        for sid, text in (("a", "文旅答辩答辩"), ("b", "文旅答辩答辩"), ("c", "答辩准备")):
            self.utterance(sid, text, kind="message.user.observed", retrieval_visibility="never")
        self._episode("e", ["a", "a", "b", "c"], topics=["文旅"])
        found = dispatch_native_memory_tool(
            "browse_memory", {"keywords": ["文旅", "答辩"], "keyword_match": "all"}, mem=self.mem
        )
        card = found["result"]["cards"][0]
        self.assertEqual(card["matched_terms"], ["文旅", "答辩"])
        self.assertEqual(card["matched_source_count"], 3)
        self.assertEqual(card["keyword_hits"], [{"query": "文旅", "term": "文旅"}])
        self.assertNotIn("matched_source_ids", card)
        self.assertEqual(card["match_preview"]["source_id"], "a")
        opened = dispatch_native_memory_tool("open_memory", {"match_ref": card["match_ref"]}, mem=self.mem)
        self.assertTrue(opened["ok"], opened)
        body = opened["result"]
        self.assertEqual(body["source_mode"], "keyword_filtered")
        self.assertEqual(body["matched_source_count"], 3)
        self.assertTrue(body["page_complete"])
        self.assertNotIn("messages", json.dumps({key: value for key, value in body.items() if key != "text"}))
        for sid in ("a", "b", "c"):
            self.assertEqual(body["text"].count(f'"{sid}"'), 1)
        self.assertEqual(body["text"].count("文旅答辩答辩"), 2)
        self.assertEqual(
            str(self.store.get_entry(namespace=self.namespace, source_id="a").retrieval_visibility), "never"
        )
        self.mem.reindex_all()
        self.assertFalse(any(item.source_id == "a" for item in self.mem.retrieve_structured(query="文旅答辩").matches))

    def test_all_can_be_split_between_tag_and_distinct_utterances(self):
        self.utterance("a", "展览准备")
        self.utterance("b", "答辩完成")
        self._episode("e", ["a", "b"], topics=["文旅"])
        card = self.mem.browse_memory(keywords=["文旅", "答辩", "展览"], keyword_match="all")["cards"][0]
        self.assertEqual(card["matched_source_count"], 2)
        evidence = self.mem.open_memory(match_ref=card["match_ref"])
        self.assertEqual([row["source_id"] for row in evidence["result"]["messages"]], ["a", "b"])

    def test_semantic_descendants_scope_and_operation_bodies(self):
        self.utterance("voice", "Ｃ＋＋ 文旅", kind="message.user.voice")
        self.store.add_message(
            namespace=self.namespace,
            role="tool_result",
            content="文旅秘密轨迹",
            source_id="tool",
            timestamp=100,
            kind="tool.search.result",
            origin="environment",
            turn_role="observation",
            correlation_id="c",
        )
        self._episode("e", ["voice", "tool"])
        self.store.add_semantic_summary(
            namespace=self.namespace,
            record={"semantic_id": "s", "timestamp": 400, "source_summary_ids": ["e"], "semantic_summary": "长期概括"},
        )
        card = self.mem.browse_memory(keywords=["C++"], node_types=["semantic"])["cards"][0]
        self.assertEqual(card["matched_source_count"], 1)
        self.assertIn("Ｃ＋＋ 文旅", self.mem.open_memory(match_ref=card["match_ref"])["text"])
        self.assertFalse(self.mem.browse_memory(keywords=["秘密轨迹"])["cards"])
        foreign = self._system(Namespace(tenant_id="t", user_id="other", domain_id="d", conversation_id="c1"))
        self.addCleanup(foreign.close)
        self.assertEqual(foreign.open_memory(match_ref=card["match_ref"])["status"], "invalid_filter")

    def test_pagination_survives_reload_and_rejects_source_changes(self):
        mem = self._system(self.namespace, native_timeline_page_token_budget=200)
        self.addCleanup(mem.close)
        for i in range(6):
            self.utterance(f"r{i}", "答辩" + "完整原话" * 30)
        self._episode("e", [f"r{i}" for i in range(6)])
        card = mem.browse_memory(keywords=["答辩"])["cards"][0]
        page = dispatch_native_memory_tool("open_memory", {"match_ref": card["match_ref"]}, mem=mem)["result"]
        self.assertFalse(page["page_complete"])
        first_cursor = page["next_cursor"]
        pages = [page]
        while not page["page_complete"]:
            page = dispatch_native_memory_tool("open_memory", {"cursor": page["next_cursor"]}, mem=mem)["result"]
            pages.append(page)
        self.assertEqual(sum(p["returned_source_count"] for p in pages), 6)
        body = "\n".join(p["text"] for p in pages)
        for i in range(6):
            self.assertEqual(body.count(f'"r{i}"'), 1)
        with self.store._conn:
            self.store._conn.execute(
                "UPDATE messages SET semantic_text='不再匹配', content='不再匹配' WHERE source_id='r5'"
            )
        self.assertEqual(mem.open_memory(cursor=first_cursor)["reason"], "source_changed_restart_required")
        self.assertEqual(mem.open_memory(match_ref=card["match_ref"])["reason"], "source_changed_restart_required")

    def test_invalid_reference_combinations_and_missing_sources_never_broaden(self):
        self.utterance("a", "答辩")
        self._episode("e", ["a", "missing"])
        card = self.mem.browse_memory(keywords=["答辩"])["cards"][0]
        self.assertFalse(card["keyword_evidence_complete"])
        opened = self.mem.open_memory(match_ref=card["match_ref"])
        self.assertEqual(opened["status"], "partial")
        self.assertEqual(opened["missing_source_count"], 1)
        for additional in ({"memory_id": "e"}, {"speaker_ids": []}, {"view": "sources"}, {"cursor": card["match_ref"]}):
            self.assertFalse(
                dispatch_native_memory_tool(
                    "open_memory", {"match_ref": card["match_ref"], **additional}, mem=self.mem
                )["ok"]
            )
