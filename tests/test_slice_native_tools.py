"""Native tool helper tests: schema generation + strict dispatch."""

from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    RetrievalMatch,
    RetrievalResult,
    SQLiteMemoryStore,
    ToolDispatchPolicy,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)


class ToolLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


def _ts(y: int, mo: int, d: int, h: int, mi: int = 0, tz: str = "Asia/Shanghai") -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp())


def _shared_mem(conversation: str = "c1"):
    emb = HashedEmbeddingProvider()
    store = SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    mem = MemorySystem(
        llm=ToolLLM(),
        namespace=Namespace(user_id="u1", conversation_id=conversation),
        timezone="Asia/Shanghai",
        store=store,
        index=index,
        embedding=emb,
        config=MemoryConfig(),
    )
    return mem, store, index, emb


class NativeToolSpecs(unittest.TestCase):
    def test_openai_tool_specs_include_memory_and_material_tools(self) -> None:
        tools = build_native_memory_tool_specs(tool_format="openai")

        names = [tool["function"]["name"] for tool in tools]

        self.assertEqual(
            names,
            [
                "retrieve_for_turn",
                "browse_memory",
                "open_memory",
                "read_timeline",
                "load_material",
            ],
        )
        retrieve = tools[0]["function"]
        self.assertTrue(retrieve["strict"])
        self.assertFalse(retrieve["parameters"]["additionalProperties"])
        self.assertEqual(
            retrieve["parameters"]["properties"]["memory_facets"]["items"]["enum"],
            [
                "profile",
                "preference",
                "viewpoint",
                "relationship",
                "event",
                "state",
                "plan",
                "decision",
                "constraint",
                "knowledge",
                "procedure",
            ],
        )
        self.assertIn("include_explicit", retrieve["parameters"]["properties"])
        self.assertIn("kind_patterns", retrieve["parameters"]["properties"])
        browse = tools[1]["function"]
        self.assertIn("time_range", browse["parameters"]["properties"])
        self.assertEqual(
            browse["parameters"]["properties"]["node_types"]["items"]["enum"],
            ["episodic", "semantic"],
        )
        self.assertIn("cursor", browse["parameters"]["properties"])
        opened = tools[2]["function"]
        self.assertIn("memory_ids", opened["parameters"]["properties"])
        self.assertEqual(
            opened["parameters"]["properties"]["view"]["enum"],
            ["card", "content", "sources", None],
        )
        self.assertEqual(
            opened["parameters"]["properties"]["projection"]["enum"],
            ["conversation", "full", "tools", None],
        )
        timeline = tools[3]["function"]
        self.assertIn("time_range", timeline["parameters"]["properties"])
        self.assertEqual(
            timeline["parameters"]["properties"]["time_range"]["required"],
            ["start_at", "end_at"],
        )
        self.assertEqual(
            timeline["parameters"]["properties"]["projection"]["enum"],
            ["conversation", "full", "tools", None],
        )
        self.assertIn("cursor", timeline["parameters"]["properties"])
        self.assertEqual(tools[4]["function"]["name"], "load_material")

    def test_openai_strict_schema_makes_optional_fields_nullable_required(self) -> None:
        tools = build_native_memory_tool_specs(tool_format="openai")
        retrieve_schema = tools[0]["function"]["parameters"]

        self.assertEqual(retrieve_schema["required"], list(retrieve_schema["properties"]))
        self.assertEqual(retrieve_schema["properties"]["entity_anchors"]["type"], ["array", "null"])
        self.assertEqual(retrieve_schema["properties"]["memory_facets"]["type"], ["array", "null"])

        time_hint = retrieve_schema["properties"]["time_hint"]
        self.assertEqual(time_hint["type"], ["object", "null"])
        self.assertEqual(time_hint["required"], list(time_hint["properties"]))
        self.assertEqual(time_hint["properties"]["start_at"]["type"], ["string", "null"])
        self.assertEqual(time_hint["properties"]["end_at"]["type"], ["string", "null"])

    def test_provider_formats_are_available(self) -> None:
        plain = build_native_memory_tool_specs(tool_format="plain", include_material_tool=False)
        anthropic = build_native_memory_tool_specs(tool_format="anthropic", include_material_tool=False)
        responses = build_native_memory_tool_specs(tool_format="openai_responses", include_material_tool=False)

        self.assertEqual(plain[0]["name"], "retrieve_for_turn")
        self.assertIn("input_schema", anthropic[0])
        self.assertEqual(responses[0]["type"], "function")
        self.assertEqual(responses[0]["name"], "retrieve_for_turn")


class NativeToolDispatch(unittest.TestCase):
    def test_explicit_kind_queries_require_and_cannot_expand_host_policy(self) -> None:
        seen: list[dict] = []

        class FakeMem:
            config = SimpleNamespace()

            def retrieve_for_turn_structured(self, **kwargs):
                seen.append(kwargs)
                return SimpleNamespace(
                    rendered_texts=("event result",),
                    to_dict=lambda: {"status": "found", "matches": []},
                )

        denied = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "财经事件", "include_explicit": True, "kind_patterns": ["event.finance.*"]},
            mem=FakeMem(),
            current={"source_id": "current"},
        )
        widened = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "工具", "include_explicit": True, "kind_patterns": ["tool.*"]},
            mem=FakeMem(),
            current={"source_id": "current"},
            policy=ToolDispatchPolicy(
                allow_explicit_trace=True,
                allowed_kind_prefixes=("event.finance",),
            ),
        )
        allowed = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "财经事件", "include_explicit": True, "kind_patterns": ["event.finance.*"]},
            mem=FakeMem(),
            current={"source_id": "current"},
            policy=ToolDispatchPolicy(
                allow_explicit_trace=True,
                allowed_kind_prefixes=("event.finance",),
            ),
        )

        self.assertEqual(denied["status"], "forbidden")
        self.assertEqual(widened["status"], "forbidden")
        self.assertTrue(allowed["ok"])
        self.assertEqual(seen[0]["kind_patterns"], ["event.finance.*"])
        self.assertTrue(seen[0]["include_explicit"])

    def test_retrieve_for_turn_dispatches_and_excludes_visible_context(self) -> None:
        mem, store, index, emb = _shared_mem(conversation="c1")
        other = MemorySystem(
            llm=ToolLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c2"),
            timezone="Asia/Shanghai",
            store=store,
            index=index,
            embedding=emb,
            config=mem.config,
        )
        mem.record_user_turn("当前会话可见 raw 可乐", timestamp=1000, source_id="visible")
        current = mem.record_user_turn("我之前说过喜欢什么饮料吗", timestamp=1001, source_id="cur")
        other.record_user_turn(
            "跨会话隐藏 raw 可乐",
            timestamp=900,
            source_id="hidden",
            memory_metadata={
                "memory_facets": ["preference"],
                "about_roles": ["user"],
                "entity_anchors": ["可乐"],
            },
        )

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "可乐", "entity_anchors": ["可乐"], "memory_facets": ["preference"]},
            mem=mem,
            current=current,
        )

        self.assertTrue(out["ok"])
        blob = "\n".join(out["result"]["snippets"])
        self.assertIn("跨会话隐藏 raw 可乐", blob)
        self.assertNotIn("当前会话可见 raw 可乐", blob)
        store.close()

    def test_retrieve_result_exposes_reloadable_navigation_and_closed_loop_guidance(self) -> None:
        class FakeMem:
            config = SimpleNamespace()

            def retrieve_for_turn_structured(self, **_kwargs):
                common = {
                    "correlation_id": "",
                    "kind": "message.user",
                    "semantic_score": 0.8,
                    "bm25_score": 0.5,
                    "fused_score": 0.9,
                    "semantic_text": "",
                }
                return RetrievalResult(
                    status="found",
                    matches=(
                        RetrievalMatch(
                            source_ids=("episode-fable",),
                            source_id="episode-fable",
                            turn_id="",
                            layer="summary",
                            timestamp=1785727000,
                            rendered_text="summary snippet",
                            lineage=("raw-a", "raw-b"),
                            **common,
                        ),
                        RetrievalMatch(
                            source_ids=("raw-c",),
                            source_id="raw-c",
                            turn_id="turn-c",
                            layer="raw",
                            timestamp=1785727800,
                            rendered_text="raw snippet",
                            **common,
                        ),
                    ),
                )

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "Fable"},
            mem=FakeMem(),
            current={"source_id": "current"},
        )

        self.assertEqual(
            out["result"]["navigation"],
            [
                {"match_index": 1, "layer": "summary", "memory_id": "episode-fable"},
                {
                    "match_index": 2,
                    "layer": "raw",
                    "source_id": "raw-c",
                    "turn_id": "turn-c",
                    "timestamp": 1785727800,
                },
            ],
        )
        self.assertIn("open_memory_sources_for_original_evidence", out["result"]["suggested_next_actions"])
        self.assertIn(
            "retrieve_again_only_with_new_entity_time_or_search_target_evidence",
            out["result"]["suggested_next_actions"],
        )

    def test_retrieve_dispatch_accepts_local_exact_time_without_unknown_entity(self) -> None:
        mem, store, index, emb = _shared_mem(conversation="live")
        history = MemorySystem(
            llm=ToolLLM(),
            namespace=Namespace(user_id="u1", conversation_id="history"),
            timezone="Asia/Shanghai",
            store=store,
            index=index,
            embedding=emb,
            config=mem.config,
        )
        history.record_user_turn(
            "misaka 和李嘉图一起来玩的",
            timestamp=_ts(2026, 8, 3, 11, 47),
            source_id="target",
            memory_metadata={
                "entity_anchors": ["misaka", "李嘉图"],
                "topic_terms": ["同行", "一起来玩"],
                "memory_facets": ["relationship"],
                "about_roles": ["third_party"],
            },
        )
        current = mem.record_user_turn("同行的人是谁", timestamp=_ts(2026, 8, 4, 20), source_id="cur")

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {
                "query": "和 misaka 一起来玩的另一个人是谁",
                "entity_anchors": ["misaka"],
                "topic_terms": ["同行", "一起来玩"],
                "memory_facets": ["relationship"],
                "about_roles": ["third_party"],
                "time_hint": {
                    "start_at": "2026-08-03 11:00",
                    "end_at": "2026-08-03 12:00",
                },
            },
            mem=mem,
            current=current,
        )

        self.assertTrue(out["ok"])
        self.assertIn("李嘉图", "\n".join(out["result"]["snippets"]))
        history.close()
        mem.close()
        store.close()

    def test_retrieve_rejects_invalid_facet_instead_of_broad_searching(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        current = mem.record_user_turn("我喜欢可乐", timestamp=1000, source_id="cur")

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "可乐", "memory_facets": ["not_a_facet"]},
            mem=mem,
            current=current,
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "invalid_arguments")
        self.assertIn("invalid_memory_facets", out["reason"])
        store.close()

    def test_retrieve_rejects_mixed_time_hint_modes_before_dispatch(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        current = mem.record_user_turn("同行的人是谁", timestamp=1000, source_id="cur")

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {
                "query": "同行的人是谁",
                "time_hint": {
                    "start_at": "2026-08-03 11:00",
                    "end_at": "2026-08-03 12:00",
                    "date_label": "2026-08-03",
                },
            },
            mem=mem,
            current=current,
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "invalid_arguments")
        self.assertEqual(out["reason"], "time_hint_modes_are_mutually_exclusive")
        store.close()

    def test_read_timeline_dispatches_and_reports_invalid_filter(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        mem.record_user_turn("上午上传了图片", timestamp=_ts(2026, 4, 10, 9), source_id="m1")

        ok = dispatch_native_memory_tool(
            "read_timeline",
            {"date_from": "2026-04-10", "time_periods": ["上午"]},
            mem=mem,
        )
        bad = dispatch_native_memory_tool("read_timeline", {"date_from": "2026/04/10"}, mem=mem)

        self.assertTrue(ok["ok"])
        self.assertEqual(ok["result"]["message_count"], 1)
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["status"], "invalid_filter")
        store.close()

    def test_browse_and_open_dispatch_form_a_catalog_to_evidence_loop(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        mem.record_user_turn("聊了扬州早茶", timestamp=_ts(2026, 4, 10, 9), source_id="raw-breakfast")
        store.add_summary(
            namespace=mem.namespace,
            record={
                "summary_id": "episode-breakfast",
                "timestamp": _ts(2026, 4, 10, 9),
                "period_start_ts": _ts(2026, 4, 10, 9),
                "period_end_ts": _ts(2026, 4, 10, 9),
                "diary_summary": "聊了扬州早茶。",
                "memory_title": "扬州早茶",
                "catalog_hint": "可回答早茶话题。",
                "catalog_schema_version": 1,
                "source_ids": ["raw-breakfast"],
            },
        )
        store.mark_messages_summarized(["raw-breakfast"], "episode-breakfast")

        catalog = dispatch_native_memory_tool(
            "browse_memory",
            {"date_from": "2026-04-10"},
            mem=mem,
        )
        evidence = dispatch_native_memory_tool(
            "open_memory",
            {"memory_id": "episode-breakfast", "view": "sources"},
            mem=mem,
        )

        self.assertTrue(catalog["ok"])
        self.assertEqual(catalog["result"]["cards"][0]["memory_id"], "episode-breakfast")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["result"]["result"]["source_count"], 1)
        self.assertNotIn("source_units", evidence["result"]["result"])
        self.assertEqual(evidence["result"]["result"]["returned_logical_unit_ids"], ["source:c1:raw-breakfast"])
        self.assertIn("聊了扬州早茶", evidence["result"]["text"])
        self.assertEqual(evidence["result"]["result_projection"], "rendered_text_with_navigation_metadata")
        self.assertEqual(evidence["receipt"]["returned_logical_unit_ids"], ["source:c1:raw-breakfast"])
        store.close()

    def test_open_memory_dispatch_batches_content_without_duplicate_bodies(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        mem.record_user_turn("第一条批量证据", timestamp=_ts(2026, 4, 10, 9), source_id="raw-a")
        mem.record_user_turn("第二条批量证据", timestamp=_ts(2026, 4, 10, 10), source_id="raw-b")

        out = dispatch_native_memory_tool(
            "open_memory",
            {"memory_ids": ["raw-b", "missing", "raw-a"], "view": "content"},
            mem=mem,
        )

        self.assertTrue(out["ok"])
        self.assertEqual(out["result"]["status"], "partial")
        self.assertEqual(
            [item["memory_id"] for item in out["result"]["result"]["items"]],
            ["raw-b", "missing", "raw-a"],
        )
        self.assertEqual(
            [item["status"] for item in out["result"]["result"]["items"]],
            ["ok", "empty", "ok"],
        )
        self.assertIn("第一条批量证据", out["result"]["text"])
        self.assertIn("第二条批量证据", out["result"]["text"])
        self.assertNotIn("第一条批量证据", str(out["result"]["result"]))
        self.assertEqual(out["receipt"]["returned_memory_ids"], ["raw-b", "raw-a"])

        rejected = dispatch_native_memory_tool(
            "open_memory",
            {"memory_ids": ["raw-a", "raw-b"], "view": "sources"},
            mem=mem,
        )
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["reason"], "batch_sources_not_supported")
        store.close()

    def test_read_timeline_dispatches_exact_time_range_without_unknown_entity(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        mem.record_user_turn("上午无关消息", timestamp=_ts(2026, 4, 10, 8), source_id="early")
        mem.record_user_turn("我们俩一起来玩的", timestamp=_ts(2026, 4, 10, 11, 47), source_id="target")

        out = dispatch_native_memory_tool(
            "read_timeline",
            {
                "time_range": {
                    "start_at": "2026-04-10 11:00",
                    "end_at": "2026-04-10 12:00",
                }
            },
            mem=mem,
        )

        self.assertTrue(out["ok"])
        self.assertNotIn("messages", out["result"])
        self.assertIn("我们俩一起来玩的", out["result"]["text"])
        self.assertEqual(out["result"]["result_projection"], "rendered_text_with_navigation_metadata")
        store.close()

    def test_read_entry_dispatches_namespace_safe_raw_expansion(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        mem.record_user_turn("原始证据", timestamp=_ts(2026, 4, 10, 11), source_id="raw-entry")

        out = dispatch_native_memory_tool(
            "read_entry",
            {"source_id": "raw-entry", "detail": "full"},
            mem=mem,
        )
        missing = dispatch_native_memory_tool(
            "read_entry",
            {"source_id": "missing", "detail": "full"},
            mem=mem,
        )

        self.assertTrue(out["ok"])
        self.assertEqual(out["result"]["entry"]["source_id"], "raw-entry")
        self.assertIn("原始证据", out["result"]["text"])
        self.assertTrue(missing["ok"])
        self.assertEqual(missing["result"]["status"], "empty")
        store.close()

    def test_timeline_cursor_rejects_repeated_selector_options(self) -> None:
        out = dispatch_native_memory_tool(
            "read_timeline",
            {"cursor": "timeline-v1:not-real:not-real", "date_from": "2026-04-10"},
            mem=SimpleNamespace(),
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "invalid_arguments")
        self.assertEqual(out["reason"], "cursor_options_are_embedded")

    def test_load_material_uses_host_loader(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        seen: list[dict] = []

        def load_material(args: dict) -> dict:
            seen.append(args)
            return {"status": "derived_ready", "text": "图片里是一道物理题。"}

        out = dispatch_native_memory_tool(
            "load_material",
            {"file_id": "file_img_001", "kind": "image", "preferred_source": "derived"},
            mem=mem,
            material_loader=load_material,
        )
        missing = dispatch_native_memory_tool("load_material", {"file_id": "file_img_001"}, mem=mem)

        self.assertTrue(out["ok"])
        self.assertEqual(seen[0]["preferred_source"], "derived")
        self.assertEqual(out["result"]["requested_file_id"], "file_img_001")
        self.assertEqual(out["result"]["status"], "derived_ready")
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["reason"], "material_loader_not_configured")
        store.close()

    def test_load_material_rejects_mismatched_file_id_result(self) -> None:
        mem, store, _index, _emb = _shared_mem()

        def load_material(_args: dict) -> dict:
            return {"file_id": "old_file", "status": "derived_ready", "text": "上一张图的摘要"}

        out = dispatch_native_memory_tool(
            "load_material",
            {"file_id": "file_img_001", "kind": "image", "preferred_source": "derived"},
            mem=mem,
            material_loader=load_material,
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "file_id_mismatch")
        self.assertIn("requested=file_img_001", out["reason"])
        store.close()


if __name__ == "__main__":
    unittest.main()
