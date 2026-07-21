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
        config=MemoryConfig(enable_verifier=False),
    )
    return mem, store, index, emb


class NativeToolSpecs(unittest.TestCase):
    def test_openai_tool_specs_include_memory_and_material_tools(self) -> None:
        tools = build_native_memory_tool_specs(categories=("preference", "material_trace"), tool_format="openai")

        names = [tool["function"]["name"] for tool in tools]

        self.assertEqual(names, ["retrieve_for_turn", "read_timeline", "load_material"])
        retrieve = tools[0]["function"]
        self.assertTrue(retrieve["strict"])
        self.assertFalse(retrieve["parameters"]["additionalProperties"])
        self.assertEqual(
            retrieve["parameters"]["properties"]["categories"]["items"]["enum"],
            ["preference", "material_trace"],
        )
        self.assertIn("include_explicit", retrieve["parameters"]["properties"])
        self.assertIn("kind_patterns", retrieve["parameters"]["properties"])

    def test_openai_strict_schema_makes_optional_fields_nullable_required(self) -> None:
        tools = build_native_memory_tool_specs(categories=("preference",), tool_format="openai")
        retrieve_schema = tools[0]["function"]["parameters"]

        self.assertEqual(retrieve_schema["required"], list(retrieve_schema["properties"]))
        self.assertEqual(retrieve_schema["properties"]["keywords"]["type"], ["array", "null"])
        self.assertEqual(retrieve_schema["properties"]["importance_min"]["type"], ["number", "null"])

        time_hint = retrieve_schema["properties"]["time_hint"]
        self.assertEqual(time_hint["type"], ["object", "null"])
        self.assertEqual(time_hint["required"], list(time_hint["properties"]))
        self.assertEqual(time_hint["properties"]["date_label"]["type"], ["string", "null"])

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
            config = SimpleNamespace(categories=("preference",))

            def retrieve_for_turn(self, **kwargs):
                seen.append(kwargs)
                return ["event result"]

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
            memory_metadata={"categories": ["preference"], "keywords": ["可乐"], "subject_scopes": ["user"]},
        )

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "可乐", "keywords": ["可乐"], "categories": ["preference"]},
            mem=mem,
            current=current,
        )

        self.assertTrue(out["ok"])
        blob = "\n".join(out["result"]["snippets"])
        self.assertIn("跨会话隐藏 raw 可乐", blob)
        self.assertNotIn("当前会话可见 raw 可乐", blob)
        store.close()

    def test_retrieve_rejects_invalid_category_instead_of_broad_searching(self) -> None:
        mem, store, _index, _emb = _shared_mem()
        current = mem.record_user_turn("我喜欢可乐", timestamp=1000, source_id="cur")

        out = dispatch_native_memory_tool(
            "retrieve_for_turn",
            {"query": "可乐", "categories": ["not_a_category"]},
            mem=mem,
            current=current,
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "invalid_arguments")
        self.assertIn("invalid_categories", out["reason"])
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
