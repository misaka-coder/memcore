"""切片 5 单测:读侧显式检索 → verifier → build_prompt_context 可见层。

canned LLM 用列表形式返回 NDJSON 事件(parse_ndjson 接受列表)。不依赖真实模型/网络。
"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from memcore import (
    Actor,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    MemoryConfig,
    MemorySystem,
    Namespace,
    NamespaceError,
    SQLiteMemoryStore,
    VectorIndex,
)
from memcore.index.entry_builder import build_raw_entry, build_semantic_entry, build_summary_entry
from memcore.index.metadata_filters import (
    INDEX_SCHEMA_KEY,
    INDEX_SCHEMA_VERSION,
    category_filter_key,
    subject_scope_filter_key,
)
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType


class ReadLLM(LLMClient):
    """verifier 按 verifier_match 决定 match/mismatch。"""

    def __init__(self, *, verifier_match: bool = True, keywords: list[str] | None = None) -> None:
        self.verifier_match = verifier_match
        self.keywords = keywords or ["可乐"]

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.VERIFIER:
            if self.verifier_match:
                return LLMResult(
                    ok=True,
                    data=[
                        {"type": "decision", "match_result": "match"},
                        {"type": "selection", "selected_indexes": [1]},
                    ],
                )
            return LLMResult(ok=True, data=[{"type": "decision", "match_result": "mismatch"}])
        return LLMResult(ok=True, data={})


def _shared_backends():
    emb = HashedEmbeddingProvider()
    return SQLiteMemoryStore(":memory:"), InMemoryVectorIndex(embedding=emb), emb


def _ts(y: int, mo: int, d: int, h: int, mi: int = 0, tz: str = "Asia/Shanghai") -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp())


def _mem(store, index, emb, *, conversation, llm=None, config=None):
    return MemorySystem(
        llm=llm or ReadLLM(),
        namespace=Namespace(user_id="u1", conversation_id=conversation),
        timezone="Asia/Shanghai",
        store=store,
        index=index,
        embedding=emb,
        config=config,
    )


class SpyIndex(VectorIndex):
    def __init__(self) -> None:
        self.semantic_wheres: list[dict] = []
        self.keyword_wheres: list[dict] = []

    def upsert(self, entries: list[dict]) -> None:
        return None

    def semantic_search(
        self, *, query_text: str, where: dict, n_results: int = 8, exclude_source_ids=None
    ) -> list[dict]:
        self.semantic_wheres.append(dict(where))
        return []

    def keyword_search(
        self, *, query_text: str, keywords: list[str], where: dict, n_results: int = 8, exclude_source_ids=None
    ) -> list[dict]:
        self.keyword_wheres.append(dict(where))
        return []

    def delete(self, source_ids: list[str]) -> None:
        return None

    def count(self) -> int:
        return 0


class ExplicitRetrieve(unittest.TestCase):
    def test_retrieve_finds_indexed_memory(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1")
        mem.record_user_turn("我最喜欢喝可乐", timestamp=1000)
        hits = mem.retrieve("可乐", keywords=["可乐"])
        self.assertTrue(any("可乐" in s for s in hits))
        store.close()

    def test_tool_trace_is_excluded_from_default_retrieve_but_explicitly_searchable(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", config=MemoryConfig(enable_verifier=False))
        mem.record_tool_exchange(
            tool_name="web_search",
            tool_input={"query": "北京天气"},
            result="北京今天 25 度晴天",
            timestamp=1000,
            source_id_prefix="tool1",
            keywords=["北京天气"],
        )

        self.assertEqual(mem.retrieve("北京天气", keywords=["北京天气"]), [])
        hits = mem.retrieve(
            "北京天气",
            keywords=["北京天气"],
            include_explicit=True,
            kind_patterns=["tool.web_search.*"],
        )
        self.assertTrue(any("25 度晴天" in s for s in hits))
        store.close()

    def test_event_trace_is_excluded_from_default_retrieve_but_explicitly_searchable(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", config=MemoryConfig(enable_verifier=False))
        mem.record_external_event(
            event_type="finance",
            source="public_news",
            fields={
                "published_at": "2026-07-20 14:29",
                "title": "虚构科创债事件",
                "summary": "只用于事件检索测试",
                "url": "https://example.com/news",
            },
            timestamp=1000,
            source_id="event1",
            keywords=["科创债"],
        )

        self.assertEqual(mem.retrieve("科创债", keywords=["科创债"]), [])
        hits = mem.retrieve(
            "科创债",
            keywords=["科创债"],
            include_explicit=True,
            kind_patterns=["event.finance.*"],
        )
        self.assertTrue(any("虚构科创债事件" in item for item in hits))
        store.close()

    def test_material_trace_is_excluded_from_default_retrieve_but_explicitly_searchable(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", config=MemoryConfig(enable_verifier=False))
        mem.record_material_reference(
            file_id="file_img_001",
            kind="image",
            filename="photo.jpg",
            mime_type="image/jpeg",
            file_status="ready",
            derived_status="ocr_ready",
            timestamp=1000,
            source_id="mat1",
            keywords=["题目图片"],
        )

        self.assertEqual(mem.retrieve("photo.jpg", keywords=["photo.jpg"]), [])
        hits = mem.retrieve(
            "photo.jpg",
            keywords=["photo.jpg"],
            include_explicit=True,
            kind_patterns=["material.*"],
        )
        self.assertTrue(any("file_img_001" in s and "photo.jpg" in s for s in hits))
        store.close()

    def test_verifier_mismatch_returns_nothing(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", llm=ReadLLM(verifier_match=False))
        mem.record_user_turn("我最喜欢喝可乐", timestamp=1000)
        self.assertEqual(mem.retrieve("可乐", keywords=["可乐"]), [])
        store.close()

    def test_verifier_disabled_passes_through(self) -> None:
        store, index, emb = _shared_backends()
        cfg = MemoryConfig(enable_verifier=False)
        mem = _mem(store, index, emb, conversation="c1", llm=ReadLLM(verifier_match=False), config=cfg)
        mem.record_user_turn("我最喜欢喝可乐", timestamp=1000)
        self.assertTrue(any("可乐" in s for s in mem.retrieve("可乐", keywords=["可乐"])))
        store.close()

    def test_index_safe_metadata_filters_are_pushed_into_where(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        index = SpyIndex()
        mem = MemorySystem(
            llm=ReadLLM(),
            namespace=Namespace(user_id="u1", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=store,
            index=index,
            embedding=HashedEmbeddingProvider(),
        )

        mem.retrieve(
            "风险偏好",
            source_layers=["summary"],
            importance_min=0.7,
            categories=["preference", "plan_goal"],
            subject_scopes=["user", "assistant"],
        )

        first = index.semantic_wheres[0]
        category_or = {"$or": [{category_filter_key("preference"): True}, {category_filter_key("plan_goal"): True}]}
        scope_or = {"$or": [{subject_scope_filter_key("user"): True}, {subject_scope_filter_key("assistant"): True}]}
        self.assertEqual(first["entry_type"], {"$in": ["summary"]})
        self.assertEqual(first["memory_importance"], {"$gte": 0.7})
        self.assertIn(category_or, first["$and"])
        self.assertIn(scope_or, first["$and"])
        self.assertNotIn("memory_categories_text", first)
        self.assertEqual(index.keyword_wheres[0], first)
        self.assertEqual(len(index.semantic_wheres), 4)
        self.assertNotIn("memory_importance", index.semantic_wheres[1])
        self.assertIn(category_or, index.semantic_wheres[1]["$and"])
        self.assertIn(scope_or, index.semantic_wheres[2]["$and"])
        self.assertNotIn(category_or, index.semantic_wheres[2]["$and"])
        self.assertNotIn(scope_or, index.semantic_wheres[3]["$and"])
        self.assertTrue(all("entry_type" in where for where in index.semantic_wheres))
        store.close()

    def test_entry_builder_writes_prefilter_flags_for_all_layers(self) -> None:
        base = {
            "tenant_id": "",
            "user_id": "u1",
            "domain_id": "",
            "conversation_id": "c1",
            "timestamp": 1000,
            "memory_metadata": {
                "categories": ["preference", "plan_goal"],
                "subject_scopes": ["user"],
                "importance": 0.8,
            },
        }
        entries = [
            build_raw_entry({**base, "source_id": "m1", "role": "user", "content": "我喜欢可乐"}),
            build_summary_entry({**base, "summary_id": "s1", "diary_summary": "用户喜欢可乐"}),
            build_semantic_entry({**base, "semantic_id": "sem1", "semantic_summary": "用户喜欢可乐"}),
        ]

        for entry in entries:
            metadata = entry["metadata"]
            self.assertTrue(metadata[category_filter_key("preference")])
            self.assertTrue(metadata[category_filter_key("plan_goal")])
            self.assertTrue(metadata[subject_scope_filter_key("user")])
            self.assertEqual(metadata["memory_importance"], 0.8)

    def test_retrieve_for_turn_excludes_visible_raw_and_context_neighbor(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", config=MemoryConfig(enable_verifier=False))
        mem.record_user_turn("当前会话可见 raw 可乐", timestamp=1000, source_id="visible")
        cur = mem.record_user_turn("我之前说过我喜欢喝可乐吗", timestamp=1001, source_id="cur")
        other = _mem(store, index, emb, conversation="c2", config=MemoryConfig(enable_verifier=False))
        other.record_user_turn("跨会话隐藏 raw 可乐", timestamp=900, source_id="hidden")

        hits = mem.retrieve_for_turn(current=cur, query="可乐", keywords=["可乐"])
        blob = "\n".join(hits)

        self.assertIn("跨会话隐藏 raw 可乐", blob)
        self.assertNotIn("当前会话可见 raw 可乐", blob)
        self.assertNotIn("我之前说过我喜欢喝可乐吗", blob)
        store.close()

    def test_retrieve_for_turn_excludes_visible_episodic_and_semantic(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", config=MemoryConfig(enable_verifier=False))
        cur = mem.record_user_turn("帮我查一下滑雪相关记忆", timestamp=2000, source_id="cur")
        other_ns = Namespace(user_id="u1", conversation_id="c2")

        visible_summary = store.add_summary(
            namespace=mem.namespace,
            record={
                "summary_id": "visible-summary",
                "timestamp": 1100,
                "diary_summary": "当前会话可见阶段摘要滑雪",
                "memory_metadata": {"keywords": ["滑雪"]},
            },
        )
        hidden_summary = store.add_summary(
            namespace=other_ns,
            record={
                "summary_id": "hidden-summary",
                "timestamp": 900,
                "diary_summary": "跨会话隐藏阶段摘要滑雪",
                "memory_metadata": {"keywords": ["滑雪"]},
            },
        )
        visible_semantic = store.add_semantic_summary(
            namespace=mem.namespace,
            record={
                "semantic_id": "visible-semantic",
                "timestamp": 1200,
                "last_reinforced_ts": 1200,
                "semantic_summary": "当前会话可见长期语义滑雪",
                "memory_metadata": {"keywords": ["滑雪"]},
            },
        )
        hidden_semantic = store.add_semantic_summary(
            namespace=other_ns,
            record={
                "semantic_id": "hidden-semantic",
                "timestamp": 950,
                "last_reinforced_ts": 950,
                "semantic_summary": "跨会话隐藏长期语义滑雪",
                "memory_metadata": {"keywords": ["滑雪"]},
            },
        )
        index.upsert(
            [
                build_summary_entry(visible_summary),
                build_summary_entry(hidden_summary),
                build_semantic_entry(visible_semantic),
                build_semantic_entry(hidden_semantic),
            ]
        )
        for source_id in ("visible-summary", "hidden-summary", "visible-semantic", "hidden-semantic"):
            store.set_index_state(
                source_id,
                "indexed",
                index_schema_version=INDEX_SCHEMA_VERSION,
                index_key=INDEX_SCHEMA_KEY,
            )

        hits = mem.retrieve_for_turn(current=cur, query="滑雪", keywords=["滑雪"])
        blob = "\n".join(hits)

        self.assertIn("跨会话隐藏阶段摘要滑雪", blob)
        self.assertIn("跨会话隐藏长期语义滑雪", blob)
        self.assertNotIn("当前会话可见阶段摘要滑雪", blob)
        self.assertNotIn("当前会话可见长期语义滑雪", blob)
        store.close()

    def test_update_turn_metadata_reindexes_raw_tags(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1", config=MemoryConfig(enable_verifier=False))
        rec = mem.record_user_turn("我最近在看一只波动很大的股票", timestamp=1000, source_id="m1")
        out = mem.update_turn_metadata(
            rec["source_id"],
            {
                "keywords": ["英伟达", "风险偏好"],
                "categories": ["preference"],
                "mood_tags": ["warm"],  # 默认 flavor 关,应被清空
                "importance": 0.9,
            },
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["status"], "updated")
        self.assertEqual(out["index_status"], "indexed")
        self.assertEqual(out["memory_metadata"]["mood_tags"], [])
        stored = store.get_record_by_source_id("m1")
        self.assertEqual(stored["memory_metadata"]["keywords"], ["英伟达", "风险偏好"])
        hits = mem.retrieve("英伟达", keywords=["英伟达"])
        self.assertTrue(any("波动很大的股票" in s for s in hits))  # 不是原文命中,是 metadata tag 命中
        store.close()

    def test_update_turn_metadata_not_found_is_structured(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1")
        out = mem.update_turn_metadata("missing", {"keywords": ["x"]})
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "not_found")
        self.assertEqual(out["reason"], "source_id_not_found_or_not_raw")
        store.close()

    def test_update_turn_metadata_accepts_matching_actor_owner(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="group-1", config=MemoryConfig(enable_verifier=False))
        actor = Actor(stable_id="qq:10001", display_name="张三")
        rec = mem.record_user_turn(
            "我最近更关注新能源板块。",
            actor=actor,
            timestamp=1000,
            source_id="actor-message-1",
        )

        out = mem.update_turn_metadata(
            rec["source_id"],
            {
                "keywords": ["新能源", "关注板块"],
                "categories": ["preference"],
                "subject_scopes": ["user"],
                "importance": 0.8,
            },
            actor=actor,
        )

        self.assertTrue(out["ok"])
        stored = store.get_record_by_source_id(rec["source_id"])
        self.assertEqual(stored["actor_id"], "qq:10001")
        self.assertEqual(stored["actor_display_name"], "张三")
        self.assertEqual(stored["memory_metadata"]["keywords"], ["新能源", "关注板块"])
        self.assertTrue(any("新能源板块" in item for item in mem.retrieve("新能源", keywords=["新能源"])))
        store.close()

    def test_update_turn_metadata_rejects_wrong_actor_owner(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="group-1")
        mem.record_user_turn(
            "这是张三的关注方向。",
            actor=Actor(stable_id="qq:10001", display_name="张三"),
            timestamp=1000,
            source_id="actor-message-1",
        )

        with self.assertRaises(NamespaceError):
            mem.update_turn_metadata(
                "actor-message-1",
                {"keywords": ["污染"]},
                actor=Actor(stable_id="qq:10002", display_name="李四"),
            )
        self.assertEqual(store.get_record_by_source_id("actor-message-1")["memory_metadata"]["keywords"], [])
        store.close()

    def test_update_turn_metadata_rejects_cross_namespace_source_id(self) -> None:
        store, index, emb = _shared_backends()
        mem_u1 = _mem(store, index, emb, conversation="c1")
        mem_u1.record_user_turn("u1 私密原文", timestamp=1000, source_id="s1")
        mem_u2 = MemorySystem(
            llm=ReadLLM(),
            namespace=Namespace(user_id="u2", conversation_id="c1"),
            timezone="Asia/Shanghai",
            store=store,
            index=index,
            embedding=emb,
        )

        with self.assertRaises(NamespaceError):
            mem_u2.update_turn_metadata("s1", {"keywords": ["污染"]})
        self.assertEqual(store.get_record_by_source_id("s1")["memory_metadata"]["keywords"], [])
        store.close()

    def test_update_turn_metadata_rejects_cross_conversation_source_id(self) -> None:
        store, index, emb = _shared_backends()
        mem_c1 = _mem(store, index, emb, conversation="c1")
        mem_c1.record_user_turn("c1 私密原文", timestamp=1000, source_id="s1")
        mem_c2 = _mem(store, index, emb, conversation="c2")

        with self.assertRaises(NamespaceError):
            mem_c2.update_turn_metadata("s1", {"keywords": ["污染"]})
        self.assertEqual(store.get_record_by_source_id("s1")["memory_metadata"]["keywords"], [])
        store.close()


class BuildContext(unittest.TestCase):
    def test_visible_context_is_per_conversation_and_retrieve_is_explicit(self) -> None:
        store, index, emb = _shared_backends()
        # c2 里存一条"可乐"记忆
        mem_c2 = _mem(store, index, emb, conversation="c2")
        mem_c2.record_user_turn("我最喜欢喝可乐", timestamp=_ts(2026, 4, 9, 9))
        # c1 当前问"之前爱喝什么";build_prompt_context 只给可见层,不做 router 自动检索
        mem_c1 = _mem(store, index, emb, conversation="c1")
        cur = mem_c1.record_user_turn("我之前说过爱喝什么", timestamp=_ts(2026, 4, 10, 9))
        ctx = mem_c1.build_prompt_context(current=cur)
        self.assertEqual(set(ctx), {"raw", "episodic", "semantic"})
        # 可见 raw 只含当前会话 c1 的消息,不含 c2
        self.assertTrue(all(r["conversation_id"] == "c1" for r in ctx["raw"]))
        rendered = mem_c1.render_prompt_context(ctx)
        self.assertIn("【近期原始对话(未摘要)】", rendered)
        self.assertIn("[日期 2026-04-10 周五]", rendered)
        self.assertIn("[09:00 | 上午] user: 我之前说过爱喝什么", rendered)
        # 聊天模型若需要旧记忆,应显式调用 retrieve 工具;检索仍按 hardkey 跨会话。
        hits = mem_c1.retrieve("可乐", keywords=["可乐"])
        self.assertTrue(any("可乐" in s for s in hits))
        store.close()


class Forgetting(unittest.TestCase):
    def test_facade_forget_namespace_clears_store_and_index(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1")
        mem.record_user_turn("我最喜欢喝可乐", timestamp=1000)
        self.assertEqual(index.count(), 1)

        result = mem.forget_namespace()
        self.assertTrue(result["ok"])
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(store.get_unsummarized_messages(namespace=mem.namespace), [])
        self.assertEqual(index.count(), 0)
        store.close()


if __name__ == "__main__":
    unittest.main()
