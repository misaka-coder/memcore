"""切片 5 单测:读侧 router → 检索 → verifier → build_prompt_context。

canned LLM 用列表形式返回 NDJSON 事件(parse_ndjson 接受列表)。不依赖真实模型/网络。
"""

from __future__ import annotations

import unittest

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    MemoryConfig,
    MemorySystem,
    Namespace,
    NamespaceError,
    SQLiteMemoryStore,
)
from memcore.llm.base import LLMClient, LLMRequest, LLMResult, TaskType


class ReadLLM(LLMClient):
    """router 恒判 need_retrieval=true;verifier 按 verifier_match 决定 match/mismatch。"""

    def __init__(self, *, verifier_match: bool = True, keywords: list[str] | None = None) -> None:
        self.verifier_match = verifier_match
        self.keywords = keywords or ["可乐"]

    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.ROUTER:
            return LLMResult(
                ok=True,
                data=[
                    {"type": "decision", "need_retrieval": True},
                    {
                        "type": "query",
                        "rewritten_query": " ".join(self.keywords),
                        "keywords": self.keywords,
                        "time_hint": None,
                    },
                ],
            )
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


class ExplicitRetrieve(unittest.TestCase):
    def test_retrieve_finds_indexed_memory(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1")
        mem.record_user_turn("我最喜欢喝可乐", timestamp=1000)
        hits = mem.retrieve("可乐", keywords=["可乐"])
        self.assertTrue(any("可乐" in s for s in hits))
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
    def test_cross_conversation_retrieval_but_visible_is_per_conversation(self) -> None:
        store, index, emb = _shared_backends()
        # c2 里存一条"可乐"记忆
        mem_c2 = _mem(store, index, emb, conversation="c2")
        mem_c2.record_user_turn("我最喜欢喝可乐", timestamp=900)
        # c1 当前问"之前爱喝什么";c1 可见层不含 c2,但检索能跨会话捞到
        mem_c1 = _mem(store, index, emb, conversation="c1")
        cur = mem_c1.record_user_turn("我之前说过爱喝什么", timestamp=1000)
        ctx = mem_c1.build_prompt_context(current=cur)
        self.assertTrue(ctx["router"].need_retrieval)
        self.assertTrue(any("可乐" in s for s in ctx["retrieved_snippets"]))
        # 可见 raw 只含当前会话 c1 的消息,不含 c2
        self.assertTrue(all(r["conversation_id"] == "c1" for r in ctx["raw"]))
        store.close()

    def test_router_gate_off_skips_retrieval(self) -> None:
        store, index, emb = _shared_backends()
        cfg = MemoryConfig(enable_pre_retrieval=False)
        mem = _mem(store, index, emb, conversation="c1", config=cfg)
        mem.record_user_turn("我最喜欢喝可乐", timestamp=900)
        cur = mem.record_user_turn("随便聊聊", timestamp=1000)
        ctx = mem.build_prompt_context(current=cur)
        self.assertFalse(ctx["router"].need_retrieval)
        self.assertEqual(ctx["retrieved_snippets"], [])
        store.close()

    def test_current_message_excluded_from_retrieval(self) -> None:
        store, index, emb = _shared_backends()
        mem = _mem(store, index, emb, conversation="c1")
        cur = mem.record_user_turn("我最喜欢喝可乐", timestamp=1000)
        ctx = mem.build_prompt_context(current=cur)
        # 当前这条已在可见层,不该又作为"检索回来的"重复出现
        self.assertFalse(any(s.count("我最喜欢喝可乐") > 0 and "原始对话片段" in s for s in ctx["retrieved_snippets"]))
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
