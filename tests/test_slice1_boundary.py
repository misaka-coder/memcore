"""切片 1 边界单测:契约、校验、命名空间、门面装配都站得住。

不依赖存储/模型/网络。验证设计文档里的几条焊死不变量真的生效。
"""

from __future__ import annotations

import unittest

from memcore import (
    Actor,
    ConfigError,
    HashedEmbeddingProvider,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    NamespaceError,
    SchemaError,
    TokenCounter,
    build_memory_metadata_instruction,
    coerce_memory_metadata,
)
from memcore.schema import ABOUT_ROLES, MEMORY_FACETS, SEMANTIC_REQUIRED_FIELDS, require_fields


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


class _StubTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(text)


class ConfigInvariants(unittest.TestCase):
    def test_default_config_valid(self) -> None:
        cfg = MemoryConfig()
        self.assertGreater(cfg.raw_token_trigger, 0)
        self.assertGreater(cfg.raw_token_batch_ratio, 0)
        self.assertLess(cfg.raw_token_batch_ratio, 1)
        self.assertEqual(cfg.retrieval_result_token_budget, 0)
        self.assertIn("knowledge", MEMORY_FACETS)
        self.assertEqual(ABOUT_ROLES, ("user", "assistant", "third_party", "external"))

    def test_retrieval_result_budget_rejects_negative_values(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(retrieval_result_token_budget=-1)

    def test_native_timeline_budget_must_be_positive(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(native_timeline_page_token_budget=0)

    def test_episodic_differential_enforced(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(episodic_compact_trigger_count=5, episodic_compact_batch_size=5)

    def test_positive_windows(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(semantic_visible_limit=0)

    def test_raw_token_batch_ratio_bounds(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_token_batch_ratio=1)
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_token_batch_ratio=0)


class MetadataCoercion(unittest.TestCase):
    def test_model_instruction_is_concise_and_preserves_field_semantics(self) -> None:
        instruction = build_memory_metadata_instruction(enable_flavor=True)

        self.assertIn("本轮记忆目标，不描述回复", instruction)
        self.assertIn("目标在查询历史时填 memory_query", instruction)
        self.assertIn("不是发言参与者", instruction)
        self.assertIn("未知答案和宽泛上位词不填", instruction)
        self.assertIn("优先保留具体、便于检索的名词和主题短语", instruction)
        self.assertIn("关键动作、关系、属性", instruction)
        self.assertIn("两组词去重", instruction)
        self.assertIn("少量补充有依据的同义词或上位词", instruction)
        self.assertIn("情感余温", instruction)
        self.assertNotIn("优先考虑未来正常聊天", instruction)

    def test_enum_filtering_and_string_deduplication(self) -> None:
        meta = coerce_memory_metadata(
            {
                "turn_intent": "memory_query",
                "memory_facets": ["preference", "not_a_facet"],
                "about_roles": ["user", "bogus"],
                "entity_anchors": ["可乐", "可乐", "饮料"],
                "topic_terms": ["喜欢", "饮用习惯"],
                "retrieval_priority": "critical",
                "mood_tags": ["warm"],
            },
            enable_flavor=True,
        )
        self.assertEqual(meta.turn_intent, "memory_query")
        self.assertEqual(meta.memory_facets, ["preference"])
        self.assertEqual(meta.about_roles, ["user"])
        self.assertEqual(meta.entity_anchors, ["可乐", "饮料"])
        self.assertEqual(meta.topic_terms, ["喜欢", "饮用习惯"])
        self.assertEqual(meta.retrieval_priority, "critical")
        self.assertEqual(meta.mood_tags, ["warm"])

    def test_legacy_fields_are_not_a_second_authority(self) -> None:
        meta = coerce_memory_metadata({"keywords": ["旧字段"], "categories": ["preference"], "confidence": 1.0})
        self.assertEqual(meta.entity_anchors, [])
        self.assertEqual(meta.memory_facets, [])

    def test_flavor_off_strips_mood(self) -> None:
        meta = coerce_memory_metadata({"mood_tags": ["warm", "sad"]}, enable_flavor=False)
        self.assertEqual(meta.mood_tags, [])

    def test_require_fields_raises_on_missing(self) -> None:
        with self.assertRaises(SchemaError):
            require_fields({"semantic_summary": "x"}, SEMANTIC_REQUIRED_FIELDS, context="semantic")


class NamespaceModel(unittest.TestCase):
    def test_user_id_required(self) -> None:
        with self.assertRaises(NamespaceError):
            Namespace(user_id="")

    def test_hard_key_excludes_conversation_and_actor(self) -> None:
        ns = Namespace(user_id="u1", tenant_id="t1", domain_id="fin", conversation_id="c1")
        self.assertEqual(ns.hard_key(), ("t1", "u1", "fin"))

    def test_actor_requires_stable_id(self) -> None:
        with self.assertRaises(NamespaceError):
            Actor(stable_id="")

    def test_rename_keeps_stable_id(self) -> None:
        a = Actor(stable_id="qq-12345", display_name="张三")
        renamed = a.with_display_name("李四")
        self.assertEqual(renamed.stable_id, "qq-12345")
        self.assertEqual(renamed.display_name, "李四")


class FacadeShell(unittest.TestCase):
    def _mk(self) -> MemorySystem:
        return MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
        )

    def test_construct_validates_deps(self) -> None:
        mem = self._mk()
        self.assertEqual(mem.timezone, "Asia/Shanghai")
        self.assertIsInstance(mem.config, MemoryConfig)

    def test_timezone_required(self) -> None:
        with self.assertRaises(ValueError):
            MemorySystem(llm=_StubLLM(), namespace=Namespace(user_id="u1"), timezone="")

    def test_llm_must_be_client(self) -> None:
        with self.assertRaises(TypeError):
            MemorySystem(llm=object(), namespace=Namespace(user_id="u1"), timezone="UTC")  # type: ignore[arg-type]

    def test_construct_wires_full_pipeline(self) -> None:
        # 切片 5 后读写侧均已接通;构造即装配完整管线。
        mem = self._mk()
        self.assertIsNotNone(mem.store)
        self.assertIsNotNone(mem.index)

    def test_missing_token_counter_uses_explicit_estimated_mode(self) -> None:
        mem = MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
        )
        self.assertIsNone(mem.token_counter)

    def test_token_counter_must_match_interface(self) -> None:
        with self.assertRaises(TypeError):
            MemorySystem(
                llm=_StubLLM(),
                namespace=Namespace(user_id="u1"),
                timezone="Asia/Shanghai",
                embedding=HashedEmbeddingProvider(),
                token_counter=object(),
            )

    def test_token_policy_accepts_token_counter(self) -> None:
        mem = MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
            token_counter=_StubTokenCounter(),
        )
        self.assertIsInstance(mem.token_counter, TokenCounter)


if __name__ == "__main__":
    unittest.main()
