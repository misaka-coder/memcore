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
    coerce_memory_metadata,
)
from memcore.schema import SEMANTIC_REQUIRED_FIELDS, require_fields


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={}, attempts=1)


class _StubTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(text)


class ConfigInvariants(unittest.TestCase):
    def test_default_config_valid(self) -> None:
        cfg = MemoryConfig()
        self.assertLess(cfg.summary_batch_size, cfg.raw_trigger_count)
        self.assertIn("tool_trace", cfg.categories)
        self.assertIn("material_trace", cfg.categories)
        self.assertIn("tool_trace", cfg.raw_compaction_excluded_categories)
        self.assertIn("material_trace", cfg.raw_compaction_excluded_categories)
        self.assertIn("tool_trace", cfg.retrieval_default_excluded_categories)
        self.assertIn("material_trace", cfg.retrieval_default_excluded_categories)

    def test_differential_relationship_enforced(self) -> None:
        # 批量 >= 触发数 必须被拒绝(防层间记忆重叠的承重约束)。
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_trigger_count=20, summary_batch_size=20)

    def test_episodic_differential_enforced(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(episodic_compact_trigger_count=5, episodic_compact_batch_size=5)

    def test_positive_windows(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(semantic_visible_limit=0)

    def test_categories_must_be_nonempty_enum(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(categories=())

    def test_raw_compaction_policy_enum(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_compaction_policy="magic")

    def test_raw_token_batch_ratio_bounds(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_token_batch_ratio=1)
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_token_batch_ratio=0)

    def test_excluded_category_configs_must_be_tuples(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(raw_compaction_excluded_categories=["tool_trace"])  # type: ignore[arg-type]
        with self.assertRaises(ConfigError):
            MemoryConfig(retrieval_default_excluded_categories=["tool_trace"])  # type: ignore[arg-type]


class MetadataCoercion(unittest.TestCase):
    def test_enum_filtering_and_clamping(self) -> None:
        meta = coerce_memory_metadata(
            {
                "keywords": ["可乐", "饮料", "可乐", "a", "b", "c"],  # 去重 + 截到 4
                "subject_scopes": ["user", "bogus"],  # bogus 丢弃
                "categories": ["preference", "not_a_category"],  # 非法丢弃
                "mood_tags": ["warm"],
                "importance": 5,  # clamp 到 1.0
                "confidence": -3,  # clamp 到 0.0
            },
            enable_flavor=True,
        )
        self.assertEqual(meta.subject_scopes, ["user"])
        self.assertEqual(meta.categories, ["preference"])
        self.assertLessEqual(len(meta.keywords), 4)
        self.assertEqual(meta.importance, 1.0)
        self.assertEqual(meta.confidence, 0.0)
        self.assertEqual(meta.mood_tags, ["warm"])

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

    def test_token_policy_requires_token_counter(self) -> None:
        with self.assertRaises(ConfigError):
            MemorySystem(
                llm=_StubLLM(),
                namespace=Namespace(user_id="u1"),
                timezone="Asia/Shanghai",
                embedding=HashedEmbeddingProvider(),
                config=MemoryConfig(raw_compaction_policy="token"),
            )

    def test_token_counter_must_match_interface(self) -> None:
        with self.assertRaises(TypeError):
            MemorySystem(
                llm=_StubLLM(),
                namespace=Namespace(user_id="u1"),
                timezone="Asia/Shanghai",
                embedding=HashedEmbeddingProvider(),
                config=MemoryConfig(raw_compaction_policy="token"),
                token_counter=object(),
            )

    def test_token_policy_accepts_token_counter(self) -> None:
        mem = MemorySystem(
            llm=_StubLLM(),
            namespace=Namespace(user_id="u1"),
            timezone="Asia/Shanghai",
            embedding=HashedEmbeddingProvider(),
            config=MemoryConfig(raw_compaction_policy="token"),
            token_counter=_StubTokenCounter(),
        )
        self.assertIsInstance(mem.token_counter, TokenCounter)


if __name__ == "__main__":
    unittest.main()
