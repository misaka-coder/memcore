"""importance 衰减:开启后长期记忆可见窗口按"随时间衰减的重要度"排序,而非纯 recency。"""

from __future__ import annotations

import unittest

from memcore import HashedEmbeddingProvider, MemoryConfig, MemorySystem, Namespace, SQLiteMemoryStore
from memcore.config import ConfigError
from memcore.decay import decayed_importance
from memcore.index.memory_index import InMemoryVectorIndex
from memcore.llm.base import LLMClient, LLMRequest, LLMResult

_DAY = 86400


class _StubLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


class DecayFunction(unittest.TestCase):
    def test_halves_each_half_life(self) -> None:
        self.assertAlmostEqual(decayed_importance(0.8, age_seconds=90 * _DAY, half_life_days=90), 0.4, places=4)
        self.assertAlmostEqual(decayed_importance(0.8, age_seconds=180 * _DAY, half_life_days=90), 0.2, places=4)

    def test_no_decay_when_fresh(self) -> None:
        self.assertEqual(decayed_importance(0.8, age_seconds=0, half_life_days=90), 0.8)


class VisibleWindowDecay(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.emb = HashedEmbeddingProvider()
        self.index = InMemoryVectorIndex(embedding=self.emb)
        self.ns = Namespace(user_id="u1", conversation_id="c1")
        self.now = 1_800_000_000
        # A:重要但 30 天没强化;B:不太重要但刚强化。
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={
                "semantic_id": "A",
                "timestamp": self.now - 30 * _DAY,
                "last_reinforced_ts": self.now - 30 * _DAY,
                "importance": 0.9,
                "semantic_summary": "A 重要但旧",
            },
        )
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={
                "semantic_id": "B",
                "timestamp": self.now,
                "last_reinforced_ts": self.now,
                "importance": 0.5,
                "semantic_summary": "B 一般但新",
            },
        )

    def tearDown(self) -> None:
        self.store.close()

    def _semantic_order(self, *, decay: bool) -> list[str]:
        mem = MemorySystem(
            llm=_StubLLM(),
            namespace=self.ns,
            timezone="Asia/Shanghai",
            store=self.store,
            index=self.index,
            embedding=self.emb,
            config=MemoryConfig(
                enable_pre_retrieval=False, enable_importance_decay=decay, importance_half_life_days=90
            ),
        )
        ctx = mem.build_prompt_context(current={"content": "随便说", "timestamp": self.now})
        return [s["semantic_id"] for s in ctx["semantic"]]

    def test_default_is_recency_order(self) -> None:
        # 不衰减:纯 last_reinforced 新者在前 → B 先。
        self.assertEqual(self._semantic_order(decay=False), ["B", "A"])

    def test_decay_promotes_important_recent_enough(self) -> None:
        # 衰减:A 0.9*0.5^(30/90)=~0.71 > B 0.5 → A 先,体现"重要度按新鲜度加权"。
        self.assertEqual(self._semantic_order(decay=True), ["A", "B"])

    def test_decay_ranks_full_candidate_set_no_recency_truncation(self) -> None:
        # 复现 review:很多条"新但低重要度" + 一条"旧但高重要度"。
        # 旧高价值的那条即使在 recency 窗口之外,衰减后仍应能进可见窗口(不被静默截断)。
        for i in range(12):  # 远超 semantic_visible_limit,且都比 HIGH_OLD 新
            self.store.add_semantic_summary(
                namespace=self.ns,
                record={
                    "semantic_id": f"low{i}",
                    "timestamp": self.now - i,
                    "last_reinforced_ts": self.now - i,
                    "importance": 0.1,
                    "semantic_summary": f"低{i}",
                },
            )
        self.store.add_semantic_summary(
            namespace=self.ns,
            record={
                "semantic_id": "HIGH_OLD",
                "timestamp": self.now - 20 * _DAY,
                "last_reinforced_ts": self.now - 20 * _DAY,
                "importance": 0.95,
                "semantic_summary": "旧但很重要",
            },
        )
        order = self._semantic_order(decay=True)
        # HIGH_OLD 衰减后 0.95*0.5^(20/90)≈0.81,远高于 0.1 → 必须出现在可见窗口里
        self.assertIn("HIGH_OLD", order)


class ConfigValidation(unittest.TestCase):
    def test_half_life_must_be_positive(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(importance_half_life_days=0)


if __name__ == "__main__":
    unittest.main()
