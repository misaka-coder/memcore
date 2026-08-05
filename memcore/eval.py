"""memcore 评测台 —— 端到端验证移植后的承重行为(设计文档 §14 可量化验收)。

默认用 hashed embedding(无语义),确定性验证:管线闭环、硬隔离零泄漏、时间过滤、精度放宽、可见排除。
真正的同义词语义召回需 BGE-M3,可把 embedding 换成真实模型复跑同一套用例拿召回数。

跑法:`python -m memcore.eval`(打印报告)或在测试里调 run_eval 断言阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import MemoryConfig
from .embedding.hashed import HashedEmbeddingProvider
from .index.memory_index import InMemoryVectorIndex
from .llm.base import LLMClient, LLMRequest, LLMResult
from .memory_system import MemorySystem
from .namespace import Namespace
from .store.sqlite_store import SQLiteMemoryStore


class EvalLLM(LLMClient):
    """确定性评测用:读侧不调用模型，压缩任务在这组用例里也不会触发。"""

    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data={})


@dataclass
class SeedTurn:
    namespace: Namespace
    content: str
    timestamp: int


@dataclass
class EvalCase:
    name: str
    namespace: Namespace
    query: str
    entity_anchors: list[str] = field(default_factory=list)
    time_hint: dict[str, Any] | None = None
    expect_substrings: list[str] = field(default_factory=list)  # 必须出现在检索结果
    forbid_substrings: list[str] = field(default_factory=list)  # 绝不能出现(隔离/时间红线)


@dataclass
class EvalReport:
    total: int
    passed: int
    hit_count: int
    expect_total: int
    leak_count: int
    failures: list[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.expect_total if self.expect_total else 1.0

    @property
    def ok(self) -> bool:
        # 验收闸:全部用例通过 + 零隔离泄漏。
        return self.passed == self.total and self.leak_count == 0

    def format(self) -> str:
        lines = [
            "=== memcore eval report ===",
            f"cases: {self.passed}/{self.total} passed",
            f"hit@k: {self.hit_count}/{self.expect_total} ({self.hit_rate:.0%})",
            f"isolation leaks: {self.leak_count} (must be 0)",
            f"verdict: {'PASS' if self.ok else 'FAIL'}",
        ]
        lines.extend(f"  - FAIL: {f}" for f in self.failures)
        return "\n".join(lines)


def run_eval(
    *,
    seed: list[SeedTurn],
    cases: list[EvalCase],
    config: MemoryConfig | None = None,
    timezone: str = "Asia/Shanghai",
) -> EvalReport:
    emb = HashedEmbeddingProvider()
    store = SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)

    def mem_for(ns: Namespace) -> MemorySystem:
        return MemorySystem(
            llm=EvalLLM(),
            namespace=ns,
            timezone=timezone,
            store=store,
            index=index,
            embedding=emb,
            config=config or MemoryConfig(),
        )

    try:
        for turn in seed:
            mem_for(turn.namespace).record_user_turn(turn.content, timestamp=turn.timestamp)

        passed = hit_count = expect_total = leak_count = 0
        failures: list[str] = []
        for case in cases:
            mem = mem_for(case.namespace)
            snippets = mem.retrieve(case.query, entity_anchors=case.entity_anchors, time_hint=case.time_hint)
            blob = "\n".join(snippets)

            case_ok = True
            expect_total += len(case.expect_substrings)
            for want in case.expect_substrings:
                if want in blob:
                    hit_count += 1
                else:
                    case_ok = False
                    failures.append(f"{case.name}: missing expected {want!r}")
            for forbid in case.forbid_substrings:
                if forbid in blob:
                    leak_count += 1
                    case_ok = False
                    failures.append(f"{case.name}: LEAKED forbidden {forbid!r}")
            passed += int(case_ok)

        return EvalReport(
            total=len(cases),
            passed=passed,
            hit_count=hit_count,
            expect_total=expect_total,
            leak_count=leak_count,
            failures=failures,
        )
    finally:
        store.close()


def default_dataset() -> tuple[list[SeedTurn], list[EvalCase]]:
    """真实形态的小数据集:偏好回忆 / 硬隔离 / 跨会话 / 时间过滤。"""
    u1_c1 = Namespace(user_id="u1", conversation_id="c1")
    u1_c2 = Namespace(user_id="u1", conversation_id="c2")
    u2_c1 = Namespace(user_id="u2", conversation_id="c1")

    seed = [
        SeedTurn(u1_c2, "我最喜欢喝可乐", 1_700_000_000),  # u1 旧会话偏好
        SeedTurn(u2_c1, "我特别讨厌可乐这种东西", 1_700_000_000),  # u2 的,绝不能串给 u1
        SeedTurn(u1_c1, "上周我们去爬山看了日出", 1_712_000_000),  # 2024-04-02 前后
        SeedTurn(u1_c1, "今天加班到很晚很累", 1_713_200_000),  # 另一天
    ]
    cases = [
        EvalCase(
            name="preference_recall_cross_conversation",
            namespace=u1_c1,
            query="我之前说过最喜欢喝可乐吗",
            entity_anchors=["可乐"],
            expect_substrings=["可乐"],
            forbid_substrings=["讨厌"],  # u2 的"讨厌可乐"不能泄漏
        ),
        EvalCase(
            name="hard_isolation_u2_not_leaked",
            namespace=u2_c1,
            query="我对可乐什么态度",
            entity_anchors=["可乐"],
            expect_substrings=["讨厌"],  # u2 自己的能看到
            forbid_substrings=["最喜欢喝可乐"],  # u1 的不能串过来
        ),
        EvalCase(
            name="explicit_tool_retrieve_cross_conversation",
            namespace=u1_c1,
            query="还记得我喜欢喝可乐吗",
            entity_anchors=["可乐"],
            expect_substrings=["可乐"],
        ),
    ]
    return seed, cases


def main() -> int:
    seed, cases = default_dataset()
    report = run_eval(seed=seed, cases=cases)
    print(report.format())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
