"""Isolated read-tool benchmark; synthetic data and local hashed embeddings only.

Run: uv run python examples/benchmark_memory_reads.py --output benchmark.json
Numbers exclude setup/indexing, network embeddings, and host/model latency.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import statistics
import time
from pathlib import Path

from memcore import (
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
)


class _NoCalls(LLMClient):
    def call(self, request):
        raise AssertionError("Read benchmark must not call an LLM")


def run(*, repeats: int, profile: str = "") -> dict:
    store = SQLiteMemoryStore(":memory:")
    embedding = HashedEmbeddingProvider()
    namespace = Namespace(user_id="synthetic-benchmark", conversation_id="test")
    mem = MemorySystem(
        namespace=namespace,
        timezone="Asia/Shanghai",
        llm=_NoCalls(),
        store=store,
        embedding=embedding,
        index=InMemoryVectorIndex(embedding=embedding),
        config=MemoryConfig(),
    )
    try:
        for episode in range(100):
            ids = []
            for offset in range(20):
                number = episode * 20 + offset
                sid = f"r{number}"
                ids.append(sid)
                store.add_message(
                    namespace=namespace,
                    source_id=sid,
                    role="user",
                    content=f"项目{episode} 文旅答辩 字幕音画同步。" + "合成测试正文，计划讨论细节。" * 100,
                    timestamp=1_700_000_000 + number,
                    memory_metadata={"topic_terms": ["文旅答辩项目", f"项目{episode}"]},
                )
            store.add_summary(
                namespace=namespace,
                record={
                    "summary_id": f"e{episode}",
                    "timestamp": 1_700_000_000 + episode * 20,
                    "diary_summary": f"项目{episode} 的文旅答辩，讨论字幕音画同步。",
                    "memory_title": f"项目{episode}",
                    "catalog_hint": "项目方案与配音",
                    "source_ids": ids,
                    "is_semanticized": 1,
                    "memory_metadata": {"topic_terms": ["文旅答辩项目", "字幕"]},
                },
            )
            store.mark_messages_summarized(ids, f"e{episode}")
        for group in range(5):
            store.add_semantic_summary(
                namespace=namespace,
                record={
                    "semantic_id": f"s{group}",
                    "timestamp": 1_700_002_000 + group,
                    "semantic_summary": "文旅项目持续讨论字幕方案。",
                    "source_summary_ids": [f"e{i}" for i in range(group * 20, group * 20 + 20)],
                    "memory_metadata": {"topic_terms": ["文旅答辩项目"]},
                },
            )
        mem.reindex_all()
        current = {"source_id": "current", "timestamp": 1_700_003_000}
        cases = {
            "browse_keywords": lambda: mem.browse_memory(keywords=["文旅答辩项目"]),
            "browse_date": lambda: mem.browse_memory(date_from="2023-11-15"),
            "open_batch_50": lambda: mem.open_memory(memory_ids=[f"e{i}" for i in range(50)], view="content"),
            "open_semantic_sources": lambda: mem.open_memory(memory_id="s0", view="sources"),
            "timeline_anchor": lambda: mem.read_timeline(anchor_source_id="r1000", before_turns=2, after_turns=2),
            "timeline_date_page": lambda: mem.read_timeline(date_from="2023-11-15", page_token_budget=4000),
            "retrieve_for_turn": lambda: mem.retrieve_for_turn_structured(current=current, query="字幕音画同步"),
        }
        report = {"fixture": {"raw": 2000, "episodic": 100, "semantic": 5}, "repeats": repeats, "cases": {}}
        profiler = cProfile.Profile() if profile else None
        for name, call in cases.items():
            statements = []
            store._conn.set_trace_callback(statements.append)
            try:
                result = call()
            finally:
                store._conn.set_trace_callback(None)
            status = result.get("status") if isinstance(result, dict) else result.status
            if status not in {"ok", "partial", "found"}:
                raise AssertionError((name, status))
            durations = []
            if profiler:
                profiler.enable()
            for _ in range(repeats):
                started = time.perf_counter()
                call()
                durations.append((time.perf_counter() - started) * 1000)
            if profiler:
                profiler.disable()
            report["cases"][name] = {
                "median_ms": round(statistics.median(durations), 3),
                "select_queries": sum(s.lstrip().upper().startswith("SELECT") for s in statements),
            }
        if profiler:
            profiler.dump_stats(profile)
        return report
    finally:
        mem.close()
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="")
    parser.add_argument("--profile", default="")
    args = parser.parse_args()
    result = run(repeats=max(1, args.repeats), profile=args.profile)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
