"""Slice 4: token accounting 反映 settled + 安全审计指标 + 回放基线与 digest 完整性。

覆盖文档 §11(压缩计数用实际投影的 settled 历史)、§13(安全 metrics 不含正文)、
§14.5(缓存前缀稳定/压缩触发延迟)、§14.6(operation digest 仍从完整真相构造)。
"""

from __future__ import annotations

import unittest

from memcore import (
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemoryConfig,
    MemorySystem,
    Namespace,
    SQLiteMemoryStore,
    TaskType,
    TimelineEntryInput,
    TurnRole,
)


class _SummaryLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        if request.task_type == TaskType.SUMMARY:
            return LLMResult(
                ok=True,
                data={
                    "diary_summary": "带工具调用的对话",
                    "importance": 0.5,
                    "key_events": [],
                    "core_facts": [],
                    "memory_title": "",
                    "catalog_hint": "",
                    "topic_headings": [],
                },
                attempts=1,
            )
        return LLMResult(ok=True, data={}, attempts=1)


def _shared_mem(policy: str = "compact_after_terminal", **config_kwargs):
    emb = HashedEmbeddingProvider()
    store = SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    mem = MemorySystem(
        llm=_SummaryLLM(),
        namespace=Namespace(user_id="u1", conversation_id="c1"),
        timezone="Asia/Shanghai",
        store=store,
        index=index,
        embedding=emb,
        config=MemoryConfig(operation_projection_policy=policy, **config_kwargs),
    )
    return mem, store


def _begin(mem: MemorySystem, turn_id: str) -> None:
    mem.begin_turn(
        turn_id=turn_id,
        opened_at=1000,
        stimuli=[
            TimelineEntryInput(
                source_id=f"{turn_id}-u",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text="问题",
                payload={"text": "问题"},
                timestamp=1000,
                compatibility_role="user",
            )
        ],
    )


def _add_tool(mem: MemorySystem, turn_id: str, call_id: str, result: str) -> str:
    exchange = mem.record_tool_exchange(
        turn_id=turn_id,
        tool_name="web_search",
        tool_call_id=call_id,
        tool_input={"query": "q"},
        result=result,
        source="web",
        source_id_prefix=f"tooltrace:{turn_id}-{call_id}",
    )
    return exchange["tool_result"]["source_id"]


def _complete(mem: MemorySystem, turn_id: str, profile: str = "openai_chat") -> None:
    mem.build_context_projection(provider_profile=profile)
    result = mem.complete_turn(
        turn_id=turn_id,
        semantic_text="ok",
        provider_output_raw="ok",
        provider_profile=profile,
        provider_projection={"role": "assistant", "content": "ok"},
    )
    assert result.completed


LONG_BODY = "网页正文 " + "记忆引擎上下文管理策略说明内容填充。" * 200


class TokenAccountingSettledTests(unittest.TestCase):
    def _fill(self, mem: MemorySystem, turns: int) -> None:
        for index in range(turns):
            _begin(mem, f"t{index}")
            _add_tool(mem, f"t{index}", f"c{index}", LONG_BODY)
            _complete(mem, f"t{index}")

    def test_compaction_counts_settled_tokens_not_full_raw(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal", raw_token_trigger=1)
        self._fill(mem, 3)
        compact_out = mem.compact_due_sync(provider_profile="openai_chat")

        full_mem, full_store = _shared_mem(policy="full_until_raw_compaction", raw_token_trigger=1)
        self._fill(full_mem, 3)
        full_out = full_mem.compact_due_sync(provider_profile="openai_chat")

        self.assertGreater(full_out["before_raw_projected_tokens"], 0)
        self.assertLess(compact_out["before_raw_projected_tokens"], full_out["before_raw_projected_tokens"])
        store.close()
        full_store.close()


class SettlementMetricsTests(unittest.TestCase):
    def test_metrics_are_structured_safe_and_include_savings(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        _begin(mem, "t1")
        _add_tool(mem, "t1", "c1", LONG_BODY)
        _complete(mem, "t1")
        metrics = mem.settlement_metrics()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric["settlement_status"], "settled")
        self.assertEqual(metric["operation_projection_policy"], "compact_after_terminal")
        self.assertEqual(len(metric["turn_id_hash"]), 16)
        self.assertGreater(metric["full_projected_tokens"], metric["settled_projected_tokens"])
        self.assertGreater(metric["saved_projected_tokens"], 0)
        self.assertGreater(metric["saved_ratio"], 0)
        self.assertEqual(metric["full_projection_hash"], metric["full_projection_hash"])
        self.assertEqual(metric["token_count_quality"], "estimated")
        self.assertEqual(metric["fallback_reason"], "")
        store.close()

    def test_metrics_never_include_prompt_body(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        _begin(mem, "t1")
        _add_tool(mem, "t1", "c1", LONG_BODY)
        _complete(mem, "t1")
        serialized = str(mem.settlement_metrics())
        self.assertNotIn(LONG_BODY, serialized)
        self.assertNotIn("content", serialized)
        store.close()

    def test_full_policy_has_no_settlement_metrics(self) -> None:
        mem, store = _shared_mem(policy="full_until_raw_compaction")
        _begin(mem, "t1")
        _add_tool(mem, "t1", "c1", LONG_BODY)
        _complete(mem, "t1")
        self.assertEqual(mem.settlement_metrics(), [])
        store.close()

    def test_settled_noop_metrics_report_no_savings(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        _begin(mem, "t1")
        _add_tool(mem, "t1", "c1", "音乐已暂停")
        _complete(mem, "t1")
        metrics = mem.settlement_metrics()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["settlement_status"], "settled_noop")
        self.assertEqual(metrics[0]["saved_projected_tokens"], 0)
        self.assertEqual(metrics[0]["saved_ratio"], 0.0)
        store.close()


class ReadbackBaselineTests(unittest.TestCase):
    def test_settled_history_stabilizes_prefix_after_compaction(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        _begin(mem, "t1")
        _add_tool(mem, "t1", "c1", LONG_BODY)
        _complete(mem, "t1")
        first = mem.build_context_projection(provider_profile="openai_chat")
        mem.compact_due_sync(provider_profile="openai_chat")
        second = mem.build_context_projection(provider_profile="openai_chat")
        self.assertEqual(first.stable_prefix_hash, second.stable_prefix_hash)
        store.close()

    def test_operation_digest_builds_from_full_truth_not_settled(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal", raw_token_trigger=1)
        _begin(mem, "t1")
        use_sid = _add_tool(mem, "t1", "c1", LONG_BODY)
        _complete(mem, "t1")
        out = mem.compact_due_sync(provider_profile="openai_chat")
        self.assertGreater(out.get("summaries_created", 0), 0)
        visible = store.get_visible_episodic_summaries(namespace=mem.namespace, limit=10)
        operation = next(item for item in visible if item["kind"] == "memory.operation_digest")
        digest_sids = set(operation.get("source_ids") or [])
        self.assertIn(use_sid, digest_sids)
        self.assertTrue(any(sid.endswith(":tool_use") for sid in digest_sids))
        # 原文仍在 messages 表, 可按 source_id 回读(与 settled 无关)
        record = store.get_retrieval_record(namespace=mem.namespace, source_id=use_sid)
        self.assertIsNotNone(record)
        store.close()


if __name__ == "__main__":
    unittest.main()
