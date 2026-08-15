"""Phase 1: operation settlement 阈值通用配置扩展(执行单 Phase 1)。

覆盖:
- MemoryConfig 新字段校验与默认兼容(256 / 0.5);
- classify_observation 边界(门槛、no-expansion、可回读卡片);
- 自定义阈值(32768)下 32 KiB 边界行为;
- begin_turn 冻结: 运行中改配置不影响已开始 turn 的 settlement;
- 重启后按冻结值稳定重放(settled 行不变、metrics 携带冻结配置);
- 幂等(first-write-wins, 重试 complete 不生成第二套 hash);
- config hash 落库与一致性;
- V5 -> V6 schema 迁移(旧库升级、默认值、旧行语义不变)。
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path

from memcore import (
    ConfigError,
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
    TimelineEntryInput,
    TurnRole,
)
from memcore.settlement import (
    COMPACT_RELOAD_MIN_INLINE_BYTES,
    COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO,
    COMPACT_RELOADABLE,
    INLINE_FULL,
    classify_observation,
    render_compact_reload,
    settlement_config_hash,
    stable_settlement_id,
    utf8_bytes,
)
from memcore.store.migrations import CURRENT_SCHEMA_VERSION


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _shared_mem(
    *,
    policy: str = "compact_after_terminal",
    conversation: str = "c1",
    store: SQLiteMemoryStore | None = None,
    config: MemoryConfig | None = None,
) -> tuple[MemorySystem, SQLiteMemoryStore]:
    emb = HashedEmbeddingProvider()
    resolved_store = store or SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    resolved_config = config or MemoryConfig(operation_projection_policy=policy)
    mem = MemorySystem(
        llm=_NoopLLM(),
        namespace=Namespace(user_id="u1", conversation_id=conversation),
        timezone="Asia/Shanghai",
        store=resolved_store,
        index=index,
        embedding=emb,
        config=resolved_config,
    )
    return mem, resolved_store


def _turn_with_tools(
    mem: MemorySystem,
    *,
    turn_id: str = "t1",
    results: list[tuple[str, str, str]],
) -> list[str]:
    mem.begin_turn(
        turn_id=turn_id,
        opened_at=1000,
        stimuli=[
            TimelineEntryInput(
                source_id=f"{turn_id}-u",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text="查一下",
                payload={"text": "查一下"},
                timestamp=1000,
                compatibility_role="user",
            )
        ],
    )
    ids: list[str] = []
    for index, (tool, call_id, result) in enumerate(results):
        exchange = mem.record_tool_exchange(
            turn_id=turn_id,
            tool_name=tool,
            tool_call_id=call_id,
            tool_input={"query": f"q{index}"},
            result=result,
            source="web",
            source_id_prefix=f"tooltrace:{turn_id}-{index}",
        )
        ids.append(exchange["tool_result"]["source_id"])
    return ids


def _complete(mem: MemorySystem, turn_id: str, *, profile: str = "openai_chat", text: str = "ok"):
    mem.build_context_projection(provider_profile=profile)
    return mem.complete_turn(
        turn_id=turn_id,
        semantic_text=text,
        provider_output_raw=text,
        provider_profile=profile,
        provider_projection={"role": "assistant", "content": text},
    )


def _settlement_row(mem: MemorySystem, turn_id: str, profile: str = "openai_chat"):
    return mem.store.get_turn_projection_settlement(
        namespace=mem.namespace,
        turn_id=turn_id,
        provider_profile=profile,
    )


def _settled_payloads(mem: MemorySystem, turn_id: str, profile: str = "openai_chat") -> list[dict]:
    settlement_id = stable_settlement_id(turn_id=turn_id, provider_profile=profile)
    rows = mem.store.list_settled_projections(settlement_id=settlement_id)
    return [json.loads(row["payload_json"]) for row in rows]


def _entry():
    return types.SimpleNamespace(
        source_id="s1",
        correlation_id="c1",
        kind="tool.t.result",
        trace_metadata={"tool_name": "t", "status": "success"},
        payload={},
    )


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_preserve_legacy_values(self) -> None:
        config = MemoryConfig()
        self.assertEqual(config.operation_settlement_min_utf8_bytes, 256)
        self.assertEqual(config.operation_settlement_min_saved_ratio, 0.5)
        self.assertEqual(config.operation_settlement_min_utf8_bytes, COMPACT_RELOAD_MIN_INLINE_BYTES)
        self.assertEqual(config.operation_settlement_min_saved_ratio, COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO)

    def test_custom_values_are_accepted(self) -> None:
        config = MemoryConfig(
            operation_projection_policy="compact_after_terminal",
            operation_settlement_min_utf8_bytes=32768,
            operation_settlement_min_saved_ratio=0.5,
        )
        self.assertEqual(config.operation_settlement_min_utf8_bytes, 32768)
        self.assertEqual(config.operation_settlement_min_saved_ratio, 0.5)

    def test_invalid_min_bytes_are_rejected(self) -> None:
        for bad in (0, -1, 1.5, "256"):
            with self.assertRaises(ConfigError):
                MemoryConfig(operation_settlement_min_utf8_bytes=bad)

    def test_invalid_ratio_is_rejected(self) -> None:
        for bad in (0.0, 1.0, 1.5, -0.1, float("nan"), "0.5"):
            with self.assertRaises(ConfigError):
                MemoryConfig(operation_settlement_min_saved_ratio=bad)


class ClassifyBoundaryTests(unittest.TestCase):
    def test_below_min_is_inline_full(self) -> None:
        kind, content = classify_observation(_entry(), "x" * 255, min_inline_bytes=256, required_savings_ratio=0.5)
        self.assertEqual(kind, INLINE_FULL)
        self.assertEqual(content, "x" * 255)

    def test_at_min_goes_through_savings_check(self) -> None:
        # 门槛边界: 等于 min 时不自动 inline, 仍走收益检查; 卡片尺寸固定,
        # 用 2c(不省) 与 2c+20(省 50% 以上) 两个长度证明门槛本身不决定结果。
        entry = _entry()
        compact_bytes = utf8_bytes(render_compact_reload(entry, full_content="x" * 512))
        no_savings_body = "x" * (compact_bytes * 2)
        kind, content = classify_observation(entry, no_savings_body, min_inline_bytes=16, required_savings_ratio=0.5)
        self.assertEqual(kind, INLINE_FULL)
        self.assertEqual(content, no_savings_body)
        savings_body = "x" * (compact_bytes * 2 + 20)
        kind, _ = classify_observation(entry, savings_body, min_inline_bytes=16, required_savings_ratio=0.5)
        self.assertEqual(kind, COMPACT_RELOADABLE)

    def test_no_expansion_keeps_full_body(self) -> None:
        # 卡片省不到 50% 时保持完整(no-expansion 硬约束)
        entry = _entry()
        compact_bytes = utf8_bytes(render_compact_reload(entry, full_content="x" * 512))
        body = "x" * (compact_bytes * 2 - 20)
        kind, content = classify_observation(entry, body, min_inline_bytes=16, required_savings_ratio=0.5)
        self.assertEqual(kind, INLINE_FULL)
        self.assertEqual(content, body)

    def test_large_body_becomes_reloadable_card(self) -> None:
        entry = _entry()
        kind, content = classify_observation(entry, "y" * 100000, min_inline_bytes=256, required_savings_ratio=0.5)
        self.assertEqual(kind, COMPACT_RELOADABLE)
        self.assertIn("[compact_reloadable]", content)
        self.assertIn("source_id: s1", content)
        self.assertIn("open_memory(memory_id=", content)


class CustomThresholdStoreTests(unittest.TestCase):
    def _config(self, *, min_bytes: int, ratio: float = 0.5) -> MemoryConfig:
        return MemoryConfig(
            operation_projection_policy="compact_after_terminal",
            operation_settlement_min_utf8_bytes=min_bytes,
            operation_settlement_min_saved_ratio=ratio,
        )

    def test_8kib_stays_full_under_32kib_threshold(self) -> None:
        mem, store = _shared_mem(config=self._config(min_bytes=32768))
        try:
            _turn_with_tools(mem, turn_id="t8k", results=[("web_search", "call8k", "z" * 8192)])
            _complete(mem, "t8k")
            row = _settlement_row(mem, "t8k")
            self.assertEqual(row["settlement_status"], "settled_noop")
            projection = mem.build_context_projection(provider_profile="openai_chat")
            payloads = [dict(getattr(m, "payload", m)) for m in projection.payloads]
            joined = json.dumps(payloads, ensure_ascii=False)
            self.assertIn("z" * 8192, joined)
        finally:
            mem.close()
            store.close()

    def test_32kib_boundary_becomes_card(self) -> None:
        mem, store = _shared_mem(config=self._config(min_bytes=32768))
        try:
            _turn_with_tools(mem, turn_id="t32k", results=[("web_search", "call32k", "z" * 32768)])
            _complete(mem, "t32k")
            row = _settlement_row(mem, "t32k")
            self.assertEqual(row["settlement_status"], "settled")
            payloads = _settled_payloads(mem, "t32k")
            joined = json.dumps(payloads, ensure_ascii=False)
            self.assertIn("[compact_reloadable]", joined)
            self.assertNotIn("z" * 32768, joined)
        finally:
            mem.close()
            store.close()

    def test_default_threshold_still_compacts_1kib(self) -> None:
        # 默认 256/0.5 下 1 KiB 结果仍会被卡片化(与历史行为一致)
        mem, store = _shared_mem()
        try:
            _turn_with_tools(mem, turn_id="t1k", results=[("web_search", "call1k", "y" * 1024)])
            _complete(mem, "t1k")
            row = _settlement_row(mem, "t1k")
            self.assertEqual(row["settlement_status"], "settled")
            self.assertEqual(int(row["settlement_min_utf8_bytes"]), 256)
            self.assertAlmostEqual(float(row["settlement_min_saved_ratio"]), 0.5)
        finally:
            mem.close()
            store.close()


class FrozenSemanticsTests(unittest.TestCase):
    def test_turn_uses_values_frozen_at_begin_not_live_config(self) -> None:
        config = MemoryConfig(
            operation_projection_policy="compact_after_terminal",
            operation_settlement_min_utf8_bytes=32768,
            operation_settlement_min_saved_ratio=0.5,
        )
        mem, store = _shared_mem(config=config)
        try:
            _turn_with_tools(mem, turn_id="tfrozen", results=[("web_search", "cfrozen", "q" * 1024)])
            # 运行中把 live 配置改回默认: 若未冻结, 1 KiB 会被卡片化
            mem.config = replace(mem.config, operation_settlement_min_utf8_bytes=256)
            _complete(mem, "tfrozen")
            row = _settlement_row(mem, "tfrozen")
            self.assertEqual(row["settlement_status"], "settled_noop")
            self.assertEqual(int(row["settlement_min_utf8_bytes"]), 32768)
        finally:
            mem.close()
            store.close()

    def test_settlement_config_hash_matches_helper(self) -> None:
        mem, store = _shared_mem()
        try:
            _turn_with_tools(mem, turn_id="thash", results=[("web_search", "chash", "q" * 4096)])
            _complete(mem, "thash")
            row = _settlement_row(mem, "thash")
            expected = settlement_config_hash(policy="compact_after_terminal", min_utf8_bytes=256, saved_ratio=0.5)
            self.assertEqual(str(row["settlement_config_hash"]), expected)
            metrics = mem.settlement_metrics()
            matching = [m for m in metrics if m["settlement_config_hash"] == expected]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["settlement_min_utf8_bytes"], 256)
        finally:
            mem.close()
            store.close()


class RestartStabilityTests(unittest.TestCase):
    def test_reopen_with_different_config_keeps_frozen_settlement_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stable.sqlite3"
            store = SQLiteMemoryStore(str(path))
            config_a = MemoryConfig(
                operation_projection_policy="compact_after_terminal",
                operation_settlement_min_utf8_bytes=32768,
                operation_settlement_min_saved_ratio=0.5,
            )
            mem1, _ = _shared_mem(store=store, config=config_a)
            _turn_with_tools(mem1, turn_id="trestart", results=[("web_search", "crestart", "w" * 40000)])
            _complete(mem1, "trestart")
            before = _settled_payloads(mem1, "trestart")
            mem1.close()

            # 用默认配置重新打开同一库
            mem2, _ = _shared_mem(
                store=store, config=MemoryConfig(operation_projection_policy="compact_after_terminal")
            )
            try:
                after = _settled_payloads(mem2, "trestart")
                self.assertEqual(after, before)
                row = _settlement_row(mem2, "trestart")
                self.assertEqual(int(row["settlement_min_utf8_bytes"]), 32768)
                metrics = mem2.settlement_metrics()
                matching = [m for m in metrics if m["settlement_min_utf8_bytes"] == 32768]
                self.assertEqual(len(matching), 1)
                # 新 turn 用重开后的默认配置, 1 KiB 会被卡片化
                _turn_with_tools(mem2, turn_id="tnew", results=[("web_search", "cnew", "v" * 1024)])
                _complete(mem2, "tnew")
                new_row = _settlement_row(mem2, "tnew")
                self.assertEqual(new_row["settlement_status"], "settled")
                self.assertEqual(int(new_row["settlement_min_utf8_bytes"]), 256)
            finally:
                mem2.close()
                store.close()

    def test_complete_turn_retry_is_idempotent(self) -> None:
        mem, store = _shared_mem()
        try:
            _turn_with_tools(mem, turn_id="tretry", results=[("web_search", "cretry", "q" * 4096)])
            first = _complete(mem, "tretry")
            self.assertTrue(first.completed)
            second = _complete(mem, "tretry")
            self.assertTrue(second.completed)
            rows = mem.store.list_turn_projection_settlements(namespace=mem.namespace)
            self.assertEqual(len(rows), 1)
            payloads = _settled_payloads(mem, "tretry")
            self.assertEqual(len(payloads), len(_settled_payloads(mem, "tretry")))
        finally:
            mem.close()
            store.close()


def _create_v5_database(path: Path) -> None:
    first = SQLiteMemoryStore(str(path))
    first.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ALTER TABLE turns DROP COLUMN operation_settlement_min_utf8_bytes")
        connection.execute("ALTER TABLE turns DROP COLUMN operation_settlement_min_saved_ratio")
        connection.execute("ALTER TABLE turn_projection_settlement DROP COLUMN settlement_min_utf8_bytes")
        connection.execute("ALTER TABLE turn_projection_settlement DROP COLUMN settlement_min_saved_ratio")
        connection.execute("ALTER TABLE turn_projection_settlement DROP COLUMN settlement_config_hash")
        connection.execute("INSERT INTO turns(turn_id, user_id, conversation_id) VALUES ('legacy-t2', 'u1', 'c1')")
        connection.execute("PRAGMA user_version = 5")
        connection.commit()
    finally:
        connection.close()


class V5ToV6MigrationTests(unittest.TestCase):
    def test_v5_database_gains_threshold_columns_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy-v5.sqlite3"
            _create_v5_database(path)
            store = SQLiteMemoryStore(str(path))
            self.assertEqual(store.schema_version, CURRENT_SCHEMA_VERSION)
            connection = sqlite3.connect(path)
            try:
                columns = {row[1] for row in connection.execute('PRAGMA table_info("turns")').fetchall()}
                self.assertIn("operation_settlement_min_utf8_bytes", columns)
                self.assertIn("operation_settlement_min_saved_ratio", columns)
                row = connection.execute(
                    "SELECT operation_settlement_min_utf8_bytes, operation_settlement_min_saved_ratio"
                    " FROM turns WHERE turn_id = 'legacy-t2'"
                ).fetchone()
                self.assertEqual(int(row[0]), 256)
                self.assertAlmostEqual(float(row[1]), 0.5)
                settle_columns = {
                    row[1] for row in connection.execute('PRAGMA table_info("turn_projection_settlement")').fetchall()
                }
                self.assertIn("settlement_min_utf8_bytes", settle_columns)
                self.assertIn("settlement_min_saved_ratio", settle_columns)
                self.assertIn("settlement_config_hash", settle_columns)
            finally:
                connection.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
