"""Slice 1: operation_projection_policy 枚举、begin_turn 冻结与 settlement schema。

覆盖文档 §4(稳定 wire value)、§5(begin_turn 冻结)、§6(settled 账本 schema) 的存储层
切片: 不接入 request builder, 只验证配置校验、迁移、冻结与 settlement 原语不变量。
"""

from __future__ import annotations

import sqlite3
import tempfile
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
    NamespaceError,
    OperationProjectionPolicy,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
)
from memcore.store.migrations import CURRENT_SCHEMA_VERSION
from memcore.timeline import TurnProjectionSettlement


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _shared_mem(policy: str = "full_until_raw_compaction", conversation: str = "c1"):
    emb = HashedEmbeddingProvider()
    store = SQLiteMemoryStore(":memory:")
    index = InMemoryVectorIndex(embedding=emb)
    mem = MemorySystem(
        llm=_NoopLLM(),
        namespace=Namespace(user_id="u1", conversation_id=conversation),
        timezone="Asia/Shanghai",
        store=store,
        index=index,
        embedding=emb,
        config=MemoryConfig(operation_projection_policy=policy),
    )
    return mem, store


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()}


def _create_v4_database(path: Path) -> None:
    first = SQLiteMemoryStore(str(path))
    first.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ALTER TABLE turns DROP COLUMN operation_projection_policy")
        connection.execute("DROP TABLE IF EXISTS turn_projection_settlement")
        connection.execute("DROP TABLE IF EXISTS settled_prompt_projection")
        connection.execute("INSERT INTO turns(turn_id, user_id, conversation_id) VALUES ('legacy-t1', 'u1', 'c1')")
        connection.execute("PRAGMA user_version = 4")
        connection.commit()
    finally:
        connection.close()


class OperationProjectionPolicyTests(unittest.TestCase):
    def test_stable_wire_values(self) -> None:
        self.assertEqual(
            [item.value for item in OperationProjectionPolicy],
            ["full_until_raw_compaction", "compact_after_terminal"],
        )
        self.assertEqual(str(OperationProjectionPolicy.FULL_UNTIL_RAW_COMPACTION), "full_until_raw_compaction")
        self.assertEqual(str(OperationProjectionPolicy.COMPACT_AFTER_TERMINAL), "compact_after_terminal")


class MemoryConfigPolicyValidation(unittest.TestCase):
    def test_default_keeps_legacy_behavior(self) -> None:
        config = MemoryConfig()
        self.assertEqual(config.operation_projection_policy, "full_until_raw_compaction")

    def test_compact_after_terminal_is_accepted(self) -> None:
        config = MemoryConfig(operation_projection_policy="compact_after_terminal")
        self.assertEqual(config.operation_projection_policy, "compact_after_terminal")

    def test_case_and_whitespace_are_normalized(self) -> None:
        config = MemoryConfig(operation_projection_policy="  COMPACT_AFTER_TERMINAL  ")
        self.assertEqual(config.operation_projection_policy, "compact_after_terminal")

    def test_unknown_value_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            MemoryConfig(operation_projection_policy="compress_everything")
        with self.assertRaises(ConfigError):
            MemoryConfig(operation_projection_policy="")


class LatestSchemaTests(unittest.TestCase):
    def test_new_database_reaches_latest_schema_with_settlement_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "new.sqlite3"
            store = SQLiteMemoryStore(str(path))
            self.assertEqual(store.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertGreaterEqual(CURRENT_SCHEMA_VERSION, 5)
            connection = sqlite3.connect(path)
            try:
                self.assertIn("operation_projection_policy", _columns(connection, "turns"))
                self.assertIn(
                    "turn_projection_settlement",
                    {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")},
                )
                self.assertIn(
                    "settled_prompt_projection",
                    {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")},
                )
            finally:
                connection.close()
            store.close()


class V4ToV5MigrationTests(unittest.TestCase):
    def test_v4_database_gains_policy_column_and_settlement_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            _create_v4_database(path)
            store = SQLiteMemoryStore(str(path))
            self.assertEqual(store.schema_version, CURRENT_SCHEMA_VERSION)
            connection = sqlite3.connect(path)
            try:
                columns = _columns(connection, "turns")
                self.assertIn("operation_projection_policy", columns)
                policy = connection.execute(
                    "SELECT operation_projection_policy FROM turns WHERE turn_id = 'legacy-t1'"
                ).fetchone()
                self.assertEqual(policy[0], "full_until_raw_compaction")
                tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("turn_projection_settlement", tables)
                self.assertIn("settled_prompt_projection", tables)
            finally:
                connection.close()
            store.close()

    def test_v5_migration_is_idempotent_across_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite3"
            _create_v4_database(path)
            first = SQLiteMemoryStore(str(path))
            first.close()
            second = SQLiteMemoryStore(str(path))
            self.assertEqual(second.schema_version, CURRENT_SCHEMA_VERSION)
            second.close()


class TurnPolicyFreezeTests(unittest.TestCase):
    def test_begin_turn_freezes_config_policy(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        mem.begin_turn(
            turn_id="freeze-t1",
            opened_at=1000,
            stimuli=[
                TimelineEntryInput(
                    source_id="u1",
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
        handle = store.get_turn(namespace=mem.namespace, turn_id="freeze-t1")
        self.assertEqual(handle.operation_projection_policy, "compact_after_terminal")
        store.close()

    def test_begin_turn_defaults_to_legacy_policy(self) -> None:
        mem, store = _shared_mem()
        mem.begin_turn(
            turn_id="freeze-t2",
            opened_at=1000,
            stimuli=[
                TimelineEntryInput(
                    source_id="u2",
                    kind="message.user",
                    origin=EntryOrigin.USER,
                    turn_role=TurnRole.STIMULUS,
                    semantic_text="hi",
                    payload={"text": "hi"},
                    timestamp=1000,
                    compatibility_role="user",
                )
            ],
        )
        handle = store.get_turn(namespace=mem.namespace, turn_id="freeze-t2")
        self.assertEqual(handle.operation_projection_policy, "full_until_raw_compaction")
        store.close()

    def test_idempotent_begin_turn_keeps_frozen_policy(self) -> None:
        mem, store = _shared_mem(policy="compact_after_terminal")
        stimuli = [
            TimelineEntryInput(
                source_id="u3",
                kind="message.user",
                origin=EntryOrigin.USER,
                turn_role=TurnRole.STIMULUS,
                semantic_text="再查",
                payload={"text": "再查"},
                timestamp=1000,
                compatibility_role="user",
            )
        ]
        first = mem.begin_turn(turn_id="freeze-t3", opened_at=1000, stimuli=stimuli)
        second = mem.begin_turn(turn_id="freeze-t3", opened_at=1000, stimuli=stimuli)
        self.assertEqual(first.operation_projection_policy, "compact_after_terminal")
        self.assertEqual(second.operation_projection_policy, "compact_after_terminal")
        store.close()


class SettlementStoreInvariants(unittest.TestCase):
    def _settlement(self, *, hash_value: str = "h2", turn_id: str = "t1"):
        return TurnProjectionSettlement(
            turn_id=turn_id,
            policy="compact_after_terminal",
            settlement_status="settled",
            provider_profile="openai_chat",
            terminal_source_id="terminal-1",
            full_projection_hash="h1",
            settled_projection_hash=hash_value,
            first_changed_projection_index=3,
            full_projected_tokens=32000,
            settled_projected_tokens=2400,
            token_count_quality="exact",
            settled_at=2000,
        )

    def test_upsert_is_first_write_wins_and_idempotent(self) -> None:
        mem, store = _shared_mem()
        ns = mem.namespace
        self.assertTrue(store.upsert_turn_projection_settlement(namespace=ns, settlement=self._settlement()))
        self.assertFalse(
            store.upsert_turn_projection_settlement(namespace=ns, settlement=self._settlement(hash_value="H2"))
        )
        row = store.get_turn_projection_settlement(namespace=ns, turn_id="t1", provider_profile="openai_chat")
        self.assertIsNotNone(row)
        self.assertEqual(row["settled_projection_hash"], "h2")
        self.assertEqual(row["policy"], "compact_after_terminal")
        self.assertEqual(row["first_changed_projection_index"], 3)
        store.close()

    def test_settlement_is_per_provider_profile(self) -> None:
        mem, store = _shared_mem()
        ns = mem.namespace
        store.upsert_turn_projection_settlement(
            namespace=ns, settlement=replace(self._settlement(), provider_profile="openai_chat")
        )
        store.upsert_turn_projection_settlement(
            namespace=ns, settlement=replace(self._settlement(), provider_profile="anthropic_messages")
        )
        openai = store.get_turn_projection_settlement(namespace=ns, turn_id="t1", provider_profile="openai_chat")
        anthropic = store.get_turn_projection_settlement(
            namespace=ns, turn_id="t1", provider_profile="anthropic_messages"
        )
        self.assertIsNotNone(openai)
        self.assertIsNotNone(anthropic)
        store.close()

    def test_cross_namespace_read_is_rejected(self) -> None:
        mem, store = _shared_mem()
        store.upsert_turn_projection_settlement(namespace=mem.namespace, settlement=self._settlement())
        other = Namespace(user_id="u1", conversation_id="c2")
        with self.assertRaises(NamespaceError):
            store.get_turn_projection_settlement(namespace=other, turn_id="t1", provider_profile="openai_chat")
        store.close()

    def test_turn_projection_settlement_carries_full_contract_fields(self) -> None:
        settlement = self._settlement()
        self.assertEqual(settlement.settlement_status, "settled")
        self.assertEqual(settlement.settlement_schema_version, 1)
        self.assertEqual(settlement.terminal_source_id, "terminal-1")
        self.assertEqual(settlement.full_projection_hash, "h1")
        self.assertEqual(settlement.settled_projected_tokens, 2400)


if __name__ == "__main__":
    unittest.main()
