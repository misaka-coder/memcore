"""Compaction's owner-checked turn lookups must not scan all stored history."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from memcore import Namespace, NamespaceError, SQLiteMemoryStore
from memcore.store.migrations import CURRENT_SCHEMA_VERSION


class TurnLookupIndexTests(unittest.TestCase):
    def assert_indexed_lookups(self, connection: sqlite3.Connection) -> None:
        queries = (
            ("SELECT * FROM messages WHERE turn_id = ? ORDER BY seq_no LIMIT 1", ("missing-turn",)),
            (
                "SELECT * FROM turn_projection_settlement WHERE turn_id = ? AND provider_profile = ?",
                ("missing-turn", "openai_chat"),
            ),
        )
        for query, params in queries:
            with self.subTest(query=query):
                plan = " ".join(str(row[3]) for row in connection.execute("EXPLAIN QUERY PLAN " + query, params))
                self.assertIn("SEARCH", plan)
                self.assertNotIn("SCAN", plan)
                self.assertNotIn("TEMP B-TREE", plan)

    def test_fresh_database_uses_indexes_for_owner_and_settlement_lookups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            store = SQLiteMemoryStore(str(path))
            store.close()
            with sqlite3.connect(path) as connection:
                self.assert_indexed_lookups(connection)
            connection.close()

    def test_current_schema_reopen_adds_indexes_without_changing_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            store = SQLiteMemoryStore(str(path))
            store.close()
            with sqlite3.connect(path) as connection:
                connection.execute("DROP INDEX idx_messages_turn_seq")
                connection.execute("DROP INDEX idx_settlement_turn_profile")
                connection.execute(
                    """INSERT INTO messages(source_id, user_id, seq_no, role, content, timestamp, turn_id)
                    VALUES ('legacy-source', 'owner', 1, 'user', 'preserved history', 123, 'legacy-turn')"""
                )
                before = connection.execute("SELECT * FROM messages").fetchall()
            connection.close()

            # Reopening the current version also installs missing derived indexes.
            for _ in range(2):
                store = SQLiteMemoryStore(str(path))
                store.close()
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], CURRENT_SCHEMA_VERSION)
                self.assertEqual(connection.execute("SELECT * FROM messages").fetchall(), before)
                self.assert_indexed_lookups(connection)
            connection.close()

            store = SQLiteMemoryStore(str(path))
            try:
                self.assertEqual(
                    store.get_turn_projections(
                        namespace=Namespace(user_id="owner"), turn_id="legacy-turn", provider_profile="openai_chat"
                    ),
                    [],
                )
                with self.assertRaises(NamespaceError):
                    store.get_turn_projections(
                        namespace=Namespace(user_id="other"), turn_id="legacy-turn", provider_profile="openai_chat"
                    )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
