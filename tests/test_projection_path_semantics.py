"""Evidence fidelity acceptance tests (projection semantics V4).

These tests pin the new contract: executable path evidence (tool call
arguments, tool results, assistant final text, PowerShell counter syntax)
survives the provider projection byte-for-byte, while secrets, media and
explicit host-internal path fields stay protected.  They also cover the
explicit legacy-projection migration and granular credential redaction.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from typing import Any

from memcore import (
    ANTHROPIC_PROFILE,
    CANONICAL_PROFILE,
    DEEPSEEK_PROFILE,
    OPENAI_PROFILE,
    OPENAI_RESPONSES_PROFILE,
    Actor,
    EntryOrigin,
    HashedEmbeddingProvider,
    InMemoryVectorIndex,
    LLMClient,
    LLMRequest,
    LLMResult,
    MemorySystem,
    Namespace,
    PROJECTION_VERSION,
    ProjectionMessageInput,
    ProjectionStatus,
    SQLiteMemoryStore,
    TimelineEntryInput,
    TurnRole,
    stable_projection_hash,
)
from memcore.projection import sanitize_projection_payload, sanitize_timeline_value

PS_COUNTER_COMMAND = r"Get-Counter '\Processor(_Total)\% Processor Time'"
WINDOWS_PATH_COMMAND = r"Get-ChildItem 'C:\Users\Public\Documents'"
POSIX_PATH_COMMAND = "find /opt/akane -maxdepth 2 -type f"


class _NoopLLM(LLMClient):
    def call(self, request: LLMRequest) -> LLMResult:
        return LLMResult(ok=True, data=request.fallback or {}, attempts=1)


def _stimulus(text: str, *, source_id: str, payload: dict[str, Any] | None = None) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="message.user",
        origin=EntryOrigin.USER,
        turn_role=TurnRole.STIMULUS,
        semantic_text=text,
        payload=payload or {"text": text},
    )


def _action(call_id: str, *, source_id: str, name: str, arguments: dict[str, Any]) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind=f"tool.{name}.call",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.ACTION,
        semantic_text=str(arguments),
        payload={"arguments": arguments},
        correlation_id=call_id,
        trace_metadata={"tool_name": name, "status": "running"},
    )


def _observation(call_id: str, *, source_id: str, text: str) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="tool.exec_run.result",
        origin=EntryOrigin.ENVIRONMENT,
        turn_role=TurnRole.OBSERVATION,
        semantic_text=text,
        payload={"output": text},
        correlation_id=call_id,
        trace_metadata={"tool_name": "exec_run", "status": "success"},
    )


def _final(text: str, *, source_id: str) -> TimelineEntryInput:
    return TimelineEntryInput(
        source_id=source_id,
        kind="message.assistant",
        origin=EntryOrigin.ASSISTANT,
        turn_role=TurnRole.FINAL,
        semantic_text=text,
        payload={"text": text, "provider_output_raw": text},
    )


def _projected_tool_commands(projection: Any) -> list[str]:
    commands: list[str] = []
    for message in projection.messages:
        raw_calls = dict(message.payload).get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            function = call.get("function") if isinstance(call, Mapping) else {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError):
                continue
            command = parsed.get("command") if isinstance(parsed, dict) else None
            if isinstance(command, str):
                commands.append(command)
    return commands


class PathEvidenceBase(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = Namespace(
            tenant_id="tenant",
            user_id="user",
            domain_id="domain",
            conversation_id="conversation",
        )
        self.store = SQLiteMemoryStore(":memory:")
        self.embedding = HashedEmbeddingProvider()
        self.mem = MemorySystem(
            llm=_NoopLLM(),
            namespace=self.namespace,
            timezone="Asia/Shanghai",
            store=self.store,
            index=InMemoryVectorIndex(embedding=self.embedding),
            embedding=self.embedding,
        )

    def tearDown(self) -> None:
        self.mem.close()
        self.store.close()


class PathEvidencePreservationTests(PathEvidenceBase):
    def _openai_tool_call_projection(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_payload = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "exec_run",
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }
        payload, _ = sanitize_projection_payload(raw_payload)
        return payload

    def test_windows_absolute_path_preserved_in_openai_tool_arguments(self) -> None:
        payload = self._openai_tool_call_projection({"command": WINDOWS_PATH_COMMAND})
        parsed = json.loads(payload["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(parsed["command"], WINDOWS_PATH_COMMAND)
        self.assertNotIn("local path omitted", payload["tool_calls"][0]["function"]["arguments"])

    def test_posix_absolute_path_preserved_in_openai_tool_arguments(self) -> None:
        payload = self._openai_tool_call_projection({"command": POSIX_PATH_COMMAND})
        parsed = json.loads(payload["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(parsed["command"], POSIX_PATH_COMMAND)

    def test_powershell_counter_preserved_in_openai_tool_arguments(self) -> None:
        payload = self._openai_tool_call_projection({"command": PS_COUNTER_COMMAND})
        parsed = json.loads(payload["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(parsed["command"], PS_COUNTER_COMMAND)
        self.assertIn("\\Processor(_Total)\\% Processor Time", parsed["command"])

    def test_openai_tool_arguments_remain_valid_json(self) -> None:
        payload = self._openai_tool_call_projection(
            {
                "command": f"{WINDOWS_PATH_COMMAND}; {POSIX_PATH_COMMAND}",
                "cwd": r"C:\Users\Public",
            }
        )
        parsed = json.loads(payload["tool_calls"][0]["function"]["arguments"])
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["cwd"], r"C:\Users\Public")

    def test_anthropic_tool_use_and_result_keep_paths(self) -> None:
        use_payload, use_status = sanitize_projection_payload(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-2",
                        "name": "exec_run",
                        "input": {"command": WINDOWS_PATH_COMMAND},
                    }
                ],
            }
        )
        self.assertEqual(use_status, ProjectionStatus.COMPLETE)
        self.assertEqual(
            use_payload["content"][0]["input"]["command"],
            WINDOWS_PATH_COMMAND,
        )
        result_payload, result_status = sanitize_projection_payload(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-2",
                        "content": "saved to C:\\Users\\Public\\Documents\\out.txt",
                    }
                ],
            }
        )
        self.assertEqual(result_status, ProjectionStatus.COMPLETE)
        self.assertIn("C:\\Users\\Public\\Documents\\out.txt", result_payload["content"][0]["content"])

    def test_assistant_final_path_visible_next_round(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("帮我保存", source_id="final-user")],
            turn_id="final-path-turn",
        )
        final_text = "已保存到 C:\\Users\\Public\\Documents\\notes.txt"
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text=final_text,
            provider_output_raw=final_text,
        )
        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        joined = json.dumps(projection.payloads, ensure_ascii=False)
        self.assertIn("C:\\\\Users\\\\Public\\\\Documents\\\\notes.txt", joined)
        repeated = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(projection.stable_prefix_hash, repeated.stable_prefix_hash)

    def test_tool_result_body_keeps_discovered_paths(self) -> None:
        payload, status = sanitize_projection_payload(
            {
                "role": "tool",
                "tool_call_id": "call-3",
                "content": "found: /opt/akane/notes.txt\ncreated C:\\Users\\Public\\Documents\\x.txt",
            }
        )
        self.assertEqual(status, ProjectionStatus.COMPLETE)
        self.assertIn("/opt/akane/notes.txt", payload["content"])
        self.assertIn("C:\\Users\\Public\\Documents\\x.txt", payload["content"])

    def test_generic_path_keys_in_tool_arguments_are_evidence(self) -> None:
        payload = self._openai_tool_call_projection(
            {"action": "copy", "path": r"C:\Users\Public\Documents", "file_path": "/opt/akane/x.txt"}
        )
        parsed = json.loads(payload["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(parsed["path"], r"C:\Users\Public\Documents")
        self.assertEqual(parsed["file_path"], "/opt/akane/x.txt")

    def test_open_memory_full_detail_recovers_operation_paths(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("run", source_id="detail-user")],
            turn_id="detail-turn",
        )
        self.mem.append_entry(
            _action("call-d", source_id="detail-action", name="exec_run", arguments={"command": POSIX_PATH_COMMAND}),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("call-d", source_id="detail-result", text="wrote /opt/akane/out.log"),
            turn_id=handle.turn_id,
        )
        projection = self.mem.build_context_projection(provider_profile=CANONICAL_PROFILE)
        full_text = "\n".join(str(payload) for payload in projection.payloads)
        self.assertIn("/opt/akane/out.log", full_text)
        self.assertIn(POSIX_PATH_COMMAND, full_text)


class PathEvidenceProtectionTests(PathEvidenceBase):
    def test_credential_variable_names_in_source_code_do_not_hide_the_tool_result(self) -> None:
        source = (
            "def fetch_token(response):\n"
            "    token = response.json()\n"
            "    pt_key = payload.get('pt_key')\n"
            "    return token, pt_key\n"
        )
        clean, status = sanitize_timeline_value(source)
        self.assertEqual(status, ProjectionStatus.COMPLETE)
        self.assertEqual(clean, source)

    def test_literal_secret_is_redacted_without_destroying_surrounding_evidence(self) -> None:
        source = (
            "before = 'keep this line'\n"
            "token = 'abcdefghijklmnopqrstuvwxyz123456'\n"
            "after = response.json()\n"
        )
        clean, status = sanitize_timeline_value(source)
        self.assertEqual(status, ProjectionStatus.SKIPPED_UNSAFE)
        self.assertIn("before = 'keep this line'", clean)
        self.assertIn("token = '[secret value omitted]'", clean)
        self.assertIn("after = response.json()", clean)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", clean)

    def test_json_text_secret_value_is_redacted_without_breaking_json(self) -> None:
        raw = '{"status":"ok","pt_key":"abcdefghijklmnopqrstuvwxyz123456","count":2}'
        clean, status = sanitize_timeline_value(raw)
        self.assertEqual(status, ProjectionStatus.SKIPPED_UNSAFE)
        parsed = json.loads(clean)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["pt_key"], "[secret value omitted]")
        self.assertEqual(parsed["count"], 2)

    def test_openai_tool_arguments_remain_valid_json_when_a_secret_field_is_redacted(self) -> None:
        original = '{"api_key":"sk-abcdefghijklmnopqrstuvwxyz","path":"/opt/akane/project"}'
        clean, status = sanitize_projection_payload(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-secret",
                        "type": "function",
                        "function": {"name": "exec", "arguments": original},
                    }
                ],
            },
            provider_profile=OPENAI_PROFILE,
        )
        self.assertEqual(status, ProjectionStatus.SKIPPED_UNSAFE)
        arguments = clean["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(arguments)
        self.assertEqual(parsed["api_key"], "[secret value omitted]")
        self.assertEqual(parsed["path"], "/opt/akane/project")

    def test_open_turn_projections_keep_auth_related_source_code_visible(self) -> None:
        source = "token = response.json()\nheaders = {'Authorization': token}\nprint(headers)"
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("inspect auth code", source_id="auth-user")],
            turn_id="auth-open-turn",
        )
        self.mem.append_entry(
            _action("auth-call", source_id="auth-action", name="exec_run", arguments={"command": "cat login.py"}),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("auth-call", source_id="auth-result", text=source),
            turn_id=handle.turn_id,
        )
        for profile in (OPENAI_PROFILE, DEEPSEEK_PROFILE, OPENAI_RESPONSES_PROFILE, ANTHROPIC_PROFILE):
            with self.subTest(profile=profile):
                projection = self.mem.build_context_projection(provider_profile=profile)
                if profile in {OPENAI_PROFILE, DEEPSEEK_PROFILE}:
                    result = next(payload for payload in projection.payloads if payload.get("role") == "tool")
                    visible = result["content"]
                elif profile == OPENAI_RESPONSES_PROFILE:
                    result = next(
                        payload for payload in projection.payloads if payload.get("type") == "function_call_output"
                    )
                    visible = result["output"]
                else:
                    result = next(
                        block
                        for payload in projection.payloads
                        for block in payload.get("content", [])
                        if isinstance(block, Mapping) and block.get("type") == "tool_result"
                    )
                    visible = result["content"]
                self.assertEqual(visible, source)

    def test_secrets_still_removed(self) -> None:
        for raw, expected in (
            ({"api_key": "sk-abc12345678901234567890"}, "[secret value omitted]"),
            (
                "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
                "[secret value omitted]",
            ),
        ):
            clean, status = sanitize_timeline_value(raw)
            self.assertEqual(status, ProjectionStatus.SKIPPED_UNSAFE)
            self.assertIn(expected, json.dumps(clean, ensure_ascii=False))

    def test_host_internal_path_fields_still_hidden(self) -> None:
        payload = {
            "cached_path": r"C:\Users\Lenovo\AppData\Local\akane\cache\x.png",
            "storage_relpath": "users/u1/media/123.png",
            "database_path": r"C:\Users\Akane\data\memcore_v01.db",
            "absolute_path": "/srv/akane/media/1.png",
            "local_path": r"C:\Temp\out.log",
        }
        clean, status = sanitize_projection_payload({"role": "user", "content": payload})
        self.assertEqual(status, ProjectionStatus.SKIPPED_UNSAFE)
        for key in payload:
            self.assertNotIn(str(payload[key]), json.dumps(clean, ensure_ascii=False))

    def test_base64_and_binary_media_still_omitted(self) -> None:
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        clean, status = sanitize_timeline_value(data_url)
        self.assertEqual(status, ProjectionStatus.MEDIA_OMITTED)
        self.assertIn("media omitted", str(clean))

    def test_opaque_ids_unchanged(self) -> None:
        text = "output_ref=runlog:2fb194c-123; gen_4aa5bae7"
        clean, status = sanitize_timeline_value(text)
        self.assertEqual(status, ProjectionStatus.COMPLETE)
        self.assertEqual(clean, text)

    def test_settled_compact_card_has_no_backing_path(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[
                _stimulus(
                    "source: attachment\nfile_id: file_1\nfilename: photo.jpg",
                    source_id="material-user",
                    payload={
                        "file_id": "file_1",
                        "kind": "image",
                        "filename": "photo.jpg",
                        "cached_path": r"C:\Users\Lenovo\AppData\Local\akane\cache\photo.png",
                    },
                )
            ],
            turn_id="material-turn",
        )
        projection = self.mem.build_context_projection(provider_profile=CANONICAL_PROFILE)
        joined = json.dumps(projection.payloads, ensure_ascii=False)
        self.assertNotIn("AppData", joined)
        self.assertIn("file_1", joined)


class LegacyPathProjectionMigrationTests(PathEvidenceBase):
    """Fabricate V2-frozen marker rows, then migrate and verify."""

    def _fabricate_legacy_action_row(self, *, store, turn_id: str, projection_id: str, source_id: str, index: int) -> None:
        legacy_arguments = json.dumps(
            {"command": "Get-ChildItem '[local path omitted from persistent history]'"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        legacy_payload = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "legacy-call", "type": "function", "function": {"name": "exec_run", "arguments": legacy_arguments}}
            ],
        }
        with store._conn:
            store._conn.execute(
                """
                INSERT OR REPLACE INTO prompt_projections(
                    projection_id, tenant_id, user_id, domain_id, conversation_id,
                    turn_id, projection_index, provider_profile, payload_json,
                    source_ids_json, payload_hash, projection_status, projection_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    projection_id,
                    "tenant",
                    "user",
                    "domain",
                    "conversation",
                    turn_id,
                    index,
                    OPENAI_PROFILE,
                    json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    json.dumps([source_id]),
                    stable_projection_hash(legacy_payload),
                    "request_frozen",
                    2,
                    1,
                ),
            )

    def _build_turn_with_raw_sources(self, turn_id: str) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("列出目录", source_id=f"{turn_id}-user")],
            turn_id=turn_id,
        )
        self.mem.append_entry(
            _action(
                "legacy-call",
                source_id=f"{turn_id}-action",
                name="exec_run",
                arguments={"command": WINDOWS_PATH_COMMAND},
            ),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("legacy-call", source_id=f"{turn_id}-result", text="listed"),
            turn_id=handle.turn_id,
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="ok",
            provider_output_raw="ok",
        )
        # Freeze the full v3 projection first, then corrupt one frozen row with
        # a legacy marker payload to simulate a v2-era database.
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)

    def _build_turn_with_long_observation(self, turn_id: str) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("列出目录", source_id=f"{turn_id}-user")],
            turn_id=turn_id,
        )
        self.mem.append_entry(
            _action(
                "legacy-call",
                source_id=f"{turn_id}-action",
                name="exec_run",
                arguments={"command": WINDOWS_PATH_COMMAND},
            ),
            turn_id=handle.turn_id,
        )
        long_text = "目录内容：" + ("文件条目、路径与时间戳。" * 60)
        self.mem.append_entry(
            _observation("legacy-call", source_id=f"{turn_id}-result", text=long_text),
            turn_id=handle.turn_id,
        )
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="ok",
            provider_output_raw="ok",
        )
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)

    def test_migration_restores_marker_rows_and_is_idempotent(self) -> None:
        self._build_turn_with_raw_sources("mig-turn")
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="mig-turn",
            projection_id="legacy-proj",
            source_id="mig-turn-action",
            index=1,
        )
        before = dict(self.store._conn.execute("SELECT * FROM prompt_projections WHERE projection_id = ?", ("legacy-proj",)).fetchone())

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["migrated"], 1)
        self.assertEqual(report["version_advanced_only"], 0)
        self.assertEqual(report["preserved_without_raw_source"], 0)

        row = dict(self.store._conn.execute("SELECT * FROM prompt_projections WHERE projection_id = ?", ("legacy-proj",)).fetchone())
        self.assertEqual(int(row["projection_version"]), PROJECTION_VERSION)
        restored = json.loads(row["payload_json"])["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(restored)["command"], WINDOWS_PATH_COMMAND)
        self.assertNotIn("local path omitted", restored)

        second = self.mem.migrate_legacy_path_projections()
        self.assertEqual(second["migrated"], 0)
        self.assertEqual(second["version_advanced_only"], 0)
        self.assertEqual(second["settled_rebuilt"], 0)

        after = dict(self.store._conn.execute("SELECT * FROM prompt_projections WHERE projection_id = ?", ("legacy-proj",)).fetchone())
        self.assertEqual(after["payload_json"], row["payload_json"])
        self.assertEqual(before["source_ids_json"], row["source_ids_json"])

        rebuilt = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertIn(WINDOWS_PATH_COMMAND, _projected_tool_commands(rebuilt))

    def test_migration_restores_v3_secret_marker_from_raw_source(self) -> None:
        source_text = "token = response.json()\npt_key = payload.get('pt_key')\n"
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("inspect source", source_id="secret-user")],
            turn_id="secret-turn",
        )
        self.mem.append_entry(
            _action(
                "secret-call",
                source_id="secret-action",
                name="read_workspace",
                arguments={"path": "login_joyclaw.py"},
            ),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("secret-call", source_id="secret-result", text=source_text),
            turn_id=handle.turn_id,
        )
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        legacy_payload = {
            "role": "tool",
            "tool_call_id": "secret-call",
            "content": "[secret omitted from persistent history]",
        }
        with self.store._conn:
            self.store._conn.execute(
                """
                UPDATE prompt_projections
                SET payload_json = ?, payload_hash = ?, projection_status = ?, projection_version = ?
                WHERE turn_id = ? AND projection_index = ? AND provider_profile = ?
                """,
                (
                    json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    stable_projection_hash(legacy_payload),
                    "skipped_unsafe",
                    3,
                    "secret-turn",
                    2,
                    OPENAI_PROFILE,
                ),
            )

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["scanned_memcore_secret_marker_rows"], 1)
        self.assertEqual(report["migrated"], 1)
        row = self.store._conn.execute(
            """
            SELECT payload_json, projection_version FROM prompt_projections
            WHERE turn_id = ? AND projection_index = ? AND provider_profile = ?
            """,
            ("secret-turn", 2, OPENAI_PROFILE),
        ).fetchone()
        self.assertIsNotNone(row)
        restored = json.loads(str(row["payload_json"]))
        self.assertEqual(restored["content"], source_text)
        self.assertEqual(int(row["projection_version"]), PROJECTION_VERSION)

        second = self.mem.migrate_legacy_path_projections()
        self.assertEqual(second["scanned_memcore_secret_marker_rows"], 0)
        self.assertEqual(second["migrated"], 0)

    def test_unaffected_rows_stay_byte_identical(self) -> None:
        self._build_turn_with_raw_sources("byte-turn")
        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="byte-turn",
            projection_id="legacy-proj-byte",
            source_id="byte-turn-action",
            index=1,
        )
        stimulus_row = self.store._conn.execute(
            "SELECT * FROM prompt_projections WHERE turn_id = ? AND projection_index = 0", ("byte-turn",)
        ).fetchone()
        stimulus_before = dict(stimulus_row)
        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 1)
        stimulus_after = dict(
            self.store._conn.execute(
                "SELECT * FROM prompt_projections WHERE turn_id = ? AND projection_index = 0", ("byte-turn",)
            ).fetchone()
        )
        self.assertEqual(stimulus_before["payload_json"], stimulus_after["payload_json"])
        self.assertEqual(stimulus_before["payload_hash"], stimulus_after["payload_hash"])
        self.assertEqual(stimulus_before["projection_version"], stimulus_after["projection_version"])

    def test_missing_raw_source_is_preserved_and_reported(self) -> None:
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="ghost-turn",
            projection_id="ghost-proj",
            source_id="ghost-action",
            index=0,
        )
        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 0)
        self.assertEqual(report["preserved_without_raw_source"], 1)
        row = dict(self.store._conn.execute("SELECT * FROM prompt_projections WHERE projection_id = ?", ("ghost-proj",)).fetchone())
        self.assertEqual(int(row["projection_version"]), 2)
        self.assertIn("local path omitted", row["payload_json"])

    def test_settled_marker_rows_are_rebuilt_with_new_cards(self) -> None:
        self._build_turn_with_long_observation("settle-turn")
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="settle-turn",
            projection_id="settle-legacy-proj",
            source_id="settle-turn-action",
            index=1,
        )
        settlement_id = f"settle-turn:{OPENAI_PROFILE}"
        marker_payload = {"role": "tool", "tool_call_id": "legacy-call", "content": "[local path omitted from persistent history]"}
        with self.store._conn:
            self.store._conn.execute(
                """
                INSERT INTO settled_prompt_projection(
                    settlement_id, projection_index, provider_profile,
                    source_ids_json, payload_json, payload_hash, projection_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'settled')
                """,
                (
                    settlement_id,
                    2,
                    OPENAI_PROFILE,
                    json.dumps(["settle-turn-result"]),
                    json.dumps(marker_payload, ensure_ascii=False),
                    stable_projection_hash(marker_payload),
                ),
            )
            self.store._conn.execute(
                """
                INSERT OR REPLACE INTO turn_projection_settlement(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                    policy, settlement_status, terminal_source_id, settled_at
                ) VALUES ('tenant', 'user', 'domain', 'conversation', 'settle-turn', ?, ?, 'settled', ?, 1)
                """,
                (OPENAI_PROFILE, "full_until_raw_compaction", "settle-turn-result"),
            )

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 1)
        self.assertEqual(report["settled_rebuilt"], 1)
        self.assertEqual(report["settled_rebuild_failed_dropped"], 0)

        settled_rows = self.store._conn.execute(
            "SELECT * FROM settled_prompt_projection WHERE settlement_id = ?", (settlement_id,)
        ).fetchall()
        self.assertGreaterEqual(len(settled_rows), 3)
        joined_rows = "\n".join(str(row["payload_json"]) for row in settled_rows)
        self.assertIn("compact_reloadable", joined_rows)
        self.assertNotIn("local path omitted", joined_rows)
        settlement_meta = dict(
            self.store._conn.execute(
                "SELECT * FROM turn_projection_settlement WHERE turn_id = ?", ("settle-turn",)
            ).fetchone()
        )
        self.assertEqual(settlement_meta["settlement_status"], "settled")
        self.assertNotEqual(settlement_meta["full_projection_hash"], "")

        projection = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertTrue(projection.has_compact_history)
        repeated = self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.assertEqual(projection.stable_prefix_hash, repeated.stable_prefix_hash)

        second = self.mem.migrate_legacy_path_projections()
        self.assertEqual(second["migrated"], 0)
        self.assertEqual(second["settled_rebuilt"], 0)

    def test_settled_rebuild_with_short_observation_reports_noop(self) -> None:
        self._build_turn_with_raw_sources("noop-turn")
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="noop-turn",
            projection_id="noop-legacy-proj",
            source_id="noop-turn-action",
            index=1,
        )
        settlement_id = f"noop-turn:{OPENAI_PROFILE}"
        marker_payload = {"role": "tool", "tool_call_id": "legacy-call", "content": "[local path omitted from persistent history]"}
        with self.store._conn:
            self.store._conn.execute(
                """
                INSERT INTO settled_prompt_projection(
                    settlement_id, projection_index, provider_profile,
                    source_ids_json, payload_json, payload_hash, projection_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'settled')
                """,
                (
                    settlement_id,
                    2,
                    OPENAI_PROFILE,
                    json.dumps(["noop-turn-result"]),
                    json.dumps(marker_payload, ensure_ascii=False),
                    stable_projection_hash(marker_payload),
                ),
            )
            self.store._conn.execute(
                """
                INSERT OR REPLACE INTO turn_projection_settlement(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                    policy, settlement_status, terminal_source_id, settled_at
                ) VALUES ('tenant', 'user', 'domain', 'conversation', 'noop-turn', ?, ?, 'settled', ?, 1)
                """,
                (OPENAI_PROFILE, "full_until_raw_compaction", "noop-turn-result"),
            )

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 1)
        self.assertEqual(report["settled_rebuilt_noop"], 1)
        self.assertEqual(report["settled_rebuilt"], 0)
        settled_rows = self.store._conn.execute(
            "SELECT * FROM settled_prompt_projection WHERE settlement_id = ?", (settlement_id,)
        ).fetchall()
        self.assertEqual(len(settled_rows), 0)
        settlement_meta = dict(
            self.store._conn.execute(
                "SELECT * FROM turn_projection_settlement WHERE turn_id = ?", ("noop-turn",)
            ).fetchone()
        )
        self.assertEqual(settlement_meta["settlement_status"], "settled_noop")

    def test_stale_full_hash_settlement_is_rebuilt_even_without_markers(self) -> None:
        self._build_turn_with_long_observation("hash-turn")
        # A settlement whose settled cards are clean but whose recorded full
        # hash matches the pre-migration (marker-containing) ledger.
        settlement_id = f"hash-turn:{OPENAI_PROFILE}"
        clean_card = {"role": "tool", "tool_call_id": "legacy-call", "content": "clean old card"}
        with self.store._conn:
            self.store._conn.execute(
                """
                INSERT INTO settled_prompt_projection(
                    settlement_id, projection_index, provider_profile,
                    source_ids_json, payload_json, payload_hash, projection_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'settled')
                """,
                (
                    settlement_id,
                    2,
                    OPENAI_PROFILE,
                    json.dumps(["hash-turn-result"]),
                    json.dumps(clean_card),
                    stable_projection_hash(clean_card),
                ),
            )
            self.store._conn.execute(
                """
                INSERT OR REPLACE INTO turn_projection_settlement(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                    policy, settlement_status, terminal_source_id, full_projection_hash,
                    settled_projection_hash, settled_at
                ) VALUES ('tenant', 'user', 'domain', 'conversation', 'hash-turn', ?, ?, 'settled', ?, 'stale-hash', 'stale-settled-hash', 1)
                """,
                (OPENAI_PROFILE, "full_until_raw_compaction", "hash-turn-result"),
            )
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="hash-turn",
            projection_id="hash-legacy-proj",
            source_id="hash-turn-action",
            index=1,
        )

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 1)
        self.assertEqual(report["settled_rebuilt"], 1)
        settlement_meta = dict(
            self.store._conn.execute(
                "SELECT * FROM turn_projection_settlement WHERE turn_id = ?", ("hash-turn",)
            ).fetchone()
        )
        self.assertEqual(settlement_meta["settlement_status"], "settled")
        self.assertNotEqual(settlement_meta["full_projection_hash"], "stale-hash")

    def test_settled_marker_is_rebuilt_even_when_full_hash_is_current(self) -> None:
        self._build_turn_with_long_observation("marker-only-turn")
        full_rows = self.store._conn.execute(
            """
            SELECT payload_json FROM prompt_projections
            WHERE turn_id = ? AND provider_profile = ?
            ORDER BY projection_index
            """,
            ("marker-only-turn", OPENAI_PROFILE),
        ).fetchall()
        current_full_hash = stable_projection_hash(
            [json.loads(str(row["payload_json"] or "{}")) for row in full_rows]
        )
        settlement_id = f"marker-only-turn:{OPENAI_PROFILE}"
        marker_payload = {
            "role": "tool",
            "tool_call_id": "legacy-call",
            "content": "[local path omitted from persistent history]",
        }
        with self.store._conn:
            self.store._conn.execute(
                """
                INSERT INTO settled_prompt_projection(
                    settlement_id, projection_index, provider_profile,
                    source_ids_json, payload_json, payload_hash, projection_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'settled')
                """,
                (
                    settlement_id,
                    2,
                    OPENAI_PROFILE,
                    json.dumps(["marker-only-turn-result"]),
                    json.dumps(marker_payload, ensure_ascii=False),
                    stable_projection_hash(marker_payload),
                ),
            )
            self.store._conn.execute(
                """
                INSERT OR REPLACE INTO turn_projection_settlement(
                    tenant_id, user_id, domain_id, conversation_id, turn_id, provider_profile,
                    policy, settlement_status, terminal_source_id, full_projection_hash,
                    settled_projection_hash, settled_at
                ) VALUES ('tenant', 'user', 'domain', 'conversation', 'marker-only-turn', ?, ?, 'settled', ?, ?, ?, 1)
                """,
                (
                    OPENAI_PROFILE,
                    "full_until_raw_compaction",
                    "marker-only-turn-result",
                    current_full_hash,
                    stable_projection_hash(marker_payload),
                ),
            )

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 0)
        self.assertEqual(report["settled_rebuilt"], 1)
        settled_rows = self.store._conn.execute(
            "SELECT payload_json FROM settled_prompt_projection WHERE settlement_id = ?",
            (settlement_id,),
        ).fetchall()
        self.assertGreaterEqual(len(settled_rows), 3)
        joined_rows = "\n".join(str(row["payload_json"] or "") for row in settled_rows)
        self.assertIn("compact_reloadable", joined_rows)
        self.assertNotIn("local path omitted", joined_rows)

        second = self.mem.migrate_legacy_path_projections()
        self.assertEqual(second["settled_rebuilt"], 0)

    def test_irrecoverable_settled_marker_is_not_rebuilt_forever(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("run", source_id="irrecoverable-settled-user")],
            turn_id="irrecoverable-settled-turn",
        )
        self.mem.append_entry(
            _action(
                "irrecoverable-call",
                source_id="irrecoverable-settled-action",
                name="exec_run",
                arguments={"command": "Get-ChildItem '[local_path]'"},
            ),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation(
                "irrecoverable-call",
                source_id="irrecoverable-settled-result",
                text="[local_path]\n" + ("long result evidence " * 80),
            ),
            turn_id=handle.turn_id,
        )
        with self.store._conn:
            self.store._conn.execute(
                "UPDATE turns SET operation_projection_policy = ? WHERE turn_id = ?",
                ("compact_after_terminal", handle.turn_id),
            )
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self.mem.complete_turn(
            turn_id=handle.turn_id,
            semantic_text="done",
            provider_output_raw="done",
            provider_profile=OPENAI_PROFILE,
            provider_projection={"role": "assistant", "content": "done"},
        )
        settlement_id = f"{handle.turn_id}:{OPENAI_PROFILE}"
        before = [
            tuple(row)
            for row in self.store._conn.execute(
                "SELECT * FROM settled_prompt_projection WHERE settlement_id = ? ORDER BY projection_index",
                (settlement_id,),
            ).fetchall()
        ]
        self.assertTrue(before)
        self.assertTrue(any("[local_path]" in str(value) for row in before for value in row))

        first = self.mem.migrate_legacy_path_projections()
        second = self.mem.migrate_legacy_path_projections()
        self.assertEqual(first["settled_rebuilt"], 0)
        self.assertEqual(second["settled_rebuilt"], 0)
        after = [
            tuple(row)
            for row in self.store._conn.execute(
                "SELECT * FROM settled_prompt_projection WHERE settlement_id = ? ORDER BY projection_index",
                (settlement_id,),
            ).fetchall()
        ]
        self.assertEqual(after, before)

    def test_host_redacted_raw_is_preserved_and_counted_irrecoverable(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("列出目录", source_id="damage-user")],
            turn_id="damage-turn",
        )
        self.mem.append_entry(
            _action(
                "legacy-call",
                source_id="damage-action",
                name="exec_run",
                arguments={"command": "Get-ChildItem '[local_path]'"},
            ),
            turn_id=handle.turn_id,
        )
        self.mem.append_entry(
            _observation("legacy-call", source_id="damage-result", text="listed"),
            turn_id=handle.turn_id,
        )
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="damage-turn",
            projection_id="damage-proj",
            source_id="damage-action",
            index=1,
        )
        with self.store._conn:
            self.store._conn.execute(
                """
                UPDATE prompt_projections
                SET payload_json = ?, payload_hash = ?, projection_version = 2
                WHERE projection_id = ?
                """,
                (
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "legacy-call",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_run",
                                        "arguments": json.dumps({"command": "Get-ChildItem '[local_path]'"}),
                                    },
                                }
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    stable_projection_hash(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "legacy-call",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_run",
                                        "arguments": json.dumps({"command": "Get-ChildItem '[local_path]'"}),
                                    },
                                }
                            ],
                        }
                    ),
                    "damage-proj",
                ),
            )

        report = self.mem.migrate_legacy_path_projections()
        self.assertEqual(report["migrated"], 0)
        self.assertGreaterEqual(report["scanned_host_local_path_rows"], 1)
        self.assertEqual(report["preserved_irrecoverable_host_redaction"], 1)
        row = dict(
            self.store._conn.execute(
                "SELECT * FROM prompt_projections WHERE projection_id = ?", ("damage-proj",)
            ).fetchone()
        )
        self.assertEqual(int(row["projection_version"]), 2)
        self.assertIn("[local_path]", row["payload_json"])

    def test_tmpdir_alias_rows_are_scanned_and_preserved(self) -> None:
        handle = self.mem.begin_turn(
            stimuli=[_stimulus("run", source_id="tmp-user")],
            turn_id="tmp-turn",
        )
        self.mem.append_entry(
            _action(
                "legacy-call",
                source_id="tmp-action",
                name="exec_run",
                arguments={"command": "cp $TMPDIR/report.txt out.txt"},
            ),
            turn_id=handle.turn_id,
        )
        self.mem.build_context_projection(provider_profile=OPENAI_PROFILE)
        with self.store._conn:
            self.store._conn.execute(
                """
                UPDATE prompt_projections SET projection_version = 2
                WHERE turn_id = ? AND projection_index = 1
                """,
                ("tmp-turn",),
            )

        report = self.mem.migrate_legacy_path_projections()
        self.assertGreaterEqual(report["scanned_host_tmpdir_rows"], 1)
        self.assertEqual(report["migrated"], 0)

    def test_dry_run_changes_nothing(self) -> None:
        self._build_turn_with_raw_sources("dry-turn")
        self._fabricate_legacy_action_row(
            store=self.store,
            turn_id="dry-turn",
            projection_id="dry-proj",
            source_id="dry-turn-action",
            index=1,
        )
        report = self.mem.migrate_legacy_path_projections(dry_run=True)
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["migrated"], 1)
        row = dict(self.store._conn.execute("SELECT * FROM prompt_projections WHERE projection_id = ?", ("dry-proj",)).fetchone())
        self.assertEqual(int(row["projection_version"]), 2)
        self.assertIn("local path omitted", row["payload_json"])

    def test_migration_survives_store_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "migration.db")
            file_store = SQLiteMemoryStore(db_path)
            file_mem = MemorySystem(
                llm=_NoopLLM(),
                namespace=self.namespace,
                timezone="Asia/Shanghai",
                store=file_store,
                index=InMemoryVectorIndex(embedding=self.embedding),
                embedding=self.embedding,
            )
            handle = file_mem.begin_turn(
                stimuli=[_stimulus("列出目录", source_id="restart-user")],
                turn_id="restart-turn",
            )
            file_mem.append_entry(
                _action(
                    "legacy-call",
                    source_id="restart-action",
                    name="exec_run",
                    arguments={"command": WINDOWS_PATH_COMMAND},
                ),
                turn_id=handle.turn_id,
            )
            file_mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            self._fabricate_legacy_action_row(
                store=file_store,
                turn_id="restart-turn",
                projection_id="restart-proj",
                source_id="restart-action",
                index=1,
            )
            report = file_mem.migrate_legacy_path_projections()
            self.assertEqual(report["migrated"], 1)
            file_mem.close()
            file_store.close()

            reopened_store = SQLiteMemoryStore(db_path)
            reopened_mem = MemorySystem(
                llm=_NoopLLM(),
                namespace=self.namespace,
                timezone="Asia/Shanghai",
                store=reopened_store,
                index=InMemoryVectorIndex(embedding=self.embedding),
                embedding=self.embedding,
            )
            projection = reopened_mem.build_context_projection(provider_profile=OPENAI_PROFILE)
            commands = _projected_tool_commands(projection)
            self.assertIn(WINDOWS_PATH_COMMAND, commands)
            row = dict(
                reopened_store._conn.execute(
                    "SELECT * FROM prompt_projections WHERE projection_id = ?", ("restart-proj",)
                ).fetchone()
            )
            self.assertEqual(int(row["projection_version"]), PROJECTION_VERSION)
            reopened_mem.close()
            reopened_store.close()


if __name__ == "__main__":
    unittest.main()
