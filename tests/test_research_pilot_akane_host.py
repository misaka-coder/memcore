"""Offline acceptance of the optional real Akane host dependency."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if __name__ == "__main__" and "--worker" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.research_pilot.akane_host import AkaneHostError, AkaneHostSession, _unix_timestamp


class AkaneHostUnitTests(unittest.TestCase):
    def test_requires_fresh_initialized_process(self):
        with self.assertRaisesRegex(AkaneHostError, "initialize_isolated_akane_required"):
            AkaneHostSession(
                Path("unused"), policy="compact_after_terminal", embedding=None, transport=None, fixture_resolver=None
            )

    def test_local_timestamp_uses_shanghai(self):
        self.assertEqual(_unix_timestamp("2026-09-06T12:00:00"), _unix_timestamp("2026-09-06T04:00:00Z"))
        with self.assertRaises(AkaneHostError):
            _unix_timestamp(None)

    @unittest.skipUnless(os.environ.get("AKANE_TEST_SOURCE_ROOT"), "optional actual Akane checkout not configured")
    def test_real_engine_unlimited_tools_native_schema_repair_and_settlement(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {
                key: value
                for key, value in os.environ.items()
                if key.upper()
                in {
                    "SYSTEMROOT",
                    "WINDIR",
                    "PATH",
                    "COMSPEC",
                    "PATHEXT",
                    "APPDATA",
                    "LOCALAPPDATA",
                    "USERPROFILE",
                    "USERNAME",
                }
            }
            result = subprocess.run(
                [
                    os.environ.get("AKANE_TEST_PYTHON", sys.executable),
                    "-E",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--worker",
                    os.environ["AKANE_TEST_SOURCE_ROOT"],
                    temp,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-3500:])
            report = json.loads((Path(temp) / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(report["guard"]["blocked_network"], 0)
            self.assertEqual(report["guard"]["blocked_reads"], 0)
            self.assertEqual(report["guard"]["blocked_writes"], 0)
            self.assertEqual(len(report["branches"]), 2)
            for row in report["branches"]:
                self.assertEqual(row["requests"], 9)
                self.assertEqual(row["fixture_calls"], 7)
                self.assertEqual(row["speech"], "修复后的完整报告。")
                self.assertTrue(row["repair_prefix_preserved"])
                self.assertTrue(row["optional_fields_preserved"])
                self.assertTrue(row["native_open_memory_optional_fields_dispatch"])
                self.assertTrue(row["raw_readback_matches_observation"])
                self.assertEqual(row["compaction_generation"], 0)
                self.assertEqual(row["assistant_sources"], 1)
            full, card = report["branches"]
            self.assertTrue(full["retains_full_fixture"])
            self.assertFalse(card["retains_full_fixture"])
            self.assertTrue(card["has_reload_card"])
            self.assertTrue(report["checks"]["actual_plaintext_terminal_recovery"])


def _worker(source: Path, output: Path) -> None:
    import time

    from examples.research_pilot.akane_host import initialize_isolated_akane

    initialized = initialize_isolated_akane(source, output)
    if os.name == "nt":
        import getpass

        # PyTorch's cache initialization needs Windows identity resolution;
        # the identity itself must never enter the captured evidence.
        assert getpass.getuser() == os.environ["USERNAME"]
    import httpx
    from memcore import HashedEmbeddingProvider

    final = json.dumps(
        {
            "emotion": "normal",
            "speech": "修复后的完整报告。",
            "tool_call": None,
            "memory_metadata": {
                "entity_anchors": [],
                "topic_terms": [],
                "memory_facets": [],
                "about_roles": [],
                "retrieval_priority": "normal",
            },
        },
        ensure_ascii=False,
    )
    fixture_body = "fixture-long-body-only-in-full-projection:" + "甲乙丙丁" * 1500

    class OfflineSDKTransport:
        def __init__(self):
            self.requests = []
            self.open_source_id = ""

        def attach_runtime(self, runtime):
            for role in ("chat", "aux", "memcore_summary", "vision"):
                bundle = getattr(runtime, role, None)
                if bundle is None or bundle.client is None:
                    continue
                bundle.client.max_retries = 0

                def send(request, *args, **kwargs):
                    wire = json.loads(request.content)
                    self.requests.append(wire)
                    number = len(self.requests)
                    if number > 11:
                        raise AssertionError("unexpected extra actual Engine request")
                    if number == 10:
                        message = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "actual-readback",
                                    "type": "function",
                                    "function": {
                                        "name": "open_memory",
                                        "arguments": json.dumps({"memory_id": self.open_source_id, "view": "content"}),
                                    },
                                }
                            ],
                        }
                        finish = "tool_calls"
                    elif number <= 7:
                        message = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"fixture-call-{number}",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_fixture",
                                        "arguments": json.dumps({"query_key": "source-v1"}),
                                    },
                                }
                            ],
                        }
                        finish = "tool_calls"
                    else:
                        message = {
                            "role": "assistant",
                            "content": ("报告前缀不应丢失。\n" if number == 8 else "") + final,
                        }
                        finish = "stop"
                    body = {
                        "id": f"offline-{number}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "deepseek-v4-flash",
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    }
                    return httpx.Response(200, request=request, json=body)

                bundle.client._client.send = send
            return ("chat", "aux", "memcore_summary")

    branches = []
    timestamp = int(time.time())
    for policy in ("full_until_raw_compaction", "compact_after_terminal"):
        transport = OfflineSDKTransport()
        host = AkaneHostSession(
            output / policy,
            policy=policy,
            embedding=HashedEmbeddingProvider(dimension=64),
            transport=transport,
            fixture_resolver=lambda step, args: {"ok": True, "text": fixture_body},
        )
        try:
            result = host.process_turn(
                {
                    "step_id": "offline-acceptance",
                    "user_text": "读取 source-v1 后给我完整报告。",
                    "timestamp": timestamp,
                }
            )
            tools = transport.requests[0]["tools"]
            retrieve = next(tool["function"] for tool in tools if tool["function"]["name"] == "retrieve_memory")
            payloads = json.dumps([row["payload"] for row in result["projection"]], ensure_ascii=False)
            old_messages = transport.requests[-2]["messages"]
            new_messages = transport.requests[-1]["messages"]
            repair_preserved = len(new_messages) == len(old_messages) + 1 and any(
                new_messages[:index] + new_messages[index + 1 :] == old_messages
                and new_messages[index].get("role") == "user"
                for index in range(1, len(new_messages))
            )
            initial_request_count = len(transport.requests)
            transport.open_source_id = next(
                row["source_id"] for row in result["memory"] if row["turn_role"] == "observation"
            )
            reopened = host.process_turn(
                {"step_id": "offline-readback", "user_text": "打开刚才的完整资料。", "timestamp": timestamp + 1}
            )
            readback_messages = [
                row
                for row in transport.requests[-1]["messages"]
                if row.get("role") == "tool" and row.get("tool_call_id") == "actual-readback"
            ]
            readback_full = len(readback_messages) == 1 and fixture_body in readback_messages[0]["content"]
            raw_readback = json.JSONDecoder().raw_decode(readback_messages[0]["content"].split("\ndata:\n", 1)[1])[0]
            exact_readback = raw_readback["output"] == json.dumps(
                {"ok": True, "text": fixture_body}, ensure_ascii=False, separators=(",", ":")
            )
            (output / (policy + "-evidence.json")).write_text(
                json.dumps(
                    {
                        "requests": transport.requests,
                        "result": result,
                        "readback_result": reopened,
                        "evidence": host.evidence(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            branches.append(
                {
                    "policy": policy,
                    "requests": initial_request_count,
                    "fixture_calls": len(result["tool_events"]),
                    "speech": result["output"]["speech"],
                    "repair_prefix_preserved": repair_preserved,
                    "native_open_memory_optional_fields_dispatch": readback_full,
                    "raw_readback_matches_observation": exact_readback,
                    "optional_fields_preserved": "entity_anchors" not in retrieve["parameters"]["required"],
                    "compaction_generation": result["metrics"]["compaction_generation"],
                    "assistant_sources": sum(row["kind"] == "message.assistant" for row in result["memory"]),
                    "retains_full_fixture": fixture_body in payloads,
                    "has_reload_card": "source_id" in payloads and "open_memory" in payloads,
                }
            )
        finally:
            host.close()

    class PlaintextRecoveryTransport:
        def __init__(self):
            self.requests = []

        def attach_runtime(self, runtime):
            for role in ("chat", "aux", "memcore_summary", "vision"):
                bundle = getattr(runtime, role, None)
                if bundle is None or bundle.client is None:
                    continue
                bundle.client.max_retries = 0

                def send(request, *args, **kwargs):
                    wire = json.loads(request.content)
                    self.requests.append(wire)
                    number = len(self.requests)
                    if number > 4:
                        raise AssertionError("unexpected plaintext recovery request")
                    content = "始终违规的JSON前缀。\n" + final if number <= 3 else "纯文本尾路完整回复。"
                    body = {
                        "id": f"offline-plaintext-{number}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "deepseek-v4-flash",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    }
                    return httpx.Response(200, request=request, json=body)

                bundle.client._client.send = send
            return ("chat", "aux", "memcore_summary")

    plain_transport = PlaintextRecoveryTransport()
    plain_host = AkaneHostSession(
        output / "plaintext-recovery",
        policy="compact_after_terminal",
        embedding=HashedEmbeddingProvider(dimension=64),
        transport=plain_transport,
        fixture_resolver=lambda step, args: {"ok": False},
    )
    try:
        plain_result = plain_host.process_turn(
            {"step_id": "offline-plaintext-recovery", "user_text": "直接给我最终回复。", "timestamp": int(time.time())}
        )
        plain_evidence = plain_host.evidence()
        plain_passed = (
            len(plain_transport.requests) == 4
            and plain_result["output"]["speech"] == "纯文本尾路完整回复。"
            and sum(
                row["kind"] == "message.assistant" and row["turn_role"] == "final" for row in plain_result["memory"]
            )
            == 1
            and not plain_transport.requests[-1].get("tools")
        )
        (output / "plaintext-recovery-evidence.json").write_text(
            json.dumps(
                {"requests": plain_transport.requests, "result": plain_result, "evidence": plain_evidence},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        plain_host.close()
    evidence = plain_host.evidence()
    source_hashes = {
        "akane/" + key: value
        for key, value in {**evidence["loaded_akane_source_hashes"], **evidence["template_source_hashes"]}.items()
    }
    for name in ("akane_host.py", "akane_transport.py"):
        path = Path(__file__).resolve().parents[1] / "examples/research_pilot" / name
        source_hashes["memcore/examples/research_pilot/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = {
        "actual_engine_control_methods_unmodified": len(evidence["engine_methods"]) == 9,
        "actual_default_tool_round_unlimited": evidence["tool_round_hard_limit"] == 0,
        "more_than_six_actual_tool_rounds": all(row["requests"] == 9 and row["fixture_calls"] == 7 for row in branches),
        "native_optional_schema_preserved": all(row["optional_fields_preserved"] for row in branches),
        "native_open_memory_optional_fields_dispatch": all(
            row["native_open_memory_optional_fields_dispatch"] for row in branches
        ),
        "exact_raw_readback": all(row["raw_readback_matches_observation"] for row in branches),
        "same_context_json_repair": all(
            row["repair_prefix_preserved"] and row["speech"] == "修复后的完整报告。" for row in branches
        ),
        "actual_plaintext_terminal_recovery": plain_passed,
        "terminal_commit_once": all(row["assistant_sources"] == 1 for row in branches),
        "natural_maintenance_no_raw_compaction": all(row["compaction_generation"] == 0 for row in branches),
        "full_projection_preserved": branches[0]["retains_full_fixture"],
        "terminal_card_reloads": not branches[1]["retains_full_fixture"] and branches[1]["has_reload_card"],
        "no_network_or_outside_file_access": all(
            initialized["audit_guard"][key] == 0
            for key in ("blocked_network", "authorized_network_events", "blocked_reads", "blocked_writes")
        ),
    }
    (output / "acceptance.json").write_text(
        json.dumps(
            {
                "status": "passed" if all(checks.values()) else "failed",
                "actual_engine_exercised": True,
                "test_mode": "offline_actual_engine_fake_sdk_http_TEST_ONLY_hashed_embedding",
                "checks": checks,
                "source_hashes": source_hashes,
                "branches": branches,
                "guard": initialized["audit_guard"],
                "limitations": [
                    "Subprocesses remain blocked after guard installation; any denied attempts are counted in guard.blocked_subprocesses.",
                    "This gate uses fake HTTP and a test-only hashed embedding. Paid model and production embedding validation are separate.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
