"""Receipt contrasts fail closed; optional worker exercises the actual Engine."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__" and "--worker" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.research_pilot.akane_empty_retrieval import normalized_continuation
from examples.research_pilot.akane_transport import AkaneTransportInterrupted


class EmptyContrastTests(unittest.TestCase):
    def test_only_bound_guidance_is_normalized_and_evidence_is_not_rewritten(self):
        old = "【MemCore 语义检索状态】\n- status=empty；returned=0\n\nold guidance"
        new = old.replace("old guidance", "new guidance")
        wire = {
            "messages": [
                {"role": "user", "content": "A specific visible fact"},
                {"role": "tool", "tool_call_id": "fixed-choice", "content": old},
            ]
        }
        original = copy.deepcopy(wire)
        left = normalized_continuation(wire, call_id="fixed-choice", expected_body=old, guidance="old guidance")
        wire["messages"][1]["content"] = new
        right = normalized_continuation(wire, call_id="fixed-choice", expected_body=new, guidance="new guidance")
        self.assertEqual(left, right)
        self.assertEqual(left["messages"][0], original["messages"][0])
        wire["messages"][0]["content"] = "Different visible evidence"
        self.assertNotEqual(
            left, normalized_continuation(wire, call_id="fixed-choice", expected_body=new, guidance="new guidance")
        )

    def test_missing_duplicate_or_changed_tool_body_is_rejected(self):
        body = "status=empty；returned=0\n\nguidance"
        message = {"role": "tool", "tool_call_id": "fixed-choice", "content": body}
        for messages in ([], [message, message], [{**message, "content": body + " changed"}]):
            with self.subTest(messages=messages), self.assertRaises(AkaneTransportInterrupted):
                normalized_continuation(
                    {"messages": messages}, call_id="fixed-choice", expected_body=body, guidance="guidance"
                )

    @unittest.skipUnless(os.environ.get("AKANE_TEST_SOURCE_ROOT"), "optional actual Akane source not configured")
    def test_actual_engine_fixed_choice_then_continuation_and_original_repair(self):
        with tempfile.TemporaryDirectory() as temp:
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
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=150,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            gate = json.loads((Path(temp) / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "passed")
            self.assertEqual(gate["real_network_calls"], 0)
            self.assertEqual(gate["conditioned_branches"], 10)
            self.assertEqual(gate["replay_responses"], 80)
            self.assertEqual(gate["mock_http_requests"], 11)
            self.assertTrue(gate["actual_final_repair_exercised"])


def _worker(source: Path, output: Path) -> None:
    import time

    from examples.research_pilot.akane_empty_retrieval import PrefixReplayTransport, load_inputs, run_contrast
    from examples.research_pilot.akane_host import initialize_isolated_akane
    from examples.research_pilot.akane_provenance import freeze_sources, verify_sources
    from examples.research_pilot.budget import BudgetLedger
    from examples.research_pilot.runner import write_json

    root = Path(__file__).resolve().parents[1]
    pack = root / "docs/research/pilot_v3"
    inputs = load_inputs(pack, root / ".research-runs/akane-retest-live-20260906-attempt02")
    evidence = freeze_sources(root, source, pack, output / "source")
    initialized = initialize_isolated_akane(source, output)
    import httpx
    from memcore import HashedEmbeddingProvider

    class TestOnlyEmbedding:
        name = "TEST_ONLY_deterministic_embedding"
        dimension = 1024
        delegate = HashedEmbeddingProvider(dimension=1024)
        embed_text = delegate.embed_text
        embed_texts = delegate.embed_texts

    count = 0
    final = json.dumps({"emotion": "normal", "speech": "离线流程验证：蓝色笔记本，周四晚上。"}, ensure_ascii=False)

    def respond(request):
        nonlocal count
        count += 1
        if count > 11:
            raise RuntimeError("unexpected_extra_mock_http")
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"TEST-ONLY-{count}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": ("前缀须由真实宿主修复。\n" if count == 1 else "") + final,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 40,
                    "prompt_cache_miss_tokens": 60,
                    "total_tokens": 120,
                },
            },
        )

    class OfflineTransport(PrefixReplayTransport):
        def attach_client(self, client, *, role="chat"):
            client._client._transport = httpx.MockTransport(respond)
            client._client._mounts = {}
            return super().attach_client(client, role=role)

    with (
        BudgetLedger(output / "TEST-ONLY-budget.sqlite3") as ledger,
        patch("examples.research_pilot.akane_transport.datetime") as clock,
    ):
        clock.now.return_value = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
        transport = OfflineTransport(
            ledger, capture_dir=output / "requests", api_key_provider=lambda: "TEST-ONLY-not-a-real-key"
        )
        result = run_contrast(
            inputs,
            output / "run",
            transport=transport,
            embedding=TestOnlyEmbedding(),
            source_check=lambda: verify_sources(evidence, root, source, pack),
            source_evidence=evidence,
            secret="TEST-ONLY-not-a-real-key",
            embedding_evidence={"mode": "TEST_ONLY_deterministic_not_production"},
        )
    gate = {
        "status": result["status"],
        "test_mode": "actual_engine_with_mock_http_and_TEST_ONLY_ledger",
        "actual_engine_exercised": True,
        "real_network_calls": 0,
        "conditioned_branches": len(result["runs"]),
        "replay_responses": sum(row["run_mode"] == "replay" for row in transport.requests),
        "mock_http_requests": count,
        "actual_final_repair_exercised": count == 11 and result["runs"][0].get("probe", {}).get("paid_requests") == 2,
        "source_evidence": evidence,
        "guard": initialized["audit_guard"],
    }
    write_json(output / "acceptance.json", gate)
    assert result["status"] == "passed", [(row["run_id"], row["error"]) for row in result["runs"]]
    assert all(row["baseline_comparison"]["normalized_equal"] for row in result["runs"])
    assert all(row["first_paid_continuation_control_passed"] for row in result["runs"])
    assert all(row["host_close"]["status"] == "stopped" for row in result["runs"])
    assert gate["actual_final_repair_exercised"]
    assert not any(gate["guard"].values())


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker(Path(sys.argv[-2]).resolve(), Path(sys.argv[-1]).resolve())
    else:
        unittest.main()
