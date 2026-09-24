"""Paid-boundary tests with an injected offline transport and temporary ledger."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.research_pilot.budget import BudgetLedger
from examples.research_pilot.preflight import MARKER, MODEL, PreflightError, run_preflight


def reply(*, tools: bool) -> dict:
    message = {"role": "assistant", "content": None, "reasoning_content": ""}
    if tools:
        message["content"] = "Checking the tool."
        message["tool_calls"] = [
            {
                "index": 0,
                "id": "original-call-id",
                "type": "function",
                "function": {"name": "pilot_echo", "arguments": '{ "value" : "' + MARKER + '" }'},
            }
        ]
    else:
        message["content"] = json.dumps({"speech": MARKER, "memory_metadata": {}})
    return {
        "model": MODEL,
        "choices": [{"finish_reason": "tool_calls" if tools else "stop", "message": message}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": 60,
            "total_tokens": 120,
        },
    }


class ProviderPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = BudgetLedger(self.root / "stage.sqlite3", limit_cny="50")
        self.requests = []

    def tearDown(self) -> None:
        self.ledger.close()
        self.temp.cleanup()

    def successful_transport(self, request: dict) -> dict:
        self.requests.append(copy.deepcopy(request))
        return reply(tools=len(self.requests) == 1)

    def run_with(self, transport) -> dict:
        return run_preflight(self.root / "output", self.ledger, {}, transport=transport)

    def test_roundtrip_preserves_entire_assistant_wire_and_does_not_claim_model_scores(self) -> None:
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            report = self.run_with(self.successful_transport)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(self.requests[1]["messages"][-2], reply(tools=True)["choices"][0]["message"])
        self.assertEqual(self.requests[1]["messages"][-1]["tool_call_id"], "original-call-id")
        self.assertEqual(self.requests[1]["thinking"], {"type": "disabled"})
        self.assertIsNone(report["model_scores"])
        self.assertIsNone(report["invoice_cost_cny"])
        self.assertFalse(report["live_transport_verified"])
        self.assertFalse(report["memory_retrieval_verified"])
        self.assertEqual(report["ledger"]["settled_requests"], 2)
        self.assertEqual(report["ledger"]["pending_count"], 0)

    def test_timeout_remains_reserved_and_prevents_another_dispatch(self) -> None:
        def fail(request):
            self.requests.append(request)
            raise PreflightError("transport_failed_or_timed_out")

        report = self.run_with(fail)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["ledger"]["pending_count"], 1)
        second = run_preflight(self.root / "second", self.ledger, {}, transport=fail)
        self.assertEqual(second["status"], "failed")
        self.assertEqual(len(self.requests), 1)
        self.assertIsNone(report["tariff_estimated_cost_cny"])

    def test_missing_usage_is_not_zero_and_stops_before_second_request(self) -> None:
        def missing(request):
            response = self.successful_transport(request)
            response.pop("usage")
            return response

        report = self.run_with(missing)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(report["ledger"]["pending_count"], 1)
        self.assertIsNone(report["requests"][0]["raw_provider_usage"])

    def test_invalid_wire_or_truncation_is_charged_but_not_retried(self) -> None:
        for change in ("bad_finish", "thinking", "unknown_wire"):
            with self.subTest(change=change):

                def bad(request):
                    response = self.successful_transport(request)
                    if change == "bad_finish":
                        response["choices"][0]["finish_reason"] = "length"
                    elif change == "thinking":
                        response["choices"][0]["message"]["reasoning_content"] = "Unexpected reasoning"
                    else:
                        response["choices"][0]["message"]["unhandled_extension"] = "must not silently discard"
                    return response

                before = len(self.requests)
                report = run_preflight(self.root / change, self.ledger, {}, transport=bad)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(self.requests), before + 1)
                self.assertEqual(report["ledger"]["pending_count"], 0)

    def test_unknown_model_cannot_be_settled_at_flash_tariff(self) -> None:
        def other_model(request):
            response = self.successful_transport(request)
            response["model"] = "another-model"
            return response

        report = self.run_with(other_model)
        self.assertEqual(report["error"], "unexpected_returned_model")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(report["ledger"]["pending_count"], 1)
        self.assertEqual(report["ledger"]["settled_requests"], 0)

    def test_insufficient_budget_does_not_dispatch(self) -> None:
        ledger = BudgetLedger(self.root / "small.sqlite3", limit_cny="0.01")
        try:
            report = run_preflight(self.root / "output", ledger, {}, transport=self.successful_transport)
        finally:
            ledger.close()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.requests, [])

    def test_missing_key_does_not_reserve_or_log_a_secret(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            report = run_preflight(self.root / "output", self.ledger, {})
        self.assertEqual(report["error"], "deepseek_api_key_missing")
        self.assertEqual(report["ledger"]["pending_count"], 0)


if __name__ == "__main__":
    unittest.main()
