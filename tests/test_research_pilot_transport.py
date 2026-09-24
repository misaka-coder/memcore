"""Offline verification of paid transport accounting and wire capture boundaries."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from examples.research_pilot.budget import BudgetLedger
from examples.research_pilot.preflight import MODEL, PreflightError, base_request
from examples.research_pilot.transport import BudgetedTransport, TransportError


def request() -> dict:
    value = base_request(
        [{"role": "user", "content": "Fixed synthetic test question."}],
        [{"type": "function", "function": {"name": "lookup_fixture", "parameters": {"type": "object"}}}],
    )
    value["tool_choice"] = "auto"
    return value


def reply() -> dict:
    return {
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"speech":"Synthetic answer","memory_metadata":{}}'},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": 60,
            "total_tokens": 120,
        },
    }


class BudgetedTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = BudgetLedger(self.root / "ledger.sqlite3")

    def tearDown(self) -> None:
        self.ledger.close()
        self.temp.cleanup()

    def transport(self, send, **kwargs) -> BudgetedTransport:
        return BudgetedTransport(self.ledger, capture_dir=self.root / "captures", transport=send, **kwargs)

    def test_success_preserves_wire_and_separates_peak_ledger_from_time_tariff(self) -> None:
        send = Mock(return_value=reply())
        client = self.transport(send)
        stamp = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)  # Sunday: off-peak.
        with patch("examples.research_pilot.transport.datetime") as clock:
            clock.now.return_value = stamp
            response = client.call(request(), scope="gallery_full", step_id="turn_1")
        self.assertEqual(response, reply())
        self.assertEqual(send.call_args.args[0], request())
        row = client.requests[0]
        self.assertEqual(row["run_mode"], "offline_transport_test")
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["request"], request())
        self.assertEqual(row["response"], reply())
        self.assertEqual(row["raw_provider_usage"], reply()["usage"])
        self.assertEqual(Decimal(row["reserved_cny"]), Decimal("3.16416"))
        peak = Decimal(row["settlement"]["usage_cost_at_reserved_rates_cny"])
        self.assertEqual(Decimal(row["tariff_estimated_cost_cny"]), peak / 2)
        self.assertEqual(row["token_count_quality"], "estimated")
        self.assertEqual(row["provider_input_reservation_tokens"], 1048576)
        self.assertIsNone(row["invoice_cost_cny"])
        self.assertGreaterEqual(row["total_latency_ms"], 0)
        saved = json.loads((self.root / "captures" / row["capture_file"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, row)
        self.assertNotIn(str(self.root), json.dumps(saved))

    def test_multiple_native_calls_retain_ids_argument_whitespace_and_content(self) -> None:
        response = reply()
        response["choices"][0] = {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "Checking two synthetic resources.",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"wire-call-{index}",
                        "type": "function",
                        "function": {"name": "lookup_fixture", "arguments": '{ "value" : "synthetic" }'},
                    }
                    for index in range(2)
                ],
            },
        }
        client = self.transport(Mock(return_value=response))
        actual = client.call(request(), scope="gallery_card", step_id="turn_1")
        self.assertEqual(actual, response)
        self.assertEqual(client.requests[0]["response"], response)

    def test_unknown_model_missing_usage_and_timeout_keep_reservation_and_never_retry(self) -> None:
        for case in ("model", "usage", "timeout"):
            with self.subTest(case=case):
                ledger = BudgetLedger(self.root / f"{case}.sqlite3")
                response = reply()
                if case == "model":
                    response["model"] = "unverified-model"
                elif case == "usage":
                    response.pop("usage")
                send = Mock(
                    return_value=response,
                    side_effect=TimeoutError("private provider failure") if case == "timeout" else None,
                )
                try:
                    client = BudgetedTransport(ledger, capture_dir=self.root / case, transport=send)
                    with self.assertRaises(TransportError):
                        client.call(request(), scope="scenario", step_id="first")
                    with self.assertRaisesRegex(TransportError, "pending_reservation_blocks_dispatch"):
                        client.call(request(), scope="other_scenario", step_id="second")
                    self.assertEqual(send.call_count, 1)
                    self.assertEqual(ledger.snapshot()["pending_count"], 1)
                    self.assertEqual(ledger.snapshot()["settled_requests"], 0)
                    self.assertNotIn("private provider failure", json.dumps(client.requests))
                finally:
                    ledger.close()

    def test_protocol_failure_is_charged_and_independent_next_scenario_can_run(self) -> None:
        bad = reply()
        bad["choices"][0]["finish_reason"] = "length"
        send = Mock(side_effect=[bad, reply()])
        client = self.transport(send)
        with self.assertRaisesRegex(TransportError, "unexpected_finish_reason"):
            client.call(request(), scope="first_scenario", step_id="turn_1")
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)
        client.call(request(), scope="second_scenario", step_id="turn_1")
        self.assertEqual(send.call_count, 2)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 2)

    def test_final_json_contract_is_left_for_live_session(self) -> None:
        response = reply()
        response["choices"][0]["message"]["content"] = "not valid final JSON"
        client = self.transport(Mock(return_value=response))
        wire = request()
        wire["response_format"] = {"type": "json_object"}
        self.assertEqual(client.call(wire, scope="scenario", step_id="final"), response)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)

    def test_actual_prompt_budget_error_is_settled_before_raising(self) -> None:
        response = reply()
        response["usage"].update(prompt_tokens=1001, prompt_cache_miss_tokens=961, total_tokens=1021)
        client = self.transport(Mock(return_value=response), input_token_budget=1000)
        with self.assertRaisesRegex(TransportError, "actual_input_token_budget_exceeded"):
            client.call(request(), scope="scenario", step_id="too_large")
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)

    def test_estimated_budget_and_generation_mismatch_do_not_reserve_or_dispatch(self) -> None:
        send = Mock(return_value=reply())
        client = self.transport(send, input_token_budget=1000)
        large = request()
        large["messages"][0]["content"] = "x" * 5000
        with self.assertRaisesRegex(TransportError, "estimated_input_token_budget_exceeded"):
            client.call(large, scope="scenario", step_id="too_large")
        for field, value in (("model", "other"), ("thinking", {"type": "enabled"}), ("max_tokens", 4096)):
            wrong = copy.deepcopy(request())
            wrong[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(TransportError, "generation_settings_mismatch"):
                client.call(wrong, scope="scenario", step_id="invalid")
        send.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)

    def test_http_failure_and_error_response_never_capture_body_or_credentials(self) -> None:
        client = self.transport(Mock(side_effect=PreflightError("private HTTP body with secret")))
        with self.assertRaisesRegex(TransportError, "transport_failed_or_timed_out"):
            client.call(request(), scope="scenario", step_id="http_failure")
        self.assertNotIn("private HTTP body", json.dumps(client.requests))
        ledger = BudgetLedger(self.root / "error.sqlite3")
        try:
            client = BudgetedTransport(
                ledger,
                capture_dir=self.root / "errors",
                transport=Mock(return_value={"error": {"message": "sensitive HTTP error body"}}),
            )
            with self.assertRaisesRegex(TransportError, "provider_error_response"):
                client.call(request(), scope="scenario", step_id="error_object")
            self.assertIsNone(client.requests[0]["response"])
            self.assertNotIn("sensitive", json.dumps(client.requests))
        finally:
            ledger.close()

    def test_default_transport_checks_current_pricing_before_reserving(self) -> None:
        with patch("examples.research_pilot.transport.official_transport") as send:
            client = BudgetedTransport(self.ledger, capture_dir=self.root / "official")
            with patch("examples.research_pilot.transport.PRICING_CHECKED", "1900-01-01"):
                with self.assertRaisesRegex(TransportError, "pricing_requires_fresh_verification"):
                    client.call(request(), scope="scenario", step_id="stale_pricing")
        send.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)
        self.assertEqual(client.requests[0]["run_mode"], "paid_scenario_request")

    def test_credential_field_is_rejected_without_capture_or_dispatch(self) -> None:
        send = Mock(return_value=reply())
        client = self.transport(send)
        wire = request()
        wire["api_key"] = "secret-sentinel-never-capture"
        with self.assertRaisesRegex(TransportError, "unsupported_request_fields"):
            client.call(wire, scope="scenario", step_id="invalid")
        send.assert_not_called()
        self.assertIsNone(client.requests[0]["request"])
        self.assertNotIn("secret-sentinel", json.dumps(client.requests))

    def test_insufficient_or_closed_ledger_never_dispatches_and_uses_transport_error(self) -> None:
        ledger = BudgetLedger(self.root / "small.sqlite3", limit_cny="0.1")
        send = Mock(return_value=reply())
        client = BudgetedTransport(ledger, capture_dir=self.root / "small", transport=send)
        try:
            with self.assertRaisesRegex(TransportError, "budget_limit_exceeded"):
                client.call(request(), scope="scenario", step_id="over_budget")
        finally:
            ledger.close()
        with self.assertRaisesRegex(TransportError, "budget_ledger_closed"):
            client.call(request(), scope="scenario", step_id="closed")
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
