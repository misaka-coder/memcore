"""Real OpenAI serializers with mocked HTTP; these tests make no paid calls."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from examples.research_pilot.akane_transport import AkaneBudgetedTransport, AkaneTransportInterrupted
from examples.research_pilot.budget import BudgetLedger
from examples.research_pilot.preflight import MODEL

try:
    import httpx
    from openai import OpenAI
except ImportError:
    httpx = None
    OpenAI = None


def request(*, stream: bool = False, text: str = "Synthetic question") -> dict:
    value = {
        "model": MODEL,
        "messages": [{"role": "system", "content": "Original synthetic persona"}, {"role": "user", "content": text}],
        "temperature": 0.8,
        "max_tokens": 2048,
        "extra_body": {"thinking": {"type": "disabled"}},
        "tools": [{"type": "function", "function": {"name": "host_lookup", "parameters": {"type": "object"}}}],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
    }
    if stream:
        value.update(stream=True, stream_options={"include_usage": True})
    return value


def reply() -> dict:
    return {
        "id": "completion-synthetic",
        "created": 1,
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Synthetic answer"}}
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": 60,
            "total_tokens": 120,
        },
    }


def sse(*, usage: bool = True, done: bool = True) -> bytes:
    source = reply()
    events = [
        {
            "id": source["id"],
            "created": 1,
            "object": "chat.completion.chunk",
            "model": MODEL,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Synthetic answer"}, "finish_reason": None}
            ],
        },
        {
            "id": source["id"],
            "created": 1,
            "object": "chat.completion.chunk",
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    if usage:
        events.append(
            {
                "id": source["id"],
                "created": 1,
                "object": "chat.completion.chunk",
                "model": MODEL,
                "choices": [],
                "usage": source["usage"],
            }
        )
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\r\n\r\n" for event in events)
    return body + (b"data: [DONE]\r\n\r\n" if done else b"")


@unittest.skipIf(httpx is None or OpenAI is None, "optional Akane HTTP/SDK dependencies are unavailable")
class AkaneTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = BudgetLedger(self.root / "ledger.sqlite3")
        self.key = Mock(return_value="synthetic-test-key-never-capture")
        self.clients = []
        self.clock = patch("examples.research_pilot.akane_transport.datetime")
        self.clock.start().now.return_value = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        for client in self.clients:
            client.close()
        self.clock.stop()
        self.ledger.close()
        self.temp.cleanup()

    def adapter(self, **kwargs) -> AkaneBudgetedTransport:
        return AkaneBudgetedTransport(
            self.ledger, capture_dir=self.root / "captures", api_key_provider=self.key, **kwargs
        )

    def client(self, adapter, handler, *, role="chat"):
        client = OpenAI(
            api_key="synthetic-unused-placeholder",
            base_url="https://api.deepseek.com",
            max_retries=4,
            http_client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
        )
        self.clients.append(client)
        adapter.attach_client(client, role=role)
        return client

    @staticmethod
    def streamed(body: bytes, *, chunk_size: int = 13):
        class Chunks(httpx.SyncByteStream):
            def __iter__(self):
                for start in range(0, len(body), chunk_size):
                    yield body[start : start + chunk_size]

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Chunks())

    def test_actual_sdk_wire_is_untouched_and_every_dispatch_is_reserved(self) -> None:
        wire_requests = []

        def handler(wire):
            self.assertEqual(self.ledger.snapshot()["pending_count"], 1)
            self.assertEqual(wire.headers["authorization"], "Bearer synthetic-test-key-never-capture")
            wire_requests.append(wire.content)
            return httpx.Response(200, json=reply())

        adapter = self.adapter()
        client = self.client(adapter, handler)
        self.assertEqual(client.max_retries, 0)
        self.assertFalse(client._client.follow_redirects)
        with adapter.scope("real_host", "turn_1"):
            actual = client.chat.completions.create(**request())
        self.assertEqual(actual.choices[0].message.content, "Synthetic answer")
        row = adapter.requests[0]
        expected = request()
        expected.update(expected.pop("extra_body"))
        self.assertEqual(row["request"], expected)
        self.assertEqual(wire_requests[0], row["request_body_utf8"].encode())
        self.assertEqual(row["request_body_sha256"], hashlib.sha256(wire_requests[0]).hexdigest())
        self.assertEqual(row["generation_settings"]["temperature"], 0.8)
        self.assertEqual(row["scope"], "real_host")
        self.assertEqual(row["step_id"], "turn_1")
        self.assertEqual(row["response"], reply())
        self.assertEqual(row["raw_provider_usage"], reply()["usage"])
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["reserved_cny"], "3.16416")
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)
        saved = json.loads((self.root / "captures" / row["capture_file"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, row)
        self.assertNotIn("synthetic-test-key", json.dumps(saved))
        self.assertNotIn(str(self.root), json.dumps(saved))

    def test_stream_uses_real_sdk_chunks_and_settles_complete_usage(self) -> None:
        body = sse()
        adapter = self.adapter()
        client = self.client(adapter, lambda _: self.streamed(body))
        chunks = list(client.chat.completions.create(**request(stream=True)))
        self.assertEqual(chunks[0].choices[0].delta.content, "Synthetic answer")
        row = adapter.requests[0]
        self.assertEqual(row["status"], "passed")
        self.assertEqual(base64.b64decode(row["response_body_base64"]), body)
        self.assertEqual(row["raw_provider_usage"], reply()["usage"])
        self.assertEqual(row["response_end_reason"], "provider_done")
        self.assertGreaterEqual(row["first_token_latency_ms"], 0)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)

    def test_infrastructure_default_output_preserves_wire_and_reserves_provider_maximum(self) -> None:
        from decimal import Decimal

        received = []
        payload = reply()
        payload["usage"].update(completion_tokens=3200, total_tokens=3300)

        def handler(wire):
            self.assertEqual(self.ledger.snapshot()["pending_count"], 1)
            self.assertGreater(Decimal(self.ledger.snapshot()["reserved_cny"]), Decimal("6"))
            received.append(json.loads(wire.content))
            return httpx.Response(200, json=payload)

        adapter = self.adapter(provider_default_output_roles=("memcore_summary",))
        client = self.client(adapter, handler, role="memcore_summary")
        actual = request()
        del actual["max_tokens"]
        client.chat.completions.create(**actual)
        self.assertNotIn("max_tokens", received[0])
        self.assertEqual(received[0], adapter.requests[0]["request"])
        self.assertEqual(adapter.requests[0]["output_reservation_tokens"], 393216)
        self.assertEqual(adapter.requests[0]["settlement"]["usage"]["completion_tokens"], 3200)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)

    def test_default_output_permission_does_not_broaden_chat_or_streaming(self) -> None:
        handler = Mock(side_effect=lambda _: httpx.Response(200, json=reply()))
        adapter = self.adapter(provider_default_output_roles=("memcore_summary",), require_non_streaming=True)
        client = self.client(adapter, handler)
        missing = request()
        del missing["max_tokens"]
        with self.assertRaisesRegex(AkaneTransportInterrupted, "generation_settings_mismatch"):
            client.chat.completions.create(**missing)
        with self.assertRaisesRegex(AkaneTransportInterrupted, "long_experiment_requires_non_streaming"):
            client.chat.completions.create(**request(stream=True))
        handler.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 0)

    def test_long_transport_serializes_chat_and_background_reservations(self) -> None:
        import threading
        import time

        from examples.research_pilot.akane_long_observation import LongRunTransport

        errors = []
        adapter = LongRunTransport(
            self.ledger, run_id="long-run", capture_dir=self.root / "long-captures", api_key_provider=self.key
        )
        self.assertEqual(adapter.input_token_budget, 65536)
        self.assertEqual(self.adapter().input_token_budget, 30720)
        with self.assertRaisesRegex(AkaneTransportInterrupted, "invalid_input_token_budget"):
            self.adapter(input_token_budget=65537)
        adapter.current_step = "step-1"

        def handler(_):
            self.assertEqual(self.ledger.snapshot()["pending_count"], 1)
            time.sleep(0.01)
            return httpx.Response(200, json=reply())

        clients = [self.client(adapter, handler, role=role) for role in ("chat", "memcore_summary")]

        def send(client):
            try:
                client.chat.completions.create(**request())
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=send, args=(client,)) for client in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 2)
        self.assertEqual({(row["scope"], row["step_id"]) for row in adapter.requests}, {("long-run", "step-1")})

    def test_missing_stream_usage_holds_reservation_and_blocks_next_dispatch(self) -> None:
        adapter = self.adapter()
        send = Mock(side_effect=lambda _: self.streamed(sse(usage=False)))
        client = self.client(adapter, send)
        with self.assertRaisesRegex(AkaneTransportInterrupted, "provider_usage_missing"):
            list(client.chat.completions.create(**request(stream=True)))
        with self.assertRaisesRegex(AkaneTransportInterrupted, "pending_reservation_blocks_dispatch"):
            client.chat.completions.create(**request())
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 1)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 0)

    def test_early_stream_close_keeps_unknown_charge_reserved(self) -> None:
        adapter = self.adapter()
        client = self.client(adapter, lambda _: self.streamed(sse()))
        stream = client.chat.completions.create(**request(stream=True))
        self.assertEqual(next(stream).choices[0].delta.content, "Synthetic answer")
        with self.assertRaisesRegex(AkaneTransportInterrupted, "provider_usage_missing"):
            stream.close()
        self.assertEqual(self.ledger.snapshot()["pending_count"], 1)
        self.assertEqual(adapter.requests[0]["status"], "interrupted")

    def test_http_error_timeout_or_unknown_model_never_retry_or_expose_error(self) -> None:
        for case in ("http", "timeout", "model"):
            with self.subTest(case=case):
                ledger = BudgetLedger(self.root / f"{case}.sqlite3")
                bad = reply()
                bad["model"] = "unknown-model"
                handler = Mock(
                    side_effect=TimeoutError("private HTTP error detail") if case == "timeout" else None,
                    return_value=httpx.Response(503, text="private HTTP error detail")
                    if case == "http"
                    else httpx.Response(200, json=bad),
                )
                adapter = AkaneBudgetedTransport(ledger, capture_dir=self.root / case, api_key_provider=self.key)
                client = self.client(adapter, handler)
                try:
                    with self.assertRaises(AkaneTransportInterrupted):
                        client.chat.completions.create(**request())
                    with self.assertRaisesRegex(AkaneTransportInterrupted, "pending_reservation_blocks_dispatch"):
                        client.chat.completions.create(**request())
                    self.assertEqual(handler.call_count, 1)
                    self.assertEqual(ledger.snapshot()["pending_count"], 1)
                    self.assertNotIn("private HTTP error detail", json.dumps(adapter.requests))
                finally:
                    ledger.close()

    def test_host_recovery_and_more_than_six_calls_are_independently_accounted(self) -> None:
        shortened = reply()
        shortened["choices"][0]["finish_reason"] = "length"
        send = Mock(
            side_effect=[httpx.Response(200, json=shortened)] + [httpx.Response(200, json=reply()) for _ in range(7)]
        )
        adapter = self.adapter()
        client = self.client(adapter, send)
        for index in range(8):
            payload = request()
            if index:
                payload["temperature"] = 0.2  # Actual host recovery configuration is retained.
            response = client.chat.completions.create(**payload)
            self.assertEqual(response.choices[0].finish_reason, "length" if not index else "stop")
        self.assertEqual(send.call_count, 8)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 8)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)

    def test_context_excess_before_send_is_free_and_after_usage_is_settled(self) -> None:
        adapter = self.adapter(input_token_budget=300)
        send = Mock(return_value=httpx.Response(200, json=reply()))
        client = self.client(adapter, send)
        with self.assertRaisesRegex(AkaneTransportInterrupted, "estimated_input_token_budget_exceeded"):
            client.chat.completions.create(**request(text="large synthetic input " * 100))
        send.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)
        large_usage = reply()
        large_usage["usage"].update(prompt_tokens=301, prompt_cache_miss_tokens=261, total_tokens=321)
        send.return_value = httpx.Response(200, json=large_usage)
        with self.assertRaisesRegex(AkaneTransportInterrupted, "actual_input_token_budget_exceeded"):
            client.chat.completions.create(**request())
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)

    def test_usage_overrun_retains_exact_response_and_keeps_reservation_blocked(self) -> None:
        response = reply()
        response["usage"].update(completion_tokens=2049, total_tokens=2149)
        response["choices"][0]["finish_reason"] = "length"
        body = json.dumps(response, ensure_ascii=False, indent=2).encode("utf-8")
        send = Mock(return_value=httpx.Response(200, content=body, headers={"content-type": "application/json"}))
        adapter = self.adapter()
        client = self.client(adapter, send)
        with self.assertRaisesRegex(AkaneTransportInterrupted, "usage_tokens_exceed_reservation"):
            client.chat.completions.create(**request())
        row = adapter.requests[0]
        saved = json.loads((self.root / "captures" / row["capture_file"]).read_text(encoding="utf-8"))
        self.assertIn("response_body_base64", saved)
        self.assertEqual(base64.b64decode(saved["response_body_base64"]), body)
        self.assertEqual(saved["response_body_sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(saved["raw_provider_usage"], response["usage"])
        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(saved["error"], "usage_tokens_exceed_reservation")
        self.assertIsNone(saved["settlement"])
        before = self.ledger.snapshot()
        self.assertEqual(before["settled_requests"], 0)
        self.assertEqual(before["pending_count"], 1)
        self.assertEqual(before["reserved_cny"], row["reserved_cny"])
        with self.assertRaisesRegex(AkaneTransportInterrupted, "pending_reservation_blocks_dispatch"):
            client.chat.completions.create(**request())
        send.assert_called_once()
        self.assertEqual(self.ledger.snapshot(), before)

    def test_unapproved_endpoint_or_secret_body_never_reserves_or_captures_secret(self) -> None:
        send = Mock(return_value=httpx.Response(200, json=reply()))
        adapter = self.adapter()
        client = self.client(adapter, send)
        bad = request()
        bad["extra_body"]["api_key"] = "synthetic-secret-body-never-capture"
        with self.assertRaisesRegex(AkaneTransportInterrupted, "unsupported_request_fields"):
            client.chat.completions.create(**bad)
        client.base_url = "https://unapproved.invalid"
        with self.assertRaisesRegex(AkaneTransportInterrupted, "unapproved_http_endpoint"):
            client.chat.completions.create(**request())
        send.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 0)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)
        self.assertNotIn("synthetic-secret-body", json.dumps(adapter.requests))

    def test_replay_actual_sdk_response_with_explicit_id_map_and_zero_additional_charge(self) -> None:
        response = reply()
        response["choices"][0]["message"]["content"] = "Synthetic memory-old"
        adapter = self.adapter()
        send = Mock(return_value=httpx.Response(200, json=response))
        client = self.client(adapter, send)
        client.chat.completions.create(**request(text="Open memory-old"))
        source = copy.deepcopy(adapter.requests[0])
        before = self.ledger.snapshot()
        self.key.reset_mock()
        with (
            adapter.scope("independent_branch", "setup_1"),
            adapter.replay([source], id_map={"memory-old": "memory-new"}),
        ):
            actual = client.chat.completions.create(**request(text="Open memory-new"))
        self.assertEqual(actual.choices[0].message.content, "Synthetic memory-new")
        self.assertEqual(self.ledger.snapshot(), before)
        self.assertEqual(send.call_count, 1)
        self.key.assert_not_called()
        replay = adapter.requests[1]
        self.assertEqual(replay["status"], "replayed")
        self.assertFalse(replay["dispatched"])
        self.assertIsNone(replay["settlement"])
        self.assertEqual(replay["replay_ledger_cost_cny"], "0")
        self.assertEqual(source, adapter.requests[0])

    def test_stream_replay_returns_original_bytes_through_real_sdk(self) -> None:
        adapter = self.adapter()
        send = Mock(side_effect=lambda _: self.streamed(sse()))
        client = self.client(adapter, send)
        first = list(client.chat.completions.create(**request(stream=True)))
        with adapter.replay_scope([copy.deepcopy(adapter.requests[0])]):
            second = list(client.chat.completions.create(**request(stream=True)))
        self.assertEqual([part.model_dump() for part in first], [part.model_dump() for part in second])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)

    def test_replay_mismatch_exhaustion_and_unconsumed_records_cannot_fall_back_to_paid(self) -> None:
        adapter = self.adapter()
        send = Mock(return_value=httpx.Response(200, json=reply()))
        client = self.client(adapter, send)
        client.chat.completions.create(**request())
        source = copy.deepcopy(adapter.requests[0])
        for expected, records, action in (
            ("replay_request_mismatch", [source], lambda: client.chat.completions.create(**request(text="different"))),
            ("replay_records_exhausted", [], lambda: client.chat.completions.create(**request())),
            ("replay_records_not_consumed", [source], lambda: None),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(AkaneTransportInterrupted, expected):
                with adapter.replay(records):
                    action()
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 1)

    def test_explicit_matcher_can_account_for_proven_time_variation(self) -> None:
        adapter = self.adapter()
        send = Mock(return_value=httpx.Response(200, json=reply()))
        client = self.client(adapter, send)
        client.chat.completions.create(**request(text="Synthetic timestamp=1"))
        source = copy.deepcopy(adapter.requests[0])

        def matcher(expected, actual):
            expected["messages"][-1]["content"] = "Synthetic timestamp=2"
            return expected == actual

        with adapter.replay([source], request_matcher=matcher):
            client.chat.completions.create(**request(text="Synthetic timestamp=2"))
        self.assertEqual(send.call_count, 1)
        self.assertEqual(source, adapter.requests[0])

    def test_runtime_attaches_every_bundle_and_is_idempotent(self) -> None:
        adapter = self.adapter()
        client = self.client(adapter, lambda _: httpx.Response(200, json=reply()))
        bundle = SimpleNamespace(client=client, model=MODEL)
        runtime = SimpleNamespace(chat=bundle, aux=bundle, memcore_summary=bundle, vision=bundle)
        self.assertEqual(adapter.attach_runtime(runtime), ("chat", "aux", "memcore_summary", "vision"))
        client.chat.completions.create(**request())
        self.assertEqual(len(adapter.requests), 1)

    def test_observer_captures_public_evidence_before_send_and_replay_compare(self) -> None:
        adapter = self.adapter()
        observed = []

        def observer(row):
            self.assertFalse(row["dispatched"])
            self.assertIsNone(row["reservation_id"])
            row["memory_evidence"] = {"source_id": "synthetic-source", "timestamp": 1}
            observed.append(row["run_mode"])

        adapter.request_observer = observer
        client = self.client(adapter, lambda _: httpx.Response(200, json=reply()))
        client.chat.completions.create(**request())
        with adapter.replay([copy.deepcopy(adapter.requests[0])]):
            client.chat.completions.create(**request())
        self.assertEqual(observed, ["paid_akane_http_request", "replay"])
        self.assertEqual(adapter.requests[1]["memory_evidence"]["source_id"], "synthetic-source")

    def test_observer_cannot_mutate_prepared_request(self) -> None:
        adapter = self.adapter()
        send = Mock(return_value=httpx.Response(200, json=reply()))
        client = self.client(adapter, send)

        def observer(row):
            row["request"]["temperature"] = 0

        adapter.request_observer = observer
        with self.assertRaisesRegex(AkaneTransportInterrupted, "request_observer_changed_capture"):
            client.chat.completions.create(**request())
        send.assert_not_called()
        self.assertEqual(adapter.requests[0]["request"]["temperature"], 0.8)
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)

    def test_sensitive_observer_rejection_never_saves_prompt_or_added_evidence(self) -> None:
        adapter = self.adapter(input_token_budget=300)
        send = Mock(return_value=httpx.Response(200, json=reply()))
        client = self.client(adapter, send)
        marker = "synthetic-private-material-do-not-capture"

        def observer(row):
            self.assertIn(marker, row["request_body_utf8"])
            row["memory_evidence"] = {"payload": marker}
            row["unexpected_extra_capture"] = marker
            row["scope"] = marker
            raise AkaneTransportInterrupted("sensitive_request_evidence_detected")

        adapter.request_observer = observer
        # A large request must still be checked for sensitive evidence before
        # the context limit's failure capture could save its original body.
        with self.assertRaisesRegex(AkaneTransportInterrupted, "sensitive_request_evidence_detected"):
            client.chat.completions.create(**request(text=marker + " synthetic filler" * 100))
        send.assert_not_called()
        self.key.assert_not_called()
        row = adapter.requests[0]
        self.assertIsNone(row["request"])
        self.assertNotIn("request_body_utf8", row)
        self.assertNotIn("memory_evidence", row)
        self.assertNotIn("unexpected_extra_capture", row)
        self.assertEqual(row["scope"], "host")
        self.assertTrue(row["request_evidence_redacted"])
        self.assertEqual(len(row["request_body_sha256"]), 64)
        self.assertGreater(row["input_tokens_estimated"], 300)
        saved = json.loads((self.root / "captures" / row["capture_file"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, row)
        self.assertNotIn(marker, json.dumps(saved))
        self.assertEqual(self.ledger.snapshot()["pending_count"], 0)
        self.assertEqual(self.ledger.snapshot()["settled_requests"], 0)

    def test_network_context_only_surrounds_physical_send_read_and_close(self) -> None:
        allowed = ContextVar("test_network_allowed", default=False)
        activations = []

        @contextmanager
        def network_context():
            token = allowed.set(True)
            activations.append(True)
            try:
                yield
            finally:
                allowed.reset(token)

        class GuardedChunks(httpx.SyncByteStream):
            def __iter__(inner):
                for line in sse().splitlines(keepends=True):
                    self.assertTrue(allowed.get())
                    yield line

            def close(inner):
                self.assertTrue(allowed.get())

        def handler(_):
            self.assertTrue(allowed.get())
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=GuardedChunks())

        adapter = self.adapter(network_context_factory=network_context)
        client = self.client(adapter, handler)
        for _ in client.chat.completions.create(**request(stream=True)):
            self.assertFalse(allowed.get())
        before = len(activations)
        with adapter.replay([copy.deepcopy(adapter.requests[0])]):
            list(client.chat.completions.create(**request(stream=True)))
        self.assertEqual(len(activations), before)
        self.assertFalse(allowed.get())


if __name__ == "__main__":
    unittest.main()
