"""Budgeted official model transport and exact local wire captures for the pilot.

Each call reserves the documented full provider input bound, independently of
the labelled research admission estimate. There are no retries. Valid usage for
the selected model is settled before checking response/context protocol errors;
uncertain charges stay reserved and block every later call through the ledger.
"""

from __future__ import annotations

import copy
import math
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import localcontext
from pathlib import Path
from typing import Any, Callable

from .budget import BudgetError, BudgetLedger, Tariff
from .preflight import (
    ENDPOINT,
    MAX_OUTPUT,
    MODEL,
    PRICING_CHECKED,
    PRICING_SOURCE,
    PROVIDER_INPUT_UPPER_BOUND,
    PreflightError,
    _tariff_estimate,
    official_transport,
    response_message,
)
from .runner import canonical, digest, write_json

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_REQUEST_FIELDS = {
    "model",
    "messages",
    "tools",
    "thinking",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "tool_choice",
    "response_format",
}
_BUDGET_ERRORS = {
    "pending_reservation_blocks_dispatch",
    "budget_limit_exceeded",
    "budget_ledger_closed",
    "provider_usage_missing",
    "invalid_provider_usage",
    "usage_cost_exceeds_reservation",
    "usage_tokens_exceed_reservation",
    "reservation_uncertain",
}
_PROVIDER_ERRORS = {
    "deepseek_api_key_missing",
    "http_redirect_refused",
    "provider_response_too_large",
    "transport_failed_or_timed_out",
    "invalid_provider_json",
    "invalid_provider_object",
    "invalid_choices",
    "unexpected_finish_reason",
    "invalid_assistant_message",
    "unsupported_assistant_wire_fields",
    "unexpected_thinking_output",
    "unexpected_returned_model",
}


class TransportError(RuntimeError):
    """Fixed error code; never forwards an HTTP body or exception detail."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _safe_exception_code(exc: BaseException) -> str:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    if isinstance(exc, BudgetError):
        return exc.code if exc.code in _BUDGET_ERRORS else "budget_ledger_failed"
    if isinstance(exc, PreflightError):
        code = str(exc)
        if code in _PROVIDER_ERRORS or re.fullmatch(r"http_status_[1-5][0-9]{2}", code):
            return code
    return "transport_failed_or_timed_out"


def _validate_request(request: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not isinstance(request, dict) or set(request) - _REQUEST_FIELDS:
        raise TransportError("unsupported_request_fields")
    if (
        request.get("model") != MODEL
        or request.get("thinking") != {"type": "disabled"}
        or type(request.get("max_tokens")) is not int
        or request["max_tokens"] != MAX_OUTPUT
        or type(request.get("temperature")) not in {int, float}
        or request["temperature"] != 0
        or type(request.get("top_p")) not in {int, float}
        or request["top_p"] != 1
        or request.get("stream") is not False
    ):
        raise TransportError("generation_settings_mismatch")
    if request.get("tool_choice", "auto") not in ("auto", "none"):
        raise TransportError("unsupported_tool_choice")
    if "response_format" in request and request["response_format"] != {"type": "json_object"}:
        raise TransportError("unsupported_response_format")
    messages, tools = request.get("messages"), request.get("tools")
    if (
        not isinstance(messages, list)
        or not messages
        or any(not isinstance(message, dict) for message in messages)
        or not isinstance(tools, list)
        or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise TransportError("invalid_request_wire")
    try:
        encoded = canonical(request).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise TransportError("invalid_request_wire") from None
    return copy.deepcopy(request), len(encoded)


def _validate_response(response: dict[str, Any]) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TransportError("invalid_choices")
    finish = choices[0].get("finish_reason")
    if finish not in ("tool_calls", "stop"):
        raise TransportError("unexpected_finish_reason")
    try:
        message = response_message(response, finish)
    except PreflightError as exc:
        raise TransportError(_safe_exception_code(exc)) from None
    calls = message.get("tool_calls")
    if finish == "stop":
        if calls or not isinstance(message.get("content"), str):
            raise TransportError("invalid_final_message_shape")
        # MemCore's final JSON contract belongs to LiveSession, after this paid call.
        return
    if (
        not isinstance(calls, list)
        or not calls
        or message.get("content") is not None
        and not isinstance(message.get("content"), str)
    ):
        raise TransportError("invalid_tool_message_shape")
    call_ids: set[str] = set()
    for call in calls:
        if (
            not isinstance(call, dict)
            or call.get("type") != "function"
            or not isinstance(call.get("id"), str)
            or not call["id"]
            or call["id"] in call_ids
            or not isinstance(call.get("function"), dict)
            or not isinstance(call["function"].get("name"), str)
            or not call["function"]["name"]
            or not isinstance(call["function"].get("arguments"), str)
        ):
            raise TransportError("invalid_native_tool_call")
        call_ids.add(call["id"])


def _capture_value(value: Any) -> Any:
    """Return only valid JSON, preserving every original wire field and string."""
    try:
        canonical(value)
        return copy.deepcopy(value)
    except (TypeError, ValueError, OverflowError):
        return None


class BudgetedTransport:
    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        capture_dir: Path,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        input_token_budget: int = 30720,
    ):
        if type(input_token_budget) is not int or not 0 < input_token_budget <= 30720:
            raise TransportError("invalid_input_token_budget")
        self.ledger = ledger
        self.requests: list[dict[str, Any]] = []
        self.input_token_budget = input_token_budget
        self._live = transport is None
        self._send = official_transport if transport is None else transport
        self._capture_dir = Path(capture_dir)
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise TransportError("capture_directory_unavailable") from None

    @property
    def is_live(self) -> bool:
        return self._live

    def _capture(self, row: dict[str, Any]) -> None:
        try:
            write_json(self._capture_dir / row["capture_file"], row)
        except (OSError, TypeError, ValueError):
            raise TransportError("capture_write_failed") from None

    def call(self, request: dict[str, Any], *, scope: str, step_id: str) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        row: dict[str, Any] = {
            "request_id": request_id,
            "capture_file": f"request_{request_id}.json",
            "scope": None,
            "step_id": None,
            "run_mode": "paid_scenario_request" if self._live else "offline_transport_test",
            "status": "preparing",
            "error": None,
            "dispatched": False,
            "requested_model": MODEL,
            "endpoint": ENDPOINT,
            "pricing_checked_at": PRICING_CHECKED,
            "pricing_source": PRICING_SOURCE,
            "request": None,
            "response": None,
            "raw_provider_usage": None,
            "settlement": None,
            "input_tokens_estimated": None,
            "token_count_quality": "estimated",
            "input_estimate_method": "ceil(canonical_request_utf8_bytes / 3)",
            "input_token_budget": self.input_token_budget,
            "provider_input_reservation_tokens": PROVIDER_INPUT_UPPER_BOUND,
            "output_reservation_tokens": MAX_OUTPUT,
            "reservation_id": None,
            "reserved_cny": None,
            "started_at": None,
            "ended_at": None,
            "total_latency_ms": None,
            "first_token_latency_ms": None,
            "tariff_band": None,
            "tariff_estimated_cost_cny": None,
            "invoice_cost_cny": None,
        }
        self.requests.append(row)
        reservation = None
        settled = False
        try:
            if any(not isinstance(value, str) or _LABEL.fullmatch(value) is None for value in (scope, step_id)):
                raise TransportError("invalid_capture_scope")
            row.update(scope=scope, step_id=step_id)
            if self._live and datetime.now(timezone.utc).date().isoformat() != PRICING_CHECKED:
                raise TransportError("pricing_requires_fresh_verification")
            wire, size = _validate_request(request)
            row.update(request=wire, request_hash=digest(wire), request_utf8_bytes=size)
            row["input_tokens_estimated"] = math.ceil(size / 3)
            if row["input_tokens_estimated"] > self.input_token_budget:
                raise TransportError("estimated_input_token_budget_exceeded")
            reservation = self.ledger.reserve(
                request_id=request_id,
                stage="scenario_request",
                input_token_upper_bound=PROVIDER_INPUT_UPPER_BOUND,
                max_output_tokens=MAX_OUTPUT,
                tariff=Tariff.peak(),
            )
            row.update(
                reservation_id=reservation.reservation_id,
                reserved_cny=reservation.reserved_cny,
                status="reserved",
            )
            # Persist the exact request and committed reservation before sending.
            self._capture(row)
            row["started_at"] = datetime.now(timezone.utc).isoformat()
            send_started = time.monotonic()
            row["dispatched"] = True
            try:
                response = self._send(copy.deepcopy(wire))
            except BaseException as exc:
                raise TransportError(_safe_exception_code(exc)) from None
            finally:
                row["ended_at"] = datetime.now(timezone.utc).isoformat()
                row["total_latency_ms"] = round((time.monotonic() - send_started) * 1000, 3)
            if not isinstance(response, dict):
                raise TransportError("invalid_provider_object")
            if "error" in response:
                # API error bodies are deliberately not written to the capture.
                raise TransportError("provider_error_response")
            row["response"] = _capture_value(response)
            row["raw_provider_usage"] = _capture_value(response.get("usage"))
            if response.get("model") != MODEL:
                raise TransportError("unexpected_returned_model")
            row["settlement"] = self.ledger.settle(reservation.reservation_id, response.get("usage"))
            settled = True
            with localcontext() as context:
                context.prec = 80
                row.update(_tariff_estimate(row["settlement"]["usage"], row["started_at"], row["ended_at"]))
            if row["response"] is None:
                raise TransportError("invalid_provider_object")
            if row["settlement"]["usage"]["prompt_tokens"] > self.input_token_budget:
                raise TransportError("actual_input_token_budget_exceeded")
            _validate_response(response)
            row["status"] = "passed"
            return copy.deepcopy(response)
        except BaseException as exc:
            if reservation is not None and not settled:
                try:
                    self.ledger.mark_uncertain(reservation.reservation_id, reason="provider_failure")
                except BudgetError:
                    # A durable existing pending/settled state is never released here.
                    pass
            code = exc.code if isinstance(exc, TransportError) else _safe_exception_code(exc)
            row.update(status="failed" if row["dispatched"] else "rejected", error=code)
            raise TransportError(code) from None
        finally:
            try:
                row["ledger"] = self.ledger.snapshot()
            except Exception as exc:
                row["ledger"] = None
                row["ledger_error"] = _safe_exception_code(exc)
            self._capture(row)
