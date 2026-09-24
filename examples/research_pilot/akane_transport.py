"""Account for actual Akane SDK HTTP requests without rebuilding their prompts.

Attach after constructing the real ``LLMRuntime``. Its normal request builder,
OpenAI SDK serialization, tool loop and compatibility recovery remain in use.
Every physical HTTP dispatch reserves from the supplied existing stage ledger.
Replay is explicit, has no network fallback, and never changes that ledger.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import localcontext
from pathlib import Path
from typing import Any, Callable, Iterator

from .budget import BudgetError, BudgetLedger, Tariff
from .preflight import MAX_OUTPUT, MODEL, PRICING_CHECKED, PRICING_SOURCE, PROVIDER_INPUT_UPPER_BOUND, _tariff_estimate
from .runner import canonical, digest, write_json

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_ENDPOINTS = {"https://api.deepseek.com/chat/completions", "https://api.deepseek.com/v1/chat/completions"}
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Verified model ceiling, used only when an explicitly approved infrastructure
# role leaves max_tokens absent. Preserve that authentic request unchanged.
PROVIDER_DEFAULT_OUTPUT_UPPER_BOUND = 384 * 1024
_FORBIDDEN_BODY_FIELDS = {
    "api_key",
    "authorization",
    "headers",
    "extra_headers",
    "base_url",
    "api_base",
    "http_client",
    "timeout",
}


class AkaneTransportInterrupted(BaseException):
    """A fixed, non-sensitive stop code that host ``except Exception`` cannot hide."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _code(exc: BaseException) -> str:
    if isinstance(exc, (AkaneTransportInterrupted, BudgetError)):
        return exc.code
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return "interrupted"
    if isinstance(exc, TimeoutError):
        return "transport_timeout"
    return "transport_failed_or_timed_out"


def _environment_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "")


def _remap(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        # One simultaneous pass prevents an ID that is also a replacement value
        # from being replaced a second time. Includes JSON tool-argument strings.
        if not mapping:
            return value
        pattern = "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
        return re.sub(pattern, lambda match: mapping[match.group(0)], value)
    if isinstance(value, list):
        return [_remap(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _remap(item, mapping) for key, item in value.items()}
    return value


class _ResponseCapture:
    def __init__(self, owner: AkaneBudgetedTransport, row: dict[str, Any]):
        self.owner, self.row = owner, row
        self.started = time.monotonic()
        self.body = bytearray()
        self.pending = bytearray()
        self.event_lines: list[bytes] = []
        self.events: list[dict[str, Any]] = []
        self.models: set[str] = set()
        self.usage: Any = None
        self.done = False
        self.finished = False
        self.settled = False

    def _event(self) -> None:
        data = b"\n".join(line[5:].lstrip(b" ") for line in self.event_lines if line.startswith(b"data:"))
        self.event_lines.clear()
        if not data:
            return
        if data == b"[DONE]":
            self.done = True
            self.finish("provider_done")
            return
        try:
            event = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise AkaneTransportInterrupted("invalid_provider_json") from None
        self._observe(event)
        self.events.append(event)

    def _observe(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise AkaneTransportInterrupted("invalid_provider_object")
        if "error" in value:
            # Never save provider error bodies, including errors in SSE events.
            raise AkaneTransportInterrupted("provider_error_response")
        model = value.get("model")
        if model is not None:
            if model != MODEL:
                raise AkaneTransportInterrupted("unexpected_returned_model")
            self.models.add(model)
        if value.get("usage") is not None:
            if self.usage is not None and self.usage != value["usage"]:
                raise AkaneTransportInterrupted("conflicting_provider_usage")
            self.usage = copy.deepcopy(value["usage"])
        if self.row["first_token_latency_ms"] is None:
            for choice in value.get("choices", []):
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if isinstance(delta, dict) and (delta.get("content") or delta.get("tool_calls")):
                    self.row["first_token_latency_ms"] = round((time.monotonic() - self.started) * 1000, 3)
                    break

    def feed(self, chunk: bytes) -> None:
        if self.finished:
            return
        if not isinstance(chunk, bytes) or len(self.body) + len(chunk) > _MAX_RESPONSE_BYTES:
            raise AkaneTransportInterrupted("provider_response_too_large")
        if self.row["first_byte_latency_ms"] is None and chunk:
            self.row["first_byte_latency_ms"] = round((time.monotonic() - self.started) * 1000, 3)
        self.body.extend(chunk)
        self.pending.extend(chunk)
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending = bytearray(rest)
            line = bytes(line).rstrip(b"\r")
            if line:
                self.event_lines.append(line)
            else:
                self._event()
            if self.finished:
                break

    def complete_json(self, body: bytes) -> None:
        if len(body) > _MAX_RESPONSE_BYTES:
            raise AkaneTransportInterrupted("provider_response_too_large")
        try:
            response = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise AkaneTransportInterrupted("invalid_provider_json") from None
        self._observe(response)
        self.body.extend(body)
        self.row["response"] = response
        self.finish("complete_http_body")

    def finish(self, reason: str) -> None:
        if self.finished:
            return
        self.row["ended_at"] = datetime.now(timezone.utc).isoformat()
        self.row["total_latency_ms"] = round((time.monotonic() - self.started) * 1000, 3)
        self.row["response_end_reason"] = reason
        # Retain received wire evidence even when usage validation rejects it.
        # fail() persists this row while leaving the reservation unresolved.
        self.row["response_body_base64"] = base64.b64encode(self.body).decode("ascii")
        self.row["response_body_sha256"] = hashlib.sha256(self.body).hexdigest()
        if self.row["stream"]:
            self.row["response_events"] = copy.deepcopy(self.events)
        if not self.models:
            raise AkaneTransportInterrupted("unexpected_returned_model")
        self.row["raw_provider_usage"] = copy.deepcopy(self.usage)
        self.row["settlement"] = self.owner.ledger.settle(self.row["reservation_id"], self.usage)
        self.settled = True
        with localcontext() as context:
            context.prec = 80
            self.row.update(
                _tariff_estimate(self.row["settlement"]["usage"], self.row["started_at"], self.row["ended_at"])
            )
        if self.row["settlement"]["usage"]["prompt_tokens"] > self.owner.input_token_budget:
            raise AkaneTransportInterrupted("actual_input_token_budget_exceeded")
        self.finished = True
        self.row["status"] = "passed"
        self.owner._save(self.row)

    def fail(self, exc: BaseException) -> AkaneTransportInterrupted:
        if self.finished:
            return AkaneTransportInterrupted(_code(exc))
        self.finished = True
        if not self.settled:
            try:
                self.owner.ledger.mark_uncertain(self.row["reservation_id"], reason="provider_failure")
            except BudgetError:
                pass
        self.row.update(status="interrupted", error=_code(exc), ended_at=datetime.now(timezone.utc).isoformat())
        self.row["total_latency_ms"] = round((time.monotonic() - self.started) * 1000, 3)
        self.owner._save(self.row)
        return AkaneTransportInterrupted(_code(exc))


def _monitored_stream(original: Any, capture: _ResponseCapture) -> Any:
    import httpx

    class MonitoredStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            try:
                iterator = iter(original)
                while True:
                    with capture.owner._network_context():
                        try:
                            chunk = next(iterator)
                        except StopIteration:
                            break
                    capture.feed(chunk)
                    yield chunk
                capture.finish("http_stream_exhausted")
            except GeneratorExit:
                # The SDK normally closes after [DONE], without exhausting raw
                # bytes. close() below handles a consumer that stopped early.
                raise
            except BaseException as exc:
                raise capture.fail(exc) from None

        def close(self) -> None:
            try:
                with capture.owner._network_context():
                    original.close()
                capture.finish("consumer_closed")
            except BaseException as exc:
                raise capture.fail(exc) from None

    return MonitoredStream()


class AkaneBudgetedTransport:
    """Audit and budget the real OpenAI HTTP client used by an isolated Akane host."""

    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        capture_dir: Path,
        input_token_budget: int = 30720,
        api_key_provider: Callable[[], str] | None = None,
        network_context_factory: Callable[[], Any] | None = None,
        provider_default_output_roles: tuple[str, ...] = (),
        require_non_streaming: bool = False,
    ):
        if type(input_token_budget) is not int or not 0 < input_token_budget <= 65536:
            raise AkaneTransportInterrupted("invalid_input_token_budget")
        self.ledger = ledger
        self.input_token_budget = input_token_budget
        if (
            not isinstance(provider_default_output_roles, tuple)
            or set(provider_default_output_roles) - {"memcore_summary", "aux"}
            or type(require_non_streaming) is not bool
        ):
            raise AkaneTransportInterrupted("invalid_infrastructure_output_policy")
        self._provider_default_output_roles = frozenset(provider_default_output_roles)
        self._require_non_streaming = require_non_streaming
        self.requests: list[dict[str, Any]] = []
        self.request_observer: Callable[[dict[str, Any]], None] | None = None
        self._capture_dir = Path(capture_dir)
        self._key = api_key_provider or _environment_key
        self._network_context = network_context_factory or nullcontext
        self._scope: ContextVar[tuple[str, str]] = ContextVar(
            "akane_paid_request_scope", default=("host", "background")
        )
        self._replay: ContextVar[dict[str, Any] | None] = ContextVar("akane_replay", default=None)
        self._replay_active = False
        self._attached: dict[int, tuple[Any, Any]] = {}
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise AkaneTransportInterrupted("capture_directory_unavailable") from None

    @contextmanager
    def scope(self, scope: str, step_id: str) -> Iterator[None]:
        if any(not isinstance(value, str) or _LABEL.fullmatch(value) is None for value in (scope, step_id)):
            raise AkaneTransportInterrupted("invalid_capture_scope")
        token = self._scope.set((scope, step_id))
        try:
            yield
        finally:
            self._scope.reset(token)

    @contextmanager
    def replay(
        self,
        records: list[dict[str, Any]],
        id_map: dict[str, str] | None = None,
        request_matcher: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ) -> Iterator[None]:
        if self._replay_active:
            raise AkaneTransportInterrupted("nested_replay_not_allowed")
        state = {
            "records": list(records),
            "index": 0,
            "id_map": id_map if id_map is not None else {},
            "matcher": request_matcher,
        }
        token = self._replay.set(state)
        self._replay_active = True
        try:
            yield
            if state["index"] != len(state["records"]):
                raise AkaneTransportInterrupted("replay_records_not_consumed")
        finally:
            self._replay_active = False
            self._replay.reset(token)

    replay_scope = replay

    def attach_runtime(self, runtime: Any) -> tuple[str, ...]:
        roles = []
        for name in ("chat", "aux", "memcore_summary", "vision"):
            bundle = getattr(runtime, name, None)
            if bundle is not None and getattr(bundle, "client", None) is not None:
                self.attach_client(bundle.client, role=name)
                roles.append(name)
        if "chat" not in roles:
            raise AkaneTransportInterrupted("akane_chat_client_missing")
        return tuple(roles)

    def attach_client(self, client: Any, *, role: str = "chat") -> Any:
        if not isinstance(role, str) or _LABEL.fullmatch(role) is None:
            raise AkaneTransportInterrupted("invalid_bundle_role")
        http_client = getattr(client, "_client", None)
        if http_client is None or not callable(getattr(http_client, "send", None)):
            raise AkaneTransportInterrupted("unsupported_akane_client")
        # Keep actual SDK objects and normal serializers. The SDK and httpx
        # redirect layer must not issue a hidden second physical request.
        client.max_retries = 0
        http_client.follow_redirects = False
        for transport in [getattr(http_client, "_transport", None), *getattr(http_client, "_mounts", {}).values()]:
            pool = getattr(transport, "_pool", None)
            if pool is not None and hasattr(pool, "_retries"):
                pool._retries = 0
        if client.max_retries != 0 or http_client.follow_redirects is not False:
            raise AkaneTransportInterrupted("sdk_retries_not_disabled")
        existing = getattr(http_client, "_akane_budget_owner", None)
        if existing is self:
            return client
        if existing is not None:
            raise AkaneTransportInterrupted("akane_client_already_attached")
        original = http_client.send

        def accounted_send(request: Any, *args: Any, **kwargs: Any) -> Any:
            # kwargs is local to this call; the source request's body is never
            # edited. This also protects SDK copies sharing the same http client.
            kwargs["follow_redirects"] = False
            return self._send(original, request, args, kwargs, role)

        http_client.send = accounted_send
        http_client._akane_budget_owner = self
        self._attached[id(http_client)] = (http_client, original)
        return client

    def _save(self, row: dict[str, Any]) -> None:
        try:
            row["ledger"] = self.ledger.snapshot()
            write_json(self._capture_dir / row["capture_file"], row)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise AkaneTransportInterrupted("interrupted") from None
            raise AkaneTransportInterrupted("capture_write_failed") from None

    def _request_row(self, request: Any, role: str) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        scope, step_id = self._scope.get()
        row: dict[str, Any] = {
            "request_id": request_id,
            "capture_file": f"request_{request_id}.json",
            "scope": scope,
            "step_id": step_id,
            "bundle_role": role,
            "run_mode": "replay" if self._replay.get() is not None else "paid_akane_http_request",
            "status": "preparing",
            "error": None,
            "dispatched": False,
            "endpoint": None,
            "requested_model": MODEL,
            "request": None,
            "response": None,
            "raw_provider_usage": None,
            "settlement": None,
            "reservation_id": None,
            "reserved_cny": None,
            "input_tokens_estimated": None,
            "token_count_quality": "estimated",
            "input_estimate_method": "ceil(canonical_request_utf8_bytes / 3)",
            "input_token_budget": self.input_token_budget,
            "provider_input_reservation_tokens": PROVIDER_INPUT_UPPER_BOUND,
            "output_reservation_tokens": MAX_OUTPUT,
            "started_at": None,
            "ended_at": None,
            "total_latency_ms": None,
            "first_byte_latency_ms": None,
            "first_token_latency_ms": None,
            "tariff_band": None,
            "tariff_estimated_cost_cny": None,
            "invoice_cost_cny": None,
            "pricing_checked_at": PRICING_CHECKED,
            "pricing_source": PRICING_SOURCE,
        }
        self.requests.append(row)
        safe_initial = copy.deepcopy(row)
        prepared: dict[str, Any] = {}
        try:
            endpoint = str(request.url)
            if endpoint not in _ENDPOINTS or request.method != "POST":
                raise AkaneTransportInterrupted("unapproved_http_endpoint")
            body = request.content
            if not isinstance(body, bytes):
                raise AkaneTransportInterrupted("invalid_request_wire")
            wire = json.loads(body)
            if not isinstance(wire, dict) or set(wire) & _FORBIDDEN_BODY_FIELDS:
                raise AkaneTransportInterrupted("unsupported_request_fields")
            provider_default_output = role in self._provider_default_output_roles and "max_tokens" not in wire
            if (
                wire.get("model") != MODEL
                or wire.get("thinking") != {"type": "disabled"}
                or type(wire.get("temperature")) not in {int, float}
                or not math.isfinite(wire["temperature"])
                or not 0 <= wire["temperature"] <= 2
                or (
                    not provider_default_output
                    and (type(wire.get("max_tokens")) is not int or wire["max_tokens"] != MAX_OUTPUT)
                )
                or type(wire.get("stream", False)) is not bool
            ):
                raise AkaneTransportInterrupted("generation_settings_mismatch")
            if self._require_non_streaming and wire.get("stream", False):
                raise AkaneTransportInterrupted("long_experiment_requires_non_streaming")
            if provider_default_output:
                row["output_reservation_tokens"] = PROVIDER_DEFAULT_OUTPUT_UPPER_BOUND
                row["output_limit_basis"] = "provider_model_maximum_preserving_absent_request_limit"
            if not isinstance(wire.get("messages"), list) or not wire["messages"]:
                raise AkaneTransportInterrupted("invalid_request_wire")
            size = len(canonical(wire).encode("utf-8"))
            row.update(
                endpoint=endpoint,
                request=wire,
                request_hash=digest(wire),
                request_body_utf8=body.decode("utf-8"),
                request_body_sha256=hashlib.sha256(body).hexdigest(),
                request_utf8_bytes=len(body),
                input_tokens_estimated=math.ceil(size / 3),
                stream=wire.get("stream", False),
                generation_settings={
                    key: copy.deepcopy(wire[key])
                    for key in ("model", "thinking", "temperature", "top_p", "max_tokens", "stream", "stream_options")
                    if key in wire
                },
            )
            prepared = copy.deepcopy(row)
            if self.request_observer is not None:
                self.request_observer(row)
                if any(row.get(key) != value for key, value in prepared.items()):
                    row.clear()
                    row.update(prepared)
                    raise AkaneTransportInterrupted("request_observer_changed_capture")
            if row["input_tokens_estimated"] > self.input_token_budget:
                raise AkaneTransportInterrupted("estimated_input_token_budget_exceeded")
            return row
        except BaseException as exc:
            if _code(exc) == "sensitive_request_evidence_detected":
                # A trusted pre-dispatch observer found forbidden material in
                # the authentic prompt or its evidence. Keep only the original
                # fixed metadata plus hashes/counts computed before the observer.
                # In particular, arbitrary observer-added fields cannot leak it.
                row.clear()
                row.update(safe_initial)
                for key in (
                    "request_hash",
                    "request_body_sha256",
                    "request_utf8_bytes",
                    "input_tokens_estimated",
                ):
                    if key in prepared:
                        row[key] = prepared[key]
                row["request_evidence_redacted"] = True
            row.update(status="interrupted", error=_code(exc))
            self._save(row)
            raise AkaneTransportInterrupted(_code(exc)) from None

    def _send(self, original: Callable, request: Any, args: tuple, kwargs: dict, role: str) -> Any:
        row = self._request_row(request, role)
        replay = self._replay.get()
        if replay is not None:
            return self._send_replay(request, row, replay)
        capture = None
        try:
            if self._replay_active:
                # A background thread does not inherit replay's ContextVar.
                # It must not accidentally turn a reconstruction into a paid call.
                raise AkaneTransportInterrupted("replay_context_missing_in_background")
            if datetime.now(timezone.utc).date().isoformat() != PRICING_CHECKED:
                raise AkaneTransportInterrupted("pricing_requires_fresh_verification")
            key = self._key()
            if not isinstance(key, str) or not key.strip():
                raise AkaneTransportInterrupted("deepseek_api_key_missing")
            request.headers["Authorization"] = "Bearer " + key.strip()
            reservation = self.ledger.reserve(
                request_id=row["request_id"],
                stage="akane_http_request",
                input_token_upper_bound=PROVIDER_INPUT_UPPER_BOUND,
                max_output_tokens=row["output_reservation_tokens"],
                tariff=Tariff.peak(),
            )
            row.update(
                reservation_id=reservation.reservation_id, reserved_cny=reservation.reserved_cny, status="reserved"
            )
            capture = _ResponseCapture(self, row)
            self._save(row)
            row.update(started_at=datetime.now(timezone.utc).isoformat(), dispatched=True)
            self._save(row)
            with self._network_context():
                response = original(request, *args, **kwargs)
            row["http_status"] = response.status_code
            if not 200 <= response.status_code < 300:
                raise AkaneTransportInterrupted(f"http_status_{response.status_code}")
            content_type = response.headers.get("content-type", "")
            row["response_headers"] = {"content-type": content_type}
            if row["stream"]:
                if not content_type.lower().startswith("text/event-stream"):
                    raise AkaneTransportInterrupted("invalid_stream_content_type")
                response.stream = _monitored_stream(response.stream, capture)
                return response
            with self._network_context():
                body = response.read()
            capture.complete_json(body)
            return response
        except BaseException as exc:
            if capture is not None:
                raise capture.fail(exc) from None
            row.update(status="interrupted", error=_code(exc))
            self._save(row)
            raise AkaneTransportInterrupted(_code(exc)) from None

    def _send_replay(self, request: Any, row: dict[str, Any], state: dict[str, Any]) -> Any:
        import httpx

        try:
            if state["index"] >= len(state["records"]):
                raise AkaneTransportInterrupted("replay_records_exhausted")
            source = state["records"][state["index"]]
            mapping = state["id_map"]
            if not isinstance(mapping, dict) or any(
                not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in mapping.items()
            ):
                raise AkaneTransportInterrupted("invalid_replay_id_map")
            if source.get("status") != "passed" or not isinstance(source.get("request"), dict):
                raise AkaneTransportInterrupted("invalid_replay_source")
            expected = _remap(source["request"], mapping)
            matcher = state["matcher"]
            matched = (
                expected == row["request"] if matcher is None else matcher(expected, copy.deepcopy(row["request"]))
            )
            if matched is not True:
                raise AkaneTransportInterrupted("replay_request_mismatch")
            body = base64.b64decode(source["response_body_base64"], validate=True)
            if hashlib.sha256(body).hexdigest() != source.get("response_body_sha256"):
                raise AkaneTransportInterrupted("replay_source_hash_mismatch")
            if mapping:
                body = _remap(body.decode("utf-8"), mapping).encode("utf-8")
            row.update(
                status="replayed",
                source_request_id=source["request_id"],
                source_request_hash=source["request_hash"],
                source_response_body_sha256=source["response_body_sha256"],
                replay_id_map=copy.deepcopy(mapping),
                response_body_base64=base64.b64encode(body).decode("ascii"),
                response_body_sha256=hashlib.sha256(body).hexdigest(),
                response=_remap(source.get("response"), mapping),
                response_events=_remap(source.get("response_events"), mapping),
                response_headers=copy.deepcopy(source["response_headers"]),
                http_status=source["http_status"],
                raw_provider_usage=copy.deepcopy(source.get("raw_provider_usage")),
                replay_ledger_cost_cny="0",
            )
            self._save(row)
            state["index"] += 1
            return httpx.Response(
                status_code=source["http_status"],
                headers=source["response_headers"],
                stream=httpx.ByteStream(body),
                request=request,
            )
        except BaseException as exc:
            row.update(status="interrupted", error=_code(exc))
            self._save(row)
            raise AkaneTransportInterrupted(_code(exc)) from None
