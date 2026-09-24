"""Two paid, synthetic DeepSeek requests; independent of Akane and memory search.

Only DEEPSEEK_API_KEY is read. Credentials never enter request captures or reports.
No automatic retries: an uncertain charge keeps its budget reservation and stops.
"""

from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from memcore import build_chat_output_contract_prompt, parse_chat_output

from .budget import BudgetError, BudgetLedger, Tariff
from .runner import canonical, digest, write_json

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
PRICING_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
PRICING_CHECKED = "2026-09-06"
# Reserve against the entire documented 1M context, not a local token estimate.
PROVIDER_INPUT_UPPER_BOUND = 1_048_576
MAX_OUTPUT = 2048
MARKER = "memcore-preflight-pass"


class PreflightError(RuntimeError):
    """Fixed diagnostic code; never contains provider bodies or credentials."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise PreflightError("http_redirect_refused")


def official_transport(request: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise PreflightError("deepseek_api_key_missing")
    wire = urllib.request.Request(
        ENDPOINT,
        data=canonical(request).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(wire, timeout=45) as response:
            body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise PreflightError("provider_response_too_large")
            result = json.loads(body)
    except urllib.error.HTTPError as exc:
        raise PreflightError(f"http_status_{exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise PreflightError("transport_failed_or_timed_out") from None
    except (ValueError, UnicodeError):
        raise PreflightError("invalid_provider_json") from None
    if not isinstance(result, dict):
        raise PreflightError("invalid_provider_object")
    return result


def base_request(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_OUTPUT,
        "stream": False,
    }


def response_message(response: dict[str, Any], expected_finish: str) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise PreflightError("invalid_choices")
    choice = choices[0]
    if choice.get("finish_reason") != expected_finish:
        raise PreflightError("unexpected_finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise PreflightError("invalid_assistant_message")
    if set(message) - {"role", "content", "tool_calls", "reasoning_content"}:
        raise PreflightError("unsupported_assistant_wire_fields")
    if message.get("reasoning_content"):
        raise PreflightError("unexpected_thinking_output")
    if response.get("model") != MODEL:
        raise PreflightError("unexpected_returned_model")
    return copy.deepcopy(message)


def _tariff_estimate(usage: dict[str, Any], started: str, ended: str) -> dict[str, Any]:
    """Use the tariff schedule only when the request stays within one tariff band."""
    from zoneinfo import ZoneInfo

    def peak(instant: datetime) -> bool:
        local = instant.astimezone(ZoneInfo("Asia/Shanghai"))
        return local.weekday() < 5 and (9 <= local.hour < 12 or 14 <= local.hour < 18)

    begin, end = datetime.fromisoformat(started), datetime.fromisoformat(ended)
    # A boundary crossing or a request lasting over a minute is conservatively left unknown.
    if peak(begin) != peak(end) or (end - begin).total_seconds() > 60:
        return {"tariff_band": "uncertain", "tariff_estimated_cost_cny": None}
    band = "peak" if peak(begin) else "off_peak"
    multiplier = Decimal("1") if band == "peak" else Decimal("0.5")
    cost = (
        (
            Decimal(usage["prompt_cache_hit_tokens"]) * Decimal("0.10")
            + Decimal(usage["prompt_cache_miss_tokens"]) * Decimal("3")
            + Decimal(usage["completion_tokens"]) * Decimal("9")
        )
        * multiplier
        / Decimal(1_000_000)
    )
    return {"tariff_band": band, "tariff_estimated_cost_cny": format(cost, "f")}


def run_preflight(
    output: Path,
    ledger: BudgetLedger,
    snapshot: dict[str, Any],
    *,
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture the wire, usage and failures. An injected transport is always labelled offline."""
    live = transport is None
    send = transport or official_transport
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "format": "akane_memcore_provider_preflight_v1",
        "run_mode": "paid_provider_preflight" if live else "offline_transport_test",
        "status": "running",
        "requested_model": MODEL,
        "endpoint": ENDPOINT,
        "pricing_source": PRICING_SOURCE,
        "pricing_checked_at": PRICING_CHECKED,
        "source_snapshot": snapshot,
        "forced_routing": True,
        "akane_runtime_verified": False,
        "memory_retrieval_verified": False,
        "model_scores": None,
        "invoice_cost_cny": None,
        "first_token_latency_ms": None,
        "requests": [],
        "error": None,
    }
    write_json(output / "report.json", report)

    def call(request: dict[str, Any], stage: str) -> dict[str, Any]:
        # This preflight has tiny synthetic inputs. This cap is bytes, not tokens.
        if len(canonical(request).encode("utf-8")) > 24_000:
            raise PreflightError("preflight_request_byte_limit")
        row: dict[str, Any] = {"stage": stage, "request": request, "request_hash": digest(request)}
        reservation = ledger.reserve(
            request_id=uuid.uuid4().hex,
            stage="provider_preflight_" + stage,
            input_token_upper_bound=PROVIDER_INPUT_UPPER_BOUND,
            max_output_tokens=MAX_OUTPUT,
            tariff=Tariff.peak(),
        )
        row.update(reservation_id=reservation.reservation_id, reserved_cny=reservation.reserved_cny)
        report["requests"].append(row)
        write_json(output / "report.json", report)
        start = time.monotonic()
        row["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            response = send(copy.deepcopy(request))
            if not isinstance(response, dict) or response.get("model") != MODEL:
                raise PreflightError("unexpected_returned_model")
        except BaseException:
            # Even an HTTP error may follow billable processing. Never assume zero.
            ledger.mark_uncertain(reservation.reservation_id, reason="provider_failure")
            raise
        finally:
            write_json(output / "report.json", report)
        row["ended_at"] = datetime.now(timezone.utc).isoformat()
        row["total_latency_ms"] = round((time.monotonic() - start) * 1000, 3)
        row["response"] = response
        row["raw_provider_usage"] = response.get("usage")
        row["settlement"] = ledger.settle(reservation.reservation_id, response.get("usage"))
        row.update(_tariff_estimate(response["usage"], row["started_at"], row["ended_at"]))
        write_json(output / "report.json", report)
        return response

    try:
        if live and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            raise PreflightError("deepseek_api_key_missing")
        if live and datetime.now(timezone.utc).date().isoformat() != PRICING_CHECKED:
            raise PreflightError("pricing_requires_fresh_verification")
        if Decimal(ledger.snapshot()["limit_cny"]) > Decimal("50"):
            raise PreflightError("stage_budget_exceeds_authorized_ceiling")
        ledger.assert_ready()
        tool = {
            "type": "function",
            "function": {
                "name": "pilot_echo",
                "description": "Synthetic transport check. Echo the supplied value once.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "enum": [MARKER]}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        messages = [
            {
                "role": "system",
                "content": "Call pilot_echo once, then put exactly its returned value in final speech.\n"
                + build_chat_output_contract_prompt(enable_flavor=False),
            },
            {"role": "user", "content": "Echo the value " + MARKER + ". After the tool, return final JSON."},
        ]
        request = base_request(messages, [tool])
        request["tool_choice"] = {"type": "function", "function": {"name": "pilot_echo"}}
        first = response_message(call(request, "tool_request"), "tool_calls")
        calls = first.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise PreflightError("expected_one_echo_call")
        selected = calls[0]
        if (
            not isinstance(selected, dict)
            or selected.get("type") != "function"
            or not isinstance(selected.get("id"), str)
            or not selected["id"]
            or not isinstance(selected.get("function"), dict)
            or selected["function"].get("name") != "pilot_echo"
        ):
            raise PreflightError("invalid_echo_call")
        try:
            arguments = json.loads(selected["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            raise PreflightError("invalid_echo_arguments") from None
        if arguments != {"value": MARKER}:
            raise PreflightError("unexpected_echo_arguments")
        # Keep the complete supported assistant message, original call ID and argument string.
        messages.extend(
            [first, {"role": "tool", "tool_call_id": selected["id"], "content": canonical({"value": MARKER})}]
        )
        request = base_request(messages, [tool])
        request.update(tool_choice="none", response_format={"type": "json_object"})
        final = response_message(call(request, "final_json"), "stop")
        if final.get("tool_calls"):
            raise PreflightError("unexpected_final_tool_call")
        parsed = parse_chat_output(final.get("content", ""), mode="memcore_json", enable_flavor=False)
        if not parsed.ok or parsed.speech != MARKER:
            raise PreflightError("final_json_contract_failed")
        report.update(status="passed", native_tool_wire_verified=True, final_json_verified=True)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="interrupted")
    except (PreflightError, BudgetError) as exc:
        report.update(status="failed", error=exc.code if isinstance(exc, BudgetError) else str(exc))
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__)
    finally:
        report["ledger"] = ledger.snapshot()
        estimates = [r.get("tariff_estimated_cost_cny") for r in report["requests"]]
        report["tariff_estimated_cost_cny"] = (
            format(sum((Decimal(n) for n in estimates), Decimal(0)), "f")
            if estimates and all(n is not None for n in estimates)
            else None
        )
        report["live_transport_verified"] = live and report["status"] == "passed"
        write_json(output / "report.json", report)
        lines = [
            "# 模型传输预检",
            "",
            f"状态：{report['status']}；请求次数：{len(report['requests'])}；模式：{report['run_mode']}。",
            "",
            f"按公开时段费率估算：{report['tariff_estimated_cost_cny']} 元；供应商账单金额尚未核对。",
            "",
            "这是强制工具路由与最终 JSON 的连接测试，不报告记忆正确率、节费效果或 Akane 宿主验收。",
            "失败请求的未决费用仍占用阶段预算。详见 report.json。",
        ]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
