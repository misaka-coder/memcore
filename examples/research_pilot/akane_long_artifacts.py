"""Bounded, synthetic plan storage/checking tools for the long experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .runner import digest, write_json


REVISIONS = ("draft_v1", "draft_v2", "final")
PLAN_FIELDS = ("option", "count", "per_unit", "total", "owner", "open_items")


class PlanWorkspace:
    """Host-owned synthetic artifact data; never a MemCore truth-store adapter."""

    def __init__(self, path: Path, requirements: Mapping[str, Any]) -> None:
        self.path = path
        self.requirements = copy.deepcopy(dict(requirements))
        self.events: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"format": "synthetic_research_plans_v1", "plans": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("format") != "synthetic_research_plans_v1" or not isinstance(data.get("plans"), dict):
            raise RuntimeError("invalid_saved_research_plan_state")
        return data

    def save(self, revision: str, plan: Any) -> dict[str, Any]:
        if revision not in REVISIONS or not isinstance(plan, dict) or set(plan) != set(PLAN_FIELDS):
            return {"status": "invalid", "reason": "exact_plan_fields_required"}
        if any(type(plan[key]) is not int or not 0 <= plan[key] <= 1000000 for key in ("count", "per_unit", "total")):
            return {"status": "invalid", "reason": "integer_amounts_required"}
        if any(not isinstance(plan[key], str) or not 0 < len(plan[key]) <= 100 for key in ("option", "owner")):
            return {"status": "invalid", "reason": "short_named_option_and_owner_required"}
        if (
            not isinstance(plan["open_items"], list)
            or len(plan["open_items"]) > 10
            or any(not isinstance(item, str) or len(item) > 150 for item in plan["open_items"])
        ):
            return {"status": "invalid", "reason": "bounded_open_items_required"}
        state = self.snapshot()
        existing = state["plans"].get(revision)
        if revision == "draft_v1" and existing is not None and existing != plan:
            return {"status": "conflict", "reason": "historical_draft_v1_is_immutable"}
        state["plans"][revision] = copy.deepcopy(plan)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".pending.json")
        write_json(temporary, state)
        temporary.replace(self.path)
        result = {
            "status": "saved",
            "synthetic": True,
            "resource_id": f"research-plan:{revision}",
            "revision": revision,
            "content_hash": digest(plan),
        }
        self.events.append({"operation": "save", "revision": revision, "plan": copy.deepcopy(plan), "result": result})
        return result

    def check(self, revision: str) -> dict[str, Any]:
        if revision not in REVISIONS:
            return {"status": "invalid", "reason": "known_revision_required"}
        plan = self.snapshot()["plans"].get(revision)
        if plan is None:
            return {"status": "not_found", "revision": revision, "reason": "plan_has_not_been_saved"}
        rules = self.requirements["v1" if revision == "draft_v1" else "v2"]
        checks = {
            "option": plan["option"] == rules["option"],
            "count": plan["count"] == rules["count"],
            "per_unit": plan["per_unit"] == rules["per_unit"],
            "total_arithmetic": plan["total"] == plan["count"] * plan["per_unit"],
            "budget": plan["total"] <= rules["budget"],
            "owner": plan["owner"] == rules["owner"],
            "unresolved_items": set(plan["open_items"]) == set(rules["open_items"]),
        }
        result = {
            "status": "passed" if all(checks.values()) else "failed",
            "synthetic": True,
            "revision": revision,
            "resource_id": f"research-plan:{revision}",
            "content_hash": digest(plan),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "scope": "Only the stated synthetic plan constraints; no external booking or execution occurred.",
        }
        self.events.append({"operation": "check", "revision": revision, "result": copy.deepcopy(result)})
        return result

    def handlers(self, host: Any) -> list[Any]:
        from capcore import CapabilityToolSpec
        from companion_v01.tool_handlers.core import BaseToolHandler, ToolExecutionResult, ToolFollowupEnvelope

        workspace = self

        class PlanHandler(BaseToolHandler):
            def tool_spec(self) -> Any:
                properties: dict[str, Any] = {"revision": {"type": "string", "enum": list(REVISIONS)}}
                if self.tool_type == "save_research_plan":
                    properties["plan"] = {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(PLAN_FIELDS),
                        "properties": {
                            "option": {"type": "string", "description": "资料中的选项名称。"},
                            "count": {"type": "integer", "minimum": 0},
                            "per_unit": {"type": "integer", "minimum": 0},
                            "total": {"type": "integer", "minimum": 0},
                            "owner": {"type": "string"},
                            "open_items": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                return CapabilityToolSpec(
                    capability_id=self.tool_type,
                    display_name=self.tool_type,
                    description=(
                        "保存虚构研究任务的结构化方案。只写入隔离测试资源，不执行预约、付款或真实部署。draft_v1 保存后不可改写；其它修订可修改。保存后应调用 check_research_plan 验收。"
                        if self.tool_type == "save_research_plan"
                        else "读取已保存方案并按已提供的虚构任务约束检查；返回检查状态、失败项和内容哈希，不返回历史资料全文，也不执行真实业务。"
                    ),
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(properties),
                        "properties": properties,
                    },
                    risk="low",
                    confirm="never",
                    effects=("synthetic_plan_write",) if self.tool_type == "save_research_plan" else (),
                    visible_in=("desktop_pet",),
                    idempotency="effectful" if self.tool_type == "save_research_plan" else "read_only",
                )

            def normalize_call(self, value: Any) -> dict[str, Any] | None:
                if not isinstance(value, dict) or value.get("type") != self.tool_type:
                    return None
                if value.get("revision") not in REVISIONS:
                    return None
                if self.tool_type == "save_research_plan" and not isinstance(value.get("plan"), dict):
                    return None
                return {key: copy.deepcopy(value[key]) for key in ("type", "revision", "plan") if key in value}

            def execute(self, *, call: dict[str, Any], context: Any) -> Any:
                result = (
                    workspace.save(call["revision"], call["plan"])
                    if self.tool_type == "save_research_plan"
                    else workspace.check(call["revision"])
                )
                body = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                host.tool_events.append(
                    {
                        "name": self.tool_type,
                        "arguments": {k: v for k, v in call.items() if k != "type"},
                        "body": body,
                        "invocation_id": context.invocation_id,
                        "source_id": context.current_user_source_id,
                    }
                )
                return ToolExecutionResult(
                    tool_type=self.tool_type,
                    followup_context=body,
                    followup_envelope=ToolFollowupEnvelope(content=body, producer_bounded=True, complete=True),
                )

        return [
            type(name, (PlanHandler,), {"tool_type": name})() for name in ("save_research_plan", "check_research_plan")
        ]
