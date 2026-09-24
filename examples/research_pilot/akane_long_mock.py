"""TEST ONLY: scripted SDK HTTP responses, never selected by the paid path."""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .akane_long_observation import LongRunTransport


class TestOnlyEmbedding:
    name = "TEST_ONLY_deterministic_embedding"
    dimension = 1024

    def __init__(self) -> None:
        from memcore import HashedEmbeddingProvider

        self.delegate = HashedEmbeddingProvider(dimension=self.dimension)

    def embed_text(self, text: str) -> list[float]:
        return self.delegate.embed_text(text)

    def embed_texts(self, texts: Any) -> list[list[float]]:
        return self.delegate.embed_texts(list(texts))


class TestOnlyLongTransport(LongRunTransport):
    def bind_case(self, case: dict[str, Any], *, phase: str) -> None:
        self.case, self.phase = copy.deepcopy(case), phase
        self.step_calls: dict[str, int] = {}
        self.count = 0

    def attach_client(self, client: Any, *, role: str = "chat") -> Any:
        import httpx

        client._client._transport = httpx.MockTransport(lambda request: self.respond(request, role=role))
        client._client._mounts = {}
        return super().attach_client(client, role=role)

    def _request_row(self, request: Any, role: str) -> dict[str, Any]:
        row = super()._request_row(request, role)
        row["run_mode"] = "TEST_ONLY_mock_http"
        return row

    def respond(self, request: Any, *, role: str) -> Any:
        import httpx

        self.count += 1
        if self.count > 200:
            raise RuntimeError("unexpected_excess_mock_requests")
        wire = json.loads(request.content)
        if role != "chat":
            content = json.dumps(
                {
                    "diary_summary": "离线测试的对话已按真实压缩流程归档。",
                    "period_label": "离线验证",
                    "event_type": "研究流程",
                    "key_events": ["保存合成任务的完整轮次"],
                    "core_facts": ["这里只验证压缩和来源链，不评估模型答案质量"],
                    "semantic_summary": "离线测试的跨轮次记录。",
                    "stable_facts": ["合成场景"],
                    "recurring_topics": ["流程验证"],
                    "important_people": [],
                    "open_loops": [],
                    "memory_metadata": {
                        "entity_anchors": [],
                        "topic_terms": ["流程验证"],
                        "memory_facets": ["event"],
                        "about_roles": ["user"],
                        "retrieval_priority": "normal",
                    },
                },
                ensure_ascii=False,
            )
            message = {"role": "assistant", "content": content}
        else:
            step = self.current_step
            call = self.step_calls.get(step, 0)
            self.step_calls[step] = call + 1
            number = int(step.rsplit("_", 1)[1])
            calls: list[tuple[str, dict[str, Any]]] = []
            if number in {2, 3, 21}:
                version = "v1" if number == 2 else "v2"
                revision = {2: "draft_v1", 3: "draft_v2", 21: "final"}[number]
                rules = self.case["plan_requirements"][version]
                plan = {
                    key: copy.deepcopy(rules[key]) for key in ("option", "count", "per_unit", "owner", "open_items")
                }
                plan["total"] = plan["count"] * plan["per_unit"]
                adjusted = call
                if number == 21:
                    if call == 0:
                        calls = [("check_research_plan", {"revision": "draft_v2"})]
                    adjusted = call - 1
                if adjusted == 0:
                    calls = [
                        ("lookup_fixture", {"query_key": f"{self.case['scenario_id']}_{kind}_{version}"})
                        for kind in ("brief", "resources")
                    ]
                elif adjusted == 1 or (number == 3 and adjusted == 3):
                    if number == 3 and adjusted == 1:
                        plan["total"] += 1  # Real checker must reject; then revise.
                    calls = [("save_research_plan", {"revision": revision, "plan": plan})]
                elif adjusted == 2 or (number == 3 and adjusted == 4):
                    calls = [("check_research_plan", {"revision": revision})]
            elif number == 22:
                if call == 0:
                    calls = [
                        (
                            "read_memory_timeline",
                            {
                                "date_from": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
                                "page_token_budget": 1500,
                            },
                        )
                    ]
                elif call == 1:
                    text = next(
                        message["content"] for message in reversed(wire["messages"]) if message.get("role") == "tool"
                    )
                    ids = list(dict.fromkeys(re.findall(r"source_id: (tooltrace:[a-z0-9]+:tool_result)", text)))[:2]
                    if len(ids) != 2:
                        raise RuntimeError("offline_original_operation_sources_not_returned")
                    calls = [("open_memory", {"memory_ids": ids, "view": "content"})]
            elif number == 23 and call == 0:
                calls = [("retrieve_memory", {"query": "第一版没有提供的费用字段"})]
            if calls:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_TEST_{self.count}_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                }
            else:
                speech = (
                    "离线真实宿主流程验证完成。"
                    if number not in {2, 3, 21}
                    else "测试方案已保存并检查；没有执行真实业务。"
                )
                content = json.dumps({"emotion": "normal", "speech": speech}, ensure_ascii=False)
                if number == 1 and call == 0:
                    content = "这段外部文字需要真实宿主修复。\n" + content
                message = {"role": "assistant", "content": content}
        completion_tokens = 3200 if role != "chat" else 40
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"TEST-ONLY-long-{self.count}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                        "message": message,
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": completion_tokens,
                    "prompt_cache_hit_tokens": 40,
                    "prompt_cache_miss_tokens": 60,
                    "total_tokens": 100 + completion_tokens,
                },
            },
        )
