"""写侧:三层压缩(raw→摘要→语义)+ 强化合并。注入 LLMClient,按 namespace 加锁同步执行。

差值关系(批量 < 触发数)由 MemoryConfig 保证。强化合并按主题重叠驱动(非位置淘汰):
新长期记忆形成时,回看最近 N 条,与重叠最高且 >= 阈值者融合,否则新建。

并发:压缩核心每 namespace 串行锁,压缩中不与自身重入(代际安全);MemorySystem 提供后台提交入口。
向量索引用 outbox:先写库(pending),upsert 成功后 set_index_status('indexed')。
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import MemoryConfig
from .index.base import VectorIndex
from .index.entry_builder import build_semantic_entry, build_summary_entry
from .llm.base import LLMClient, LLMRequest, ResponseFormat, TaskType
from .namespace import Namespace
from .prompts import PromptOverrides, build_reinforcement_prompts, build_semantic_prompts, build_summary_prompts
from .schema import coerce_memory_metadata
from .store.base import MemoryStore
from .text_utils import normalize_text
from .time_anchor import infer_time_of_day, timestamp_to_date_label


@dataclass
class JsonCallResult:
    ok: bool
    data: dict[str, Any]


class Compaction:
    def __init__(
        self,
        *,
        store: MemoryStore,
        index: VectorIndex,
        llm: LLMClient,
        config: MemoryConfig,
        timezone: str,
        overrides: PromptOverrides | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.llm = llm
        self.config = config
        self.timezone = timezone
        self.overrides = overrides or PromptOverrides()
        self._locks: dict[tuple[str, str, str], threading.RLock] = defaultdict(threading.RLock)

    def run_due(self, *, namespace: Namespace) -> dict[str, int]:
        result = {
            "summaries_created": 0,
            "semantic_created": 0,
            "reinforced": 0,
            "summary_retry_pending": 0,
            "semantic_retry_pending": 0,
        }
        with self._locks[namespace.hard_key()]:  # 同 namespace 串行,压缩中不自重入
            self._summarize_raw(namespace, result)
            self._semanticize_episodic(namespace, result)
        return result

    # --- raw → 阶段摘要 ---

    def _summarize_raw(self, namespace: Namespace, result: dict[str, int]) -> None:
        cfg = self.config
        msgs = self.store.get_unsummarized_messages(namespace=namespace)
        while len(msgs) >= cfg.raw_trigger_count:
            batch = msgs[: cfg.summary_batch_size]
            call = self._call_json(
                TaskType.SUMMARY,
                *build_summary_prompts(
                    transcript=self._render_transcript(batch),
                    batch_size=len(batch),
                    overrides=self.overrides,
                    enable_flavor=self.config.enable_flavor,
                    reference_summary_text=self._render_reference_summaries(namespace),
                ),
                fallback={"diary_summary": "", "importance": 0.3, "key_events": [], "core_facts": []},
            )
            if not call.ok or not _has_summary_content(call.data):
                result["summary_retry_pending"] += 1
                return
            payload = call.data
            start_ts = min(int(m["timestamp"]) for m in batch)
            end_ts = max(int(m["timestamp"]) for m in batch)
            record = {
                "summary_id": uuid.uuid4().hex,
                "timestamp": end_ts,
                "period_start_ts": start_ts,
                "period_end_ts": end_ts,
                "date_label": timestamp_to_date_label(end_ts, self.timezone),
                "time_of_day": infer_time_of_day(end_ts, self.timezone),
                "period_label": str(payload.get("period_label") or ""),
                "event_type": str(payload.get("event_type") or ""),
                "importance": _clamp01(payload.get("importance")),
                "diary_summary": str(payload.get("diary_summary") or ""),
                "key_events": _str_list(payload.get("key_events")),
                "core_facts": _str_list(payload.get("core_facts")),
                "memory_metadata": coerce_memory_metadata(
                    payload.get("memory_metadata"), categories=cfg.categories, enable_flavor=cfg.enable_flavor
                ).to_dict(),
                "semantic_tags": coerce_memory_metadata(payload.get("memory_metadata")).keywords,
            }
            saved = self.store.add_summary(namespace=namespace, record=record)
            self.store.mark_messages_summarized([str(m["source_id"]) for m in batch], saved["summary_id"])
            self._index(build_summary_entry(saved), saved["summary_id"])
            result["summaries_created"] += 1
            msgs = self.store.get_unsummarized_messages(namespace=namespace)

    # --- 阶段摘要 → 长期语义记忆(含强化合并) ---

    def _semanticize_episodic(self, namespace: Namespace, result: dict[str, int]) -> None:
        cfg = self.config
        eps = self.store.get_uncompacted_episodic_summaries(namespace=namespace)
        while len(eps) >= cfg.episodic_compact_trigger_count:
            batch = eps[: cfg.episodic_compact_batch_size]
            call = self._call_json(
                TaskType.SEMANTIC,
                *build_semantic_prompts(
                    source_text=self._render_episodes(batch),
                    overrides=self.overrides,
                    enable_flavor=self.config.enable_flavor,
                ),
                fallback={"semantic_summary": "", "importance": 0.4, "stable_facts": []},
            )
            if not call.ok or not _has_semantic_content(call.data):
                result["semantic_retry_pending"] += 1
                return
            payload = call.data
            start_ts = min(int(s.get("period_start_ts") or s.get("timestamp") or 0) for s in batch)
            end_ts = max(int(s.get("period_end_ts") or s.get("timestamp") or 0) for s in batch)
            incoming = {
                "timestamp": end_ts,
                "period_start_ts": start_ts,
                "period_end_ts": end_ts,
                "date_label": timestamp_to_date_label(end_ts, self.timezone),
                "time_of_day": infer_time_of_day(end_ts, self.timezone),
                "importance": _clamp01(payload.get("importance")),
                "semantic_summary": str(payload.get("semantic_summary") or ""),
                "stable_facts": _str_list(payload.get("stable_facts")),
                "recurring_topics": _str_list(payload.get("recurring_topics")),
                "important_people": _str_list(payload.get("important_people")),
                "open_loops": _str_list(payload.get("open_loops")),
                "memory_metadata": coerce_memory_metadata(
                    payload.get("memory_metadata"), categories=cfg.categories, enable_flavor=cfg.enable_flavor
                ).to_dict(),
                "source_summary_ids": [str(s["summary_id"]) for s in batch],
            }
            incoming["semantic_tags"] = list(incoming["memory_metadata"].get("keywords") or [])

            target = self._find_reinforcement_target(namespace, incoming)
            if target is not None:
                record = self._merge_reinforcement(target, incoming)
                result["reinforced"] += 1
            else:
                record = {
                    "semantic_id": uuid.uuid4().hex,
                    "reinforcement_count": 1,
                    "last_reinforced_ts": end_ts,
                    **incoming,
                }
                result["semantic_created"] += 1

            saved = self.store.add_semantic_summary(namespace=namespace, record=record)
            self.store.mark_summaries_semanticized([str(s["summary_id"]) for s in batch], saved["semantic_id"])
            self._index(build_semantic_entry(saved), saved["semantic_id"])
            eps = self.store.get_uncompacted_episodic_summaries(namespace=namespace)

    def _find_reinforcement_target(self, namespace: Namespace, incoming: dict[str, Any]) -> dict[str, Any] | None:
        cfg = self.config
        candidates = self.store.get_recent_semantic_summaries(
            namespace=namespace, limit=cfg.semantic_reinforcement_lookback
        )
        best, best_score = None, 0
        for cand in candidates:
            score = _overlap_score(cand, incoming)
            if score >= cfg.semantic_reinforcement_min_overlap and score > best_score:
                best, best_score = cand, score
        return best

    def _merge_reinforcement(self, target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        existing_count = max(1, int(target.get("reinforcement_count") or 1))
        blended = (float(target.get("importance") or 0.5) * existing_count + float(incoming["importance"])) / (
            existing_count + 1
        )
        fallback = {
            "semantic_summary": target.get("semantic_summary") or incoming["semantic_summary"],
            "importance": max(float(target.get("importance") or 0.0), float(incoming["importance"]), blended),
            "stable_facts": _merge_unique(target.get("stable_facts"), incoming["stable_facts"]),
            "recurring_topics": _merge_unique(target.get("recurring_topics"), incoming["recurring_topics"]),
            "important_people": _merge_unique(target.get("important_people"), incoming["important_people"]),
            "open_loops": _merge_unique(target.get("open_loops"), incoming["open_loops"]),
        }
        call = self._call_json(
            TaskType.REINFORCEMENT,
            *build_reinforcement_prompts(
                existing_text=_render_semantic_text(target),
                incoming_text=_render_semantic_text(incoming),
                overrides=self.overrides,
                enable_flavor=self.config.enable_flavor,
            ),
            fallback=fallback,
        )
        payload = call.data
        return {
            "semantic_id": target["semantic_id"],  # 同 id 覆盖 = 强化
            "timestamp": incoming["timestamp"],
            "period_start_ts": min(
                int(target.get("period_start_ts") or 0) or incoming["period_start_ts"], incoming["period_start_ts"]
            ),
            "period_end_ts": max(int(target.get("period_end_ts") or 0), incoming["period_end_ts"]),
            "date_label": incoming["date_label"],
            "time_of_day": incoming["time_of_day"],
            "importance": _clamp01(payload.get("importance", fallback["importance"])),
            "semantic_summary": str(payload.get("semantic_summary") or fallback["semantic_summary"]),
            "stable_facts": _str_list(payload.get("stable_facts")) or fallback["stable_facts"],
            "recurring_topics": _str_list(payload.get("recurring_topics")) or fallback["recurring_topics"],
            "important_people": _str_list(payload.get("important_people")) or fallback["important_people"],
            "open_loops": _str_list(payload.get("open_loops")) or fallback["open_loops"],
            "memory_metadata": incoming["memory_metadata"],
            "semantic_tags": _merge_unique(target.get("semantic_tags"), incoming.get("semantic_tags")),
            "source_summary_ids": _merge_unique(target.get("source_summary_ids"), incoming["source_summary_ids"]),
            "reinforcement_count": int(target.get("reinforcement_count") or 1) + 1,
            "last_reinforced_ts": incoming["timestamp"],
        }

    # --- 工具 ---

    def _call_json(
        self, task_type: TaskType, system_prompt: str, user_prompt: str, *, fallback: dict
    ) -> JsonCallResult:
        res = self.llm.call(
            LLMRequest(
                task_type=task_type,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=ResponseFormat.JSON,
                max_retries=self.config.llm_max_retries,
                fallback=fallback,
            )
        )
        if res.ok and isinstance(res.data, dict):
            return JsonCallResult(ok=True, data=res.data)
        return JsonCallResult(ok=False, data=fallback)

    def _index(self, entry: dict[str, Any], source_id: str) -> None:
        # outbox:库已 pending,upsert 成功才置 indexed;失败保持 pending,留给 reindex_pending 自愈。
        try:
            self.index.upsert([entry])
            self.store.set_index_status(source_id, "indexed")
        except Exception:
            return

    def _render_reference_summaries(self, namespace: Namespace) -> str:
        """取本会话最近的既有阶段摘要作参考,帮新摘要与旧摘要保持一致、避免冲突。"""
        existing = self.store.get_visible_episodic_summaries(
            namespace=namespace, limit=self.config.episodic_visible_max
        )
        lines = []
        for s in existing:
            diary = normalize_text(s.get("diary_summary"))
            facts = "; ".join(str(f) for f in (s.get("core_facts") or []))
            label = str(s.get("date_label") or "")
            lines.append(f"- [{label}] {diary}" + (f"(事实:{facts})" if facts else ""))
        return "\n".join(lines)

    @staticmethod
    def _render_transcript(batch: list[dict[str, Any]]) -> str:
        return "\n".join(f"{m.get('role', '')}: {normalize_text(m.get('content'))}" for m in batch)

    @staticmethod
    def _render_episodes(batch: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"- {normalize_text(s.get('diary_summary'))} | 事实: {'; '.join(s.get('core_facts') or [])}" for s in batch
        )


def _clamp01(value: Any) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, n))


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out, seen = [], set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _has_summary_content(payload: dict[str, Any]) -> bool:
    return bool(
        normalize_text(payload.get("diary_summary"))
        or _str_list(payload.get("key_events"))
        or _str_list(payload.get("core_facts"))
    )


def _has_semantic_content(payload: dict[str, Any]) -> bool:
    return bool(
        normalize_text(payload.get("semantic_summary"))
        or _str_list(payload.get("stable_facts"))
        or _str_list(payload.get("recurring_topics"))
        or _str_list(payload.get("important_people"))
        or _str_list(payload.get("open_loops"))
    )


def _merge_unique(*lists: Any, limit: int = 8) -> list[str]:
    out, seen = [], set()
    for lst in lists:
        for item in lst or []:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
                if len(out) >= limit:
                    return out
    return out


def _overlap_score(candidate: dict[str, Any], incoming: dict[str, Any]) -> int:
    def norm_set(record: dict[str, Any], key: str) -> set[str]:
        return {normalize_text(x) for x in (record.get(key) or []) if normalize_text(x)}

    score = 0
    for key in ("semantic_tags", "recurring_topics", "important_people"):
        score += len(norm_set(candidate, key) & norm_set(incoming, key))
    if norm_set(candidate, "stable_facts") & norm_set(incoming, "stable_facts"):
        score += 1
    return score


def _render_semantic_text(record: dict[str, Any]) -> str:
    parts = [str(record.get("semantic_summary") or "")]
    parts += [str(x) for x in (record.get("stable_facts") or [])]
    parts += [str(x) for x in (record.get("recurring_topics") or [])]
    return " ".join(p for p in parts if p)
