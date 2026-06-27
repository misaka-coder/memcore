"""把记忆记录构建成 VectorIndex 条目。

向量 text 只放语义内容(不含时间字符串,防污染);时间/隔离/标签进 metadata。
关键词侧的 metadata tag 文本(memory_keywords_text 等)由 InMemoryVectorIndex 拼进 BM25 文本。
"""

from __future__ import annotations

from typing import Any

from ..text_utils import join_tags


def _safe_importance(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def _scope_meta(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "tenant_id": str(record.get("tenant_id") or ""),
        "user_id": str(record.get("user_id") or ""),
        "domain_id": str(record.get("domain_id") or ""),
        "conversation_id": str(record.get("conversation_id") or ""),
        "timestamp": int(record.get("timestamp") or 0),
        "date_label": str(record.get("date_label") or ""),
        "time_of_day": str(record.get("time_of_day") or ""),
    }


def _metadata_tags(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("memory_metadata") if isinstance(record.get("memory_metadata"), dict) else {}
    return {
        "memory_keywords_text": join_tags(meta.get("keywords")),
        "memory_subject_scopes_text": join_tags(meta.get("subject_scopes")),
        "memory_categories_text": join_tags(meta.get("categories")),
        "memory_mood_tags_text": join_tags(meta.get("mood_tags")),
        "memory_importance": _safe_importance(
            meta.get("importance") if "importance" in meta else record.get("importance")
        ),
        "semantic_tags_text": join_tags(record.get("semantic_tags")),
    }


def build_raw_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": record["source_id"],
        "text": str(record.get("content") or ""),
        "metadata": {
            **_scope_meta(record),
            "entry_type": "raw",
            "speaker": str(record.get("role") or ""),
            **_metadata_tags(record),
        },
    }


def build_summary_entry(record: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [str(record.get("diary_summary") or "")]
        + [str(x) for x in (record.get("key_events") or [])]
        + [str(x) for x in (record.get("core_facts") or [])]
    )
    return {
        "source_id": record["summary_id"],
        "text": text,
        "metadata": {**_scope_meta(record), "entry_type": "summary", **_metadata_tags(record)},
    }


def build_semantic_entry(record: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [str(record.get("semantic_summary") or "")]
        + [str(x) for x in (record.get("stable_facts") or [])]
        + [str(x) for x in (record.get("recurring_topics") or [])]
        + [str(x) for x in (record.get("important_people") or [])]
        + [str(x) for x in (record.get("open_loops") or [])]
    )
    return {
        "source_id": record["semantic_id"],
        "text": text,
        "metadata": {**_scope_meta(record), "entry_type": "semantic_summary", **_metadata_tags(record)},
    }
