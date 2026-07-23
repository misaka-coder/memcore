"""把记忆记录构建成 VectorIndex 条目。

向量 text 只放语义内容(不含时间字符串,防污染);时间/隔离/标签进 metadata。
关键词侧的 metadata tag 文本(memory_keywords_text 等)由 InMemoryVectorIndex 拼进 BM25 文本。
"""

from __future__ import annotations

from typing import Any

from ..text_utils import join_tags
from .metadata_filters import (
    ENTITY_FLAG_SCHEMA_VERSION,
    INDEX_SCHEMA_KEY,
    INDEX_SCHEMA_VERSION,
    KIND_FLAG_SCHEMA_VERSION,
    VISIBILITY_SCHEMA_VERSION,
    kind_filter_flags,
    metadata_filter_flags,
)


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
        "memory_entity_text": join_tags(meta.get("entity_anchors")),
        "memory_topic_text": join_tags(meta.get("topic_terms")),
        "memory_facets_text": join_tags(meta.get("memory_facets")),
        "memory_about_roles_text": join_tags(meta.get("about_roles")),
        "memory_mood_tags_text": join_tags(meta.get("mood_tags")),
        "memory_priority": str(meta.get("retrieval_priority") or "normal"),
        "semantic_tags_text": join_tags(record.get("semantic_tags")),
        **metadata_filter_flags(meta),
    }


def _retrieval_metadata(record: dict[str, Any], *, default_kind: str) -> dict[str, Any]:
    kind = str(record.get("kind") or default_kind).strip().lower()
    visibility = str(record.get("retrieval_visibility") or "explicit").strip().lower()
    annotation_status = str(record.get("annotation_status") or "unannotated").strip().lower()
    trust = str(record.get("trust") or "untrusted_data").strip().lower()
    lineage_status = str(record.get("lineage_status") or "raw").strip().lower()
    trace_metadata = record.get("trace_metadata") if isinstance(record.get("trace_metadata"), dict) else {}
    root = kind.split(".", 1)[0]
    return {
        "kind_exact": kind,
        **kind_filter_flags(kind),
        "is_trace_kind": root in {"material", "tool"},
        "retrieval_visibility": visibility,
        "retrieval_policy": str(record.get("retrieval_policy") or "auto").strip().lower(),
        "annotation_status": annotation_status,
        "trust": trust,
        "lineage_status": lineage_status,
        "is_legacy_migration": annotation_status == "accepted_legacy" or bool(trace_metadata.get("legacy_categories")),
        "turn_id": str(record.get("turn_id") or ""),
        "turn_role": str(record.get("turn_role") or ""),
        "correlation_id": str(record.get("correlation_id") or ""),
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "index_schema_key": INDEX_SCHEMA_KEY,
        "kind_flag_schema_version": KIND_FLAG_SCHEMA_VERSION,
        "visibility_schema_version": VISIBILITY_SCHEMA_VERSION,
        "entity_flag_schema_version": ENTITY_FLAG_SCHEMA_VERSION,
    }


def build_raw_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": record["source_id"],
        "text": str(record.get("semantic_text") or record.get("content") or ""),
        "metadata": {
            **_scope_meta(record),
            "entry_type": "raw",
            "speaker": str(record.get("role") or ""),
            **_retrieval_metadata(record, default_kind="legacy.unknown"),
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
        "metadata": {
            **_scope_meta(record),
            "entry_type": "summary",
            **_retrieval_metadata(record, default_kind="memory.episode_summary"),
            **_metadata_tags(record),
        },
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
        "metadata": {
            **_scope_meta(record),
            "entry_type": "semantic_summary",
            **_retrieval_metadata(record, default_kind="memory.semantic_summary"),
            **_metadata_tags(record),
        },
    }
