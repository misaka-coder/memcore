"""Catalog tags derived from existing metadata, without a second taxonomy."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any


def normalize_catalog_terms(value: Any, *, strict: bool = False) -> list[str]:
    """Deduplicate complete terms; retain punctuation and the first display spelling."""

    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        if strict:
            raise ValueError("keywords_must_be_string_array")
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            if strict:
                raise ValueError("keywords_must_contain_nonempty_strings")
            continue
        term = " ".join(unicodedata.normalize("NFKC", item).split())
        key = term.casefold()
        if key and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def catalog_tag_fields(metadata: Any) -> dict[str, list[str]]:
    """Keep entity/topic display roles, deduplicating across both fields."""

    source = metadata if isinstance(metadata, Mapping) else {}
    entities = normalize_catalog_terms(source.get("entity_anchors"))
    entity_keys = {term.casefold() for term in entities}
    topics = [term for term in normalize_catalog_terms(source.get("topic_terms")) if term.casefold() not in entity_keys]
    return {"entity_anchors": entities, "topic_terms": topics}


def catalog_metadata_terms(metadata: Any) -> list[str]:
    fields = catalog_tag_fields(metadata)
    return [*fields["entity_anchors"], *fields["topic_terms"]]


def catalog_keyword_hits(pool: list[str], keywords: list[str]) -> list[dict[str, str]]:
    """Return one stored witness per query: exact first, then contains.

    Inputs already have NFKC/whitespace normalization. Matching is deliberately
    one-way: a short stored tag cannot substantiate a longer query. Preserve the
    first stored spelling and never copy the full source pool into a result.
    """

    stored = {term.casefold(): term for term in reversed(pool)}
    hits = []
    for query in keywords:
        key = query.casefold()
        witness = stored.get(key)
        if witness is None:
            witness = next((term for term in pool if key in term.casefold()), None)
        if witness is not None:
            hits.append({"query": query, "term": witness})
    return hits
