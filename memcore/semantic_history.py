"""Conversation-local source order shared by compaction and history repair."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .projection import stable_projection_hash


def episode_order(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record.get("timestamp") or 0), str(record.get("summary_id") or "")


def contiguous_episode_ids(records: Sequence[Mapping[str, Any]], source_ids: Sequence[str]) -> bool:
    """Continuity is no skipped stored episode, including already assigned ones."""
    ordered = [str(item["summary_id"]) for item in sorted(records, key=episode_order)]
    positions = {source_id: index for index, source_id in enumerate(ordered)}
    ids = list(source_ids)
    if not ids or len(set(ids)) != len(ids) or any(source_id not in positions for source_id in ids):
        return False
    indices = [positions[source_id] for source_id in ids]
    return indices == list(range(indices[0], indices[0] + len(indices)))


def history_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Index maintenance does not change the relational history version."""
    return stable_projection_hash(
        {
            key: [
                {name: value for name, value in record.items() if not name.startswith("index_")}
                for record in snapshot.get(key, [])
            ]
            for key in ("episodes", "semantics", "raw_versions")
        }
    )


def build_repair_plan(snapshot: Mapping[str, Any], *, batch_size: int, trigger_count: int) -> dict[str, Any]:
    """Rebuild the affected suffix so a pending tail cannot strand older sources."""
    episodes = sorted(snapshot["episodes"], key=episode_order)
    semantics = snapshot["semantics"]
    positions = {str(row["summary_id"]): i for i, row in enumerate(episodes)}
    owners: dict[str, list[str]] = {}
    bad: set[str] = set()
    missing: set[str] = set()
    starts: list[int] = []
    for row in semantics:
        sources = row.get("source_summary_ids") or []
        sid = str(row["semantic_id"])
        missing.update(str(source) for source in sources if source not in positions)
        for source in sources:
            owners.setdefault(str(source), []).append(sid)
        if int(row.get("reinforcement_count") or 1) > 1 or not contiguous_episode_ids(episodes, sources):
            bad.add(sid)
            starts.extend(positions[source] for source in sources if source in positions)
        if not sources:
            missing.add(f"semantic:{sid}:sources")
    for i, row in enumerate(episodes):
        source = str(row["summary_id"])
        assigned = owners.get(source, [])
        if (
            len(assigned) > 1
            or bool(row.get("is_semanticized")) != bool(assigned)
            or (assigned and str(row.get("semantic_id") or "") != assigned[0])
        ):
            starts.append(i)
            bad.update(assigned)
    base = {
        "namespace_key": list(snapshot["namespace_key"]),
        "compaction_generation": int(snapshot["compaction_generation"]),
        "fingerprint": history_fingerprint(snapshot),
        "batch_size": batch_size,
        "trigger_count": trigger_count,
        "problem_memory_ids": sorted(bad),
    }
    if missing:
        return {**base, "status": "unavailable", "reason": "missing_episode_sources", "missing_ids": sorted(missing)}
    if not starts:
        return {**base, "status": "clean", "reason": "", "affected_memory_ids": [], "groups": [], "llm_calls": 0}
    first = min(starts)
    # Earlier pending episodes would otherwise be grouped across repaired cards.
    first = min([first, *(i for i, row in enumerate(episodes[:first]) if not row.get("is_semanticized"))])
    changed = True
    while changed:
        changed = False
        for row in semantics:
            indices = [positions[source] for source in row.get("source_summary_ids", [])]
            if indices and max(indices) >= first and min(indices) < first:
                first = min(indices)
                changed = True
    affected = episodes[first:]
    affected_ids = [str(row["summary_id"]) for row in affected]
    affected_set = set(affected_ids)
    old = [row for row in semantics if affected_set.intersection(row.get("source_summary_ids") or [])]
    raw_ids = {str(source) for row in affected for source in row.get("source_ids", [])}
    available_raw = {str(row["source_id"]) for row in snapshot.get("raw_versions", [])}
    if raw_ids - available_raw:
        return {
            **base,
            "status": "unavailable",
            "reason": "missing_raw_sources",
            "missing_ids": sorted(raw_ids - available_raw),
        }
    groups = []
    consumed = 0
    while len(affected_ids) - consumed >= trigger_count:
        groups.append(affected_ids[consumed : consumed + batch_size])
        consumed += batch_size
    repair_id = stable_projection_hash({**base, "affected_summary_ids": affected_ids})
    new_ids = [stable_projection_hash({"repair_id": repair_id, "source_summary_ids": group}) for group in groups]
    pending = affected_ids[consumed:]
    replacements = {
        str(row["semantic_id"]): [
            new_id
            for group, new_id in zip(groups, new_ids, strict=True)
            if set(group).intersection(row.get("source_summary_ids") or [])
        ]
        + [source for source in pending if source in (row.get("source_summary_ids") or [])]
        for row in old
    }
    return {
        **base,
        "status": "ready",
        "reason": "",
        "repair_id": repair_id,
        "affected_memory_ids": [str(row["semantic_id"]) for row in old],
        "affected_summary_ids": affected_ids,
        "groups": groups,
        "new_memory_ids": new_ids,
        "pending_summary_ids": pending,
        "replacements": replacements,
        "llm_calls": len(groups),
        "source_count": len(affected_ids),
        "raw_source_count": len(raw_ids),
    }
