"""确定性终局紧凑投影(文档 §7/§12): 无 LLM、无 per-tool renderer、无 Mermaid。

只组合 action/observation 已保证存在的协议字段, 按单一全局字节门槛分类:
  - inline_full: 短反馈/低收益, 保留完整正文(逐字节不变);
  - compact_reloadable: 长正文且 compact 引用明显更小, 替换为可回读引用;
  - explicit_empty: 无正文(不替换, 保持宿主记录的 observation 形状)。
硬约束: compact 必须严格小于 full(no-expansion); 默认节省严格超过一半才执行。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from functools import partial

from .errors import SchemaError
from .config import OperationProjectionPolicy
from .projection import (
    ANTHROPIC_PROFILE,
    CANONICAL_PROFILE,
    OPENAI_PROFILE,
    ProjectionMessageInput,
    canonical_json_bytes,
    stable_projection_hash,
)
from .time_anchor import timestamp_to_datetime_label
from .token_counter import estimate_text_tokens

COMPACT_RELOAD_MIN_INLINE_BYTES = 256  # 正文低于该字节数直接 inline_full(短反馈)
COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO = 0.5  # compact 必须 < full 的 1-ratio(明显收益)
ARGUMENT_MIN_UTF8_BYTES = 1024
ARGUMENT_PREVIEW_CHARS = 96
SETTLED_PROJECTION_SCHEMA_VERSION = 1
TOKEN_COUNT_QUALITY = "estimated"

INLINE_FULL = "inline_full"
COMPACT_RELOADABLE = "compact_reloadable"
EXPLICIT_EMPTY = "explicit_empty"

_PROJECTION_KINDS = (INLINE_FULL, COMPACT_RELOADABLE, EXPLICIT_EMPTY)


def utf8_bytes(value: Any) -> int:
    return len(str(value or "").encode("utf-8"))


def _tool_label(entry: Any) -> str:
    trace_name = str((entry.trace_metadata or {}).get("tool_name") or "").strip()
    if trace_name:
        return trace_name[:64]
    name = str(getattr(entry, "kind", "") or "").removeprefix("tool.")
    for suffix in (".call", ".result"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name[:64] or "tool"


def _observation_status(entry: Any) -> str:
    return str((entry.trace_metadata or {}).get("status") or (entry.payload or {}).get("status") or "").strip()


def render_compact_reload(entry: Any, *, full_content: str, timezone: str = "", include_call_id: bool = True) -> str:
    """通用可回读引用(文档 §7.2 推荐形态的确定性文本化)。

    只保留识别与回读所需字段; 调用配对和校验哈希仍由投影账本保存。
    ``timezone`` 非空时从 entry 既有 timestamp 派生可读时间。
    只有原生消息已携带调用 ID 时才省略重复行; 纯文本回放保留配对标识。
    已冻结的历史卡片从账本原样读取, 不经本函数重新渲染。
    """
    payload_chars = len(str(full_content or ""))
    status = _observation_status(entry)
    lines = ["[compact_reloadable]"]
    if str(timezone or "").strip():
        timestamp = int(getattr(entry, "timestamp", 0) or 0)
        if timestamp > 0:
            lines.append(f"time: {timestamp_to_datetime_label(timestamp, str(timezone).strip())}")
    lines.append(f"tool: {_tool_label(entry)}")
    if include_call_id:
        lines.append(f"call_id: {str(getattr(entry, 'correlation_id', '') or '')}")
    if status:
        lines.append(f"status: {status}")
    lines.extend(
        [
            f"source_id: {str(getattr(entry, 'source_id', '') or '')}",
            f"stored_chars: {payload_chars}",
        ]
    )
    return "\n".join(lines)


def classify_observation(
    entry: Any,
    full_content: str,
    *,
    min_inline_bytes: int = COMPACT_RELOAD_MIN_INLINE_BYTES,
    required_savings_ratio: float = COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO,
    timezone: str = "",
    include_call_id: bool = True,
) -> tuple[str, str]:
    """确定性分类, 返回 (kind, projection_content)。

    - 空正文 → explicit_empty(不替换);
    - 正文低于门槛 → inline_full;
    - compact 引用未明显更小(no-expansion 硬约束) → inline_full;
    - 否则 → compact_reloadable。
    ``timezone`` 非空时紧凑卡片附带通用可读时间(仅新增字段, 不复制工具入参)。
    """
    if not str(full_content or "").strip():
        return (EXPLICIT_EMPTY, str(full_content or ""))
    full_bytes = utf8_bytes(full_content)
    if full_bytes < min_inline_bytes:
        return (INLINE_FULL, str(full_content))
    compact = render_compact_reload(
        entry, full_content=full_content, timezone=timezone, include_call_id=include_call_id
    )
    compact_bytes = utf8_bytes(compact)
    savings_ratio = min(0.99, max(0.0, float(required_savings_ratio)))
    if compact_bytes >= full_bytes * (1.0 - savings_ratio):
        return (INLINE_FULL, str(full_content))
    return (COMPACT_RELOADABLE, compact)


@dataclass(frozen=True)
class SettledProjectionPlan:
    """一次 complete_turn 的 settled 投影结果(文档 §6/§12)。"""

    turn_id: str
    provider_profile: str
    settlement_status: str  # settled | settled_noop | full_fallback
    messages: tuple[ProjectionMessageInput, ...] = ()
    full_messages: tuple[ProjectionMessageInput, ...] = ()
    first_changed_projection_index: int = -1
    full_projected_tokens: int = 0
    settled_projected_tokens: int = 0
    full_projection_hash: str = ""
    settled_projection_hash: str = ""
    token_count_quality: str = TOKEN_COUNT_QUALITY
    reason: str = ""
    observation_kinds: dict[str, str] = field(default_factory=dict)
    action_kinds: dict[str, str] = field(default_factory=dict)
    settlement_min_utf8_bytes: int = COMPACT_RELOAD_MIN_INLINE_BYTES
    settlement_min_saved_ratio: float = COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO
    settlement_config_hash: str = ""
    settlement_argument_min_utf8_bytes: int = ARGUMENT_MIN_UTF8_BYTES
    settlement_argument_preview_chars: int = ARGUMENT_PREVIEW_CHARS

    @property
    def replaced(self) -> bool:
        return self.settlement_status == "settled" and self.first_changed_projection_index >= 0

    @property
    def saved_projected_tokens(self) -> int:
        return max(0, self.full_projected_tokens - self.settled_projected_tokens)


def _count_payload_tokens(
    messages: Sequence[ProjectionMessageInput],
    count_text: Callable[[str], int] | None,
) -> int:
    """Use the same canonical provider-payload accounting shape as raw compaction."""

    total = 0
    for message in messages:
        text = canonical_json_bytes(message.payload).decode("utf-8")
        count = estimate_text_tokens(text) if count_text is None else int(count_text(text))
        if count < 0:
            raise ValueError("TokenCounter.count_text() must return a non-negative int")
        total += count + 4
    return total


def _as_projection_input(message: Any) -> ProjectionMessageInput:
    return ProjectionMessageInput(
        provider_profile=str(getattr(message, "provider_profile", "") or ""),
        payload=dict(getattr(message, "payload", {}) or {}),
        source_ids=tuple(getattr(message, "source_ids", ()) or ()),
        projection_index=int(getattr(message, "projection_index", -1)),
        projection_status=getattr(message, "projection_status", "complete"),
        projection_version=int(getattr(message, "projection_version", 1) or 1),
    )


def _is_observation(entry: Any) -> bool:
    return str(getattr(entry, "turn_role", "") or "").endswith("observation")


def compact_argument_value(
    value: Any,
    *,
    source_id: str,
    min_utf8_bytes: int = ARGUMENT_MIN_UTF8_BYTES,
    preview_chars: int = ARGUMENT_PREVIEW_CHARS,
    saved_ratio: float = 0.5,
) -> tuple[Any, bool]:
    """Keep argument shape; shorten only eligible string leaves of objects.

    Arrays are deliberately opaque. The source ID opens the immutable action,
    so field names, call IDs and tool names need no duplicate markers.
    """
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            result[key], replaced = compact_argument_value(
                item,
                source_id=source_id,
                min_utf8_bytes=min_utf8_bytes,
                preview_chars=preview_chars,
                saved_ratio=saved_ratio,
            )
            changed = changed or replaced
        return (result if changed else value), changed
    if not isinstance(value, str) or utf8_bytes(value) < min_utf8_bytes:
        return value, False
    preview = value[:preview_chars] + f"\n[truncated source_id={source_id}]"
    if utf8_bytes(preview) >= utf8_bytes(value) * (1.0 - saved_ratio):
        return value, False
    return preview, True


def _has_opaque_signature(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in {"signature", "thought_signature", "thoughtSignature", "encrypted_content"}
            or (str(key) not in {"arguments", "input"} and _has_opaque_signature(item))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_opaque_signature(item) for item in value)
    return False


def _settle_action_payload(
    payload: dict[str, Any],
    actions: Sequence[Any],
    *,
    provider_profile: str,
    min_utf8_bytes: int,
    preview_chars: int,
    saved_ratio: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    kinds = {entry.source_id: INLINE_FULL for entry in actions}
    by_call = {str(entry.correlation_id): entry for entry in actions}
    if len(by_call) != len(actions):
        raise SchemaError("settlement_action_correlation_ambiguous")

    def compact(value: Any, entry: Any) -> Any:
        replacement, changed = compact_argument_value(
            value,
            source_id=entry.source_id,
            min_utf8_bytes=min_utf8_bytes,
            preview_chars=preview_chars,
            saved_ratio=saved_ratio,
        )
        if changed:
            kinds[entry.source_id] = COMPACT_RELOADABLE
        return replacement

    result = dict(payload)
    if provider_profile == OPENAI_PROFILE:
        calls = payload.get("tool_calls")
        if payload.get("role") != "assistant" or not isinstance(calls, list):
            raise SchemaError("settlement_action_projection_shape_unsupported")
        updated_calls = []
        seen: set[str] = set()
        for call in calls:
            if not isinstance(call, Mapping) or call.get("type") != "function":
                raise SchemaError("settlement_action_projection_shape_unsupported")
            call_id = str(call.get("id") or "")
            function = call.get("function")
            if call_id not in by_call or call_id in seen or not isinstance(function, Mapping):
                raise SchemaError("settlement_action_projection_shape_unsupported")
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                raise SchemaError("settlement_action_arguments_invalid")
            try:
                parsed = json.loads(arguments)
            except (ValueError, TypeError) as exc:
                raise SchemaError("settlement_action_arguments_invalid") from exc
            replacement = compact(parsed, by_call[call_id])
            # Keep untouched JSON strings byte-identical, including whitespace.
            if replacement is not parsed:
                call = {
                    **call,
                    "function": {
                        **function,
                        "arguments": json.dumps(replacement, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            updated_calls.append(call)
            seen.add(call_id)
        if seen != set(by_call):
            raise SchemaError("settlement_action_source_coverage_mismatch")
        result["tool_calls"] = updated_calls
    elif provider_profile == ANTHROPIC_PROFILE:
        blocks = payload.get("content")
        if payload.get("role") != "assistant" or not isinstance(blocks, list):
            raise SchemaError("settlement_action_projection_shape_unsupported")
        updated_blocks = []
        seen = set()
        for block in blocks:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                updated_blocks.append(block)
                continue
            call_id = str(block.get("id") or "")
            if call_id not in by_call or call_id in seen or "input" not in block:
                raise SchemaError("settlement_action_projection_shape_unsupported")
            replacement = compact(block["input"], by_call[call_id])
            updated_blocks.append({**block, "input": replacement} if replacement is not block["input"] else block)
            seen.add(call_id)
        if seen != set(by_call):
            raise SchemaError("settlement_action_source_coverage_mismatch")
        result["content"] = updated_blocks
    else:
        content = payload.get("content")
        if len(actions) != 1 or not isinstance(content, str):
            raise SchemaError("settlement_action_projection_shape_unsupported")
        prefix, separator, data = content.rpartition("\ndata:\n")
        if not separator:
            return payload, kinds  # Custom canonical renderer: no trusted input section.
        try:
            parsed = json.loads(data)
        except ValueError as exc:
            raise SchemaError("settlement_action_arguments_invalid") from exc
        if not isinstance(parsed, dict):
            return payload, kinds
        key = "input" if "input" in parsed else "arguments" if "arguments" in parsed else ""
        if key:
            replacement = compact(parsed[key], actions[0])
            parsed = {**parsed, key: replacement}
        else:
            parsed = compact(parsed, actions[0])
        if any(kind == COMPACT_RELOADABLE for kind in kinds.values()):
            result["content"] = (
                prefix + separator + json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
    if any(kind == COMPACT_RELOADABLE for kind in kinds.values()) and _has_opaque_signature(payload):
        raise SchemaError("settlement_signed_action_requires_full_replay")
    return result, kinds


def _settle_authoritative_messages(
    entries: Sequence[Any],
    full_messages: Sequence[Any],
    *,
    provider_profile: str,
    observation_decider: Callable[[Any, str], tuple[str, str]],
    compact_arguments: bool = False,
    argument_min_utf8_bytes: int = ARGUMENT_MIN_UTF8_BYTES,
    argument_preview_chars: int = ARGUMENT_PREVIEW_CHARS,
    saved_ratio: float = 0.5,
) -> tuple[tuple[ProjectionMessageInput, ...], dict[str, str], dict[str, str]]:
    """Settle eligible values inside the exact frozen provider projection."""

    if provider_profile not in {CANONICAL_PROFILE, OPENAI_PROFILE, ANTHROPIC_PROFILE}:
        raise SchemaError("settlement_provider_profile_unsupported")
    entry_by_source = {
        str(getattr(entry, "source_id", "") or ""): entry
        for entry in entries
        if str(getattr(entry, "source_id", "") or "")
    }
    settled: list[ProjectionMessageInput] = []
    kinds: dict[str, str] = {}
    action_kinds: dict[str, str] = {}
    for raw_message in full_messages:
        message = _as_projection_input(raw_message)
        observations = [
            entry_by_source[source_id]
            for source_id in message.source_ids
            if source_id in entry_by_source and _is_observation(entry_by_source[source_id])
        ]
        actions = [
            entry_by_source[source_id]
            for source_id in message.source_ids
            if source_id in entry_by_source and str(entry_by_source[source_id].turn_role).endswith("action")
        ]
        if not observations and not (compact_arguments and actions):
            settled.append(message)
            continue

        payload = dict(message.payload)
        if compact_arguments and actions:
            payload, replacements = _settle_action_payload(
                payload,
                actions,
                provider_profile=provider_profile,
                min_utf8_bytes=argument_min_utf8_bytes,
                preview_chars=argument_preview_chars,
                saved_ratio=saved_ratio,
            )
            action_kinds.update(replacements)
        if not observations:
            pass
        elif provider_profile in {CANONICAL_PROFILE, OPENAI_PROFILE}:
            expected_role = "tool" if provider_profile == OPENAI_PROFILE else "user"
            if len(observations) != 1 or str(payload.get("role") or "") != expected_role:
                raise SchemaError("settlement_observation_projection_shape_unsupported")
            content = payload.get("content")
            if not isinstance(content, str):
                raise SchemaError("settlement_observation_content_must_be_text")
            entry = observations[0]
            kind, replacement = observation_decider(entry, content)
            source_id = str(getattr(entry, "source_id", "") or "")
            kinds[source_id] = kind if kind in _PROJECTION_KINDS else INLINE_FULL
            payload["content"] = replacement
        else:
            content = payload.get("content")
            if str(payload.get("role") or "") != "user" or not isinstance(content, list):
                raise SchemaError("settlement_observation_projection_shape_unsupported")
            by_call_id = {str(getattr(entry, "correlation_id", "") or ""): entry for entry in observations}
            seen: set[str] = set()
            blocks: list[Any] = []
            for raw_block in content:
                if not isinstance(raw_block, Mapping) or str(raw_block.get("type") or "") != "tool_result":
                    blocks.append(raw_block)
                    continue
                block = dict(raw_block)
                call_id = str(block.get("tool_use_id") or "")
                entry = by_call_id.get(call_id)
                if entry is None or call_id in seen or not isinstance(block.get("content"), str):
                    raise SchemaError("settlement_observation_projection_shape_unsupported")
                kind, replacement = observation_decider(entry, str(block["content"]))
                source_id = str(getattr(entry, "source_id", "") or "")
                kinds[source_id] = kind if kind in _PROJECTION_KINDS else INLINE_FULL
                block["content"] = replacement
                seen.add(call_id)
                blocks.append(block)
            if seen != set(by_call_id):
                raise SchemaError("settlement_observation_projection_shape_unsupported")
            payload["content"] = blocks

        settled.append(
            ProjectionMessageInput(
                provider_profile=message.provider_profile,
                payload=payload,
                source_ids=message.source_ids,
                projection_index=message.projection_index,
                projection_status=message.projection_status,
                projection_version=message.projection_version,
            )
        )
    return tuple(settled), kinds, action_kinds


def build_settlement_plan(
    adapter: Any,
    entries: Sequence[Any],
    *,
    provider_profile: str,
    start_index: int = 0,
    observation_decider: Callable[[Any, str], tuple[str, str]] = classify_observation,
    authoritative_messages: Sequence[Any] | None = None,
    count_text: Callable[[str], int] | None = None,
    token_count_quality: str = TOKEN_COUNT_QUALITY,
    settlement_min_utf8_bytes: int = COMPACT_RELOAD_MIN_INLINE_BYTES,
    settlement_min_saved_ratio: float = COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO,
    settlement_config_hash: str = "",
    timezone: str = "",
    compact_arguments: bool = False,
    settlement_argument_min_utf8_bytes: int = ARGUMENT_MIN_UTF8_BYTES,
    settlement_argument_preview_chars: int = ARGUMENT_PREVIEW_CHARS,
) -> SettledProjectionPlan:
    """确定性生成 settled projection。

    start_index 是 turn 内相对基准(0); 与 build_context_projection 的全局绝对 index
    对齐留到 Slice 3(request builder 消费 settled 账本时)。
    ``timezone`` 非空且使用默认 decider 时, 紧凑卡片附带通用可读时间。
    """
    if observation_decider is classify_observation:
        observation_decider = partial(
            classify_observation,
            timezone=str(timezone or "").strip(),
            include_call_id=provider_profile == CANONICAL_PROFILE,
        )
    action_kinds: dict[str, str] = {}
    if authoritative_messages is not None or compact_arguments:
        if authoritative_messages is None:
            authoritative_messages = adapter.project_entries(
                entries, provider_profile=provider_profile, start_index=start_index
            )
        full_messages = tuple(_as_projection_input(message) for message in authoritative_messages)
        settled_messages, kinds, action_kinds = _settle_authoritative_messages(
            entries,
            full_messages,
            provider_profile=provider_profile,
            observation_decider=observation_decider,
            compact_arguments=compact_arguments,
            argument_min_utf8_bytes=settlement_argument_min_utf8_bytes,
            argument_preview_chars=settlement_argument_preview_chars,
            saved_ratio=settlement_min_saved_ratio,
        )
    else:
        full_messages = tuple(
            adapter.project_entries(entries, provider_profile=provider_profile, start_index=start_index)
        )
        settled_messages = tuple(
            adapter.project_entries(
                entries,
                provider_profile=provider_profile,
                start_index=start_index,
                observation_decider=observation_decider,
            )
        )
        kinds = {
            str(getattr(entry, "source_id", "") or ""): _observed_kind(entry, observation_decider)
            for entry in entries
            if _is_observation(entry)
        }

    changed_index = -1
    for index, (full_message, settled_message) in enumerate(zip(full_messages, settled_messages)):
        if full_message.payload != settled_message.payload:
            changed_index = index
            break
    any_compact = COMPACT_RELOADABLE in (*kinds.values(), *action_kinds.values())
    if any_compact and changed_index >= 0:
        status = "settled"
    else:
        status = "settled_noop"

    full_tokens = _count_payload_tokens(full_messages, count_text)
    settled_tokens = _count_payload_tokens(settled_messages, count_text)
    return SettledProjectionPlan(
        turn_id=str(getattr(entries[0], "turn_id", "") or "") if entries else "",
        provider_profile=provider_profile,
        settlement_status=status,
        messages=settled_messages,
        full_messages=full_messages,
        first_changed_projection_index=changed_index,
        full_projected_tokens=full_tokens,
        settled_projected_tokens=settled_tokens,
        full_projection_hash=stable_projection_hash([m.payload for m in full_messages]),
        settled_projection_hash=stable_projection_hash([m.payload for m in settled_messages]),
        token_count_quality=str(token_count_quality or TOKEN_COUNT_QUALITY),
        observation_kinds=dict(kinds),
        action_kinds=dict(action_kinds),
        settlement_min_utf8_bytes=int(settlement_min_utf8_bytes),
        settlement_min_saved_ratio=float(settlement_min_saved_ratio),
        settlement_config_hash=str(settlement_config_hash or ""),
        settlement_argument_min_utf8_bytes=int(settlement_argument_min_utf8_bytes),
        settlement_argument_preview_chars=int(settlement_argument_preview_chars),
    )


def _observation_content(entry: Any) -> str:
    payload = dict(getattr(entry, "payload", None) or {})
    output = payload.get("output")
    if output is not None:
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(getattr(entry, "semantic_text", "") or "")


def _observed_kind(entry: Any, decider: Callable[[Any, str], tuple[str, str]]) -> str:
    kind, _ = decider(entry, _observation_content(entry))
    return kind if kind in _PROJECTION_KINDS else INLINE_FULL


def full_fallback_plan(turn_id: str, provider_profile: str, reason: str) -> SettledProjectionPlan:
    return SettledProjectionPlan(
        turn_id=turn_id,
        provider_profile=provider_profile,
        settlement_status="full_fallback",
        reason=reason,
    )


def stable_settlement_id(*, turn_id: str, provider_profile: str) -> str:
    return f"{turn_id}:{provider_profile}"


def settlement_config_hash(
    *,
    policy: str,
    min_utf8_bytes: int,
    saved_ratio: float,
    argument_min_utf8_bytes: int = ARGUMENT_MIN_UTF8_BYTES,
    argument_preview_chars: int = ARGUMENT_PREVIEW_CHARS,
) -> str:
    """确定性配置指纹: 冻结阈值与策略共同进入 settlement 身份, 重启后稳定。

    只用于一致性校验与审计, 不参与 provider payload hash。
    """

    canonical = f"{str(policy or 'full_until_raw_compaction').strip()}|{int(min_utf8_bytes)}|{float(saved_ratio):.6f}"
    if policy == OperationProjectionPolicy.COMPACT_ARGUMENTS_AFTER_TERMINAL.value:
        canonical += f"|arguments_v1|{int(argument_min_utf8_bytes)}|{int(argument_preview_chars)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_settled_projection(
    store: Any,
    *,
    namespace: Any,
    turn_id: str,
    provider_profile: str,
    settlement_policies: set[str] | None = None,
) -> "list[Any] | None":
    """读取已冻结的 settled 历史投影(仅 status==settled), 无则返回 None。

    request builder 与 compaction token 计数共用此入口, 保证 token accounting
    反映实际会投影给 provider 的 settled 历史(文档 §11)。settled_noop / full_fallback
    返回 None(保持 full 投影与 full token 计数)。
    """
    try:
        settlement = store.get_turn_projection_settlement(
            namespace=namespace,
            turn_id=turn_id,
            provider_profile=provider_profile,
        )
    except NotImplementedError:
        return None
    if not settlement or settlement.get("settlement_status") != "settled":
        return None
    rows = store.list_settled_projections(
        settlement_id=stable_settlement_id(turn_id=turn_id, provider_profile=provider_profile)
    )
    if not rows:
        raise SchemaError("settled_projection_rows_missing")
    messages = settled_rows_to_messages(settlement, rows)
    if settlement_policies is not None:
        settlement_policies.add(str(settlement.get("policy") or ""))
    return messages


def settled_rows_to_messages(
    settlement_row: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> "list[Any]":
    """把 settled 账本行还原为稳定的 ProjectionMessage(文档 §6 历史投影权威)。

    projection_index 沿用 turn 内相对 index(与 build_context_projection 冻结一致),
    因此 settled turn 与 full turn 在 request builder 中可无缝按序拼接。
    payload hash 校验失败(被篡改/损坏)时结构化报错, 不静默回退。
    """
    from .errors import SchemaError
    from .projection import ProjectionMessage, ProjectionStatus, stable_projection_hash

    namespace_key = (
        str(settlement_row.get("tenant_id") or ""),
        str(settlement_row.get("user_id") or ""),
        str(settlement_row.get("domain_id") or ""),
        str(settlement_row.get("conversation_id") or ""),
    )
    turn_id = str(settlement_row.get("turn_id") or "")
    ordered_rows = list(rows)
    indices = [int(row.get("projection_index") or 0) for row in ordered_rows]
    if indices != list(range(len(indices))):
        raise SchemaError("settled_projection_index_sequence_corrupt")
    messages: list[Any] = []
    payloads: list[dict[str, Any]] = []
    for row in ordered_rows:
        payload = json.loads(str(row.get("payload_json") or "{}"))
        payload_hash = str(row.get("payload_hash") or "")
        if payload_hash != stable_projection_hash(payload):
            raise SchemaError("settled_projection_payload_hash_mismatch")
        if str(row.get("provider_profile") or "") != str(settlement_row.get("provider_profile") or ""):
            raise SchemaError("settled_projection_profile_mismatch")
        payloads.append(payload)
        settlement_id = str(row.get("settlement_id") or "")
        projection_index = int(row.get("projection_index") or 0)
        messages.append(
            ProjectionMessage(
                projection_id=f"settled:{settlement_id}:{projection_index}",
                namespace_key=namespace_key,
                turn_id=turn_id,
                projection_index=projection_index,
                provider_profile=str(row.get("provider_profile") or ""),
                payload=payload,
                source_ids=tuple(json.loads(str(row.get("source_ids_json") or "[]"))),
                payload_hash=payload_hash,
                projection_status=ProjectionStatus.SETTLED,
                projection_version=SETTLED_PROJECTION_SCHEMA_VERSION,
                created_at=int(settlement_row.get("settled_at") or 0),
            )
        )
    if stable_projection_hash(payloads) != str(settlement_row.get("settled_projection_hash") or ""):
        raise SchemaError("settled_projection_set_hash_mismatch")
    return messages


__all__ = [
    "COMPACT_RELOAD_MIN_INLINE_BYTES",
    "COMPACT_RELOAD_REQUIRED_SAVINGS_RATIO",
    "SETTLED_PROJECTION_SCHEMA_VERSION",
    "TOKEN_COUNT_QUALITY",
    "INLINE_FULL",
    "COMPACT_RELOADABLE",
    "EXPLICIT_EMPTY",
    "utf8_bytes",
    "classify_observation",
    "render_compact_reload",
    "build_settlement_plan",
    "full_fallback_plan",
    "stable_settlement_id",
    "settlement_config_hash",
    "load_settled_projection",
    "settled_rows_to_messages",
    "SettledProjectionPlan",
]
