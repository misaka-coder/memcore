"""Export real Akane prompt-builder output without starting its runtime.

The export runs in an isolated, network-disabled child process. It reads only
Akane source/persona files and uses a new data root and an empty settings file.
The resulting bundle reuses the desktop-pet prompt profile with care disabled;
it is not an acceptance test of Akane's Engine, tools, UI, or memory integration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


FORMAT = "akane_research_prompt_bundle_v1"
TIMEZONE = "Asia/Shanghai"
BUNDLE_FILENAME = "akane_prompt_bundle.json"
_SOURCE_FILES = (
    "config.py",
    "akane_paths.py",
    "companion_v01/__init__.py",
    "companion_v01/client_protocol.py",
    "companion_v01/persona_config.py",
    "companion_v01/persona_profiles.toml",
    "companion_v01/prompt_blocks.py",
    "companion_v01/prompt_profiles.py",
    "companion_v01/prompt_builder.py",
    "desktop_pet_creator_kit/characters/akane_v1/persona.md",
)
_OS_ENV_KEYS = frozenset(
    {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC", "PATHEXT", "TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "USERPROFILE"}
)
_CARE_TOKENS = ("state_request", "care.state", "care_runtime", "affinity", "hunger", "energy", "affection_tier")
_LOCAL_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|file://|\\\\[a-z0-9])")

# Explicitly document the only output-template adaptation. Both experimental
# conditions receive the same block through the real builder's extension port.
OUTPUT_CONTRACT_ADAPTATION = """【纯文本研究回合的最终输出契约】
本次回合复用角色的说话方式，只使用本请求实际提供的原生工具。没有实时养成数值、视觉、桌面或投递状态输入，不自行编造这些当前状态或动作结果。
最终回复只输出一个 JSON 对象，且仅包含 speech 和 memory_metadata 两个字段，speech 在前。代码或其它用户需要阅读的内容也放在 speech 内。
本段关于最终字段和字段顺序的约定覆盖前面的桌宠界面模板；记忆标注的内容遵循 MemCore 记忆元数据契约。工具调用仍走原生工具通道，不放在最终 JSON 中。
""".strip()


class AkanePromptError(RuntimeError):
    """A fixed diagnostic code without credentials, prompts, or local paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _timestamp(value: Any) -> datetime:
    try:
        if type(value) is int and value > 0:
            result = datetime.fromtimestamp(value, ZoneInfo(TIMEZONE))
        elif isinstance(value, str) and value.strip():
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if result.tzinfo is None:
                result = result.replace(tzinfo=ZoneInfo(TIMEZONE))
            result = result.astimezone(ZoneInfo(TIMEZONE))
        else:
            raise ValueError
    except (ValueError, OverflowError, OSError):
        raise AkanePromptError("invalid_script_timestamp") from None
    return result


def _normalize_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(steps, (list, tuple)) or not steps or len(steps) > 1000:
        raise AkanePromptError("invalid_prompt_steps")
    normalized: dict[str, dict[str, Any]] = {}
    for step in steps:
        # Do not accept evaluator-only answers, fixture bodies, or arbitrary
        # extra host state at this boundary.
        if not isinstance(step, Mapping) or set(step) != {"step_id", "user_text", "timestamp"}:
            raise AkanePromptError("prompt_step_fields_must_be_current_input_only")
        step_id, text = step["step_id"], step["user_text"]
        if not isinstance(step_id, str) or not step_id or len(step_id) > 160 or "\x00" in step_id:
            raise AkanePromptError("invalid_prompt_step_id")
        if not isinstance(text, str) or not text.strip():
            raise AkanePromptError("invalid_prompt_user_text")
        row = {"step_id": step_id, "user_text": text, "timestamp": _timestamp(step["timestamp"]).isoformat()}
        if step_id in normalized and normalized[step_id] != row:
            raise AkanePromptError("conflicting_prompt_step_id")
        normalized[step_id] = row
    return list(normalized.values())


def _isolated_environment(output: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    env = {key: value for key, value in inherited.items() if key.upper() in _OS_ENV_KEYS}
    env.update(
        AKANE_DATA_ROOT=str(output / "isolated-data"),
        AKANE_ENV_FILE=str(output / "empty.env"),
        AKANE_INSTANCE_ID="research-prompt-export",
        MEMCORE_ENABLE_FLAVOR="false",
        PERSONA_VARIANT="default",
        TZ=TIMEZONE,
    )
    return env


def export_akane_prompt_bundle(
    akane_root: Path,
    output_dir: Path,
    steps: Sequence[Mapping[str, Any]],
    *,
    python_executable: Path | str | None = None,
) -> dict[str, Any]:
    """Call the real Akane builder for every scripted input, without a model call.

    ``steps`` accepts only step_id, user_text, and timestamp (Unix seconds or
    ISO 8601). Repeated identical IDs are deduplicated; conflicting IDs fail.
    ``python_executable`` may select an existing Akane dependency environment.
    No dependency installation, credential lookup, or Akane Engine occurs here.
    """
    normalized = _normalize_steps(steps)
    root, output = Path(akane_root).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise AkanePromptError("export_must_not_write_to_akane_checkout")
    for name in _SOURCE_FILES:
        candidate = root / name
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise AkanePromptError("akane_prompt_source_missing_or_external")
    output.mkdir(parents=True, exist_ok=False)
    (output / "empty.env").write_text("", encoding="utf-8")
    env = _isolated_environment(output, os.environ)
    try:
        result = subprocess.run(
            [
                str(python_executable or sys.executable),
                "-E",
                "-B",
                str(Path(__file__).resolve()),
                "--export-worker",
                str(root),
                str(output),
            ],
            input=_canonical({"steps": normalized}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            cwd=output,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise AkanePromptError("akane_prompt_export_process_failed") from None
    if result.returncode != 0:
        # A dependency's traceback may include local paths. Do not return it.
        raise AkanePromptError("akane_prompt_export_failed_see_validation")
    return load_akane_prompt_bundle(output / BUNDLE_FILENAME)


def load_akane_prompt_bundle(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_bundle(value)
    return value


def _validate_bundle(bundle: Any) -> None:
    if not isinstance(bundle, Mapping) or bundle.get("format") != FORMAT:
        raise AkanePromptError("invalid_akane_prompt_bundle")
    body = {key: value for key, value in bundle.items() if key != "bundle_hash"}
    if bundle.get("bundle_hash") != _hash(body):
        raise AkanePromptError("akane_prompt_bundle_hash_mismatch")
    if not isinstance(bundle.get("shared"), Mapping) or not isinstance(bundle.get("steps"), Mapping):
        raise AkanePromptError("invalid_akane_prompt_bundle")


def compose_akane_messages(
    bundle: Mapping[str, Any],
    *,
    history_messages: Sequence[Mapping[str, Any]],
    step_id: str,
    current_message_index: int,
    extra_system_blocks: Sequence[str] = (),
) -> dict[str, Any]:
    """Compose the complete request and the exact history-only freeze payload.

    MemCore's projection already contains the current user stimulus. Replace
    that one indexed message; never append another copy. All other history,
    including original native assistant/tool messages, remains unchanged.
    ``history_message_indexes`` maps history payloads to complete request slots.
    Only ``history_payloads`` belongs in record_request_projection. The stable
    host prefix and per-request ephemeral blocks must not be frozen as history.
    """
    _validate_bundle(bundle)
    if not isinstance(history_messages, (list, tuple)) or any(not isinstance(m, Mapping) for m in history_messages):
        raise AkanePromptError("invalid_provider_history")
    if (
        type(current_message_index) is not int
        or not 0 <= current_message_index < len(history_messages)
        or history_messages[current_message_index].get("role") != "user"
    ):
        raise AkanePromptError("current_stimulus_index_must_select_user")
    if step_id not in bundle["steps"]:
        raise AkanePromptError("unknown_prompt_step")
    if not isinstance(extra_system_blocks, (list, tuple)) or any(not isinstance(b, str) for b in extra_system_blocks):
        raise AkanePromptError("invalid_extra_system_blocks")
    shared, step = bundle["shared"], bundle["steps"][step_id]
    system = [
        {"role": "system", "content": block}
        for block in [shared["system_prompt"], *shared["system_extra_blocks"], *extra_system_blocks]
        if block.strip()
    ]
    host_prefix = copy.deepcopy(shared["history_prefix_messages"])
    ephemeral = copy.deepcopy(step["ephemeral_turns"])
    history = copy.deepcopy(list(history_messages))
    # Preserve any provider extension fields on the stimulus itself too.
    history[current_message_index]["content"] = step["user_prompt"]
    messages = copy.deepcopy(system) + copy.deepcopy(host_prefix)
    indexes: list[int] = []
    for index, message in enumerate(history):
        indexes.append(len(messages))
        messages.append(copy.deepcopy(message))
        if index == current_message_index:
            messages.extend(copy.deepcopy(ephemeral))
    return {
        "messages": messages,
        "history_payloads": history,
        "history_message_indexes": indexes,
        "system_prefix": system,
        "host_prefix_messages": host_prefix,
        "ephemeral_messages": ephemeral,
        "prompt_bundle_hash": bundle["bundle_hash"],
        "step_id": step_id,
    }


def _install_export_audit_guard(akane_root: Path, output: Path) -> dict[str, int]:
    """Deny network, writes outside the export, and private Akane file reads."""
    counts = {"network_attempts": 0, "blocked_reads": 0, "blocked_writes": 0}
    source_paths = {(akane_root / name).resolve() for name in _SOURCE_FILES}
    empty_env = (output / "empty.env").resolve()

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "urllib.Request"}:
            counts["network_attempts"] += 1
            raise AkanePromptError("network_forbidden_during_prompt_export")
        if event not in {"open", "os.mkdir", "os.remove", "os.rename", "os.rmdir"}:
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if event == "open":
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
        else:
            writing = True
        if writing and not path.is_relative_to(output):
            counts["blocked_writes"] += 1
            raise AkanePromptError("write_outside_prompt_export_forbidden")
        if writing or path.is_relative_to(output):
            return
        if path == empty_env:
            return
        if path.is_relative_to(akane_root):
            # Python may read an existing compiled version of an allowed source;
            # -B prevents it from creating or changing Akane bytecode caches.
            cached_source = None
            if path.suffix == ".pyc" and path.parent.name == "__pycache__":
                cached_source = path.parent.parent / (path.name.split(".")[0] + ".py")
            if path not in source_paths and cached_source not in source_paths:
                counts["blocked_reads"] += 1
                raise AkanePromptError("private_akane_file_read_forbidden")
        elif path.name.casefold() in {".env", "auth.json", "credentials.json", "model_service.json"} or any(
            part.casefold() in {"users_data", "runtime_logs"} for part in path.parts
        ):
            counts["blocked_reads"] += 1
            raise AkanePromptError("private_runtime_file_read_forbidden")

    sys.addaudithook(audit)
    return counts


def _worker(akane_root: Path, output: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _install_export_audit_guard(akane_root, output)
    source_hashes = {name: hashlib.sha256((akane_root / name).read_bytes()).hexdigest() for name in _SOURCE_FILES}
    memcore_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(memcore_root))
    sys.path.insert(0, str(akane_root))
    # All configuration-bearing imports occur after environment/file guards.
    import config
    from companion_v01.client_protocol import ClientMode
    from companion_v01.persona_config import load_persona_config
    from companion_v01.prompt_builder import PromptBuilder
    from companion_v01.prompt_profiles import PromptProfileRegistry

    for module_name in (
        "companion_v01.persona_config",
        "companion_v01.prompt_builder",
        "companion_v01.prompt_profiles",
    ):
        expected = akane_root / (module_name.replace(".", "/") + ".py")
        if Path(sys.modules[module_name].__file__).resolve() != expected:
            raise AkanePromptError("unexpected_akane_prompt_module_origin")
    if (
        Path(config.__file__).resolve() != akane_root / "config.py"
        or Path(config.DATA_ROOT).resolve() != output / "isolated-data"
        or Path(config._configured_env_file()).resolve() != output / "empty.env"
        or config.MEMCORE_ENABLE_FLAVOR
    ):
        raise AkanePromptError("akane_prompt_isolation_not_effective")
    persona_config = load_persona_config(path=akane_root / "companion_v01/persona_profiles.toml", variant="default")
    profile = PromptProfileRegistry().get(ClientMode.DESKTOP_PET, care_enabled=False)
    if {"state_request", "care_runtime"}.intersection(profile.system_block_ids):
        raise AkanePromptError("care_profile_was_not_disabled")
    persona_text = (akane_root / _SOURCE_FILES[-1]).read_text(encoding="utf-8").strip()
    if not persona_text:
        raise AkanePromptError("fixed_akane_persona_empty")
    builder = PromptBuilder(persona_config, stable_system_blocks_provider=lambda: (OUTPUT_CONTRACT_ADAPTATION,))
    shared: dict[str, Any] | None = None
    exported_steps: dict[str, Any] = {}
    actual_calls = 0
    for step in steps:
        stamp = _timestamp(step["timestamp"])
        weekday = "周" + "一二三四五六日"[stamp.weekday()]
        current_text = f"time: {stamp:%Y-%m-%d} {weekday} {stamp:%H:%M}\nUser: {step['user_text']}"
        scripted_now = f"当前时间：{stamp:%Y-%m-%d %H:%M}（{TIMEZONE}）"
        values = {
            "now_ts": int(stamp.timestamp()),
            "raw_text": "",
            "history_turns": [],
            "current_message_text": current_text,
            "episodic_summary_text": "",
            "semantic_summary_text": "",
            "memory_text": "",
            "current_visual_context": "",
            "resource_context": "",
            "extra_context": "",
            "volatile_extra_context": scripted_now,
            "visual_defaults": {
                "major": "home",
                "minor": "room",
                "background": "morning",
                "bgm": "",
                "outfit": "default",
                "emotion": "normal",
            },
            "allow_tool_call": True,
            "tool_prompt_context": "",
            "debug_enabled": False,
            "persona_system_context": persona_text,
            "persona_active_id": "akane_v1",
            "system_prompt_override": profile.system_prompt_override,
            "mode_prompt_override": profile.mode_prompt_override(debug_enabled=False),
            "current_message_in_raw": True,
        }
        built = builder.build_final_generation_context(**values)
        repeated = builder.build_final_generation_context(**values)
        actual_calls += 2
        if built != repeated:
            raise AkanePromptError("akane_builder_not_deterministic_for_fixed_input")
        candidate = {
            "system_prompt": built["system_prompt"],
            "system_extra_blocks": built["system_extra_blocks"],
            "history_prefix_messages": built["history_turns"],
        }
        if shared is not None and shared != candidate:
            raise AkanePromptError("akane_stable_prefix_changed_between_steps")
        shared = candidate
        visible = _canonical(candidate)
        if any(token in visible for token in _CARE_TOKENS):
            raise AkanePromptError("care_contract_leaked_into_prompt")
        if built["user_prompt"] != current_text or built["ephemeral_turns"] != [
            {"role": "user", "content": scripted_now}
        ]:
            raise AkanePromptError("unexpected_akane_current_input_layout")
        if _LOCAL_PATH.search(visible) or str(akane_root) in visible or str(output) in visible:
            raise AkanePromptError("local_path_leaked_into_shared_prompt")
        exported_steps[step["step_id"]] = {
            "timestamp": stamp.isoformat(),
            "input_user_text_hash": hashlib.sha256(step["user_text"].encode("utf-8")).hexdigest(),
            "user_prompt": built["user_prompt"],
            "ephemeral_turns": built["ephemeral_turns"],
        }
    if source_hashes != {name: hashlib.sha256((akane_root / name).read_bytes()).hexdigest() for name in _SOURCE_FILES}:
        raise AkanePromptError("akane_prompt_source_changed_during_export")
    evidence = {
        "actual_builder_called": True,
        "actual_builder_calls": actual_calls,
        "builder": "companion_v01.prompt_builder.PromptBuilder.build_final_generation_context",
        "imported_source_modules_verified": True,
        "profile": profile.to_public_dict(),
        "care_enabled": False,
        "removed_care_block_ids": ["care_runtime", "state_request"],
        "care_contract_absent": True,
        "fixed_persona": "akane_v1",
        "fixed_persona_source": _SOURCE_FILES[-1],
        "fixed_persona_sha256": hashlib.sha256(persona_text.encode("utf-8")).hexdigest(),
        "stable_system_and_persona_across_steps": True,
        "output_template_adaptation": OUTPUT_CONTRACT_ADAPTATION,
        "dynamic_visual_input": False,
        "dynamic_care_input": False,
        "dynamic_resource_input": False,
        "random_tools_or_timers_started": False,
        "isolated_data_root_verified": True,
        "empty_env_file_verified": True,
        "credential_environment_inherited": False,
        "audit_guard": counts,
        "akane_engine_verified": False,
        "akane_tools_verified": False,
        "paid_requests": 0,
        "python_version": sys.version.split()[0],
        "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_file_hashes": source_hashes,
    }
    result = {"format": FORMAT, "timezone": TIMEZONE, "shared": shared, "steps": exported_steps, "evidence": evidence}
    result["bundle_hash"] = _hash(result)
    return result


def _worker_main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] != "--export-worker":
        return 2
    root, output = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    try:
        payload = json.loads(sys.stdin.read())
        bundle = _worker(root, output, _normalize_steps(payload["steps"]))
        _write_json(output / BUNDLE_FILENAME, bundle)
        _write_json(
            output / "validation.json", {"status": "passed", "bundle_hash": bundle["bundle_hash"], **bundle["evidence"]}
        )
    except Exception as exc:
        code = exc.code if isinstance(exc, AkanePromptError) else type(exc).__name__
        _write_json(output / "validation.json", {"status": "failed", "error": code, "paid_requests": 0})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
