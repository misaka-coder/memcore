"""Isolated real Akane Engine host for the research pilot.

The only Engine overrides are dependency factories for an externally supplied
embedding and the five approved handlers. Turn control, native schema building,
prompt construction, output repair, memory recording and maintenance are Akane's.
Initialize in a fresh child process before importing any Akane modules.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo


_INITIALIZED: dict[str, Any] | None = None
_NETWORK = ContextVar("akane_research_authorized_network", default=False)
_SAFE_ENV = {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC", "PATHEXT", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "USERNAME"}
_SOURCE_EXTENSIONS = {".py", ".pyi", ".pyc", ".pyd", ".dll", ".toml", ".md", ".json", ".txt", ".yaml", ".yml"}
_POLICIES = {"full_until_raw_compaction", "compact_after_terminal"}


class AkaneHostError(RuntimeError):
    """Fixed diagnostics; never include credentials or deployment paths."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@contextmanager
def allow_model_network() -> Iterator[None]:
    """Used only by the budgeted transport after reservation and URL validation."""
    token = _NETWORK.set(True)
    try:
        yield
    finally:
        _NETWORK.reset(token)


def initialize_isolated_akane(
    akane_root: Path,
    run_root: Path,
    *,
    read_roots: Sequence[Path] = (),
    write_paths: Sequence[Path] = (),
    raw_token_trigger: int = 200000,
    embedding_reindex_batch_size: int = 64,
) -> dict[str, Any]:
    """Install filesystem/network guards and replace process settings before imports.

    ``write_paths`` is for the existing shared budget SQLite file, including its
    SQLite sidecars. It is not a directory grant. Secrets must be retained only
    in the caller's closure before this function clears inherited settings.
    """
    global _INITIALIZED
    if type(raw_token_trigger) is not int or not 1000 <= raw_token_trigger <= 200000:
        raise AkaneHostError("invalid_research_raw_token_trigger")
    if type(embedding_reindex_batch_size) is not int or not 1 <= embedding_reindex_batch_size <= 64:
        raise AkaneHostError("invalid_research_embedding_reindex_batch_size")
    if (
        _INITIALIZED is not None
        or "config" in sys.modules
        or any(name == "companion_v01" or name.startswith("companion_v01.") for name in sys.modules)
    ):
        raise AkaneHostError("akane_requires_fresh_isolated_process")
    source, output = Path(akane_root).resolve(), Path(run_root).resolve()
    if output.is_relative_to(source) or not (source / "companion_v01/engine.py").is_file():
        raise AkaneHostError("invalid_isolated_host_roots")
    output.mkdir(parents=True, exist_ok=True)
    (output / "empty.env").write_text("", encoding="utf-8")
    (output / "tmp").mkdir(exist_ok=True)
    allowed_writes = set()
    for value in write_paths:
        path = Path(value).resolve()
        if path.is_dir() or path.is_relative_to(source):
            raise AkaneHostError("write_exception_must_be_file")
        allowed_writes.update({path, *(Path(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal"))})
    counts = {
        "blocked_network": 0,
        "authorized_network_events": 0,
        "blocked_reads": 0,
        "blocked_writes": 0,
        "blocked_subprocesses": 0,
    }
    safe_roots = tuple(Path(value).resolve() for value in read_roots)
    runtime_roots = {Path(value).resolve() for value in sys.path if value and Path(value).is_dir()}
    runtime_roots.update({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()})
    interpreter_roots = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    # Editable dependencies are already registered by the selected interpreter.
    # Their declared source roots are readable, but private runtime names below
    # remain denied. No discovery of user settings or credential files occurs.
    for module in tuple(sys.modules.values()):
        if str(getattr(module, "__name__", "")).startswith("__editable__"):
            mapping = getattr(module, "MAPPING", {})
            if isinstance(mapping, dict):
                runtime_roots.update(Path(value).resolve() for value in mapping.values() if isinstance(value, str))
    # Python's Windows platform probe may run `ver`; cache its OS-only answer
    # before the host guard so imported dependencies do not launch subprocesses.
    platform.uname()
    platform.processor()
    platform.platform()

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "urllib.Request"}:
            if _NETWORK.get():
                counts["authorized_network_events"] += 1
                return
            counts["blocked_network"] += 1
            raise AkaneHostError("network_outside_budgeted_transport_forbidden")
        if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            counts["blocked_subprocesses"] += 1
            # Optional dependency hardware probes conventionally recover from
            # OSError on unsupported Windows commands. Deny the process while
            # preserving that normal operating-system error contract.
            raise PermissionError("host_subprocess_forbidden")
        if event not in {"open", "os.mkdir", "os.remove", "os.rename", "os.rmdir", "sqlite3.connect"}:
            return
        paths = args[:2] if event == "os.rename" else args[:1]
        for raw in paths:
            if not isinstance(raw, (str, bytes, os.PathLike)):
                continue
            if raw == ":memory:":
                continue
            if os.fsdecode(raw).casefold() == os.devnull.casefold():
                continue
            path = Path(os.fsdecode(raw)).resolve()
            if event == "open":
                mode = args[1] if len(args) > 1 else None
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                )
            else:
                writing = True
            if writing:
                if not path.is_relative_to(output) and path not in allowed_writes:
                    counts["blocked_writes"] += 1
                    raise AkaneHostError("write_outside_isolated_run_forbidden")
                continue
            if path.is_relative_to(output) or path in allowed_writes:
                continue
            parts = {part.casefold() for part in path.parts}
            private = path.name.casefold() in {".env", "auth.json", "credentials.json", "model_service.json"} or bool(
                parts & {"users_data", "runtime_logs", ".ssh", ".aws", ".azure"}
            )
            if path.is_relative_to(source) and not any(path.is_relative_to(root) for root in interpreter_roots):
                private = (
                    private
                    or bool(parts & {"logs", "backups", "backup", "work"})
                    or path.suffix not in _SOURCE_EXTENSIONS
                )
            if private:
                counts["blocked_reads"] += 1
                raise AkaneHostError("private_runtime_read_forbidden")
            # Declared model paths are read-only; no implicit model downloads.
            if any(path.is_relative_to(root) for root in safe_roots):
                return
            if path.is_relative_to(source) or any(path.is_relative_to(root) for root in runtime_roots):
                return
            counts["blocked_reads"] += 1
            raise AkaneHostError("read_outside_declared_runtime_forbidden")

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    inherited = {key: value for key, value in os.environ.items() if key.upper() in _SAFE_ENV}
    os.environ.clear()
    os.environ.update(inherited)
    os.environ.update(
        AKANE_DATA_ROOT=str(output / "bootstrap-data"),
        AKANE_ENV_FILE=str(output / "empty.env"),
        AKANE_INSTANCE_ID="research-host",
        PERSONA_VARIANT="default",
        MEMORY_BACKEND="memcore",
        MEMCORE_VISIBLE_SCOPE="user",
        MEMCORE_ENABLE_FLAVOR="false",
        MEMCORE_RAW_TOKEN_TRIGGER=str(raw_token_trigger),
        EMBEDDING_REINDEX_BATCH_SIZE=str(embedding_reindex_batch_size),
        MEMCORE_SUMMARY_API_KEY="research-placeholder-not-a-credential",
        MEMCORE_SUMMARY_BASE_URL="https://api.deepseek.com",
        MEMCORE_SUMMARY_MODEL_NAME="deepseek-v4-flash",
        MEMCORE_SUMMARY_API_PROTOCOL="openai",
        TEMP=str(output / "tmp"),
        TMP=str(output / "tmp"),
        TZ="Asia/Shanghai",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        ANONYMIZED_TELEMETRY="False",
        USERPROFILE=str(output / "home"),
        APPDATA=str(output / "appdata"),
        LOCALAPPDATA=str(output / "localappdata"),
    )
    os.chdir(output)
    sys.path.insert(0, str(source))
    _INITIALIZED = {"source": source, "output": output, "audit_guard": counts, "raw_token_trigger": raw_token_trigger}
    return {"isolation": "fresh_process_new_data_root", "audit_guard": counts}


def _unix_timestamp(value: Any) -> int:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return int(stamp.timestamp())
    raise AkaneHostError("invalid_turn_timestamp")


def _safe_metadata(value: Any) -> list[dict[str, Any]]:
    """Extract typed public raw records without persisting tool bodies twice."""
    rows: dict[str, dict[str, Any]] = {}
    fields = (
        "source_id",
        "timestamp",
        "kind",
        "turn_role",
        "turn_id",
        "correlation_id",
        "seq_no",
        "origin",
        "role",
        "entry_type",
        "date_label",
        "time_of_day",
        "renderer_id",
        "renderer_version",
    )

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if (
                item.get("node_type") in {"episodic", "semantic"}
                and item.get("view") == "content"
                and item.get("status") == "ok"
                and item.get("memory_id")
                and isinstance(item.get("result"), dict)
            ):
                memory_id = str(item["memory_id"])
                card = item["result"].get("card", {})
                encoded = json.dumps(item["result"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                rows[memory_id] = {
                    "source_id": memory_id,
                    "node_type": item["node_type"],
                    "kind": card.get("kind", ""),
                    "content_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                }
                return
            source_id = item.get("source_id")
            if source_id and ("timestamp" in item or "kind" in item):
                rows[str(source_id)] = {key: copy.deepcopy(item[key]) for key in fields if key in item}
                for field, target in (("semantic_text", "content_hash"), ("memory_metadata", "annotation_hash")):
                    if field in item:
                        encoded = json.dumps(item[field], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        rows[str(source_id)][target] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            for key, child in item.items():
                if key not in {"payload", "provider_output_raw", "semantic_text", "text", "content", "speech"}:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return list(rows.values())


class AkaneHostSession:
    def __init__(
        self,
        output_dir: Path,
        *,
        policy: str,
        embedding: Any,
        transport: Any,
        fixture_resolver: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
        stable_system_blocks_provider: Callable[[], tuple[str, ...]] | None = None,
        user_id: str = "research-user",
        session_id: str = "research-session",
        model: str = "deepseek-v4-flash",
        reopen_existing: bool = False,
        extra_handlers_factory: Callable[[Any], Sequence[Any]] | None = None,
    ) -> None:
        if _INITIALIZED is None:
            raise AkaneHostError("initialize_isolated_akane_required")
        if policy not in _POLICIES or model != "deepseek-v4-flash":
            raise AkaneHostError("invalid_host_policy_or_model")
        output = Path(output_dir).resolve()
        if (
            not output.is_relative_to(_INITIALIZED["output"])
            or output == _INITIALIZED["output"]
            or (output.exists() and not reopen_existing)
            or (reopen_existing and not output.is_dir())
        ):
            raise AkaneHostError("session_requires_new_isolated_directory")
        output.mkdir(parents=True, exist_ok=reopen_existing)
        import config
        from capcore import CapabilityToolSpec
        from companion_v01.capability_registry import CapabilityModule, CapabilityRegistry
        from companion_v01.client_protocol import ClientMode
        from companion_v01.desktop_pet_character_resources import DesktopPetCharacterResourceService
        from companion_v01.embedding_provider import BaseEmbeddingProvider
        from companion_v01.engine import AkaneMemoryEngine
        from companion_v01.instance_profile import (
            ChannelSnapshot,
            FeatureSnapshot,
            InstanceContext,
            InstanceManifest,
            QQChannelSelection,
        )
        from companion_v01.instance_runtime import bind_instance_runtime
        from companion_v01.runtime_settings import BotSettingsView
        from companion_v01.tool_handlers.core import BaseToolHandler, ToolExecutionResult, ToolFollowupEnvelope
        from companion_v01.tool_handlers.memory import (
            BrowseMemoryToolHandler,
            OpenMemoryToolHandler,
            ReadMemoryTimelineToolHandler,
            RetrieveMemoryToolHandler,
        )

        self.policy, self.user_id, self.session_id = policy, user_id, session_id
        self.transport = transport
        self._step: Mapping[str, Any] = {}
        self.tool_events: list[dict[str, Any]] = []
        host = self

        class InjectedEmbedding(BaseEmbeddingProvider):
            provider_name = str(embedding.name)
            version = str(getattr(embedding, "version", "research-provider-v1"))

            def __init__(self) -> None:
                super().__init__(dimension=embedding.dimension)

            def embed_text(self, text: str) -> list[float]:
                return embedding.embed_text(text)

            def embed_texts(self, texts: Any) -> list[list[float]]:
                return embedding.embed_texts(list(texts))

            def embed_query(self, text: str) -> list[float]:
                return getattr(embedding, "embed_query", embedding.embed_text)(text)

            def embed_documents(self, texts: Any) -> list[list[float]]:
                return getattr(embedding, "embed_documents", embedding.embed_texts)(list(texts))

        class FixtureHandler(BaseToolHandler):
            tool_type = "lookup_fixture"

            def tool_spec(self) -> Any:
                return CapabilityToolSpec(
                    capability_id="lookup_fixture",
                    display_name="lookup_fixture",
                    description="读取本次研究场景明确允许的静态资料。query_key 必须来自当前用户输入；历史事实优先使用记忆工具。",
                    input_schema={
                        "type": "object",
                        "properties": {"query_key": {"type": "string"}},
                        "required": ["query_key"],
                        "additionalProperties": False,
                    },
                    risk="low",
                    confirm="never",
                    effects=(),
                    visible_in=("desktop_pet",),
                    idempotency="read_only",
                )

            def normalize_call(self, value: Any) -> dict[str, Any] | None:
                if (
                    not isinstance(value, dict)
                    or value.get("type") != self.tool_type
                    or not isinstance(value.get("query_key"), str)
                ):
                    return None
                return {"type": self.tool_type, "query_key": value["query_key"]}

            def execute(self, *, call: dict[str, Any], context: Any) -> Any:
                arguments = {"query_key": call["query_key"]}
                result = fixture_resolver(host._step, arguments)
                body = (
                    result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                )
                host.tool_events.append(
                    {
                        "name": self.tool_type,
                        "arguments": arguments,
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

        class ResearchEngine(AkaneMemoryEngine):
            def _build_embedding_provider(self) -> Any:
                self._embedding_startup_status = {
                    "ok": True,
                    "status": "injected",
                    "provider": str(embedding.name),
                    "reason": "caller_verified_provider",
                }
                return InjectedEmbedding()

            def _build_tool_handlers(self) -> dict[str, Any]:
                service = self._build_memory_timeline_tool_service()
                handlers = [
                    RetrieveMemoryToolHandler(retrieve_fn=self._execute_retrieve_memory_tool),
                    ReadMemoryTimelineToolHandler(timeline_service=service),
                    BrowseMemoryToolHandler(timeline_service=service),
                    OpenMemoryToolHandler(timeline_service=service),
                    FixtureHandler(),
                ]
                if extra_handlers_factory is not None:
                    handlers.extend(extra_handlers_factory(host))
                names = [handler.tool_type for handler in handlers]
                if len(names) != len(set(names)):
                    raise AkaneHostError("duplicate_research_tool_handler")
                return {handler.tool_type: handler for handler in handlers}

        config.MEMCORE_OPERATION_PROJECTION_POLICY = policy
        instance = InstanceContext(
            InstanceManifest(
                1,
                "research-host",
                "akane_v1",
                FeatureSnapshot(care=False),
                ChannelSnapshot(QQChannelSelection(False, "")),
                (),
            ),
            source="research",
        )
        self.lease = bind_instance_runtime(instance, data_root=output, explicit_data_root=True)
        placeholder = "research-placeholder-not-a-credential"
        settings = BotSettingsView(
            text_api_key=placeholder,
            text_base_url="https://api.deepseek.com",
            text_model_name=model,
            text_api_protocol="openai",
            aux_api_key=placeholder,
            aux_base_url="https://api.deepseek.com",
            aux_model_name=model,
            aux_api_protocol="openai",
            chat_api_key=placeholder,
            chat_base_url="https://api.deepseek.com",
            chat_model_name=model,
            chat_api_protocol="openai",
            vision_enabled=False,
            image_generation_enabled=False,
            llm_thinking_mode="disabled",
            llm_chat_max_output_tokens=2048,
        )
        resources = DesktopPetCharacterResourceService(
            characters_dir=_INITIALIZED["source"] / "desktop_pet_creator_kit/characters",
            public_prefix="/desktop-pet-character-packs",
        )
        self.engine = ResearchEngine(
            self.lease.layout.engine_dir,
            instance_context=instance,
            runtime_layout=self.lease.layout,
            desktop_pet_character_resources=resources,
            stable_system_blocks_provider=stable_system_blocks_provider,
            settings=settings,
        )
        registry = self.engine.capability_registry
        module = CapabilityModule(
            name="research_fixture",
            layer="core",
            modes=(ClientMode.DESKTOP_PET,),
            tools=tuple(
                name
                for name in self.engine.tool_handlers
                if name not in {"retrieve_memory", "read_memory_timeline", "browse_memory", "open_memory"}
            ),
            light_hint="可按当前输入指定的 query_key 读取受控研究资料。",
            trigger=lambda _: True,
        )
        self.engine.capability_registry = CapabilityRegistry(
            modules=(*registry.modules, module),
            offer_source=registry.offer_source,
            server_offer_index=registry.server_offer_index,
        )
        self.attached_roles = transport.attach_runtime(self.engine.llm)
        self.method_evidence = {}
        for name in (
            "process_turn",
            "_run_turn_core",
            "_prepare_final_response_context",
            "_build_final_response",
            "_final_attempt_terminal_output",
            "_record_memcore_tool_batch",
            "_append_tool_history_batch",
            "_finalize_memcore_input_turn_for_delivery",
            "_schedule_memcore_compaction",
        ):
            original = getattr(AkaneMemoryEngine, name, None)
            if original is None or getattr(ResearchEngine, name) is not original:
                raise AkaneHostError("host_control_method_modified_or_missing")
            self.method_evidence[name] = {"module": original.__module__, "qualname": original.__qualname__}
        if not self.engine.memcore_manager.available:
            raise AkaneHostError("actual_memcore_manager_unavailable")

    def snapshot(self) -> dict[str, Any]:
        manager = self.engine.memcore_manager
        coordinates = {"profile_user_id": self.user_id, "session_id": self.session_id, "character_pack_id": "akane_v1"}
        projection = manager.build_context_projection(provider_profile="openai_chat", **coordinates)
        if not projection.get("ok"):
            raise AkaneHostError("public_context_projection_failed")
        metadata = []
        source_ids = projection.get("source_ids", [])
        for offset in range(0, len(source_ids), 20):
            opened = manager.open_memory(
                arguments={"memory_ids": source_ids[offset : offset + 20], "view": "content"}, **coordinates
            )
            if not opened.get("ok") or opened.get("result", {}).get("failed_count", 0):
                raise AkaneHostError("public_source_metadata_read_failed")
            metadata.extend(_safe_metadata(opened))
        if {row["source_id"] for row in metadata} != set(source_ids):
            raise AkaneHostError("public_source_metadata_incomplete")
        metadata.sort(key=lambda row: (row.get("seq_no", 0), row["source_id"]))
        for row in metadata:
            associated = [
                message["payload"] for message in projection["messages"] if row["source_id"] in message["source_ids"]
            ]
            contents = [payload.get("content") for payload in associated]
            encoded = json.dumps(contents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            row["projection_content_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            row["projection_payload_hashes"] = [
                hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                for payload in associated
            ]
            row["annotation_hash_available"] = "annotation_hash" in row
        state = {key: value for key, value in manager.status().items() if key != "storage_path"}
        state["active_persona_card"] = self.engine.store.get_active_persona_card(
            profile_user_id=self.user_id, session_id=self.session_id
        )
        state["session"] = self.engine.store.get_session(self.user_id, self.session_id)
        return {
            "projection": copy.deepcopy(projection["messages"]),
            "memory": metadata,
            "host_state": state,
            "metrics": {
                key: projection.get(key)
                for key in (
                    "message_count",
                    "source_count",
                    "stable_prefix_hash",
                    "projection_version",
                    "compaction_generation",
                    "projection_generation",
                    "has_compact_history",
                )
            },
        }

    def process_turn(self, step: Mapping[str, Any]) -> dict[str, Any]:
        self._step = {key: step[key] for key in ("step_id", "user_text", "timestamp")}
        before = self.snapshot()
        first_event = len(self.tool_events)
        payload = {
            "user_id": self.session_id,
            "real_user_id": self.user_id,
            "session_id": self.session_id,
            "character_pack_id": "akane_v1",
            "client_mode": "desktop_pet",
            "message": step["user_text"],
            "timestamp": _unix_timestamp(step["timestamp"]),
            "trace_id": step["step_id"],
            "memory_idempotency_key": step["step_id"],
        }
        output = self.engine.process_turn(payload)
        return {
            **self.snapshot(),
            "before": before,
            "output": output,
            "tool_events": copy.deepcopy(self.tool_events[first_event:]),
            "step_id": step["step_id"],
            "input_timestamp": payload["timestamp"],
        }

    def evidence(self) -> dict[str, Any]:
        assert _INITIALIZED is not None
        import config

        source = _INITIALIZED["source"]
        module_hashes = {}
        interpreter_roots = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if (
                    path.is_relative_to(source)
                    and path.suffix == ".py"
                    and not any(path.is_relative_to(root) for root in interpreter_roots)
                ):
                    module_hashes[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "engine_methods": self.method_evidence,
            "loaded_akane_source_hashes": module_hashes,
            "audit_guard": dict(_INITIALIZED["audit_guard"]),
            "attached_roles": list(self.attached_roles),
            "operation_policy": self.policy,
            "care_enabled": False,
            "raw_token_trigger": _INITIALIZED["raw_token_trigger"],
            "embedding_reindex_batch_size": config.EMBEDDING_REINDEX_BATCH_SIZE,
            "tool_round_hard_limit": config.TOOL_ROUND_HARD_LIMIT,
            "decision_max_attempts": config.CHAT_MODEL_DECISION_MAX_ATTEMPTS,
            "template_source_hashes": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in (
                    "desktop_pet_creator_kit/characters/akane_v1/persona.md",
                    "desktop_pet_creator_kit/characters/akane_v1/character.json",
                    "desktop_pet_creator_kit/characters/akane_v1/character.toml",
                )
                if (source / name).is_file()
            },
            "namespace": {"user_id": self.user_id, "conversation_id": self.session_id, "domain_id": "akane_v1"},
        }

    def close(self) -> Any:
        try:
            return self.engine.close()
        finally:
            self.lease.release()
