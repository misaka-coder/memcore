"""memcore —— 领域无关、可扩展、可授权的分层记忆内核。

公共入口。后续切片只在此追加导出,不改既有契约名(字段契约焊死)。
"""

from __future__ import annotations

from .config import MemoryConfig, OperationProjectionPolicy
from .conformance import (
    ConformanceCheck,
    ConformanceReport,
    validate_context_adapter,
    validate_provider_wire_capture,
)
from .context_contract import (
    CONTEXT_SURFACE_MESSAGE_METADATA_VERSION,
    CONTEXT_SURFACE_VERSION,
    ContextDiagnostic,
    ContextSurface,
    RequestBindingGroup,
    RequestBindingMessage,
    RequestBindingResult,
    bind_request_projection_messages,
)
from .context_session import ContextSessionStatus, MemCoreContextSession
from .compaction_v2 import (
    CompactionResult,
    CompactionSnapshot,
    SemanticBatchCommitResult,
    SemanticCommitInput,
    SemanticSnapshot,
    SummaryBatchCommitResult,
    SummaryRecordInput,
    TurnBundle,
)
from .chat_output import (
    ChatOutputConfig,
    ChatOutputMode,
    ChatOutputParseResult,
    ChatOutputStatus,
    MemoryMetadataStatus,
    StreamingSpeechParser,
    build_chat_output_contract_prompt,
    parse_chat_output,
    segment_speech,
)
from .embedding.base import EmbeddingProvider
from .embedding.hashed import HashedEmbeddingProvider
from .embedding.huggingface import HuggingFaceEmbeddingProvider
from .embedding.http import HTTPEmbeddingProvider, RoleAwareHTTPEmbeddingProvider
from .embedding.verify import verify_embedding
from .errors import ConfigError, MemcoreError, NamespaceError, PromptError, SchemaError
from .prompts import PromptOverrides
from .projection import (
    ANTHROPIC_PROFILE,
    CANONICAL_PROFILE,
    DEEPSEEK_PROFILE,
    OPENAI_PROFILE,
    OPENAI_RESPONSES_PROFILE,
    PROJECTION_VERSION,
    ContextProjection,
    EntryProjectionHash,
    ProjectionAdapter,
    ProjectionAudit,
    ProjectionAuditInput,
    ProjectionLedger,
    ProjectionMessage,
    ProjectionMessageInput,
    ProjectionStatus,
    RendererRegistry,
    RequestProjectionResult,
    build_projection_audit_input,
    canonical_json_bytes,
    default_renderer_registry,
    is_strict_message_prefix,
    stable_projection_hash,
)
from .provider_adapters import (
    AnthropicMessagesContextAdapter,
    ContextProviderAdapter,
    DeepSeekChatContextAdapter,
    NormalizedContextEvent,
    OpenAIChatContextAdapter,
    OpenAIResponsesContextAdapter,
    normalize_context_event,
    official_context_adapters,
)
from .rendering import (
    render_external_event_text,
    render_material_cleanup_text,
    render_material_reference_text,
    render_prompt_message,
    render_tool_result_text,
    render_tool_use_text,
)
from .index.base import VectorIndex
from .index.memory_index import InMemoryVectorIndex
from .index.rrf import fuse_with_rrf
from .llm.base import LLMClient, LLMRequest, LLMResult, ResponseFormat, TaskType
from .memory_system import MemorySystem
from .memory_receipt import build_memory_operation_receipt
from .native_tools import (
    NATIVE_MEMORY_TOOL_NAMES,
    ToolDispatchPolicy,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)
from .namespace import Actor, Namespace
from .schema import (
    ABOUT_ROLES,
    MEMORY_FACETS,
    MOOD_TAGS,
    RETRIEVAL_PRIORITIES,
    TURN_INTENTS,
    MemoryMetadata,
    SemanticRecord,
    SummaryRecord,
    build_memory_metadata_instruction,
    coerce_memory_metadata,
    memory_metadata_has_signal,
)
from .runtime import ConversationLockRegistry, MemCoreRuntime
from .retrieval import (
    HardFilterPlan,
    RetrievalMatch,
    RetrievalQueryPlan,
    RetrievalRequest,
    RetrievalResult,
    RelationExpansionPlan,
    SemanticFilterPlan,
)
from .store.base import LineageClosure, MemoryStore, RawTurnWindow
from .store.migrations import migrate_legacy_memory_metadata
from .store.sqlite_store import SQLiteMemoryStore
from .token_counter import TokenCounter
from .timeline import (
    AnnotationStatus,
    CompletionCommitResult,
    EntryOrigin,
    EntryTrust,
    MAX_OPERATION_RETENTION_ANCHOR_BYTES,
    MemoryAnnotation,
    OPERATION_RETENTION_ANCHOR_KEY,
    OPERATION_RETENTION_ANCHOR_STATUS_KEY,
    OPERATION_RETENTION_ANCHOR_BYTES_KEY,
    RetrievalPolicy,
    RetrievalVisibility,
    TimelineEntry,
    TimelineEntryInput,
    TurnAbortResult,
    TurnCompletion,
    TurnHandle,
    TurnRole,
    TurnStatus,
    build_action_entry,
    build_observation_entry,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 门面
    "MemorySystem",
    "build_memory_operation_receipt",
    "NATIVE_MEMORY_TOOL_NAMES",
    "ToolDispatchPolicy",
    "build_native_memory_tool_specs",
    "dispatch_native_memory_tool",
    # 配置 / 命名空间
    "MemoryConfig",
    "OperationProjectionPolicy",
    "ConformanceCheck",
    "ConformanceReport",
    "validate_context_adapter",
    "validate_provider_wire_capture",
    "CONTEXT_SURFACE_VERSION",
    "CONTEXT_SURFACE_MESSAGE_METADATA_VERSION",
    "ContextDiagnostic",
    "ContextSurface",
    "RequestBindingGroup",
    "RequestBindingMessage",
    "RequestBindingResult",
    "bind_request_projection_messages",
    "ContextSessionStatus",
    "MemCoreContextSession",
    "ChatOutputConfig",
    "ChatOutputMode",
    "ChatOutputParseResult",
    "ChatOutputStatus",
    "MemoryMetadataStatus",
    "StreamingSpeechParser",
    "build_chat_output_contract_prompt",
    "parse_chat_output",
    "segment_speech",
    "Namespace",
    "Actor",
    # 契约
    "MemoryMetadata",
    "SummaryRecord",
    "SemanticRecord",
    "coerce_memory_metadata",
    "memory_metadata_has_signal",
    "build_memory_metadata_instruction",
    "MEMORY_FACETS",
    "ABOUT_ROLES",
    "RETRIEVAL_PRIORITIES",
    "TURN_INTENTS",
    "MOOD_TAGS",
    # 接口
    "LLMClient",
    "LLMRequest",
    "LLMResult",
    "TaskType",
    "ResponseFormat",
    "MemoryStore",
    "RawTurnWindow",
    "migrate_legacy_memory_metadata",
    "SQLiteMemoryStore",
    "VectorIndex",
    "InMemoryVectorIndex",
    "fuse_with_rrf",
    "EmbeddingProvider",
    "HashedEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "HTTPEmbeddingProvider",
    "RoleAwareHTTPEmbeddingProvider",
    "verify_embedding",
    "TokenCounter",
    # Unified Timeline V2
    "AnnotationStatus",
    "CompletionCommitResult",
    "EntryOrigin",
    "EntryTrust",
    "MAX_OPERATION_RETENTION_ANCHOR_BYTES",
    "MemoryAnnotation",
    "OPERATION_RETENTION_ANCHOR_KEY",
    "OPERATION_RETENTION_ANCHOR_STATUS_KEY",
    "OPERATION_RETENTION_ANCHOR_BYTES_KEY",
    "RetrievalPolicy",
    "RetrievalVisibility",
    "TimelineEntry",
    "TimelineEntryInput",
    "TurnAbortResult",
    "TurnCompletion",
    "TurnHandle",
    "TurnRole",
    "TurnStatus",
    "build_action_entry",
    "build_observation_entry",
    # Retrieval V2
    "HardFilterPlan",
    "RetrievalMatch",
    "RetrievalQueryPlan",
    "RetrievalRequest",
    "RetrievalResult",
    "RelationExpansionPlan",
    "SemanticFilterPlan",
    "LineageClosure",
    # Compaction V2 / shared runtime
    "CompactionResult",
    "CompactionSnapshot",
    "ConversationLockRegistry",
    "MemCoreRuntime",
    "SemanticBatchCommitResult",
    "SemanticCommitInput",
    "SemanticSnapshot",
    "SummaryBatchCommitResult",
    "SummaryRecordInput",
    "TurnBundle",
    # Projection Ledger
    "ANTHROPIC_PROFILE",
    "CANONICAL_PROFILE",
    "DEEPSEEK_PROFILE",
    "OPENAI_PROFILE",
    "OPENAI_RESPONSES_PROFILE",
    "PROJECTION_VERSION",
    "ContextProjection",
    "EntryProjectionHash",
    "ProjectionAdapter",
    "ProjectionAudit",
    "ProjectionAuditInput",
    "ProjectionLedger",
    "ProjectionMessage",
    "ProjectionMessageInput",
    "ProjectionStatus",
    "RendererRegistry",
    "RequestProjectionResult",
    "build_projection_audit_input",
    "canonical_json_bytes",
    "default_renderer_registry",
    "is_strict_message_prefix",
    "stable_projection_hash",
    # Provider adapters
    "AnthropicMessagesContextAdapter",
    "ContextProviderAdapter",
    "DeepSeekChatContextAdapter",
    "NormalizedContextEvent",
    "normalize_context_event",
    "OpenAIChatContextAdapter",
    "OpenAIResponsesContextAdapter",
    "official_context_adapters",
    # 异常
    "MemcoreError",
    "ConfigError",
    "SchemaError",
    "NamespaceError",
    "PromptError",
    # 提示词治理
    "PromptOverrides",
    "render_external_event_text",
    "render_prompt_message",
    "render_material_reference_text",
    "render_material_cleanup_text",
    "render_tool_use_text",
    "render_tool_result_text",
]
