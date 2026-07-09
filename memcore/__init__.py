"""memcore —— 领域无关、可扩展、可授权的分层记忆内核。

公共入口。后续切片只在此追加导出,不改既有契约名(字段契约焊死)。
"""

from __future__ import annotations

from .config import MemoryConfig
from .chat_output import (
    ChatOutputConfig,
    ChatOutputMode,
    ChatOutputParseResult,
    ChatOutputStatus,
    StreamingSpeechParser,
    build_chat_output_contract_prompt,
    parse_chat_output,
    segment_speech,
)
from .embedding.base import EmbeddingProvider
from .embedding.hashed import HashedEmbeddingProvider
from .embedding.http import HTTPEmbeddingProvider
from .embedding.verify import verify_embedding
from .errors import ConfigError, MemcoreError, NamespaceError, PromptError, SchemaError
from .prompts import PromptOverrides
from .rendering import (
    render_material_cleanup_text,
    render_material_reference_text,
    render_tool_result_text,
    render_tool_use_text,
)
from .index.base import VectorIndex
from .index.memory_index import InMemoryVectorIndex
from .index.rrf import fuse_with_rrf
from .llm.base import LLMClient, LLMRequest, LLMResult, ResponseFormat, TaskType
from .memory_system import MemorySystem
from .native_tools import (
    NATIVE_MEMORY_TOOL_NAMES,
    build_native_memory_tool_specs,
    dispatch_native_memory_tool,
)
from .namespace import Actor, Namespace
from .schema import (
    DEFAULT_CATEGORIES,
    MOOD_TAGS,
    SUBJECT_SCOPES,
    MemoryMetadata,
    SemanticRecord,
    SummaryRecord,
    coerce_memory_metadata,
)
from .store.base import MemoryStore
from .store.sqlite_store import SQLiteMemoryStore
from .token_counter import TokenCounter

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 门面
    "MemorySystem",
    "NATIVE_MEMORY_TOOL_NAMES",
    "build_native_memory_tool_specs",
    "dispatch_native_memory_tool",
    # 配置 / 命名空间
    "MemoryConfig",
    "ChatOutputConfig",
    "ChatOutputMode",
    "ChatOutputParseResult",
    "ChatOutputStatus",
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
    "DEFAULT_CATEGORIES",
    "SUBJECT_SCOPES",
    "MOOD_TAGS",
    # 接口
    "LLMClient",
    "LLMRequest",
    "LLMResult",
    "TaskType",
    "ResponseFormat",
    "MemoryStore",
    "SQLiteMemoryStore",
    "VectorIndex",
    "InMemoryVectorIndex",
    "fuse_with_rrf",
    "EmbeddingProvider",
    "HashedEmbeddingProvider",
    "HTTPEmbeddingProvider",
    "verify_embedding",
    "TokenCounter",
    # 异常
    "MemcoreError",
    "ConfigError",
    "SchemaError",
    "NamespaceError",
    "PromptError",
    # 提示词治理
    "PromptOverrides",
    "render_material_reference_text",
    "render_material_cleanup_text",
    "render_tool_use_text",
    "render_tool_result_text",
]
