"""MemoryConfig —— 开放的可配置面(窗口/阈值/枚举/开关),带焊死的不变量校验。

设计原则:**给配置项 = 给用户把记忆系统写烂的权限**,所以开放项必须在构造时校验,
非法值直接报错,而不是让它在运行时悄悄把记忆系统拖垮。

焊死的承重不变量(不可绕过):
- raw 压缩只按 provider projection token 触发，并按完整 terminal turn/component 落切点。
- `episodic_compact_batch_size < episodic_compact_trigger_count`。
- 所有窗口/阈值为正;重叠阈值 >= 1;模型元数据枚举由 schema 固定而非宿主配置。
- TokenCounter 可由宿主显式注入；缺失时 raw 规划使用带 quality 标记的估算，不伪装成精确 tokenizer。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .errors import ConfigError


class OperationProjectionPolicy(str, Enum):
    """稳定 wire value: 持久化并进入 settled hash, 已发布值不得改名/删除/复用或改语义。

    已弃用值仍必须支持历史读取与确定性重放; schema 升级只允许新增值。
    """

    FULL_UNTIL_RAW_COMPACTION = "full_until_raw_compaction"
    COMPACT_AFTER_TERMINAL = "compact_after_terminal"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def stable_values(cls) -> list[str]:
        return [item.value for item in cls]


@dataclass
class MemoryConfig:
    # --- 写侧:统一时间线与三层窗口 ---
    raw_token_trigger: int = 12000  # 未摘要 raw provider projection 达到多少 tokens 后触发摘要
    raw_token_batch_ratio: float = 0.67  # 触发后计划压缩的最旧 raw token 比例
    episodic_visible_max: int = 8  # 可见阶段摘要数
    episodic_compact_trigger_count: int = 10  # 阶段摘要达到多少条触发语义压缩
    episodic_compact_batch_size: int = 5  # 每次压缩多少条阶段摘要(必须 < trigger)
    semantic_visible_limit: int = 5  # 可见长期记忆数

    # --- provider projection 与检索预算 ---
    # token 决定何时压缩和大致压缩多少，完整 terminal turn/component 决定实际边界。
    # 0 表示 MemCore 不裁剪检索结果；正值仅供宿主显式选择，不由 MemCore 自动推断。
    retrieval_result_token_budget: int = 0
    # provider-native read_timeline 的模型可见单页上限。可信宿主仍可直接调用
    # MemorySystem.read_timeline(page_token_budget=0) 进行显式无限诊断读取。
    native_timeline_page_token_budget: int = 12000
    projection_profile: str = "canonical_user_assistant"
    compaction_min_recent_turns: int = 1
    compaction_schema_version: int = 2
    summary_profile: str = "timeline_v2"

    # --- 强化合并 ---
    semantic_reinforcement_lookback: int = 8  # 回看最近多少条长期记忆找合并目标
    semantic_reinforcement_min_overlap: int = 2  # 触发合并的最小重叠分

    # --- 检索 ---
    retrieval_limit: int = 6  # 默认返回片段数
    relaxation_stop_candidate_count: int = 12  # 候选达到多少就停止放宽
    retrieval_min_dense_score: float = 0.0
    retrieval_min_bm25_score: float = 0.0
    retrieval_min_fused_score: float = 0.0
    llm_timeout_s: float = 30.0  # 单次结构化请求传输上限；宿主停机仍以协作取消为准
    llm_max_retries: int = 2  # 结构化 LLM 调用建议重试次数;失败仍不提交空记忆

    # --- 可见层作用域 ---
    # "conversation":可见 episodic/semantic 只看当前会话(金融/客服安全,默认)。
    # "user":跨会话带同一用户的长期记忆(陪伴/桌宠连续感)。
    # raw 层在两种模式下都只看当前会话。
    visible_memory_scope: str = "conversation"

    # --- 开关 ---
    enable_flavor: bool = False  # 温度层(mood/口吻);默认关 = 纯事实
    enable_importance_decay: bool = False  # 开启后:长期记忆可见窗口按"随时间衰减的重要度"排序,而非纯recency
    importance_half_life_days: float = 90.0  # 衰减半衰期(天):越久未强化,重要度按指数减半

    # --- 工具结果终局紧凑投影 ---
    # 枚举值见 OperationProjectionPolicy(稳定 wire value, 永不改名)。默认保持旧行为:
    # full_until_raw_compaction = 完整 action/observation 直到 raw compaction 由
    # operation_digest 接替, 不新增 settlement 元数据。compact_after_terminal 由后续
    # 切片接入, 本字段仅用于 begin_turn 冻结与 schema 落库。
    operation_projection_policy: str = "full_until_raw_compaction"
    # 卡片替换的字节/收益门槛(通用扩展, 默认与历史行为逐字节一致):
    # 正文 UTF-8 字节数低于 min 直接 inline_full; 卡片不能至少比正文小 ratio 时
    # 也保持 inline_full(no-expansion 硬约束)。两者在 begin_turn 冻结并进入
    # settlement 配置 hash, 重启后按冻结值稳定重放, 不受运行中配置变化影响。
    operation_settlement_min_utf8_bytes: int = 256
    operation_settlement_min_saved_ratio: float = 0.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        positives = {
            "raw_token_trigger": self.raw_token_trigger,
            "episodic_visible_max": self.episodic_visible_max,
            "episodic_compact_trigger_count": self.episodic_compact_trigger_count,
            "episodic_compact_batch_size": self.episodic_compact_batch_size,
            "semantic_visible_limit": self.semantic_visible_limit,
            "compaction_min_recent_turns": self.compaction_min_recent_turns,
            "compaction_schema_version": self.compaction_schema_version,
            "semantic_reinforcement_lookback": self.semantic_reinforcement_lookback,
            "retrieval_limit": self.retrieval_limit,
            "relaxation_stop_candidate_count": self.relaxation_stop_candidate_count,
            "llm_max_retries": self.llm_max_retries,
            "native_timeline_page_token_budget": self.native_timeline_page_token_budget,
        }
        for name, value in positives.items():
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{name} must be a positive int, got {value!r}")

        if self.semantic_reinforcement_min_overlap < 1:
            raise ConfigError(
                f"semantic_reinforcement_min_overlap must be >= 1, got {self.semantic_reinforcement_min_overlap!r}"
            )

        for name in (
            "retrieval_min_dense_score",
            "retrieval_min_bm25_score",
            "retrieval_min_fused_score",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or float(value) < 0.0 or not math.isfinite(float(value)):
                raise ConfigError(f"{name} must be a non-negative finite number, got {value!r}")

        if (
            not isinstance(self.llm_timeout_s, (int, float))
            or float(self.llm_timeout_s) <= 0.0
            or not math.isfinite(float(self.llm_timeout_s))
        ):
            raise ConfigError(f"llm_timeout_s must be a positive finite number, got {self.llm_timeout_s!r}")

        if not isinstance(self.raw_token_batch_ratio, (int, float)) or not (0 < self.raw_token_batch_ratio < 1):
            raise ConfigError(f"raw_token_batch_ratio must be > 0 and < 1, got {self.raw_token_batch_ratio!r}")
        if not isinstance(self.retrieval_result_token_budget, int) or self.retrieval_result_token_budget < 0:
            raise ConfigError(
                f"retrieval_result_token_budget must be a non-negative int, got {self.retrieval_result_token_budget!r}"
            )
        self.projection_profile = str(self.projection_profile or "").strip().lower()
        if not self.projection_profile:
            raise ConfigError("projection_profile must be non-empty")
        self.summary_profile = str(self.summary_profile or "").strip()
        if not self.summary_profile:
            raise ConfigError("summary_profile must be non-empty")
        if self.episodic_compact_batch_size >= self.episodic_compact_trigger_count:
            raise ConfigError(
                "episodic_compact_batch_size must be < episodic_compact_trigger_count "
                f"(got {self.episodic_compact_batch_size} >= {self.episodic_compact_trigger_count})"
            )

        if not isinstance(self.importance_half_life_days, (int, float)) or self.importance_half_life_days <= 0:
            raise ConfigError(
                f"importance_half_life_days must be a positive number, got {self.importance_half_life_days!r}"
            )

        if self.visible_memory_scope not in ("conversation", "user"):
            raise ConfigError(
                f"visible_memory_scope must be 'conversation' or 'user', got {self.visible_memory_scope!r}"
            )

        policy_value = str(self.operation_projection_policy or "").strip().lower()
        if policy_value not in OperationProjectionPolicy.stable_values():
            raise ConfigError(
                "operation_projection_policy must be a stable wire value in "
                f"{OperationProjectionPolicy.stable_values()}, got {self.operation_projection_policy!r}"
            )
        self.operation_projection_policy = policy_value

        if (
            not isinstance(self.operation_settlement_min_utf8_bytes, int)
            or self.operation_settlement_min_utf8_bytes <= 0
        ):
            raise ConfigError(
                "operation_settlement_min_utf8_bytes must be a positive int, "
                f"got {self.operation_settlement_min_utf8_bytes!r}"
            )
        ratio = self.operation_settlement_min_saved_ratio
        if not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)) or not (0.0 < float(ratio) < 1.0):
            raise ConfigError(f"operation_settlement_min_saved_ratio must be > 0 and < 1, got {ratio!r}")
        self.operation_settlement_min_saved_ratio = float(ratio)
