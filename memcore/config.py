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

from .errors import ConfigError


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
    enable_verifier: bool = True  # verifier 门:片段进 prompt 前先校验筛选
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
