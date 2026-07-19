"""MemoryConfig —— 开放的可配置面(窗口/阈值/枚举/开关),带焊死的不变量校验。

设计原则:**给配置项 = 给用户把记忆系统写烂的权限**,所以开放项必须在构造时校验,
非法值直接报错,而不是让它在运行时悄悄把记忆系统拖垮。

焊死的承重不变量(不可绕过):
- 差值关系:`summary_batch_size < raw_trigger_count`(防层间记忆重叠 / no blind window)。
- 同理:`episodic_compact_batch_size < episodic_compact_trigger_count`。
- 所有窗口/阈值为正;重叠阈值 >= 1;categories 为非空固定枚举。
- raw token policy 只改变 raw->episodic 批次选择;token 参数必须合法,TokenCounter 由门面注入校验。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigError
from .schema import DEFAULT_CATEGORIES


@dataclass
class MemoryConfig:
    # --- 写侧:三层窗口与差值 ---
    raw_trigger_count: int = 30  # raw 达到多少条触发摘要
    summary_batch_size: int = 20  # 每次总结最老多少条(必须 < raw_trigger_count)
    raw_compaction_policy: str = "count"  # "count" | "token"
    raw_token_trigger: int = 12000  # token policy:未摘要 raw content token 总量达到多少触发摘要
    raw_token_batch_ratio: float = 0.67  # token policy:触发后压缩 trigger 的多少比例,默认约等于 20/30
    raw_token_min_remainder_messages: int = 1  # token policy:压缩后至少保留多少条 raw 近期上下文
    raw_token_boundary_role: str = "assistant"  # token policy:批次边界对齐到 assistant 回复
    raw_compaction_excluded_categories: tuple[str, ...] = (
        "material_trace",
    )  # count policy:材料锚点不计数;工具轨迹参与正常 raw 生命周期
    episodic_visible_max: int = 8  # 可见阶段摘要数
    episodic_compact_trigger_count: int = 10  # 阶段摘要达到多少条触发语义压缩
    episodic_compact_batch_size: int = 5  # 每次压缩多少条阶段摘要(必须 < trigger)
    semantic_visible_limit: int = 5  # 可见长期记忆数

    # --- 强化合并 ---
    semantic_reinforcement_lookback: int = 8  # 回看最近多少条长期记忆找合并目标
    semantic_reinforcement_min_overlap: int = 2  # 触发合并的最小重叠分

    # --- 检索 ---
    retrieval_limit: int = 6  # 默认返回片段数
    relaxation_stop_candidate_count: int = 12  # 候选达到多少就停止放宽
    retrieval_default_excluded_categories: tuple[str, ...] = (
        "event_trace",
        "tool_trace",
        "material_trace",
    )  # 普通检索默认不捞外部事件/工具/材料轨迹
    enable_verifier: bool = True  # verifier 门:片段进 prompt 前先校验筛选
    llm_max_retries: int = 2  # 结构化 LLM 调用建议重试次数;失败仍不提交空记忆

    # --- 领域词表(固定枚举,可换内容,不可空) ---
    categories: tuple[str, ...] = DEFAULT_CATEGORIES

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
            "raw_trigger_count": self.raw_trigger_count,
            "summary_batch_size": self.summary_batch_size,
            "raw_token_trigger": self.raw_token_trigger,
            "raw_token_min_remainder_messages": self.raw_token_min_remainder_messages,
            "episodic_visible_max": self.episodic_visible_max,
            "episodic_compact_trigger_count": self.episodic_compact_trigger_count,
            "episodic_compact_batch_size": self.episodic_compact_batch_size,
            "semantic_visible_limit": self.semantic_visible_limit,
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

        if self.raw_compaction_policy not in ("count", "token"):
            raise ConfigError(f"raw_compaction_policy must be 'count' or 'token', got {self.raw_compaction_policy!r}")
        if not isinstance(self.raw_token_batch_ratio, (int, float)) or not (0 < self.raw_token_batch_ratio < 1):
            raise ConfigError(f"raw_token_batch_ratio must be > 0 and < 1, got {self.raw_token_batch_ratio!r}")
        if self.raw_token_boundary_role != "assistant":
            raise ConfigError(
                f"raw_token_boundary_role currently supports only 'assistant', got {self.raw_token_boundary_role!r}"
            )
        self.raw_compaction_excluded_categories = self._clean_tuple(
            self.raw_compaction_excluded_categories, "raw_compaction_excluded_categories"
        )
        self.retrieval_default_excluded_categories = self._clean_tuple(
            self.retrieval_default_excluded_categories, "retrieval_default_excluded_categories"
        )

        # 焊死的差值关系:批量必须严格小于触发数,否则层间记忆会重叠 / 出现空窗。
        if self.summary_batch_size >= self.raw_trigger_count:
            raise ConfigError(
                "summary_batch_size must be < raw_trigger_count "
                f"(got {self.summary_batch_size} >= {self.raw_trigger_count}) —— 差值关系是防层间重叠的承重约束"
            )
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

        # categories 必须是非空、去重的固定枚举。
        cats = tuple(str(c).strip() for c in self.categories if str(c).strip())
        if not cats:
            raise ConfigError("categories must be a non-empty enum")
        if len(set(cats)) != len(cats):
            raise ConfigError(f"categories must not contain duplicates: {self.categories!r}")
        self.categories = cats

    @staticmethod
    def _clean_tuple(value: tuple[str, ...], name: str) -> tuple[str, ...]:
        if not isinstance(value, tuple):
            raise ConfigError(f"{name} must be a tuple[str, ...], got {type(value).__name__}")
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return tuple(out)
