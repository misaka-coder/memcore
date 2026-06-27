"""重要度时间衰减 —— 让"很久没被强化的长期记忆"在默认视野里自然淡出(见设计文档 §13)。

指数半衰期:每过 half_life_days,有效重要度减半。强化合并会更新 last_reinforced_ts,
所以真正反复出现的记忆 age 重置、不衰减;一次性的旧记忆则随时间退场。
"""

from __future__ import annotations

_SECONDS_PER_DAY = 86400.0


def decayed_importance(base_importance: float, *, age_seconds: float, half_life_days: float) -> float:
    base = max(0.0, float(base_importance or 0.0))
    if base == 0.0 or age_seconds <= 0 or half_life_days <= 0:
        return base
    half_lives = (float(age_seconds) / _SECONDS_PER_DAY) / float(half_life_days)
    return base * (0.5**half_lives)
