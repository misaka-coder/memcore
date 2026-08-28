"""memcore 的结构化异常。

设计原则(见设计文档 §"失败要结构化"):非法配置/契约/命名空间在**接口层**就被拒绝,
带明确原因,而不是静默接受后让记忆系统在深处烂掉。
"""

from __future__ import annotations


class MemcoreError(Exception):
    """所有 memcore 异常的基类。"""


class ConfigError(MemcoreError):
    """MemoryConfig 取值非法(如批量 >= 触发数、窗口 <= 0)。"""


class SchemaError(MemcoreError):
    """输出字段契约被违反且无法安全回退(焊死的字段契约)。"""


class StaleSnapshotError(MemcoreError):
    """A valid concurrent append made a previously read snapshot obsolete."""


class NamespaceError(MemcoreError):
    """命名空间缺少必填硬隔离键,或键格式非法。"""


class PromptError(MemcoreError):
    """提示词插槽取值非法(类型错误、超长)——焊死骨架不可被外部覆盖。"""
