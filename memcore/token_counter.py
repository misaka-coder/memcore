"""TokenCounter 接口。

token 计数与具体模型 tokenizer 绑定。memcore 只定义接口,不静默猜测字符/token 比例。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TokenCounter(ABC):
    @abstractmethod
    def count_text(self, text: str) -> int:
        """返回 text 在接入方模型 tokenizer 下的 token 数。"""
        raise NotImplementedError
