"""通用文本工具:归一化 + 分词(供 embedding 哈希与关键词 BM25 共用)。

⚠️ 分词与旧项目 `text_utils.tokenize` 不必逐字节一致,但接评测台回归(切片 5)时需对齐口径。
"""

from __future__ import annotations

import re
import unicodedata

_LATIN_RUN = re.compile(r"[a-z0-9]+")
_CJK_CHAR = re.compile(r"[一-鿿]")


def normalize_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def tokenize(text: object) -> list[str]:
    """拉丁词整体成 token;中文按字(unigram)+ 相邻字(bigram)切,适配中文 BM25。"""
    normalized = normalize_text(text).lower()
    if not normalized:
        return []
    tokens: list[str] = _LATIN_RUN.findall(normalized)
    # 连续 CJK 段:逐字 + 相邻二元
    for run in re.findall(r"[一-鿿]+", normalized):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def join_tags(tags: object) -> str:
    if not isinstance(tags, (list, tuple)):
        return ""
    return " ".join(normalize_text(item) for item in tags if normalize_text(item))
