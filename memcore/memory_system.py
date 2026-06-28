"""MemorySystem —— 对外唯一门面。

写侧(切片 4)已接通:record_user_turn / record_assistant_turn / compact_due_background / compact_due_sync / embedding_status。
读侧(切片 5)已接通:build_prompt_context / retrieve。遗忘会同步清 store 与 VectorIndex。

生命周期(见设计文档 §3.3),使用方按顺序接:
    record_user_turn → build_prompt_context → record_assistant_turn → compact_due_background
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from zoneinfo import ZoneInfo

from .compaction import Compaction
from .config import MemoryConfig
from .embedding.base import EmbeddingProvider
from .errors import ConfigError
from .index.base import VectorIndex
from .index.entry_builder import build_raw_entry
from .index.memory_index import InMemoryVectorIndex
from .llm.base import LLMClient
from .namespace import Actor, Namespace
from .prompts import PromptOverrides
from .retrieval import ReadPipeline
from .schema import coerce_memory_metadata
from .store.base import MemoryStore
from .store.sqlite_store import SQLiteMemoryStore
from .time_anchor import infer_time_of_day, timestamp_to_date_label
from .token_counter import TokenCounter


class MemorySystem:
    def __init__(
        self,
        *,
        llm: LLMClient,
        namespace: Namespace,
        timezone: str,
        storage_dir: str | None = None,
        config: MemoryConfig | None = None,
        store: MemoryStore | None = None,
        index: VectorIndex | None = None,
        embedding: EmbeddingProvider | str | None = None,
        token_counter: TokenCounter | None = None,
        enable_flavor: bool | None = None,
        persona_text: str = "",
        prompt_overrides: PromptOverrides | None = None,
    ) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError("llm must be an LLMClient instance (inject your model adapter)")
        if not isinstance(namespace, Namespace):
            raise TypeError("namespace must be a Namespace instance")
        if not str(timezone or "").strip():
            # 时间锚点是命根,时区缺失会让相对时间换算全错(见 §5)。
            raise ValueError("timezone is required (e.g. 'Asia/Shanghai'); time anchoring depends on it")
        tz_name = str(timezone).strip()
        try:  # 构造时就验合法 IANA 时区,别拖到渲染才炸
            ZoneInfo(tz_name)
        except Exception as exc:
            raise ValueError(f"invalid timezone {tz_name!r}; must be a valid IANA name like 'Asia/Shanghai'") from exc

        self.llm = llm
        self.namespace = namespace
        self.timezone = tz_name
        self.storage_dir = storage_dir
        self.config = config or MemoryConfig()
        if enable_flavor is not None:  # 显式传入覆盖 config 默认
            self.config.enable_flavor = bool(enable_flavor)
        if token_counter is not None and not isinstance(token_counter, TokenCounter):
            raise TypeError("token_counter must be a TokenCounter instance or None")
        if self.config.raw_compaction_policy == "token" and token_counter is None:
            raise ConfigError("token_counter is required when raw_compaction_policy='token'")
        self.persona_text = str(persona_text or "")
        # 提示词治理:persona_text 便捷参数填进 overrides 的对应插槽(显式 overrides 优先)。
        self.prompt_overrides = self._resolve_overrides(prompt_overrides, self.persona_text)

        # 依赖装配:缺省自带 SQLite + 内存索引;embedding 必须显式(生产不静默退 hashed,见 §13.1)。
        self.embedding = self._resolve_embedding(embedding)
        self.token_counter = token_counter
        self.store: MemoryStore = store or SQLiteMemoryStore(storage_dir or ":memory:")
        self.index: VectorIndex = index or InMemoryVectorIndex(embedding=self.embedding)
        self._compaction = Compaction(
            store=self.store,
            index=self.index,
            llm=self.llm,
            config=self.config,
            timezone=self.timezone,
            overrides=self.prompt_overrides,
            token_counter=self.token_counter,
        )
        self._read = ReadPipeline(
            store=self.store, index=self.index, llm=self.llm, config=self.config, timezone=self.timezone
        )
        self._compact_executor: ThreadPoolExecutor | None = None  # 懒建,单 worker 串行后台压缩

    @staticmethod
    def _resolve_overrides(prompt_overrides: PromptOverrides | None, persona_text: str) -> PromptOverrides:
        import dataclasses

        if prompt_overrides is not None and not isinstance(prompt_overrides, PromptOverrides):
            # 坏配置在构造时就拒绝,别拖到压缩时 AttributeError 或被空 dict 静默忽略。
            raise TypeError(
                f"prompt_overrides must be a PromptOverrides or None, got {type(prompt_overrides).__name__}"
            )
        ov = prompt_overrides or PromptOverrides()
        if persona_text and not ov.persona_text:  # 便捷参数只在 overrides 没填 persona 时生效
            ov = dataclasses.replace(ov, persona_text=persona_text)
        return ov

    @staticmethod
    def _resolve_embedding(embedding: EmbeddingProvider | str | None) -> EmbeddingProvider:
        if isinstance(embedding, EmbeddingProvider):
            return embedding
        if isinstance(embedding, str) and embedding.strip():
            from .embedding.huggingface import HuggingFaceEmbeddingProvider  # 延迟导入重依赖

            return HuggingFaceEmbeddingProvider(model_name=embedding.strip())
        raise ValueError(
            "embedding is required: pass an EmbeddingProvider or a model name string. "
            "Do not rely on a silent hashed fallback in production (see design §13.1)."
        )

    # --- 写侧生命周期 ---

    def record_user_turn(self, content: str, *, actor: Actor | None = None, **fields: Any) -> dict[str, Any]:
        return self._record(role="user", content=content, actor=actor, **fields)

    def record_assistant_turn(
        self, reply: str, *, in_reply_to: dict[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        return self._record(role="assistant", content=reply, actor=None, **fields)

    def _record(self, *, role: str, content: str, actor: Actor | None, **fields: Any) -> dict[str, Any]:
        ts = int(fields.pop("timestamp", None) or time.time())
        ns = self.namespace if actor is None else self._with_actor(actor)
        rec = self.store.add_message(
            namespace=ns,
            role=role,
            content=content,
            timestamp=ts,
            source_id=fields.pop("source_id", None) or uuid.uuid4().hex,
            date_label=timestamp_to_date_label(ts, self.timezone),
            time_of_day=infer_time_of_day(ts, self.timezone),
            memory_metadata=coerce_memory_metadata(
                fields.pop("memory_metadata", None),
                categories=self.config.categories,
                enable_flavor=self.config.enable_flavor,
            ).to_dict(),
        )
        # outbox:写库已成功(pending);向量 upsert 失败就留 pending,交给 reindex_pending 自愈,不阻断记录。
        try:
            self.index.upsert([build_raw_entry(rec)])
            self.store.set_index_status(rec["source_id"], "indexed")
            rec["index_status"] = "indexed"  # 返回值与 store 同步,别让调用方误判
        except Exception:
            rec["index_status"] = "pending"
        return rec

    def _with_actor(self, actor: Actor) -> Namespace:
        ns = self.namespace
        return Namespace(
            user_id=ns.user_id,
            tenant_id=ns.tenant_id,
            domain_id=ns.domain_id,
            conversation_id=ns.conversation_id,
            actor=actor,
        )

    def compact_due_sync(self) -> dict[str, int]:
        """同步压缩(阻塞直到完成)。

        仅建议用于单测、CLI、管理脚本或进程退出前的确定性 flush。聊天请求链路应使用
        compact_due_background(),避免压缩 LLM 调用阻塞用户可见回复。
        """
        return self._compaction.run_due(namespace=self.namespace)

    def compact_due_background(self) -> Future:
        """异步压缩:提交到单 worker 后台线程,立刻返回 Future,不阻塞聊天链路。

        单 worker = 所有压缩串行,不会压垮 LLM/库;同 namespace 还有 Compaction 内部锁兜底。
        Future.result() 可取压缩统计;聊天产品里通常 fire-and-forget。用完调 close() 收线程。
        """
        if self._compact_executor is None:
            self._compact_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memcore-compact")
        return self._compact_executor.submit(self._compaction.run_due, namespace=self.namespace)

    def close(self, *, wait: bool = True) -> None:
        """收掉后台压缩线程。store/index 生命周期由调用方自理。"""
        if self._compact_executor is not None:
            self._compact_executor.shutdown(wait=wait)
            self._compact_executor = None

    def acquaintance_note(self, *, now_ts: int | None = None, cross_conversation: bool = True) -> str:
        """(opt-in,陪伴向)相处时间感:"第一次聊天是哪天、到今天认识第几天"。

        无历史返回 ""。其它领域(金融/客服)不调用即可,不强加。
        """
        first = self.store.get_first_message_timestamp(namespace=self.namespace, cross_conversation=cross_conversation)
        if not first:
            return ""
        tz = ZoneInfo(self.timezone)
        from datetime import datetime

        now = int(now_ts or time.time())
        first_date = datetime.fromtimestamp(first, tz).date()
        now_date = datetime.fromtimestamp(now, tz).date()
        days = max(1, (now_date - first_date).days + 1)
        return (
            f"按本地留下的记录,你和该用户第一次留下对话是在 {first_date.isoformat()};到今天是认识的第 {days} 天。"
            "这是一条相处时间线索,只在谈到初识/陪伴/纪念/久未联系时自然带入,不必每轮报数,"
            "也不要把『首次留下记录』夸张成你能证明的现实起点。"
        )

    def reindex_pending(self, *, limit: int = 100) -> dict[str, int]:
        """outbox 自愈:把 index_status=pending 的记录补做向量 upsert。可定期/启动时调用。"""
        from .index.entry_builder import build_raw_entry, build_semantic_entry, build_summary_entry

        builders = {
            "raw": build_raw_entry,
            "summary": build_summary_entry,
            "semantic_summary": build_semantic_entry,
        }
        pending = self.store.list_pending_index(limit=limit)
        repaired = failed = 0
        for rec in pending:
            builder = builders.get(str(rec.get("entry_type")), build_raw_entry)
            entry = builder(rec)
            try:
                self.index.upsert([entry])
                self.store.set_index_status(entry["source_id"], "indexed")
                repaired += 1
            except Exception:
                failed += 1
        return {"scanned": len(pending), "repaired": repaired, "failed": failed}

    def update_turn_metadata(self, source_id: str, memory_metadata: dict[str, Any] | None) -> dict[str, Any]:
        """回写 raw turn 的 memory_metadata,并重建 raw 向量索引。

        用于 Chat Output Adapter:先安全记录用户原文,等聊天模型 final JSON 出来后,
        再把模型本人给出的 memory_metadata 回写到该 raw message。
        """
        sid = str(source_id or "").strip()
        if not sid:
            return {
                "ok": False,
                "status": "not_found",
                "source_id": "",
                "memory_metadata": {},
                "index_status": "",
                "reason": "source_id_required",
            }
        metadata = coerce_memory_metadata(
            memory_metadata,
            categories=self.config.categories,
            enable_flavor=self.config.enable_flavor,
        ).to_dict()
        rec = self.store.update_message_memory_metadata(
            namespace=self.namespace, source_id=sid, memory_metadata=metadata
        )
        if rec is None:
            return {
                "ok": False,
                "status": "not_found",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "",
                "reason": "source_id_not_found_or_not_raw",
            }
        try:
            self.index.upsert([build_raw_entry(rec)])
            self.store.set_index_status(sid, "indexed")
            rec["index_status"] = "indexed"
            return {
                "ok": True,
                "status": "updated",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "indexed",
                "reason": "",
            }
        except Exception as exc:
            self.store.set_index_status(sid, "pending")
            return {
                "ok": False,
                "status": "pending",
                "source_id": sid,
                "memory_metadata": metadata,
                "index_status": "pending",
                "reason": str(exc) or exc.__class__.__name__,
            }

    def embedding_status(self) -> dict[str, Any]:
        return {
            "provider": self.embedding.name,
            "version": self.embedding.version,
            "dimension": self.embedding.dimension,
            "degraded": self.embedding.name == "hashed",  # hashed = 无语义,只应出现在测试/显式降级
        }

    def verify_embedding(self, **kwargs: Any) -> dict[str, Any]:
        """语义自检:确认当前 embedding 真有语义(近义词更近)。上线前调用,ok=False 说明已降级。"""
        from .embedding.verify import verify_embedding as _verify

        return _verify(self.embedding, **kwargs)

    # --- 读侧生命周期 ---

    def build_prompt_context(self, *, current: dict[str, Any]) -> dict[str, Any]:
        """拼"可见三层",供使用方拼最终聊天 prompt。

        当前轮是否需要检索由聊天模型通过 retrieve/read_timeline 工具自行决定;
        memcore 不做前置 router,避免每轮额外 LLM 判定带来的成本和误判。
        """
        now_ts = int(current.get("timestamp") or 0)
        return self._read.build_context(namespace=self.namespace, now_ts=now_ts)

    def retrieve(self, query: str, **filters: Any) -> list[str]:
        """显式检索(工具式),返回经 verifier 确认的记忆片段文本。"""
        return self._read.retrieve(namespace=self.namespace, query=query, **filters)

    def retrieve_for_turn(self, *, current: dict[str, Any], query: str, **filters: Any) -> list[str]:
        """当前聊天轮的安全检索工具:默认排除本轮 prompt 已可见记忆。

        生命周期是先 record_user_turn(current) 再让聊天模型决定是否调用检索工具。此时当前消息已经入索引,
        且 build_prompt_context 会把当前未摘要 raw/可见摘要/可见语义放进 prompt。宿主工具包装层应优先调用
        这个方法,避免工具检索把已可见内容重复搜回来。
        """
        exclude_ids = {
            str(item).strip() for item in (filters.pop("exclude_source_ids", None) or []) if str(item or "").strip()
        }
        now_ts = int((current or {}).get("timestamp") or 0)
        exclude_ids |= self._read.visible_source_ids(namespace=self.namespace, now_ts=now_ts)
        current_id = str((current or {}).get("source_id") or "").strip()
        if current_id:
            exclude_ids.add(current_id)
        return self.retrieve(query, exclude_source_ids=sorted(exclude_ids), **filters)

    def read_timeline(
        self,
        *,
        date_from: str,
        date_to: str = "",
        time_periods: list[str] | None = None,
        cross_conversation: bool = False,
    ) -> dict[str, Any]:
        """时间线工具:按日期(范围)精确读原始对话,不走向量。与 retrieve 互补。

        date_to 省略时等于 date_from(单日)。time_periods 接受上午/下午/晚上/凌晨等别名。
        这是精确读工具:日期/时间段非法时**结构化报 invalid_filter**,绝不静默放宽成"查整天"。
        """
        from datetime import date

        from .rendering import render_timeline
        from .time_anchor import normalize_time_periods

        def _invalid(reason: str) -> dict[str, Any]:
            return {"status": "invalid_filter", "reason": reason, "messages": [], "message_count": 0, "text": ""}

        def _is_iso_date(value: str) -> bool:
            try:
                date.fromisoformat(value)
                return True
            except ValueError:
                return False

        start = str(date_from or "").strip()
        end = str(date_to or "").strip() or start
        if not _is_iso_date(start) or not _is_iso_date(end):
            return _invalid("date_must_be_YYYY-MM-DD")
        if start > end:
            return _invalid("date_from_after_date_to")

        raw_periods = [str(p).strip() for p in (time_periods or []) if str(p or "").strip()]
        unknown = [p for p in raw_periods if not normalize_time_periods([p])]
        if unknown:
            return _invalid(f"unknown_time_periods:{unknown}")
        periods = normalize_time_periods(raw_periods)
        messages = self.store.get_messages_by_date_range(
            namespace=self.namespace,
            date_from=start,
            date_to=end,
            time_periods=periods,
            cross_conversation=cross_conversation,
        )
        return {
            "status": "ok" if messages else "empty",
            "date_from": start,
            "date_to": end,
            "time_periods": periods,
            "message_count": len(messages),
            "messages": messages,
            "text": render_timeline(messages, tz=self.timezone),
        }

    def forget_namespace(self, namespace: Namespace | None = None) -> dict[str, Any]:
        """定向遗忘:删除 store 记录,并尽力同步清 VectorIndex;失败时结构化报告 partial。"""
        target = namespace or self.namespace
        deleted_ids = self.store.delete_namespace(namespace=target)
        try:
            self.index.delete(deleted_ids)
        except Exception as exc:
            return {
                "ok": False,
                "status": "partial",
                "deleted_count": len(deleted_ids),
                "source_ids": deleted_ids,
                "index_deleted": False,
                "reason": str(exc) or exc.__class__.__name__,
            }
        return {
            "ok": True,
            "status": "deleted",
            "deleted_count": len(deleted_ids),
            "source_ids": deleted_ids,
            "index_deleted": True,
        }
