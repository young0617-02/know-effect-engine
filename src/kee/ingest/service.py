"""M1 摄取编排：真实文件 → 解析 → 元数据 → 准入 → 权威源 → Outbox → 索引 → 审计。

与 Gate 0 的 build_index() 的关系：那条路是给自建语料批量建库用的；这里给的是
面向上传的单文档路径，两者共用 admission / chunking / Outbox / 物料仓，
所以「先权威源、后索引」「索引可重建」这两条纪律在两条路径上是同一套实现。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kee.db.outbox import OutboxDispatcher
from kee.db.pg import AuthoritativeStore
from kee.db.vector_store import VectorStore
from kee.ingest.admission import admit, checksum_of
from kee.ingest.chunking import (
    attach_block_metadata,
    chunk_blocks,
    chunk_document,
    chunk_strategy_from_settings,
    granularity_from_settings,
)
from kee.ingest.content import count_blocks
from kee.ingest.media import MediaPipeline
from kee.ingest.metadata import resolve_meta
from kee.ingest.parsers import ParsedDocument, parse_file
from kee.retrieval.milvus_store import KbStore


class AdmissionRejected(Exception):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("；".join(reasons))
        self.reasons = reasons


class DuplicateDocument(Exception):
    def __init__(self, existing_doc_id: str) -> None:
        super().__init__(f"内容指纹已存在于文档 {existing_doc_id}")
        self.existing_doc_id = existing_doc_id


@dataclass
class IngestResult:
    doc_id: str
    version_id: str
    title: str
    new_document: bool
    chunks_parent: int
    chunks_child: int
    indexed: int
    parse_notes: list[str] = field(default_factory=list)
    meta_warnings: list[str] = field(default_factory=list)
    outbox: dict[str, Any] = field(default_factory=dict)
    timings_sec: dict[str, float] = field(default_factory=dict)
    trace_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "version_id": self.version_id,
            "title": self.title,
            "new_document": self.new_document,
            "chunks": {"parent": self.chunks_parent, "child": self.chunks_child},
            "indexed": self.indexed,
            "parse_notes": self.parse_notes,
            "meta_warnings": self.meta_warnings,
            "outbox": self.outbox,
            "timings_sec": self.timings_sec,
            "trace_id": self.trace_id,
        }


class IngestionService:
    def __init__(
        self,
        pg: AuthoritativeStore | None = None,
        kb: KbStore | None = None,
        vectors: VectorStore | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.pg = pg or AuthoritativeStore()
        self.kb = kb or KbStore()
        self.vectors = vectors or VectorStore()
        self.scheduler = scheduler
        self.media = MediaPipeline.from_settings()

    # ---- 解析与校验 ----
    def parse(self, *, data: bytes | None, path: str | Path | None, filename: str) -> ParsedDocument:
        target = Path(path) if path else Path(filename)
        return parse_file(target, raw=data, filename=filename, media=self.media)

    def _card_from(self, parsed: ParsedDocument, form: dict[str, Any], filename: str) -> tuple[dict, list[str]]:
        sample = "\n".join(p for s in parsed.sections[:3] for p in s.paragraphs[:3])
        res = resolve_meta(form, filename=filename, parsed_meta=parsed.meta_hint, sample_text=sample)
        override = res.meta.pop("title", "") or str(form.get("title") or "")
        # status 是写入期决定的文档状态，不属于解析卡片字段
        res.meta.pop("status", None)
        card = parsed.to_card(title_override=override, **res.meta)
        return card, res.warnings

    # ---- 主流程 ----
    def ingest(
        self,
        *,
        filename: str,
        data: bytes | None = None,
        path: str | Path | None = None,
        form: dict[str, Any] | None = None,
        actor: str = "uploader",
        trace_id: str | None = None,
        idempotency_key: str | None = None,
        skip_embedding: bool = False,
    ) -> IngestResult:
        trace_id = trace_id or f"ing-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        if idempotency_key:
            prior = self.pg.reserve_idempotency(idempotency_key, scope="document_upload")
            if prior:
                return IngestResult(
                    doc_id=prior.get("doc_id", "?"),
                    version_id=prior.get("version_id", "?"),
                    title=prior.get("title", ""),
                    new_document=False,
                    chunks_parent=0,
                    chunks_child=0,
                    indexed=0,
                    trace_id=trace_id,
                    outbox={"replayed": True, "idempotency_key": idempotency_key},
                )

        parsed = self.parse(data=data, path=path, filename=filename)
        t_parse = time.perf_counter()
        card, meta_warnings = self._card_from(parsed, form or {}, filename)
        source_bytes = data if data is not None else (Path(path).read_bytes() if path else b"")
        if source_bytes:
            card["_source_checksum"] = hashlib.sha256(source_bytes).hexdigest()

        rep = admit([card])
        if rep.rejected:
            raise AdmissionRejected(rep.rejected[0][1])
        card = rep.accepted[0]
        checksum = card["_checksum"]

        existing_by_hash = None
        with self.pg.connect() as conn:
            existing_by_hash = conn.execute(
                "SELECT id FROM document WHERE checksum=%s AND id <> %s", (checksum, card["doc_id"])
            ).fetchone()
        if existing_by_hash:
            raise DuplicateDocument(existing_by_hash["id"])

        doc_row = {
            "id": card["doc_id"],
            "title": card["title"],
            "industry": card["industry"],
            "customer": card["customer"],
            "project": card["project"],
            "doc_type": card["doc_type"],
            "source_level": card["source_level"],
            "security_level": card["security_level"],
            "status": "active",
            "checksum": checksum,
        }
        self.pg.init_schema()
        # 必须在 upsert 之前判断是否新文档：bump_version 之后版本列表必然非空
        new_document = not self.pg.versions_of(card["doc_id"])
        self.pg.upsert_document(doc_row)
        granularity = granularity_from_settings()
        strategy = chunk_strategy_from_settings()
        block_mode = strategy == "block-v1" and bool(parsed.blocks)
        scheme = "block-v1" if block_mode else (
            f"parent-child:v1{granularity.scheme_tag()}" if granularity else "parent-child:v1"
        )
        version_id = self.pg.bump_version(card["doc_id"], scheme, checksum)
        # bump_version 已把旧版本置 superseded；这里必须把旧版本的索引行撤掉，
        # 否则同一文档 v1/v2 会同时被检索命中，去重要靠检索层兜底是错的。
        for stale in self.pg.versions_of(card["doc_id"]):
            if stale["status"] == "superseded" and stale["id"] != version_id:
                self.kb.delete_version(stale["id"])
        rows = (
            chunk_blocks(parsed.blocks, version_id, card["doc_id"], card["title"])
            if block_mode
            else attach_block_metadata(
                chunk_document(card, version_id, granularity=granularity),
                parsed.blocks,
                version_id,
            )
        )
        self.pg.insert_chunks(rows)
        # Phase 1：结构块与 chunk 解耦写入；不参与本次分块/索引读数。
        block_count = self.pg.insert_blocks(version_id, card["doc_id"], parsed.blocks)
        children = [r for r in rows if not r["is_parent"]]
        parents = [r for r in rows if r["is_parent"]]
        self.pg.enqueue_outbox(
            [
                {
                    "aggregate_type": "chunk_batch",
                    "aggregate_id": version_id,
                    "event_type": "index_upsert",
                    "payload": {"version_id": version_id, "chunk_ids": [c["id"] for c in children]},
                }
            ]
        )
        t_write = time.perf_counter()

        if children:
            import numpy as np

            texts = [c["content"] for c in children]
            if skip_embedding:
                # 干跑（不占 GPU）：给占位向量，用于跑通链路与权限/幂等测试
                arr = np.zeros((len(texts), 1024), dtype="float32")
                arr[:, 0] = 1.0
            else:
                if self.scheduler is None:
                    raise RuntimeError("未提供 scheduler/embedder，且 skip_embedding=False")
                with self.scheduler.use("embed"):
                    arr = self.scheduler.embedder.embed_passages(texts, batch_size=16)
            self.vectors.save(version_id, [c["id"] for c in children], arr)
        t_embed = time.perf_counter()

        dispatcher = OutboxDispatcher(self.pg, self.kb, self.vectors)
        disp = dispatcher.drain_with_backoff()
        self.pg.append_audit(
            actor,
            "document_ingested",
            card["doc_id"],
            trace_id,
            {
                "version_id": version_id,
                "source_format": parsed.source_format,
                "child_chunks": len(children),
                "content_blocks": block_count,
                "block_summary": count_blocks(parsed.blocks),
                "chunk_scheme": scheme,
                "chunk_strategy": strategy,
                "meta_warnings": meta_warnings,
                "parse_notes": parsed.notes,
            },
        )
        t_end = time.perf_counter()

        result = IngestResult(
            doc_id=card["doc_id"],
            version_id=version_id,
            title=card["title"],
            new_document=bool(new_document),
            chunks_parent=len(parents),
            chunks_child=len(children),
            indexed=len(self.kb.all_chunk_ids(version_id=version_id)),
            parse_notes=parsed.notes,
            meta_warnings=meta_warnings,
            outbox={k: v for k, v in disp.items() if k != "attempts"},
            timings_sec={
                "parse": round(t_parse - t0, 3),
                "authoritative_write": round(t_write - t_parse, 3),
                "embed": round(t_embed - t_write, 3),
                "index": round(t_end - t_embed, 3),
                "total": round(t_end - t0, 3),
            },
            trace_id=trace_id,
        )
        if idempotency_key:
            self.pg.store_idempotency(idempotency_key, result.as_dict())
        return result

    # ---- 复核与回滚 ----
    def review(self, doc_id: str, decision: str, *, actor: str, note: str = "", trace_id: str | None = None) -> dict[str, Any]:
        if decision not in ("pass", "reject"):
            raise ValueError("decision 必须是 pass 或 reject")
        doc = self.pg.get_document(doc_id)
        if not doc:
            raise KeyError(f"文档不存在：{doc_id}")
        # pass → 文档可用（保持 active）；reject → 文档 rejected 且生效版本下线、索引撤掉，
        # 否则被驳回的内容仍会被检索命中，这是知识准入的最后一道闸
        if decision == "pass":
            self.pg.set_document_status(doc_id, "active")
            # 盖复核章（处置 C 的凭据）：章落在**版本**上，不落在文档上——
            # 否则一次复核就把这个 doc 以后所有新版本都自动放行了，那正是投毒路径要防的形状。
            for ver in self.pg.versions_of(doc_id):
                if ver["status"] == "effective":
                    self.pg.mark_reviewed(ver["id"], actor)
        else:
            self.pg.set_document_status(doc_id, "rejected")
            for ver in self.pg.versions_of(doc_id):
                if ver["status"] == "effective":
                    self.pg.set_version_status(ver["id"], "superseded")
                    self.kb.delete_version(ver["id"])
        self.pg.append_audit(
            actor, f"document_review_{decision}", doc_id, trace_id or f"rev-{doc_id}", {"note": note}
        )
        return {"doc_id": doc_id, "status": self.pg.get_document(doc_id)["status"], "decision": decision}

    def rollback(self, doc_id: str, to_version_id: str, *, actor: str, trace_id: str | None = None) -> dict[str, Any]:
        versions = {v["id"]: v for v in self.pg.versions_of(doc_id)}
        if to_version_id not in versions:
            raise KeyError(f"目标版本不属于该文档：{to_version_id}")
        for vid in versions:
            if vid != to_version_id:
                self.kb.delete_version(vid)
        self.pg.set_version_status(to_version_id, "effective")
        indexed = self.index_version(to_version_id)
        self.pg.append_audit(
            actor, "document_rollback", doc_id, trace_id or f"rb-{doc_id}", {"to_version": to_version_id}
        )
        return {"doc_id": doc_id, "effective_version": to_version_id, **indexed}

    def index_version(self, version_id: str) -> dict[str, Any]:
        """只靠权威源 + 物料仓把某个版本重新灌进索引（升版/回滚共用）。"""
        t0 = time.perf_counter()
        with self.pg.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.doc_id, c.version_id, c.parent_id, c.section_no, c.page, c.order_no, "
                "c.content, d.industry, d.customer, d.project, d.doc_type, d.source_level, d.security_level "
                "FROM chunk c JOIN document d ON d.id = c.doc_id "
                "WHERE c.version_id = %s AND c.is_parent = false ORDER BY c.order_no",
                (version_id,),
            ).fetchall()
        vecs = self.vectors.vector_of(version_id)
        docs = [
            {
                "chunk_id": r["id"],
                "doc_id": r["doc_id"],
                "version_id": r["version_id"],
                "parent_id": r["parent_id"] or "",
                "industry": r["industry"],
                "customer": r["customer"],
                "project": r["project"],
                "doc_type": r["doc_type"],
                "source_level": r["source_level"],
                "security_level": int(r["security_level"]),
                "section_no": r["section_no"],
                "page": int(r["page"]),
                "order_no": int(r["order_no"]),
                "text": r["content"],
                "dense": vecs[r["id"]],
            }
            for r in rows
        ]
        written = self.kb.insert(docs) if docs else 0
        self.kb.flush()
        return {"reindexed": written, "seconds": round(time.perf_counter() - t0, 2)}
