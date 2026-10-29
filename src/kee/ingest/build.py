"""离线构建链路：准入 → 分块 → 权威源写入 → Outbox → 向量索引（方案 4.2-B）。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
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
from kee.ingest.content import blocks_from_sections
from kee.ingest.parsers import ParsedSection
from kee.retrieval.milvus_store import KbStore


def load_cards() -> list[dict[str, Any]]:
    from kee.ingest.corpus_data import FACT_CARDS

    return list(FACT_CARDS)


def version_id_of(doc_id: str, version_tag: str) -> str:
    return f"{doc_id}@{version_tag}"


def rebuild_index_from_authoritative(
    pg: AuthoritativeStore, kb: KbStore, vector_store: VectorStore, batch: int = 200
) -> dict[str, Any]:
    """只靠权威源 + 向量物料重建索引（「向量索引可重建」原则的可执行证据）。"""
    t0 = time.perf_counter()
    kb.drop()
    kb.ensure()
    n = 0
    versions = 0
    with pg.connect() as conn:
        for ver in pg.effective_versions():
            rows = conn.execute(
                "SELECT c.id, c.doc_id, c.version_id, c.parent_id, c.section_no, c.page, c.order_no, "
                "c.content, d.industry, d.customer, d.project, d.doc_type, d.source_level, d.security_level "
                "FROM chunk c JOIN document d ON d.id = c.doc_id "
                "WHERE c.version_id = %s AND c.is_parent = false ORDER BY c.order_no",
                (ver["id"],),
            ).fetchall()
            vecs = vector_store.vector_of(ver["id"])
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
            for i in range(0, len(docs), batch):
                n += kb.insert(docs[i : i + batch])
            versions += 1
    kb.flush()
    return {
        "collection": kb.collection,
        "analyzer": kb.analyzer,
        "versions": versions,
        "inserted": n,
        "milvus_count": kb.count(),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def build_index(
    pg: AuthoritativeStore,
    kb: KbStore,
    vector_store: VectorStore,
    embedder,
    version_tag: str = "v1",
    reset: bool = True,
    batch_size: int = 16,
    trace_id: str = "gate0-build",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cards = load_cards()
    report = admit(cards)

    pg.init_schema()
    if reset:
        pg.truncate_all(include_audit=False)
    kb.drop() if reset else None
    kb.ensure()

    child_rows: list[dict[str, Any]] = []
    parent_count = 0
    per_version: dict[str, list[str]] = {}
    now = datetime.now(timezone.utc)
    granularity = granularity_from_settings()
    strategy = chunk_strategy_from_settings()
    block_mode = strategy == "block-v1"
    scheme = "block-v1" if block_mode else (
        f"parent-child:v1{granularity.scheme_tag()}" if granularity else "parent-child:v1"
    )

    for card in report.accepted:
        vid = version_id_of(card["doc_id"], version_tag)
        sections = [
            ParsedSection(
                str(sec["section_no"]),
                str(sec.get("heading", "") or ""),
                int(sec.get("page", 0) or 0),
                [str(p) for p in sec.get("paragraphs", []) or []],
            )
            for sec in card["sections"]
        ]
        blocks = blocks_from_sections(sections)
        rows = (
            chunk_blocks(blocks, vid, card["doc_id"], card["title"])
            if block_mode
            else attach_block_metadata(chunk_document(card, vid, granularity=granularity), blocks, vid)
        )
        doc_row = {
            "id": card["doc_id"],
            "title": card["title"],
            "industry": card["industry"],
            "customer": card["customer"],
            "project": card["project"],
            "doc_type": card["doc_type"],
            "source_level": card["source_level"],
            "security_level": int(card["security_level"]),
            "status": "active",
            "checksum": card["_checksum"],
        }
        pg.upsert_document(doc_row)
        pg.insert_version(
            {
                "id": vid,
                "doc_id": card["doc_id"],
                "version": 1,
                "chunk_scheme": scheme,
                "checksum": checksum_of(card),
                "status": "effective",
                "effective_at": now,
            }
        )
        pg.insert_chunks(rows)
        # 离线语料同样落结构块：来源统一标 native_text，分块读数不受影响。
        pg.insert_blocks(vid, card["doc_id"], blocks)
        children = [r for r in rows if not r["is_parent"]]
        parents = [r for r in rows if r["is_parent"]]
        parent_count += len(parents)
        child_rows.extend(children)
        per_version[vid] = [c["id"] for c in children]
        pg.enqueue_outbox(
            [
                {
                    "aggregate_type": "chunk_batch",
                    "aggregate_id": vid,
                    "event_type": "index_upsert",
                    "payload": {"version_id": vid, "chunk_ids": per_version[vid]},
                }
            ]
        )
        pg.append_audit("ingest", "document_admitted", card["doc_id"], trace_id, {"version_id": vid})

    t_write = time.perf_counter()
    texts = [c["content"] for c in child_rows]
    vectors = embedder.embed_passages(texts, batch_size=batch_size)
    t_embed = time.perf_counter()

    by_version_child = {vid: [] for vid in per_version}
    for c in child_rows:
        by_version_child[c["version_id"]].append(c["id"])
    idx_by_id = {c["id"]: i for i, c in enumerate(child_rows)}
    for vid, ids in by_version_child.items():
        import numpy as np

        arr = np.stack([vectors[idx_by_id[i]] for i in ids])
        vector_store.save(vid, ids, arr)

    dispatcher = OutboxDispatcher(pg, kb, vector_store)
    disp = dispatcher.drain_with_backoff()

    return {
        "version_tag": version_tag,
        "collection": kb.collection,
        "analyzer": kb.analyzer,
        "admission": report.summary(),
        "rejected_detail": report.rejected[:10],
        "duplicates": report.duplicates[:10],
        "docs_indexed": len(report.accepted),
        "parent_chunks": parent_count,
        "child_chunks": len(child_rows),
        "pg_stats": pg.stats(),
        "milvus_count": kb.count(),
        "timings_sec": {
            "authoritative_write": round(t_write - t0, 2),
            "embed": round(t_embed - t_write, 2),
            "total": round(time.perf_counter() - t0, 2),
        },
        "dispatch": {k: v for k, v in disp.items() if k != "attempts"},
    }
