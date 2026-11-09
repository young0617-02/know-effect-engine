"""Phase 2：legacy vs block-v1 的 scratch 检索 A/B。

临时建库/集合，批量 embed + 一次 Milvus flush；不碰生产索引，也不改生产 Outbox 路径。
gold 按各集合 chunk 内容重新装配；结论只用于判断是否值得进入答案/重切评估。

跑法：
    docker compose -f infra/docker-compose.gate0.yml up -d
    .venv/Scripts/python.exe scripts/block_v1_ab.py --out var/gate0/block_v1_ab.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.ingest.admission import admit  # noqa: E402
from kee.ingest.build import load_cards  # noqa: E402
from kee.ingest.chunking import chunk_blocks, chunk_document  # noqa: E402
from kee.ingest.content import blocks_from_sections  # noqa: E402
from kee.ingest.parsers import ParsedSection  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from m2_eval import BASELINE_CFG, metric, run_config  # noqa: E402

SCRATCH = {
    "legacy": ("kee_ab_legacy", "kb_ab_legacy"),
    "block-v1": ("kee_ab_block", "kb_ab_block"),
}


def psql(sql: str) -> None:
    r = subprocess.run(
        ["docker", "exec", "kee-postgres", "psql", "-U", "kee_admin", "-d", "postgres", "-tAc", sql],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])


def create_db(name: str) -> None:
    psql(f"DROP DATABASE IF EXISTS {name} WITH (FORCE);")
    psql(f"CREATE DATABASE {name} OWNER kee_app;")


def drop_db(name: str) -> None:
    try:
        psql(f"DROP DATABASE IF EXISTS {name} WITH (FORCE);")
    except Exception as e:  # noqa: BLE001
        print(f"清理提示 {name}: {e}")


def sections_of(card: dict[str, Any]) -> list[ParsedSection]:
    return [
        ParsedSection(str(sec["section_no"]), str(sec.get("heading", "") or ""),
                      int(sec.get("page", 0) or 0),
                      [str(p) for p in sec.get("paragraphs", []) or []])
        for sec in card["sections"]
    ]


def build_scratch(pg: AuthoritativeStore, kb: KbStore, embedder, strategy: str) -> dict[str, Any]:
    cards = admit(load_cards()).accepted
    pg.init_schema()
    pg.truncate_all(include_audit=False)
    kb.drop()
    kb.ensure()
    block_mode = strategy == "block-v1"
    scheme = "block-v1" if block_mode else "parent-child:v1"
    now = datetime.now(timezone.utc)
    child_rows: list[dict[str, Any]] = []
    parents = 0
    doc_meta: dict[str, dict[str, Any]] = {}
    for card in cards:
        vid = f"{card['doc_id']}@v1"
        pg.upsert_document({
            "id": card["doc_id"], "title": card["title"], "industry": card["industry"],
            "customer": card["customer"], "project": card["project"], "doc_type": card["doc_type"],
            "source_level": card["source_level"], "security_level": int(card["security_level"]),
            "status": "active", "checksum": card["_checksum"],
        })
        pg.insert_version({
            "id": vid, "doc_id": card["doc_id"], "version": 1, "chunk_scheme": scheme,
            "checksum": card["_checksum"], "status": "effective", "effective_at": now,
        })
        blocks = blocks_from_sections(sections_of(card))
        rows = (chunk_blocks(blocks, vid, card["doc_id"], card["title"])
                if block_mode else chunk_document(card, vid))
        pg.insert_chunks(rows)
        pg.insert_blocks(vid, card["doc_id"], blocks)
        doc_meta[card["doc_id"]] = card
        parents += sum(1 for r in rows if r["is_parent"])
        child_rows.extend(r for r in rows if not r["is_parent"])

    t0 = time.perf_counter()
    texts = [r["content"] for r in child_rows]
    vectors = embedder.embed_passages(texts, batch_size=16)
    docs = []
    for i, row in enumerate(child_rows):
        card = doc_meta[row["doc_id"]]
        docs.append({
            "chunk_id": row["id"], "doc_id": row["doc_id"], "version_id": row["version_id"],
            "parent_id": row["parent_id"] or "", "industry": card["industry"],
            "customer": card["customer"], "project": card["project"], "doc_type": card["doc_type"],
            "source_level": card["source_level"], "security_level": int(card["security_level"]),
            "section_no": row["section_no"], "page": int(row["page"]),
            "order_no": int(row["order_no"]), "text": row["content"],
            "dense": vectors[i].tolist(),
        })
    for i in range(0, len(docs), 200):
        kb.insert(docs[i:i + 200])
    kb.flush()
    return {
        "version_tag": "v1", "scheme": scheme, "docs": len(cards),
        "parents": parents, "children": len(child_rows), "milvus_count": kb.count(),
        "embed_sec": round(time.perf_counter() - t0, 2),
    }


def _scratch_dsn(db: str) -> str:
    base = get_settings().postgres_dsn
    if not base:
        raise SystemExit("KEE_POSTGRES_DSN 未配置；先按 .env.example 填好 .env")
    return urlunsplit(urlsplit(base)._replace(path=f"/{db}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="var/gate0/block_v1_ab.json")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    if subprocess.run(["docker", "exec", "kee-postgres", "pg_isready", "-U", "kee"],
                      capture_output=True).returncode != 0:
        print("PostgreSQL 未就绪，先起 infra/docker-compose.gate0.yml")
        return 2

    result: dict[str, Any] = {"baseline_cfg": BASELINE_CFG, "top_k": args.top_k, "runs": {}}
    sched = ResidencyScheduler(policy="co_resident")
    sched.embedder.load()
    try:
        for strategy, (db, coll) in SCRATCH.items():
            create_db(db)
            pg = AuthoritativeStore(dsn=_scratch_dsn(db))
            kb = KbStore(coll, "jieba")
            t0 = time.perf_counter()
            built = build_scratch(pg, kb, sched.embedder, strategy)
            built["seconds"] = round(time.perf_counter() - t0, 1)
            eval_set = build_eval_set(pg, write_files=False)
            metrics = run_config("baseline_declared", BASELINE_CFG, eval_set["questions"],
                                 kb, sched, top_k=args.top_k)
            result["runs"][strategy] = {
                "db": db, "collection": coll, "build": built,
                "eval_stats": eval_set["stats"], "n_chunks": kb.count(),
                "metrics": {
                    **{k: metrics.get(k) for k in ("n_graded", "n", "recall@5", "hit@5", "mrr")},
                    "strict": metrics.get("strict_gold"),
                    "latency_ms": metrics.get("latency_ms"),
                    "degraded_count": metrics.get("degraded_count"),
                    "recall@5_ci95": metrics.get("recall@5_ci95"),
                },
                "per_question": metrics.get("per_question"),
            }
            print(f"{strategy}: chunks={kb.count()} build={built}")
            print(f"  metrics={result['runs'][strategy]['metrics']}")
            kb.close()
            drop_db(db)
            try:
                k = KbStore(coll, "jieba"); k.drop(); k.close()
            except Exception as e:  # noqa: BLE001
                print(f"集合清理提示 {coll}: {e}")
    finally:
        sched.release("embed")
        sched.release("rerank")
        for db, coll in SCRATCH.values():
            drop_db(db)
            try:
                k = KbStore(coll, "jieba"); k.drop(); k.close()
            except Exception as e:  # noqa: BLE001
                print(f"集合清理提示 {coll}: {e}")

    legacy = result["runs"].get("legacy", {}).get("metrics", {})
    block = result["runs"].get("block-v1", {}).get("metrics", {})
    result["delta_block_minus_legacy"] = {
        "recall@5": round(metric(block, "recall", args.top_k) - metric(legacy, "recall", args.top_k), 4),
        "hit@5": round(metric(block, "hit", args.top_k) - metric(legacy, "hit", args.top_k), 4),
        "mrr": round((block.get("mrr") or 0.0) - (legacy.get("mrr") or 0.0), 4),
        "strict_recall@5": round(
            metric(block.get("strict") or {}, "recall", args.top_k)
            - metric(legacy.get("strict") or {}, "recall", args.top_k), 4),
    }
    result["note"] = (
        "scratch 批量建库（一次 flush），不是生产 Outbox 逐文档投影；"
        "gold 按各自集合 chunk 内容重新装配，legacy v1 id 与现网一致，block-v1 未搬运 547 行人工标注。"
        "本对照只回答两种分块方案各自的可检索性，不替代答案/重切裁决。"
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"delta={result['delta_block_minus_legacy']}")
    print(f"证据：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
