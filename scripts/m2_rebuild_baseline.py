"""M2-01 重建基线索引：把 Gate 0 遗留的 193 块旧双前缀命名统一到修复后的 chunk-id 口径。

为什么这是一次「必须自证等价」的重建：
  truncate document/chunk/version/outbox 是破坏性动作，而语料是 `corpus_data.FACT_CARDS`
  直接生成的卡片、不经过任何解析器，所以**正文逐块不变、只有 id 变**是本次重建唯一的
  正确性不变量。若这条不成立，说明我改 `_clean`/分块时动了正文，那必须先解释清楚再重建。

流程：备份 → 记录 before → 重建（含向量物料重嵌）→ 记录 after → 三项校验 → 落证据。
任一项不过即非零退出，不做「看起来没问题」的软判定。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.db.vector_store import VectorStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.gate0.g02_chinese_bm25 import COLLECTIONS  # noqa: E402
from kee.ingest.build import build_index, rebuild_index_from_authoritative  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402

# 对齐键：版本 + 章节号 + 块序。这三者在改名前后都稳定，能唯一定位一块正文
KEY = lambda r: (r["version_id"], str(r["section_no"]), int(r["order_no"]))  # noqa: E731
NEW_ID = re.compile(r"^[^#]+#(\d+(?:\.\d+)*)#c\d+$")
OLD_DUP = re.compile(r"^[^#@]+@[^#@]+@v\d+#")


def inventory(pg: AuthoritativeStore) -> dict[str, dict]:
    with pg.connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.version_id, c.section_no, c.order_no, c.content, c.is_parent, c.page, c.parent_id "
            "FROM chunk c ORDER BY c.id"
        ).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        out[KEY(r)] = {
            "id": r["id"],
            "is_parent": bool(r["is_parent"]),
            "page": int(r["page"]),
            "sha1": hashlib.sha1(r["content"].encode("utf-8")).hexdigest()[:12],
            "chars": len(r["content"]),
        }
    return out


def snapshot(art: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("vectors", "eval_set.json", "qrels.jsonl"):
        src = art / name
        if not src.exists():
            continue
        target = dst / name
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            shutil.copy2(src, target)
        copied.append(name)
    return copied


def gold_by_qid(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            out[d["qid"]] = list(d.get("gold_chunk_ids") or [])
    return out


def main() -> int:
    s = get_settings()
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    vs = VectorStore()

    before = inventory(pg)
    gold_before = gold_by_qid(art / "qrels.jsonl")
    with pg.connect() as conn:
        docs_before = conn.execute("SELECT count(*) n FROM document").fetchone()["n"]
    old_ids = {v["id"] for v in before.values()}
    n_old_dup = sum(1 for i in old_ids if OLD_DUP.match(i))
    print(f"[before] 分块 {len(before)}（父 {sum(1 for v in before.values() if v['is_parent'])}）"
          f" 文档 {docs_before} 旧双前缀命名 {n_old_dup} 块")

    snap = snapshot(art, art / "pre_m2_rebuild")
    (art / "pre_m2_rebuild" / "inventory_before.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in before.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[备份] {art / 'pre_m2_rebuild'} ← {snap} + inventory_before.json")

    sched = ResidencyScheduler(policy="co_resident")
    t0 = time.perf_counter()
    with sched.use("embed"):
        rep = build_index(
            pg, KbStore(s.milvus_collection, "jieba"), vs, sched.embedder, reset=True, trace_id="m2-rebuild"
        )
    build_s = round(time.perf_counter() - t0, 1)
    print(f"[重建] 文档 {rep['docs_indexed']} 父 {rep['parent_chunks']} 子 {rep['child_chunks']} "
          f"索引 {rep['milvus_count']} 用时 {build_s}s（含嵌入 {rep['timings_sec']['embed']}s）")

    refreshed = {}
    for name, coll in COLLECTIONS.items():
        if coll == s.milvus_collection:
            continue
        r = rebuild_index_from_authoritative(pg, KbStore(coll, name), vs)
        refreshed[name] = {"inserted": r["inserted"], "count": r["milvus_count"]}
    print(f"[对照集合刷新] {json.dumps(refreshed, ensure_ascii=False)}")

    after = inventory(pg)
    ev = build_eval_set(pg)
    gold_after = {r["qid"]: list(r["gold_chunk_ids"]) for r in ev["questions"]}

    checks: dict[str, dict] = {}

    # 校验 1：对齐键集合与正文逐块一致（只有 id 允许变）
    keys_same = set(before) == set(after)
    content_diff = [
        {"key": f"{k[0]}|{k[1]}|{k[2]}", "before": before[k]["sha1"], "after": after[k]["sha1"]}
        for k in sorted(set(before) & set(after))
        if before[k]["sha1"] != after[k]["sha1"]
    ]
    page_diff = [k for k in set(before) & set(after) if before[k]["page"] != after[k]["page"]]
    checks["正文逐块不变"] = {
        "通过": keys_same and not content_diff and not page_diff,
        "对齐键数": len(set(before) & set(after)),
        "键集合相同": keys_same,
        "内容漂移块数": len(content_diff),
        "样例": content_diff[:5],
        "页码漂移块数": len(page_diff),
    }

    # 校验 2：命名口径统一（无旧双前缀，且全部匹配新式）
    new_ids = {v["id"] for v in after.values()}
    still_dup = sorted(i for i in new_ids if OLD_DUP.match(i))
    unmatched = sorted(i for i in new_ids if not NEW_ID.match(i) and not i.endswith("#parent"))
    checks["chunk-id 命名统一"] = {
        "通过": not still_dup and not unmatched,
        "重建前旧双前缀块数": n_old_dup,
        "重建后旧双前缀块数": len(still_dup),
        "不匹配新式命名的块数": len(unmatched),
        "样例": (still_dup or unmatched)[:3],
    }

    # 校验 3：gold 映射在改名下一一对应（题数、每题 gold 数、成员都能对上）
    id_map = {}
    for k in set(before) & set(after):
        id_map[before[k]["id"]] = after[k]["id"]
    drift = []
    qids = sorted(set(gold_before) | set(gold_after))
    for q in qids:
        gb, ga = gold_before.get(q, []), gold_after.get(q, [])
        mapped = sorted(id_map.get(x, x) for x in gb)
        if mapped != sorted(ga):
            drift.append({"qid": q, "before映射后": len(mapped), "after": len(ga)})
    checks["gold 映射等价"] = {
        "通过": bool(gold_before) and not drift,
        "题数": len(qids),
        "gold 漂移题数": len(drift),
        "样例": drift[:5],
        "重建后评测集统计": ev["stats"],
    }

    # 校验 4：索引与权威源归零对账
    eff = [v["id"] for v in pg.effective_versions()]
    auth_children = set(pg.chunk_ids(version_ids=eff))
    child_ids = {v["id"] for v in after.values() if not v["is_parent"]}
    kb = KbStore(s.milvus_collection, "jieba")
    indexed = kb.count()
    kb.close()
    checks["索引与权威源一致"] = {
        "通过": auth_children == child_ids and indexed == len(auth_children),
        "权威源子块": len(auth_children),
        "分块表子块": len(child_ids),
        "Milvus 条数": indexed,
    }

    out = {
        "before": {"chunks": len(before), "old_double_prefix": n_old_dup, "documents": docs_before},
        "build": {k: rep[k] for k in ("docs_indexed", "parent_chunks", "child_chunks", "milvus_count", "admission")},
        "build_seconds": build_s,
        "timings_sec": rep["timings_sec"],
        "analyzer_collections": refreshed,
        "after": {"chunks": len(after), "documents": pg.list_documents(limit=1, status=None)["total"]},
        "checks": checks,
        "snapshot_dir": str(art / "pre_m2_rebuild"),
    }
    (art / "m2_rebuild.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = True
    for name, c in checks.items():
        v = c["通过"]
        ok &= bool(v)
        print(f"[{'✅' if v else '❌'}] {name}: " + json.dumps(
            {k: (vv if not isinstance(vv, list) else vv[:3]) for k, vv in c.items() if k != "通过"},
            ensure_ascii=False)[:300])
    print("\n证据：", art / "m2_rebuild.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
