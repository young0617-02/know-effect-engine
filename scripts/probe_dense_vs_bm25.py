"""三条问句上的双腿对照：稠密(BGE-M3) vs 服务端 BM25，各取 Top-5 并记录名次差异。

问句类型固定为「纯编号 / 口语化 / 编号+口语混合」，用来把两腿各自的失效面照出来。
构造参数照 `g03_ablation.py` 的 `dense_only` / `sparse_only` 两格：`use_rerank=False`、
`cache=None`、`rewrite_mode="off"`、同一 collection、同一常驻策略——这是唯一能让每腿
Top-5 保持为该腿纯排名的配置（加了重排就不是腿的排序了）。

只比名次不比分数：稠密腿是 COSINE ∈ [-1,1]，BM25 是原始量纲（几十）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

# gold 一律取自评测集里已核对过的题，不为本实验新造标注
GOLD_Q018 = [
    "fin-ifspec-prj-fin-2024-017@v1#1.2#c1",
    "fin-ifspec-prj-fin-2024-017@v1#1.2#c2",
]

QUERIES: list[dict[str, Any]] = [
    {
        "key": "Q1",
        "type": "纯编号",
        "query": "IF-FIN-0012",
        "gold": GOLD_Q018,
        "gold_from": "q-018（同一接口的金额上限题）",
    },
    {
        "key": "Q2",
        "type": "口语化",
        "query": "操作员连着拒了几条系统建议，平台会把它这条回路晾多久？",
        "gold": ["chm-ifspec-prj-chm-2024-003@v1#2.1#c1"],
        "gold_from": "q-050",
    },
    {
        "key": "Q3",
        "type": "混合",
        "query": "IF-FIN-0012 那笔汇款超了上限，钱还能不能走，得怎么接着办？",
        "gold": GOLD_Q018,
        "gold_from": "q-018",
    },
]

LEGS = ("dense", "sparse")
WINDOW = 20  # 腿内取前 20 名次，Top-5 之外的名次用来解释「谁把谁挤掉了」

# 对照组：`--control` 才跑。构造目的是把词面重合压到接近零，用来看两腿的失效机制；
# 它不在评测集里，所以**不能当作「口语化题普遍表现」的证据**——那件事的证据是评测集 14 道口语化题
# 与 g03 的分类别 Recall。gold 仍用关键词命中规则在语料里逐字核过。
CONTROL_QUERIES: list[dict[str, Any]] = [
    {
        "key": "C1",
        "type": "口语化-低词面重合（构造对照）",
        "query": "厂里那套自动给数值的玩意儿，被人连着压了三回不搭理，它自己会消停多长时间？",
        "gold": ["chm-ifspec-prj-chm-2024-003@v1#2.1#c1"],
        "gold_from": "q-050 同一块（答案：同一回路连续 3 条建议被操作员拒绝后暂停 30 分钟）",
    },
]


def run_leg(retriever: HybridRetriever, ctx: AccessContext, query: str) -> dict[str, Any]:
    res = retriever.retrieve(query, ctx, top_k=WINDOW)
    return {
        "ranked": [c.chunk_id for c in res.chunks],
        "scores": [round(c.score, 4) for c in res.chunks],
        "texts": [c.text for c in res.chunks],
        "fields": [c.fields for c in res.chunks],
        "timings_ms": res.timings_ms,
        "degraded": res.degraded,
        "expr": res.expr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_chunks", help="生产集合；g03 当时用的是 kb_bm25_jieba")
    ap.add_argument("--out", default=str(ROOT / "var" / "gate0" / "dense_vs_bm25_probe.json"))
    ap.add_argument("--control", action="store_true", help="只跑构造的低词面重合对照问句，产物须用 --out 另存")
    args = ap.parse_args()
    queries = CONTROL_QUERIES if args.control else QUERIES

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[abort] 产物已存在，不覆盖：{out_path}（要重跑请用 --out 另存）", file=sys.stderr)
        return 2

    store = KbStore(args.collection, "jieba")
    n_chunks = store.count()
    if n_chunks < 100:
        print(f"[abort] 集合 {args.collection} 只有 {n_chunks} 块，索引未就绪", file=sys.stderr)
        return 2

    sched = ResidencyScheduler(policy="co_resident")
    ctx = AccessContext(role="knowledge_owner")
    legs = {
        m: HybridRetriever(
            store, scheduler=sched, cache=None, mode=m, use_rerank=False, rewrite_mode="off"
        )
        for m in LEGS
    }

    with sched.use("embed"):
        pass
    # 真查询预热一次：模型懒加载 6~13s，冷启动会把第一条问句算进 dense_timeout
    warm = run_leg(legs["dense"], ctx, "系统可用性要求")
    print(f"[warmup] dense 腿首查 {warm['timings_ms'].get('retrieval_total_ms')}ms，degraded={warm['degraded']}")

    report: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collection": args.collection,
        "analyzer": "jieba",
        "chunk_count": n_chunks,
        "config": "HybridRetriever(mode=dense|sparse, use_rerank=False, cache=None, rewrite_mode=off), top_k=5",
        "query_set": "control" if args.control else "headline3",
        "warmup": {
            "retrieval_total_ms": warm["timings_ms"].get("retrieval_total_ms"),
            "degraded": warm["degraded"],
        },
        "queries": [],
    }

    for q in queries:
        r = {m: run_leg(legs[m], ctx, q["query"]) for m in LEGS}
        entry: dict[str, Any] = {
            "key": q["key"],
            "type": q["type"],
            "query": q["query"],
            "gold": q["gold"],
            "gold_from": q["gold_from"],
            "legs": {},
        }
        for m in LEGS:
            other = r[LEGS[1 - LEGS.index(m)]]["ranked"]
            top5 = r[m]["ranked"][:5]
            entry["legs"][m] = {
                "top5": [
                    {
                        "rank": i + 1,
                        "chunk_id": cid,
                        "score_raw": r[m]["scores"][i],
                        "gold": cid in q["gold"],
                        "other_leg_rank": (other.index(cid) + 1) if cid in other else None,
                        "doc_id": r[m]["fields"][i].get("doc_id"),
                        "section_no": r[m]["fields"][i].get("section_no"),
                        "table_row": "（表行）" in r[m]["texts"][i],
                        "text_head": r[m]["texts"][i][:80],
                    }
                    for i, cid in enumerate(top5)
                ],
                "window": r[m]["ranked"],
                "gold_ranks": [
                    {"gold_id": g, "rank": (r[m]["ranked"].index(g) + 1) if g in r[m]["ranked"] else None}
                    for g in q["gold"]
                ],
                "timings_ms": r[m]["timings_ms"],
                "degraded": r[m]["degraded"],
            }
        d5, s5 = set(entry["legs"]["dense"]["window"][:5]), set(entry["legs"]["sparse"]["window"][:5])
        entry["overlap"] = {
            "top5_common": sorted(d5 & s5),
            "top5_only_dense": sorted(d5 - s5),
            "top5_only_sparse": sorted(s5 - d5),
            "jaccard_top5": round(len(d5 & s5) / len(d5 | s5), 4) if (d5 | s5) else None,
            "top1_equal": (
                entry["legs"]["dense"]["window"][0] == entry["legs"]["sparse"]["window"][0]
                if d5 and s5
                else None
            ),
        }
        report["queries"].append(entry)

    # 两腿各自窗口内的集合相似度：Top-5 之外还重合多少
    for entry in report["queries"]:
        dw, sw = set(entry["legs"]["dense"]["window"]), set(entry["legs"]["sparse"]["window"])
        entry["overlap"]["jaccard_window20"] = round(len(dw & sw) / len(dw | sw), 4) if (dw | sw) else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for entry in report["queries"]:
        print(f"\n=== {entry['key']} {entry['type']}｜{entry['query']}")
        print(f"    gold: {entry['gold']}（取自 {entry['gold_from']}）")
        for m in LEGS:
            g = entry["legs"][m]
            print(
                f"  [{m}] gold 名次={[(x['gold_id'].split('#')[-2] + '#' + x['gold_id'].split('#')[-1], x['rank']) for x in g['gold_ranks']]} "
                f"degraded={g['degraded']}"
            )
            for h in g["top5"]:
                mark = "GOLD" if h["gold"] else "    "
                othr = f"另一腿#{h['other_leg_rank']}" if h["other_leg_rank"] else "另一腿未进前20"
                print(
                    f"    #{h['rank']} {mark} {othr:>14} {h['score_raw']:>9} "
                    f"{h['chunk_id'].split('@')[0]:<28}#{h['section_no']:<4} {h['text_head'][:34]}"
                )
        ov = entry["overlap"]
        print(
            f"  [对照] Top-5 共有 {len(ov['top5_common'])}｜稠密独占 {len(ov['top5_only_dense'])}｜"
            f"BM25 独占 {len(ov['top5_only_sparse'])}｜Jaccard@5={ov['jaccard_top5']}｜@20={ov['jaccard_window20']}"
        )

    print("\nJSON:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
