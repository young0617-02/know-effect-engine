"""Gate 0 ③：混合检索相对单一稠密的增益（消融对照）。

四组对照：Dense-only / Sparse-only / Dense+Sparse+RRF / +Rerank。
除检索配置外，其余变量（同一 jieba collection、同一评测集、同一 top_k、同一常驻策略）全部锁死，
保证「增益可归因」。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import aggregate_qa_metrics, latency_report  # noqa: E402
from kee.gate0.common import env_snapshot, md_table, verdict, write_json, write_md  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

CONFIGS = [
    ("dense_only", {"mode": "dense", "use_rerank": False}),
    ("sparse_only", {"mode": "sparse", "use_rerank": False}),
    ("hybrid_rrf", {"mode": "hybrid", "use_rerank": False}),
    ("hybrid_rrf_rerank", {"mode": "hybrid", "use_rerank": True}),
]


def run_config(name: str, kwargs: dict, questions: list[dict], store: KbStore, sched: ResidencyScheduler, top_k: int) -> dict[str, Any]:
    retriever = HybridRetriever(store, scheduler=sched, cache=None, **kwargs)
    ctx = AccessContext(role="knowledge_owner")
    rows, lat, per_cat, degraded = [], [], {}, []
    for q in questions:
        if q["expected_behavior"] != "answer" or not q["gold_chunk_ids"]:
            continue
        res = retriever.retrieve(q["question"], ctx, top_k=top_k)
        ranked = [c.chunk_id for c in res.chunks]
        rows.append({"ranked_ids": ranked, "gold_ids": set(q["gold_chunk_ids"])})
        lat.append(res.timings_ms.get("retrieval_total_ms", 0.0))
        if res.degraded:
            degraded.append({"qid": q["qid"], "degraded": res.degraded})
        per_cat.setdefault(q["category"], []).append({"ranked_ids": ranked, "gold_ids": set(q["gold_chunk_ids"])})
    out = aggregate_qa_metrics(rows, k=top_k)
    out["latency_ms"] = latency_report(lat)
    out["by_category"] = {cat: aggregate_qa_metrics(v, k=top_k) for cat, v in per_cat.items()}
    out["degraded_events"] = degraded
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--collection", default="kb_bm25_jieba")
    args = ap.parse_args()

    pg = AuthoritativeStore()
    ev = build_eval_set(pg, write_files=False)
    questions = ev["questions"]
    store = KbStore(args.collection, "jieba")
    sched = ResidencyScheduler(policy="co_resident")

    report: dict[str, Any] = {
        "env": env_snapshot(),
        "collection": args.collection,
        "top_k": args.top_k,
        "n_questions_gradable": sum(1 for q in questions if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]),
        "chunk_count": store.count(),
        "configs": {},
    }
    with sched.use("embed"):
        pass  # 预热：让两组对照的第一次查询不背冷启动的账
    with sched.use("rerank"):
        pass

    for name, kwargs in CONFIGS:
        report["configs"][name] = run_config(name, kwargs, questions, store, sched, args.top_k)

    # 复现性：同一配置连跑两次，排序完全一致才算「可复现」
    rep = HybridRetriever(store, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    ctx = AccessContext(role="knowledge_owner")
    sample = [q for q in questions if q["expected_behavior"] == "answer"][:8]
    a = [tuple(c.chunk_id for c in rep.retrieve(q["question"], ctx).chunks) for q in sample]
    b = [tuple(c.chunk_id for c in rep.retrieve(q["question"], ctx).chunks) for q in sample]
    report["reproducibility"] = {"n": len(sample), "identical_ordering": a == b}

    write_json("g03_ablation.json", report)

    dense = report["configs"]["dense_only"]
    sparse = report["configs"]["sparse_only"]
    hyb = report["configs"]["hybrid_rrf"]
    hyb_r = report["configs"]["hybrid_rrf_rerank"]
    best = hyb_r
    gain_rrf = round((hyb.get("recall@5") or 0) - (dense.get("recall@5") or 0), 4)
    gain_full = round((best.get("recall@5") or 0) - (dense.get("recall@5") or 0), 4)
    rows = [
        [
            name,
            report["configs"][name].get("recall@5"),
            report["configs"][name].get("hit@5"),
            report["configs"][name].get("precision@5"),
            report["configs"][name].get("mrr"),
            report["configs"][name].get("zero_recall_rate"),
            report["configs"][name]["latency_ms"].get("p95"),
            round((report["configs"][name].get("recall@5") or 0) - (dense.get("recall@5") or 0), 4),
        ]
        for name, _ in CONFIGS
    ]
    cat_rows = [
        [
            cat,
            *(report["configs"][n]["by_category"].get(cat, {}).get(f"recall@{args.top_k}") for n, _ in CONFIGS),
        ]
        for cat in ("术语", "编号", "方案名", "口语化", "跨文档")
    ]

    passed = gain_full > 0.05 and (best.get("recall@5") or 0) >= 0.85
    partial = gain_full > 0.05 and (best.get("recall@5") or 0) < 0.85

    lines = [
        "# Gate 0 ③ 检索消融对照报告（Dense / Sparse / 混合 / +Rerank）",
        "",
        f"- 验证时间：{report['env']['ts']}｜集合 {args.collection}（jieba 分析器）｜子块 {report['chunk_count']}｜可判分题 {report['n_questions_gradable']}",
        f"- 锁定变量：同一评测集、同一 collection、top_k={args.top_k}、嵌入/重排同卡常驻、无缓存",
        "",
        "## 总表",
        "",
        *md_table(
            ["配置", f"Recall@{args.top_k}", "hit@5", f"P@{args.top_k}", "MRR", "零召回率", "P95 检索 ms", "ΔRecall@5 vs Dense"],
            rows,
        ),
        "",
        "## 分类别 Recall@5（术语/编号/方案名是稀疏路的靶心）",
        "",
        *md_table(["类别", "Dense", "Sparse", "Hybrid+RRF", "Hybrid+RRF+Rerank"], cat_rows),
        "",
        "## 归因",
        "",
        f"- 加 RRF 融合（Dense→Hybrid）：Recall@5 {dense.get('recall@5')} → {hyb.get('recall@5')}，Δ={gain_rrf}",
        f"- 再加 Rerank（Hybrid→Hybrid+Rerank）：{hyb.get('recall@5')} → {best.get('recall@5')}，Δ={round(gain_full - gain_rrf, 4)}",
        f"- 合计相对单一稠密：ΔRecall@5={gain_full}；MRR {dense.get('mrr')} → {best.get('mrr')}",
        f"- 排序可复现性：连跑 8 题顺序一致 = {report['reproducibility']['identical_ordering']}",
        "",
        "## 默认检索参数基线（③出口）",
        "",
        "- dense_top_k = 20，sparse_top_k = 20，RRF k = 60，rerank 候选 20 → Top-5。",
        "- 稠密索引 FLAT/COSINE（语料规模下取精确检索，ANN 分层留到规模化阶段）。",
        "- 生产默认配置：`mode=hybrid + rerank=True`。",
        "",
        verdict(
            "③",
            "pass" if passed else ("partial" if partial else "fail"),
            f"混合(+Rerank) Recall@5={best.get('recall@5')}，Dense={dense.get('recall@5')}，Δ={gain_full}；MRR={best.get('mrr')}",
            ""
            if passed
            else f"Recall@5 距 85% 门禁还差 {round(0.85 - (best.get('recall@5') or 0), 4)}；自建语料下绝对值受 gold 粒度与语料规模影响，见「已知边界」",
        ),
        "",
        "## 已知边界",
        "",
        "- 自建语料 + 构造式 gold（章节级）：语料只有数百子块、且每题 gold 可能含 2~4 个子块，Recall@5 的分母偏大，绝对值天然偏保守；"
        "跨配置之间的**相对差**才是本项的结论载体。",
        "- 评测集为单人（构造时）标注，第二人复核表已生成待填，Kappa 未出。",
    ]
    p = write_md("g03_ablation.md", lines)
    print("\n".join(lines))
    print("\nMD:", p)
    return 0 if (passed or partial) else 1


if __name__ == "__main__":
    raise SystemExit(main())
