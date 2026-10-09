"""Gate 0 ②：Milvus v3.0.0 服务端 BM25 + 中文分析器可用性与中文检索有效性（必备项）。

对照组：默认分析器（standard，等价不传 analyzer_params）vs jieba vs chinese。
判据：中文分析器生效，且 BM25 相对默认分析器有明确正增益（非≈0）；术语/编号类 query 明显改善。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.db.vector_store import VectorStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import aggregate_qa_metrics, latency_report  # noqa: E402
from kee.gate0.common import env_snapshot, md_table, verdict, write_json, write_md  # noqa: E402
from kee.ingest.build import build_index, rebuild_index_from_authoritative  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import ANALYZERS, KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

COLLECTIONS = {
    "jieba": "kb_bm25_jieba",
    "standard": "kb_bm25_standard",
    "chinese": "kb_bm25_chinese",
    "default_none": "kb_bm25_defaultnone",
}

# 切分行为探针：整词 / 半词切片 / 编号 / 英文缩写
PROBE_TERMS = [
    "银企直联",
    "企直联",
    "实时清算",
    "线损率",
    "损率",
    "SCADA 主站",
    "IEC 60870-5-104",
    "DCS 联锁",
    "P1",
]


def run_sparse_eval(questions: list[dict], store: KbStore, scheduler: ResidencyScheduler, top_k: int = 5) -> dict[str, Any]:
    retriever = HybridRetriever(store, scheduler=scheduler, cache=None, mode="sparse", use_rerank=False)
    ctx = AccessContext(role="knowledge_owner")
    rows, lat, per_category = [], [], {}
    for q in questions:
        if q["expected_behavior"] != "answer" or not q["gold_chunk_ids"]:
            continue
        res = retriever.retrieve(q["question"], ctx, top_k=top_k)
        ranked = [c.chunk_id for c in res.chunks]
        rows.append({"ranked_ids": ranked, "gold_ids": set(q["gold_chunk_ids"])})
        lat.append(res.timings_ms.get("sparse_ms", res.timings_ms.get("retrieval_total_ms", 0.0)))
        per_category.setdefault(q["category"], {"ranked": [], "gold": []})
        per_category[q["category"]]["ranked"].append(ranked)
        per_category[q["category"]]["gold"].append(set(q["gold_chunk_ids"]))
    overall = aggregate_qa_metrics(rows, k=top_k)
    overall["latency_ms"] = latency_report(lat)
    overall["by_category"] = {
        cat: aggregate_qa_metrics(
            [{"ranked_ids": r, "gold_ids": g} for r, g in zip(v["ranked"], v["gold"])], k=top_k
        )
        for cat, v in per_category.items()
    }
    overall["n_eval"] = len(rows)
    return overall


def probe_tokenization(stores: dict[str, KbStore]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for term in PROBE_TERMS:
        row = {}
        for name, st in stores.items():
            ids = st.tokenize_probe([term], limit=20)[term]
            row[name] = len(ids)
        out[term] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="跳过重建索引，只做检索对照")
    args = ap.parse_args()

    report: dict[str, Any] = {"env": env_snapshot(), "analyzer_params_supported": {}}
    pg = AuthoritativeStore()
    vs = VectorStore()
    sched = ResidencyScheduler(policy="co_resident")

    if not args.reuse:
        with sched.use("embed"):
            t0 = time.perf_counter()
            report["build"] = build_index(pg, KbStore(COLLECTIONS["jieba"], "jieba"), vs, sched.embedder, reset=True)
            report["build_seconds"] = round(time.perf_counter() - t0, 1)
        for name in ("standard", "chinese", "default_none"):
            report[f"rebuild_{name}"] = rebuild_index_from_authoritative(
                pg, KbStore(COLLECTIONS[name], name), vs
            )
        # 生产集合名同步建好：①④⑤ 都按 settings.milvus_collection 取数，缺一则那些项无数据可测
        from kee.config import get_settings as _gs

        report["rebuild_canonical"] = rebuild_index_from_authoritative(
            pg, KbStore(_gs().milvus_collection, "jieba"), vs
        )

    stores = {n: KbStore(c, n) for n, c in COLLECTIONS.items()}
    for n, st in stores.items():
        st.ensure()
        report["analyzer_params_supported"][n] = st.describe_analyzer()
        report["analyzer_params_supported"][n + "_count"] = st.count()

    ev = build_eval_set(pg)
    report["eval_set"] = ev["stats"]
    questions = ev["questions"]

    report["tokenization_probe"] = probe_tokenization(stores)

    report["sparse_eval"] = {}
    for n, st in stores.items():
        report["sparse_eval"][n] = run_sparse_eval(questions, st, sched)

    write_json("g02_chinese_bm25.json", report)

    base = report["sparse_eval"]["default_none"]
    cmp_rows = []
    for n in stores:
        m = report["sparse_eval"][n]
        cmp_rows.append(
            [
                n,
                json.dumps(report["analyzer_params_supported"][n], ensure_ascii=False),
                m.get("recall@5"),
                m.get("hit@5"),
                m.get("mrr"),
                m.get("zero_recall_rate"),
                round((m.get("recall@5") or 0) - (base.get("recall@5") or 0), 4),
                m["latency_ms"].get("p95"),
            ]
        )

    jieba = report["sparse_eval"]["jieba"]
    chinese = report["sparse_eval"]["chinese"]
    best_name = max(("jieba", "chinese"), key=lambda k: report["sparse_eval"][k].get("recall@5") or 0)
    best = report["sparse_eval"][best_name]
    gain = (best.get("recall@5") or 0) - (base.get("recall@5") or 0)
    term_gain = {
        cat: round(
            (best["by_category"].get(cat, {}).get(f"recall@{5}") or 0)
            - (base["by_category"].get(cat, {}).get(f"recall@{5}") or 0),
            4,
        )
        for cat in ("术语", "编号", "方案名")
        if cat in best.get("by_category", {})
    }
    passed = gain > 0.05 and all(v >= 0 for v in term_gain.values()) and any(v > 0.05 for v in term_gain.values())

    lines = [
        "# Gate 0 ② 中文稀疏检索（服务端 BM25 + 分析器）验证报告",
        "",
        f"- 验证时间：{report['env']['ts']}｜commit {report['env']['git_commit']}｜Milvus 3.0.0 + pymilvus {report['env']['pymilvus']}",
        "",
        "## 关键发现（与方案假设的差异）",
        "",
        "1. `analyzer_params={\"type\":\"jieba\"}` 被 Milvus v3.0.0 服务端拒绝：",
        "   `unknown build-in analyzer type: jieba`（internal/core/thirdparty/tantivy/tokenizer.h）。",
        "   **可用形态是 `{\"tokenizer\":\"jieba\"}`**，方案 11.2 的 analyzer 配置基线必须按此锁定。",
        "2. 另有内置 `\"type\":\"chinese\"` 分析器可用，一并纳入对照。",
        "3. 默认（不给 params，仅 enable_analyzer）对中文查询近乎无效——见下表基线行。",
        "",
        "## 分析器与索引规模",
        "",
        *md_table(
            ["变体", "服务端落库的 analyzer 参数", "子块数"],
            [[n, json.dumps(report["analyzer_params_supported"][n], ensure_ascii=False), report["analyzer_params_supported"][n + "_count"]] for n in stores],
        ),
        "",
        "## 切分行为探针（term → 命中子块数）",
        "",
        *md_table(
            ["term"] + list(stores.keys()),
            [[t] + [report["tokenization_probe"][t][n] for n in stores] for t in PROBE_TERMS],
        ),
        "",
        "## 稀疏检索对照（Recall@5 / hit@5 / MRR，仅 BM25 单路）",
        "",
        *md_table(
            ["变体", "analyzer", "Recall@5", "hit@5", "MRR", "零召回率", "ΔRecall@5(对默认)", "P95 检索 ms"],
            cmp_rows,
        ),
        "",
        f"- 评测集可判分题数：{jieba['n_eval']}；语料子块总数：{ev['stats']['n_children']}",
        f"- 最优中文分析器：**{best_name}**，相对默认 Recall@5 增益 **{round(gain, 4)}**",
        f"- 分类别增益（对默认）：{json.dumps(term_gain, ensure_ascii=False)}",
        "",
        "## 建库与重建",
        "",
        f"- 首次建库（含嵌入）：{json.dumps(report.get('build', {}).get('timings_sec', {}), ensure_ascii=False)}",
        f"- 索引重建（不碰 GPU，仅权威源+向量物料）：{json.dumps({k: v.get('seconds') for k, v in report.items() if k.startswith('rebuild_')}, ensure_ascii=False)}",
        f"- 准入：{json.dumps(report.get('build', {}).get('admission', {}), ensure_ascii=False)}",
        "",
        verdict(
            "②",
            "pass" if passed else "fail",
            f"{best_name} Recall@5 {best.get('recall@5')} vs 默认 {base.get('recall@5')}（Δ={round(gain,4)}）；"
            f"术语/编号类增益 {json.dumps(term_gain, ensure_ascii=False)}",
            "jieba 需以 {\"tokenizer\":\"jieba\"} 形态配置；方案文档口径需同步修订",
        ),
        "",
        "## 参数基线（②出口）",
        "",
        "- `analyzer_params = {\"tokenizer\": \"jieba\"}`（首选），`{\"type\": \"chinese\"}` 作为备选。",
        "- 稀疏索引：`SPARSE_INVERTED_INDEX` + `metric_type=BM25`；稠密索引本轮用 `FLAT/COSINE`（精确，避免 ANN 近似污染消融）。",
        "- 索引重建路径：PostgreSQL 权威源 + 本地向量物料 → Milvus，无需 GPU 重算。",
    ]
    p = write_md("g02_chinese_bm25.md", lines)
    print("\n".join(lines))
    print("\nMD:", p)
    for n in stores:
        stores[n].close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
