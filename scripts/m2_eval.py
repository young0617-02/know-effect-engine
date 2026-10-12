"""M2 检索评测作业：固定基线配置 + 诊断网格 + 回归比对，产出可复跑的评测基线。

为什么要 dev/test 拆分而不是「跑一遍取最好」：
  Gate 0 ③ 的 Recall@5=0.822 是在全部 58 道可判题上出的数，如果我在这里把参数网格挨个跑一遍、
  挑最高的那格当结论，那就是**在同一份测试集上调参再报它的分数**，数字会系统性虚高。
  所以：参数网格只跑 dev 半区做诊断，**test 半区只在预先声明的基线配置上跑一次**，
  门禁判定只认基线配置的 test 值。网格里最好的那一格要转正，必须改基线声明后重跑，
  而不是直接引用网格数字。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import aggregate_qa_metrics, hit_at_k, latency_report, recall_at_k  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402
from kee.retrieval.rewrite import QueryRewriter  # noqa: E402

DEV_FRACTION_MOD = 10  # qid 哈希取模：<4 为 dev，其余 test
BASELINE_NAME = "baseline_declared"
# 预先声明的基线配置：与 settings 里的检索参数基线一致，改动这里等于改门禁口径，需在报告里留痕
BASELINE_CFG: dict[str, Any] = {
    "mode": "hybrid",
    "use_rerank": True,
    "rerank_candidates": 20,
    "rewrite_mode": "off",
}
# OFAT 诊断网格：每次只动一个因子，其余回到基线
GRID: list[tuple[str, dict[str, Any]]] = [
    ("dense_only", {"mode": "dense"}),
    ("sparse_only", {"mode": "sparse", "use_rerank": False}),
    ("hybrid_no_rerank", {"use_rerank": False}),
    ("rrf_k_20", {"rrf_k": 20}),
    ("rrf_k_120", {"rrf_k": 120}),
    ("candidates_50", {"rerank_candidates": 50}),
    ("topk_wide_40", {"dense_top_k": 40, "sparse_top_k": 40}),
    ("rewrite_sparse", {"rewrite_mode": "sparse"}),
    ("rewrite_extra_leg", {"rewrite_mode": "extra_leg"}),
]
GATE_HIT_AT_5 = 0.95
GATE_RECALL_AT_5 = 0.85
GATE_MRR = 0.75
GATE_P95_MS = 800.0


def split_of(qid: str) -> str:
    return "dev" if zlib.crc32(qid.encode("utf-8")) % DEV_FRACTION_MOD < 4 else "test"


def run_config(
    label: str,
    cfg: dict[str, Any],
    questions: list[dict],
    store: KbStore,
    sched: ResidencyScheduler,
    top_k: int,
    rewriter_cache=None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mode": cfg.get("mode", "hybrid"),
        "use_rerank": cfg.get("use_rerank", True),
        "rerank_candidates": cfg.get("rerank_candidates", 20),
        "rewrite_mode": cfg.get("rewrite_mode", "off"),
        "rrf_k": cfg.get("rrf_k"),
        "dense_top_k": cfg.get("dense_top_k"),
        "sparse_top_k": cfg.get("sparse_top_k"),
    }
    rewriter = None
    if kwargs["rewrite_mode"] != "off":
        rewriter = QueryRewriter(cache=rewriter_cache, always=False)
    retriever = HybridRetriever(store, scheduler=sched, cache=None, rewriter=rewriter, **kwargs)
    ctx = AccessContext(role="knowledge_owner")
    rows: list[dict] = []
    rows_strict: list[dict] = []
    lat: list[float] = []
    per_q: list[dict] = []
    per_cat: dict[str, list[dict]] = {}
    per_split: dict[str, list[dict]] = {}
    degraded: list[dict] = []
    rewrites: list[dict] = []
    for q in questions:
        if q["expected_behavior"] != "answer" or not q["gold_chunk_ids"]:
            continue
        t0 = time.perf_counter()
        res = retriever.retrieve(q["question"], ctx, top_k=top_k)
        ranked = [c.chunk_id for c in res.chunks]
        row = {"ranked_ids": ranked, "gold_ids": set(q["gold_chunk_ids"])}
        rows.append(row)
        strict = set(q.get("gold_strict_ids") or [])
        if strict:
            rows_strict.append({"ranked_ids": ranked, "gold_ids": strict})
        per_q.append(
            {
                "qid": q["qid"],
                "recall": recall_at_k(ranked, row["gold_ids"], top_k),
                "recall_strict": recall_at_k(ranked, strict, top_k) if strict else None,
                "hit": hit_at_k(ranked, row["gold_ids"], top_k),
                "n_gold": len(row["gold_ids"]),
                "category": q["category"],
                "split": split_of(q["qid"]),
            }
        )
        per_cat.setdefault(q["category"], []).append(row)
        per_split.setdefault(split_of(q["qid"]), []).append(row)
        lat.append(res.timings_ms.get("retrieval_total_ms", 0.0))
        if res.degraded:
            degraded.append({"qid": q["qid"], "degraded": res.degraded})
        if res.rewritten:
            rewrites.append({"qid": q["qid"], "原": q["question"], "改写": res.rewritten, "ms": res.rewrite_ms})
    out = aggregate_qa_metrics(rows, k=top_k)
    # 两套 gold 口径同时报：section 级（章节内所有子块都算相关）会把「已经答对但没取全章节」
    # 记成漏召，strict 级（正文真含关键词的子块）才对应「证据是否被取到」。只报前者会误导。
    out["strict_gold"] = aggregate_qa_metrics(rows_strict, k=top_k) if rows_strict else {"n_graded": 0}
    out["latency_ms"] = latency_report(lat)
    out["latencies_ms"] = [round(x, 1) for x in lat]
    out["per_question"] = per_q
    lo, hi = bootstrap_ci([r["recall"] for r in per_q])
    out[f"recall@{top_k}_ci95"] = [lo, hi]
    out["n"] = len(rows)
    out["by_category"] = {c: aggregate_qa_metrics(v, k=top_k) for c, v in per_cat.items()}
    out["by_split"] = {c: {**aggregate_qa_metrics(v, k=top_k), "n": len(v)} for c, v in per_split.items()}
    out["degraded_events"] = degraded[:20]
    out["degraded_count"] = len(degraded)
    out["rewrites"] = rewrites[:10]
    out["config"] = cfg
    out["label"] = label
    return out


def metric(res: dict, name: str, k: int, default=0.0):
    return res.get(f"{name}@{k}", res.get(f"{name}@5", default))


def bootstrap_ci(vals: list[float], n: int = 2000, seed: int = 20261010) -> tuple[float, float]:
    """按题 bootstrap 的 95% 置信区间。Recall@5 是逐题召回率的均值（可取小数），
    不是二项比例，所以不能用 Wilson；固定 seed 保证同一份数据出同一个区间，可复跑核对。"""
    if not vals:
        return (0.0, 0.0)
    import random

    rng = random.Random(seed)
    m = len(vals)
    means = sorted(sum(rng.choices(vals, k=m)) / m for _ in range(n))
    return (round(means[max(0, int(0.025 * n))], 4), round(means[min(n - 1, int(0.975 * n))], 4))


def _ollama_resident() -> list[str]:
    """记录测量时显存里到底有没有 7B：8GB 卡上「有/无 7B 常驻」是两套延迟分布，不能混着报。"""
    import httpx

    try:
        r = httpx.get(get_settings().ollama_url.rstrip("/") + "/api/ps", timeout=5.0)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return [f"unreachable:{type(exc).__name__}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--skip-grid", action="store_true", help="只跑基线（回归比对用）")
    ap.add_argument("--fail-on-gold-defect", action="store_true",
                    help="关键词命中面 > 3 段且无豁免时直接拒绝跑（改完尺子后的自检用）")
    ap.add_argument("--prev", default=None,
                    help="回归对照用的上一轮读数路径（默认 var/gate0/m2_eval_prev.json；"
                         "换了尺子要指到改前留档那份，由指纹检查判定可不可比）")
    ap.add_argument("--update-prev", action="store_true",
                    help="跑完把本轮读数写成对照基线（**只有确认这轮是新尺子的基准时才用**）")
    args = ap.parse_args()

    s = get_settings()
    coll = args.collection or s.milvus_collection
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    built = build_eval_set(pg, write_files=False, fail_on_gold_defect=args.fail_on_gold_defect)
    questions = built["questions"]
    dts = built["stats"]["gold_defects"]
    if dts:
        print(f"[gold 缺陷检测] 关键词命中面 > {dts[0]['limit']} 段的题 {len(dts)} 道：" +
              "、".join(f"{d['qid']}({d['n_strict']})" + ("[已豁免]" if d["waived"] else "[未豁免]") for d in dts))
        for d in dts:
            if d["waived"]:
                print(f"    豁免理由 {d['qid']}：{d['waiver_reason'][:70]}")
    dev = [q for q in questions if split_of(q["qid"]) == "dev"]
    test = [q for q in questions if split_of(q["qid"]) == "test"]
    store = KbStore(coll, "jieba")
    sched = ResidencyScheduler(policy="co_resident")
    print(f"[数据] 题 {len(questions)} → dev {len(dev)} / test {len(test)}；集合 {coll} 块 {store.count()}")

    # 真预热：跑一次完整检索，把两个模型的冷加载（实测各 6~13s）挡在计时之外。
    # 只 with sched.use(...) 里 pass 一把是假预热——模型是懒加载的，第一道题照样背冷启的账。
    warm = HybridRetriever(store, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    tw = time.perf_counter()
    warm.retrieve("预热查询：配电自动化终端心跳超时阈值", AccessContext(role="knowledge_owner"), top_k=5)
    print(f"[预热] 首查 {round((time.perf_counter() - tw) * 1000)}ms（含模型冷加载），不计入延迟统计")

    t0 = time.perf_counter()
    base_full = run_config(BASELINE_NAME, BASELINE_CFG, questions, store, sched, args.top_k)
    base_test = run_config(BASELINE_NAME, BASELINE_CFG, test, store, sched, args.top_k)
    base_dev = run_config(BASELINE_NAME, BASELINE_CFG, dev, store, sched, args.top_k)
    k = args.top_k
    print(f"[基线] full Recall@{k}={metric(base_full, 'recall', k)} "
          f"CI95={base_full[f'recall@{k}_ci95']} n={base_full['n']} "
          f"P95={base_full['latency_ms']['p95']}ms（{round(time.perf_counter() - t0, 1)}s）")
    print(f"[基线] 分半 dev={metric(base_dev, 'recall', k)}(n={base_dev['n']}) "
          f"test={metric(base_test, 'recall', k)}(n={base_test['n']}) —— 门禁只认 full，分半只用于调参隔离")

    grid: dict[str, Any] = {}
    if not args.skip_grid:
        for name, delta in GRID:
            cfg = {**BASELINE_CFG, **delta}
            t1 = time.perf_counter()
            grid[name] = run_config(name, cfg, dev, store, sched, args.top_k)
            print(f"[网格] {name:20s} dev Recall@{k}={metric(grid[name], 'recall', k)} "
                  f"MRR={grid[name].get('mrr')} P95={grid[name]['latency_ms']['p95']}ms "
                  f"({round(time.perf_counter() - t1, 1)}s)")

    # 尺子指纹：题量、可判行数、gold 块总数。任一项变了，与 prev 的差就**不是回归读数**，
    # 而是两把不同尺子上的两个数——过去几批 gold 改动后 prev 指针一直没动，
    # 于是一行「delta=+0.0179 / regressed=false」其实是在跟四批之前、70 题那把尺子比。
    ruler = {"n_questions": len(questions), "n_graded": base_full.get("n"),
             "gold_section_total": sum(r.get("n_gold", 0) for r in base_full.get("per_question", [])),
             # 题面哈希也要进指纹：改题干等于换了查询串，检索结果本来就该变——
             # 只看题量与 gold 数的话，一次纯题面改动会被读成「同尺子可比」，那是假的。
             "question_hash": zlib.crc32("|".join(f"{q['qid']}::{q['question']}"
                                                  for q in sorted(questions, key=lambda x: x["qid"]))
                                         .encode("utf-8"))}
    prev_path = Path(args.prev) if args.prev else art / "m2_eval_prev.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None
    regression = None
    if prev:
        p_full = prev["baseline_full"] or {}
        p_ruler = prev.get("ruler") or {}
        comparable = p_ruler == ruler and bool(p_ruler)
        reason = None
        if not p_ruler:
            reason = "prev 没有尺子指纹（那是加指纹之前留下的快照），不可比"
        elif not comparable:
            reason = f"尺子变了：prev {p_ruler} ≠ now {ruler}"
        d = round(metric(base_full, "recall", k) - metric(p_full, "recall", k), 4)
        # 回归判定用 strict 那把尺——section 刚被证实在注水，拿它判回归等于用错温度测温。
        # 上一版这里 `regressed` 实际算的是 section 差，而注释与 note 都写「以 strict 为准」：
        # 现在判定与文字对齐，section 差留作对照。
        d_strict = round(metric(base_full.get("strict_gold") or {}, "recall", k)
                         - metric((p_full or {}).get("strict_gold") or {}, "recall", k), 4)
        regression = {
            "prev_path": prev_path.name, "comparable": comparable, "reason": reason,
            "ruler_now": ruler, "ruler_prev": p_ruler or None,
            "prev": metric(p_full, "recall", k),
            "now": metric(base_full, "recall", k),
            "delta_section": d if comparable else None,
            "prev_strict": metric((p_full or {}).get("strict_gold") or {}, "recall", k),
            "now_strict": metric(base_full.get("strict_gold") or {}, "recall", k),
            "delta_strict": d_strict if comparable else None,
            "regressed": (d_strict < -0.02) if comparable else None,
            "note": ("同尺子对照，判定看 strict 差" if comparable
                     else f"不可比：{reason} ⇒ 不给 delta、不给 regressed=false"),
        }
    # 门禁一律看 full：基线配置是**预先声明**、没在数据上调过，所以全样本判定是干净的；
    # 反过来，若哪天把网格里赢的那格转正，就必须先改这里的声明再重跑，不能直接引用网格数字。
    #
    # 判定口径按 2026-10-10 的 M2 裁决第 1 项（《项目定义》V1.3 /《技术方案》V1.0.3）：
    #   主判定 = Hit@5 ≥ 95% + **strict gold** Recall@5 ≥ 85%；section gold Recall@5 只是报告值。
    # 此前这里实现错了——把 section 读数喂进 GATE_RECALL_AT_5 且根本不查 Hit@5，
    # 于是门禁判定一直在用一个「已被独立第二人证成为注水尺子」的口径（详见 reports/m2 第十节）。
    # 阈值数值一个都没动，改的是「它测哪把尺子」。
    strict_full = base_full.get("strict_gold") or {}
    gates = {
        f"Hit@{k}(strict)≥{GATE_HIT_AT_5}": metric(strict_full, "hit", k) >= GATE_HIT_AT_5,
        f"Recall@{k}(strict)≥{GATE_RECALL_AT_5}": metric(strict_full, "recall", k) >= GATE_RECALL_AT_5,
        f"MRR≥{GATE_MRR}": base_full.get("mrr", 0) >= GATE_MRR,
        f"P95检索≤{GATE_P95_MS:.0f}ms": base_full["latency_ms"]["p95"] <= GATE_P95_MS,
    }
    # section 口径降为报告值，按裁决必须带 n 与 95% CI；两份 Hit@5 在本语料上数值相同（0.9828），
    # 所以「Hit 用哪把尺子」这个歧义在本机上是空歧义——仍按 strict 判，并把两把都报出来。
    reported = {
        "section_recall@5": metric(base_full, "recall", k),
        "section_recall@5_ci95": base_full.get(f"recall@{k}_ci95"),
        "section_hit@5": metric(base_full, "hit", k),
        "strict_hit@5": metric(strict_full, "hit", k),
        "n": base_full.get("n"),
        "gate_basis": "Hit@5(strict) + Recall@5(strict)，section 仅报告——依 2026-10-10 M2 裁决第 1 项",
    }
    summ = sched.summary()
    from kee.models.gpu import vram_used_mib

    out = {
        "env": {"collection": coll, "chunks": store.count(), "top_k": args.top_k,
                "split_mod": DEV_FRACTION_MOD, "baseline_cfg": BASELINE_CFG,
                "vram_used_mib_at_end": round(vram_used_mib(), 1),
                "ollama_resident": _ollama_resident()},
        "model_loads": {f: summ[f] for f in ("policy", "device", "n_cold_loads", "cold_load_sec_max",
                                            "embed_load_sec", "rerank_load_sec")},
        "baseline_full": base_full,
        "baseline_test": base_test,
        "baseline_dev": base_dev,
        "grid_dev": grid,
        "gates": gates,
        "reported_not_gated": reported,
        "gates_passed": all(gates.values()),
        "ruler": ruler,
        "gold_defects": built["stats"]["gold_defects"],
        "regression": regression,
        "best_grid_cell_is_not_evidence": "网格只在 dev 上跑，用于诊断；门禁只认基线配置的全样本值",
    }
    art.mkdir(parents=True, exist_ok=True)
    (art / "m2_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.update_prev:
        # 排在落盘之后：要留作基准的是**本轮**读数，而不是上一轮那份
        shutil.copyfile(art / "m2_eval.json", prev_path)
        print(f"[prev 指针] 本轮读数已设为对照基线：{prev_path.name}")
    store.close()
    print("\n判定:", json.dumps(gates, ensure_ascii=False), "→", "通过" if out["gates_passed"] else "不通过")
    if regression:
        print("回归:", json.dumps(regression, ensure_ascii=False))
    print("证据：", art / "m2_eval.json")
    if regression and regression["regressed"] is True:
        return 2
    return 0 if out["gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
