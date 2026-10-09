"""Gate 0 ①：8GB 显存下「嵌入/重排按需加载释放 + 本地生成常驻」的分时调度可行性（必备项）。

被试策略：
  time_sharded : 嵌入与重排互斥驻留，用完即释（方案原文口径）
  co_resident  : 嵌入与重排同时常驻，与 7B 生成共存
  cpu_sharded  : 嵌入与重排落 CPU（回退分支①第一档）
每种策略都跑「短 query / 长 query 混合」的真实问答链路（检索+生成），采全局显存、
本进程 torch 显存、OOM 次数、冷启切换耗时、单问答耗时与 P95 端到端。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import latency_report, percentile  # noqa: E402
from kee.gate0.common import env_snapshot, md_table, verdict, write_json, write_md  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.gpu import VramSampler  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa.answer import AnswerEngine  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

POLICIES = ("time_sharded", "co_resident", "cpu_sharded")
FALLBACK_QUERIES = [
    "P1 级故障多久要启动响应",
    "银企直联实时清算方案的 RTO 目标是多少，双活部署下 RPO 又有何要求？",
    "ITSS 事件分级",
    "配网线损治理方案 V2.1 把线损率从多少降到多少，用了哪些手段和验收口径？",
    "FAT 与 SAT",
    "HAZOP 分析在连续重整装置智能优化方案 V1.4 的交付流程里处于哪个环节，输出物是什么？",
]


def pick_queries(n: int) -> list[dict[str, Any]]:
    """长短混合负载：评测集问题普遍 19~44 字（中位 32），没有真短查询，
    因此短 query 用评测集的 gold 术语/编号词组现造，保证「长短混合」这条前提成立。"""
    try:
        ev = build_eval_set(AuthoritativeStore(), write_files=False)
        qs = [
            {"question": q["question"], "category": q["category"], "expected_behavior": q["expected_behavior"]}
            for q in ev["questions"]
            if q["expected_behavior"] in ("answer", "refuse")
        ]
        long_ = [q for q in qs if len(q["question"]) > 12]
        seen: set[str] = set()
        short: list[dict] = []
        for q in ev["questions"]:
            for kw in q.get("gold_keywords", []) or []:
                if 2 <= len(kw) <= 10 and kw not in seen:
                    seen.add(kw)
                    short.append({"question": kw, "category": "术语短查", "expected_behavior": "answer"})
        mix = []
        for i in range(max(len(short), len(long_))):
            if i < len(short):
                mix.append(short[i])
            if i < len(long_):
                mix.append(long_[i])
        if len(mix) >= n:
            return mix[:n]
    except Exception as e:  # noqa: BLE001
        print(f"[g01] 评测集不可用，退回内置混合 query：{type(e).__name__}: {e}")
    return [{"question": q, "category": "内置", "expected_behavior": "answer"} for q in FALLBACK_QUERIES[:n]]


def run_policy(policy: str, queries: list[dict], max_tokens: int, vram_total_mib: float, warmup: bool) -> dict[str, Any]:
    s = get_settings()
    sampler = VramSampler(0.02)
    store = KbStore(s.milvus_collection, "jieba")
    sched = ResidencyScheduler(policy=policy)
    retriever = HybridRetriever(store, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    engine = AnswerEngine(retriever, OllamaClient(), max_tokens=max_tokens)
    ctx = AccessContext(role="knowledge_owner")

    if warmup:
        with sched.use("embed"), sched.use("rerank"):
            pass
    sampler.start()
    oom = 0
    wall_sec = 0.0
    per_q, retr_q, gen_q, statuses, json_valid, tps = [], [], [], [], [], []
    for q in queries:
        t_start = time.perf_counter()
        try:
            res = engine.answer(q["question"], ctx)
            per_q.append(res.timings_ms.get("e2e_ms", 0.0))
            retr_q.append(res.timings_ms.get("retrieval_ms", 0.0))
            gen_q.append(res.timings_ms.get("generate_ms", 0.0))
            statuses.append(res.status)
            if res.llm:
                json_valid.append(1 if res.llm.get("first_attempt_json_valid") else 0)
                m = res.llm.get("metrics") or {}
                if m.get("tokens_per_sec"):
                    tps.append(m["tokens_per_sec"])
        except RuntimeError as e:
            oom += 1
            per_q.append(round((time.perf_counter() - t_start) * 1000, 1))
            statuses.append(f"error:{type(e).__name__}")
            print(f"[g01] {policy} 异常：{e}")
        wall_sec += time.perf_counter() - t_start
    samples = sampler.stop()
    peak = max((x["gpu_used_mib"] for x in samples), default=0.0)
    breach = sum(1 for x in samples if x["gpu_used_mib"] >= vram_total_mib - 32)

    summ = sched.summary()
    return {
        "policy": policy,
        "n_queries": len(queries),
        "max_tokens": max_tokens,
        "wall_sec": round(wall_sec, 1),
        "statuses": statuses,
        "oom_count": oom,
        "vram": {
            "peak_mib": round(peak, 1),
            "mean_mib": round(sum(x["gpu_used_mib"] for x in samples) / max(len(samples), 1), 1),
            "peak_torch_alloc_mib": round(max((x["torch_alloc_mib"] for x in samples), default=0.0), 1),
            "n_breach_samples": breach,
            "n_samples": len(samples),
        },
        "switch": {
            "n_cold_loads": summ["n_cold_loads"],
            "cold_load_sec_max": summ["cold_load_sec_max"],
            "cold_load_sec_mean": summ["cold_load_sec_mean"],
            "embed_load_sec": summ["embed_load_sec"],
            "rerank_load_sec": summ["rerank_load_sec"],
            "events": summ["events"][:12],
        },
        "e2e_ms": latency_report(per_q),
        "e2e_samples": [round(x, 1) for x in per_q],
        "retrieval_ms": latency_report(retr_q),
        "generate_ms": latency_report(gen_q),
        "llm": {
            "tokens_per_sec_median": _median(tps),
            "first_attempt_json_valid_rate": round(sum(json_valid) / max(len(json_valid), 1), 4),
        },
    }


def _median(vals: list[float]) -> float:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    return round(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="每种策略跑的问答轮数")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()

    s = get_settings()
    # 前置校验：索引必须存在且有数据。否则每条 query 都会「无证据→拒答」，
    # 端到端延迟全为 0，判定逻辑会把 0ms 误读成「远优于 3s 门禁」。
    probe = KbStore(s.milvus_collection, "jieba")
    if not probe.exists() or probe.count() == 0:
        msg = f"前置数据缺失：collection {s.milvus_collection} 不存在或为空，先跑 g02 建库再测 ①"
        print(f"[g01] {msg}")
        write_json("g01_vram_scheduler.json", {"env": env_snapshot(), "abort": msg})
        verdict("①", "fail", msg, "未产生有效测量，禁止据此放行")
        return 2
    n_chunks = probe.count()
    probe.close()

    report: dict[str, Any] = {"env": env_snapshot(), "policies": {}, "indexed_chunks": n_chunks}

    sampler = VramSampler(0.05)
    base_samples = []
    sampler.start()
    time.sleep(2.0)
    base_samples = sampler.stop()
    baseline_mib = max(x["gpu_used_mib"] for x in base_samples)
    total_mib = report["env"]["gpu"]["total_mib"]

    cli = OllamaClient()
    t0 = time.perf_counter()
    warm = cli.warmup()
    report["ollama_warmup"] = {**warm, "wall_sec": round(time.perf_counter() - t0, 2)}
    time.sleep(1.0)
    sampler.start()
    time.sleep(1.5)
    with_7b_mib = max(x["gpu_used_mib"] for x in sampler.stop())
    report["vram_attribution"] = {
        "baseline_mib": round(baseline_mib, 1),
        "with_7b_resident_mib": round(with_7b_mib, 1),
        "seven_b_footprint_mib": round(with_7b_mib - baseline_mib, 1),
        "headroom_for_embed_rerank_mib": round(total_mib - with_7b_mib, 1),
        "gpu_total_mib": total_mib,
    }

    queries = pick_queries(args.n)
    report["workload"] = {
        "n": len(queries),
        "short": sum(1 for q in queries if len(q["question"]) <= 12),
        "long": sum(1 for q in queries if len(q["question"]) > 12),
        "categories": sorted({q["category"] for q in queries}),
    }

    policies = [p for p in args.policies.split(",") if p in POLICIES]
    for p in policies:
        print(f"\n[g01] === policy {p} ===")
        report["policies"][p] = run_policy(p, queries, args.max_tokens, total_mib, warmup=(p != "time_sharded"))
        print(json.dumps({k: report["policies"][p][k] for k in ("oom_count", "vram", "switch", "e2e_ms")}, ensure_ascii=False)[:600])

    # 生成长度扫描：P95 端到端 < 3s 到底需要什么预算
    # 必须 warmup=True：否则每轮首问把嵌入/重排冷启算进 P95，n 小的时候 P95≈max，数字被污染
    sweep = {}
    # 只在「真的跑出了端到端延迟」的策略里选最优，否则全零时 p95=0 会被当成最优解
    measurable = [
        p
        for p in policies
        if report["policies"][p]["oom_count"] == 0 and report["policies"][p]["e2e_ms"].get("p50", 0) > 0
    ]
    best_policy = (
        min(measurable, key=lambda p: report["policies"][p]["e2e_ms"].get("p95", 1e9))
        if measurable
        else "co_resident"
    )
    sweep_queries = pick_queries(6)
    for mt in (96, 256, 700):
        one = run_policy(best_policy, sweep_queries, mt, total_mib, warmup=True)
        sweep[mt] = {
            "e2e_ms": one["e2e_ms"],
            "e2e_samples": one["e2e_samples"],
            "retrieval_ms": one["retrieval_ms"],
            "generate_ms": one["generate_ms"],
            "vram_peak_mib": one["vram"]["peak_mib"],
            "oom": one["oom_count"],
        }
        print(
            f"[g01] max_tokens={mt} → e2e p50={sweep[mt]['e2e_ms'].get('p50')} "
            f"p95={sweep[mt]['e2e_ms'].get('p95')} | 检索 p95={sweep[mt]['retrieval_ms'].get('p95')} "
            f"| 生成 p95={sweep[mt]['generate_ms'].get('p95')} | OOM={one['oom_count']}"
        )
    report["max_tokens_sweep"] = sweep
    report["sweep_policy"] = best_policy
    report["sweep_n"] = len(sweep_queries)

    rows = [
        [
            p,
            d["oom_count"],
            d["vram"]["peak_mib"],
            d["vram"]["mean_mib"],
            d["vram"]["n_breach_samples"],
            d["switch"]["n_cold_loads"],
            d["switch"]["cold_load_sec_max"],
            d["e2e_ms"].get("p50"),
            d["e2e_ms"].get("p95"),
            d["llm"]["tokens_per_sec_median"],
        ]
        for p, d in report["policies"].items()
    ]
    zero_oom = all(d["oom_count"] == 0 for d in report["policies"].values())
    viable = [p for p, d in report["policies"].items() if d["oom_count"] == 0 and d["vram"]["n_breach_samples"] == 0]
    measured = [p for p, d in report["policies"].items() if d["e2e_ms"].get("p50", 0) > 0]
    data_valid = bool(measured)
    co = report["policies"].get("co_resident", {})
    resident_ok = bool(co) and co.get("oom_count") == 0 and co["vram"]["n_breach_samples"] == 0
    retr_p95 = min(
        (d["retrieval_ms"].get("p95", float("inf")) for p, d in report["policies"].items() if p in measured),
        default=float("inf"),
    )
    # 同配置的两组样本（主循环 + 长度扫描）合并算 P95：挑较小的那个数会直接把判定翻掉——
    # 上一轮 co_resident 主循环 P95 6916ms vs 扫描 5799ms，差异就是采样方差，不能择优。
    combined = list(co.get("e2e_samples", [])) + list(sweep.get(256, {}).get("e2e_samples", []))
    e2e_p95 = percentile(combined, 95) if combined else float("inf")
    e2e_by_source = {
        "loop_p95": co.get("e2e_ms", {}).get("p95"),
        "sweep_p95": sweep.get(256, {}).get("e2e_ms", {}).get("p95"),
        "combined_p95": round(e2e_p95, 1),
        "n_samples": len(combined),
    }
    gate_retr = retr_p95 < 800
    gate_e2e = e2e_p95 < 6000
    passed = data_valid and zero_oom and resident_ok and gate_retr and gate_e2e
    partial = data_valid and zero_oom and resident_ok and not (gate_retr and gate_e2e)
    report["amended_gate"] = {
        "retrieval_p95_ms": retr_p95,
        "e2e_p95_at_256_ms": round(e2e_p95, 1),
        "e2e_p95_by_source": e2e_by_source,
        "retrieval_gate_800ms": gate_retr,
        "e2e_gate_6s": gate_e2e,
        "original_3s_gate_at_96_tokens": {
            "p95_ms": sweep.get(96, {}).get("e2e_ms", {}).get("p95"),
            "reachable": bool((sweep.get(96, {}).get("e2e_ms", {}).get("p95") or 9e9) < 3000),
            "note": "原口径「P95 端到端 < 3s」在 96 token 短答案档可达；不可达的是 256 token 完整结构化答案档",
        },
        "first_token_gate_1_2s": "未测（流式在 M4 实现后补测）",
    }
    margin_ms = round(6000 - e2e_p95, 1)
    report["amended_gate"]["margin_to_6s_gate_ms"] = margin_ms
    report["amended_gate"]["knife_edge"] = margin_ms < 500
    write_json("g01_vram_scheduler.json", report)

    lines = [
        "# Gate 0 ① 显存调度验证报告",
        "",
        f"- 验证时间：{report['env']['ts']}｜GPU {report['env']['gpu']['name']} {total_mib}MiB｜commit {report['env']['git_commit']}",
        f"- 负载：{report['workload']['n']} 轮（短 query {report['workload']['short']} / 长 query {report['workload']['long']}），max_tokens={args.max_tokens}",
        "",
        "## 显存归因（先立基线，再谈共存）",
        "",
        f"- 空闲基线（未加载任何模型）：{report['vram_attribution']['baseline_mib']} MiB",
        f"- qwen2.5:7b 常驻后：{report['vram_attribution']['with_7b_resident_mib']} MiB"
        f"（净增 {report['vram_attribution']['seven_b_footprint_mib']} MiB，Ollama warmup {report['ollama_warmup']['wall_sec']}s）",
        f"- 留给嵌入+重排的余量：{report['vram_attribution']['headroom_for_embed_rerank_mib']} MiB",
        "",
        "## 三种驻留策略对照",
        "",
        *md_table(
            ["策略", "OOM", "显存峰值 MiB", "均值 MiB", "越界采样", "冷启次数", "冷启最大 s", "P50 e2e ms", "P95 e2e ms", "tok/s"],
            rows,
        ),
        "",
        "## 生成长度预算扫描（修订后门禁：P95 完整答案 < 6s）",
        "",
        *md_table(
            ["max_tokens", "e2e P50", "e2e P95", "e2e max", "检索 P95", "生成 P95", "显存峰值", "OOM"],
            [
                [
                    mt,
                    v["e2e_ms"].get("p50"),
                    v["e2e_ms"].get("p95"),
                    v["e2e_ms"].get("max"),
                    v["retrieval_ms"].get("p95"),
                    v["generate_ms"].get("p95"),
                    v["vram_peak_mib"],
                    v["oom"],
                ]
                for mt, v in sweep.items()
            ],
        ),
        f"- 扫描策略：{best_policy}（预热后测，每档 {report['sweep_n']} 轮）",
        f"- 判定用合并样本（主循环 + 256 档扫描）：n={e2e_by_source['n_samples']}，"
        f"P95 = {e2e_by_source['combined_p95']}ms；分开看是主循环 {e2e_by_source['loop_p95']}ms / 扫描 {e2e_by_source['sweep_p95']}ms"
        "——两组差异属采样方差，判定不挑较小的那个。",
        f"- 判定余量：距 6s 门禁 {margin_ms}ms"
        + ("——**临界通过**，不构成「已达标」结论；M2 需以 ≥50 样本复测并设线上 P95 告警" if margin_ms < 500 else ""),
        f"- 原口径 < 3s 的可达性：96 token 短答案档 P95 {sweep.get(96, {}).get('e2e_ms', {}).get('p95')}ms"
        f"（{'可达' if (sweep.get(96, {}).get('e2e_ms', {}).get('p95') or 9e9) < 3000 else '不可达'}），"
        "256 token 完整答案档不可达——修订只放宽「完整答案」档，短答案档仍按原口径跟踪。",
        "",
        verdict(
            "①",
            "pass" if passed else ("partial" if partial else "fail"),
            f"零 OOM={zero_oom}；可与 7B 共存驻留={resident_ok}；P95 检索 {retr_p95:.0f}ms；"
            f"P95 完整答案(256，合并 n={e2e_by_source['n_samples']}) {e2e_p95:.0f}ms（有效测量：{measured}）",
            (
                "无有效端到端测量（query 全部在生成前被拒），本轮判定不成立；"
                if not data_valid
                else ""
            )
            + (
                ""
                if (gate_retr and gate_e2e)
                else f"修订后门禁（P95 检索<800ms 且 P95 完整答案<6s）未达：检索={gate_retr}，端到端={gate_e2e}"
            ),
        ),
        "",
        "## 分时调度策略参数（①出口）",
        "",
        "- 详见 `var/g01_vram_scheduler.json`；推荐参数在结论段给出（驻留策略、冷启预算、生成长度上限、空闲释放秒数）。",
        "",
        "## 已知边界",
        "",
        "- Windows WDDM 下 `nvidia-smi --query-compute-apps` 的 per-process 显存为 N/A，进程级归因只能用「全局用量差分 + 本进程 torch reserved」。",
        "- 采样期间宿主仍有桌面/浏览器占用（基线已单独记录），因此峰值含非本项目开销。",
        "- Ollama 侧 `deepseek-r1:7b` 与 `qwen2.5:7b` 不能同时驻留 8GB 单卡，这也是方案否决本地双模型的直接实测依据。",
    ]
    p = write_md("g01_vram_scheduler.md", lines)
    print("\n".join(lines))
    print("\nMD:", p)
    return 0 if (passed or partial) else 1


if __name__ == "__main__":
    raise SystemExit(main())
