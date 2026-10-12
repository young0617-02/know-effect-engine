"""把净环境延迟的**多个重复**合成一份可引用的读数（#56）。

为什么要有这支脚本：同一配置的两次跑，客户端 p50 相差 20% 以上并不稀奇（这台机器上实测过
4502ms 与 3193ms），而门禁线（3s / 6s）恰好落在这个跨度里。这时挑哪一次当"读数"，
就直接决定了结论是达标还是不达标——那叫择优取样本。所以规则是：**合并样本再判，各来源值一起列出来。**

三档来源分开，不混：
- `uncached`：起跑前清了 Redis 检索缓存的重复（真正走完整检索腿的那条路径）——主口径。
- `cached`：命中检索缓存的轮次（整条检索腿被短路，`retrieval_ms=0.0`）——另一条路径，
  单独列出来是为了防止下次有人复跑快了就写「延迟改善了」。
- `transcribed`：修 `e2e_ms` 之前那一轮，产物被同名覆盖，只剩终端读数（明确标注为转录，可信度低于文件）。

用法：uv run python scripts/m5_latency_merge.py
产物：var/gate0/m5_latency_replicates.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402

# 修 e2e 前那一轮：文件已被同名覆盖，这里是当时终端读数的转录，不是产物。
TRANSCRIBED = {
    "source": "run1（修 e2e_ms 前，转录自终端读数，无产物文件）",
    "n_generating": 20,
    "cold_start_ms": 23507.8,
    "client_p50": 4502.3, "client_p95": 5617.7, "client_max": 5750.2,
    "share_over_3s": 0.95, "share_over_6s": 0.0,
    "note": "这一轮拒答轮次 total_ms 为 NULL，所以它的 n 里没有拒答行；"
            "客户端读数还含当轮的其他背景负载。与后三轮不同构，只能并列不能合并。",
}

TAGS = ("run2", "rep1", "rep2", "rep3")


def quantile(vals: list[float], q: float) -> float:
    v = sorted(vals)
    return round(v[min(len(v) - 1, max(0, int(q * len(v)) - 1))], 1) if v else 0.0


def band(vals: list[float]) -> dict:
    return {"n": len(vals), "p50": quantile(vals, 0.5), "p95": quantile(vals, 0.95),
            "max": round(max(vals), 1) if vals else 0.0,
            "mean": round(statistics.mean(vals), 1) if vals else 0.0,
            "share_over_3s": round(sum(1 for v in vals if v > 3000) / len(vals), 4) if vals else None,
            "share_over_6s": round(sum(1 for v in vals if v > 6000) / len(vals), 4) if vals else None}


def is_cached(row: dict) -> bool:
    """run2 那轮还没记 `cache_hit`，用 `retrieval_ms==0.0` 反推（缓存命中时检索腿被整体短路）。"""
    if "cache_hit" in row:
        return bool(row["cache_hit"])
    return row.get("retrieval_ms") == 0.0


def main() -> int:
    art = Path(get_settings().resolved_artifact_dir())
    per_run, pooled_unc, pooled_cached = {}, [], []
    for tag in TAGS:
        p = art / f"m5_latency_clean.{tag}.json"
        if not p.exists():
            print(f"[跳过] 没有 {p.name}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d["rows"]
        unc = [r for r in rows if not is_cached(r)]
        gen = [r for r in unc if r["generates"]]
        per_run[tag] = {
            "cold_start_ms": d["cold_start_ms"],
            "env_gpu": d["env_at_measurement"].get("gpu"),
            "rows": len(rows), "cached_rows": len(rows) - len(unc),
            "generating_uncached": band([r["client_ms"] for r in gen]),
            "all_uncached": band([r["client_ms"] for r in unc]),
            "server_all_uncached": band([r["server_total_ms"] for r in unc
                                         if r["server_total_ms"]]),
            "retrieval": band([r["retrieval_ms"] for r in unc if r["retrieval_ms"]]),
            "first_token": d.get("first_token"),
            "stream_rows": d.get("stream_rows") or [],
            "stream_flushed_before_stream": (d.get("protocol") or {}).get(
                "stream_cache_flushed_keys"),
            "stream_flushed_keys": (d.get("protocol") or {}).get("stream_cache_flushed_keys"),
            "null_total_ms_rows": [r["qid"] for r in rows if not r["server_total_ms"]],
        }
        pooled_unc.extend(unc)
        pooled_cached.extend([r for r in rows if is_cached(r)])

    if not pooled_unc:
        print("没有可用的重复产物，先跑 m5_latency_clean.py --tag <名字>")
        return 2

    gen = [r for r in pooled_unc if r["generates"]]
    # 只合并**流前清过缓存**的那几轮：run2 的 SSE 段跑在端到端段之后、同一批题，
    # 检索腿全部命中缓存（它那个版本也还没记 `stream_cache_flushed_keys`）。
    # 把它的 p50=407ms 混进来，首 token 会凭空快一整趟 embedding。
    ft_runs = {t: v for t, v in per_run.items()
               if v.get("stream_rows") and v.get("stream_flushed_keys") is not None}
    excluded_ft = sorted(set(per_run) - set(ft_runs))
    pooled_ft: dict[str, list[float]] = {"envelope_first_char": [], "first_event_ms": [],
                                         "first_visible_ms": []}
    for t, d in ft_runs.items():
        for s in d["stream_rows"]:
            for k in pooled_ft:
                if s.get(k) is not None:
                    pooled_ft[k].append(s[k])
    out = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": ("同配置多轮 → **合并样本再判**，各来源值与样本量一起列出；"
                 "禁止挑其中一轮当唯一读数"),
        "sources": {"uncached_replicates": {t: {k: v for k, v in d.items() if k != "stream_rows"}
                                            for t, d in per_run.items()},
                    "transcribed": TRANSCRIBED,
                    "cached_sample": "var/gate0/m5_latency_clean.cachedpath.json"},
        "pooled_uncached": {
            "n_rows": len(pooled_unc),
            "n_generating": len(gen),
            "n_questions": len({r["qid"] for r in pooled_unc}),
            "runs": len(per_run),
            "client_all": band([r["client_ms"] for r in pooled_unc]),
            "client_generating_only": band([r["client_ms"] for r in gen]),
            "server_all": band([r["server_total_ms"] for r in pooled_unc
                                if r["server_total_ms"]]),
            "server_generating_only": band([r["server_total_ms"] for r in gen
                                            if r["server_total_ms"]]),
        },
        "pooled_cached": {
            "n_rows": len(pooled_cached),
            "client_all": band([r["client_ms"] for r in pooled_cached]),
            "note": "命中检索缓存的轮次，检索腿被短路。单列，不与 uncached 合并。",
        },
        "pooled_first_token": {
            "runs": sorted(ft_runs),
            "excluded_runs": excluded_ft,
            "口径": {
                "envelope_first_char": "收到响应第一个字节（几乎不代表『有用』）",
                "first_event_ms": "首个带内容的 SSE 事件（通常是检索骨架，用户已能看到查到了什么）",
                "first_visible_ms": "首个非空 delta 文本（第一个真正可读的字）——**最严的那把**",
            },
            "bands": {k: band(v) for k, v in pooled_ft.items()},
            "note": ("三个口径量的是完全不同的东西，① 门禁「1.2s」认哪个必须先由人裁；"
                     "默认取最严的 first_visible_ms，取宽松的那个就是择优取样本。"),
        },
        "spread_across_runs": {
            "client_generating_p50": {t: v["generating_uncached"]["p50"] for t, v in per_run.items()},
            "client_generating_p95": {t: v["generating_uncached"]["p95"] for t, v in per_run.items()},
            "p50_range_ms": round(max(v["generating_uncached"]["p50"] for v in per_run.values())
                                  - min(v["generating_uncached"]["p50"] for v in per_run.values()), 1),
            "note": ("同一配置重复跑之间的 p50 极差就是这台机器的测量噪声地板；"
                     "任何小于这个极差的 Δ 都不许写成「改善/恶化」。"),
        },
        "gate_relevance": {
            "3s": "合并样本后 >3s 的超标率见 pooled_uncached.client_generating_only.share_over_3s",
            "6s": "同上 >6s",
            "statement": ("这一支脚本**不给达标/不达标**：DoD#5 认客户端还是服务端、"
                          "取 3s 还是 6s，属口径裁决（待用户裁），裁决前给结论就是把尺子和读数混为一谈。"),
        },
        "e2e_fix_verification": {
            "rows_without_total_ms": sum(len(v["null_total_ms_rows"]) for v in per_run.values()),
            "rows_covered": len(pooled_unc),
            "statement": ("修 e2e_ms 前，不进生成的轮次（拒答/转人工）total_ms 恒为 NULL，"
                          "从 session_turn 算的每个分位数都在悄悄丢掉最快的轮次。"
                          "现在每轮都跑了 6 道不进生成的题，上表 rows_without_total_ms 必须为 0。"),
        },
    }
    (art / "m5_latency_replicates.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    pu = out["pooled_uncached"]
    print(f"合并 {out['pooled_uncached']['runs']} 轮 / {pu['n_rows']} 行"
          f"（其中进生成 {pu['n_generating']} 行，覆盖 {pu['n_questions']} 题）")
    print("  客户端·仅进生成", json.dumps(pu["client_generating_only"], ensure_ascii=False))
    print("  客户端·含拒答  ", json.dumps(pu["client_all"], ensure_ascii=False))
    print("  服务端·仅进生成", json.dumps(pu["server_generating_only"], ensure_ascii=False))
    print("  服务端·含拒答  ", json.dumps(pu["server_all"], ensure_ascii=False))
    print("  缓存命中轮次   ", json.dumps(out["pooled_cached"]["client_all"], ensure_ascii=False),
          f"(n={out['pooled_cached']['n_rows']})")
    sp = out["spread_across_runs"]
    print(f"  轮间 p50 {sp['client_generating_p50']} / p95 {sp['client_generating_p95']}"
          f" → p50 极差 {sp['p50_range_ms']}ms（这就是这台机器的延迟噪声地板）")
    pft = out["pooled_first_token"]["bands"]
    print(f"  首 token（合并 {out['pooled_first_token']['runs']}，流前已清缓存；"
          f"排除 {out['pooled_first_token']['excluded_runs']}=流段跑在热缓存上）")
    for k in ("envelope_first_char", "first_event_ms", "first_visible_ms"):
        b = pft[k]
        print(f"    {k:20s} n={b['n']} p50={b['p50']} p95={b['p95']} max={b['max']}")
    print(f"  转录轮（修 e2e 前）p50={TRANSCRIBED['client_p50']} p95={TRANSCRIBED['client_p95']}"
          f" >3s={TRANSCRIBED['share_over_3s']} —— 与上面三轮不同构，只并列不合并")
    ev = out["e2e_fix_verification"]
    print(f"  e2e 修复核对：{ev['rows_covered']} 行里 {ev['rows_without_total_ms']} 行缺 total_ms")
    print("  ⚠", out["gate_relevance"]["statement"])
    print("证据：", art / "m5_latency_replicates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
