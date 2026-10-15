"""#56 留下的那条尾巴：拒答轮次为什么比作答轮次还慢（q-063 实测 7.8s，是含拒答口径下唯一 >6s 的样本）。

不要凭叙述解释——生产入口的响应里就带着分段耗时（`retrieval_ms` / `generate_ms` / `e2e_ms` /
`first_token_ms` / `stage_*`），把它们**逐次摊开**再看分布，才知道时间花在哪一段。

同一道题连跑 N 次而不是 1 次：单次的 7.8s 可能是一次抖动，
`#56` 量到的延迟噪声地板是 93.6ms，但也见过冷装载差一个量级——**多次才分得清"这道题慢"和"这台机器慢"**。

跑法：uv run python scripts/m5_latency_outlier_probe.py [--qid q-063 --n 6 --compare q-060]
产物：var/gate0/m5_latency_outlier.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m5_latency_clean import flush_search_cache  # noqa: E402

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402

from _identity import ensure_accounts, login_http  # noqa: E402

BASE = os.environ.get("KEE_LAT_BASE", "http://127.0.0.1:38000")
# 身份走登录令牌：`KEE_ALLOW_HEADER_IDENTITY` 的默认已于 2026-10-15 翻成 False（用户裁「交付物默认 false、脚本显式带令牌」）；
# `Authorization` 由 main() 登录后补进 HDRS，session_turn/audit 里的 actor 因此是账号名 verify-owner，不再是 outlier-probe。
HDRS = {"Content-Type": "application/json"}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers=HDRS, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def one(question: str) -> dict:
    sid = f"s-out-{uuid.uuid4().hex[:10]}"
    t = time.perf_counter()
    a = post("/api/v1/answers", {"session_id": sid, "question": question})
    client = round((time.perf_counter() - t) * 1000, 1)
    tm = a.get("timings_ms") or {}
    return {"session_id": sid, "client_ms": client, "status": a.get("status"),
            "needs_human": bool(a.get("needs_human")),
            "conclusion_chars": len((a.get("conclusion") or "").strip()),
            "n_citations": len(a.get("citations") or []),
            "e2e_ms": tm.get("e2e_ms"), "retrieval_ms": tm.get("retrieval_ms"),
            "generate_ms": tm.get("generate_ms"), "first_token_ms": tm.get("first_token_ms"),
            "stages": {k: v for k, v in tm.items() if str(k).startswith("stage_")},
            "notes": a.get("notes") or [], "degraded": (a.get("retrieval") or {}).get("degraded") or []}


def band(vals: list[float]) -> dict:
    v = sorted(vals)
    return {"n": len(v), "min": v[0], "median": statistics.median(v), "max": v[-1]} \
        if v else {"n": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="q-063")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--compare", default="q-060",
                    help="对照组：同样是不作答的轮次，但历史上只要 0.4s——差异本身就是答案")
    args = ap.parse_args()

    try:
        h = json.loads(urllib.request.urlopen(BASE + "/healthz", timeout=10).read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[拒绝起跑] 生产入口不在 {BASE}（{e}）")
        return 2

    ensure_accounts()
    HDRS.update(login_http(BASE, "verify-owner"))

    from kee.eval.build_eval import build_eval_set

    pg = AuthoritativeStore()
    by_q = {q["qid"]: q for q in build_eval_set(pg, write_files=False)["questions"]}
    todo = [x for x in (args.qid, args.compare) if x and x in by_q]

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "healthz": h, "cases": {}}
    for qid in todo:
        q = by_q[qid]
        # 缓存要在**每一跑前**清：只在整批前清一次，第 2~N 跑照样命中缓存（检索列全成 0.0），
        # 而我关心的是"同一道题重复跑的样子"——那每一跑都必须是同一条路径。
        reps = []
        for _ in range(args.n):
            flush_search_cache()
            reps.append(one(q["question"]))
        for r in reps:  # 慢的真相常在最后一列：期望行为与本轮实际走的闸
            r["expected_behavior"] = q.get("expected_behavior")
            r["category"] = q.get("category")
            print(f"  {qid} client={r['client_ms']} e2e={r['e2e_ms']} 检索={r['retrieval_ms']} "
                  f"生成={r['generate_ms']} 首token={r['first_token_ms']} status={r['status']} "
                  f"结论字数={r['conclusion_chars']} 引用={r['n_citations']} notes={r['notes'][:2]}")
        out["cases"][qid] = {
            "question": q["question"], "category": q["category"],
            "cache_flushed_before_each_rep": True,
            "expected_behavior": q.get("expected_behavior"),
            "n_gold": len(q.get("gold_chunk_ids") or []),
            "reps": reps,
            "summary": {k: band([r[k] for r in reps if r.get(k) is not None])
                        for k in ("client_ms", "e2e_ms", "retrieval_ms", "generate_ms", "first_token_ms")},
        }

    art = Path(get_settings().resolved_artifact_dir())
    (art / "m5_latency_outlier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for qid, c in out["cases"].items():
        s = c["summary"]
        print(f"\n{qid}（{c['category']}/{c['expected_behavior']}，n={len(c['reps'])}）"
              f"客户端 {s['client_ms']} ｜ 检索 {s.get('retrieval_ms')} ｜ 生成 {s.get('generate_ms')}")
    print("证据：", art / "m5_latency_outlier.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
