"""净环境端到端延迟实测（#56）——走**生产入口**（宿主 API + HTTP），不 in-process 调引擎。

为什么单独一支脚本，而不是继续读 `session_turn` 全表：
那张表里混着 verify/probe/前端走查自己打的流量，p50/p95 是"这台机器这段时间所有流量的分布"，
不是"用户点一次发送等多久"。这里每一问都开**自己的 session**，统计只取本次创建的 session，
污染面从"整张表"缩到"这一次跑"。

口径写死在产物里，免得主张又变：
- 冷启动（模型首次装载）**单列**，不与热态混在一个数里；
- 同时给「客户端 elapsed」与「服务端 total_ms」两把尺——前者含 HTTP/Nginx 之外的宿主往返，
  后者是链路自身；DoD#5 到底认哪一把，等这批数出来后由用户裁；
- 3s 与 6s 两个门禁值各给超标率，不预选对自己有利的那个。

跑法：
    KEE_API_PORT=38000 uv run python scripts/serve_api.py     # 另开一个终端
    uv run python scripts/m5_latency_clean.py [--n 20]
产物：var/gate0/m5_latency_clean.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402

from _identity import ensure_accounts, login_http  # noqa: E402

BASE = os.environ.get("KEE_LAT_BASE", "http://127.0.0.1:38000")
# 身份走登录令牌：`KEE_ALLOW_HEADER_IDENTITY` 的默认已于 2026-10-15 翻成 False（用户裁「交付物默认 false、脚本显式带令牌」）；
# `Authorization` 由 main() 登录后补进 HDRS，session_turn/audit 里的 actor 因此是账号名 verify-owner，不再是 lat-clean。
HDRS = {"Content-Type": "application/json"}


def post(path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"), headers=HDRS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def quantile(vals: list[float], q: float) -> float:
    v = sorted(vals)
    return round(v[min(len(v) - 1, max(0, int(q * len(v)) - 1))], 1) if v else 0.0


def env_snapshot() -> dict:
    """把「当时有多干净」写进证据：不写这句，这批数过两周就没法判断可不可信。"""
    out = {}
    try:
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out["gpu"] = f"取失败 {e}"
    try:
        ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=20).stdout
        out["containers"] = sorted(ps.split())
    except Exception as e:  # noqa: BLE001
        out["containers"] = f"取失败 {e}"
    out["non_kee_containers"] = [c for c in out.get("containers", []) if isinstance(c, str) and not c.startswith("kee-")]
    return out


def flush_search_cache() -> int:
    """删掉 `kee:search:*`（Redis，非权威、TTL 300s 可驱逐），返回删了多少个键。

    为什么必须由跑的人决定：检索缓存命中时整条检索腿被短路（`retrieval_ms` 直接是 0.0），
    实测一轮 20 题约 2 分钟，下一轮在 TTL 内跑**同一批题**就全走缓存——
    我这台机器上第二次跑 p50 从 3193ms 掉到 2947ms，看着像噪声，其实是换了被测系统。
    重复测量前不清缓存，得到的不是「同一个配置的多个样本」，是两个配置。
    """
    import redis  # 延迟导入：不跑这支脚本的人不必装 redis

    s = get_settings()
    r = redis.Redis.from_url(s.redis_url, socket_connect_timeout=5)
    n = 0
    for k in r.scan_iter(match="kee:search:*", count=500):
        r.delete(k)
        n += 1
    return n


def stream_once(sid: str, question: str) -> dict:
    """走真 SSE 通路量三个口径，一次跑同时记：
      envelope_first_char = 收到响应第一个字节（信封最前面那个 `{`，几乎不等于「有用」）
      first_event_ms      = 首个带内容的 SSE 事件（通常是 retrieval 骨架 —— 用户已能看到「查到了什么」）
      first_visible_ms    = 首个非空 delta 文本（第一个真正可读的字）
    三者差一个量级是常态，所以① 门禁「1.2s」到底认哪个必须先定，否则任何一边都能自称达标。"""
    body = json.dumps({"session_id": sid, "question": question}).encode("utf-8")
    req = urllib.request.Request(BASE + "/api/v1/answers/stream", data=body, headers=HDRS, method="POST")
    t0 = time.perf_counter()
    out = {"envelope_first_char": None, "first_event_ms": None, "first_visible_ms": None,
           "retract_rate": None, "final_status": None}
    with urllib.request.urlopen(req, timeout=300) as r:
        ev = ""
        while True:
            line = r.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace").rstrip("\r\n")
            if s.startswith("event:"):
                ev = s.split(":", 1)[1].strip()
                if out["first_event_ms"] is None:
                    out["first_event_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            elif s.startswith("data:"):
                if out["envelope_first_char"] is None:
                    out["envelope_first_char"] = round((time.perf_counter() - t0) * 1000, 1)
                payload = s[5:].strip()
                if ev == "delta" and out["first_visible_ms"] is None:
                    try:
                        if json.loads(payload).get("text"):
                            out["first_visible_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    except json.JSONDecodeError:
                        pass
                elif ev == "answer":
                    try:
                        a = json.loads(payload)
                        out["final_status"] = a.get("status")
                        out["retract_rate"] = a.get("provisional_retracted")
                    except json.JSONDecodeError:
                        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--no-stream", action="store_true",
                    help="跳过 SSE 首 token 三口径（只跑端到端）")
    ap.add_argument("--keep-cache", action="store_true",
                    help="不清 Redis 检索缓存（默认清）——同题在 TTL 内复跑会整条检索腿短路，"
                         "那是另一个配置，不是同一配置的第二次采样")
    ap.add_argument("--stream-n", type=int, default=12)
    ap.add_argument("--tag", default="",
                    help="产物文件名后缀。同配置重复跑必须各留一份——"
                         "覆盖上一份就只剩一个读数，而门禁线恰好落在两次读数的中间。")
    args = ap.parse_args()

    try:
        h = json.loads(urllib.request.urlopen(BASE + "/healthz", timeout=10).read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[拒绝起跑] 宿主 API 不在 {BASE}（{e}）；先起 serve_api.py——"
              "延迟必须在生产入口上量，in-process 跑出来的数不算")
        return 2
    if not h.get("ok", True):
        print(f"[拒绝起跑] /healthz 报不健康：{h}")
        return 2
    print(f"[前置] healthz={h}")

    ensure_accounts()
    HDRS.update(login_http(BASE, "verify-owner"))

    pg = AuthoritativeStore()
    qs = build_eval_set(pg, write_files=False)["questions"]
    pool = [q for q in qs if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]]
    step = max(1, len(pool) // args.n)
    picked = pool[::step][: args.n]
    # 不进生成的路径（拒答/转人工）也要测：它们的 total_ms 曾长期是 NULL，
    # 于是从 session_turn 算的每个分位数都在悄悄丢掉最快的轮次。不跑这几题，
    # 「修了」这件事就只是我的一句话。
    edge = [q for q in qs if q["expected_behavior"] in ("refuse", "escalate", "no_answer")][:6]
    print(f"[样本] {len(picked)} 题（等距抽样自 {len(pool)} 道可答题）+ {len(edge)} 道不进生成的题")

    flushed = None if args.keep_cache else flush_search_cache()
    if flushed is not None:
        print(f"[清缓存] 删除 kee:search:* 键 {flushed} 个")
    else:
        print("[保留缓存] 若同题仍在 TTL 内，检索腿会被短路——分位数将只取未命中的轮次")

    env = env_snapshot()
    t0 = time.perf_counter()
    cold = post("/api/v1/answers", {"session_id": f"s-cold-{uuid.uuid4().hex[:8]}",
                                    "question": picked[0]["question"]})
    cold_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[冷启动] {cold_ms}ms（模型首次装载，单列、不进热态统计）status={cold.get('status')}")

    rows: list[dict] = []
    for q in picked + edge:
        sid = f"s-lat-{uuid.uuid4().hex[:10]}"
        t = time.perf_counter()
        r = post("/api/v1/answers", {"session_id": sid, "question": q["question"]})
        el = round((time.perf_counter() - t) * 1000, 1)
        rows.append({"qid": q["qid"], "category": q["category"], "session_id": sid,
                     "client_ms": el, "status": r.get("status"),
                     "cache_hit": bool((r.get("retrieval") or {}).get("cache_hit")),
                     "expected_behavior": q.get("expected_behavior", "answer"),
                     "generates": q["expected_behavior"] == "answer",
                     "n_citations": len(r.get("citations") or [])})
        print(f"  {q['qid']} {q['category']:4s} client={el}ms status={r.get('status')}")

    sids = tuple(r["session_id"] for r in rows)
    with pg.connect() as conn:
        srv = {x["session_id"]: x for x in conn.execute(
            "SELECT session_id, total_ms, retrieval_ms, status FROM session_turn WHERE session_id = ANY(%s)",
            (list(sids),)).fetchall()}
    for r in rows:
        s = srv.get(r["session_id"], {})
        r["server_total_ms"] = s.get("total_ms")
        r["retrieval_ms"] = s.get("retrieval_ms")

    # 缓存命中的轮次走的是另一条路径（检索腿被整体短路，retrieval_ms=0.0），
    # 与未命中不可混在同一个分位数里——混了就是拿两个配置出一个数。
    unc = [r for r in rows if not r["cache_hit"]]
    hot_c = [r["client_ms"] for r in unc if r["generates"]]
    all_c = [r["client_ms"] for r in unc]
    hot_s = [r["server_total_ms"] for r in unc if r["server_total_ms"]]
    ret = [r["retrieval_ms"] for r in unc if r["retrieval_ms"]]
    missing_e2e = [r["qid"] for r in rows if r["server_total_ms"] is None]

    def band(vals: list[float]) -> dict:
        return {"n": len(vals), "p50": quantile(vals, 0.5), "p95": quantile(vals, 0.95),
                "max": round(max(vals), 1) if vals else 0.0,
                "mean": round(statistics.mean(vals), 1) if vals else 0.0,
                "share_over_3s": round(sum(1 for v in vals if v > 3000) / len(vals), 4) if vals else None,
                "share_over_6s": round(sum(1 for v in vals if v > 6000) / len(vals), 4) if vals else None}

    # ---- 首 token 三口径：走真 SSE 通路（/answers/stream），与 /answers 同一张图 ----
    stream_rows: list[dict] = []
    re_flushed: int | None = None
    if not args.no_stream:
        # 同一批题刚被端到端跑过一遍，缓存必然是热的；不重刷就量到「短路检索」的首 token，
        # 比真实首查快一整趟 embedding——那是不利方向上对我们太有利的数。
        re_flushed = None if args.keep_cache else flush_search_cache()
        print(f"[流前清缓存] {re_flushed} 个键" if re_flushed is not None else "[流前保留缓存]")
        for q in picked[: args.stream_n]:
            sid = f"s-str-{uuid.uuid4().hex[:10]}"
            s = stream_once(sid, q["question"])
            s.update({"qid": q["qid"], "session_id": sid})
            stream_rows.append(s)
            print(f"  [流] {q['qid']} 信封={s['envelope_first_char']}ms 首事件={s['first_event_ms']}ms "
                  f"首个可读字={s['first_visible_ms']}ms status={s['final_status']}")

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env_at_measurement": env,
           "protocol": {"entrypoint": BASE + "/api/v1/answers（宿主 API，生产入口）",
                        "sessions": "一问一 session，统计只取本次创建的 session_id",
                        "cold": "单次冷启动单列，不进热态分位数",
                        "sample": f"{len(picked)} 题等距抽样 + {len(edge)} 道不进生成的题（验 total_ms 不再为 NULL）",
                        "stream": f"{len(stream_rows)} 题走 /api/v1/answers/stream 量首 token 三口径",
                        "cache_flushed_keys": flushed,
                        "stream_cache_flushed_keys": re_flushed},
           "cold_start_ms": cold_ms,
           "cache_hits": {"rows": len(rows), "cached": len(rows) - len(unc),
                          "note": "以下端到端分位数只取**未命中检索缓存**的轮次；"
                                  "命中轮次整条检索腿被短路（retrieval_ms=0.0），是另一条路径"},
           "client_side": band(all_c),
           "client_side_generating_only": band(hot_c),
           "server_side": band(hot_s),
           "retrieval_only": band(ret),
           # 与上一轮（修 e2e 前，样本全是进生成的题）可比的那把尺：加进拒答轮次后
           # 样本构成变了，不并列出来的话，p50 的移动会被读成「延迟变了」。
           "server_side_generating_only": band([r["server_total_ms"] for r in unc
                                                if r["generates"] and r["server_total_ms"]]),
           "first_token": {k: band([r[k] for r in stream_rows if r.get(k) is not None])
                           for k in ("envelope_first_char", "first_event_ms", "first_visible_ms")},
           "missing_e2e_qids": missing_e2e,
           "rows": rows, "stream_rows": stream_rows,
           "caveats": [
               "本机仍跑着 IDE 与桌面服务（我就在里面），做不到零背景；已把当时的 GPU 占用与容器清单记进 env_at_measurement。",
               "客户端与服务端两把尺都给了——DoD#5 认哪把、门禁取 3s 还是 6s，属口径裁决，不由这批数自己决定。",
               f"冷启动 {cold_ms}ms 与热态差一个量级；对外引用延迟必须分开说，否则下次拿到冷值会被读成 regression。",
               "首 token 三个口径量的是完全不同的东西（信封字节 ≠ 检索骨架事件 ≠ 可读文本），"
               "① 门禁认哪一个必须先由人裁；把最有利的那个当默认口径就是择优取样本。",
           ]}
    art = Path(get_settings().resolved_artifact_dir())
    name = ("m5_latency_clean.json" if not args.tag
            else f"m5_latency_clean.{args.tag}.json")
    (art / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[缓存] 命中 {out['cache_hits']['cached']}/{out['cache_hits']['rows']} 行，"
          f"以下端到端分位数只用未命中的 {len(unc)} 行")
    print("热态·客户端（未命中缓存，全部轮次）", json.dumps(out["client_side"], ensure_ascii=False))
    print("热态·客户端（仅进生成）", json.dumps(out["client_side_generating_only"], ensure_ascii=False))
    print("热态·服务端（含拒答轮次）", json.dumps(out["server_side"], ensure_ascii=False))
    print("热态·服务端（仅进生成，与上一轮可比）",
          json.dumps(out["server_side_generating_only"], ensure_ascii=False))
    print("仅检索", json.dumps(out["retrieval_only"], ensure_ascii=False))
    ft = out["first_token"]
    if stream_rows:
        print(f"首 token 三口径（{len(stream_rows)} 题，SSE 真通路）："
              f"信封首字节 p50={ft['envelope_first_char']['p50']} / "
              f"首个事件 p50={ft['first_event_ms']['p50']} / "
              f"首个可读字 p50={ft['first_visible_ms']['p50']}")
    if out["missing_e2e_qids"]:
        print(f"⚠ 仍有 {len(out['missing_e2e_qids'])} 个轮次 total_ms 为空：{out['missing_e2e_qids']}")
    else:
        print("所有轮次都有 total_ms（含拒答/转人工）")
    print("证据：", art / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
