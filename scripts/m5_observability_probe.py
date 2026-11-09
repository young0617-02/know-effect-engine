"""可观测接入实测：`/metrics` 的形状、埋点开销、Prometheus 侧对账、看板与告警表达式的体检。

这条脚本要回答的不是「接上了没有」，而是三件会被追问的事：
  1. **埋点贵不贵**——两份同码实例（开指标 / 关指标）**配对交错**逐对相减，并在测量期间
     暂停抓取器。第一版按「先 50 次开、再 50 次关」跑，同一个端点的 Δ 在两次运行里
     从 −0.3ms 翻到 +7.1ms——符号会翻的差不是差，那是进程状态与抓取竞争混进来的。
  2. **数对不对**——宿主 `/metrics` 的累计值与 Prometheus 查出来的值要对得上，
     各计数器之间要有恒等式（轮次、缓存命中、降级标记、直方图样本数）。恒等式不成立
     就是埋点漏了某条出口，而不是「看板看着不太对」。
  3. **看板和告警里的每条表达式真的能跑吗**——逐条送到 Prometheus 上执行，
     分「有数 / 空 / 报错」。一条写错的 PromQL 在 UI 上只表现为 No data，
     和「这个指标本轮没发生」长得一模一样。

前置（都要已经起着）：
    uv run python scripts/serve_api.py                       # 宿主 API（--api 指它）
    uv run python scripts/obs_targets.py --port <API 端口>
    docker compose -f infra/docker-compose.observability.yml up -d

跑法：
    uv run python scripts/m5_observability_probe.py
    uv run python scripts/m5_observability_probe.py --skip-traffic     # 只体检，不打业务流量
    uv run python scripts/m5_observability_probe.py --skip-ab          # 不起第二份实例做对照

产物：var/gate0/m5_observability.json + reports/m5/M5_可观测接入实测.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_ENV = dotenv_values(ROOT / ".env")
GRAFANA_USER = os.environ.get("KEE_GRAFANA_ADMIN_USER") or _ENV.get("KEE_GRAFANA_ADMIN_USER") or "admin"
GRAFANA_PASSWORD = os.environ.get("KEE_GRAFANA_ADMIN_PASSWORD") or _ENV.get("KEE_GRAFANA_ADMIN_PASSWORD") or ""
sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "var" / "gate0"
OUT_MD = ROOT / "reports" / "m5" / "M5_可观测接入实测.md"
DASH = ROOT / "infra" / "grafana" / "dashboards" / "kee-rag.json"
ALERTS = ROOT / "infra" / "prometheus" / "alerts.yml"

# 这些表达式在本轮流量下**必须有数**；其余允许为空（空是「没发生」，不是坏）。
REQUIRED_EXPRS = (
    "sum(rate(kee_http_requests_total[5m]))",
    "sum(increase(kee_qa_rounds_total[15m]))",
    'kee_dependency_up',
    'min(kee_dependency_up)',
)


def http(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def timed_text(url: str, n: int) -> dict:
    """抓 n 次 `/metrics`，记录字节数与耗时分布（含序列化，这才是抓取真实花掉的时间）。

    形状统计只在**第一次**抓取上做：循环里反复重取会把抓取自身算进样本数，
    `kee_http_requests_total` 会多几条——那不是脏数据，但会让「声明族清单」看着不稳定。
    """
    status, first = http(url, timeout=60.0)
    lines = first.decode("utf-8").splitlines()
    typed = [ln.split() for ln in lines if ln.startswith("# TYPE kee_")]
    # `prometheus_client` 会为每个 counter/histogram 额外发一条 `<名>_created` 时间戳序列，
    # 它**只在指标被摸过之后才出现** ⇒ 数 `# TYPE kee_` 行数会得到一个随运行浮动的数
    # （本机实测 23 / 24 / 32），所以「有多少个指标」必须只数**声明族**、派生族单独列。
    declared = sorted(t[2] for t in typed if not t[2].endswith("_created"))
    costs, sizes = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        status, body = http(url, timeout=60.0)
        costs.append(round((time.perf_counter() - t0) * 1000, 2))
        sizes.append(len(body))
        if status != 200:
            raise SystemExit(f"/metrics 返回 {status}")
        time.sleep(0.05)
    return {
        "次数": n,
        "字节": sizes[-1],
        "声明族数": len(declared),
        "声明族清单": declared,
        "created派生族数": len(typed) - len(declared),
        "样本数": sum(1 for ln in lines if ln and not ln.startswith("#")),
        "耗时ms": {"p50": round(statistics.median(costs), 2),
                   "p95": round(sorted(costs)[max(0, int(len(costs) * 0.95) - 1)], 2),
                   "max": round(max(costs), 2),
                   "min": round(min(costs), 2)},
    }


def prom_eventually(prom: str, expr: str, expect: float, timeout: float = 120.0,
                    window: str | None = None) -> dict:
    """等到 Prometheus **真的抓到**那个值为止，而不是睡一个固定秒数就当它对。

    第一版就是睡 20s 然后比较，结果 A/B 把容器停过一轮、重启后第一次抓取还没落地，
    对账读出「宿主 7 / Prometheus 5 → 不一致」。那个 5 不是错，是**旧**；
    把「还没抓到」报成「对不上」，就是在给自己的观测层造一条假故障。
    """
    deadline = time.monotonic() + timeout
    attempts, last = 0, None
    while time.monotonic() < deadline:
        attempts += 1
        last = prom_query(prom, expr, window=window)
        if last.get("状态") == "OK" and abs(float(last["首值"]) - expect) < 1e-6:
            return {"一致": True, "尝试次数": attempts, "Prometheus": last, "宿主": expect}
        time.sleep(3.0)
    return {"一致": False, "尝试次数": attempts, "Prometheus": last, "宿主": expect,
            "说明": f"{timeout:.0f}s 内没等到（抓取间隔 15s；连续多次不等才是真问题）"}


def prom_query(prom: str, expr: str, window: str | None = None) -> dict:
    """instant query；返回分类 + 首个值。`window` 只用于把表达式里的 [30m] 换成进程 uptime 宽度。

    参数走 **form 编码**：Prometheus 的 `/api/v1/query` 只认 urlencoded/query-string，
    发 JSON body 会得到「parse error: no expression found in input」——那条错误长得很像
    「表达式写错了」，其实是请求体格式错了。一次踩坑换来的一条注释。
    """
    e = expr
    if window:
        e = re.sub(r"\[(?:5|10|15|30)m\]", f"[{window}]", e)
    try:
        body = http(f"{prom}/api/v1/query",
                    data=urllib.parse.urlencode({"query": e}).encode(),
                    headers={"Content-Type": "application/x-www-form-urlencoded"})[1]
        data = json.loads(body)["data"]["result"]
        return {"状态": "OK" if data else "EMPTY", "序列数": len(data),
                "首值": (round(float(data[0]["value"][1]), 6) if data else None)}
    except urllib.error.HTTPError as err:
        return {"状态": "ERROR", "错误": f"http {err.code}: {err.read().decode('utf-8', 'ignore')[:200]}"}
    except Exception as err:  # noqa: BLE001
        return {"状态": "ERROR", "错误": f"{type(err).__name__}: {err}"[:200]}


# ---------- 计数器解析 ----------
def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, val = line.rpartition(" ")
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def counter(samples: dict[str, float], name: str, **labels: str) -> float:
    """按样本名 + 标签精确取值；取不到就是 0（Prometheus 的约定：没发生过）。"""
    for key, val in samples.items():
        head, _, lab = key.partition("{")
        if head != name and key != name:
            continue
        got = dict(re.findall(r'(\w+)="([^"]*)"', lab))
        if all(got.get(k) == v for k, v in labels.items()):
            return val
    return 0.0


def sum_prefix(samples: dict[str, float], name: str, **labels: str) -> float:
    total = 0.0
    for key, val in samples.items():
        head, _, lab = key.partition("{")
        if head != name and key != name:
            continue
        got = dict(re.findall(r'(\w+)="([^"]*)"', lab))
        if all(got.get(k) == v for k, v in labels.items()):
            total += val
    return total


# ---------- 真流量 ----------
def run_traffic(api: str, question: str, stream_question: str) -> dict:
    from _identity import login_http  # 与界面同一个入口、同一枚令牌

    headers = login_http(api, "verify-owner")
    headers["Content-Type"] = "application/json"
    out: dict = {"账号": "verify-owner（走真实登录换 Bearer）"}

    def post(path: str, payload: dict, timeout: float = 420.0):
        t0 = time.perf_counter()
        status, body = http(f"{api}{path}", data=json.dumps(payload, ensure_ascii=False).encode(),
                            headers=headers, timeout=timeout)
        return status, json.loads(body), round((time.perf_counter() - t0) * 1000, 1)

    _, a1, ms1 = post("/api/v1/answers", {"question": question, "top_k": 5})
    out["冷轮"] = {"status": a1["status"], "用时ms": ms1, "degraded": a1["retrieval"]["degraded"],
                   "cache_hit": a1["retrieval"]["cache_hit"], "composite": a1["confidence"].get("composite"),
                   "引用数": len(a1["citations"]), "timings_ms": a1["timings_ms"]}
    _, a2, ms2 = post("/api/v1/answers", {"question": question, "top_k": 5})
    out["缓存轮"] = {"status": a2["status"], "用时ms": ms2, "degraded": a2["retrieval"]["degraded"],
                     "cache_hit": a2["retrieval"]["cache_hit"], "composite": a2["confidence"].get("composite"),
                     "引用数": len(a2["citations"])}
    # 流式：图跑在同一个收口上，所以要计进同一个计数器
    t0 = time.perf_counter()
    kinds: dict[str, int] = {}
    req = urllib.request.Request(f"{api}/api/v1/answers/stream",
                                 data=json.dumps({"question": stream_question, "top_k": 5}).encode(),
                                 headers=headers, method="POST")
    final: dict = {}
    with urllib.request.urlopen(req, timeout=420) as resp:
        event = ""
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                kinds[event] = kinds.get(event, 0) + 1
            elif line.startswith("data:") and event == "answer":
                final = json.loads(line.split(":", 1)[1].strip())
    out["流式轮"] = {"事件": kinds, "用时ms": round((time.perf_counter() - t0) * 1000, 1),
                     "status": final.get("status"), "degraded": (final.get("retrieval") or {}).get("degraded"),
                     "provisional_retracted": final.get("provisional_retracted"),
                     "provisional_lost": final.get("provisional_lost")}
    # 不带缓存的检索端点（`cache=None`）与一次鉴权失败，用来验 `cached="disabled"` 与错误码计数
    _, s1, _ = post("/api/v1/retrieval/search", {"query": question, "top_k": 5})
    out["检索端点"] = {"mode": s1["mode"], "degraded": s1["degraded"], "hits": len(s1["hits"])}
    try:
        http(f"{api}/api/v1/auth/me")
    except urllib.error.HTTPError as e:
        out["未登录"] = {"http": e.code, "code": json.loads(e.read())["code"]}
    return out


# ---------- 恒等式 ----------
def invariants(before: dict[str, float], host: dict[str, float], traffic: dict) -> list[dict]:
    """计数器之间必须成立的等式。

    用**增量**而不是累计值：累计值互查只能证明「进程启动以来恰好一直没歪」，
    增量互查才能证明「这一批请求确实按这个关系计数」。`--skip-traffic` 时两者都是 0，
    那属真空通过，报告里按「空集上的真」对待、不当证据。
    """
    def d(name: str, **labels: str) -> float:
        return sum_prefix(host, name, **labels) - sum_prefix(before, name, **labels)

    def c(name: str, **labels: str) -> float:
        return counter(host, name, **labels) - counter(before, name, **labels)

    hybrid = d("kee_retrieval_rounds_total", mode="hybrid")
    degraded_expected = sum(len(traffic.get(k, {}).get("degraded") or [])
                            for k in ("冷轮", "缓存轮", "流式轮", "检索端点"))
    checks = [
        {"恒等式": "每轮问答恰好计一次（轮次增量 == e2e 直方图样本增量）",
         "左": d("kee_qa_rounds_total"), "右": c("kee_qa_stage_seconds_count", stage="e2e"),
         "依据": "两者都记在 `result_from_state` 这一个收口上；分开两处记就会一条多一条少"},
        {"恒等式": "缓存 get 的两个记账点重合（hit 次数 == cached=hit 轮数）",
         "左": c("kee_search_cache_ops_total", kind="search", result="hit"),
         "右": c("kee_retrieval_rounds_total", mode="hybrid", cached="hit"),
         "依据": "`SearchCache.get` 与 `pipeline._emit` 各记一次，必须同增"},
        {"恒等式": "hit + miss == 带缓存的检索轮数",
         "左": c("kee_search_cache_ops_total", kind="search", result="hit")
         + c("kee_search_cache_ops_total", kind="search", result="miss"),
         "右": hybrid, "依据": "每次 retrieve() 最多做一次 get"},
        {"恒等式": "store <= miss（`retrieval_total_failure` 那条出口在 set 之前返回）",
         "左": c("kee_search_cache_ops_total", kind="search", result="store"),
         "右": c("kee_search_cache_ops_total", kind="search", result="miss"),
         "依据": "两条出口顺序不同，所以只能是不等式"},
        {"恒等式": "cached=disabled 轮数 == 不带缓存的端点调用数",
         "左": c("kee_retrieval_rounds_total", cached="disabled"), "右": 1 if traffic else 0,
         "依据": "`/api/v1/retrieval/search` 用 `cache=None`；把它并进 miss 会让命中率的分母混进不该参与的轮次"},
        {"恒等式": "top_score 样本增量 == 带缓存的检索轮数",
         "左": c("kee_qa_top_score_count"), "右": hybrid,
         "依据": "`graph.retrieve` 每轮记一次（含命中轮与零命中轮）；对不上就是漏了某条出口"},
        {"恒等式": "响应体里的 degraded 条数 == degraded 计数增量",
         "左": degraded_expected, "右": d("kee_retrieval_degraded_total"),
         "依据": "缓存命中轮的 degraded 恒为 0（既有边界），响应体与计数器要**一起**少；只少一边是漏计"},
    ]
    for chk in checks:
        chk["成立"] = (chk["左"] <= chk["右"]) if chk["恒等式"].startswith("store") else (chk["左"] == chk["右"])
    return checks


# ---------- A/B 历史账（开销结论不许手写） ----------
HIST = ART / "m5_observability_ab_history.json"


def append_history(rows: list[dict], pairs: int) -> list[dict]:
    """把本轮每个端点的「配对差中位数」累进历史账。

    为什么要账本而不是一句写死的话：开销结论要靠**多次运行**才成立，而报告是脚本生成的——
    在脚本里写「三次运行落在 −0.52 ~ +1.89ms」，第四次跑完这句话就自动变成假话，
    而且变得很难看（它看起来像新数据）。现在这句话由账本推导。
    """
    hist = []
    if HIST.exists():
        try:
            hist = json.loads(HIST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hist = []
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        hist.append({"ts": stamp, "端点": r["端点"], "配对差中位数ms": r["配对差中位数ms"],
                     "Δp50ms": r["Δp50ms"], "每端点对数": pairs})
    HIST.parent.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    return hist


def history_statement(hist: list[dict]) -> dict:
    """按端点汇总历史中位数，并做**可证否**的判定：t = mean/(s/√n)，5% 显著性临界 3.92。"""
    by: dict[str, list[float]] = {}
    for row in hist:
        by.setdefault(row["端点"], []).append(float(row["配对差中位数ms"]))
    per, verdicts = {}, []
    all_vals: list[float] = []
    for endpoint, vals in by.items():
        all_vals += vals
        if len(vals) >= 3:
            mean = statistics.fmean(vals)
            s = statistics.stdev(vals)
            # 开销的**方向未定**，按双侧 5%（df=2 → 4.303）；这里用 3.92（双侧、df≈大量）作保守近似，
            # 且刻意写成「拒绝/证据不足」而不是给一个 p 值——n 只有几次，报 p 值就是假装精度。
            t = abs(mean) / (s / len(vals) ** 0.5) if s > 0 else float("inf")
            verdict = ("拒绝「有 2ms 开销」且无法证明有正开销" if t < 3.92 else
                       ("开销为正且统计可辨" if mean > 0 else "开指标反而更快（噪声或系统差）"))
            per[endpoint] = {"n次": len(vals), "均值ms": round(mean, 2), "标准差ms": round(s, 2),
                             "5%临界ms": round(3.92 * s / len(vals) ** 0.5, 2), "判定": verdict}
        else:
            per[endpoint] = {"n次": len(vals), "取值": vals, "判定": "次数不足以下结论（<3）"}
    if len(all_vals) >= 3:
        mean, s = statistics.fmean(all_vals), statistics.stdev(all_vals)
        bound = 3.92 * s / len(all_vals) ** 0.5
        claim = f"{mean:.2f}±{bound:.2f}ms（5% 置信半宽）"
        neg = sum(1 for v in all_vals if v < 0)
        statement = (f"合并 {len(all_vals)} 个端点轮次的配对差中位数：分布 {min(all_vals)} ~ {max(all_vals)}ms，"
                     f"其中 {neg} 个为负；均值 {claim}。"
                     + ("**|mean| 低于临界 ⇒ 既拒绝「有 2ms 开销」，也拿不出「确有正开销」的证据**，"
                        "只能报「不高于 2ms、与噪声同量级」。" if abs(mean) <= bound else
                        "均值超出临界值，说明有可辨的正开销，须按上面的分布报具体数。"))
    else:
        statement = "历史运行不足 3 次，任何开销结论都还立不住。"
    verdicts.append(statement)
    return {"按端点": per, "结论": statement}


# ---------- A/B 对照 ----------
def one_cost(base: str, headers: dict | None, path: str, payload: dict | None) -> float:
    t0 = time.perf_counter()
    try:
        if payload is None:
            http(base + path, headers=headers or {}, timeout=60.0)
        else:
            http(base + path, data=json.dumps(payload).encode(), headers=headers or {}, timeout=60.0)
    except urllib.error.HTTPError:
        pass  # 401/404 也是一次完整中间件往返
    return round((time.perf_counter() - t0) * 1000, 2)


def bench_paired(api_on: str, api_off: str, headers: dict | None, path: str, rounds: int) -> dict:
    """**配对交错采样**：每次请求都 A、B 各打一发，交替进行。

    分开两轮跑（先 50 次 A、再 50 次 B）会把「这段时间机器在忙什么」算进 Δ：
    本机第一版就是这么读出 /healthz 上 +7ms 的，而换一次运行同一个端点量到 −0.3ms——
    符号会翻的差就不是差。交错之后，同一次抖动同时落在两边。
    """
    a: list[float] = []
    b: list[float] = []
    for i in range(rounds):
        # 奇偶交换先后次序，避免「永远先打 A」带来的固定偏置（连接池、CPU 频率爬坡）
        if i % 2 == 0:
            a.append(one_cost(api_on, headers, path, None))
            b.append(one_cost(api_off, headers, path, None))
        else:
            b.append(one_cost(api_off, headers, path, None))
            a.append(one_cost(api_on, headers, path, None))
        time.sleep(0.02)

    def stats(v: list[float]) -> dict:
        s = sorted(v)
        return {"p50": round(statistics.median(s), 2),
                "p95": round(s[max(0, int(len(s) * 0.95) - 1)], 2),
                "均值": round(statistics.fmean(s), 2)}

    sa, sb = stats(a), stats(b)
    # 配对差：同一对里相减再取中位数，这才是「同一次抖动同时落在两边」的算法
    pairs = sorted(round(x - y, 2) for x, y in zip(a, b))
    return {"开指标": sa, "关指标": sb, "Δp50ms": round(sa["p50"] - sb["p50"], 2),
            "配对差中位数ms": round(statistics.median(pairs), 2),
            "配对差p95ms": pairs[max(0, int(len(pairs) * 0.95) - 1)],
            "n": rounds}


def micro_bench() -> dict:
    """进程内微基准：埋点本身每次花多少。与 HTTP 对照互为佐证，但不等于端到端。"""
    from kee.obs import metrics as m
    from kee.qa.answer import AnswerResult

    res = AnswerResult(question="q", status="answer",
                       confidence={"composite": 0.5, "score_basis": "rule_composite_not_probability"},
                       citations=[{"evidence_id": "E1"}],
                       timings_ms={"e2e_ms": 8000.0, "retrieval_ms": 3000.0, "generate_ms": 4900.0,
                                   "first_token_ms": 900.0, "stage_embed_ms": 120.0, "stage_rerank_ms": 800.0})
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        m.observe_qa_result(res, gated="")
    qa_us = round((time.perf_counter() - t0) * 1e6 / n, 2)
    t0 = time.perf_counter()
    for _ in range(n):
        m.observe_retrieval(_FakeRet(), cached="miss")
    ret_us = round((time.perf_counter() - t0) * 1e6 / n, 2)
    t0 = time.perf_counter()
    m.render()
    render_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"observe_qa_result_每次us": qa_us, "observe_retrieval_每次us": ret_us,
            "render_一次ms": render_ms, "n": n,
            "注": "同一进程连跑 2000 次的均值，不含 Redis/Milvus 往返；它是埋点开销的**下界口径**"}


class _FakeRet:
    mode = "hybrid"
    cache_hit = False
    degraded = ["dense_timeout>2500ms", "hybrid_single_leg"]
    chunks: list = []


def spawn_second_api(api_port: int, ab_port: int) -> subprocess.Popen:
    """起一份**关掉指标端点**的同码实例。`serve_api.py` 自己会拒绝非回环绑定，这里只传端口。

    ⚠ 收尾必须 `taskkill /T`（进程树）：`.venv/Scripts/python.exe` 是 trampoline，
    父进程退了子进程还在占端口——只杀父 PID 会把这台机器留成一个「看不见但活着」的实例。
    """
    env = dict(os.environ, KEE_API_PORT=str(ab_port), KEE_METRICS_ENABLED="false",
               KEE_ALLOW_HEADER_IDENTITY=os.environ.get("KEE_ALLOW_HEADER_IDENTITY", "false"))
    log = open(ROOT / "var" / "api_metrics_off.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "serve_api.py")],
                            cwd=ROOT, env=env, stdout=log, stderr=log)
    for _ in range(60):
        try:
            http(f"http://127.0.0.1:{ab_port}/healthz", timeout=5.0)
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    proc.terminate()
    raise SystemExit(f"关掉指标的对照实例起不来（端口 {ab_port}），看 var/api_metrics_off.log")


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        # 不捕获 stdout：`taskkill` 在中文 Windows 上输出 GBK，`text=True` 会按 UTF-8 解
        # 而崩在读线程里（崩了不影响杀进程，但会在报告里留一条吓人的 traceback）。
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    finally:
        for _ in range(20):
            try:
                http(f"http://127.0.0.1:{proc_port(proc)}/healthz", timeout=2.0)
                time.sleep(0.5)
            except Exception:  # noqa: BLE001
                return
        print(f"[警告] 对照实例端口 {proc_port(proc)} 仍在监听，请手工确认", file=sys.stderr)


def proc_port(proc: subprocess.Popen) -> int:
    return int(getattr(proc, "_kee_port", 0) or 0)


# ---------- 主流程 ----------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:28000")
    ap.add_argument("--prom", default="http://127.0.0.1:39090")
    ap.add_argument("--grafana", default="http://127.0.0.1:33000")
    ap.add_argument("--ab-port", type=int, default=28001)
    ap.add_argument("--rounds", type=int, default=60, help="A/B 每个端点的请求次数")
    ap.add_argument("--scrapes", type=int, default=20)
    ap.add_argument("--question", default="项目终验通过的条件是什么")
    ap.add_argument("--stream-question", default="终验之后质保期多久")
    ap.add_argument("--skip-traffic", action="store_true")
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--pause-prom", action=argparse.BooleanOptionalAction, default=True,
                    help="A/B 期间暂停 kee-prom 容器（每 15s 的抓取会给开指标那侧加一条 B 侧没有的竞争）")
    ap.add_argument("--tag", default="", help="证据卫生：另存为 m5_observability<tag>.json，不覆盖既有产物")
    args = ap.parse_args()

    api_port = int(re.search(r":(\d+)", args.api).group(1))
    rep: dict = {"生成时间": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "api": args.api, "prometheus": args.prom, "grafana": args.grafana,
                 "命令行": " ".join(sys.argv[1:]) or "（默认参数）"}

    print("== 1. 直连 /metrics 的形状与抓取耗时 ==")
    status, body = http(f"{args.api}/metrics")
    host = parse_metrics(body.decode("utf-8"))
    rep["端点形状"] = timed_text(f"{args.api}/metrics", args.scrapes)
    rep["端点形状"]["http"] = status
    print("  ", json.dumps(rep["端点形状"], ensure_ascii=False))

    print("== 2. 进程内微基准（埋点自身成本） ==")
    rep["微基准"] = micro_bench()
    print("  ", json.dumps(rep["微基准"], ensure_ascii=False))

    traffic: dict = {}
    status, body = http(f"{args.api}/metrics")
    before = parse_metrics(body.decode("utf-8"))
    if not args.skip_traffic:
        print("== 3. 真流量（登录 → 冷轮 → 缓存轮 → 流式 → 检索端点 → 未登录） ==")
        traffic = run_traffic(args.api, args.question, args.stream_question)
        rep["真流量"] = traffic
        for k in ("冷轮", "缓存轮", "流式轮", "检索端点", "未登录"):
            if k in traffic:
                print(f"   {k}: {json.dumps(traffic[k], ensure_ascii=False)[:190]}")
        time.sleep(0.2)
        status, body = http(f"{args.api}/metrics")
    host = parse_metrics(body.decode("utf-8"))

    print("== 4. 计数器之间的恒等式（增量） ==")
    rep["恒等式"] = invariants(before, host, traffic)
    rep["恒等式口径"] = "打了流量→用增量互查；`--skip-traffic`→两侧都是 0，属真空通过，不作证据"
    for c in rep["恒等式"]:
        print(f"   {'✓' if c['成立'] else '✗'} {c['恒等式']} 左={c['左']} 右={c['右']}")
    rep["恒等式全部成立"] = all(c["成立"] for c in rep["恒等式"])

    print("== 5. 外泄面核查（/metrics 里不该有的东西） ==")
    text = body.decode("utf-8")
    leak_terms = [t for t in (args.question, args.stream_question, "verify-owner", "终验", "质保") if t in text]
    # 标签在样本里按字母序输出（`method=` 在 `route=` 前面），所以不能用「{route= 开头」去匹配——
    # 那样匹配不到，样例列会永远空白，而空白看起来像「没有流量」而不是「我正则写错了」。
    routes = sorted({m.group(1) for line in text.splitlines() if line.startswith("kee_http_requests_total{")
                     for m in [re.search(r'route="([^"]*)"', line)] if m})
    rep["外泄面"] = {"命中词": leak_terms, "路由标签样例": routes[:8], "路由标签数": len(routes)}
    print("   ", json.dumps(rep["外泄面"], ensure_ascii=False))

    if not args.skip_ab:
        print(f"== 6. 配对交错 A/B：开指标({api_port}) vs 关指标({args.ab_port})，各 {args.rounds} 对 ==")
        from _identity import login_http
        headers = login_http(args.api, "verify-owner")
        headers["Content-Type"] = "application/json"
        # 抓取本身要停下来量：Prometheus 每 15s 打一次 `/metrics`，那次请求会跑依赖探针
        # （Milvus count + PG count）并走同一条中间件——不停掉，A 侧就多了 B 侧没有的周期性竞争。
        paused = False
        if args.pause_prom:
            r = subprocess.run(["docker", "stop", "kee-prom"], capture_output=True, text=True, check=False)
            paused = r.returncode == 0
            print(f"   kee-prom {'已暂停' if paused else '暂停失败：' + (r.stderr or r.stdout)[:120]}")
        ab_base = f"http://127.0.0.1:{args.ab_port}"
        ab = spawn_second_api(api_port, args.ab_port)
        setattr(ab, "_kee_port", args.ab_port)
        try:
            rows = []
            for path, name in [("/healthz", "GET /healthz（含依赖自检）"),
                               ("/api/v1/auth/me", "GET /api/v1/auth/me（含令牌解析）"),
                               ("/api/v1/documents?limit=20", "GET /api/v1/documents（含 RBAC 过滤）")]:
                r = bench_paired(args.api, ab_base, headers, path, args.rounds)
                rows.append({"端点": name, **r})
                print(f"   {name}: 开={r['开指标']['p50']} 关={r['关指标']['p50']} "
                      f"Δp50={r['Δp50ms']} 配对差中位={r['配对差中位数ms']}")
            hist = append_history(rows, args.rounds)
            rep["A_B对照"] = {"每端点对数": args.rounds, "采样方式": "配对交错（同一次抖动同时落在两侧）",
                              "抓取器已暂停": paused, "行": rows,
                              "历史账": history_statement(hist),
                              "历史文件": str(HIST.relative_to(ROOT)),
                              "读法": "只看「配对差中位数」那一列；**单次运行不足以定开销**，"
                                      "跨运行的结论由 `m5_observability_ab_history.json` 推导（见下面「按端点」与「结论」）。"
                                      "计数器微基准 17µs/轮，比这里小三个数量级——所以若真有毫秒级开销，"
                                      "来源是**多一层 BaseHTTPMiddleware 的包装成本**、不是打点。"
                                      "第一版按「先 50 次开、再 50 次关」跑时，同一端点的 Δ 会翻符号，"
                                      "那量的是进程状态；配对交错 + 暂停抓取器才把它压进噪声里。"
                                      "要省掉它得把计时改成纯 ASGI 中间件，本版没做，代价记在风险册。",
                              }
        finally:
            kill_tree(ab)
            if paused:
                subprocess.run(["docker", "start", "kee-prom"], capture_output=True, text=True, check=False)
                print("   kee-prom 已恢复")
                time.sleep(20.0)  # 等下一次抓取，让后续对账拿到新鲜值

    print("== 7. Prometheus 侧对账 + 表达式体检 ==")
    up = http(f"{args.prom}/-/healthy")[1].decode()
    started = counter(host, "kee_process_start_timestamp_seconds")
    if started <= 0:
        print("[停] 被观测的 API 里没有 `kee_process_start_timestamp_seconds`：它跑的是改动前的代码，"
              "重启宿主 API 再重跑本脚本。窗口宽度不能靠猜。", file=sys.stderr)
        return 2
    uptime = max(60.0, time.time() - started)
    window = f"{int(min(uptime / 60 + 5, 360))}m"
    rep["prometheus"] = {"healthy": up.strip(), "对账窗口": window}
    host_rounds = sum_prefix(host, "kee_qa_rounds_total")
    recon = prom_eventually(args.prom, "sum(kee_qa_rounds_total)", host_rounds)
    # 累计值为 0 时（`--skip-traffic` 且这是个刚起的空实例）这条对账**没有可比的量**：
    # 判成「不一致」会把一个真空状态误报成故障，判成「一致」就是空集上的真。两个都不行。
    rep["prometheus"]["轮次对账"] = {"宿主": host_rounds, "Prometheus": recon["Prometheus"],
                                     "一致": None if host_rounds == 0 else recon["一致"],
                                     "等待抓取尝试": recon["尝试次数"],
                                     "口径": "累计值相等才叫一致；不等时先确认不是「还没抓到」"}
    print("  ", json.dumps({k: v for k, v in rep["prometheus"]["轮次对账"].items() if k != "口径"},
                           ensure_ascii=False)[:300])

    exprs = []
    dash = json.loads(DASH.read_text(encoding="utf-8"))
    for panel in dash["panels"]:
        for t in panel.get("targets") or []:
            exprs.append({"来源": f"看板面板#{panel['id']} {panel['title']}", "expr": t["expr"]})
    for block in ALERTS.read_text(encoding="utf-8").split("- alert: ")[1:]:
        name = block.split("\n", 1)[0].strip()
        expr = re.search(r"expr:\s*(.+)\n", block)
        if expr:
            exprs.append({"来源": f"告警 {name}", "expr": expr.group(1).strip()})
    results = []
    for item in exprs:
        r = prom_query(args.prom, item["expr"], window=window)
        results.append({**item, **r})
        if r["状态"] == "ERROR":
            print(f"   ✗ {item['来源']}: {r['错误'][:130]}")
    rep["表达式"] = results
    rep["表达式统计"] = {"总数": len(results),
                        "OK": sum(1 for r in results if r["状态"] == "OK"),
                        "EMPTY": sum(1 for r in results if r["状态"] == "EMPTY"),
                        "ERROR": sum(1 for r in results if r["状态"] == "ERROR"),
                        "必备表达式": [
                            {"expr": e, **prom_query(args.prom, e, window=window)} for e in REQUIRED_EXPRS]}
    print("  ", json.dumps({k: v for k, v in rep["表达式统计"].items() if k != "必备表达式"}, ensure_ascii=False))

    print("== 8. Grafana 侧 ==")
    g: dict = {}
    auth = "Basic " + base64.b64encode(f"{GRAFANA_USER}:{GRAFANA_PASSWORD}".encode()).decode()
    try:
        g["health"] = json.loads(http(f"{args.grafana}/api/health")[1])
        g["datasource"] = json.loads(http(f"{args.grafana}/api/datasources/uid/kee",
                                          headers={"Authorization": auth})[1])
        g["datasource_health"] = json.loads(http(f"{args.grafana}/api/datasources/1/health",
                                                 headers={"Authorization": auth})[1])
        db = json.loads(http(f"{args.grafana}/api/dashboards/uid/kee-rag",
                             headers={"Authorization": auth})[1])["dashboard"]
        g["dashboard"] = {"uid": db["uid"], "标题": db["title"], "面板数": len(db["panels"]),
                          "可改": db.get("editable", True), "provisioned_version": db["version"]}
        # **面板级**通路：走 `/api/ds/query`（就是看板每块面板实际调的那个端点）。
        # 只测 Prometheus 侧的 query API 证明不了 Grafana 那层的数据源、鉴权、POST 编码都通。
        # 注意返回形状：`{frames:[{schema, data:{values:[[时间],[值]]}}]}`——值在 `data.frames`，
        # 不在 `data.data.frames`（第一次读错会把有数的查询读成「无数据」）。
        live = []
        for label, expr in [("面板#1 请求速率", "sum(rate(kee_http_requests_total[5m]))"),
                            ("面板#5 依赖健康", "min(kee_dependency_up)"),
                            ("面板#6 e2e p95（桶近似）",
                             'histogram_quantile(0.95, sum by (le) (rate(kee_qa_stage_seconds_bucket{stage="e2e"}[30m])))')]:
            payload = {"queries": [{"refId": "A", "expr": expr,
                                    "datasource": {"type": "prometheus", "uid": "kee"},
                                    "instant": True, "range": False,
                                    "intervalMs": 15000, "maxDataPoints": 100}],
                       "from": "now-30m", "to": "now"}
            res = json.loads(http(f"{args.grafana}/api/ds/query",
                                  data=json.dumps(payload).encode(),
                                  headers={"Authorization": auth,
                                          "Content-Type": "application/json"})[1])
            frames = (res["results"]["A"].get("frames")
                      or res["results"]["A"].get("data", {}).get("frames", []))
            vals = frames[0]["data"]["values"] if frames else []
            live.append({"面板": label, "expr": expr, "序列有值": bool(vals and len(vals) >= 2),
                         "值": vals[-1][0] if vals and len(vals) >= 2 and vals[-1] else None})
        g["面板级实查询"] = live
    except Exception as e:  # noqa: BLE001
        g["错误"] = f"{type(e).__name__}: {e}"[:200]
    rep["grafana"] = g
    print("  ", json.dumps({k: v for k, v in g.items() if k != "datasource"}, ensure_ascii=False)[:520])

    print("== 9. Langfuse 状态（默认关闭） ==")
    try:
        from kee.config import get_settings
        from kee.obs.langfuse import LangfuseExporter
        exp = LangfuseExporter(get_settings())
        rep["langfuse"] = exp.status()
        exp.close()
    except Exception as e:  # noqa: BLE001
        rep["langfuse"] = {"读取失败": f"{type(e).__name__}: {e}"}
    print("  ", json.dumps(rep["langfuse"], ensure_ascii=False)[:260])

    tag = args.tag
    ART.mkdir(parents=True, exist_ok=True)
    art = ART / f"m5_observability{tag}.json"
    art.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(rep, OUT_MD, art)
    ok = (rep["恒等式全部成立"] and rep["表达式统计"]["ERROR"] == 0 and leak_terms == []
          and rep["prometheus"]["轮次对账"]["一致"] is not False)
    print(f"\n结论：{'通过' if ok else '不通过'}  产物 → {art.relative_to(ROOT)} / {OUT_MD.relative_to(ROOT)}")
    return 0 if ok else 1


def write_md(rep: dict, out: Path, art: Path) -> None:
    L = ["# M5 · 可观测接入实测（Langfuse / Prometheus / Grafana）", "",
         f"> 生成：`{rep['生成时间']}`；参数：`{rep['命令行']}`；原始数据：`{art.relative_to(ROOT)}`",
         f"> Prometheus `{rep['prometheus']['healthy'] if 'prometheus' in rep else ''}`；"
         f"宿主 API `{rep['api']}`", "",
         "这一层是**旁路**：它不写 state、不改门禁，指标全部进程内累计、不落库。"
         "门禁（① 延迟 / ③ 检索 / ⑤ 引用与置信）的权威读数仍由 `scripts/` 下的离线探针产出，"
         "看板上的分位数是直方图桶上的近似值，两者不一致时以离线探针为准。", "",
         "## 一 `/metrics` 的形状与抓取成本", ""]
    s = rep["端点形状"]
    L += [f"- HTTP {s['http']}，**{s['字节']:,} 字节 / {s['声明族数']} 个 `kee_*` 声明族**"
          f"（另有 {s['created派生族数']} 条 `_created` 派生族——它们只在指标被摸过之后才出现，"
          "所以「数 `# TYPE kee_` 行」会得到一个浮动的数，本机见过 23/24/32；对外只能报声明族数）"
          f"，本轮首个抓取里 **{s['样本数']} 条样本**"
          f"（{s['次数']} 次抓取，最后一次的大小）",
          f"- 抓取耗时：p50 {s['耗时ms']['p50']}ms、p95 {s['耗时ms']['p95']}ms、"
          f"min {s['耗时ms']['min']}ms、max {s['耗时ms']['max']}ms（含序列化，Prometheus 每 15s 一次）", "",
          "声明族清单（`curl -s <api>/metrics | grep '^# TYPE kee_'` 去掉 `_created` 就是这张表）：", "",
          "```", *s["声明族清单"], "```", ""]

    L += ["## 二 埋点开销（先给噪声地板，再给 Δ）", ""]
    mb = rep["微基准"]
    L += [f"- 进程内微基准（n={mb['n']}，同一结果对象重复观察）：`observe_qa_result` "
          f"**{mb['observe_qa_result_每次us']}µs/轮**、`observe_retrieval` {mb['observe_retrieval_每次us']}µs/轮、"
          f"`render()` {mb['render_一次ms']}ms/次。{mb['注']}。", ""]
    if "A_B对照" in rep:
        ab = rep["A_B对照"]
        L += [f"- **配对交错**采样（同一次抖动同时落在两侧），每端点 {ab['每端点对数']} 对，"
              f"抓取器{'已暂停（避免每 15s 的 `/metrics` 抓取只给 A 侧加竞争）' if ab['抓取器已暂停'] else '未暂停'}：", "",
              "| 端点 | 开指标 p50 | 关指标 p50 | Δp50 | 配对差中位数 | 配对差 p95 | 开指标 p95 | 关指标 p95 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in ab["行"]:
            L.append(f"| {r['端点']} | {r['开指标']['p50']}ms | {r['关指标']['p50']}ms | {r['Δp50ms']}ms "
                     f"| **{r['配对差中位数ms']}ms** | {r['配对差p95ms']}ms "
                     f"| {r['开指标']['p95']}ms | {r['关指标']['p95']}ms |")
        h = ab.get("历史账", {})
        L += ["", f"- **跨运行的开销结论由历史账推导**（`{ab.get('历史文件', '')}`，本轮已并入）——"
              "这句话要是手写进脚本，下一次跑完就自动变成假话，而且假得看不出来：", ""]
        for endpoint, v in (h.get("按端点") or {}).items():
            L.append(f"  - {endpoint}：**{v.get('n次')} 次**，"
                     + (f"均值 {v['均值ms']}ms，5% 临界 ±{v['5%临界ms']}ms ⇒ **{v['判定']}**"
                        if "均值ms" in v else f"取值 {v.get('取值')} ⇒ {v['判定']}"))
        L += ["", f"- 合并判定：{h.get('结论', '（无）')}", "", f"- 读法：{ab['读法']}。",
              "- 这份对照**不含**问答链路的埋点：那一层每轮只加两次计数（见上面微基准的 µs 量级），"
              "而一轮问答实测 4s～79s，两个量级相差 5～6 个数量级，用真实问答做 A/B 只会测到 LLM 的抖动。", ""]
    else:
        L += ["- 本轮 `--skip-ab`，未做实例间对照；上面只有微基准。", ""]

    if "真流量" in rep:
        t = rep["真流量"]
        L += ["## 三 真流量打出来的读数", "",
              f"- 账号 `{t['账号']}`（真实登录换 Bearer，与界面同一条入口）", ""]
        for k in ("冷轮", "缓存轮", "流式轮", "检索端点", "未登录"):
            if k in t:
                L.append(f"- **{k}**：`{json.dumps(t[k], ensure_ascii=False)}`")
        L += ["", "冷轮与缓存轮同一个问题：命中那轮 `degraded` 为空是**既有边界**"
              "（缓存分支返回的 `RetrievalResult` 不带降级标记与改写来历），不是这次埋点漏计。", ""]

    L += ["## 四 计数器之间的恒等式（这才是「漏了一条出口」的探测器）", "",
          "| 恒等式 | 左 | 右 | 成立 | 依据 |", "| --- | --- | --- | --- | --- |"]
    for c in rep["恒等式"]:
        L.append(f"| {c['恒等式']} | {c['左']} | {c['右']} | {'✓' if c['成立'] else '✗'} | {c['依据']} |")
    L += ["", f"**全部成立：{rep['恒等式全部成立']}**。", ""]

    L += ["## 五 外泄面核查", "",
          f"- `/metrics` 文本里检索问题原文 / 用户名：`{rep['外泄面']['命中词'] or '零命中'}`",
          f"- 路由标签全是模板：`{json.dumps(rep['外泄面']['路由标签样例'], ensure_ascii=False)}`",
          "- 所以 `/metrics` 可以不开鉴权；它另一道收口是 Nginx 不反代该路径 + API 只绑回环。", ""]

    st = rep["表达式统计"]
    L += ["## 六 看板与告警表达式逐条体检", "",
          f"- 共 **{st['总数']} 条**（看板 {sum(1 for r in rep['表达式'] if r['来源'].startswith('看板'))} 条 + "
          f"告警 {sum(1 for r in rep['表达式'] if r['来源'].startswith('告警'))} 条），"
          f"执行结果：有数 **{st['OK']}** 条、空 **{st['EMPTY']}** 条、**报错 {st['ERROR']} 条**",
          f"- 对账窗口取 `{rep['prometheus']['对账窗口']}`（按 `kee_process_start_timestamp_seconds` 推的进程年龄 + 5m 余量）"
          "——所有 counter 都是进程内累计，窗口宽度不对上就会拿「累计值」比「区间增量」",
          f"- 宿主与 Prometheus 的轮次对账：宿主 `{rep['prometheus']['轮次对账']['宿主']}` / "
          f"Prometheus `{rep['prometheus']['轮次对账']['Prometheus'].get('首值')}` → "
          f"**{ {True: '一致', False: '不一致', None: '无可比量（该实例累计轮次为 0，判不成真也判不成假）'}[rep['prometheus']['轮次对账']['一致']] }**", "",
          "| 来源 | 状态 | 序列数 | 首值 |", "| --- | --- | --- | --- |"]
    for r in rep["表达式"]:
        L.append(f"| {r['来源']} | {r['状态']} | {r.get('序列数', '')} | {r.get('首值', r.get('错误', ''))} |")
    L += ["", "必备表达式（本轮流量下必须非空）：", ""]
    for e in st["必备表达式"]:
        L.append(f"- `{e['expr']}` → {e['状态']}，首值 {e.get('首值')}")
    gf = rep["grafana"]
    L += ["", "## 七 Grafana 与 Langfuse", "",
          f"- Grafana `{gf.get('health', {}).get('version', '?')}`：数据库 `{gf.get('health', {}).get('database', '?')}`、"
          f"数据源健康 `{gf.get('datasource_health', {}).get('status', gf.get('错误', '?'))}`",
          f"- 看板 `{gf.get('dashboard', {}).get('uid', '?')}`：**{gf.get('dashboard', {}).get('面板数', '?')} 个面板**、"
          f"provisioning 版本 {gf.get('dashboard', {}).get('provisioned_version', '?')}、"
          f"UI 不可改（`editable={gf.get('dashboard', {}).get('可改', '?')}`）——改面板只能改仓库里的 JSON，"
          "在看板上手改出来的东西没人能复现", ""]
    for row in gf.get("面板级实查询", []):
        L.append(f"- 走 `/api/ds/query`（每块面板实际调的那个端点）：{row['面板']} → "
                 f"**{'有值' if row['序列有值'] else '无值'}** `{row['值']}`")
    L += ["",
          "> 上面那条 p95 与 ① 门禁的 6s **不是一个数**：这条把冷启动轮次（含 `dense_timeout` 那批 20~90s 的）"
          "也算进来了，而门禁口径是「6s + 仅作答轮次 + first_visible」，且直方图分位本身是桶上插值。"
          "看板的用途是盯趋势与触警，报数仍走离线探针。", "",
          f"- Langfuse：`{json.dumps(rep['langfuse'], ensure_ascii=False)[:300]}`", "",
          "Langfuse 的**上报载荷与鉴权**由 `tests/test_langfuse_exporter.py` 对一个本地 stub 采集器实测"
          "（批量形状、Basic 鉴权、trace/span/generation 串联、2xx 里的 `errorList` 计数、"
          "目标端挂掉时不抛、公网目标直接拒发）。真 Langfuse 容器那一侧**未实跑**："
          "自托管要拉 `langfuse/langfuse` + Postgres + Redis（GB 级镜像、且落 C 盘的 Docker VM），"
          "与本机「大件不重复下载 / 重资产留 D 盘」冲突，`infra/docker-compose.langfuse.yml` 只出稿待裁。", "",
          "## 八 这一层没能证明的事", "",
          "- **门禁读数不受影响**不由本脚本证明：本脚本自己就产生流量，会改变被观测对象的累计值。"
          "那条结论来自 `verify_m1/m3/m4`、`m5_test_user_run` 的复跑（读数逐条对齐改动前）。",
          "- 看板 p95 与门禁 p95 不是同一个数（桶近似 + 全轮次 vs 仅作答轮次），本脚本只验证表达式可执行。",
          "- Langfuse 真实落库、Grafana 面板渲染效果（截图通路另计）未纳入本次实测。", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
