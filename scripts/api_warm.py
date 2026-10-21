"""冷启动预热：把「模型冷装载」的账从验收读数里挪出来，单独打印。

为什么需要它（2026-10-16 第一次全量复验照出来的）：`verify_m1` / `verify_m3` 要在自己进程里
加载嵌入与重排，8GB 卡零余量，所以复验脚本必须先停宿主 API、跑完再起回来。起回来那一刻进程是
**冷的**：`/healthz` 立刻 200（它不碰模型），但第一个真问答要把 BGE-M3 与 reranker 装进一张
已被 Ollama 占掉 4.7GB 的卡上。本轮实测该请求 **>180s 没有首字节**，于是
`verify_m4`（httpx 超时 180s）与观测探针（真流量段）把「服务刚起来」读成了三次失败。

这不是测量，是没预热。所以这里只做一件事：登录 → 打一次非流式问答 → 报告耗时。
不写产物、不进门禁读数；冷装载的耗时**照样打印出来**，冷/热两个数分开记是本仓库的纪律
（首个可见 delta 冷 21118ms / 热 1218ms 那次就是这么定的口径）。

跑法：
    uv run python scripts/api_warm.py --api http://127.0.0.1:28000
退出码：0 预热完成；1 预热失败或超预算（这时**不要**继续跑真 HTTP 验收，读数只会是假的）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from _identity import BY_ROLE, ensure_accounts, login_http  # noqa: E402

# 这道题在本语料里稳定出结论（`verify_m4` 用的就是它），拒答轮不会走生成，预热不到 7B。
QUESTION = "SIS 安全仪表系统采用的表决逻辑是什么？"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:28000")
    ap.add_argument("--role", default="pm", help="用哪个角色的验收账号（见 _identity.BY_ROLE）")
    ap.add_argument("--budget-sec", type=float, default=600.0,
                    help="冷装载的容忍上限；超了就判预热失败，别让后面的验收带病读数")
    args = ap.parse_args()

    base = args.api.rstrip("/")
    username = BY_ROLE.get(args.role)
    if not username:
        print(f"未知角色 {args.role}，可选：{sorted(BY_ROLE)}")
        return 1

    t0 = time.perf_counter()
    ensure_accounts()
    headers = login_http(base, username)
    login_ms = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    try:
        r = httpx.post(f"{base}/api/v1/answers",
                       json={"question": QUESTION, "top_k": 5},
                       headers=headers, timeout=args.budget_sec)
    except Exception as e:  # noqa: BLE001
        print(f"预热请求没打通（{type(e).__name__}: {e}）——这一步就该慢，但不该失败到读不出东西")
        return 1
    warm_ms = round((time.perf_counter() - t1) * 1000)

    if r.status_code != 200:
        print(f"预热返回 {r.status_code}：{r.text[:200]}")
        return 1
    body = r.json()
    timings = body.get("timings_ms") or {}
    print(f"预热完成 登录 {login_ms}ms ｜ 首个问答 {warm_ms}ms ｜ status={body.get('status')} "
          f"引用 {len(body.get('citations') or [])} 条 ｜ 阶段 "
          + json.dumps({k: timings.get(k) for k in ("retrieval_ms", "generate_ms", "e2e_ms")},
                       ensure_ascii=False))
    if warm_ms / 1000.0 > args.budget_sec:
        print(f"超出预算 {args.budget_sec}s，后面的验收别指望热态读数")
        return 1
    # 再打一次：第二次才代表「热态」，两个数一起报出来才有对照意义
    t2 = time.perf_counter()
    r2 = httpx.post(f"{base}/api/v1/answers", json={"question": QUESTION, "top_k": 5},
                    headers=headers, timeout=args.budget_sec)
    print(f"热态复打 {round((time.perf_counter() - t2) * 1000)}ms ｜ status={r2.json().get('status')}"
          f"（第二次可能命中检索缓存，所以它是「进程已热」的下界，不是热态延迟的读数）")
    return 0 if r2.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
