"""⑥ 的加强版断网验证：容器无出网通路时，核心链路能否端到端跑完。

与 Gate 0 ⑥ 的区别写在 `infra/docker-compose.gate0.nodev.yml` 头里：
⑥ 用的 `internal: true` 连宿主↔容器通路一起切了，所以它只能证「容器封公网」，
证不了「断网还能问答」。本脚本用 `enable_ip_masquerade: false`（只关 SNAT）把这条补上。

拓扑前提：API 在宿主（ADR-12 未裁决），Ollama 在宿主，存储与探针在容器。

跑法：uv run python scripts/m4_airgap_check.py
副作用：存储容器会被 down/up 重建一轮（**数据卷不动**），kee-web 会先 stop 后 --force-recreate 起回。
        无论中途成败，finally 都会复原；复原后自检「文档数/chunk 数与跑前一致」，不一致就打 FAILED。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _identity import ensure_accounts, login_http  # noqa: E402

MAIN = "infra/docker-compose.gate0.yml"
NODEV = "infra/docker-compose.gate0.nodev.yml"
M4 = "infra/docker-compose.m4.yml"
# 宿主 API 端口不是常量（同上：Windows 保留段），换端口时传 KEE_API_PORT
API = f"http://127.0.0.1:{os.environ.get('KEE_API_PORT', '38000')}"
OUT = ROOT / "var" / "gate0" / "m4_airgap_nodev.json"

# 身份走登录令牌（`KEE_ALLOW_HEADER_IDENTITY` 的默认已于 2026-10-15 翻成 False，用户裁「交付物默认 false、脚本显式带令牌」），
# TOKEN 在 main() 里登录后填充：审计里的 actor 因此是账号名 verify-owner，不再是请求头里随手写的 airgap-probe。
TOKEN = ""

PUBLIC_TARGETS = ["http://1.1.1.1", "http://8.8.8.8", "https://registry-1.docker.io/v2/"]


def sh(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def compose(*files: str, args: list[str], timeout: int = 400) -> dict:
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", f]
    cmd += args
    r = sh(cmd, timeout=timeout)
    return {"rc": r.returncode, "out": r.stdout.strip()[:400], "err": r.stderr.strip()[:400]}


def health() -> dict:
    r = sh(["curl", "-s", "-m", "10", f"{API}/healthz"])
    try:
        return json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return {"status": "unreachable", "raw": r.stdout[:120]}


def counts(h: dict) -> tuple:
    d = h.get("deps", {})
    return (d.get("milvus", {}).get("chunks"), d.get("postgres", {}).get("documents"))


def docker_up_running() -> bool:
    r = sh(["docker", "ps", "--format", "{{.Names}}"], timeout=60)
    names = set(r.stdout.split())
    return {"kee-milvus", "kee-postgres", "kee-etcd"} <= names


def start_api() -> tuple[subprocess.Popen | None, bool]:
    """核心链路测试要走真实 HTTP 服务，所以这里自己把宿主 API 拉起来。

    返回 (进程, 是否复用已有服务)。判定必须看**依赖真的通**，不能只看 `/healthz` 有回话——
    存储栈整个没起来时 API 照样会答 `degraded`，那时候复用等于把「跑前对照」写成假的。
    """
    if counts(health())[0] is not None:
        return None, True
    p = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "serve_api.py")],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(45):
        if counts(health())[0] is not None:
            return p, False
        time.sleep(2)
    return p, False


def stop_api(p: subprocess.Popen | None) -> None:
    if p is None:
        return
    p.terminate()
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:  # noqa: BLE01
        p.kill()


def wait_ready(timeout_sec: int = 240) -> dict:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_sec:
        h = health()
        d = h.get("deps", {})
        if d.get("milvus", {}).get("ok") and d.get("postgres", {}).get("ok"):
            return {"ok": True, "waited_sec": round(time.perf_counter() - t0, 1), "health": h}
        time.sleep(5)
    return {"ok": False, "waited_sec": timeout_sec, "health": health()}


def container_egress(container: str) -> list[dict]:
    """在容器里打公网靶：期望全部失败。curl 退出码 7=连不上、6=解析不了、28=超时。"""
    res = []
    for url in PUBLIC_TARGETS:
        r = sh(["docker", "exec", container, "curl", "-sS", "-m", "6", "-o", "/dev/null", "-w", "%{http_code}", url], timeout=30)
        res.append({"url": url, "rc": r.returncode, "http": r.stdout.strip()[:8], "err": r.stderr.strip()[:80]})
    dns = sh(["docker", "exec", container, "getent", "hosts", "github.com"], timeout=20)
    res.append({"url": "dns:github.com", "rc": dns.returncode, "http": dns.stdout.strip()[:40], "err": dns.stderr.strip()[:60]})
    return res


def container_mesh() -> dict:
    """容器间互联必须仍然可用（否则「断网」是把系统自己弄死，不是有效测试）。"""
    etcd = sh(["docker", "exec", "kee-milvus", "curl", "-s", "-m", "6", "http://kee-etcd:2379/health"], timeout=30)
    return {"etcd_health": etcd.stdout.strip()[:80] or etcd.stderr.strip()[:80], "rc": etcd.returncode}


def ask(question: str) -> dict:
    body = json.dumps({"question": question}, ensure_ascii=False)
    r = sh(
        [
            "curl", "-s", "-m", "180", "-X", "POST", f"{API}/api/v1/answers",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {TOKEN}",
            "-d", body,
        ],
        timeout=200,
    )
    try:
        d = json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return {"parse_error": r.stdout[:200]}
    # 响应是**扁平**的：`{**res.as_dict(), session_id, turn_no, new_session}`，没有 `answer` 这层外壳。
    # 之前按 d["answer"][...] 取，取到的是空 dict → 报出「status=answer 但 0 字 0 引用」这种自相矛盾的数。
    return {
        "status": d.get("status"),
        "结论字数": len(d.get("conclusion", "") or ""),
        "引用数": len(d.get("citations", []) or []),
        "needs_human": d.get("needs_human"),
        "turn_no": d.get("turn_no"),
    }


def stream_first_event(timeout: int = 120) -> dict:
    """流式端点在断网容器下是否仍按 retrieval → delta 出帧（只取前几帧，不等全文）。"""
    r = sh(
        [
            "curl", "-s", "-N", "-m", str(timeout), "-X", "POST", f"{API}/api/v1/answers/stream",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {TOKEN}",
            "-d", json.dumps({"question": "银企直联方案 V3.2 的 RTO 是多少？"}, ensure_ascii=False),
        ],
        timeout=timeout + 30,
    )
    kinds = [ln[6:] for ln in r.stdout.splitlines() if ln.startswith("event:")]
    return {"事件序列": kinds[:6], "事件数": len(kinds)}


def main() -> int:
    # 上一轮被中断过一次，存储栈可能整个不在——先把它补起来，否则「跑前对照」无从谈起。
    if not docker_up_running():
        print("存储栈未在跑，先 compose up（数据卷不动）")
        compose(MAIN, args=["up", "-d", "--wait", "--wait-timeout", "240"], timeout=500)
    api, reused = start_api()
    print(f"宿主 API：{'复用已在监听的进程' if reused else f'本脚本自己起（PID {api.pid}，结束时会停掉）'}")
    pre = health()
    pre_counts = counts(pre)
    if pre_counts[0] is None:
        print("[拒绝开始] 宿主 API 起不来或依赖未就绪，跑前状态就无法作对照。")
        stop_api(api)
        return 2
    print(f"跑前：chunks={pre_counts[0]} docs={pre_counts[1]}")

    # 令牌在断网步骤之前取好：会话行存在数据卷里，容器重建后这枚令牌照样解得开
    global TOKEN
    ensure_accounts()
    TOKEN = login_http(API, "verify-owner")["Authorization"].removeprefix("Bearer ")

    rep: dict = {
        "env": {"ts": datetime.now().isoformat(timespec="seconds"), "commit": sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()},
        "pre": {"health": pre, "counts": list(pre_counts)},
        "steps": {},
    }
    core: dict = {}
    try:
        # kee-web 与存储栈同属项目 kee-gate0，它的 endpoint 会挡住网络删除（⑥ 复跑第一轮就是这么空跑的）
        rep["steps"]["stop_web"] = compose(M4, args=["stop", "kee-web"])
        rep["steps"]["down_main"] = compose(MAIN, args=["down", "--remove-orphans"], timeout=300)
        rep["steps"]["up_nodev"] = compose(MAIN, NODEV, args=["up", "-d", "--wait", "--wait-timeout", "240"], timeout=500)
        if rep["steps"]["up_nodev"]["rc"] != 0:
            # 起不来就别往下测：上一轮正是 compose schema 拒了非法字段导致 up 失败，
            # 而探针继续跑会把「容器根本不在」读成「封网成功」。
            core["blocked"] = f"nodev 变体 up 失败 rc={rep['steps']['up_nodev']['rc']}：{rep['steps']['up_nodev']['err'][:200]}"
            return 3
        rep["steps"]["ready"] = wait_ready()

        nets = sh(["docker", "network", "inspect", "kee-gate0_kee-net", "-f", "{{json .Options}}"])
        rep["steps"]["network_options"] = nets.stdout.strip()[:300]

        if not rep["steps"]["ready"]["ok"]:
            core["blocked"] = "断网变体起来后依赖没就绪，后续读数全部作废"
            return 3

        core["容器出网探测"] = container_egress("kee-milvus")
        core["容器间互联"] = container_mesh()
        core["宿主→容器发布端口"] = {"healthz": rep["steps"]["ready"]["health"]}
        core["核心链路_单轮问答"] = ask("银企直联实时清算方案 V3.2 的灾备切换目标里，RTO 要求是多少？")
        core["核心链路_流式"] = stream_first_event()
    finally:
        rep["steps"]["down_nodev"] = compose(MAIN, NODEV, args=["down", "--remove-orphans"], timeout=300)
        rep["steps"]["up_main"] = compose(MAIN, args=["up", "-d", "--wait", "--wait-timeout", "240"], timeout=500)
        rep["steps"]["restore_web"] = compose(M4, args=["up", "-d", "--force-recreate", "kee-web"], timeout=300)
        back = wait_ready()
        post = health()
        rep["post"] = {"ready": back, "health": post, "counts": list(counts(post))}
        core["复原后_计数一致"] = counts(post) == pre_counts
        stop_api(api)

    blocked = [c for c in core.get("容器出网探测", []) if c.get("rc") == 0 and c.get("http") not in ("", "000")]
    mesh_ok = core.get("容器间互联", {}).get("rc") == 0
    ans = core.get("核心链路_单轮问答", {})
    chain_ok = ans.get("status") == "answer" and (ans.get("引用数") or 0) >= 1
    core["判定"] = {
        "容器封公网": bool(core.get("容器出网探测")) and not blocked,
        "容器间互联可用": mesh_ok,
        "宿主可达发布端口": rep["steps"]["ready"]["ok"],
        "断网下核心链路跑通": chain_ok,
        "复原无损伤": core.get("复原后_计数一致", False),
    }
    rep["core"] = core
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n--- 容器出网探测 ---")
    for c in core.get("容器出网探测", []):
        print(f"  {c['url']:34s} rc={c['rc']} http={c['http'] or '-'} {c['err'][:50]}")
    print("--- 容器间互联 ---", json.dumps(core.get("容器间互联"), ensure_ascii=False))
    print("--- 核心链路 ---")
    print("  单轮:", json.dumps(ans, ensure_ascii=False))
    print("  流式:", json.dumps(core.get("核心链路_流式"), ensure_ascii=False))
    print("--- 判定 ---", json.dumps(core["判定"], ensure_ascii=False))
    print("--- 复原 --- counts", rep["post"]["counts"], "一致:", core.get("复原后_计数一致"))
    print("证据：", OUT)
    return 0 if all(core["判定"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
