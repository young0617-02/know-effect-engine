"""Gate 0 ⑥：单机 Docker Compose 全组件本地化与数据不出域（必备项）。

证据链：一键起栈 → 仅绑定 127.0.0.1 → 容器出网被切断时核心链路仍可跑 →
宿主进程连接审计 → 代码/配置出站端点静态扫描 → 云端兜底开关为关。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.gate0.common import (  # noqa: E402
    PROJECT_ROOT,
    env_snapshot,
    md_table,
    verdict,
    write_json,
    write_md,
)

COMPOSE_MAIN = "infra/docker-compose.gate0.yml"
COMPOSE_ISO = "infra/docker-compose.gate0.iso.yml"
SERVICES = ["kee-etcd", "kee-minio", "kee-milvus", "kee-postgres", "kee-redis"]

# 允许的出站目标：回环 + 私网（Docker/WSL 网段）+ 本机文件路径
ALLOWED_REMOTE_PREFIXES = ("127.", "::1", "0.0.0.0")
PRIVATE_RANGES = re.compile(
    r"^(10\.|127\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|255\.255\.255\.255|169\.254\.)"
)
URL_RE = re.compile(r"https?://([A-Za-z0-9_.\-]+)(:[0-9]+)?", re.I)
LOCAL_ALLOWLIST = {
    "localhost",
    "0.0.0.0",
    "host.docker.internal",
    "etcd",
    "minio",
    "kee-milvus",
    "kee-etcd",
    "kee-minio",
    "milvus",
}
# 连通性探测专用地址：它们是「出网测试的靶子」，不是数据出域的目的地
PROBE_HOSTS = ("1.1.1.1", "8.8.8.8", "detectportal", "example.com", "ipinfo", "ifconfig.me")


def host_of(url: str) -> str:
    m = URL_RE.search(url)
    return m.group(1) if m else url


def sh(cmd: list[str], timeout: int = 300, check: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, check=check)


def compose(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return sh(["docker", "compose", "-f", COMPOSE_MAIN, *args], timeout=timeout)


def wait_for_milvus(timeout_sec: int = 150) -> dict[str, Any]:
    """一键启动后的就绪轮询：Milvus 在 etcd 竞态下会 panic 退出，靠 restart 策略自愈，
    宿主侧必须等它真正可用，否则「启动成功率」会被伪造成失败。"""
    from pymilvus import MilvusClient

    from kee.config import get_settings

    s = get_settings()
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout_sec:
        try:
            c = MilvusClient(uri=s.milvus_uri, timeout=10)
            v = c.get_server_version()
            c.close()
            return {"ok": True, "waited_sec": round(time.perf_counter() - t0, 1), "server": v}
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(3)
    return {"ok": False, "waited_sec": timeout_sec, "error": last}


def step_validate_and_up(rebuild: bool, from_scratch: bool = False) -> dict[str, Any]:
    cfg = compose("config", "--quiet")
    res: dict[str, Any] = {"config_ok": cfg.returncode == 0, "config_error": cfg.stderr[:300]}
    if rebuild or from_scratch:
        # down -v 会连数据卷一起删（Milvus 集合与 PG 权威源全没），只在 from_scratch 显式要求时用；
        # 常规 --rebuild 只重建容器不动数据，否则会把已建好的索引抹掉。
        args = ["down", "-v", "--remove-orphans"] if from_scratch else ["down", "--remove-orphans"]
        compose(*args, timeout=300)
        res["volume_cleanup"] = "down -v（数据卷已清空，需重新建库）" if from_scratch else "down（保留数据卷）"
    t0 = time.perf_counter()
    up = compose("up", "-d", "--wait", "--wait-timeout", "240", timeout=400)
    res["up_rc"] = up.returncode
    res["up_error"] = up.stderr[:400]
    res["startup_sec"] = round(time.perf_counter() - t0, 1)
    ps = compose("ps", "--format", "{{.Name}}\t{{.Status}}\t{{.Ports}}")
    lines = [l.split("\t") for l in ps.stdout.strip().splitlines() if l]
    res["services"] = [{"name": p[0], "status": p[1] if len(p) > 1 else "", "ports": p[2] if len(p) > 2 else ""} for p in lines]
    res["all_healthy"] = all("healthy" in s["status"] and "unhealthy" not in s["status"] for s in res["services"]) and len(
        res["services"]
    ) == len(SERVICES)
    return res


def step_loopback_binding(res: dict[str, Any]) -> dict[str, Any]:
    bad = []
    for s in res["services"]:
        for m in re.finditer(r"(\d+\.\d+\.\d+\.\d+):(\d+)->", s["ports"]):
            if not m.group(1).startswith("127."):
                bad.append(f"{s['name']}:{m.group(0)}")
    published = [
        f"{s['name']} {m.group(2)}" for s in res["services"] for m in re.finditer(r"([\d.]+):(\d+)->", s["ports"])
    ]
    return {"non_loopback_bindings": bad, "published_ports": published}


def step_connectivity() -> dict[str, Any]:
    from kee.config import get_settings

    s = get_settings()
    out: dict[str, Any] = {}
    try:
        from pymilvus import MilvusClient

        c = MilvusClient(uri=s.milvus_uri, timeout=20)
        out["milvus"] = {"ok": True, "server": c.get_server_version(), "collections": len(c.list_collections())}
        c.close()
    except Exception as e:  # noqa: BLE001
        out["milvus"] = {"ok": False, "error": str(e)[:200]}
    try:
        import psycopg

        with psycopg.connect(s.postgres_dsn, connect_timeout=10) as conn:
            out["postgres"] = {"ok": True, "version": conn.execute("select version()").fetchone()[0][:30]}
    except Exception as e:  # noqa: BLE001
        out["postgres"] = {"ok": False, "error": str(e)[:200]}
    try:
        import redis

        r = redis.Redis.from_url(s.redis_url, socket_connect_timeout=5)
        out["redis"] = {"ok": bool(r.ping()), "version": r.info("server")["redis_version"]}
        r.close()
    except Exception as e:  # noqa: BLE001
        out["redis"] = {"ok": False, "error": str(e)[:200]}
    try:
        from kee.llm.ollama import OllamaClient

        c = OllamaClient()
        out["ollama"] = {"ok": c.model_available(), "model": c.model, "resident": c.is_resident()}
        c.close()
    except Exception as e:  # noqa: BLE001
        out["ollama"] = {"ok": False, "error": str(e)[:200]}
    return out


def host_connections(pid: int) -> list[dict[str, str]]:
    """宿主侧该进程的全部 TCP 连接（用于「无外呼」的动态证据）。

    State 是枚举，ConvertTo-Json 会打成数字，这里显式转字符串，
    否则过滤条件永不命中会把「无外呼」伪造成真。
    """
    ps = (
        "Get-NetTCPConnection -OwningProcess "
        f"{pid} -ErrorAction SilentlyContinue | "
        "Select-Object RemoteAddress,RemotePort,@{n='State';e={[string]$_.State}} | ConvertTo-Json -Compress"
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60
    )
    txt = (r.stdout or "").strip()
    if not txt:
        return []
    data = json.loads(txt)
    if isinstance(data, dict):
        data = [data]
    return [{"remote": d.get("RemoteAddress", ""), "port": str(d.get("RemotePort", "")), "state": d.get("State", "")} for d in data]


def step_core_path_with_audit() -> dict[str, Any]:
    """跑一遍核心链路（嵌入→混合检索→生成），期间审计本进程出网连接。"""
    import threading

    from kee.config import get_settings
    from kee.llm.ollama import OllamaClient
    from kee.models.registry import ResidencyScheduler
    from kee.retrieval.milvus_store import KbStore

    mypid = os.getpid()
    observed: list[dict[str, str]] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                observed.extend(host_connections(mypid))
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.3)

    kb = KbStore("kb_chunks_iso_probe", analyzer="jieba")
    kb.drop()
    kb.ensure()
    sched = ResidencyScheduler(policy="co_resident")
    emb = sched.acquire("embed")
    rows = []
    for i, t in enumerate(["P1 级重大故障须在十五分钟内启动应急响应流程。", "配网线损率从 6.8% 降至 4.9%。"]):
        rows.append(
            {
                "chunk_id": f"iso#{i}",
                "doc_id": "iso",
                "version_id": "iso#v1",
                "parent_id": "",
                "industry": "能源",
                "customer": "某省级电网公司",
                "project": "PRJ-ENE-2024-001",
                "doc_type": "运维手册",
                "source_level": "B",
                "security_level": 2,
                "section_no": "1.1",
                "page": 1,
                "order_no": i,
                "text": t,
                "dense": emb.embed_query(t),
            }
        )
    kb.insert(rows)
    kb.flush()
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    vec = emb.embed_query("P1 级故障多久要启动响应")
    dense_hits = kb.search_dense(vec, limit=2)
    sparse_hits = kb.search_sparse("P1 级故障 响应", limit=2)
    cli = OllamaClient()
    llm = cli.chat(
        [{"role": "user", "content": '只输出 JSON：{"conclusion":"...","citations":["E1"]}。证据 E1：P1 级重大故障须在十五分钟内启动应急响应。问题：多久响应？'}],
        json_mode=True,
        max_tokens=120,
    )
    sched.release_all()
    kb.drop()
    kb.close()
    stop.set()
    t.join(timeout=5)

    uniq = sorted({(c["remote"], c["port"], c["state"]) for c in observed})
    skip = {"", "0.0.0.0", "::", "::1", "127.0.0.1"}
    # 保守判定：不看状态，只要远端不是回环/未指定地址就算一次潜在出网
    remote = [c for c in observed if c["remote"] not in skip and not c["remote"].startswith("127.")]
    public = [c for c in remote if not PRIVATE_RANGES.match(c["remote"])]
    return {
        "dense_hits": len(dense_hits),
        "sparse_hits": len(sparse_hits),
        "llm_ok": llm.ok,
        "llm_json_valid": llm.json_valid,
        "llm_metrics": llm.metrics,
        "observed_connections_uniq": sorted(uniq),
        "non_loopback_connections": remote,
        "public_egress_connections": public,
    }


def parse_proc_net_tcp(text: str) -> list[dict[str, str]]:
    """解析 /proc/net/tcp(6)：返回 [{remote, port, state}]，点分十进制表示。"""
    rows: list[dict[str, str]] = []
    for line in text.splitlines()[1:]:
        cols = line.split()
        if len(cols) < 4:
            continue
        rem, state = cols[2], cols[3]
        addr, _, port = rem.partition(":")
        if len(addr) == 8:  # IPv4 little-endian hex
            ip = ".".join(str(int(addr[i : i + 2], 16)) for i in (6, 4, 2, 0))
        else:
            ip = addr[:8] + "…"  # IPv6 折叠显示
        rows.append({"remote": ip, "port": str(int(port, 16)), "state": state})
    return rows


def container_connections(container: str) -> dict[str, Any]:
    """容器侧动态出网审计：ESTAB（state 01）且远端非回环/非 Docker 私网即视为可疑。"""
    out: dict[str, Any] = {"container": container}
    rows: list[dict[str, str]] = []
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        r = sh(["docker", "exec", container, "cat", f], timeout=30)
        if r.returncode == 0:
            rows.extend(parse_proc_net_tcp(r.stdout or ""))
    estab = [x for x in rows if x["state"] == "01"]
    sus = [
        x
        for x in estab
        if not (x["remote"].startswith(("127.", "172.", "10.", "192.168.")) or x["remote"] in ("0.0.0.0", "00000000…"))
    ]
    out["n_estab"] = len(estab)
    out["suspicious_external"] = sus
    out["peers"] = sorted({(x["remote"], x["port"]) for x in estab})
    return out


def step_container_egress_probe(internal_mode: bool) -> dict[str, Any]:
    """容器出网可达性探测 + 容器侧连接审计（数据面是否真的不需要外网）。"""
    out: dict[str, Any] = {"internal_mode": internal_mode}
    r = sh(["docker", "exec", "kee-milvus", "curl", "-m", "6", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "http://1.1.1.1/"], timeout=40)
    out["milvus_to_public"] = {"rc": r.returncode, "body": (r.stdout or r.stderr)[:160]}
    r2 = sh(["docker", "exec", "kee-milvus", "curl", "-m", "6", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:9091/healthz"], timeout=40)
    out["milvus_healthz_in_container"] = (r2.stdout or "").strip()
    r3 = sh(["docker", "exec", "kee-milvus", "curl", "-m", "6", "-s", "http://kee-etcd:2379/health"], timeout=40)
    out["etcd_peer_reachable"] = (r3.stdout or r3.stderr)[:160]
    public_hit = out["milvus_to_public"]["rc"] == 0 and (out["milvus_to_public"]["body"] or "").strip() not in ("", "000")
    out["containers_blocked_from_public"] = internal_mode and not public_hit
    out["containers_can_reach_public"] = (not internal_mode) and public_hit
    out["connections"] = {c: container_connections(c) for c in SERVICES}
    out["any_suspicious_external"] = any(
        v["suspicious_external"] for v in out["connections"].values() if isinstance(v, dict)
    )
    return out


def step_static_scan() -> dict[str, Any]:
    roots = ["src", "infra", "scripts", ".env.example", "pyproject.toml"]
    findings = []
    for rel in roots:
        p = PROJECT_ROOT / rel
        files = [p] if p.is_file() else list(p.rglob("*")) if p.is_dir() else []
        for f in files:
            if not f.is_file() or f.suffix not in (".py", ".yml", ".yaml", ".toml", ".example", ".json", ".ps1"):
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            for m in URL_RE.finditer(text):
                host = m.group(1).lower()
                if host.startswith("127.") or host in LOCAL_ALLOWLIST:
                    continue
                findings.append({"file": str(f.relative_to(PROJECT_ROOT)), "url": m.group(0), "line_hint": text[: m.start()].count("\n") + 1})
    # 分类：出域嫌疑 vs 连通性探测地址 vs 容器内部名，三类不能混为一谈
    probe_hits = [f for f in findings if any(h in f["url"] for h in PROBE_HOSTS)]
    internal = [f for f in findings if "." not in host_of(f["url"])]
    data_egress = [f for f in findings if f not in probe_hits and f not in internal]
    return {"findings": findings, "probe_targets": probe_hits, "internal_names": internal, "runtime_data_egress": data_egress}


def step_cloud_switch_off() -> dict[str, Any]:
    from kee.config import get_settings

    s = get_settings()
    return {"cloud_fallback_enabled": s.cloud_fallback_enabled}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="先 down（保留数据卷）再一键起，验证重启可用性")
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="先 down -v 清空数据卷再从零起，验证真正的一键部署；会抹掉已建索引，跑完需重建",
    )
    ap.add_argument("--isolation", action="store_true", help="额外跑 internal 网络（切断容器出网）变体验证")
    args = ap.parse_args()

    report: dict[str, Any] = {"env": env_snapshot(), "steps": {}}
    report["steps"]["up"] = step_validate_and_up(rebuild=args.rebuild, from_scratch=args.from_scratch)
    report["steps"]["milvus_ready"] = wait_for_milvus()
    report["steps"]["binding"] = step_loopback_binding(report["steps"]["up"])
    report["steps"]["connectivity"] = step_connectivity()

    iso_res: dict[str, Any] = {"skipped": True}
    if args.isolation:
        compose("down", timeout=300)
        up = sh(
            ["docker", "compose", "-f", COMPOSE_MAIN, "-f", COMPOSE_ISO, "up", "-d", "--wait", "--wait-timeout", "240"],
            timeout=500,
        )
        ps = sh(["docker", "compose", "-f", COMPOSE_MAIN, "-f", COMPOSE_ISO, "ps", "--format", "{{.Name}}\t{{.Status}}"])
        host_probe = step_connectivity()
        iso_res = {
            "up_rc": up.returncode,
            "up_error": up.stderr[:300],
            "services": ps.stdout.strip().splitlines(),
            "all_healthy": "unhealthy" not in ps.stdout and "exited" not in ps.stdout.lower(),
            "probe": step_container_egress_probe(internal_mode=True),
            # 实测结论：internal 网络无 NAT 网关，宿主经发布端口访问容器会被拒绝，
            # 因此断网变体只能在「客户端也在同一网络内」时跑通完整链路（应用容器化属 M4）。
            "host_can_reach_published_ports": bool(host_probe.get("milvus", {}).get("ok")),
        }
        sh(["docker", "compose", "-f", COMPOSE_MAIN, "-f", COMPOSE_ISO, "down", "--wait", "60"], timeout=300)
        compose("up", "-d", "--wait", "--wait-timeout", "240")
        iso_res["milvus_ready_after_restore"] = wait_for_milvus()
    report["steps"]["isolation"] = iso_res
    report["steps"]["core_path"] = step_core_path_with_audit()
    # 核心链路跑完后立刻采一次容器侧连接（覆盖「宿主在查、容器有没有偷偷出网」）
    report["steps"]["container_probe"] = step_container_egress_probe(internal_mode=False)
    report["steps"]["static_scan"] = step_static_scan()
    report["steps"]["cloud_switch"] = step_cloud_switch_off()

    write_json("g06_localization.json", report)

    up = report["steps"]["up"]
    core = report["steps"]["core_path"]
    scan = report["steps"]["static_scan"]
    one_click_ok = up["config_ok"] and up.get("up_rc") == 0 and up["all_healthy"] and report["steps"]["milvus_ready"]["ok"]
    conn_ok = all(v.get("ok") for v in report["steps"]["connectivity"].values())
    loopback_ok = not report["steps"]["binding"]["non_loopback_bindings"]
    egress_ok = (
        not core["public_egress_connections"]
        and not scan["runtime_data_egress"]
        and not report["steps"]["container_probe"]["any_suspicious_external"]
    )
    passed = one_click_ok and conn_ok and loopback_ok and egress_ok and not report["steps"]["cloud_switch"]["cloud_fallback_enabled"]

    lines = [
        "# Gate 0 ⑥ 单机本地化部署与数据不出域验证报告",
        "",
        f"- 验证时间：{report['env']['ts']}",
        f"- 环境：{report['env']['platform']} / GPU {report['env']['gpu'].get('name')} {report['env']['gpu'].get('total_mib')}MiB / commit {report['env']['git_commit']}",
        "",
        "## 一键启动",
        "",
        f"- compose config：{'通过' if up['config_ok'] else '失败'}",
        f"- `docker compose up -d --wait` 用时：{up.get('startup_sec')}s，返回码 {up.get('up_rc')}",
        f"- 全组件 healthy：{'是' if up['all_healthy'] else '否'}（期望 {len(SERVICES)} 个服务）",
        f"- Milvus 就绪轮询：{'通过' if report['steps']['milvus_ready']['ok'] else '超时'}"
        f"（等待 {report['steps']['milvus_ready']['waited_sec']}s，服务端 {report['steps']['milvus_ready'].get('server', report['steps']['milvus_ready'].get('error', ''))}）",
        "",
        *md_table(
            ["服务", "状态", "宿主端口"],
            [[s["name"], s["status"], s["ports"] or "-"] for s in up["services"]],
        ),
        "",
        "## 连通性与绑定",
        "",
        *md_table(
            ["组件", "结果"],
            [[k, json.dumps(v, ensure_ascii=False)] for k, v in report["steps"]["connectivity"].items()],
        ),
        f"- 仅绑定 127.0.0.1：{'是' if loopback_ok else '否（发现非回环绑定）'}",
        "- 云端兜底开关：`False`" if not report["steps"]["cloud_switch"]["cloud_fallback_enabled"] else "- ⚠ 云端兜底开关为开",
        "",
        "## 核心链路（嵌入→混合检索→生成）与宿主出网审计",
        "",
        f"- dense 命中 {core['dense_hits']} 条 / sparse 命中 {core['sparse_hits']} 条 / 生成 {'成功' if core['llm_ok'] else '失败'}（JSON 合法 {core['llm_json_valid']}）",
        f"- 本进程非回环连接：{len(core['non_loopback_connections'])} 条；其中公网目标：{len(core['public_egress_connections'])} 条",
        f"- 观测到的连接（去重）：{core['observed_connections_uniq']}",
        "",
        "## 静态出站端点扫描",
        "",
        f"- 命中 URL 总数：{len(scan['findings'])}；分类：连通性探测靶 {len(scan['probe_targets'])} 处、"
        f"容器内部名 {len(scan['internal_names'])} 处、**运行期数据出域嫌疑 {len(scan['runtime_data_egress'])} 处**",
        "- 分类口径：探测靶（1.1.1.1 等，只用于验证出网被堵）与容器内部服务名不构成数据出域；只有真实外部目的地才算嫌疑。",
        *md_table(["文件", "行", "URL"], [[f["file"], f["line_hint"], f["url"]] for f in scan["findings"][:20]]),
        "",
        "## 容器侧连接审计（正常网络，核心链路运行期间）",
        "",
        *md_table(
            ["容器", "ESTAB 连接数", "可疑对外连接", "对端"],
            [
                [
                    name,
                    info.get("n_estab"),
                    len(info.get("suspicious_external", [])),
                    json.dumps(info.get("peers", []), ensure_ascii=False)[:90],
                ]
                for name, info in report["steps"]["container_probe"]["connections"].items()
            ],
        ),
        f"- 容器能否访问公网（正常模式下应为可，说明 NAT 存在）：{report['steps']['container_probe']['containers_can_reach_public']}",
        f"- Milvus 容器内 healthz：{report['steps']['container_probe']['milvus_healthz_in_container']}；etcd 健康应答：{report['steps']['container_probe']['etcd_peer_reachable'][:60]}",
        "",
        "## 断网（internal 网络）变体",
        "",
        (
            "- 本轮跳过（未加 `--isolation`）"
            if iso_res.get("skipped")
            else "\n".join(
                [
                    f"- internal 网络下全组件 healthy：{'是' if iso_res['all_healthy'] else '否'}",
                    f"- 容器出网被封堵（公网探测全部失败）：{'确认' if iso_res['probe']['containers_blocked_from_public'] else '未确认'}",
                    f"- 容器间互联仍可用（milvus→etcd 健康应答）：{iso_res['probe']['etcd_peer_reachable'][:60] or '无应答'}",
                    f"- 容器侧可疑对外连接：{iso_res['probe']['any_suspicious_external']}",
                    f"- 宿主经发布端口访问 internal 网络：{'可用' if iso_res['host_can_reach_published_ports'] else '不可用（预期实测结论）'}",
                    "- ⚠ 结论：`internal: true` 会同时切断宿主↔容器的发布端口通路，因此「宿主跑应用 + 容器全内网」这一组合无法直接验证；"
                    "完整断网链路验证需应用容器化后在同网段内跑（列入 M4），本轮以「容器侧无可疑对外连接 + 仅绑定 127.0.0.1 + 宿主进程连接审计」三条证据支撑无外呼判定。",
                ]
            )
        ),
        "",
        verdict(
            "⑥",
            "pass" if passed else "fail",
            f"一键启动 {up.get('startup_sec')}s 全 healthy={up['all_healthy']}；"
            f"非回环绑定 {len(report['steps']['binding']['non_loopback_bindings'])}；"
            f"公网出网连接 {len(core['public_egress_connections'])}；"
            f"静态扫描运行期出域嫌疑 {len(scan['runtime_data_egress'])}",
            "" if not iso_res.get("skipped") else "断网变体未在本轮执行，需补跑 --isolation",
        ),
        "",
        "## 已知边界",
        "",
        "- 应用与 Nginx 容器本轮未纳入（按方案 3.4 回退分支⑥：先收敛到最小核心组件），列入 M4。",
        "- Neo4j 图谱为可选增强，本轮未起。",
        "- 宿主进程连接审计只覆盖本进程与 Ollama 侧可见连接；Ollama 自身出网需单独审计（本机 Ollama 绑定 127.0.0.1）。",
    ]
    p = write_md("g06_localization.md", lines)
    print("\n".join(lines))
    print("\nJSON:", write_json("g06_localization.json", report))
    print("MD:", p)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
