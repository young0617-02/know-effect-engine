"""静态检查（`ci_local.sh` 的 static 阶段）：能编译、能解析、配置合法。

这一层拦住的正是今天真实发生过的两类错：
  * 一个 `dict` 少收尾的 `}` —— 它只在**运行时**报，而 `py_compile` 当场就报；
    今天两次改脚本都踩了这同一个形状，第二次是靠重跑才发现的。
  * 告警 YAML 写成「两行各自带引号」—— 症状是 Prometheus 整个起不来，
    `promtool check config` 一行就判出来。

刻意**不装任何东西**：`ruff` 不在依赖里，装了它属于改环境；它在本仓库只作参考读数，
不判失败（理由写在 `ruff_advisory` 里）。判失败的只有会当场炸的形状：语法、JSON、compose、promtool。
"""

from __future__ import annotations

import compileall
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = [
    "infra/docker-compose.gate0.yml",
    "infra/docker-compose.m4.yml",
    "infra/docker-compose.observability.yml",
]
JSONS = ["infra/grafana/dashboards/kee-rag.json", "infra/prometheus/targets/kee.json"]
PROM_IMAGE = "prom/prometheus:v2.53.1"
fails: list[str] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  ✓ {label}{(' — ' + detail) if detail else ''}")


def bad(label: str, detail: str = "") -> None:
    fails.append(label)
    print(f"  ✗ {label}{(' — ' + detail) if detail else ''}")


def note(label: str, detail: str = "") -> None:
    """既不通过也不算失败的状态（工具缺失、只供参考的检查跑不起来）。"""
    print(f"  ! {label}{(' — ' + detail) if detail else ''}")


def run(cmd: list[str], label: str, hint: str = "") -> None:
    if shutil.which(cmd[0]) is None:
        # 判失败而不是「跳过」：命令不在就等于这项没测，读成通过是空集上的真
        bad(label, f"{cmd[0]} 不在 PATH → 该项未测")
        return
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        tail = (r.stdout or "").strip().splitlines()
        ok(label, tail[-1][:90] if tail else "")
    else:
        msg = ((r.stderr or "") + (r.stdout or "")).strip().replace("\n", " / ")[:220]
        bad(label, msg + (f"｜{hint}" if hint else ""))


def main() -> int:
    print("静态检查")
    for d in ("src", "scripts", "tests"):
        # quiet=2 只报错误；force 让它真的走完全部文件（默认遇到 .pyc 新鲜就跳过，会漏）
        good = compileall.compile_dir(str(ROOT / d), quiet=2, force=True)
        (ok if good else bad)(f"compileall {d}/")
    for f in JSONS:
        p = ROOT / f
        try:
            json.loads(p.read_text(encoding="utf-8"))
            ok(f"JSON {f}")
        except Exception as e:  # noqa: BLE001
            bad(f"JSON {f}", f"{type(e).__name__}: {e}")
    for c in COMPOSE:
        run(["docker", "compose", "-f", c, "config", "-q"], f"compose config {c}")
    # langfuse 那份是「出稿未跑」，但语法必须一直合法，否则裁决下来那天第一次 up 就炸
    run(["docker", "compose", "-f", "infra/docker-compose.langfuse.yml", "--profile", "langfuse",
         "config", "-q"], "compose config infra/docker-compose.langfuse.yml")
    images = subprocess.run(["docker", "image", "ls", "-q", PROM_IMAGE], cwd=ROOT,
                            capture_output=True, text=True)
    if images.stdout.strip():
        run(["docker", "run", "--rm", "-v", f"{ROOT / 'infra' / 'prometheus'}:/etc/prometheus:ro",
             "--entrypoint", "/bin/promtool", PROM_IMAGE, "check", "config",
             "/etc/prometheus/prometheus.yml"], "promtool check config")
    else:
        ok("promtool check config", f"{PROM_IMAGE} 不在本机，跳过")
    ruff_advisory()
    print(f"\n静态检查：{'全部通过' if not fails else '失败 ' + str(len(fails)) + ' 项：' + ', '.join(fails)}")
    return 0 if not fails else 1


def ruff_advisory() -> None:
    """ruff 只报数、**不判失败**（这一条是决定，不是遗漏）。

    仓库里没有 `[tool.ruff]` 配置，ruff 0.16.8 的默认规则集在本代码上常驻 **590 项**
    （263 项还是 `RUF100`：满仓的 `# noqa: BLE001` 注的全是没启用的规则）。
    让这种状态去判失败，等于给复验脚本装一条永远红的闸门——信号为零，还会把真错淹掉。
    把它升级成门禁需要「写配置 + 清 590 项」，那是独立一次改动，不属于复验脚本；
    同理不擅自往依赖里加 ruff（本脚本刻意不装任何东西）。
    报数的价值：`F821 / E9 / F7` 这类会当场炸的形状一旦跳数，日志里看得见。
    """
    if shutil.which("ruff") is None:
        ok("ruff", "未安装 → 跳过（不在依赖里，本脚本不擅自装环境）")
        return
    r = subprocess.run(
        ["ruff", "check", "src", "scripts", "tests", "--statistics", "--quiet"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    rows = []
    for line in (r.stdout or "").splitlines():
        head, _, rest = line.partition("\t")
        if head.strip().isdigit():
            tok = rest.split()
            rows.append((int(head.strip()), tok[0] if tok else "?"))
    if not rows:
        if r.returncode not in (0, 1):  # ruff 自己起不来（配置/权限）不能读成「无发现」
            note("ruff", "工具报错：" + ((r.stderr or "") + (r.stdout or "")).strip().replace("\n", " / ")[:180])
        else:
            ok("ruff（仅供参考，不判失败）", "无发现")
        return
    rows.sort(reverse=True)
    ok(
        "ruff（仅供参考，不判失败）",
        f"共 {sum(n for n, _ in rows)} 项：" + "、".join(f"{name} {n}" for n, name in rows[:5])
        + (f"…另 {len(rows) - 5} 类" if len(rows) > 5 else ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
