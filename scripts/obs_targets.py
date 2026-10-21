#!/usr/bin/env python
"""生成 Prometheus 的 file_sd 目标文件：把「宿主 API 在哪个端口」这件事收在一处。

为什么要这么绕一层：Windows 会随机保留一段 TCP 端口范围，本机 API 已经从 38000 挪到过 28000。
端口写死在 `prometheus.yml` 里，改端口就会「抓取目标一直 down、看板一直 No data」，
而那副样子和「指标层坏了」长得一模一样——上一轮 Nginx 反代 502 就是同一个形状的坑。
`file_sd` + `refresh_interval` 让 Prometheus 自己 20s 刷一次，改端口不用重启任何容器。

跑法：
    uv run python scripts/obs_targets.py --port 28000
    uv run python scripts/obs_targets.py            # 用 .env 里的 KEE_API_PORT，默认 38000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "infra" / "prometheus" / "targets" / "kee.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("KEE_API_PORT", "38000")))
    ap.add_argument("--host", default="host.docker.internal",
                    help="Prometheus 在容器里，够得到宿主回环靠的是这个名字")
    args = ap.parse_args()

    payload = [
        {
            "targets": [f"{args.host}:{args.port}"],
            "labels": {
                "service": "kee-api",
                "note": f"由 scripts/obs_targets.py 生成于端口 {args.port}",
            },
        }
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    OUT.write_text(text, encoding="utf-8")
    print(f"{'已更新' if old != text else '无变化'}：{OUT}")
    print(f"  target = {args.host}:{args.port}（Prometheus 容器内视角）")
    print("  20s 内自动生效，不用重启 Prometheus。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
