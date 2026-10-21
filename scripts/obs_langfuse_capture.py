"""Langfuse 上报载荷的**实采**：一个本地采集器 + 一个打印器。

为什么要有这个脚本：`infra/docker-compose.langfuse.yml` 那套真容器还没拉（等裁决），
于是「Langfuse 里的一条 trace 长什么样」这句话本来只能靠我口述。
本脚本把话说实：**它就是 KEE 通过官方 `/api/public/ingestion` 批量接口送出去的那批字节**，
一个字段都不改地落盘。区别只在于收下这批字节的是本机采集器，而不是 Langfuse 的 UI。

跑法（两个终端，或按 `obs_langfuse_capture.py --help` 的顺序）：
    # 1) 起采集器（只绑回环）
    uv run python scripts/obs_langfuse_capture.py --serve --port 30099 \
        --out var/observability/langfuse_capture.json
    # 2) 让宿主 API 把上报指到它（同批设 pk/sk，值随便，采集器只回显不校验）
    KEE_API_PORT=28000 KEE_LANGFUSE_ENABLED=true \
    KEE_LANGFUSE_HOST=http://127.0.0.1:30099 \
    KEE_LANGFUSE_PUBLIC_KEY=待填_KEE_LANGFUSE_PUBLIC_KEY KEE_LANGFUSE_SECRET_KEY=待填_KEE_LANGFUSE_SECRET_KEY \
    uv run python scripts/serve_api.py
    # 3) 走真实 HTTP 问一轮，然后回来看落盘文件
    uv run python scripts/obs_langfuse_capture.py --print var/observability/langfuse_capture.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE: dict = {"out": None, "batches": []}


class Handler(BaseHTTPRequestHandler):
    """Langfuse ingestion 的最小可信实现：收 batch、按「每条成功」回 207。

    刻意**不校验** pk/sk 的真伪（那是 Langfuse 服务端的事），但会把收到的 Authorization
    原样记下来——「鉴权头到底带没带、是不是 Basic」正是这条通路上最该被看见的一件事。
    """

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        try:
            batch = json.loads(raw.decode("utf-8")).get("batch") or []
        except json.JSONDecodeError:
            batch = []
        rec = {
            "path": self.path,
            "authorization": self.headers.get("authorization") or self.headers.get("Authorization"),
            "content_type": self.headers.get("content-type"),
            "n_events": len(batch),
            "event_types": [e.get("type") for e in batch],
            "events": batch,
        }
        STATE["batches"].append(rec)
        if STATE["out"]:
            STATE["out"].write_text(json.dumps(STATE["batches"], ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        body = json.dumps({"successList": [{"id": e.get("id"), "status": 201} for e in batch],
                           "errorList": []}).encode()
        self.send_response(207)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 静音：这是被抓的那一方，不该往验收输出里打字
        pass


def serve(port: int, out: Path, once_sec: float) -> int:
    STATE["out"] = out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("[]", encoding="utf-8")
    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Langfuse 采集器 http://127.0.0.1:{port}/api/public/ingestion → {out}", flush=True)
    if once_sec > 0:
        threading.Timer(once_sec, srv.shutdown).start()
    try:
        srv.serve_forever()
    finally:
        print(f"已收 {len(STATE['batches'])} 批", flush=True)
    return 0


def show(path: Path, want_trace_id: str = "") -> int:
    """把实采载荷按「一条 trace + 它的 span/generation」摊开打印。"""
    batches = json.loads(path.read_text(encoding="utf-8"))
    events = [e for b in batches for e in b["events"]]
    traces = [e for e in events if e["type"] == "trace-create"]
    if not traces:
        print("没有 trace-create 事件——上报链路根本没跑起来", file=sys.stderr)
        return 1
    pick = next((t for t in traces if t["body"]["id"] == want_trace_id), traces[0]) if want_trace_id \
        else traces[0]
    tid = pick["body"]["id"]
    kids = [e for e in events if e.get("body", {}).get("traceId") == tid]
    print(f"# 采集 {len(batches)} 批 / {len(events)} 个事件；下面摊开其中一条 trace\n")
    print(f"traceId = {tid}")
    b = pick["body"]
    print(f"name    = {b['name']}    tags = {b.get('tags')}")
    print(f"session = {b.get('sessionId')}   user = {b.get('userId')}   ts = {b['timestamp']}")
    print(f"input   = {json.dumps(b.get('input'), ensure_ascii=False)}")
    print(f"output  = {json.dumps(b.get('output'), ensure_ascii=False)[:300]}")
    print("metadata= " + json.dumps({k: v for k, v in (b.get('metadata') or {}).items()
                                     if k != 'timings_ms'}, ensure_ascii=False)[:520])
    print("timings = " + json.dumps((b.get('metadata') or {}).get('timings_ms'), ensure_ascii=False))
    print(f"\n## 挂在同一条 trace 上的 {len(kids)} 个子事件")
    for e in kids:
        x = e["body"]
        span = f"{x.get('startTime','')[11:23]} → {x.get('endTime','')[11:23]}" if x.get("startTime") else "-"
        extra = ""
        if e["type"] == "generation-create":
            extra = f" model={x.get('model')} usage={json.dumps(x.get('usage'), ensure_ascii=False)}"
            md = x.get("metadata") or {}
            extra += (f" prompt_chars={md.get('prompt_chars')} done_reason={md.get('done_reason')}"
                      f" json_valid={md.get('json_valid_first_attempt')}")
        elif x.get("metadata"):
            extra = f" duration_ms={x['metadata'].get('duration_ms')}"
        io = f"  in={json.dumps(x.get('input'), ensure_ascii=False)[:60]!r}" if x.get("input") else ""
        outv = json.dumps(x.get("output"), ensure_ascii=False)[:120] if x.get("output") is not None else "null"
        print(f"- {e['type']:18} {x.get('name'):12} {span}{extra}")
        print(f"    output = {outv}{io}")
    print("\n## 鉴权与端点（每批一次 POST）")
    for i, batch in enumerate(batches, 1):
        auth = (batch["authorization"] or "")[:12] + "…" + (batch["authorization"] or "")[-6:]
        print(f"- 批{i}: {batch['path']}  auth={auth}  {batch['n_events']} 事件  "
              f"types={batch['event_types']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=30099)
    ap.add_argument("--out", default="var/observability/langfuse_capture.json")
    ap.add_argument("--auto-stop-sec", type=float, default=0.0)
    ap.add_argument("--print", dest="show", default="")
    ap.add_argument("--trace-id", default="")
    a = ap.parse_args()
    if a.serve:
        return serve(a.port, Path(a.out), a.auto_stop_sec)
    if a.show:
        return show(Path(a.show), a.trace_id)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
