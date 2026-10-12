"""把上传路径在**真 HTTP 服务进程**上走一遍（`ensure_scheduler` 修复的验收）。

为什么还要这个脚本：`m4_kb_write_path.py` 用的是 `TestClient`（进程内 app），而那个缺陷恰恰藏在
**进程启动方式**里——`serve_api.py` 调 `create_app()` 不传 scheduler，TestClient 的调用方却自己传了。
所以「TestClient 通过」根本照不到生产那条路。这里起的是**真的 uvicorn 子进程**，
用的是一模一样的 `scripts/serve_api.py`。

为什么不跑在浏览器上：UI 上传会写进**生产库**（24 篇 / 193 块），而线上没有删除文档的端点，
写进去撤不回来，还可能改掉按关键词回查的 gold。所以同样打真 HTTP，但后端指向
**独立 scratch 库 + scratch 集合 + 独立端口**，跑完删净并自检生产计数未变。

跑法：uv run python scripts/m4_upload_http_check.py
产物：var/gate0/m4_upload_http.json
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

DB = "kee_uploadhttp"
COLL = "kb_chunks_uploadhttp"
PORT = 38001
PROD_PORT = 38000
OUT = ROOT / "var" / "gate0" / "m4_upload_http.json"

DOC = """# 上传链路验证说明 v1

## 1.1 值班要求
每 2 小时核对一次采集网关心跳，连续 3 次缺失即开工单并在 30 分钟内响应。
"""

checks: list[dict] = []


def ck(name: str, ok, detail="") -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)[:400]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — {str(detail)[:200]}"))
    return bool(ok)


def psql(sql: str) -> str:
    r = subprocess.run(["docker", "exec", "kee-postgres", "psql", "-U", "kee", "-d", "postgres", "-tAc", sql],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return r.stdout.strip()


def curl(args: list[str], timeout: int = 240) -> tuple[int, str]:
    r = subprocess.run(["curl", "-s", "-m", str(timeout), "-w", "\\n__HTTP__%{http_code}", *args],
                       capture_output=True, text=True, cwd=ROOT, timeout=timeout + 30)
    body, _, code = r.stdout.rpartition("__HTTP__")
    return int(code or 0), body.strip().rstrip("\n")


def prod_health() -> dict:
    code, body = curl(["http://127.0.0.1:%d/healthz" % PROD_PORT], timeout=20)
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    pre = prod_health()
    pre_counts = (pre.get("deps", {}).get("milvus", {}).get("chunks"), pre.get("deps", {}).get("postgres", {}).get("documents"))
    if pre_counts[0] is None:
        print("[拒绝开始] 生产 API 不可达，拿不到「跑前对照」的计数。")
        return 2
    print(f"生产侧跑前：chunks={pre_counts[0]} docs={pre_counts[1]}（本脚本不该改动它）")

    env = dict(os.environ)
    env["KEE_POSTGRES_DSN"] = f"postgresql://kee:待填_KEE_PG_SUPERUSER_PASSWORD@127.0.0.1:35432/{DB}"
    env["KEE_MILVUS_COLLECTION"] = COLL
    env["KEE_API_PORT"] = str(PORT)

    psql(f"CREATE DATABASE {DB} OWNER kee")
    print(f"scratch 库已建：{DB} / 集合 {COLL} / 端口 {PORT}")
    proc = None
    try:
        from kee.db.pg import AuthoritativeStore
        from kee.retrieval.milvus_store import KbStore

        AuthoritativeStore(dsn=env["KEE_POSTGRES_DSN"]).init_schema()
        KbStore(COLL, "jieba").ensure()

        # 真起一个 serve_api.py 子进程——与生产完全同一份启动代码
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "serve_api.py")],
                                cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{PORT}"
        ready = False
        for _ in range(60):
            code, body = curl([f"{base}/healthz"], timeout=10)
            if code == 200 and COLL in body:
                ready = True
                break
            time.sleep(3)
        ck("scratch 后端起来了，且 healthz 报的就是 scratch 集合", ready, {"healthz": body})
        if not ready:
            return 3

        owner = ["-H", "X-Kee-Role: knowledge_owner", "-H", "X-Kee-Actor: reviewer-b"]
        eng = ["-H", "X-Kee-Role: delivery_engineer", "-H", "X-Kee-Actor: uploader-a"]
        f = ROOT / "var" / ".upload_probe.md"
        f.write_text(DOC, encoding="utf-8")
        try:


            code, body = curl(["-X", "POST", f"{base}/api/v1/documents/upload", *eng,
                               "-F", f"file=@{f};type=text/markdown",
                               "-F", "doc_id=upload-http-probe", "-F", "industry=能源",
                               "-F", "customer=某省级电网公司", "-F", "project=PRJ-ENE-2024-001",
                               "-F", "doc_type=运维手册", "-F", "source_level=B", "-F", "security_level=2"], timeout=300)
            j = json.loads(body) if code else {}
            # 这一条就是 ensure_scheduler 的验收：修之前这里必 500
            ck("真 HTTP 服务上上传成功（缺陷修复的验收点）", code == 201 and j.get("version_id"),
               {"http": code, "body": body[:240]})
            v1 = j.get("version_id", "")
            doc_id = j.get("doc_id", "")

            code, body = curl([f"{base}/api/v1/documents", *owner])
            lst = json.loads(body) if code == 200 else {}
            # 响应键是 items（不是 documents）——第一版按 documents 取，取到空就把一条正常的接口判成失败
            got = [d for d in (lst.get("items") or []) if d.get("id") == doc_id]
            ck("文档列表读得到刚上传的那篇", code == 200 and len(got) == 1,
               {"http": code, "total": lst.get("total"), "doc_status": got[0].get("status") if got else None})

            # 更要紧的一条：文档一上传就是 active，只有**版本**是 pending。
            # 那未经复核的内容是不是已经能被检索到？如果是，「知识准入」这道闸其实没有拦住索引。
            code, body = curl(["-X", "POST", f"{base}/api/v1/retrieval/search", *owner,
                               "-H", "Content-Type: application/json",
                               "-d", json.dumps({"query": "采集网关 心跳 工单 响应", "top_k": 5}, ensure_ascii=False)])
            pre_review_hits = [h.get("chunk_id") for h in (json.loads(body).get("hits", []) if code == 200 else [])]
            leaked = [h for h in pre_review_hits if h and h.startswith(f"{doc_id}@v1")]
            code2, body2 = curl([f"{base}/api/v1/documents/{doc_id}", *owner])
            vers = {(v["id"], v["status"]) for v in (json.loads(body2).get("versions", []) if code2 == 200 else [])}
            # C 的边界：**未复核也允许被检索到**（复核人得看得见才能去复核），
            # 所以这一条断言的是「检索照旧命中」，不是「检索挡住」。
            ck("复核**之前**仍可被检索命中（C 的边界：闸不在检索层）", bool(leaked),
               {"版本状态": sorted(vers), "复核前命中": pre_review_hits})
            assert leaked, "C 语义下未复核内容应当仍可检索；查不到说明闸装错层了"

            # 检索命中只是半句；真正要紧的是「未经复核的内容能不能直接被引用作答」。
            # 用一个只存在于刚上传文档里的问法，看答案是否引用了那个未复核版本。
            code, body = curl(["-X", "POST", f"{base}/api/v1/answers", *owner,
                               "-H", "Content-Type: application/json",
                               "-d", json.dumps({"question": "采集网关心跳连续缺失几次要开工单，工单多久内响应？"}, ensure_ascii=False)],
                              timeout=300)
            a = json.loads(body) if code == 200 else {}
            cited = [c.get("chunk_id") for c in (a.get("citations") or [])]
            answered_unreviewed = (a.get("status") in ("answer", "tentative")
                                   and any(c and c.startswith(f"{doc_id}@v1") for c in cited))
            ck("复核**之前**不得被引用作答（未复核内容不进答案）", not answered_unreviewed,
               {"status": a.get("status"), "结论": (a.get("conclusion") or "")[:90], "引用": cited})


            code, body = curl(["-X", "POST", f"{base}/api/v1/documents/{doc_id}/review", *owner,
                               "-H", "Content-Type: application/json", "-d", json.dumps({"decision": "pass", "note": "UI 等价操作"})])
            ck("复核 pass 生效（角色合法、actor 与 uploader 不同）", code == 200, {"http": code, "body": body[:200]})

            code, body = curl(["-X", "POST", f"{base}/api/v1/retrieval/search", *owner,
                               "-H", "Content-Type: application/json",
                               "-d", json.dumps({"query": "采集网关 心跳 工单 响应", "top_k": 5}, ensure_ascii=False)])
            hits = [h.get("chunk_id") for h in (json.loads(body).get("hits", []) if code == 200 else [])]
            ck("复核**之后**可被检索命中（索引真的写进去了）", any(h and h.startswith(f"{doc_id}@v1") for h in hits),
               {"http": code, "hits": hits[:5]})

            # C 的另一半：复核**之后**同一条内容必须立刻可被引用作答，否则闸是死的。
            code, body = curl(["-X", "POST", f"{base}/api/v1/answers", *owner,
                               "-H", "Content-Type: application/json",
                               "-d", json.dumps({"question": "采集网关心跳连续缺失几次要开工单，工单多久内响应？"}, ensure_ascii=False)],
                              timeout=300)
            a2 = json.loads(body) if code == 200 else {}
            cited2 = [c.get("chunk_id") for c in (a2.get("citations") or [])]
            ck("复核**之后**可被引用作答（闸不是永久封死）",
               a2.get("status") in ("answer", "tentative") and any(x and x.startswith(f"{doc_id}@v1") for x in cited2),
               {"status": a2.get("status"), "结论": (a2.get("conclusion") or "")[:90], "引用": cited2})

            code, body = curl(["-X", "POST", f"{base}/api/v1/documents/upload", *eng,
                               "-F", f"file=@{f};type=text/markdown",
                               "-F", "doc_id=upload-http-probe", "-F", "industry=能源", "-F", "customer=某省级电网公司",
                               "-F", "project=PRJ-ENE-2024-001", "-F", "doc_type=运维手册",
                               "-F", "source_level=B", "-F", "security_level=2"], timeout=300)
            dup = json.loads(body) if code else {}
            ck("重复上传同一 doc_id 走的是新版本（不是新建文档）", code == 201 and dup.get("new_document") is False
               and dup.get("version_id", "").endswith("@v2"), {"http": code, "resp": body[:200]})
            v2 = dup.get("version_id", "")

            code, body = curl(["-X", "POST", f"{base}/api/v1/documents/{doc_id}/rollback", *owner,
                               "-H", "Content-Type: application/json", "-d", json.dumps({"to_version_id": v1})])
            rb = json.loads(body) if code == 200 else {}
            ck("回滚到 v1 生效", code == 200 and rb.get("effective_version") == v1, {"http": code, "body": body[:200]})

            kb = KbStore(COLL)
            left = kb.all_chunk_ids()
            kb.close()
            ck("索引里只剩 v1 的子块", all(not i.startswith(v2) for i in left) and len(left) >= 1, left)

            code, body = curl(["-X", "POST", f"{base}/api/v1/reconcile/run", *owner, "--data-urlencode", "repair=false", "-G"])
            rec = json.loads(body) if code == 200 else {}
            ck("对账接口在真服务上可读（dry-run）", code == 200 and "diff" in rec, {"http": code, "diff": rec.get("diff")})

            code, body = curl([f"{base}/api/v1/audit?limit=50", *owner])
            acts = sorted({a["action"] for a in (json.loads(body).get("items", []) if code == 200 else {})})
            ck("上传/复核/回滚/对账都进了审计", {"document_ingested", "document_review_pass", "document_rollback", "reconciliation_run"} <= set(acts), acts)
        finally:
            f.unlink(missing_ok=True)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            KbStore(COLL).drop()
        except Exception as e:  # noqa: BLE001
            print("集合清理提示：", e)
        psql(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
        post = prod_health()
        post_counts = (post.get("deps", {}).get("milvus", {}).get("chunks"), post.get("deps", {}).get("postgres", {}).get("documents"))
        ck("生产侧未被改动（chunks/docs 与跑前一致）", post_counts == pre_counts, {"pre": pre_counts, "post": post_counts})
        print(f"scratch 已删：库 {DB} / 集合 {COLL}；API 子进程（PID {proc.pid if proc else '-'}）已停")

    rep = {"env": {"ts": datetime.now().isoformat(timespec="seconds"),
                   "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                   "scratch_db": DB, "scratch_collection": COLL, "port": PORT},
           "prod_before": pre_counts, "checks": checks, "passed": all(c["ok"] for c in checks)}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共 {len(checks)} 项，失败 {sum(1 for c in checks if not c['ok'])} 项；证据：{OUT}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
