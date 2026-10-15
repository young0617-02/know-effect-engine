"""#48 知识库写路径在**真库**上跑一遍：上传→复核→升版→回滚→对账。

为什么单独开一个脚本而不是塞进 verify_m4：这几步会**写** PostgreSQL 与 Milvus，
拿生产语料库跑等于永久改语料状态（没有删除文档的端点）。所以这里把写路径指向
**同一容器里的另一套库表 + 另一个集合**（`kee_writepath` / `kb_chunks_writepath`），
跑完删掉——真引擎、真 SQL、真 Outbox，但不动基线索引与 24 篇语料。

跑法：uv run python scripts/m4_kb_write_path.py
前置：存储栈在跑；宿主 API 不必在跑（本脚本用 TestClient，进程内起 app）。
产物：var/gate0/m4_kb_write_path.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCRATCH_DB = "kee_writepath"
SCRATCH_COLL = "kb_chunks_writepath"
OUT = ROOT / "var" / "gate0" / "m4_kb_write_path.json"

# 必须在 import kee.config 之前盖掉，否则 Settings 会把生产 DSN 缓存住
os.environ["KEE_POSTGRES_DSN"] = f"postgresql://kee_app:待填_KEE_PG_APP_PASSWORD@127.0.0.1:35432/{SCRATCH_DB}"
os.environ["KEE_MILVUS_COLLECTION"] = SCRATCH_COLL

# 身份走登录令牌：`KEE_ALLOW_HEADER_IDENTITY` 的默认已于 2026-10-15 翻成 False（用户裁「交付物默认 false、脚本显式带令牌」），
# 这四个字典在 main() 里登录后填充；审计里的 actor 因此是账号名（verify-owner / verify-admin），不再是请求头里随手写的 uploader-a / reviewer-b。
OWNER: dict = {}
UPLOADER: dict = {}
# 职责分离测的是「同一个人」，不是「同一个角色」：令牌身份下「同一个人」只能靠复用上传者那枚令牌来表达
SELF_REVIEW: dict = {}
ADMIN: dict = {}
DOC_ID = "prj-ene-2024-001-handover-wtest"

DOC_V1 = """# 运维交接说明 v1

## 1.1 巡检要求
每日 09:00 前完成核心链路巡检，异常须在 15 分钟内上报值班群。
"""

DOC_V2 = """# 运维交接说明 v2

## 1.1 巡检要求
每日 08:30 前完成核心链路巡检，异常须在 10 分钟内上报值班群，并附巡检截图。
"""

checks: list[dict] = []


def ck(name: str, ok: bool, detail: dict | str = "") -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {json.dumps(detail, ensure_ascii=False)[:180]}" if isinstance(detail, dict) else f" — {detail}"))
    return bool(ok)


def psql(sql: str) -> str:
    r = subprocess.run(
        ["docker", "exec", "kee-postgres", "psql", "-U", "kee_admin", "-d", "postgres", "-tAc", sql],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql 失败：{r.stderr[:200]}")
    return r.stdout.strip()


def make_db() -> None:
    psql(f"CREATE DATABASE {SCRATCH_DB} OWNER kee_app")


def drop_db() -> None:
    psql(f"DROP DATABASE IF EXISTS {SCRATCH_DB} WITH (FORCE)")


def upload(client, headers: dict, text: str, filename: str) -> dict:
    # doc_id 必须显式带上：升版是「同一个 doc_id 再传一次」，
    # 交给文件名推 doc_id 会变成两篇独立文档，测不到版本切换。
    return client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": (filename, text.encode("utf-8"), "text/markdown")},
        data={
            "doc_id": DOC_ID,
            "industry": "能源",
            "customer": "某省级电网公司",
            "project": "PRJ-ENE-2024-001",
            "doc_type": "运维手册",
            "source_level": "B",
            "security_level": "2",
        },
    )


def main() -> int:
    make_db()
    print(f"scratch 库已建：{SCRATCH_DB} / 集合 {SCRATCH_COLL}")
    try:
        from fastapi.testclient import TestClient

        from kee.api.app import create_app
        from kee.db.pg import AuthoritativeStore
        from kee.retrieval.milvus_store import KbStore

        from _identity import ensure_accounts, login_headers  # noqa: E402

        # scratch 库是新建的空库：表结构与集合都得先建出来（app 不负责建，它假设依赖已就绪）
        AuthoritativeStore().init_schema()
        KbStore(SCRATCH_COLL, "jieba").ensure()

        app = create_app()
        pg = app.state.pg  # AuthoritativeStore 是 app 内的实例，不是模块单例
        ensure_accounts(pg)  # 账号要播种进**本脚本的 scratch 库**：登录与写路径读的必须是同一个库

        def search_text(client, query: str) -> str:
            """检索接口只回元数据（`hits`），正文要经 /chunks/batch 二段取——顺带把这条契约也验了。"""
            got = client.post("/api/v1/retrieval/search", headers=OWNER, json={"query": query, "top_k": 5}).json()
            ids = [h["chunk_id"] for h in got.get("hits", [])]
            if not ids:
                return ""
            batch = client.post("/api/v1/chunks/batch", headers=OWNER, json={"ids": ids}).json()
            return "".join(i.get("content", "") for i in batch.get("items", []))

        with TestClient(app) as client:
            global OWNER, UPLOADER, SELF_REVIEW, ADMIN
            OWNER = login_headers(client, "verify-owner")   # knowledge_owner：检索、换人复核、回滚（与原来同一个角色）
            ADMIN = login_headers(client, "verify-admin")   # system_admin：对账、审计
            # 上传者换成带复核权的账号：delivery_engineer 自审只会撞「角色不够」的 403，那条断言就不再测职责分离了
            UPLOADER = ADMIN
            SELF_REVIEW = UPLOADER  # 同一账号的**同一枚令牌**——令牌身份下「同一个人」只能这样造
            h = client.get("/healthz").json()
            coll = h["deps"]["milvus"].get("collection")
            ck("写路径跑在 scratch 上（没碰生产库与基线集合）", coll == SCRATCH_COLL, {"collection": coll, "chunks": h["deps"]["milvus"].get("chunks")})

            r1 = upload(client, UPLOADER, DOC_V1, "handover-v1.md")
            body1 = r1.json() if r1.status_code == 201 else {}
            doc_id = body1.get("doc_id", "")
            v1 = body1.get("version_id", "")
            ck("上传 v1 返回 201 并给出 doc_id/version_id", r1.status_code == 201 and doc_id and v1,
               {"http": r1.status_code, "body": r1.text[:300], "doc_id": doc_id, "version_id": v1,
                "chunks": body1.get("indexed_chunks")})

            # 职责分离：同一个**人**上传又复核（角色合法、actor 即登录账号名）必须被拒——故 SELF_REVIEW 用的就是 UPLOADER 那枚令牌
            self_review = client.post(f"/api/v1/documents/{doc_id}/review", headers=SELF_REVIEW, json={"decision": "pass"})
            ck("上传者复核自己被拒（E403 职责分离）",
               self_review.status_code == 403 and "职责分离" in self_review.text,
               {"http": self_review.status_code, "detail": self_review.json().get("detail", "")[:60]})

            rv = client.post(f"/api/v1/documents/{doc_id}/review", headers=OWNER, json={"decision": "pass", "note": "复核通过"})
            st = pg.get_document(doc_id)
            vers = pg.versions_of(doc_id)
            eff_v1 = next((v["id"] for v in vers if v["status"] == "effective"), None)
            ck("他人复核 pass 后文档转 active 且版本生效",
               rv.status_code == 200 and st["status"] == "active" and eff_v1 == v1,
               {"doc_status": st["status"], "effective": eff_v1})

            txt1 = search_text(client, "核心链路巡检 异常 上报 值班群")
            ck("v1 生效后可被检索命中（15 分钟那条在）", "15 分钟" in txt1, {"命中正文字数": len(txt1)})

            r2 = upload(client, UPLOADER, DOC_V2, "handover-v2.md")
            v2 = r2.json().get("version_id", "")
            client.post(f"/api/v1/documents/{doc_id}/review", headers=OWNER, json={"decision": "pass"})
            eff_v2 = next((v["id"] for v in pg.versions_of(doc_id) if v["status"] == "effective"), None)
            ck("升版：v2 复核后生效版本切到 v2", r2.status_code == 201 and eff_v2 == v2, {"v1": v1, "v2": v2, "effective": eff_v2})

            txt2 = search_text(client, "核心链路巡检 异常 上报 值班群 截图")
            ck("升版后检索到新口径（10 分钟），旧口径不再是生效内容",
               "10 分钟" in txt2 and "15 分钟" not in txt2,
               {"含10分钟": "10 分钟" in txt2, "含15分钟": "15 分钟" in txt2})

            rb = client.post(f"/api/v1/documents/{doc_id}/rollback", headers=OWNER, json={"to_version_id": v1})
            eff_rb = next((v["id"] for v in pg.versions_of(doc_id) if v["status"] == "effective"), None)
            kb = KbStore(SCRATCH_COLL)
            left = sorted(kb.all_chunk_ids())
            kb.close()
            ck("回滚到 v1：生效版本回 v1，且索引里不再有 v2 的子块",
               rb.status_code == 200 and eff_rb == v1 and not any(i.startswith(v2) for i in left),
               {"http": rb.status_code, "effective": eff_rb, "v2 残留": [i for i in left if i.startswith(v2)], "索引内": left})

            rec0 = client.post("/api/v1/reconcile/run", headers=ADMIN, params={"repair": "false"}).json()
            d0 = rec0.get("diff", {})
            ck("对账 dry-run 读得出漂移面", set(d0) >= {"authoritative_children", "indexed", "missing_in_vector", "extra_in_vector"}, d0)

            # 人为制造漂移：从向量侧删掉一个子块，看对账能不能发现并修好
            with pg.connect() as conn:
                victim = conn.execute(
                    "SELECT id FROM chunk WHERE version_id=%s AND NOT is_parent ORDER BY order_no LIMIT 1", (v1,)
                ).fetchone()["id"]
            kb = KbStore(SCRATCH_COLL)
            kb.delete_chunks([victim])
            kb.flush()
            kb.close()
            rec1 = client.post("/api/v1/reconcile/run", headers=ADMIN, params={"repair": "true"}).json()
            ck("漂移被发现（missing_in_vector ≥ 1）", (rec1.get("diff", {}).get("missing_in_vector") or 0) >= 1,
               {"被删子块": victim, "diff": rec1.get("diff")})
            recheck = rec1.get("recheck") or {}
            ck("修复后缺口归零", recheck.get("missing_in_vector") == 0 and recheck.get("extra_in_vector") == 0, recheck)

            aud = client.get("/api/v1/audit?limit=100", headers=ADMIN).json()
            acts = sorted({a["action"] for a in aud.get("items", [])})
            need = {"document_ingested", "document_review_pass", "document_rollback", "reconciliation_run"}
            ck("四步写操作都留了审计", need <= set(acts), {"actions": acts[:12]})

            vers_final = pg.versions_of(doc_id)
            st_final = pg.get_document(doc_id)
            ck("终态自洽：生效版本唯一、文档 active",
               sum(1 for v in vers_final if v["status"] == "effective") == 1 and st_final["status"] == "active",
               {"versions": [(v["id"][-6:], v["status"]) for v in vers_final]})
    finally:
        try:
            from kee.retrieval.milvus_store import KbStore as _KB
            _KB(SCRATCH_COLL).drop()
        except Exception as e:  # noqa: BLE001
            print("集合清理提示：", e)
        drop_db()
        print(f"scratch 已删：库 {SCRATCH_DB} / 集合 {SCRATCH_COLL}")

    prod = json.loads((ROOT / "var" / "gate0" / "m2_eval.json").read_text(encoding="utf-8"))
    rep = {
        "env": {"ts": datetime.now().isoformat(timespec="seconds"),
                "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                "scratch_db": SCRATCH_DB, "scratch_collection": SCRATCH_COLL},
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
        "生产侧未受影响（对照 m2_eval 记录）": {"collection": prod["env"]["collection"], "chunks": prod["env"]["chunks"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共 {len(checks)} 项，失败 {sum(1 for c in checks if not c['ok'])} 项；证据：{OUT}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
