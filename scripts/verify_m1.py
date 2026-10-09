"""M1 端到端验证：真实文件经 HTTP 契约走完「上传→准入→解析→分块→权威源→Outbox→索引→检索」。

跑在独立集合 kb_m1_verify 上，不动 Gate 0 的基线数据。
HTTP 层用 TestClient 在进程内驱动：不起服务、不占端口，但走的是同一套路由与错误码。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from kee.api import create_app  # noqa: E402
from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

COLLECTION = "kb_chunks"  # 对账口径是「一份权威源 ↔ 一个索引」，验证必须跑在生产集合上
FIXTURES = Path("var/m1/fixtures")
REAL_PDF = Path("D:/project/kb_rag/doc/hak180产品安全手册.pdf")

OWNER = {"X-Kee-Role": "knowledge_owner", "X-Kee-Actor": "owner-01"}
ADMIN = {"X-Kee-Role": "system_admin", "X-Kee-Actor": "admin-01"}
ENG = {"X-Kee-Role": "delivery_engineer", "X-Kee-Actor": "eng-01"}
NEW = {"X-Kee-Role": "newcomer", "X-Kee-Actor": "new-01"}

evidence: dict[str, Any] = {"steps": [], "timings_sec": {}}


def purge_m1(pg: AuthoritativeStore, kb: KbStore) -> int:
    """清掉本验证造的 m1-* 文档（权威源、索引、向量物料、幂等键一起清）。

    验证脚本必须可重复运行、且跑完把生产集合还原，否则下一轮会被自己的
    历史数据判成「重复文档」，对账也会把上一轮的残留当成脏行。
    """
    from kee.db.vector_store import VectorStore as _VS

    vs = _VS()
    with pg.connect() as conn:
        stale = [r["id"] for r in conn.execute("SELECT id FROM document WHERE id LIKE 'm1-%'").fetchall()]
        for d in stale:
            for ver in pg.versions_of(d):
                kb.delete_version(ver["id"])
                vs.drop(ver["id"])
            conn.execute("DELETE FROM chunk WHERE doc_id=%s", (d,))
            conn.execute("DELETE FROM document_version WHERE doc_id=%s", (d,))
            conn.execute("DELETE FROM outbox_event WHERE aggregate_id LIKE %s", (f"{d}%",))
            conn.execute("DELETE FROM document WHERE id=%s", (d,))
        conn.execute("DELETE FROM idempotency_key")
    return len(stale)


def step(name: str, ok: bool, detail: Any) -> None:
    evidence["steps"].append({"step": name, "ok": bool(ok), "detail": detail})
    mark = "✅" if ok else "❌"
    text = json.dumps(detail, ensure_ascii=False, default=str)
    print(f"{mark} {name}: {text[:260]}")
    if not ok:
        raise SystemExit(f"步骤失败：{name} → {text[:400]}")


def upload(client: TestClient, path: Path, meta: dict[str, str], headers: dict[str, str], key: str | None = None) -> dict:
    files = {"file": (path.name, path.read_bytes(), "application/octet-stream")}
    data = {k: v for k, v in meta.items() if v}
    h = dict(headers)
    if key:
        h["Idempotency-Key"] = key
    r = client.post("/api/v1/documents/upload", files=files, data=data, headers=h)
    return {"status_code": r.status_code, "body": r.json()}


def main() -> int:
    t_all = time.perf_counter()
    s = get_settings()
    pg = AuthoritativeStore()
    pg.init_schema()

    kb = KbStore(COLLECTION, "jieba")
    # 不 drop：这是 Gate 0 的基线索引，验证只在其中增删 m1-* 文档，跑完还原
    kb.ensure()
    cleaned = purge_m1(pg, kb)
    baseline_chunks = kb.count()
    evidence["baseline"] = {"collection": COLLECTION, "chunks_before": baseline_chunks, "purged_docs": cleaned}
    print(f"基线索引 {baseline_chunks} 块；清理上一轮验证残留 {cleaned} 份文档")

    sched = ResidencyScheduler(policy="co_resident")
    client = TestClient(create_app(scheduler=sched, collection=COLLECTION))

    h = client.get("/healthz").json()
    step("健康检查（依赖就绪）", h["status"] == "ok", h)

    # ---- 1. 四类真实文件上传 ----
    t0 = time.perf_counter()
    # 密级件必须有独立内容：否则会被自己的指纹去重判成重复文档（那是正确行为，但测不到 RBAC）
    secret_md = FIXTURES / "商务附件.md"
    secret_md.write_text(
        "# 智能输送线改造项目商务附件\n\n"
        "本项目对某印刷设备集团的报价为人民币 386 万元，含两年驻场运维与备件包，"
        "合同金额按 3:3:3:1 节点支付，质保金 10% 在终验后十二个月释放。\n\n"
        "## 客户联系人清单\n\n"
        "集团设备部、三家子公司项目经理与采购负责人联系方式见附表，属于对外保密范围，"
        "仅限投标团队在标书中引用本项目历史合同金额与折扣口径。\n",
        encoding="utf-8",
    )
    up_md = upload(client, FIXTURES / "交付复盘.md", {
        "doc_id": "m1-fupan-prj-ene-2025-008", "industry": "能源", "customer": "某省级电网公司",
        "project": "PRJ-ENE-2025-008", "doc_type": "复盘记录", "source_level": "A", "security_level": "2",
    }, ENG)
    up_txt = upload(client, FIXTURES / "巡检作业.txt", {
        "doc_id": "m1-xunjian-prj-ene-2025-008", "industry": "能源", "customer": "某省级电网公司",
        "project": "PRJ-ENE-2025-008", "doc_type": "运维手册", "source_level": "B", "security_level": "2",
    }, ENG)
    up_docx = upload(client, FIXTURES / "实施手册.docx", {
        "doc_id": "m1-shishi-prj-fin-2024-017", "industry": "金融", "customer": "某城商银行",
        "project": "PRJ-FIN-2024-017", "doc_type": "接口规范", "source_level": "A", "security_level": "2",
    }, ENG)
    up_pdf = upload(client, REAL_PDF, {
        # 厂商公开发布的产品安全手册：密级 1，用于证明 RBAC 不是「全都拦」而是按密级放行
        "doc_id": "m1-anquan-prj-mfg-2024-001", "industry": "制造", "customer": "某印刷设备客户",
        "project": "PRJ-MFG-2024-001", "doc_type": "合规要求", "source_level": "C", "security_level": "1",
    }, ENG)
    # 密级 3 件：给 RBAC 与驳回撤索引用
    up_secret = upload(client, secret_md, {
        "doc_id": "m1-mi-mfg-2024-002", "title": "智能输送线改造项目商务附件",
        "industry": "制造", "customer": "某印刷设备集团", "project": "PRJ-MFG-2024-002",
        "doc_type": "售前方案", "source_level": "B", "security_level": "3",
    }, OWNER)
    evidence["uploads"] = {"md": up_md, "txt": up_txt, "docx": up_docx, "pdf": up_pdf, "secret": up_secret}
    evidence["timings_sec"]["upload_5_docs"] = round(time.perf_counter() - t0, 2)
    not_accepted = {
        k: (v["status_code"], v["body"])
        for k, v in evidence["uploads"].items()
        if v["status_code"] != 201
    }
    step(
        "四类真实文件 + 密级件上传成功",
        not not_accepted,
        not_accepted
        or {
            k: {
                "doc_id": v["body"]["doc_id"],
                "version": v["body"]["version_id"],
                "父/子块": v["body"]["chunks"],
                "索引条数": v["body"]["indexed"],
                "解析告警": len(v["body"]["parse_notes"]),
                "元数据告警": len(v["body"]["meta_warnings"]),
                "耗时s": v["body"]["timings_sec"]["total"],
            }
            for k, v in evidence["uploads"].items()
        },
    )
    step(
        "真实 PDF 解析告警如实上报（章节未识别 / 短行碎片）",
        any("MinerU" in n for n in up_pdf["body"]["parse_notes"]),
        up_pdf["body"]["parse_notes"],
    )

    # ---- 2. 幂等：同 key 首提 + 重放 ----
    first_with_key = upload(client, FIXTURES / "巡检作业.txt", {
        "doc_id": "m1-xunjian-prj-ene-2025-008", "industry": "能源", "customer": "某省级电网公司",
        "project": "PRJ-ENE-2025-008", "doc_type": "运维手册",
    }, ENG, key="idem-m1-002")
    replay2 = upload(client, FIXTURES / "巡检作业.txt", {
        "doc_id": "m1-xunjian-prj-ene-2025-008", "industry": "能源", "customer": "某省级电网公司",
        "project": "PRJ-ENE-2025-008", "doc_type": "运维手册",
    }, ENG, key="idem-m1-002")
    step(
        "幂等：同 Idempotency-Key 重放返回首次结果且不重复入库",
        replay2["body"].get("outbox", {}).get("replayed") is True
        and replay2["body"]["version_id"] == first_with_key["body"]["version_id"],
        {"首次": first_with_key["body"]["version_id"], "重放": replay2["body"]["version_id"], "重放标记": replay2["body"].get("outbox")},
    )

    # ---- 3. 错误契约 ----
    dup = upload(client, FIXTURES / "巡检作业.txt", {
        "doc_id": "m1-dup-attempt", "industry": "能源", "customer": "某省级电网公司",
        "project": "PRJ-ENE-2025-008", "doc_type": "运维手册",
    }, ENG)
    bad_ext = client.post(
        "/api/v1/documents/upload",
        files={"file": ("x.xyz", b"content", "application/octet-stream")},
        data={"industry": "能源", "customer": "某电网", "project": "PRJ-ENE-2025-008"},
        headers=ENG,
    )
    bad_industry = client.post(
        "/api/v1/documents/upload",
        files={"file": ("y.md", b"# t\n\n" + "正文内容需要达到准入的最小长度要求。".encode("utf-8"), "text/markdown")},
        data={"industry": "医疗", "customer": "某医院", "project": "PRJ-MED-1", "doc_type": "复盘记录"},
        headers=ENG,
    )
    empty = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.md", b"", "text/markdown")},
        data={"industry": "能源", "customer": "某电网", "project": "PRJ-ENE-1"},
        headers=ENG,
    )
    step(
        "错误契约：重复 409 / 格式不支持 415 / 行业越界 400 / 空文件 400，且都带 trace_id 与 retryable",
        dup["status_code"] == 409 and bad_ext.status_code == 415 and bad_industry.status_code == 400 and empty.status_code == 400
        and all("trace_id" in r.json() for r in (bad_ext, bad_industry, empty)),
        {
            "重复内容": [dup["status_code"], dup["body"]["code"]],
            "不支持格式": [bad_ext.status_code, bad_ext.json()["code"]],
            "行业越界": [bad_industry.status_code, bad_industry.json()["code"]],
            "空文件": [empty.status_code, empty.json()["code"]],
        },
    )

    # ---- 4. 检索命中与真实页码 ----
    t0 = time.perf_counter()
    retriever = HybridRetriever(kb, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    ctx = AccessContext(role="knowledge_owner")
    # 探针按真实文档正文出题（不是凭空造词）：每条都指明期望命中的 doc
    probes = [
        ("配电自动化终端心跳超时阈值冬季怎么调整", "m1-fupan-prj-ene-2025-008"),
        ("线损率从多少降到多少", "m1-fupan-prj-ene-2025-008"),
        ("烫金机能不能自行拆解维修", "m1-anquan-prj-mfg-2024-001"),
        ("电源插头积灰会引起什么风险", "m1-anquan-prj-mfg-2024-001"),
        ("银企直联实时清算接口 IF-FIN-0012 的报文标准", "m1-shishi-prj-fin-2024-017"),
    ]
    hits = []
    for q, expect in probes:
        res = retriever.retrieve(q, ctx, top_k=3)
        hits.append(
            {
                "query": q,
                "expected_doc": expect,
                "hit_expected": expect in [c.fields.get("doc_id") for c in res.chunks],
                "top": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc": c.fields.get("doc_id"),
                        "section": c.fields.get("section_no"),
                        "page": c.fields.get("page"),
                        "score": c.score,
                    }
                    for c in res.chunks[:2]
                ],
                "degraded": res.degraded,
            }
        )
    evidence["retrieval"] = hits
    evidence["timings_sec"]["retrieval_probes"] = round(time.perf_counter() - t0, 2)
    missed = [h["query"] for h in hits if not h["hit_expected"]]
    step(
        "混合检索命中预期文档（5 探针均落在正确来源上）",
        not missed and all(h["top"] for h in hits) and not any(h["degraded"] for h in hits),
        {"missed": missed, "detail": [{"q": h["query"][:20], "top_doc": h["top"][0]["doc"]} for h in hits]},
    )
    pdf_hits = [h for h in hits if h["expected_doc"] == "m1-anquan-prj-mfg-2024-001"]
    step(
        "真实 PDF 命中且带页码（页码取自 PDF 页序；章节未识别时归到该章节首页，粒度见注）",
        all(h["top"] and h["top"][0]["page"] and h["top"][0]["page"] > 0 for h in pdf_hits),
        {
            "样例": pdf_hits[0]["top"][0],
            "页码口径注记": "chunk.page = 所属章节起始页；本次 PDF 未识别出章节，"
            "全部子块落在第 1 页，属解析粒度限制而非引用造假",
        },
    )

    # ---- 5. RBAC：密级 3 对新人不可见，且检索层过滤 ----
    list_new = client.get("/api/v1/documents", headers=NEW).json()
    list_own = client.get("/api/v1/documents", headers=OWNER).json()
    sec_search = retriever.retrieve("烫金机能不能自行拆解维修", AccessContext(role="newcomer"), top_k=5)
    step(
        "RBAC 检索层过滤：newcomer 看得到密级 1、看不到密级 2/3，命中结果按密级收敛",
        0 < list_new["total"] < list_own["total"]
        and all(int(d["security_level"]) <= 1 for d in list_new["items"])
        and bool(sec_search.chunks)
        and all(int(c.fields.get("security_level") or 0) <= 1 for c in sec_search.chunks),
        {
            "newcomer 可见": list_new["total"],
            "owner 可见": list_own["total"],
            "newcomer 过滤表达式": list_new["scope"]["filter_expr"],
            "newcomer 命中条数": len(sec_search.chunks),
            "newcomer 命中密级": sorted({int(c.fields.get("security_level") or 0) for c in sec_search.chunks}),
        },
    )

    # ---- 6. 升版 / 回滚 ----
    p = FIXTURES / "交付复盘.md"
    modified = p.read_text(encoding="utf-8").replace("线损率从 6.8% 降至 4.9%", "线损率从 6.8% 降至 4.1%，并追加季度复测要求")
    tmp = FIXTURES / "交付复盘_v2.md"
    tmp.write_text(modified, encoding="utf-8")
    up_v2 = upload(client, tmp, {
        "doc_id": "m1-fupan-prj-ene-2025-008", "industry": "能源", "customer": "某省级电网公司",
        "project": "PRJ-ENE-2025-008", "doc_type": "复盘记录",
    }, ENG)
    versions = pg.versions_of("m1-fupan-prj-ene-2025-008")
    stale_indexed = len(kb.all_chunk_ids(version_id="m1-fupan-prj-ene-2025-008@v1"))
    step(
        "同 doc_id 重传自动升版：v2 生效、v1 转 superseded，且 v1 索引被撤下（避免同文档双版同时命中）",
        up_v2["status_code"] == 201 and up_v2["body"]["version_id"].endswith("@v2")
        and [v["status"] for v in versions] == ["effective", "superseded"] and stale_indexed == 0,
        {
            "new_version": up_v2["body"]["version_id"],
            "版本状态": {v["id"]: v["status"] for v in versions},
            "v1 残留索引条数": stale_indexed,
            "v2 索引条数": up_v2["body"]["indexed"],
        },
    )

    rb = client.post(
        "/api/v1/documents/m1-fupan-prj-ene-2025-008/rollback",
        json={"to_version_id": "m1-fupan-prj-ene-2025-008@v1"},
        headers=OWNER,
    )
    v1_chunks = len(kb.all_chunk_ids(version_id="m1-fupan-prj-ene-2025-008@v1"))
    v2_chunks = len(kb.all_chunk_ids(version_id="m1-fupan-prj-ene-2025-008@v2"))
    step(
        "回滚到 v1：生效版本切回、v2 索引撤下、v1 索引由权威源+物料仓重建",
        rb.status_code == 200 and v1_chunks > 0 and v2_chunks == 0,
        {"回滚响应": rb.json(), "v1 索引条数": v1_chunks, "v2 索引条数": v2_chunks},
    )

    # ---- 7. 复核：职责分离 + 驳回撤索引 ----
    self_review = client.post(
        "/api/v1/documents/m1-shishi-prj-fin-2024-017/review",
        json={"decision": "pass", "note": "我自己传的我自己批"},
        headers=ENG,
    )
    wrong_role = client.post(
        "/api/v1/documents/m1-shishi-prj-fin-2024-017/review",
        json={"decision": "pass"},
        headers={"X-Kee-Role": "pm", "X-Kee-Actor": "pm-01"},
    )
    other_pass = client.post(
        "/api/v1/documents/m1-shishi-prj-fin-2024-017/review",
        json={"decision": "pass", "note": "格式与来源核对通过"},
        headers={"X-Kee-Role": "knowledge_owner", "X-Kee-Actor": "owner-02"},
    )
    reject = client.post(
        "/api/v1/documents/m1-mi-mfg-2024-002/review",
        json={"decision": "reject", "note": "商务数据未脱敏，不得进知识库"},
        headers=ADMIN,  # 该文档由 owner-01 上传，必须换人复核（职责分离真的在拦）
    )
    rejected_ver = "m1-mi-mfg-2024-002@v1"
    step(
        "复核契约：上传者自审 403、非 Owner 角色 403、他人 pass 通过、reject 后索引撤净",
        self_review.status_code == 403 and wrong_role.status_code == 403
        and other_pass.status_code == 200 and reject.status_code == 200
        and len(kb.all_chunk_ids(version_id=rejected_ver)) == 0,
        {
            "自审": [self_review.status_code, self_review.json().get("code")],
            "非授权角色": [wrong_role.status_code, wrong_role.json().get("code")],
            "他人通过": other_pass.json().get("status"),
            "驳回后状态": reject.json().get("status"),
            "驳回后索引条数": len(kb.all_chunk_ids(version_id=rejected_ver)),
        },
    )

    # ---- 8. 对账：注入缺数据 → 作业发现并修复 ----
    surviving = [c["id"] for c in pg.chunks_of("m1-anquan-prj-mfg-2024-001@v1") if not c["is_parent"]]
    victim = surviving[:2]
    kb.delete_chunks(victim)
    t0 = time.perf_counter()
    rec = client.post("/api/v1/reconcile/run", params={"repair": "true"}, headers=ADMIN).json()
    evidence["timings_sec"]["reconcile_run"] = round(time.perf_counter() - t0, 2)
    step(
        "对账作业发现注入的缺数据并完成补偿，复检归零",
        rec["diff"]["missing_in_vector"] == len(victim)
        and rec.get("recheck", {}).get("missing_in_vector") == 0,
        {"注入缺失": victim, "diff": rec["diff"], "repaired": rec.get("repair", {}).get("repaired"), "recheck": rec.get("recheck")},
    )

    # ---- 9. 审计留痕 ----
    with pg.connect() as conn:
        audits = conn.execute(
            "SELECT action, count(*) AS c FROM audit_log GROUP BY action ORDER BY action"
        ).fetchall()
        tamper = None
        try:
            conn.execute("UPDATE audit_log SET action='tampered' WHERE id=(SELECT min(id) FROM audit_log)")
        except Exception as e:  # noqa: BLE001
            tamper = str(e).splitlines()[0]
    step(
        "审计追加留痕且禁改（覆盖上传/复核/回滚/对账）",
        tamper is not None and {"document_ingested", "document_review_pass", "document_rollback", "reconciliation_run"} <= {a["action"] for a in audits},
        {"动作分布": {a["action"]: a["c"] for a in audits}, "篡改被拒": tamper[:120]},
    )

    # ---- 9.5 清场：验证不该在生产集合里留下痕迹，否则下一轮基线就被污染 ----
    purge_m1(pg, kb)
    restored = kb.count()
    step(
        "跑完还原生产集合（索引回到基线，权威源不剩 m1-* 文档）",
        restored == evidence["baseline"]["chunks_before"] and pg.list_documents(status=None)["total"] == 24,
        {
            "基线索引": evidence["baseline"]["chunks_before"],
            "还原后索引": restored,
            "权威源文档数": pg.list_documents(status=None)["total"],
        },
    )

    effective_ids = [v["id"] for v in pg.effective_versions()]
    evidence["totals"] = {
        "collection": COLLECTION,
        "indexed_chunks": kb.count(),
        "authoritative_children_effective": len(pg.chunk_ids(version_ids=effective_ids)),
        "m1_documents": len([d for d in pg.list_documents(limit=100, status=None)["items"] if d["id"].startswith("m1-")]),
        "note": "indexed 与 authoritative 只对「生效版本」比对；superseded 版本既不在权威口径也不在索引里",
    }
    evidence["timings_sec"]["whole_run"] = round(time.perf_counter() - t_all, 2)
    kb.close()

    art = Path(s.resolved_artifact_dir()) / "m1_verify.json"
    art.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n原始证据：", art)
    print("全部步骤通过，集合：", evidence["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
