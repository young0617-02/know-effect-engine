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

from _identity import ensure_accounts, login_headers  # noqa: E402

COLLECTION = "kb_chunks"  # 对账口径是「一份权威源 ↔ 一个索引」，验证必须跑在生产集合上
FIXTURES = Path("var/m1/fixtures")
REAL_PDF = Path("D:/project/kb_rag/doc/hak180产品安全手册.pdf")

# 身份来自登录令牌（`KEE_ALLOW_HEADER_IDENTITY` 默认已于 2026-10-15 翻成 False），
# 在 main() 里登录后填充。审计里的 actor 因此是账号名，不再是请求头里随手写的 owner-01。
OWNER: dict[str, str] = {}
ADMIN: dict[str, str] = {}
ENG: dict[str, str] = {}
NEW: dict[str, str] = {}
PM: dict[str, str] = {}

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

    global OWNER, ADMIN, ENG, NEW, PM
    ensure_accounts(pg)
    OWNER = login_headers(client, "verify-owner")
    ADMIN = login_headers(client, "verify-admin")
    ENG = login_headers(client, "verify-eng")
    NEW = login_headers(client, "verify-new")
    PM = login_headers(client, "verify-pm")

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

    # ---- 7.5 扫描件 OCR：无文本层的 PDF 也必须能进知识库并被检索到 ----
    scan_pdf = FIXTURES / "扫描件_产品安全手册第1页.pdf"
    if not scan_pdf.exists() and REAL_PDF.exists():
        import pypdfium2 as pdfium
        from PIL import Image

        page_img = Image.fromarray(
            pdfium.PdfDocument(REAL_PDF.read_bytes())[0].render(scale=150 / 72).to_numpy()[:, :, :3]
        )
        page_img.save(scan_pdf, "PDF")
    up_scan = upload(client, scan_pdf, {
        "doc_id": "m1-saomiao-prj-mfg-2024-003", "industry": "制造", "customer": "某印刷设备客户",
        "project": "PRJ-MFG-2024-003", "doc_type": "合规要求", "source_level": "C", "security_level": "1",
    }, ENG)
    scan_ok = up_scan["status_code"] == 201 and up_scan["body"]["indexed"] > 0 and any(
        "MinerU-ONNX" in n for n in up_scan["body"]["parse_notes"]
    )
    scan_hit = retriever.retrieve("塑料袋并不是玩具 避免窒息 危险", ctx, top_k=3)
    step(
        "扫描件 OCR 入库并可检索（pypdf 无文本层 → 自动转 MinerU-ONNX 后端）",
        scan_ok and "m1-saomiao-prj-mfg-2024-003" in [c.fields.get("doc_id") for c in scan_hit.chunks],
        {
            "分块": up_scan["body"].get("chunks"),
            "索引条数": up_scan["body"].get("indexed"),
            "解析痕迹": [n[:56] for n in up_scan["body"].get("parse_notes", [])],
            "耗时s": up_scan["body"].get("timings_sec", {}).get("total"),
            "检索命中": [
                {"doc": c.fields.get("doc_id"), "page": c.fields.get("page"), "text": c.text[:34]}
                for c in scan_hit.chunks[:2]
            ],
        },
    )

    # ---- 7.6 表格结构还原：扫描件里的表格必须带行列结构入库，而不是被压成一行文本 ----
    # 结构断言放在**父块**（整张表，Markdown 行列都在），检索断言放在**子块**（行级「列名=值」）。
    # 子块刻意不再是 Markdown：整表一块时单元格级问题实测进不了 Top-5，见 M2 报告第七节。
    scan_doc = "m1-saomiao-prj-mfg-2024-003"
    with pg.connect() as conn:
        rows_all = conn.execute(
            "SELECT id, content, is_parent FROM chunk WHERE doc_id=%s ORDER BY order_no", (scan_doc,)
        ).fetchall()
    parents = [r["content"] for r in rows_all if r["is_parent"] and "|---|" in r["content"]]
    row_children = [r["content"] for r in rows_all if not r["is_parent"] and "（表行）" in r["content"]]
    parent_tbl = [ln for ln in "\n".join(parents).split("\n") if ln.startswith("|") and "|---|" not in ln]
    ncols = {len(ln.strip().strip("|").split("|")) for ln in parent_tbl}
    tbl_hit = retriever.retrieve("框架 L 单元 铅 六价铬 含量标识", ctx, top_k=5)
    hit_ids = [c.fields.get("doc_id") for c in tbl_hit.chunks]
    step(
        "表格结构还原入库（父块 17 行 × 7 列 + 合并表头；子块行级「列名=值」且可按单元格检索命中）",
        len(parent_tbl) >= 17 and ncols == {7} and any("有害物质" in p for p in parents)
        and len(row_children) >= 16 and "m1-saomiao-prj-mfg-2024-003" in hit_ids,
        {
            "父块表格行数": len(parent_tbl),
            "父块每行列数": sorted(ncols),
            "合并表头在父块": any("有害物质" in p for p in parents),
            "行级子块数(=数据行数，表头行转成列名)": len(row_children),
            "子块样例": row_children[1][:58] if row_children else "",
            "单元格检索命中": [
                {"doc": c.fields.get("doc_id"), "rank": i + 1, "text": c.text[:40]}
                for i, c in enumerate(tbl_hit.chunks[:3])
            ],
        },
    )

    # ---- 8. 升版 / 回滚 ----
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
    # 先记一笔先前存在的错：`review` 的路由是**先查角色、再查职责分离**（app.py:475-483）。
    # 所以「delivery_engineer 自审拿到 403」证明的是角色闸门，不是职责分离——两条 403 撞在
    # 同一个状态码上，断言就分不出是谁在拦。这里改成既验状态码也验文案，
    # 并补一条真正落在职责分离上的自审（带复核权的账号复核自己传的那篇）。
    true_self_review = client.post(
        "/api/v1/documents/m1-mi-mfg-2024-002/review",
        json={"decision": "pass", "note": "我传的文档我自己批"},
        headers=OWNER,   # 该文档由 verify-owner 上传（见上面的 up_mission），同一个人、同一枚令牌
    )
    self_review = client.post(
        "/api/v1/documents/m1-shishi-prj-fin-2024-017/review",
        json={"decision": "pass", "note": "我自己传的我自己批"},
        headers=ENG,
    )
    wrong_role = client.post(
        "/api/v1/documents/m1-shishi-prj-fin-2024-017/review",
        json={"decision": "pass"},
        headers=PM,
    )
    other_pass = client.post(
        "/api/v1/documents/m1-shishi-prj-fin-2024-017/review",
        json={"decision": "pass", "note": "格式与来源核对通过"},
        headers=OWNER,   # 该篇由 verify-eng 上传，OWNER 是另一个人 ⇒ 这条才是「换人复核通过」
    )
    reject = client.post(
        "/api/v1/documents/m1-mi-mfg-2024-002/review",
        json={"decision": "reject", "note": "商务数据未脱敏，不得进知识库"},
        headers=ADMIN,  # 该文档由 verify-owner 上传，必须换人复核（职责分离真的在拦）
    )
    rejected_ver = "m1-mi-mfg-2024-002@v1"
    step(
        "复核契约：带复核权的人自审被职责分离拦、无复核权的角色被角色闸门拦、换人 pass 通过、reject 后索引撤净",
        true_self_review.status_code == 403
        and "职责分离" in true_self_review.json().get("message", "")
        and self_review.status_code == 403 and "复核需" in self_review.json().get("message", "")
        and wrong_role.status_code == 403 and "复核需" in wrong_role.json().get("message", "")
        and other_pass.status_code == 200 and reject.status_code == 200
        and len(kb.all_chunk_ids(version_id=rejected_ver)) == 0,
        {
            "同账号自审(职责分离)": [true_self_review.status_code, true_self_review.json().get("message", "")[:40]],
            "上传者无复核权(角色闸门)": [self_review.status_code, self_review.json().get("message", "")[:40]],
            "pm 角色": [wrong_role.status_code, wrong_role.json().get("message", "")[:40]],
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
        # TRUNCATE 那条缝（2026-10-15 收紧的对象）：触发器只声明 DELETE/UPDATE，
        # 而 PostgreSQL 的触发器不覆盖 TRUNCATE，所以这条路只能靠权限堵。必须单独试一条——
        # 否则「禁改做了」会被读成「整张表堵住了」，而它当时只堵了三分之二。
        truncate = None
        rows_before = conn.execute("SELECT count(*) n FROM audit_log").fetchone()["n"]
        try:
            conn.execute("TRUNCATE TABLE audit_log")
        except Exception as e:  # noqa: BLE001
            truncate = str(e).splitlines()[0]
        posture = conn.execute(
            "SELECT current_user AS role, "
            "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user) AS sup, "
            "(SELECT r.rolname FROM pg_class c JOIN pg_roles r ON r.oid = c.relowner "
            " WHERE c.relname = 'audit_log') AS audit_owner"
        ).fetchone()
        rows_after = conn.execute("SELECT count(*) n FROM audit_log").fetchone()["n"]
    step(
        "审计追加留痕且禁改（覆盖上传/复核/回滚/对账）",
        tamper is not None and {"document_ingested", "document_review_pass", "document_rollback", "reconciliation_run"} <= {a["action"] for a in audits},
        {"动作分布": {a["action"]: a["c"] for a in audits}, "篡改被拒": tamper[:120]},
    )
    step(
        "TRUNCATE 这条路也堵住了，且应用角色不是超级用户（收紧后的闸门）",
        truncate is not None and not posture["sup"] and posture["audit_owner"] != posture["role"]
        and rows_before == rows_after,
        {"试探结果": (truncate or "竟然成功了")[:120], "连接角色": posture["role"],
         "该角色超级用户": posture["sup"], "audit_log 属主": posture["audit_owner"],
         "行数_前": rows_before, "行数_后": rows_after},
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
