"""M2 表格检索专项：结构还原入库之后，单元格级问题能不能真的检索到那张表。

为什么单独立项：主评测集是生成的事实卡、不含表格，所以「表格结构已接」这句话
在 M1 只验证到「块存在且被切分」，没有验证到「用户问某个单元格时能命中它」。
这一项补的就是这个缺口，并且明确不参与主基线（跑在独立集合 kb_m2_tables 上），
否则会把与 Gate 0 可比的 193 块基线污染掉。

gold 口径是**结构式**的而不是关键词回查：一道题的 gold = 应答文档里所有含 Markdown
表格的块。这样定义的理由写进报告，不假装它是人工相关性标注。
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.db.vector_store import VectorStore  # noqa: E402
from kee.eval.metrics import aggregate_qa_metrics, latency_report  # noqa: E402
from kee.ingest.service import IngestionService  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

COLLECTION = "kb_m2_tables"
FIX = ROOT / "var" / "m1" / "fixtures"
SCAN_PDF = FIX / "扫描件_产品安全手册第1页.pdf"

META = {
    "industry": "制造",
    "customer": "表格检索专项用合成客户",
    "project": "PRJ-M2-TABLES",
    "doc_type": "合规要求",
    "source_level": "C",
    "security_level": "1",
}

# (doc_id, 文件, [(问题, gold 行特征词)])。
# 问题与特征词一律从**实际还原出来的表格文本**里出，不凭空造术语——凭空造的探针词
# 命中不到任何东西，测的是探针而不是系统。
# gold 定义随行级分块一起改：分块下沉到一行一块后，「该文档所有表格块」不再是正确 gold
# （答对一行会被记成 recall 1/7），gold 收窄到**含答案那一行的表格块**。
CASES: list[tuple[str, Path, list[tuple[str, str]]]] = [
    (
        "m2tbl-hak180",
        SCAN_PDF,
        [
            ("HAK180 的框架 L 单元里铅的含量标识是什么", "框架 L 单元"),
            ("顶盖单元是否含有六价铬", "顶盖单元"),
            ("包装材料的多溴二苯醚标识结果", "包装材料"),
            ("主电路板在有害物质表里汞的标记", "主电路板"),
        ],
    ),
    (
        "m2tbl-env3",
        FIX / "表格_普通三列表.pdf",
        [("高温工作的要求值是多少", "高温工作"), ("振动等级的判定结果", "振动等级")],
    ),
    (
        "m2tbl-mat7",
        FIX / "表格_合并表头七列表.pdf",
        [("电源模块的铅标识是什么", "电源模块"), ("线缆组件的汞含量标识", "线缆组件")],
    ),
    (
        "m2tbl-ms2",
        FIX / "表格_两列长表.pdf",
        [("试运行阶段要交付什么", "试运行"), ("结项阶段的交付物", "结项")],
    ),
    (
        "m2tbl-wide6",
        FIX / "表格_宽表六列.pdf",
        [("华东区域的年成本是多少", "华东"), ("西南区域的单位能耗", "西南")],
    ),
]


def purge(pg: AuthoritativeStore, kb: KbStore, vs: VectorStore) -> int:
    with pg.connect() as conn:
        stale = [r["id"] for r in conn.execute("SELECT id FROM document WHERE id LIKE 'm2tbl-%'").fetchall()]
        for d in stale:
            for ver in pg.versions_of(d):
                kb.delete_version(ver["id"])
                vs.drop(ver["id"])
            conn.execute("DELETE FROM chunk WHERE doc_id=%s", (d,))
            conn.execute("DELETE FROM document_version WHERE doc_id=%s", (d,))
            conn.execute("DELETE FROM outbox_event WHERE aggregate_id LIKE %s", (f"{d}%",))
            conn.execute("DELETE FROM document WHERE id=%s", (d,))
    return len(stale)


def main() -> int:
    s = get_settings()
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    vs = VectorStore()
    sched = ResidencyScheduler(policy="co_resident")
    for f, _ in [(c[1], 0) for c in CASES]:
        if not f.exists():
            print(f"缺样本：{f}（先跑 scripts/table_fixtures.py 生成合成表格）")
            return 2
    missing = [c[0] for c in CASES if not c[1].exists()]
    if missing:
        print(f"缺文件，无法执行：{missing}")
        return 2

    cleaned = purge(pg, KbStore(COLLECTION, "jieba"), vs)
    kb = KbStore(COLLECTION, "jieba")
    kb.drop()
    kb.ensure()
    svc = IngestionService(pg=pg, kb=kb, vectors=vs, scheduler=sched)

    ingested: dict[str, Any] = {}
    with sched.use("embed"):
        for doc_id, path, _qs in CASES:
            t0 = time.perf_counter()
            r = svc.ingest(
                filename=path.name,
                data=path.read_bytes(),
                form={"doc_id": doc_id, **META, "title": doc_id},
                actor="m2-table-probe",
                trace_id=f"m2tbl-{doc_id}",
            )
            ingested[doc_id] = {
                "version": r.version_id,
                "parent": r.chunks_parent,
                "child": r.chunks_child,
                "indexed": r.indexed,
                "sec": round(time.perf_counter() - t0, 1),
                "table_note": [n for n in r.parse_notes if "表格结构" in n],
            }
            print(f"[入库] {doc_id:14s} 父{r.chunks_parent} 子{r.chunks_child} 索引{r.indexed} "
                  f"{ingested[doc_id]['sec']}s {r.parse_notes and ''}")

    # 每个文档的表格块（含 Markdown 分隔行），gold 再按行特征词收窄
    with pg.connect() as conn:
        table_chunks: dict[str, list[dict[str, str]]] = {}
        for doc_id, _p, _q in CASES:
            rows = conn.execute(
                "SELECT id, content FROM chunk WHERE doc_id=%s AND is_parent=false ORDER BY order_no", (doc_id,)
            ).fetchall()
            table_chunks[doc_id] = [
                {"id": x["id"], "content": x["content"]} for x in rows if "（表行）" in x["content"]
            ]
    n_no_table = [d for d, g in table_chunks.items() if not g]
    if n_no_table:
        print(f"❌ 这些文档没找到表格块，专项无法判定：{n_no_table}")

    def gold_of(doc_id: str, term: str) -> set[str]:
        # 空格不敏感：OCR 对同一列名在不同行会给出不一致的空白（实测「框架 L 单元」与
        # 「框架L单元」并存），按字面匹配会把探针自己的写法差异误判成检索失败
        key = re.sub(r"\s+", "", term)
        return {c["id"] for c in table_chunks[doc_id] if key in re.sub(r"\s+", "", c["content"])}

    retriever = HybridRetriever(kb, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    ctx = AccessContext(role="knowledge_owner")
    rows, lat, detail, miss = [], [], [], []
    with sched.use("embed"):
        pass
    warm = time.perf_counter()
    retriever.retrieve("预热：高温工作的要求值", ctx, top_k=5)
    print(f"[预热] {round((time.perf_counter() - warm) * 1000)}ms 不计入延迟")
    for doc_id, _path, questions in CASES:
        for q, term in questions:
            gold = gold_of(doc_id, term)
            if not gold:
                print(f"❌ {doc_id} 里找不到含「{term}」的表格块，题目或还原有问题")
                continue
            res = retriever.retrieve(q, ctx, top_k=5)
            ranked = [c.chunk_id for c in res.chunks]
            rows.append({"ranked_ids": ranked, "gold_ids": gold})
            lat.append(res.timings_ms.get("retrieval_total_ms", 0.0))
            top1 = ranked[0] if ranked else ""
            ok = bool(set(ranked) & gold)
            detail.append({"doc": doc_id, "q": q, "term": term, "hit": ok, "top1_is_gold": top1 in gold,
                           "n_gold": len(gold)})
            if not ok:
                miss.append({"doc": doc_id, "q": q, "gold": sorted(gold), "top5": ranked})

    out = aggregate_qa_metrics(rows, k=5)
    out["latency_ms"] = latency_report(lat)
    out["n"] = len(rows)
    out["top1_is_gold_rate"] = round(sum(1 for d in detail if d["top1_is_gold"]) / len(detail), 4)
    out["per_question"] = detail
    out["misses"] = miss
    out["ingest"] = ingested
    out["table_chunk_count_by_doc"] = {d: len(v) for d, v in table_chunks.items()}
    out["collection"] = COLLECTION
    out["chunk_count"] = kb.count()

    print(f"\n[表格检索专项] n={out['n']} Recall@5={out.get('recall@5')} Hit@5={out.get('hit@5')} "
          f"MRR={out.get('mrr')} Top1即gold率={out['top1_is_gold_rate']} P95={out['latency_ms']['p95']}ms")
    for d in detail:
        print(f"  {'✅' if d['hit'] else '❌'}{'(top1)' if d['top1_is_gold'] else '      '} {d['doc']:14s} {d['q']}")

    # 跑完还原：专项不留残渣，主基线集合不受影响
    purge(pg, kb, vs)
    kb.drop()
    kb.close()
    canon = KbStore(s.milvus_collection, "jieba")
    baseline = canon.count()
    canon.close()
    out["restored_baseline_chunks"] = baseline
    art.mkdir(parents=True, exist_ok=True)
    (art / "m2_table_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[还原] 专项文档已清、集合已删；主基线 {s.milvus_collection} 剩 {baseline} 块")
    print("证据：", art / "m2_table_probe.json")
    return 0 if out.get("hit@5", 0) >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
