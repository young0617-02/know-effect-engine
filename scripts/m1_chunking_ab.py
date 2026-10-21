"""Parent-Child vs 固定分块：把同批内容重塑成「长章节」，两套分块各建一个 scratch 集合做真对照。

**为什么必须换形状**（C7 的来历，实测不是推测）：生产语料里父块正文被它自家子块覆盖的比例中位 **1.000**、
父块字数减子块字数合计中位 **28 字**、**62.14% 的父块只含 1 个子块** ⇒ 「父块」实质是子块副本加一行抬头，
两方案在这份语料上近乎产出同一批块。所以上一轮四档消融那句「接上父块无收益」只对**这个形状**成立，
拿它当「Parent-Child 没用」是把不可分辨的对照当成结论。本轮把相邻小节并成**长章节**（父块相对子块有数倍增量），
再回答「相对固定分块到底有没有效果」。

**三处设计是刻意的**：
1. **不写 PG、不动生产**：只建两个 scratch Milvus 集合（`kb_chunkab_p` / `kb_chunkab_f`），跑完删。
   ⇒ 生产 193 块 / 24 篇、strict gold、547 行绑子块的人工标注**全部零作废**（G5 跑前后核计数）。
   这条把用户选项 ③ 原本「子块 id 全变 ⇒ 人工标注作废」的代价直接消掉了。
2. **gold 不靠人标、也不靠关键词**：每题的 gold 由**原生产子块正文里的答案句**（≥12 字，切句后）派生，两臂用**同一个包含判定**
   （去掉全部空白后取子串）派生。关键词法天然偏爱「按章节切」的那臂，是这类实验最容易造出的假绿。
   代价：表行子块渲染成「列名=值」，原文里不存在，这类题**结构性剔除**并报数（G0）。
3. **两臂唯一的差别是分块函数**：P 用生产 `chunk_document`（章内按段、≤280、overlap 40、带 `标题｜` 前缀），
   F 用**整篇文档 280/40 全局滑窗**（无前缀、可跨章）。题面、检索参数、模型、驻留策略全同（G1）。

生成层四档，前两档比检索方案，后两档比「父块怎么接」：
  F_flat        F 窗口块直接作答案面
  P_child       P 子块作答案面（现状那套，只是语料变长）
  P_parent      P 子块 + 整章原文逐条贴（**复刻上一轮 A1 的笨接法**，含重复）
  P_parent_trim P 子块 + 整章中**没被任何被引子块覆盖的段落**（同章只贴一次）
加第四档的理由：上一轮「无收益」可能测的是「把在场文字重贴五遍」而不是「长上下文没用」。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import percentile  # noqa: E402
from kee.ingest.chunking import TABLE_ROW_MARK, chunk_document  # noqa: E402
from kee.ingest.corpus_data import FACT_CARDS  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa import graph as graph_mod  # noqa: E402
from kee.qa.answer import AnswerEngine, format_evidence, load_calibration_temperature  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

ART = ROOT / "var" / "gate0"
REPORT = ROOT / "reports" / "m1" / "M1_分块策略对照.md"
COLL_P = "kb_chunkab_p"
COLL_F = "kb_chunkab_f"
PROD_COLL = "kb_chunks"
CHILD_MAX, OVERLAP, PARENT_MAX = 280, 40, 2400
GEN_ARMS = ("F_flat", "P_child", "P_parent", "P_parent_trim")
ARM_INDEX = {"F_flat": "F", "P_child": "P", "P_parent": "P", "P_parent_trim": "P"}
MATERIAL_QUESTION_UNITS = 2.0
RET_KEYS = ("hit@5", "sentence_recall@5", "mrr")
WARM_QUERIES = ("预热：配电自动化终端心跳超时阈值", "预热：清算链路灾备切换目标 RTO")

CURRENT = {"arm": "P_child", "parents": {}, "seen": set(), "injected": 0, "evidence_chars": 0}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


# ---------------------------------------------------------------- 语料重塑
def reshape_card(card: dict[str, Any], per: int) -> dict[str, Any]:
    """相邻 `per` 个小节并成一章：段落数不变、正文一字不改，只把章节边界抬高。

    每个原小节合成**一个长段**——这正是让 P 臂也经历「段落被窗口切开」的原因：
    不这么做的话 P 的子块会永远落在段落边界上，等于给对照组送一个它结构上拿不到的便宜。
    """
    secs = card["sections"]
    groups = [secs[i:i + per] for i in range(0, len(secs), per)]
    sections = []
    for gi, g in enumerate(groups, start=1):
        paras, heads = [], []
        for s in g:
            head = f"{s['section_no']} {s.get('heading', '') or ''}".strip()
            heads.append(str(s.get("heading", "") or s["section_no"]))
            body = " ".join(str(p) for p in (s.get("paragraphs") or []))
            if body.strip():
                paras.append(f"{head}　{body}")
        sections.append({"section_no": f"{gi}", "page": int(g[0]["page"]),
                         "heading": "／".join(heads)[:36], "paragraphs": paras})
    return {**{k: v for k, v in card.items() if k != "sections"}, "sections": sections}


def fixed_windows(text: str, size: int = CHILD_MAX, overlap: int = OVERLAP) -> list[tuple[int, str]]:
    stride = max(size - overlap, 1)
    out = []
    for start in range(0, len(text), stride):
        piece = text[start:start + size]
        if piece.strip():
            out.append((start, piece))
        if start + size >= len(text):
            break
    return out


def build_two_arm_chunks(merged: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """P 臂走生产分块器；F 臂把同一篇正文展平成一条流再滑窗。两臂内容字面完全同源。"""
    p_children: list[dict[str, Any]] = []
    parents: dict[str, str] = {}
    f_children: list[dict[str, Any]] = []
    f_span_chapters: dict[str, list[str]] = {}
    for card in merged:
        vid = f"{card['doc_id']}@ab"
        rows = chunk_document(card, vid, child_max=CHILD_MAX, parent_max=PARENT_MAX, overlap=OVERLAP)
        pars = [r for r in rows if r["is_parent"]]
        kids = [r for r in rows if not r["is_parent"]]
        for pr in pars:
            parents[pr["id"]] = pr["content"]
        for c in kids:
            c["version_id"] = vid
            p_children.append(c)
        # F 臂：整篇文档流（章序、段序不变）
        stream_parts, marks = [], []
        pos = 0
        for si, sec in enumerate(card["sections"]):
            for para in sec["paragraphs"]:
                stream_parts.append(para)
                marks.append((pos, pos + len(para), sec["section_no"]))
                pos += len(para) + 1
            stream_parts.append("")  # 段间换行
        stream = "\n".join(stream_parts)
        for wi, (start, piece) in enumerate(fixed_windows(stream), start=1):
            touched = sorted({m[2] for m in marks if m[0] < start + len(piece) and m[1] > start})
            head_chapter = touched[0] if touched else card["sections"][0]["section_no"]
            sec = next(s for s in card["sections"] if s["section_no"] == head_chapter)
            cid = f"{vid}#w{wi}"
            f_children.append({
                "id": cid, "version_id": vid, "doc_id": card["doc_id"], "parent_id": None,
                "section_no": head_chapter, "page": int(sec["page"]), "order_no": wi,
                "content": piece, "char_count": len(piece), "is_parent": False,
            })
            f_span_chapters[cid] = touched
    return ({"children": p_children, "parents": parents},
            {"children": f_children, "chapters": f_span_chapters})


# ---------------------------------------------------------------- 索引
def insert_arm(kb: KbStore, rows: list[dict[str, Any]], card_meta: dict[str, dict],
               vectors: dict[str, list[float]]) -> int:
    kb.ensure()
    docs = []
    for i, r in enumerate(rows):
        m = card_meta[r["doc_id"]]
        docs.append({
            "chunk_id": r["id"], "doc_id": r["doc_id"], "version_id": r["version_id"],
            "parent_id": r["parent_id"] or "", "industry": m["industry"], "customer": m["customer"],
            "project": m["project"], "doc_type": m["doc_type"], "source_level": m["source_level"],
            "security_level": int(m["security_level"]), "section_no": r["section_no"],
            "page": int(r["page"]), "order_no": i + 1, "text": r["content"],
            "dense": vectors[r["id"]],
        })
    n = 0
    for i in range(0, len(docs), 200):
        n += kb.insert(docs[i:i + 200])
    kb.flush()
    return n


# ---------------------------------------------------------------- gold：span 派生
def spans_for(q: dict[str, Any], prod_content: dict[str, str]) -> tuple[list[str], str]:
    """返回 (答案句列表, 剔除原因)。

    **判据单位是句子，不是整条子块正文**——review 时改的：原本拿整条 span（中位 251 字）做
    「块必须整条包含」的判定，而窗口块只有 280 字 ⇒ 判据天然只奖励与 span 同形状的那臂
    （P 的子块本来就是按段落切的，几乎总能整条包含自己）。那是在比分据形状，不是在比分块。
    切成句（≥12 字）之后两臂都要重新挣分，而**把答案句切断正是固定滑窗的真实代价**，
    这条代价此前被我的判据掩盖成「F 臂不行」。
    """
    out: list[str] = []
    for cid in q["gold_strict_ids"]:
        text = prod_content.get(cid)
        if text is None:
            return [], "gold 子块不在权威源里"
        if TABLE_ROW_MARK in text:
            return [], "表行子块渲染成「列名=值」，原文里没有这个串"
        body = text.split("｜", 1)[-1] if "｜" in text else text
        for piece in re.split(r"[。；！？;\n]", body):
            n = norm(piece)
            if len(n) >= 12:
                out.append(n)
    if not out:
        return [], "gold 子块里没有 ≥ 12 字的句子"
    return out, ""


def arm_gold(arm_rows: list[dict[str, Any]], spans: dict[str, list[str]]) -> dict[str, set]:
    """某臂里「包含该题任一 span」的块 = 该臂该题的 gold。**两臂同一函数**。"""
    out = {q: set() for q in spans}
    for r in arm_rows:
        t = norm(r["content"])
        for q, ss in spans.items():
            if any(s in t for s in ss):
                out[q].add(r["id"])
    return out


def retrieval_metrics(res_by_q: dict[str, Any], spans: dict[str, list[str]],
                      gold_ids: dict[str, set]) -> dict[str, Any]:
    hit = rec_sum = mrr_sum = n = 0
    recs = []
    for q, res in res_by_q.items():
        ss = spans[q]
        texts = [norm(c.text) for c in res.chunks]
        covered = [any(s in t for t in texts) for s in ss]
        ranks = [i + 1 for i, t in enumerate(texts) if any(s in t for s in ss)]
        r5 = sum(covered) / len(ss)
        recs.append(r5)
        hit += 1 if any(covered) else 0
        mrr_sum += (1.0 / ranks[0]) if ranks else 0.0
        n += 1
    return {"n": n, "hit@5": round(hit / n, 4) if n else None,
            "sentence_recall@5": round(sum(recs) / n, 4) if n else None,
            "mrr": round(mrr_sum / n, 4) if n else None,
            "n_gold_chunks_total": sum(len(v) for v in gold_ids.values())}


def trimmed_parent_body(parent_text: str, cited_children: list[str], min_run: int = 12) -> str:
    """父块里**没被任何被引子块覆盖**的部分（按去空白字符掩码算，粒度比段落细）。

    去重单元必须是「子块的子串」而不是「段落」：重塑后一个段落就是一整个小节（~500 字），
    而子块 ≤280 字 ⇒ 判「段落被子块包含」永远不成立，trim 会退化成整章照贴，
    于是 P_parent 与 P_parent_trim 变成同一个档，这一档就白设了（写的时候没看出来，review 时照出来的）。
    """
    body = parent_text.split("】", 1)[-1] if parent_text.startswith("【") else parent_text
    pn = norm(body)
    if not pn:
        return ""
    mask = bytearray(b"\x01") * len(pn)
    for ct_raw in cited_children:
        cn = norm(ct_raw.split("｜", 1)[-1] if "｜" in ct_raw else ct_raw)
        if len(cn) < 4:
            continue
        start = pn.find(cn)
        while start >= 0:
            for i in range(start, start + len(cn)):
                mask[i] = 0
            start = pn.find(cn, start + 1)
    runs, cur = [], []
    for ch, keep in zip(pn, mask):
        if keep:
            cur.append(ch)
        elif cur:
            if len(cur) >= min_run:
                runs.append("".join(cur))
            cur = []
    if cur and len(cur) >= min_run:
        runs.append("".join(cur))
    return "／".join(runs)


# ---------------------------------------------------------------- 答案面改写
def arm_evidence(res: Any) -> tuple[str, dict[str, dict]]:
    text, index = format_evidence(res)
    arm = CURRENT["arm"]
    if arm in ("F_flat", "P_child"):
        CURRENT["evidence_chars"] = len(text)
        return text, index
    by_parent: dict[str, list[Any]] = {}
    for i, c in enumerate(res.chunks):
        pid = (index.get(f"E{i + 1}") or {}).get("parent_id") or ""
        by_parent.setdefault(pid, []).append(c)
    out, injected = [], 0
    for i, line in enumerate(lines := text.split("\n")):
        eid = f"E{i + 1}"
        pid = (index.get(eid) or {}).get("parent_id") or ""
        body = CURRENT["parents"].get(pid) or ""
        if arm == "P_parent":
            out.append(line + (f"\n    【所属章节全文】{body}" if body else ""))
            injected += 1 if body else 0
        elif pid not in CURRENT["seen"]:  # trim：同章只贴一次，且只贴没被覆盖的部分
            CURRENT["seen"].add(pid)
            rest = trimmed_parent_body(body, [c.text for c in by_parent.get(pid, [])]) if body else ""
            if rest:
                out.append(line + f"\n    【本章其余内容（去重后 {len(rest)} 字）】{rest}")
                injected += 1
            else:
                out.append(line + "\n    【本章其余内容】（子块已覆盖整章，无新增）")
    CURRENT["injected"] = injected
    CURRENT["evidence_chars"] = len("\n".join(out))
    return "\n".join(out), index


class ReplayRetriever:
    def __init__(self, by_query: dict[str, Any]) -> None:
        self.by_query = by_query

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        return self.by_query[query]


class CountingLlm:
    def __init__(self, inner: OllamaClient) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, **kw):
        res = self.inner.chat(messages, **kw)
        m = dict(getattr(res, "metrics", {}) or {})
        self.calls.append({"prompt_tokens": int(m.get("prompt_tokens") or 0),
                           "completion_tokens": int(m.get("completion_tokens") or 0),
                           "total_ms": float(m.get("total_ms") or 0.0),
                           "done_reason": m.get("done_reason", "")})
        return res

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self) -> None:
        self.calls = []

    def summary(self) -> dict[str, Any]:
        if not self.calls:
            return {"calls": 0}
        p = [c["prompt_tokens"] for c in self.calls]
        return {"calls": len(self.calls), "prompt_tokens_p50": percentile(p, 50),
                "prompt_tokens_max": max(p), "prompt_tokens_mean": round(sum(p) / len(p), 1), "truncated": sum(1 for c in self.calls if c["done_reason"] == "length"),
                "ms_p50": round(percentile([c["total_ms"] for c in self.calls], 50), 1)}


NUM = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|％|分钟|小时|天|个月|周|年|人|条|项|个|家|台|次|分|秒|毫秒|ms|万元|亿元|亿|万|元)?")


def key_tokens(spans: list[str], reference: str) -> list[str]:
    """答案串落地率的判据：gold span 里的数字/单位 token；没有数字就取首个 span 的前 12 字。

    这是**机械代理，不是正确率**——它只问「结论里有没有出现答案原文里那个串」。
    """
    toks = [t.strip() for s in spans for t in NUM.findall(s) if re.search(r"\d", t)]
    uniq = sorted(set(t for t in toks if len(t) >= 2))
    if len(uniq) >= 2:
        return uniq[:6]
    return [s[:12] for s in spans[:1]]


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    ap.add_argument("--per", type=int, default=2, help="几小节并一章（默认 2）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="检索层重复几轮以量轮间噪声地板（跨臂比 Δ 之前必须先过这个）")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--keep-collections", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--render-only", default="", help="已有产物 json：只重出报告（不打 GPU、不查库）")
    ap.add_argument("--floor-from", default="",
                    help="另一轮同配置产物 json：把两轮之间各档的差当作**生成层轮间地板**，"
                         "跨档 Δ 与之相比才定结论（temp=0 也不逐位复现，实测见报告）")
    args = ap.parse_args()

    if args.render_only:
        src = Path(args.render_only).resolve()
        rep = json.loads(src.read_text(encoding="utf-8"))
        if args.floor_from:
            other = json.loads(Path(args.floor_from).read_text(encoding="utf-8"))
            rep["generation_floor"] = generation_floor(rep, other)
            rep["generation_floor_source"] = {"this": src.name, "other": Path(args.floor_from).name,
                                              "other_generated_at": other.get("generated_at")}
        derive_retrieval_verdicts(rep)
        tag = src.stem.replace("chunking_ab", "").lstrip(".").replace(".pre", "")
        out_md = REPORT if not tag else REPORT.with_name(REPORT.stem + f"_{tag}.md")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render(rep), encoding="utf-8")
        print(f"[仅重出] 报告 {out_md}")
        return 0
    t0 = time.perf_counter()
    s = get_settings()

    prod_counts_before = _prod_counts(s)

    merged = [reshape_card(c, args.per) for c in FACT_CARDS]
    card_meta = {c["doc_id"]: c for c in FACT_CARDS}
    P, F = build_two_arm_chunks(merged)
    p_rows, f_rows = P["children"], F["children"]
    print(f"[重塑] 文档 {len(merged)}｜P 子块 {len(p_rows)}｜F 窗口块 {len(f_rows)}｜父块 {len(P['parents'])}")

    # ---- 题目与 span gold（纯 PG 读，不打 GPU）
    ev = build_eval_set(AuthoritativeStore(), write_files=False)
    with psycopg.connect(s.postgres_dsn) as conn:
        prod_content = {r[0]: r[1] for r in conn.execute(
            "SELECT id, content FROM chunk WHERE is_parent = false")}
    questions, spans, dropped = [], {}, []
    for q in ev["questions"]:
        if q.get("expected_behavior") != "answer" or not q.get("gold_strict_ids"):
            continue
        ss, why = spans_for(q, prod_content)
        if not ss:
            dropped.append({"qid": q["qid"], "why": why})
            continue
        spans[q["qid"]] = ss
        questions.append(q)
    doc_norm = {c["doc_id"]: norm(" ".join(p for sec in c["sections"] for p in sec["paragraphs"]))
                for c in merged}
    all_norm = list(doc_norm.values())
    locatable = [all(any(s in d for d in all_norm) for s in spans[q["qid"]]) for q in questions]
    for q, ok in zip(questions, locatable):
        if not ok:
            dropped.append({"qid": q["qid"], "why": "span 在重塑后的文档流里定位不到"})
    questions = [q for q, ok in zip(questions, locatable) if ok]
    if args.limit:
        questions = questions[:args.limit]
    spans = {q["qid"]: spans[q["qid"]] for q in questions}
    print(f"[题面] 可判 {len(questions)}｜剔除 {len(dropped)}（例：{dropped[:2]}）")

    gold_p = arm_gold(p_rows, spans)
    gold_f = arm_gold(f_rows, spans)

    # ---- 嵌入 + 建两个 scratch 集合
    sched = ResidencyScheduler(policy="co_resident")
    emb = sched.acquire("embed")
    texts = [r["content"] for r in p_rows] + [r["content"] for r in f_rows]
    t_e = time.perf_counter()
    vecs = emb.embed_passages(texts, batch_size=16)
    vp: dict[str, list[float]] = {r["id"]: [float(x) for x in vecs[i]] for i, r in enumerate(p_rows)}
    vf: dict[str, list[float]] = {r["id"]: [float(x) for x in vecs[len(p_rows) + i]] for i, r in enumerate(f_rows)}
    print(f"[嵌入] {len(texts)} 块 {(vecs.shape[1] if hasattr(vecs, 'shape') else 1024)} 维，"
          f"{round(time.perf_counter() - t_e, 1)}s")
    sched.release_all()

    kb_p, kb_f = KbStore(COLL_P, "jieba"), KbStore(COLL_F, "jieba")
    for kb in (kb_p, kb_f):
        kb.drop()
    n_p = insert_arm(kb_p, p_rows, card_meta, vp)
    n_f = insert_arm(kb_f, f_rows, card_meta, vf)
    print(f"[索引] {COLL_P} {n_p}/{len(p_rows)}｜{COLL_F} {n_f}/{len(f_rows)}")

    # ---- 检索（两臂同一套参数，逐臂两轮预热再计量）
    # `--repeat` 把同一套检索原样再跑几轮，量**轮间噪声地板**：跨臂的 Δ 必须先过这个数，
    # 否则「Hit@5 差 1 题」这类差值分不清是分块造成的还是同机抖动造成的。
    degraded: dict[str, list[str]] = {}
    by_arm_runs: list[dict[str, dict[str, Any]]] = []
    for rep_i in range(max(1, args.repeat)):
        by_arm: dict[str, dict[str, Any]] = {}
        for name, kb, rows in (("P", kb_p, p_rows), ("F", kb_f, f_rows)):
            rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                                 mode="hybrid", use_rerank=True)
            for wq in WARM_QUERIES:
                rt.retrieve(wq, AccessContext(role="knowledge_owner"))
            got = {}
            for i, q in enumerate(questions, start=1):
                res = rt.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
                got[q["qid"]] = res
                if rep_i == 0:
                    degraded[f"{name}/{q['qid']}"] = list(res.degraded or [])
                if i % 20 == 0:
                    print(f"[R:{name}#{rep_i + 1}] {i}/{len(questions)}")
            rt.sched.release_all()
            by_arm[name] = got
        by_arm_runs.append(by_arm)
    by_arm = by_arm_runs[0]
    n_deg = sum(1 for v in degraded.values() if v)
    print(f"[R] 两臂检索完成 ×{len(by_arm_runs)}｜降级行 {n_deg}")

    runs_metrics = [{nm: retrieval_metrics(r[nm], spans, gold_p if nm == "P" else gold_f)
                     for nm in ("P", "F")} for r in by_arm_runs]
    ret_metrics = runs_metrics[0]
    floor = {k: {nm: round(max(m[nm][k] for m in runs_metrics) - min(m[nm][k] for m in runs_metrics), 4)
                 for nm in ("P", "F")} for k in RET_KEYS}
    floor["n_runs"] = len(by_arm_runs)
    floor["top5_member_flip_rows"] = {
        nm: sum(1 for qid in by_arm_runs[0][nm]
                if [c.chunk_id for c in by_arm_runs[0][nm][qid].chunks]
                != [c.chunk_id for c in by_arm_runs[-1][nm][qid].chunks])
        for nm in ("P", "F")} if len(by_arm_runs) > 1 else {"P": 0, "F": 0}
    # **实质门槛 = 2 题当量**（与上一轮消融同一条纪律）：1 题当量就是单题分辨率，
    # 「可复现」不等于「成差别」——零抖动下差 1 题只说明这 1 题稳定地不一样，不说明分块有效。
    _derived = {"retrieval": ret_metrics, "retrieval_floor": floor, "shape": {"questions": len(questions)}}
    derive_retrieval_verdicts(_derived)
    ret_delta = _derived["retrieval_delta"]
    ret_units = _derived["retrieval_question_units"]
    ret_verdict = _derived["retrieval_verdict"]

    # ---- 生成四档
    gen: dict[str, list[dict[str, Any]]] = {}
    costs: dict[str, Any] = {}
    if not args.skip_generation:
        ollama = OllamaClient()
        print(f"[A] warm 7B：{ollama.warmup()['load_ms']}ms")
        counting = CountingLlm(ollama)
        for arm in GEN_ARMS:
            CURRENT["arm"] = arm
            CURRENT["seen"] = set()
            CURRENT["parents"] = P["parents"]
            replay = ReplayRetriever({q["question"]: by_arm[ARM_INDEX[arm]][q["qid"]] for q in questions})
            eng = AnswerEngine(retriever=replay, llm=counting)
            eng.calibration_temperature = load_calibration_temperature(s)
            graph_mod.format_evidence = arm_evidence
            out = []
            for i, q in enumerate(questions, start=1):
                CURRENT["seen"] = set()
                CURRENT["injected"] = 0
                CURRENT["evidence_chars"] = 0
                res = eng.answer(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
                gold = gold_p if ARM_INDEX[arm] == "P" else gold_f
                cited = {c.get("chunk_id") for c in res.citations}
                keys = key_tokens(spans[q["qid"]], q.get("reference_answer", ""))
                cn = norm(res.conclusion or "")
                out.append({"qid": q["qid"], "status": res.status, "conclusion": res.conclusion or "",
                            "injected": CURRENT["injected"], "evidence_chars": CURRENT["evidence_chars"],
                            "n_citations": len(res.citations),
                            "cited_in_gold": bool(cited & gold.get(q["qid"], set())),
                            "key_hit": any(norm(k) in cn for k in keys) if keys else None,
                            "keys": keys})
                if i % 20 == 0:
                    print(f"[G:{arm}] {i}/{len(questions)}")
            gen[arm] = out
            costs[arm] = counting.summary()
            print(f"[G:{arm}] 完成｜串落地 {sum(1 for r in out if r['key_hit'])}/{len(out)}｜"
                  f"可完成率 {round(sum(1 for r in out if r['conclusion'].strip()) / len(out), 4)}")
        graph_mod.format_evidence = format_evidence

    gen_metrics = {}
    for arm, rows in gen.items():
        with_key = [r for r in rows if r["key_hit"] is not None]
        gen_metrics[arm] = {
            "n": len(rows),
            "key_grounding": round(sum(1 for r in with_key if r["key_hit"]) / len(with_key), 4) if with_key else None,
            "completion_rate": round(sum(1 for r in rows if r["conclusion"].strip()) / len(rows), 4),
            "cited_in_gold": round(sum(1 for r in rows if r["cited_in_gold"]) / len(rows), 4),
            "status_dist": {k: sum(1 for r in rows if r["status"] == k)
                            for k in ("answer", "tentative", "refuse", "escalate")},
        }

    # ---- 结构性诊断（回答「两臂到底像不像」）
    kids_per_parent: dict[str, int] = {}
    for r in p_rows:
        kids_per_parent[r["parent_id"] or ""] = kids_per_parent.get(r["parent_id"] or "", 0) + 1
    chars_by_parent: dict[str, int] = {}
    for r in p_rows:
        chars_by_parent[r["parent_id"] or ""] = chars_by_parent.get(r["parent_id"] or "", 0) + r["char_count"]
    ratios = [round(len(P["parents"][pid]) / max(chars_by_parent.get(pid, 1), 1), 3) for pid in P["parents"]]
    cross = [len(v) for v in F["chapters"].values()]
    struct = {
        "parent_chars_median": int(st.median([len(v) for v in P["parents"].values()])),
        "child_chars_median": int(st.median([r["char_count"] for r in p_rows])),
        "children_per_parent_dist": dict(sorted(st_kids(kids_per_parent).items())),
        "parents_with_ge2_children_rate": round(sum(1 for v in kids_per_parent.values() if v >= 2) / len(kids_per_parent), 4),
        "f_mean_chapters_touched": round(sum(cross) / len(cross), 3),
        "f_cross_chapter_rate": round(sum(1 for v in cross if v > 1) / len(cross), 4),
        "p_len_vs_f_len_p50": [percentile([r["char_count"] for r in p_rows], 50),
                               percentile([r["char_count"] for r in f_rows], 50)],
    }

    def p50(arm: str) -> Any:
        return (costs.get(arm) or {}).get("prompt_tokens_p50")

    inj_p50 = {a: percentile([r["injected"] for r in rows], 50) for a, rows in gen.items()} or "未跑生成"
    def mean(arm: str) -> Any:
        return (costs.get(arm) or {}).get("prompt_tokens_mean")

    def echars(arm: str, q: float = 50) -> Any:
        rows = gen.get(arm) or []
        return percentile([r["evidence_chars"] for r in rows], q) if rows else None

    # **主判据是证据文本字数，不是 token**：token 里混着系统提示与题面，字数才是这条改写本身的量。
    # 均值而不用 p50：小样本里各档 p50 会撞在同一个值上（smoke 实测四档 p50 全等），中位数在此是退化统计量。
    g4_ok = args.skip_generation or (
        (echars("P_parent") or 0) > (echars("P_child") or 0)
        and (echars("P_parent_trim") or 0) < (echars("P_parent") or 0)
        and (mean("P_parent") or 0) > (mean("P_child") or 0))
    trunc = {a: (costs.get(a) or {}).get("truncated") for a in GEN_ARMS}
    g5_ok = args.skip_generation or all((trunc.get(a) or 0) == 0
                                        for a in ("F_flat", "P_child", "P_parent_trim"))
    gates = {
        "G0 span 全部可在重塑语料定位（否则两臂量的不是同一批事实）": {
            "pass": not any(d["why"].startswith("span 在重塑") for d in dropped),
            "basis": f"因结构性原因剔除 {len(dropped)} 题，其中 span 定位失败 "
                     f"{sum(1 for d in dropped if d['why'].startswith('span 在重塑'))} 题"},
        "G1 检索零降级（单腿答会让两臂都不可信）": {
            "pass": n_deg == 0, "basis": f"降级行 {n_deg}/{2 * len(questions)}"
                    + ("" if n_deg else f"（例 {[k for k, v in degraded.items() if v][:3]}）")},
        "G2 两臂块长可比（否则比的是谁塞的字多）": {
            "pass": abs(struct["p_len_vs_f_len_p50"][0] - struct["p_len_vs_f_len_p50"][1]) / max(struct["p_len_vs_f_len_p50"]) < 0.30,
            "basis": f"子块 p50 {struct['p_len_vs_f_len_p50'][0]} vs 窗口块 p50 {struct['p_len_vs_f_len_p50'][1]}"},
        "G3 对照真的成立（F 臂必须跨章、P 臂父块必须多子）": {
            "pass": struct["f_cross_chapter_rate"] > 0 and struct["parents_with_ge2_children_rate"] > 0.5,
            "basis": f"F 跨章块占比 {struct['f_cross_chapter_rate']}（平均触章 {struct['f_mean_chapters_touched']}）；"
                     f"含 ≥2 子块的父块占比 {struct['parents_with_ge2_children_rate']}；"
                     f"「父块字数 / 其全部子块字数合计」中位 {round(st.median(ratios), 3)}"
                     "（≈1 说明子块已覆盖整章，父块是副本而非增量）"},
        "G4 生成层三档提示成本必须分层（证明注入与去重都真的落地了）": {
            "pass": g4_ok,
            "basis": f"证据字数 p50：F_flat {echars('F_flat')}｜P_child {echars('P_child')}｜"
                     f"P_parent {echars('P_parent')}｜P_parent_trim {echars('P_parent_trim')}"
                     f"（最大值 {echars('P_parent', 100)} vs {echars('P_parent_trim', 100)}）；"
                     f"prompt 均值 {mean('P_child')} → {mean('P_parent')} → {mean('P_parent_trim')}；"
                     f"p50 仅供参考：F_flat {p50('F_flat')}｜P_child {p50('P_child')}｜"
                     f"P_parent {p50('P_parent')}｜P_parent_trim {p50('P_parent_trim')}；"
                     f"注入生效要求 P_parent 均值 > P_child，去重生效要求 P_parent_trim 均值 < P_parent"
                     f"（两者同高说明 trim 其实是同一个档）；"
                     f"逐题注入次数 p50 {inj_p50}；p50 列一并给出供参考，不参与判定"},
        "G5 主对照三档不撞上下文窗口（撞了量到的就是截断不是分块）": {
            "pass": g5_ok,
            "basis": f"truncated 计数 {trunc}"
                     "｜P_parent 撞窗**不判废**——「整章逐条贴」在长章节上先撞 num_ctx=4096 正是这一档的实测缺陷，"
                     "但如果 F_flat / P_child / P_parent_trim 也撞，主对照就整体失真"},
    }
    all_pass = all(g["pass"] for g in gates.values())
    print("\n[护栏] " + " ｜ ".join(f"{k.split(' ')[0]}:{'过' if v['pass'] else '不过'}" for k, v in gates.items()))

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag, "void": not all_pass,
        "why_this_exists": (__doc__ or "").split("**三处设计是刻意的**")[0][-320:],
        "shape": {"sections_per_chapter": args.per, "child_max": CHILD_MAX, "overlap": OVERLAP,
                  "n_docs": len(merged), "p_children": len(p_rows), "f_children": len(f_rows),
                  "p_parents": len(P["parents"]), "questions": len(questions), "dropped": dropped},
        "structure": struct, "retrieval": ret_metrics,
        "retrieval_floor": floor, "retrieval_delta": ret_delta, "retrieval_verdict": ret_verdict,
        "retrieval_question_units": ret_units,
        "generation": gen_metrics, "cost_by_arm": costs, "gates": gates,
        "rows": {a: gen.get(a, []) for a in GEN_ARMS},
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }

    # ---- 复原与生产零改动自检
    if not args.keep_collections:
        kb_p.drop()
        kb_f.drop()
    prod_counts_after = _prod_counts(s)
    rep["production_untouched"] = {
        "before": prod_counts_before, "after": prod_counts_after,
        "same": prod_counts_before == prod_counts_after,
    }
    print(f"[复原] 生产计数 {prod_counts_before} → {prod_counts_after}｜scratch 集合已删={not args.keep_collections}")

    out_json = ART / (f"chunking_ab{('.' + args.tag) if args.tag else ''}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():  # 覆盖前留底：这份产物每次要 20 分钟 GPU，判废或重跑都不该把它顶掉
        import shutil
        keep = out_json.with_name(out_json.stem + f".pre-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(out_json, keep)
        print(f"[留底] {keep.name}")
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out_md = REPORT if not args.tag else REPORT.with_name(REPORT.stem + f"_{args.tag}.md")
    out_md.write_text(render(rep), encoding="utf-8")
    print(f"\n证据： {out_json}\n报告： {out_md}\n用时 {rep['elapsed_sec']}s")
    return 0 if all_pass else 3


def st_kids(d: dict[str, int]) -> dict[int, int]:
    from collections import Counter
    return dict(Counter(d.values()))


def _prod_counts(s) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        kb = KbStore(PROD_COLL, "jieba")
        out["milvus_chunks"] = kb.count()
    except Exception as e:  # noqa: BLE001
        out["milvus_chunks"] = f"ERR {type(e).__name__}"
    try:
        with psycopg.connect(s.postgres_dsn) as conn:
            out["pg_documents"] = conn.execute("SELECT count(*) FROM document").fetchone()[0]
            out["pg_child_chunks"] = conn.execute(
                "SELECT count(*) FROM chunk WHERE is_parent = false").fetchone()[0]
    except Exception as e:  # noqa: BLE001
        out["pg"] = f"ERR {type(e).__name__}"
    return out


def derive_retrieval_verdicts(rep: dict[str, Any]) -> None:
    """跨臂 Δ、题当量与判读**都是推导值**：只要 `retrieval` 与 `retrieval_floor` 在产物里就能重算，
    所以补判据不必重打 GPU。（读数本身一律沿用产物原值，这里一个都不改。）"""
    r = rep["retrieval"]
    fl = rep.get("retrieval_floor") or {}
    n = rep["shape"]["questions"] or 1
    ret_delta = {k: round(r["P"][k] - r["F"][k], 4) for k in RET_KEYS}
    units = {k: round(abs(ret_delta[k]) * n, 2) for k in RET_KEYS}
    verdict = {}
    for k in RET_KEYS:
        floor_k = max(fl[k].values()) if fl.get(k) else None
        beyond = (floor_k is not None and abs(ret_delta[k]) > floor_k)
        material = units[k] >= MATERIAL_QUESTION_UNITS
        verdict[k] = ("**实质差**" if (beyond and material) else
                      (f"{'可复现但' if beyond else '落在轮间地板内且'}只有 {units[k]} 题当量"
                       f"（<{MATERIAL_QUESTION_UNITS} 题门槛），**测不出差别**")
                      if fl else "本轮未配轮间地板，**不下结论**")
    rep["retrieval_delta"], rep["retrieval_question_units"] = ret_delta, units
    rep["retrieval_verdict"] = verdict


def generation_floor(rep: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """两轮同配置独立全量之间，各档各指标的最大漂移 = 生成层的轮间地板。

    这一条不是形式主义：本次两轮全量（间隔约 20 分钟、同一模型同一温度）四档各有 1–3 题漂移，
    连「P_parent 可完成率下降」这种看起来很大的差都在漂移范围内。**没有这个地板，
    生成层四档的表就会被读成四条结论。**
    """
    keys = ("key_grounding", "completion_rate", "cited_in_gold")
    out = {k: {} for k in keys}
    for arm in GEN_ARMS:
        a, b = rep["generation"].get(arm), other["generation"].get(arm)
        if not a or not b:
            continue
        for k in keys:
            if a.get(k) is None or b.get(k) is None:
                continue
            out[k][arm] = round(abs(a[k] - b[k]), 4)
    return {"per_arm_max": {k: max(v.values()) if v else None for k, v in out.items()},
            "detail": out,
            "note": "取两轮各档漂移的最大值；跨档 Δ 只有超过它才写「有效差」"}


def render(rep: dict[str, Any]) -> str:
    r, g, c, s = rep["retrieval"], rep["generation"], rep["cost_by_arm"], rep["structure"]
    L = ["# M1 · Parent-Child vs 固定分块（长章节语料，两臂同题同参数）", ""]
    if rep.get("void"):
        L.append("> **本轮判废**：护栏未全过 ⇒ 跨臂差值不采信，只作存档。")
        L.append("")
    L.append(f"生成时点 {rep['generated_at']}｜文档 {rep['shape']['n_docs']} 篇重塑为长章节"
             f"（每章并 {rep['shape']['sections_per_chapter']} 小节）｜P 子块 {rep['shape']['p_children']}、"
             f"F 窗口块 {rep['shape']['f_children']}｜可判 {rep['shape']['questions']} 题"
             f"（剔除 {len(rep['shape']['dropped'])}）｜用时 {rep['elapsed_sec']}s")
    L.append("")
    L.append(f"> **为什么要换语料**：{rep['why_this_exists']}")
    L.append("")
    L.append("## 一、对照成不成立（先看这个，再看读数）")
    L.append("")
    L.append(f"- 父块中位 **{s['parent_chars_median']} 字** vs 子块中位 **{s['child_chars_median']} 字**；"
             f"含 ≥2 子块的父块占比 **{s['parents_with_ge2_children_rate']}**；"
             f"每父块子块数分布 {s['children_per_parent_dist']}")
    L.append(f"- F 臂**跨章块占比 {s['f_cross_chapter_rate']}**、平均每块触及 {s['f_mean_chapters_touched']} 章 "
             "⇒ 两臂确实不是同一个东西（这正是上一轮那份对照缺的条件）")
    L.append(f"- 块长 p50：P {s['p_len_vs_f_len_p50'][0]} / F {s['p_len_vs_f_len_p50'][1]} 字")
    L.append(f"- **生产零改动**：{rep['production_untouched']['same']}"
             f"（前后计数 {rep['production_untouched']['before']}）")
    L.append("")
    L.append("## 二、检索层（同一批答案句、同一个包含判定）")
    L.append("")
    L.append("| 分块方案 | Hit@5 | 答案句覆盖 Recall@5 | MRR | 该臂 gold 块总数 |")
    L.append("|---|---|---|---|---|")
    for name, lab in (("P", "Parent-Child（子块）"), ("F", "固定滑窗 280/40")):
        m = r[name]
        L.append(f"| {lab} | {m['hit@5']} | {m['sentence_recall@5']} | {m['mrr']} | {m['n_gold_chunks_total']} |")
    fl = rep.get("retrieval_floor", {})
    if fl:
        L.append("")
        L.append(f"**轮间噪声地板**（同一套检索原样跑 {fl.get('n_runs', 1)} 轮，取每把尺子的轮间极差）："
                 + "；".join(f"{k} P 极差 {fl[k]['P']} / F 极差 {fl[k]['F']}"
                             for k in ("hit@5", "sentence_recall@5", "mrr"))
                 + f"；Top-5 成员翻转行数 {fl.get('top5_member_flip_rows')}")
        L.append("")
        for k in ("hit@5", "sentence_recall@5", "mrr"):
            L.append(f"- **{k}**：P − F = **{rep['retrieval_delta'][k]:+.4f}** = "
                     f"**{rep.get('retrieval_question_units', {}).get(k)} 题当量** ⇒ {rep['retrieval_verdict'][k]}"
                     f"（轮间地板取两臂较大者 {max(fl[k].values())}；实质门槛 "
                     f"{MATERIAL_QUESTION_UNITS} 题当量）")
    L.append("")
    L.append("## 三、生成层四档")
    L.append("")
    if not g:
        L.append("（本轮 `--skip-generation`，未跑生成层）")
    else:
        L.append("| 档 | 答案串落地率 | 可完成率 | 引用落 gold | escalate | prompt p50/max tok | 截断 |")
        L.append("|---|---|---|---|---|---|---|")
        for arm in GEN_ARMS_LABEL:
            m, cc = g.get(arm), c.get(arm, {})
            if not m:
                continue
            L.append(f"| {arm}｜{GEN_ARMS_LABEL[arm]} | {m['key_grounding']} | {m['completion_rate']} | "
                     f"{m['cited_in_gold']} | {m['status_dist']['escalate']} | "
                     f"{cc.get('prompt_tokens_p50')} / {cc.get('prompt_tokens_max')} | {cc.get('truncated')} |")
        L.append("")
        L.append("> 「答案串落地率」是**机械代理**：结论里是否出现该题 gold 原文中的数字/单位串或首 12 字。"
                 "**不是正确率**，也不进门禁——同一函数对所有档，跨档可比。")
        fl = rep.get("generation_floor", {}).get("per_arm_max", {})
        nq = next((m["n"] for m in g.values() if m.get("n")), 0) or 1
        L.append("")
        L.append("### 生成层判读（跨档 Δ 先对两轮漂移地板，再看门槛）")
        L.append("")
        if not fl:
            L.append("> **本轮未配地板**（跑时没带 `--floor-from` 另一轮产物）⇒ 下面只列差值，**不下结论**："
                     "temp=0 下生成层跨轮**并不逐位复现**（实测两轮四档各漂 1–3 题），"
                     "没有这个数，四档的表会被读成四条结论。")
        for x, y in (("P_child", "F_flat"), ("P_parent", "P_child"), ("P_parent_trim", "P_child")):
            if not g.get(x) or not g.get(y):
                continue
            parts = []
            for k, lab in (("key_grounding", "串落地"), ("completion_rate", "可完成"), ("cited_in_gold", "引用落gold")):
                dv = (g[x].get(k) or 0) - (g[y].get(k) or 0)
                units = round(abs(dv) * nq, 2)
                if not fl:
                    parts.append(f"{lab} Δ {dv:+.4f}（{units} 题当量）")
                    continue
                f_units = round(fl[k] * nq, 2)
                parts.append(f"{lab} Δ {dv:+.4f}（{units} 题当量，地板 {f_units} 题当量）⇒ "
                             + ("**有效差**" if abs(dv) > fl[k] and units >= MATERIAL_QUESTION_UNITS
                                else "**测不出差别**"))
            L.append(f"- **{x} vs {y}**：" + "；".join(parts))
    L.append("")
    L.append("## 四、护栏")
    L.append("")
    for k, v in rep["gates"].items():
        L.append(f"- {k}：{'过' if v['pass'] else '**不过**'} ｜ {v['basis']}")
    L.append("")
    L.append("## 五、剔除了哪些题、为什么")
    L.append("")
    from collections import Counter
    for why, n in Counter(d["why"] for d in rep["shape"]["dropped"]).most_common():
        L.append(f"- {n} 题：{why}")
    L.append("")
    L.append("## 六、口径与边界（引用本实验必须连着这一段）")
    L.append("")
    L.append("- **观测域是这份重塑语料**：内容取自生产 24 篇，但章节边界是我按「相邻小节并章」造的，"
             "题面与答案句也派生自生产 gold ⇒ 结论说的是「在长章节形状下两臂差多少」，"
             "**不是**「在真交付文档上差多少」。")
    L.append("- **gold 用答案句包含判定，不用人标也不用关键词**：关键词法偏爱按章节切的那臂（等于给对照组送便宜）；"
             "整条子块正文当 span 也一样偏心——span 中位 251 字 vs 窗口 280 字，"
             "「必须整条包含」只奖励与 span 同形状的分块。这两处都是 review 时改掉的，不是默认设定。")
    L.append("- **检索层的 Δ 已与轮间地板同表给出**（`--repeat`）；跨臂判读只认超出地板的那几条。")
    gf = rep.get("generation_floor")
    if gf and gf.get("detail"):
        gfs = rep.get("generation_floor_source", {})
        L.append("- **生成层配了轮间漂移地板，但它只有 2 个样本点**：地板取自本实验两轮独立全量"
                 f"（`{gfs.get('this')}` vs `{gfs.get('other')}`，另一轮跑于 {gfs.get('other_generated_at')}），"
                 "同装置同题同参数。**两点只能给出最大漂移，给不出分布** ⇒ 这条地板偏乐观，"
                 "跨档 Δ 判「有效差」时应当要求**明显超出**它，而不是刚刚压线。")
    else:
        L.append("- **生成层本轮未配地板**（没带 `--floor-from`）⇒ 四档之间的差**一律不足以定结论**："
                 "temp=0 下生成层跨轮不逐位复现（本实验两轮实测各档漂 1–3 题），"
                 "没有这个数，四档的表会被读成四条结论。")
    L.append("- **「答案串落地率」是机械代理**（结论里有没有出现答案原文的数值串/首 12 字），不是正确率，也不进门禁。")
    a = rep.get("cost_by_arm", {})
    if a:
        L.append(f"- **P_parent 已贴到上下文窗口边缘**：prompt max "
                 f"{max((v or {}).get('prompt_tokens_max', 0) for v in a.values())} / num_ctx=4096"
                 f"，本轮截断计数 {({k: (v or {}).get('truncated') for k, v in a.items()})}"
                 "——再长一点就会从**头部静默截掉**，那时量的就不是分块而是截断了。")
    L.append("- F 臂跨章块的 `章节号/页/密级` 只能取**窗口起点**那一章的值——这是固定滑窗固有的歧义，"
             "不是我给它加的惩罚；本轮把它作为结构性代价单列（§一 跨章块占比）。")
    L.append("")
    return "\n".join(L)


GEN_ARMS_LABEL = {
    "F_flat": "固定滑窗块直接作答",
    "P_child": "子块作答（现状那套，语料变长）",
    "P_parent": "子块 + 整章原文逐条贴（复刻 A1 笨接法）",
    "P_parent_trim": "子块 + 整章去重后的其余段落（同章只贴一次）",
}


if __name__ == "__main__":
    sys.exit(main())
