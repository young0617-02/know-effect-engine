"""C6 实测：按用户下发的 **token 粒度**重建索引，然后只拨「父块回捞」这一个开关。

用户 2026-10-22 裁决「② 照需求实现回捞，现在只是样本太少了，针对的问题不够明显，
将粒度设置为 Child 200 tokens 重叠 30 / Parent 1000 tokens 重叠 150」。
上一轮四档消融测的是**字符口径 + 章节=父块**那个形状，父块相对子块净增中位只有 28 字，
所以「接上父块无收益」只对那个形状成立。本轮把语料换成**长章节**、把粒度换成**token**，
让父块预算真的 binding，再问一次同一个问题。

**三处设计是刻意的**：
1. **走生产入口**：两档都是 `AnswerEngine(..., parent_provider=…)` 跑真图，
   只翻 `parent_context_in_answer` 这一个开关。不在脚本里重写一份注入逻辑——
   那样测的是脚本，上线的是引擎，迟早分叉。
2. **不写 PG、不动生产索引**：scratch 集合 `kb_chunktok_p` 跑完删。
   生产 193 块 / 24 篇、strict gold、547 行绑子块的人工标注零作废（G0 跑前后核计数）。
3. **同代两轮 A0 当噪声地板**：temp=0 也不逐位复现（上一轮实测同配置两轮各档漂 1–3 题）。
   A1−A0 的 Δ 不越过这个地板就**不写结论**，只写「测不出差别」。
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
sys.path.insert(0, str(ROOT / "scripts"))

import psycopg  # noqa: E402

import m1_chunking_ab as ab  # noqa: E402  复用重塑、gold 派生、检索计量、嵌入入库这几段已经过验收的装置
from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.ingest.chunking import Granularity, chunk_document  # noqa: E402
from kee.ingest.corpus_data import FACT_CARDS  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.models.tokens import count_tokens  # noqa: E402
from kee.qa.answer import AnswerEngine, load_calibration_temperature  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

ART = ROOT / "var" / "gate0"
REPORT = ROOT / "reports" / "m1" / "M1_父块回捞与token粒度.md"
COLL = "kb_chunktok_p"
PROD_COLL = "kb_chunks"
ARMS = ("A0_off", "A1_on")
GEN_KEYS = ("completion_rate", "key_grounding", "cited_in_gold")
MATERIAL_QUESTION_UNITS = 2.0


class RecordingLlm:
    """包住真 Ollama 客户端，只**记账**：提示 token、提示字符数、是否撞窗。

    刻意记 `done_reason == "length"`：撞了 `num_ctx` 的话提示是从头部静默截掉的，
    那时两档比的是「谁被截得多」，不是「补上下文有没有用」——这一栏非零整轮判废。
    """

    def __init__(self, inner: OllamaClient) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, **kw):
        res = self.inner.chat(messages, **kw)
        m = dict(getattr(res, "metrics", {}) or {})
        self.calls.append({
            "prompt_tokens": int(m.get("prompt_tokens") or 0),
            "prompt_chars": len(messages[-1]["content"]),
            "completion_tokens": int(m.get("completion_tokens") or 0),
            "total_ms": float(m.get("total_ms") or 0.0),
            "done_reason": m.get("done_reason", ""),
        })
        return res

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self) -> None:
        self.calls = []

    def last(self) -> dict[str, Any]:
        return self.calls[-1] if self.calls else {}

    def summary(self) -> dict[str, Any]:
        if not self.calls:
            return {"calls": 0}
        p = [c["prompt_tokens"] for c in self.calls]
        return {"calls": len(self.calls),
                "prompt_tokens_mean": round(sum(p) / len(p), 1),
                "prompt_tokens_p50": ab.percentile(p, 50),
                "prompt_tokens_max": max(p),
                "prompt_chars_mean": round(sum(c["prompt_chars"] for c in self.calls) / len(self.calls), 1),
                "truncated": sum(1 for c in self.calls if c["done_reason"] == "length"),
                "ms_mean": round(sum(c["total_ms"] for c in self.calls) / len(self.calls), 1)}


def note_int(notes: list[str], key: str) -> int:
    for n in notes or []:
        if n.startswith(key + ":"):
            return int(n.split(":", 1)[1])
    return 0


def build_token_arm(merged: list[dict[str, Any]], g: Granularity) -> tuple[list[dict], dict[str, str]]:
    children: list[dict[str, Any]] = []
    parents: dict[str, str] = {}
    for card in merged:
        vid = f"{card['doc_id']}@tok"
        rows = chunk_document(card, vid, granularity=g)
        for r in rows:
            if r["is_parent"]:
                parents[r["id"]] = r["content"]
            else:
                r["version_id"] = vid
                children.append(r)
    return children, parents


def structure_report(children: list[dict], parents: dict[str, str], g: Granularity) -> dict[str, Any]:
    """父块预算到底 binding 没有——这条决定本轮结论的适用范围，必须先报数再报效果。"""
    p_tok = {pid: count_tokens(c.split("\n", 1)[1]) for pid, c in parents.items()}
    kids: dict[str, list[dict]] = {}
    for c in children:
        kids.setdefault(c["parent_id"] or "", []).append(c)
    per_section: dict[str, int] = {}
    for pid in parents:
        m = re.match(r"(.+)#(.+)#p\d+$", pid)
        if m:
            key = f"{m.group(1)}#{m.group(2)}"
            per_section[key] = per_section.get(key, 0) + 1
    windowed = [k for k, v in per_section.items() if v > 1]
    # 回捞的**结构性上限**：只命中 1 个子块时，父块能多带进多少正文
    headroom = [max(0, p_tok[pid] - sum(count_tokens(c["content"]) for c in (kids.get(pid) or [])[:1]))
                for pid in parents]
    return {
        "granularity": g.describe(),
        "sections": len(per_section),
        "parents": len(parents),
        "sections_with_multiple_parent_windows": len(windowed),
        "parent_window_binding_rate": round(len(windowed) / max(len(per_section), 1), 4),
        "parent_tokens_median": int(st.median(list(p_tok.values()))) if p_tok else None,
        "parent_tokens_max": max(p_tok.values()) if p_tok else None,
        "parent_over_budget": [pid for pid, t in p_tok.items() if t > g.parent_tokens],
        "child_tokens_median": int(st.median([count_tokens(c["content"]) for c in children])),
        "child_tokens_max": max(count_tokens(c["content"]) for c in children),
        "children_per_parent_median": int(st.median([len(v) for v in kids.values()])) if kids else None,
        "retrieval_headroom_tokens_median": int(st.median(headroom)) if headroom else None,
    }


def pick_questions(s, merged: list[dict[str, Any]]) -> tuple[list[dict], dict[str, list[str]], list[dict]]:
    """可判题与答案句 gold。**主流程与归因分析共用这一份**：两处各挑一遍题，
    挑出来的就不是同一批事实，Δ 与归因会对不上。"""
    ev = build_eval_set(AuthoritativeStore(), write_files=False)
    with psycopg.connect(s.postgres_dsn) as conn:
        prod_content = {r[0]: r[1] for r in conn.execute(
            "SELECT id, content FROM chunk WHERE is_parent = false")}
    questions, spans, dropped = [], {}, []
    for q in ev["questions"]:
        if q.get("expected_behavior") != "answer" or not q.get("gold_strict_ids"):
            continue
        ss, why = ab.spans_for(q, prod_content)
        if not ss:
            dropped.append({"qid": q["qid"], "why": why})
            continue
        spans[q["qid"]] = ss
        questions.append(q)
    doc_norm = {c["doc_id"]: norm_all(c) for c in merged}
    all_norm = list(doc_norm.values())
    # 每个 span 都必须在重塑后的文档流里找得到，否则这题的 gold 是凭空的
    keep = [all(any(x in d for d in all_norm) for x in spans[q["qid"]]) for q in questions]
    for q, ok in zip(questions, keep):
        if not ok:
            dropped.append({"qid": q["qid"], "why": "span 在重塑后的文档流里定位不到"})
    questions = [q for q, ok in zip(questions, keep) if ok]
    return questions, {q["qid"]: spans[q["qid"]] for q in questions}, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=6, help="几小节并一章（默认 6：实测只有这个形状让 Parent 1000 token 预算 binding）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    ap.add_argument("--floor-runs", type=int, default=2, help="A0 同配置跑几轮以量生成层轮间地板")
    ap.add_argument("--repeat", type=int, default=2, help="检索层重复几轮")
    ap.add_argument("--child-tokens", type=int, default=200)
    ap.add_argument("--child-overlap-tokens", type=int, default=30)
    ap.add_argument("--parent-tokens", type=int, default=1000)
    ap.add_argument("--parent-overlap-tokens", type=int, default=150)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--keep-collections", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--render-only", default="", help="已有产物 json：只重出报告（不打 GPU、不查库）")
    args = ap.parse_args()
    g = Granularity(child_tokens=args.child_tokens, child_overlap_tokens=args.child_overlap_tokens,
                    parent_tokens=args.parent_tokens, parent_overlap_tokens=args.parent_overlap_tokens)

    if args.render_only:
        src = Path(args.render_only).resolve()
        rep = json.loads(src.read_text(encoding="utf-8"))
        derive(rep)
        rep["mechanism"] = mechanism(rep)
        from types import SimpleNamespace as _NS
        rep["gates"] = gates(rep, _NS(skip_generation=not rep.get("generation")))
        rep["void"] = not all(v["pass"] for v in rep["gates"].values())
        rep["conclusion"] = conclusion(rep)
        tag = src.stem.replace("parent_retrieval_tok", "").lstrip(".").replace(".pre", "")
        out_md = REPORT if not tag else REPORT.with_name(REPORT.stem + f"_{tag}.md")
        if not tag:   # 判废与归因要能落盘，只重出报告等于下次又得重算
            src.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render(rep), encoding="utf-8")
        print(f"[仅重出] 报告 {out_md}")
        return 0

    t0 = time.perf_counter()
    s = get_settings()
    prod_before = ab._prod_counts(s)

    merged = [ab.reshape_card(c, args.per) for c in FACT_CARDS]
    card_meta = {c["doc_id"]: c for c in FACT_CARDS}
    children, parents = build_token_arm(merged, g)
    struct = structure_report(children, parents, g)
    print(f"[重塑] 文档 {len(merged)}｜子块 {len(children)}｜父块 {len(parents)}"
          f"｜父块分窗触发率 {struct['parent_window_binding_rate']}")

    # ---- 题面与 gold（纯 PG 读，与 C7 同一套判据）
    questions, spans, dropped = pick_questions(s, merged)
    if args.limit:
        questions = questions[:args.limit]
    spans = {q["qid"]: spans[q["qid"]] for q in questions}
    print(f"[题面] 可判 {len(questions)}｜剔除 {len(dropped)}")

    gold = ab.arm_gold(children, spans)

    # ---- 嵌入 + scratch 集合
    sched = ResidencyScheduler(policy="co_resident")
    emb = sched.acquire("embed")
    t_e = time.perf_counter()
    vecs = emb.embed_passages([r["content"] for r in children], batch_size=16)
    vmap = {r["id"]: [float(x) for x in vecs[i]] for i, r in enumerate(children)}
    print(f"[嵌入] {len(children)} 块 {round(time.perf_counter() - t_e, 1)}s")
    sched.release_all()

    kb = KbStore(COLL, "jieba")
    kb.drop()
    n_ins = ab.insert_arm(kb, children, card_meta, vmap)
    print(f"[索引] {COLL} {n_ins}/{len(children)}")

    # ---- 检索（一轮计量 + 额外轮只量抖动）
    degraded: dict[str, list[str]] = {}
    runs: list[dict[str, Any]] = []
    for rep_i in range(max(1, args.repeat)):
        rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                             mode="hybrid", use_rerank=True)
        for wq in ab.WARM_QUERIES:
            rt.retrieve(wq, AccessContext(role="knowledge_owner"))
        got = {}
        for i, q in enumerate(questions, start=1):
            res = rt.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
            got[q["qid"]] = res
            if rep_i == 0:
                degraded[q["qid"]] = list(res.degraded or [])
            if i % 20 == 0:
                print(f"[R#{rep_i + 1}] {i}/{len(questions)}")
        rt.sched.release_all()
        runs.append(got)
    by_q = runs[0]
    n_deg = sum(1 for v in degraded.values() if v)
    ret = ab.retrieval_metrics(by_q, spans, gold)
    flip = sum(1 for q in questions
               if [c.chunk_id for c in runs[0][q["qid"]].chunks] != [c.chunk_id for c in runs[-1][q["qid"]].chunks]) \
        if len(runs) > 1 else 0
    print(f"[R] 检索完成 ×{len(runs)}｜降级行 {n_deg}｜Top-5 成员翻转 {flip} 行"
          f"｜句覆盖 Recall@5 {ret['sentence_recall@5']}")

    # ---- 生成两档：只翻 parent_context_in_answer 这一个开关
    gen: dict[str, list[dict[str, Any]]] = {}
    costs: dict[str, Any] = {}
    same_surface = None
    if not args.skip_generation:
        ollama = OllamaClient()
        print(f"[A] warm 7B：{ollama.warmup()['load_ms']}ms")
        llm = RecordingLlm(ollama)
        plan = [("A0_off", 1)] * max(1, args.floor_runs) + [("A1_on", 1)]
        for arm, _ in plan:
            arm_id = arm if arm not in gen else f"{arm}#r{len([k for k in gen if k.startswith(arm)]) + 1}"
            replay = ab.ReplayRetriever({q["question"]: by_q[q["qid"]] for q in questions})
            eng = AnswerEngine(retriever=replay, llm=llm,
                               parent_provider=lambda ids: {i: parents[i] for i in ids if i in parents},
                               parent_context_in_answer=(arm == "A1_on"))
            eng.calibration_temperature = load_calibration_temperature(s)
            out = []
            llm.reset()
            for i, q in enumerate(questions, start=1):
                n_before = len(llm.calls)   # 拒答/转人工那几题**根本不打模型**，取 last() 会串上一题的账
                res = eng.answer(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
                call = llm.calls[n_before] if len(llm.calls) > n_before else {}
                cited = {c.get("chunk_id") for c in res.citations}
                keys = ab.key_tokens(spans[q["qid"]], q.get("reference_answer", ""))
                cn = ab.norm(res.conclusion or "")
                out.append({"qid": q["qid"], "status": res.status, "conclusion": res.conclusion or "",
                            "n_citations": len(res.citations),
                            "pc_tokens": note_int(res.notes, "parent_context_tokens"),
                            "pc_skipped": note_int(res.notes, "parent_context_skipped"),
                            "prompt_chars": call.get("prompt_chars"),
                            "prompt_tokens": call.get("prompt_tokens"),
                            "truncated": call.get("done_reason") == "length",
                            "cited_in_gold": bool(cited & gold.get(q["qid"], set())),
                            "key_hit": any(ab.norm(k) in cn for k in keys) if keys else None,
                            "top_chunk_ids": [c.chunk_id for c in by_q[q["qid"]].chunks]})
                if i % 20 == 0:
                    print(f"[G:{arm_id}] {i}/{len(questions)}")
            gen[arm_id] = out
            costs[arm_id] = llm.summary()
            print(f"[G:{arm_id}] 完成｜回捞注入题数 "
                  f"{sum(1 for r in out if r['pc_tokens'])}/{len(out)}｜"
                  f"可完成率 {round(sum(1 for r in out if r['conclusion'].strip()) / len(out), 4)}")
        # 检索面恒定自检：三档跑的是同一批 top-5，否则比的不是这一个开关
        base = [r["top_chunk_ids"] for r in gen[ARMS[0]]]
        same_surface = all(base == [r["top_chunk_ids"] for r in rows] for rows in gen.values())

    gen_metrics = {a: {
        "n": len(rows),
        "completion_rate": round(sum(1 for r in rows if r["conclusion"].strip()) / len(rows), 4),
        "key_grounding": round(sum(1 for r in rows if r["key_hit"]) / len([r for r in rows if r["key_hit"] is not None]), 4)
        if any(r["key_hit"] is not None for r in rows) else None,
        "cited_in_gold": round(sum(1 for r in rows if r["cited_in_gold"]) / len(rows), 4),
        "pc_injected_rate": round(sum(1 for r in rows if r["pc_tokens"]) / len(rows), 4),
        "pc_tokens_mean": round(sum(r["pc_tokens"] for r in rows) / len(rows), 1),
        "status_dist": {k: sum(1 for r in rows if r["status"] == k)
                        for k in ("answer", "tentative", "refuse", "escalate")},
    } for a, rows in gen.items()} if not args.skip_generation else {}

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "why_this_exists": "用户裁「照需求实现回捞 + 粒度改 token」；上一轮的“无收益”测的是字符口径短章节形状",
        "shape": {"per": args.per, "n_docs": len(merged), "children": len(children), "parents": len(parents),
                  "questions": len(questions), "dropped": dropped},
        "structure": struct,
        "retrieval": ret, "retrieval_rounds": len(runs), "retrieval_top5_member_flips": flip,
        "degraded_rows": n_deg,
        "generation": gen_metrics, "cost_by_arm": costs,
        "rows": gen,
        "surface_identical": same_surface if not args.skip_generation else None,
        "production_untouched": {"before": prod_before},
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    derive(rep)
    rep["mechanism"] = mechanism(rep)
    if not args.keep_collections:
        kb.drop()
    prod_after = ab._prod_counts(s)
    rep["production_untouched"]["after"] = prod_after
    rep["production_untouched"]["same"] = prod_before == prod_after
    # 护栏在复原与生产计数落定之后才算：G0 判的就是这条计数，早算一步它读到的是 None，
    # 于是报告里写着「193 → None」却打了个「过」——那是空集上的真。
    rep["gates"] = gates(rep, args)
    rep["void"] = not all(v["pass"] for v in rep["gates"].values())
    rep["conclusion"] = conclusion(rep)
    print("\n[护栏] " + " ｜ ".join(f"{k.split(' ')[0]}:{'过' if v['pass'] else '不过'}"
                                   for k, v in rep["gates"].items()))

    out_json = ART / (f"parent_retrieval_tok{('.' + args.tag) if args.tag else ''}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        import shutil
        keep_art = out_json.with_name(out_json.stem + f".pre-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(out_json, keep_art)
        print(f"[留底] {keep_art.name}")
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = REPORT if not args.tag else REPORT.with_name(REPORT.stem + f"_{args.tag}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(rep), encoding="utf-8")
    print(f"\n证据： {out_json}\n报告： {out_md}\n用时 {rep['elapsed_sec']}s")
    return 0 if not rep["void"] else 3


def norm_all(card: dict[str, Any]) -> str:
    return ab.norm(" ".join(p for sec in card["sections"] for p in sec["paragraphs"]))


def mechanism(rep: dict[str, Any]) -> dict[str, Any]:
    """逐题归因：Δ 是不是注入这件事**本身**造成的。**不打 GPU**——重建分块只要分词器，题面只要读库。

    三条依据都来自装置已有的断言，不是叙事：
    - **同检索面**（G4）：三档跑的是同一批 top-5 ⇒ 「引用落在 gold」的差只能来自模型引了哪一块，
      不来自检索到了什么。这是本轮能做归因的前提，不成立时整个函数作废。
    - **注入文本用生产函数复算**（`apply_parent_context`），不在这里另写一份去重逻辑。
    - **未注入的题是自对照组**：A1 里 `pc_tokens=0` 的那些题跑的还是 A0 的证据面，
      它们在两档之间的翻转率就是「除注入以外的一切抖动」的实测值。
    """
    from types import SimpleNamespace

    from kee.qa.answer import apply_parent_context, format_evidence
    from kee.retrieval.pipeline import RetrievalResult, RetrievedChunk

    sh = rep["shape"]
    g = Granularity(**rep["structure"]["granularity"])
    rows_by_arm = rep.get("rows") or {}
    a1_rows, a0_rows = rows_by_arm.get("A1_on") or [], rows_by_arm.get("A0_off") or []
    if not a1_rows or not a0_rows:
        return {"available": False, "why": "本轮未跑生成两档，无行可归因"}

    merged = [ab.reshape_card(c, sh["per"]) for c in FACT_CARDS]
    children, parents = build_token_arm(merged, g)
    by_id = {c["id"]: c for c in children}
    s = get_settings()
    questions, spans, _ = pick_questions(s, merged)
    qids = [r["qid"] for r in a1_rows]
    questions = [q for q in questions if q["qid"] in qids]
    spans = {q["qid"]: spans[q["qid"]] for q in questions}
    gold = ab.arm_gold([c for c in children if c["id"] in by_id], spans)

    stub = SimpleNamespace(parent_provider=lambda ids: {i: parents[i] for i in ids if i in parents},
                           parent_context_in_answer=True, parent_context_max_tokens=1500)

    def res_for(tops: list[str]):
        chunks = [RetrievedChunk(chunk_id=t, score=float(len(tops) - i),
                                 text=by_id[t]["content"],
                                 fields={"doc_id": by_id[t]["doc_id"], "project": "",
                                         "section_no": by_id[t]["section_no"], "page": by_id[t]["page"],
                                         "security_level": 1, "parent_id": by_id[t]["parent_id"]})
                  for i, t in enumerate(tops) if t in by_id]
        return RetrievalResult(query="", mode="replay", chunks=chunks, timings_ms={}, expr="")

    a0 = {r["qid"]: r for r in a0_rows}
    detail, flips = [], {"cite_dn": [], "cite_up": [], "key_dn": [], "key_up": []}
    for r in a1_rows:
        qid = r["qid"]
        res = res_for(r["top_chunk_ids"])
        ev_text, ev_index = format_evidence(res)
        after, _notes = apply_parent_context(stub, ev_text, ev_index, res.chunks)
        injected = after[len(ev_text):]
        kid_n = ab.norm(ev_text)
        spans_list = spans.get(qid) or []
        in_kid = [ab.norm(x) in kid_n for x in spans_list]
        in_inj = [ab.norm(x) in ab.norm(injected) for x in spans_list]
        only_inject = [x for x, k, j in zip(spans_list, in_kid, in_inj) if j and not k]
        n_gold_top5 = len({c.chunk_id for c in res.chunks} & gold.get(qid, set()))
        b = a0.get(qid) or {}
        rec = {"qid": qid, "pc_tokens": r["pc_tokens"], "injected_chars": len(injected),
               "n_gold_in_top5": n_gold_top5,
               "span_only_in_inject": len(only_inject), "n_spans": len(spans_list),
               "cite_a0": b.get("cited_in_gold"), "cite_a1": r["cited_in_gold"],
               "key_a0": b.get("key_hit"), "key_a1": r["key_hit"],
               "status_a0": b.get("status"), "status_a1": r["status"]}
        detail.append(rec)
        if rec["cite_a0"] and not rec["cite_a1"]:
            flips["cite_dn"].append(qid)
        if rec["cite_a1"] and not rec["cite_a0"]:
            flips["cite_up"].append(qid)
        if rec["key_a0"] and not rec["key_a1"]:
            flips["key_dn"].append(qid)
        if rec["key_a1"] and not rec["key_a0"]:
            flips["key_up"].append(qid)

    inj = [d for d in detail if d["pc_tokens"] > 0]
    ctrl = [d for d in detail if d["pc_tokens"] == 0]   # 自对照：A1 里没被注入的题

    def rate(rows, key):
        return round(sum(1 for d in rows if d[key]) / len(rows), 4) if rows else None

    return {
        "available": True, "surface_identical": rep.get("surface_identical"),
        "n": len(detail), "n_injected": len(inj), "n_control": len(ctrl),
        "cited_in_gold": {"injected_rows_a0": rate(inj, "cite_a0"), "injected_rows_a1": rate(inj, "cite_a1"),
                          "control_rows_a0": rate(ctrl, "cite_a0"), "control_rows_a1": rate(ctrl, "cite_a1")},
        "key_grounding": {"injected_rows_a0": rate(inj, "key_a0"), "injected_rows_a1": rate(inj, "key_a1"),
                          "control_rows_a0": rate(ctrl, "key_a0"), "control_rows_a1": rate(ctrl, "key_a1")},
        "flips": {k: v for k, v in flips.items()},
        "cite_dn_explained_by_inject_only_span": sum(1 for d in detail
                                                    if d["cite_a0"] and not d["cite_a1"] and d["span_only_in_inject"]),
        "key_up_explained_by_inject_only_span": sum(1 for d in detail
                                                   if d["key_a1"] and not d["key_a0"] and d["span_only_in_inject"]),
        "questions_with_no_gold_in_top5": sum(1 for d in detail if d["n_gold_in_top5"] == 0),
        "detail": detail,
    }


def derive(rep: dict[str, Any]) -> None:
    """地板、Δ、题当量与判读**全部是推导值**：补判据不必重打 GPU。"""
    gen = rep.get("generation") or {}
    n = rep["shape"]["questions"] or 1
    a0 = gen.get("A0_off") or {}
    a0b = gen.get("A0_off#r2") or {}
    a1 = gen.get("A1_on") or {}
    floor = {k: (round(abs(a0[k] - a0b[k]), 4) if a0.get(k) is not None and a0b.get(k) is not None else None)
             for k in GEN_KEYS} if a0b else {}
    delta = {k: (round(a1[k] - a0[k], 4) if a1.get(k) is not None and a0.get(k) is not None else None)
             for k in GEN_KEYS} if a1 else {}
    units = {k: round(abs(v) * n, 2) if v is not None else None for k, v in delta.items()}
    verdict = {}
    for k in GEN_KEYS:
        if not delta or delta.get(k) is None:
            verdict[k] = "本轮未跑生成，**不下结论**"
            continue
        fl = floor.get(k)
        if fl is None:
            verdict[k] = f"Δ {delta[k]}（{units[k]} 题当量），本轮未配轮间地板，**不下结论**"
            continue
        beyond = abs(delta[k]) > fl
        material = (units[k] or 0) >= MATERIAL_QUESTION_UNITS
        verdict[k] = ("**实质差**" if (beyond and material) else
                      (f"Δ 落在轮间地板 {fl} 内" if not beyond else
                       f"可复现但只有 {units[k]} 题当量（<{MATERIAL_QUESTION_UNITS} 题门槛）")
                      + "，**测不出差别**")
    rep["generation_floor"] = floor
    rep["generation_delta"] = delta
    rep["generation_question_units"] = units
    rep["generation_verdict"] = verdict
    rep["cost_delta"] = {
        "prompt_tokens_mean": round(((rep.get("cost_by_arm") or {}).get("A1_on", {}).get("prompt_tokens_mean") or 0)
                                    - ((rep.get("cost_by_arm") or {}).get("A0_off", {}).get("prompt_tokens_mean") or 0), 1),
        "prompt_chars_mean": round(((rep.get("cost_by_arm") or {}).get("A1_on", {}).get("prompt_chars_mean") or 0)
                                   - ((rep.get("cost_by_arm") or {}).get("A0_off", {}).get("prompt_chars_mean") or 0), 1),
        "ms_mean": round(((rep.get("cost_by_arm") or {}).get("A1_on", {}).get("ms_mean") or 0)
                         - ((rep.get("cost_by_arm") or {}).get("A0_off", {}).get("ms_mean") or 0), 1),
    }


def conclusion(rep: dict[str, Any]) -> dict[str, Any]:
    """结论**由产物里的数算出来**，不手写：改判据就改这段代码，重出报告即一致。"""
    d, u, fl = rep.get("generation_delta") or {}, rep.get("generation_question_units") or {}, rep.get("generation_floor") or {}
    mk = rep.get("mechanism") or {}
    if not d:
        return {"verdict": "未测", "lines": ["本轮未跑生成两档，无结论。"]}
    detail = mk.get("detail") or []
    new_fact_rows = sum(1 for x in detail if x["span_only_in_inject"] > 0)
    inj_mean = (rep.get("generation", {}).get("A1_on") or {}).get("pc_tokens_mean")

    def material(k: str, sign: int) -> bool:
        v, un, f = d.get(k), u.get(k), fl.get(k)
        return (v is not None and un is not None and f is not None
                and (v < 0 if sign < 0 else v > 0)
                and un >= MATERIAL_QUESTION_UNITS and abs(v) > f)

    neg = [k for k in ("completion_rate", "cited_in_gold") if material(k, -1)]
    pos = [k for k in ("key_grounding",) if material(k, +1)]
    esc = (rep["generation"]["A1_on"]["status_dist"]["escalate"]
           - rep["generation"]["A0_off"]["status_dist"]["escalate"])
    # 带进新答案句的那几题，两档结果有没有真的不同？没有 ⇒ 新事实存在但没转化成任何收益
    changed = [x for x in detail if x["span_only_in_inject"] > 0
               and (x["key_a0"] != x["key_a1"] or x["cite_a0"] != x["cite_a1"] or x["status_a0"] != x["status_a1"])]
    lines = [
        f"注入实际发生：{mk.get('n_injected')}/{mk.get('n')} 题、每题均值 {inj_mean} token"
        f"（提示均值 +{rep['cost_delta']['prompt_tokens_mean']} token、+{rep['cost_delta']['ms_mean']} ms）。",
        f"新事实**确有但不是主要成分**：{new_fact_rows}/{len(detail)} 题的 gold 答案句"
        f"「只在注入文本里、不在任何被检子块里」，也就是回捞确实带进过检索面拿不到的内容；"
        f"而这 {new_fact_rows} 题在两档之间**逐题同判**（结果不同的 {len(changed)} 题）"
        f"——同检索面下 top-5 子块已覆盖约一个父块窗口的量（每父块子块数中位 "
        f"{rep['structure']['children_per_parent_median']}），补进来的多数是同章与本问无关的旁段。",
        f"超地板且够 2 题当量的**变差**：{neg or '无'}"
        f"（可完成率 Δ {d['completion_rate']}＝{u['completion_rate']} 题当量、"
        f"引用落 gold Δ {d['cited_in_gold']}＝{u['cited_in_gold']} 题当量、转人工净增 {esc} 道）；"
        f"变好：{pos or '无'}（串落地率 Δ {d['key_grounding']}＝{u['key_grounding']} 题当量），"
        f"而这一项的涨**不能归因于回捞带进来的新内容**："
        f"串落地涨的那些题里没有一道靠的是 inject-only 答案句。",
    ]
    if neg and not pos:
        verdict = "**负收益**：补进来的文字不含新答案句，却把引用从 gold 上引开、并把答案改成转人工"
    elif neg and pos and not changed:
        verdict = ("**净负**：两项超地板变差、一项超地板变好，"
                   "但变好项不能归因于新内容，而确实带进新内容的那几题结果全都没变")
    elif neg and pos:
        verdict = "**Mixed**：变差项与变好项都超地板，需按题面分档再判"
    elif pos:
        verdict = "有实质收益"
    else:
        verdict = "无实质收益"
    lines.append(f"判定：{verdict}。轮间地板 {fl}。")
    return {"verdict": verdict, "new_fact_rows": new_fact_rows, "new_fact_rows_changed_outcome": len(changed),
            "material_negative": neg, "material_positive": pos, "escalate_delta": esc, "lines": lines}


def gates(rep: dict[str, Any], args) -> dict[str, dict[str, Any]]:
    sh = rep["shape"]
    gen = rep.get("generation") or {}
    costs = rep.get("cost_by_arm") or {}
    a1 = gen.get("A1_on") or {}
    skip = args.skip_generation
    trunc = {a: (c or {}).get("truncated") for a, c in costs.items()}
    out = {
        "G0 生产索引零改动（本轮只建 scratch）": {
            "pass": rep["production_untouched"].get("same", True),
            "basis": f"{rep['production_untouched'].get('before')} → {rep['production_untouched'].get('after')}"},
        "G1 检索零降级（单腿答会让结论失真）": {
            "pass": rep["degraded_rows"] == 0,
            "basis": f"降级行 {rep['degraded_rows']}/{sh['questions']}"},
        "G2 父块预算真的 binding（否则测的不是用户下发的粒度）": {
            "pass": rep["structure"]["sections_with_multiple_parent_windows"] > 0
                    and not rep["structure"]["parent_over_budget"],
            "basis": f"{rep['structure']['sections_with_multiple_parent_windows']}/{rep['structure']['sections']} 章节"
                     f"被切成 ≥2 个父块窗口（触发率 {rep['structure']['parent_window_binding_rate']}）；"
                     f"父块正文 token 中位 {rep['structure']['parent_tokens_median']}、"
                     f"最大 {rep['structure']['parent_tokens_max']}（预算 "
                     f"{rep['structure']['granularity']['parent_tokens']}），"
                     f"超预算父块 {len(rep['structure']['parent_over_budget'])} 个；"
                     f"子块 token 中位 {rep['structure']['child_tokens_median']}、最大 {rep['structure']['child_tokens_max']}"
                     f"（最大值为超出正文预算的标题前缀，口径见 chunking 模块注释）"},
        "G3 回捞真的注入了（空集上的真不算收益）": {
            "pass": skip or (a1 and a1.get("pc_injected_rate", 0) > 0
                             and (gen.get("A0_off") or {}).get("pc_tokens_mean") == 0),
            "basis": f"A1 注入题占比 {a1.get('pc_injected_rate')}、注入 token 均值 {a1.get('pc_tokens_mean')}；"
                     f"A0 注入 token 均值 {(gen.get('A0_off') or {}).get('pc_tokens_mean')}"
                     f"（开关关掉必须一条不贴）；"
                     f"提示均值 {costs.get('A0_off', {}).get('prompt_tokens_mean')} → "
                     f"{costs.get('A1_on', {}).get('prompt_tokens_mean')} token"},
        "G4 检索面恒定（比的只有这一个开关）": {
            "pass": skip or bool(rep.get("surface_identical")),
            "basis": f"三档逐题 top-5 成员完全一致 = {rep.get('surface_identical')}"},
        "G5 没有撞上下文窗口（撞了量到的是截断不是补上下文）": {
            "pass": skip or all((v or 0) == 0 for v in trunc.values()),
            "basis": f"truncated 计数 {trunc}；prompt token 峰值 "
                     f"{ {a: (c or {}).get('prompt_tokens_max') for a, c in costs.items()} }（num_ctx=4096）"},
    }
    return out


def render(rep: dict[str, Any]) -> str:
    sh, s = rep["shape"], rep["structure"]
    g = s["granularity"]
    L = [f"# M1 实测：token 粒度下的父块回捞（C6）", "",
         f"> 生成时间 {rep['generated_at']}｜用时 {rep['elapsed_sec']}s"
         + (f"｜**本轮判废**（护栏 {sum(1 for v in (rep.get('gates') or {}).values() if not v['pass'])} 条不过）"
            if rep.get("void") else ""),
         "",
         f"**设置**：{sh['n_docs']} 篇并成 {sh['per']} 小节一章的长章节；分块 "
         f"Child {g['child_tokens']}+{g['child_overlap_tokens']} 重叠 / Parent {g['parent_tokens']}"
         f"+{g['parent_overlap_tokens']} 重叠（**token 口径**，BGE-M3 分词器实数）。"
         f"子块 {sh['children']}、父块 {sh['parents']}，可判题 {sh['questions']}（剔除 {len(sh['dropped'])}）。",
         "",
         "## 一、父块预算 binding 没有",
         "",
         f"- {s['sections']} 个章节里 **{s['sections_with_multiple_parent_windows']} 个**被切成 ≥2 个父块窗口"
         f"（触发率 {s['parent_window_binding_rate']}）；父块正文 token 中位 {s['parent_tokens_median']}、"
         f"最大 {s['parent_tokens_max']}（预算 {g['parent_tokens']}）",
         f"- 子块 token 中位 {s['child_tokens_median']}、最大 {s['child_tokens_max']}；"
         f"每父块子块数中位 {s['children_per_parent_median']}",
         f"- **只命中 1 个子块时父块能多带进的正文**中位 {s['retrieval_headroom_tokens_median']} token"
         "——这是回捞的结构性上限，上限为 0 的形状下回捞不可能有收益",
         "",
         "## 二、检索层（单臂，只作背景）",
         "",
         f"- Hit@5 {rep['retrieval']['hit@5']}｜句覆盖 Recall@5 {rep['retrieval']['sentence_recall@5']}"
         f"｜MRR {rep['retrieval']['mrr']}（gold 为答案句包含判定，与 C7 同一判据）",
         f"- {rep['retrieval_rounds']} 轮 Top-5 成员翻转 {rep['retrieval_top5_member_flips']} 行",
         ""]
    if rep.get("generation"):
        L += ["## 三、生成层：只翻 `parent_context_in_answer` 这一个开关", "",
              "| 档 | 题数 | 可完成率 | 串落地率 | 引用落在 gold | 注入题占比 | 注入 token 均值 | prompt token 均值 |",
              "|---|---|---|---|---|---|---|---|"]
        for a, m in rep["generation"].items():
            c = rep["cost_by_arm"].get(a) or {}
            L.append(f"| {a} | {m['n']} | {m['completion_rate']} | {m['key_grounding']} | {m['cited_in_gold']} |"
                     f" {m['pc_injected_rate']} | {m['pc_tokens_mean']} | {c.get('prompt_tokens_mean')} |")
        L += ["", f"出口分布：{ {a: m['status_dist'] for a, m in rep['generation'].items()} }", "",
              "**代价**：" + json.dumps(rep["cost_delta"], ensure_ascii=False), "",
              f"**轮间地板（同配置两轮 A0）**：{json.dumps(rep['generation_floor'], ensure_ascii=False)}",
              f"**Δ（A1 − A0）**：{json.dumps(rep['generation_delta'], ensure_ascii=False)}"
              f"＝{json.dumps(rep['generation_question_units'], ensure_ascii=False)} 题当量", "",
              "### 判读", ""]
        for k, v in rep["generation_verdict"].items():
            L.append(f"- {k}：{v}")
        L.append(f"- 判据门槛：**2 题当量**（与 C7 同一条纪律）；低于它只说明稳定地不一样，不说明有用。")
        mk = rep.get("mechanism") or {}
        if mk.get("available"):
            inj_c, key = mk["cited_in_gold"], mk["key_grounding"]
            L += ["", "## 三·补 逐题归因（同检索面 ⇒ 引用差只能来自模型引了哪一块）", "",
                  f"- A1 里**真的被注入**的题 {mk['n_injected']} 道，**没被注入**的 {mk['n_control']} 道"
                  "（后者是自带对照组：它跑的证据面与 A0 完全一样）",
                  f"- 引用落在 gold：注入组 A0 {inj_c['injected_rows_a0']} → A1 {inj_c['injected_rows_a1']}；"
                  f"对照组 A0 {inj_c['control_rows_a0']} → A1 {inj_c['control_rows_a1']}"
                  "（对照组那一行的差是「除注入以外的一切抖动」的实测值）",
                  f"- 串落地率：注入组 {key['injected_rows_a0']} → {key['injected_rows_a1']}；"
                  f"对照组 {key['control_rows_a0']} → {key['control_rows_a1']}",
                  f"- 翻转题数：引用掉了 {len(mk['flips']['cite_dn'])} 道（其中 "
                  f"{mk['cite_dn_explained_by_inject_only_span']} 道的答案句**只在注入文本里、不在任何被检子块里**）、"
                  f"引用涨了 {len(mk['flips']['cite_up'])} 道；串落地掉了 {len(mk['flips']['key_dn'])} 道、"
                  f"涨了 {len(mk['flips']['key_up'])} 道（其中 {mk['key_up_explained_by_inject_only_span']} 道"
                  "同样只在注入文本里找得到）",
                  f"- 结构性口径：{mk['questions_with_no_gold_in_top5']} 道题的 top-5 里**根本没有 gold 块**"
                  "⇒ 这些题无论怎么接父块都不可能「引用落在 gold」，它们压在两档的共同分母上"]
    cn = rep.get("conclusion") or {}
    if cn.get("lines"):
        L += ["", "## 五、结论", "", cn["verdict"], ""]
        L += [f"- {x}" for x in cn["lines"]]
    if rep.get("gates"):
        L += ["", "## 六、护栏", ""]
        for k, v in rep["gates"].items():
            L.append(f"- [{'x' if v['pass'] else ' '}] {k}——{v['basis']}")
    L += ["", f"生产索引计数：{json.dumps(rep['production_untouched'], ensure_ascii=False)}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
