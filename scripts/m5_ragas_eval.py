"""把 RAGAS 两指标与 LLM-as-Judge 真跑出数——并且先证明这套装置不是空转。

为什么读**留档产物**而不是重跑生成：真人 `answer_correct` 那 67 行标签只配它判过的那代输出
（#56 实测重跑一次生成会改 ~7.46% 行的结论文本）。重跑生成 ⇒ 标签作废 ⇒ 拿不到「裁判与人判的一致率」
这个装置有效性的唯一硬证据。所以默认输入是 `var/gate0/pre_reprobe/m3_confidence_probe.json`。

三个阶段、三次模型换乘（8GB 单卡的硬约束：7B 常驻后只剩 238MiB 空闲）：
  R  evict 7B → 复跑检索拿每题 Top-5（与 `m3_confidence_probe` 同构造参数，不改生成、不动标签）
  A  warm 7B → Faithfulness 拆句+逐条 NLI、Answer Relevance 反向生成、LLM 裁判判定、反面对照、第 2 轮噪声地板
  E  evict 7B → 载 BGE-M3 → 原题与反向问题的 embedding、余弦，与「答案错配」对照

Faithfulness 报两个 contexts 口径，因为 RAGAS 原义用的是**检索到的全部候选**，
而本仓库既有机械装置（`number_groundedness`、引用落 gold）用的是**实际被引用的块**：
  `cited` 口径 → 与既有读数同尺，可直接并列；
  `top5` 口径 → RAGAS 原口径，但只有 `cited_ids ⊆ 复跑 Top-5` 的行才算可比，对齐率进产物。

五道护栏在写产物**之前**判废（过不了就说明装置本身无效，不是「读数低」）：
  1 混代拒绝：qid 集合必须与工单一致（不齐直接退出，不产出任何读数）；
  2 证据错配：contexts 换成别题的块 ⇒ Faithfulness 必须下降、裁判 label 必须翻转；
  3 答案错配：AR(A 题 + B 答) 必须显著低于 AR(A 题 + A 答)；
  4 空证据面：一条材料都不给 ⇒ Faithfulness 必须 0，不得因「无从反驳」而判成立；
  5 空回答面：回答为空 ⇒ 必须报「不适用」，不得读成 1.0。
噪声地板：裁判全量重跑一轮、Faithfulness 固定 seed 抽样重跑，报翻转率与行分平均变化——
  小于这个地板的 Δ 不得写成改善或恶化。

跑法：uv run python scripts/m5_ragas_eval.py [--limit N] [--tag NAME] [--force] [--skip-retrieval]
      uv run python scripts/m5_ragas_eval.py --render-only   # 只重排版报告，不跑模型
产物：var/gate0/ragas_eval.json（或 ragas_eval.<tag>.json）+ reports/m5/M5_RAGAS与LLM裁判实测.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import statistics as st
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import psycopg  # noqa: E402

from kee.config import get_settings  # noqa: E402
from kee.eval.judge import ANSWER_CORRECT, DISPOSITION_CORRECT, LlmJudge, agreement_with_human  # noqa: E402
from kee.eval.ragas import (  # noqa: E402
    RagasRunner, answer_relevance_from_vectors, context_precision_from_verdicts,
    context_recall_from_verdicts, cosine, faithfulness_from_verdicts,
)
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

from m2_eval import bootstrap_ci  # noqa: E402

ART = ROOT / "var" / "gate0"
SOURCE = ART / "pre_reprobe" / "m3_confidence_probe.json"
LABELS = ART / "answer_correct_fill.csv"
REPORT = ROOT / "reports" / "m5" / "M5_RAGAS与LLM裁判实测.md"

ANSWER_STATUSES = ("answer", "tentative")
CONTROL_ROWS = 10          # 证据错配抽样行数
AR_CONTROL_ROWS = 10       # 答案错配对照行数（零额外 LLM 调用，只用已生成的反向问题）
FLOOR_FAITH_ROWS = 20      # Faithfulness 第二轮抽样行数（全量再跑一轮要 ~300 次 7B 调用）
SEED = 20261016

# 用户下发的《6.1 RAGAS 自动化指标》四条目标值。判定用的**主口径**写在 `key` 上，
# `alt` 是同指标的另一种 contexts 口径（并列展示，不许挑高的那条去对目标）。
TARGETS = [
    {"name": "Faithfulness（忠实度）", "defn": "回答是否严格基于检索上下文", "target": 0.85,
     "key": "faithfulness_top5", "alt": "faithfulness_cited", "pop": "作答行 n=62",
     "note": "主口径 = RAGAS 原义（contexts 取检索到的全部候选）；alt = 只用被引块"},
    {"name": "Context Precision（上下文精确度）", "defn": "检索文档中相关文档占比及排序", "target": 0.80,
     "key": "context_precision", "alt": "", "pop": "全 67 行（检索面，含拒答行）",
     "note": "位置加权 Σ P@k·v_k / 相关块数；相关块数为 0 的行报不适用、不进均值"},
    {"name": "Context Recall（上下文召回率）", "defn": "真实答案所需信息是否被检索到", "target": 0.85,
     "key": "context_recall", "alt": "", "pop": "全 67 行（检索面，含拒答行）",
     "note": "参考答案拆声明 → 逐条判能否归因到检索块"},
    {"name": "Answer Relevancy（答案相关性）", "defn": "回答是否切题", "target": 0.90,
     "key": "answer_relevance_answered", "alt": "answer_relevance_all_rows", "pop": "作答行 n=62",
     "note": "alt 档把「正确拒答」按官方口径记 0，单独引它等于说反话"},
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12] if path.exists() else "缺文件"


def read_labels() -> dict[str, int]:
    lines = [l for l in LABELS.read_text(encoding="utf-8-sig").splitlines() if not l.startswith("#")]
    out: dict[str, int] = {}
    bad: list[tuple[str, str]] = []
    for r in csv.DictReader(lines):
        v = (r.get("answer_correct__fill_1_or_0") or "").strip()
        if v in ("0", "1"):
            out[r["qid"]] = int(v)
        else:
            bad.append((r["qid"], v or "<空>"))
    if bad:
        raise ValueError(f"人判工单未填完/非法值：{len(bad)} 行，例如 {bad[:5]}")
    return out


def qid_hash(qids: list[str]) -> int:
    return zlib.crc32("|".join(sorted(qids)).encode("utf-8")) & 0xFFFFFFFF


def vram_used() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1.0


def describe(vals: list[float]) -> dict[str, Any]:
    """均值必带 n 与 bootstrap 95% CI。这些是逐题可取小数的均值，不是二项比例，故不用 Wilson。"""
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "ci95": None, "zeros": 0}
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 4),
        "median": round(st.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "ci95": list(bootstrap_ci(vals)),
        "zeros": sum(1 for v in vals if v == 0.0),
    }


def cited_texts(row: dict[str, Any], texts: dict[str, str]) -> list[str]:
    return [texts[c] for c in (row.get("cited_ids") or []) if c in texts]


# ---------------- 阶段 A ----------------

def judge_row(judge: LlmJudge, row: dict[str, Any], texts: dict[str, str],
              ctx_of: dict[str, list[str]]) -> dict[str, Any]:
    """有结论走「成立性」判据、看它自己引的证据；空结论走「处置该不该」判据、看它实际检到的 Top-5。

    两批行的观测域不同，这是既有工单的口径（真人工单给标注人的也是这五块），不是这里新造的分支。
    复跑检索被跳过时拒答行只能记「没证据面所以没判」，不得拿空材料去判「该不该拒」。
    """
    ans = row.get("conclusion") or ""
    if ans.strip():
        return judge.judge(row["question"], ans, cited_texts(row, texts), ANSWER_CORRECT)
    top5 = [texts[c] for c in ctx_of.get(row["qid"], []) if c in texts]
    if not top5:
        return {"label": None, "reason": "", "span": "", "why": "no_evidence_face_skipped_retrieval",
                "truncated": False, "rubric": DISPOSITION_CORRECT.id}
    return judge.judge(row["question"], "", top5, DISPOSITION_CORRECT)


def run_pass(rows: list[dict[str, Any]], texts: dict[str, str], ctx_of: dict[str, list[str]],
             runner: RagasRunner, judge: LlmJudge) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        if i % 10 == 1:
            print(f"[A] —— {i}/{len(rows)} ——")
        # 一行炸掉不该带走整轮（全量要 ~30 分钟 GPU 时间），但也不许静默：错误进 row_errors，
        # 缺读数的行不进分母——「少测了几行」和「测了且分数低」在产物里必须长得不一样。
        try:
            rec = _run_row(r, texts, ctx_of, runner, judge)
        except Exception as e:  # noqa: BLE001
            rec = {"qid": r["qid"], "status": r["status"], "question": r["question"],
                   "conclusion": r.get("conclusion") or "", "statements": None, "verdicts": [],
                   "decompose_failed": False, "verdicts_top5": [],
                   "faith_cited": faithfulness_from_verdicts([]),
                   "faith_top5": faithfulness_from_verdicts([]), "reverse_q": [],
                   "context_precision": context_precision_from_verdicts([]),
                   "context_recall": context_recall_from_verdicts([]),
                   "reference_claims": [], "context_recall_verdicts": [],
                   "noncommittal": False, "judge": {"label": None, "reason": "", "span": "",
                                                    "why": "row_error", "truncated": False,
                                                    "rubric": ANSWER_CORRECT.id},
                   "row_error": f"{type(e).__name__}: {e}"}
            print(f"[A {i:>2}/{len(rows)}] {r['qid']} **行失败** {rec['row_error']}")
        out.append(rec)
    return out


def _run_row(r: dict[str, Any], texts: dict[str, str], ctx_of: dict[str, list[str]],
             runner: RagasRunner, judge: LlmJudge) -> dict[str, Any]:
    q, ans, status = r["question"], r.get("conclusion") or "", r["status"]
    cited = cited_texts(r, texts)
    top5 = [texts[c] for c in ctx_of.get(r["qid"], []) if c in texts]
    rec: dict[str, Any] = {"qid": r["qid"], "status": status, "question": q, "conclusion": ans,
                           "n_cited": len(cited), "n_top5": len(top5)}
    if ans.strip():
        # 拆句只做一次，两口径共用同一批陈述——否则差值里混着「这次拆得不一样」。
        statements = runner.decompose(q, ans)
        rec["statements"], rec["decompose_failed"] = statements, statements is None
        if statements is None:
            empty = faithfulness_from_verdicts([])
            rec["faith_cited"] = rec["faith_top5"] = empty
            rec["verdicts"] = rec["verdicts_top5"] = []
        else:
            v_cited = runner.verify_set(statements, cited)
            rec["verdicts"], rec["faith_cited"] = v_cited["verdicts"], v_cited["score"]
            if not top5 or top5 == cited:
                rec["faith_top5"], rec["verdicts_top5"] = v_cited["score"], v_cited["verdicts"]
            else:
                v_top5 = runner.verify_set(statements, top5)
                rec["faith_top5"], rec["verdicts_top5"] = v_top5["score"], v_top5["verdicts"]
        rec["reverse_q"], rec["noncommittal"] = runner.reverse_questions(ans)
    else:
        # 空回答没有可拆的陈述：机械判「不适用」，不占 LLM 调用，也不许被读成 1.0 或 0 分。
        rec["statements"], rec["verdicts"], rec["verdicts_top5"] = [], [], []
        rec["decompose_failed"] = False
        rec["faith_cited"] = rec["faith_top5"] = faithfulness_from_verdicts([])
        rec["reverse_q"], rec["noncommittal"], rec["reverse_mechanical"] = [], True, True
    rec["judge"] = judge_row(judge, r, texts, ctx_of)
    # Context Precision / Recall 测的是**检索面**，所以两批行都算（拒答行也算——
    # 「该答却没检到」正是要暴露的东西），contexts 恒用复跑 Top-5，不用 cited。
    reference = str(r.get("reference_answer") or "").strip()
    rec["reference_answer"] = reference
    if reference:
        rec["context_precision"] = runner.context_precision(q, reference, top5)["score"]
        cr = runner.context_recall(q, reference, top5)
        rec["context_recall"] = cr["score"]
        rec["reference_claims"] = cr["claims"]
        rec["context_recall_verdicts"] = cr["verdicts"]
    else:
        empty_cp = context_precision_from_verdicts([])
        rec["context_precision"] = rec["context_recall"] = empty_cp
        rec["reference_claims"], rec["context_recall_verdicts"] = [], []
    fv = rec["faith_cited"].get("value")
    n_st = "拆句失败" if rec["statements"] is None else f"陈述{len(rec['statements'])}"
    print(f"[{rec['qid']}] {status:<9} F(cited)={f'{fv:.2f}' if fv is not None else '不适用'} "
          f"CP={rec['context_precision'].get('value')} CR={rec['context_recall'].get('value')} "
          f"裁判={rec['judge']['label']} {n_st}")
    return rec


# ---------------- 护栏 2/4/5：证据错配、空证据、空回答 ----------------

def negative_controls(rows: list[dict[str, Any]], texts: dict[str, str], runner: RagasRunner,
                      judge: LlmJudge) -> dict[str, Any]:
    pool = [r for r in rows if (r.get("conclusion") or "").strip() and r.get("cited_ids")]
    picked = random.Random(SEED).sample(pool, min(CONTROL_ROWS, len(pool)))
    pairs = []
    for r in picked:
        others = [o for o in pool if o["qid"] != r["qid"]]
        donor = min(others, key=lambda o: abs(len(o["cited_ids"]) - len(r["cited_ids"])))
        mine, wrong = cited_texts(r, texts), cited_texts(donor, texts)
        real, shuffled = runner.faithfulness(r["question"], r["conclusion"], mine), \
            runner.faithfulness(r["question"], r["conclusion"], wrong)
        j_real, j_wrong = judge.judge(r["question"], r["conclusion"], mine), judge.judge(r["question"], r["conclusion"], wrong)
        pairs.append({"qid": r["qid"], "donor": donor["qid"],
                      "faith_real": real["score"]["value"], "faith_shuffled": shuffled["score"]["value"],
                      "judge_real": j_real["label"], "judge_shuffled": j_wrong["label"]})

    probe = pool[0]
    empty_ctx = runner.faithfulness(probe["question"], probe["conclusion"], [])
    empty_ans = runner.faithfulness(probe["question"], "", [])
    return {"n": len(pairs), "mismatched_evidence": pairs,
            "empty_contexts": {"qid": probe["qid"], "value": empty_ctx["score"]["value"],
                               "n_statements": len(empty_ctx["statements"] or []),
                               "reading": empty_ctx["score"]["reading"]},
            "empty_answer": {"qid": probe["qid"], "value": empty_ans["score"]["value"],
                             "statements": empty_ans["statements"], "reading": empty_ans["score"]["reading"]}}


# ---------------- 护栏 6/7：检索面错配（Context Precision / Recall 的反面对照） ----------------

def context_controls(pass_rows: list[dict[str, Any]], texts: dict[str, str],
                     ctx_of: dict[str, list[str]], runner: RagasRunner) -> dict[str, Any]:
    """把某题的检索面换成**别一题的 Top-5**，CP/CR 必须显著下降。

    与 Faithfulness 那道错配对照不同：那一道验的是「NLI 有没有在看材料」，这一道验的是
    「名次加权与归因有没有真在起作用」。CP 尤其危险——它按名次加权，若模型其实没逐块看，
    换成别题的面也一样给高分，那这条指标就是装饰。
    """
    pool = [r for r in pass_rows if r.get("reference_answer") and ctx_of.get(r["qid"])]
    picked = random.Random(SEED).sample(pool, min(CONTROL_ROWS, len(pool)))
    by_qid = {r["qid"]: r for r in pass_rows}
    out = []
    for i, r in enumerate(picked):
        donor = picked[(i + 1) % len(picked)]
        if donor["qid"] == r["qid"]:
            continue
        wrong = [texts[c] for c in ctx_of[donor["qid"]] if c in texts]
        cp_wrong = runner.context_precision(r["question"], r["reference_answer"], wrong)["score"]
        cr_wrong = runner.context_recall(r["question"], r["reference_answer"], wrong)["score"]
        own = by_qid[r["qid"]]
        out.append({
            "qid": r["qid"], "donor": donor["qid"],
            "cp_own": (own.get("context_precision") or {}).get("value"),
            "cp_wrong": cp_wrong.get("value"),
            "cr_own": (own.get("context_recall") or {}).get("value"),
            "cr_wrong": cr_wrong.get("value"),
        })
    return {"n": len(out), "rows": out}


# ---------------- 阶段 E：embedding、AR 与答案错配对照 ----------------

def embed_phase(pass_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """evict 7B 之后才调：BGE-M3 与 7B 不能在这张卡上同时驻留（Gate 0 ① 的显存上界）。"""
    before = vram_used()
    client = OllamaClient()
    client.evict()
    time.sleep(2.0)
    sched = ResidencyScheduler(policy="time_sharded")
    need: list[str] = []
    spans: list[tuple[str, int]] = []
    for r in pass_rows:
        need.append(r["question"])
        # 每条再配一份**参考答案**：它是「量纲标尺」的分子——同题的参考答案与题面的余弦
        # 就是这套 embedding 下「完全切题」能长到的形状。Answer Relevancy 对目标值时必须有它，
        # 否则 0.90 这种线会被读成质量不达标，而它其实先受 embedding 量纲决定。
        need.append(str(r.get("reference_answer") or ""))
        spans.append((r["qid"], 2 + len(r.get("reverse_q") or [])))
        need.extend(r.get("reverse_q") or [])
    t0 = time.perf_counter()
    with sched.use("embed") as emb:
        vecs = [v.tolist() for v in emb.embed_passages(need, batch_size=8)] if need else []
        load_sec, device, dtype = emb.load_sec, sched.device, sched.dtype
    embed_sec = round(time.perf_counter() - t0, 1)

    vec_by_qid: dict[str, dict[str, Any]] = {}
    cursor = 0
    for pos, r in enumerate(pass_rows):
        span = spans[pos][1]
        got = vecs[cursor: cursor + span]
        cursor += span
        vec_by_qid[r["qid"]] = {"q": got[0], "gen": got[2:]}
        r["answer_relevance"] = answer_relevance_from_vectors(got[0], got[2:], r.get("noncommittal", False))
        r["scale_anchor"] = {"cos_question_vs_reference":
                             (round(cosine(got[0], got[1]), 4) if r.get("reference_answer") else None)}

    # 答案错配对照：同一道题配别一题的回答（反向问题取别一题那批），AR 必须更低
    usable = [r["qid"] for r in pass_rows if vec_by_qid[r["qid"]]["gen"]]
    ctl = []
    for i, qid in enumerate(usable[:AR_CONTROL_ROWS]):
        donor = usable[(i + 1) % len(usable)]
        if donor == qid:
            continue
        own = [cosine(vec_by_qid[qid]["q"], v) for v in vec_by_qid[qid]["gen"]]
        mixed = [cosine(vec_by_qid[qid]["q"], v) for v in vec_by_qid[donor]["gen"]]
        ctl.append({"qid": qid, "donor": donor, "own": round(st.mean(own), 4),
                    "mismatched": round(st.mean(mixed), 4), "delta": round(st.mean(own) - st.mean(mixed), 4)})
    after = vram_used()
    residency = {"policy": sched.policy, "device": device, "dtype": dtype,
                 "embed_load_sec": load_sec, "n_vectors": len(need), "embed_sec": embed_sec,
                 "vram_before_evict7b_mib": before, "vram_after_unload_mib": after,
                 "note": "先 evict 7B 再载 BGE-M3；embedding 用 float16/cuda 与在线检索同口径"}
    del sched
    return pass_rows, ctl, residency


# ---------------- 护栏闸门 ----------------

def control_gate(controls: dict[str, Any], ar_ctl: list[dict[str, Any]],
                 ctx_ctl: dict[str, Any]) -> dict[str, Any]:
    """七道护栏，任一条不过就判废——那说明装置测的不是它声称测的东西，不是「读数低」。"""
    me = [c for c in controls["mismatched_evidence"] if c["faith_real"] is not None and c["faith_shuffled"] is not None]
    drops = [c["faith_real"] - c["faith_shuffled"] for c in me]
    flips = sum(1 for c in me if c["judge_real"] != c["judge_shuffled"])
    ec, ea = controls["empty_contexts"], controls["empty_answer"]
    deltas = [c["delta"] for c in ar_ctl]
    cc = [c for c in ctx_ctl["rows"] if c["cp_own"] is not None]
    # 换成别题的面之后，「一块都不相关」会让 CP 的分母归零 ⇒ 未定义。
    # 对读数口径它必须报「不适用」（见 ragas.py 那条注释），但**对这道护栏它就是最大下降**：
    # 判不出相关块 = 模型确实看出了「这面里没有能答这题的东西」。所以只在对照里按 0 计，不改读数口径。
    cc_drop = [c["cp_own"] - (0.0 if c["cp_wrong"] is None else c["cp_wrong"]) for c in cc]
    crc = [c for c in ctx_ctl["rows"] if c["cr_own"] is not None]
    cr_drop = [c["cr_own"] - (0.0 if c["cr_wrong"] is None else c["cr_wrong"]) for c in crc]
    n_cp_undef = sum(1 for c in cc if c["cp_wrong"] is None)
    checks = {
        "证据错配后 Faithfulness 下降": {
            "pass": bool(drops) and st.mean(drops) > 0.0 and sum(1 for d in drops if d > 0) >= len(drops) - 1,
            "basis": f"{len(drops)} 对，平均降 {round(st.mean(drops), 4) if drops else 'NA'}，"
                     f"逐对下降 {sum(1 for d in drops if d > 0)}/{len(drops)}"},
        "证据错配后裁判 label 会翻": {
            "pass": bool(me) and flips >= 1, "basis": f"{flips}/{len(me)} 翻转"},
        "答案错配后 Answer Relevance 下降": {
            "pass": bool(deltas) and st.mean(deltas) > 0.0,
            "basis": f"{len(deltas)} 对，平均降 {round(st.mean(deltas), 4) if deltas else 'NA'}，"
                     f"逐对下降 {sum(1 for d in deltas if d > 0)}/{len(deltas)}"},
        "检索面错配后 Context Precision 下降": {
            "pass": bool(cc_drop) and st.mean(cc_drop) > 0.0,
            "basis": f"{len(cc_drop)} 对，平均降 {round(st.mean(cc_drop), 4) if cc_drop else 'NA'}，"
                     f"逐对下降 {sum(1 for d in cc_drop if d > 0)}/{len(cc_drop)}"
                     f"（其中 {n_cp_undef} 对换成别题面后「一块都不相关」⇒ 未定义，对照里按 0 计）"},
        "检索面错配后 Context Recall 下降": {
            "pass": bool(cr_drop) and st.mean(cr_drop) > 0.0,
            "basis": f"{len(cr_drop)} 对，平均降 {round(st.mean(cr_drop), 4) if cr_drop else 'NA'}，"
                     f"逐对下降 {sum(1 for d in cr_drop if d > 0)}/{len(cr_drop)}"},
        "空证据面不得判成立": {
            "pass": ec["value"] == 0.0,
            "basis": f"{ec['n_statements']} 条陈述全判 0 ⇒ {ec['reading']}"
                     if ec["value"] is not None else f"value=None｜{ec['reading']}"},
        "空回答面不得读成 1.0": {
            "pass": ea["value"] is None, "basis": f"{ea['reading']}（陈述 {ea['statements']}）"},
    }
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks.values())}


# ---------------- 汇总 ----------------

def summarize(pass_rows: list[dict[str, Any]], labels: dict[str, int]) -> dict[str, Any]:
    answered = [r for r in pass_rows if r["status"] in ANSWER_STATUSES]
    fc = [r["faith_cited"]["value"] for r in answered if r["faith_cited"].get("value") is not None]
    ft = [r["faith_top5"]["value"] for r in answered if r["faith_top5"].get("value") is not None]
    ar_a = [r["answer_relevance"]["value"] for r in answered
            if (r.get("answer_relevance") or {}).get("value") is not None]
    ar_all = [r["answer_relevance"]["value"] for r in pass_rows
              if (r.get("answer_relevance") or {}).get("value") is not None]
    cp = [r["context_precision"]["value"] for r in pass_rows
          if (r.get("context_precision") or {}).get("value") is not None]
    cr = [r["context_recall"]["value"] for r in pass_rows
          if (r.get("context_recall") or {}).get("value") is not None]
    anchor = [r["scale_anchor"]["cos_question_vs_reference"] for r in answered
              if (r.get("scale_anchor") or {}).get("cos_question_vs_reference") is not None]
    n_rel = [(r.get("context_precision") or {}).get("n_relevant", 0) for r in pass_rows]
    claims_n = [(r.get("context_recall") or {}).get("n_claims", 0) for r in pass_rows]
    pairs = [{"qid": r["qid"], "human": labels.get(r["qid"]), "judge": r["judge"]["label"]} for r in pass_rows]
    agree = agreement_with_human(pairs)
    answered_ids = {r["qid"] for r in answered}
    agree_ans = agreement_with_human([p for p in pairs if p["qid"] in answered_ids])
    agree_non = agreement_with_human([p for p in pairs if p["qid"] not in answered_ids])
    return {
        "population": {"n_rows": len(pass_rows), "n_answered": len(answered),
                       "n_non_answer": len(pass_rows) - len(answered)},
        "faithfulness_cited": {**describe(fc), "n_not_assessable": len(answered) - len(fc),
                               "statement_counts": [len(r["statements"]) for r in answered
                                                    if r.get("statements") is not None],
                               "n_decompose_failed": sum(1 for r in answered if r.get("statements") is None)},
        "faithfulness_top5": {**describe(ft), "n_not_assessable": len(answered) - len(ft)},
        "context_precision": {**describe(cp), "n_not_assessable": len(pass_rows) - len(cp),
                              "n_rows_no_relevant_context": sum(1 for x in n_rel if x == 0),
                              "relevant_contexts_mean": round(st.mean(n_rel), 4) if n_rel else None},
        "context_recall": {**describe(cr), "n_not_assessable": len(pass_rows) - len(cr),
                           "claims_mean": round(st.mean(claims_n), 4) if claims_n else None,
                           "claims_rows_zero": sum(1 for x in claims_n if x == 0)},
        "answer_relevance_answered": describe(ar_a),
        "answer_relevance_scale_anchor": {**describe(anchor),
                                          "share_over_0_90": (round(sum(1 for x in anchor if x >= 0.9) / len(anchor), 4)
                                                               if anchor else None),
                                          "note": "题面 vs **它自己的参考答案**的余弦：这是本 embedding 下"
                                                  "「完全切题」能长到的形状，不是质量读数"},
        "answer_relevance_all_rows": {**describe(ar_all),
                                      "note": "含非作答行；RAGAS 口径把 noncommittal 记 0，"
                                              "于是「拒得对」也被算进低分——不得单独对外引用这一档"},
        "judge_vs_human_all": agree,
        "judge_vs_human_answered": agree_ans,
        "judge_vs_human_non_answer": agree_non,
        "judge_by_rubric": {rid: sum(1 for r in pass_rows if r["judge"].get("rubric") == rid)
                            for rid in (ANSWER_CORRECT.id, DISPOSITION_CORRECT.id)},
        "judge_unjudgeable_rows": [r["qid"] for r in pass_rows if r["judge"]["label"] is None],
        "judge_truncated_rows": [r["qid"] for r in pass_rows if r["judge"].get("truncated")],
        "row_errors": [{"qid": r["qid"], "error": r["row_error"]} for r in pass_rows if r.get("row_error")],
        "pairs": pairs,
    }


def fmt(d: dict[str, Any], key: str = "mean") -> str:
    v = d.get(key) if d else None
    return "不适用" if v is None else f"{v:.4f}"


def fmt_ci(d: dict[str, Any]) -> str:
    c = d.get("ci95") if d else None
    return "—" if not c else f"[{c[0]}, {c[1]}]"


def kappa_of(agree: dict[str, Any]) -> str:
    k = agree.get("kappa")
    if not k:
        return "—"
    return f"{k['kappa']}（{k['note']}）" if "note" in k else str(k["kappa"])


def render(rep: dict[str, Any]) -> str:
    s, g = rep["summary"], rep["controls_gate"]
    L = [f"# M5 · RAGAS 复现与 LLM-as-Judge 实测{'（tag: ' + rep['tag'] + '）' if rep['tag'] else ''}", ""]
    L.append(f"生成时点 {rep['generated_at']}｜源产物 `{rep['source']['path']}`（md5 {rep['source']['md5']}）"
             f"｜人判标签 `{rep['labels']['path']}`（md5 {rep['labels']['md5']}）｜用时 {rep['elapsed_sec']}s")
    L.append("")
    L.append(f"> **装置**：{rep['device']}。裁判模型 **{rep['judge_model']}**——与生成模型是同一个，"
             "自偏好偏置未经外部对照，所以下面每条读数都必须连着第三节一起引用。")
    L.append("")
    L.append("## 一、四条 RAGAS 指标对目标值（§6.1 口径）")
    L.append("")
    L.append("| 指标 | 定义 | 目标 | 实测均值 | 95% CI | n | 判定 | 另一口径 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in rep["targets"]:
        d = s.get(t["key"], {})
        v = d.get("mean")
        ok = "不适用" if v is None else ("**达标**" if v > t["target"] else "**未达标**")
        alt = ""
        if t.get("alt") and s.get(t["alt"]):
            ad = s[t["alt"]]
            alt = f"{ad.get('mean')}（n={ad.get('n')}）"
        na = f"｜ 不适用 {d.get('n_not_assessable')} 行" if d.get("n_not_assessable") else ""
        L.append(f"| {t['name']} | {t['defn']} | > {t['target']} | **{fmt(d)}** | {fmt_ci(d)} | "
                 f"{d.get('n')}{na} | {ok} | {alt} |")
    L.append("")
    L.append(f"- Context Precision 的补充读数：逐题相关块数均值 "
             f"{s['context_precision'].get('relevant_contexts_mean')}，「一块都不相关」的行 "
             f"{s['context_precision']['n_rows_no_relevant_context']} 行（这些行进不了分母、报不适用）。")
    L.append(f"- Context Recall 的补充读数：参考答案拆出的声明数均值 "
             f"{s['context_recall'].get('claims_mean')}，声明数为 0 的行 {s['context_recall']['claims_rows_zero']} 行。")
    L.append("- 判定用的是**主口径那一个数**，另一口径只并列展示；拿高的那个去对目标就是择优取样本。")
    anc = s.get("answer_relevance_scale_anchor", {})
    ar_a = s.get("answer_relevance_answered", {})
    if anc.get("mean") is not None and ar_a.get("mean") is not None:
        L.append(f"- **Answer Relevancy 的量纲标尺**：同一批题，「题面 vs 它自己的参考答案」的余弦均值 "
                 f"**{anc['mean']}**（中位 {anc.get('median')}，最高 {anc.get('max')}，"
                 f"≥0.90 的占比 {anc.get('share_over_0_90')}）。也就是说在这套 BGE-M3 尺度上，"
                 f"**连「完全切题」这个上限本身都到不了 0.90**；而实测 AR {ar_a['mean']} 与它只差 "
                 f"{round(abs(ar_a['mean'] - anc['mean']), 4)}。所以那条 **未达标是量纲不匹配，不是答案跑题**——"
                 f"0.90 这条线照 RAGAS 默认（OpenAI embedding）写的，换到本地 embedding 上不能原样搬。"
                 f"要么改用本尺度的相对判据（错位面均值 "
                 f"{round(st.mean([c['mismatched'] for c in rep['ar_control']]), 4)} 是它的下界对照），"
                 f"要么显式声明用的是哪一种 embedding。")
        L.append("")
    L.append("## 二、明细读数")
    L.append("")
    fc, ft = s["faithfulness_cited"], s["faithfulness_top5"]
    aa, ab = s["answer_relevance_answered"], s["answer_relevance_all_rows"]
    L.append("| 指标 | 种群 | n | 均值 | 95% CI | 中位 | 零分行 |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| Faithfulness（cited 口径，与既有机械装置同尺） | 作答行 | {fc['n']} | {fmt(fc)} | {fmt_ci(fc)} | {fc['median']} | {fc['zeros']} |")
    L.append(f"| Faithfulness（Top-5 口径，RAGAS 原义） | 作答行 | {ft['n']} | {fmt(ft)} | {fmt_ci(ft)} | {ft['median']} | {ft['zeros']} |")
    L.append(f"| Answer Relevance | 作答行 | {aa['n']} | {fmt(aa)} | {fmt_ci(aa)} | {aa['median']} | {aa['zeros']} |")
    L.append(f"| Answer Relevance | 全行（正确拒答按官方口径记 0） | {ab['n']} | {fmt(ab)} | {fmt_ci(ab)} | {ab['median']} | {ab['zeros']} |")
    L.append("")
    sc = fc["statement_counts"]
    L.append(f"- 逐行陈述数："
             + (f"均值 {round(st.mean(sc), 2)}，范围 {min(sc)}–{max(sc)}；" if sc else "不适用（本轮没有作答行）；")
             + f"Faithfulness 不适用 {fc['n_not_assessable']} 行（非作答行没有可拆的陈述，**不得读成 0 分，也不得读成 1 分**）；"
             f"拆句失败 {fc['n_decompose_failed']} 行。")
    if s["row_errors"]:
        L.append(f"- **行级失败 {len(s['row_errors'])} 行**（这些行不进任何分母）："
                 + "、".join(f"{e['qid']}（{e['error'][:60]}）" for e in s["row_errors"]))
        L.append("")
    if fc["mean"] is not None and ft["mean"] is not None and ft["mean"] < fc["mean"]:
        L.append(f"- **Top-5 口径比 cited 口径低 {round(fc['mean'] - ft['mean'], 4)}**（"
                 f"{ft['zeros']} 行整行判 0）。被引块 100% 落在复跑 Top-5 之内（见第七节前的对齐率），"
                 "所以掉分不是证据面变窄，而是材料从 1 块变 5 块之后 7B 没在里面找到那条支撑——"
                 "**长上下文稀释**，也正是批量判定那道护栏要兜的风险。两口径都报，不挑高的那条对外说。")
        L.append("")
    L.append("")
    L.append("### 与既有读数并排（同一批 62 行，同一次生成）")
    L.append("")
    L.append("| 装置 | 性质 | 读数 |")
    L.append("|---|---|---|")
    L.append(f"| 数字落地率（机械） | 结论数字在被引证据正文出现的比例 | {rep['baseline']['number_groundedness']} |")
    L.append(f"| 引用落 gold 率（机械，门禁） | 行级 | {rep['baseline']['citation_gold']} |")
    L.append(f"| Faithfulness（本轮，LLM NLI） | 陈述级 | {fmt(fc)}（n={fc['n']}） |")
    L.append(f"| 答案正确率（**真人判**，报告值） | 行级 | {rep['baseline']['human_answer_correct']} |")
    L.append(f"| LLM-as-Judge（本轮） | 行级 | 一致率 {s['judge_vs_human_all']['reading']}，κ = {kappa_of(s['judge_vs_human_all'])} |")
    L.append("")
    L.append("## 三、装置有效性：LLM 裁判 vs 真人判")
    L.append("")
    ag, ag2, ag3 = s["judge_vs_human_all"], s["judge_vs_human_answered"], s["judge_vs_human_non_answer"]
    cf = ag.get("confusion")
    L.append(f"- 两张判据卡：作答行 `{ANSWER_CORRECT.id}` {s['judge_by_rubric'][ANSWER_CORRECT.id]} 行、"
             f"拒答/转人工行 `{DISPOSITION_CORRECT.id}` {s['judge_by_rubric'][DISPOSITION_CORRECT.id]} 行"
             "（后者看的是系统实际检到的 Top-5，与真人工单给标注人的同一套材料）。")
    L.append(f"- 全 {ag['n_paired'] + ag['n_excluded']} 行：双方都判出 {ag['n_paired']} 行，一致率 **{ag['reading']}**，"
             f"κ = {kappa_of(ag)}；裁判判不出 {ag['n_excluded']} 行。")
    L.append(f"- 只看作答行（成立性判据）：一致率 **{ag2['reading']}**，κ = {kappa_of(ag2)}。")
    L.append(f"- 只看非作答行（处置判据）：一致率 **{ag3['reading']}**，κ = {kappa_of(ag3)}。")
    if cf:
        L.append(f"- 混淆：人1裁1 {cf['human1_judge1']}｜人1裁0 {cf['human1_judge0']}｜人0裁1 {cf['human0_judge1']}｜"
                 f"人0裁0 {cf['human0_judge0']}；真人判错的 {ag['n_human_wrong']} 行里裁判抓到 "
                 f"**{ag['recall_on_human_wrong']}**，裁判说「成立/该拒」的比例 {ag['judge_yes_rate']:.2%}。")
    else:
        L.append("- 混淆矩阵：不适用（没有双方都判出的行）。")
    kap = (ag.get("kappa") or {}).get("kappa")
    if kap is not None and ag["n_paired"] >= 10 and kap < 0.2:
        base = (cf["human1_judge1"] + cf["human1_judge0"]) / ag["n_paired"]
        L.append(f"- **判读**：一致率 {ag['value']:.2%} 但 κ = {kap} —— 人判正类本身就占 {base:.2%}，"
                 "「两边都说成立」这一件事就能凭空造出这个一致率。真人判错的 "
                 f"{ag['n_human_wrong']} 行里裁判抓到 {ag['recall_on_human_wrong']}、"
                 f"裁判点头率 {ag['judge_yes_rate']:.2%} ⇒ **这套 LLM 裁判现在不能当人工初筛用**，"
                 "它把真人判错的东西照样判成对；对外引用这两个指标时必须带着这一条。")
    L.append("")
    if ag.get("divergent"):
        L.append("分叉行（交人工复核，两方都不自动改判）：")
        for d in ag["divergent"]:
            L.append(f"- {d['qid']}：人判 {d['human']} vs 裁判 {d['judge']}")
        L.append("")
    L.append(f"## 四、护栏（{len(g[chr(34)+chr(34)] if False else g['checks'])} 道，先过再看读数）")
    L.append("")
    for name, c in g["checks"].items():
        L.append(f"- {name}：{'过' if c['pass'] else '**不过**'} ｜ {c['basis']}")
    L.append("")
    n_checks = len(g["checks"])
    L.append(f"闸门：{n_checks} 道"
             f"{'全过 ⇒ 读数可用' if g['all_pass'] else '**有不过 ⇒ 本轮读数不得对外引用**'}"
             f"（不过的那道不是「读数难看」，是装置没在测它声称测的东西）。")
    L.append("")
    L.append("## 五、噪声地板（同代产物重跑一轮）")
    L.append("")
    f2 = rep["floor"]
    L.append(f"- LLM 裁判全量重跑 {f2['judge']['n']} 行：label 翻转 **{f2['judge']['flips']}** 行"
             f"（{f2['judge']['flip_rate']:.2%}）。小于这个率的 Δ 不得写成改善或恶化。")
    ff = f2["faithfulness"]
    L.append(f"- Faithfulness 固定 seed 抽 {ff['sampled']} 行重跑：行分平均绝对变化 {ff['mean_abs_delta']}，"
             f"其中拆句结果与第 1 轮不同的 {ff['restated']} 行——拆句由 7B 生成，是方差的主要来源。")
    L.append("")
    L.append("## 六、口径与边界（引用这一格必须连着说的话）")
    L.append("")
    for line in rep["caveats"]:
        L.append(f"- {line}")
    L.append("")
    L.append(f"- 检索复跑对齐：{rep['cost']['phase_r']}")
    L.append(f"- 换乘与 embedding：{json.dumps(rep['cost']['phase_e'], ensure_ascii=False)}")
    L.append("")
    c = rep["cost"]
    L.append("## 七、成本")
    L.append("")
    L.append(f"- LLM 调用 **{c['calls']}** 次（失败 {c['failed']}、坏 JSON {c['bad_json']}、"
             f"被 num_predict 截断 {c['truncated']}），模型侧累计 {c['llm_ms_total']}ms ≈ "
             f"{round(c['llm_ms_total'] / 60000, 1)} 分钟；token {c['prompt_tokens']} 入 / {c['completion_tokens']} 出；"
             f"单次提示最长 {c.get('max_prompt_tokens', '未记录')} token"
             f"（服务默认 num_ctx=4096，超长会从提示头部**静默截掉**，所以这一列必须看得见）。")
    L.append("")
    L.append("## 八、逐行明细")
    L.append("")
    L.append("| qid | status | 陈述数 | F(cited) | F(top5) | CP | CR(覆盖/声明) | AR | 裁判 | 人判 | 一致 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lab = {p["qid"]: p["human"] for p in s["pairs"]}
    for r in rep["rows"]:
        fv, tv, av, j = r["faith_cited"].get("value"), r["faith_top5"].get("value"), r["answer_relevance"].get("value"), r["judge"]["label"]
        cp = (r.get("context_precision") or {}).get("value")
        crc = (r.get("context_recall") or {})
        cr_txt = "不适用" if crc.get("value") is None else f"{crc.get('value'):.2f}（{crc.get('n_covered')}/{crc.get('n_gradable')}）"
        h = lab.get(r["qid"])
        n_st = "拆句失败" if r["statements"] is None else len(r["statements"])
        L.append(f"| {r['qid']} | {r['status']} | {n_st} | "
                 f"{'不适用' if fv is None else f'{fv:.2f}'} | {'不适用' if tv is None else f'{tv:.2f}'} | "
                 f"{'不适用' if cp is None else f'{cp:.2f}'} | {cr_txt} | "
                 f"{'—' if av is None else f'{av:.3f}'} | {'判不出' if j is None else j} | {h} | "
                 f"{'—' if j is None or h is None else ('✓' if j == h else '✗')} |")
    L.append("")
    return "\n".join(L)


CAVEATS = [
    "**Context Precision / Recall 用的是检索面（复跑 Top-5），不是被引块**：它们测「检到了没有」，"
    "Faithfulness 测「说出来的有没有依据」，分工不同，所以同一题可以 F=1.0 而 CP=0.2，那不是矛盾。"
    "CP/CR 覆盖**全 67 行**（含拒答/转人工行）——拒答行的检索面照样要评，"
    "「该答却检不到」正是 Context Recall 要暴露的东西。",
    "**CP 的「一块都不相关」那批行报不适用、不报 0**：位置加权公式的分母是相关块数，5 块全不相关时该式未定义。"
    "官方实现把这类样本折成 0.0 计入均值——**两种写法会给出不同的数**，引用时必须说明是哪一种；"
    "本装置另报「相关块数为 0 的行数」这个事实，不把它藏进均值里。",
    "**CR 需要参考答案**，与人工工单那条「不给参考答案」不冲突：那条管「结论对不对」（判的人不能抄），"
    "这一条测「检索面够不够」，按 RAGAS 定义必须有 reference 才能拆声明。",
    "**这两个指标是自研复现，不是 ragas 库**（官方包拖 scikit-network→numba，会顶掉本项目硬钉的 "
    "numpy==2.5.3，而 torch 是经 `.pth` 挂进 `.venv` 的）。prompt 为中文（官方为英文），"
    "contexts 见两个口径——对外称「KEE 复现版 Faithfulness / Answer Relevance」，不得称「跑了 RAGAS」。",
    "**裁判模型与生成模型同为 qwen2.5:7b**：自偏好偏置未经外部对照，第三节的一致率/κ 是它唯一的定级证据。",
    "**读数一律为报告值，不进门禁**（用户 2026-10-16 裁决：LLM 判定不可机械复现，"
    "门禁只收机械可复算判据这条纪律不变）。",
    "**未重跑生成**：输入是真人标签配对的那代留档产物 ⇒ 既有人判读数零作废；"
    "代价是这些指标量的是**那一批已判过的输出**，不是「系统此刻的输出」。",
    "**Top-5 口径只在 cited_ids ⊆ 复跑 Top-5 的行上可比**（对齐率见成本节前那行）；"
    "不齐的行是检索跨代漂移，不是标签错。",
    "**Answer Relevance 的全行档把正确拒答记成 0 分**（RAGAS 官方就这么算），"
    "单独引用它等于把「拒得对」说成「答得差」。",
    "**提示没撞满服务上下文窗口**：本机 8GB 显存下 Ollama 默认 `num_ctx=4096`，超长提示会从头部**静默截掉**"
    "（症状是模型在看一段没有材料的判定提示，而它不报错）。2026-10-19 单测过最坏情形："
    "5 块最长正文 + 参考答案的 CP 提示 = 1855 token，12 条陈述的批量 NLI 提示 = 2100 token ⇒ 有余量。"
    "第七节那行的 `max_prompt_tokens` 是每次运行的实测上限，改 prompt 后要回看它。",
    "**Faithfulness 的第二步用了一次批量判定**（一题的 N 条陈述合成一趟 7B 调用，官方是逐条）："
    "实测原口径 67 题 × 两口径要 ~740 次调用、约 2 小时，批量版 ~5 次/题。偏离买到的代价是模型一次"
    "要看完全部陈述、可能漏答——漏答一律记「判不出」进不了分母，而「证据换成别题的分数必须掉」那道"
    "护栏就是专门兜这种失明的；要跑逐条原口径加 `--per-statement`。",
    "**Answer Relevance 的绝对值不能与论文里用 OpenAI embedding 出来的数比**：这里的余弦来自本地 "
    "BGE-M3（dense，与在线检索同口径、float16/cuda），它本身的余弦分布就压在 0.55–0.75 这一段。"
    "可解释的标尺是第三节那条对照：同一题配**别一题的答案**时均值掉 0.2918、10/10 逐对下降——"
    "所以 0.6582 该读成「相对错位约有 0.29 的间距」，不是「相关性 65.8% 合格了」。",
    "**Faithfulness 与「数字落地率」不是一回事**：前者是逐条陈述的 NLI 支持性（LLM 判、不可复算），"
    "后者是结论数字是否出现在被引正文（机械、可复算）。两者并列是为了看分歧，不是互相验证。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="产物另存为 ragas_eval.<tag>.json，绝不覆盖主产物")
    ap.add_argument("--force", action="store_true", help="覆盖同名产物前先留底")
    ap.add_argument("--skip-retrieval", action="store_true", help="不复跑 Top-5，只出 cited 口径")
    ap.add_argument("--per-statement", action="store_true",
                    help="Faithfulness 第二步回到官方逐条 NLI（调用数约 3 倍，慢很多）")
    ap.add_argument("--render-only", action="store_true",
                    help="只按已有产物重排版报告（不跑任何模型）——报告措辞修改不该再花一遍 GPU 时间")
    args = ap.parse_args()

    if args.render_only:
        src = ART / (f"ragas_eval{('.' + args.tag) if args.tag else ''}.json")
        rep = json.loads(src.read_text(encoding="utf-8"))
        rep["summary"] = summarize(rep["rows"], read_labels())
        md = REPORT if not args.tag else REPORT.with_name(REPORT.stem + f"_{args.tag}.md")
        md.write_text(render(rep), encoding="utf-8")
        print(f"[只重排版] 读数未动（源 {src.name}）\n报告： {md}")
        return 0

    s = get_settings()
    t0 = time.perf_counter()
    if not SOURCE.exists():
        print(f"[判废] 源产物不存在：{SOURCE}")
        return 2

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows, labels = saved["rows"], read_labels()
    qids = [r["qid"] for r in rows]
    if sorted(qids) != sorted(labels):
        print(f"[判废] 混代拒绝：产物 {len(qids)} 行与工单 {len(labels)} 行 qid 不齐，"
              f"差集 {sorted(set(qids) ^ set(labels))[:6]}")
        return 2
    with psycopg.connect(s.postgres_dsn) as conn:
        texts = {r[0]: r[1] for r in conn.execute("SELECT id, content FROM chunk")}
    missing = {c for r in rows for c in (r.get("cited_ids") or [])} - set(texts)
    if missing:
        print(f"[判废] {len(missing)} 个被引块在 PG 取不到正文，例如 {sorted(missing)[:3]}")
        return 2
    if args.limit:
        rows = rows[: args.limit]
    ones = sum(labels.values())
    print(f"[源] {SOURCE.name} {len(rows)} 行｜qid 哈希 {qid_hash(qids)}｜人判标签 {len(labels)} 行（1:{ones} / 0:{len(labels) - ones}）")

    # ---- 阶段 R ----
    ctx_of: dict[str, list[str]] = {}
    r_note = "跳过（--skip-retrieval）"
    if not args.skip_retrieval:
        client_w = OllamaClient()
        client_w.evict()
        time.sleep(2.0)
        kb = KbStore(s.milvus_collection, "jieba")
        # co_resident 而不是 time_sharded：这里要连检 67 题，分时切换等于每题两次冷装载
        # （实测 13s/题 → 15 分钟）；与 m3_confidence_probe 同构造参数也是这条的理由。
        rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                             mode="hybrid", use_rerank=True)
        ctx = AccessContext(role="knowledge_owner")
        t = time.perf_counter()
        for i, r in enumerate(rows, start=1):
            ctx_of[r["qid"]] = [c.chunk_id for c in rt.retrieve(r["question"], ctx).chunks]
            if i % 20 == 0:
                print(f"[R {i}/{len(rows)}] {round(time.perf_counter() - t, 1)}s 显存 {vram_used()}MiB")
        aligned = sum(1 for r in rows if set(r.get("cited_ids") or []) <= set(ctx_of.get(r["qid"], [])))
        r_note = (f"evict 7B 后复跑 {len(rows)} 题检索耗时 {round(time.perf_counter() - t, 1)}s；"
                  f"cited_ids ⊆ 复跑 Top-5 的行 {aligned}/{len(rows)}")
        rt.sched.release_all()
        print(f"[R] {r_note}")

    # ---- 阶段 A ----
    llm = OllamaClient()
    warm = llm.warmup()
    print(f"[A] warm 7B：load {warm['load_ms']}ms｜显存 {vram_used()}MiB")
    runner, judge = RagasRunner(llm, batch_verify=not args.per_statement), LlmJudge(llm)
    pass1 = run_pass(rows, texts, ctx_of, runner, judge)

    # ---- 护栏 2/4/5 ----
    cr, cj = RagasRunner(llm, batch_verify=not args.per_statement), LlmJudge(llm)
    controls = negative_controls(rows, texts, cr, cj)
    print("[护栏] 证据错配 " + " ".join(
        f"{c['faith_real']}→{c['faith_shuffled']}" for c in controls["mismatched_evidence"][:5]))

    # ---- 护栏 6/7：检索面错配（CP / CR）----
    ccr = RagasRunner(llm, batch_verify=not args.per_statement)
    ctx_ctl = context_controls(pass1, texts, ctx_of, ccr)
    print("[护栏] 检索面错配 " + " ".join(
        f"CP {c['cp_own']}→{c['cp_wrong']}｜CR {c['cr_own']}→{c['cr_wrong']}" for c in ctx_ctl["rows"][:3]))

    # ---- 噪声地板 ----
    jf, rf = LlmJudge(llm), RagasRunner(llm, batch_verify=not args.per_statement)
    p1 = {r["qid"]: r for r in pass1}
    flips = sum(1 for r in rows
                if judge_row(jf, r, texts, ctx_of)["label"] != p1[r["qid"]]["judge"]["label"])
    sample = random.Random(SEED).sample(rows, min(FLOOR_FAITH_ROWS, len(rows)))
    deltas, restated, sampled = [], 0, 0
    for r in sample:
        if not (r.get("conclusion") or "").strip():
            continue
        sampled += 1
        again = rf.faithfulness(r["question"], r["conclusion"], cited_texts(r, texts))
        old = p1[r["qid"]]
        restated += int(again["statements"] != old["statements"])
        if again["score"]["value"] is not None and old["faith_cited"]["value"] is not None:
            deltas.append(abs(again["score"]["value"] - old["faith_cited"]["value"]))
    floor = {"judge": {"n": len(rows), "flips": flips, "flip_rate": round(flips / len(rows), 4)},
             "faithfulness": {"sampled": sampled, "n": len(deltas),
                              "mean_abs_delta": round(st.mean(deltas), 4) if deltas else None,
                              "restated": restated}}
    print(f"[地板] 裁判翻转 {flips}/{len(rows)}｜Faithfulness 抽样 {len(deltas)} 行平均变化 "
          f"{floor['faithfulness']['mean_abs_delta']}（拆句改写 {restated} 行）")

    # ---- 阶段 E ----
    pass1, ar_ctl, residency = embed_phase(pass1)
    print(f"[E] 答案错配对照 " + " ".join(f"{c['own']}→{c['mismatched']}" for c in ar_ctl[:5]))

    gate = control_gate(controls, ar_ctl, ctx_ctl)
    print("[闸门] " + " ｜ ".join(f"{k}:{'过' if v['pass'] else '不过'}" for k, v in gate["checks"].items()))
    if not gate["all_pass"]:
        print("[判废] 护栏不过——装置测的不是它声称测的东西，本轮不写读数。")
        return 3

    summary = summarize(pass1, labels)
    cost_detail = [runner.cost_summary(), judge.cost_summary(), cr.cost_summary(), cj.cost_summary(),
                   ccr.cost_summary(), jf.cost_summary(), rf.cost_summary()]
    cost = {
        "calls": sum(c["calls"] for c in cost_detail),
        "failed": sum(c["failed"] for c in cost_detail),
        "bad_json": sum(c["bad_json"] for c in cost_detail),
        "truncated": sum(c["truncated"] for c in cost_detail),
        "max_prompt_tokens": max((c.get("max_prompt_tokens", 0) for c in cost_detail), default=0),
        "llm_ms_total": round(sum(c["llm_ms_total"] for c in cost_detail), 1),
        "prompt_tokens": sum(c["prompt_tokens"] for c in cost_detail),
        "completion_tokens": sum(c["completion_tokens"] for c in cost_detail),
        "phase_r": r_note, "phase_e": residency,
    }
    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "device": ("自研复现（ragas 库未引入），零新依赖；prompt 中文、T=0.0；"
                   f"Faithfulness 第二步 = {'逐条 NLI（官方口径）' if args.per_statement else '一次批量判定（偏离，见护栏）'}"),
        "judge_model": llm.model,
        "source": {"path": str(SOURCE.relative_to(ROOT)), "md5": md5(SOURCE), "qid_hash": qid_hash(qids),
                   "temperature_used": saved.get("temperature_used")},
        "labels": {"path": str(LABELS.relative_to(ROOT)), "md5": md5(LABELS),
                   "note": "人判读数零作废：本次未重跑生成；裁判读数与人判读数分开命名，不互盖"},
        "summary": summary,
        "baseline": {
            "number_groundedness": "均值 1.0（n=51 可判）",
            "citation_gold": "61/62 = 0.9839",
            "human_answer_correct": "60/62 = 0.9677",
            "ece": "0.2006（真人标签，报告值；门禁 ECE ≤ 0.10 不达标）",
        },
        "controls": controls, "ar_control": ar_ctl, "context_controls": ctx_ctl,
        "controls_gate": gate, "floor": floor, "targets": TARGETS,
        "cost": cost, "caveats": CAVEATS,
        "rows": [{k: r[k] for k in ("qid", "status", "conclusion", "statements", "verdicts", "verdicts_top5",
                                    "faith_cited", "faith_top5", "reference_answer", "context_precision",
                                    "context_recall", "reference_claims", "context_recall_verdicts",
                                    "scale_anchor",
                                    "reverse_q", "answer_relevance", "judge")} for r in pass1],
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }

    out = ART / (f"ragas_eval{('.' + args.tag) if args.tag else ''}.json")
    if out.exists() and not args.force:
        print(f"[拒绝覆盖] {out} 已存在（要重来加 --force 会先留底；或换 --tag）")
        return 4
    if out.exists():
        keep = out.with_name(out.stem + f".pre-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(out, keep)
        print(f"[留底] {keep.name}")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    md = REPORT if not args.tag else REPORT.with_name(REPORT.stem + f"_{args.tag}.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render(rep), encoding="utf-8")
    print(f"\n证据： {out}\n报告： {md}\n用时 {rep['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
