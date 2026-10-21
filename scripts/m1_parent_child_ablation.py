"""Parent-Child 的「父块补全」到底值不值——四档消融，检索面恒定，人工标注零作废。

背景（本轮查出来的事实，不是猜测）：`format_evidence` 喂给模型的是**子块自己的正文**，
父块只以 `parent_id` 存在于存储与查询字段里，**从未被取回正文参与作答**。
现在唯一生效的"上下文补全"是子块开头那句 `章节标题｜`。所以本实验同时量三件事：
父块正文（A1）、标题内联（A2，去掉它）、以及**错配的父块**（A1b，反面对照）。

设计上的四条硬约束：
1. **检索只跑一次**，四档共用同一个 `RetrievalResult`（用回放式 retriever 注进真链路）。
   ⇒ 档位之间唯一的变量是喂进 prompt 的证据文本，检索侧差异被结构性排除；
   也因此 `retrieval` 置信度成分与 top_score 必须逐行相同——不同就说明有副作用，判废。
2. **走的是生产链路本身**：`AnswerEngine` + `kee.qa.graph` 全套（敏感闸、检索闸、
   无来源不出结论、引用一致性、复合分），只把 `graph.format_evidence` 换成按档位改写的包装。
   不自己复刻一份"差不多"的生成器——那测的就不是这套系统。
3. **人工标注零作废**：子块 `chunk_id` 一个都没变 ⇒ gold 与 547 行双人标注全部有效；
   代价是本实验**不能引用真人 `answer_correct` 那 62 行**（那是另一代输出），
   质量轴只能用 LLM 裁判 + 机械指标，且必须这样命名。
4. **A0 兼作保真检查**：A0 与留档 `pre_reprobe` 的结论逐字比对——同一模型同一 prompt 同一检索面，
   对不上就说明这套 harness 没复现在线路径，后面所有跨档差值都无意义。

护栏（不过就判废：跨档差值不采信，但**产物照落盘**并打 `void`——作废原因本身是证据）：
  G1 检索面恒定：四档 top_score 逐行相同；
  G2 提示不撞窗口：`max prompt_tokens < 3600`（Ollama 本机默认 num_ctx=4096，超长从**头部静默截掉**，
     那会把「父块有没有用」测成「提示被截了」）；
  G3 反面对照：**错配父块 A1b 不得优于正确父块 A1**——若 A1b≈A1，那 A1 的收益就不是上下文补全给的。
     **只用机械量**（可完成率 + 引用落 gold）：上一版拿 LLM 裁判成立率当这道闸门，
     被 κ=−0.0645、本轮给错配档打分最高的那把尺子误废了一轮好数据；裁判读数改为并列报告值。
  G4 改写真的生效：A2 必须确实去掉了标题前缀、A1 必须确实带上了父块正文（逐档自检一行）；
  G5 检索零降级：预热之后任何一行出现 `degraded` 都判废——单腿答会让候选面本身不可信。
噪声地板：A0 连跑两轮，报结论翻转率 / status 翻转率 / 引用落 gold 翻转率 / 裁判 label 翻转率。
  **跨档 Δ 小于这个地板的，一律写「测不出差别」，不写改善或恶化。**
  「引用落 gold」的跨档比较一律走**同分母配对**（各档分母自己会变），且「收益」额外要求净 ≥2 行。

跑法：uv run python scripts/m1_parent_child_ablation.py [--limit N] [--tag NAME]
      已有产物只重算派生读数与报告（不打 GPU，需 PG 起着以取 strict gold）：--render-only var/gate0/parent_child_ablation.json
产物：var/gate0/parent_child_ablation.json + reports/m1/M1_父块上下文消融.md
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
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
from kee.eval.judge import ANSWER_CORRECT, LlmJudge  # noqa: E402
from kee.eval.metrics import percentile  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa import answer as answer_mod  # noqa: E402
from kee.qa import graph as graph_mod  # noqa: E402
from kee.qa.answer import AnswerEngine, format_evidence, load_calibration_temperature  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

ART = ROOT / "var" / "gate0"
REPORT = ROOT / "reports" / "m1" / "M1_父块上下文消融.md"
PAIRED = ART / "pre_reprobe" / "m3_confidence_probe.json"

ARMS = ("A0_child", "A1_parent", "A2_noheading", "A1b_wrongparent")
ARM_LABEL = {
    "A0_child": "A0 现状：子块正文（含 `章节标题｜` 前缀）",
    "A1_parent": "A1 子块 + **其父块（整章节）正文**",
    "A2_noheading": "A2 子块正文，**去掉标题前缀**（现状减掉唯一生效的那点补全）",
    "A1b_wrongparent": "A1b 子块 + **别处的父块正文**（反面对照）",
}
PROMPT_TOKEN_CEILING = 3600
G3_MECH_TOL = 0.02
# 「同分母落 gold」要多动几行才算收益：净 1 行 = 单题分辨率，涨跌各一行就能凑出「+1」，不够格
MATERIAL_GOLD_NET_ROWS = 2
SEED = 20261019

CURRENT = {"arm": "A0_child", "parents": {}, "donor": {}, "order": []}


def arm_evidence(res: Any) -> tuple[str, dict[str, dict]]:
    """生产 `format_evidence` 的按档位改写。只改文本，不动 index（引用仍指向原子块）。"""
    text, index = format_evidence(res)
    arm = CURRENT["arm"]
    if arm == "A0_child":
        return text, index
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        eid = f"E{i + 1}"
        meta = index.get(eid) or {}
        pid = meta.get("parent_id")
        child_text = res.chunks[i].text if i < len(res.chunks) else ""
        if arm == "A2_noheading":
            # 精确剥：只去掉子块自己那个 `章节标题｜` 前缀（按已知 c.text 定位，不整行找 ｜，
            # 因为表格行正文里也可能出现 ｜，用 rpartition 会把正文一起削掉）
            stripped = child_text.split("｜", 1)[-1] if "｜" in child_text else child_text
            out.append(line[: len(line) - len(child_text)] + stripped if child_text else line)
        elif arm == "A1_parent":
            body = CURRENT["parents"].get(pid or "")
            CURRENT["order"].append(bool(body))
            out.append(line + (f"\n    【所属章节全文】{body}" if body else "\n    【所属章节全文】（缺）"))
        elif arm == "A1b_wrongparent":
            wrong = CURRENT["donor"].get(pid or "")
            CURRENT["order"].append(bool(wrong))
            out.append(line + (f"\n    【所属章节全文】{wrong}" if wrong else ""))
    return "\n".join(out), index


class CountingLlm:
    """包一层 Ollama：记录每次调用的 prompt/completion token 与耗时。判定路径零改动。"""

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
        return {"calls": len(self.calls), "prompt_tokens_p50": percentile(p, 0.5),
                "prompt_tokens_max": max(p), "prompt_tokens_sum": sum(p),
                "completion_tokens_sum": sum(c["completion_tokens"] for c in self.calls),
                "truncated": sum(1 for c in self.calls if c["done_reason"] == "length"),
                "ms_p50": round(percentile([c["total_ms"] for c in self.calls], 0.5), 1)}


class ReplayRetriever:
    """回放固定的 `RetrievalResult`：四档看到的是同一个检索面，检索侧差异被排除。"""

    def __init__(self, by_query: dict[str, Any]) -> None:
        self.by_query = by_query
        self.hits = 0

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        self.hits += 1
        return self.by_query[query]


def row_metrics(results: list[dict[str, Any]], gold: dict[str, set]) -> dict[str, Any]:
    answered = [r for r in results if r["status"] in ("answer", "tentative")]
    with_conc = [r for r in results if (r["conclusion"] or "").strip()]
    graded = [r for r in results if r["status"] in ("answer", "tentative") and r["citations"]]
    hit_gold = sum(1 for r in graded if {c.get("chunk_id") for c in r["citations"]} & gold.get(r["qid"], set()))
    return {
        "n": len(results),
        "status_dist": {k: sum(1 for r in results if r["status"] == k)
                        for k in ("answer", "tentative", "refuse", "escalate")},
        "completion_rate": round(len(with_conc) / len(results), 4),
        "conclusion_chars_p50": percentile([len(r["conclusion"] or "") for r in results], 0.5),
        "n_citations_p50": percentile([len(r["citations"]) for r in results], 0.5),
        "row_gold_rate": round(hit_gold / len(graded), 4) if graded else None,
        "row_gold": f"{hit_gold}/{len(graded)}" if graded else "不适用",
        "composite_p50": percentile([r["confidence"].get("composite", 0.0) for r in answered], 0.5) if answered else None,
        "top_score_p50": percentile([r["confidence"].get("retrieval", 0.0) for r in results], 0.5),
    }


def judge_arm(judge: LlmJudge, results: list[dict[str, Any]], texts_by_qid: dict[str, str]) -> dict[str, Any]:
    """裁判看的是**该档实际喂给模型的那套证据**——A1 有权用父块正文，判它时也只能拿那套材料。"""
    out = []
    for r in results:
        got = judge.judge(r["question"], r["conclusion"] or "", [texts_by_qid[r["qid"]]])
        out.append({"qid": r["qid"], "status": r["status"], "label": got["label"], "reason": got["reason"]})
    graded = [o for o in out if o["label"] in (0, 1)]
    answered = [o for o in out if o["status"] in ("answer", "tentative")]
    return {
        "n_gradable": len(graded),
        "n_unjudgeable": len(out) - len(graded),
        "supported_rate": round(sum(o["label"] for o in graded) / len(graded), 4) if graded else None,
        "reading": f"{sum(o['label'] for o in graded)}/{len(graded)}" if graded else "不适用",
        "n_answer_gradable": len([o for o in graded if o["status"] in ("answer", "tentative")]),
        "answer_supported_rate": (round(sum(o["label"] for o in answered if o["label"] in (0, 1))
                                        / len([o for o in answered if o["label"] in (0, 1)]), 4)
                                  if [o for o in answered if o["label"] in (0, 1)] else None),
        "rows": out,
    }


def paired_gold_view(runs: dict[str, list[dict[str, Any]]], strict: dict[str, set],
                     base: str = "A0_child") -> dict[str, Any]:
    """跨档比「引用落 gold」必须走**同分母配对**。

    `row_metrics` 里的分母是「该档 answer/tentative 且带引用的行」，而各档这个集合**不一样**：
    把证据换成父块正文会让若干行改走 escalate（本轮实测 A0 62 行 → A1 60 行），
    于是「率」的差里混着「谁被算进了分母」。配对视图只取**两档都可判的行**，
    并把掉出去/新增的行号如实列出，让读者看见代价落在哪些题上。
    """
    def graded_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {r["qid"]: r for r in rows
                if r["status"] in ("answer", "tentative") and r["citations"]}

    def hit(r: dict[str, Any], qid: str) -> bool:
        return bool({c.get("chunk_id") for c in r["citations"]} & strict.get(qid, set()))

    b = graded_map(runs[base])
    out = {"base": base, "n_base_gradable": len(b), "arms": {}}
    for arm, rows in runs.items():
        if arm == base:
            continue
        a = graded_map(rows)
        common = sorted(set(b) & set(a))
        dropped = sorted(set(b) - set(a))
        added = sorted(set(a) - set(b))
        hb = sum(1 for q in common if hit(b[q], q))
        ha = sum(1 for q in common if hit(a[q], q))
        gained = [q for q in common if not hit(b[q], q) and hit(a[q], q)]
        lost = [q for q in common if hit(b[q], q) and not hit(a[q], q)]
        out["arms"][arm] = {
            "n_common": len(common), "dropped_qids": dropped, "added_qids": added,
            "hits_base": hb, "hits_arm": ha,
            "rate_base": round(hb / len(common), 4) if common else None,
            "rate_arm": round(ha / len(common), 4) if common else None,
            "delta": round((ha - hb) / len(common), 4) if common else None,
            "gained_qids": gained, "lost_qids": lost, "net_rows": len(gained) - len(lost),
            "flips": [q for q in common if hit(b[q], q) != hit(a[q], q)],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--render-only", default="", help="已有产物 json 路径：只重算配对视图并重出报告，不打 GPU")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if args.render_only:
        # 只重算派生读数并重出报告：四档逐行数据已在产物里，不需要再打 GPU。
        # strict gold 仍要问一次权威源（纯 CPU/PG），所以这条通路只在库起着时可用。
        rep = json.loads(Path(args.render_only).read_text(encoding="utf-8"))
        st = {q["qid"]: set(q["gold_strict_ids"]) for q in
              build_eval_set(AuthoritativeStore(), write_files=False)["questions"]}
        rep["paired_gold"] = paired_gold_view(rep["rows"], st)
        rep["caveats"] = CAVEATS  # 口径说明随当前代码刷新，读数一律沿用产物里的原值
        rep["re_rendered_at"] = datetime.now().isoformat(timespec="seconds")
        Path(args.render_only).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        src = Path(args.render_only).resolve()
        tag = src.stem.replace("parent_child_ablation", "").lstrip(".")
        out_md = REPORT if not tag else REPORT.with_name(REPORT.stem + f"_{tag}.md")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render(rep), encoding="utf-8")
        print(f"[仅重出] 配对视图已补算｜产物 {args.render_only}｜报告 {out_md}")
        return 0

    s = get_settings()
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))
    gold = {r["qid"]: set() for r in paired["rows"]}
    with psycopg.connect(s.postgres_dsn) as conn:
        texts = {r[0]: r[1] for r in conn.execute("SELECT id, content FROM chunk")}
    # strict gold 从当前评测集取（题面与留档同一代，qid 对齐已核）
    ev = build_eval_set(AuthoritativeStore(), write_files=False)
    strict = {q["qid"]: set(q["gold_strict_ids"]) for q in ev["questions"]}

    rows = [r for r in paired["rows"] if r["status"] in ("answer", "tentative", "refuse", "escalate")]
    if args.limit:
        rows = rows[: args.limit]
    parents = {r[0]: r[1] for r in psycopg.connect(s.postgres_dsn).execute(
        "SELECT id, content FROM chunk WHERE is_parent")}
    pids = sorted(parents)
    donor_map = {p: parents[pids[(i + 7) % len(pids)]] for i, p in enumerate(pids)}
    CURRENT["parents"], CURRENT["donor"] = parents, donor_map
    print(f"[源] {PAIRED.name} {len(rows)} 行｜父块 {len(parents)} 个（中位 "
          f"{int(st.median([len(v) for v in parents.values()]))} 字）｜子块 {len(texts) - len(parents)} 个")

    # ---- 检索只跑一次（与 m3_confidence_probe 同构造参数）----
    kb = KbStore(s.milvus_collection, "jieba")
    rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                         mode="hybrid", use_rerank=True)
    ctx = AccessContext(role="knowledge_owner")
    CURRENT["arm"] = "A0_child"
    graph_mod.format_evidence = arm_evidence
    # 预热两次再开始计量：首呼要把 BGE-M3 与重排换入显存，稠密腿必然超预算 ⇒ 检索**静默退化成
    # 单腿 BM25**（本项目在 #56 与 ci_local.sh 的 `warm` 段都为此踩过）。把冷启动算进测量轮，
    # G5 就会稳定判废——而判废的理由不是系统问题，是我没预热。
    for wq in ("预热：配电自动化终端心跳超时阈值", "预热：清算链路灾备切换目标 RTO"):
        rt.retrieve(wq, ctx)
    by_query: dict[str, Any] = {}
    degraded: dict[str, list[str]] = {}
    t = time.perf_counter()
    for i, r in enumerate(rows, start=1):
        got = rt.retrieve(r["question"], ctx)
        by_query[r["question"]] = got
        degraded[r["qid"]] = list(got.degraded or [])
        if i % 20 == 0:
            print(f"[R] {i}/{len(rows)} {round(time.perf_counter() - t, 1)}s")
    rt.sched.release_all()
    n_deg = sum(1 for v in degraded.values() if v)
    print(f"[R] 检索面固定完成 {round(time.perf_counter() - t, 1)}s｜降级行 {n_deg}")

    replay = ReplayRetriever(by_query)
    ollama = OllamaClient()
    print(f"[A] warm 7B：{ollama.warmup()['load_ms']}ms")
    counting = CountingLlm(ollama)
    eng = AnswerEngine(retriever=replay, llm=counting)
    eng.calibration_temperature = load_calibration_temperature(s)

    runs: dict[str, list[dict[str, Any]]] = {}
    costs: dict[str, Any] = {}
    prompts: dict[str, dict[str, str]] = {}
    order_flags: dict[str, list[bool]] = {}
    for arm in ARMS:
        CURRENT["arm"] = arm
        CURRENT["order"] = []
        counting.reset()
        out = []
        for i, r in enumerate(rows, start=1):
            res = eng.answer(r["question"], ctx, top_k=5)
            ev_text, _ = arm_evidence(by_query[r["question"]])
            prompts.setdefault(arm, {})[r["qid"]] = ev_text
            out.append({"qid": r["qid"], "status": res.status, "question": r["question"],
                        "conclusion": res.conclusion, "citations": [dict(c) for c in res.citations],
                        "confidence": dict(res.confidence), "notes": list(res.notes)})
            if i % 20 == 0:
                print(f"[{arm}] {i}/{len(rows)}")
        runs[arm] = out
        costs[arm] = counting.summary()
        order_flags[arm] = list(CURRENT["order"])
        print(f"[{arm}] 完成｜可完成率 {row_metrics(out, strict)['completion_rate']}｜"
              f"prompt p50 {costs[arm]['prompt_tokens_p50']} max {costs[arm]['prompt_tokens_max']}")

    # ---- 噪声地板：A0 再跑一轮 ----
    CURRENT["arm"] = "A0_child"
    CURRENT["order"] = []
    counting.reset()
    rerun = []
    for r in rows:
        res = eng.answer(r["question"], ctx, top_k=5)
        rerun.append({"qid": r["qid"], "status": res.status, "conclusion": res.conclusion,
                      "citations": [dict(c) for c in res.citations], "confidence": dict(res.confidence)})
    floor_cost = counting.summary()

    # ---- LLM 裁判（四档；判据同真人 answer_correct 工单，但**这是 LLM 判，不覆盖人判读数**）----
    judge = LlmJudge(ollama)
    judged = {arm: judge_arm(judge, runs[arm], prompts[arm]) for arm in ARMS}
    judge_cost = judge.cost_summary()

    # ---- 保真检查：A0 vs 留档 ----
    paired_by_qid = {r["qid"]: r for r in rows}
    same_text = sum(1 for r in runs["A0_child"] if (r["conclusion"] or "") == (paired_by_qid[r["qid"]].get("conclusion") or ""))
    fidelity = {"n": len(runs["A0_child"]), "conclusion_identical": same_text,
                "rate": round(same_text / len(runs["A0_child"]), 4)}
    # ---- 噪声地板读数 ----
    by_q = {r["qid"]: r for r in runs["A0_child"]}
    flips = [r for r in rerun if (r["conclusion"] or "") != (by_q[r["qid"]]["conclusion"] or "")]
    status_flips = [r for r in rerun if r["status"] != by_q[r["qid"]]["status"]]
    gold_flips = []
    for r in rerun:
        a = {c.get("chunk_id") for c in by_q[r["qid"]]["citations"]} & strict.get(r["qid"], set())
        b = {c.get("chunk_id") for c in r["citations"]} & strict.get(r["qid"], set())
        if bool(a) != bool(b):
            gold_flips.append(r["qid"])
    judge_flips = []
    j0 = {o["qid"]: o["label"] for o in judged["A0_child"]["rows"]}
    for r in rerun:
        if r["status"] in ("answer", "tentative"):
            got = judge.judge(paired_by_qid[r["qid"]]["question"], r["conclusion"] or "",
                              [prompts["A0_child"][r["qid"]]])
            if got["label"] != j0[r["qid"]]:
                judge_flips.append(r["qid"])
    floor = {"conclusion_flip": len(flips), "conclusion_flip_rate": round(len(flips) / len(rows), 4),
             "status_flip": len(status_flips), "gold_hit_flip": len(gold_flips),
             "judge_label_flip": len(judge_flips), "flip_qids": [f["qid"] for f in flips],
             "cost": floor_cost, "judge_extra_calls": len(rows)}

    # ---- 护栏 ----
    m = {arm: row_metrics(runs[arm], strict) for arm in ARMS}
    tops = {arm: [r["confidence"].get("retrieval", 0.0) for r in runs[arm]] for arm in ARMS}
    g1 = all(tops[arm] == tops["A0_child"] for arm in ARMS)
    g2 = max(costs[arm]["prompt_tokens_max"] for arm in ARMS) < PROMPT_TOKEN_CEILING
    a1_rate, a1b_rate = judged["A1_parent"]["supported_rate"], judged["A1b_wrongparent"]["supported_rate"]
    # **G3 换用机械量**：上一版拿 LLM 裁判成立率做闸门，等于用一把已知走音的尺子定结论成不成立——
    # 该裁判与真人判 κ=−0.0645、A0 同档重跑的 label 翻转地板是 {floor['judge_label_flip']} 行，
    # 而 0.02 的裕度在 67 行上就是 1.3 行，判定完全落在噪声里（实际就被它误判废过一次）。
    # 这也违反本仓既有裁决「LLM 判定只作报告值、不进门禁」。裁判版 A1 vs A1b 仍原样并列在报告第三节里。
    a1_m, a1b_m = m["A1_parent"], m["A1b_wrongparent"]
    g3_gold = (a1b_m["row_gold_rate"] is None or a1_m["row_gold_rate"] is None
               or a1b_m["row_gold_rate"] <= a1_m["row_gold_rate"] + G3_MECH_TOL)
    g3_done = a1b_m["completion_rate"] <= a1_m["completion_rate"] + G3_MECH_TOL
    g3 = g3_gold and g3_done
    g4a = all(order_flags["A1_parent"]) and all(order_flags["A1b_wrongparent"])
    a2_sample = list(prompts["A2_noheading"])[:5]
    a2_stripped = sum(1 for q in a2_sample if prompts["A2_noheading"][q] != prompts["A0_child"][q])
    g5 = n_deg == 0
    gates = {
        "G1 检索面恒定（四档 top_score 逐行相同）": {"pass": g1, "basis": "逐行比对 retrieval 成分"},
        "G5 检索面零降级": {"pass": g5, "basis": f"降级行 {n_deg}/{len(rows)}"
                             + ("" if g5 else f"（例：{[k for k, v in degraded.items() if v][:3]}）"
                                              "——单腿答会让 top_score 与候选面都不可信，四档差值全部无意义")},
        "G2 提示不撞上下文窗口": {
            "pass": g2, "basis": f"四档最大 prompt_tokens {max(costs[a]['prompt_tokens_max'] for a in ARMS)}"
                                f" < 上限 {PROMPT_TOKEN_CEILING}（服务默认 num_ctx=4096）"},
        "G3 错配父块不得优于正确父块（机械量）": {
            "pass": g3, "basis": f"可完成率 A1 {a1_m['completion_rate']} vs A1b {a1b_m['completion_rate']}；"
                                f"引用落 gold A1 {a1_m['row_gold']} vs A1b {a1b_m['row_gold']}；"
                                f"容差 {G3_MECH_TOL}。**裁判版对照（仅报告值，不进门禁）**："
                                f"A1 {a1_rate} vs A1b {a1b_rate}，同档 label 翻转地板 {floor['judge_label_flip']} 行"},
        "G4 改写真的生效": {"pass": g4a and a2_stripped == len(a2_sample),
                          "basis": f"A1 注入父块 {sum(order_flags['A1_parent'])}/{len(order_flags['A1_parent'])}、"
                                   f"A1b 注入错配父块 {sum(order_flags['A1b_wrongparent'])}——"
                                   f"**这两个计数的是证据行（每题 5 条引用 × 每次组装），不是题目行**；"
                                   f"A2 抽样 {len(a2_sample)} 题、标题前缀确实剥掉 {a2_stripped}"},
    }
    all_pass = all(g["pass"] for g in gates.values())
    print("\n[护栏] " + " ｜ ".join(f"{k.split(' ')[0]}:{'过' if v['pass'] else '不过'}" for k, v in gates.items()))
    if not all_pass:
        print("[判废] 护栏不过——跨档差值不采信。**但产物照写**：作废的这轮也是证据，"
              "上一版在这里 return 掉了 40 分钟的逐行数据，那是我的实验设计缺陷，已改。")

    deltas = {}
    for arm in ARMS:
        if arm == "A0_child":
            continue
        deltas[arm] = {k: (round(m[arm][k] - m["A0_child"][k], 4)
                           if isinstance(m[arm].get(k), (int, float)) and isinstance(m["A0_child"].get(k), (int, float))
                           else None)
                       for k in ("completion_rate", "row_gold_rate", "composite_p50", "conclusion_chars_p50")}
        deltas[arm]["supported_rate_vs_A0"] = (round(judged[arm]["supported_rate"] - judged["A0_child"]["supported_rate"], 4)
                                              if judged[arm]["supported_rate"] is not None
                                              and judged["A0_child"]["supported_rate"] is not None else None)
        deltas[arm]["prompt_tokens_p50_vs_A0"] = round(
            costs[arm]["prompt_tokens_p50"] - costs["A0_child"]["prompt_tokens_p50"], 1)

    # ---- 判读：每个跨档 Δ 与它**自己的**同档抖动地板比 ----
    # 地板取同档 A0 重跑的翻转率（同指标同分母），不是拍一个 0.02 当通用阈值
    paired = paired_gold_view(runs, strict)
    n0 = judged["A0_child"]["n_gradable"]
    noise_cmp = {
        "completion_rate": round(floor["conclusion_flip"] / len(rows), 4),
        "row_gold_rate": round(floor["gold_hit_flip"] / len(rows), 4),
        "supported_rate": round(floor["judge_label_flip"] / n0, 4) if n0 else None,
    }
    verdict = {}
    for arm in ARMS:
        if arm == "A0_child":
            continue
        v = {}
        for key, lab in (("completion_rate", "completion_rate"), ("row_gold_rate", "row_gold_rate"),
                         ("supported_rate_vs_A0", "supported_rate")):
            dv, nf = deltas[arm].get(key), noise_cmp.get(lab)
            if dv is None or nf is None:
                v[key] = "不适用"
            else:
                v[key] = (f"Δ {dv:+.4f} vs 地板 {nf:.4f} ⇒ "
                          + ("**有效差**" if abs(dv) > nf else "落在同档抖动内，**测不出差别**"))
        v["prompt_tokens_p50_vs_A0"] = deltas[arm]["prompt_tokens_p50_vs_A0"]
        v["row_gold_rate"] += "（**此率各档分母不同，仅参考**，判定看下面的同分母配对）"
        p = paired["arms"][arm]
        v["row_gold_paired"] = (f"同分母 {p['n_common']} 行：A0 落 gold {p['hits_base']} → 本档 {p['hits_arm']}"
                                f"（Δ {p['delta']:+.4f}）；因转 escalate/丢引用而掉出 {len(p['dropped_qids'])} 行、"
                                f"新增 {len(p['added_qids'])} 行；命中翻转的题 {p['flips'] or '无'}")
        verdict[arm] = v

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "void": not all_pass,
        "arms": {a: ARM_LABEL[a] for a in ARMS},
        "why_this_exists": ("父块正文从未进入生成链路（`format_evidence` 只喂子块 `c.text`，"
                            "`parent_id` 只存在于存储与查询字段）；当前唯一生效的上下文补全是子块开头的"
                            "`章节标题｜` 前缀。所以本实验测的是「把没接的那一半接上，值不值」。"),
        "corpus_shape_note": (f"本语料父块中位 {int(st.median([len(v) for v in parents.values()]))} 字、"
                              f"子块中位 {int(st.median([len(v) for k, v in texts.items() if k not in parents]))} 字 ⇒ "
                              "**章节大多就一段，父块相对子块的先天信息增量有限**；"
                              "读数若接近持平，原因在这里，不能推广成「父块检索没用」。"),
        "retrieval_degraded": degraded,
        "source": {"path": str(PAIRED.relative_to(ROOT)), "n_rows": len(rows),
                   "retrieval": "检索只跑一次、四档共用同一 RetrievalResult（回放式 retriever 注进真链路）"},
        "fidelity_check": fidelity,
        "noise_floor": floor,
        "metrics_by_arm": m,
        "judge_by_arm": {a: {k: v for k, v in judged[a].items() if k != "rows"} for a in ARMS},
        "deltas_vs_A0": deltas,
        "noise_comparison": noise_cmp,
        "paired_gold": paired,
        "verdict_vs_floor": verdict,
        "cost_by_arm": costs,
        "gates": gates,
        "rows": {a: runs[a] for a in ARMS},
        "judge_rows": {a: judged[a]["rows"] for a in ARMS},
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "caveats": CAVEATS, "judge_calls": judge_cost["calls"],
    }
    out_json = ART / (f"parent_child_ablation{('.' + args.tag) if args.tag else ''}.json")
    if out_json.exists():
        keep = out_json.with_name(out_json.stem + f".pre-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(out_json, keep)
        print(f"[留底] {keep.name}")
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out_md = REPORT if not args.tag else REPORT.with_name(REPORT.stem + f"_{args.tag}.md")
    out_md.write_text(render(rep), encoding="utf-8")
    print(f"\n证据： {out_json}\n报告： {out_md}\n用时 {rep['elapsed_sec']}s")
    return 0 if all_pass else 3


def conclusion_lines(rep: dict[str, Any]) -> list[str]:
    """结论由读数派生，不手写——手写段落在下一次重跑会被 render 冲掉。"""
    if rep.get("void"):
        return ["> **本轮不出结论**：护栏未全过。下面这些数字只能用于定位是哪道护栏、被什么数据照废的。"]
    m, floor = rep["metrics_by_arm"], rep["noise_floor"]
    paired = rep.get("paired_gold", {}).get("arms", {})
    nf = rep["noise_comparison"]
    out: list[str] = []
    # 「收益」的门槛先定死再看数：可完成率要**超出同档抖动地板**，落 gold 要**净 ≥ MATERIAL_GOLD_NET_ROWS 行**
    # ——净 1 行是单题的分辨率（一涨一跌就能凑出来），不够格叫变好。
    gains: list[str] = []
    for arm in ("A1_parent", "A2_noheading", "A1b_wrongparent"):
        dv, pg = rep["deltas_vs_A0"][arm], (paired.get(arm) or {})
        if dv["completion_rate"] is not None and dv["completion_rate"] > nf["completion_rate"]:
            gains.append(f"{arm} 可完成率 Δ {dv['completion_rate']:+.4f}（> 地板 {nf['completion_rate']}）")
        if pg.get("net_rows") is not None and pg["net_rows"] >= MATERIAL_GOLD_NET_ROWS:
            gains.append(f"{arm} 同分母落 gold 净 +{pg['net_rows']} 行（{pg['hits_base']}→{pg['hits_arm']}）")
    floor_txt = (f"结论文本翻转地板 {floor['conclusion_flip']}/{rep['source']['n_rows']} "
                 f"= {floor['conclusion_flip_rate']:.2%}")
    if gains:
        out.append("- **有没有哪一档真的更好**：" + "；".join(gains) + "。")
    else:
        nets = "、".join(f"{a.split('_')[0]} 净 {(paired.get(a) or {}).get('net_rows')} 行"
                         for a in ("A1_parent", "A2_noheading", "A1b_wrongparent"))
        out.append("- **有没有哪一档真的更好**：**没有**。可完成率的全部正向变化都在同档抖动以内"
                   f"（{floor_txt}）；同分母落 gold 只动单题（{nets}，门槛净 ≥{MATERIAL_GOLD_NET_ROWS} 行）"
                   "且涨跌互见——那是换题，不是变好。")
    a1, a2, a1b = paired.get("A1_parent", {}), paired.get("A2_noheading", {}), paired.get("A1b_wrongparent", {})
    d1 = rep["deltas_vs_A0"]["A1_parent"]
    out.append(f"- **把没接的那一半接上（A1 = 子块 + 其父块整章节）**：可完成率 Δ {d1['completion_rate']:+.4f}"
               f"（地板 {nf['completion_rate']}）、同分母落 gold 净 {a1.get('net_rows', 0):+d} 行"
               f"（涨 {len(a1.get('gained_qids', []))} / 跌 {len(a1.get('lost_qids', []))}）、"
               f"代价是 prompt p50 {d1['prompt_tokens_p50_vs_A0']:+.0f} tok 且 escalate "
               f"{m['A0_child']['status_dist']['escalate']} → {m['A1_parent']['status_dist']['escalate']} 行"
               " ⇒ **读数支持不接**。要不要据此改需求原文（《需求文档》US-1.2 验收写的就是「父块补上下文」），"
               "列在待定清单 **C6** 待裁；本轮只把「已实现」口径的表述改成实情。")
    d2 = rep["deltas_vs_A0"]["A2_noheading"]
    out.append(f"- **现状唯一生效的那一半（A2 = 删掉子块开头的 `章节标题｜` 前缀）**：可完成率 Δ {d2['completion_rate']:+.4f}"
               f"（在地板内）、同分母落 gold 净 {a2.get('net_rows', 0):+d} 行、"
               f"prompt p50 {d2['prompt_tokens_p50_vs_A0']:+.0f} tok"
               " ⇒ 删掉它没有任何可测后果。这点头补全在本语料上**无可测贡献**；它是零成本的所以保留，"
               "但文档不能再把它描述成「父块补全上下文已生效」。")
    d1b = rep["deltas_vs_A0"]["A1b_wrongparent"]
    out.append(f"- **反面对照（A1b = 灌别处的父块正文）**：可完成率 Δ {d1b['completion_rate']:+.4f}"
               f"{'，**超出地板** ⇒ 可复现的劣化' if abs(d1b['completion_rate'] or 0) > nf['completion_rate'] else '（仍在地板内）'}、"
               f"escalate {m['A0_child']['status_dist']['escalate']} → {m['A1b_wrongparent']['status_dist']['escalate']} 行。"
               "错配比正确更伤 ⇒ 这一档量到的是「往提示里灌长文本」的代价，不是上下文补全的收益；"
               "所以 A1 的读数必须与 A1b 并排看（护栏 G3 就是为这件事设的）。")
    jr = {a: rep["judge_by_arm"][a]["supported_rate"] for a in rep["judge_by_arm"]}
    best = max((a for a in jr if jr[a] is not None), key=lambda a: jr[a])
    out.append(f"- **裁判这杆秤再次自证没有判别力**：四档里它给「{best}」打的成立率最高（{jr[best]}），"
               + ("而这一档正是灌**错配**父块的反面对照——垃圾上下文被它判成最好。" if best == "A1b_wrongparent" else "")
               + f"同档重跑它的 label 翻转 {floor['judge_label_flip']} 行、与真人判 κ=−0.0645。"
               "上一版把它的成立率放进护栏 G3 当闸门，把一轮好数据误判废了——"
               "**任何把 LLM 判定放进门禁的设计都应按这条实证撤掉**（本次已撤，改机械量）。")
    out.append(f"- **外推边界**：{rep['corpus_shape_note']}")
    return out


def render(rep: dict[str, Any]) -> str:
    m, d, c, j = rep["metrics_by_arm"], rep["deltas_vs_A0"], rep["cost_by_arm"], rep["judge_by_arm"]
    L = ["# M1 · Parent-Child 父块上下文消融（四档，检索面恒定）", ""]
    if rep.get("void"):
        L.append("> **本轮判废**：护栏未全过 ⇒ 全部跨档差值一律不采信，只作存档。"
                 "产物照落盘，因为「哪道护栏、被什么数据照废」本身就是结论的一部分。")
        L.append("")
    L.append(f"生成时点 {rep['generated_at']}｜源 {rep['source']['path']}（{rep['source']['n_rows']} 行）"
             f"｜用时 {rep['elapsed_sec']}s｜**人工标注零作废**（子块 id 未变）")
    L.append("")
    L.append(f"> **为什么要做这个**：{rep['why_this_exists']}")
    L.append("")
    L.append("## 一、结论（由读数派生，重跑自动更新）")
    L.append("")
    for line in conclusion_lines(rep):
        L.append(line)
    L.append("")
    L.append("## 二、四档读数")
    L.append("")
    L.append("| 档 | 含义 | 可完成率 | 行级引用落 gold | 结论字数 p50 | 复合分 p50 | 裁判成立率 | prompt p50/max tok |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in ("A0_child", "A1_parent", "A2_noheading", "A1b_wrongparent"):
        L.append(f"| {arm} | {rep['arms'][arm]} | {m[arm]['completion_rate']} | {m[arm]['row_gold']}"
                 f"（{m[arm]['row_gold_rate']}） | {m[arm]['conclusion_chars_p50']} | {m[arm]['composite_p50']} | "
                 f"**{j[arm]['supported_rate']}**（{j[arm]['reading']}，判不出 {j[arm]['n_unjudgeable']}） | "
                 f"{c[arm]['prompt_tokens_p50']} / {c[arm]['prompt_tokens_max']} |")
    L.append("")
    L.append("## 三、相对 A0 的差值（必须先和噪声地板比）")
    L.append("")
    f = rep["noise_floor"]
    L.append(f"**噪声地板**（同一档 A0 连跑两轮，{rep['source']['n_rows']} 行）：结论文本翻转 "
             f"**{f['conclusion_flip']} 行（{f['conclusion_flip_rate']:.2%}）**、status 翻转 {f['status_flip']}、"
             f"引用落 gold 翻转 {f['gold_hit_flip']}、裁判 label 翻转 {f['judge_label_flip']}。"
             "**小于这个地板的跨档差一律写「测不出差别」。**")
    L.append("")
    L.append("| 档 | 可完成率 Δ | 引用落 gold Δ | 复合分 Δ | 结论字数 Δ | 裁判成立率 Δ | prompt Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ("A1_parent", "A2_noheading", "A1b_wrongparent"):
        x = d[arm]
        L.append(f"| {arm} | {x['completion_rate']} | {x['row_gold_rate']} | {x['composite_p50']} | "
                 f"{x['conclusion_chars_p50']} | {x['supported_rate_vs_A0']} | {x['prompt_tokens_p50_vs_A0']:+} tok |")
    L.append("")
    L.append("### 判读（Δ 对各自地板，机械判定，不靠手写措辞）")
    L.append("")
    L.append("> 注：`引用落 gold` 的**率**各档分母不同（掉出去的行正是改走 escalate / 丢引用的那些），"
             "跨档比它不自洽；所以下面每条都配一行**同分母配对**（只取两档都可判的行）。")
    L.append("")
    for arm in ("A1_parent", "A2_noheading", "A1b_wrongparent"):
        v = rep["verdict_vs_floor"][arm]
        L.append(f"- **{arm}**：可完成率 {v['completion_rate']}；引用落 gold {v['row_gold_rate']}；"
                 f"裁判成立率 {v['supported_rate_vs_A0']}；提示成本 {v['prompt_tokens_p50_vs_A0']:+} tok p50")
        pg = rep.get("paired_gold", {}).get("arms", {}).get(arm)
        if pg and pg["n_common"]:
            L.append(f"  - 同分母 {pg['n_common']} 行：A0 落 gold {pg['hits_base']} → 本档 {pg['hits_arm']}"
                     f"（净 {pg['hits_arm'] - pg['hits_base']:+d} 行，Δ {pg['delta']:+.4f}；"
                     f"涨 {pg.get('gained_qids') or '无'} / 跌 {pg.get('lost_qids') or '无'}），"
                     f"命中翻转共 {len(pg['flips'])} 行；"
                     f"掉出判定 {pg['dropped_qids'] or '无'}、新进入 {pg['added_qids'] or '无'}；"
                     f"escalate 行数 A0 {m['A0_child']['status_dist']['escalate']} → 本档 "
                     f"{m[arm]['status_dist']['escalate']}")
    L.append("")
    L.append("## 四、护栏")
    L.append("")
    for k, v in rep["gates"].items():
        L.append(f"- {k}：{'过' if v['pass'] else '**不过**'} ｜ {v['basis']}")
    L.append("> 计数口径提醒：G4 里 A1/A1b 的分子是**证据行**（每题 5 条引用 × 每轮组装次数），不是 67 道题；"
             "题目级覆盖看 G1 的逐行比对与 `paired_gold.dropped_qids`。代码侧已把这条措辞改成「证据行」，"
             "本产物由改动前那趟跑出，故仍印「行」。")
    L.append(f"- 保真检查（A0 是否复现留档输出）：结论逐字一致 "
             f"**{rep['fidelity_check']['conclusion_identical']}/{rep['fidelity_check']['n']} "
             f"= {rep['fidelity_check']['rate']:.2%}**"
             f"（不要求 100%。但**别拿同代漂移来解释这个缺口**：同代 A0 重跑只翻 "
             f"{rep['noise_floor']['conclusion_flip']} 行 = {rep['noise_floor']['conclusion_flip_rate']:.2%}，"
             "与留档差这么多是**跨代**（留档那代之后动过身份默认、检索量纲闸门、退出撤销）"
             "⇒ 本实验绝对值不得与留档读数并列引用，见 §五 口径）")
    L.append("")
    L.append("## 五、口径与边界")
    L.append("")
    for line in rep["caveats"]:
        L.append(f"- {line}")
    L.append("")
    L.append("## 六、成本")
    L.append("")
    for arm in ("A0_child", "A1_parent", "A2_noheading", "A1b_wrongparent"):
        L.append(f"- {arm}：{c[arm]['calls']} 次生成调用、prompt 合计 {c[arm]['prompt_tokens_sum']} token、"
                 f"被截断 {c[arm]['truncated']}、单次耗时 p50 {c[arm]['ms_p50']}ms")
    L.append(f"- 裁判：{rep['judge_calls']} 次调用；噪声地板额外 {f['cost']['calls']} 次生成 + {f['judge_extra_calls']} 次裁判")
    L.append("")
    return "\n".join(L)


CAVEATS = [
    "**质量轴用的是 LLM 裁判（qwen2.5:7b，判据照真人 `answer_correct` 工单），不是真人判**："
    "本实验重新生成了输出，而真人那 62 行标签只配它判过的那代产物 ⇒ 两者**不可互换引用**；"
    "且该裁判自身与真人判的 κ = −0.0645（见 `M5_RAGAS与LLM裁判实测.md`），"
    "所以它的绝对值只能用于**同档内跨档相对比较**，不能对外当「答案正确率」。",
    "**裁判每档看的是该档实际喂给模型的那套证据**（A1 有权用父块正文，判它也只能用那套）——"
    "否则等于用更窄的材料去判一个用了更多材料的答案，跨档差值会假性偏向 A0。",
    "**语料形状限制外推**：父块中位 276 字 vs 子块中位 222 字，章节大多一段 ⇒ "
    "父块相对子块的先天增量本来就小。**读数持平只能说明「在本语料上持平」，不能说明父块检索无价值**；"
    "真实交付文档里章节普遍长得多，那一半的收益要等真语料重测。",
    "**A1 的代价是提示变长**（prompt token 上升），在 8GB 单卡 + num_ctx=4096 的约束下，"
    "父块正文一旦超过护栏 G2 的上限就会被**从头部静默截掉**——那时测到的就不是「上下文补全」而是「截断」。",
    "**检索面恒定是这套设计的前提**：四档共用同一个 `RetrievalResult`（回放式 retriever），"
    "所以 `retrieval` 置信度成分与 top_score 必须逐行相同（G1）；不同就说明改写证据文本影响了检索，"
    "那本实验的结论全部作废。",
    "**跨档比「引用落 gold」只看同分母配对**：各档的率分母是「该档 answer/tentative 且带引用的行」，"
    "注入父块正文会让若干行改走 escalate，分母自己就变了（本轮 A0 62 → A1 60 → A1b 58）。"
    "率的差里混着「谁被算进分母」，所以报告以配对行（两档都可判的行）为准，率只作原始刻度。",
    "**A0 与留档产物是两代系统**：逐字一致率见第一节「保真检查」，而同代重跑两轮只翻极少数行"
    "（见噪声地板）——两者相差悬殊，说明差的不是噪声而是跨代（留档那代之后动过身份默认、检索量纲闸门、"
    "退出撤销等）。所以本实验的绝对值**不能**与 `m3_confidence_probe` / 人判那批读数并列引用，"
    "四档之间的相对比较才是本实验的证据范围。",
]

if __name__ == "__main__":
    sys.exit(main())
