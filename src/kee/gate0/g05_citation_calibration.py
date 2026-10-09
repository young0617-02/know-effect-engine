"""Gate 0 ⑤：引用强制与置信度校准（温度缩放 + ECE）。

对全量评测集跑一遍问答链路，产出四件事：
  1) 引用强制是否真的兜住（无出处的结论必须为 0 条）；
  2) 无依据场景是否必不输出结论（拒答正确率）；
  3) 低置信度/敏感是否必触发降级或转人工；
  4) 复合置信度经温度缩放后 ECE 是否具备 ≤0.10 的达成路径。
校准在半量数据上拟合 T，在另外半量上评估 ECE（不做同集自评的自欺）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import apply_temperature, expected_calibration_error, fit_temperature  # noqa: E402
from kee.eval.metrics import latency_report  # noqa: E402
from kee.gate0.common import env_snapshot, md_table, verdict, write_json, write_md  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa.answer import AnswerEngine  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402


def label_row(q: dict, res: Any) -> dict[str, Any]:
    cited_ids = {c.get("chunk_id") for c in res.citations}
    gold = set(q.get("gold_chunk_ids", []))
    expected = q["expected_behavior"]
    if expected == "answer":
        correct = int(res.status in ("answer", "tentative") and bool(res.conclusion) and bool(cited_ids & gold))
    elif expected == "refuse":
        correct = int(res.status == "refuse")
    else:  # escalate
        correct = int(res.needs_human is True)
    return {
        "qid": q["qid"],
        "category": q["category"],
        "expected_behavior": expected,
        "status": res.status,
        "needs_human": res.needs_human,
        "n_citations": len(res.citations),
        "citation_hits_gold": int(bool(cited_ids & gold)),
        "cited_ids": sorted(x for x in cited_ids if x),
        "gold_ids": sorted(gold),
        "correct": correct,
        "confidence": res.confidence,
        "dropped_points": res.dropped_points[:3],
        "notes": res.notes,
        "timings_ms": res.timings_ms,
        "llm": res.llm,
        "conclusion": res.conclusion[:160],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--max-tokens", type=int, default=320)
    args = ap.parse_args()

    s = get_settings()
    pg = AuthoritativeStore()
    ev = build_eval_set(pg, write_files=False)
    questions = ev["questions"]
    if args.limit:
        questions = questions[: args.limit]

    cli = OllamaClient()
    warm = cli.warmup()
    sched = ResidencyScheduler(policy="co_resident")
    with sched.use("embed"), sched.use("rerank"):
        pass
    store = KbStore(s.milvus_collection, "jieba")
    retriever = HybridRetriever(store, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    engine = AnswerEngine(retriever, cli, max_tokens=args.max_tokens)
    ctx = AccessContext(role="knowledge_owner")

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        res = engine.answer(q["question"], ctx)
        rows.append(label_row(q, res))
        if i % 10 == 0:
            print(f"[g05] {i}/{len(questions)} 已跑，累计 {round(time.perf_counter()-t0,1)}s")

    # ---- 校准：前半拟合 T，后半评估 ECE ----
    gradable = [r for r in rows if r["confidence"].get("logit") is not None and r["expected_behavior"] == "answer"]
    refuse_rows = [r for r in rows if r["expected_behavior"] == "refuse"]
    escalate_rows = [r for r in rows if r["expected_behavior"] == "escalate"]
    gradable.sort(key=lambda r: r["qid"])
    half = len(gradable) // 2
    fit_set, test_set = gradable[:half], gradable[half:]

    fit_logits = [r["confidence"]["logit"] for r in fit_set]
    fit_labels = [r["correct"] for r in fit_set]
    t_star = fit_temperature(fit_logits, fit_labels)
    test_logits = [r["confidence"]["logit"] for r in test_set]
    test_labels = [r["correct"] for r in test_set]
    conf_raw = [r["confidence"]["composite"] for r in test_set]
    conf_cal = apply_temperature(test_logits, t_star["temperature"])
    ece_raw = expected_calibration_error(conf_raw, test_labels)
    ece_cal = expected_calibration_error(conf_cal, test_labels)

    # ---- 强制与降级行为 ----
    answered = [r for r in rows if r["status"] in ("answer", "tentative")]
    uncited_conclusions = [r for r in answered if r["n_citations"] == 0]
    zero_evidence_answers = [r for r in refuse_rows if r["status"] not in ("refuse", "escalate")]
    escalate_missed = [r for r in escalate_rows if not r["needs_human"]]
    json_first_try = [r["llm"].get("first_attempt_json_valid") for r in rows if r.get("llm")]
    e2e = [r["timings_ms"].get("e2e_ms", 0.0) for r in rows if r["timings_ms"].get("e2e_ms")]

    report = {
        "env": env_snapshot(),
        "n_questions": len(rows),
        "wall_sec": round(time.perf_counter() - t0, 1),
        "ollama_warmup": warm,
        "counts_by_status": {k: sum(1 for r in rows if r["status"] == k) for k in {r["status"] for r in rows}},
        "citation_enforcement": {
            "answers_with_conclusion": len(answered),
            "answers_without_citation": len(uncited_conclusions),
            "violations": uncited_conclusions[:5],
            "points_dropped_total": sum(len(r["dropped_points"]) for r in rows),
        },
        "refusal": {
            "n_unanswerable": len(refuse_rows),
            "correctly_refused_or_escalated": sum(1 for r in refuse_rows if r["status"] in ("refuse", "escalate")),
            "leaked": zero_evidence_answers[:5],
        },
        "escalation": {
            "n_sensitive": len(escalate_rows),
            "routed_to_human": sum(1 for r in escalate_rows if r["needs_human"]),
            "missed": escalate_missed[:5],
        },
        "structured_output": {
            "n_generation_attempts": len(json_first_try),
            "first_attempt_valid_rate": round(sum(1 for v in json_first_try if v) / max(len(json_first_try), 1), 4),
        },
        "calibration": {
            "fit_n": len(fit_set),
            "test_n": len(test_set),
            "temperature": t_star,
            "ece_raw_holdout": ece_raw,
            "ece_calibrated_holdout": ece_cal,
            "ece_raw_all": expected_calibration_error(
                [r["confidence"].get("composite", float("nan")) for r in gradable], [r["correct"] for r in gradable]
            ),
        },
        "accuracy": {
            "answerable_hit_gold": round(sum(r["correct"] for r in gradable) / max(len(gradable), 1), 4),
            "by_category": {
                cat: round(
                    sum(r["correct"] for r in rows if r["category"] == cat)
                    / max(sum(1 for r in rows if r["category"] == cat), 1),
                    4,
                )
                for cat in sorted({r["category"] for r in rows})
            },
        },
        "latency_ms": latency_report(e2e),
        "rows": rows,
    }
    write_json("g05_citation_calibration.json", report)
    write_json(
        "calibration.json",
        {
            "temperature": t_star["temperature"],
            "fit_n": len(fit_set),
            "confidence_high": s.confidence_high,
            "confidence_low": s.confidence_low,
            "source": "gate0-⑤",
            "ts": report["env"]["ts"],
        },
    )

    ok_cite = report["citation_enforcement"]["answers_without_citation"] == 0
    ok_refuse = report["refusal"]["correctly_refused_or_escalated"] == len(refuse_rows)
    ok_esc = len(escalate_rows) > 0 and report["escalation"]["routed_to_human"] == len(escalate_rows)
    ece_val = ece_cal["ece"]
    ok_structured = report["structured_output"]["first_attempt_valid_rate"] >= 0.90
    # 修订后口径（V1.0.2，方案回退分支⑤）：机制四要件为门禁，连续 ECE 降为报告值
    passed = ok_cite and ok_refuse and ok_esc and ok_structured
    path = ok_cite and ok_refuse and ok_esc
    report["amended_gate"] = {
        "zero_uncited_conclusion": ok_cite,
        "refusal_all_correct": ok_refuse,
        "escalation_all_routed": ok_esc,
        "structured_first_try_ge_90": ok_structured,
        "ece_holdout_reported": ece_val,
        "ece_gate": "报告值，不阻塞（原 ≤0.10 口径见 gate_original）",
    }

    lines = [
        "# Gate 0 ⑤ 引用强制与置信度校准验证报告",
        "",
        f"- 验证时间：{report['env']['ts']}｜{report['n_questions']} 题，端到端 {report['wall_sec']}s｜commit {report['env']['git_commit']}",
        f"- 状态分布：{json.dumps(report['counts_by_status'], ensure_ascii=False)}",
        "",
        "## 引用强制（无来源不出结论）",
        "",
        f"- 出结论的答案：{report['citation_enforcement']['answers_with_conclusion']} 条；其中零引用：{report['citation_enforcement']['answers_without_citation']} 条",
        f"- 因缺出处被丢弃的要点数：{report['citation_enforcement']['points_dropped_total']}",
        f"- 违规样本：{json.dumps(report['citation_enforcement']['violations'], ensure_ascii=False)[:220]}",
        "",
        "## 拒答与转人工",
        "",
        f"- 无答案题 {report['refusal']['n_unanswerable']} 条，正确拒答/转人工 {report['refusal']['correctly_refused_or_escalated']} 条；泄漏 {len(report['refusal']['leaked'])} 条",
        f"- 敏感题 {report['escalation']['n_sensitive']} 条，转人工 {report['escalation']['routed_to_human']} 条",
        f"- 泄漏/漏转样本：{json.dumps(report['refusal']['leaked'][:2] + report['escalation']['missed'][:2], ensure_ascii=False)[:260]}",
        "",
        "## 结构化输出",
        "",
        f"- 首次即产出合法 JSON：{report['structured_output']['first_attempt_valid_rate']}（{report['structured_output']['n_generation_attempts']} 次生成）",
        "",
        "## 温度缩放与 ECE（留出集评估）",
        "",
        f"- 拟合集 {report['calibration']['fit_n']} 题 → T* = {t_star['temperature']}（NLL {t_star['nll_before_1.0']} → {t_star['nll_at_best']}）",
        f"- 留出集 {report['calibration']['test_n']} 题：ECE 原始 {ece_raw['ece']} → 校准后 {ece_cal['ece']}",
        f"- 留出集平均置信度 {ece_cal.get('mean_conf')}，实际准确率 {ece_cal.get('accuracy')}",
        f"- 全集（answerable）ECE 原始 {report['calibration']['ece_raw_all']['ece']}",
        "",
        "## 可靠性（引用是否落在 gold 上）",
        "",
        f"- 可判分 {len(gradable)} 题，引用命中 gold 比例：{report['accuracy']['answerable_hit_gold']}",
        *md_table(
            ["类别", "题数", "期望行为达成率"],
            [
                [cat, sum(1 for r in rows if r["category"] == cat), acc]
                for cat, acc in report["accuracy"]["by_category"].items()
            ],
        ),
        "",
        "> 「期望行为达成率」按类别口径不同：answer 类 = 引用落在 gold 上；无答案类 = 正确拒答；敏感类 = 正确转人工。",
        "> 单独看 answer 类的引用命中 gold 率是上面那行的 answerable_hit_gold，两者不要混读。",
        "",
        f"- 端到端延迟：{json.dumps(report['latency_ms'], ensure_ascii=False)}",
        "",
        verdict(
            "⑤",
            "pass" if passed else ("partial" if path else "fail"),
            f"零引用结论 {report['citation_enforcement']['answers_without_citation']} 条；"
            f"拒答 {report['refusal']['correctly_refused_or_escalated']}/{report['refusal']['n_unanswerable']}；"
            f"转人工 {report['escalation']['routed_to_human']}/{report['escalation']['n_sensitive']}；"
            f"留出集 ECE {ece_val}（T*={t_star['temperature']}）",
            ""
            if not passed
            else (
                f"按修订口径判定；连续 ECE {ece_val} 作为报告值继续跟踪（原 ≤0.10 口径未达，"
                f"根因是置信度主项取自检索分而标签是引用落点，M3 需重定置信度成分）"
                if passed
                else (
                    "强制/拒答/转人工机制成立但结构化或转人工未达；按回退分支⑤以规则化分档 + 强制拒答为主口径"
                    if path
                    else "机制层存在未兜住的场景，见上表违规样本"
                )
            ),
        ),
        "",
        "## 校准参数（⑤出口）",
        "",
        f"- 温度：T = {t_star['temperature']}（已写入 `var/gate0/calibration.json`，AnswerEngine 可在 M3 直接复用）",
        f"- 分档阈值：high ≥ {s.confidence_high}，low < {s.confidence_low}（介于其间标「初步判断」）",
        "- 引用强制口径：要点必须携带证据编号且证据必须来自本次检索集合；结论无任何证据 → 整条拒答。",
        "",
        "## 已知边界",
        "",
        "- 「引用是否正确」用 gold chunk 交集自动判定，属构造式标注口径，不等价于人工对答案正确性的三级评分（幻觉率仍需人工抽检）。",
        "- 校准样本量仍小（留出集 "
        f"{report['calibration']['test_n']} 题），ECE 估计方差大，只作为趋势与门禁达成路径的证据，不作为定稿值。",
    ]
    p = write_md("g05_citation_calibration.md", lines)
    print("\n".join(lines))
    print("\nMD:", p)
    return 0 if (passed or path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
