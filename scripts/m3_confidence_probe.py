"""M3-04a 诊断：置信度到底能不能预测任何东西——先量标签方差，再谈拟合。

现状的根子问题：`m3_answer_probe` 报的 ECE 0.2422 是在**全 1 标签**上算的
（53 行 `citation_hits_gold` 全为 1），于是 ECE 退化成「mean_conf 与 1.0 的差」，
它测的是我们比完美欠了多少自信，**不是答案对不对**。
在这种标签上拟合温度只会往「全体更自信」的方向走——label 全 1 时 NLL 的单调解就是最小 T，
M3 那次 T=0.05 的退化拟合就是这么来的。所以顺序必须是：先证明标签有方差，才有资格谈校准。

四个候选标签，全部由评测集既有字段确定性算出，不引入人工判断：
  L1 `section_any`       任一条引用 ∈ section 级 gold（**现在用的这个**，预期饱和）
  L2 `strict_any`        任一条引用 ∈ strict gold（正文自身含答案关键词那层）
  L3 `strict_precision`  引用里落在 strict gold 的**比例**（条目级；row 级 max 会把不一致抹平）
  L4 `ref_number_recall` reference_answer 里的数字有多少出现在结论里——「答对了没有」的代理。
                         数字是这批题的承重内容（RTO 15 分钟 / 拨备 150% / 214 项 / 800 毫秒），
                         q-059 那种编造「2024年12月20日、93.7%」正好会被它抓到。
                         ref 里没有阿拉伯数字的行标 not_assessable，**不算对也不算错**。

再量**推理期也算得出来的候选成分**（不需要 gold，所以它们才有资格进置信度公式）：
  `number_groundedness` 结论里的数字有多少**真的出现在被引用的证据正文里**——
                        L4 的无标准答案版，且天然就是编造检测器。
  `n_citations`         引用条数（支持面越宽越可信？实测说话，不预设结论）
  `evidence_number_count` 被引证据里数字的个数（证据含不含可搬的硬事实）

n=58 且语料是模拟的，AUC 的区间会很宽——这里只用来判「有没有方向」，不用来给方案排名。

用法：
  uv run python scripts/m3_confidence_probe.py [--limit N]      # 跑生成 + 分析
  uv run python scripts/m3_confidence_probe.py --analyze-only   # 复用已存证据只重算分析
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import expected_calibration_error, fit_temperature  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa.answer import AnswerEngine, load_calibration_temperature  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

from m2_eval import bootstrap_ci  # noqa: E402

_NUM = re.compile(r"\d+(?:\.\d+)?")
CURRENT_COMPONENTS = ("retrieval", "citation_consistency", "self_assessed", "composite", "calibrated")
RUNTIME_SIGNALS = ("number_groundedness", "n_citations", "evidence_number_count")


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUC，并列记 0.5。两类齐全才有定义，否则返回 None——不硬凑 0.5 或 1.0。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = ties = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(text or ""))


def label_row(q: dict, res) -> dict:
    """四个标签 + 后续复算所需的原始材料。引用取**实际被引用的 chunk**，不是候选集。"""
    cited = [c.get("chunk_id") for c in res.citations]
    section, strict = set(q["gold_chunk_ids"]), set(q["gold_strict_ids"])
    ref_nums = numbers(str(q.get("reference_answer") or ""))
    conc_nums = numbers(res.conclusion)
    assessable = bool(ref_nums)
    return {
        "cited_ids": cited,
        "n_citations": len(cited),
        "section_any": int(any(c in section for c in cited)) if cited else 0,
        "strict_any": int(any(c in strict for c in cited)) if cited else 0,
        "strict_precision": round(sum(1 for c in cited if c in strict) / len(cited), 4) if cited else 0.0,
        "ref_number_recall": round(len(ref_nums & conc_nums) / len(ref_nums), 4) if assessable else None,
        # 编造面：结论里出现了 ref 没有的数字。刻意**不**豁免「问题里已有的数字」——
        # 把问题里的数字原样搬进结论同样是一种「没从证据来」的输出，该一起看。
        "invented_numbers": sorted(conc_nums - ref_nums) if assessable else [],
        "assessable_by_numbers": assessable,
    }


def add_runtime_signals(rows: list[dict], texts: dict[str, str]) -> None:
    """补上推理期也算得出的成分：只需要被引证据正文，不需要 gold。"""
    for r in rows:
        ev = " ".join(texts.get(c, "") for c in r.get("cited_ids") or [])
        ev_nums, conc_nums = numbers(ev), numbers(r.get("conclusion"))
        r["number_groundedness"] = round(len(conc_nums & ev_nums) / len(conc_nums), 4) if conc_nums else None
        r["evidence_number_count"] = len(ev_nums)


def var_report(rows: list[dict], key: str, binary: bool) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    if binary:
        ones = sum(1 for v in vals if v == 1)
        return {"n": len(vals), "ones": ones, "zeros": len(vals) - ones,
                "mean": round(ones / len(vals), 4), "degenerate": ones == 0 or ones == len(vals)}
    return {"n": len(vals), "mean": round(st.mean(vals), 4),
            "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "distinct": len(set(vals)), "degenerate": len(set(vals)) == 1}


def analyze(rows: list[dict], temperature_used: float) -> dict:
    grad = [r for r in rows if r["confidence"].get("logit") is not None
            and r["status"] in ("answer", "tentative")]
    l4 = [r for r in grad if r.get("ref_number_recall") is not None]
    l4_y = [1 if r["ref_number_recall"] >= 1.0 else 0 for r in l4]

    def auc_on(src: str) -> float | None:
        xs = [r[src] if src in r else r["confidence"].get(src) for r in l4]
        xs = [float(x) for x in xs if x is not None]
        if len(xs) != len(l4) or len(set(xs)) == 1:
            return None
        return auc(xs, l4_y)

    dist = {}
    for c in CURRENT_COMPONENTS:
        vals = [float(r["confidence"][c]) for r in grad if r["confidence"].get(c) is not None]
        dist[c] = {"distinct": len(set(vals)), "min": round(min(vals), 4) if vals else None,
                   "max": round(max(vals), 4) if vals else None,
                   "mean": round(st.mean(vals), 4) if vals else None, "constant": len(set(vals)) == 1}

    cal = [float(r["confidence"]["calibrated"]) for r in grad]
    l4_cal = [float(r["confidence"]["calibrated"]) for r in l4]
    return {
        "n_run": len(rows),
        "n_gradable": len(grad),
        "n_l4_assessable": len(l4),
        "l4_positives": sum(l4_y),
        "temperature_used": temperature_used,
        "label_variance": {
            "L1_section_any": var_report(grad, "section_any", True),
            "L2_strict_any": var_report(grad, "strict_any", True),
            "L3_strict_precision": var_report(grad, "strict_precision", False),
            "L4_ref_number_recall": var_report(l4, "ref_number_recall", False),
        },
        "component_distribution": dist,
        "auc_vs_L4": {c: auc_on(c) for c in CURRENT_COMPONENTS + RUNTIME_SIGNALS},
        "ece_and_fit": {
            "L1_current_label": {**expected_calibration_error(cal, [r["section_any"] for r in grad]),
                                 "note": "现在报告里那条 0.24 的口径：标签全 1"},
            "L4_answer_correct": {
                **expected_calibration_error(l4_cal, l4_y),
                "ci95_gap": list(bootstrap_ci([abs(c - y) for c, y in zip(l4_cal, l4_y)])),
                "fit": fit_temperature([float(r["confidence"]["logit"]) for r in l4], l4_y),
            },
        },
        "invented_number_rows": [{"qid": r["qid"], "nums": r["invented_numbers"]}
                                 for r in grad if r["invented_numbers"]],
        "rows": rows,
    }


def print_report(out: dict) -> None:
    print(f"\n=== 标签方差（可判集 n={out['n_gradable']}，其中可按数字判 {out['n_l4_assessable']}）===")
    for k, v in out["label_variance"].items():
        print(f"  {k:<22} {v}{'  ← 零方差，不可用于校准' if v.get('degenerate') else ''}")
    print("\n=== 现有成分分布（常数项只贡献固定偏移，不贡献信息）===")
    for k, v in out["component_distribution"].items():
        print(f"  {k:<22} {v}{'  ← 常数' if v['constant'] else ''}")
    print(f"\n=== 成分判别力 AUC（对 L4「ref 数字全对」，pos={out['l4_positives']}）===")
    for k, v in out["auc_vs_L4"].items():
        print(f"  {k:<26} {v if v is not None else '无定义（常数或标签退化）'}")
    print("\n=== ECE 与温度可辨识性 ===")
    for k, d in out["ece_and_fit"].items():
        line = f"  {k:<20} ece={d['ece']} acc={d.get('accuracy')} mean_conf={d.get('mean_conf')}"
        if "fit" in d:
            f = d["fit"]
            line += f" | T*={f['temperature']}"
            line += f"（拒绝：{f['refused']}）" if "refused" in f else \
                    f" nll 1.0→{f['nll_before_1.0']} best→{f['nll_at_best']}"
        print(line + (f"  {d['note']}" if d.get("note") else ""))
    if out["invented_number_rows"]:
        print("\n=== 结论里出现 ref 没有的数字（编造面）===")
        for r in out["invented_number_rows"]:
            print(f"  {r['qid']} {r['nums']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true", help="复用已存证据只重算分析，不跑模型")
    args = ap.parse_args()

    s = get_settings()
    art = Path(s.resolved_artifact_dir()) / "m3_confidence_probe.json"
    pg = AuthoritativeStore()
    with pg.connect() as conn:
        texts = {r["id"]: r["content"] for r in conn.execute("SELECT id, content FROM chunk")}

    if args.analyze_only:
        saved = json.loads(art.read_text(encoding="utf-8"))
        rows, temperature_used = saved["rows"], saved.get("temperature_used", 1.0)
    else:
        qs = build_eval_set(pg, write_files=False)["questions"]
        pool = [q for q in qs if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]]
        if args.limit:
            pool = pool[: args.limit]
        kb = KbStore(s.milvus_collection, "jieba")
        rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                             mode="hybrid", use_rerank=True)
        eng = AnswerEngine(retriever=rt, llm=OllamaClient())
        # 与 API 同源，不再直接读 Gate 0 那份 calibration.json——那正是「两套温度」的由来
        temperature_used = load_calibration_temperature(s)
        eng.calibration_temperature = temperature_used
        ctx = AccessContext(role="knowledge_owner")
        t0 = time.perf_counter()
        eng.answer("预热：配电自动化终端心跳超时阈值", ctx)
        print(f"[预热] {round(time.perf_counter() - t0, 1)}s")

        rows = []
        for i, q in enumerate(pool, start=1):
            r = eng.answer(q["question"], ctx)
            lab = label_row(q, r)
            rows.append({"qid": q["qid"], "category": q["category"], "status": r.status,
                         "question": q["question"], "conclusion": r.conclusion,
                         "reference_answer": str(q.get("reference_answer") or ""),
                         "confidence": dict(r.confidence), **lab})
            l4 = "—" if lab["ref_number_recall"] is None else f'{lab["ref_number_recall"]:.2f}'
            print(f"[{i:>2}/{len(pool)}] {q['qid']} {r.status:<10} L1={lab['section_any']} "
                  f"L2={lab['strict_any']} L3={lab['strict_precision']:.2f} L4={l4} "
                  f"编造{len(lab['invented_numbers'])} conf={r.confidence.get('calibrated')}")

    add_runtime_signals(rows, texts)
    out = analyze(rows, temperature_used)
    art.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(out)
    print(f"\n证据： {art}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
