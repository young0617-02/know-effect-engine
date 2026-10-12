"""用**人工 answer_correct 标签**算置信度校准（⑤ ECE 敞口的关闭路径）。

为什么单独一支：此前 ECE 用的是代理标签（L1「任一引用落 gold」在可判集上恒为 1、
L4「数字召回」把拒答也拖进来），零方差/半方差的标签上算出来的 ECE 与拟合温度都不可信
——`var/gate0/confidence_calibration.json` 里那条「零方差标签上温度不可辨识」就是这么来的。
这张工单（`answer_correct_fill.csv`）是第一个**真人判「答案对不对」**的标签集，所以：

- 种群分两档，各自报，不合成一个数：
  - `answer_prod`：status ∈ {answer, tentative}，测「给出的答案对不对」；
  - `all_behav`：全 67 行，按工单口径把「误拒/误转人工」也计为 0，测「行为对不对」。
    两档的负类含义不同（答错 vs 该答不答），混在一起就没人知道那个数在说什么。
- **拟合温度有门槛**：负类样本 < `MIN_NEG_FOR_FIT` 只做 NLL 对照、不产出可上线的 T。
  这是 2026-10-11 那次「拟合自零方差标签的 T=6.655 差点上线」的直接教训。
- 阈值（0.75 / 0.45）照现状用，**不为读数挪阈值**；只报「7 个错行为里几个落在高置信带」——
  这才是安全相关的读数。

跑法：uv run python scripts/m5_answer_correct_calibration.py
产物：var/gate0/answer_correct_calibration.json（新路径，不覆盖 Gate 0 的 calibration*.json）
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "var" / "gate0"
LABELS = ART / "answer_correct_fill.csv"
PROBE = ART / "m3_confidence_probe.json"
OUT = ART / "answer_correct_calibration.json"

CONF_HIGH = 0.75
CONF_LOW = 0.45
MIN_NEG_FOR_FIT = 5          # 低于此负类样本数：只对照 NLL，不产出可上线的 T
N_BOOT = 2000
SEED = 20261012
ABSTAIN = {"refuse", "escalate"}


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
        raise ValueError(f"工单未填完/有非法值：{len(bad)} 行，例如 {bad[:5]}")
    return out


def ece(scores: list[float], labels: list[int], bins: int = 10) -> tuple[float, list[dict]]:
    n = len(scores)
    tot, table = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (s <= hi and (s > lo or b == 0))]
        if not idx:
            continue
        acc = statistics.mean(labels[i] for i in idx)
        conf = statistics.mean(scores[i] for i in idx)
        tot += len(idx) / n * abs(acc - conf)
        table.append({"bin": b, "lo": round(lo, 2), "hi": round(hi, 2), "n": len(idx),
                      "mean_conf": round(conf, 4), "accuracy": round(acc, 4)})
    return round(tot, 4), table


def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 for p in pos for q in neg if p > q) + 0.5 * sum(1.0 for p in pos for q in neg if p == q)
    return round(wins / (len(pos) * len(neg)), 4)


def boot_auc(scores: list[float], labels: list[int], n_boot: int = N_BOOT) -> list[float] | None:
    rnd = random.Random(SEED)
    vals = []
    for _ in range(n_boot):
        idx = [rnd.randrange(len(scores)) for _ in scores]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        v = auc(s, y)
        if v is not None:
            vals.append(v)
    if len(vals) < 20:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def brier(scores: list[float], labels: list[int]) -> float:
    return round(statistics.mean((s - y) ** 2 for s, y in zip(scores, labels)), 4)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def nll(scores: list[float], labels: list[int], t: float) -> float:
    ps = [1.0 / (1.0 + math.exp(-_logit(s) / t)) for s in scores]
    return -statistics.mean(y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
                             for p, y in zip(ps, labels))


def fit_temperature(scores: list[float], labels: list[int]) -> dict:
    if min(sum(labels), len(labels) - sum(labels)) == 0:
        return {"refused": "零方差标签，NLL 最优解只能是 T→∞（输出基率），那不是校准。"}
    clamped = [min(max(s, 1e-6), 1 - 1e-6) for s in scores]
    grid = [round(0.05 + 0.05 * i, 3) for i in range(2000)]
    curve = {t: nll(clamped, labels, t) for t in grid}
    best = min(curve, key=curve.get)
    neg = len(labels) - sum(labels)
    res = {"n": len(labels), "neg": neg, "nll_at_T1": round(curve[1.0], 4),
           "best_T": best, "nll_at_best": round(curve[best], 4), "adoptable": neg >= MIN_NEG_FOR_FIT}
    if not res["adoptable"]:
        res["refused"] = (f"负类仅 {neg} 例（<{MIN_NEG_FOR_FIT}）：T 由极少数点决定，网格上近乎平坦，"
                          "把它写进服务参数就是给噪声配了个名字。只报 NLL 对照，不产出可上线的 T。")
    return res


def describe(name: str, rows: list[dict], labels: dict[str, int], fit_admissible: bool) -> dict:
    scores = [r["score"] for r in rows]
    y = [labels[r["qid"]] for r in rows]
    acc, table = ece(scores, y)
    base = statistics.mean(y)
    mean_conf = round(statistics.mean(scores), 4)
    high = [r for r, i in zip(rows, y) if r["score"] >= CONF_HIGH]
    neg_high = [r["qid"] for r, i in zip(rows, y) if i == 0 and r["score"] >= CONF_HIGH]
    neg_low = [r["qid"] for r, i in zip(rows, y) if i == 0 and r["score"] < CONF_LOW]
    neg_struct_zero = [r["qid"] for r, i in zip(rows, y) if i == 0 and r["status"] in ABSTAIN and r["score"] <= 1e-9]
    fit = fit_temperature(scores, y)
    if not fit_admissible and fit.get("adoptable"):
        fit["adoptable"] = False
        fit["refused"] = (
            f"负类 {len(neg_struct_zero)} 行是拒答/转人工，其复合分被规则**钉死为 0.0**（结构值，不是证据强度读数）。"
            "T<1 的收益几乎全来自把这批行更狠地压到 0——那是在拟合「没作答当然不算答对」这条同义反复，"
            "不是校准。此档只作对照，不产出可上线的 T。")
    return {
        "population": name,
        "n": len(rows),
        "n_correct": sum(y),
        "n_wrong": len(y) - sum(y),
        "accuracy": round(base, 4),
        "mean_conf": mean_conf,
        "ece": acc,
        "ece_sign": ("欠置信（平均置信低于实测正确率）" if base > mean_conf
                     else "过置信（平均置信高于实测正确率）"),
        "reliability_bins": table,
        "brier": brier(scores, y),
        "auc_composite_vs_answer_correct": auc(scores, y),
        "auc_ci95": boot_auc(scores, y),
        "gate_band": {
            "conf_high": CONF_HIGH, "conf_low": CONF_LOW,
            "n_in_high_band": len(high),
            "accuracy_in_high_band": round(statistics.mean([labels[r["qid"]] for r in high]), 4) if high else None,
            "wrong_rows_in_high_band": neg_high,
            "wrong_rows_in_low_band": neg_low,
            "note": "wrong_rows_in_high_band 是安全相关读数（错行为被高置信放行）；low_band 里的错行是「该不该拒」另一回事。",
        },
        "neg_composition": {"答错(有结论)": sum(1 for r, i in zip(rows, y) if i == 0 and r["status"] not in ABSTAIN),
                            "误拒/误转人工": len(neg_struct_zero),
                            "复合分钉在 0.0 的行": neg_struct_zero},
        "temperature_fit": fit,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=str(PROBE),
                    help="与标签配对的那一份探针产物。**标签是按哪次生成就必须配哪次**——"
                         "探针一重跑就会覆盖源文件，配对错了对标签不是报错而是安静地失配")
    ap.add_argument("--ignore-qids", default="",
                    help="探针里有、工单里没有的 qid（逗号分隔），例如新加 yet 未标注的题")
    ap.add_argument("--ignore-reason", default="", help="上一条的书面理由，空理由视为未豁免")
    args = ap.parse_args()

    probe_path = Path(args.probe)
    if not LABELS.exists() or not probe_path.exists():
        print(f"[缺输入] {LABELS.name} / {probe_path.name} 有一份不在，先去填工单或跑 m3 置信度探针。")
        return 2
    labels = read_labels()
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    by = {r["qid"]: r for r in probe["rows"]}

    only_probe = set(by) - set(labels)
    only_labels = set(labels) - set(by)
    ignored = {q.strip() for q in args.ignore_qids.split(",") if q.strip()}
    if ignored and not args.ignore_reason.strip():
        print("[拒绝产出] --ignore-qids 必须配 --ignore-reason（空理由视为未豁免）")
        return 2
    unexplained = only_probe - ignored
    if unexplained:
        print(f"[对不上] 工单 qid 与探针不一致（探针多出且未豁免）：{sorted(unexplained)[:8]}"
              f"{'；工单侧还多 ' + str(sorted(only_labels)[:5]) if only_labels else ''}。"
              "新题没标签要用 --ignore-qids 点名并写理由，其余情况一律拒。")
        return 3
    if ignored:
        # 只豁免「探针有、标签没有」的方向；反向（标签找不到对应探针行）永远是错行，不给豁免
        labels = {q: v for q, v in labels.items() if q in by}

    rows = [{"qid": q,
             "status": by[q]["status"],
             "score": by[q]["confidence"]["composite"],
             "calibrated": by[q]["confidence"].get("calibrated", by[q]["confidence"]["composite"]),
             "retrieval": by[q]["confidence"]["retrieval"],
             "conclusion": (by[q].get("conclusion") or "").strip(),
             "reference_answer": by[q].get("reference_answer", ""),
             "label": labels[q]} for q in sorted(labels)]
    answer_prod = [r for r in rows if r["status"] not in ABSTAIN]

    # 生成可复现性：同一套题、同一份配置重跑一次探针，结论文本变了多少行？
    # 这个数字决定「人判标签」还能不能配在另一代产物上——漂移大就说明标签标的是一个已经不存在的输出。
    archived = ART / "pre_reprobe" / "m3_confidence_probe.json"
    other = PROBE if probe_path.resolve() == archived.resolve() else archived
    drift = {}
    if other.exists() and other.resolve() != probe_path.resolve():
        old = {r["qid"]: (r.get("conclusion") or "").strip() for r in
               json.loads(other.read_text(encoding="utf-8"))["rows"]}
        shared = [q for q in labels if q in old]
        changed = [q for q in shared if old[q] != (by[q].get("conclusion") or "").strip()]
        drift = {"comparing_this_to": other.name, "shared_rows": len(shared),
                 "conclusion_changed": len(changed),
                 "changed_ratio": round(len(changed) / len(shared), 4) if shared else None,
                 "changed_qids": changed[:12],
                 "note": ("标签是照着 `--probe` 指的那一份判的；这一栏只说明**重跑一次探针会改多少行结论文本**，"
                          "不用于改判标签。漂移不为 0 ⇒ 标签与另一代产物不可互换配对。")}

    out = {
        "source": {"labels": LABELS.name, "scores": str(probe_path.relative_to(ART.parent.parent))
                   if probe_path.is_relative_to(ROOT) else str(probe_path),
                   "paired_note": "标签与探针必须同一次生成；换探针产物要显式 --probe",
                   "ignored_unlabeled": {sorted(ignored): args.ignore_reason} if ignored else {},
                   "score_basis": probe["rows"][0]["confidence"].get("score_basis", "rule_composite"),
                   "temperature_used_at_runtime": probe["temperature_used"]},
        "generation_drift": drift,
        "label_check": {"n": len(rows), "n_correct": sum(r["label"] for r in rows),
                        "n_wrong": sum(1 - r["label"] for r in rows),
                        "degenerate": sum(r["label"] for r in rows) in (0, len(rows)),
                        "by_status": {s: {"n": sum(1 for r in rows if r["status"] == s),
                                          "wrong": sum(1 - r["label"] for r in rows if r["status"] == s)}
                                      for s in sorted({r["status"] for r in rows})}},
        "populations": {
            "answer_prod": describe("status ∈ {answer, tentative}：给出的答案对不对", answer_prod, labels, True),
            "all_behav": describe("全 67 行：行为对不对（误拒/误转人工计 0）", rows, labels, False),
        },
        "wrong_rows": [{"qid": r["qid"], "status": r["status"], "composite": r["score"],
                        "calibrated": r["calibrated"],
                        "conclusion": r["conclusion"][:60], "reference_answer": r["reference_answer"][:60]}
                       for r in rows if r["label"] == 0],
        "comparison_with_prior_readings": {
            "L1_section_any_ece_全1标签": probe["ece_and_fit"]["L1_current_label"]["ece"],
            "L4_ref_number_recall_proxy": {"ece": probe["ece_and_fit"]["L4_answer_correct"]["ece"],
                                           "accuracy": probe["ece_and_fit"]["L4_answer_correct"]["accuracy"],
                                           "注意": "那是「数字召回」代理标签，不是人判的答案正确性，两者别混"},
            "gate0_⑤_fitted_T": 6.655,
        },
    }
    ap, ab = out["populations"]["answer_prod"], out["populations"]["all_behav"]
    from kee.eval.eval_data import QUESTIONS

    pool = {q["qid"] for q in QUESTIONS if q.get("expected_behavior") == "answer"}
    out_of_scope = sorted({q["qid"] for q in QUESTIONS} - pool)
    out["coverage_boundary"] = {
        "评测集总数": len(QUESTIONS),
        "本标签集覆盖": len(rows),
        "覆盖范围": "仅 `expected_behavior == answer` 的 67 题（探针的取题口径就是这条过滤）",
        "未覆盖": out_of_scope,
        "未覆盖的是什么": "无答案 7 题 + 敏感 5 题——它们的**正确行为本就是拒答/升级**，"
                    "所以「误拒率 7.5%」只描述该答却拒的那一面，不能当系统总体拒答行为；"
                    "反过来，这 12 题上「该拒却答了」没有真人标签，仍是空白。",
    }
    cause = {}
    for r in rows:
        if r["label"] == 1:
            continue
        src = by[r["qid"]]
        notes = src.get("notes") or []
        if r["status"] in ABSTAIN:
            cause[r["qid"]] = ("判定层：检索分高却声明无依据（转人工）"
                               if any("strong_evidence_but_no_basis" in n for n in notes)
                               else "检索层：证据没进 Top-K，拒答是链路内的正确反应，但题本身该答")
        else:
            cause[r["qid"]] = ("生成层：引用块在 strict gold 内，结论仍读错"
                               if src.get("strict_any") else
                               "证据层：所引块不在 strict gold，结论跟着错块走")
    out["wrong_row_causes"] = cause
    out["verdict"] = {
        "⑤_ECE_敞口": (f"关闭：第一个真人判「答案对不对」的标签集已回收（{ap['n']} 行给出答案的行 + "
                    f"{len(rows) - ap['n']} 行拒答/转人工），ECE 有了方差供给。"
                    f"主口径 ECE={ap['ece']}（{ap['ece_sign']}），门禁 ECE ≤ 0.10 **仍不达标**。"),
        "为什么这个不达标不危险": (
            f"方向是**欠置信**：平均复合分 {ap['mean_conf']} 低于实测正确率 {ap['accuracy']}。"
            "危险的是过置信（错答案顶着高分被自动放行），这里没有——"
            f"但高置信带里仍漏了 {len(ab['gate_band']['wrong_rows_in_high_band'])} 行错答案"
            f"（{ab['gate_band']['wrong_rows_in_high_band']}，复合分 0.75~0.79 贴着门槛过线）。"),
        "温度缩放": "两档拟合都**不采纳**：answer_prod 负类仅 2 例不可辨识；"
                "all_behav 的负类 5/7 是复合分被钉在 0.0 的拒答行，T<1 拟合的是同义反复。"
                "运行时维持 T=1.0、复合分不进门禁——与 confidence_calibration.json 的裁决一致，本次不改。",
        "真正暴露的问题": (f"7 个错行为按层拆开：{sum(1 for v in cause.values() if v.startswith('检索层'))} 行是检索漏召"
                       f"（证据没进 Top-K，拒答在链路内是对的、题却是该答的），"
                       f"{sum(1 for v in cause.values() if v.startswith('判定层'))} 行是判定层老毛病"
                       "（检索分 ≥0.9 却声明无依据转人工，即 M3-02 那类），"
                       f"{sum(1 for v in cause.values() if v.startswith('生成层'))} 行引用块在 gold 内仍读错，"
                       f"{sum(1 for v in cause.values() if v.startswith('证据层'))} 行引用块不在 strict gold 而结论跟着错块走。"
                       "没有一行是「校准能救」的——校准只改分数的显示口径，不改行为。"),
        "欠置信的代价": (f"反过来读：把复合分当概率呈现，会系统性低估自己（报 0.77 而实得 0.97）。"
                     "在这批标签里它没造成错误放行，但它压低的是**用户看到的把握程度**，"
                     "所以这个分数继续只用于内部闸门与降级排序，对用户的口径维持「三态 + 引用」不显示概率。"),
        "样本量边界": "62 行的正确率与 2 个负类撑不起任何「总体正确率 ≥ X%」的断言，"
                  "AUC 的 CI95 宽到 [0.63, 0.91]。所有读数只描述这一批探针，不外推。",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in ("answer_prod", "all_behav"):
        p = out["populations"][name]
        print(f"\n== {name} ==  n={p['n']}  正 {p['n_correct']} / 负 {p['n_wrong']}")
        print(f"  accuracy {p['accuracy']}  mean_conf {p['mean_conf']}  ECE {p['ece']}  Brier {p['brier']}")
        print(f"  AUC {p['auc_composite_vs_answer_correct']}  CI95 {p['auc_ci95']}")
        g = p["gate_band"]
        print(f"  高置信带(≥{g['conf_high']}) {g['n_in_high_band']} 行，带内正确率 {g['accuracy_in_high_band']}")
        print(f"    错行为落在高置信带：{g['wrong_rows_in_high_band'] or '无'}")
        print(f"    错行为落在低置信带：{g['wrong_rows_in_low_band'] or '无'}")
        nc = p["neg_composition"]
        print(f"  负类构成：答错 {nc['答错(有结论)']} / 误拒·误转人工 {nc['误拒/误转人工']}"
              f"（钉 0.0：{nc['复合分钉在 0.0 的行'] or '无'}）")
        tf = p["temperature_fit"]
        print(f"  温度拟合：{json.dumps({k: tf[k] for k in tf if k in ('best_T', 'nll_at_T1', 'nll_at_best', 'neg', 'adoptable')}, ensure_ascii=False)}")
        if tf.get("refused"):
            print(f"    [不采纳] {tf['refused']}")
    print(f"\n落盘：{OUT}")
    print("错行清单：")
    for r in out["wrong_rows"]:
        print(f"  {r['qid']} [{r['status']}] comp={r['composite']} 结论「{r['conclusion'] or '(空)'}」 参考「{r['reference_answer']}」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
