"""算第二人标注的一致性（Cohen's Kappa），双口径 + 允许半成品 + 把分歧摊开。

为什么不直接用 `kee.eval.kappa`：那个模块把 A 侧固定成 section 级 gold，
而 M2 已经证明 section 级是把错尺子（gold=13 的题 Recall@5 上限只有 5/13）。
第二人按「这块正文里到底含不含答案」判，天然接近 strict 口径，
拿 section gold 去比，算出来的分歧里**混着口径差异**，不是人在不一致。
所以这里两个口径都算，并排报出来。

三件刻意的设计：

1. **允许只标了一部分就算**。计划是先做 25 题停一次，看判据对不对。
   若必须全填完才出数，中途就无法发现问题，只能标完 63 题再返工。
   未填完时输出 `complete: false` 与进度，数字照算但明确标为「部分」。
2. **Kappa 的适用边界写死在输出里**：观测域是**候选池**（第二人只看过这些块），
   不是全库。所以这个数是「在候选池上的一致性」，不能写成「标注全库一致」。
   这句话不写，Kappa 会被当成比它实际含义更强的证据。
3. **分歧要摊开且分类**，因为两个方向的含义完全不同：
   - `A=1/B=0`：第一人多标了（section gold 的注水，M2 已怀疑的那件事）
   - `A=0/B=1`：**第一人漏标了**——这方向更危险，它意味着评测集的 gold 有缺口，
     而检索指标会把「其实答对了」的块判成错。扩样本之前这个数必须知道。

用法：uv run python scripts/m4_kappa.py [--fail-if-incomplete]
退出码：0 正常；2 一题都没填；填完且 --fail-if-incomplete 时未完成返回 1。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.eval.kappa import cohens_kappa  # noqa: E402

FILL_COL = "b_directly_answers__fill_1_or_0"
SENTINEL = "todo"


def load(art: Path) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    with (art / "annotations_a2_fill.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    key = {(r["qid"], int(r["cand_no"])): r
           for r in json.loads((art / "annotations_a2_key.json").read_text(encoding="utf-8"))}
    return rows, key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-if-incomplete", action="store_true")
    args = ap.parse_args()

    art = Path(get_settings().resolved_artifact_dir())
    rows, key = load(art)
    if not rows:
        print("没有工单，先跑 scripts/m4_build_annotation_sheet.py")
        return 2

    filled: list[tuple[str, int, int, int, int]] = []  # qid, cand_no, b, a_section, a_strict
    blank_qids: list[str] = []
    for r in rows:
        raw = (r.get(FILL_COL) or "").strip().lower()
        if not raw or raw.startswith(SENTINEL):
            blank_qids.append(r["qid"])
            continue
        if raw not in ("0", "1"):
            print(f"[判废] {r['qid']}#{r['cand_no']} 填了 {raw!r}，只接受 1 或 0")
            return 2
        k = key[(r["qid"], int(r["cand_no"]))]
        filled.append((r["qid"], int(r["cand_no"]), int(raw),
                       int(k["in_a_section_gold"]), int(k["in_a_strict_gold"])))

    if not filled:
        print("一题都没填，无法计算。建议先填前 25 题跑一次，确认判据没跑偏再续标。")
        return 2

    done_qids = {f[0] for f in filled}
    total_qids = {r["qid"] for r in rows}
    complete = not blank_qids

    out = {
        "unit": "(qid, 候选块) 的相关判定",
        "scope_note": ("观测域是**候选池**（第二人只看过这些块），不是全库。"
                       "这个 Kappa 是「在候选池上的一致性」，不得表述成全库标注一致。"),
        "n_rows_judged": len(filled),
        "n_questions_done": len(done_qids),
        "n_questions_total": len(total_qids),
        "complete": complete,
        "blank_rows": len(blank_qids),
    }

    for name, idx in (("against_section_gold", 3), ("against_strict_gold", 4)):
        la = [f[idx] for f in filled]
        lb = [f[2] for f in filled]
        res = cohens_kappa(la, lb)
        agree = sum(1 for a, b in zip(la, lb) if a == b)
        res["disagreement"] = {
            # A=1/B=0：第一人多标（section gold 注水方向）
            "a_yes_b_no": sum(1 for a, b in zip(la, lb) if a == 1 and b == 0),
            # A=0/B=1：第一人**漏标**——评测集 gold 有缺口的信号，比上一类更要紧
            "a_no_b_yes": sum(1 for a, b in zip(la, lb) if a == 0 and b == 1),
        }
        res["raw_agreement"] = round(agree / max(len(la), 1), 4)
        res["b_positive_rate"] = round(sum(lb) / max(len(lb), 1), 4)
        out[name] = res

    # 分歧样本摊开：数值只说明有没有一致性，样本才说明不一致在哪一类
    samples = []
    for qid, no, b, a_s, a_t in filled:
        if a_s == b and a_t == b:
            continue
        samples.append({"qid": qid, "cand_no": no, "b": b,
                        "a_section": a_s, "a_strict": a_t,
                        "kind": ("A 多标（strict 也不认则为注水）" if a_s == 1 and b == 0
                                 else "A 漏标（gold 有缺口）" if a_s == 0 and b == 1 else "口径差异")})
    out["disagreement_samples"] = samples[:40]
    out["kind_counts"] = dict(Counter(s["kind"] for s in samples))

    (art / "kappa.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    s_, t_ = out["against_section_gold"], out["against_strict_gold"]
    status = "（完整）" if complete else f"（未完成，还有 {out['blank_rows']} 行空着）"
    print(f"已判 {out['n_rows_judged']} 行 / 覆盖 {out['n_questions_done']}/{out['n_questions_total']} 题{status}")
    print(f"  Kappa vs section gold : {s_.get('kappa')}  po={s_.get('po')} 分歧 A多标={s_['disagreement']['a_yes_b_no']} A漏标={s_['disagreement']['a_no_b_yes']}")
    print(f"  Kappa vs strict gold  : {t_.get('kappa')}  po={t_.get('po')} 分歧 A多标={t_['disagreement']['a_yes_b_no']} A漏标={t_['disagreement']['a_no_b_yes']}")
    print(f"  分歧分类: {out['kind_counts']}")
    print("  口径提醒:", out["scope_note"])
    print(f"证据： {art / 'kappa.json'}")

    if args.fail_if_incomplete and not complete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
