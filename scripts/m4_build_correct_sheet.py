"""生成「答案正确性」工单（#40）：每题一行，判**结论对不对**，不是判块相关。

为什么单独一张：⑤ 的 ECE 敞口一直没关，根因是标签零方差——复合分要校准，
需要的是**有方差的正确性标签**，而块级相关性工单（`annotations_a2_fill.csv`）测的不是这件事。

工单形状（照 V1.10 定的口径）：
- 源 = `var/gate0/m3_confidence_probe.json` 的 `conclusion` + `cited_ids`（线上那次真实答案）；
- **不显示 `reference_answer`**——显示出来就是抄答案，标注不再是独立判断；
- 每行填 `1/0`：这条结论相对于它自己引用的证据是否成立；
- 允许留空/填 `?`（判不了就明说，不许凑数）。

跑法：uv run python scripts/m4_build_correct_sheet.py [--force]
产物：var/gate0/answer_correct_fill.csv（待填）+ answer_correct_key.json（含 reference，**标注后才可看**）
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "var" / "gate0"
SRC = ART / "m3_confidence_probe.json"
FILL = ART / "answer_correct_fill.csv"
KEY = ART / "answer_correct_key.json"

COLS = ["qid", "category", "status", "question", "conclusion", "n_citations", "cited_ids", "answer_correct__fill_1_or_0"]
JUDGE_LINE = (
    "# 判据：只判「这条 conclusion 相对于它自己引用的证据是否成立」——1=成立，0=不成立，?=判不了。"
    "**不给参考答案**（给了就是抄）。不看措辞是否漂亮，看结论里的数字/口径/主体在不在引用块里、有没有反着说。"
    "拒答/转人工的行（status=refuse|escalate 且 conclusion 为空）按「该拒吗」判：该拒=1，不该拒=0。"
)


def guard_existing_fill(force: bool) -> str | None:
    if not FILL.exists() or force:
        return None
    with FILL.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    filled = [r for r in rows if (r.get("answer_correct__fill_1_or_0") or "").strip() not in ("", None)]
    if not filled:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = FILL.with_suffix(f".filled-{stamp}.csv")
    shutil.copyfile(FILL, bak)
    return (
        f"已有 {len(filled)}/{len(rows)} 行填过。为避免覆盖已填标注，本次拒绝写入；"
        f"已把旧表复制到 {bak.name}。确认要重来请加 --force（旧表已在上面那份里）。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="确认覆盖已填的工单（会先复制留底）")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"[缺源] 找不到 {SRC}，先跑 m3 置信度探针。")
        return 2
    rows = json.loads(SRC.read_text(encoding="utf-8"))["rows"]
    blocked = guard_existing_fill(args.force)
    if blocked:
        print(f"[拒绝覆盖] {blocked}")
        return 4

    with FILL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([JUDGE_LINE])
        w.writerow([f"# 生成时点 {datetime.now().isoformat(timespec='seconds')}｜源 {SRC.name}｜共 {len(rows)} 行"])
        w.writerow(COLS)
        for r in rows:
            concl = (r.get("conclusion") or "").strip()
            w.writerow([
                r["qid"], r.get("category", ""), r.get("status", ""), r.get("question", ""),
                concl, r.get("n_citations", len(r.get("cited_ids") or [])),
                " | ".join(r.get("cited_ids") or []), "",
            ])

    # 答案卷单独存：标注人阶段看不到，收回来再算一致性
    KEY.write_text(
        json.dumps(
            {"generated_at": datetime.now().isoformat(timespec="seconds"),
             "note": "含 reference_answer，**只在标注回收之后**才能打开比对",
             "rows": [{"qid": r["qid"], "conclusion": r.get("conclusion", ""),
                       "reference_answer": r.get("reference_answer", ""),
                       "cited_ids": r.get("cited_ids", []), "status": r.get("status")} for r in rows]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    empty_concl = sum(1 for r in rows if not (r.get("conclusion") or "").strip())
    print(f"工单：{FILL}")
    print(f"  {len(rows)} 行（每题一行）；其中空结论/拒答类 {empty_concl} 行按「该不该拒」判")
    print(f"  答案卷：{KEY}（标注回收前不要打开）")
    print(f"  提醒：这张测的是 answer_correct，不是块级相关性；两张表的判据不同，别混用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
