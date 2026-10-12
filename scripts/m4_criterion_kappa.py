"""裁决前置第二半：乙口径会不会「靠负例把一致性算高」。

这是本项目已经栽过一次的那类陷阱的变体——**标签零方差时，一致率会假装得很好**。
乙（必须答全所有子问）会把两问题上的部分回答块全部改判 0；
如果这些题里剩下的正例极少，那么「两人都说 0」就会把 Kappa 抬上去，
抬的幅度与检索质量无关。这里把两种口径下的读数并排算出来，供人工裁决。

只读 var/gate0/ 下的三个工件，不开 PG、不改任何数据。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.eval.kappa import cohens_kappa  # noqa: E402

INTERROGATIVE = re.compile(r"什么|哪些|怎么|怎样|多久|多少|能不能|可不可以|是否|依据哪|哪个|哪几|从哪里|吗")
FILL_COL = "b_directly_answers__fill_1_or_0"
art = Path(get_settings().resolved_artifact_dir())

rows = list(csv.DictReader((art / "annotations_a2_fill.csv").open(encoding="utf-8-sig", newline="")))
key = {(r["qid"], int(r["cand_no"])): r
       for r in json.loads((art / "annotations_a2_key.json").read_text(encoding="utf-8"))}
parts = {json.loads(l)["qid"]: len(INTERROGATIVE.findall(json.loads(l)["question"]))
         for l in (art / "qrels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

judged = [(r["qid"], int(r["cand_no"]), int((r.get(FILL_COL) or "").strip()),
           int(key[(r["qid"], int(r["cand_no"]))]["in_a_strict_gold"]))
          for r in rows if (r.get(FILL_COL) or "").strip() in ("0", "1")]


def stats(label_fn, name: str) -> dict:
    la = [label_fn(qid, no, b, a_s)[1] for qid, no, b, a_s in judged]
    lb = [label_fn(qid, no, b, a_s)[0] for qid, no, b, a_s in judged]
    res = cohens_kappa(la, lb)
    both_yes = sum(1 for a, b in zip(la, lb) if a == 1 and b == 1)
    agree_no = sum(1 for a, b in zip(la, lb) if a == 0 and b == 0)
    tp_two = sum(1 for (qid, no, b, a_s), bb in zip(judged, lb) if bb == 1 and parts[qid] >= 2)
    out = {"口径": name, "Kappa": res["kappa"], "po": res["po"], "pe": res["pe"],
           "B正例数": sum(lb), "gold正例数": sum(la), "双正例": both_yes, "双负例": agree_no,
           "B正例落在两问题上": tp_two}
    print(f"  {name:34s} Kappa={out['Kappa']:<7} po={out['po']:<7} "
          f"正例 B={out['B正例数']:<4} gold={out['gold正例数']:<4} 双正例={out['双正例']:<4} "
          f"双负例={out['双负例']}（两问题上的 B 正例 {tp_two} 条）")
    return out


def jia(qid, no, b, a_s):  # 甲：维持第二人的判法（答到任一子问即算）
    return b, a_s


def yi(qid, no, b, a_s):
    """乙：两问题上，只有「答全所有子问」才算 1。
    第二人当时按甲判，所以他在两问题上给的 1 **都是部分回答**——这些改判 0；
    而他在两问题上判 1 且本来就在 strict gold 里的（gold 那块经原文核对确实答全，见 m4_criterion_impact），
    在乙下仍然是 1，不能一并抹零——抹了就是自己造分歧。"""
    if parts[qid] >= 2 and a_s == 0:
        return 0, a_s
    return b, a_s


print("并排读数（同 210 行、同一批 gold）：")
a = stats(jia, "甲（答任一问即 1）")
y = stats(yi, "乙（必须答全）")
print(f"\n差值：Kappa 甲 {a['Kappa']} → 乙 {y['Kappa']}"
      f"；B 正例 {a['B正例数']} → {y['B正例数']}，双正例 {a['双正例']} → {y['双正例']}"
      f"；双负例 {a['双负例']} → {y['双负例']}")
print("读法：如果乙把 Kappa 抬上去的同时**双正例大幅减少**，那这个「更一致」是负例堆出来的，")
print("     它不说明第二人和第一人更合，只说明标签方差的可供量变小了——正是零方差标签那次踩过的坑。")

# 乙口径下的另一个代价：两问题上剩下的正例有多少
tp_rows = [(q, n, b, a) for q, n, b, a in judged if parts[q] >= 2]
tp_gold = sum(a for _, _, _, a in tp_rows)
tp_b_jia = sum(b for _, _, b, _ in tp_rows)
tp_b_yi = sum(1 for q, n, b, a in tp_rows if b == 1 and a == 1)
print(f"\n两问题上的行数 {len(tp_rows)}（已判 210 行内）：strict gold 正例 {tp_gold}、"
      f"第二人甲口径给 {tp_b_jia}、乙口径只剩 {tp_b_yi}")
print(f"→ 乙把这几题的正例从 {tp_b_jia} 压到 {tp_b_yi}，一致率上升主要来自「双方都说 0」的行变多。")
