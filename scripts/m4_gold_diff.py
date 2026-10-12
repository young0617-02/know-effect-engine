"""全库分歧清单：把 31 行「两人真分歧」按可自动识别的成因分类，产出 gold 待改清单。

为什么不能只报 Kappa：一致性数字告诉你「有多合」，但 gold 要改的是**具体哪几行**。
首轮的 +2 −2 是按 25 题外推的，全量实测把 110 桶从 2 行放大到 18 行——
外推错了，所以这次逐行列出，不再给估算。

自动信号只是**候选归类**，不是判定：
- `cross_industry`：题面点名单一行业而金块属另一行业（词典法，与 m4_gold_defects 同一套词）；
- `two_part`：题面有 ≥2 个疑问标记（乙口径下 B 的 1 已被机械归 0，这里只用于解释）；
- `descriptive`：金块正文没有任何带强制性的词（必须/不得/只能/一律/须/禁止）——描述型的信号。
最终删/留要人工读，脚本不替人裁。

只读本地工件（CSV / key / qrels），不开 PG。输出 var/gate0/gold_diff.json。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402

INTERROGATIVE = re.compile(r"什么|哪些|怎么|怎样|多久|多少|能不能|可不可以|是否|依据哪|哪个|哪几|从哪里|吗")
MODAL = re.compile(r"必须|不得|只能|一律|禁止|应当|须")
INDUSTRY = {
    "fin": ("金融", "清算", "资金", "日切", "对账", "信用损失", "报表", "IFRS", "银企"),
    "ene": ("电力", "能源", "电网", "调度", "厂站", "SCADA", "终端", "台区", "线损",
            "配电", "分布式电源", "用电信息"),
    "chm": ("化工", "装置", "联锁", "SIS", "防爆", "批次", "炼化", "催化剂", "GMP",
            "药品", "工艺", "重整"),
}
FILL_COL = "b_directly_answers__fill_1_or_0"
art = Path(get_settings().resolved_artifact_dir())

rows = list(csv.DictReader((art / "annotations_a2_fill.csv").open(encoding="utf-8-sig", newline="")))
key = {(r["qid"], int(r["cand_no"])): r
       for r in json.loads((art / "annotations_a2_key.json").read_text(encoding="utf-8"))}
qrels = {json.loads(l)["qid"]: json.loads(l)
         for l in (art / "qrels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def named_industries(text: str) -> set:
    return {i for i, ts in INDUSTRY.items() if any(t in text for t in ts)}


def gold_ind(cid: str) -> str:
    return str(cid).split("-")[0]


out = {"scope_note": "观测域是候选池（第二人只看系统检出 top-8 ∪ 第一人 gold），不是全库",
       "n_rows": len(rows), "gap_rows": [], "reject_rows": []}
no_positive_questions = []

for qid in sorted({r["qid"] for r in rows}):
    qrows = [r for r in rows if r["qid"] == qid]
    rec = qrels[qid]
    parts = len(INTERROGATIVE.findall(rec["question"]))
    named = named_industries(rec["question"])
    if not any((r.get(FILL_COL) or "").strip() == "1" for r in qrows):
        no_positive_questions.append(qid)
    for r in qrows:
        b = int((r.get(FILL_COL) or "").strip())
        a_s = int(key[(qid, int(r["cand_no"]))]["in_a_section_gold"])
        a_t = int(key[(qid, int(r["cand_no"]))]["in_a_strict_gold"])
        text = r["text_head"] or ""
        signals = {
            "cross_industry": len(named) == 1 and gold_ind(r["chunk_id"]) not in named,
            "two_part": parts >= 2,
            "no_modal": not MODAL.search(text),
        }
        # 漏标候选：B 判是而 strict 不认。两问题上的这种行在乙口径下是「部分回答」，
        # 已被机械归 0，不算 gold 缺口——所以这里显式排除两问题（不是遗漏，是裁决结果）。
        if a_t == 0 and b == 1 and parts < 2:
            out["gap_rows"].append({"qid": qid, "cand_no": int(r["cand_no"]), "chunk_id": r["chunk_id"],
                                    "a_section": a_s, "signals": signals,
                                    "question": rec["question"], "reference": rec.get("reference_answer", ""),
                                    "text": text[:300]})
        if a_t == 1 and b == 0:
            out["reject_rows"].append({"qid": qid, "cand_no": int(r["cand_no"]), "chunk_id": r["chunk_id"],
                                       "a_section": a_s, "signals": signals,
                                       "question": rec["question"], "reference": rec.get("reference_answer", ""),
                                       "text": text[:300]})

print(f"候选池内没有任何一块被 B 判相关的题：{len(no_positive_questions)} 道 -> {no_positive_questions}")
print("  （题面点名主体或描述型 gold 都可能造成；这类题的 gold 值得单独复核）\n")
print(f"B 判是而 strict 不认（gold **漏标**候选）：{len(out['gap_rows'])} 行")
for g in out["gap_rows"]:
    sig = "/".join(k for k, v in g["signals"].items() if v) or "-"
    print(f"  {g['qid']}#{g['cand_no']} [{sig}] {g['chunk_id']}")
    print(f"      题：{g['question'][:70]}")
    print(f"      块：{g['text'][:150]}")
print(f"\nstrict gold 认而 B 判否（gold **扩收/偏松**候选）：{len(out['reject_rows'])} 行")
by_sig = Counter("/".join(k for k, v in g["signals"].items() if v) or "-" for g in out["reject_rows"])
print("  信号分布:", dict(by_sig))
for g in out["reject_rows"]:
    sig = "/".join(k for k, v in g["signals"].items() if v) or "-"
    print(f"  {g['qid']}#{g['cand_no']} [{sig}] {g['chunk_id']}")
    print(f"      题：{g['question'][:70]}")
    print(f"      块：{g['text'][:150]}")

(art / "gold_diff.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n证据： {art / 'gold_diff.json'}")
print("⚠ 信号是候选归类不是判定：cross_industry 靠字面行业词（隐含主体的问法扫不出）；"
      "no_modal 只是描述型的必要条件。删哪块要人工读。")
