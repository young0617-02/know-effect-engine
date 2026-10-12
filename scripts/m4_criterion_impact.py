"""裁决前置：把甲/乙两个判据各自的后果量成数，供人工拍板（不替人判）。

甲 = 逐块判「答到任一子问」即算相关；乙 = 必须答全所有子问才算相关。
乙 的隐藏代价是：**如果某题的 gold 本身就没有一块能答全所有子问，这题在 strict 口径下直接不可判**。
所以这里对已判的两问题，把 gold 原文与每个子问并排打出来，逐题看 gold 覆没覆盖全。

只读 var/gate0/{qrels.jsonl,kappa_a2.json,annotations_a2_fill.csv}，不开 PG。
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

INTERROGATIVE = re.compile(r"什么|哪些|怎么|怎样|多久|多少|能不能|可不可以|是否|依据哪|哪个|哪几|从哪里|吗")
art = Path(get_settings().resolved_artifact_dir())

rows = list(csv.DictReader((art / "annotations_a2_fill.csv").open(encoding="utf-8-sig", newline="")))
key = {(r["qid"], int(r["cand_no"])): r
       for r in json.loads((art / "annotations_a2_key.json").read_text(encoding="utf-8"))}
qrels = {}
for line in (art / "qrels.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        rec = json.loads(line)
        qrels[rec["qid"]] = rec
div = json.loads((art / "kappa_a2.json").read_text(encoding="utf-8"))["divergence_samples"]

judged = {r["qid"] for r in rows if (r.get("b_directly_answers__fill_1_or_0") or "").strip() in ("0", "1")}
two_part = [q for q in judged if len(INTERROGATIVE.findall(qrels[q]["question"])) >= 2]

print(f"已判 {len(judged)} 题，其中一问两问 {len(two_part)} 题：{sorted(two_part)}\n")

for qid in sorted(two_part):
    rec = qrels[qid]
    marks = INTERROGATIVE.findall(rec["question"])
    strict = rec.get("gold_strict_ids") or []
    print("=" * 96)
    print(f"{qid}  子问标记 {len(marks)} 个（{marks}）")
    print(f"  题面：{rec['question']}")
    print(f"  参考应答：{rec.get('reference_answer','')[:120]}")
    print(f"  strict gold {len(strict)} 块 —— 乙口径下要看它是否答全：")
    for cid in strict:
        head = next((r["text_head"] for r in rows if r["chunk_id"] == cid), "")
        print(f"    - {cid}")
        print(f"      {head[:300]}")
    b_extra = [d for d in div if d["qid"] == qid and d["b"] == 1 and d["a_strict"] == 0]
    print(f"  甲口径下会新增进 gold 的块（B 判是、strict 未收）：{len(b_extra)}")
    for d in b_extra:
        print(f"    - cand{d['cand_no']} {d['chunk_id']}")
        print(f"      {d['chunk_text'][:200]}")

n_extra = sum(1 for d in div if d["b"] == 1 and d["a_strict"] == 0)
n_drop = sum(1 for d in div if d["b"] == 0 and d["a_strict"] == 1)
print("=" * 96)
print(f"\n甲口径的净效果（仅已判 25 题）：strict gold 需新增 {n_extra} 块；B 否掉的 strict gold {n_drop} 块需人工裁（不等于删）")
print(f"全 70 题里两问题的下限（同一词典）：", sum(1 for r in qrels.values() if len(INTERROGATIVE.findall(r['question'])) >= 2))
