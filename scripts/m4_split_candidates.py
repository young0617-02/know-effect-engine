"""拆题候选核对（离线、要点 token 版）。

为什么不用字符串包含：参考答案是原文的**压缩改写**，字面不等不等于语义不覆盖
（实测把 q-013 误报成「撑不起」，而那块明明写着「其 2oo3 表决逻辑与整定值不随优化目标调整」）。
本语料的答点几乎都是**可精确匹配的 token**：标准号、数字、单位、帧名、流程号——
所以核对「参考答案里的关键 token 有没有出现在金块正文里」才有判别力。

两类题排除在拆题判断之外：`expected_behavior` 为 refuse / escalate 的题。
它们的参考答案是**处置政策**（「应转人工、不得直接对外输出」），本来就不由任何金块正文承载，
按覆盖度判它们「撑不起」是把题型判成了缺陷。

局限连着数一起报：`text_head` 截到 320 字，金块后半段的 token 可能看不到（漏 → 误报要拆）；
token 抽取也认不全中文短语（漏判 → 误报够）。所以这份给的是**候选面**，最终拆哪几道要人工读金块全文。
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
FILL_COL = "b_directly_answers__fill_1_or_0"
# 答点 token：字母数字串（2oo3 / TESTFH / GB/T 22239-2019 / Q/KEE-SOP-032 / IF-CHM-0007）、
# 数字+单位（15 分钟 / 60 秒 / 8% / 4 小时 / 3 年 / 7 个自然日）、纯数字（5 次 / 30 分钟）
TOKEN = re.compile(
    r"[A-Za-z][A-Za-z0-9./\-]{2,}"          # 标准号/代号/帧名
    r"|\d+(?:\.\d+)?\s*(?:分钟|小时|秒|天|年|个月|个自然日|%|次|期|万元|元|人|路|项|处|台)"
    r"|\d+(?:\.\d+)?"
)
POLICY_WORDS = ("转人工", "不得直接对外", "授权", "商务", "拒答", "敏感")

art = Path(get_settings().resolved_artifact_dir())
rows = list(csv.DictReader((art / "annotations_a2_fill.csv").open(encoding="utf-8-sig", newline="")))
key = {(r["qid"], int(r["cand_no"])): r
       for r in json.loads((art / "annotations_a2_key.json").read_text(encoding="utf-8"))}
qrels = {json.loads(l)["qid"]: json.loads(l)
         for l in (art / "qrels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

pool: dict[str, list[dict]] = {}
for r in rows:
    pool.setdefault(r["qid"], []).append(r)

two_part = [qid for qid, q in qrels.items() if len(INTERROGATIVE.findall(q["question"])) >= 2]
print(f"两问题 {len(two_part)} 道（词典法，下界）\n")

rows_out = []
for qid in sorted(two_part):
    q = qrels[qid]
    behavior = q.get("expected_behavior", "")
    ref = (q.get("reference_answer") or "").strip()
    tokens = sorted(set(TOKEN.findall(ref) and [m.group(0) for m in TOKEN.finditer(ref)]))
    policy_only = not tokens and any(w in ref for w in POLICY_WORDS)
    gold_rows = [r for r in pool.get(qid, []) if key[(qid, int(r["cand_no"]))]["in_a_strict_gold"] == 1]
    gold_text = " ".join((g["text_head"] or "") for g in gold_rows)

    if behavior in ("refuse", "escalate") or policy_only:
        verdict, note = "不参与拆题判断", f"题型 {behavior or '政策型'}，参考答案是处置政策不是事实点"
        missing = []
    elif not gold_rows:
        verdict, note = "无法离线核对", "工单候选集里没有该题金块"
        missing = []
    else:
        missing = [t for t in tokens if t.replace(" ", "") not in gold_text.replace(" ", "")]
        if not tokens:
            verdict, note = "要点无法 token 化", "参考答案是纯中文短语，需人工读金块"
        elif not missing:
            verdict, note = "够（不必拆）", f"{len(tokens)} 个答点 token 全在金块里"
        elif len(missing) <= len(tokens) / 2:
            verdict, note = "边界（人工裁）", f"缺 {len(missing)}/{len(tokens)}"
        else:
            verdict, note = "撑不起（候选要拆）", f"缺 {len(missing)}/{len(tokens)}"
    rows_out.append((qid, verdict, behavior))
    print(f"[{verdict}] {qid}  题型={behavior or '-'}｜答点 token {len(tokens)} 个｜缺 {len(missing)}")
    print(f"      题面：{q['question']}")
    print(f"      参考：{ref[:110]}")
    if tokens:
        print(f"      token：{tokens}｜缺：{missing}")
    print(f"      判定依据：{note}")

print("=" * 92)
from collections import Counter
print("汇总:", dict(Counter(v for _, v, _ in rows_out)))
print("候选要拆:", [q for q, v, _ in rows_out if v == "撑不起（候选要拆）"])
print("边界待人工:", [q for q, v, _ in rows_out if v == "边界（人工裁）"])
print("不参与:", [q for q, v, _ in rows_out if v == "不参与拆题判断"])
print("无法核对:", [q for q, v, _ in rows_out if v in ("无法离线核对", "要点无法 token 化")])
print("\n⚠ 局限：text_head 截到 320 字（金块后半段的 token 看不到 → 会误报「要拆」）；"
      "token 化只认得住数字/代号，纯中文答点会漏。所以这是**候选面**，最终拆题要人工读金块全文。")
