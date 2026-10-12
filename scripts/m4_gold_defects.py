"""扫全库 gold 的两类结构性缺陷：一问两问 与「题面点名单一行业但 gold 跨行业」。

为什么单开一个脚本（而不是塞进 m4_kappa）：`m4_kappa` 量的是**两个人的判断是否一致**，
这里量的是**尺子本身长什么样**——后者不需要第二人参与，全库 70 题都能扫，
所以它的输出是「判据要补哪条规则、影响面多大」的依据，不是又一组一致性读数。

两条计数法都踩过坑，都写死在代码里而不是靠人记得：
* **一问两问按问号数会漏**：这套题的两问只挂一个问号（q-013「…是什么，…能不能…」），
  按问号只数出 1 题，按疑问标记数出 11 题。**词典不全时结果是下界**（q-020「从哪里开始」数出 0），
  报数时按「≥」读。
* **行业词典漏词会把正确题判成缺陷**：第一版词典没有「能源/化工/金融」，
  于是 q-055（题面明写三条线并列）被误报成 gold 跨行业缺陷。补全后命中数从 2 回到 1。

局限必须连着数一起说：主体识别是**字面**的。题面靠项目名或系统名隐含主体时这招扫不出来，
所以这里只能说「词典法命中 N 处」，**不能说「全库只有 N 处」**。

只读 `var/gate0/qrels.jsonl`，不开 PG、不开 Milvus：
uv run python scripts/m4_gold_defects.py [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402

INTERROGATIVE = re.compile(
    r"什么|哪些|怎么|怎样|多久|多少|能不能|可不可以|是否|依据哪|哪个|哪几|从哪里|吗")
# 行业线索词。三类都要含「该行业在题面里的自然叫法」，否则会把并列问三条线的题误判成点名单一行业。
INDUSTRY = {
    "fin": ("金融", "清算", "资金", "日切", "对账", "信用损失", "报表", "IFRS", "银企"),
    "ene": ("电力", "能源", "电网", "调度", "厂站", "SCADA", "终端", "台区", "线损",
            "配电", "分布式电源", "用电信息"),
    "chm": ("化工", "装置", "联锁", "SIS", "防爆", "批次", "炼化", "催化剂", "GMP",
            "药品", "工艺", "重整"),
}


def scan(rows: list[dict]) -> dict:
    two_part, cross_named = [], []
    for q in rows:
        marks = len(INTERROGATIVE.findall(q["question"]))
        if marks >= 2:
            two_part.append({"qid": q["qid"], "marks": marks, "question": q["question"]})
        named = {i for i, ts in INDUSTRY.items() if any(t in q["question"] for t in ts)}
        gold_ind = {str(cid).split("-")[0]
                    for cid in set(q["gold_chunk_ids"]) | set(q.get("gold_strict_ids") or [])}
        gold_ind.discard("")
        if len(named) == 1 and len(gold_ind) > 1:
            cross_named.append({"qid": q["qid"], "named": sorted(named),
                                "gold_industries": sorted(gold_ind),
                                "question": q["question"],
                                "n_strict_gold": len(q.get("gold_strict_ids") or [])})
    return {"n_questions": len(rows),
            "two_part": {"count": len(two_part), "note": "下界：疑问词典不全", "rows": two_part},
            "gold_cross_industry": {"count": len(cross_named), "rows": cross_named,
                                    "note": "题面点名单一行业但 gold 含其他行业；主体识别是字面的，"
                                            "靠项目名/系统名隐含主体的问法扫不出来"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    art = Path(get_settings().resolved_artifact_dir())
    p = art / "qrels.jsonl"
    if not p.exists():
        print(f"缺 {p}（先跑 g02 建评测集）")
        return 2
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = scan(rows)
    if args.json:
        (art / "m4_gold_defects.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全 {out['n_questions']} 题")
    print(f"  一问两问: {out['two_part']['count']} 题（{out['two_part']['note']}）"
          f" -> {[r['qid'] for r in out['two_part']['rows']]}")
    print(f"  题面点名单一行业但 gold 跨行业: {out['gold_cross_industry']['count']} 题")
    for r in out["gold_cross_industry"]["rows"]:
        print(f"     {r['qid']} 点名 {r['named']} 但 gold 含 {r['gold_industries']}"
              f"（strict gold {r['n_strict_gold']} 块）：{r['question']}")
    print("  局限:", out["gold_cross_industry"]["note"])
    if args.json:
        print(f"证据： {art / 'm4_gold_defects.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
