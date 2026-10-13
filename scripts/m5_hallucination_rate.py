"""#57 的另一半：把 12 行「处置 vs 所见证据」的人判标签算成真幻觉率那半边读数。

三种模式：
    --emit            出一张**小表**（`q-059 1=? 2=? 类型=`），避免在 1.6KB 证据正文的宽表里横向找格子错行
    --merge           把小表写回工单（只接受 0/1/?、已填格默认不许改、改前整份留底）
    （无参数）          读数：要求 qid 集合双向对齐，未填完照算但显式标 `complete: false`

读数怎么来的（为什么不让工单直接问"这是幻觉吗"）：
    真幻觉（无依据却出结论）= **结论非空 且 evidence_sufficient==0**  ← 机械导出
    误拒（该答却拒）        = **结论为空 且 evidence_sufficient==1**
    行为正确率              = behavior_correct==1 / 已判行
    ? 与空**都不进分母**——空是没填，? 是结构性不可判（例如系统压根没检索，判不了"证据够不够"）。

一条防假绿的硬规则：本轮 12 行系统**全部没给结论**（7 拒答 + 5 转人工），
所以"无依据却出结论"的分母是 **0**。分母为 0 时**必须报「不适用 / 无此类样本」**，
不能报 0.00%——那是把「没测到」写成「测了且优秀」，与 M4 那次「空容器集上的公网探测全失败 = 已封堵」同一类假绿。
这张表此刻真正能量化的是**误拒**与**该转未转**两件事；真幻觉率要有"系统给出了结论"的非作答题样本才出得来。

跑法：uv run python scripts/m5_hallucination_rate.py [--emit|--merge]
产物：var/gate0/hallucination_rate.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "var" / "gate0"
FILL = ART / "hallucination_fill.csv"
RUN = ART / "hallucination_run.json"
KEY = ART / "hallucination_key.json"
OUT = ART / "hallucination_rate.json"
TINY = ART / "hallucination_tiny.txt"

C1, C2, C3 = ("evidence_sufficient__fill_1_or_0", "behavior_correct__fill_1_or_0",
              "failure_kind__fill_optional")
VALS = ("0", "1", "?")


def _raw_rows() -> list[list[str]]:
    """整份文件用 csv.reader 读：**证据单元格里有换行**（块之间用 `---` 分隔），
    按行 split 再找表头会把一条记录切成几行——那是错行的另一种发生方式。"""
    with FILL.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.reader(f) if r]


def _header_idx(raw: list[list[str]]) -> int:
    for i, r in enumerate(raw):
        if r and r[0] == "qid":
            return i
    raise ValueError("工单里没有以 qid 开头的表头行——文件被改坏了？")


def read_rows() -> list[dict]:
    raw = _raw_rows()
    hi = _header_idx(raw)
    cols = raw[hi]
    return [dict(zip(cols, r)) for r in raw[hi + 1:] if r and r[0].strip()]


def header_and_comments() -> tuple[list[str], list[str]]:
    raw = _raw_rows()
    hi = _header_idx(raw)
    return [r[0] for r in raw[:hi]], raw[hi]


# ---------------- 填写脚手架（不碰判断） ----------------

def emit() -> int:
    if not FILL.exists():
        print("[缺工单] 先跑 scripts/m5_hallucination_sheet.py 生成工单")
        return 2
    rows = read_rows()
    lines = ["# 一行一题：`q-059 1=0 2=1 类型=该答却拒`。1/2 只接受 0、1、?；类型可留空。",
             "# 判据全文在 hallucination_fill.csv 顶部注释里；这三列是人的判断，我只做往返读写。"]
    for r in rows:
        v1 = (r.get(C1) or "").strip() or "?"
        v2 = (r.get(C2) or "").strip() or "?"
        v3 = (r.get(C3) or "").strip()
        lines.append(f"{r['qid']} 1={v1} 2={v2} 类型={v3}")
    TINY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    done = sum(1 for r in rows if (r.get(C1) or "").strip() in VALS)
    print(f"小表：{TINY}（{len(rows)} 行，其中已填 {done} 行）")
    print("  填完跑：uv run python scripts/m5_hallucination_rate.py --merge")
    return 0


def merge() -> int:
    if not TINY.exists():
        print("[缺小表] 先跑 --emit")
        return 2
    rows = read_rows()
    by_q = {r["qid"]: r for r in rows}
    vals: dict[str, dict[str, str]] = {}
    for ln in TINY.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        qid = parts[0]
        if qid not in by_q:
            print(f"[拒绝写回] 小表里有工单外的 qid：{qid!r}（错行就是这么发生的）")
            return 3
        d = {}
        for p in parts[1:]:
            if "=" not in p:
                print(f"[拒绝写回] {qid} 的片段 {p!r} 不是 `键=值`")
                return 3
            k, v = p.split("=", 1)
            d[k] = v.strip()
        for k in ("1", "2"):
            if k not in d:
                print(f"[拒绝写回] {qid} 缺 {k} 列——要么填值要么填 ?，不许缺席")
                return 3
            if d[k] not in VALS:
                print(f"[拒绝写回] {qid} 的 {k}={d[k]!r}，只接受 0 / 1 / ?")
                return 3
        if "3" in d or "类型" in d:
            d["类型"] = d.get("类型", d.get("3", ""))
        if qid in vals:
            print(f"[拒绝写回] {qid} 在小表里出现两次")
            return 3
        vals[qid] = d

    changed, refused = 0, []
    for qid, d in vals.items():
        r = by_q[qid]
        for col, key in ((C1, "1"), (C2, "2")):
            old = (r.get(col) or "").strip()
            new = d[key]
            if old and old != new:
                refused.append(f"{qid}.{col}: {old}→{new}")
            if old != new:
                r[col] = new
                changed += 1
        if d.get("类型"):
            old3 = (r.get(C3) or "").strip()
            if old3 and old3 != d["类型"]:
                refused.append(f"{qid}.类型: {old3}→{d['类型']}")
            if old3 != d["类型"]:
                r[C3] = d["类型"]
                changed += 1
    if refused:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = FILL.with_name(f"hallucination_fill.pre-merge-{stamp}.csv")
        shutil.copyfile(FILL, bak)
        print(f"[改判留底] 有 {len(refused)} 个格子在已有值上被改动：{refused[:8]}；旧表 → {bak.name}")

    comments, cols = header_and_comments()
    with FILL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for c in comments:
            w.writerow([c])
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(k, "") for k in cols])
    print(f"写回 {changed} 个格子 / {len(vals)} 行 → {FILL}")
    return 0


# ---------------- 读数 ----------------

def rate(num: int, den: int) -> dict:
    """分母为 0 时**不给 0.0%**——那是"没测到"，不是"测了且优秀"。"""
    if not den:
        return {"n": 0, "numerator": 0, "value": None,
                "reading": "不适用（此类样本为 0，不得读成 0%）"}
    return {"n": den, "numerator": num, "value": round(num / den, 4),
            "reading": f"{num}/{den} = {num / den:.2%}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.emit:
        return emit()
    if args.merge:
        return merge()

    for p in (FILL, RUN, KEY):
        if not p.exists():
            print(f"[缺输入] {p.name} 不在")
            return 2
    rows, run = read_rows(), json.loads(RUN.read_text(encoding="utf-8"))
    key = {k["qid"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))["rows"]}
    by_q = {r["qid"]: r for r in run["rows"]}

    # 双向对齐：工单与运行记录必须同一批 qid，且**题面哈希一致**——标签只能配它判过的那一代产物
    miss = (set(by_q) - set({r["qid"] for r in rows})) | ({r["qid"] for r in rows} - set(by_q))
    if miss:
        print(f"[对不上] 工单与运行记录 qid 不一致：{sorted(miss)}")
        return 3
    cur_hash = run.get("question_hash")
    if cur_hash and cur_hash != _hash_of(rows, by_q):
        print(f"[拒绝读数] 工单题面与运行记录不同代（哈希 {cur_hash} vs 现算）——"
              "题面改过就要重出工单，混代算出来的率没有意义")
        return 3

    judged: list[dict] = []
    blanks: list[str] = []
    for r in rows:
        v1, v2 = (r.get(C1) or "").strip(), (r.get(C2) or "").strip()
        if v1 == "" or v2 == "":
            blanks.append(r["qid"])
            continue
        if v1 not in VALS or v2 not in VALS:
            print(f"[判废] {r['qid']} 填了 {v1!r}/{v2!r}，只接受 0 / 1 / ?")
            return 3
        x = by_q[r["qid"]]
        judged.append({
            "qid": r["qid"], "c1": v1, "c2": v2,
            "failure_kind": (r.get(C3) or "").strip(),
            "has_conclusion": bool((x.get("conclusion") or "").strip()),
            "status": x.get("status"), "needs_human": bool(x.get("needs_human")),
            "evidence_n": len(x.get("retrieved_ids") or []),
            "expected_behavior": key[r["qid"]]["expected_behavior"],
            "category": key[r["qid"]]["category"],
        })

    if not judged:
        print(f"一题都没填（{len(rows)} 行全空），无法计算。"
              "先跑 --emit 出小表、填完 --merge，再跑本脚本。\n"
              "不落任何产物：写一份全是「不适用」的读数文件，下次会被当成测过了。")
        return 2

    # 判据一的可判分母还要排除"系统没检索"的行——那类行的 c1 按工单说明填 ?，这里再兜一层
    def assessable(rows_: list[dict], want_conclusion: bool) -> list[dict]:
        return [j for j in rows_ if j["c1"] in ("0", "1")
                and j["has_conclusion"] == want_conclusion and j["evidence_n"] > 0]

    halluc_den = assessable(judged, True)
    halluc_num = [j for j in halluc_den if j["c1"] == "0"]
    over_den = assessable(judged, False)
    over_num = [j for j in over_den if j["c1"] == "1"]
    bc = [j for j in judged if j["c2"] in ("0", "1")]
    bc_num = [j for j in bc if j["c2"] == "1"]

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope_note": ("观测域 = 系统实际检索到的 Top-5 证据面（候选池），不是全库；"
                       "「该检到却没检到」不在本表职责内（属 #58 检索缺口）。"),
        "pairing_note": run.get("pairing_note", ""),
        "question_hash": cur_hash,
        "n_rows_total": len(rows), "n_rows_judged": len(judged), "blank_rows": blanks,
        "complete": not blanks,
        "system_behavior_shape": {
            "rows_with_conclusion": sum(1 for j in judged if j["has_conclusion"]),
            "rows_without_conclusion": sum(1 for j in judged if not j["has_conclusion"]),
            "status_dist": dict(Counter(j["status"] for j in judged)),
            "note": ("这个分布决定哪几个率有分母。本次生成的 12 行系统全部没给结论 ⇒ "
                     "真幻觉率的分母为 0，只能报「不适用」，不能报 0%"),
        },
        "metrics": {
            "hallucination_unsupported_conclusion": rate(len(halluc_num), len(halluc_den)),
            "over_refusal_had_evidence_but_refused": rate(len(over_num), len(over_den)),
            "behavior_correct": rate(len(bc_num), len(bc)),
        },
        "lists": {
            "hallucination_rows": [j["qid"] for j in halluc_num],
            "over_refusal_rows": [j["qid"] for j in over_num],
            "behavior_wrong_rows": [{"qid": j["qid"], "kind": j["failure_kind"],
                                     "status": j["status"], "expected": j["expected_behavior"]}
                                    for j in bc if j["c2"] == "0"],
            "not_assessable_no_evidence_face": [j["qid"] for j in judged if j["evidence_n"] == 0],
        },
        "failure_kind_dist": dict(Counter(j["failure_kind"] for j in judged if j["failure_kind"])),
        "escalate_structural": {
            "rows": sum(1 for j in judged if j["expected_behavior"] == "escalate"),
            "needs_human_all": all(j["needs_human"] for j in judged
                                   if j["expected_behavior"] == "escalate") if any(
                                       j["expected_behavior"] == "escalate" for j in judged) else None,
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    m = out["metrics"]
    st = "" if out["complete"] else f"（未完成：还有 {len(blanks)} 行空着 {blanks[:5]}）"
    print(f"已判 {out['n_rows_judged']}/{out['n_rows_total']} 行{st}")
    print(f"  真幻觉（无依据却出结论）: {m['hallucination_unsupported_conclusion']['reading']}")
    print(f"  误拒（证据够却拒答）    : {m['over_refusal_had_evidence_but_refused']['reading']}")
    print(f"  处置正确率              : {m['behavior_correct']['reading']}")
    if out["lists"]["behavior_wrong_rows"]:
        print(f"  判错的行: {out['lists']['behavior_wrong_rows']}")
    print(f"  ⚠ {out['scope_note']}")
    print(f"  ⚠ {out['system_behavior_shape']['note']}")
    print("证据：", OUT)
    return 0


def _hash_of(rows: list[dict], by_q: dict) -> int:
    pairs = sorted((r["qid"], by_q[r["qid"]]["question"]) for r in rows)
    return __import__("zlib").crc32("|".join(f"{q}::{t}" for q, t in pairs).encode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
