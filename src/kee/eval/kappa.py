"""双人标注一致性：Cohen's Kappa。

口径：以「(qid, chunk_id) 对被判定为相关/不相关」为观测单元，
A = 构造时 gold（qrels.jsonl 的 gold_chunk_ids），B = 第二人复核（annotations_a2.csv）。
未填的第二人结果会直接报错退出，避免把「没标」当成「不一致」。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kee.config import get_settings  # noqa: E402


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> dict[str, float]:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("两份标注必须等长且非空")
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    pa1 = Counter(labels_a)[1] / n
    pb1 = Counter(labels_b)[1] / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return {"kappa": float("nan"), "po": round(po, 4), "pe": 1.0, "n": n, "note": "单一类别，Kappa 无定义"}
    return {"kappa": round((po - pe) / (1 - pe), 4), "po": round(po, 4), "pe": round(pe, 4), "n": n}


def load_pair_labels(qrels_path: Path, a2_path: Path) -> tuple[list[int], list[int], list[tuple[str, str]]]:
    qa = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        qa[obj["qid"]] = set(obj["gold_chunk_ids"])

    pairs: set[tuple[str, str]] = set()
    qb: dict[str, set[str]] = {}
    unfilled: list[str] = []
    with a2_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["qid"]
            raw = (row.get("annotator2_gold_chunk_ids") or "").strip()
            if not raw or "TODO" in raw:
                unfilled.append(qid)
                ids = set()
                pairs |= {(qid, c) for c in qa.get(qid, set())}
            else:
                ids = {x.strip() for x in raw.replace(",", ";").split(";") if x.strip()}
                pairs |= qa.get(qid, set()) | ids
            qb[qid] = ids
    if unfilled:
        raise ValueError(f"第二人标注未完成，缺 {len(unfilled)} 题（示例 {unfilled[:3]}）；搜 TODO_ANNOTATOR2 定位待填项")

    la, lb = [], []
    for qid, cid in sorted(pairs):
        la.append(1 if cid in qa.get(qid, set()) else 0)
        lb.append(1 if cid in qb.get(qid, set()) else 0)
    return la, lb, sorted(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--a2", default=None)
    args = ap.parse_args()

    art = Path(get_settings().resolved_artifact_dir())
    qrels = Path(args.qrels) if args.qrels else art / "qrels.jsonl"
    a2 = Path(args.a2) if args.a2 else art / "annotations_a2_template.csv"
    for p in (qrels, a2):
        if not p.exists():
            print(f"缺文件：{p}（先跑 g02 建库并生成评测集）")
            return 2
    la, lb, pairs = load_pair_labels(qrels, a2)
    out = cohens_kappa(la, lb)
    out["unit"] = "(qid, chunk_id) 相关判定"
    out["n_positive_a"] = sum(la)
    out["n_positive_b"] = sum(lb)
    art.joinpath("kappa.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
