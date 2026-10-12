"""把 124 行重标工单变成「读一行、敲一行」的形状，并在回收时做合法性自检。

为什么不直接在 CSV 里填：那张表每行带 231~287 字正文，最后一列在最右边，
人在浏览器/编辑器里要横向找列、还得保证不错行——错一行的代价是**一整行判断挂到别的块上**，
而 Kappa 看不出来（它只看每行的两列标签一不一致）。

这里的做法：
- `--emit`  从 `_fill.csv` 生成一张**只有键**的小表 `<prefix>_tiny.txt`，形如 `q-013#1=`，
  按题分组、带题面当分组头；你只管在等号后敲 0/1，列不会错位。
- `--merge` 校验后把值写回 CSV 的 `b_directly_answers__fill_1_or_0`，**写前先备份**。
- `--check` 只验不写。

自检的四条都是防「假完整」：键集必须与工单**双向**一致（少一行就是没填完，多一行就是题号写错）、
值只能是 0/1（`?`/空/`2` 一律报出来而不是被静默当成 0）、已经填过的格子**默认不许改**
（要改加 `--force`——已收到的标注不能被子流程覆盖）、合并后重数一遍分布。

跑法：
    uv run python scripts/m4_annotation_fill_helper.py --prefix annotations_a2_relabel --emit
    ...在 var/gate0/annotations_a2_relabel_tiny.txt 的等号后填 0/1...
    uv run python scripts/m4_annotation_fill_helper.py --prefix annotations_a2_relabel --merge
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "var" / "gate0"
FILL_COL = "b_directly_answers__fill_1_or_0"
COMMENT_COLS = ("qid", "cand_no")


def read_fill(path: Path) -> list[dict]:
    return list(csv.DictReader(commented_lines(path)[0]))


def commented_lines(path: Path) -> tuple[list[str], list[str]]:
    """返回 (数据行, 原文件里的 # 注释行)。合并时必须把注释原样写回——
    那两行是**判据原文与生成时点**，收了标注之后才发现判据丢了，等于事后说不清按什么判的。
    表头上残留的 BOM 一并剥掉：它会让第一列变成 `qid`，然后报一个标注人看不懂的 KeyError。"""
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    data = [l.lstrip("﻿") for l in raw if not l.startswith("#")]
    return data, [l.lstrip("﻿") for l in raw if l.startswith("#")]


def key_of(row: dict) -> str:
    return f"{row['qid']}#{row['cand_no']}"


def emit(prefix: str) -> int:
    csv_path = ART / f"{prefix}_fill.csv"
    if not csv_path.exists():
        print(f"[缺工单] 找不到 {csv_path.name}")
        return 2
    rows = read_fill(csv_path)
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    qtext = {}
    ws = ART / f"{prefix}_worksheet.md"
    if ws.exists():
        for ln in ws.read_text(encoding="utf-8").splitlines():
            if ln.startswith("## "):
                qid, _, rest = ln[3:].partition("｜")
                qtext[qid] = rest
    out = ART / f"{prefix}_tiny.txt"
    L = [f"# 重标小表｜源 {csv_path.name}｜{len(by_q)} 题 / {len(rows)} 行｜生成 {datetime.now().isoformat(timespec='seconds')}",
         "# 每行等号后只敲 0 或 1；不许留空、不许填 ?。整题的每一行都要填。",
         "# 判据（乙口径）：这一块正文**单独**能不能答全该题；跨项目/跨行业同名章节=0；"
         "题面点名单一文档时同项目其他文档=0；描述型块=0。",
         "# 这张表只当「填得不错行」的脚手架，判什么仍以 _worksheet.md 的正文为准。", ""]
    for qid in sorted(by_q):
        L.append(f"# {qid}｜{qtext.get(qid, '')}（{len(by_q[qid])} 行）")
        for r in sorted(by_q[qid], key=lambda x: int(x["cand_no"])):
            v = (r.get(FILL_COL) or "").strip()
            v = v if v in ("0", "1") else ""
            L.append(f"{key_of(r)}={v}")
        L.append("")
    if out.exists() and any(l.split("=")[-1].strip() in ("0", "1") for l in out.read_text(encoding="utf-8").splitlines() if "=" in l):
        bak = out.with_suffix(f".{datetime.now().strftime('%H%M%S')}.bak")
        shutil.copyfile(out, bak)
        print(f"[留底] 小表里已有填过的值，复制到 {bak.name} 后重写")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"小表：{out}")
    print(f"  {len(rows)} 行键；已填 {sum(1 for r in rows if (r.get(FILL_COL) or '').strip() in ('0','1'))} 行")
    return 0


def parse_tiny(path: Path) -> dict:
    vals: dict[str, str] = {}
    empty: list[str] = []
    illegal: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        v = v.strip()
        if v in ("0", "1"):
            vals[k.strip()] = v
        elif not v:
            empty.append(k.strip())
        else:
            illegal.append(f"{k.strip()}='{v}'")
    return {"vals": vals, "empty": empty, "illegal": illegal}


def report(rows: list[dict]) -> None:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    dist = Counter((r.get(FILL_COL) or "").strip() or "TODO" for r in rows)
    print(f"  分布 {dict(dist)}｜{len(by_q)} 题")
    allzero = [q for q, rs in by_q.items() if rs and all((r.get(FILL_COL) or "").strip() == "0" for r in rs)]
    if allzero:
        print(f"  提示：这些题**整题判 0**——{sorted(allzero)}。不是错误，但按首轮经验"
              "（跨文档题在乙口径下确实可能无单块可答全）值得回头核一遍题面是不是又长回了集合型。")


def merge(prefix: str, force: bool) -> int:
    csv_path = ART / f"{prefix}_fill.csv"
    tiny_path = ART / f"{prefix}_tiny.txt"
    if not csv_path.exists() or not tiny_path.exists():
        print("[缺输入] _fill.csv 或 _tiny.txt 有一份不在")
        return 2
    rows = read_fill(csv_path)
    parsed = parse_tiny(tiny_path)
    vals, empty, illegal = parsed["vals"], parsed["empty"], parsed["illegal"]

    keys = {key_of(r) for r in rows}
    missing = sorted((keys - set(vals)) | set(empty))
    extra = sorted(set(vals) - keys)
    if illegal:
        print(f"[拒绝合并] 有 {len(illegal)} 行填的不是 0/1：{illegal[:6]}"
              "——`?` 会被下游当 0 用，所以这里不当没填，直接退回")
        return 4
    if missing:
        print(f"[拒绝合并] 还差 {len(missing)} 行没填，例如 {missing[:6]}（一行没填就别算 Kappa）")
        return 4
    if extra:
        print(f"[拒绝合并] 小表里有 {len(extra)} 个工单外的键，例如 {extra[:6]}（题号或编号写错过不了）")
        return 4

    changed, blocked = 0, []
    for r in rows:
        cur = (r.get(FILL_COL) or "").strip()
        new = vals[key_of(r)]
        if cur in ("0", "1") and cur != new:
            if not force:
                blocked.append(f"{key_of(r)}: CSV 里是 {cur}、小表要改成 {new}")
                continue
            r[FILL_COL] = new
            changed += 1
        elif cur != new:
            r[FILL_COL] = new
            changed += 1
    if blocked:
        print(f"[拒绝覆盖已收标注] {len(blocked)} 行要把已填的格子改成不同值：")
        for b in blocked[:10]:
            print("   " + b)
        print("  确认是有意重判再加 --force（改前会先备份整份 CSV）")
        return 5

    bak = csv_path.with_suffix(f".pre-merge-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    shutil.copyfile(csv_path, bak)
    _data, comments = commented_lines(csv_path)
    # BOM 必须在**最前面**（Excel 靠它认中文），所以这里用 utf-8 手写 BOM，
    # 而不是 utf-8-sig——后者会把 BOM 写在第一次 write 的位置，一旦上面有 # 注释行，
    # BOM 就落到文件中间，表头变成 `qid`，下一行读取直接 KeyError（自检时踩过）。
    had_bom = csv_path.read_bytes().startswith(b"\xef\xbb\xbf") or (comments and comments[0].startswith("﻿"))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if had_bom:
            f.write("﻿")
        for c in comments:
            f.write(c.lstrip("﻿") + "\r\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[已合并] 写入 {changed} 行｜整份备份：{bak.name}")
    report(read_fill(csv_path))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="annotations_a2")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="允许覆盖 CSV 里已填的格子（改前备份）")
    args = ap.parse_args()
    if not (args.emit or args.merge or args.check):
        print("要选一条：--emit / --merge / --check")
        return 2
    if args.emit:
        return emit(args.prefix)
    if args.check:
        rows = read_fill(ART / f"{args.prefix}_fill.csv")
        report(rows)
        todo = [key_of(r) for r in rows if (r.get(FILL_COL) or "").strip() not in ("0", "1")]
        print(f"  未填/非法 {len(todo)} 行" + (f"，例如 {todo[:8]}" if todo else ""))
        return 0 if not todo else 4
    return merge(args.prefix, args.force)


if __name__ == "__main__":
    sys.exit(main())
