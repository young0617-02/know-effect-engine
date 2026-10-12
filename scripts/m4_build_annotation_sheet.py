"""生成第二人标注工单（可填表）。

为什么要单独做这个而不是直接让人填 `annotations_a2_template.csv`：
模板要填的是 `fin-closeout-prj-fin-2024-017@v1#2.1#c2` 这种 chunk ID。
那等于要求标注人边开语料边数块号——70 题会做半天以上，而且填错的 ID 根本对不上。
工单不可填，是这份标注会烂尾的真正原因，不是「没时间」。

更要紧的是防一个必然的假绿：**若只让第二人在第一人标好的 gold 里勾「同意吗」，
他没有机会表示异议，Kappa 会结构性趋近 1。** 那比没有 Kappa 更糟——
它会被当成「双人一致性已验证」写进对外材料。

所以这张表有三条设计约束：

1. **候选集 = 系统检出的 top-k ∪ 第一人的 gold（含 strict）**，
   于是必然包含「第一人没标但系统召回了」的块——异议空间是构造出来的，不是碰运气。
   脚本会**断言**至少 60% 的题含此类候选，不满足就直接失败（这个比例是防假绿的闸，不是统计兴趣）。
2. **工单里不显示哪些块是第一人标过的。** 一起看就锚定了。
   对应关系另存 `annotations_a2_key.json`，算完再对。
3. **判据只有一句**，写在表头：这块正文里是否**直接含有能回答该问题的内容**；
   同主题、同章节但不含答案 → 0。不这么写，第二人会按「相关吗」判，
   而「相关」正是 section 级 gold 的口径——那 Kappa 测的就又变成口径差异了。

候选从**服务端 BM25**（jieba 分词）取，默认不跑稠密检索。
两个理由：① 稠密那条要加载 BGE-M3，与常驻的 7B 抢 8GB 显存，实测跑满 70 题会
在客户端里崩掉（`memory allocation of 4325392 bytes failed`）——工单不该为此背一个
显存调度问题；② 词法命中的性质正是我们想要的：它会召回「同章节同主题但不含答案」的块，
也就是好的反例。代价说清楚——这份候选**不等于系统实际拿去答题的那五块**，
所以工量单用来标 gold 合适，用来复现检索质量不合适（那是 m2_eval 的活）。
`--hybrid` 可切回稠密+词法，显存有余量时再用。

用法：uv run python scripts/m4_build_annotation_sheet.py [--top-k 8] [--limit N] [--hybrid]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

MIN_OUTSIDE_A_GOLD_SHARE = 0.60  # 低于此比例说明候选集没能给出异议空间
FILL_COL = "b_directly_answers__fill_1_or_0"
# 正文截断长度。160 字太紧：答案值常落在段落后半，
# 截短了会逼标注人猜，那测出来的就不是他的判断。全文可用 qid#编号 现场取。
TEXT_HEAD_CHARS = 320
JUDGEMENT_LINE = (
    "判据只有一条：**这一块的正文里，是否直接含有能答全该问题的内容。**\n"
    "同主题、同章节、相关但不含答案 → 填 0。别按「相关吗」判——那测的就又变回 gold 口径差异了。\n"
    "三条细则（由首轮 25 题的 9 行真分歧裁决得出，2026-10-12 用户选定乙口径）：\n"
    "1. 题面有两问时（例：「表决逻辑是什么，**能不能改整定值**」），必须**同一块内答全所有问**才算 1；"
    "只答其中一问 → 0。\n"
    "2. 题面点名了主体（清算平台 / 某装置 / 某条线）时，**其他行业或别的项目**的同名章节即使数字相同也算 0；"
    "题面未点名主体、把几条线并列问到时，各线的块都算。\n"
    "   2b. 题面**点名了某一份文档**（如「化工运维手册里…」「接口规范里…」「合规摘录里…」）时，"
    "同项目**别那份文档**里内容相同的块也算 0（2026-10-12 补，重标轮第二人在 q-058 上判过一例，"
    "其余 7 道点名文档的题他都遵守了这条，说明这是判据没写清而不是人不合）。\n"
    "3. **描述型**块（部署了几台、达成率多少、周期几天）不算直接可答，除非正文里有带强制性"
    "（必须 / 不得 / 只能 / 一律）的句子正面回答该问。\n"
)


def _guard_existing_fill(csv_path: Path, *, force: bool) -> str | None:
    """**不许无声覆盖已填的标注。**

    这张 CSV 是人工劳动的唯一载体，重跑生成器会把它整表刷回 TODO。
    首轮的 210 行就是靠这条守卫才没被后面任何一次「顺手重跑」抹掉。
    确实要重来时必须显式 `--force`，且先把旧表另存一份带时间戳的备份。
    """
    if not csv_path.exists():
        return None
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        filled = sum(1 for r in csv.DictReader(f) if (r.get(FILL_COL) or "").strip() in ("0", "1"))
    if not filled:
        return None
    if not force:
        return (f"已填 {filled} 行的工单已存在：{csv_path}\n"
                "        重跑会整表刷回 TODO、抹掉人工标注。要继续填这张表就直接填；\n"
                "        确实要重建请加 --force（会先备份已填表）。")
    bak = csv_path.with_name(f"{csv_path.stem}.filled-{time.strftime('%Y%m%dT%H%M%S')}.csv")
    shutil.copyfile(csv_path, bak)
    print(f"[--force] 已填表先备份到：{bak.name}（{filled} 行）")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="已填工单存在时先备份再重建（默认拒绝，防止抹掉人工标注）")
    ap.add_argument("--top-k", type=int, default=8, help="每题给第二人看的检索候选数")
    ap.add_argument("--hybrid", action="store_true", help="候选改用稠密+词法（要加载嵌入模型，吃显存）")
    ap.add_argument("--limit", type=int, default=0, help="只出前 N 题的工单（试标用；建议先做 25 题）")
    ap.add_argument("--only-qids", default="",
                    help="只出这些题号的工单，逗号分隔（拆题后重标用：题面一改，原来那几行的判断就不再是同一件事了）")
    ap.add_argument("--out-prefix", default="annotations_a2",
                    help="输出文件名前缀；重标轮要另起一套，别覆盖已收 full 的那份证据")
    ap.add_argument("--worksheet-only", action="store_true",
                    help="只按已存在的 _fill.csv 重排人读版式（不碰 CSV、不连存储栈、不重跑检索）")
    args = ap.parse_args()

    if args.worksheet_only:
        return _rewrite_worksheet_only(Path(args.out_prefix), args.hybrid)

    s = get_settings()
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    rows = build_eval_set(pg, write_files=False)["questions"]
    pool = [q for q in rows if q["gold_chunk_ids"]]
    if args.only_qids:
        want = {x.strip() for x in args.only_qids.split(",") if x.strip()}
        pool = [q for q in pool if q["qid"] in want]
        missing = want - {q["qid"] for q in pool}
        if missing:
            print(f"[拒绝生成] 这些题号没有可标的 gold 块或不存在：{sorted(missing)}")
            return 2
    if args.limit:
        pool = pool[: args.limit]

    with pg.connect() as conn:
        texts = {r["id"]: r["content"] for r in conn.execute("SELECT id, content FROM chunk")}
        meta = {
            r["id"]: r
            for r in conn.execute("SELECT id, doc_id, section_no, page, is_parent FROM chunk")
        }

    kb = KbStore(s.milvus_collection, "jieba")
    rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                         mode="hybrid" if args.hybrid else "sparse", use_rerank=False)
    ctx = AccessContext(role="knowledge_owner")
    rt.retrieve("预热：配电自动化终端心跳超时阈值", ctx)

    fill_rows, key_rows, no_dissent = [], [], []
    for q in pool:
        res = rt.retrieve(q["question"], ctx, top_k=args.top_k)
        a_gold = set(q["gold_chunk_ids"])
        strict = set(q.get("gold_strict_ids") or [])
        # 顺序：先检出的（这是系统真实会拿去答的块），再补 gold 里的，保证 gold 不漏
        cand: list[str] = []
        for cid in [c.chunk_id for c in res.chunks] + sorted(a_gold) + sorted(strict):
            if cid not in cand and cid in texts:
                cand.append(cid)
        if not any(c not in a_gold for c in cand):
            no_dissent.append(q["qid"])
        for n, cid in enumerate(cand, start=1):
            m = meta.get(cid) or {}
            fill_rows.append(
                {
                    "qid": q["qid"],
                    "cand_no": n,
                    "chunk_id": cid,
                    "doc_id": m.get("doc_id", ""),
                    "section_no": m.get("section_no", ""),
                    "page": m.get("page", ""),
                    "is_parent": int(bool(m.get("is_parent"))),
                    "text_head": " ".join((texts.get(cid) or "").split())[:TEXT_HEAD_CHARS],
                    FILL_COL: "TODO",
                }
            )
            key_rows.append({"qid": q["qid"], "cand_no": n, "chunk_id": cid,
                             "in_a_section_gold": int(cid in a_gold),
                             "in_a_strict_gold": int(cid in strict)})

    outside_share = 1 - len(no_dissent) / max(len(pool), 1)
    # 这是防假绿的闸：候选全是 A 标过的块，第二人就只能同意。
    if outside_share < MIN_OUTSIDE_A_GOLD_SHARE:
        print(f"[判废] 只有 {outside_share:.0%} 的题含「第一人未标」的候选，"
              f"低于要求的 {MIN_OUTSIDE_A_GOLD_SHARE:.0%}（无异议空间题：{no_dissent[:8]}）。"
              "这样的工单做出来 Kappa 会结构性偏高，不能拿去当一致性证据。")
        kb.close()
        return 3

    csv_path = art / f"{args.out_prefix}_fill.csv"
    blocked = _guard_existing_fill(csv_path, force=args.force)
    if blocked:
        print(f"[拒绝覆盖] {blocked}")
        kb.close()
        return 4

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fill_rows[0].keys()))
        w.writeheader()
        w.writerows(fill_rows)

    (art / f"{args.out_prefix}_key.json").write_text(json.dumps(key_rows, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    (art / f"{args.out_prefix}_worksheet.md").write_text(
        _worksheet(fill_rows, {q["qid"]: q["question"] for q in pool}, outside_share, args.hybrid),
        encoding="utf-8")
    print(f"工单：{len(pool)} 题 / {len(fill_rows)} 个候选块，待填列「{FILL_COL}」全为 TODO")
    print(f"含「第一人未标」候选的题占比：{outside_share:.0%}（闸值 {MIN_OUTSIDE_A_GOLD_SHARE:.0%}）")
    print(f"人读：{art / f'{args.out_prefix}_worksheet.md'}")
    print(f"机填：{csv_path}")
    kb.close()
    return 0


def _head_stats(rows: list[dict]) -> dict:
    """正文是不是真被截过——这句必须算出来，不能照抄模板话术。
    上一版工单写着「正文只截前 320 字」，而实测最长 287 字、**一行都没截**：
    那等于让标注人以为看到的是节选，于是对「块里其实写全了」的那几行倾向给 0。"""
    lens = [len(r["text_head"]) for r in rows]
    truncated = [r for r in rows if len(r["text_head"]) >= TEXT_HEAD_CHARS or r["text_head"].rstrip().endswith("…")]
    return {"max": max(lens) if lens else 0, "n_truncated": len(truncated),
            "truncated_keys": [f"{r['qid']}#{r['cand_no']}" for r in truncated][:12]}


def _worksheet(fill_rows: list[dict], qtext: dict[str, str], outside_share: float, hybrid: bool) -> str:
    hs = _head_stats(fill_rows)
    head_note = (f"正文**本批 {len(fill_rows)} 行没有一行被截**（最长 {hs['max']} 字，上限 {TEXT_HEAD_CHARS}），"
                 "你看到的就是整块内容——不要因为「怕后面还有」而压成 0。"
                 if not hs["n_truncated"] else
                 f"正文只截前 {TEXT_HEAD_CHARS} 字，**本批有 {hs['n_truncated']} 行被截**"
                 f"（如 {hs['truncated_keys'][:4]}），要全文报 `qid#编号` 现场取。")
    by_q: dict[str, list[dict]] = {}
    for r in fill_rows:
        by_q.setdefault(r["qid"], []).append(r)
    lines = [
        "# 第二人标注工单",
        "",
        JUDGEMENT_LINE,
        "",
        "**怎么填**：只改最后一列 `1` 或 `0`，其它列别动。一个题的每一行都要填，不留 TODO。"
        f"本批 **{len(by_q)} 题 / {len(fill_rows)} 行**，逐题行数见各标题后。",
        "**嫌横向找列容易错行**：`uv run python scripts/m4_annotation_fill_helper.py --prefix <前缀> --emit`"
        " 会生成一张只有键的小表（`q-013#1=`），在等号后敲 0/1，再 `--merge` 写回——"
        "写回前自动备份整份，漏一行、填 `?`、键写错都会**被拒绝而不是被当成 0**。",
        f"**防假绿设计**：{outside_share:.0%} 的题里含第一人未标出的候选块，所以你有机会表示异议。",
        "**刻意不告诉你**第一人标了哪些——一起看就会被锚定，那测出来的不是独立性。",
        head_note,
        f"**候选来源**：{'稠密+词法' if hybrid else '服务端 BM25（词法）'}——"
        "用来定 gold 合适，不代表系统实际答题用的那五块。",
        "",
        "**细则 1 在本批几乎不会触发**：这 15 题是拆题后重出的，题面已收成**单一问点**，"
        "所以「必须同一块答全两问」这条只在块里出现**两件事、题只问一件**时用来防误读，不是用来加严。"
        "真正吃紧的是细则 2（别的项目/行业的同名章节）与细则 3（描述型块 vs 带强制性句子）。",
        "",
        "---",
        "",
    ]
    for qid in sorted(by_q):
        lines.append(f"## {qid}｜{qtext.get(qid, '')}")
        lines.append("")
        lines.append(f"共 {len(by_q[qid])} 行，全部要填。")
        lines.append("")
        lines.append(f"| # | 章节 | 页 | 正文起头 | {FILL_COL} |")
        lines.append("|---|---|---|---|---|")
        for r in by_q[qid]:
            lines.append(f"| {r['cand_no']} | {r['section_no']} | {r['page']} | {r['text_head']} | "
                         f"{(r.get(FILL_COL) or 'TODO').strip() or 'TODO'} |")
        lines.append("")
    return "\n".join(lines)


def _rewrite_worksheet_only(prefix: str, hybrid: bool) -> int:
    """人读版式改了（截断话术、行数提示）但工单内容一个字没动时用这条路径：
    不连 Milvus/PG、不重跑检索、**不重写 CSV**，所以不会碰到已填或待填的任何一格。"""
    from kee.eval.eval_data import QUESTIONS

    art = Path(get_settings().resolved_artifact_dir())
    csv_path = art / f"{prefix}_fill.csv"
    if not csv_path.exists():
        print(f"[缺工单] {csv_path.name} 不在；worksheet-only 只能重排版式，不能凭空造工单。")
        return 2
    lines = [l for l in csv_path.read_text(encoding="utf-8-sig").splitlines() if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    qtext = {q["qid"]: q["question"] for q in QUESTIONS}
    out = art / f"{prefix}_worksheet.md"
    filled = sum(1 for r in rows if (r.get(FILL_COL) or "").strip() in ("0", "1"))
    # 「有几成题含第一人未标的块」必须**算**出来。写死一个数（哪怕碰巧对上）就是给没人复核的断言配读数；
    # 没有 key 文件可算时这行宁可不出现。
    share = _outside_a_share(art / f"{prefix}_key.json")
    out.write_text(_worksheet(rows, qtext, share if share is not None else 0.0, hybrid), encoding="utf-8")
    print(f"只重排版式：{out.name}｜{len({r['qid'] for r in rows})} 题 / {len(rows)} 行｜已填 {filled} 行（未动）")
    print("防假绿读数：" + (f"{share:.0%} 的题含第一人未标的候选块（由 _key.json 聚合算出）"
                          if share is not None else "无 _key.json 可算 ⇒ 不写这句，不编数"))
    print(f"CSV 未被改写：{csv_path}")
    return 0


def _outside_a_share(key_path: Path) -> float | None:
    """只取聚合值：有几成题里存在 A 未标的块。**不输出任何逐行 A 标签**——
    逐行标签会让第二人锚定，聚合值不会。"""
    import json
    from collections import defaultdict

    if not key_path.exists():
        return None
    by: dict[str, list[int]] = defaultdict(list)
    for r in json.loads(key_path.read_text(encoding="utf-8")):
        by[r["qid"]].append(int(r.get("in_a_strict_gold") or 0))
    if not by:
        return None
    return sum(1 for v in by.values() if 0 in v) / len(by)


if __name__ == "__main__":
    sys.exit(main())
