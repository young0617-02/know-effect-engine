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
    args = ap.parse_args()

    s = get_settings()
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    rows = build_eval_set(pg, write_files=False)["questions"]
    pool = [q for q in rows if q["gold_chunk_ids"]]
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

    csv_path = art / "annotations_a2_fill.csv"
    blocked = _guard_existing_fill(csv_path, force=args.force)
    if blocked:
        print(f"[拒绝覆盖] {blocked}")
        kb.close()
        return 4

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fill_rows[0].keys()))
        w.writeheader()
        w.writerows(fill_rows)

    (art / "annotations_a2_key.json").write_text(json.dumps(key_rows, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    (art / "annotations_a2_worksheet.md").write_text(_worksheet(fill_rows, pool, outside_share, args.hybrid),
                                                     encoding="utf-8")
    print(f"工单：{len(pool)} 题 / {len(fill_rows)} 个候选块，待填列「{FILL_COL}」全为 TODO")
    print(f"含「第一人未标」候选的题占比：{outside_share:.0%}（闸值 {MIN_OUTSIDE_A_GOLD_SHARE:.0%}）")
    print(f"人读：{art / 'annotations_a2_worksheet.md'}")
    print(f"机填：{csv_path}")
    kb.close()
    return 0


def _worksheet(fill_rows: list[dict], pool: list[dict], outside_share: float, hybrid: bool) -> str:
    lines = [
        "# 第二人标注工单",
        "",
        JUDGEMENT_LINE,
        "",
        "**怎么填**：只改最后一列 `1` 或 `0`，其它列别动。一个题的每一行都要填，不留 TODO。",
        f"**防假绿设计**：{outside_share:.0%} 的题里含第一人未标出的候选块，所以你有机会表示异议。",
        "**刻意不告诉你**第一人标了哪些——一起看就会被锚定，那测出来的不是独立性。",
        f"正文只截前 {TEXT_HEAD_CHARS} 字；要哪一条的全文，报 `qid#编号` 即可现场取。",
        f"**候选来源**：{'稠密+词法' if hybrid else '服务端 BM25（词法）'}——"
        "用来定 gold 合适，不代表系统实际答题用的那五块。",
        "",
        "---",
        "",
    ]
    by_q: dict[str, list[dict]] = {}
    for r in fill_rows:
        by_q.setdefault(r["qid"], []).append(r)
    qtext = {q["qid"]: q["question"] for q in pool}
    for qid in sorted(by_q):
        lines.append(f"## {qid}｜{qtext.get(qid, '')}")
        lines.append("")
        lines.append(f"| # | 章节 | 页 | 正文起头 | {FILL_COL} |")
        lines.append("|---|---|---|---|---|")
        for r in by_q[qid]:
            lines.append(f"| {r['cand_no']} | {r['section_no']} | {r['page']} | {r['text_head']} | TODO |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
