"""算第二人标注的一致性（Cohen's Kappa），双口径 + 允许半成品 + 把分歧摊开。

为什么不直接用 `kee.eval.kappa`：那个模块把 A 侧固定成 section 级 gold，
而 M2 已经证明 section 级是把错尺子（gold=13 的题 Recall@5 上限只有 5/13）。
第二人按「这块正文里到底含不含答案」判，天然接近 strict 口径，
拿 section gold 去比，算出来的分歧里**混着口径差异**，不是人在不一致。
所以这里两个口径都算，并排报出来。

三件刻意的设计：

1. **允许只标了一部分就算**。计划是先做 25 题停一次，看判据对不对。
   若必须全填完才出数，中途就无法发现问题，只能标完 63 题再返工。
   未填完时输出 `complete: false` 与进度，数字照算但明确标为「部分」。
2. **Kappa 的适用边界写死在输出里**：观测域是**候选池**（第二人只看过这些块），
   不是全库。所以这个数是「在候选池上的一致性」，不能写成「标注全库一致」。
   这句话不写，Kappa 会被当成比它实际含义更强的证据。
3. **分歧按 (section, strict, B) 三元组分桶摊开**，而不是只按「A 多/少标」两个方向：
   - `100`：只有 section 认，strict 与第二人都不认 → section 口径注水的直接证据（M2 怀疑的那件事）
   - `001`：两个口径都不认、第二人认 → **gold 漏标**，这方向更危险，检索指标会把
     「其实答对了」的块判成错；扩样本之前这个数必须知道
   - `110` / `101`：三人里两个认、一个不认 → 真分歧，成因要读原文才能裁
     （gold 跨项目扩收、判据没覆盖的问法、第二人判松，三类都会落到这里）
   桶名只描述形状不写结论：上一版把 `110` 也算进「注水」，等于让计数替人工裁决。

用法：uv run python scripts/m4_kappa.py [--fail-if-incomplete]
退出码：0 正常；2 一题都没填；填完且 --fail-if-incomplete 时未完成返回 1。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.eval.kappa import cohens_kappa  # noqa: E402

FILL_COL = "b_directly_answers__fill_1_or_0"
SENTINEL = "todo"
# 疑问标记计数：这套题「一问两问」只挂一个问号（例：表决逻辑是什么，能不能改整定值？），
# 按问号数会漏（实测只数出 1 题，按标记数是 11 题）。判据缺规则那类分歧要靠它显形。
INTERROGATIVE = re.compile(r"什么|哪些|怎么|怎样|多久|多少|能不能|可不可以|是否|依据哪|哪个|哪几|吗")


def load(art: Path, prefix: str = "annotations_a2") -> tuple[list[dict], dict[tuple[str, int], dict]]:
    """支持逗号分隔的多张工单合并成一个观测域（同一把尺子下分批收的标注该能合起来算）。
    每行带上它来自哪张工单，剔除才能精确到「某工单的某题」而不是全局按题号砍。"""
    rows: list[dict] = []
    key: dict[tuple[str, int], dict] = {}
    for one in [x.strip() for x in prefix.split(",") if x.strip()]:
        with (art / f"{one}_fill.csv").open(encoding="utf-8-sig", newline="") as f:
            part = list(csv.DictReader(f))
        for r in part:
            r["_src"] = one
        rows += part
        for x in json.loads((art / f"{one}_key.json").read_text(encoding="utf-8")):
            x["_src"] = one
            key[(one, x["qid"], int(x["cand_no"]))] = x
    qp = art / "qrels.jsonl"
    if qp.exists():
        cur = {}
        for line in qp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cur[rec["qid"]] = (set(rec["gold_chunk_ids"]), set(rec.get("gold_strict_ids") or []))
        stale = 0
        for k, v in key.items():
            pair = cur.get(k[1])
            if not pair:
                continue
            cid = v["chunk_id"]
            new_s, new_t = int(cid in pair[0]), int(cid in pair[1])
            if (int(v["in_a_section_gold"]), int(v["in_a_strict_gold"])) != (new_s, new_t):
                stale += 1
                v["in_a_section_gold"], v["in_a_strict_gold"] = str(new_s), str(new_t)
        v_gold = {"source": "qrels.jsonl（当前尺子）", "rows_relabeled_from_snapshot": stale}
    else:
        v_gold = {"source": f"{prefix}_key.json（工单快照，可能滞后于 gold 修订）",
                  "rows_relabeled_from_snapshot": 0}
    return rows, key, v_gold


def load_questions(art: Path) -> dict[str, dict]:
    """从 qrels.jsonl 读题面（**不开 PG 也能读**：分歧行的证据必须能离线复看）。"""
    p = art / "qrels.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def cluster_bootstrap(filled: list[tuple], draws: int = 2000,
                      seed: int = 7) -> dict[str, Any]:
    """按**题**聚类重采样，一次抽样同时给三个区间：两个口径的 Kappa 与它们的差。

    重采样单元是题不是行：同一题的 8–15 个候选块共用一个题面，彼此不独立，
    按行自助会把区间算得比实际窄。

    差值必须在**同一批重采样**里算（每次抽样同时算出 section 与 strict 两个 Kappa 再相减），
    否则得到的是「两条区间碰巧分开」这种弱得多的证据——两个独立区间的间隙并不等价于差值显著。
    退化抽样（某次重采样的题全正或全负，Kappa 无定义）单独计数，不混进分位数。
    """
    by_q: dict[str, list[tuple]] = {}
    for row in filled:
        by_q.setdefault(row[0], []).append(row)
    qids = sorted(by_q)
    rng = random.Random(seed)
    ks, kt, diff, degenerate = [], [], [], 0
    for _ in range(draws):
        sample = [r for q in (rng.choice(qids) for _ in qids) for r in by_q[q]]
        b_labels = [r[2] for r in sample]
        s = cohens_kappa([r[3] for r in sample], b_labels)["kappa"]
        t = cohens_kappa([r[4] for r in sample], b_labels)["kappa"]
        if s != s or t != t:  # NaN：该次重采样只剩一个类别
            degenerate += 1
            continue
        ks.append(s)
        kt.append(t)
        diff.append(t - s)
    if not diff:
        return {"note": "全部重采样退化，无法给区间", "degenerate_draws": degenerate}

    def pct(vals: list[float]) -> tuple[float, float]:
        v = sorted(vals)
        return round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)

    lo_s, hi_s = pct(ks)
    lo_t, hi_t = pct(kt)
    lo_d, hi_d = pct(diff)
    return {
        "unit": "题（聚类重采样）", "draws": len(diff), "degenerate_draws": degenerate,
        "section_ci95": [lo_s, hi_s], "strict_ci95": [lo_t, hi_t],
        "diff_strict_minus_section": {"point": round(sum(diff) / len(diff), 4),
                                      "ci95": [lo_d, hi_d],
                                      "share_draws_positive": round(
                                          sum(1 for d in diff if d > 0) / len(diff), 4)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-if-incomplete", action="store_true")
    ap.add_argument("--prefix", default="annotations_a2",
                    help="读 <prefix>_fill.csv + <prefix>_key.json；重标轮是 annotations_a2_relabel")
    ap.add_argument("--out", default="kappa_a2.json",
                    help="产物文件名。**重标轮必须另起名**——kappa_a2.json 是 M4 已收证据，同名会静默覆盖")
    ap.add_argument("--already-yi", action="store_true",
                    help="本轮工单抬头写的就是乙口径（第二人按乙填），不再机械把「部分回答」压成 0；"
                         "首轮工单抬头是甲，默认仍走机械改判")
    ap.add_argument("--exclude-qids", default="",
                    help="把这些题从一致性观测域里剔除（只在「题面已改、旧行判的不是同一件事」时用），"
                         "必须配 --exclude-note 说明理由，否则拒绝")
    ap.add_argument("--exclude-note", default="",
                    help="剔除理由，会原样写进产物——没有理由的剔除就是择优取样本")
    ap.add_argument("--retest-vs", default="",
                    help="与另一张工单比**同一人跨轮**的判断（重测信度），按 (qid, chunk_id) 对齐；"
                         "这不是双人一致性，是同一个标注者的自一致——它给出所有 Kappa 的噪声地板")
    args = ap.parse_args()
    if args.exclude_qids and not args.exclude_note.strip():
        print("[拒绝] 剔除题目必须给 --exclude-note：没有书面理由的剔除等于挑好看的样本")
        return 2

    art = Path(get_settings().resolved_artifact_dir())
    rows, key, gold_version = load(art, args.prefix)
    if not rows:
        print("没有工单，先跑 scripts/m4_build_annotation_sheet.py")
        return 2

    filled: list[tuple[str, int, int, int, int, str]] = []  # qid, cand_no, b, a_section, a_strict, doc_id
    blank_qids: list[str] = []
    for r in rows:
        raw = (r.get(FILL_COL) or "").strip().lower()
        if not raw or raw.startswith(SENTINEL):
            blank_qids.append(r["qid"])
            continue
        if raw not in ("0", "1"):
            print(f"[判废] {r['qid']}#{r['cand_no']} 填了 {raw!r}，只接受 1 或 0")
            return 2
        k = key[(r["_src"], r["qid"], int(r["cand_no"]))]
        filled.append((r["qid"], int(r["cand_no"]), int(raw),
                       int(k["in_a_section_gold"]), int(k["in_a_strict_gold"]), r["doc_id"],
                       r["_src"]))

    # 剔除可以精确到「某张工单的某题」——q-071 在两张工单里都有，但那是两个不同的题面，
    # 全局按题号砍会把新问法那 9 行也一起剔掉。写法：`qid` 对所有工单生效，`工单名/qid` 只剔那一张。
    ex_items = [x.strip() for x in args.exclude_qids.split(",") if x.strip()]
    ex_qids = {e for e in ex_items if "/" not in e}
    ex_pairs = {(e.split("/", 1)[0], e.split("/", 1)[1]) for e in ex_items if "/" in e}

    def dropped(f) -> bool:
        return f[0] in ex_qids or (f[6], f[0]) in ex_pairs

    excluded_rows = sum(1 for f in filled if dropped(f))
    if ex_items:
        filled = [f for f in filled if not dropped(f)]
    excluded = set(ex_items)

    if not filled:
        print("一题都没填，无法计算。建议先填前 25 题跑一次，确认判据没跑偏再续标。")
        return 2

    done_qids = {f[0] for f in filled}
    total_qids = {r["qid"] for r in rows}
    complete = not blank_qids

    out = {
        "unit": "(qid, 候选块) 的相关判定",
        "scope_note": ("观测域是**候选池**（第二人只看过这些块），不是全库。"
                       "这个 Kappa 是「在候选池上的一致性」，不得表述成全库标注一致。"),
        "n_rows_judged": len(filled),
        "sources": sorted({f[6] for f in filled}),
        "n_questions_done": len(done_qids),
        "n_questions_total": len(total_qids),
        "complete": complete,
        "blank_rows": len(blank_qids),
        "excluded": ({"spec": sorted(excluded), "rows_dropped": excluded_rows,
                      "reason": args.exclude_note.strip()} if excluded else None),
        "gold_version": gold_version,
    }

    for name, idx in (("against_section_gold", 3), ("against_strict_gold", 4)):
        la = [f[idx] for f in filled]
        lb = [f[2] for f in filled]
        res = cohens_kappa(la, lb)
        agree = sum(1 for a, b in zip(la, lb) if a == b)
        res["disagreement"] = {
            # A=1/B=0：第一人多标（section gold 注水方向）
            "a_yes_b_no": sum(1 for a, b in zip(la, lb) if a == 1 and b == 0),
            # A=0/B=1：第一人**漏标**——评测集 gold 有缺口的信号，比上一类更要紧
            "a_no_b_yes": sum(1 for a, b in zip(la, lb) if a == 0 and b == 1),
        }
        res["raw_agreement"] = round(agree / max(len(la), 1), 4)
        res["b_positive_rate"] = round(sum(lb) / max(len(lb), 1), 4)
        out[name] = res
    out["bootstrap"] = cluster_bootstrap(filled)

    # 重测信度：同一个第二人、隔几轮、题面微调后重判同一批块。
    # 为什么单列：所有 Kappa 都默认「标注者对自己上一轮的判断是稳的」，这条假设从没量过。
    # 翻标的比例就是这台测量仪的噪声地板——它比任何一次「A 与 B 差多少」都更该先知道。
    if args.retest_vs:
        base = {r["qid"] + "|" + r["chunk_id"]: (r.get(FILL_COL) or "").strip() for r in rows}
        with (art / f"{args.retest_vs}_fill.csv").open(encoding="utf-8-sig", newline="") as f:
            prior = {r["qid"] + "|" + r["chunk_id"]: (r.get(FILL_COL) or "").strip()
                     for r in csv.DictReader(f)}
        shared = sorted(set(base) & set(prior))
        pair = [(int(prior[k]), int(base[k])) for k in shared if prior[k] in ("0", "1") and base[k] in ("0", "1")]
        flips = [{"qid": k.split("|")[0], "chunk_id": k.split("|")[1], "then": prior[k], "now": base[k]}
                 for k in shared if prior[k] in ("0", "1") and base[k] in ("0", "1") and prior[k] != base[k]]
        rt = cohens_kappa([a for a, _ in pair], [b for _, b in pair]) if pair else {}
        out["retest"] = {
            "against": args.retest_vs,
            "shared_rows": len(pair),
            "kappa_self": rt.get("kappa"), "po_self": rt.get("po"),
            "flipped": flips,
            "note": ("同一人跨轮的自一致，不是双人一致性；它给的是这台测量仪的噪声地板。"
                     "用它约束结论：分歧数小于这个地板的『改善』不能声称成立。"),
        }

    # ---- 裁决后的判据（乙）：一块要算 1，必须能独立答全该题的全部内容 ----
    # 第二人当时是按甲填的，所以乙的改判是**机械的、不需要人工重标**：
    # 两问题上「不在 strict gold 里却给了 1」的块 → 0（那些是部分回答，已逐行读过原文确认）；
    # 在金块里给的 1 **保留**——本语料的章节是聚合写的，那块本来就答全了两问。
    qtexts_c = load_questions(art)

    def parts_of(qid: str) -> int:
        return len(INTERROGATIVE.findall((qtexts_c.get(qid) or {}).get("question", "")))

    def yi_label(qid: str, b: int, a_t: int) -> int:
        if args.already_yi:
            return b
        return 0 if (parts_of(qid) >= 2 and a_t == 0) else b

    gold_t = [f[4] for f in filled]
    b_jia = [f[2] for f in filled]
    b_yi = [yi_label(f[0], f[2], f[4]) for f in filled]

    # 细则 2 的文档级补充（2026-10-12 裁决）：题面点名单一文档类型的题，第二人判「是」的块
    # 若不属那份文档，按判据应为 0。这条**会把人判的 1 改成 0、方向对第一人有利**，
    # 所以两版必须并排报，且只用在相关性叙述里，不进 `m2_eval` 的门禁判定。
    from kee.eval.eval_data import DOC_SCOPED_QIDS

    def doc_scoped(label: int, qid: str, doc_id: str) -> int:
        kind = DOC_SCOPED_QIDS.get(qid)
        return 0 if (kind and label == 1 and kind not in (doc_id or "")) else label

    b_doc = [doc_scoped(y, f[0], f[5]) for y, f in zip(b_yi, filled)]
    doc_flipped = [{"qid": f[0], "cand_no": f[1], "doc_id": f[5],
                    "named": DOC_SCOPED_QIDS.get(f[0])}
                   for y, f in zip(b_yi, filled) if doc_scoped(y, f[0], f[5]) != y]

    def snapshot(labels: list[int]) -> dict[str, Any]:
        res = cohens_kappa(gold_t, labels)
        return {
            "kappa_vs_strict": res["kappa"], "po": res["po"],
            "positive_b": sum(labels), "positive_gold": sum(gold_t),
            "both_positive": sum(1 for a, b in zip(gold_t, labels) if a == 1 and b == 1),
            "both_negative": sum(1 for a, b in zip(gold_t, labels) if a == 0 and b == 0),
        }

    snap_jia, snap_yi = snapshot(b_jia), snapshot(b_yi)
    snap_doc = snapshot(b_doc)

    # ---- 测量面（用户 2026-10-12 裁决：敏感题不测相关性，只测路由）----
    # 理由必须是结构性的、先于数值：`escalate`/`refuse` 题按设计就没有「该对外给的答案块」，
    # 逐块问「这块能不能直接答」对它们是范畴错误——能直接答的那块恰好是不该给的内容。
    # 但读数**两份都报**，且不拿高的那份对外：按题型层层剔除会让 Kappa 一路走高，
    # 那是择优取样本的斜坡，不是质量提升。
    def scope_sel(behaviors: set) -> list[tuple]:
        return [f for f in filled
                if (qtexts_c.get(f[0]) or {}).get("expected_behavior", "answer") not in behaviors]

    def snap_for(sub: list[tuple]) -> dict[str, Any]:
        la = [f[4] for f in sub]
        lb = [doc_scoped(yi_label(f[0], f[2], f[4]), f[0], f[5]) for f in sub]
        res = cohens_kappa(la, lb)
        return {"rows": len(sub), "questions": len({f[0] for f in sub}),
                "kappa_vs_strict": res["kappa"], "po": res["po"],
                "positive_b": sum(lb), "positive_gold": sum(la),
                "both_positive": sum(1 for a, b in zip(la, lb) if a == 1 and b == 1)}

    out["relevance_scope"] = {
        "decided": ("敏感/拒答题（expected_behavior=escalate|refuse）不参与相关性测量，按 ⑤ 的路由正确性判；"
                    "剔除理由是其结构（按设计就没有可给的答案块），不是数值。两份读数并列，对外只用「含敏感题」那份"),
        "including_sensitive": snap_for(filled),
        "excluding_sensitive": snap_for(scope_sel({"escalate", "refuse"})),
        "ladder_warning": ("按题型剔除会一路把 Kappa 抬上去（含敏感 < 不含敏感 < 再剔跨文档）。"
                           "**跨文档类不得剔除**：那是本系统最难也最值钱的能力，剔它就是为了好看的数请掉真问题。"),
    }
    out["criterion"] = {
        "decided": ("乙：一块必须能独立答全该题全部内容才算 1（2026-10-12 用户裁决）。"
                    "细则另两条：题面点名主体时别的项目/行业的同名章节算 0；描述型块（部署几台、达成率）算 0"),
        "label_basis": ("本轮工单抬头写的就是**乙**，第二人按乙填 ⇒ 不再机械改判（`--already-yi`）。"
                        "as_filled 与 adjudicated 两行因此相同，读 as_filled 那份即可"
                        if args.already_yi else
                        "工单抬头是甲，乙口径的改判是机械的（两问题上「不在 strict gold 却判 1」→ 0）"),
        "as_filled_jia": snap_jia,
        "adjudicated_yi": snap_yi,
        "adjudicated_yi_with_doc_scope": snap_doc,
        "part_answer_rows_zeroed": sum(1 for j, y in zip(b_jia, b_yi) if j == 1 and y == 0),
        "doc_scope_amendment": {
            "rule": ("细则 2 的文档级补充（2026-10-12 裁决）：题面点名单一文档类型的题（`DOC_SCOPED_QIDS`，"
                     "现 9 道），第二人判是但块不属那份文档 ⇒ 按判据为 0"),
            "applied_to_qids": sorted(DOC_SCOPED_QIDS),
            "q-044_excluded": "题面说「规范里」而它的两块 gold 在售前方案与运维手册里——"
                              "那是限定语与答题块不同源的**题面缺陷**，不入表；硬配 doc_scope 等于用判据把第一人自己的 gold 清零",
            "rows_flipped": doc_flipped,
            "kappa_before": snap_yi["kappa_vs_strict"],
            "kappa_after": snap_doc["kappa_vs_strict"],
            "direction_warning": ("这条补充把第二人的 1 改成 0、方向对第一人有利，且规则是在看过分歧之后才补的。"
                                  "所以两版必须并排报，**且只用于相关性叙述，不进 `m2_eval` 的门禁判定**；"
                                  "对外引用时以「补充前」为保守值、「补充后」为判据现值，不得只报高的那份"),
        },
        "uplift_warning": (
            "本轮抬头即乙口径，甲/乙两行数值相同，**不存在抬升**，也就没有「负例变多冒充一致性」的空间"
            if args.already_yi else
            f"Kappa {snap_jia['kappa_vs_strict']} → {snap_yi['kappa_vs_strict']} 的抬升**不能读成两人更合**："
            f"双正例 {snap_jia['both_positive']} → {snap_yi['both_positive']}（一个没多），"
            f"双负例 {snap_jia['both_negative']} → {snap_yi['both_negative']}——抬幅全部来自负例变多。"
            "这与复合分那次的零方差标签是同一类陷阱：可判定的方差变小了，一致率自然变高。"
        ),
        "split_check": ("**2026-10-12 全量复核后作废**此前写在此处的「拆题是空集」：那句靠「≥2 个疑问标记」"
                        "的检测器得出，而集合型题往往只有一个问号。实测定为必须拆 3 道——"
                        "q-053（哪些文档引用过）、q-055（三条线各自门限）、q-058（两份文档口径一致吗）；"
                        "q-054 的第二问「各项目是怎么用的」同属集合型。q-056/057 经核单块可答，不拆。"
                        "检测器 scripts/m4_split_candidates.py 的输出只当候选面，结论要逐题人工读。"),
    }

    # 分歧按 (a_section, a_strict, b) 三元组分桶。
    # 上一版把桶名写成「A 多标（strict 也不认则为注水）」，可代码只判了 section 一个条件，
    # 于是「两个 gold 都认、B 判否」这种**真分歧**被并进了「注水」桶——桶名替数据下了结论。
    # 现在桶只描述三元组的形状；成因（gold 扩收 / gold 漏标 / 判据缺口）要逐行读原文才能裁，
    # 不是一次计数能给的。
    bucket = {
        (1, 0, 0): "section 独认（strict 与 B 都不认）",
        (1, 1, 0): "双 gold 认、B 判否",
        (1, 0, 1): "section 与 B 认、strict 不认",
        (0, 0, 1): "两口径都不认、B 判是（gold 缺口方向）",
    }
    samples: list[dict] = []
    joint: Counter = Counter()
    row_by_key = {(r["_src"], r["qid"], int(r["cand_no"])): r for r in rows}
    qtexts = load_questions(art)
    for qid, no, b, a_s, a_t, doc_i, src_name in filled:
        joint[(a_s, a_t, b)] += 1
        kind = bucket.get((a_s, a_t, b))
        if not kind:
            continue
        rec = qtexts.get(qid) or {}
        qtext = rec.get("question", "")
        src = row_by_key[(src_name, qid, no)]
        samples.append({
            "src": src_name, "qid": qid, "cand_no": no, "b": b, "a_section": a_s, "a_strict": a_t, "kind": kind,
            # 下面这些字段是为了让「成因」能在 artifact 里被复看，而不是只活在某次对话里
            "question": qtext,
            "n_sub_questions": len(INTERROGATIVE.findall(qtext)) if qtext else 0,
            "doc_id": src["doc_id"], "chunk_id": src["chunk_id"],
            "reference_answer": rec.get("reference_answer", ""),
            "chunk_text": (src["text_head"] or "")[:320],
        })
    out["joint_section_strict_b"] = {"".join(map(str, k)): v for k, v in sorted(joint.items())}
    out["n_divergence"] = len(samples)
    # 「判据缺规则」的直接证据：分歧行里有多少落在一问两问的题上（不是人不一致，是我没写规则）
    out["divergence_on_two_part_questions"] = sum(1 for s in samples if s["n_sub_questions"] >= 2)
    out["divergence_samples"] = samples
    out["kind_counts"] = dict(Counter(s["kind"] for s in samples))

    (art / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    s_, t_ = out["against_section_gold"], out["against_strict_gold"]
    status = "（完整）" if complete else f"（未完成，还有 {out['blank_rows']} 行空着）"
    print(f"已判 {out['n_rows_judged']} 行 / 覆盖 {out['n_questions_done']}/{out['n_questions_total']} 题{status}")
    if out["excluded"]:
        e = out["excluded"]
        print(f"  ⚠ 剔除观测域：{e['spec']}（{e['rows_dropped']} 行）——理由：{e['reason']}")
    print(f"  Kappa vs section gold : {s_.get('kappa')}  po={s_.get('po')} 分歧 A多标={s_['disagreement']['a_yes_b_no']} A漏标={s_['disagreement']['a_no_b_yes']}")
    print(f"  Kappa vs strict gold  : {t_.get('kappa')}  po={t_.get('po')} 分歧 A多标={t_['disagreement']['a_yes_b_no']} A漏标={t_['disagreement']['a_no_b_yes']}")
    print(f"  分歧分类: {out['kind_counts']}（共 {out['n_divergence']} 行；三元组分布 {out['joint_section_strict_b']}）")
    bs = out["bootstrap"]
    d = bs["diff_strict_minus_section"]
    print(f"    区间(按题聚类重采样 {bs.get('draws')} 次，退化 {bs.get('degenerate_draws')} 次): "
          f"section {bs['section_ci95']} / strict {bs['strict_ci95']} | "
          f"差值 strict−section = {d['point']}，CI {d['ci95']}，{d['share_draws_positive']:.0%} 的重采样里 strict 更高")
    c = out["criterion"]
    print(f"  判据（已裁决）: {c['decided']}")
    print(f"    按甲填的原样 vs strict = {c['as_filled_jia']['kappa_vs_strict']}"
          f"（正例 {c['as_filled_jia']['positive_b']}、双正例 {c['as_filled_jia']['both_positive']}）；"
          f"改判乙后 = {c['adjudicated_yi']['kappa_vs_strict']}"
          f"（正例 {c['adjudicated_yi']['positive_b']}、双正例 {c['adjudicated_yi']['both_positive']}、"
          f"机械改判 {c['part_answer_rows_zeroed']} 行）")
    print(f"    ⚠ {c['uplift_warning']}")
    ds = c["doc_scope_amendment"]
    flipped_keys = "、".join(f"{r['qid']}#{r['cand_no']}" for r in ds["rows_flipped"]) or "无"
    print(f"  细则 2 文档级补充：改判 {len(ds['rows_flipped'])} 行（{flipped_keys}），"
          f"vs strict {ds['kappa_before']} → {ds['kappa_after']}（方向对第一人有利，两版并排报，不进 m2_eval 门禁）")
    print(f"    拆题核对：{c['split_check']}")
    rs = out["relevance_scope"]
    inc, exc = rs["including_sensitive"], rs["excluding_sensitive"]
    print(f"  相关性测量面（已裁决）: 敏感/拒答题只测路由，不测相关性；两份读数并列，**对外只用含敏感题那份**")
    print(f"    含敏感题 {inc['questions']} 题 / {inc['rows']} 行 → vs strict {inc['kappa_vs_strict']}"
          f"（正例 B {inc['positive_b']}／gold {inc['positive_gold']}，双正例 {inc['both_positive']}）")
    print(f"    剔除敏感 {exc['questions']} 题 / {exc['rows']} 行 → vs strict {exc['kappa_vs_strict']}"
          f"（正例 B {exc['positive_b']}／gold {exc['positive_gold']}，双正例 {exc['both_positive']}）")
    print(f"    ⚠ {rs['ladder_warning']}")
    gv = out["gold_version"]
    print(f"  gold 来源: {gv['source']}"
          f"{'；从工单快照改判 ' + str(gv['rows_relabeled_from_snapshot']) + ' 行' if gv['rows_relabeled_from_snapshot'] else ''}")
    if gv["rows_relabeled_from_snapshot"]:
        print("    ⚠ 本轮 gold 已修订（#41），修订方向与第二人判断同向 ⇒ 修订后的一致率**不是独立证据**，"
              "只能读成「第一人按复核收窄了 gold」；独立证据是修订前那份（pre_goldfix/kappa_a2.json）")
    if out.get("retest"):
        rt = out["retest"]
        keys = "、".join(f"{f['qid']}#{f['chunk_id'].split('#')[-1]}" for f in rt["flipped"]) or "无"
        print(f"  重测信度（同一人跨轮 vs {rt['against']}）：共有 {rt['shared_rows']} 行、翻标 {len(rt['flipped'])} 行"
              f"（{keys}），自身 Kappa {rt['kappa_self']} / po {rt['po_self']}")
        print(f"    ⚠ {rt['note']}")
    print("  口径提醒:", out["scope_note"])
    print(f"  判据读数基准: {out['criterion']['label_basis']}")
    print(f"证据： {art / args.out}")

    if args.fail_if_incomplete and not complete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
