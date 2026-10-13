"""#58 检索缺口的**归因**测量：把「gold 没进前 5」拆成三种完全不同的病。

为什么要这支脚本：M2 一直报 Recall@5 / Hit@5，`m2_eval.json` 的 per_question 只有
`{qid, recall, recall_strict, hit, n_gold, category, split}`——**没进前 5 的 gold 到底排第几、
有没有进过候选池、与第 1 名差多少分，全都看不出来**。看不出来就只能猜，
而猜出来的修法（扩池 / 加改写腿 / 改词法映射）花的代价差一个量级。

诊断按**成员关系**推，先后可判定性（每个病因对应不同代价的修法）：
  hit         生产前 5 就有 gold
  misranked   生产池里有、但排在第 5 名之外 ⇒ 扩池无效，是**排序/证据表述**问题
  needs_rc    只有把 `rerank_candidates` 抬到 40 才看得见 ⇒ 花的是**重排时间**
  needs_legs  只有把两条腿各取 40 才看得见 ⇒ 花的是**向量库与 BM25 的取回量**
  needs_both  只有两个旋钮一起动才看得见（宽配置）⇒ 单独修哪个都不够
  absent      宽池里也没有 ⇒ 只能靠**查询侧**（改写/同义词）

第一版把 needs_* 一律写成 `cut_by_pool`，因为宽配置同时动了两个旋钮；实测 rc=40 单跑**不翻转**，
才看清真因在腿宽。**混着跑会把修法指到不生效的那一个上**——这就是两条 what-if 必须分开跑的理由。

外加一条对「口语化查询 vs 书面证据」这个假说的**直接检验**：
逐题算题面与 gold 块 / 与当届第 1 名的词法重合度（jieba 分词 + 字符二元组两种量法），
如果 miss 题上「gold 重合度 < 第 1 名重合度」成片出现，假说成立；否则就得换解释。

两遍跑的用法（这是关键，不能只跑一遍）：
  A 生产配置（top_k=5、rerank_candidates=20、dense/sparse 各 20）→ 拿到**线上真实结果**；
  B 宽配置（top_k=50、rerank_candidates=50、dense/sparse 各 40）→ 拿到池成员关系与完整重排序。
  「A 里没有而 B 里有」= cut_by_pool；「B 里也没有」= absent。
  注意 B 的读数**不能**当成 Recall@5 报——top_k 变了 k 就变了，那不是同一把尺。

前置校验照 #56 的教训来：依赖不健康 / 检索降级腿 / 缓存命中，任一出现就拒绝出数。
分数只在**同一题内**比较（sigmoid 后的重排分是 query-pair 尺度，跨题平均没有意义）。

跑法：uv run python scripts/m2_retrieval_gap.py [--limit N] [--out NAME]
产物：var/gate0/m2_retrieval_gap.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jieba  # noqa: E402

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set, load_children  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）()【】\[\]《》<>|｜/\\\-_+*#&@~^$,.?:;!\"'“”‘’]+")
HEADING_PREFIX = re.compile(r"^[^｜|]{0,80}｜")  # chunking 给块加的前缀，会污染词法重合度


def tokens(text: str) -> set[str]:
    text = HEADING_PREFIX.sub("", text or "")
    return {w.strip() for w in jieba.lcut(text) if w.strip() and not PUNCT.fullmatch(w.strip())}


def bigrams(text: str) -> set[str]:
    s = "".join((text or "").split())
    s = HEADING_PREFIX.sub("", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def overlap(a: str, b: str) -> float:
    """重合度 = |交集| / min(|A|,|B|)。用 min 而不是并集：题面本来就比块短，
    用 Jaccard 会因为长度差把什么都压到 0.0x，看不出差异。"""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / min(len(ta), len(tb)), 4)


def bg_overlap(a: str, b: str) -> float:
    ta, tb = bigrams(a), bigrams(b)
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / min(len(ta), len(tb)), 4)


def mk_retriever(store, sched, *, top_k: int, rc: int, legs: int, mode: str = "hybrid",
                 rrf_k: int | None = None) -> HybridRetriever:
    kw = {} if rrf_k is None else {"rrf_k": rrf_k}
    return HybridRetriever(store, scheduler=sched, cache=None, mode=mode, use_rerank=True,
                           rerank_candidates=rc, dense_top_k=legs, sparse_top_k=legs,
                           rewrite_mode="off", **kw)


def explain(store, sched, q, ctx, limit=50):
    """机制探针：gold 到底卡在**哪条腿**的哪一名。
    三种查询同源对比：
      原查询        —— 用户真会打的那句（口语）
      原查询 + gold_keywords —— **上限检验，不是可上线的功能**：gold_keywords 是从 gold 正文里取的词，
                                用它等于偷看答案。它只回答一个问题：「如果查询里有书面语的词，检得到吗？」
                                答"是"⇒ 瓶颈在词法（值得做同义词/改写）；答"否"⇒ 连词都对上也排不上，是排序/embedding 的问题。
      只查关键词    —— 拆掉口语外壳后剩下的那串术语
    """
    from kee.retrieval.fusion import rrf_fuse
    from kee.retrieval.rbac import build_filter_expr

    expr = build_filter_expr(ctx)
    gold = set(q["gold_chunk_ids"])
    variants = {"原查询": q["question"],
                "原查询+gold_keywords": q["question"] + " " + " ".join(q.get("gold_keywords") or []),
                "只查关键词": " ".join(q.get("gold_keywords") or [])}
    out = {}
    for name, text in variants.items():
        with sched.use("embed"):  # 与 pipeline._dense_hits 同一取法：模型是懒加载的
            vec = sched.embedder.embed_query(text)
        d = store.search_dense(vec, limit=limit, expr=expr)
        s = store.search_sparse(text, limit=limit, expr=expr)
        rd = {h.chunk_id: i + 1 for i, h in enumerate(d)}
        rs = {h.chunk_id: i + 1 for i, h in enumerate(s)}
        fused = rrf_fuse([d, s], k=60)
        rf = {h.chunk_id: i + 1 for i, (h, _) in enumerate(fused)}
        out[name] = {"query": text[:60], "per_gold": [
            {"chunk_id": g, "dense_rank": rd.get(g), "sparse_rank": rs.get(g), "rrf_rank": rf.get(g)}
            for g in sorted(gold)]}
        print(f"    [{name}] " + " ｜ ".join(
            f"{v['chunk_id'].split('#')[0]}: 稠密#{v['dense_rank']} BM25#{v['sparse_rank']} 融合#{v['rrf_rank']}"
            for v in out[name]["per_gold"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--out", default="m2_retrieval_gap.json")
    ap.add_argument("--explain", default="",
                    help="只做机制探针（逗号分隔 qid），不跑全量：gold 卡在哪条腿的哪一名")
    args = ap.parse_args()

    s = get_settings()
    pg = AuthoritativeStore()
    built = build_eval_set(pg, write_files=False)
    qs = [q for q in built["questions"]
          if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]]
    if args.limit:
        qs = qs[: args.limit]
    children = {c["id"]: c for c in load_children(pg)}
    qhash = zlib.crc32("|".join(f"{q['qid']}::{q['question']}"
                                for q in sorted(qs, key=lambda x: x["qid"])).encode("utf-8"))

    store = KbStore(s.milvus_collection, "jieba")
    # 前置校验①：集合里到底有多少块。空/骤减时后面每个数都是假的（#56 同一类防线）
    n_vec = store.count()
    if n_vec < 100:
        print(f"[拒绝起跑] 集合 {s.milvus_collection} 只有 {n_vec} 个向量，索引不像建好过——"
              "在这种状态下算出来的「缺口」全是重建事故，不是检索质量")
        return 2
    sched = ResidencyScheduler(policy="co_resident")

    narrow = mk_retriever(store, sched, top_k=5, rc=20, legs=20)     # 与生产同一套参数
    wide = mk_retriever(store, sched, top_k=50, rc=50, legs=40)      # 只为看池成员与完整排序
    # 单因子 what-if **两条**，各自只动一个旋钮：
    #   rc40  ：只把 `rerank_candidates` 20→40（重排看得到更多融合结果）
    #   legs40：只把 `dense_top_k`/`sparse_top_k` 20→40（两条腿各多取一倍，融合列表才更长）
    # 为什么必须分开：宽配置同时动了两个旋钮，第一次我把 q-044 报成 cut_by_pool，
    # 而 rc=40 单跑**根本没有翻转**——那说明真因在腿宽不在候选上限。混着跑就会把修法指错。
    whatif_rc = mk_retriever(store, sched, top_k=5, rc=40, legs=20)
    whatif_legs = mk_retriever(store, sched, top_k=5, rc=20, legs=40)
    # 第三个是**组合**（两个旋钮一起动），不当它是什么单因子归因：
    # 前两条都不翻转、而宽配置能翻，说明这道题需要两者同时到位——那就要把组合本身量出来，
    # 不能靠"两个各加一半应该也行"去推。
    whatif_both = mk_retriever(store, sched, top_k=5, rc=40, legs=40)
    # 第四个：宽配置的**生产形态版**（top_k 回到 5，只留腿宽 40 + 上限 50）。
    # 前三条的差值已经把机制框出来了：gold 落在融合名次 41–50 之间——那么 rc=40 必然看不见它，
    # 需要的是 rc=50。这一跑就是去验这句话，而不是继续推。
    whatif_deep = mk_retriever(store, sched, top_k=5, rc=50, legs=40)
    # 两条**不要时间**的：只改融合曲线 `rrf_k`（60→20 / 60→10），腿宽与上限都不动。
    # 机制探针显示 q-044 是「BM25#7 但稠密腿前 50 没有」——这种单腿强的块正是被 RRF 的
    # 双票结构埋掉的，所以先试最便宜的那一刀，而不是直接去买 2.4 倍延迟。
    whatif_rrf20 = mk_retriever(store, sched, top_k=5, rc=20, legs=20, rrf_k=20)
    whatif_rrf10 = mk_retriever(store, sched, top_k=5, rc=20, legs=20, rrf_k=10)

    # 真预热：两个模型是懒加载的（各 6~13s），不挡在计时外第一题会撞 dense_budget_ms 超时，
    # 于是我的降级腿前置校验会拒绝整批数——那是一次假故障（m2_eval 当年踩过同一处）。
    tw = time.perf_counter()
    w = narrow.retrieve("预热查询：配电自动化终端心跳超时阈值",
                        AccessContext(role="knowledge_owner"), top_k=5)
    print(f"[预热] 首查 {round((time.perf_counter() - tw) * 1000)}ms（含模型冷加载），"
          f"degraded={w.degraded} cache_hit={w.cache_hit}")
    if w.degraded:  # 冷加载那一下撞超时是常态，再跑一次；还降级就是真故障，不是一冷
        w = narrow.retrieve("预热查询二：灾备切换 RTO 阈值", AccessContext(role="knowledge_owner"), top_k=5)
        if w.degraded:
            print(f"[拒绝起跑] 预热两次仍有降级腿 {w.degraded}——现在不是量缺口的时机")
            return 2

    rows: list[dict] = []
    deg: list[tuple[str, list]] = []

    if args.explain:  # 只跑机制探针就退出：这条路要单独看，混在全量里只会多花两分钟
        want = {x.strip() for x in args.explain.split(",") if x.strip()}
        ex = {}
        for q in qs:
            if q["qid"] not in want:
                continue
            print(f"  {q['qid']} {q['category']} 题面={q['question'][:44]}")
            ex[q["qid"]] = {"question": q["question"], "category": q["category"],
                            "gold_keywords": q.get("gold_keywords"),
                            "variants": explain(store, sched, q,
                                                AccessContext(role="knowledge_owner"))}
        art = Path(s.resolved_artifact_dir())
        (art / "m2_retrieval_gap_explain.json").write_text(json.dumps({
            "purpose": ("机制探针：gold 卡在稠密/BM25/融合各自的第几名。"
                        "**第二个变体用了 gold_keywords，属上限检验，不可当功能上线**——"
                        "它只回答「词都对上了还检不到吗」"),
            "n_vectors": n_vec, "cases": ex}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  ⚠ 「原查询+gold_keywords」这一行是偷看答案的上限检验，不是可用的改写策略。")
        print("证据：", art / "m2_retrieval_gap_explain.json")
        return 0
    for q in qs:
        qid, gold = q["qid"], set(q["gold_strict_ids"] or q["gold_chunk_ids"])
        gold_any = set(q["gold_chunk_ids"])
        a = narrow.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        b = wide.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=50)
        wf = whatif_rc.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        lg = whatif_legs.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        bt = whatif_both.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        dp = whatif_deep.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        r20 = whatif_rrf20.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        r10 = whatif_rrf10.retrieve(q["question"], AccessContext(role="knowledge_owner"), top_k=5)
        if any(x.degraded or x.cache_hit for x in (a, b, wf, lg, bt, dp, r20, r10)):
            deg.append((qid, list(a.degraded) + list(b.degraded) + list(wf.degraded)
                        + list(lg.degraded) + list(bt.degraded) + list(dp.degraded)
                        + list(r20.degraded) + list(r10.degraded)))
        ids_a = [c.chunk_id for c in a.chunks]
        ids_b = [c.chunk_id for c in b.chunks]
        ids_w = [c.chunk_id for c in wf.chunks]
        ids_l = [c.chunk_id for c in lg.chunks]
        ids_t = [c.chunk_id for c in bt.chunks]
        ids_d = [c.chunk_id for c in dp.chunks]
        ids_20 = [c.chunk_id for c in r20.chunks]
        ids_10 = [c.chunk_id for c in r10.chunks]
        # best rank 用**宽列表**里的位置；但 @5 判定只看生产那次，两件事不能混
        ranks_b = [i + 1 for i, cid in enumerate(ids_b) if cid in gold_any]
        top5 = ids_a[:5]
        hit5 = any(cid in gold_any for cid in top5)
        strict_in_a = any(cid in gold for cid in top5)
        # 成员关系分别记下来，诊断只由这些布尔推出（见 DIAGNOSIS_PRECEDENCE），不靠叙述
        mem = {"in_prod": any(cid in gold_any for cid in ids_a),
               "in_rc40": any(cid in gold_any for cid in ids_w),
               "in_legs40": any(cid in gold_any for cid in ids_l),
               "in_both40": any(cid in gold_any for cid in ids_t),
               "in_deep40_50": any(cid in gold_any for cid in ids_d),
               "in_wide": any(cid in gold_any for cid in ids_b)}
        diag_kind = ("hit" if hit5 else
                     "misranked" if mem["in_prod"] else
                     "needs_rc" if mem["in_rc40"] else
                     "needs_legs" if mem["in_legs40"] else
                     "needs_both" if mem["in_both40"] else
                     "needs_deep" if mem["in_deep40_50"] else
                     "absent" if not mem["in_wide"] else "absent_wide_only")
        top1 = a.chunks[0] if a.chunks else None
        top2 = a.chunks[1] if len(a.chunks) > 1 else None
        gtext = " ".join(children[g]["content"] for g in list(gold_any)[:3] if g in children)
        rows.append({
            "qid": qid, "category": q["category"], "n_gold": len(gold_any), "n_strict": len(gold),
            "hit5_prod": hit5, "strict_in_top5": strict_in_a,
            "hit5_rc40": any(cid in gold_any for cid in ids_w[:5]),
            "hit5_legs40": any(cid in gold_any for cid in ids_l[:5]),
            "hit5_both40": any(cid in gold_any for cid in ids_t[:5]),
            "hit5_deep40_50": any(cid in gold_any for cid in ids_d[:5]),
            "hit5_rrf20": any(cid in gold_any for cid in ids_20[:5]),
            "hit5_rrf10": any(cid in gold_any for cid in ids_10[:5]),
            "strict_rrf20": any(cid in gold for cid in ids_20[:5]),
            "strict_rrf10": any(cid in gold for cid in ids_10[:5]),
            "ms_rrf20": r20.timings_ms.get("retrieval_total_ms"),
            "ms_rrf10": r10.timings_ms.get("retrieval_total_ms"),
            "hit5_rrf20": any(cid in gold_any for cid in ids_20[:5]),
            "hit5_rrf10": any(cid in gold_any for cid in ids_10[:5]),
            "strict_rc40": any(cid in gold for cid in ids_w[:5]),
            "strict_legs40": any(cid in gold for cid in ids_l[:5]),
            "strict_both40": any(cid in gold for cid in ids_t[:5]),
            "strict_deep40_50": any(cid in gold for cid in ids_d[:5]),
            "ms_prod": a.timings_ms.get("retrieval_total_ms"),
            "ms_wide": b.timings_ms.get("retrieval_total_ms"),
            "ms_rc40": wf.timings_ms.get("retrieval_total_ms"),
            "ms_legs40": lg.timings_ms.get("retrieval_total_ms"),
            "ms_both40": bt.timings_ms.get("retrieval_total_ms"),
            "ms_deep40_50": dp.timings_ms.get("retrieval_total_ms"),
            "pool_membership": mem,
            "best_rank_wide": min(ranks_b) if ranks_b else None,
            "diagnosis": diag_kind,
            # 分数只在同题内可比：rank1 vs rank2，且必须两条都是重排给的（表行走融合位时分数不可比）
            "gap_r1_r2": (round(top1.score - top2.score, 6)
                          if top1 is not None and top2 is not None
                          and top1.fields.get("ranked_by") == "rerank"
                          and top2.fields.get("ranked_by") == "rerank" else None),
            "top1_ranked_by": (top1.fields.get("ranked_by") if top1 else None),
            "top1_score": round(top1.score, 6) if top1 else None,
            "ov_gold_vs_question": overlap(q["question"], gtext),
            "bg_gold_vs_question": bg_overlap(q["question"], gtext),
            "ov_top1_vs_question": overlap(q["question"], top1.text) if top1 else None,
            "bg_top1_vs_question": bg_overlap(q["question"], top1.text) if top1 else None,
            "gold_ids": sorted(gold_any)[:6],
        })
        print(f"  {qid} {q['category']:4s} hit5={hit5!s:5s} 宽表 best={rows[-1]['best_rank_wide']} "
              f"诊断={rows[-1]['diagnosis']} 重合 gold={rows[-1]['ov_gold_vs_question']} "
              f"top1={rows[-1]['ov_top1_vs_question']}")

    if deg:
        print(f"[拒绝出数] {len(deg)} 题的检索出现降级腿或缓存命中：{deg[:5]}——"
              "那是另一条路径，混进缺口归因会得到假结论")
        return 3

    miss = [r for r in rows if not r["hit5_prod"]]
    diag = Counter(r["diagnosis"] for r in rows)
    by_cat: dict[str, dict] = {}
    for c in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == c]
        m = [r for r in sub if not r["hit5_prod"]]
        by_cat[c] = {"n": len(sub), "miss": len(m),
                     "hit5": round(1 - len(m) / len(sub), 4),
                     "diag": dict(Counter(r["diagnosis"] for r in sub)),
                     "mean_ov_gold": round(sum(r["ov_gold_vs_question"] for r in sub) / len(sub), 4),
                     "mean_ov_top1": round(sum(r["ov_top1_vs_question"] or 0 for r in sub) / len(sub), 4)}

    def mean(vals: list[float]) -> float:
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def q(vals: list[float], p: float) -> float:
        v = sorted(x for x in vals if x is not None)
        return round(v[min(len(v) - 1, max(0, int(p * len(v)) - 1))], 1) if v else 0.0

    def one_whatif(tag: str, ms_key: str, hit_key: str, strict_key: str) -> dict:
        gained = [r["qid"] for r in rows if not r["hit5_prod"] and r[hit_key]]
        lost = [r["qid"] for r in rows if r["hit5_prod"] and not r[hit_key]]
        gained_s = [r["qid"] for r in rows if not r["strict_in_top5"] and r[strict_key]]
        lost_s = [r["qid"] for r in rows if r["strict_in_top5"] and not r[strict_key]]
        ms = [r[ms_key] for r in rows]
        return {"gained": gained, "lost": lost, "gained_strict": gained_s, "lost_strict": lost_s,
                "hit5_gold_any": sum(1 for r in rows if r[hit_key]),
                "hit5_strict": sum(1 for r in rows if r[strict_key]),
                "p50_ms": q(ms, .5), "p95_ms": q(ms, .95)}

    ms_p = [r["ms_prod"] for r in rows]
    whatif_block = {
        "why_two": ("宽配置同时动了两个旋钮，第一次我把 q-044 报成 cut_by_pool 而 rc=40 单跑**没有翻转**——"
                    "说明真因在腿宽不在候选上限。混着跑就会把修法指错，所以这里各自只动一个因子。"),
        "prod": {"hit5_gold_any": sum(1 for r in rows if r["hit5_prod"]),
                 "hit5_strict": sum(1 for r in rows if r["strict_in_top5"]),
                 "p50_ms": q(ms_p, .5), "p95_ms": q(ms_p, .95)},
        "rc_40": one_whatif("rc40", "ms_rc40", "hit5_rc40", "strict_rc40"),
        "legs_40": one_whatif("legs40", "ms_legs40", "hit5_legs40", "strict_legs40"),
        "both_40": one_whatif("both40", "ms_both40", "hit5_both40", "strict_both40"),
        "deep_40_50": one_whatif("deep40_50", "ms_deep40_50", "hit5_deep40_50", "strict_deep40_50"),
        "rrf_k_20": one_whatif("rrf20", "ms_rrf20", "hit5_rrf20", "strict_rrf20"),
        "rrf_k_10": one_whatif("rrf10", "ms_rrf10", "hit5_rrf10", "strict_rrf10"),
        "gate": {"p95_retrieval_ms": 800.0},
        "not_a_ruler_change": ("这一栏是 what-if，不是已经改了的门禁口径；"
                               "真要落地得按「原口径/修订口径/实测依据」三件套改 config 并重跑 m2_eval"),
    }

    # 假说的直接检验：漏召题上「gold 重合度」是不是显著低于「第 1 名重合度」——
    # 若是，说明题面的词没落在证据里（口语 vs 书面），扩池/重排都救不了；
    # 若否（两边都低或 gold 反而更高），那假说就不成立，得换解释。
    gold_lower = [r for r in miss if r["ov_top1_vs_question"] is not None
                  and r["ov_gold_vs_question"] < r["ov_top1_vs_question"]]
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ruler": {"n_graded": len(rows), "question_hash": qhash,
                  "collection": s.milvus_collection, "n_vectors": n_vec,
                  "prod_cfg": {"top_k": 5, "rerank_candidates": 20, "dense_top_k": 20,
                               "sparse_top_k": 20, "rewrite_mode": "off", "mode": "hybrid"},
                  "wide_cfg": {"top_k": 50, "rerank_candidates": 50, "dense_top_k": 40,
                               "sparse_top_k": 40}},
        "reading_rules": {
            "diagnosis": ("按 precedence 从**成员关系**推出，不靠叙述："
                          "hit=生产前 5 有 gold → misranked=生产池里有但排在 5 名外（扩池无效，是排序问题）→ "
                          "needs_rc=只有把候选上限抬到 40 才看得见 → needs_legs=只有把两条腿各取 40 才看得见 → "
                          "needs_both=只有宽配置（两个旋钮一起动）才看得见 → absent=宽池里也没有"),
            "why_precedence_matters": ("misranked 与 needs_* 的修法代价完全不同：前者要动排序/证据表述，"
                                        "后者只是把池子开宽（花的是重排时间）。第一版我把 needs_* 一律写成 "
                                        "cut_by_pool，因为宽配置同时动了两个旋钮——结果会把修法指到不生效的那个上。"),
            "warning": ("宽配置那次**不是** Recall@5——top_k 变了 k 就变了，不是同一把尺。"
                        "宽表只用来回答「它到底在不在池里、排第几」。"),
            "scores": "重排分是 query-pair 尺度，只在同题内比 rank1−rank2；跨题平均无意义。",
            "overlap": "重合度=|交集|/min(|A|,|B|)，块正文的标题前缀已剔除（那是分块加的，不是证据的词）。",
        },
        "totals": {"n": len(rows), "hit5": diag.get("hit", 0), **{k: v for k, v in diag.items() if k != "hit"}},
        "lexical_hypothesis": {
            "miss_rows": len(miss),
            "gold_overlap_lower_than_top1": len(gold_lower),
            "share": round(len(gold_lower) / len(miss), 4) if miss else None,
            "mean_ov_gold_on_miss": mean([r["ov_gold_vs_question"] for r in miss]),
            "mean_ov_top1_on_miss": mean([r["ov_top1_vs_question"] for r in miss]),
            "mean_ov_gold_on_hit": mean([r["ov_gold_vs_question"] for r in rows if r["hit5_prod"]]),
            "mean_ov_top1_on_hit": mean([r["ov_top1_vs_question"] for r in rows if r["hit5_prod"]]),
            "bigram_same_test": {"gold": mean([r["bg_gold_vs_question"] for r in miss]),
                                 "top1": mean([r["bg_top1_vs_question"] for r in miss])},
        },
        "by_category": by_cat,
        "whatif_single_factor": whatif_block,
        "worst": sorted(miss, key=lambda r: (r["best_rank_wide"] or 999))[:12],
        "rows": rows,
    }
    art = Path(s.resolved_artifact_dir())
    (art / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    t = out["totals"]
    print(f"\n生产配置 @5：{t['hit5']}/{t['n']} 题命中 gold；漏召 {t['n'] - t['hit5']} 题")
    print(f"  归因（全量 {len(rows)} 题）：{dict(diag)}")
    lh = out["lexical_hypothesis"]
    print(f"  词法假说：漏召题里 gold 重合度 < 第 1 名重合度的有 {lh['gold_overlap_lower_than_top1']}"
          f"/{lh['miss_rows']}（{lh['share']}）；漏召均值 gold={lh['mean_ov_gold_on_miss']} vs "
          f"top1={lh['mean_ov_top1_on_miss']}；命中均值 gold={lh['mean_ov_gold_on_hit']} vs "
          f"top1={lh['mean_ov_top1_on_hit']}")
    for c, v in by_cat.items():
        print(f"  {c:4s} n={v['n']:2d} hit5={v['hit5']:.3f} 诊断={v['diag']} "
              f"重合 gold/top1={v['mean_ov_gold']}/{v['mean_ov_top1']}")
    wi = out["whatif_single_factor"]
    pr, rc40, lg40 = wi["prod"], wi["rc_40"], wi["legs_40"]
    print(f"  what-if（同一趟、各自只动一个因子）：生产 @5={pr['hit5_gold_any']}/"
          f"{len(rows)}（strict {pr['hit5_strict']}）p95 {pr['p95_ms']}ms")
    for name, blk in (("rrf_k 60→20（免费）", wi["rrf_k_20"]), ("rrf_k 60→10（免费）", wi["rrf_k_10"]),
                      ("rerank_candidates 20→40", rc40), ("dense/sparse_top_k 20→40", lg40),
                      ("两者同时 20→40（组合）", wi["both_40"]),
                      ("腿 40 + 上限 50（宽配置的生产形态）", wi["deep_40_50"])):
        print(f"    {name}: @5={blk['hit5_gold_any']}（strict {blk['hit5_strict']}）"
              f" 翻转 +{blk['gained']} −{blk['lost']}｜p95 {blk['p95_ms']}ms"
              f"（门禁 800ms，{'仍有余量' if blk['p95_ms'] < wi['gate']['p95_retrieval_ms'] else '**顶破门禁**'}）")
    print("  ⚠ 这一栏是 what-if，不是已改的口径；落地要走三件套改 config 并重跑 m2_eval。")
    print("证据：", art / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
