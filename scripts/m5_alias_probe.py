"""#58 的下一步验证：**只在 scratch 集合上**试"给块加一层书面语别名"能不能收掉口语化缺口。

为什么走这条路而不是调参数：#58 已量清——q-044 的三块 gold 在原查询下**稠密腿前 50 全无**、
BM25 排 7/17/28 却被 RRF 埋到 26 名以后；而唯一有效的参数配置（腿 40 + 上限 50）要把检索
p95 从 319 推到 760ms、门禁余量剩 40ms（低于 #56 的噪声地板 93.6ms）。**那是拿门禁稳定性换一道题**。
别名这条路是离线一次性成本、在线不加延迟，所以值得先花小钱验一次，而不是直接裁"做/不做"。

三件让这条路可判的前提（都是硬约束，不是愿望）：

1. **不重新分块**。别名写成同一行的 `text` 追加内容，`chunk_id` 一字不改 ⇒
   547 行块级双人标注、67 行 answer_correct、gold 映射全部不作废。
   （一旦重新分块，那些人工标注判的对象就不存在了——那才是真的代价。）
2. **不碰生产集合**。只写 `kb_chunks_alias_probe`；脚本开头就断言两者不同名，
   同名直接拒绝起跑——线上没有"删文档"端点，写错集合撤不回来。
3. **别名只许改写自正文**。生成本地 7B 的产出要过**数字落地检查**：别名里出现正文没有的阿拉伯数字
   ⇒ 判为不落地，**该块不注入别名**（fail-closed，宁可少一个样本也不给 BM25 喂假词——
   那是自己造投毒面，与复核闸要防的是同一类东西）。

跑法：
    uv run python scripts/m5_alias_probe.py                 # 建 scratch + 对照，保留集合
    uv run python scripts/m5_alias_probe.py --drop           # 跑完删掉 scratch
产物：var/gate0/m5_alias_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext, build_filter_expr  # noqa: E402

ALIAS_PROMPT = (
    "你在为一段交付文档做「检索别名」。只依据给出的【正文】，写一句**用户可能怎么问这段话**。\n"
    "硬性规则：\n"
    '1) 只输出 JSON：{{"alias": "一句话问法"}}；\n'
    "2) 问法里出现的数字、编号、产品名、指标名**必须都在正文里出现过**，不许引入正文没有的说法；\n"
    "3) 用**书面术语**写（正文怎么写就怎么问），不要口语化、不要评价、不要总结成长段；\n"
    "4) 不超过 40 个字。"
)
DIGITS = re.compile(r"\d+(?:\.\d+)?")
SCRATCH = "kb_chunks_alias_probe"


def grounded(alias: str, content: str) -> tuple[bool, list[str]]:
    """别名里的阿拉伯数字必须都在正文里——这是"只许改写自正文"唯一可机械检验的部分。"""
    bad = [d for d in DIGITS.findall(alias or "") if d not in (content or "")]
    return (not bad, bad)


def fetch_effective_children(pg) -> list[dict]:
    with pg.connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT c.id, c.doc_id, c.version_id, c.parent_id, c.section_no, c.page, c.order_no, "
            "       c.content, d.industry, d.customer, d.project, d.doc_type, d.source_level, "
            "       d.security_level "
            "FROM chunk c JOIN document d ON d.id = c.doc_id "
            "JOIN document_version v ON v.id = c.version_id AND v.status = 'effective' "
            "WHERE c.is_parent = false ORDER BY c.id").fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", action="store_true", help="跑完删掉 scratch 集合（默认保留，便于复核）")
    ap.add_argument("--max-alias", type=int, default=0, help="只给前 N 个目标块生成别名（冒烟用）")
    args = ap.parse_args()

    s = get_settings()
    if SCRATCH == s.milvus_collection:
        print(f"[拒绝起跑] scratch 集合名与生产集合同名（{SCRATCH}）——生产库没有删除端点，写错撤不回")
        return 2
    pg = AuthoritativeStore()
    built = build_eval_set(pg, write_files=False)
    prod = KbStore(s.milvus_collection, "jieba")
    n_prod = prod.count()
    if n_prod < 100:
        print(f"[拒绝起跑] 生产集合只有 {n_prod} 条，索引不像建好过")
        return 2

    # 目标：口语化那 14 道题的 gold 块（并集）。只给它们加别名，其余块与生产逐字相同——
    # 这样"有没有变好"才只可能来自别名，而不是我顺手改了别的东西。
    colloquial = [q for q in built["questions"] if q["category"] == "口语化" and q["gold_chunk_ids"]]
    targets = sorted({g for q in colloquial for g in q["gold_chunk_ids"]})
    print(f"[样本] 口语化 {len(colloquial)} 题，目标块 {len(targets)} 个（去重）")

    llm = OllamaClient()
    if not llm.model_available():
        print(f"[拒绝起跑] 本地推理服务没有 {s.llm_model}——别名要靠它生成，没有就整批都该停")
        return 2

    rows = fetch_effective_children(pg)
    by_id = {r["id"]: r for r in rows}
    missing = [t for t in targets if t not in by_id]
    if missing:
        print(f"[中止] gold 块在权威源里找不到生效版本的有 {len(missing)} 个：{missing[:5]}——"
              "多半是版本被回滚/驳回，此时对照没有意义")
        return 3

    aliases: dict[str, dict] = {}
    for cid in (targets[: args.max_alias] if args.max_alias else targets):
        content = by_id[cid]["content"]
        t0 = time.perf_counter()
        r = llm.chat([{"role": "user", "content":
                       ALIAS_PROMPT.format() + f"\n\n【正文】{content[:1200]}"}],
                     json_mode=True, temperature=0.0, max_tokens=120)
        alias = ""
        if r.ok and isinstance(r.parsed, dict):
            alias = str(r.parsed.get("alias") or "").strip()
        else:
            try:
                alias = json.loads((r.text or "").strip()).get("alias", "").strip()
            except json.JSONDecodeError:
                alias = ""
        ok, bad = grounded(alias, content)
        aliases[cid] = {"alias": alias, "grounded": ok and bool(alias), "ungrounded_digits": bad,
                        "ms": round((time.perf_counter() - t0) * 1000, 1)}
        print(f"  {cid[:52]:52s} {'OK ' if aliases[cid]['grounded'] else '弃 '} {alias[:34]} "
              f"{('编造数字:' + ','.join(bad)) if bad else ''}")
    usable = {k: v["alias"] for k, v in aliases.items() if v["grounded"]}
    print(f"[别名] 可用 {len(usable)}/{len(aliases)}（未过数字落地检查的一律不注入）")

    # ---- 建 scratch：同一批块、同一个 chunk_id，只有目标块的 text 末尾多一行别名 ----
    scratch = KbStore(SCRATCH, "jieba")
    scratch.drop()
    scratch.ensure()
    sched = ResidencyScheduler(policy="co_resident")
    texts = [by_id[r["id"]]["content"] + (f" 可能问法：{usable[r['id']]}" if r["id"] in usable else "")
             for r in rows]
    with sched.use("embed"):
        em = sched.embedder
        vecs = [list(map(float, v)) for v in np.asarray(em.embed_passages(texts))]
    docs = [{"chunk_id": r["id"], "doc_id": r["doc_id"], "version_id": r["version_id"],
             "parent_id": r["parent_id"] or "", "industry": r["industry"], "customer": r["customer"],
             "project": r["project"], "doc_type": r["doc_type"], "source_level": r["source_level"],
             "security_level": int(r["security_level"]), "section_no": r["section_no"],
             "page": int(r["page"]), "order_no": int(r["order_no"]),
             "text": t, "dense": v} for r, t, v in zip(rows, texts, vecs, strict=True)]
    written = scratch.insert(docs)
    scratch.flush()
    print(f"[scratch] 写入 {written}/{len(docs)} 条（生产集合 {n_prod} 条未动）")

    def run(store: KbStore) -> dict:
        rt = HybridRetriever(store, scheduler=sched, cache=None, mode="hybrid", use_rerank=True,
                             rerank_candidates=20, dense_top_k=20, sparse_top_k=20, rewrite_mode="off")
        ctx = AccessContext(role="knowledge_owner")
        out = {}
        for q in colloquial:
            gold = set(q["gold_chunk_ids"])
            strict = set(q["gold_strict_ids"] or q["gold_chunk_ids"])
            t1 = time.perf_counter()
            res = rt.retrieve(q["question"], ctx, top_k=5)
            ids = [c.chunk_id for c in res.chunks]
            out[q["qid"]] = {
                "hit_any": any(i in gold for i in ids[:5]),
                "hit_strict": any(i in strict for i in ids[:5]),
                "n_found": sum(1 for i in ids[:5] if i in gold),
                "top_ids": ids[:5], "ms": round((time.perf_counter() - t1) * 1000, 1),
                "degraded": list(res.degraded), "cache_hit": res.cache_hit,
            }
        return out

    base = run(prod)
    aug = run(scratch)
    if any(v["degraded"] or v["cache_hit"] for v in list(base.values()) + list(aug.values())):
        print("[拒绝出数] 出现降级腿或缓存命中，对照不成立")
        return 3

    flips_up = [k for k in base if not base[k]["hit_strict"] and aug[k]["hit_strict"]]
    flips_dn = [k for k in base if base[k]["hit_strict"] and not aug[k]["hit_strict"]]
    strict_before = sum(1 for v in base.values() if v["hit_strict"])
    strict_after = sum(1 for v in aug.values() if v["hit_strict"])
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": ("小样本验证「书面语别名」，**生产集合未写一行**；只给 14 道口语化题的 gold 块加别名，"
                    "其余块与生产逐字相同 ⇒ 变化只可能来自别名"),
        "hard_constraints": {
            "chunk_id 不变": "别名写进同一行的 text，不重新分块 ⇒ 人工标注与 gold 映射不作废",
            "生产集合只读": f"写入目标 {SCRATCH}，与 {s.milvus_collection} 不同名（脚本开头断言）",
            "别名必须落地": "别名里的数字若在正文中不存在 ⇒ 该块不注入（fail-closed）",
        },
        "aliases": aliases,
        "usable_aliases": len(usable),
        "questions": {"before": base, "after": aug},
        "summary": {
            "n_questions": len(colloquial),
            "hit_strict_before": strict_before, "hit_strict_after": strict_after,
            "flips_up": flips_up, "flips_down": flips_dn,
            "p50_ms_before": sorted(v["ms"] for v in base.values())[len(base) // 2],
            "p50_ms_after": sorted(v["ms"] for v in aug.values())[len(aug) // 2],
        },
        "scope_limits": (
            "这是**上界探针，不是可上线的效果**：别名是从 gold 块自己的正文生成的（等于知道答案在哪块再"
            "给它写问法），全库推广时必须由文档侧独立生成、并另做一轮盲测。"
            "本轮只回答一个问题：'块面上多一层书面语问法，检索认不认'。"),
    }
    art = Path(s.resolved_artifact_dir())
    (art / "m5_alias_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    sm = out["summary"]
    print(f"\n口语化 {sm['n_questions']} 题：strict @5 命中 {sm['hit_strict_before']} → {sm['hit_strict_after']}"
          f" ｜ 翻上来 {sm['flips_up']} ｜ 掉下去 {sm['flips_down']}")
    print(f"检索 p50：{sm['p50_ms_before']}ms → {sm['p50_ms_after']}ms")
    print("  ⚠ 这是上界探针（别名由 gold 块自身正文生成），不是可上线效果的证据；全库推广要另做盲测。")
    if args.drop:
        scratch.drop()
        print(f"[清理] 已删除 {SCRATCH}")
    else:
        print(f"[保留] scratch 集合 {SCRATCH} 未删（--drop 可删）；生产集合 {s.milvus_collection} 全程只读")
    print("证据：", art / "m5_alias_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
