"""M3 答案链实测：契约校验上线后，答案状态分布与引用落点到底变成什么样。

要回答的问题很具体：Gate 0 ⑤ 那 39 行「有要点、有出处、无结论」的输出，
在加了出口契约之后去了哪里，以及这件事让「成功答案数」变好还是变差。
答案大概率是变差——而变差才是对的，因为原来那部分是假成功。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import expected_calibration_error  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa.answer import AnswerEngine, load_calibration_temperature  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    args = ap.parse_args()

    s = get_settings()
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    questions = build_eval_set(pg, write_files=False)["questions"]
    ans = [q for q in questions if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]]
    if args.limit:
        ans = ans[: args.limit]

    sched = ResidencyScheduler(policy="co_resident")
    kb = KbStore(s.milvus_collection, "jieba")
    rt = HybridRetriever(kb, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    eng = AnswerEngine(retriever=rt, llm=OllamaClient())
    # 与 API 同源取温度：否则「探针测出来的分布」又不是服务路径的那个分布了
    eng.calibration_temperature = load_calibration_temperature(s)
    ctx = AccessContext(role="knowledge_owner")

    # 真预热：把 7B 与两个模型的冷加载挡在计时之外
    t_w = time.perf_counter()
    eng.answer("预热：配电自动化终端心跳超时阈值", ctx)
    print(f"[预热] {round(time.perf_counter() - t_w, 1)}s")

    rows = []
    for i, q in enumerate(ans, start=1):
        t0 = time.perf_counter()
        r = eng.answer(q["question"], ctx)
        gold = set(q["gold_chunk_ids"])
        cited = {c.get("chunk_id") for c in r.citations}
        rows.append(
            {
                "qid": q["qid"],
                "category": q["category"],
                "status": r.status,
                "needs_human": r.needs_human,
                "conclusion_len": len(r.conclusion.strip()),
                "n_points": len(r.key_points),
                "n_citations": len(r.citations),
                "citation_hits_gold": int(bool(cited & gold)),
                "confidence": r.confidence,
                "notes": r.notes,
                "e2e_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        )
        print(f"  [{i:2d}/{len(ans)}] {q['qid']} {r.status:9s} 结论{rows[-1]['conclusion_len']:3d}字 "
              f"要点{len(r.key_points)} 引用{len(r.citations)} 落gold={rows[-1]['citation_hits_gold']} "
              f"{rows[-1]['e2e_ms']:.0f}ms")

    st = Counter(r["status"] for r in rows)
    gradable = [r for r in rows if r["confidence"].get("logit") is not None]
    hit_gold = sum(r["citation_hits_gold"] for r in rows if r["n_citations"])
    n_cit_rows = sum(1 for r in rows if r["n_citations"])
    empty_concl_answer = sum(1 for r in rows if r["status"] in ("answer", "tentative") and not r["conclusion_len"])
    out = {
        "collection": s.milvus_collection,
        "n_answerable": len(rows),
        "status_dist": dict(st),
        "needs_human": sum(1 for r in rows if r["needs_human"]),
        "契约拦下的空结论": sum(1 for r in rows if "contract_violation:empty_conclusion" in r["notes"]),
        "行级引用落gold": f"{hit_gold}/{n_cit_rows} = {round(hit_gold / max(n_cit_rows, 1), 4)}",
        "仍被判answer/tentative且结论为空": empty_concl_answer,
        "可判校准集(有logit)": len(gradable),
        "e2e_ms": {
            "p50": sorted(r["e2e_ms"] for r in rows)[len(rows) // 2],
            "max": max(r["e2e_ms"] for r in rows),
        },
        "rows": rows,
    }
    if gradable:
        conf = [r["confidence"]["composite"] for r in gradable]
        labels = [r["citation_hits_gold"] for r in gradable]
        out["calibration_on_gradable"] = expected_calibration_error(conf, labels)
    (art / "m3_answer_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    kb.close()
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("证据：", art / "m3_answer_probe.json")
    # 判定：不允许任何 answer/tentative 状态带空结论
    return 1 if empty_concl_answer else 0


if __name__ == "__main__":
    raise SystemExit(main())
