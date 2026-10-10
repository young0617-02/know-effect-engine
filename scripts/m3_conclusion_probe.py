"""M3-01c 诊断：58 道可判答案题里 39 道结论为空，逐条分型找机制，不靠猜。

四类互斥的可能原因，必须分开计数，因为修法完全不同：
  truncated        done_reason=length → num_predict 截断（M2 的查询改写踩过同型坑）
  empty_field      JSON 完整、有 conclusion 键、但值是空串 → 模型主动留空（提示词/服从率问题）
  missing_field    JSON 完整但没有 conclusion 键 → 模型没按 schema 出
  parse_failed     JSON 解析失败 → 输出形态问题
另外单独记 no_evidence=true 的题（那本来就该拒答，不属于「该答没答」）。
"""

from __future__ import annotations

import argparse
import collections
import json
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
from kee.qa.answer import SYSTEM_PROMPT, USER_TEMPLATE, format_evidence  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402


def classify(r) -> str:
    """把一次生成归到上面四类之一。"""
    if r.metrics.get("done_reason") == "length":
        return "truncated"
    if not r.json_valid:
        return "parse_failed"
    obj = r.parsed if isinstance(r.parsed, dict) else {}
    if "conclusion" not in obj:
        return "missing_field"
    return "empty_field" if not str(obj.get("conclusion", "")).strip() else "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--reps", type=int, default=1, help="每题重复几次，用来量同一题的输出稳定性")
    args = ap.parse_args()

    s = get_settings()
    art = Path(s.resolved_artifact_dir())
    pg = AuthoritativeStore()
    questions = [
        q for q in build_eval_set(pg, write_files=False)["questions"]
        if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]
    ]
    if args.limit:
        questions = questions[: args.limit]

    sched = ResidencyScheduler(policy="co_resident")
    kb = KbStore(s.milvus_collection, "jieba")
    rt = HybridRetriever(kb, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)
    llm = OllamaClient()
    ctx = AccessContext(role="knowledge_owner")

    llm.warmup(max_tokens=1)
    rows = []
    for i, q in enumerate(questions, start=1):
        res = rt.retrieve(q["question"], ctx)
        ev, _idx = format_evidence(res)
        prompt = USER_TEMPLATE.format(evidence=ev, question=q["question"])
        for rep in range(args.reps):
            r = llm.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            kind = classify(r)
            obj = r.parsed if isinstance(r.parsed, dict) else {}
            rows.append(
                {
                    "qid": q["qid"],
                    "rep": rep,
                    "category": q["category"],
                    "kind": kind,
                    "done_reason": r.metrics.get("done_reason", ""),
                    "completion_tokens": r.metrics.get("completion_tokens", 0),
                    "prompt_tokens": r.metrics.get("prompt_tokens", 0),
                    "max_tokens": args.max_tokens,
                    "no_evidence": obj.get("no_evidence"),
                    "n_points": len(obj.get("key_points") or []),
                    "conclusion": str(obj.get("conclusion", ""))[:80],
                    "raw_head": r.text[:120],
                    "raw_tail": r.text[-90:],
                }
            )
        last = rows[-1]
        print(f"[{i:2d}/{len(questions)}] {q['qid']} {last['kind']:13s} "
              f"done={last['done_reason'] or '-':6s} tok={last['completion_tokens']:3d} "
              f"要点={last['n_points']} no_evid={last['no_evidence']} | {last['conclusion'][:34]!r}")

    kinds = collections.Counter(r["kind"] for r in rows)
    trunc = [r for r in rows if r["kind"] == "truncated"]
    empty = [r for r in rows if r["kind"] == "empty_field"]
    out = {
        "max_tokens": args.max_tokens,
        "reps": args.reps,
        "n_questions": len(questions),
        "n_generations": len(rows),
        "分型": dict(kinds),
        "被截断数": len(trunc),
        "主动留空数": len(empty),
        "截断样本tok": sorted(r["completion_tokens"] for r in trunc)[:12],
        "主动留空但要点非空": sum(1 for r in empty if r["n_points"] > 0),
        "主动留空且no_evidence为真": sum(1 for r in empty if r["no_evidence"] is True),
        "rows": rows,
    }
    (art / "m3_conclusion_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    kb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
