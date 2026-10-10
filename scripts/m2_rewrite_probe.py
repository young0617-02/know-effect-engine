"""查询改写的真机探针：量出「触发率 + 单次耗时 + 改写质量」，而不是假设它免费。

检索 P95 门禁是 800ms，而本地 7B 一次生成就要 0.6~0.9s（Gate 0 ① 实测），
所以改写能不能常开，取决于这两个数：
  1) 70 题评测集里有多少题会触发改写（触发率决定它是偶发开销还是系统性开销）；
  2) 触发后的 P50/P95 是多少（决定 800ms 门禁是否还成立）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.retrieval.rewrite import QueryRewriter, needs_rewrite  # noqa: E402


def pctl(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def main() -> int:
    art = ROOT / "var" / "gate0"
    questions = [json.loads(ln) for ln in (art / "qrels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    texts = [q["question"] for q in questions]
    trig = [t for t in texts if needs_rewrite(t)]
    print(f"[触发率] 评测集 {len(texts)} 题，启发式判定需要改写 {len(trig)} 题 = {len(trig) / len(texts):.1%}")
    for t in trig[:8]:
        print("   触发:", t)

    llm = OllamaClient()
    w = llm.warmup(max_tokens=1)
    print(f"[预热] {json.dumps(w, ensure_ascii=False)[:160]}")

    rw = QueryRewriter(client=llm, always=True)  # always=True：全量实测，不受触发条件影响
    rows = []
    for t in texts:
        r = rw.rewrite(t)
        rows.append(r)
        flag = "改" if r.changed else ("拒" if r.rejected_reason else ("降" if r.degraded else "同"))
        print(f"  [{flag}] {r.ms:6.0f}ms  {t[:34]}  →  {r.expanded[:40]}"
              + (f"  |{r.rejected_reason}|" if r.rejected_reason else ""))

    ms = [r.ms for r in rows]
    n_changed = sum(1 for r in rows if r.changed)
    n_rej = sum(1 for r in rows if r.rejected_reason)
    n_deg = sum(1 for r in rows if r.degraded)
    out = {
        "全量_题数": len(rows),
        "全量_改写数": n_changed,
        "被校验拒数": n_rej,
        "降级数": n_deg,
        "单次ms": {"p50": pctl(ms, 0.5), "p95": pctl(ms, 0.95), "max": round(max(ms), 1)},
        "触发子集_题数": len(trig),
        "触发子集ms": {"p50": pctl([r.ms for r in rows if needs_rewrite(r.original)], 0.5),
                   "p95": pctl([r.ms for r in rows if needs_rewrite(r.original)], 0.95)},
        "检索门禁_ms": 800,
        "样例": [{"原": r.original, "改写": r.expanded, "ms": r.ms, "拒": r.rejected_reason} for r in rows[:20]],
        "rewriter_stats": rw.summary(),
    }
    (art / "m2_rewrite_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "样例"}, ensure_ascii=False, indent=2))
    print("证据：", art / "m2_rewrite_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
