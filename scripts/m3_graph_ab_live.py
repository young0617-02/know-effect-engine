"""真机 A/B：同一批问题、同一个模型，分别喂给「重构前的实现」和「图化后的实现」。

差分网（`m3_graph_equivalence.py`）用的是**钉死的 JSON 载荷**，它能证明控制流等价，
但证不了「真机跑出来的分类结果没变」——因为本地 7B 即使 temperature=0 也不是逐次可复现的，
同一个问题两次跑可能一次声明无依据、一次给出一条要点。

所以这里做的是**同机同模型 A/B**：
  1. 12 道守门题各跑 legacy 与 graph 一遍，比 (status, needs_human, 结论字数, 引用数)；
  2. 凡出现不一致的题，**再用同一个实现跑一遍**——
     若同实现两次自己就不一致，那是模型抖动，不能记到重构头上；
     若同实现两次一致、而两实现之间不一致，那才是真回归。

归因规则写在这里是因为它容易被混用：见过不一致就喊「重构改坏了」，
等于把模型抖动记成代码缺陷，然后去修一个不存在的问题。

用法：uv run python scripts/m3_graph_ab_live.py
退出码 = 「同实现可复现、跨实现不一致」的题数（即真回归数）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa.answer import AnswerEngine, load_calibration_temperature  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

from m3_graph_equivalence import load_legacy  # noqa: E402

OBSERVED = ("status", "needs_human", "conclusion_len", "n_citations")


def observe(r) -> dict:
    return {
        "status": r.status,
        "needs_human": r.needs_human,
        "conclusion_len": len(r.conclusion.strip()),
        "n_citations": len(r.citations),
        "notes": r.notes,
    }


def main() -> int:
    s = get_settings()
    legacy = load_legacy()
    questions = build_eval_set(AuthoritativeStore(), write_files=False)["questions"]
    guarded = [q for q in questions if q["expected_behavior"] in ("refuse", "escalate")]

    def make_retriever():
        sched = ResidencyScheduler(policy="co_resident")
        kb = KbStore(s.milvus_collection, "jieba")
        return HybridRetriever(kb, scheduler=sched, cache=None, mode="hybrid", use_rerank=True)

    # 两个实现各挂一个**检索器**（不是各挂一个引擎——构造第一个参数是 retriever，
    # 传错了会在跑第一题时才炸出 AttributeError）。缓存一律关掉，
    # 免得一方把结果喂给另一方，A/B 就变成同一份结果比自己。
    eng_graph = AnswerEngine(retriever=make_retriever(), llm=OllamaClient())
    eng_legacy = legacy.AnswerEngine(retriever=make_retriever(), llm=OllamaClient())
    # 两边都用服务路径的温度，否则 A/B 比的是「两个都没上线的配置」
    t_cal = load_calibration_temperature(s)
    eng_graph.calibration_temperature = t_cal
    eng_legacy.calibration_temperature = t_cal
    ctx = AccessContext(role="knowledge_owner")

    t_w = time.perf_counter()
    eng_graph.answer("预热：配电自动化终端心跳超时阈值", ctx)
    print(f"[预热] {round(time.perf_counter() - t_w, 1)}s")

    rows, flips, real_regressions = [], [], []
    for i, q in enumerate(guarded, start=1):
        a = observe(eng_legacy.answer(q["question"], ctx))
        b = observe(eng_graph.answer(q["question"], ctx))
        same = all(a[k] == b[k] for k in OBSERVED)
        rows.append({"qid": q["qid"], "expected": q["expected_behavior"], "legacy": a, "graph": b, "agree": same})
        mark = "一致" if same else "不一致"
        print(f"[{i:>2}/{len(guarded)}] {q['qid']} {mark}  legacy={a['status']}/{a['conclusion_len']}字 "
              f"graph={b['status']}/{b['conclusion_len']}字")
        if not same:
            flips.append(q["qid"])

    # 抖没抖动，用同一个实现再跑一次来判，而不是靠猜。
    print("\n[归因] 对不一致的题各跑两次同一实现：")
    for qid in flips:
        q = next(x for x in guarded if x["qid"] == qid)
        g2 = observe(eng_graph.answer(q["question"], ctx))
        l2 = observe(eng_legacy.answer(q["question"], ctx))
        prev = next(r for r in rows if r["qid"] == qid)
        graph_repro = all(prev["graph"][k] == g2[k] for k in OBSERVED)
        legacy_repro = all(prev["legacy"][k] == l2[k] for k in OBSERVED)
        # 抖动判定要分开看：一侧抖不能把另一侧的真实差异一起洗掉。
        # 只有两实现各自都复现不了自己时，才是真的无法归因。
        if graph_repro and legacy_repro:
            verdict = "真回归（两实现各自可复现、彼此不同）"
        elif graph_repro:
            verdict = "graph 稳定而 legacy 抖 → 差异落在 graph 侧，记入回归待人工看"
        elif legacy_repro:
            verdict = "legacy 稳定而 graph 抖 → 差异落在 graph 侧，记入回归待人工看"
        else:
            verdict = "两实现自己都不可复现 → 模型抖动，不计入重构回归"
        rec = {
            "qid": qid,
            "graph_second_run": g2,
            "legacy_second_run": l2,
            "graph_reproducible": graph_repro,
            "legacy_reproducible": legacy_repro,
            "attribution": verdict,
        }
        if "不计入" not in verdict:
            real_regressions.append(qid)
        rows.append({"qid": qid, "attribution": rec})
        print(f"  {qid}: graph 复现={graph_repro} legacy 复现={legacy_repro} → {verdict}")

    n_compared = sum(1 for r in rows if "agree" in r)
    out = {
        "n_questions": n_compared,
        "agree": sum(1 for r in rows if r.get("agree")),
        "disagree": flips,
        "real_regressions": real_regressions,
        "attribution_rule": "同一实现两次不一致 = 模型抖动；两实现各自一致而彼此不一致 = 真回归",
        "rows": rows,
    }
    art = Path(s.resolved_artifact_dir()) / "m3_graph_ab_live.json"
    art.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n同机 A/B：{out['agree']}/{n_compared} 题两实现观测一致，不一致 {len(flips)} 题，"
          f"其中判为真回归 {len(real_regressions)} 题")
    print(f"证据： {art}")
    return len(real_regressions)


if __name__ == "__main__":
    sys.exit(main())
