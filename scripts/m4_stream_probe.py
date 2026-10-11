"""token 级流式的真机核对：首 token 延迟过不过 ① 门禁，以及撤回率有多高。

这两件事决定这个功能该不该默认开，而它们**都不能靠读代码回答**：

1. **首 token < 1.2s** 是方案 2.2 / Gate 0 ① 留到 M4 补测的门禁项。
   本地 7B + 几千字的证据提示词，prompt eval 就可能是秒级——真数是多少只有跑才知道。
2. **撤回率**：`conclusion` 在 JSON 信封最前，引用强制要等整段解析完，
   所以流给用户看的文字必然早于校验。若「流过非空文字、最后没成为对外答案」的比例不低，
   那 token 级流式对知识服务是**负收益**（用户先看到一条答案、然后它被作废），
   该默认关掉而不是硬上。这个数必须量，不能假设它接近 0。

顺带做一件对照：同一批题分别用流式与非流式跑，最终答案载荷必须逐字段一致——
两条交付方式若悄悄给出不同结果，那测的就不是上线的那条。为省时间只抽前 N 题做这组对照。

用法：uv run python scripts/m4_stream_probe.py [--limit N] [--equiv M]
退出码 0 = 跑通并落证据；3 = 依赖不健康（判废，不写 artifact）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402
from kee.eval.metrics import percentile  # noqa: E402
from kee.llm.ollama import OllamaClient  # noqa: E402
from kee.models.registry import ResidencyScheduler  # noqa: E402
from kee.qa.answer import AnswerEngine, load_calibration_temperature  # noqa: E402
from kee.qa.graph import run_qa  # noqa: E402
from kee.qa.streaming import iter_qa_events  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402
from kee.retrieval.pipeline import HybridRetriever  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

GATE_FIRST_TOKEN_MS = 1200.0
ANSWERED = ("answer", "tentative")


def make_engine(s):
    kb = KbStore(s.milvus_collection, "jieba")
    rt = HybridRetriever(kb, scheduler=ResidencyScheduler(policy="co_resident"), cache=None,
                         mode="hybrid", use_rerank=True)
    eng = AnswerEngine(retriever=rt, llm=OllamaClient())
    eng.calibration_temperature = load_calibration_temperature(s)
    return eng, kb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--equiv", type=int, default=12, help="流式/非流式逐字段对照的题数")
    args = ap.parse_args()

    s = get_settings()
    qs = build_eval_set(AuthoritativeStore(), write_files=False)["questions"]
    pool = [q for q in qs if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]]
    if args.limit:
        pool = pool[: args.limit]

    eng, kb = make_engine(s)
    ctx = AccessContext(role="knowledge_owner")
    t_w = time.perf_counter()
    eng.answer("预热：配电自动化终端心跳超时阈值", ctx)
    print(f"[预热] {round(time.perf_counter() - t_w, 1)}s")

    rows = []
    for i, q in enumerate(pool, start=1):
        kinds, t_first_event, payload = [], None, {}
        t_first_conclusion = None
        deltas = 0
        t0 = time.perf_counter()
        for kind, body in iter_qa_events(eng, q["question"], ctx):
            if t_first_event is None:
                t_first_event = round((time.perf_counter() - t0) * 1000, 1)
            kinds.append(kind)
            if kind == "delta":
                deltas += 1
                # 「用户看到第一个字」要按**结论内容**算，不能按生成层首 token 算：
                # 信封的 `{` 也在 1.2s 内到，但屏幕上什么都没有。
                # 而且 39/58 的结论是搬运出来的（模型没写 conclusion），
                # 那种题根本没有可流的内容，first_conclusion_ms 为空是**事实**不是漏测。
                if t_first_conclusion is None and body.get("text"):
                    t_first_conclusion = round((time.perf_counter() - t0) * 1000, 1)
            if kind in ("answer", "error"):
                payload = body
        tm = payload.get("timings_ms") or {}
        rows.append(
            {
                "qid": q["qid"],
                "status": payload.get("status"),
                "event_seq_head": kinds[:2],
                "n_delta": deltas,
                "first_token_ms": tm.get("first_token_ms"),
                "first_conclusion_ms": t_first_conclusion,
                "first_event_ms": t_first_event,
                "retrieval_ms": tm.get("retrieval_ms"),
                "generate_ms": tm.get("generate_ms"),
                "e2e_ms": tm.get("e2e_ms"),
                "streamed_chars": payload.get("provisional_streamed_chars", 0),
                "retracted": bool(payload.get("provisional_retracted")),
                "conclusion_len": len((payload.get("conclusion") or "").strip()),
                "notes": payload.get("notes") or [],
            }
        )
        r = rows[-1]
        print(f"[{i:2d}/{len(pool)}] {q['qid']} {r['status']:<10} "
              f"信封首token={r['first_token_ms']} 首字可见={r['first_conclusion_ms']} "
              f"delta={r['n_delta']:2d} 流过{r['streamed_chars']}字 撤回={r['retracted']}")

    # ---- 有效性前置：整批没有生成产出时，这些数字一个都不该被写下 ----
    # 判据用 notes 里的 generation_failed/unparseable_output，而不是延迟或 status：
    # 依赖故障会伪装成「整齐的 degraded + 空结论」，而无守卫的撤回统计会把它读成 0 撤回 = 安全。
    from kee.eval.probe_guard import generation_failure_share, unusable_run

    reason = unusable_run(generation_failure_share([r["notes"] for r in rows]))
    if reason:
        print(f"\n[判废] {reason}")
        kb.close()
        return 3

    # ---- 流式与非流式的交付方式对照（抽前 N 题）----
    # 刻意**不**比结论文本：本地 7B 两次生成本来就可能不同，比文本会把生成抖动误报成
    # 「两种交付方式给出不同答案」，那种假阳性会把人引去查一个不存在的分叉。
    # 能稳定比的是：字段集合一致、answer/tentative 且未撤回时必有非空结论、事件序一致。
    mismatch = []
    for q in pool[: args.equiv]:
        kinds, streamed = [], {}
        for kind, body in iter_qa_events(eng, q["question"], ctx):
            kinds.append(kind)
            if kind == "answer":
                streamed = body
        sync = run_qa(eng, q["question"], ctx).as_dict()
        if set(streamed) - {"provisional_streamed_chars", "provisional_retracted"} != set(sync):
            mismatch.append({"qid": q["qid"], "why": "响应字段集合不同（交付方式真的分叉了）"})
        elif kinds[:2] != ["retrieval", "delta"] and kinds[:2] != ["retrieval", "answer"]:
            mismatch.append({"qid": q["qid"], "why": f"事件序异常 {kinds[:3]}"})
        elif (streamed.get("status") in ANSWERED and not streamed.get("provisional_retracted")
              and not (streamed.get("conclusion") or "").strip()):
            mismatch.append({"qid": q["qid"], "why": "答出了状态却无结论"})

    ft = [r["first_token_ms"] for r in rows if r["first_token_ms"] is not None]
    fc = [r["first_conclusion_ms"] for r in rows if r["first_conclusion_ms"] is not None]
    fe = [r["first_event_ms"] for r in rows if r["first_event_ms"] is not None]
    retracted = [r for r in rows if r["retracted"]]
    streamed_any = [r for r in rows if r["streamed_chars"]]
    out = {
        "n": len(rows),
        "gate_first_token_ms": GATE_FIRST_TOKEN_MS,
        "first_token_ms": {
            "n": len(ft),
            "p50": round(percentile(ft, 50), 1) if ft else None,
            "p95": round(percentile(ft, 95), 1) if ft else None,
            "max": max(ft) if ft else None,
            "share_under_gate": round(sum(1 for x in ft if x < GATE_FIRST_TOKEN_MS) / len(ft), 4) if ft else None,
            "warning": "这是**信封首字符**的到达时间，不是用户看到字的时间；"
                       "门禁若照这个数报就是把 `{` 当成答案",
        },
        "first_conclusion_ms": {  # 用户真正看到第一个答案字符
            "n": len(fc),
            "coverage": round(len(fc) / len(rows), 4) if rows else None,
            "p50": round(percentile(fc, 50), 1) if fc else None,
            "p95": round(percentile(fc, 95), 1) if fc else None,
            "share_under_gate": round(sum(1 for x in fc if x < GATE_FIRST_TOKEN_MS) / len(fc), 4) if fc else None,
            "note": "coverage 低于 1 是因为搬运出来的结论没有可流的原文（模型没写 conclusion）——"
                    "那些题只能等 answer 事件",
        },
        "first_event_ms": {  # 用户真正「看到东西」的时刻，含检索
            "p50": round(percentile(fe, 50), 1) if fe else None,
            "p95": round(percentile(fe, 95), 1) if fe else None,
        },
        "retraction": {
            "n_streamed": len(streamed_any),
            "n_retracted": len(retracted),
            "rate_of_streamed": round(len(retracted) / len(streamed_any), 4) if streamed_any else None,
            "rate_of_all": round(len(retracted) / len(rows), 4),
            "retracted_qids": [r["qid"] for r in retracted],
        },
        "delivery_equivalence": {
            "n_compared": min(args.equiv, len(rows)),
            "mismatches": mismatch,
            "note": "比的是响应字段集合、事件序、状态与结论的一致性；"
                    "结论文本不作对比——本地 7B 两次生成本来就可能不同，比文本会把生成抖动误报成分叉",
        },
        "rows": rows,
    }
    art = Path(s.resolved_artifact_dir()) / "m4_stream_probe.json"
    art.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    f = out["first_token_ms"]
    print(f"\n信封首 token（生成层第一个字符）：n={f['n']} p50={f['p50']} p95={f['p95']} max={f['max']} "
          f"低于 {GATE_FIRST_TOKEN_MS:.0f}ms 的比例={f['share_under_gate']}")
    g = out["first_conclusion_ms"]
    print(f"首个**可见结论字**：覆盖 {g['n']}/{len(rows)}（{g['coverage']}） "
          f"p50={g['p50']} p95={g['p95']} 低于门禁比例={g['share_under_gate']}")
    print(f"首个事件（含检索）：p50={out['first_event_ms']['p50']} p95={out['first_event_ms']['p95']}")
    rc = out["retraction"]
    print(f"撤回：流过文字的 {rc['n_streamed']} 题中撤回 {rc['n_retracted']} 题 "
          f"（{rc['rate_of_streamed']}）；占全批 {rc['rate_of_all']}  {rc['retracted_qids']}")
    print(f"交付方式对照：{out['delivery_equivalence']['n_compared']} 题，"
          f"结构不一致 {len(mismatch)} 题  {mismatch[:3]}")
    print(f"证据： {art}")
    kb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
