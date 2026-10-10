"""多轮指代消解的**收益与代价**：机械构造追问，整批跑，不挑好看的。

M3-05 之前只证明了「补全进了检索与提示词」（单测），没证明它对召回有帮助。
第一次真机对照就撞到一个反例：同一句「那采集成功率目标是多少？」，
无上文时 `tentative` 答出来了，有上文被补全成「采集成功率目标是多少？」反而 `escalate` 空结论——
**补全把句子里区分度的字去掉了，检索就变差了**。所以这件事必须整批量，不能靠挑一两个成功的例子收口。

追问是机械构造的，不是手写的：
    取可答题的原问句 → 删掉第一个「的」之前的主题词 → 前面冠以「那个」
例如「配电自动化终端的遥测数据上送有哪些测点」→「那个的遥测数据上送有哪些测点」。
按评测集顺序取前 N 道**全部**参与统计（含构造后仍自足、以及补全反而帮倒忙的），
这样报出来的收益才是这批题的收益，不是样本选择的结果。

两类结果分别记：
  * benefit：无上文答不出（refuse/degraded），有上文答得出 → 上文的功劳
  * harm   ：无上文答得出，有上文反而答不出 → 消解把区分度弄丢了
两者都要报，只报 benefit 等于把这份证据拿去替新功能说好话。

用法：uv run python scripts/m3_multiturn_probe.py [--limit 14]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from kee.api.app import create_app  # noqa: E402
from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402

OWNER = {"X-Kee-Role": "knowledge_owner", "X-Kee-Actor": "m3-mt-01"}


def answered(body: dict) -> bool:
    """「答出来了」按**有没有给出结论**判，不按 status 判。

    用 status∈(answer,tentative) 会把 refuse→escalate 这种真实变化读成「两边都没答出」，
    而 escalate 是可以带结论的（有权限 + 密级 3 的正常形态就是 answer/escalate 带结论转人工）。
    判据要贴着用户实际拿到什么，而不是贴着内部分类标签。
    """
    return bool((body.get("conclusion") or "").strip())


def to_anaphora(q: str) -> str:
    """机械造追问：去掉主题词，冠以「那个」。造不出（没有「的」）就返回空串跳过。"""
    i = q.find("的")
    if i <= 0:
        return ""
    tail = q[i + 1:].strip()
    if not tail or len(tail) < 4:
        return ""
    return f"那个的{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=14, help="按评测集顺序取前 N 道可答题")
    args = ap.parse_args()

    s = get_settings()
    questions = build_eval_set(AuthoritativeStore(), write_files=False)["questions"]
    pool = [q for q in questions if q["expected_behavior"] == "answer" and q["gold_chunk_ids"]]

    triples = []
    for q in pool:
        a = to_anaphora(q["question"])
        if a and a != q["question"]:
            triples.append((a, q))
        if len(triples) >= args.limit:
            break
    if not triples:
        print("一道追问都构造不出来：to_anaphora 的规则与这批题的句式不匹配，先看原始问句再改规则")
        return 2

    with TestClient(create_app()) as client:
        rows = []
        for anaphora, q in triples:
            # 无上文：全新会话 + rewrite=false。
            # 必须显式关掉改写，否则单轮扩展会把「那个的X」补成「X」，
            # 那是**另一种**改写，会把「有无上文」这组对照变成「两种改写方式」的对照。
            bare = client.post(
                "/api/v1/answers", json={"question": anaphora, "rewrite": False}, headers=OWNER
            ).json()
            # 有上文：同一会话里先问原句（让历史真的含主题），再问追问
            sid = f"s-mt-{q['qid']}"
            seed = client.post(
                "/api/v1/answers", json={"question": q["question"], "session_id": sid}, headers=OWNER
            ).json()
            follow = client.post(
                "/api/v1/answers",
                json={"question": anaphora, "session_id": sid, "rewrite": True},
                headers=OWNER,
            ).json()
            resolved = follow.get("retrieval", {}).get("query_used", "") or ""
            b_ok, f_ok = answered(bare), answered(follow)
            outcome = "benefit" if (not b_ok and f_ok) else "harm" if (b_ok and not f_ok) else "flat"
            rows.append(
                {
                    "qid": q["qid"],
                    "原句": q["question"],
                    "构造追问": anaphora,
                    "补全结果": resolved,
                    "消解是否发生": resolved != anaphora,
                    "无上文": {"status": bare.get("status"), "结论字数": len((bare.get("conclusion") or "").strip()),
                               "top": bare.get("confidence", {}).get("retrieval")},
                    "有上文": {"status": follow.get("status"), "结论字数": len((follow.get("conclusion") or "").strip()),
                               "top": follow.get("confidence", {}).get("retrieval")},
                    "种子轮状态": seed.get("status"),
                    "结果": outcome,
                    "状态是否变化": bare.get("status") != follow.get("status"),
                }
            )
            print(f"[{len(rows):>2}] {q['qid']} {outcome:<7} 追问「{anaphora[:22]}」→ 补全「{resolved[:26]}」 "
                  f"无上文={bare.get('status')} 有上文={follow.get('status')}")

    n = len(rows)
    dist = {k: sum(1 for r in rows if r["结果"] == k) for k in ("benefit", "harm", "flat")}
    resolved_n = sum(1 for r in rows if r["消解是否发生"])
    # 检索层面的收益与答案层面的收益是两件事，必须分开报：
    # 消解的作用域到检索为止，后面还有密级/置信度两道闸会改写结局。
    # 只看答案层面会把 q-003 这种「检索从 0.039 救回 0.970、但正确地转人工」记成没收益。
    GATE = 0.20
    def top(r, side):
        v = r[side]["top"]
        return float(v) if v is not None else 0.0
    recovered = sum(1 for r in rows if top(r, "无上文") < GATE <= top(r, "有上文"))
    lost = sum(1 for r in rows if top(r, "有上文") < GATE <= top(r, "无上文"))
    out = {
        "n": n,
        "resolved_rate": round(resolved_n / n, 4),
        "benefit": dist["benefit"],
        "harm": dist["harm"],
        "flat": dist["flat"],
        "net": dist["benefit"] - dist["harm"],
        "status_changed": sum(1 for r in rows if r["状态是否变化"]),
        "retrieval_gate": GATE,
        "retrieval_recovered_by_history": recovered,
        "retrieval_lost_by_history": lost,
        "note": (
            "benefit/harm 按「能否答出」判，且「答出」= 结论非空（不是 status∈answer/tentative——"
            "escalate 也可以带结论，反过来也会把真实变化读成 flat）。"
            "检索层面另算救回/丢掉：以 retrieval_gate 为界，比较两侧用于检索的证据强度。"
            "本表按评测集顺序整批构造，不剔除帮倒忙的样本。"
        ),
        "rows": rows,
    }
    art = Path(s.resolved_artifact_dir()) / "m3_multiturn_probe.json"
    art.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n样本 {n} 道｜消解发生 {resolved_n}/{n}｜答案层面 benefit {dist['benefit']}｜harm {dist['harm']}｜"
          f"flat {dist['flat']}｜净收益 {out['net']}｜状态有变化 {out['status_changed']} 道")
    print(f"检索层面（阈值 {GATE}）：有上文把检索救回 {recovered} 道｜因消解而丢掉 {lost} 道")
    print(f"证据： {art}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
