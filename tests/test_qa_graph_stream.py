"""SSE 端点依赖的是 `graph.stream(..., stream_mode="updates")`，而 `/answers` 依赖 `invoke`。
两条入口跑的是同一张图，但**状态回收方式不同**：invoke 直接拿终态，stream 要自己把每个节点的
增量累加回去。累加漏一个键，流式版答案就会和同步版悄悄不一样——而同步版是有测试的，流式版没有，
这类不对称正是「测过的代码没问题、上线的是没测的那条」的标准成因。

所以这里钉的是：**stream 累加出来的状态，必须和 invoke 的终态收成同一个答案。**
不需要连任何服务：检索与生成用桩，验的是图与状态回收本身。
"""

from __future__ import annotations

import json

from kee.qa.answer import AnswerEngine
from kee.qa.graph import initial_state, result_from_state, run_qa
from kee.retrieval.rbac import AccessContext

from test_qa_multiturn import FakeLlm, SpyRetriever, chunk


def engine():
    r = SpyRetriever([chunk("c1"), chunk("c2")])
    return AnswerEngine(r, FakeLlm()), r


def collect(engine_obj, question, ctx, **kw):
    """复刻 SSE 端点里那段累加逻辑（而不是调端点）：两边必须逐行同构才可比。"""
    state = initial_state(question, ctx, **kw)
    order: list[str] = []
    for update in engine_obj.graph.stream(state, stream_mode="updates"):
        for node, delta in update.items():
            order.append(node)
            state.update(delta or {})
    return order, result_from_state(state), state


def test_nodes_fire_in_declared_topology():
    eng, _ = engine()
    order, _, _ = collect(eng, "灾备的 RTO 是多少", AccessContext(role="knowledge_owner"))
    assert order == [
        "sensitive_gate", "rewrite", "retrieve", "retrieval_gate", "generate", "enforce", "confidence"
    ]
    # rewrite 一定在图上，但没历史时它返回空增量：节点会出现在轨迹里，状态却不多一个键。
    # 这正是端点「只在 retrieve 之后发事件」而不是「按节点数发事件」的理由。


def test_stream_and_invoke_agree_on_the_same_answer():
    eng, _ = engine()
    ctx = AccessContext(role="knowledge_owner")
    _, streamed, _ = collect(eng, "灾备的 RTO 是多少", ctx)
    invoked = run_qa(eng, "灾备的 RTO 是多少", ctx)
    a, b = streamed.as_dict(), invoked.as_dict()
    for k in ("timings_ms", "trace_id"):
        a.pop(k), b.pop(k)
    assert a == b, f"流式与同步收敛不一致：{ {k: (a.get(k), b.get(k)) for k in a if a.get(k) != b.get(k)} }"


def test_sensitive_question_stream_stops_before_retrieval():
    """早退路径没有 retrieval 事件可发——端点因此不能无条件读 state["retrieval"]。"""
    eng, r = engine()
    order, res, state = collect(eng, "合同金额是多少", AccessContext(role="knowledge_owner"))
    assert order == ["sensitive_gate"]
    assert "retrieve" not in state
    assert r.calls == []
    assert res.status == "escalate"


def test_weak_retrieval_stream_still_emits_its_retrieval_event():
    """检索跑了但被判弱证据拦下：retrieval 事件已经发给前端了，答案事件必须补上，
    否则前端会停在「已收到检索、永远等不到答案」的状态。"""
    eng, _ = engine()
    eng.retriever = SpyRetriever([chunk("c1", score=0.03)])
    order, res, _ = collect(eng, "灾备的 RTO 是多少", AccessContext(role="knowledge_owner"))
    assert "retrieve" in order
    assert order[-1] == "retrieval_gate"
    assert res.status == "refuse"


def test_stream_delta_accumulation_does_not_drop_late_keys():
    """累加语义的前提是「后跑的节点覆盖前一个的同名字段」。
    enforce 先写 notes，confidence 再追加——若端点按节点顺序整体替换而非合并，
    前半段的降级原因会被后半段吃掉。"""
    eng, _ = engine()
    eng.retriever = SpyRetriever([chunk("c1", score=0.9, level=3)])
    order, res, _ = collect(eng, "灾备的 RTO 是多少", AccessContext(role="knowledge_owner"))
    assert order[-1] == "confidence"
    assert any("secret_level_cited" in n for n in res.notes)
    assert json.dumps(res.as_dict(), ensure_ascii=False, default=str)
