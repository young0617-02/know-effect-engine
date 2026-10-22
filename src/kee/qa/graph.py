"""LangGraph 类型化状态机：把在线链路从「一个大方法」拆成可单测、可回归的节点。

方案 6.3 要求「确定性优先，不做开放式自治 Agent」，所以这是一张**固定拓扑**的图，
不是让模型自己决定下一步：

    START → sensitive_gate → rewrite → retrieve → retrieval_gate
              → generate → enforce → confidence → END

三条纪律决定了这么拆：
  1. **节点纯函数化**：每个节点只读 state、只返回要更新的键，可以单独构造 state 断言。
  2. **早退显式成边**：拒答/降级/转人工是图上的分支，由 `gated` 字段驱动，
     而不是散落在方法各处的 return——否则「这条链路一共有几个出口」只能靠读代码确认。
  3. **敏感闸门放在最前**：被拦的问题不该再花一次检索与一次生成。原实现是先检索再判敏感，
     那是顺序缺陷，顺手在这张图里修正（RBAC 已在检索层过滤，所以不是泄漏，只是白烧算力）。

多轮历史由调用方从 `session_turn` 读出来注入 `state.history`。
**没有接 PostgresSaver**：当前不需要中断-恢复，而 session_turn 已经是可查的权威历史，
再加一套 checkpointer 等于同一份状态存两处。等人工复核闭环（中断-恢复）落地时再接，
这条偏差记在 M3 报告里。
"""

from __future__ import annotations

import math
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from kee.obs import metrics
from kee.qa.answer import (
    ESCORT_SECURITY_LEVEL,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    AnswerResult,
    _clamp,
    _logit,
    apply_parent_context,
    composite_score,
    format_evidence,
    is_sensitive,
)


class QAState(TypedDict, total=False):
    """图上的类型化状态。缺键即「该节点还没跑到」，所以 total=False。"""

    question: str
    ctx: Any
    top_k: int | None
    history: list[dict[str, Any]]
    trace_id: str
    t_start: float

    effective_question: str
    resolved: bool  # 图上真的产出了一条与原文不同的查询；被拒的改写不算
    rewrite_ms: float

    retrieval_res: Any
    retrieval: dict[str, Any]
    top_score: float
    # 复核闸摘掉的条数。放进状态而不是只写进 notes：出口判定要能用机械字段，
    # 「有资料但没复核」与「真没资料」这两件事今后靠它区分，不靠读 notes 字符串。
    dropped_unreviewed: int
    refusal_reason: str
    evidence_text: str
    evidence_index: dict[str, dict]
    timings_ms: dict[str, float]

    gen_obj: dict[str, Any] | None
    gen_ok: bool
    gen_error: str
    llm_meta: dict[str, Any]
    # token 级流式的回调（由 SSE 端点注入）。放进 state 而不是塞进 engine：
    # 引擎实例是跨请求复用的，回调是每次请求各自的，放引擎上会把上一请求的管道接进下一请求。
    on_delta: Any

    conclusion: str
    key_points: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    dropped_points: list[str]

    status: str
    needs_human: bool
    confidence: dict[str, Any]
    notes: list[str]
    gated: str  # 非空表示已被某个闸门终结，路由据此决定是否继续


def _components(obj: dict, top_score: float, points: list, dropped: list) -> dict[str, float]:
    """复合分的三个成分 + 由它们算出的合成分。enforce 与 confidence 共用同一份算法。

    enforce 的两个契约出口也要吐 citation_consistency / self_assessed：
    原实现在那里就记录了它们，漏掉会让「拒答的原因构成」在审计里看不全。

    `composite_candidate` 是给 enforce 的搬运下限判据用的**中间量**，
    它不该出现在对外字典里（那里的 composite 一律是 0.0，表示「这条没成立」），
    所以由 `_public()` / `_gate()` 统一剥掉，不靠每个出口各自记得。
    """
    citation_consistency = 1.0 if not dropped else round(len(points) / max(len(points) + len(dropped), 1), 4)
    self_assessed = _clamp(float(obj.get("self_assessed_confidence", 0.0) or 0.0))
    return {
        "retrieval": round(top_score, 4),
        "citation_consistency": round(citation_consistency, 4),
        "self_assessed": round(self_assessed, 4),
        "composite_candidate": composite_score(top_score, citation_consistency, self_assessed),
    }


def _public(comp: dict[str, float]) -> dict[str, float]:
    """剥掉中间量，得到可以对外发的成分字典。"""
    return {k: v for k, v in comp.items() if k != "composite_candidate"}


def _gate(comp: dict[str, float], rule: str, temperature: float) -> dict[str, Any]:
    """契约出口的置信度块：成分照登，但 composite/calibrated 记 0——这条答案没成立。

    `score_basis` 与 `temperature` 必须与作答出口**同形状**带出。此前这两个字段只在作答出口有，
    于是拒答轮给的是「composite=0 但没有一句说明这个 0 是什么」——而 0 恰恰最容易被下游读成
    「系统自评 0% 正确率」。《前端流式渲染契约》第 4 条写的是「原样带出 score_basis」，没分出口，
    所以这不是前端要兜的边界，是后端少发了字段（2026-10-13 真机走查抓到，前端当时显示「score_basis 缺失」）。
    """
    return {**_public(comp), "composite": 0.0, "calibrated": 0.0, "rule": rule,
            "temperature": temperature, "score_basis": "rule_composite_not_probability"}


def build_qa_graph(engine):
    """把节点闭包绑到具体引擎上（引擎持有 retriever / llm / 阈值）。"""

    def _n(state: QAState, *items: str) -> list[str]:
        return list(state.get("notes") or []) + list(items)

    # ---------- 闸门 ----------
    def sensitive_gate(state: QAState) -> dict[str, Any]:
        hit = is_sensitive(state["question"])
        if not hit:
            return {}
        return {
            "gated": "sensitive",
            "status": "escalate",
            "needs_human": True,
            "notes": _n(state, f"sensitive_pattern:{hit}"),
            "confidence": _gate({}, "sensitive_block", engine.calibration_temperature),
        }

    def rewrite(state: QAState) -> dict[str, Any]:
        """多轮指代消解：有历史且引擎开了改写器时，把「那汞呢」补全成自足查询。

        改写作用在**检索之前**，所以补全出来的词真的能进 BM25 与稠密向量；
        原实现把改写塞在 retriever 内部，拿不到历史，等于只能做单查询扩展。
        """
        rw = getattr(engine.retriever, "rewriter", None)
        if rw is None or not state.get("history"):
            return {}
        context = "\n".join(
            f"第{h.get('turn_no')}轮 问：{h.get('question')}\n第{h.get('turn_no')}轮 答：{h.get('conclusion', '')}"
            for h in state["history"][-3:]
        )
        res = rw.rewrite(state["question"], context=context)
        return {
            "effective_question": res.effective,
            # 只有真的补全出不同查询才算「已改写」。被校验门拒掉时 effective 就是原话，
            # 此时若也报 True，检索层会跟着跳过自己的单查询扩展——追问反而比单轮召回更差。
            "resolved": res.changed,
            "rewrite_ms": res.ms,
            "notes": _n(state, *[d for d in res.degraded]),
        }

    def retrieve(state: QAState) -> dict[str, Any]:
        timings = dict(state.get("timings_ms") or {})
        query = state.get("effective_question") or state["question"]
        # 图已经用多轮上文消解出一条**不同的**查询时，就别让检索层再改写一次：
        # 同一件事付两趟 LLM 是纯浪费。
        resolved = bool(state.get("resolved"))
        res = engine.retriever.retrieve(
            query, state["ctx"], top_k=state.get("top_k"), already_rewritten=resolved
        )
        dropped_unreviewed = engine.gate_unreviewed(res)
        top_score = max((c.score for c in res.chunks), default=0.0)
        # 指标记在**闸门实际比较的那个数**上（复核闸摘完之后），不是摘之前——
        # 否则 `le="0.2"` 那一档算出来的是「没过检索闸的比例」，看着像过闸率却少算了被摘掉的那批。
        metrics.observe_top_score(top_score)
        timings["retrieval_ms"] = res.timings_ms.get("retrieval_total_ms", 0.0)
        timings.update({f"stage_{k}": v for k, v in res.timings_ms.items() if k != "retrieval_total_ms"})
        extra_notes = [f"unreviewed_evidence_dropped:{dropped_unreviewed}"] if dropped_unreviewed else []
        return {
            "notes": _n(state, *extra_notes),
            "retrieval_res": res,
            "dropped_unreviewed": dropped_unreviewed,
            "timings_ms": timings,
            "retrieval": {
                "mode": res.mode,
                "cache_hit": res.cache_hit,
                "degraded": res.degraded,
                "expr": res.expr,
                "n_chunks": len(res.chunks),
                "top_chunk_ids": [c.chunk_id for c in res.chunks],
                "query_used": query,
            },
            "top_score": top_score,
        }

    def retrieval_gate(state: QAState) -> dict[str, Any]:
        res = state["retrieval_res"]
        top_score = float(state["top_score"])
        if res.chunks and top_score >= engine.min_retrieval_score:
            return {}
        dropped = int(state.get("dropped_unreviewed") or 0)
        # **「库里没资料」与「资料在、只是还没复核」是两件事**：前者真无解，后者是流程没走完。
        # 复核闸摘在检索闸之前，所以摘光时 `res.chunks` 为空、top_score 归 0——
        # 不分开就两种情形共用同一个拒答出口，业务读到的是「没找到相关资料」。
        # 判据是机械的（摘除条数 > 0 且摘完一条不剩），因此这里同时落对外字段与新出口标签。
        review_gated = bool(dropped) and not res.chunks
        return {
            "gated": "evidence_all_unreviewed" if review_gated else "weak_retrieval",
            "status": "refuse",
            # 复核是人力动作 ⇒ 这一类必须让下游看得见「有人可以推进」，而不是一句干巴巴的拒答。
            "needs_human": True if review_gated else top_score >= engine.min_retrieval_score,
            "refusal_reason": "all_evidence_unreviewed" if review_gated else "no_evidence",
            "unreviewed_evidence_count": dropped,
            "notes": _n(state, "all_evidence_unreviewed" if review_gated else "no_sufficient_evidence"),
            "confidence": _gate({"retrieval": round(top_score, 4)},
                                "retrieval_gate", engine.calibration_temperature),
        }

    # ---------- 生成与出口 ----------
    def generate(state: QAState) -> dict[str, Any]:
        ev_text, ev_index = format_evidence(state["retrieval_res"])
        # 父块回捞（需求 US-1.2）刻意做在**这里、提示词组装之前**：
        # 引用强制与 citations 都是按 ev_index 算的，父块只补上下文、不占证据编号，
        # 所以注入晚一步会让模型看见一批无法被引用的文字，注入进 ev_text 而不进 ev_index 正是这个原因。
        ev_text, pc_notes = apply_parent_context(engine, ev_text, ev_index, state["retrieval_res"].chunks)
        # 用补全后的查询喂生成：多轮里原文是「那个月结的呢？」，模型只看这句无法定指。
        # 补全后的查询已在检索之前生效，原文仍留在 state["question"] 里对外返回与审计。
        # 单轮时 effective_question 不设置，这里回落到原文——所以与重构前等价。
        query = state.get("effective_question") or state["question"]
        prompt = USER_TEMPLATE.format(evidence=ev_text, question=query)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        timings = dict(state.get("timings_ms") or {})
        on_delta = state.get("on_delta")
        t1 = time.perf_counter()
        first_token_ms: float | None = None
        # 流式与非流式**共用同一个 generate 节点**：拓扑、门禁、出口契约一处都不分叉。
        # 分叉成两条链是这类功能最容易犯的错——测的是一条、上线的是另一条。
        if on_delta is not None and hasattr(engine.llm, "chat_stream"):
            from kee.qa.streaming import scan_streamable

            llm_res = None
            err = ""
            buf = ""
            for kind, payload in engine.llm.chat_stream(
                messages, json_mode=True, temperature=engine.temperature, max_tokens=engine.max_tokens
            ):
                if kind == "delta":
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - t1) * 1000, 1)
                    # payload 是内容**片段**；扫描器需要「到目前为止的完整缓冲」，
                    # 因为它得从信封头部判断自己处在哪个字段里。
                    buf += payload
                    for ev in scan_streamable(buf):
                        on_delta(ev)
                elif kind == "final":
                    llm_res = payload
                elif kind == "error":
                    err = payload
            if llm_res is None:
                from kee.llm.ollama import LlmResult

                llm_res = LlmResult(ok=False, error=err or "no_output")
        else:
            llm_res = engine.llm.chat(
                messages, json_mode=True, temperature=engine.temperature, max_tokens=engine.max_tokens
            )
        timings["generate_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        timings["e2e_ms"] = round((time.perf_counter() - state["t_start"]) * 1000, 1)
        if first_token_ms is not None:
            # ① 门禁「流式首 token < 1.2s」的测量点。刻意记在从**发起请求**算起的位置，
            # 不含检索耗时——门禁要的是「用户按下回车到看见第一个字」里属于生成那一段。
            # 注意这是生成层的首 token，不是页面首屏；页面另有 retrieval 事件更早到。
            timings["first_token_ms"] = first_token_ms
        return {
            "evidence_text": ev_text,
            "evidence_index": ev_index,
            "timings_ms": timings,
            # 回捞的实际代价必须落在审计里：`parent_context_tokens` 是贴进去的 token 数，
            # `parent_context_skipped:N` 是被注入预算挡掉的块数。没有这两条，
            # 「补了上下文」与「预算不够所以没补」在日志上长得一样。
            "notes": _n(state, *pc_notes),
            "llm_meta": {
                "ok": llm_res.ok,
                "first_attempt_json_valid": llm_res.json_valid,
                "metrics": llm_res.metrics,
                "error": llm_res.error,
                "prompt_chars": len(prompt),
            },
            "gen_ok": llm_res.ok,
            "gen_error": llm_res.error,
            "gen_obj": engine._extract_json(llm_res.text) if llm_res.ok else None,
        }

    def enforce(state: QAState) -> dict[str, Any]:
        """引用强制 → 出口契约 → 逐字搬运。三者都是确定性规则，不请模型复核。"""
        top_score = float(state.get("top_score") or 0.0)
        if not state["gen_ok"]:
            res = state["retrieval_res"]
            ev_index = state["evidence_index"]
            return {
                "gated": "generation_failed",
                "status": "degraded",
                "needs_human": True,
                "notes": _n(state, f"generation_failed:{state['gen_error'][:60]}"),
                "key_points": [
                    {"point": c.text[:120], "evidence_ids": [f"E{i}"]} for i, c in enumerate(res.chunks, start=1)
                ],
                "citations": [
                    {**ev_index[f"E{i}"], "evidence_id": f"E{i}"} for i in range(1, len(res.chunks) + 1)
                ],
            }
        obj = state["gen_obj"]
        if obj is None:
            return {
                "gated": "unparseable",
                "status": "degraded",
                "needs_human": True,
                "notes": _n(state, "unparseable_output"),
            }
        if obj.get("no_evidence") is True:
            # 反向兜底：模型声明无依据时，① 一律挂人工；② 只有检索层也不同意「无依据」时
            # 才把 refuse 改成 escalate——此时 refuse 是在断言一件检索层反对的事。
            strong = top_score >= engine.strong_evidence_floor
            notes = ["model_declared_no_evidence"]
            if strong:
                notes.append(
                    f"strong_evidence_but_no_basis_claimed:{top_score:.4f}>={engine.strong_evidence_floor}"
                )
            return {
                "gated": "declared_no_evidence",
                "status": "escalate" if strong else "refuse",
                "needs_human": True,
                "notes": _n(state, *notes),
                "confidence": _gate({"retrieval": round(top_score, 4)},
                                    "declared_no_evidence", engine.calibration_temperature),
            }

        points, cites, dropped = engine._enforce_citations(obj, state["evidence_index"])
        comp = _components(obj, top_score, points, dropped)
        out: dict[str, Any] = {"key_points": points, "citations": cites, "dropped_points": dropped}
        if not points or not cites:
            return {
                **out,
                "gated": "no_citation",
                "status": "refuse",
                "notes": _n(state, "citation_enforcement_rejected"),
                "confidence": _gate(comp, "contract_no_citation", engine.calibration_temperature),
            }
        conclusion = str(obj.get("conclusion", "") or "").strip()
        carried: list[str] = []
        if not conclusion and points:
            # 实测 45/58 是「答案写在要点里、结论留空」的字段错位：逐字搬运而不是判死。
            # 不用「要求结论非空」的提示词修法——那会把 7 道无答案题从 0 道出结论变成 2 道。
            #
            # 但搬运有前提：**只有当其它独立信号都站得住时，空结论才算字段错位**。
            # q-061（top_score 0.2074、自评 0、composite 0.4433）是「证据勉强过检索门槛 +
            # 模型自己既没写结论也没给信心」，这时把要点拼成结论，等于由我们把一条站不住的答案
            # 说出来——那是「无依据必拒」的漏口，不是完成率的成绩。
            # 下限直接用现成的 confidence_low，不另造阈值：低于它本来就要转人工，
            # 那它同时也该阻止我们替模型补结论。安全性已核：39 行搬运无一低于 0.45
            # （最近的是 q-046 的 0.4588），所以这条不伤完成率——但余量只有 0.009，见报告。
            if comp["composite_candidate"] < engine.settings.confidence_low:
                return {
                    **out,
                    "key_points": [],
                    "citations": [],
                    "gated": "marginal_evidence_no_derivation",
                    "status": "refuse",
                    "needs_human": False,
                    "notes": _n(state, "refused_low_confidence_no_basis",
                                f"composite<{engine.settings.confidence_low} 且模型未给结论，不代为补全"),
                    "confidence": _gate(comp, "marginal_evidence_refusal", engine.calibration_temperature),
                }
            derived = "；".join(
                str(p.get("point", "")).strip() for p in points if str(p.get("point", "")).strip()
            )
            if derived:
                conclusion = derived
                carried.append(f"conclusion_derived_from_points:{len(points)}")
        if not conclusion:
            return {
                **out,
                "gated": "empty_conclusion",
                "status": "degraded",
                "needs_human": True,
                "notes": _n(state, *carried, "contract_violation:empty_conclusion"),
                "confidence": _gate(comp, "contract_empty_conclusion", engine.calibration_temperature),
            }
        out["conclusion"] = conclusion
        if carried:
            out["notes"] = _n(state, *carried)
        return out

    def confidence(state: QAState) -> dict[str, Any]:
        obj = state.get("gen_obj") or {}
        top_score = float(state.get("top_score") or 0.0)
        comp = _components(obj, top_score, state.get("key_points") or [], state.get("dropped_points") or [])
        # 与 enforce 的搬运下限共用同一个 composite_score()：一处改权重另一处不会跟着走，
        # 那就是「分档依据」和「要不要替模型补结论」用了两个不同的分数——不可解释。
        composite = comp["composite_candidate"]
        calibrated = round(
            1.0 / (1.0 + math.exp(-(_logit(composite) / engine.calibration_temperature))), 4
        )
        notes = list(state.get("notes") or [])
        # 引用落在密级 3 的证据上 → 答案可给（用户有权限）但强制挂人工复核。
        # 触发点是「实际被引用的证据」而不是「候选证据集」：后者让有权限的用户几乎问不出答案。
        secret = [c for c in state.get("citations") or [] if int(c.get("security_level") or 0) >= ESCORT_SECURITY_LEVEL]
        if secret:
            notes.append(f"secret_level_cited:{len(secret)} 条需人工复核")
        if calibrated < engine.settings.confidence_low:
            status, needs = "escalate", True
            notes.append("low_confidence_route_to_human")
        elif calibrated < engine.settings.confidence_high:
            status, needs = "tentative", False
            notes.append("初步判断（低置信度显式降级）")
        else:
            status, needs = "answer", False
        return {
            "status": status,
            "needs_human": bool(needs or secret),
            "notes": notes,
            "confidence": {
                **_public(comp),
                "composite": composite,
                "logit": round(_logit(composite), 4),
                "calibrated": calibrated,
                "temperature": engine.calibration_temperature,
                # 明确标注这个数的性质，防止下游（前端、审计、对外材料）把它当概率使用。
                # calibrated 只在 temperature != 1.0 时才经过缩放；两者都**不是**正确概率。
                "score_basis": "rule_composite_not_probability",
            },
        }

    # ---------- 路由：只看 gated，不看散落的字段组合 ----------
    def route(state: QAState) -> str:
        return END if state.get("gated") else "continue"

    g = StateGraph(QAState)
    for name, fn in {
        "sensitive_gate": sensitive_gate,
        "rewrite": rewrite,
        "retrieve": retrieve,
        "retrieval_gate": retrieval_gate,
        "generate": generate,
        "enforce": enforce,
        "confidence": confidence,
    }.items():
        g.add_node(name, fn)

    g.add_edge(START, "sensitive_gate")
    g.add_conditional_edges("sensitive_gate", route, {END: END, "continue": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "retrieval_gate")
    g.add_conditional_edges("retrieval_gate", route, {END: END, "continue": "generate"})
    g.add_edge("generate", "enforce")
    g.add_conditional_edges("enforce", route, {END: END, "continue": "confidence"})
    g.add_edge("confidence", END)
    return g.compile()


def initial_state(question: str, ctx, top_k=None, history=None) -> QAState:
    """组装图的初始状态。SSE 端点要用同一份，不能让两条入口各写一套默认值。"""
    return {
        "question": question,
        "ctx": ctx,
        "top_k": top_k,
        "history": list(history or []),
        "trace_id": f"qa-{int(time.time() * 1000) % 10**12}",
        "t_start": time.perf_counter(),
        "notes": [],
        "timings_ms": {},
        "status": "refuse",
        "needs_human": False,
        "confidence": {},
    }


def result_from_state(state: QAState) -> AnswerResult:
    """把（可能是中途结束的）状态收回对外的 AnswerResult。"""
    timings = dict(state.get("timings_ms") or {})
    if state.get("rewrite_ms"):
        timings["rewrite_ms"] = state["rewrite_ms"]
    # e2e_ms 原先只在 generate 节点里写。拒答/转人工/契约失败这些**不进生成**的路径
    # 就一个耗时都没有，于是 session_turn.total_ms 成 NULL——
    # 任何从那张表算的 p50/p95 都在悄悄丢掉最快的轮次（此前那份 n=205 就是这么偏的）。
    # 收口放在这个唯一的出口处：所有结束路径都要有 e2e。
    if "e2e_ms" not in timings and state.get("t_start"):
        timings["e2e_ms"] = round((time.perf_counter() - state["t_start"]) * 1000, 1)
    res = AnswerResult(
        question=state["question"],
        status=state.get("status", "refuse"),
        conclusion=state.get("conclusion", ""),
        key_points=state.get("key_points", []),
        citations=state.get("citations", []),
        dropped_points=state.get("dropped_points", []),
        needs_human=bool(state.get("needs_human")),
        refusal_reason=str(state.get("refusal_reason") or ""),
        # 计数在**部分摘除**时也不为零：答出来了但「本题还有 N 条相关资料没复核」这句话
        # 对业务同样有用，所以它不只属于拒答路径。
        unreviewed_evidence_count=int(state.get("dropped_unreviewed") or 0),
        confidence=state.get("confidence", {}),
        retrieval=state.get("retrieval", {}),
        llm=state.get("llm_meta", {}),
        timings_ms=timings,
        trace_id=state.get("trace_id", ""),
        notes=list(state.get("notes") or []),
    )
    # 指标只在**这一个收口**记：run_qa（非流式）与 iter_qa_events（流式）都经过这里，
    # 分两处记就会一条链路算两次、或某条早退路径静默失踪。出口标签用 `gated` 原值。
    metrics.observe_qa_result(res, gated=state.get("gated") or "")
    return res


def run_qa(engine, question: str, ctx, top_k=None, history=None) -> AnswerResult:
    """图的唯一入口：组装初始状态、跑图、把结果收回 AnswerResult。"""
    return result_from_state(engine.graph.invoke(initial_state(question, ctx, top_k, history)))
