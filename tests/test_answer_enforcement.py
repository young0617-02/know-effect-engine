from __future__ import annotations

import json

import pytest

from kee.qa.answer import AnswerEngine, format_evidence, is_sensitive
from kee.retrieval.pipeline import RetrievedChunk, RetrievalResult


class FakeRetriever:
    def __init__(self, chunks, degraded=None):
        self._chunks = chunks
        self.degraded = degraded or []

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        return RetrievalResult(
            query=query,
            mode="hybrid",
            chunks=self._chunks,
            timings_ms={"retrieval_total_ms": 10.0, "retrieval_ms": 10.0},
            expr="security_level <= 3",
            degraded=self.degraded,
        )


class FakeLlm:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok
        self.last_prompt = ""

    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        from kee.llm.ollama import LlmResult

        self.last_prompt = messages[-1]["content"]
        text = json.dumps(self._payload, ensure_ascii=False) if isinstance(self._payload, dict) else self._payload
        return LlmResult(
            ok=self.ok,
            text=text,
            parsed=None,
            json_valid=isinstance(self._payload, dict),
            metrics={"total_ms": 500.0, "tokens_per_sec": 40.0, "completion_tokens": 20},
        )


def chunk(cid, score=0.9, doc="fin-doc", section="2.1", page=5, level=2):
    return RetrievedChunk(
        chunk_id=cid,
        text=f"证据正文 {cid}",
        score=score,
        fields={
            "chunk_id": cid,
            "doc_id": doc,
            "project": "PRJ-FIN-2024-017",
            "section_no": section,
            "page": page,
            "security_level": level,
            "parent_id": f"{doc}#p",
        },
    )


def ctx():
    from kee.retrieval.rbac import AccessContext

    return AccessContext(role="knowledge_owner")


def good_payload(eids=("E1",)):
    return {
        "conclusion": "RTO 小于 15 分钟。",
        "key_points": [{"point": "灾备切换 RTO < 15 分钟", "evidence_ids": list(eids)}],
        "citations": [{"evidence_id": e for e in eids}],
        "no_evidence": False,
        "self_assessed_confidence": 0.8,
    }


def test_citation_required_points_must_carry_evidence():
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(good_payload()))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status in ("answer", "tentative")
    assert res.citations and res.citations[0]["chunk_id"] == "c1"
    assert res.confidence["citation_consistency"] == 1.0
    assert res.confidence["calibrated"] > res.confidence["composite"] * 0.5


def test_points_without_evidence_are_dropped():
    payload = good_payload()
    payload["key_points"].append({"point": "凭空补充的要点", "evidence_ids": ["E9"]})
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert len(res.key_points) == 1
    assert res.dropped_points == ["凭空补充的要点"]
    assert res.confidence["citation_consistency"] == 0.5


def test_conclusion_without_any_citation_is_refused():
    payload = good_payload()
    payload["key_points"] = [{"point": "无出处要点", "evidence_ids": ["E7"]}]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "refuse"
    assert res.citations == []
    assert "conclusion_without_citation" in res.dropped_points


def test_model_declared_no_evidence_leads_to_refuse():
    """弱证据下声明无依据 → 仍是 refuse（「没有可用依据」这个断言成立）。
    强证据下的分支已变，见 M3-02 的 test_strong_evidence_with_declared_no_basis_escalates...。"""
    payload = {"conclusion": "", "key_points": [], "no_evidence": True, "self_assessed_confidence": 0.1}
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.42)]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "refuse"
    assert "model_declared_no_evidence" in res.notes


def test_low_retrieval_score_short_circuits_before_llm():
    llm = FakeLlm(good_payload())
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.01)]), llm)
    res = eng.answer("无关问题", ctx())
    assert res.status == "refuse"
    assert llm.last_prompt == ""  # 根本没调用生成


def test_empty_retrieval_refuses():
    eng = AnswerEngine(FakeRetriever([]), FakeLlm(good_payload()))
    res = eng.answer("无关问题", ctx())
    assert res.status == "refuse"
    assert res.retrieval["n_chunks"] == 0


def test_sensitive_question_esculates_without_generating():
    llm = FakeLlm(good_payload())
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), llm)
    res = eng.answer("这家客户的合同金额和报价是多少", ctx())
    assert res.status == "escalate"
    assert res.needs_human is True
    assert llm.last_prompt == ""
    assert is_sensitive("生产系统口令策略") == "口令"


def test_generation_failure_degrades_to_fragments_only():
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm("服务不可用", ok=False))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "degraded"
    assert res.needs_human is True
    assert res.citations  # 仍给出可溯源片段
    assert any(n.startswith("generation_failed") for n in res.notes)


def test_unparseable_output_is_flagged_and_escalated():
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm("这是一段散文，不是 JSON"))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "degraded"
    assert "unparseable_output" in res.notes


def test_evidence_ids_are_whitelisted_by_this_retrieval():
    _, index = format_evidence(RetrievalResult("q", "hybrid", [chunk("c1"), chunk("c2")], {}, ""))
    assert set(index) == {"E1", "E2"}
    assert index["E2"]["chunk_id"] == "c2"


def test_temperature_hook_applies_to_calibrated_confidence():
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(good_payload()))
    eng.calibration_temperature = 4.0
    res = eng.answer("RTO 是多少", ctx())
    assert res.confidence["temperature"] == 4.0
    assert res.confidence["calibrated"] < res.confidence["composite"]  # T>1 拉回置信度


def test_cited_secret_level_answer_is_flagged_for_human_review():
    # 触发点是「被引用的密级3证据」，不是候选集：有权限的用户仍应拿到答案，但必须挂复核标记
    llm = FakeLlm(good_payload())
    eng = AnswerEngine(FakeRetriever([chunk("c1", level=3)]), llm)
    res = eng.answer("这次交付的验收流程分几步，每步的输出物是什么？", ctx())
    assert llm.last_prompt != ""
    assert res.status in ("answer", "tentative", "escalate")
    assert res.needs_human is True
    assert any(n.startswith("secret_level_cited") for n in res.notes)


def test_non_secret_citation_does_not_demand_review():
    llm = FakeLlm(good_payload())
    eng = AnswerEngine(FakeRetriever([chunk("c1", level=2)]), llm)
    res = eng.answer("RTO 是多少", ctx())
    assert not any(n.startswith("secret_level_cited") for n in res.notes)
    assert res.needs_human is False


@pytest.mark.parametrize("bad", ["报价", "客户名单", "密钥", "薪酬"])
def test_sensitive_patterns(bad):
    assert is_sensitive(f"给我看看{bad}") == bad


# ---- M3-01b 答案结构契约：有出处不等于答案成立 ----
# 契约的触发条件是「结论为空**且搬运不出结论**」。结论为空但要点可用时，
# 走 M3-01c 的逐字搬运（那是字段错位，不是没有答案），见下方 derive 用例。


def test_empty_conclusion_cannot_be_reported_as_answer():
    payload = good_payload()
    payload["conclusion"] = ""
    payload["key_points"] = [{"point": "   ", "evidence_ids": ["E1"]}]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "degraded"
    assert res.needs_human is True
    assert "contract_violation:empty_conclusion" in res.notes
    assert res.conclusion == ""
    # 出处仍然留给人工，不因为契约不过就把线索丢掉
    assert res.citations


def test_whitespace_only_conclusion_is_still_a_violation():
    payload = good_payload()
    payload["conclusion"] = "     "
    payload["key_points"] = [{"point": "  ", "evidence_ids": ["E1"]}]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "degraded" and res.confidence["rule"] == "contract_empty_conclusion"


def test_contract_violated_answer_stays_out_of_calibration_set():
    """可判集按 logit 是否存在筛；空结论若仍带 logit，ECE 就会被测成「答案完整率」。"""
    payload = good_payload()
    payload["conclusion"] = ""
    payload["key_points"] = [{"point": "  ", "evidence_ids": ["E1"]}]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert "logit" not in res.confidence
    assert res.confidence["composite"] == 0.0


def test_valid_conclusion_still_produces_calibratable_answer():
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(good_payload()))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status in ("answer", "tentative")
    assert res.needs_human is False
    assert "logit" in res.confidence and res.conclusion


# ---- M3-01c：答案写在要点里、结论留空时，逐字搬运而不是判死 ----
# 实测 45/58 是这个形态。不能改成「提示词要求结论非空」——那会把 7 道无答案题从
# 0 道出结论变成 2 道出结论（q-059 编出「一次通过率93.7%」且不转人工），方向更坏。


def test_empty_conclusion_is_derived_verbatim_from_key_points():
    payload = good_payload()
    payload["conclusion"] = ""
    payload["key_points"] = [
        {"point": "灾备切换 RTO < 15 分钟", "evidence_ids": ["E1"]},
        {"point": "切换窗口每月两个", "evidence_ids": ["E1"]},
    ]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status in ("answer", "tentative")
    assert res.needs_human is False
    # 逐字搬运，不生成任何新表述
    assert res.conclusion == "灾备切换 RTO < 15 分钟；切换窗口每月两个"
    assert "conclusion_derived_from_points:2" in res.notes


def test_derived_conclusion_does_not_invent_text():
    """搬运只能用要点原文：结论里出现的每个字都必须来自某个要点。"""
    payload = good_payload()
    payload["conclusion"] = ""
    payload["key_points"] = [{"point": "采集成功率 99.3%", "evidence_ids": ["E1"]}]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("采集成功率多少", ctx())
    assert res.conclusion == "采集成功率 99.3%"
    assert all(ch in "采集成功率 99.3%；" for ch in res.conclusion)


def test_all_points_blank_still_degrades_to_human():
    """要点也空时搬运不出东西，契约仍然要兜住，不能出现空答案冒充成功。"""
    payload = good_payload()
    payload["conclusion"] = ""
    payload["key_points"] = [{"point": "   ", "evidence_ids": ["E1"]}]
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "degraded" and res.needs_human is True
    assert "contract_violation:empty_conclusion" in res.notes


# ---- M3-02 反向兜底：强证据却声明无依据 ----
# 复现的是 Gate 0 ⑤ 唯一未闭环项 q-070：检索分 0.9683、模型声明无依据、
# 系统只是 refuse 且 needs_human=False，既没复核也没承认自己判错。


def no_basis_payload():
    return {"conclusion": "", "key_points": [], "citations": [], "no_evidence": True,
            "self_assessed_confidence": 0.0}


def test_strong_evidence_with_declared_no_basis_escalates_instead_of_refusing():
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.9683)]), FakeLlm(no_basis_payload()))
    res = eng.answer("电子记录那个项目的应急储备金比例是多少", ctx())
    assert res.status == "escalate"
    assert res.needs_human is True
    assert any(n.startswith("strong_evidence_but_no_basis_claimed") for n in res.notes)
    assert "model_declared_no_evidence" in res.notes


def test_weak_evidence_with_declared_no_basis_still_refuses_but_flags_review():
    """第一层不需要阈值：知识服务里「查不出来」恰恰是最该被人看见的出口。"""
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.55)]), FakeLlm(no_basis_payload()))
    res = eng.answer("今年食堂供应商的中标名单", ctx())
    assert res.status == "refuse"
    assert res.needs_human is True
    assert not any(n.startswith("strong_evidence_but_no_basis_claimed") for n in res.notes)


def test_below_the_floor_is_not_called_strong():
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.8745)]), FakeLlm(no_basis_payload()))
    res = eng.answer("某项内部指标是多少", ctx())
    assert res.status == "refuse" and res.needs_human is True


def test_no_basis_answer_stays_out_of_calibration_set():
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.9683)]), FakeLlm(no_basis_payload()))
    res = eng.answer("电子记录那个项目的应急储备金比例是多少", ctx())
    assert "logit" not in res.confidence and res.confidence["composite"] == 0.0


# ---- M3-04b 搬运下限：证据勉强过线且模型没写结论时，不许由我们替它补 ----
# 复现的是 ⑤ 唯一未闭环的那一条（q-061：top_score=0.2074，比检索门槛 0.20 只高 0.0074，
# 自评 0，结论留空，1 条要点）。它 composite=0.4433 < confidence_low=0.45。


def test_marginal_evidence_must_not_have_its_conclusion_written_for_it():
    payload = good_payload()
    payload["conclusion"] = ""
    payload["self_assessed_confidence"] = 0.0
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.2074)]), FakeLlm(payload))
    res = eng.answer("手册里有配置依据吗", ctx())
    assert res.status == "refuse"
    assert res.conclusion == "", "低置信下搬运出的结论就是「无依据必拒」的漏口"
    # 补出来的要点也不能留着：它们正是被搬运拼成结论的那些字
    assert res.key_points == [] and res.citations == []
    assert "refused_low_confidence_no_basis" in res.notes


def test_derivation_still_fires_when_evidence_is_solid():
    """反向对照：同一份 payload 只把证据强度抬上去，搬运必须照常发生。

    没有这一条，上一条测试可能只是「搬运整条被关掉了」——那会把完成率打回 24% 而无人察觉。
    """
    payload = good_payload()
    payload["conclusion"] = ""
    payload["self_assessed_confidence"] = 0.0
    eng = AnswerEngine(FakeRetriever([chunk("c1", score=0.95)]), FakeLlm(payload))
    res = eng.answer("RTO 是多少", ctx())
    assert res.conclusion != ""
    assert any(str(n).startswith("conclusion_derived_from_points") for n in res.notes)


def test_derivation_floor_uses_the_same_composite_as_the_grader():
    """下限与分档必须是同一个分数，否则「要不要补结论」和「这答案多可信」各算各的。

    取 composite 恰在 0.45 两侧的两个证据强度：低的那条 refuse、高的那条不许 refuse。
    """
    from kee.config import get_settings

    low = get_settings().confidence_low
    # 目标 composite 取门槛**两侧**而不是门槛上：score=(target-0.35)/0.45（自评 0、一致性 1）
    def score_for(target_composite: float) -> float:
        return round((target_composite - 0.35) / 0.45, 4)

    for score, want_refused in ((score_for(low - 0.01), True), (score_for(low + 0.02), False)):
        payload = good_payload()
        payload["conclusion"] = ""
        payload["self_assessed_confidence"] = 0.0
        eng = AnswerEngine(FakeRetriever([chunk("c1", score=score)]), FakeLlm(payload))
        res = eng.answer("RTO 是多少", ctx())
        is_refused = "refused_low_confidence_no_basis" in res.notes
        assert want_refused == is_refused, f"score={score} composite≈{0.45 * score + 0.35:.4f} 判据与分档不同源"
