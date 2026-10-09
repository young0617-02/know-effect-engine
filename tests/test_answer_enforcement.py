from __future__ import annotations

import json

import pytest

from kee.qa.answer import AnswerEngine, format_evidence, is_sensitive
from kee.retrieval.pipeline import RetrievedChunk, RetrievalResult


class FakeRetriever:
    def __init__(self, chunks, degraded=None):
        self._chunks = chunks
        self.degraded = degraded or []

    def retrieve(self, query, ctx, top_k=None):
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
    payload = {"conclusion": "", "key_points": [], "no_evidence": True, "self_assessed_confidence": 0.1}
    eng = AnswerEngine(FakeRetriever([chunk("c1")]), FakeLlm(payload))
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
