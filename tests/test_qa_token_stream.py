"""token 级流式的行为测试。

重点是两件容易在实现时悄悄丢掉的事：
  1. **流式与非流式必须给出同一个最终答案**。两条链一旦分叉，测的就不是上线的那条。
  2. **撤回要能被算出来**。引用强制跑在校验之后，所以「先显示、后作废」在设计上必然会发生；
     不把它算成一个可观测的布尔值，就等于把风险从可见变成不可见。
"""

from __future__ import annotations

import json

from kee.llm.ollama import LlmResult
from kee.qa.answer import AnswerEngine
from kee.qa.graph import run_qa
from kee.qa.streaming import iter_qa_events
from kee.retrieval.pipeline import RetrievedChunk, RetrievalResult
from kee.retrieval.rbac import AccessContext


class StreamLlm:
    """既支持 chat() 也支持 chat_stream()，后者把整段文本按固定片长切开吐。

    刻意用「定长切片」而不是逐字符：切点会落在转义序列中间、落在键名中间，
    那些位置才是扫描器真正会被咬的地方。
    """

    def __init__(self, text: str, chunk_size: int = 3, fail: bool = False):
        self.text = text
        self.chunk_size = chunk_size
        self.fail = fail

    def _result(self) -> LlmResult:
        if self.fail:
            return LlmResult(ok=False, error="http 500: llama-server binary not found", metrics={})
        return LlmResult(ok=True, text=self.text, parsed=None, json_valid=True,
                         metrics={"total_ms": 100.0, "completion_tokens": 20, "done_reason": "stop"})

    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        return self._result()

    def chat_stream(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        if self.fail:
            yield ("error", "http 500: llama-server binary not found")
            return
        for i in range(0, len(self.text), self.chunk_size):
            yield ("delta", self.text[i : i + self.chunk_size])
        yield ("final", self._result())


def chunk(cid, score=0.9, level=2):
    return RetrievedChunk(
        chunk_id=cid, text=f"证据正文 {cid}", score=score,
        fields={"chunk_id": cid, "doc_id": "fin-doc", "project": "PRJ-FIN-2024-017",
                "section_no": "2.1", "page": 5, "security_level": level, "parent_id": "fin-doc#p"},
    )


class SpyRetriever:
    rewriter = None

    def __init__(self, chunks=None):
        self.chunks = chunks or [chunk("c1"), chunk("c2")]

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        return RetrievalResult(query=query, mode="hybrid", chunks=self.chunks,
                               timings_ms={"retrieval_total_ms": 10.0}, expr="", degraded=[])


def payload_json(**over):
    base = {
        "conclusion": "灾备切换 RTO 小于 15 分钟。",
        "key_points": [{"point": "灾备 RTO < 15 分钟", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1"}],
        "no_evidence": False,
        "self_assessed_confidence": 0.9,
    }
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


def build(text: str):
    return AnswerEngine(SpyRetriever(), StreamLlm(text))


def collect(engine, question="RTO 是多少", token_stream=True):
    kinds, deltas, final = [], [], {}
    for kind, payload in iter_qa_events(engine, question, AccessContext(role="knowledge_owner"),
                                        token_stream=token_stream):
        kinds.append(kind)
        if kind == "delta":
            deltas.append(payload)
        if kind in ("answer", "error"):
            final = payload
    return kinds, deltas, final


def test_event_sequence_is_retrieval_then_deltas_then_answer():
    kinds, deltas, final = collect(build(payload_json()))
    assert kinds[0] == "retrieval" and kinds[-1] == "answer"
    assert set(kinds[1:-1]) == {"delta"}, f"中间只该是 delta 事件，实得 {kinds}"


def test_every_streamed_field_is_a_prefix_of_what_it_became():
    """逐段给用户看的文字，必须是它最终成为的那段内容的前缀。

    结论与每条要点是**各自独立**的流，所以要按 (field,index) 分开收，
    不能拿最后一条 delta 当「流出去的文字」——那可能是第二条要点，与结论无关。
    """
    kinds, deltas, final = collect(build(payload_json()))
    assert deltas, "一次 delta 都没有，等于没在流式"
    shown: dict = {}
    for d in deltas:
        assert d["provisional"] is True
        shown[(d["field"], d["index"])] = d["text"]
    pool = [final["conclusion"]] + [p["point"] for p in final["key_points"]]
    for (field, idx), text in shown.items():
        assert any(p.startswith(text) for p in pool), f"{field}[{idx}] 流出了最终内容之外的文字：{text!r}"


def test_key_points_are_streamed_so_most_answers_have_something_to_show():
    """实测 79% 的答案 conclusion 是搬运来的、模型没写——只流结论对它们无事可做。

    这条断言的存在理由就是这个测量结果：要点文字是模型生成的，流它才有覆盖率。
    """
    text = payload_json(conclusion="")  # 模型没写结论，答案全靠要点
    kinds, deltas, final = collect(build(text))
    point_events = [d for d in deltas if d["field"] == "point"]
    assert point_events, "模型没写结论时一个要点事件都没有，等于对这类题没流式"
    assert final["provisional_streamed_chars"] > 0
    # 要点全部流完且最终答案给出了搬运出来的结论 → 不算撤回
    assert final["provisional_retracted"] is False


def test_final_answer_matches_the_non_streaming_path_exactly():
    """同一条链路、两种交付方式，收敛必须一致（除流式专属的两个字段）。"""
    text = payload_json()
    _, _, streamed = collect(build(text))
    sync = run_qa(build(text), "RTO 是多少", AccessContext(role="knowledge_owner")).as_dict()
    a, b = dict(streamed), dict(sync)
    for k in ("timings_ms", "trace_id", "session_id", "turn_no"):
        a.pop(k, None), b.pop(k, None)
    for k in ("provisional_streamed_chars", "provisional_retracted",
              "provisional_lost", "provisional_lost_samples"):
        a.pop(k, None)
    assert a == b, f"流式与同步收敛不一致：{ {k: (a.get(k), b.get(k)) for k in a if a.get(k) != b.get(k)} }"


def test_retraction_is_recorded_when_the_text_never_becomes_an_answer():
    """流完一段文字、最后因引用不成立而拒答——这必须被算成撤回，而不是悄悄发生。

    构造：结论有内容，但唯一要点的 evidence_ids 指向不存在的证据 → 引用强制全弃 → refuse。
    """
    bad = payload_json(key_points=[{"point": "无出处的要点", "evidence_ids": ["E9"]}])
    kinds, deltas, final = collect(build(bad))
    assert final["status"] == "refuse"
    assert deltas and deltas[-1]["text"], "这条用例要求确实流出了文字，否则撤回断言是空转"
    assert final["provisional_retracted"] is True
    assert final["provisional_streamed_chars"] > 0


def test_no_retraction_flag_when_the_streamed_text_became_the_answer():
    _, _, final = collect(build(payload_json()))
    assert final["provisional_retracted"] is False


def test_token_stream_off_is_the_old_stage_level_contract():
    kinds, deltas, final = collect(build(payload_json()), token_stream=False)
    assert kinds == ["retrieval", "answer"]
    assert deltas == []
    assert final["provisional_streamed_chars"] == 0


def test_first_token_latency_is_recorded_for_the_stream_gate():
    """① 门禁「流式首 token < 1.2s」要有测量点；不记就等于这条门禁无法核对。"""
    _, _, final = collect(build(payload_json()))
    ms = final["timings_ms"].get("first_token_ms")
    assert isinstance(ms, float) and ms >= 0
    # 首 token 必然不晚于整段生成结束
    assert ms <= final["timings_ms"]["generate_ms"]


def test_generation_error_ends_the_stream_with_an_error_free_answer_event():
    """流断了也不能让前端停在「收到一半」。这里选择以 answer 事件收尾并带 degraded。"""
    eng = AnswerEngine(SpyRetriever(), StreamLlm("", fail=True))
    kinds, _, final = collect(eng)
    assert kinds[-1] == "answer"
    assert final["status"] == "degraded" and final["needs_human"] is True
