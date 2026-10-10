"""M3-05 多轮指代消解：图上的 rewrite 节点只在有上文时介入，且消解结果必须真的进检索。

要钉住的是四件容易悄悄退化掉的事：
  1. 补全出来的查询**进了检索**（而不是只记在响应里好看）——否则多轮等于没做。
  2. 补全出来的查询**进了生成提示词**——「那个月结的呢」原样喂给模型，模型无从定指。
  3. 对外的 `question` 仍是用户原话——审计与用户看到的必须是同一句话，补全只是内部手段。
  4. 无上文时一个改写调用都不发——单轮延迟不能被多轮能力拖累。
"""

from __future__ import annotations

import hashlib
import json

from kee.qa.answer import AnswerEngine
from kee.retrieval.pipeline import RetrievedChunk, RetrievalResult
from kee.retrieval.rbac import AccessContext
from kee.retrieval.rewrite import QueryRewriter, RewriteResult


class SpyRetriever:
    """记录图实际拿来检索的查询，以及检索层自己有没有又改写一次。"""

    def __init__(self, chunks=None):
        self.chunks = chunks or [chunk("c1")]
        self.rewriter = None
        self.calls: list[tuple[str, bool]] = []

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        self.calls.append((query, already_rewritten))
        return RetrievalResult(
            query=query,
            mode="hybrid",
            chunks=self.chunks,
            timings_ms={"retrieval_total_ms": 10.0},
            expr="security_level <= 3",
        )


class SpyRewriter:
    def __init__(self, expanded: str, degraded=None):
        self._expanded = expanded
        self.degraded = degraded or []
        self.contexts: list[str | None] = []

    def rewrite(self, query, context=None):
        self.contexts.append(context)
        changed = self._expanded != query
        return RewriteResult(
            original=query,
            expanded=self._expanded,
            changed=changed,
            triggered=bool(context),
            ms=42.0,
            degraded=list(self.degraded),
        )


class FakeLlm:
    def __init__(self, payload=None):
        self.payload = payload or {
            "conclusion": "月结在 T+3 完成。",
            "key_points": [{"point": "月结 T+3", "evidence_ids": ["E1"]}],
            "citations": [{"evidence_id": "E1"}],
            "no_evidence": False,
            "self_assessed_confidence": 0.9,
        }
        self.last_prompt = ""

    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        from kee.llm.ollama import LlmResult

        self.last_prompt = messages[-1]["content"]
        return LlmResult(
            ok=True,
            text=json.dumps(self.payload, ensure_ascii=False),
            parsed=None,
            json_valid=True,
            metrics={"total_ms": 500.0},
        )


def chunk(cid, score=0.9, level=2):
    return RetrievedChunk(
        chunk_id=cid,
        text=f"证据正文 {cid}",
        score=score,
        fields={
            "chunk_id": cid,
            "doc_id": "fin-doc",
            "project": "PRJ-FIN-2024-017",
            "section_no": "2.1",
            "page": 5,
            "security_level": level,
            "parent_id": "fin-doc#p",
        },
    )


def ctx():
    return AccessContext(role="knowledge_owner")


def history(*pairs):
    return [{"turn_no": i + 1, "question": q, "conclusion": a} for i, (q, a) in enumerate(pairs)]


def build(expanded=None, degraded=None):
    r = SpyRetriever()
    llm = FakeLlm()
    eng = AnswerEngine(r, llm)
    if expanded is not None:
        r.rewriter = SpyRewriter(expanded, degraded=degraded)
    return eng, r, llm


HIST = history(("灾备方案的 RTO 是多少", "灾备切换 RTO < 15 分钟。"))


def test_resolved_query_is_the_one_actually_retrieved():
    eng, r, _ = build(expanded="灾备方案的月结完成时限是多少")
    res = eng.answer("那个月结的呢", ctx(), history=HIST)
    assert r.calls == [("灾备方案的月结完成时限是多少", True)]
    assert res.retrieval["query_used"] == "灾备方案的月结完成时限是多少"


def test_resolved_query_is_what_the_model_is_asked():
    eng, _, llm = build(expanded="灾备方案的月结完成时限是多少")
    eng.answer("那个月结的呢", ctx(), history=HIST)
    assert "灾备方案的月结完成时限是多少" in llm.last_prompt
    assert "那个月结的呢" not in llm.last_prompt


def test_outward_question_stays_the_users_own_words():
    eng, _, _ = build(expanded="灾备方案的月结完成时限是多少")
    res = eng.answer("那个月结的呢", ctx(), history=HIST)
    assert res.question == "那个月结的呢"


def test_context_carries_previous_turns_into_the_prompt():
    eng, r, _ = build(expanded="灾备方案的月结完成时限是多少")
    eng.answer("那个月结的呢", ctx(), history=HIST)
    context = r.rewriter.contexts[0]
    assert "灾备方案的 RTO 是多少" in context and "RTO < 15 分钟" in context


def test_retriever_does_not_rewrite_again_after_the_graph_did():
    """already_rewritten=True 是防双趟 LLM 的那道闸：追问不能为同一件事付两次钱。"""
    eng, r, _ = build(expanded="灾备方案的月结完成时限是多少")
    eng.answer("那个月结的呢", ctx(), history=HIST)
    assert r.calls[0][1] is True


def test_no_history_means_zero_rewrite_calls_and_single_turn_behaviour():
    eng, r, llm = build(expanded="绝不该被用到")
    res = eng.answer("灾备方案的 RTO 是多少", ctx())
    assert r.rewriter.contexts == []
    assert r.calls == [("灾备方案的 RTO 是多少", False)]
    assert "灾备方案的 RTO 是多少" in llm.last_prompt
    assert res.retrieval["query_used"] == "灾备方案的 RTO 是多少"


def test_rewriter_without_history_is_skipped_even_when_present():
    """图上的判据是「有没有上文」，不是「有没有改写器」——单轮不该为指代消解付费。"""
    eng, r, _ = build(expanded="X")
    eng.answer("灾备方案的 RTO 是多少", ctx(), history=[])
    assert r.calls == [("灾备方案的 RTO 是多少", False)]


def test_rejected_rewrite_falls_back_to_the_literal_question():
    """改写被拒（changed=False）时检索必须照用原话，而不是把半截补全塞进去。"""
    eng, r, _ = build(expanded="那个月结的呢", degraded=["rewrite_rejected:引入了原查询没有的数字"])
    res = eng.answer("那个月结的呢", ctx(), history=HIST)
    assert r.calls == [("那个月结的呢", False)]
    assert res.retrieval["query_used"] == "那个月结的呢"
    assert any("rewrite_rejected" in n for n in res.notes)


def test_over_budget_note_is_carried_into_the_answer():
    eng, _, _ = build(expanded="灾备方案的月结完成时限是多少", degraded=["rewrite_over_budget(900ms>800ms)"])
    res = eng.answer("那个月结的呢", ctx(), history=HIST)
    assert any("rewrite_over_budget" in n for n in res.notes)


def test_sensitive_question_with_history_never_reaches_the_rewriter():
    """敏感闸在图首，带历史也一样：被拦的问题不该再花改写、检索、生成任何一次调用。"""
    eng, r, _ = build(expanded="合同金额的付款周期")
    res = eng.answer("合同金额是多少", ctx(), history=HIST)
    assert res.status == "escalate"
    assert r.calls == []
    assert r.rewriter.contexts == []


# ---- 改写缓存的键必须把上文算进去 ----
# 同一句「那个月结的呢」在不同上文下应当消解成不同查询。键里不带上文，
# 第二条追问就会命中第一条的缓存结果，拿到一个**属于别的会话的查询**——
# 这比不缓存更糟，因为它看起来是通的。


class MemoryCache:
    """按 SearchCache.make_key 的同样算法实现，这样测的才是真的键构造。"""

    def __init__(self):
        self.store: dict[str, dict] = {}

    def make_key(self, query, mode, expr, top_k):
        payload = json.dumps([query.strip(), mode, expr, top_k], ensure_ascii=False)
        return "kee:test:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


class ScriptedLlm:
    def __init__(self, by_marker):
        self.by_marker = by_marker
        self.n = 0

    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        from kee.llm.ollama import LlmResult

        self.n += 1
        content = messages[-1]["content"]
        expanded = next((v for k, v in self.by_marker.items() if k in content), None)
        return LlmResult(
            ok=True,
            text=json.dumps({"expanded": expanded}, ensure_ascii=False),
            parsed={"expanded": expanded},
            json_valid=True,
            metrics={"total_ms": 10.0},
        )


def test_different_contexts_do_not_share_a_rewrite_entry():
    cache = MemoryCache()
    llm = ScriptedLlm({"RTO 是多少": "灾备方案 RTO 月结时限", "巡检频次": "巡检频次的月结时限"})
    rw = QueryRewriter(client=llm, cache=cache, always=True)

    a = rw.rewrite("那个月结的呢", context="上一轮问：灾备方案的 RTO 是多少")
    b = rw.rewrite("那个月结的呢", context="上一轮问：变电站的巡检频次")
    assert (a.changed, b.changed) == (True, True)
    assert a.expanded != b.expanded, "两条不同上文的追问拿到了同一个改写——缓存串了"
    assert llm.n == 2


def test_same_context_is_served_from_cache_without_a_second_call():
    cache = MemoryCache()
    llm = ScriptedLlm({"RTO 是多少": "灾备方案 RTO 月结时限"})
    rw = QueryRewriter(client=llm, cache=cache, always=True)
    context = "上一轮问：灾备方案的 RTO 是多少"
    first = rw.rewrite("那个月结的呢", context=context)
    second = rw.rewrite("那个月结的呢", context=context)
    assert llm.n == 1, "同一(查询, 上文)对第二次仍然打了模型，缓存没生效"
    assert second.from_cache and second.expanded == first.expanded


def test_context_free_rewrite_and_context_rewrite_are_separate_entries():
    """无上下文的单轮改写与带上下文的追问必须各占一个键，否则单轮结果会喂给追问。"""
    cache = MemoryCache()
    llm = ScriptedLlm({"RTO 是多少": "灾备方案 RTO 月结时限", "那个月结的呢": "单轮扩展结果"})
    rw = QueryRewriter(client=llm, cache=cache, always=True)
    with_ctx = rw.rewrite("那个月结的呢", context="上一轮问：灾备方案的 RTO 是多少")
    bare = rw.rewrite("那个月结的呢")
    assert with_ctx.expanded == "灾备方案 RTO 月结时限"
    assert bare.expanded == "单轮扩展结果"
    assert bare.from_cache is False


def test_two_contexts_of_equal_length_and_equal_tail_still_get_own_entries():
    """专门钉「指纹式缓存键」的那次收紧。

    键若按「上文长度 + 末 60 字」算，下面这两条上文**长度相同、尾部也完全相同**（只差首字），
    于是第二条会命中第一条的缓存，拿到一个属于别的上文的查询——表面通、实质串。
    这条用例对旧的指纹写法会失败，对整段上文进键的写法必须通过。
    """
    cache = MemoryCache()
    filler = "共用尾部内容" * 12  # 72 字，保证两条上文的末 60 字完全一致
    c1 = "甲项目问句：" + filler
    c2 = "乙项目问句：" + filler
    assert len(c1) == len(c2) and c1[-60:] == c2[-60:] and c1 != c2

    llm = ScriptedLlm({"甲项目": "甲项目的月结时限", "乙项目": "乙项目的月结时限"})
    rw = QueryRewriter(client=llm, cache=cache, always=True)
    a = rw.rewrite("那个月结的呢", context=c1)
    b = rw.rewrite("那个月结的呢", context=c2)
    assert a.expanded == "甲项目的月结时限"
    assert b.expanded == "乙项目的月结时限", f"上文串了：拿到 {b.expanded}"
    assert llm.n == 2, "两条不同上文只发了一次模型调用，说明键撞在一起"
