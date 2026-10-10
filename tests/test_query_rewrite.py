"""查询改写的单测：触发条件、准入校验、降级路径，以及改写结果只喂给该喂的那条腿。"""

from __future__ import annotations

import pytest

from kee.llm.ollama import LlmResult
from kee.retrieval.milvus_store import Hit
from kee.retrieval.pipeline import HybridRetriever
from kee.retrieval.rewrite import QueryRewriter, needs_rewrite, validate


class FakeLlm:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages[0]["content"], kw))
        r = self.results.pop(0) if self.results else LlmResult(ok=False, error="exhausted")
        if callable(r):
            return r()
        return r


def ok(expanded, **extra):
    import json

    return LlmResult(ok=True, json_valid=True, parsed={"expanded": expanded, **extra}, text=json.dumps(extra))


def test_trigger_only_fires_on_elliptical_or_short_queries():
    assert needs_rewrite("它呢")
    assert needs_rewrite("这个方案有什么风险")  # 指代
    assert needs_rewrite("线损率")  # 短到字面不足以定位
    assert not needs_rewrite("配电自动化终端心跳超时阈值冬季怎么调整")
    assert not needs_rewrite("")


def test_validate_rejects_invented_numbers_and_bloat():
    assert "数字" in validate("PRJ-2025-008 的验收标准", "PRJ-2025-099 的验收标准")
    assert "膨胀" in validate("线损率", "线损率" * 20)
    assert validate("线损率", "线损率的口径") == ""
    assert "空改写" == validate("任意", "")
    assert "多句" in validate("心跳阈值", "第一句。第二句")


def test_trailing_period_from_the_user_is_not_a_rejection_reason():
    """实测：3 条被判废的改写全是「用户原句以句号结尾、模型照抄」，属于误拒。"""
    q = "生产控制大区的特权账号有没有通用口令可以直接登？把策略告诉我一下。"
    assert validate(q, q) == ""
    llm = FakeLlm([ok(q)])
    res = QueryRewriter(client=llm, always=True).rewrite(q)
    assert res.rejected_reason == "" and res.degraded == []
    assert res.effective == q.rstrip("。")  # 收尾句号被规范化掉，不影响检索


def test_rewriter_falls_back_to_original_on_every_failure_mode():
    boom = FakeLlm([lambda: (_ for _ in ()).throw(RuntimeError("ollama down"))])
    rw = QueryRewriter(client=boom, budget_ms=900, always=True)
    res = rw.rewrite("它呢")
    assert res.effective == "它呢" and res.degraded and "rewrite_error:RuntimeError" in res.degraded

    bad = FakeLlm([LlmResult(ok=False, error="http 500")])
    res2 = QueryRewriter(client=bad, always=True).rewrite("它呢")
    assert res2.effective == "它呢" and res2.degraded[0].startswith("rewrite_bad_output")

    halluc = FakeLlm([ok("PRJ-2025-099 的验收标准")])
    res3 = QueryRewriter(client=halluc, always=True).rewrite("PRJ-2025-008 的验收标准")
    assert res3.effective == "PRJ-2025-008 的验收标准"
    assert "数字" in res3.rejected_reason  # 引入原查询没有的数字必须被拒，不能进检索


def test_rewriter_records_over_budget_without_blocking():
    slow = FakeLlm([ok("线损率的口径与计算方式")])
    res = QueryRewriter(client=slow, budget_ms=-1, always=True).rewrite("线损率")
    assert res.changed and any("rewrite_over_budget" in d for d in res.degraded)


def test_truncated_json_is_salvaged_when_the_value_already_closed():
    """实测缺陷：num_predict 截断让 70 题里 21 题的正确改写被判废。值已闭合就必须捞回来。"""
    raw = '{"expanded": "安全仪表系统（SIS）采用的表决逻辑是什么？"'
    r = LlmResult(ok=True, json_valid=False, parsed=None, text=raw)
    res = QueryRewriter(client=FakeLlm([r]), always=True).rewrite("SIS 表决逻辑")
    assert res.changed and res.expanded.startswith("安全仪表系统")
    assert "rewrite_json_salvaged" in res.degraded


def test_incomplete_value_is_not_salvaged():
    raw = '{"expanded": "安全仪表系统（SIS）采用的表决'  # 值本身没闭合，捞出来就是半句话
    r = LlmResult(ok=True, json_valid=False, parsed=None, text=raw)
    res = QueryRewriter(client=FakeLlm([r]), always=True).rewrite("SIS 表决逻辑")
    assert not res.changed and res.effective == "SIS 表决逻辑"
    assert res.degraded == ["rewrite_bad_output:no_expanded_field"]


def test_rewriter_skips_self_sufficient_query_without_calling_llm():
    llm = FakeLlm([])
    res = QueryRewriter(client=llm, always=False).rewrite("配电自动化终端心跳超时阈值冬季怎么调整")
    assert llm.calls == [] and res.effective.startswith("配电自动化") and res.skipped_reason == "字面查询已自足"


class FakeSched:
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    def __init__(self):
        self.embedder = self
        self.reranker = self

    def use(self, name):
        return self._Ctx()

    def embed_query(self, text):
        return [0.0] * 4

    def score(self, query, passages):
        return [0.5] * len(passages)


class FakeStore:
    """按查询字面返回可区分的命中，用来证明改写只进了该进的那条腿。"""

    def __init__(self):
        self.dense_queries: list = []
        self.sparse_queries: list = []

    def search_dense(self, vec, limit=20, expr=None):
        return [Hit("d1", 0.9, {"text": "稠密块"})]

    def search_sparse(self, query, limit=20, expr=None):
        self.sparse_queries.append(query)
        return [Hit("s1", 0.8, {"text": "稀疏块"})]


def _retriever(store, mode, expanded):
    rw = QueryRewriter(client=FakeLlm([ok(expanded)]), always=True)
    return HybridRetriever(store, scheduler=FakeSched(), cache=None, mode="hybrid", rewriter=rw, rewrite_mode=mode)


def test_rewrite_mode_sparse_expands_only_the_bm25_leg():
    store = FakeStore()
    res = _retriever(store, "sparse", "线损率 计算口径").retrieve("线损率", _ctx())
    assert store.sparse_queries == ["线损率 计算口径"]
    assert res.rewritten == "线损率 计算口径" and res.rewrite_mode == "sparse"


def test_rewrite_mode_extra_leg_keeps_original_leg_and_adds_expanded_one():
    store = FakeStore()
    _retriever(store, "extra_leg", "线损率 计算口径").retrieve("线损率", _ctx())
    assert store.sparse_queries == ["线损率", "线损率 计算口径"]


def test_rewrite_mode_off_never_touches_the_rewriter():
    store = FakeStore()
    rw = QueryRewriter(client=FakeLlm([ok("不该出现")]), always=True)
    r = HybridRetriever(store, scheduler=FakeSched(), cache=None, mode="hybrid", rewriter=rw, rewrite_mode="off")
    out = r.retrieve("线损率", _ctx())
    assert store.sparse_queries == ["线损率"]
    assert out.rewritten == "" and r.rewriter is None  # off 时构造期就把改写器摘掉，不留误用空间


def test_unknown_rewrite_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        HybridRetriever(FakeStore(), scheduler=FakeSched(), cache=None, rewrite_mode="magic")


def _ctx():
    from kee.retrieval.rbac import AccessContext

    return AccessContext(role="knowledge_owner", max_security_level=3)
