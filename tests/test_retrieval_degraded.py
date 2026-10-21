"""检索降级标记与量纲闸门。全是假 store / 假调度器，不加载模型、不占显存。

要钉的是 2026-10-13 复跑 verify_m3 抓到的那个形状（用户 2026-10-15 裁「乙为主、甲为辅」）：
同一份代码两次跑出 `confidence.retrieval` = 0.9937 与 **36.9408**。后者不是分变高，是换了量纲——
稠密腿交白卷时 hybrid 不报错也不留痕，单腿于是跳过 RRF 直接把 BM25 原始分往下传，
而 `min_retrieval_score=0.20` 与置信度成分都假定这个数在 0–1。

每条测试都同时断言「标记出现」与「标记不该出现时没出现」，因为这类闸门一旦天天响，
就没人再把它当回事了——和 `degraded=[]` 也亮 badge 那次同一类错。
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from kee.eval.probe_guard import score_scale_violation
from kee.retrieval.cache import SearchCache
from kee.retrieval.milvus_store import Hit
from kee.retrieval.pipeline import HybridRetriever
from kee.retrieval.rbac import AccessContext

CTX = AccessContext(role="knowledge_owner")


class FakeSched:
    """分时调度器的替身：只保证 `use()` 是可进入的上下文，模型本身不加载。"""

    def __init__(self, embed_ok: bool = True, rerank_ok: bool = True, rerank_score: float = 0.9937):
        self.embed_ok = embed_ok
        self.rerank_ok = rerank_ok
        self.rerank_score = rerank_score

    @contextmanager
    def use(self, _slot: str):
        yield

    @property
    def embedder(self):
        outer = self

        class _E:
            def embed_query(_self, _q: str):
                if not outer.embed_ok:
                    raise RuntimeError("CUDA out of memory")
                return [0.0] * 8

        return _E()

    @property
    def reranker(self):
        outer = self

        class _R:
            def score(_self, _q: str, passages: list[str]):
                if not outer.rerank_ok:
                    raise RuntimeError("CUDA out of memory")
                return [outer.rerank_score] * len(passages)

        return _R()


class FakeStore:
    def __init__(self, dense: list[Hit], sparse: list[Hit]):
        self._dense = dense
        self._sparse = sparse

    def search_dense(self, _vec, limit: int = 20, expr: str = "") -> list[Hit]:
        return self._dense[:limit]

    def search_sparse(self, _q, limit: int = 20, expr: str = "") -> list[Hit]:
        return self._sparse[:limit]


def sparse_hits(*scores: float) -> list[Hit]:
    return [Hit(chunk_id=f"c{i}", score=s, fields={"text": f"正文{i}"}) for i, s in enumerate(scores, start=1)]


def make(store: FakeStore, sched: FakeSched, **kw) -> HybridRetriever:
    return HybridRetriever(store, scheduler=sched, cache=None, rewrite_mode="off",
                           rerank_candidates=20, **kw)


def test_hybrid_dense_leg_empty_is_no_longer_silent():
    """甲：hybrid 下稠密腿交白卷必须留痕。以前只在 mode=="dense" 才补 dense_empty。"""
    rt = make(FakeStore([], sparse_hits(36.9408, 17.2)), FakeSched(), mode="hybrid")
    res = rt.retrieve("遥测上送覆盖哪些测点", CTX)
    assert "dense_empty" in res.degraded
    assert "hybrid_single_leg" in res.degraded


def test_dense_mode_empty_still_marks_dense_empty():
    """不回归：dense 模式交白卷原本就标，改条件后仍要标。"""
    rt = make(FakeStore([], []), FakeSched(), mode="dense")
    res = rt.retrieve("任意问题", CTX)
    assert "dense_empty" in res.degraded
    # 两条腿全空是「彻底没检索到」，另有自己的标记，不该被当成 hybrid 单腿
    assert "retrieval_total_failure" in res.degraded


def test_dense_exception_is_not_double_marked_as_empty():
    """稠密腿抛错时只报 dense_error，别再叠一个语义重复的 dense_empty。"""
    rt = make(FakeStore([], sparse_hits(12.5)), FakeSched(embed_ok=False), mode="hybrid")
    res = rt.retrieve("心跳超时阈值", CTX)
    assert any(d.startswith("dense_error") for d in res.degraded)
    assert "dense_empty" not in res.degraded


def test_legitimate_single_leg_modes_do_not_carry_hybrid_marker():
    """mode="sparse" 本来就是单腿（BM25 对照端点与消融格用它），不该报「hybrid 退化」。"""
    rt = make(FakeStore([], sparse_hits(30.0, 20.0)), FakeSched(rerank_ok=False), mode="sparse",
              use_rerank=False)
    res = rt.retrieve("点表", CTX)
    assert "hybrid_single_leg" not in res.degraded
    assert "score_scale_out_of_range" not in " ".join(res.degraded)


def test_rerank_failure_leaves_bm25_scale_and_gets_flagged():
    """乙的可见性：开了重排却没拿到重排分，原始稀疏分漏到下游时必须在 payload 里看得见。

    这一条就是 36.9408 那个读数：重排抛错 → 回落 `_to_chunk` → `chunks[].score` 是 BM25 分。
    """
    rt = make(FakeStore(sparse_hits(36.9408, 17.2), sparse_hits(36.9408, 17.2)),
              FakeSched(rerank_ok=False), mode="hybrid")
    res = rt.retrieve("遥测上送覆盖哪些测点", CTX)
    assert res.chunks and res.chunks[0].score > 1.0
    assert any(d.startswith("rerank_error") for d in res.degraded)
    assert any(d.startswith("score_scale_out_of_range") for d in res.degraded)


def test_rerank_healthy_run_stays_silent():
    """反面对照：正常 hybrid 两腿都有、重排给 0–1 分时，一个量纲标记都不该出现。"""
    rt = make(FakeStore(sparse_hits(0.91, 0.88), sparse_hits(36.9408, 17.2)),
              FakeSched(), mode="hybrid")
    res = rt.retrieve("遥测上送覆盖哪些测点", CTX)
    assert res.degraded == []
    assert res.top_score == pytest.approx(0.9937)


def test_score_scale_guard_rejects_out_of_range_reading():
    assert score_scale_violation([0.9937, 0.8712]) is None
    bad = score_scale_violation([0.9937, 36.9408])
    assert bad and "不是测量结果" in bad and "36.9408" in bad


def test_score_scale_guard_on_empty_scores_reports_nothing():
    """空集不判废也不判通过——它压根没读数。这里刻意断言 None 并写明理由，
    免得日后有人把「没拒绝」读成「量纲检查过了」。"""
    assert score_scale_violation([]) is None


class RecordingCache:
    """只记录键的缓存替身。`make_key` 直接委托生产实现，
    这样钉的是真的键构造，而不是替身自己定的约定。"""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def make_key(self, query: str, mode: str, expr: str, top_k: int) -> str:
        return SearchCache.make_key(query, mode, expr, top_k)

    def get(self, key: str, kind: str = "search"):
        self.keys.append(key)
        return None

    def set(self, key: str, value: object, kind: str = "search") -> None:
        return None


def _sparse_retriever(cache: RecordingCache) -> HybridRetriever:
    # sparse + 不重排：稠密腿不进门，测试只跑键构造这一件事
    return HybridRetriever(FakeStore([], sparse_hits(0.9, 0.8)), scheduler=FakeSched(), cache=cache,
                           rewrite_mode="off", mode="sparse", use_rerank=False)


def test_rbac_expression_reaches_the_cache_key_through_the_real_call_site():
    """权限表达式进缓存键是**越权问题**不是命中率问题：键里没有 expr，
    金融新人问过的结果就会被能源工程师直接命中——检索层的过滤被缓存旁路掉了。"""
    low = AccessContext(role="newcomer", industries=["金融"], max_security_level=1)
    high = AccessContext(role="delivery_engineer", industries=["能源"], max_security_level=2)
    cache = RecordingCache()
    rt = _sparse_retriever(cache)
    q = "清算链路 P1 多久响应"
    rt.retrieve(q, low, top_k=5)
    rt.retrieve(q, high, top_k=5)
    assert len(cache.keys) == 2 and cache.keys[0] != cache.keys[1]

    # 同身份同参数重复问 ⇒ 同一个键，否则缓存等于没开
    again = RecordingCache()
    rt2 = _sparse_retriever(again)
    rt2.retrieve(q, low, top_k=5)
    rt2.retrieve(q, low, top_k=5)
    assert again.keys[0] == again.keys[1]


def test_cache_key_components_are_all_live():
    """改写模式、mode、top_k 任一分量变了都得换键——`rewrite_mode` 是折在 mode 字段里的，
    所以这条同时钉住「换消融配置不会读到上一格的结果」。"""
    expr = 'security_level <= 2 and industry in ["金融"]'
    base = SearchCache.make_key("点表", "sparse+rw:off", expr, 5)
    assert base.startswith("kee:search:")  # clear_namespace 扫这个前缀，前缀改了清场就静默失效
    assert SearchCache.make_key("点表", "sparse+rw:off", expr, 5) == base
    assert SearchCache.make_key("点表", "sparse+rw:sparse", expr, 5) != base
    assert SearchCache.make_key("点表", "hybrid+rw:off", expr, 5) != base
    assert SearchCache.make_key("点表", "sparse+rw:off", expr, 10) != base
    assert SearchCache.make_key("点表", "sparse+rw:off", 'security_level <= 1', 5) != base
    assert SearchCache.make_key("点表 清单", "sparse+rw:off", expr, 5) != base


def test_cache_key_only_strips_the_query():
    """归一化只有 `strip()`：大小写与句中空白不同仍是两个键。这是**有意的弱归一化**
    （宁可少命中，也不要把两个语义可能不同的查询并成一个键），别读成"缓存做了查询规范化"。"""
    expr = 'security_level <= 1'
    assert SearchCache.make_key("RTO 要求", "sparse+rw:off", expr, 5) == \
        SearchCache.make_key("  RTO 要求  ", "sparse+rw:off", expr, 5)
    assert SearchCache.make_key("rto 要求", "sparse+rw:off", expr, 5) != \
        SearchCache.make_key("RTO 要求", "sparse+rw:off", expr, 5)
    assert SearchCache.make_key("RTO  要求", "sparse+rw:off", expr, 5) != \
        SearchCache.make_key("RTO 要求", "sparse+rw:off", expr, 5)
