"""混合检索流水线：Dense(BGE-M3) ‖ Sparse(服务端 BM25) → RRF → Rerank，含缓存与降级。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kee.config import get_settings
from kee.ingest.chunking import TABLE_ROW_MARK
from kee.models.registry import ResidencyScheduler
from kee.retrieval.cache import SearchCache
from kee.retrieval.fusion import rrf_fuse
from kee.retrieval.milvus_store import Hit, KbStore
from kee.retrieval.rbac import AccessContext, build_filter_expr

MODES = ("dense", "sparse", "hybrid")
REWRITE_MODES = ("off", "sparse", "both", "extra_leg")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    fields: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "text": self.text, "score": self.score, "fields": self.fields}


@dataclass
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]
    expr: str
    degraded: list[str] = field(default_factory=list)
    cache_hit: bool = False
    rewritten: str = ""
    rewrite_ms: float = 0.0
    rewrite_mode: str = "off"

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0


def _to_chunk(h: Hit) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=h.chunk_id, text=h.fields.get("text", ""), score=h.score, fields=h.fields
    )


class HybridRetriever:
    def __init__(
        self,
        store: KbStore,
        scheduler: ResidencyScheduler | None = None,
        cache: SearchCache | None = None,
        mode: str = "hybrid",
        use_rerank: bool = True,
        rerank_candidates: int = 20,
        dense_budget_ms: float = 2500.0,
        rewriter=None,
        rewrite_mode: str | None = None,
        rrf_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        row_rank_by_fusion: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}")
        s = get_settings()
        self.store = store
        self.sched = scheduler or ResidencyScheduler()
        self.cache = cache
        self.mode = mode
        self.use_rerank = use_rerank
        self.rerank_candidates = rerank_candidates
        self.dense_budget_ms = dense_budget_ms
        self.settings = s
        # 消融要逐格换参数，所以给成实例字段而不是改全局 settings：
        # 改 settings 会在下一格没显式赋值时把上一格的值带过去，网格数字就成了互相污染的产物
        self.rrf_k = s.rrf_k if rrf_k is None else rrf_k
        self.dense_top_k = s.dense_top_k if dense_top_k is None else dense_top_k
        self.sparse_top_k = s.sparse_top_k if sparse_top_k is None else sparse_top_k
        # True：表格行块守住融合名次（同表各行重排分只差 1e-3，用它排序等于用噪声）
        self.row_rank_by_fusion = row_rank_by_fusion
        self.rewrite_mode = rewrite_mode or s.rewrite_mode
        if self.rewrite_mode not in REWRITE_MODES:
            raise ValueError(f"rewrite_mode 必须是 {REWRITE_MODES}")
        if self.rewrite_mode == "off":
            self.rewriter = None
        else:
            if rewriter is None:
                from kee.retrieval.rewrite import QueryRewriter

                rewriter = QueryRewriter(cache=cache)
            self.rewriter = rewriter

    def _dense_hits(self, query: str, expr: str, top_k: int, t: dict, degraded: list[str]):
        t0 = time.perf_counter()
        with self.sched.use("embed"):
            model = self.sched.embedder
            vec = model.embed_query(query)
            t["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            t1 = time.perf_counter()
            hits = self.store.search_dense(vec, limit=top_k, expr=expr)
            t["dense_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        if t["embed_ms"] + t["dense_ms"] > self.dense_budget_ms:
            degraded.append(f"dense_timeout>{self.dense_budget_ms:.0f}ms")
            return []
        return hits

    def _sparse_hits(self, query: str, expr: str, top_k: int, t: dict, key: str = "sparse_ms"):
        t0 = time.perf_counter()
        hits = self.store.search_sparse(query, limit=top_k, expr=expr)
        t[key] = round((time.perf_counter() - t0) * 1000, 1)
        return hits

    def _rerank(self, query: str, candidates: list[Hit], top_k: int, degraded: list[str]) -> list[RetrievedChunk]:
        """重排**打分**照用交叉编码器，但表格行块的**名次**沿用融合结果。

        实测依据（M2 报告第七节 + 打分对照）：正确表行块的重排分本身很高（0.9976），
        问题不在「键值碎片被打分器歧视」，而在**同表各行之间的分差只有 1e-3 量级**
        （正确行 0.9976 vs 隔壁行 0.9971 vs 旁注 0.9897）——打分器在这一档已经饱和，
        用它的名次去区分同表两行等于用噪声排序。而 BM25/RRF 靠行内术语能把它们分开
        （关掉重排时正确行实测排第 1）。
        所以：分数全部保留（下游置信度与 0.20 检索闸门仍在同一量纲上），
        名次上让行块守住自己的融合位置，只重排正文槽位。
        """
        texts = [c.fields.get("text", "") for c in candidates]
        with self.sched.use("rerank"):
            scores = self.sched.reranker.score(query, texts)
        score_of = {id(candidates[i]): float(scores[i]) for i in range(len(candidates))}
        is_row = [TABLE_ROW_MARK in t for t in texts] if self.row_rank_by_fusion else [False] * len(candidates)
        if not any(is_row):
            order = [h for h, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])]
            marks = [False] * len(order)
        else:
            prose_ranked = [h for h, s in sorted(
                (zip([c for c, r in zip(candidates, is_row) if not r],
                     [s for s, r in zip(scores, is_row) if not r])), key=lambda x: -x[1])]
            prose_iter = iter(prose_ranked)
            order, marks = [], []
            for h, r in zip(candidates, is_row):
                if r:
                    order.append(h)
                    marks.append(True)
                else:
                    picked = next(prose_iter, None)
                    if picked is not None:
                        order.append(picked)
                        marks.append(False)
        out: list[RetrievedChunk] = []
        for h, row in zip(order, marks):
            c = _to_chunk(h)
            c.score = round(score_of[id(h)], 6)
            c.fields = {**c.fields, "ranked_by": "fusion" if row else "rerank"}
            out.append(c)
        return out[:top_k]

    def retrieve(
        self, query: str, ctx: AccessContext, top_k: int | None = None, already_rewritten: bool = False
    ) -> RetrievalResult:
        """`already_rewritten=True` 表示这条 query 已经过同一个改写器按多轮上文消解过，
        本次不要再自己改写一次——追问因此不会为同一件事付两趟 0.6s 的 LLM 调用。
        """
        s = self.settings
        top_k = top_k or s.rerank_top_n
        expr = build_filter_expr(ctx)
        # 缓存键带上改写模式：否则「off」跑出来的结果会被「sparse」直接命中，消融数字全废
        key = (
            self.cache.make_key(query, f"{self.mode}+rw:{self.rewrite_mode}", expr, top_k)
            if self.cache
            else None
        )
        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        if key and self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return RetrievalResult(
                    query=query,
                    mode=self.mode,
                    chunks=[RetrievedChunk(**c) for c in cached],
                    timings_ms={"cache_ms": 0.0},
                    expr=expr,
                    cache_hit=True,
                )

        degraded: list[str] = []
        rw = None
        if self.rewriter is not None and not already_rewritten:
            rw = self.rewriter.rewrite(query)
            degraded += rw.degraded
            timings["rewrite_ms"] = rw.ms
        # 重排一律用**原查询**打分：改写是为召回服务的，判「这段是否回答了用户问的话」
        # 不能拿改写后的字面去判，否则改写得越激进、重排越容易被带偏
        q_dense = rw.effective if rw and self.rewrite_mode == "both" else query
        q_sparse = rw.effective if rw and self.rewrite_mode in ("sparse", "both") else query
        rankings: list[list[Hit]] = []
        if self.mode in ("dense", "hybrid"):
            try:
                d = self._dense_hits(q_dense, expr, self.dense_top_k, timings, degraded)
                if d:
                    rankings.append(d)
                elif not any(x.startswith("dense_timeout") for x in degraded):
                    # 此前这个标记只在 mode=="dense" 时补，于是 hybrid 下稠密腿交白卷既不报错也不留痕，
                    # 整条链路静默退化成 BM25 单腿（实测同一份代码两次跑出 0.9937 与 36.9408）。
                    # 超时那条腿自己会标 dense_timeout，不重复追加。
                    degraded.append("dense_empty")
            except Exception as e:  # noqa: BLE001 - 依赖故障必须显式降级而非抛错
                degraded.append(f"dense_error:{type(e).__name__}")
        if self.mode in ("sparse", "hybrid"):
            try:
                rankings.append(self._sparse_hits(q_sparse, expr, self.sparse_top_k, timings))
            except Exception as e:  # noqa: BLE001
                degraded.append(f"sparse_error:{type(e).__name__}")
        if (
            rw is not None
            and rw.changed
            and self.rewrite_mode == "extra_leg"
            and self.mode in ("sparse", "hybrid")
        ):
            # 原查询与改写查询各算一路：召回覆盖最大，代价是多一次 BM25 与更宽的候选
            try:
                rankings.append(self._sparse_hits(rw.effective, expr, self.sparse_top_k, timings, key="sparse_rw_ms"))
            except Exception as e:  # noqa: BLE001
                degraded.append(f"sparse_rw_error:{type(e).__name__}")

        if not rankings:
            return RetrievalResult(
                query=query,
                mode=self.mode,
                chunks=[],
                timings_ms=timings,
                expr=expr,
                degraded=degraded + ["retrieval_total_failure"],
                rewritten=rw.effective if rw else "",
                rewrite_ms=rw.ms if rw else 0.0,
                rewrite_mode=self.rewrite_mode,
            )

        if len(rankings) == 1:
            fused = [(h, h.score) for h in rankings[0]]
            if self.mode == "hybrid":
                # 只有 hybrid 才标：mode="sparse"/"dense" 本来就是单腿，标它是噪声。
                # 单腿时 RRF 没有发生，`h.score` 是该腿的原始量纲——BM25 是几十，不是 0–1。
                degraded.append("hybrid_single_leg")
        else:
            fused = rrf_fuse(rankings, k=self.rrf_k)
            timings["fuse_ms"] = 0.0

        candidates = [h for h, _ in fused[: self.rerank_candidates]]
        if self.use_rerank and candidates:
            t0 = time.perf_counter()
            try:
                chunks = self._rerank(query, candidates, top_k, degraded)
            except Exception as e:  # noqa: BLE001
                degraded.append(f"rerank_error:{type(e).__name__}")
                chunks = [_to_chunk(h) for h in candidates[:top_k]]
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            chunks = [_to_chunk(h) for h in candidates[:top_k]]

        timings["retrieval_total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        if self.use_rerank and chunks and max(c.score for c in chunks) > 1.0:
            # 交叉编码器的分恒在 (0,1)（`reranker.score` 走 sigmoid）。出现 >1 就说明这个分
            # 不是重排给的，而是某条腿的原始量纲漏到了下游——下游的 `min_retrieval_score=0.20`
            # 与置信度成分都假定 retrieval ∈ [0,1]，在这个尺度上「过闸」等于没比。
            # 只标记、不动阈值：动阈值就是动证据链。
            degraded.append(f"score_scale_out_of_range:{round(max(c.score for c in chunks), 4)}")
        if key and self.cache:
            self.cache.set(key, [c.as_dict() for c in chunks])
        return RetrievalResult(
            query=query,
            mode=self.mode,
            chunks=chunks,
            timings_ms=timings,
            expr=expr,
            degraded=degraded,
            rewritten=rw.effective if rw else "",
            rewrite_ms=rw.ms if rw else 0.0,
            rewrite_mode=self.rewrite_mode,
        )
