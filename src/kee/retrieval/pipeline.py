"""混合检索流水线：Dense(BGE-M3) ‖ Sparse(服务端 BM25) → RRF → Rerank，含缓存与降级。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kee.config import get_settings
from kee.models.registry import ResidencyScheduler
from kee.retrieval.cache import SearchCache
from kee.retrieval.fusion import rrf_fuse
from kee.retrieval.milvus_store import Hit, KbStore
from kee.retrieval.rbac import AccessContext, build_filter_expr

MODES = ("dense", "sparse", "hybrid")


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

    def _sparse_hits(self, query: str, expr: str, top_k: int, t: dict):
        t0 = time.perf_counter()
        hits = self.store.search_sparse(query, limit=top_k, expr=expr)
        t["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return hits

    def retrieve(self, query: str, ctx: AccessContext, top_k: int | None = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.rerank_top_n
        expr = build_filter_expr(ctx)
        key = self.cache.make_key(query, self.mode, expr, top_k) if self.cache else None
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
        rankings: list[list[Hit]] = []
        if self.mode in ("dense", "hybrid"):
            try:
                d = self._dense_hits(query, expr, s.dense_top_k, timings, degraded)
                if d:
                    rankings.append(d)
            except Exception as e:  # noqa: BLE001 - 依赖故障必须显式降级而非抛错
                degraded.append(f"dense_error:{type(e).__name__}")
            if self.mode == "dense" and not rankings:
                degraded.append("dense_empty")
        if self.mode in ("sparse", "hybrid"):
            try:
                rankings.append(self._sparse_hits(query, expr, s.sparse_top_k, timings))
            except Exception as e:  # noqa: BLE001
                degraded.append(f"sparse_error:{type(e).__name__}")

        if not rankings:
            return RetrievalResult(
                query=query,
                mode=self.mode,
                chunks=[],
                timings_ms=timings,
                expr=expr,
                degraded=degraded + ["retrieval_total_failure"],
            )

        if len(rankings) == 1:
            fused = [(h, h.score) for h in rankings[0]]
        else:
            fused = rrf_fuse(rankings, k=s.rrf_k)
            timings["fuse_ms"] = 0.0

        candidates = [h for h, _ in fused[: self.rerank_candidates]]
        if self.use_rerank and candidates:
            t0 = time.perf_counter()
            try:
                with self.sched.use("rerank"):
                    scores = self.sched.reranker.score(
                        query, [c.fields.get("text", "") for c in candidates]
                    )
                ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
                chunks = []
                for h, sc in ranked[:top_k]:
                    c = _to_chunk(h)
                    c.score = round(float(sc), 6)
                    chunks.append(c)
            except Exception as e:  # noqa: BLE001
                degraded.append(f"rerank_error:{type(e).__name__}")
                chunks = [_to_chunk(h) for h in candidates[:top_k]]
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            chunks = [_to_chunk(h) for h in candidates[:top_k]]

        timings["retrieval_total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        if key and self.cache:
            self.cache.set(key, [c.as_dict() for c in chunks])
        return RetrievalResult(
            query=query,
            mode=self.mode,
            chunks=chunks,
            timings_ms=timings,
            expr=expr,
            degraded=degraded,
        )
