"""RRF（Reciprocal Rank Fusion）融合。"""

from __future__ import annotations

from kee.retrieval.milvus_store import Hit


def rrf_fuse(rankings: list[list[Hit]], k: int = 60) -> list[tuple[Hit, float]]:
    """多路排序 → RRF 分数。同一 chunk 跨路去重（保留首次出现的 Hit 元数据）。"""
    scores: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            keep.setdefault(hit.chunk_id, hit)
    return sorted(((keep[cid], s) for cid, s in scores.items()), key=lambda x: -x[1])
