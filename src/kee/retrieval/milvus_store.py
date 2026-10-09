"""Milvus v3.0.0 混合检索存储：服务端 BM25（可切分析器）+ BGE-M3 稠密向量。

分析器实测结论（2026-10-09，Milvus v3.0.0 + pymilvus 3.0.2）：
  - analyzer_params={"type": "jieba"}      → 服务端拒绝：unknown build-in analyzer type: jieba
  - analyzer_params={"tokenizer": "jieba"} → 接受，中文分词生效
  - analyzer_params={"type": "chinese"}    → 接受
  - 默认（仅 enable_analyzer，不给 params）  → 接受，等价 standard，中文查询几乎无命中
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from kee.config import get_settings

DENSE_DIM = 1024

# 分析器别名 → analyzer_params（None 表示只开 enable_analyzer 不给参数）
ANALYZERS: dict[str, dict | None] = {
    "jieba": {"tokenizer": "jieba"},
    "chinese": {"type": "chinese"},
    "standard": {"type": "standard"},
    "default_none": None,
}

VARIANT_FIELDS = [
    "chunk_id",
    "doc_id",
    "version_id",
    "parent_id",
    "industry",
    "customer",
    "project",
    "doc_type",
    "source_level",
    "security_level",
    "section_no",
    "page",
    "order_no",
    "text",
]


@dataclass
class Hit:
    chunk_id: str
    score: float
    fields: dict[str, Any] = field(default_factory=dict)


class KbStore:
    """一个实例绑定一个 collection（即一种分析器变体），便于消融对照。"""

    def __init__(self, collection: str, analyzer: str = "jieba", uri: str | None = None) -> None:
        if analyzer not in ANALYZERS:
            raise ValueError(f"analyzer 必须是 {list(ANALYZERS)} 之一")
        self.collection = collection
        self.analyzer = analyzer
        self._client = MilvusClient(uri=uri or get_settings().milvus_uri, timeout=60)

    # ---- DDL ----
    def exists(self) -> bool:
        return self._client.has_collection(self.collection)

    def drop(self) -> None:
        if self.exists():
            self._client.drop_collection(self.collection)

    def ensure(self) -> None:
        if self.exists():
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=160)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("version_id", DataType.VARCHAR, max_length=128)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=160)
        schema.add_field("industry", DataType.VARCHAR, max_length=64)
        schema.add_field("customer", DataType.VARCHAR, max_length=128)
        schema.add_field("project", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=64)
        schema.add_field("source_level", DataType.VARCHAR, max_length=16)
        schema.add_field("security_level", DataType.INT8)
        schema.add_field("section_no", DataType.VARCHAR, max_length=32)
        schema.add_field("page", DataType.INT32)
        schema.add_field("order_no", DataType.INT32)

        text_kwargs: dict[str, Any] = {"max_length": 8192, "enable_analyzer": True}
        params = ANALYZERS[self.analyzer]
        if params is not None:
            text_kwargs["analyzer_params"] = params
        schema.add_field("text", DataType.VARCHAR, **text_kwargs)

        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )

        ip = self._client.prepare_index_params()
        # 语料规模小，dense 用 FLAT 精确检索，避免 ANN 近似损失污染消融结论
        ip.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
        ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self._client.create_collection(collection_name=self.collection, schema=schema, index_params=ip)

    def describe_analyzer(self) -> Any:
        for f in self._client.describe_collection(self.collection).get("fields", []):
            if f.get("name") == "text":
                return f.get("params", {})
        return None

    # ---- DML ----
    def insert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        res = self._client.insert(collection_name=self.collection, data=rows)
        return int(res.get("insert_count", len(rows)))

    def flush(self) -> None:
        self._client.flush(collection_name=self.collection)

    def count(self) -> int:
        agg = self._client.query(
            collection_name=self.collection, filter="", output_fields=["count(*)"]
        )
        return int(agg[0]["count(*)"]) if agg else 0

    def delete_version(self, version_id: str) -> None:
        self._client.delete(collection_name=self.collection, filter=f'version_id == "{version_id}"')
        self.flush()

    # ---- 检索 ----
    @staticmethod
    def _to_hits(res: list[list[dict]]) -> list[Hit]:
        hits: list[Hit] = []
        for h in res[0]:
            ent = h.get("entity", {}) or {}
            cid = ent.get("chunk_id", h.get("chunk_id"))
            fields = {k: ent[k] for k in VARIANT_FIELDS if k in ent}
            hits.append(Hit(chunk_id=cid, score=float(h["distance"]), fields=fields))
        return hits

    def search_dense(
        self, vector: list[float], limit: int = 20, expr: str | None = None
    ) -> list[Hit]:
        res = self._client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="dense",
            limit=limit,
            filter=expr or "",
            output_fields=VARIANT_FIELDS,
        )
        return self._to_hits(res)

    def search_sparse(
        self, query: str, limit: int = 20, expr: str | None = None
    ) -> list[Hit]:
        res = self._client.search(
            collection_name=self.collection,
            data=[query],
            anns_field="sparse",
            limit=limit,
            filter=expr or "",
            output_fields=VARIANT_FIELDS,
        )
        return self._to_hits(res)

    def tokenize_probe(self, terms: list[str], limit: int = 3) -> dict[str, list[str]]:
        """用单个 term 探服务端切分行为：命中的 chunk_id 列表（②的分词生效证据）。"""
        out: dict[str, list[str]] = {}
        for t in terms:
            hits = self.search_sparse(t, limit=limit)
            out[t] = [h.chunk_id for h in hits]
        return out

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        ids = ", ".join('"' + c.replace('"', "") + '"' for c in chunk_ids)
        res = self._client.delete(collection_name=self.collection, filter=f"chunk_id in [{ids}]")
        self.flush()
        return len(chunk_ids)

    def all_chunk_ids(self, version_id: str | None = None, page_limit: int = 16384) -> list[str]:
        expr = f'version_id == "{version_id}"' if version_id else 'chunk_id != ""'
        rows = self._client.query(
            collection_name=self.collection, filter=expr, output_fields=["chunk_id"], limit=page_limit
        )
        return [r["chunk_id"] for r in rows]

    def close(self) -> None:
        self._client.close()
