"""向量物料仓：把嵌入结果落成本地 .npy + ids.json 的派生物。

为什么不让 PostgreSQL 存向量：向量索引的权威归属是 Milvus，PostgreSQL 只当元数据/版本/审计
唯一权威源。Outbox 补偿与每日对账要能在「无 GPU」的情况下重建索引，所以把嵌入产物留在本地
物料仓（派生、可删可重建），既不让权威源膨胀，也让一致性验证不依赖显存。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kee.config import get_settings


class VectorStore:
    def __init__(self, root: Path | None = None) -> None:
        base = root or (get_settings().resolved_artifact_dir() / "vectors")
        self.root = Path(base)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, version_id: str) -> tuple[Path, Path]:
        safe = version_id.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.npy", self.root / f"{safe}.ids.json"

    def save(self, version_id: str, chunk_ids: list[str], vectors: np.ndarray) -> dict[str, int]:
        npy, ids = self._paths(version_id)
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError(f"chunk_ids({len(chunk_ids)}) 与向量行数({vectors.shape[0]}) 不一致")
        np.save(npy, vectors.astype("float32"))
        ids.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
        return {"rows": int(vectors.shape[0]), "dim": int(vectors.shape[1])}

    def load(self, version_id: str) -> tuple[list[str], np.ndarray]:
        npy, ids = self._paths(version_id)
        if not npy.exists() or not ids.exists():
            raise FileNotFoundError(f"向量物料缺失：{version_id}")
        chunk_ids = json.loads(ids.read_text(encoding="utf-8"))
        return chunk_ids, np.load(npy)

    def vector_of(self, version_id: str) -> dict[str, list[float]]:
        chunk_ids, arr = self.load(version_id)
        return {cid: arr[i].tolist() for i, cid in enumerate(chunk_ids)}

    def drop(self, version_id: str) -> None:
        npy, ids = self._paths(version_id)
        npy.unlink(missing_ok=True)
        ids.unlink(missing_ok=True)
