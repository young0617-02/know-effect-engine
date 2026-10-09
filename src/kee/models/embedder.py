"""BGE-M3 稠密嵌入：本地权重直读，CLS pooling + L2 归一化，支持显式加载/释放。"""

from __future__ import annotations

import time

import numpy as np

from kee.config import get_settings


class BgeM3Embedder:
    """dense 向量维度 1024；查询侧不加指令前缀（BGE-M3 官方口径无需 instruction）。"""

    dim = 1024

    def __init__(
        self,
        path: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        max_length: int | None = None,
    ) -> None:
        s = get_settings()
        self.path = path or s.bge_m3_path
        self.device = device or s.device
        self.dtype = dtype or s.model_dtype
        self.max_length = max_length or s.embed_max_length
        self._tokenizer = None
        self._model = None
        self.load_sec: float = 0.0

    @property
    def resident(self) -> bool:
        return self._model is not None

    def load(self) -> float:
        if self._model is not None:
            return 0.0
        import torch
        from transformers import AutoModel, AutoTokenizer

        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.path)
        model = AutoModel.from_pretrained(self.path, dtype=getattr(torch, self.dtype))
        self._model = model.to(self.device)
        self._model.eval()
        self.load_sec = round(time.perf_counter() - t0, 3)
        return self.load_sec

    def unload(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("BGE-M3 未加载：先调用 load()")
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self._model(**enc)
                cls = out.last_hidden_state[:, 0]  # 1_Pooling: pooling_mode_cls_token=true
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                vecs.append(cls.float().cpu().numpy())
        return np.concatenate(vecs, axis=0)

    def embed_passages(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts, batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], 1)[0].tolist()
