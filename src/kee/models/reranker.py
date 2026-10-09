"""BGE-reranker-v2-m3 交叉编码器重排：本地权重直读，显式加载/释放。"""

from __future__ import annotations

import time

from kee.config import get_settings

# bge-reranker-v2-m3 官方推荐 query 侧前缀模板
PROMPT_TEMPLATE = "A: {query}?\nB: {passage}"


class BgeReranker:
    def __init__(
        self,
        path: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        max_length: int | None = None,
    ) -> None:
        s = get_settings()
        self.path = path or s.bge_reranker_path
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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.path)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.path, dtype=getattr(torch, self.dtype)
        )
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

    def score(self, query: str, passages: list[str], batch_size: int = 8) -> list[float]:
        """返回 sigmoid(logit) 作为相关性概率，顺序与 passages 一致。"""
        import torch

        if self._model is None:
            raise RuntimeError("reranker 未加载：先调用 load()")
        if not passages:
            return []
        pairs = [(PROMPT_TEMPLATE.format(query=query, passage=p), p) for p in passages]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                chunk = pairs[i : i + batch_size]
                enc = self._tokenizer(
                    [q for q, _ in chunk],
                    [p for _, p in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits[:, 0]
                scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        return scores
