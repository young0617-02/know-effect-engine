"""嵌入/重排与本地生成模型的分时驻留调度器（Gate 0 ① 的被试机制）。

policy:
  - time_sharded : 嵌入与重排互斥驻留，用完即释（8GB 单卡的推荐路线）
  - co_resident  : 两者同时常驻（用于量出显存上界，作为反例证据）
  - cpu_sharded  : 两者落 CPU（回退分支①的第一档）
生成模型由 Ollama 常驻，不在本调度器管辖内，其显存占用通过全局采样体现。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from kee.config import get_settings
from kee.models.embedder import BgeM3Embedder
from kee.models.gpu import VramSampler
from kee.models.reranker import BgeReranker

POLICIES = ("time_sharded", "co_resident", "cpu_sharded")


@dataclass
class ResidencyEvent:
    role: str
    action: str  # load / unload / switch
    sec: float
    vram_before_mib: float
    vram_after_mib: float
    policy: str


class ResidencyScheduler:
    def __init__(self, policy: str = "time_sharded", dtype: str | None = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy 必须是 {POLICIES} 之一，收到 {policy}")
        s = get_settings()
        self.policy = policy
        self.device = "cpu" if policy == "cpu_sharded" else s.device
        self.dtype = dtype or s.model_dtype
        self._embedder = BgeM3Embedder(device=self.device, dtype=self.dtype)
        self._reranker = BgeReranker(device=self.device, dtype=self.dtype)
        self._probe = VramSampler()
        self.events: list[ResidencyEvent] = []
        self.resident_role: str | None = None

    # ---- 观测 ----
    @property
    def embedder(self) -> BgeM3Embedder:
        return self._embedder

    @property
    def reranker(self) -> BgeReranker:
        return self._reranker

    def _vram(self) -> float:
        return round(self._probe._global_used_mib(), 1)  # noqa: SLF001

    def _record(self, role: str, action: str, sec: float, before: float, after: float) -> None:
        self.events.append(
            ResidencyEvent(
                role=role,
                action=action,
                sec=round(sec, 3),
                vram_before_mib=before,
                vram_after_mib=after,
                policy=self.policy,
            )
        )

    def _release_other(self, keep: str) -> None:
        if self.policy != "time_sharded":
            return
        if keep == "embed" and self._reranker.resident:
            self.release("rerank")
        elif keep == "rerank" and self._embedder.resident:
            self.release("embed")

    def acquire(self, role: str) -> object:
        if role not in ("embed", "rerank"):
            raise ValueError(role)
        before = self._vram()
        t0 = time.perf_counter()
        self._release_other(role)
        model = self._embedder if role == "embed" else self._reranker
        if not model.resident:
            model.load()
        switch_sec = round(time.perf_counter() - t0, 3)
        after = self._vram()
        self.resident_role = role if self.policy == "time_sharded" else "both"
        action = "load" if switch_sec > 0.5 else "reuse"
        self._record(role, action, switch_sec, before, after)
        return model

    def release(self, role: str) -> None:
        model = self._embedder if role == "embed" else self._reranker
        before = self._vram()
        t0 = time.perf_counter()
        model.unload()
        self._record(role, "unload", round(time.perf_counter() - t0, 3), before, self._vram())
        if self.resident_role == role:
            self.resident_role = None

    def release_all(self) -> None:
        self.release("embed")
        self.release("rerank")

    @contextmanager
    def use(self, role: str):
        model = self.acquire(role)
        try:
            yield model
        finally:
            if self.policy == "time_sharded":
                self.release(role)

    def summary(self) -> dict:
        loads = [e for e in self.events if e.action in ("load", "switch")]
        cold = [e.sec for e in self.events if e.action == "load"]
        return {
            "policy": self.policy,
            "device": self.device,
            "dtype": self.dtype,
            "n_events": len(self.events),
            "n_cold_loads": len(cold),
            "cold_load_sec_max": round(max(cold), 3) if cold else 0.0,
            "cold_load_sec_mean": round(sum(cold) / len(cold), 3) if cold else 0.0,
            "embed_load_sec": self._embedder.load_sec,
            "rerank_load_sec": self._reranker.load_sec,
            "events": [asdict(e) for e in self.events],
        }


def _global_used_mib() -> float:
    sampler = VramSampler()
    return sampler._global_used_mib()  # noqa: SLF001
