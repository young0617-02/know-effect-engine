"""显存观测与模型驻留调度（Gate 0 ① 的机制层）。"""

from kee.models.embedder import BgeM3Embedder
from kee.models.gpu import VramSampler, vram_used_mib
from kee.models.reranker import BgeReranker
from kee.models.registry import ResidencyScheduler

__all__ = [
    "BgeM3Embedder",
    "BgeReranker",
    "ResidencyScheduler",
    "VramSampler",
    "vram_used_mib",
]
