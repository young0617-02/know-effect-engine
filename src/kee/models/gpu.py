"""显存采样：优先 NVML（毫秒级），不可用时退化为 nvidia-smi 轮询。

Windows WDDM 下 `nvidia-smi --query-compute-apps` 的 per-process 显存为 N/A，
因此进程级归因只能用「全局用量差分 + 本进程 torch reserved」两条证据链。
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


class VramSampler:
    """后台线程按固定间隔采集整卡显存与本进程 torch 显存。"""

    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
            self.source = "nvml"
        except Exception:  # noqa: BLE001
            self.source = "nvidia-smi"

    def _global_used_mib(self) -> float:
        if self._handle is not None:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / 1048576.0
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip().splitlines()[0])

    @staticmethod
    def _torch_mib() -> tuple[float, float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0, 0.0
            return (
                torch.cuda.memory_allocated() / 1048576.0,
                torch.cuda.memory_reserved() / 1048576.0,
            )
        except Exception:  # noqa: BLE001
            return 0.0, 0.0

    def _loop(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            alloc, reserved = self._torch_mib()
            self.samples.append(
                {
                    "t": round(time.perf_counter() - t0, 3),
                    "gpu_used_mib": round(self._global_used_mib(), 1),
                    "torch_alloc_mib": round(alloc, 1),
                    "torch_reserved_mib": round(reserved, 1),
                }
            )
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            return {"peak_mib": 0.0, "mean_mib": 0.0, "n_samples": 0, "peak_torch_alloc_mib": 0.0}
        used = [s["gpu_used_mib"] for s in self.samples]
        alloc = [s["torch_alloc_mib"] for s in self.samples]
        return {
            "peak_mib": max(used),
            "mean_mib": round(sum(used) / len(used), 1),
            "peak_torch_alloc_mib": max(alloc),
            "n_samples": len(used),
        }


def vram_used_mib() -> float:
    s = VramSampler()
    return s._global_used_mib()  # noqa: SLF001 - 单次取样快捷入口


@dataclass
class OomWatchdog:
    """记录采样期间的显存峰值与越界次数（OOM 判定用）。"""

    limit_mib: float
    events: list[dict[str, float]] = field(default_factory=list)

    def check(self, samples: list[dict[str, float]]) -> int:
        breaches = [s for s in samples if s["gpu_used_mib"] > self.limit_mib]
        self.events = breaches
        return len(breaches)
