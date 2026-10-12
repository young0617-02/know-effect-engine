"""ADR-12 决策缺的那个数：**容器化只能带 CPU torch 时，嵌入/重排要慢多少**。

为什么必须实测而不是估算：ADR-12 的僵局是「API 要不要进容器」，而反对进容器的人（包括我此前写的）
说的都是「要拉 GB 级轮子 / 再要一份 CUDA 运行时」——**没人量过代价的另一头**：
就算不装 GPU 版，CPU 路径下的检索延迟还成不成立 P95 < 800ms 这条门禁。
这句话没有数，就不该拿它当论据。

零下载做法：本机 CUDA 环境里同一套权重强制 `device=cpu` 跑一遍，与 GPU 跑同一份工作对照。
不是模拟容器，是**容器里唯一可行的那条路径的下界**（容器还要多付一层 gRPC/网络，只会更慢不会更快）。

跑法：uv run python scripts/m4_cpu_path_probe.py
产物：var/gate0/m4_cpu_path.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "var" / "gate0" / "m4_cpu_path.json"
N_QUERY = 20          # 与基线同一量级的查询数
N_PAIR = 20          # 每次重排的候选数
TEXTS = [
    "银企直联实时清算方案 V3.2 的灾备切换目标：RTO 小于 15 分钟，RPO 等于 0。",
    "生产控制大区与管理信息大区之间的数据交换只能经正反向隔离装置以单向方式完成。",
    "SIS 安全仪表系统采用 2oo3 表决逻辑，独立于优化平台部署，整定值不随优化目标调整。",
    "装置区按防爆区域划分原则划分为 1 区与 2 区，进入 1 区作业须办理受限空间与动火两项作业票。",
    "运维对象含记录模板引擎、电子签名服务、审计追踪存储与采集网关四类，纳管产线 4 条。",
    "ITSS 事件分级在金融线把清算链路 P1 定为 15 分钟内启动应急响应。",
    "五步交付法：现状诊断、方案定型、试点验证、规模推广、成效固化。",
    "客户数据不出域，四道防线为环境隔离、传输受控、密级过滤、审计追溯。",
    "配电自动化终端的遥测数据上送覆盖三相电流、三相电压、有功与无功功率及功率因数共 14 个测点。",
    "CMMI 3 级过程域映射要求登记度量项并按季度回灌组织过程资产库。",
]
QUERIES = [
    "RTO 要求是多少", "跨区数据交换要经过什么装置", "SIS 用什么表决逻辑", "1 区作业要办哪些票",
    "运维对象有几类", "清算链路 P1 多久响应", "五步交付法有哪五个阶段", "数据不出域有几道防线",
    "遥测上送多少个测点", "过程域度量项登记哪些", "灾备切换目标", "隔离装置部署几台",
    "优化平台能不能改整定值", "防爆区域按什么国标", "审计追踪能不能改", "结项验收指标是什么",
    "采集成功率提升到多少", "头寸预测偏差收敛到多少", "拨备覆盖率最低要求", "两地三中心怎么部署",
]


def timed(fn, reps: int = 3) -> dict:
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append(round((time.perf_counter() - t0) * 1000, 1))
    return {"per_rep_ms": xs, "p50": statistics.median(xs), "max": max(xs)}


def run(device: str, dtype: str | None = None) -> dict:
    from kee.models.embedder import BgeM3Embedder
    from kee.models.reranker import BgeReranker

    kw = {"device": device} if dtype is None else {"device": device, "dtype": dtype}

    t0 = time.perf_counter()
    emb = BgeM3Embedder(**kw)
    emb.load()
    load_embed_ms = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    rer = BgeReranker(**kw)
    rer.load()
    load_rerank_ms = round((time.perf_counter() - t1) * 1000, 1)

    def embed_passages() -> None:
        emb.embed_passages(TEXTS * (N_QUERY // len(TEXTS) + 1), batch_size=16)

    embed_query_stats: dict = {}

    def embed_query() -> None:
        """测的是**单条查询**的耗时：取每条的均值，而不是把 N 条的总时长当成一次检索。
        第一版就把这个搞错了，报出 GPU 合成 909.7ms，而基线明明是 326ms。"""
        xs = []
        for q in QUERIES:
            t = time.perf_counter()
            emb.embed_query(q)
            xs.append(round((time.perf_counter() - t) * 1000, 1))
        embed_query_stats.update({"per_query_ms": xs})

    passages = (TEXTS + TEXTS)[:N_PAIR]

    def rerank() -> None:
        # 一条查询对 20 个候选打分——正是线上取 Top-5 之前那一趟
        rer.score(QUERIES[0], passages)

    out = {
        "device": device,
        "load_embed_ms": load_embed_ms,
        "load_rerank_ms": load_rerank_ms,
        "embed_passages_20": timed(embed_passages),
        "embed_query_per": {
            **timed(embed_query),
            "单条_p50": statistics.median(embed_query_stats["per_query_ms"]),
            "单条_p95": sorted(embed_query_stats["per_query_ms"])[int(0.95 * len(embed_query_stats["per_query_ms"])) - 1],
        },
        "rerank_20x1": timed(rerank),
    }
    # 一次检索的粗略合成：1 次查询嵌入 + 1 趟重排（BM25 腿在 Milvus 服务端，与设备无关）
    out["retrieval_composite_p50_ms"] = round(out["embed_query_per"]["单条_p50"] + out["rerank_20x1"]["p50"], 1)
    out["retrieval_composite_worst_ms"] = round(out["embed_query_per"]["单条_p95"] + out["rerank_20x1"]["max"], 1)
    del emb, rer
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    return out


def main() -> int:
    gpu = run("cuda")
    cpu = run("cpu")                       # fp16 在 CPU 上是软件模拟，未必是容器里该有的配法
    cpu32 = run("cpu", "float32")
    gate = 800.0
    rep = {
        "env": {"ts": datetime.now().isoformat(timespec="seconds"),
                "commit": subprocess_short("git rev-parse --short HEAD"),
                "gpu": subprocess_short("nvidia-smi --query-gpu=name --format=csv,noheader")},
        "note": "同一套本地权重、同一份工作，只换 device。这是「容器里只能 CPU」那条路径的**下界**："
                "容器还要多付一层网络与 gRPC，实际只会更慢。",
        "cuda": gpu, "cpu": cpu, "cpu_float32": cpu32,
        "gate_P95_retrieval_ms": gate,
        "cpu_over_gate": {
            "composite_p50": cpu["retrieval_composite_p50_ms"],
            "worst": cpu["retrieval_composite_worst_ms"],
            "p50_超门禁": cpu["retrieval_composite_p50_ms"] > gate,
            "worst_超门禁": cpu["retrieval_composite_worst_ms"] > gate,
        },
        "adr12_measured_facts": {
            "本地有没有 GPU 基础镜像": any("cuda" in l.lower() for l in subprocess_short("docker images --format '{{.Repository}}:{{.Tag}}'").split("\n")),
            "CPU 权重装载耗时对比_gpu": {"embed_gpu_ms": gpu["load_embed_ms"], "embed_cpu_ms": cpu["load_embed_ms"],
                                    "rerank_gpu_ms": gpu["load_rerank_ms"], "rerank_cpu_ms": cpu["load_rerank_ms"]},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in ("cuda", "cpu", "cpu_float32"):
        d = rep[k]
        print(f"{k:12s} 重排 {d['rerank_20x1']['p50']:.1f}ms ｜ "
              f"单条查询嵌入 {d['embed_query_per']['单条_p50']:.1f}ms ｜ "
              f"合成一次检索 {d['retrieval_composite_p50_ms']}ms（最差 {d['retrieval_composite_worst_ms']}ms）｜ "
              f"装载 embed {d['load_embed_ms']:.0f}ms / rerank {d['load_rerank_ms']:.0f}ms")
    print(f"\n门禁 P95 检索 {gate:.0f}ms：CPU 路径 p50 超门禁={rep['cpu_over_gate']['p50_超门禁']}、"
          f"最差值超门禁={rep['cpu_over_gate']['worst_超门禁']}")
    print("证据：", OUT)
    return 0


def subprocess_short(cmd: str) -> str:
    import subprocess

    try:
        return subprocess.run(cmd.split(), cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"（取不到：{type(e).__name__}）"


if __name__ == "__main__":
    raise SystemExit(main())
