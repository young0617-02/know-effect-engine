"""评测指标与置信度校准（纯函数，可单测、可回归）。"""

from __future__ import annotations

import math
from collections import Counter


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    if not gold_ids:
        return float("nan")
    top = set(ranked_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def hit_at_k(ranked_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    return 1.0 if set(ranked_ids[:k]) & gold_ids else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    if k <= 0:
        return float("nan")
    return len(set(ranked_ids[:k]) & gold_ids) / k


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位（与 numpy.percentile 同口径，避免为探针引入额外依赖）。"""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def latency_report(values_ms: list[float]) -> dict[str, float | int]:
    if not values_ms:
        return {"n": 0, "p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "n": len(values_ms),
        "p50": round(percentile(values_ms, 50), 1),
        "p95": round(percentile(values_ms, 95), 1),
        "p99": round(percentile(values_ms, 99), 1),
        "max": round(max(values_ms), 1),
        "mean": round(sum(values_ms) / len(values_ms), 1),
    }


def aggregate_qa_metrics(rows: list[dict], k: int = 5) -> dict:
    """rows: [{"ranked_ids": [...], "gold_ids": set([...])}]，按题平均。"""
    scored = [r for r in rows if r["gold_ids"]]
    n = len(scored)
    if not n:
        return {"n_graded": 0}
    return {
        "n_graded": n,
        "n_total": len(rows),
        f"recall@{k}": round(sum(recall_at_k(r["ranked_ids"], r["gold_ids"], k) for r in scored) / n, 4),
        f"hit@{k}": round(sum(hit_at_k(r["ranked_ids"], r["gold_ids"], k) for r in scored) / n, 4),
        f"precision@{k}": round(sum(precision_at_k(r["ranked_ids"], r["gold_ids"], k) for r in scored) / n, 4),
        "mrr": round(sum(reciprocal_rank(r["ranked_ids"], r["gold_ids"]) for r in scored) / n, 4),
        "zero_recall_rate": round(
            sum(1 for r in scored if not set(r["ranked_ids"][:k]) & r["gold_ids"]) / n, 4
        ),
    }


# ---------------- 校准 ----------------


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def expected_calibration_error(confidences: list[float], labels: list[int], bins: int = 10) -> dict:
    """ECE = Σ (n_b/N)·|acc_b − conf_b|；只统计非 NaN 的样本。"""
    pairs = [(c, y) for c, y in zip(confidences, labels) if not (isinstance(c, float) and math.isnan(c))]
    n = len(pairs)
    if n == 0:
        return {"ece": float("nan"), "n": 0, "bins": []}
    bucket: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for c, y in pairs:
        idx = min(int(c * bins), bins - 1)
        bucket[idx].append((c, y))
    ece = 0.0
    rows = []
    for b, items in enumerate(bucket):
        if not items:
            continue
        conf = sum(c for c, _ in items) / len(items)
        acc = sum(y for _, y in items) / len(items)
        weight = len(items) / n
        ece += weight * abs(acc - conf)
        rows.append({"bin": b, "n": len(items), "conf": round(conf, 4), "acc": round(acc, 4)})
    return {
        "ece": round(ece, 4),
        "n": n,
        "mean_conf": round(sum(c for c, _ in pairs) / n, 4),
        "accuracy": round(sum(y for _, y in pairs) / n, 4),
        "bins": rows,
    }


def nll_with_temperature(logits: list[float], labels: list[int], t: float) -> float:
    if t <= 0:
        return float("inf")
    total = 0.0
    for z, y in zip(logits, labels):
        p = sigmoid(z / t)
        p = min(max(p, 1e-9), 1 - 1e-9)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / max(len(labels), 1)


def fit_temperature(logits: list[float], labels: list[int], grid: list[float] | None = None) -> dict:
    """一维网格 + 局部细化求 T*（最小化 NLL）。透明优先于花哨。

    **零方差标签直接拒绝拟合**：标签全 1 时 NLL 的单调解是「越自信越好」，网格上必然选中最小 T；
    全 0 时同理选中最大 T。那不是校准，是把温度当成缩放常数用。
    M3 那次 n=7 全对拟合出 T=0.05 就是这个坑——返回值看着正常，实际把置信度整体推向过度自信。
    拒绝时返回 T=1.0（恒等）并带 `refused`，让调用方无法把结果当拟合成果拿去用。
    """
    if not logits:
        return {"temperature": 1.0, "n": 0}
    if len(labels) != len(logits):
        raise ValueError("logits 与 labels 长度必须一致")
    ones = sum(labels)
    if ones == 0 or ones == len(labels):
        return {
            "temperature": 1.0,
            "n": len(labels),
            "refused": f"标签零方差（{ones}/{len(labels)} 为正类），温度不可辨识",
            "nll_before_1.0": round(nll_with_temperature(logits, labels, 1.0), 4),
        }
    coarse = grid or [round(0.05 * i, 2) for i in range(1, 101)]  # 0.05 ~ 5.0
    best_t, best_loss = 1.0, float("inf")
    for t in coarse:
        loss = nll_with_temperature(logits, labels, t)
        if loss < best_loss:
            best_t, best_loss = t, loss
    for _ in range(3):  # 两轮细化
        step = best_t * 0.1
        for t in [best_t - step, best_t - step / 2, best_t, best_t + step / 2, best_t + step]:
            if t <= 0:
                continue
            loss = nll_with_temperature(logits, labels, t)
            if loss < best_loss:
                best_t, best_loss = round(t, 4), loss
    return {
        "temperature": round(best_t, 4),
        "nll_before_1.0": round(nll_with_temperature(logits, labels, 1.0), 4),
        "nll_at_best": round(best_loss, 4),
        "n": len(labels),
    }


def apply_temperature(logits: list[float], t: float) -> list[float]:
    return [round(sigmoid(z / t), 4) for z in logits]


def confusion_counts(labels: list[int], preds: list[int]) -> dict:
    c = Counter((y, p) for y, p in zip(labels, preds))
    return {f"y{y}_p{p}": n for (y, p), n in sorted(c.items())}
