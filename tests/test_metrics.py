import math

import pytest

from kee.eval.metrics import (
    aggregate_qa_metrics,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_mrr_precision_basics():
    ranked = ["a", "b", "c", "d", "e", "f"]
    gold = {"c", "x"}
    assert recall_at_k(ranked, gold, k=5) == 0.5
    assert reciprocal_rank(ranked, gold) == 1 / 3
    assert precision_at_k(ranked, gold, k=5) == 0.2
    assert math.isnan(recall_at_k(ranked, set(), k=5))


def test_aggregate_skips_ungradeable():
    rows = [
        {"ranked_ids": ["a", "b"], "gold_ids": {"a"}},
        {"ranked_ids": ["a", "b"], "gold_ids": set()},
    ]
    out = aggregate_qa_metrics(rows, k=5)
    assert out["n_graded"] == 1 and out["n_total"] == 2
    assert out["recall@5"] == 1.0 and out["mrr"] == 1.0


def test_percentile_linear_interpolation():
    vals = [10, 20, 30, 40]
    assert percentile(vals, 50) == 25
    assert percentile(vals, 0) == 10
    assert percentile(vals, 100) == 40


def test_temperature_scaling_moves_confidence_toward_accuracy():
    # 过度自信：预测 0.9 但只有 40% 正确
    logits = [math.log(0.9 / 0.1)] * 10
    labels = [1] * 4 + [0] * 6
    fitted = fit_temperature(logits, labels)
    assert fitted["temperature"] > 1.0
    cal = apply_temperature(logits, fitted["temperature"])
    ece_raw = expected_calibration_error([0.9] * 10, labels)["ece"]
    ece_cal = expected_calibration_error(cal, labels)["ece"]
    assert ece_cal < ece_raw


def test_ece_uses_bins_and_ignores_nan():
    out = expected_calibration_error([0.5, float("nan"), 0.5], [1, 0, 0])
    assert out["n"] == 2
    assert abs(out["ece"] - 0.0) < 1e-9  # 两样本 conf=acc=0.5


def test_temperature_refuses_to_fit_a_zero_variance_label():
    """标签零方差时温度不可辨识——网格上必然选极端 T，那是缩放不是校准。

    钉的是 M3 真实踩过的坑：n=7 全对的集合拟合出 T=0.05，看着是「校准过了」，
    实际把置信度整体推向过度自信。拒绝必须返回恒等 T=1.0 且带 refused 说明，
    让调用方拿不到一个可以冒充拟合成果的数字。
    """
    logits = [math.log(0.9 / 0.1)] * 7
    all_pos = fit_temperature(logits, [1] * 7)
    assert all_pos["temperature"] == 1.0
    assert "零方差" in all_pos["refused"]
    assert all_pos["n"] == 7
    # 全负类同样拒
    assert "refused" in fit_temperature(logits, [0] * 7)
    # 有方差时不受影响，仍然正常拟合
    mixed = fit_temperature(logits, [1] * 3 + [0] * 4)
    assert "refused" not in mixed and mixed["temperature"] != 1.0


def test_temperature_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_temperature([0.1, 0.2, 0.3], [1, 0])


def test_probe_guard_refuses_to_report_a_dead_generator_as_a_pass():
    """一次依赖故障会伪装成「所有题都干净地拒答」——那是假绿里最难发现的一类。

    2026-10-11 真机撞见过：Ollama /api/tags 返回 200 但 llama-server 二进制缺失，
    整批题都走到 generation_failed（degraded + 0 字结论）。
    此时无守卫的守门脚本会报「refuse 类 0 泄漏 → PASS」。
    """
    from kee.eval.probe_guard import generation_failure_share, unusable_run

    dead_run = [["generation_failed:http 500"], ["no_sufficient_evidence"],
                ["generation_failed:http 500"], ["model_declared_no_evidence"]]
    share = generation_failure_share(dead_run)
    assert share == 0.5
    assert "不是测量结果" in unusable_run(share)

    # 健康的一轮不该被拦（个别题失败本来就该允许）
    healthy = [["contract_violation:empty_conclusion"], ["no_sufficient_evidence"], ["ok"], ["ok"], ["ok"]]
    assert generation_failure_share(healthy) == 0.0
    assert unusable_run(generation_failure_share(healthy)) is None

    # 只有 1/20 行失败：在阈值内，不拦
    mostly_ok = [["ok"]] * 19 + [["generation_failed:timeout"]]
    assert unusable_run(generation_failure_share(mostly_ok), threshold=0.2) is None


def test_cohens_kappa_rejects_string_labels_loudly():
    """字符串标注会让 Counter[1] 恒为 0 ⇒ pe==1 ⇒ 返回 NaN：一个"看起来算过"的坏数。
    我在重测信度那段真踩到了，所以这里挡住而不是留个 NaN 让下游猜。"""
    import pytest

    from kee.eval.kappa import cohens_kappa

    with pytest.raises(TypeError, match="int 0/1"):
        cohens_kappa(["1", "0"], ["1", "1"])
    assert cohens_kappa([1, 0], [1, 1])["kappa"] is not None
