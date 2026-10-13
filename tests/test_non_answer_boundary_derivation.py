"""非作答 12 题那半边的边界句**必须由 #57 产物推导**，不许写死。

起因（2026-10-13 检查项目状态时抓到）：`answer_correct_calibration.json` 的
`coverage_boundary` 里硬编码着「这 12 题没有真人标签，仍是空白」，而 #57 早已把那 12 行判完——
产物在标签回收之后继续谎报空白。写死的边界句会朝两个方向说谎：标签回来了还说没回来；
真幻觉率哪天有了分母还会继续说「不适用」。所以把它做成函数，三种状态各测一条。

钉的是措辞与推导，不是读数：ECE / accuracy 那些数由别的测试与产物本身负责。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m5_answer_correct_calibration as cal  # noqa: E402

FULL = {
    "complete": True, "n_rows_total": 12, "n_rows_judged": 12, "blank_rows": [],
    "question_hash": 3508928384,
    "system_behavior_shape": {"rows_with_conclusion": 0, "rows_without_conclusion": 12},
    "metrics": {"hallucination_unsupported_conclusion":
                {"n": 0, "numerator": 0, "value": None, "reading": "不适用（此类样本为 0，不得读成 0%）"}},
}


def test_absent_labels_says_blank():
    """#57 之前：产物不在 ⇒ 老实说空白，且点名缺的是哪个文件。"""
    s = cal.non_answer_boundary({})
    assert "仍是空白" in s and "hallucination_rate.json" in s


def test_partial_labels_are_not_reported_as_blank_or_reading():
    """填了一半时既不能说「没标」也不能说「不适用」——要报出判了几行、哪些还空着。"""
    s = cal.non_answer_boundary({**FULL, "complete": False, "n_rows_judged": 5, "blank_rows": ["q-063"]})
    assert "5/12" in s and "q-063" in s
    assert "空白" not in s and "不适用" not in s


def test_recollected_labels_are_not_reported_as_blank():
    """本次修的那条：标签回收后不许再说「没有真人标签」；分母为 0 也不是 0%。"""
    s = cal.non_answer_boundary(FULL)
    assert "已回收" in s and "没有真人标签" not in s and "仍是空白" not in s
    assert "不适用" in s and "0%" in s              # 「不是没标，也不是 0%」那半句
    assert "3508928384" in s                         # 边界跟着具体那代产物走，不泛化


def test_a_real_denominator_stops_saying_not_applicable():
    """反向：真给出过结论的行 >0 时读数就是实数，那句「按构造不可判」必须自己消失。"""
    with_face = {**FULL,
                 "system_behavior_shape": {"rows_with_conclusion": 3, "rows_without_conclusion": 9},
                 "metrics": {"hallucination_unsupported_conclusion":
                             {"n": 3, "numerator": 1, "value": 0.3333, "reading": "1/3 = 33.33%"}}}
    s = cal.non_answer_boundary(with_face)
    assert "1/3 = 33.33%" in s and "n=3" in s
    assert "按构造不可判" not in s and "不适用" not in s
    assert "3508928384" in s                        # 代次仍要同句带：换生成就要整套重测
