"""测试用户剧本的分母规则：钉「谁进分母」，不钉读数。

为什么单独测：第一版把 q-047（gold 在密级 3）手标成「该答」交给交付工程师（上限 ≤2），
系统**按权限正确拒答**却被算成一次失败；反过来若把「该答」的分母交给脚本作者手挑，
冒烟那轮就会跑出 5/5 = 100% —— n 小且题由我选，这个数只会复现挑题这个动作。
两件事都是**分母错**，而分母错不会让任何数字变难看，只会让它错得可信。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m5_test_user_run as tu  # noqa: E402


def test_invisible_gold_makes_refusal_correct_not_a_failure():
    """交付工程师（≤2）问 gold 在密级 3 的题 ⇒ 桶必须是「权限拦截」且不许给结论。"""
    e = tu.role_expectation("answer", [3], 2)
    assert e["桶"] == "权限拦截" and e["该给结论"] is False


def test_visible_gold_stays_in_the_self_serve_denominator():
    e = tu.role_expectation("answer", [2], 2)
    assert e["桶"] == "作答" and e["该给结论"] is True and e["该转人工"] is False


def test_escalate_only_counts_where_the_role_can_see_it():
    """转人工的触发条件是「引用里含密级 3 证据」——看不见就无从触发，这一格必须落到拦截桶，
    否则「转人工触发率」的分母会混进根本测不到这条路径的角色。"""
    assert tu.role_expectation("escalate", [3], 3)["桶"] == "转人工"
    assert tu.role_expectation("escalate", [3], 2)["桶"] == "权限拦截"


def test_no_gold_means_refuse_not_permission_block():
    """语料内本就无依据（gold 为空）与「被权限挡掉」是两件事，混在一个桶里就读不出归因。"""
    e = tu.role_expectation("refuse", [], 3)
    assert e["桶"] == "拒答" and e["该给结论"] is False


def test_plan_is_derived_from_the_whole_eval_set_not_from_hand_picked_qids():
    by_q = {
        "q-001": {"question": "A", "expected_behavior": "answer", "gold_chunk_ids": ["c2"]},
        "q-002": {"question": "B", "expected_behavior": "answer", "gold_chunk_ids": ["c3"]},
        "q-003": {"question": "C", "expected_behavior": "refuse", "gold_chunk_ids": []},
        "q-004": {"question": "D", "expected_behavior": "escalate", "gold_chunk_ids": ["c3"]},
    }
    lv = {"c2": 2, "c3": 3}
    plan = tu.build_plan(by_q, lv)
    assigned = {p["qid"]: p["user"] for p in plan if p["评测集期望"] != "answer" or p["桶"] != "权限拦截"}
    # 每题都出现（全量派生 ⇒ 分母由评测集决定）
    assert {p["qid"] for p in plan} >= set(by_q)
    # 密级 2 的题交给「刚好看得见」的最低角色（delivery ≤2），不是无脑给最高权限
    assert assigned["q-001"] == "u-delivery-01"
    # 密级 3 的题只能给 pm（≤3）
    assert assigned["q-002"] == "u-pm-01"
    # 无 gold 的拒答题给最高权限：让它的拒答归因于「语料没有」而不是「看不见」
    assert assigned["q-003"] == "u-pm-01"


def test_blocking_bucket_is_never_empty():
    """拦截任务必须**显式加**：自动派生只会把题交给看得见的角色，
    于是「权限拦截正确率」会是一个空集上的真——那正是本项目反复登记要防的形状。"""
    by_q = {"q-001": {"question": "A", "expected_behavior": "answer", "gold_chunk_ids": ["c3"]},
            "q-002": {"question": "B", "expected_behavior": "answer", "gold_chunk_ids": ["c3"]}}
    plan = tu.build_plan(by_q, {"c3": 3})
    assert any(p["桶"] == "权限拦截" for p in plan)
    assert any(p["桶"] == "作答" for p in plan)


def test_smoke_subset_is_still_declared_not_silently_used():
    """`--smoke` 传进来的 qid 只影响题面，不改分母规则；冒烟读数为什么不算业务结论由产物字段说明。"""
    by_q = {f"q-00{i}": {"question": "x", "expected_behavior": "answer", "gold_chunk_ids": ["c2"]} for i in (1, 2, 3)}
    plan = tu.build_plan(by_q, {"c2": 2}, ["q-001"])
    assert len([p for p in plan if p["桶"] == "作答"]) == 1
