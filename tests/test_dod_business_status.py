"""DoD#6 状态推导的四种落点。

危险方向各不相同，所以四种都要钉住：
- 缺产物却写「不可测」＝把**没跑过**说成**结构性拿不到**，之后没人会去补跑；
- 冒烟读数当业务结论＝拿自己挑的题冒充用户（本项目第一版就跑出过 5/5=100%）；
- 链路有失败还报比率＝系统根本没在正常跑，数字无从谈起；
- 门禁过了就写「达标」＝把三条里只有一条可测的格子说成全过。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m5_dod_check as dod  # noqa: E402


def _tu(value=0.94, source="评测集全量自动派生（题与角色都不由脚本作者挑）", n=68):
    return {"剧本来源": source, "metrics": {"模拟自助解答率": {"value": value, "n_tasks": n, "reading": f"{n}n"}}}


def test_missing_artifact_is_not_marked_不可测_with_a_clean_sounding_word():
    status, say = dod.business_status({}, [])
    assert status == "待复测" or status == "不可测"
    assert "产物缺失" in say and "m5_test_user_run.py" in say


def test_smoke_reading_never_becomes_a_business_verdict():
    """手挑题的比率不能进门禁判定，哪怕数值远高于 60%。"""
    status, say = dod.business_status(_tu(source="冒烟：SMOKE_QIDS 手挑 11 题"), [])
    assert status == "待复测"
    assert "不能当业务读数" in say


def test_link_failure_dominates_the_ratio():
    status, say = dod.business_status(_tu(), ["SSE 流式可通且有可读首字"])
    assert status == "不达标" and "链路检查" in say


def test_empty_denominator_is_not_zero_percent():
    status, say = dod.business_status(_tu(value=None), [])
    assert status == "不可测" and "分母为空" in say


def test_passing_the_gate_still_caps_at_部分达标():
    """自助解答率过了 60%，另两条仍不可测 ⇒ 这一格最多「部分达标」，不许写「达标」。"""
    assert dod.business_status(_tu(value=0.9412), [])[0] == "部分达标"
    assert dod.business_status(_tu(value=0.60), [])[0] == "部分达标"
    assert dod.business_status(_tu(value=0.59), [])[0] == "不达标"
