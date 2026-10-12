"""gold 装配层的两条闸门：关键词命中面过宽要**被检测出来**，判据补充不能反过来把第一人的 gold 清零。

为什么值得单测：「命中面 > 3 段即缺陷」这条在 `eval_data` 的 docstring 里写了一年，
但从来没有代码执行它——结果就是发 9 道新题时这道检查一次都没跑（q-071 的 5 段命中就是这么漏出去的）。
把口径写进文档而不做成闸门，等于没有口径。
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from kee.eval import build_eval
from kee.eval.eval_data import DOC_SCOPED_QIDS, QUESTIONS


def _fake_corpus(monkeypatch, blocks: list[tuple[str, str]], parents: dict[str, str]) -> None:
    children = [{"id": cid, "content": text, "parent_id": None} for cid, text in blocks]
    monkeypatch.setattr(build_eval, "load_children", lambda pg: children)
    monkeypatch.setattr(build_eval, "load_parents", lambda pg: parents)


def test_wide_keyword_hit_is_flagged(monkeypatch):
    """4 段命中 = 缺陷；恰好 3 段不是（阈值边界要钉住，不能靠感觉）。"""
    blocks = [(f"doc@v1#1.1#c{i}", f"含 {'宽词' if i < 4 else '别的'} 的正文") for i in range(5)]
    _fake_corpus(monkeypatch, blocks, {})
    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {"qid": "q-t1", "question": "t", "category": "术语", "gold_keywords": ["宽词"],
         "expected_behavior": "answer"}])
    out = build_eval.build_eval_set(pg=None, write_files=False)
    d = out["stats"]["gold_defects"]
    assert [x["n_strict"] for x in d] == [4] and d[0]["waived"] is False
    assert out["stats"]["gold_defect_unwaived"] == ["q-t1"]


def test_hit_at_limit_not_flagged(monkeypatch):
    blocks = [(f"doc@v1#1.1#c{i}", "含 宽词 的正文") for i in range(3)]
    _fake_corpus(monkeypatch, blocks, {})
    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {"qid": "q-t2", "question": "t", "category": "术语", "gold_keywords": ["宽词"],
         "expected_behavior": "answer"}])
    assert build_eval.build_eval_set(pg=None, write_files=False)["stats"]["gold_defects"] == []


def test_waiver_requires_a_written_reason(monkeypatch):
    """豁免必须带理由，否则等于没豁免——空理由的 waiver 仍按未豁免处理。"""
    blocks = [(f"doc@v1#1.1#c{i}", "含 宽词 的正文") for i in range(4)]
    _fake_corpus(monkeypatch, blocks, {})
    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {"qid": "q-t3", "question": "t", "category": "术语", "gold_keywords": ["宽词"],
         "expected_behavior": "answer"}])
    monkeypatch.setitem(build_eval.GOLD_DEFECT_WAIVER, "q-t3", "")
    with pytest.raises(ValueError, match="q-t3"):
        build_eval.build_eval_set(pg=None, write_files=False, fail_on_gold_defect=True)


def test_defect_measured_on_final_gold_not_on_keyword_approximation(monkeypatch):
    """量在**最终 strict gold** 上：manual_drop 把宽命中收窄到位，就不该再报缺陷
    （第一版按覆盖前的数量，q-056 报 7 块，可它最终 gold 是 3 块且逐块与第二人判是对齐——误伤修正本身）；
    反过来 manual_add 把 gold 抬过上限，同样得被看一眼。"""
    blocks = [(f"doc@v1#1.1#c{i}", "含 宽词 的正文") for i in range(5)]
    _fake_corpus(monkeypatch, blocks, {})
    parent = {"qid": "q-t5", "question": "t", "category": "术语", "gold_keywords": ["宽词"],
              "expected_behavior": "answer"}
    monkeypatch.setattr(build_eval, "load_questions", lambda: [parent])
    assert build_eval.build_eval_set(pg=None, write_files=False)["stats"]["gold_defect_count"] == 1

    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {**parent, "gold_manual_drop": [f"doc@v1#1.1#c{i}" for i in range(2)]}])
    assert build_eval.build_eval_set(pg=None, write_files=False)["stats"]["gold_defects"] == []

    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {**parent, "gold_keywords": ["窄词"], "gold_manual_add": [f"doc@v1#1.1#c{i}" for i in range(4)]}])
    out = build_eval.build_eval_set(pg=None, write_files=False)
    assert [d["n_strict"] for d in out["stats"]["gold_defects"]] == [4], "人工加块抬过上限也要被看见"


def test_doc_scoped_qids_actually_name_a_document():
    """`DOC_SCOPED_QIDS` 会让第二人的 1 变 0，所以每条都必须题面真点了文档名——
    题面没点而表里配了，就是拿判据去改数据。"""
    by_id = {q["qid"]: q for q in QUESTIONS}
    for qid in DOC_SCOPED_QIDS:
        assert qid in by_id, f"{qid} 不在评测集里"
        assert re.search(r"(手册|规范|摘录)", by_id[qid]["question"]), f"{qid} 题面没点文档名"


def test_doc_scope_detector_flags_question_naming_a_doc_kind_with_gold_elsewhere(monkeypatch):
    """丙口径的操作化：题面点了「规范里」而 gold 全在别的文档类型 ⇒ 这是题面缺陷，必须被代码报出来。
    之前只靠我在注释里记着 q-044 是这么一道，注释会烂，检测器不会。"""
    blocks = [("fin-presale-prj-x@v1#2.1#c1", "含 堵死 的正文"), ("chm-runbook-prj-x@v1#2.2#c1", "含 堵死 的正文")]
    _fake_corpus(monkeypatch, blocks, {})
    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {"qid": "q-d1", "question": "想直接改库里字段，规范里哪句话把这条路堵死了？", "category": "口语化",
         "gold_keywords": ["堵死"], "expected_behavior": "answer"}])
    s = build_eval.build_eval_set(pg=None, write_files=False)["stats"]
    d = s["doc_scope_defects"]
    assert [x["qid"] for x in d] == ["q-d1"]
    assert d[0]["named_kinds"] == ["ifspec"] and set(d[0]["gold_kinds"]) == {"presale", "runbook"}
    assert s["doc_scope_defect_unwaived"] == ["q-d1"]
    with pytest.raises(ValueError, match="q-d1"):
        build_eval.build_eval_set(pg=None, write_files=False, fail_on_gold_defect=True)


def test_doc_scope_detector_stays_quiet_when_gold_is_in_the_named_doc(monkeypatch):
    """题面点「运维手册里」而 gold 确实在 runbook ⇒ 不报。
    检测器只抓限定词与证据不同源，不管别的宽度问题。"""
    blocks = [("chm-runbook-prj-x@v1#4.1#c1", "含 单元 的正文")]
    _fake_corpus(monkeypatch, blocks, {})
    monkeypatch.setattr(build_eval, "load_questions", lambda: [
        {"qid": "q-d2", "question": "化工运维手册里，防爆区域划分为几个单元？", "category": "跨文档",
         "gold_keywords": ["单元"], "expected_behavior": "answer"}])
    s = build_eval.build_eval_set(pg=None, write_files=False)["stats"]
    assert s["doc_scope_defects"] == []


def test_bing_landing_keeps_umbrella_block_and_drops_descriptive_one():
    """丙（《项目定义》V1.17）落到数据上的两处形状：
    q-071 问的是**约束** ⇒ 那份自我声明跨项目适用的合规摘录**不能被 drop**（第二人判 0，按裁决记为误判），
    同时补该项目自己的接口规范块；q-080 的「整定值不随优化目标调整」是描述型 ⇒ 必须落在 manual_drop 里。"""
    by = {q["qid"]: q for q in QUESTIONS}
    assert "chm-compliance-prj-chm-2023-021@v1#2.1#c2" not in (by["q-071"].get("gold_manual_drop") or [])
    assert "chm-ifspec-prj-chm-2024-003@v1#1.2#c1" in by["q-071"]["gold_manual_add"]
    assert "chm-closeout-prj-chm-2024-003@v1#2.2#c1" in by["q-080"]["gold_manual_drop"]


def test_q044_is_deliberately_outside_doc_scope():
    """q-044 题面说「规范里」，而它的两块 gold 在售前方案与运维手册里 —— 一份规范都不是。
    那是题面缺陷（登记待裁），硬配 doc_scope 等于用判据把第一人自己的 gold 清零。"""
    assert "q-044" not in DOC_SCOPED_QIDS


def test_module_docstring_question_count_matches_data():
    """docstring 里的题量/分类计数与真实数据必须一致——分段头曾经写着 15 题而实际早改了。"""
    from kee.eval import eval_data

    doc = eval_data.__doc__ or ""
    m = re.search(r"（共 (\d+) 题）：([^。\n]+)。", doc)
    assert m, "docstring 里的题量行没找到（改题量要同步改这里）"
    assert int(m.group(1)) == len(QUESTIONS)
    claimed = dict(p.rsplit(" ", 1) for p in m.group(2).split("、"))
    real = {k: str(v) for k, v in Counter(q["category"] for q in QUESTIONS).items()}
    assert claimed == real, f"docstring 说 {claimed}，实际 {real}"


def test_batch4_edits_are_in_place():
    """第四批尺子改动的落点：q-071 收窄成单对象、整定值另立 q-080、q-025 补了漏收的那块。"""
    q71 = next(q for q in QUESTIONS if q["qid"] == "q-071")
    q80 = next(q for q in QUESTIONS if q["qid"] == "q-080")
    q25 = next(q for q in QUESTIONS if q["qid"] == "q-025")
    assert "或整定值" not in q71["question"], "q-071 还是双对象题面"
    assert q71["gold_keywords"] == ["输出回路", "优化类应用"]
    assert q80["expected_behavior"] == "answer" and "整定值" in q80["question"]
    assert "chm-compliance-prj-chm-2023-021@v1#1.1#c1" in q25["gold_manual_add"]
