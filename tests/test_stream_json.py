"""增量 JSON 字段扫描器的测试。

这个模块的 bug 会直接把错的内容显示给用户看，所以核心断言不是「结果对不对」一条，
而是一条**性质**：对同一份完整输出的**每一个前缀**，扫出来的值必须是最终值的**前缀**。
也就是说流式过程中绝不许出现「先显示一段、后面又被改掉」的内容——那是撤回之外更糟的形态，
用户看到的那几秒里它是完整的、错的。

其余用例覆盖真实会撞上的坑：深层同名键、转义被拆在两帧之间、值不是字符串、字段缺失。
"""

from __future__ import annotations

import json

from kee.qa.stream_json import partial_string_field

FULL = json.dumps(
    {
        "conclusion": "灾备切换 RTO < 15 分钟。",
        "key_points": [{"point": "conclusion 这个词出现在要点里也不该被认成结论", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "doc_id": "d"}],
        "no_evidence": False,
        "self_assessed_confidence": 0.8,
    },
    ensure_ascii=False,
)


def _final(buf: str, key: str) -> str:
    return json.loads(buf)[key]


def test_every_prefix_yields_only_a_prefix_of_the_final_value():
    """核心性质：流式过程中扫出的内容，永远不能超过最终值、也不能与之前矛盾。"""
    shown = ""
    for i in range(len(FULL) + 1):
        val, closed = partial_string_field(FULL[:i], "conclusion")
        assert _final(FULL, "conclusion").startswith(val), f"第 {i} 帧扫出了最终值之外的内容：{val!r}"
        assert val.startswith(shown) or shown.startswith(val), "同一字段的内容在帧之间发生了回退"
        shown = max(shown, val, key=len)
    assert partial_string_field(FULL, "conclusion") == (_final(FULL, "conclusion"), True)


def test_nested_key_with_the_same_name_does_not_leak_into_the_top_level_field():
    """深层对象里的同名键必须不算。否则一条要点会把结论区染成要点文字。"""
    buf = '{"conclusion": "", "key_points": [{"point": "x", "conclusion": "不该被读到的深层值"}]}'
    assert partial_string_field(buf, "conclusion") == ("", True)


def test_boolean_field_is_not_mistaken_for_text():
    """`no_evidence` 是布尔：读不到文本就明说读不到，别把 `False` 当字符串吐出去。"""
    assert partial_string_field(FULL, "no_evidence")[0] == ""


def test_absent_field_reads_as_empty_and_not_closed():
    assert partial_string_field('{"conclusion": "a"}', "nonexistent") == ("", False)


def test_escapes_survive_being_split_across_frames():
    """转义序列被拆在两帧之间时，宁可暂时少显示那个字符，也不许露出 `\\` 残骸。"""
    obj = {"conclusion": '他说"要停机"，然后\n换行', "key_points": []}
    buf = json.dumps(obj, ensure_ascii=False)
    assert partial_string_field(buf, "conclusion")[0] == obj["conclusion"]
    # 逐帧看：任何一帧的输出里都不该出现孤立的反斜杠
    for i in range(len(buf) + 1):
        val, _ = partial_string_field(buf[:i], "conclusion")
        assert not val.endswith("\\"), f"第 {i} 帧把半个转义显示出来了：{val!r}"
        assert obj["conclusion"].startswith(val)


def test_unicode_escape_is_decoded():
    buf = '{"conclusion": "\\u707e\\u5907 RTO", "key_points": []}'
    assert partial_string_field(buf, "conclusion") == ("灾备 RTO", True)


def test_partial_key_name_is_not_matched_to_the_wrong_field():
    """键名还没写完时不能猜。`concl` 前缀匹配到 conclusion 会把别的字段的值端出去。"""
    buf = '{"conclu'
    assert partial_string_field(buf, "conclusion") == ("", False)
    buf2 = '{"conclusion_prefix_not_a_real_field": "别读错", "conclusion": "真的"}'
    assert partial_string_field(buf2, "conclusion") == ("真的", True)


def test_field_before_its_value_arrives_reads_empty():
    for probe in ('{"conclusion"', '{"conclusion":', '{"conclusion": ', '{"conclusion": "灾'):
        val, closed = partial_string_field(probe, "conclusion")
        assert closed is False, probe
        assert val in ("", "灾"), (probe, val)


# ---- 要点（数组内对象的文本项）流式 ----
# 加这条是因为实测发现：79% 的答案结论是搬运来的、模型没写 conclusion，
# 只流 conclusion 等于对四分之三的题无事可做。流要点就把「会被引用强制丢弃」的东西
# 也提前显示给用户了——所以这里的下标正确性是安全性问题，不只是美观问题。

POINTS = json.dumps(
    {
        "conclusion": "",
        "key_points": [
            {"point": "灾备 RTO < 15 分钟", "evidence_ids": ["E1", "E2"]},
            {"point": '含"引号"与\反斜杠的要点', "evidence_ids": ["E1"]},
            {"point": "第 3 条要点：月结 T+3", "evidence_ids": ["E2"]},
        ],
        "no_evidence": False,
        "self_assessed_confidence": 0.7,
    },
    ensure_ascii=False,
)


def _final_points():
    return [p["point"] for p in json.loads(POINTS)["key_points"]]


def test_point_items_match_the_final_array_exactly_when_complete():
    from kee.qa.stream_json import partial_array_items

    assert [t for t, _ in partial_array_items(POINTS, "key_points", "point")] == _final_points()


def test_every_prefix_yields_only_prefixes_at_the_right_indexes():
    """每个前缀帧里，第 k 个要点的文本必须是最终第 k 个要点的**前缀**。

    下标错位（把第三条显示成第二条）在这一条下会当场失败——那比显示不全更糟。
    """
    from kee.qa.stream_json import partial_array_items

    finals = _final_points()
    for i in range(len(POINTS) + 1):
        items = partial_array_items(POINTS[:i], "key_points", "point")
        assert len(items) <= len(finals), f"第 {i} 帧凭空多出了要点"
        for k, (text, _closed) in enumerate(items):
            assert finals[k].startswith(text), f"第 {i} 帧第 {k} 条与最终内容矛盾：{text!r} vs {finals[k]!r}"


def test_index_is_taken_from_position_not_arrival_order():
    """`point` 在 `evidence_ids` 之后的对象里时，缺席中间项也不能让后面的一条顶上错号。"""
    from kee.qa.stream_json import partial_array_items

    buf = '{"key_points": [{"evidence_ids": ["E1"], "point": "第一条"}, {"point": "第二条"}]}'
    assert [t for t, _ in partial_array_items(buf, "key_points", "point")] == ["第一条", "第二条"]


def test_top_level_key_named_point_is_not_mixed_into_the_array():
    from kee.qa.stream_json import partial_array_items

    buf = '{"point": "这是顶层的同名键", "key_points": [{"point": "真要点"}]}'
    assert [t for t, _ in partial_array_items(buf, "key_points", "point")] == ["真要点"]


def test_numbers_and_booleans_are_consumed_so_indexes_do_not_drift():
    """标量必须被消费掉，否则数组下标会漏位——这里直接盯住这个后果。"""
    from kee.qa.stream_json import partial_array_items

    buf = '{"key_points": [{"point": "甲", "score": 0.91, "ok": true}, {"point": "乙"}]}'
    assert [t for t, _ in partial_array_items(buf, "key_points", "point")] == ["甲", "乙"]
