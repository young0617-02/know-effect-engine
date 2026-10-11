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
