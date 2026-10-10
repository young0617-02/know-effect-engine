"""表格结构还原的纯函数单测：不加载模型，只验网格→Markdown 的几何与文本落位。"""

from __future__ import annotations

import numpy as np
import pytest

from kee.ingest.chunking import chunk_document
from kee.ingest.mineru_onnx import (
    Box,
    TableStructureModel,
    _quad_to_box,
    table_span_boxes,
    table_to_markdown,
)
from kee.ingest.parsers import _clean, is_table_block, split_table_block


def cell(x1, y1, x2, y2, label="cell"):
    return Box(x1, y1, x2, y2, 1.0, label)


def line(x1, y1, x2, y2, text):
    return Box(x1, y1, x2, y2, 1.0, "line", text)


def test_grid_becomes_markdown_with_header_separator():
    rows = [
        [cell(0, 0, 100, 30), cell(100, 0, 200, 30)],
        [cell(0, 30, 100, 60), cell(100, 30, 200, 60)],
    ]
    lines = [
        line(2, 8, 40, 22, "部件"),
        line(105, 8, 150, 22, "铅"),
        line(2, 38, 40, 52, "主板"),
        line(105, 38, 150, 52, "超标"),
    ]
    md = table_to_markdown(rows, [], lines)
    assert md.split("\n") == [
        "| 部件 | 铅 |",
        "|---|---|",
        "| 主板 | 超标 |",
    ]


def test_merged_cell_anchors_at_leftmost_covered_column():
    rows = [
        [cell(0, 0, 60, 30)],
        [cell(0, 30, 60, 60), cell(60, 30, 120, 60), cell(120, 30, 180, 60)],
    ]
    spans = [cell(60, 0, 180, 30, label="span")]
    lines = [
        line(2, 8, 40, 22, "型号"),
        line(90, 8, 150, 22, "有害物质"),
        line(2, 38, 40, 52, "主板"),
        line(70, 38, 110, 52, "铅"),
        line(130, 38, 170, 52, "汞"),
    ]
    md = table_to_markdown(rows, spans, lines)
    body = md.split("\n")
    assert body[0] == "| 型号 | 有害物质 |  |"  # 合并格落在最左覆盖列，其余列留空而不是丢掉
    assert body[2] == "| 主板 | 铅 | 汞 |"


def test_multiline_cjk_cell_joins_without_space_but_latin_keeps_it():
    rows = [[cell(0, 0, 100, 60), cell(100, 0, 200, 60)], [cell(0, 60, 100, 90), cell(100, 60, 200, 90)]]
    lines = [
        line(2, 4, 90, 24, "多溴联"),
        line(2, 28, 90, 48, "苯"),
        line(102, 4, 190, 24, "Poly"),
        line(102, 28, 190, 48, "Brominated"),
        line(2, 64, 90, 84, "x"),
        line(102, 64, 190, 84, "y"),
    ]
    md = table_to_markdown(rows, [], lines)
    assert "多溴联苯" in md  # 中文换行不是词边界，插空格会让 jieba 分词出错
    assert "Poly Brominated" in md


def test_empty_cell_does_not_steal_a_far_line():
    rows = [[cell(0, 0, 100, 30), cell(100, 0, 200, 30)], [cell(0, 30, 100, 60), cell(100, 30, 200, 60)]]
    lines = [line(2, 38, 40, 52, "只有左下角有字"), line(102, 8, 150, 22, "右上")]
    md = table_to_markdown(rows, [], lines)
    assert md.split("\n")[0] == "|  | 右上 |"
    assert md.split("\n")[2] == "| 只有左下角有字 |  |"


def test_single_column_grid_is_not_reported_as_table():
    rows = [[cell(0, 0, 100, 30)], [cell(0, 30, 100, 60)]]
    assert table_to_markdown(rows, [], [line(2, 8, 40, 22, "一行字")]) == ""


def test_span_candidates_reject_quads_that_overlap_cells():
    # 4 点序 (x1,y1,x2,y2,x3,y3,x4,y4)，像素坐标 × 归一化到 (W,H)=(200,90)
    quads = [(0, 0, 100, 30), (100, 0, 200, 30), (0, 30, 100, 60), (100, 0, 200, 60), (100, 60, 200, 90)]
    ids = np.array([48, 48, 48, 7, 8], dtype="int64")
    loc = np.zeros((len(quads), 8), dtype="float32")
    for i, (x1, y1, x2, y2) in enumerate(quads):
        loc[i] = [x1 / 200, y1 / 90, x2 / 200, y1 / 90, x2 / 200, y2 / 90, x1 / 200, y2 / 90]
    cells = [_quad_to_box(loc[i], 200.0, 90.0) for i in range(3)]
    spans = table_span_boxes(loc, ids, cells, W=200.0, H=90.0)
    assert len(spans) == 1  # id=7 与 cell 大面积重叠 → 无效坐标；id=8 在 cell 之外 → 合并格
    assert (spans[0].x1, spans[0].y1, spans[0].x2, spans[0].y2) == pytest.approx((100, 60, 200, 90))
    assert spans[0].label == "span"


def test_skewed_quad_snaps_to_consensus_column_and_keeps_its_text():
    """复现真实缺陷：宽表首行首格的 quad 右边界画到 393（真值 301）、下边界短一截，
    导致「区域」被隔壁格抢走。吸附到列共识边界后必须回到本格。"""
    rows = [
        [cell(8, 2, 393, 28), cell(291, 1, 444, 41), cell(430, 2, 590, 45)],
        [cell(4, 57, 301, 105), cell(290, 59, 440, 106), cell(423, 58, 580, 106)],
        [cell(5, 105, 298, 154), cell(290, 107, 439, 153), cell(423, 107, 578, 153)],
    ]
    lines = [
        line(120, 24, 180, 40, "区域"),
        line(120, 76, 180, 92, "华东"),
        line(330, 24, 400, 40, "年产量"),
        line(330, 76, 400, 92, "1200 台"),
        line(470, 24, 550, 40, "单位能耗"),
        line(470, 76, 550, 92, "8.4"),
        line(120, 124, 180, 140, "华南"),
        line(330, 124, 400, 140, "980 台"),
        line(470, 124, 550, 140, "9.1"),
    ]
    md = table_to_markdown(rows, [], lines)
    body = md.split("\n")
    assert body[0] == "| 区域 | 年产量 | 单位能耗 |"
    assert body[2] == "| 华东 | 1200 台 | 8.4 |"


def test_clean_folds_prose_wraps_but_keeps_table_rows():
    assert _clean("第一段\n还是第一段\n\n第二段") == "第一段 还是第一段\n\n第二段"
    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert _clean(f"说明文字\n{md}") == f"说明文字\n{md}"


def test_table_block_detection_and_split_repeat_header():
    md = "\n".join(["| 列1 | 列2 |", "|---|---|"] + [f"| 值{i} | 备{i} |" for i in range(6)])
    assert is_table_block(md)
    assert not is_table_block("普通段落\n第二行")
    pieces = split_table_block(md, 24)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("| 列1 | 列2 |\n|---|---|\n")
        assert is_table_block(piece)


def test_chunking_keeps_table_in_its_own_child_with_header():
    md = "\n".join(
        ["| 部件 | 铅含量判定 |", "|---|---|"]
        + [f"| 组件框架单元{i} | 超出限值{i} 微克 |" for i in range(12)]
    )
    assert len(md) > 280  # 必须超过子块长度，否则测不到切分
    card = {
        "doc_id": "d1",
        "title": "测试文档",
        "sections": [{"section_no": "1", "page": 1, "heading": "有害物质表", "paragraphs": [md]}],
    }
    rows = chunk_document(card, "d1@v1")
    children = [r for r in rows if not r["is_parent"]]
    assert len(children) > 1
    for c in children:
        head = c["content"].split("\n")
        assert head[0] == "有害物质表"  # 章节标题独占一行，不能把表头行挤下去
        assert head[1] == "| 部件 | 铅含量判定 |"
        assert c["char_count"] <= 280 * 2


def test_table_structure_model_reports_missing_weight(tmp_path, monkeypatch):
    monkeypatch.setattr("kee.ingest.mineru_onnx.model_root", lambda: tmp_path)
    assert TableStructureModel.available() is False
