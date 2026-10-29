"""Phase 1：ContentBlock 与 sections 文本等价、来源标记、持久化参数。

不连数据库：`insert_blocks` 用假连接检查 SQL 参数与 JSON 序列化；
真实 schema 由 `pg.init_schema()` 建表，后续 M1 写路径会走同一方法。
"""

from __future__ import annotations

import json

from kee.db.pg import AuthoritativeStore
from kee.ingest.content import (
    ContentBlock,
    ParagraphMeta,
    TableCell,
    TableData,
    blocks_to_markdown,
    blocks_to_text,
    count_blocks,
)
from kee.ingest.mineru_onnx import Box, MinerUPdfParser
from kee.ingest.parsers import ParsedDocument, ParsedSection


def _document() -> ParsedDocument:
    doc = ParsedDocument(
        title="结构块测试",
        source_format="text",
        sections=[
            ParsedSection(
                "1",
                "第一章 总则",
                None,
                ["这是一段足够长的正文，用来验证 blocks 与 sections 的文本完全一致。"],
                source="native_text",
                paragraph_meta=[ParagraphMeta(type="text", source="native_text")],
            ),
            ParsedSection(
                "2",
                "第 1 页图片与图表",
                1,
                ["【图片 1】图内文字：现场照片中的设备铭牌。"],
                source="mineru_ocr",
                paragraph_meta=[ParagraphMeta(type="figure", source="mineru_ocr")],
            ),
        ],
    )
    return doc.normalize()


def test_blocks_roundtrip_text_and_have_provenance() -> None:
    doc = _document()
    expected = "\n".join(
        part for section in doc.sections for part in [section.heading, *section.paragraphs]
    )

    assert blocks_to_text(doc.blocks) == expected
    assert [(b.type, b.source) for b in doc.blocks] == [
        ("heading", "native_text"),
        ("text", "native_text"),
        ("heading", "mineru_ocr"),
        ("figure", "mineru_ocr"),
    ]
    assert "## 第一章 总则" in blocks_to_markdown(doc.blocks)
    assert count_blocks(doc.blocks)["by_source"] == {"native_text": 2, "mineru_ocr": 2}


def test_mineru_blocks_carry_type_source_and_bbox() -> None:
    table = TableData(
        n_rows=2, n_cols=2,
        cells=[TableCell(0, 0, "列A"), TableCell(0, 1, "列B"),
               TableCell(1, 0, "1"), TableCell(1, 1, "2")],
    )
    pages = [(1, [
        (Box(0, 0, 100, 20, 0.9, "paragraph_title", "标题"), "paragraph_title"),
        (Box(0, 30, 200, 50, 0.9, "text", "这是一段足够长的正文，用来验证来源与类型。"), "text"),
        (Box(0, 60, 200, 100, 0.9, "table", "| 列A | 列B |\n|---|---|\n| 1 | 2 |", table=table), "table"),
        (Box(0, 110, 200, 150, 0.9, "formula", "E=mc^2"), "formula"),
        (Box(0, 160, 200, 200, 0.9, "figure", "【图表 第1页】图内文字：Q1 10"), "figure"),
    ])]
    doc = MinerUPdfParser._to_document(object(), pages, "论文.pdf", [])

    by_type = {b.type: b for b in doc.blocks}
    assert by_type["heading"].source == "mineru_ocr"
    assert by_type["text"].source == "mineru_ocr"
    assert by_type["table"].source == "mineru_table"
    assert by_type["formula"].source == "mineru_formula"
    assert by_type["figure"].source == "mineru_figure"
    assert by_type["table"].bbox == (0, 60, 200, 100)
    assert by_type["table"].coord_space == "rendered_px"
    assert by_type["table"].table is not None
    assert by_type["table"].table.n_rows == 2
    assert blocks_to_text(doc.blocks) == "\n".join(
        part for section in doc.sections for part in [section.heading, *section.paragraphs]
    )


class _FakeCursor:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def executemany(self, sql: str, rows: list[dict]) -> None:
        self.sql = sql
        self.rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    def __init__(self) -> None:
        self.cur = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cur

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeStore(AuthoritativeStore):
    def __init__(self) -> None:
        self.conn = _FakeConn()

    def connect(self) -> _FakeConn:
        return self.conn


def test_insert_blocks_serializes_source_bbox_and_marks_table_json_absent() -> None:
    store = _FakeStore()
    block = ContentBlock(
        block_id="b0001",
        page=1,
        order=1,
        type="table",
        source="mineru_table",
        text="| A | B |",
        markdown="| A | B |",
        bbox=(1.0, 2.0, 3.0, 4.0),
        coord_space="rendered_px",
        table=TableData(n_rows=1, n_cols=2, cells=[TableCell(0, 0, "A"), TableCell(0, 1, "B")]),
    )

    n = store.insert_blocks("v1", "doc-1", [block])

    assert n == 1
    row = store.conn.cur.rows[0]
    assert row["id"] == "v1#b0001"
    assert row["source"] == "mineru_table"
    assert json.loads(row["bbox"]) == [1.0, 2.0, 3.0, 4.0]
    table = json.loads(row["table_json"])
    assert table["n_rows"] == 1 and table["cells"][0]["text"] == "A"
