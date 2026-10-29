"""Phase 3：DOCX 表格保留原生结构，而不是只展平成文本。"""

from __future__ import annotations

import io

import docx

from kee.ingest.chunking import chunk_blocks
from kee.ingest.parsers import parse_docx


class _FakeMedia:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def begin_document(self) -> None:
        self.notes.clear()

    def docx_images(self, data: bytes, *, context: str = ""):
        return []


def test_docx_table_becomes_native_table_block() -> None:
    d = docx.Document()
    d.add_heading("测试表格", level=1)
    d.add_paragraph("这是一段足够长的正文，用来保证章节有正文段落。")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "部件"
    table.cell(0, 1).text = "状态"
    table.cell(1, 0).text = "主板"
    table.cell(1, 1).text = "正常"
    buf = io.BytesIO()
    d.save(buf)

    doc = parse_docx(buf.getvalue(), "table.docx", media=_FakeMedia())
    table_blocks = [b for b in doc.blocks if b.type == "table"]
    assert len(table_blocks) == 1
    block = table_blocks[0]
    assert block.source == "native_table"
    assert block.table is not None
    assert block.table.n_rows == 2 and block.table.n_cols == 2
    assert "| 部件 | 状态 |" in block.text
    by_pos = {(c.row, c.col): c.text for c in block.table.cells}
    assert by_pos[(1, 0)] == "主板"
    rows = chunk_blocks(doc.blocks, "doc@v1", "doc", doc.title)
    assert any("部件=主板" in r["content"] for r in rows if not r["is_parent"])
