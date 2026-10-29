"""Phase 2：block-v1 从 ContentBlock 直接分块，表格优先读 TableData。"""

from __future__ import annotations

from kee.config import Settings
from kee.ingest.chunking import attach_block_metadata, chunk_blocks, chunk_strategy_from_settings, chunk_document
from kee.ingest.content import ContentBlock, ParagraphMeta, TableCell, TableData
from kee.ingest.parsers import ParsedDocument, ParsedSection


def _doc() -> ParsedDocument:
    return ParsedDocument(
        title="block-v1 测试",
        source_format="text",
        sections=[
            ParsedSection(
                "1",
                "第一章 总则",
                1,
                [
                    "这是一段足够长的正文，用来验证 block-v1 的正文子块。",
                    "| 部件 | 状态 |\n|---|---|\n| 主板 | 正常 |\n| 风扇 | 异常 |",
                ],
                source="native_text",
                paragraph_meta=[
                    ParagraphMeta(type="text", source="native_text"),
                    ParagraphMeta(
                        type="table", source="mineru_table",
                        table=TableData(
                            n_rows=3,
                            n_cols=2,
                            cells=[
                                TableCell(0, 0, "部件"), TableCell(0, 1, "状态"),
                                TableCell(1, 0, "主板"), TableCell(1, 1, "正常"),
                                TableCell(2, 0, "风扇"), TableCell(2, 1, "异常"),
                            ],
                        ),
                    ),
                ],
            )
        ],
    ).normalize()


def test_block_v1_uses_table_json_for_row_chunks() -> None:
    doc = _doc()
    rows = chunk_blocks(doc.blocks, "doc@v1", "doc", doc.title)

    parents = [r for r in rows if r["is_parent"]]
    children = [r for r in rows if not r["is_parent"]]
    assert len(parents) == 1
    assert len(children) == 3  # 一段正文 + 两行表格
    table_rows = [r for r in children if r["block_type"] == "table"]
    assert [r["content"] for r in table_rows] == [
        "第一章 总则（表行）｜部件=主板；状态=正常",
        "第一章 总则（表行）｜部件=风扇；状态=异常",
    ]
    assert table_rows[0]["source"] == "mineru_table"
    assert table_rows[0]["block_id"] == "doc@v1#b0003"
    assert len({r["id"] for r in rows}) == len(rows)


def test_block_v1_default_remains_legacy() -> None:
    s = Settings(chunk_strategy="legacy")
    assert chunk_strategy_from_settings(s) == "legacy"
    s = Settings(chunk_strategy="block_v1")
    assert chunk_strategy_from_settings(s) == "block-v1"


def test_legacy_chunks_can_carry_block_metadata_without_changing_text_or_id() -> None:
    doc = _doc()
    card = {
        "doc_id": "doc", "title": doc.title,
        "sections": [
            {
                "section_no": s.section_no, "page": s.page, "heading": s.heading,
                "paragraphs": list(s.paragraphs),
            }
            for s in doc.sections
        ],
    }
    legacy = chunk_document(card, "doc@v1")
    before = [(r["id"], r["content"]) for r in legacy]
    attach_block_metadata(legacy, doc.blocks, "doc@v1")
    assert [(r["id"], r["content"]) for r in legacy] == before
    table_rows = [r for r in legacy if not r["is_parent"] and "（表行）" in r["content"]]
    assert table_rows and all(r["block_type"] == "table" for r in table_rows)
    assert all(r["block_id"].startswith("doc@v1#b") for r in table_rows)
