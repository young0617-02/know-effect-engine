"""解析层的统一内容块：结构真相在这里，Markdown 只是渲染视图。

Phase 1 只做“结构落地”，不改现有分块/索引：
  - 每个段落携带 `ParagraphMeta`（类型、来源、bbox、坐标空间），
    `ParsedSection` 保留原 paragraphs，归一化时元数据与文本一起变换；
  - `ParsedDocument.normalize()` 之后由 `blocks_from_sections()` 生成 `ContentBlock`；
  - `blocks_to_text()` 与 sections 的文本逐字一致，用于证明 Phase 1 没有改变解析输出；
  - `blocks_to_markdown()` 供后续渲染/导出使用，不参与当前分块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

BLOCK_TYPES = {"title", "heading", "text", "table", "formula", "figure", "image"}
BLOCK_SOURCES = {
    "native_text",
    "mineru_ocr",
    "mineru_table",
    "mineru_formula",
    "mineru_figure",
    "native_table",
    "legacy",
}


@dataclass
class TableCell:
    """表格单元格的权威结构；bbox 与 ContentBlock 同一坐标空间。"""

    row: int
    col: int
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    bbox: tuple[float, float, float, float] | None = None
    is_span: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "bbox": list(self.bbox) if self.bbox else None,
            "is_span": self.is_span,
        }


@dataclass
class TableData:
    n_rows: int
    n_cols: int
    cells: list[TableCell] = field(default_factory=list)
    markdown: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "cells": [c.as_dict() for c in self.cells],
            "markdown": self.markdown,
        }


@dataclass
class ParagraphMeta:
    """一个段落的结构元数据；与 `ParsedSection.paragraphs` 一一对应。"""

    type: str = "text"
    source: str = "native_text"
    bbox: tuple[float, float, float, float] | None = None
    coord_space: str | None = None
    table: TableData | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "bbox": list(self.bbox) if self.bbox else None,
            "coord_space": self.coord_space,
            "table": self.table.as_dict() if self.table else None,
        }


@dataclass
class ContentBlock:
    block_id: str
    page: int | None
    order: int
    type: str
    source: str
    text: str
    markdown: str = ""
    section_no: str = ""
    bbox: tuple[float, float, float, float] | None = None
    coord_space: str | None = None
    table: TableData | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "order": self.order,
            "type": self.type,
            "source": self.source,
            "text": self.text,
            "markdown": self.markdown,
            "section_no": self.section_no,
            "bbox": list(self.bbox) if self.bbox else None,
            "coord_space": self.coord_space,
            "table": self.table.as_dict() if self.table else None,
        }


def _default_meta(section: Any) -> ParagraphMeta:
    return ParagraphMeta(type="text", source=getattr(section, "source", "native_text"))


def blocks_from_sections(sections: Iterable[Any]) -> list[ContentBlock]:
    """把归一化后的 sections 转成带类型/来源的块列表。

    标题块取 section.source；段落块优先取 `paragraph_meta`，缺失时退化为
    text + section.source。不修改 sections，也不依赖 Markdown 反推类型。
    """
    blocks: list[ContentBlock] = []
    order = 0
    for section in sections:
        order += 1
        heading = str(getattr(section, "heading", "") or "")
        source = str(getattr(section, "source", "native_text") or "native_text")
        if heading:
            blocks.append(
                ContentBlock(
                    block_id=f"b{order:04d}",
                    page=section.page,
                    order=order,
                    type="heading",
                    source=source,
                    text=heading,
                    markdown=heading,
                    section_no=str(section.section_no),
                )
            )
        paragraphs = list(getattr(section, "paragraphs", []) or [])
        metas = list(getattr(section, "paragraph_meta", []) or [])
        if len(metas) != len(paragraphs):
            metas = [_default_meta(section) for _ in paragraphs]
        for paragraph, meta in zip(paragraphs, metas):
            order += 1
            text = str(paragraph)
            blocks.append(
                ContentBlock(
                    block_id=f"b{order:04d}",
                    page=section.page,
                    order=order,
                    type=meta.type,
                    source=meta.source,
                    text=text,
                    markdown=text,
                    section_no=str(section.section_no),
                    bbox=meta.bbox,
                    coord_space=meta.coord_space,
                    table=meta.table,
                )
            )
    return blocks


def blocks_to_text(blocks: Iterable[ContentBlock]) -> str:
    """按块序拼回纯文本；与 sections 的 heading + paragraphs 拼接结果一致。"""
    return "\n".join(block.text for block in blocks if block.text)


def blocks_to_markdown(blocks: Iterable[ContentBlock]) -> str:
    """面向人/LLM 的可读渲染；Phase 1 不把它作为分块输入。"""
    parts: list[str] = []
    for block in blocks:
        if not block.text:
            continue
        if block.type == "title":
            parts.append(f"# {block.markdown or block.text}")
        elif block.type == "heading":
            parts.append(f"## {block.markdown or block.text}")
        else:
            parts.append(block.markdown or block.text)
    return "\n\n".join(parts)


def count_blocks(blocks: Iterable[ContentBlock]) -> dict[str, dict[str, int]]:
    """审计用：按来源/类型计数，不携带正文。"""
    by_source: dict[str, int] = {}
    by_type: dict[str, int] = {}
    total = 0
    for block in blocks:
        total += 1
        by_source[block.source] = by_source.get(block.source, 0) + 1
        by_type[block.type] = by_type.get(block.type, 0) + 1
    return {"total": {"all": total}, "by_source": by_source, "by_type": by_type}
