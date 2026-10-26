import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from docx import Document
from docx.shared import Inches

from kee.ingest.admission import admit
from kee.ingest.chunking import chunk_document
from kee.ingest.media import MediaPipeline
from kee.ingest.parsers import (
    ParsedDocument,
    ParsedSection,
    _append_pdf_images,
    parse_docx,
    parse_file,
    parse_image,
)


class FakeOcr:
    def det_boxes(self, bgr):
        return [object()]

    def crop(self, bgr, box):
        return bgr

    def rec_text(self, bgr):
        return "图内文字：交付金额 100 万元"


class FakeVlm:
    enabled = True
    disabled_reason = ""

    def describe_image(self, image, **kwargs):
        return "一张柱状图，展示四个季度的交付金额上升。"

    def summary(self):
        return {"enabled": True}


def _png_bytes(text: str = "KEE IMAGE", size=(400, 200)) -> bytes:
    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).text((20, 80), text, fill="black")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _docx_with_image() -> bytes:
    doc = Document()
    doc.add_heading("测试文档", level=1)
    doc.add_paragraph("这是一段足够长的正文，用于通过准入的段落长度校验。")
    doc.add_picture(io.BytesIO(_png_bytes()), width=Inches(2))
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def test_docx_embedded_image_becomes_indexable_paragraph():
    p = MediaPipeline(vlm=FakeVlm(), ocr=FakeOcr())
    doc = parse_docx(_docx_with_image(), "带图片.docx", media=p)

    assert any(s.heading == "图片与图表" for s in doc.sections)
    text = "\n".join(x for s in doc.sections for x in s.paragraphs)
    assert "图内文字" in text
    assert "内容描述" in text
    assert len(doc.figures) == 1
    assert doc.figures[0].kind == "image"
    assert "word/media/" in doc.figures[0].source_name


def test_standalone_image_is_a_document():
    p = MediaPipeline(vlm=FakeVlm(), ocr=FakeOcr())
    doc = parse_image(_png_bytes(), "流程图.png", media=p)
    assert doc.title == "流程图"
    assert len(doc.sections) == 1
    assert "【图片 1】" in doc.sections[0].paragraphs[0]
    assert "内容描述" in doc.sections[0].paragraphs[0]


def test_image_description_enters_chunking_and_admission():
    p = MediaPipeline(vlm=FakeVlm(), ocr=FakeOcr())
    doc = parse_file(filename="流程图.png", raw=_png_bytes(), media=p)
    card = doc.to_card(
        doc_id="img-demo",
        industry="制造",
        customer="某制造企业",
        project="PRJ-IMG-001",
        doc_type="复盘记录",
        source_level="B",
        security_level=2,
    )
    rep = admit([card])
    assert len(rep.accepted) == 1 and not rep.rejected
    rows = chunk_document(card, "img-demo@v1")
    child_text = "\n".join(r["content"] for r in rows if not r["is_parent"])
    assert "内容描述" in child_text
    assert any("图内文字" in r["content"] for r in rows if not r["is_parent"])


def test_pdf_embedded_images_are_appended_on_text_path():
    p = MediaPipeline(vlm=FakeVlm(), ocr=FakeOcr())
    doc = ParsedDocument(
        title="文字型 PDF",
        sections=[ParsedSection("1", "正文", 1, ["这是一段足够长的正文，用于通过准入的段落长度校验。"])],
        source_format="pdf",
    )
    raw = Path("var/m1/fixtures/表格_普通三列表.pdf").read_bytes()
    out = _append_pdf_images(doc, raw, "表格.pdf", p)
    assert any(s.section_no.startswith("IMG-") for s in out.sections)
    assert out.figures
    assert "内容描述" in "\n".join(x for s in out.sections for x in s.paragraphs)


def test_pdf_images_extraction_returns_paragraphs():
    p = MediaPipeline(vlm=FakeVlm(), ocr=FakeOcr())
    raw = Path("var/m1/fixtures/表格_普通三列表.pdf").read_bytes()
    items = p.pdf_images(raw, context="表格")
    assert items
    assert "图内文字" in items[0].paragraph()
    assert "内容描述" in items[0].paragraph()
