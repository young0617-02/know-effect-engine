"""按页路由：文本页保留 pypdf，文本不足的页只在该页跑 MinerU。

这些用例不加载任何 ONNX 模型：用假 PdfReader 提供逐页文本，用假 MinerU 返回
「该页 OCR 结果」，因此可以在没有 GPU/存储栈的环境里把路由决策、页序、回退
和统计口径钉住。
"""

from __future__ import annotations

import io

import pypdf
import pytest

from kee.config import Settings
from kee.ingest import mineru_onnx
from kee.ingest.parsers import ParsedDocument, ParsedSection, parse_pdf


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, texts: list[str]) -> None:
        self.pages = [_FakePage(t) for t in texts]


class _FakeMedia:
    """替代 MediaPipeline：记录哪些页被交给内嵌图片 OCR。"""

    def __init__(self) -> None:
        self.notes: list[str] = []
        self.calls: list[set[int] | None] = []

    def begin_document(self) -> None:
        self.notes.clear()

    def pdf_images(self, data: bytes, *, context: str = "", pages: set[int] | None = None):
        self.calls.append(set(pages) if pages is not None else None)
        return []


def _install_reader(monkeypatch: pytest.MonkeyPatch, texts: list[str]) -> None:
    reader = _FakeReader(texts)
    monkeypatch.setattr(pypdf, "PdfReader", lambda *args, **kwargs: reader)


@pytest.fixture()
def page_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(
        pdf_parse_backend="auto",
        pdf_page_routing=True,
        pdf_min_chars_per_page=40,
        media_scan_text_pdf=True,
    )
    monkeypatch.setattr("kee.config.get_settings", lambda: s)
    return s


@pytest.fixture()
def fake_mineru(monkeypatch: pytest.MonkeyPatch) -> list[set[int] | None]:
    calls: list[set[int] | None] = []

    class FakeMinerU:
        def parse_pages(self, data: bytes, filename: str, page_numbers=None):
            calls.append(set(page_numbers) if page_numbers is not None else None)
            nums = sorted(page_numbers or [])
            sections = [
                ParsedSection(
                    section_no=str(i),
                    heading=f"OCR页{p}",
                    page=p,
                    paragraphs=[f"第{p}页扫描文字足够长，可以被分块" * 2],
                )
                for i, p in enumerate(nums, start=1)
            ]
            return ParsedDocument(
                title="OCR标题",
                sections=sections,
                source_format="pdf",
                notes=["fake MinerU"],
            )

        def parse(self, data: bytes, filename: str):
            calls.append(None)
            return self.parse_pages(data, filename, page_numbers={1})

    monkeypatch.setattr(mineru_onnx, "MinerUPdfParser", FakeMinerU)
    return calls


def _mixed_texts() -> list[str]:
    return [
        "1 总则\n" + "正文" * 80,   # 第 1 页文本充足
        "",                            # 第 2 页扫描/图片
        "2 附录\n" + "正文" * 80,   # 第 3 页文本充足
    ]


def test_page_routing_routes_only_the_thin_page(
    monkeypatch: pytest.MonkeyPatch,
    page_settings: Settings,
    fake_mineru: list[set[int] | None],
) -> None:
    texts = _mixed_texts()
    _install_reader(monkeypatch, texts)
    media = _FakeMedia()

    doc = parse_pdf(b"fake", "混合文档.pdf", media=media)

    assert fake_mineru == [{2}], "只有第 2 页应交给 MinerU"
    assert media.calls == [{1, 3}], "内嵌位图 OCR 只应处理文本页，不能重复处理 OCR 页"
    assert {s.page for s in doc.sections} == {1, 2, 3}
    assert any("第2页扫描文字" in p for s in doc.sections for p in s.paragraphs)
    assert any("按页路由：文本层 2 页，MinerU-ONNX 1 页" in n for n in doc.notes)
    assert "OCR 页 [2]" in "\n".join(doc.notes)
    assert doc.title == "1 总则"


def test_page_routing_is_per_page_not_document_average(
    monkeypatch: pytest.MonkeyPatch,
    page_settings: Settings,
    fake_mineru: list[set[int] | None],
) -> None:
    texts = _mixed_texts()
    # 历史文档级口径会算成 (100+0+100)/3 ≈ 66 > 40，整份都不会转 OCR；
    # 按页路由必须仍然把第 2 页送进 MinerU。
    assert sum(len(t) for t in texts) / len(texts) > page_settings.pdf_min_chars_per_page
    _install_reader(monkeypatch, texts)

    parse_pdf(b"fake", "混合文档.pdf", media=_FakeMedia())

    assert fake_mineru == [{2}]


def test_all_text_pages_skip_mineru(
    monkeypatch: pytest.MonkeyPatch,
    page_settings: Settings,
    fake_mineru: list[set[int] | None],
) -> None:
    _install_reader(monkeypatch, ["1 总则\n" + "正文" * 80, "2 附录\n" + "正文" * 80])
    media = _FakeMedia()

    doc = parse_pdf(b"fake", "纯文本.pdf", media=media)

    assert fake_mineru == []
    assert media.calls == [{1, 2}]
    assert any("按页路由：文本层 2 页，MinerU-ONNX 0 页" in n for n in doc.notes)


def test_all_ocr_pages_use_mineru_without_image_double_processing(
    monkeypatch: pytest.MonkeyPatch,
    page_settings: Settings,
    fake_mineru: list[set[int] | None],
) -> None:
    _install_reader(monkeypatch, ["", ""])
    media = _FakeMedia()

    doc = parse_pdf(b"fake", "扫描件.pdf", media=media)

    assert fake_mineru == [{1, 2}]
    assert media.calls == [], "整份都交给 MinerU 时不应再跑内嵌位图 OCR"
    assert {s.page for s in doc.sections} == {1, 2}
    assert any("按页路由：文本层 0 页，MinerU-ONNX 2 页" in n for n in doc.notes)


def test_page_routing_can_be_disabled_for_legacy_document_gate(
    monkeypatch: pytest.MonkeyPatch,
    fake_mineru: list[set[int] | None],
) -> None:
    s = Settings(
        pdf_parse_backend="auto",
        pdf_page_routing=False,
        pdf_min_chars_per_page=40,
        media_scan_text_pdf=True,
    )
    monkeypatch.setattr("kee.config.get_settings", lambda: s)
    _install_reader(monkeypatch, _mixed_texts())
    media = _FakeMedia()

    doc = parse_pdf(b"fake", "混合文档.pdf", media=media)

    # 文档级平均 66 > 40 ⇒ 不触发整份 MinerU；保留旧口径用于回归对照。
    assert fake_mineru == []
    assert media.calls == [None]
    assert not any("按页路由" in n for n in doc.notes)


def test_page_routing_falls_back_to_text_and_images_when_mineru_fails(
    monkeypatch: pytest.MonkeyPatch,
    page_settings: Settings,
) -> None:
    class BrokenMinerU:
        def parse_pages(self, data: bytes, filename: str, page_numbers=None):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(mineru_onnx, "MinerUPdfParser", BrokenMinerU)
    _install_reader(monkeypatch, _mixed_texts())
    media = _FakeMedia()

    doc = parse_pdf(b"fake", "混合文档.pdf", media=media)

    assert media.calls == [None], "页级 MinerU 失败时应对全文补内嵌位图 OCR"
    assert any("页级 MinerU-ONNX 不可用" in n for n in doc.notes)
    assert {s.page for s in doc.sections} == {1, 3}


def test_page_routing_falls_back_per_page_when_one_ocr_page_has_no_sections(
    monkeypatch: pytest.MonkeyPatch,
    page_settings: Settings,
) -> None:
    class PartialMinerU:
        def parse_pages(self, data: bytes, filename: str, page_numbers=None):
            # 第 1 页有 OCR 结果，第 2 页没产出章节（模拟空白/单页失败）
            return ParsedDocument(
                title="OCR标题",
                sections=[ParsedSection("1", "OCR页1", 1, ["第1页扫描文字足够长，可以被分块"] )],
                source_format="pdf",
                notes=["fake partial MinerU"],
            )

    monkeypatch.setattr(mineru_onnx, "MinerUPdfParser", PartialMinerU)
    _install_reader(monkeypatch, ["", "第2页文本", "正文" * 80])
    media = _FakeMedia()

    doc = parse_pdf(b"fake", "部分失败.pdf", media=media)

    assert any("未覆盖页 [2]" in n for n in doc.notes)
    assert doc.title == "第2页文本"
    assert media.calls == [{2, 3}]


def test_render_pdf_pages_can_render_only_selected_page() -> None:
    from kee.ingest.docimg import render_pdf_pages

    fixture = "var/m1/fixtures/扫描件_产品安全手册第1页.pdf"
    try:
        src = pypdf.PdfReader(fixture)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"缺扫描件 fixture：{type(e).__name__}: {e}")
    writer = pypdf.PdfWriter()
    writer.add_page(src.pages[0])
    writer.add_page(src.pages[0])
    buf = io.BytesIO()
    writer.write(buf)

    rendered = list(render_pdf_pages(buf.getvalue(), dpi=72, pages={2}))

    assert [pno for pno, _ in rendered] == [2]
    assert rendered[0][1].ndim == 3
