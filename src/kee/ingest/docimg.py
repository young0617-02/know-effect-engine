"""PDF 页面渲染：pypdfium2 出位图，交给版面/OCR 模型。

不用 PyMuPDF：它是 AGPL，而本项目要对外展示；pypdfium2 是 Apache/BSD 系且只有 3.8MB。
"""

from __future__ import annotations

import io
from typing import Iterator

import numpy as np


def render_pdf_pages(data: bytes, dpi: int = 150, max_pages: int = 60) -> Iterator[tuple[int, np.ndarray]]:
    """逐页产出 (页码从 1 开始, HWC 的 BGR uint8 位图)。"""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    try:
        n = min(len(doc), max_pages)
        for i in range(n):
            page = doc[i]
            bitmap = page.render(scale=dpi / 72.0)
            rgb = bitmap.to_numpy()  # RGBA
            bgr = rgb[:, :, [2, 1, 0]].copy()  # 模型按 BGR 训练
            yield i + 1, bgr
    finally:
        doc.close()


def page_count(data: bytes) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    try:
        return len(doc)
    finally:
        doc.close()


def image_to_pdf_bgr(img_path: str) -> tuple[bytes, np.ndarray]:
    """给测试用的夹具：把位图包成单页 PDF（模拟扫描件）。"""
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PDF")
    return buf.getvalue(), np.asarray(im)[:, :, ::-1].copy()
