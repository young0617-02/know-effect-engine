"""图片/图表入库辅助：抽取 → 本地 OCR → 可检索段落。

这里只做「把图片里的文字变成文本」；不调用外部视觉模型，也不做图表语义理解。
识别出的文字会像普通段落一样进 Parent-Child 分块，被 BGE-M3 嵌入、被 BM25 命中。
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from kee.config import Settings, get_settings

log = logging.getLogger("kee.ingest.media")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
_KIND_LABEL = {"chart": "图表", "figure": "插图", "image": "图片", "scanned_page": "扫描页", "formula": "公式"}


@dataclass
class MediaText:
    kind: str
    index: int
    ocr_text: str = ""
    page: int | None = None
    source_name: str = ""

    def paragraph(self) -> str:
        label = _KIND_LABEL.get(self.kind, "图片")
        text = self.ocr_text.strip()
        if not text:
            return ""
        return f"【{label} {self.index}】图内文字：{text}"


class MediaPipeline:
    """按需装载本地 PP-OCR；没有 OCR 模型时跳过图片并写 note，不伪造内容。"""

    def __init__(self, settings: Settings | None = None, *, ocr: Any | None = None) -> None:
        self.s = settings or get_settings()
        self._ocr = ocr
        self._ocr_error = ""
        self.notes: list[str] = []

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "MediaPipeline":
        return cls(settings=settings)

    def begin_document(self) -> None:
        """同一服务进程会连续摄取多篇文档；notes 必须按文档隔离，不能串到下一条审计里。"""
        self.notes.clear()

    @property
    def ocr(self) -> Any | None:
        if self._ocr is not None:
            return self._ocr
        if self._ocr_error:
            return None
        try:
            from kee.ingest.mineru_onnx import OcrEngine

            self._ocr = OcrEngine()
        except Exception as exc:  # noqa: BLE001 - 没有 OCR 模型时跳过图片
            self._ocr_error = f"{type(exc).__name__}: {exc}"[:200]
            self.notes.append(f"OCR 模型不可用：{self._ocr_error}")
            log.warning("OCR 模型不可用：%s", type(exc).__name__)
            return None
        return self._ocr

    def ocr_bgr(self, bgr: np.ndarray) -> str:
        eng = self.ocr
        if eng is None:
            return ""
        try:
            boxes = eng.det_boxes(bgr)
            texts = []
            for b in boxes:
                text = eng.rec_text(eng.crop(bgr, b))
                if text.strip():
                    texts.append(text.strip())
            return " ".join(texts).strip()
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"图片 OCR 失败：{type(exc).__name__}: {exc}"[:200])
            log.warning("图片 OCR 失败：%s", type(exc).__name__)
            return ""

    def process_bgr(
        self,
        bgr: np.ndarray,
        *,
        kind: str = "image",
        index: int = 1,
        page: int | None = None,
        context: str = "",
        source_name: str = "",
    ) -> MediaText:
        if bgr.size == 0:
            return MediaText(kind=kind, index=index, page=page, source_name=source_name)
        if bgr.shape[0] * bgr.shape[1] < max(1, self.s.media_min_image_pixels):
            return MediaText(kind=kind, index=index, page=page, source_name=source_name)
        return MediaText(
            kind=kind,
            index=index,
            ocr_text=self.ocr_bgr(bgr),
            page=page,
            source_name=source_name,
        )

    def process_image_bytes(
        self,
        data: bytes,
        *,
        kind: str = "image",
        index: int = 1,
        page: int | None = None,
        context: str = "",
        source_name: str = "",
    ) -> MediaText:
        try:
            bgr = image_bytes_to_bgr(data)
        except Exception as exc:  # noqa: BLE001 - 坏图不能拖垮整篇文档
            self.notes.append(f"图片解码失败（{source_name or index}）：{type(exc).__name__}: {exc}"[:200])
            return MediaText(kind=kind, index=index, page=page, source_name=source_name)
        return self.process_bgr(
            bgr, kind=kind, index=index, page=page, context=context, source_name=source_name
        )

    @staticmethod
    def extract_docx_images(data: bytes) -> list[tuple[str, bytes]]:
        """从 docx 的 zip 包中抽 `word/media/*`；按包内顺序返回。"""
        out: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(BytesIO(data)) as zf:
            for name in zf.namelist():
                if not name.startswith("word/media/") or name.endswith("/"):
                    continue
                if Path(name).suffix.lower() not in _IMAGE_EXTS:
                    continue
                try:
                    out.append((name, zf.read(name)))
                except KeyError:
                    continue
        return out

    def pdf_images(
        self, data: bytes, *, context: str = "", pages: set[int] | None = None
    ) -> list[MediaText]:
        """抽取 PDF 内嵌图片（文本层路径用）；扫描页整页图在 MinerU 路径里处理。

        `pages` 给定时只处理这些页（1-based），按页路由用它把 OCR 限制在文本页，
        避免对已经交给 MinerU 的页重复识别同一张图。
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"PDF 内嵌图片抽取失败：{type(exc).__name__}: {exc}"[:200])
            return []
        limit = max(0, self.s.media_max_images_per_doc)
        out: list[MediaText] = []
        total = 0
        for pno, page in enumerate(reader.pages, start=1):
            if pages is not None and pno not in pages:
                continue
            try:
                images = list(page.images)
            except Exception as exc:  # noqa: BLE001 - 某些 PDF 的图片对象损坏
                self.notes.append(f"第 {pno} 页图片枚举失败：{type(exc).__name__}"[:120])
                continue
            for img in images:
                total += 1
                if len(out) >= limit:
                    continue
                try:
                    raw = img.data
                    name = getattr(img, "name", f"page{pno}-image{len(out) + 1}")
                except Exception:  # noqa: BLE001
                    continue
                mt = self.process_image_bytes(
                    raw, kind="image", index=len(out) + 1, page=pno,
                    context=context, source_name=str(name),
                )
                if mt.paragraph():
                    out.append(mt)
        if total > len(out):
            self.notes.append(f"PDF 内嵌图片 {total} 张，按上限/可读性处理 {len(out)} 张")
        if total:
            self.notes.append(f"PDF 内嵌图片：枚举 {total} 张，产出文字 {len(out)} 张")
        return out

    def docx_images(self, data: bytes, *, context: str = "") -> list[MediaText]:
        """抽取并处理 DOCX 内嵌图片；超过上限的图片丢弃并写 note。"""
        try:
            images = self.extract_docx_images(data)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"DOCX 图片抽取失败：{type(exc).__name__}: {exc}"[:200])
            return []
        if not images:
            return []
        limit = max(0, self.s.media_max_images_per_doc)
        kept = images[:limit]
        if len(images) > len(kept):
            self.notes.append(f"DOCX 内嵌图片 {len(images)} 张，按上限只处理前 {len(kept)} 张")
        results: list[MediaText] = []
        for i, (name, raw) in enumerate(kept, start=1):
            mt = self.process_image_bytes(
                raw, kind="image", index=i, page=None, context=context, source_name=name
            )
            if mt.paragraph():
                results.append(mt)
        self.notes.append(
            f"DOCX 内嵌图片：抽取 {len(images)} 张，处理 {len(kept)} 张，产出文字 {len(results)} 张"
        )
        return results

    def summary(self) -> dict[str, Any]:
        return {
            "ocr_available": self.ocr is not None,
            "ocr_error": self._ocr_error,
            "notes": list(self.notes),
        }


def image_bytes_to_bgr(data: bytes) -> np.ndarray:
    """任意常见位图 → HWC BGR uint8；EMF/WMF 等 PIL 不支持的格式会抛异常。"""
    from PIL import Image

    im = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(im)[:, :, ::-1].copy()


def bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    """给测试/取证用：BGR → PNG 字节。"""
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(bgr[:, :, ::-1]).save(buf, format="PNG")
    return buf.getvalue()
