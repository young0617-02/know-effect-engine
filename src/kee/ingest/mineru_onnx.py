"""MinerU 4.x pipeline 的本地 ONNX 后端：版面结构（PP-DocLayoutV2）+ 中文 OCR（PP-OCRv6）。

为什么自己驱动 ONNX，而不是 pip 装 mineru：
  - mineru 包本身只有 3.5MB，但依赖树里带 lmdeploy / gradio / mineru-llama-cpp / accelerate，
    与本 venv 通过 .pth 挂载的 CUDA torch 底座同装极易互相顶版本——为了省事把 8GB 显存
    那条已验证的调度基线搞坏，是亏本买卖。
  - 只用 onnxruntime(14MB) + pypdfium2(4MB)，模型走本机 D:\\modelscope 已有的 819MB 权重，
    零 GB 下载、零外网调用（数据不出域）。
  - 推理放在 CPU：GPU 已被 7B + 嵌入 + 重排按 Gate 0 ① 的实测占满（峰值 8098/8188 MiB），
    版面/OCR 再挤进去就是把临界通过变成 OOM。

公式（PP-FormulaNet-M 564MB）与表格结构（SLANet-plus/UniNet）模型本地已有但未接：
交付文档里公式罕见，表格先按版面框定位 + OCR 成文本行，行列结构还原留到 M2 按需接。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from kee.config import get_settings
from kee.ingest.parsers import ParsedDocument, ParsedSection, _SectionNumbering

# 版面标签里会形成阅读结构的这几类，其余（页眉页脚图片印章等）丢弃
TEXT_LABELS = {
    "paragraph_title",
    "text",
    "table",
    "figure_title",
    "vertical_text",
    "reference_content",
    "content",
    "abstract",
}
HEADING_LABELS = {"paragraph_title", "doc_title", "figure_title"}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = ""
    text: str = ""

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def model_root() -> Path:
    return Path(get_settings().mineru_model_dir)


@lru_cache(maxsize=8)
def _yml(path: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class LayoutDetector:
    """PP-DocLayoutV2（DETR 导出）：800x800 输入，输出 [N,8] = x1,y1,x2,y2,score,cls,-,-。"""

    def __init__(self, path: str | None = None, score_thr: float = 0.30) -> None:
        import onnxruntime as ort

        root = model_root()
        self.model = Path(path) if path else root / "Layout/PP-DocLayoutV2/inference.onnx"
        cfg = _yml(str(self.model.parent / "inference.yml"))
        self.labels: list[str] = cfg["label_list"]
        self.score_thr = score_thr
        self.sess = ort.InferenceSession(str(self.model), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def detect(self, bgr: np.ndarray) -> list[Box]:
        h, w = bgr.shape[:2]
        canvas = self._letterbox(bgr, 800)
        # PaddleX 的 NormalizeImage 是 mean=0/std=1 且 norm_type=none，但导出的 ONNX 图
        # 内部按 0~1 输入训练：实测不除 255 时 300 个框全是分数 0.05 的垃圾，除了才对
        chw = (canvas[:, :, ::-1].astype("float32") / 255.0).transpose(2, 0, 1)
        feeds = {
            "image": chw[None],
            "im_shape": np.array([[800, 800]], dtype="float32"),
            "scale_factor": np.array([[800.0 / h, 800.0 / w]], dtype="float32"),
        }
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        out = self.sess.run(None, feeds)[0]
        return self._decode(np.asarray(out), (h, w))

    @staticmethod
    def _letterbox(bgr: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        im = Image.fromarray(bgr[:, :, ::-1]).resize((size, size), Image.BILINEAR)
        return np.asarray(im)[:, :, ::-1].copy()  # 回到 BGR 通道序，与 Paddle 解码一致

    def _decode(self, out: np.ndarray, orig_hw: tuple[int, int]) -> list[Box]:
        """导出张量的列序是 [cls_id, score, x1, y1, x2, y2, ...]，坐标已是原图尺度。"""
        h, w = orig_hw
        boxes: list[Box] = []
        matrix = out.reshape(-1, out.shape[-1]) if out.ndim > 1 else out[None, :]
        for row in matrix:
            if row.size < 6:
                continue
            cls_id, score = int(row[0]), float(row[1])
            x1, y1, x2, y2 = (float(v) for v in row[2:6])
            if score < self.score_thr or cls_id < 0 or cls_id >= len(self.labels):
                continue
            label = self.labels[cls_id]
            if label not in TEXT_LABELS:
                continue
            if max(x1, y1, x2, y2) <= 1.5:  # 归一化输出
                x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            elif max(x1, x2) > w * 1.05 or max(y1, y2) > h * 1.05:  # 800 画布尺度
                x1, x2 = x1 * (w / 800.0), x2 * (w / 800.0)
                y1, y2 = y1 * (h / 800.0), y2 * (h / 800.0)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            boxes.append(Box(x1, y1, x2, y2, score, label))
        boxes.sort(key=lambda b: -b.score)
        kept: list[Box] = []
        for b in boxes:
            if any(_iou(b, k) > 0.6 for k in kept):
                continue
            kept.append(b)
        kept.sort(key=lambda b: (round(b.y1 / 12), b.x1))  # 阅读顺序：先按行带、再按横向
        return kept


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class OcrEngine:
    """PP-OCRv6 tiny det + small rec，ONNX 直推，DB 后处理用 scipy 连通域（不引 cv2）。"""

    DET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
    DET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")

    def __init__(self, det_path: str | None = None, rec_path: str | None = None) -> None:
        import onnxruntime as ort

        root = model_root() / "OCR/paddleocr"
        det = Path(det_path) if det_path else root / "ch_PP-OCRv6_tiny_det_infer.onnx"
        rec = Path(rec_path) if rec_path else root / "ch_PP-OCRv6_small_rec_infer.onnx"
        self.det = ort.InferenceSession(str(det), providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(rec), providers=["CPUExecutionProvider"])
        dcfg = _yml(str(det.parent / "ch_PP-OCRv6_tiny_det_inference.yml"))
        rcfg = _yml(str(rec.parent / "ch_PP-OCRv6_small_rec_inference.yml"))
        pp = dcfg["PostProcess"]
        self.thresh = float(pp.get("thresh", 0.2))
        self.box_thresh = float(pp.get("box_thresh", 0.4))
        self.unclip = float(pp.get("unclip_ratio", 1.4))
        chars = list(rcfg["PostProcess"]["character_dict"])
        if rcfg["PostProcess"].get("use_space_char", rcfg.get("MinerU", {}).get("use_space_char")):
            chars.append(" ")
        self.dict = ["blank"] + chars  # CTC 第 0 位是 blank

    # ---- 检测 ----
    def det_boxes(self, bgr: np.ndarray, limit_side: int = 960) -> list[Box]:
        from scipy import ndimage

        h, w = bgr.shape[:2]
        ratio = min(1.0, limit_side / max(h, w))
        nh, nw = max(32, int(h * ratio) // 32 * 32), max(32, int(w * ratio) // 32 * 32)
        small = self._resize(bgr, nw, nh)
        img = small.astype("float32") / 255.0
        img = (img - self.DET_MEAN) / self.DET_STD
        x = img.transpose(2, 0, 1)[None].astype("float32")
        prob = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        mask = prob > self.thresh
        if not mask.any():
            return []
        lab, n = ndimage.label(mask)
        if n == 0:
            return []
        boxes: list[Box] = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            ys, xs = sl[0], sl[1]
            y1, y2 = int(ys.start), int(ys.stop)
            x1, x2 = int(xs.start), int(xs.stop)
            area = (y2 - y1) * (x2 - x1)
            if area < 12 or (y2 - y1) < 4:
                continue
            sub = prob[y1:y2, x1:x2]
            sel = (lab[y1:y2, x1:x2] == i)
            score = float(sub[sel].mean()) if sel.any() else 0.0
            if score < self.box_thresh:
                continue
            bw, bh = x2 - x1, y2 - y1
            pad = max(1, int((bw * bh) * (self.unclip - 1.0) / max(2 * (bw + bh), 1)))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(nw, x2 + pad), min(nh, y2 + pad)
            sx, sy = w / nw, h / nh
            boxes.append(Box(x1 * sx, y1 * sy, x2 * sx, y2 * sy, score))
        boxes.sort(key=lambda b: (round(b.y1 / 14), b.x1))
        return boxes

    # ---- 识别 ----
    def rec_text(self, bgr: np.ndarray) -> str:
        crops = self._resize_keep_h(bgr, 48)
        if crops is None:
            return ""
        x = ((crops.astype("float32") / 255.0) - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None].astype("float32")
        logits = self.rec.run(None, {self.rec.get_inputs()[0].name: x})[0][0]
        idx = logits.argmax(axis=-1)
        out: list[str] = []
        prev = -1
        for i in idx:
            i = int(i)
            if i != prev and 0 < i < len(self.dict):
                out.append(self.dict[i])
            prev = i
        return "".join(out).strip()

    @staticmethod
    def _resize(bgr: np.ndarray, nw: int, nh: int) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.fromarray(bgr[:, :, ::-1]).resize((nw, nh), Image.BILINEAR))[:, :, ::-1].copy()

    @staticmethod
    def _resize_keep_h(bgr: np.ndarray, h: int, max_w: int = 2560) -> np.ndarray | None:
        from PIL import Image

        ih, iw = bgr.shape[:2]
        if ih == 0 or iw == 0:
            return None
        nw = min(max_w, max(16, int(round(iw * h / ih))))
        return OcrEngine._resize(bgr, nw, h)

    def crop(self, bgr: np.ndarray, b: Box) -> np.ndarray:
        H, W = bgr.shape[:2]
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(W, int(b.x2)), min(H, int(b.y2))
        return bgr[y1:y2, x1:x2]


class MinerUPdfParser:
    """PDF → 整页 det 出行框 → 逐行 rec → 按版面框归属 → 带真实页码的章节结构。

    为什么不能把版面块直接丢给识别模型：PP-OCR 的 rec 是**单行**模型，喂多行块
    会把跨行像素当成一行来认，实测出一串看着像中文的乱码。所以必须 det 出行框、
    逐行 rec，再用版面框决定这段文字是标题、正文还是表格。
    """

    def __init__(self, dpi: int | None = None, use_layout: bool = True) -> None:
        s = get_settings()
        self.dpi = dpi or s.ocr_dpi
        self.ocr = OcrEngine()
        # 版面模型 204MB，整份文档复用一次加载，不能逐页重建会话
        self.layout = (
            LayoutDetector()
            if use_layout and (model_root() / "Layout/PP-DocLayoutV2/inference.onnx").exists()
            else None
        )

    @staticmethod
    def _region_of(line: Box, regions: list[Box]) -> Box | None:
        """行中心落在哪个版面框里；多重包含取面积最小的（更具体的块优先）。"""
        cx, cy = (line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2
        hits = [r for r in regions if r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2]
        return min(hits, key=lambda r: r.area) if hits else None

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        from kee.ingest.docimg import render_pdf_pages

        notes = [
            "MinerU-ONNX 后端：PP-OCRv6(tiny det + small rec) 整页行级识别 + "
            f"PP-DocLayoutV2 版面归属（{'开' if self.layout else '关'}）；CPU 推理"
        ]
        pages: list[list[tuple[Box, str]]] = []
        for _pno, bgr in render_pdf_pages(data, dpi=self.dpi):
            lines = self.ocr.det_boxes(bgr)
            for ln in lines:
                ln.text = self.ocr.rec_text(self.ocr.crop(bgr, ln))
            lines = [ln for ln in lines if len(ln.text.strip()) >= 1]
            regions = self.layout.detect(bgr) if self.layout else []
            assigned: list[tuple[Box, str]] = []
            for ln in lines:
                reg = self._region_of(ln, regions) if regions else None
                label = reg.label if reg is not None else "text"
                if label in {"image", "seal", "header", "footer", "figure"}:
                    continue  # 图形/页眉页脚类区域不参与正文
                assigned.append((ln, label))
            assigned.sort(key=lambda x: (round(x[0].y1 / 10), x[0].x1))
            pages.append(assigned)
        return self._to_document(pages, filename, notes)

    def _to_document(self, pages: list[list[tuple[Box, str]]], filename: str, notes: list[str]) -> ParsedDocument:
        stem = Path(filename).stem
        sections: list[ParsedSection] = []
        cur: ParsedSection | None = None
        num = _SectionNumbering()
        title = ""
        chars = 0
        for pno, items in enumerate(pages, start=1):
            buf: list[str] = []
            prev_y: float | None = None

            def flush() -> None:
                if not buf:
                    return
                text = " ".join(buf).strip()
                buf.clear()
                if len(text) < 2 or cur is None:
                    return
                cur.paragraphs.append(text)

            for b, label in items:
                text = b.text.strip()
                if not text:
                    continue
                chars += len(text)
                if label in HEADING_LABELS and len(text) <= 30:
                    flush()
                    prev_y = None
                    if label == "doc_title" and not title:
                        title = text
                        continue
                    cur = ParsedSection(num.next(1), text, pno, [])
                    sections.append(cur)
                    continue
                if cur is None:
                    cur = ParsedSection(num.next(1), f"第 {pno} 页", pno, [])
                    sections.append(cur)
                # 行间距明显变大就断段，避免整页糊成一段
                if prev_y is not None and b.y1 - prev_y > 18:
                    flush()
                prev_y = b.y1
                buf.append(text)
            flush()
        dropped = [s for s in sections if not s.paragraphs]
        sections = [s for s in sections if s.paragraphs]
        if dropped:
            notes.append(f"丢弃 {len(dropped)} 个无正文的标题块（多栏页面上版面归属会串行，M2 待修）")
        if not sections and dropped:
            sections = [
                ParsedSection("1", dropped[0].heading, dropped[0].page, [d.heading for d in dropped if d.heading])
            ]
        if not sections:
            notes.append("OCR 未产出可用文本：可能是超低分辨率扫描或空白页，应回退 pypdf 文本层")
        if not sections:
            notes.append("OCR 未产出可用文本：可能是超低分辨率扫描或空白页，应回退 pypdf 文本层")
        notes.append(f"OCR 产出 {chars} 字符，覆盖 {len(pages)} 页")
        return ParsedDocument(
            title=title or stem,
            sections=sections,
            source_format="pdf",
            notes=notes,
            meta_hint={},
        ).normalize()  # OCR 会产出「•」这类碎行，必须走同一套段落归一化，否则撞准入长度规则
