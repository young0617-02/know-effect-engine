"""MinerU 4.x pipeline 的本地 ONNX 后端：版面结构（PP-DocLayoutV2）+ 中文 OCR（PP-OCRv6）。

为什么自己驱动 ONNX，而不是 pip 装 mineru：
  - mineru 包本身只有 3.5MB，但依赖树里带 lmdeploy / gradio / mineru-llama-cpp / accelerate，
    与本 venv 通过 .pth 挂载的 CUDA torch 底座同装极易互相顶版本——为了省事把 8GB 显存
    那条已验证的调度基线搞坏，是亏本买卖。
  - 只用 onnxruntime(14MB) + pypdfium2(4MB)，模型走本机 D:\\modelscope 已有的 819MB 权重，
    零 GB 下载、零外网调用（数据不出域）。
  - 推理放在 CPU：GPU 已被 7B + 嵌入 + 重排按 Gate 0 ① 的实测占满（峰值 8098/8188 MiB），
    版面/OCR 再挤进去就是把临界通过变成 OOM。

公式（PP-FormulaNet_plus-M 564MB）本地已有，按需加载：只有版面模型真的判出公式区域才把
564MB 的 ONNX 会话装进来，普通文档不付这笔内存/时间；模型失败时退回 OCR 文本行。
图片/图表区域不再整块丢弃：区域文字照常 OCR；只保留图内文字，不做图表语义理解，
也不引入额外视觉模型。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from kee.config import get_settings
from kee.ingest.parsers import (
    ParsedDocument,
    ParsedFigure,
    ParsedSection,
    ParseFailure,
    _SectionNumbering,
)

# 会形成阅读结构的文字类标签
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
# 不能再在 `LayoutDetector._decode` 里被过滤掉：图片/图表/公式也要拿到框，
# 否则 parser 根本看不见它们，只能整块丢弃。
MEDIA_LABELS = {
    "image",
    "figure",
    "chart",
    "display_formula",
    "inline_formula",
    "formula_number",
    "seal",
    "header",
    "footer",
    "footer_image",
    "header_image",
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
            if label not in TEXT_LABELS and label not in MEDIA_LABELS:
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


# ---- 表格结构（SLANet-plus 坐标头）----
# 本地 Table/ 目录只有 3 个 onnx、没有随附的 50 词 HTML 词表，所以 token id 的语义
# 不靠猜，用 scripts/probe_slanet.py 从坐标头实测反推（真值：人工读 hak180 有害物质表）：
#   id=48 的 quad 恰好铺满网格 —— 113 个 cell 切成 17 行、每行 7 列，与真值一致；
#     且 y 中心聚类在阈值 0.005~0.02、左边界聚类在 0.02~0.05 下都稳定给出 17×7，不是调出来的。
#   id∈{5,6} 各出现 17 次、成对跟在每行 cell 之后 —— 行终止符。
#   其余 id 的 quad 与 cell 大面积重叠（实测重叠率 0.72~1.02），是无效坐标；
#   只有跨列合并格那类 quad 落在所有 cell 之外（实测 id=7 的 quad 正是「有害物质」那格），
#   故按「与所有 cell 几乎不相交」回收，重叠的候选只留面积最大的。
TABLE_CELL_ID = 48
TABLE_ROW_END_IDS = frozenset({5, 6})
TABLE_PAD_ID = 1
TABLE_INPUT = 488


def _quad_to_box(q: np.ndarray, w: float, h: float) -> Box:
    """SLANet 的 loc 是 4 点 ×(x,y)，点序不保证，所以取 min/max 包成轴对齐框。"""
    xs = [float(q[i]) for i in (0, 2, 4, 6)]
    ys = [float(q[i]) for i in (1, 3, 5, 7)]
    return Box(min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h, 1.0, "cell")


def _overlap_ratio(b: Box, cells: list[Box]) -> float:
    if b.area <= 0:
        return 1.0
    hit = 0.0
    for c in cells:
        ix = max(0.0, min(b.x2, c.x2) - max(b.x1, c.x1))
        iy = max(0.0, min(b.y2, c.y2) - max(b.y1, c.y1))
        hit += ix * iy
    return hit / b.area


class TableStructureModel:
    """表格位图 → 逐行 cell 四边形 + 合并格；只用坐标头，不依赖 HTML 词表。"""

    def __init__(self, path: str | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(path) if path else model_root() / "Table/slanet-plus.onnx"
        so = ort.SessionOptions()
        # 这个导出的 shape 推断不完整，ORRuntime 会对每个 token 打一条 shape 合并告警，
        # 实测单张表格刷 68KB 日志；关掉，取证只看返回值。
        so.log_severity_level = 3
        so.log_verbosity_level = 0
        self.sess = ort.InferenceSession(
            str(self.path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def available(cls) -> bool:
        return (model_root() / "Table/slanet-plus.onnx").exists()

    def _infer(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        if bgr.shape[0] < 8 or bgr.shape[1] < 8:
            raise ParseFailure("表格裁剪区域过小，结构模型无法处理")
        bgr = OcrEngine._resize(bgr, TABLE_INPUT, TABLE_INPUT)
        x = ((bgr.astype("float32") / 255.0) - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None].astype("float32")
        loc, tok = self.sess.run(None, {self.sess.get_inputs()[0].name: x})
        return np.asarray(loc)[0], np.asarray(tok)[0]

    def structure(self, bgr: np.ndarray) -> tuple[list[list[Box]], list[Box]]:
        """返回 (按阅读顺序的行, 每行的 cell 框) 与 (跨列合并格框)，坐标为裁剪像素。"""
        loc, tok = self._infer(bgr)
        ids = tok.argmax(axis=-1)
        H, W = float(bgr.shape[0]), float(bgr.shape[1])
        rows: list[list[Box]] = []
        cur: list[Box] = []
        for i in range(len(ids)):
            d = int(ids[i])
            if d == TABLE_CELL_ID:
                cur.append(_quad_to_box(loc[i], W, H))
            elif d in TABLE_ROW_END_IDS:
                if cur:
                    cur.sort(key=lambda b: b.x1)
                    rows.append(cur)
                    cur = []
        if cur:
            cur.sort(key=lambda b: b.x1)
            rows.append(cur)
        if not rows:
            return [], []
        return rows, table_span_boxes(loc, ids, [b for r in rows for b in r], W, H)


def table_span_boxes(
    loc: np.ndarray, ids: np.ndarray, cells: list[Box], W: float, H: float
) -> list[Box]:
    """合并格：quad 与所有 cell 都几乎不相交、且面积够大的非 cell token。"""
    table = Box(
        min(c.x1 for c in cells),
        min(c.y1 for c in cells),
        max(c.x2 for c in cells),
        max(c.y2 for c in cells),
        1.0,
        "table",
    )
    cands: list[Box] = []
    for i in range(len(ids)):
        d = int(ids[i])
        if d == TABLE_CELL_ID or d == TABLE_PAD_ID:
            continue
        b = _quad_to_box(loc[i], W, H)
        if b.area < table.area * 0.003:
            continue
        if _overlap_ratio(b, cells) > 0.15:
            continue
        b.label = "span"
        cands.append(b)
    cands.sort(key=lambda b: -b.area)
    kept: list[Box] = []
    for b in cands:
        if any(_iou(b, k) > 0.3 for k in kept):
            continue
        kept.append(b)
    return kept


def snap_rows_to_grid(rows: list[list[Box]]) -> tuple[list[list[Box]], list[tuple[float, float]]]:
    """把每个 cell 的框吸附到「所在行 × 所在列」的共识边界。

    坐标头的单格 quad 会画歪：实测宽表首行首格右边界画到 393（该列真值 301），
    把「区域」那格撑成跨两列、下边界还短了一截，文本行因此落到隔壁格子里。
    行列边界是全表多格共识，所以按列取 x 中位、按行取 y 中位重画，比信单格 quad 稳。
    """
    if not rows:
        return rows, []
    ref = max(rows, key=len)
    col_edges: list[list[tuple[float, float]]] = [[] for _ in ref]
    bands: list[tuple[float, float]] = []

    def axis_of(b: Box, axes: list[tuple[float, float]]) -> int:
        return max(
            range(len(axes)),
            key=lambda k: max(0.0, min(b.x2, axes[k][1]) - max(b.x1, axes[k][0])),
        )

    for r in rows:
        for b in r:
            col_edges[axis_of(b, [(c.x1, c.x2) for c in ref])].append((b.x1, b.x2))
        bands.append(
            (float(np.median([b.y1 for b in r])), float(np.median([b.y2 for b in r])))
        )
    axes = [
        (float(np.median([e[0] for e in c])), float(np.median([e[1] for e in c])))
        for c in col_edges
        if c
    ]
    out: list[list[Box]] = []
    for r, (y1, y2) in zip(rows, bands):
        snapped = []
        for b in r:
            cx1, cx2 = axes[axis_of(b, axes)]
            snapped.append(Box(cx1, y1, cx2, y2, b.score, b.label, b.text))
        out.append(snapped)
    return out, axes


def assign_lines_to_cells(cells: list[Box], lines: list[Box]) -> list[list[Box]]:
    """把文本行**唯一**归属到格子：中心在格内优先，否则按距离兜底。

    必须一次性分配而不是逐格查找，否则同一行会被相邻两格同时认领，
    一个数字在表里出现两次是比丢字更糟的错。
    """
    owner: list[list[Box]] = [[] for _ in cells]
    for ln in lines:
        cx, cy = (ln.x1 + ln.x2) / 2, (ln.y1 + ln.y2) / 2
        hit = [
            i
            for i, b in enumerate(cells)
            if b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2
        ]
        if not hit:
            cand = min(
                range(len(cells)),
                key=lambda i: _dist(ln, (cells[i].x1 + cells[i].x2) / 2, (cells[i].y1 + cells[i].y2) / 2),
                default=None,
            )
            if cand is None:
                continue
            b = cells[cand]
            # 兜底阈值取半格：再远就说明这格本来就是空的（合并格、空白标记格），不能硬塞
            if _dist(ln, (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2) > 0.5 * (b.x2 - b.x1) + 0.5 * (b.y2 - b.y1):
                continue
            hit = [cand]
        else:
            # 格子边界不严格互斥（合并格会覆盖邻格），取中心最近的
            hit = [min(hit, key=lambda i: _dist(ln, (cells[i].x1 + cells[i].x2) / 2, (cells[i].y1 + cells[i].y2) / 2))]
        owner[hit[0]].append(ln)
    for grp in owner:
        grp.sort(key=lambda ln: (round(ln.y1 / 8), ln.x1))
    return owner


def lines_text(group: list[Box]) -> str:
    return _join_lines([ln.text.strip() for ln in group if ln.text.strip()])


def _join_lines(parts: list[str]) -> str:
    out = ""
    for p in parts:
        if out and _is_cjk(out[-1]) and _is_cjk(p[0]):
            out += p  # 中文换行不是词边界，插空格会把「多溴联苯」变成「多溴联 苯」，jieba 分词跟着错
        else:
            out = f"{out} {p}" if out else p
    return out


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def _dist(ln: Box, cx: float, cy: float) -> float:
    return abs((ln.x1 + ln.x2) / 2 - cx) + abs((ln.y1 + ln.y2) / 2 - cy)


def table_to_markdown(rows: list[list[Box]], spans: list[Box], lines: list[Box]) -> str:
    """cell 框 + 已识别文本行 → Markdown 表格。

    列轴取自「cell 最多的那一行」的横向分区，再用区间重叠把每格落到列上：
    这样不需要行列聚类阈值这类魔数，也天然容得下合并格。
    """
    if not rows:
        return ""
    if max(len(r) for r in rows) < 2:
        return ""
    rows, _axes = snap_rows_to_grid(rows)
    ref = max(rows, key=len)
    ncols = len(ref)
    grid: list[list[Box]] = [list(r) for r in rows]
    for s in spans:
        ri = max(
            range(len(grid)),
            key=lambda k: max((min(s.y2, c.y2) - max(s.y1, c.y1) for c in grid[k]), default=-1e9),
        )
        band = grid[ri][0]
        grid[ri].append(Box(s.x1, band.y1, s.x2, band.y2, s.score, "span", s.text))
    out: list[str] = []
    for r in grid:
        r.sort(key=lambda b: b.x1)
    flat = [b for r in grid for b in r]
    text_of = {
        id(b): lines_text(g)
        for b, g in zip(flat, assign_lines_to_cells(flat, lines))
    }
    for ri, r in enumerate(grid):
        cells = [""] * ncols
        for b in r:
            ov = [max(0.0, min(b.x2, ref[k].x2) - max(b.x1, ref[k].x1)) for k in range(ncols)]
            # 合并格锚在最左覆盖列（读起来是「有害物质」统领后面 6 列）；普通格落在重叠最多的列
            ci = next((k for k, v in enumerate(ov) if v > 0), 0) if b.label == "span" else max(
                range(ncols), key=lambda k: ov[k]
            )
            text = text_of[id(b)].replace("|", "｜").strip()
            if text and cells[ci]:
                text = _join_lines([cells[ci], text])
            cells[ci] = text
        out.append("| " + " | ".join(cells) + " |")
        if ri == 0:
            out.append("|" + "---|" * ncols)
    return "\n".join(out)


class FormulaModel:
    """PP-FormulaNet_plus-M ONNX：公式位图 → LaTeX。

    输入是 **单通道 384×384**，输出是 token id 序列。tokenizer 直接从模型目录的
    `*_inference.yml` 里取（`PostProcess.character_dict.fast_tokenizer_file`），
    不额外下载 50k 词表；`tokenizers` 由 CUDA/transformers 底座提供，缺失时抛错交给
    上层回退 OCR 文本行。
    """

    INPUT = 384

    def __init__(self, path: str | None = None, session: Any | None = None,
                 tokenizer: Any | None = None) -> None:
        self.path = Path(path) if path else _formula_model_path()
        if session is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.log_verbosity_level = 0
            session = ort.InferenceSession(
                str(self.path), sess_options=so, providers=["CPUExecutionProvider"]
            )
        self.sess = session
        self.tok = tokenizer or self._load_tokenizer()

    @classmethod
    def available(cls) -> bool:
        return _formula_model_path().exists()

    def _load_tokenizer(self) -> Any:
        from tokenizers import Tokenizer

        cfg = _yml(str(self.path.parent / f"{self.path.stem}_inference.yml"))
        torch_cfg = (cfg.get("PostProcess") or {}).get("character_dict") or {}
        fast = torch_cfg.get("fast_tokenizer_file")
        if not fast:
            raise ParseFailure("公式模型 inference.yml 缺少 fast_tokenizer_file")
        return Tokenizer.from_str(json.dumps(fast, ensure_ascii=False))

    def recognize(self, bgr: np.ndarray) -> str:
        x = self._preprocess(bgr)
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        ids = [int(v) for v in np.asarray(out)[0].tolist()]
        ids = [i for i in ids if i >= 0]
        if not ids:
            return ""
        text = self.tok.decode(ids, skip_special_tokens=True)
        text = text.replace("[object Object]", "").strip()
        return text[:2000]

    @classmethod
    def _preprocess(cls, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        if bgr.ndim != 3 or bgr.shape[0] < 8 or bgr.shape[1] < 8:
            raise ValueError(f"公式裁剪区域过小或形状不对：{bgr.shape}")
        # 先裁掉白边：版面框通常比公式宽，白边会被 resize 成有效像素。
        rgb = bgr[:, :, ::-1]
        mask = np.asarray(rgb).min(axis=2) < 245
        if mask.any():
            ys, xs = np.where(mask)
            y1, y2 = max(0, int(ys.min()) - 6), min(rgb.shape[0], int(ys.max()) + 7)
            x1, x2 = max(0, int(xs.min()) - 6), min(rgb.shape[1], int(xs.max()) + 7)
            rgb = rgb[y1:y2, x1:x2]
        im = Image.fromarray(rgb).convert("L")
        w, h = im.size
        # 小公式也要放大到 384，不能因为原图小于 384 就原样居中（那会让模型看到一张很小的图）
        scale = cls.INPUT / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (cls.INPUT, cls.INPUT), 255)
        canvas.paste(im, ((cls.INPUT - nw) // 2, (cls.INPUT - nh) // 2))
        arr = np.asarray(canvas).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        return arr[None, None]


def _formula_model_path() -> Path:
    custom = (get_settings().formula_model_path or "").strip()
    if custom:
        return Path(custom)
    return model_root() / "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.onnx"


class MinerUPdfParser:
    """PDF → 整页 det 出行框 → 逐行 rec → 按版面框归属 → 带真实页码的章节结构。

    为什么不能把版面块直接丢给识别模型：PP-OCR 的 rec 是**单行**模型，喂多行块
    会把跨行像素当成一行来认，实测出一串看着像中文的乱码。所以必须 det 出行框、
    逐行 rec，再用版面框决定这段文字是标题、正文还是表格。
    """

    def __init__(
        self,
        dpi: int | None = None,
        use_layout: bool = True,
        use_table: bool = True,
        use_formula: bool | None = None,
    ) -> None:
        s = get_settings()
        self.dpi = dpi or s.ocr_dpi
        self.use_formula = s.parse_formula if use_formula is None else use_formula
        self._formula: FormulaModel | None = None
        self._formula_error = ""
        self.ocr = OcrEngine()
        # 版面模型 204MB，整份文档复用一次加载，不能逐页重建会话
        self.layout = (
            LayoutDetector()
            if use_layout and (model_root() / "Layout/PP-DocLayoutV2/inference.onnx").exists()
            else None
        )
        # 表格结构模型只有 7.4MB，加载成本可忽略，但只在版面框判出 table 时才推理
        self.table = (
            TableStructureModel()
            if use_table and TableStructureModel.available()
            else None
        )

    @staticmethod
    def _region_of(line: Box, regions: list[Box]) -> Box | None:
        """行中心落在哪个版面框里；多重包含取面积最小的（更具体的块优先）。"""
        cx, cy = (line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2
        hits = [r for r in regions if r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2]
        return min(hits, key=lambda r: r.area) if hits else None

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        return self.parse_pages(data, filename)

    def parse_pages(
        self, data: bytes, filename: str, page_numbers: set[int] | None = None
    ) -> ParsedDocument:
        """整份或指定页解析。

        `page_numbers` 是 1-based 页号集合；给定时只渲染这些页并保留真实页码，
        供 `parse_pdf` 的按页路由调用。整份调用仍受 `render_pdf_pages` 的页数上限约束。
        """
        from kee.ingest.docimg import render_pdf_pages

        notes = [
            "MinerU-ONNX 后端：PP-OCRv6(tiny det + small rec) 整页行级识别 + "
            f"PP-DocLayoutV2 版面归属（{'开' if self.layout else '关'}）；CPU 推理"
        ]
        pages: list[tuple[int, list[tuple[Box, str]]]] = []
        figures: list[ParsedFigure] = []
        counts = {"table_ok": 0, "table_fail": 0, "formula_ok": 0, "formula_fail": 0, "figure": 0}
        rendered: set[int] = set()
        for pno, bgr in render_pdf_pages(data, dpi=self.dpi, pages=page_numbers):
            rendered.add(pno)
            assigned, page_figures, page_counts = self._parse_page(bgr, pno, notes)
            pages.append((pno, assigned))
            figures.extend(page_figures)
            for key, value in page_counts.items():
                counts[key] += value
        if page_numbers is not None:
            missing = sorted(set(page_numbers) - rendered)
            note = f"页级 MinerU：请求 {len(page_numbers)} 页，处理 {len(pages)} 页"
            if missing:
                note += f"；未渲染 {missing[:10]}{'…' if len(missing) > 10 else ''}（不存在或超过单次上限）"
            notes.append(note)
        if self.table is not None:
            notes.append(
                f"表格结构还原（SLANet-plus）：成功 {counts['table_ok']} 张，"
                f"失败回退文本行 {counts['table_fail']} 张"
            )
        if counts["formula_ok"] or counts["formula_fail"]:
            notes.append(
                f"公式识别（PP-FormulaNet_plus-M）：成功 {counts['formula_ok']}，"
                f"回退文本 {counts['formula_fail']}"
            )
        if counts["figure"]:
            notes.append(f"图片/图表区域 {counts['figure']} 个：仅保留图内 OCR 文字，不做语义描述")
        return self._to_document(pages, filename, notes, figures=figures)

    def _parse_page(
        self, bgr: np.ndarray, pno: int, notes: list[str]
    ) -> tuple[list[tuple[Box, str]], list[ParsedFigure], dict[str, int]]:
        """单页：整页 det/rec → 版面归属 → 表格/公式/图表 → 其余文本行。"""
        lines = self.ocr.det_boxes(bgr)
        for ln in lines:
            ln.text = self.ocr.rec_text(self.ocr.crop(bgr, ln))
        lines = [ln for ln in lines if len(ln.text.strip()) >= 1]
        regions = self.layout.detect(bgr) if self.layout else []
        tables = [r for r in regions if r.label == "table"] if self.table else []
        formula_regions = self._dedupe_regions(
            [r for r in regions if r.label in {"display_formula", "inline_formula"}]
        )
        figure_regions = self._dedupe_regions(
            [r for r in regions if r.label in {"image", "figure", "chart"}]
        )
        consumed: set[int] = set()
        assigned: list[tuple[Box, str]] = []
        figures: list[ParsedFigure] = []
        counts = {"table_ok": 0, "table_fail": 0, "formula_ok": 0, "formula_fail": 0, "figure": 0}

        def inside(reg: Box) -> list[int]:
            return [
                i
                for i, ln in enumerate(lines)
                if reg.x1 <= (ln.x1 + ln.x2) / 2 <= reg.x2
                and reg.y1 <= (ln.y1 + ln.y2) / 2 <= reg.y2
            ]

        # 1) 表格优先：表格结构模型会消费框内行，避免行块与表格内容重复。
        for reg in tables:
            idxs = inside(reg)
            md = self._table_markdown(bgr, reg, [lines[i] for i in idxs], notes)
            if md is None:
                counts["table_fail"] += 1
                continue  # 结构没出来就退回文本行，不能把已经 det 到的行丢掉
            consumed.update(idxs)
            counts["table_ok"] += 1
            assigned.append((Box(reg.x1, reg.y1, reg.x2, reg.y2, reg.score, "table", md), "table"))

        # 2) 公式：PP-FormulaNet 优先；模型不可用/失败时保留 OCR 文本行。
        for reg in formula_regions:
            idxs = inside(reg)
            crop = self._crop(bgr, reg)
            latex = self._formula_latex(crop, notes)
            consumed.update(idxs)
            if latex:
                counts["formula_ok"] += 1
                box = Box(reg.x1, reg.y1, reg.x2, reg.y2, reg.score, "formula", latex)
                assigned.append((box, "formula"))
                figures.append(ParsedFigure(
                    kind="formula", page=pno, text=latex,
                    bbox=(reg.x1, reg.y1, reg.x2, reg.y2),
                ))
            else:
                counts["formula_fail"] += 1
                for i in idxs:
                    assigned.append((lines[i], "text"))

        # 3) 图片/图表：区域文字已由 PP-OCR 识别，直接作为可检索文本；不做图表语义理解。
        for reg in figure_regions:
            idxs = inside(reg)
            ocr_text = " ".join(lines[i].text.strip() for i in idxs if lines[i].text.strip())
            consumed.update(idxs)
            if not ocr_text:
                continue
            kind = "chart" if reg.label == "chart" else "figure"
            label = "图表" if kind == "chart" else "插图"
            text = f"【{label} 第{pno}页】图内文字：{ocr_text}"
            counts["figure"] += 1
            assigned.append((Box(reg.x1, reg.y1, reg.x2, reg.y2, reg.score, "figure", text), "figure"))
            figures.append(ParsedFigure(
                kind=kind, page=pno, text=ocr_text,
                bbox=(reg.x1, reg.y1, reg.x2, reg.y2),
            ))

        # 4) 其余行按原来的阅读结构走；页眉页脚/印章/图形区不参与正文。
        for i, ln in enumerate(lines):
            if i in consumed:
                continue
            reg = self._region_of(ln, regions) if regions else None
            label = reg.label if reg is not None else "text"
            if label in {
                "image", "figure", "chart", "display_formula", "inline_formula",
                "formula_number", "seal", "header", "footer", "footer_image", "header_image",
            }:
                continue
            assigned.append((ln, label))
        assigned.sort(key=lambda x: (round(x[0].y1 / 10), x[0].x1))
        return assigned, figures, counts

    @staticmethod
    def _dedupe_regions(regions: list[Box]) -> list[Box]:
        """版面模型偶尔对同一公式/图表输出多个高度重叠的框；保留最大的那个。"""
        kept: list[Box] = []
        for r in sorted(regions, key=lambda b: b.area, reverse=True):
            duplicate = False
            for k in kept:
                ix1, iy1 = max(r.x1, k.x1), max(r.y1, k.y1)
                ix2, iy2 = min(r.x2, k.x2), min(r.y2, k.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter <= 0:
                    continue
                union = r.area + k.area - inter
                if union > 0 and (inter / union >= 0.5 or inter / max(1.0, min(r.area, k.area)) >= 0.8):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(r)
        return kept

    @staticmethod
    def _crop(bgr: np.ndarray, reg: Box) -> np.ndarray:
        H, W = bgr.shape[:2]
        x1, y1 = max(0, int(reg.x1)), max(0, int(reg.y1))
        x2, y2 = min(W, int(reg.x2)), min(H, int(reg.y2))
        return bgr[y1:y2, x1:x2]

    def _formula_latex(self, crop: np.ndarray, notes: list[str]) -> str:
        if crop.size == 0:
            return ""
        model = self._formula_or_none(notes)
        if model is None:
            return ""
        try:
            return model.recognize(crop)
        except Exception as exc:  # noqa: BLE001 - 公式模型失败保留 OCR 文本行，不拖垮整份 PDF
            notes.append(f"公式模型推理失败（回退 OCR 文本）：{type(exc).__name__}: {exc}"[:200])
            return ""

    def _formula_or_none(self, notes: list[str]) -> FormulaModel | None:
        if not self.use_formula:
            return None
        if self._formula is not None:
            return self._formula
        if self._formula_error:
            return None
        if not FormulaModel.available():
            self._formula_error = "公式模型文件不存在"
            notes.append(f"公式模型不可用：{self._formula_error}")
            return None
        try:
            self._formula = FormulaModel()
        except Exception as exc:  # noqa: BLE001
            self._formula_error = f"{type(exc).__name__}: {exc}"[:200]
            notes.append(f"公式模型加载失败（回退 OCR 文本）：{self._formula_error}")
            return None
        return self._formula

    def _table_markdown(self, bgr: np.ndarray, reg: Box, page_lines: list[Box], notes: list[str]) -> str | None:
        """版面框内的表格 → Markdown；结构不可用时返回 None，让调用方按文本行兜底。"""
        assert self.table is not None
        ox, oy = int(reg.x1), int(reg.y1)
        crop = bgr[oy : min(bgr.shape[0], int(reg.y2)), ox : min(bgr.shape[1], int(reg.x2))]
        inside = [
            Box(ln.x1 - ox, ln.y1 - oy, ln.x2 - ox, ln.y2 - oy, ln.score, "line", ln.text)
            for ln in page_lines
        ]
        try:
            rows, spans = self.table.structure(crop)
        except Exception as exc:  # 结构失败不能把整篇文档拖回 pypdf
            notes.append(f"表格结构还原失败（按文本行处理）：{type(exc).__name__}: {exc}")
            return None
        md = table_to_markdown(rows, spans, inside)
        if not md:
            notes.append("表格结构还原为空网格（按文本行处理）")
        return md or None

    def _to_document(
        self,
        pages: list[tuple[int, list[tuple[Box, str]]]],
        filename: str,
        notes: list[str],
        figures: list[ParsedFigure] | None = None,
    ) -> ParsedDocument:
        stem = Path(filename).stem
        sections: list[ParsedSection] = []
        cur: ParsedSection | None = None
        num = _SectionNumbering()
        title = ""
        chars = 0
        for pno, items in pages:
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
                if label == "table":
                    flush()
                    prev_y = None
                    if cur is None:
                        cur = ParsedSection(num.next(1), f"第 {pno} 页", pno, [])
                        sections.append(cur)
                    cur.paragraphs.append(text)  # Markdown 表格整块成段，不能被行序打散
                    continue
                if label in {"formula", "figure"}:
                    flush()
                    prev_y = None
                    if cur is None:
                        cur = ParsedSection(num.next(1), f"第 {pno} 页", pno, [])
                        sections.append(cur)
                    cur.paragraphs.append(
                        f"【公式】${text}$" if label == "formula" else text
                    )
                    continue
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
        notes.append(f"OCR 产出 {chars} 字符，覆盖 {len(pages)} 页")
        return ParsedDocument(
            title=title or stem,
            sections=sections,
            source_format="pdf",
            notes=notes,
            meta_hint={},
            figures=figures or [],
        ).normalize()  # OCR 会产出「•」这类碎行，必须走同一套段落归一化，否则撞准入长度规则
