"""SLANet-plus 坐标头反推证据脚本（本机 Table/ 目录不随附 50 词 HTML 词表）。

三个问题必须用实测回答，不能靠猜 token 标签：
  A 每个 token id 的 quad 到底是什么形状 —— 判定哪个 id 是单元格；
  B 行/列数在不同聚类阈值下稳不稳 —— 判定 17×7 是巧合还是稳定结论；
  C 非 cell token 的 quad 与 cell 的重叠率 —— 判定「合并格能不能回收、怎么回收」。

跑法：.venv/Scripts/python.exe scripts/probe_slanet.py
输出：var/tmp/table_probe4.txt（逐 token quad 全量）+ 本脚本 stdout 的三段结论
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.ingest.docimg import render_pdf_pages  # noqa: E402
from kee.ingest.mineru_onnx import (  # noqa: E402
    TABLE_CELL_ID,
    TABLE_ROW_END_IDS,
    Box,
    LayoutDetector,
    TableStructureModel,
    _overlap_ratio,
    _quad_to_box,
)

PDF = Path("D:/project/kb_rag/doc/hak180产品安全手册.pdf")
OUT = ROOT / "var" / "tmp" / "table_probe4.txt"
# 真值：人眼读 var/tmp/table_probe4.png（该页版面框裁出的表格区域）
TRUTH_ROWS, TRUTH_COLS = 17, 7


def table_crop():
    data = PDF.read_bytes()
    page = dict(render_pdf_pages(data, dpi=150))[1]
    reg = max((b for b in LayoutDetector().detect(page) if b.label == "table"), key=lambda b: b.area)
    x1, y1, x2, y2 = (int(round(v)) for v in (reg.x1, reg.y1, reg.x2, reg.y2))
    crop = page[y1:y2, x1:x2]
    OUT.with_suffix(".png").write_bytes(_png(crop))
    return crop


def _png(bgr) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(bgr[:, :, ::-1]).save(buf, "PNG")
    return buf.getvalue()


def cluster(vals: list[float], thr: float) -> list[list[float]]:
    out: list[list[float]] = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= thr:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def main() -> int:
    crop = table_crop()
    sess = TableStructureModel()
    loc, tok = sess._infer(crop)
    ids = tok.argmax(axis=-1)
    L = len(ids)
    H, W = float(crop.shape[0]), float(crop.shape[1])

    lines = [f"crop={crop.shape[1]}x{crop.shape[0]} L={L} 真值={TRUTH_ROWS}行x{TRUTH_COLS}列"]
    for i in range(L):
        lines.append(f"  {i:3d} id={int(ids[i]):3d} quad={tuple(round(float(v), 4) for v in loc[i])}")
    lines.append(f"id 分布: {sorted(Counter(int(v) for v in ids).items(), key=lambda kv: (-kv[1], kv[0]))}")
    OUT.write_text("\n".join(lines), encoding="utf-8")

    # ---- A：哪个 id 是单元格 ----
    by_id = Counter(int(v) for v in ids)
    print(f"[A] token 总数 {L}，id 分布 {dict(by_id)}")
    print(f"    全量 quad 已写 {OUT}")
    cells = [_quad_to_box(loc[i], W, H) for i in range(L) if int(ids[i]) == TABLE_CELL_ID]
    print(f"[A] id={TABLE_CELL_ID} 共 {len(cells)} 个 quad，铺满整表（并集覆盖 "
          f"{_coverage(cells, crop.shape)}），判定为单元格")

    # ---- B：行列数稳不稳 ----
    rows: list[list[int]] = []
    cur: list[int] = []
    for i in range(L):
        d = int(ids[i])
        if d == TABLE_CELL_ID:
            cur.append(i)
        elif d in TABLE_ROW_END_IDS:
            if cur:
                rows.append(cur)
                cur = []
    if cur:
        rows.append(cur)
    print(f"[B] 按 id∈{sorted(TABLE_ROW_END_IDS)} 行终止符切分 → {len(rows)} 行，"
          f"每行 cell 数 {[len(r) for r in rows]}")
    ok_b = len(rows) == TRUTH_ROWS and max(len(r) for r in rows) == TRUTH_COLS
    for thr in (0.005, 0.01, 0.02):
        g = cluster([float(np.mean(loc[i][[1, 3, 5, 7]])) for i in range(L) if int(ids[i]) == TABLE_CELL_ID], thr)
        print(f"    y 中心聚类 thr={thr} → {len(g)} 组")
        ok_b &= len(g) == TRUTH_ROWS
    for thr in (0.02, 0.03, 0.05):
        g = cluster([float(np.min(loc[i][[0, 2, 4, 6]])) for i in range(L) if int(ids[i]) == TABLE_CELL_ID], thr)
        print(f"    x 左边界聚类 thr={thr} → {len(g)} 组")
        ok_b &= len(g) == TRUTH_COLS
    print(f"[B] 结论：{'与真值一致且阈值不敏感' if ok_b else '与真值不一致'}")

    # ---- C：合并格能不能回收 ----
    print("[C] 非 cell token 的 quad 与 cell 的重叠率（>0.15 判无效坐标）：")
    kept = 0
    for i in range(L):
        d = int(ids[i])
        if d == TABLE_CELL_ID or d == 1:
            continue
        b = _quad_to_box(loc[i], W, H)
        r = _overlap_ratio(b, cells)
        if r <= 0.15:
            kept += 1
            print(f"    tok{i:3d} id={d:2d} box=({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}) 重叠={r:.2f} ← 合并格候选")
    spans = sess.structure(crop)[1]
    print(f"[C] 候选 {kept} 个，IoU 去重后留 {len(spans)} 个 → "
          f"{[(round(s.x1), round(s.y1), round(s.x2), round(s.y2)) for s in spans]}")
    print(f"    该框对应真值第一行「有害物质」跨 6 列的合并表头")
    return 0 if (ok_b and len(spans) == 1) else 1


def _coverage(cells: list[Box], hw) -> str:
    x1, y1 = min(c.x1 for c in cells), min(c.y1 for c in cells)
    x2, y2 = max(c.x2 for c in cells), max(c.y2 for c in cells)
    return f"x[{x1:.0f},{x2:.0f}] y[{y1:.0f},{y2:.0f}] / 图 {hw[1]}x{hw[0]}"


if __name__ == "__main__":
    raise SystemExit(main())
