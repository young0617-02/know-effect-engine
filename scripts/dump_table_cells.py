from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kee.ingest.docimg import render_pdf_pages  # noqa: E402
from kee.ingest.mineru_onnx import (  # noqa: E402
    Box,
    LayoutDetector,
    OcrEngine,
    TableStructureModel,
    assign_lines_to_cells,
)

pdf = Path(sys.argv[1])
row_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
data = pdf.read_bytes()
bgr = dict(render_pdf_pages(data, dpi=150))[1]
reg = [r for r in LayoutDetector().detect(bgr) if r.label == "table"][0]
ox, oy = int(reg.x1), int(reg.y1)
crop = bgr[oy : int(reg.y2), ox : int(reg.x2)]
print("region", [round(v) for v in (reg.x1, reg.y1, reg.x2, reg.y2)], "crop", crop.shape[1], crop.shape[0])
rows, spans = TableStructureModel().structure(crop)
for i, r in enumerate(rows[: 3 + row_i]):
    print(f"row{i}", [[round(v) for v in (b.x1, b.y1, b.x2, b.y2)] for b in r])
print("spans", [[round(v) for v in (b.x1, b.y1, b.x2, b.y2)] for b in spans])

ocr = OcrEngine()
plines = ocr.det_boxes(crop)
for ln in plines:
    ln.text = ocr.rec_text(crop[max(0, int(ln.y1)) : int(ln.y2), max(0, int(ln.x1)) : int(ln.x2)])
flat = [b for r in rows for b in r]
owner = assign_lines_to_cells(flat, plines)
print("lines:", len(plines))
for i, (b, grp) in enumerate(zip(flat, owner)):
    if i > 12:
        break
    print(
        f"cell{i} box={[round(v) for v in (b.x1, b.y1, b.x2, b.y2)]}",
        "<<<",
        [(round((g.x1 + g.x2) / 2), round((g.y1 + g.y2) / 2), g.text) for g in grp],
    )
