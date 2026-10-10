"""表格结构还原的多样本验证：合成表格 PDF → 解析链 → 与真值逐格比对。

判定只看两件事：
  1) 结构（行数、列数）对不对 —— 这是「结构还原」的本职；
  2) 单元格文本落位对不对 —— 允许 OCR 字符级错误，但内容不能跑到别的格子里。
OCR 本身的错字不在本步骤的判定范围内（那是 PP-OCRv6 small 模型的精度问题，另记）。
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from table_fixtures import build_all  # noqa: E402

FIX_DIR = ROOT / "var" / "m1" / "fixtures"


def parse_md_table(md: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for ln in md.split("\n"):
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def find_table(doc) -> str:
    for sec in doc.sections:
        for p in sec.paragraphs:
            if p.lstrip().startswith("|"):
                return p
    return ""


def similarity(a: str, b: str) -> float:
    """字符级相似率：OCR 错字按部分得分，不按整格判负。"""
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    hits = sum(1 for ch in b if ch in a)
    return hits / max(len(a), len(b))


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main() -> int:
    from kee.ingest.mineru_onnx import MinerUPdfParser

    FIX_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = build_all(str(FIX_DIR))
    parser = MinerUPdfParser()
    report: list[dict] = []
    all_ok = True
    for fx in fixtures:
        t0 = time.time()
        doc = parser.parse(Path(fx.path).read_bytes(), Path(fx.path).name)
        el = time.time() - t0
        md = find_table(doc)
        got = parse_md_table(md)
        truth = fx.spec.grid
        nrows_ok = len(got) == len(truth)
        ncols_ok = bool(got) and all(len(r) == fx.spec.n_cols for r in got)
        cell_scores: list[float] = []
        misses: list[str] = []
        ws_only = raw_exact = 0
        for ri, trow in enumerate(truth):
            if ri >= len(got):
                misses.append(f"第 {ri} 行整行缺失")
                continue
            gi = 0
            for text, span in trow:
                if gi >= len(got[ri]):
                    break
                s = similarity(text, got[ri][gi])
                cell_scores.append(s)
                if text == got[ri][gi]:
                    raw_exact += 1
                if s < 0.6:
                    misses.append(f"r{ri}c{gi} 期望「{text}」实得「{got[ri][gi]}」")
                elif _norm(text) != text or _norm(got[ri][gi]) != got[ri][gi]:
                    ws_only += 1
                gi += span
        exact = sum(1 for s in cell_scores if s == 1.0)
        # 判定只看结构与落位；OCR 的空格/字符差异另记，不让精度问题冒充结构问题
        ok = nrows_ok and ncols_ok and not misses
        all_ok &= ok
        report.append(
            {
                "样本": fx.spec.name,
                "真值行列": f"{len(truth)}x{fx.spec.n_cols}",
                "还原行列": f"{len(got)}x{len(got[0]) if got else 0}",
                "结构判定": "通过" if (nrows_ok and ncols_ok) else "不通过",
                "落位错误": misses[:6],
                "逐格内容一致(忽略空格)": f"{exact}/{len(cell_scores)}",
                "逐格原样全等": f"{raw_exact}/{len(cell_scores)}",
                "仅空格差异": ws_only,
                "相似率均值": round(sum(cell_scores) / max(1, len(cell_scores)), 3),
                "耗时s": round(el, 1),
                "解析痕迹": [n for n in doc.notes if "表格结构" in n],
            }
        )
        print(f"[{'OK ' if ok else 'BAD'}] {fx.spec.name} 真值{len(truth)}x{fx.spec.n_cols} "
              f"还原{len(got)}x{len(got[0]) if got else 0} 内容一致{exact}/{len(cell_scores)} "
              f"原样全等{raw_exact}/{len(cell_scores)} {el:.1f}s")
        for m in misses[:6]:
            print("      ", m)
    (ROOT / "var" / "m1" / "table_structure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n总体:", "通过" if all_ok else "不通过")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
