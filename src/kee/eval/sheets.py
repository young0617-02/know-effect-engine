"""人工工单 CSV 的读取——只为一件事：**护栏必须真的看得见已填的行**。

工单文件开头有几行 `#` 注释（判据原文 + 生成时点），它们不是数据。
直接 `csv.DictReader(f)` 会把**第一行注释当成表头**，于是每一行数据的键都变成那串判据文字，
`row.get("answer_correct__fill_1_or_0")` 恒为 None ⇒ 「已填行数」永远算出 0。

这个形状的后果不是报错，是**护栏静默失效**：`guard_existing_fill()` 的本意是
「有人填过就拒绝覆盖，覆盖等于销毁人工标注」，而它看到的是「一行都没填」——
于是它放行覆盖。两张工单（`answer_correct_fill.csv` 67 行、`hallucination_fill.csv` 12 行）
都是真人判完回收的，是 ECE 与处置正确率那两个读数的唯一来源。

2026-10-15 复跑 `m5_hallucination_sheet.py` 时撞出来的：那次靠 `--force` 才没被拦，
而"没被拦"这件事本身才是缺陷——不 `--force` 它同样会放行。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path


def read_fill_rows(path: Path) -> list[dict[str, str]]:
    """跳过 `#` 注释与空行后按真表头解析。返回的每一行都能用判断列的列名取到值。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        body = "\n".join(ln for ln in f.read().splitlines()
                         if ln.strip() and not ln.lstrip().startswith("#"))
    return list(csv.DictReader(io.StringIO(body)))
