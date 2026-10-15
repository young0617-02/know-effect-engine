"""工单「拒绝覆盖已填标注」这道护栏自己的回归测试。

起因（2026-10-15 复跑 `m5_hallucination_sheet.py` 时撞出来）：两张人工工单开头都有几行
`#` 判据注释，而两道护栏都用裸 `csv.DictReader(f)` 读它 —— 第一行注释被当成表头，
于是 `row.get("answer_correct__fill_1_or_0")` 恒为 None，「已填行数」永远算出 0，
**护栏在有人填过的表上也放行覆盖**。它保护的东西恰好是不可再生的人工标注。

这个缺陷的形状值得记一下：它不会让任何读数变错，也不会报错，
只会在某次"顺手重跑一下生成脚本"时把 67 行 / 12 行真人判断抹掉，
而事后从产物上看不出来——因为文件仍然"格式正确"。

真工单全程只读：写测试一律在 tmp_path 的副本上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from kee.eval.sheets import read_fill_rows  # noqa: E402

COMMENT = "# 判据：只判「这条结论相对它自己引用的证据」对不对。1=对，0=不对，?=判不了"
HEADER = "qid,status,answer_correct__fill_1_or_0"


def write_sheet(path: Path, filled: list[str]) -> Path:
    body = "\n".join(f"q-{i:03d},answer,{v}" for i, v in enumerate(filled, start=1))
    path.write_text(f"{COMMENT}\n{HEADER}\n{body}\n", encoding="utf-8-sig", newline="")
    return path


def test_read_fill_rows_skips_the_comment_header(tmp_path):
    p = write_sheet(tmp_path / "s.csv", ["1", "0", ""])
    rows = read_fill_rows(p)
    assert len(rows) == 3, "行数按数据算，不是按注释行算"
    assert rows[0]["answer_correct__fill_1_or_0"] == "1"


def test_naive_dictreader_is_what_breaks_the_guard(tmp_path):
    """把缺陷本身钉住：不跳注释就会把注释行当表头，已填数读成 0。"""
    import csv

    p = write_sheet(tmp_path / "s.csv", ["1", "1", "1"])
    with open(p, encoding="utf-8-sig", newline="") as f:
        naive = list(csv.DictReader(f))
    assert all(r.get("answer_correct__fill_1_or_0") is None for r in naive)
    assert len(read_fill_rows(p)) == 3


@pytest.mark.parametrize("mod,fill_col", [
    ("m4_build_correct_sheet", "answer_correct"),
    ("m5_hallucination_sheet", "evidence_sufficient__fill_1_or_0"),
])
def test_guards_refuse_to_overwrite_a_filled_sheet(tmp_path, monkeypatch, mod, fill_col):
    mod_obj = __import__(mod)
    sheet = tmp_path / ("correct_fill.csv" if "correct" in mod else "hallucination_fill.csv")
    write_sheet(sheet, ["1", "0", ""]) if "correct" in mod else _hallu_sheet(sheet)
    monkeypatch.setattr(mod_obj, "FILL", sheet)
    why = mod_obj.guard_existing_fill(False)
    assert why, f"{mod} 的护栏在已填过的表上放行了——它瞎了"
    assert "拒绝" in why or "填过" in why


def _hallu_sheet(path: Path) -> Path:
    cols = ["evidence_sufficient__fill_1_or_0", "behavior_correct__fill_1_or_0", "failure_kind"]
    body = "\n".join(f"q-{i:03d},refuse,0,1," for i in range(1, 4))
    path.write_text(f"{COMMENT}\n{'qid,status,' + ','.join(cols)}\n{body}\n",
                    encoding="utf-8-sig", newline="")
    return path


@pytest.mark.parametrize("mod", ["m4_build_correct_sheet", "m5_hallucination_sheet"])
def test_guards_still_allow_a_first_run(tmp_path, monkeypatch, mod):
    """反面对照：没填过就该放行。只测「拒绝」这一边，会把护栏改成永远拒绝也测不出来。"""
    mod_obj = __import__(mod)
    sheet = tmp_path / "fill.csv"
    write_sheet(sheet, ["", "", ""]) if "correct" in mod else _hallu_sheet_blank(sheet)
    monkeypatch.setattr(mod_obj, "FILL", sheet)
    assert mod_obj.guard_existing_fill(False) is None


def _hallu_sheet_blank(path: Path) -> Path:
    cols = ["evidence_sufficient__fill_1_or_0", "behavior_correct__fill_1_or_0", "failure_kind"]
    body = "\n".join(f"q-{i:03d},refuse,,," for i in range(1, 4))
    path.write_text(f"{COMMENT}\n{'qid,status,' + ','.join(cols)}\n{body}\n",
                    encoding="utf-8-sig", newline="")
    return path


@pytest.mark.parametrize("name,expect", [("answer_correct_fill.csv", 67),
                                         ("hallucination_fill.csv", 12)])
def test_the_real_sheets_are_seen_as_filled(name, expect):
    """最强的一条：拿**仓库里真的人工工单**验护栏看得见。

    行数写死是有意的——67 与 12 分别是 #40 与 #57 回收的人工标注行数，
    少了就说明有人覆盖过或漏了；这条测试与"读数是否变化"无关，它守的是证据本身。
    """
    p = ROOT / "var" / "gate0" / name
    if not p.exists():
        pytest.skip(f"工单不在树里（{p.name}）——跳过不是通过")
    rows = read_fill_rows(p)
    assert len(rows) == expect, f"{name} 行数 {len(rows)} ≠ {expect}"
    filled = [r for r in rows
              if any((v or "").strip() for k, v in r.items() if k.endswith("__fill_1_or_0"))]
    assert len(filled) == expect, f"{name} 有 {expect - len(filled)} 行没被看见填过——护栏会放行覆盖"
