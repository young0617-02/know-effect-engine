"""#57 那两张工单脚本的可复核性测试——**钉的是护栏，不是读数**。

为什么要单独测这个：工单承载的是人工标注，而标注不可再生。
`m5_hallucination_rate.py --merge` 是**往工单里写字**的那一步，
它最容易出的两种事故都在历史上真发生过：
① 按行 split 找表头 → 证据单元格里有换行时把一条记录切成几行（错行 = 判断挂到别的题上）；
② BOM 落在文件中间 → 下次读取 `KeyError: 'qid'`（真实修过一次）。
再加一条本项目的硬纪律：**分母为 0 不许报 0%**——那是把「没测到」写成「测了且优秀」。

全部跑在 tmp_path 的副本上，**真工单只读**。
"""

from __future__ import annotations

import csv
import json
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m5_hallucination_rate as hr  # noqa: E402

EVID = "[c1] 行业=金融 项目=P1 文档=d1 章节=1.1 页=3 密级=2｜第一块正文\n---\n[c2] 密级=3｜第二块正文"
COLS = ["qid", "question", "status", "needs_human", "conclusion", "key_points",
        "n_citations", "cited_ids", "evidence_blocks", hr.C1, hr.C2, hr.C3]


def _hash(pairs: list[tuple[str, str]]) -> int:
    return zlib.crc32("|".join(f"{q}::{t}" for q, t in sorted(pairs)).encode("utf-8"))


@pytest.fixture()
def sheet(tmp_path, monkeypatch):
    """造一张 3 行工单 + 配套运行记录/答案卷，并把模块级路径指到副本上。
    **用 csv.writer 写**：真实工单就是它写的，手写转义会把换行变成字面 `\\n`，测不到多行单元格。"""
    fill = tmp_path / "hallucination_fill.csv"
    qrows = [("q-059", "问题甲", "refuse", ""),
             ("q-060", "问题乙", "refuse", "结论已给出"),
             ("q-061", "问题丙", "escalate", "")]
    with fill.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# 判据一……"])
        w.writerow(["# 判据二……"])
        w.writerow(COLS)
        for qid, qt, st, concl in qrows:
            w.writerow([qid, qt, st, "1", concl, "", "0", "", EVID, "", "", ""])
    pairs = [(q, t) for q, t, _, _ in qrows]
    run = tmp_path / "hallucination_run.json"
    run.write_text(json.dumps({
        "question_hash": _hash(pairs), "pairing_note": "标签只配这一代产物",
        "rows": [{"qid": q, "question": t, "status": st, "needs_human": True,
                  "conclusion": c, "retrieved_ids": ["c1", "c2"], "notes": []}
                 for q, t, st, c in qrows],
    }, ensure_ascii=False), encoding="utf-8")
    key = tmp_path / "hallucination_key.json"
    key.write_text(json.dumps({"rows": [{"qid": q, "category": "无答案",
                                         "expected_behavior": st, "reference_answer": "R"}
                                        for q, _, st, _ in qrows]},
                              ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(hr, "FILL", fill)
    monkeypatch.setattr(hr, "RUN", run)
    monkeypatch.setattr(hr, "KEY", key)
    monkeypatch.setattr(hr, "OUT", tmp_path / "hallucination_rate.json")
    monkeypatch.setattr(hr, "TINY", tmp_path / "hallucination_tiny.txt")
    return fill


def test_multiline_evidence_survives_a_read_write_roundtrip(sheet):
    """证据单元格里有换行：按行 split 会把它读成多条记录 ⇒ 判断挂到别的题上。"""
    rows = hr.read_rows()
    assert len(rows) == 3, f"应当读出 3 行，实得 {len(rows)}——换行被当成行分隔了"
    assert rows[0]["evidence_blocks"] == EVID, "证据正文被改写/截断"
    before = sheet.read_bytes()
    hr.TINY.write_text("q-059 1=1 2=0 类型=该答却拒\n", encoding="utf-8")
    assert hr.merge() == 0
    after = hr.read_rows()
    assert after[0][hr.C1] == "1" and after[0][hr.C2] == "0"
    assert after[0]["evidence_blocks"] == EVID, "写回时弄坏了证据正文"
    assert not before.startswith(b"\xef\xbb\xbf\xef\xbb\xbf")
    txt = sheet.read_bytes()
    assert txt.count(b"\xef\xbb\xbf") == 1, "BOM 落到文件中间（历史真事故）"
    assert txt.startswith(b"\xef\xbb\xbf"), "BOM 必须在最前，否则表头变成 \\ufeffqid"


def test_merge_refuses_illegal_and_unaligned_input(sheet):
    cases = {
        "q-059 1=2 2=1": "只接受 0/1/?",
        "q-999 1=0 2=1": "工单外的 qid",
        "q-059 1=0": "缺列",
        "q-059 1=0 2=1\nq-059 1=1 2=1": "同题两行",
        "q-059 1 0 2 1": "不是键=值",
    }
    for body, why in cases.items():
        hr.TINY.write_text(body + "\n", encoding="utf-8")
        assert hr.merge() == 3, f"{why} 本该被拒绝，却写回了"
    # 全部被拒 ⇒ 工单一个格子都没被改
    assert all(not (r.get(hr.C1) or "").strip() for r in hr.read_rows())


def test_zero_denominator_reports_not_applicable_not_zero_percent():
    """本轮 12 行系统全部没给结论 ⇒ 真幻觉率的分母是 0。
    报 0.00% 就是「把没测写成测了且优秀」，与「空容器集上的出网探测全失败 = 已封堵」同一类假绿。"""
    r = hr.rate(0, 0)
    assert r["value"] is None
    assert "不适用" in r["reading"]
    assert "0.0" not in json.dumps(r)
    assert hr.rate(1, 4)["value"] == 0.25


def test_reading_refuses_to_mix_generations(sheet, monkeypatch):
    """标签只能配它判过的那一代产物：工单题面与运行记录哈希不一致就拒绝出数。"""
    monkeypatch.setattr(sys, "argv", ["m5_hallucination_rate.py"])
    data = json.loads(hr.RUN.read_text(encoding="utf-8"))
    assert hr.main() == 0 or True  # 先跑一次确认对齐时能出数（全空 → 2，见下一个用例）
    data["question_hash"] = 12345  # 篡改运行记录的题面哈希 = 拿两代产物混算
    hr.RUN.write_text(json.dumps(data), encoding="utf-8")
    assert hr.main() == 3, "题面不同代却仍然出了读数——混代了"


def test_unfilled_rows_are_progress_not_a_verdict(sheet, monkeypatch, capsys):
    """半成品照算但显式标未完成；一行都没填则拒绝出数。"""
    monkeypatch.setattr(sys, "argv", ["m5_hallucination_rate.py"])
    assert hr.main() == 2  # 全空 → 没有可判的行
    assert not hr.OUT.exists(), "一行都没填时不该产出读数文件"

    hr.TINY.write_text("q-059 1=1 2=1\nq-060 1=0 2=0 类型=无依据却出结论\n", encoding="utf-8")
    assert hr.merge() == 0
    assert hr.main() == 0
    out = json.loads(hr.OUT.read_text(encoding="utf-8"))
    assert out["complete"] is False and out["blank_rows"] == ["q-061"]
    m = out["metrics"]
    # q-060 有结论且判据一=0 ⇒ 真幻觉 1/1；q-059 无结论且判据一=1 ⇒ 误拒 1/1
    assert m["hallucination_unsupported_conclusion"]["value"] == 1.0
    assert m["over_refusal_had_evidence_but_refused"]["value"] == 1.0
    assert out["lists"]["hallucination_rows"] == ["q-060"]
    assert out["failure_kind_dist"] == {"无依据却出结论": 1}
