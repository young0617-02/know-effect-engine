from __future__ import annotations

from pathlib import Path

import pytest

from kee.ingest.metadata import _doc_id_from, resolve_meta
from kee.ingest.parsers import (
    ParsedDocument,
    ParsedSection,
    UnsupportedFormat,
    _front_matter,
    _split_long,
    parse_file,
    parse_markdown,
    parse_plain_text,
)

MD = """---
title: 测试文档
industry: 能源
project: PRJ-ENE-2025-999
---
# 背景

本项目覆盖 1200 台配电自动化终端的接入，线损率从 6.8% 降至 4.9%。

## 子节

心跳超时阈值为 30 秒，冬季低温下超时率升至 12%。

| 事项 | 期限 |
|---|---|
| 复测 SLA-A1 | 2025-11-30 |
"""


def test_front_matter_extracted_and_stripped():
    meta, body = _front_matter(MD)
    assert meta["industry"] == "能源"
    assert not body.lstrip().startswith("---")
    assert "# 背景" in body


def test_markdown_hierarchy_and_table_flattened():
    doc = parse_markdown(MD, "a.md")
    assert doc.title == "测试文档"
    assert [s.section_no for s in doc.sections] == ["1", "1.1"]
    assert any("复测 SLA-A1" in p for s in doc.sections for p in s.paragraphs)


def test_markdown_without_front_matter_uses_first_h1_as_title():
    doc = parse_markdown("# 真标题\n\n一段足够长的正文内容，用来通过准入的段落长度下限校验。\n\n# 第二节\n\n另一段足够长的正文，用于验证第二个 H1 不被当成标题。", "b.md")
    assert doc.title == "真标题"
    # 首个 H1 既是标题也是 1 号章节，它名下的正文不能被甩进幽灵「正文」章节
    assert [s.section_no for s in doc.sections] == ["1", "2"]
    assert doc.sections[0].heading == "真标题"
    assert any("一段足够长的正文内容" in p for p in doc.sections[0].paragraphs)


def test_plain_text_heading_is_first_line_of_block_not_whole_block():
    raw = "1 作业范围\n变电设备月度巡检覆盖主变与开关柜，单站作业时长控制在 4 小时以内。\n\n2 安全要求\n进入生产控制大区须由运行人员陪同，禁止未报备介质接入 SCADA 主站。"
    doc = parse_plain_text(raw, "c.txt")
    assert [s.section_no for s in doc.sections] == ["1", "2"]
    assert doc.sections[0].paragraphs and "变电设备" in doc.sections[0].paragraphs[0]


def test_number_prefixed_body_line_is_not_a_heading():
    """正文里「26572 规定的限量要求以下。」这类句子不能被判成章节标题。"""
    doc = parse_plain_text("26572 规定的限量要求以下，且不超过 Class A 等级的限值范围说明。", "d.txt")
    assert doc.sections[0].heading != "26572 规定的限量要求以下，且不超过 Class A 等级的限值范围说明。"


def test_split_long_keeps_sentences_under_limit():
    long = "这是一段用于测试长段落切分的中文文本，" * 40
    parts = _split_long(long)
    assert all(len(p) <= 800 for p in parts)
    assert "".join(parts).startswith(long[:20])


def test_normalize_drops_nothing_silently():
    doc = ParsedDocument(title="t", sections=[ParsedSection("1", "标题", None, ["短"])], source_format="text")
    out = doc.normalize()
    assert any("无达标长度段落" in n or "保留最长" in n for n in out.notes) or out.sections[0].paragraphs


def test_unsupported_format_message_names_the_backend_gap(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"fake")
    with pytest.raises(UnsupportedFormat) as e:
        parse_file(p)
    assert "MinerU" in str(e.value)


def test_parse_bytes_without_temp_leak_for_text(tmp_path):
    doc = parse_file(Path("示例.md"), raw=MD.encode("utf-8"), filename="示例.md")
    assert doc.sections and doc.source_format == "markdown"


def test_doc_id_is_deterministic_and_keeps_cjk():
    # 内建 hash() 按进程随机化，绝不能用来生成 doc_id
    assert _doc_id_from("实施手册-v2.DOCX", None) == "实施手册-v2"
    assert _doc_id_from("巡检 作业.txt", None) == "巡检-作业"
    assert _doc_id_from("a_b c.docx", None) == "a_b-c"
    assert _doc_id_from("x.docx", "My-Doc") == "My-Doc"
    assert _doc_id_from("x.docx", None) == _doc_id_from("x.docx", None)


def test_doc_id_survives_admission_id_rule():
    from kee.ingest.admission import check_format

    card = {
        "doc_id": "实施手册-v2",
        "title": "t",
        "industry": "制造",
        "customer": "某客户",
        "project": "PRJ-MFG-1",
        "doc_type": "运维手册",
        "source_level": "A",
        "security_level": 2,
        "sections": [{"section_no": "1", "page": 1, "heading": "h", "paragraphs": ["一段足够长的中文正文内容用于通过长度校验。"]}],
    }
    assert check_format(card) == []


def test_meta_prefers_form_over_front_matter():
    r = resolve_meta({"industry": "金融"}, filename="a.md", parsed_meta={"industry": "能源"}, sample_text="结项")
    assert r.meta["industry"] == "金融"
    assert not any("front-matter" in w for w in r.warnings)


def test_meta_falls_back_to_front_matter_with_warning():
    r = resolve_meta({}, filename="a.md", parsed_meta={"industry": "能源", "project": "PRJ-ENE-1"}, sample_text="")
    assert r.meta["project"] == "PRJ-ENE-1"
    assert any("front-matter" in w for w in r.warnings)


def test_meta_infers_doc_type_and_warns_loudly():
    r = resolve_meta({"industry": "能源", "customer": "某电网", "project": "PRJ-ENE-2"}, filename="x.md", sample_text="本文给出接口编号 IF-FIN-0012 的报文要求")
    assert r.meta["doc_type"] == "接口规范"
    assert any("推断" in w for w in r.warnings)


def test_meta_missing_isolation_fields_are_flagged():
    r = resolve_meta({"industry": "能源"}, filename="x.md", sample_text="运维手册正文")
    assert any("customer 未提供" in w for w in r.warnings)
    assert any("project 未提供" in w for w in r.warnings)


def test_meta_rejects_industry_outside_domain():
    with pytest.raises(ValueError):
        resolve_meta({"industry": "医疗"}, filename="x.md")


def test_secret_hint_raises_security_level():
    r = resolve_meta({"industry": "金融", "customer": "某行", "project": "P", "doc_type": "售前方案"}, filename="x.md", sample_text="附件为客户名单与各自合同金额")
    assert r.meta["security_level"] == 3
    assert any("密级" in w or "security_level" in w for w in r.warnings)
