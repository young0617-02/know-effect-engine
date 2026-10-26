from kee.ingest.admission import admit, checksum_of

BASE = {
    "doc_id": "fin-ok-doc",
    "title": "某城商银行结项报告",
    "industry": "金融",
    "customer": "某城商银行",
    "project": "PRJ-FIN-2024-017",
    "doc_type": "结项报告",
    "source_level": "A",
    "security_level": 2,
    "sections": [
        {"section_no": "1.1", "page": 2, "heading": "目标", "paragraphs": ["银企直联实时清算方案 V3.2 的 RTO 小于十五分钟。"]}
    ],
}


def card(**over):
    c = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in BASE.items()}
    c.update(over)
    return c


def test_accepts_valid_card():
    rep = admit([card()])
    assert len(rep.accepted) == 1 and not rep.rejected
    assert rep.accepted[0]["_checksum"] == checksum_of(BASE)


def test_source_checksum_keeps_vlm_description_out_of_dedupe():
    a = card(_source_checksum="raw-file-sha", sections=[
        {"section_no": "1", "page": 1, "heading": "正文", "paragraphs": ["这是一段足够长的正文，用于通过准入校验。"]},
        {"section_no": "IMG-1", "page": 1, "heading": "图片", "paragraphs": ["【图片 1】内容描述：柱状图上升。"]},
    ])
    b = card(_source_checksum="raw-file-sha", sections=[
        {"section_no": "1", "page": 1, "heading": "正文", "paragraphs": ["这是一段足够长的正文，用于通过准入校验。"]},
        {"section_no": "IMG-1", "page": 1, "heading": "图片", "paragraphs": ["【图片 1】内容描述：柱状图下降。"]},
    ])
    assert checksum_of(a) == checksum_of(b) == "raw-file-sha"
    rep = admit([a, b])
    assert len(rep.accepted) == 1 and len(rep.duplicates) == 1


def test_rejects_bad_industry_doc_type_and_level():
    rep = admit(
        [
            card(industry="医疗"),
            card(doc_type="内部八卦"),
            card(security_level=9),
            card(source_level="D"),
            card(doc_id="BAD ID"),
            card(security_level=2, sections=[]),
        ]
    )
    reasons = "|".join(" ".join(r) for _, r in rep.rejected)
    assert len(rep.rejected) == 6
    for frag in ("industry", "doc_type", "security_level", "source_level", "doc_id", "sections"):
        assert frag in reasons


def test_duplicate_content_is_deduped_by_fingerprint():
    rep = admit([card(), card(doc_id="fin-ok-doc-2", title="换个编号但正文一样")])
    assert len(rep.accepted) == 1
    assert rep.duplicates and rep.duplicates[0][0] == "fin-ok-doc-2"


def test_whitespace_only_change_is_not_a_new_document():
    a = card(sections=[{"section_no": "1.1", "page": 2, "heading": "目标", "paragraphs": ["RTO 小于十五分钟。"]}])
    b = card(
        doc_id="fin-ok-doc-3",
        sections=[{"section_no": "1.1", "page": 2, "heading": "目标", "paragraphs": ["RTO  小于十五分钟 。"]}],
    )
    rep = admit([a, b])
    assert len(rep.accepted) == 1 and len(rep.duplicates) == 1


def test_paragraph_length_bounds_enforced():
    rep = admit([card(sections=[{"section_no": "1.1", "page": 1, "heading": "短", "paragraphs": ["太短。"]}])])
    assert rep.rejected and "段落长度越界" in rep.rejected[0][1][0]
