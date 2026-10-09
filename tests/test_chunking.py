from kee.ingest.chunking import chunk_document

CARD = {
    "doc_id": "fin-test-doc",
    "title": "测试结项报告",
    "industry": "金融",
    "customer": "某城商银行",
    "project": "PRJ-FIN-2024-999",
    "doc_type": "结项报告",
    "source_level": "A",
    "security_level": 2,
    "sections": [
        {
            "section_no": "1.1",
            "page": 3,
            "heading": "可用性目标",
            "paragraphs": [
                "银企直联实时清算方案 V3.2 采用双活部署，RTO 目标小于十五分钟。",
                "头寸管理接口的超时阈值统一设置为五秒，重试三次后转人工介入处理。",
            ],
        },
        {
            "section_no": "1.2",
            "page": 4,
            "heading": "验收口径",
            "paragraphs": [
                "拨备覆盖率不低于 150%。",
            ],
        },
    ],
}


def test_parent_and_child_ids_and_order():
    rows = chunk_document(CARD, "fin-test-doc@v1")
    parents = [r for r in rows if r["is_parent"]]
    children = [r for r in rows if not r["is_parent"]]
    assert len(parents) == 2
    assert parents[0]["id"] == "fin-test-doc@v1#1.1#parent"
    assert all(c["parent_id"] for c in children)
    assert [r["order_no"] for r in rows] == list(range(1, len(rows) + 1))


def test_child_ids_follow_naming():
    children = [r for r in chunk_document(CARD, "fin-test-doc@v1") if not r["is_parent"]]
    assert all(c["id"].startswith("fin-test-doc@v1#") for c in children)
    assert all("#c" in c["id"] for c in children)


def test_child_never_crosses_section():
    children = [r for r in chunk_document(CARD, "fin-test-doc@v1") if not r["is_parent"]]
    for c in children:
        assert c["section_no"] in c["id"]


def test_long_paragraph_is_split_with_overlap():
    card = dict(CARD)
    long_para = "风险分类要求逾期九十天以上归为可疑类，" * 30
    card["sections"] = [
        {"section_no": "2.1", "page": 9, "heading": "长段落", "paragraphs": [long_para, "补充说明段落内容。"]}
    ]
    children = [r for r in chunk_document(card, "fin-test-doc@v1") if not r["is_parent"]]
    assert len(children) > 2
    assert all(len(c["content"]) <= 280 * 3 for c in children)
    # 重叠：相邻子块尾部/首部存在共享片段
    assert any(children[i]["content"][-30:] in children[i + 1]["content"] for i in range(len(children) - 1))


def test_parent_respects_max_length():
    card = dict(CARD)
    card["sections"] = [
        {"section_no": "3.1", "page": 1, "heading": "超长章节", "paragraphs": ["甲方验收意见记录如下，" * 200, "乙方回复。"]}
    ]
    parents = [r for r in chunk_document(card, "fin-test-doc@v1", parent_max=600) if r["is_parent"]]
    assert len(parents[0]["content"]) <= 600
