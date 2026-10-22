"""token 口径分块（用户 2026-10-22 下发 Child 200/30、Parent 1000/150）的防回归测试。

这里钉的是四条**结构性**约束，不是效果：
1. 子块与父块都是原文**子串**——分块不许改正文，否则引用出去的段落与权威源对不上；
2. `【标题 / 章节】` 抬头是标签，**不计进 token 预算**；
3. 表格仍走行级「列名=值」，**不做 token 切分**（把一行切成两半等于把语义切没）；
4. 父块归属按**字符区间重叠**算，不按字符串相等找——重叠区里同一段文字属于两个父块，`find` 会认错。

真分词器加载不了（权重不在本机）时整组跳过：这条通路**刻意不做字符兜底**，
兜底等于把声明的粒度悄悄换掉，那正是本模块要防的事。
"""

from __future__ import annotations

import pytest

from kee.ingest.chunking import Granularity, chunk_document, granularity_from_settings
from kee.models.tokens import count_tokens


def _tok_probe() -> tuple[int | None, str]:
    """跑一次真实计数。失败原因**原样带进 skip reason**：本机最常见的不是权重缺失，
    而是宿主 API 与测试进程同时载 torch ⇒ Windows 提交空间不够（`WinError 1455 页面文件太小`）。
    只写一句「分词器不可达」会把「环境撞车」报成「功能没做」，跳过也就成了免检的通过。"""
    try:
        return count_tokens("可用性目标"), ""
    except Exception as e:  # noqa: BLE001 - 权重缺失 / 内存撞车 / transformers 不可用都算跑不了
        return None, f"{type(e).__name__}: {str(e)[:180]}"


_TOK_N, _TOK_ERR = _tok_probe()
needs_tok = pytest.mark.skipif(_TOK_N is None, reason=f"BGE-M3 分词器本次载不起来（{_TOK_ERR}）")


class _S:
    def __init__(self, granularity="chars", **kw):
        self.chunk_granularity = granularity
        self.child_tokens = kw.get("child_tokens", 200)
        self.child_overlap_tokens = kw.get("child_overlap_tokens", 30)
        self.parent_tokens = kw.get("parent_tokens", 1000)
        self.parent_overlap_tokens = kw.get("parent_overlap_tokens", 150)


def card(paras, section_no="1.1", heading="可用性目标", title="测试结项报告"):
    return {
        "doc_id": "fin-test-doc",
        "title": title,
        "sections": [{"section_no": section_no, "page": 3, "heading": heading, "paragraphs": list(paras)}],
    }


# ---- 口径选择：默认必须是 legacy ----


def test_default_settings_keep_the_char_path():
    """`chunk_granularity` 默认 `chars` ⇒ 返回 None ⇒ 走 legacy。
    默认值要是被改成 tokens，下一次重索引会悄悄作废现网 547 行标注与全部门禁读数。"""
    assert granularity_from_settings(_S()) is None


def test_tokens_setting_produces_the_declared_granularity():
    g = granularity_from_settings(_S("tokens", child_tokens=200, child_overlap_tokens=30,
                                     parent_tokens=1000, parent_overlap_tokens=150))
    assert (g.child_tokens, g.child_overlap_tokens, g.parent_tokens, g.parent_overlap_tokens) == (200, 30, 1000, 150)
    assert g.scheme_tag() == "+tok:c200/30-p1000/150"


def test_legacy_path_is_unchanged_when_granularity_given_as_none():
    rows = chunk_document(card(["银企直联实时清算方案 V3.2 采用双活部署，RTO 目标小于十五分钟。"]),
                          "fin-test-doc@v1", granularity=None)
    assert [r["id"] for r in rows if r["is_parent"]] == ["fin-test-doc@v1#1.1#parent"]


# ---- 结构约束 ----


@needs_tok
def test_children_are_exact_substrings_of_the_body():
    p = "头寸管理接口的超时阈值统一设置为五秒，重试三次后转人工介入处理。" * 40
    g = Granularity(child_tokens=120, child_overlap_tokens=10, parent_tokens=400, parent_overlap_tokens=40)
    rows = chunk_document(card([p]), "fin-test-doc@v1", granularity=g)
    children = [r for r in rows if not r["is_parent"]]
    assert len(children) > 3, "这条语料在给定预算下必须真的被切开，否则下面所有断言都是空集上的真"
    for r in children:
        piece = r["content"].split("｜", 1)[1]
        assert piece in p
        # 预算约束的是**正文**；`章节标题｜` 是与父块标签同一性质的抬头，不计进预算（口径见下条测试）
        assert count_tokens(piece) <= g.child_tokens, f"子块正文超出 token 预算：{r['id']}"
        assert r["content"] == f"可用性目标｜{piece}"


@needs_tok
def test_parent_label_is_not_charged_to_the_budget():
    """父块正文必须 ≤ 预算，**不含** `【标题 / 章节】` 那行标签。
    标签算进去的话，报告里的「Parent 1000 tokens」量的就不是正文窗口。"""
    p = "拨备覆盖率不低于 150%，逐季按监管口径重算一次并留痕。" * 60
    g = Granularity(child_tokens=100, child_overlap_tokens=10, parent_tokens=200, parent_overlap_tokens=30)
    rows = chunk_document(card([p], heading="很长的章节抬头标题也不该占预算" * 6), "fin-test-doc@v1", granularity=g)
    parents = [r for r in rows if r["is_parent"]]
    assert len(parents) >= 2, "父块必须真的分窗，否则这条测试没测到窗口"
    for r in parents:
        body = r["content"].split("\n", 1)[1]
        assert body in p
        assert count_tokens(body) <= g.parent_tokens, f"父块正文超预算：{r['id']}"


@needs_tok
def test_parent_ownership_uses_span_overlap_not_string_equality():
    """重叠区里的同一段文字同时属于两个父块。按 `find` 认归属会把它固定判给第一个，
    按重叠长度认才会判给真正覆盖它更多的那个。"""
    p = "灾备切换演练每半年一次，RTO 实测值记录在演练报告附表里。" * 50
    g = Granularity(child_tokens=60, child_overlap_tokens=12, parent_tokens=150, parent_overlap_tokens=60)
    rows = chunk_document(card([p]), "fin-test-doc@v1", granularity=g)
    parents = [r["id"] for r in rows if r["is_parent"]]
    children = [r for r in rows if not r["is_parent"]]
    assert len(parents) >= 2
    assert {c["parent_id"] for c in children} <= set(parents)
    assert len({c["parent_id"] for c in children}) >= 2, "跨窗口的子块必须分散到多个父块，否则归属逻辑没被跑到"
    for c in children:
        piece = c["content"].split("｜", 1)[1]
        own = next(r for r in rows if r["id"] == c["parent_id"])
        assert piece in own["content"], f"{c['id']} 的正文不在它所属父块里 ⇒ 归属算错了"


@needs_tok
def test_table_rows_are_never_split_by_token_budget():
    """行级「列名=值」是被实测过的检索单元；把一行切成两半，key=value 的语义就没了。
    预算 20 token 远小于一张 30 行的表，如果表格真走了 token 切分，这里块数会爆到 30 以上。"""
    md = "\n".join(["| 部件 | 铅含量判定 |", "|---|---|"]
                   + [f"| 组件框架单元{i} | 超出限值{i} 微克 |" for i in range(30)])
    g = Granularity(child_tokens=20, child_overlap_tokens=5, parent_tokens=60, parent_overlap_tokens=10)
    rows = chunk_document(card([md], heading="有害物质表"), "d1@v1", granularity=g)
    children = [r for r in rows if not r["is_parent"]]
    assert len(children) == 30
    for i, c in enumerate(children):
        assert c["content"] == f"有害物质表（表行）｜部件=组件框架单元{i}；铅含量判定=超出限值{i} 微克"


@needs_tok
def test_no_content_is_lost_between_parents():
    """父块窗口带重叠 ⇒ 拼接去重后必须仍能覆盖整段正文（漏字就是「分块丢了内容」）。"""
    p = "月结批处理在每月最后一个工作日 22:00 起跑，预计四十分钟完成。" * 30
    g = Granularity(child_tokens=80, child_overlap_tokens=8, parent_tokens=180, parent_overlap_tokens=45)
    rows = chunk_document(card([p]), "fin-test-doc@v1", granularity=g)
    covered = "".join(dict.fromkeys(r["content"].split("\n", 1)[1] for r in rows if r["is_parent"]))
    for probe in (p[:40], p[len(p) // 2: len(p) // 2 + 40], p[-40:]):
        assert probe in covered, f"父块窗口没有覆盖到正文这一段：{probe[:20]}…"
