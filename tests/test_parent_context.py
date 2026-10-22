"""父块回捞（需求 US-1.2「父块补上下文」）的防回归测试。

与复核闸那组同样的分层：注入规则用桩验证（不打 GPU、不打库），
`pg.parents_content` 那两条打真 PostgreSQL、连不上整组跳过——
「跳过要看得见，不能算通过」。

预算计数刻意 **monkeypatch 掉真分词器**：这里要钉的是「超预算就整块不贴并记数」这条规则，
用 4 字符/token 的确定性近似反而比真实 token 数更稳（真分词器一变，测试会红在没有规则错的地方）。
真实 token 计数的保真度另有 `tests/test_token_granularity.py` 负责。
"""

from __future__ import annotations

import pytest

from kee.models import tokens as tokens_mod
from kee.qa.answer import AnswerEngine, format_evidence
from kee.retrieval.pipeline import RetrievedChunk

from test_answer_enforcement import FakeLlm, FakeRetriever, ctx, good_payload

PARENT_MARK = "【父块补上下文"


@pytest.fixture(autouse=True)
def _cheap_token_count(monkeypatch):
    """把 count_tokens 换成确定性的 len/4，让预算断言不依赖分词器与权重路径。"""
    monkeypatch.setattr(tokens_mod, "count_tokens", lambda t: max(1, len(t or "") // 4))


def child(cid, parent, text, doc="fin-doc", score=0.9):
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        score=score,
        fields={
            "chunk_id": cid,
            "doc_id": doc,
            "project": "PRJ-FIN-2024-017",
            "section_no": "2.1",
            "page": 5,
            "security_level": 2,
            "parent_id": parent,
        },
    )


def run(chunks, parent_provider=None, **over):
    """`parent_context_in_answer` 这里**默认给 True**：产品默认是关的（见下面那条默认值测试），
    而这些用例要测的是「开着的时候规则对不对」，不默认打开它们就全在测同一条空路径。"""
    over.setdefault("parent_context_in_answer", True)
    llm = FakeLlm(good_payload())
    eng = AnswerEngine(FakeRetriever(chunks), llm, parent_provider=parent_provider, **over)
    res = eng.answer("RTO 是多少", ctx())
    return eng, llm, res


def test_shipped_default_is_off_and_reads_the_field_not_the_instance():
    """钉默认值。读 `Settings.model_fields` 而**不读实例**：读实例测的是本机 `.env`/环境变量，
    换台机器就永远绿。裁决依据（用户 2026-10-22「默认关（按实测）」）写在 `kee/config.py` 那段注释里。
    """
    from kee.config import Settings

    assert Settings.model_fields["parent_context_in_answer"].default is False
    assert Settings.model_fields["chunk_granularity"].default == "chars", \
        "改默认 = 让下一次重索引悄悄作废 strict gold 与 547 行绑子块的人工标注"


def test_default_engine_with_provider_injects_nothing():
    """服务默认态：provider 接上了、开关没打开 ⇒ 一条都不贴，notes 里也不许出现父块计数。"""
    pid = "fin-doc@v1#2.1#parent"
    chunks = [child("fin-doc@v1#2.1#c1", pid, "证据正文 c1")]
    llm = FakeLlm(good_payload())
    eng = AnswerEngine(FakeRetriever(chunks), llm, parent_provider=lambda ids: {pid: f"{pid}\n父块独有一句"},
                       parent_context_in_answer=False)
    res = eng.answer("RTO 是多少", ctx())
    assert PARENT_MARK not in llm.last_prompt
    assert not any(n.startswith("parent_context") for n in res.notes)


def test_no_provider_leaves_the_prompt_byte_identical():
    """不接 provider = 完全直通。离线复跑与既有全部测试的行为不许被这道改造悄悄改掉。"""
    chunks = [child("fin-doc@v1#2.1#c1", "fin-doc@v1#2.1#parent", "证据正文 c1")]
    _, llm, _ = run(chunks, parent_provider=None)
    assert PARENT_MARK not in llm.last_prompt


def test_off_flag_leaves_the_prompt_byte_identical():
    chunks = [child("fin-doc@v1#2.1#c1", "fin-doc@v1#2.1#parent", "证据正文 c1")]
    _, llm, _ = run(chunks, parent_provider=lambda ids: {"fin-doc@v1#2.1#parent": "第一节 父块正文\n没被引用的一句"})
    assert PARENT_MARK in llm.last_prompt  # 先证明这条数据确实能触发注入，否则下面的断言是空的
    _, llm2, _ = run(chunks, parent_provider=lambda ids: {
        "fin-doc@v1#2.1#parent": "第一节 父块正文\n没被引用的一句"}, parent_context_in_answer=False)
    assert PARENT_MARK not in llm2.last_prompt


def test_one_parent_is_pasted_once_even_with_three_children():
    """上一版的缺陷就是逐条贴：三块同属一章 ⇒ 同一段在场文字重贴三遍。"""
    pid = "fin-doc@v1#2.1#parent"
    chunks = [child(f"fin-doc@v1#2.1#c{i}", pid, f"证据正文 c{i}") for i in (1, 2, 3)]
    _, llm, res = run(chunks, parent_provider=lambda ids: {pid: f"{pid}\n未被引用的父块独有一句"})
    assert llm.last_prompt.count(PARENT_MARK) == 1
    assert any(n.startswith("parent_context_tokens:") for n in res.notes)


def test_text_already_carried_by_children_is_not_repeated():
    """父块相对子块净增很少（实测中位 28 字），重复内容只有挤窗口的代价 ⇒ 一句不贴。"""
    pid = "fin-doc@v1#2.1#parent"
    chunks = [child(f"fin-doc@v1#2.1#c{i}", pid, f"第{i}句 完整正文") for i in (1, 2)]
    _, llm, res = run(chunks, parent_provider=lambda ids: {pid: f"{pid}\n第1句 完整正文\n第2句 完整正文"})
    assert PARENT_MARK not in llm.last_prompt
    assert not any(n.startswith("parent_context") for n in res.notes)


def test_over_budget_parent_is_dropped_whole_and_counted():
    """超预算**整块不贴**并记数，不做静默截断：本机 num_ctx 会从头部截，
    那时补的不是上下文而是截断，报出来的数字就没有意义了。"""
    p1 = "fin-doc@v1#2.1#parent"
    p2 = "fin-doc@v1#2.2#parent"
    chunks = [child("fin-doc@v1#2.1#c1", p1, "证据正文 c1"), child("fin-doc@v1#2.2#c1", p2, "证据正文 c2")]
    bodies = {p1: f"{p1}\n短", p2: f"{p2}\n" + "长" * 2000}
    _, llm, res = run(chunks, parent_provider=lambda ids: bodies, parent_context_max_tokens=40)
    assert f"{PARENT_MARK} {p1}】" in llm.last_prompt
    assert f"{PARENT_MARK} {p2}】" not in llm.last_prompt
    assert "parent_context_skipped:1" in res.notes


def test_provider_sees_only_parents_of_chunks_that_survived_both_gates():
    """复核闸摘掉的版本，其父块不许再从权威源捞回来——否则「未复核内容不进答案」这条裁决
    会经由父块正文漏实现：子块被摘了，父块里那段原话却贴着答出去了。"""
    unrev = "fin-doc@v1"
    seen: list[list[str]] = []

    def provider(ids):
        seen.append(list(ids))
        return {i: f"{i}\n父块独有一句" for i in ids}

    chunks = [
        child(f"{unrev}#2.1#c1", f"{unrev}#2.1#parent", "未复核的子块正文", doc=unrev),
        child("fin-doc@v2#2.1#c1", "fin-doc@v2#2.1#parent", "已复核的子块正文", doc="fin-doc"),
    ]
    _, _, res = run(chunks, parent_provider=provider, unreviewed_provider=lambda: {unrev})
    assert res.unreviewed_evidence_count == 1
    assert seen and all(not i.startswith(unrev) for batch in seen for i in batch)


def test_missing_parent_row_is_counted_not_silent():
    """库里查不到父块（例如权威源被清理过）也要记数：
    悄悄少贴一块，读数上看不出来，日后对不上账就是又一笔「数字照常出来、量的不是那件事」。"""
    pid = "fin-doc@v1#2.1#parent"
    chunks = [child("fin-doc@v1#2.1#c1", pid, "证据正文 c1")]
    _, llm, res = run(chunks, parent_provider=lambda ids: {})
    assert PARENT_MARK not in llm.last_prompt
    assert "parent_context_skipped:1" in res.notes


@pytest.fixture(scope="module")
def store():
    from kee.db.pg import AuthoritativeStore

    st = AuthoritativeStore()
    try:
        with st.connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - 连不上就是没起容器，如实跳过
        pytest.skip(f"PostgreSQL 不可达，父块回捞取数无法验证：{type(e).__name__}: {e}")
    return st


def test_empty_id_list_does_not_touch_the_database(store):
    """空列表直接返回空字典——不然每次「没有父块」的问答都要白打一趟库。"""
    assert store.parents_content([]) == {}


def test_parents_content_returns_parent_rows_only(store):
    """真库里按 id 取父块正文：这条通路是需求 US-1.2 唯一的取数点，桩测不出来。"""
    store.init_schema()
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, content FROM chunk WHERE is_parent = true ORDER BY id LIMIT 2").fetchall()
    if not rows:
        pytest.skip("库里没有父块行，无法验证取数通路")
    got = store.parents_content([r["id"] for r in rows] + ["不存在的@v1#9.9#parent"])
    assert set(got) == {r["id"] for r in rows}          # 查不到的键不返回，不返回空串冒充
    assert all(got[r["id"]] == r["content"] for r in rows)
