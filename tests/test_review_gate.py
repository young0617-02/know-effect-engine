"""复核闸（处置 C）的防回归测试：章盖在**版本**上，闸装在**答案层**。

分两层是刻意的：
- `gate_unreviewed` 的单元断言不需要 PG、不需要 GPU，永远会跑——它钉的是「摘哪几条、摘完走哪条出口」；
- 涉及 schema 迁移与 `review(pass)` 盖章的那几条**打真 PostgreSQL、不 mock**：
  复核章的全部价值就在于它是库里的一行事实，换成假对象就只剩「函数被调用了」。
  PG 连不上就整组跳过，**跳过要看得见，不能算通过**。
"""

from __future__ import annotations

import pytest

from kee.db.pg import AuthoritativeStore
from kee.qa.answer import AnswerEngine
from kee.retrieval.pipeline import RetrievalResult, RetrievedChunk


def _res(chunk_ids: list[str]) -> RetrievalResult:
    chunks = [
        RetrievedChunk(chunk_id=c, score=0.9, text=f"正文 {c}",
                       fields={"doc_id": c.split("@")[0], "section_no": "1.1", "page": 1,
                               "security_level": 1, "project": "PRJ-X-0001", "parent_id": ""})
        for c in chunk_ids
    ]
    return RetrievalResult(query="q", mode="hybrid", chunks=chunks, timings_ms={"retrieval_total_ms": 1.0}, expr="")


class _FakeRetriever:
    def __init__(self, res):
        self.res = res

    def retrieve(self, *a, **k):
        return self.res


def _engine(provider=None):
    eng = AnswerEngine(retriever=_FakeRetriever(_res([])), llm=object(), unreviewed_provider=provider)
    return eng


def test_no_provider_means_no_gate():
    """离线复跑与既有测试的行为不能被这道闸悄悄改掉：不接 provider 就是完全直通。"""
    eng = _engine(provider=None)
    res = _res(["a@v1#1.1#c1", "b@v2#1.1#c1"])
    assert eng.gate_unreviewed(res) == 0
    assert len(res.chunks) == 2


def test_empty_unreviewed_set_leaves_evidence_untouched():
    eng = _engine(provider=lambda: set())
    res = _res(["a@v1#1.1#c1"])
    assert eng.gate_unreviewed(res) == 0 and len(res.chunks) == 1


def test_unreviewed_version_dropped_but_reviewed_kept():
    """按**版本**摘，不是按文档摘：a@v1 未复核、a@v2 已复核时只掉 v1。"""
    eng = _engine(provider=lambda: {"a@v1"})
    res = _res(["a@v1#1.1#c1", "a@v2#1.1#c1", "b@v1#1.1#c1"])
    assert eng.gate_unreviewed(res) == 1
    assert [c.chunk_id for c in res.chunks] == ["a@v2#1.1#c1", "b@v1#1.1#c1"]


def test_gate_is_per_version_not_per_document():
    """章在版本上：只未复核那一个版本被摘，同文档其余版本照旧。
    这条是防止「一次复核放行该文档以后所有版本」——那正是投毒路径的形状。"""
    eng = _engine(provider=lambda: {"doc@v3"})
    res = _res(["doc@v1#1.1#c1", "doc@v2#1.1#c1", "doc@v3#1.1#c1"])
    assert eng.gate_unreviewed(res) == 1
    assert all(c.chunk_id != "doc@v3#1.1#c1" for c in res.chunks)


def test_all_evidence_dropped_empties_chunks_for_downstream_refusal():
    """全摘光时 chunks 变空——下游既有「无充分依据」出口会据此拒答，不新造一条路径。"""
    eng = _engine(provider=lambda: {"a@v1"})
    res = _res(["a@v1#1.1#c1"])
    assert eng.gate_unreviewed(res) == 1 and res.chunks == []


@pytest.fixture(scope="module")
def store() -> AuthoritativeStore:
    st = AuthoritativeStore()
    try:
        with st.connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - 连不上就是没起容器，如实跳过
        pytest.skip(f"PostgreSQL 不可达，复核章无法验证：{type(e).__name__}: {e}")
    return st


def test_migration_backfills_existing_versions(store):
    """存量必须回填成已复核：不回填就等于让现有语料一夜之间全部「无依据可答」——
    那是断服务，不是加闸。回填的 `reviewed_by` 明确写的是迁移身份，不是某个人的复核结论。"""
    store.init_schema()
    unrev = store.unreviewed_versions()
    assert isinstance(unrev, set)
    # 生产语料全部是迁移前入库的，回填后不应有任何未复核版本
    with store.connect() as conn:
        seeded = conn.execute("SELECT count(*) AS n FROM document_version WHERE reviewed_by IS NULL").fetchone()["n"]
    assert seeded == 0, f"有 {seeded} 个存量版本没被回填，它们会突然从答案里消失"


def test_mark_reviewed_flips_only_that_version(store):
    with store.connect() as conn:
        row = conn.execute("SELECT id FROM document_version ORDER BY id LIMIT 1").fetchone()
    vid = row["id"]
    original = None
    with store.connect() as conn:
        original = conn.execute("SELECT reviewed_by FROM document_version WHERE id=%s", (vid,)).fetchone()["reviewed_by"]
        conn.execute("UPDATE document_version SET reviewed_by=NULL WHERE id=%s", (vid,))
    try:
        assert vid in store.unreviewed_versions()
        store.mark_reviewed(vid, "reviewer-test")
        assert vid not in store.unreviewed_versions()
        with store.connect() as conn:
            got = conn.execute("SELECT reviewed_by FROM document_version WHERE id=%s", (vid,)).fetchone()["reviewed_by"]
        assert got == "reviewer-test"
    finally:  # 复原：测试不许在生产库里留下章
        with store.connect() as conn:
            conn.execute("UPDATE document_version SET reviewed_by=%s WHERE id=%s", (original, vid))
        assert vid not in store.unreviewed_versions()


def test_versions_of_exposes_the_stamp(store):
    """管理页要靠这个字段判断「哪一版还没复核」，所以章必须在 versions_of 里可见。"""
    doc = store.list_documents(status="active", limit=1)["items"][0]["id"]
    vs = store.versions_of(doc)
    assert vs and all("reviewed_by" in v and "reviewed_at" in v for v in vs)
