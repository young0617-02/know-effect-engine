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


# ---- 拒答得说清是「库里没资料」还是「有资料但没复核」 ----
# 这两件事过去共用 `refuse + no_sufficient_evidence` 一个出口，业务读到的是「没找到相关资料」，
# 而真相是「资料在，只是没人复核」——后者是可推进的，前者是真无解。下面三条钉的就是这个区分。

from test_answer_enforcement import FakeLlm, FakeRetriever, chunk, ctx, good_payload  # noqa: E402

UNREV_VERSION = "fin-doc@v1"
C_UNREV = f"{UNREV_VERSION}#2.1#c1"
C_REVIEWED = "fin-doc@v2#2.1#c1"


def test_all_evidence_unreviewed_says_so_instead_of_no_evidence():
    """摘光 ⇒ 新出口标签 + 对外字段 + needs_human，且**不再**冒充 `no_sufficient_evidence`。"""
    eng = AnswerEngine(retriever=FakeRetriever([chunk(C_UNREV, score=0.95)]), llm=FakeLlm(good_payload()),
                       unreviewed_provider=lambda: {UNREV_VERSION})
    res = eng.answer("RTO 是多少", ctx())
    assert res.status == "refuse"
    assert res.refusal_reason == "all_evidence_unreviewed"
    assert res.unreviewed_evidence_count == 1
    assert res.needs_human is True
    assert "all_evidence_unreviewed" in res.notes and "no_sufficient_evidence" not in res.notes
    # SSE 与前端读的是 as_dict()，字段掉在这里就等于对外不存在
    as_dict = res.as_dict()
    assert as_dict["refusal_reason"] == "all_evidence_unreviewed"
    assert as_dict["unreviewed_evidence_count"] == 1


def test_genuine_missing_evidence_still_reports_no_evidence():
    """反面对照：没摘除任何东西、检索本来就空 ⇒ 必须还是 `no_evidence`。
    少了这条，把 refusal_reason 硬编码成 review 也能过上一条测试。"""
    eng = AnswerEngine(retriever=FakeRetriever([]), llm=FakeLlm(good_payload()),
                       unreviewed_provider=lambda: {UNREV_VERSION})
    res = eng.answer("无关问题", ctx())
    assert res.status == "refuse" and res.refusal_reason == "no_evidence"
    assert res.unreviewed_evidence_count == 0 and "no_sufficient_evidence" in res.notes


def test_partial_drop_keeps_answering_but_still_counts():
    """摘掉一部分、剩一条够答 ⇒ 正常出答案，但计数照样带出去：
    「本题还有 N 条相关资料没复核」这句话在答对的时候同样值钱。"""
    eng = AnswerEngine(retriever=FakeRetriever([chunk(C_UNREV), chunk(C_REVIEWED)]),
                       llm=FakeLlm(good_payload()), unreviewed_provider=lambda: {UNREV_VERSION})
    res = eng.answer("RTO 是多少", ctx())
    assert res.status in ("answer", "tentative"), res.status
    assert res.refusal_reason == "" and res.unreviewed_evidence_count == 1


def test_refusal_discloses_count_only_never_titles():
    """用户 2026-10-22 裁决：**拒答只披露数量，不披露是哪份资料**。
    数量是"这事可推进"的信号；标题/文档号是额外的元信息泄漏面（跨部门场景尤其敏感）。
    这条测试就是那条裁决的闸门——日后谁把标题拼进 hint，这里当场红。
    """
    import json

    secret_version = "prd-handover@v9"          # 刻意不与保留块共用文档前缀，否则断言会假失败
    eng = AnswerEngine(retriever=FakeRetriever([chunk(f"{secret_version}#3.1#c1", score=0.93, doc="prd-handover")]),
                       llm=FakeLlm(good_payload()), unreviewed_provider=lambda: {secret_version})
    res = eng.answer("交维前的遗留风险清单是什么", ctx())
    assert res.status == "refuse" and res.refusal_reason == "all_evidence_unreviewed"
    assert res.unreviewed_evidence_count == 1
    blob = json.dumps(res.as_dict(), ensure_ascii=False)
    for forbidden in ("prd-handover", secret_version, f"{secret_version}#3.1#c1"):
        assert forbidden not in blob, f"拒答载荷里出现了被摘资料的标识（{forbidden}）——违反「只给数量」裁决"


def test_pending_reviews_queue_is_scope_filtered(store):
    """待复核队列必须按调用者可见范围过滤：拿最高密级那篇当样本，
    低密级上限的角色不能看见它——队列开给所有读权角色，但范围判定不许另开一套。"""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT v.id AS vid, d.security_level AS lvl FROM document_version v "
            "JOIN document d ON d.id = v.doc_id ORDER BY d.security_level DESC LIMIT 1").fetchone()
    vid, lvl = row["vid"], int(row["lvl"])
    original = None
    with store.connect() as conn:
        original = conn.execute("SELECT reviewed_by FROM document_version WHERE id=%s",
                                (vid,)).fetchone()["reviewed_by"]
        conn.execute("UPDATE document_version SET reviewed_by=NULL WHERE id=%s", (vid,))
    try:
        full = store.pending_reviews(max_security_level=3)
        assert full["total"] >= 1 and any(i["version_id"] == vid for i in full["items"])
        if lvl > 1:
            narrow = store.pending_reviews(max_security_level=lvl - 1)
            assert all(i["version_id"] != vid for i in narrow["items"]), "密级过滤没生效：越权看得见待复核清单"
    finally:  # 生产库里不许留下我摘掉的章
        with store.connect() as conn:
            conn.execute("UPDATE document_version SET reviewed_by=%s WHERE id=%s", (original, vid))
        assert vid not in store.unreviewed_versions()


def test_reviews_pending_endpoint_echoes_scope(store):
    """路由存在且回显范围——运维要靠这个字段判断「N 到底是哪个范围内的 N」。"""
    from fastapi.testclient import TestClient

    from kee.api.app import create_app

    owner = {"X-Kee-Role": "knowledge_owner", "X-Kee-Actor": "owner-01"}
    with TestClient(create_app(skip_embedding=True)) as client:
        r = client.get("/api/v1/reviews/pending", headers=owner)
        assert r.status_code == 200, r.text
        body = r.json()
        assert {"total", "items", "scope"} <= set(body)
        assert body["scope"]["role"] == "knowledge_owner"
        assert client.get("/api/v1/reviews/pending?limit=0", headers=owner).status_code == 400
