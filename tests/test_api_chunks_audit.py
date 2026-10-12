"""M4 新增的两个读接口：引用片段批量读 + 审计读。

两条都直接打权威源（PostgreSQL），**不 mock**：这两个端点的全部价值就在「按身份收敛读取范围」，
把数据库换成假对象就只剩「函数被调用了」这种什么都证明不了的断言。
PG 不可达时整组跳过并在输出里说明——跳过要看得见，不能算成通过。

`skip_embedding=True` 起应用：这两个端点不碰稠密检索，用不到 GPU，也不该去抢显存。
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from kee.api.app import create_app
from kee.db.pg import AuthoritativeStore

ADMIN = {"X-Kee-Role": "system_admin", "X-Kee-Actor": "admin-01"}
ENGINEER = {"X-Kee-Role": "delivery_engineer", "X-Kee-Actor": "eng-01"}
NEWCOMER = {"X-Kee-Role": "newcomer", "X-Kee-Actor": "new-01"}

# 生效版本 + 子块，且密级恰好是新人看不见 / 管理员可见的两端
_SQL_SECRET = (
    "SELECT c.id FROM chunk c JOIN document d ON d.id = c.doc_id "
    "JOIN document_version v ON v.id = c.version_id AND v.status = 'effective' "
    "WHERE d.security_level = 3 AND c.is_parent = false ORDER BY c.id LIMIT 1"
)
_SQL_OPEN = (
    "SELECT c.id FROM chunk c JOIN document d ON d.id = c.doc_id "
    "JOIN document_version v ON v.id = c.version_id AND v.status = 'effective' "
    "WHERE d.security_level <= 1 AND c.is_parent = false ORDER BY c.id LIMIT 1"
)


@pytest.fixture(scope="module")
def store() -> AuthoritativeStore:
    st = AuthoritativeStore()
    try:
        with st.connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - 连不上就是没起容器，如实跳过
        pytest.skip(f"PostgreSQL 不可达，这两个读接口无法验证：{type(e).__name__}: {e}")
    return st


@pytest.fixture()
def client():
    with TestClient(create_app(skip_embedding=True)) as c:
        yield c


def _first_id(store: AuthoritativeStore, sql: str) -> str | None:
    with store.connect() as conn:
        row = conn.execute(sql).fetchone()
    return row["id"] if row else None


def test_chunks_batch_requires_explicit_identity(client):
    r = client.post("/api/v1/chunks/batch", json={"ids": ["whatever"]})
    assert r.status_code == 401
    assert r.json()["code"] == "E401"


def test_chunks_batch_caps_the_id_list(client):
    """一次最多 50 个 id：不设上限等于给人一个批量拖正文的入口。"""
    r = client.post("/api/v1/chunks/batch", json={"ids": [f"x{i}" for i in range(51)]}, headers=ADMIN)
    assert r.status_code == 400
    assert r.json()["code"] == "E400"


def test_chunks_batch_hides_what_the_role_cannot_see(client, store):
    """同一个 chunk id，在两个角色下结果必须不同——这条断言是自反的：
    把 SQL 里的密级条件删掉，它当场就红。"""
    secret = _first_id(store, _SQL_SECRET)
    if not secret:
        pytest.skip("语料里没有密级 3 的子块，越权面无法验证")

    as_newcomer = client.post("/api/v1/chunks/batch", json={"ids": [secret]}, headers=NEWCOMER).json()
    assert as_newcomer["items"] == [], "新人不该拿到密级 3 的正文"
    assert as_newcomer["hidden"] == [secret]

    as_admin = client.post("/api/v1/chunks/batch", json={"ids": [secret]}, headers=ADMIN).json()
    assert len(as_admin["items"]) == 1
    assert as_admin["items"][0]["content"], "管理员拿到的应该是正文本身而不是空串"
    # 越权与不存在合并成同一个答案，不告诉调用者属于哪一类
    assert "exists" not in str(as_newcomer).lower()


def test_chunks_batch_honours_industry_scope(client, store):
    """行业范围外的块读不到。必须用库里**真实的行业取值**：
    先前版本拿 chunk id 前缀（fin/ene/chm）当行业名传，那是个必然匹配不到的值，
    测试会因为完全错误的原因变绿——等于没测。"""
    open_id = _first_id(store, _SQL_OPEN)
    if not open_id:
        pytest.skip("语料里没有密级 ≤1 的子块")
    with store.connect() as conn:
        mine = conn.execute(
            "SELECT d.industry FROM chunk c JOIN document d ON d.id = c.doc_id WHERE c.id = %s", (open_id,)
        ).fetchone()["industry"]
        others = [
            r["industry"]
            for r in conn.execute(
                "SELECT DISTINCT industry FROM document WHERE industry <> %s ORDER BY industry", (mine,)
            ).fetchall()
        ]
    if not others:
        pytest.skip("语料只剩一个行业，行业维度无法验证")

    # 头里只能放 latin-1，中文取值一律百分号编码（与前端 identityHeaders 的约定对称）
    outside = client.post(
        "/api/v1/chunks/batch",
        json={"ids": [open_id]},
        headers={**ENGINEER, "X-Kee-Industries": quote(others[0])},
    ).json()
    assert outside["items"] == [] and outside["hidden"] == [open_id], (
        f"行业 {others[0]} 的范围里不该读到属于 {mine} 的 {open_id}"
    )

    inside = client.post(
        "/api/v1/chunks/batch",
        json={"ids": [open_id]},
        headers={**ENGINEER, "X-Kee-Industries": quote(mine)},
    ).json()
    assert len(inside["items"]) == 1, f"同一条在 {mine} 范围内应当可读（编码/解码不闭环时这里会红）"


def test_plain_ascii_scope_value_still_works(client, store):
    """纯 ASCII 的范围值不编码也要照旧可用：编码约定不能变成「只有会编码的客户端能连」。"""
    open_id = _first_id(store, _SQL_OPEN)
    if not open_id:
        pytest.skip("语料里没有密级 ≤1 的子块")
    r = client.post(
        "/api/v1/chunks/batch",
        json={"ids": [open_id]},
        headers={**ENGINEER, "X-Kee-Projects": "PRJ-NOSUCH-9999"},
    )
    assert r.status_code == 200
    assert r.json()["items"] == [] and r.json()["hidden"] == [open_id]


def test_chunks_batch_leaves_an_audit_trail(client, store):
    """引用片段的读取必须留痕：文档 DoD 要「引用」这一环也可追溯。"""
    open_id = _first_id(store, _SQL_OPEN)
    if not open_id:
        pytest.skip("无可读子块")
    actor = "audit-check-01"
    client.post("/api/v1/chunks/batch", json={"ids": [open_id]}, headers={**ADMIN, "X-Kee-Actor": actor})
    rows = store.list_audit(actor=actor, action="chunks_read", limit=5)["items"]
    assert rows, "chunks_read 没落审计"
    assert rows[0]["payload"]["returned"] == 1


def test_audit_read_is_admin_only(client):
    denied = client.get("/api/v1/audit", headers=ENGINEER)
    assert denied.status_code == 403
    body = denied.json()
    assert body["code"] == "E403" and body["retryable"] is False and body["trace_id"]

    allowed = client.get("/api/v1/audit?limit=5", headers=ADMIN)
    assert allowed.status_code == 200
    data = allowed.json()
    assert data["scope"]["role"] == "system_admin"
    assert len(data["items"]) <= 5


def test_audit_rows_come_back_newest_first(client, store):
    r = client.get("/api/v1/audit?limit=3", headers=ADMIN).json()
    ids = [row["id"] for row in r["items"]]
    assert ids == sorted(ids, reverse=True), "按 id 倒序是翻页不重不漏的前提"


def test_audit_limit_bounds(client):
    assert client.get("/api/v1/audit?limit=0", headers=ADMIN).json()["code"] == "E400"
    assert client.get("/api/v1/audit?limit=201", headers=ADMIN).json()["code"] == "E400"
