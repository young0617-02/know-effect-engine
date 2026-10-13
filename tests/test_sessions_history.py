"""会话历史端点：左侧「会话历史」栏读的是这两条，判的是**归属**而不是角色。

不跑答案链路（那要占显存），直接往权威源里写轮次再读回来——这里要验的是
「列表按 actor 收窄」「详情带得出完整答案整包」「别人的会话读不到」这三条读路径，
跟生成无关，跑真链路只会让测试慢十倍而不多证明任何东西。
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from kee.api.app import create_app
from kee.db.pg import AuthoritativeStore

SUF = uuid.uuid4().hex[:8]
STORE = AuthoritativeStore()


def sid(kind: str) -> str:
    return f"t-hist-{kind}-{SUF}"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(skip_embedding=True)) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def seeded():
    """两个身份各一轮会话 + 一轮空会话（空的那条不该出现在列表里）。"""
    for s in (sid("mine"), sid("theirs"), sid("empty")):
        STORE.open_session(s, "actor-a" if s != sid("theirs") else "actor-b", {"role": "pm"})
    STORE.append_turn(sid("mine"), {
        "question": "SIS 表决逻辑是什么？", "status": "answer", "conclusion": "2oo3 表决",
        "trace_id": "tr-a",
        "answer": {"status": "answer", "conclusion": "2oo3 表决",
                   "key_points": [{"point": "三取二", "evidence_ids": ["E1"]}],
                   "citations": [{"evidence_id": "E1", "chunk_id": "doc-a@v1#1.1#c1",
                                  "doc_id": "doc-a", "section_no": "1.1", "page": 3, "security_level": 2}],
                   "confidence": {"composite": 0.7813, "score_basis": "rule_composite_not_probability"}},
    })
    STORE.append_turn(sid("theirs"), {"question": "别人的问题", "status": "answer",
                                      "conclusion": "不该被 actor-a 读到", "trace_id": "tr-b"})
    yield
    with STORE.connect() as conn:
        conn.execute("DELETE FROM session WHERE id LIKE %s", (f"t-hist-%-{SUF}",))


A = {"X-Kee-Role": "pm", "X-Kee-Actor": "actor-a"}
B = {"X-Kee-Role": "pm", "X-Kee-Actor": "actor-b"}


def test_list_only_returns_the_callers_own_sessions(client):
    ids = [i["id"] for i in client.get("/api/v1/sessions", headers=A).json()["items"]]
    assert sid("mine") in ids
    assert sid("theirs") not in ids, "列表按 actor 收窄，不是按角色"


def test_empty_session_is_not_listed(client):
    ids = [i["id"] for i in client.get("/api/v1/sessions", headers=A).json()["items"]]
    assert sid("empty") not in ids, "没有轮次的会话在侧栏里是一条点不开的空白"


def test_list_row_carries_what_the_sidebar_needs_to_render(client):
    row = next(i for i in client.get("/api/v1/sessions", headers=A).json()["items"] if i["id"] == sid("mine"))
    assert row["n_turns"] == 1
    assert row["title"] == "SIS 表决逻辑是什么？", "侧栏的标题取首轮问题，不是会话 id"
    assert row["last_status"] == "answer"
    assert row["last_at"]


def test_detail_returns_the_full_answer_payload(client):
    """恢复历史必须带整包：只回结论的话，要点/引用落点/规则分会在「翻回旧会话」时凭空消失。"""
    turns = client.get(f"/api/v1/sessions/{sid('mine')}", headers=A).json()["turns"]
    ans = turns[0]["answer"]
    assert ans["key_points"][0]["point"] == "三取二"
    assert ans["citations"][0]["doc_id"] == "doc-a"
    assert ans["confidence"]["score_basis"] == "rule_composite_not_probability"


def test_reading_another_actors_session_is_forbidden(client):
    """会话 id 是 `s-<12 hex>`，猜中不难——挡它的理由是「不是你的」，不是「你猜不到」。"""
    assert client.get(f"/api/v1/sessions/{sid('mine')}", headers=B).status_code == 403
    assert client.get(f"/api/v1/sessions/{sid('theirs')}", headers=A).status_code == 403


def test_detail_of_missing_session_is_404_not_403(client):
    r = client.get("/api/v1/sessions/s-nope", headers=A)
    assert r.status_code == 404 and r.json()["code"] == "E404"


def test_list_requires_identity(client):
    r = client.get("/api/v1/sessions")
    assert r.status_code == 401 and r.json()["code"] == "E401"


def test_reading_a_session_leaves_an_audit_row(client):
    """列表不落审计（逐条列表会把表灌成噪声），但**打开某一条**必须留痕——读自己的历史也一样要留。"""
    client.get(f"/api/v1/sessions/{sid('mine')}", headers=A)
    rows = STORE.list_audit(actor="actor-a", action="session_read", target=sid("mine"))["items"]
    assert rows, "session_read 没落审计：读历史这个动作在系统里等于没发生过"
