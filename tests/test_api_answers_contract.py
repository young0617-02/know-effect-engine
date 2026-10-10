"""问答端点的契约面测试：不占 GPU、不依赖模型底座，只验「拒绝得对不对」。

用 skip_embedding=True 起应用：答案链路必须有稠密检索，所以该模式下的正确行为是
明确拒绝（E902/503），而不是静默降级成 sparse-only 出一版更差的答案。
这条区分很重要——降级成稀疏检索在功能上「能跑」，但会把检索质量悄悄换掉。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kee.api.app import create_app

OWNER = {"X-Kee-Role": "knowledge_owner", "X-Kee-Actor": "owner-01"}


@pytest.fixture()
def client():
    with TestClient(create_app(skip_embedding=True)) as c:
        yield c


def test_answers_requires_explicit_identity(client):
    r = client.post("/api/v1/answers", json={"question": "RTO 是多少"})
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == "E401"
    assert body["retryable"] is False
    assert body["trace_id"]


def test_answers_refuses_to_run_without_dense_backend(client):
    """skip_embedding 下必须拒绝而不是偷偷降级成 sparse-only。"""
    r = client.post("/api/v1/answers", json={"question": "RTO 是多少"}, headers=OWNER)
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "E902"
    assert body["retryable"] is True  # E9xx 依赖不可用属于可重试
    assert "稠密检索" in body["message"]


def test_stream_endpoint_has_the_same_guard(client):
    r = client.post("/api/v1/answers/stream", json={"question": "RTO 是多少"}, headers=OWNER)
    assert r.status_code == 503
    assert r.json()["code"] == "E902"


def test_empty_question_is_rejected_before_any_work(client):
    r = client.post("/api/v1/answers", json={"question": ""}, headers=OWNER)
    assert r.status_code == 400
    assert r.json()["code"] == "E400"


def test_top_k_is_bounded(client):
    for bad in (0, 11):
        r = client.post("/api/v1/answers", json={"question": "RTO 是多少", "top_k": bad}, headers=OWNER)
        assert r.status_code == 400, bad


def test_unknown_session_returns_404_not_empty_history(client):
    r = client.get("/api/v1/sessions/s-does-not-exist", headers=OWNER)
    assert r.status_code == 404
    assert r.json()["code"] == "E404"


def test_trace_id_is_echoed_on_error_responses(client):
    r = client.post(
        "/api/v1/answers",
        json={"question": "RTO 是多少"},
        headers={**OWNER, "X-Trace-Id": "trace-keep-001"},
    )
    assert r.status_code == 503
    assert r.json()["trace_id"] == "trace-keep-001"
    assert r.headers["X-Trace-Id"] == "trace-keep-001"
