"""Langfuse 上报器的实测：载荷形状、鉴权、失败不抛、越域拒发。

这里**不连真 Langfuse**：自托管要拉 ClickHouse + Postgres + web 一整套镜像（GB 级），
而本机约束是「大件不重复下载、C 盘紧」。所以用一个本地 HTTP 采集器把
`/api/public/ingestion` 的**契约面**测实——路径、Basic 鉴权、批量事件形状、部分拒绝计数。
真容器那一侧能不能收下这些事件，属未验证项，如实记在 `reports/m5/M5_可观测接入实测.md`。
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kee.config import Settings
from kee.obs import metrics as m
from kee.obs.langfuse import LangfuseExporter, in_domain_host
from kee.qa.answer import AnswerResult


def _wait(exp: LangfuseExporter, key: str, at_least: int, timeout: float = 8.0) -> bool:
    """后台线程是异步落账的，`flush_now` 只保证队列排空、不保证这一批已经写完计数。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if exp.stats.get(key, 0) >= at_least:
            return True
        time.sleep(0.05)
    return False


def _sink_factory(response: dict, code: int = 207):
    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(n)
            captured.append({
                "path": self.path,
                "auth": self.headers.get("authorization") or self.headers.get("Authorization"),
                "json": json.loads(raw.decode("utf-8")),
            })
            body = json.dumps(response).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # 静音：测试里不需要 access log
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, captured


def _res(status: str = "answer") -> AnswerResult:
    return AnswerResult(
        question="终验通过率是多少",
        status=status,
        conclusion="终验一次通过率 93.7%（E1）",
        key_points=[{"point": "一次通过率 93.7%", "evidence_ids": ["E1"]}],
        citations=[{"evidence_id": "E1", "doc_id": "d1", "score": 0.91}],
        confidence={"composite": 0.7971, "retrieval": 0.91, "score_basis": "rule_composite_not_probability"},
        retrieval={"mode": "hybrid", "cache_hit": False, "degraded": ["dense_timeout>2500ms"],
                   "n_chunks": 3, "top_chunk_ids": ["d1#v1#3"], "query_used": "终验通过率"},
        llm={"ok": True, "prompt_chars": 4200, "metrics": {"prompt_tokens": 1800, "completion_tokens": 120,
                                                            "done_reason": "stop"}},
        timings_ms={"e2e_ms": 8150.0, "retrieval_ms": 3000.0, "generate_ms": 5000.0, "rewrite_ms": 150.0},
        trace_id="api-abc123",
        notes=["初步判断（低置信度显式降级）"],
    )


@pytest.fixture()
def sink():
    srv, captured = _sink_factory({"successList": [], "errorList": []})
    yield srv, captured
    srv.shutdown()
    srv.server_close()


def _settings(srv: HTTPServer, **kw) -> Settings:
    base = dict(
        langfuse_enabled=True,
        langfuse_host=f"http://127.0.0.1:{srv.server_port}",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_flush_interval_sec=0.2,
        langfuse_max_batch=20,
        langfuse_timeout_sec=3.0,
    )
    base.update(kw)
    return Settings(**base)


def test_ingestion_payload_shape_and_basic_auth(sink) -> None:
    srv, captured = sink
    exp = LangfuseExporter(_settings(srv))
    try:
        assert exp.enabled is True
        exp.record_qa(_res(), ctx=None, session_id="s-1", actor="verify-owner", stream=True)
        assert _wait(exp, "flushed", exp.stats["enqueued"]), exp.stats
    finally:
        exp.close()
    assert captured, "一个批次都没送到：上报链路根本没跑"
    req = captured[0]
    assert req["path"] == "/api/public/ingestion"
    assert req["auth"] == "Basic " + base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    batch = req["json"]["batch"]
    types = [e["type"] for e in batch]
    assert types.count("trace-create") == 1
    assert "generation-create" in types
    assert {"span-create"} <= set(types)
    trace = next(e for e in batch if e["type"] == "trace-create")
    # trace/span/generation 必须挂在同一个 traceId 上，否则 Langfuse 里是一条条孤儿
    ids = {e["body"]["traceId"] for e in batch if e["type"] != "trace-create"}
    assert ids == {trace["body"]["id"]}
    assert trace["body"]["input"] == "终验通过率是多少"
    assert trace["body"]["sessionId"] == "s-1"
    assert trace["body"]["metadata"]["residual_unmeasured_ms"] >= 0
    gen = next(e for e in batch if e["type"] == "generation-create")
    assert gen["body"]["usage"] == {"input": 1800, "output": 120, "unit": "TOKENS"}
    assert gen["body"]["metadata"]["done_reason"] == "stop"
    assert exp.stats["flushed"] == len(batch)


def test_refuse_round_has_no_generation_span(sink) -> None:
    """没进生成节点就不该编造一次模型调用——`generate_ms` 缺失时 generation 事件必须不出现。"""
    srv, captured = sink
    exp = LangfuseExporter(_settings(srv))
    try:
        r = _res("refuse")
        r.timings_ms.pop("generate_ms", None)
        exp.record_qa(r)
        assert _wait(exp, "flushed", exp.stats["enqueued"])
    finally:
        exp.close()
    batch = captured[0]["json"]["batch"]
    assert [e["type"] for e in batch].count("generation-create") == 0
    assert any(e["type"] == "span-create" and e["body"]["name"] == "qa.retrieve" for e in batch)


def test_partial_rejection_is_counted_not_hidden() -> None:
    """批量接口是 2xx + 逐条结果：整体成功不等于每条都落了库，`errorList` 必须进计数。"""
    srv, captured = _sink_factory({"successList": [1], "errorList": [2, 3]})
    exp = LangfuseExporter(_settings(srv))
    try:
        exp.record_qa(_res())
        assert _wait(exp, "errors", 2)
        assert "partial_rejected" in (exp.stats["last_error"] or "")
    finally:
        exp.close()
        srv.shutdown()
        srv.server_close()


def test_dead_sink_never_breaks_the_answer() -> None:
    """采集器挂了（端口都不存在）时，`record_qa` 只计数不抛——观测手段不许反噬被观测对象。"""
    exp = LangfuseExporter(Settings(
        langfuse_enabled=True, langfuse_host="http://127.0.0.1:1",
        langfuse_public_key="pk", langfuse_secret_key="sk",
        langfuse_timeout_sec=1.0, langfuse_flush_interval_sec=0.2,
    ))
    try:
        exp.record_qa(_res())
        assert _wait(exp, "errors", 1), exp.stats
        assert exp.stats["last_error"]
    finally:
        exp.close()


def test_disabled_exporter_is_a_noop() -> None:
    exp = LangfuseExporter(Settings(langfuse_enabled=False))
    assert exp.enabled is False
    assert exp.stats["enqueued"] == 0
    exp.record_qa(_res())
    exp.flush_now(0.1)
    exp.close()


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:30090", True),
        ("http://localhost:3000", True),
        ("http://host.docker.internal:30090", True),
        ("http://192.168.1.20:30090", True),
        ("http://10.0.0.8:30090", True),
        ("https://cloud.langfuse.com", False),
        ("http://8.8.8.8:30090", False),
        ("http://langfuse.example.com", False),
        ("ftp://127.0.0.1", False),
        ("", False),
    ],
)
def test_in_domain_gate(url: str, ok: bool) -> None:
    assert in_domain_host(url)[0] is ok


def test_public_host_is_refused_even_when_enabled(sink) -> None:
    """云端 SaaS 那扇门必须由代码关上，不能靠部署说明里的一句「请不要配公网」。"""
    srv, _ = sink
    exp = LangfuseExporter(_settings(srv, langfuse_host="https://cloud.langfuse.com"))
    try:
        assert exp.enabled is False
        assert "越域" in exp.disabled_reason
        assert exp.status()["reason"] == exp.disabled_reason
    finally:
        exp.close()


def test_allow_remote_override_is_explicit(sink) -> None:
    """越过那道闸门只能靠一个名字里写着「allow remote」的开关——它存在本身就是一条风险登记。"""
    srv, _ = sink
    exp = LangfuseExporter(_settings(srv, langfuse_host="https://cloud.langfuse.com",
                                     langfuse_allow_remote=True))
    try:
        assert exp.enabled is True
    finally:
        exp.close()


def test_span_times_ordered_and_generate_not_duplicated(sink) -> None:
    """起止方向与重复上报——两条都是**实采**照出来的，不是推演出来的。

    第一版把「距本轮结束的毫秒偏移」当成起点用，实采的 trace 里出现
    `qa.retrieve 03:42:26 → 03:41:36`（结束早于开始）；同时 `qa.generate`
    既出 span 又出 generation，同一条阶段在 Langfuse 里挂两遍。
    """
    srv, captured = sink
    exp = LangfuseExporter(_settings(srv))
    try:
        exp.record_qa(_res())
        assert _wait(exp, "flushed", exp.stats["enqueued"]), exp.stats
    finally:
        exp.close()
    batch = captured[0]["json"]["batch"]
    timed = [e for e in batch if e["type"] in ("span-create", "generation-create")]
    assert timed, "一条带完整 timings 的结果应该出 span"
    for e in timed:
        b = e["body"]
        assert b["startTime"] < b["endTime"], f"{b['name']} 结束早于开始：{b['startTime']} → {b['endTime']}"
    names = [e["body"]["name"] for e in timed]
    assert names.count("qa.generate") == 1, f"同一阶段重复上报：{names}"
    by = {e["body"]["name"]: e["body"] for e in timed}
    assert by["qa.retrieve"]["endTime"] <= by["qa.generate"]["startTime"], "阶段顺序被排反了"


def test_metrics_counters_follow_exporter_stats(sink) -> None:
    srv, _ = sink
    before = sample_langfuse("flushed")
    exp = LangfuseExporter(_settings(srv))
    try:
        exp.record_qa(_res())
        assert _wait(exp, "flushed", exp.stats["enqueued"])
        assert sample_langfuse("flushed") >= before + 1
    finally:
        exp.close()


def sample_langfuse(result: str) -> float:
    for metric in m.REGISTRY.collect():
        for s in metric.samples:
            if s.name == "kee_langfuse_events_total" and s.labels.get("result") == result:
                return float(s.value)
    return 0.0
