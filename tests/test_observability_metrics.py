"""可观测层的钉桩：标签闭合、表格与生产者同源、收口只计一次、/metrics 的形状。

这个文件里最值钱的不是断言本身，是两条**反向扫描**：
`KNOWN_DEGRADED` / `GATED_LABELS` 是从 `src/` 的真实源码里扫出来对照的。白名单类表格最容易
坏在「代码加了新出口、表格没人记得补」——那之后看板上的 `answered` 会把新出口算进去，
读数看起来一切正常。扫描让这种漏登记当场变红。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from kee.config import get_settings
from kee.obs import metrics as m
from kee.qa.answer import AnswerResult
from kee.retrieval.pipeline import RetrievalResult, RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = (ROOT / "src/kee/retrieval/pipeline.py").read_text(encoding="utf-8")
REWRITE_SRC = (ROOT / "src/kee/retrieval/rewrite.py").read_text(encoding="utf-8")
GRAPH_SRC = (ROOT / "src/kee/qa/graph.py").read_text(encoding="utf-8")


def sample(name: str, **labels: str) -> float:
    """按**样本名** + 标签取当前值；取不到就是 0（Prometheus 的约定：没发生过）。

    匹配 `s.name` 而不是 `metric.name`：Counter 族名是 `kee_qa_rounds`，样本名才是
    `kee_qa_rounds_total`——按族名查会永远查到 0，那种假 0 在看板上没人起疑。
    """
    for metric in m.REGISTRY.collect():
        for s in metric.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return float(s.value)
    return 0.0


def hist_counts(name: str, label: str) -> dict[str, float]:
    """直方图各桶的 `_count` 之和按标签拆开（用来断言「这一阶段真的被记了一次」）。"""
    out: dict[str, float] = {}
    for metric in m.REGISTRY.collect():
        if metric.name != name:
            continue
        for s in metric.samples:
            if s.name.endswith("_count"):
                out[s.labels[label]] = out.get(s.labels[label], 0.0) + float(s.value)
    return out


# ---------- 表格与真实源码必须同源 ----------
def test_degraded_whitelist_covers_every_producer() -> None:
    found = set(re.findall(r'degraded\.append\(f?"([a-z_]+)', PIPELINE_SRC + REWRITE_SRC))
    assert len(found) >= 9, f"扫描器本身失效了（源码格式变了？）：只找到 {sorted(found)}"
    missing = {x for x in found if x not in m.KNOWN_DEGRADED}
    assert not missing, f"pipeline/rewrite 新增降级标记但 KNOWN_DEGRADED 没登记：{missing}"


def test_gated_labels_cover_every_graph_exit() -> None:
    found = set(re.findall(r'"gated": "([a-z_]+)"', GRAPH_SRC))
    assert len(found) >= 6, f"扫描器本身失效了（源码格式变了？）：只找到 {sorted(found)}"
    missing = {x for x in found if x not in m.GATED_LABELS}
    assert not missing, f"图上新开了出口但 GATED_LABELS 没登记：{missing}"


def test_unregistered_gate_becomes_other_not_answered() -> None:
    """反面对照：陌生出口必须落到 `other`，不能被默认值吸收成「答成功了」。"""
    assert m.exit_gate_label("a_brand_new_gate") == "other"
    assert m.exit_gate_label("") == "answered"
    assert m.exit_gate_label("weak_retrieval") == "retrieval_gate"


def test_degraded_class_strips_values_but_keeps_known_reasons() -> None:
    assert m.degraded_class("dense_timeout>2500ms") == "dense_timeout"
    assert m.degraded_class("score_scale_out_of_range:36.9408") == "score_scale_out_of_range"
    assert m.degraded_class("rewrite_over_budget(901ms>900ms)") == "rewrite_over_budget"
    assert m.degraded_class("rewrite_error:TimeoutError") == "rewrite_error"
    # 未登记的原因不能凭空造出一条无限基数的时序
    assert m.degraded_class("brand_new_marker:42") == "other"


# ---------- 收口只计一次 ----------
def _res(status: str = "answer", **kw) -> AnswerResult:
    base = dict(
        question="问",
        status=status,
        confidence={"composite": 0.5, "score_basis": "rule_composite_not_probability"},
        citations=[{"evidence_id": "E1"}],
        timings_ms={"e2e_ms": 1200.0, "retrieval_ms": 300.0},
    )
    base.update(kw)
    return AnswerResult(**base)


def test_observe_qa_result_counts_exactly_one_round() -> None:
    before = sample("kee_qa_rounds_total", status="answer", exit_gate="answered")
    m.observe_qa_result(_res(), gated="")
    assert sample("kee_qa_rounds_total", status="answer", exit_gate="answered") == before + 1
    before_esc = sample("kee_qa_rounds_total", status="escalate", exit_gate="sensitive")
    m.observe_qa_result(_res(status="escalate"), gated="sensitive")
    assert sample("kee_qa_rounds_total", status="escalate", exit_gate="sensitive") == before_esc + 1


def test_only_measured_stages_are_observed() -> None:
    """`timings_ms` 里没有的阶段不补 0——拿 0 填进去会把 p50 拉到地板上，
    而「generate 的 p50 只有几毫秒」这种数一旦出现是查不出来的。"""
    before = hist_counts("kee_qa_stage_seconds", "stage")
    m.observe_qa_result(_res(), gated="")
    after = hist_counts("kee_qa_stage_seconds", "stage")
    assert after["e2e"] == before["e2e"] + 1
    assert after["retrieval"] == before["retrieval"] + 1
    assert after.get("generate", 0.0) == before.get("generate", 0.0)


def test_error_exit_is_counted() -> None:
    before = sample("kee_qa_rounds_total", status="degraded", exit_gate="graph_exception")
    m.observe_error_exit("graph_exception")
    assert sample("kee_qa_rounds_total", status="degraded", exit_gate="graph_exception") == before + 1


def test_retrieval_observation_leaves_no_business_text() -> None:
    """查询原文与表达式进不了标签——这是 `/metrics` 可以不开鉴权的前提。"""
    res = RetrievalResult(
        query="客户名单里张三的薪酬",
        mode="hybrid",
        chunks=[RetrievedChunk(chunk_id="d1#v1#3", text="正文", score=0.9, fields={})],
        timings_ms={},
        expr='security_level <= 2 && project == "某客户"',
        degraded=["dense_timeout>2500ms", "hybrid_single_leg"],
    )
    m.observe_retrieval(res)
    text = generate_latest(m.REGISTRY).decode("utf-8")
    assert "张三" not in text and "薪酬" not in text and "某客户" not in text
    assert 'mode="hybrid"' in text
    assert 'reason="dense_timeout"' in text


# ---------- HTTP 层 ----------
@pytest.fixture()
def client() -> TestClient:
    from kee.api.app import create_app

    with TestClient(create_app(skip_embedding=True)) as c:
        yield c


def test_metrics_endpoint_shape(client: TestClient) -> None:
    body = client.get("/metrics").text
    assert "# TYPE kee_http_requests_total counter" in body
    assert "# TYPE kee_qa_stage_seconds histogram" in body
    assert "kee_dependency_up" in body
    # 抓取自身不计数，否则 QPS 图会被每 15s 一次的抓取灌满
    assert 'route="/metrics"' not in body


def test_route_label_uses_template_not_actual_path(client: TestClient) -> None:
    r = client.get("/api/v1/documents/机密文档XYZ")
    assert r.status_code in (401, 404)
    # 错误体里明写着这个文档 id，但 /metrics 里只能有路由模板
    body = client.get("/metrics").text
    assert 'route="/api/v1/documents/{doc_id}"' in body
    assert "机密文档XYZ" not in body


def test_error_code_counter_has_single_choke_point(client: TestClient) -> None:
    before = sample("kee_api_errors_total", code="E401")
    client.get("/api/v1/auth/me")
    assert sample("kee_api_errors_total", code="E401") == before + 1


def test_metrics_disabled_turns_endpoint_into_404(monkeypatch: pytest.MonkeyPatch) -> None:
    from kee.api.app import create_app

    monkeypatch.setenv("KEE_METRICS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(skip_embedding=True)) as c:
            assert c.get("/metrics").status_code == 404
    finally:
        monkeypatch.delenv("KEE_METRICS_ENABLED", raising=False)
        get_settings.cache_clear()


def test_scrape_probe_failure_does_not_break_metrics() -> None:
    """探针挂了就只标记，不能把 /metrics 打成 500——那时坏掉的恰好是
    「正在看依赖挂没挂」的那只眼（比依赖本身故障更难查）。"""

    def boom() -> None:
        raise RuntimeError("依赖探针炸了")

    before = sample("kee_scrape_probe_errors_total")
    m.set_scrape_probe(boom)
    try:
        # 必须走 render()：直接 generate_latest 会跳过抓取期探针，那条路径上探针炸没炸都不知道
        assert m.render().startswith(b"# HELP")
    finally:
        m.set_scrape_probe(None)
    assert sample("kee_scrape_probe_errors_total") >= before + 1


def test_error_code_labels_keep_their_shape() -> None:
    """错误码是 `/metrics` 上唯一直接来自异常路径的标签：形状不对就归 `other`。

    调用点传的都是源码字面量，但拼错一个码就会多一条永久时间序列，而这种基数泄漏
    只会让看板慢慢变卡，不会在任何断言里报错。
    """
    before = sample("kee_api_errors_total", code="other")
    m.observe_api_error("E-whatever-用户输入")
    assert sample("kee_api_errors_total", code="other") == before + 1
    m.observe_api_error("E902")
    assert sample("kee_api_errors_total", code="E902") > 0


def test_pipeline_returns_all_route_through_emit() -> None:
    """`pipeline.py` 里不许有未接 `_emit()` 的 `return RetrievalResult(`。

    漏一条 = 那条出口的降级计数永久为 0，而「永久为 0」在看板上和「真的没降级」长得一样。
    """
    assert PIPELINE_SRC.count("return _emit(") >= 3
    assert "return RetrievalResult(" not in PIPELINE_SRC


# ---------- 结构化日志 ----------
def test_json_log_line_is_parseable_and_config_is_idempotent(capsys) -> None:
    import json
    import logging

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_kee_logging_configured", False):
            root.removeHandler(h)
    from kee.obs.logging import configure_logging

    h1 = configure_logging("INFO", True)
    h2 = configure_logging("INFO", True)
    assert h1 is h2, "configure_logging 不幂等：验收脚本一晚上调十几次，每条日志会打十几遍"
    logging.getLogger("kee.test").warning("检索超时", extra={"trace_id": "api-xyz", "stage": "dense"})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    row = json.loads(line)
    assert row["msg"] == "检索超时" and row["level"] == "WARNING"
    assert row["trace_id"] == "api-xyz" and row["stage"] == "dense"
    root.removeHandler(h1)


def test_third_party_info_is_not_dumped_onto_the_root(capsys) -> None:
    """给根 logger 装 handler 会把第三方库的 INFO 一起放出来——验收输出会被淹。

    `verify_m1` 第一次跑结构化日志时，每次依赖调用都多一行 `httpx` 的
    `HTTP Request: POST http://testserver/...`，把步骤表冲得看不见。
    所以 `NOISY_LOGGERS` 显式压到 WARNING；反面对照是 `kee.*` 的 INFO 必须仍然出得来。
    """
    import logging

    from kee.obs.logging import NOISY_LOGGERS, configure_logging

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_kee_logging_configured", False):
            root.removeHandler(h)
    handler = configure_logging("INFO", True)
    try:
        logging.getLogger("httpx").info("HTTP Request: POST /api/v1/documents/upload 201")
        logging.getLogger("kee.retrieval").info("稠密腿超时")
        err = capsys.readouterr().err
        assert "HTTP Request" not in err
        assert "稠密腿超时" in err
        assert "httpx" in NOISY_LOGGERS
    finally:
        root.removeHandler(handler)
