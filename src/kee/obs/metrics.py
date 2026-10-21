"""Prometheus 指标层：把「只能翻表、翻日志才看得见」的东西换成可被抓取的时序。

三条口径纪律，不遵守就别看这些数：

1. **看板上的 P95 不是门禁那条 P95。** 这里的分位数由 `histogram_quantile()` 在固定的桶上
   线性插值而来，是**近似值**；① 端到端延迟那条门禁（6s、仅作答轮次、first_visible 口径）的
   权威读数仍由 `scripts/` 下的离线探针产出。看板用来看趋势与告警，不用来对外报数。
2. **标签一律闭合枚举。** 查询文本、用户名、会话 id、文档 id、分数值都不进标签——高基数会把
   Prometheus 压垮，而这些恰恰是最不该外泄的字段；数值一律走直方图。唯一像开放集合的是
   `degraded` 的原因类，已按生产者白名单收敛（`KNOWN_DEGRADED`，并有单测从真实源码反向核对）。
3. **注册表是本模块私有的**，不用 `prometheus_client` 的全局默认注册表：同进程里 `create_app()`
   会被调很多次（契约测试、验收脚本、宿主 API 各一次），全局注册表第二次就撞名，
   而撞名后的症状是 `/metrics` 「部分可用」——那种半坏最难被发现。

计时口径的两处如实交代：
  * `kee_http_request_seconds{route="/api/v1/answers/stream"}` 量的是**响应头/首字节**，
    不含整条 SSE 流（`BaseHTTPMiddleware` 在 `call_next` 返回时就结束了）；整轮耗时看
    `kee_qa_stage_seconds{stage="e2e"}`。
  * 命中 Redis 检索缓存的那一轮，`degraded` 与改写来历在 `pipeline` 里就被抹掉了
    （既有边界，见 `docs/技术方案与MVP设计`），所以缓存命中轮的降级计数**必然为 0**——
    这是「用户当时看到的」口径，不是统计漏了。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

# 统一错误码的形状：`E400` / `E901` / `E500`。见 `observe_api_error`。
_CODE_RE = re.compile(r"E\d{2,3}")

# 在线链路的真实分布很宽：缓存命中 0ms，冷启动嵌入要加载权重实测 86.9s、整轮 98.3s。
# 桶按这个跨度铺，否则 +Inf 里装着绝大多数样本，p95 就等于没测。
BUCKETS_LATENCY = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0, 150.0)
# 文档上传要过解析 + 嵌入，尾巴比问答还长（整库重灌能到几分钟），单独一套桶。
BUCKETS_HTTP = BUCKETS_LATENCY + (240.0, 600.0, 1200.0)
# 0.20 是 `min_retrieval_score` 那条检索闸门，桶边界刻意含它：
# `le="0.2"` 直接就是「没过闸」的占比，不用另立指标。
BUCKETS_SCORE = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BUCKETS_COUNT = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0)

ANSWER_STATUSES = ("answer", "tentative", "refuse", "escalate", "degraded")

# 早退闸门 → 指标标签。键是 `kee/qa/graph.py` 里 `state["gated"]` 的**实际取值**（不是 notes 的
# 子串——notes 是给人看的，会夹带 `secret_level_cited` 这类与出口无关的东西，拿它判出口迟早看走眼）。
# 空串表示七个节点全走完、没有被任何闸门终结。
# `tests/test_observability_metrics.py` 反向扫 graph.py 的 `"gated": "<值>"` 字面量：
# 图上开了新出口而这张表没登记，测试当场红——「一共有几个出口」不能靠读代码确认。
GATED_LABELS = {
    "": "answered",
    "sensitive": "sensitive",
    "weak_retrieval": "retrieval_gate",
    "generation_failed": "generation_failed",
    "unparseable": "unparseable",
    "declared_no_evidence": "declared_no_evidence",
    "no_citation": "no_citation",
    "marginal_evidence_no_derivation": "marginal_evidence_refusal",
    "empty_conclusion": "empty_conclusion",
    # 不是图上的 gated，但确实是「这轮没产出答案」的一条出口：流式 worker 捕获到图抛异常。
    "graph_exception": "graph_exception",
}

# `degraded` 标记的原因类白名单，逐条来自真实生产者：
# `kee/retrieval/pipeline.py`（dense_timeout / dense_empty / dense_error / sparse_error /
# sparse_rw_error / rerank_error / hybrid_single_leg / retrieval_total_failure /
# score_scale_out_of_range）与 `kee/retrieval/rewrite.py`（rewrite_error / rewrite_over_budget /
# rewrite_bad_output / rewrite_json_salvaged）。
KNOWN_DEGRADED = frozenset(
    {
        "dense_timeout",
        "dense_empty",
        "dense_error",
        "sparse_error",
        "sparse_rw_error",
        "rerank_error",
        "hybrid_single_leg",
        "retrieval_total_failure",
        "score_scale_out_of_range",
        "rewrite_error",
        "rewrite_over_budget",
        "rewrite_bad_output",
        "rewrite_json_salvaged",
    }
)

# `timings_ms` 的键 → 指标上的 stage 标签。刻意用固定表而不是把键名直接当标签：
# 键是内部实现细节（`stage_sparse_rw_ms` 这种），进时序标签后改名就是断线。
STAGE_KEYS = {
    "e2e_ms": "e2e",
    "retrieval_ms": "retrieval",
    "generate_ms": "generate",
    "first_token_ms": "first_token",
    "rewrite_ms": "rewrite",
    "stage_embed_ms": "embed",
    "stage_dense_ms": "dense",
    "stage_sparse_ms": "sparse",
    "stage_sparse_rw_ms": "sparse_rw",
    "stage_rerank_ms": "rerank",
    "stage_fuse_ms": "fuse",
    "stage_cache_ms": "cache_read",
    "stage_rewrite_ms": "rewrite_leg",
}

HTTP_ROUTES_UNMATCHED = "__unmatched__"

REQUESTS = Counter(
    "kee_http_requests_total",
    "HTTP 请求数（按路由模板 / 方法 / 状态码）。不含 /metrics 自身。",
    ["route", "method", "status"],
    registry=REGISTRY,
)
REQUEST_SECONDS = Histogram(
    "kee_http_request_seconds",
    "HTTP 处理耗时（秒）。SSE 端点量到响应头为止，整轮耗时看 kee_qa_stage_seconds。",
    ["route"],
    buckets=BUCKETS_HTTP,
    registry=REGISTRY,
)
API_ERRORS = Counter(
    "kee_api_errors_total",
    "统一错误体里带 code 的失败数（E4xx 客户端 / E5xx 服务端 / E9xx 依赖）。",
    ["code"],
    registry=REGISTRY,
)

QA_ROUNDS = Counter(
    "kee_qa_rounds_total",
    "问答轮次（按出口状态 / 早退闸门）。answered 表示七个节点全走完。",
    ["status", "exit_gate"],
    registry=REGISTRY,
)
QA_STAGE_SECONDS = Histogram(
    "kee_qa_stage_seconds",
    "在线链路各阶段耗时（秒）。与 session_turn / 离线探针同源，但分位数是桶上近似值。",
    ["stage"],
    buckets=BUCKETS_LATENCY,
    registry=REGISTRY,
)
QA_TOP_SCORE = Histogram(
    "kee_qa_top_score",
    "每轮检索的最高分分布（重排 sigmoid 分，∈(0,1)）。le=0.2 即没过检索闸门那一档。",
    buckets=BUCKETS_SCORE,
    registry=REGISTRY,
)
QA_COMPOSITE = Histogram(
    "kee_qa_composite_score",
    "规则合成分分布。**不是概率**（basis 标签明写 rule_composite_not_probability），"
    "且实测 93% 是常数偏移，用来看漂移不要用来判对错。",
    ["basis"],
    buckets=BUCKETS_SCORE,
    registry=REGISTRY,
)
QA_CITATIONS = Histogram(
    "kee_qa_citation_count",
    "每轮对外给出的引用条数。",
    buckets=BUCKETS_COUNT,
    registry=REGISTRY,
)
QA_STREAM_RETRACTIONS = Counter(
    "kee_qa_stream_retractions_total",
    "流式撤回：provisional_retracted 的结论级轮次数与 provisional_lost 的条目数。",
    ["kind"],
    registry=REGISTRY,
)

RETRIEVAL_ROUNDS = Counter(
    "kee_retrieval_rounds_total",
    "检索轮次。cached 三值而不是两值：`disabled` 是**没配缓存**（/search 端点与所有离线评测跑的都是这条），"
    "把它并进 `miss` 会让「缓存命中率」的分母里混进一批根本不该参与命中计算轮次——那条比率是 F 因子"
    "那条门禁的报值口径，分母混了就是读数错。",
    ["mode", "cached"],
    registry=REGISTRY,
)
RETRIEVAL_DEGRADED = Counter(
    "kee_retrieval_degraded_total",
    "检索降级标记数（按原因类，白名单收敛）。",
    ["reason"],
    registry=REGISTRY,
)
CACHE_OPS = Counter(
    "kee_search_cache_ops_total",
    "Redis 缓存操作数：kind=search|rewrite，result=hit|miss|error|store。",
    ["kind", "result"],
    registry=REGISTRY,
)

DEPENDENCY_UP = Gauge("kee_dependency_up", "依赖自检结果（1 可用 / 0 不可用），抓取时实测。", ["dep"], registry=REGISTRY)
MILVUS_CHUNKS = Gauge("kee_milvus_chunks", "Milvus 集合里的分块数，抓取时实测。", registry=REGISTRY)
PG_DOCUMENTS = Gauge("kee_pg_documents", "权威源 document 表行数，抓取时实测。", registry=REGISTRY)
PROBE_ERRORS = Counter(
    "kee_scrape_probe_errors_total",
    "抓取期依赖探针异常数（探针挂了只标记不抛，否则 /metrics 会 500）。",
    registry=REGISTRY,
)

LANGFUSE_EVENTS = Counter(
    "kee_langfuse_events_total",
    "Langfuse 上报事件计数：enqueued 入队 / flushed 成功 / error 失败 / dropped 队列满丢弃 / disabled 未启用。",
    ["result"],
    registry=REGISTRY,
)
LANGFUSE_QUEUE = Gauge("kee_langfuse_queue_depth", "Langfuse 上报队列当前深度。", registry=REGISTRY)
# 所有 counter/gauge 都是**进程内累计、不落库**的：API 重启即归零。这条起点让「累计值」与
# 「increase(…[窗口])」之间可以对账——`scripts/m5_observability_probe.py` 用它推窗口宽度，
# 否则同一份脚本换个时间重跑，对账结论会随进程年龄翻转（那种翻转看起来像指标坏了）。
PROCESS_START = Gauge("kee_process_start_timestamp_seconds", "本进程启动时刻（Unix 秒）。", registry=REGISTRY)
PROCESS_START.set(time.time())


def _closed(value: str, allowed: Iterable[str]) -> str:
    return value if value in set(allowed) else "other"


def degraded_class(note: str) -> str:
    """把 `dense_timeout>2500ms` / `score_scale_out_of_range:36.9408` / `rewrite_error:TimeoutError`
    这类带数值的标记收敛成原因类。不收敛的话每个新分数就是一个新时间序列。"""
    head = note.split(":", 1)[0].split(">", 1)[0].split("(", 1)[0].strip()
    return head if head in KNOWN_DEGRADED else "other"


def exit_gate_label(gated: str | None) -> str:
    """`state["gated"]` 的取值翻成闭合标签；没登记过的一律 `other`（宁可多一条 `other` 时序，
    也不要让一个陌生的出口在图上伪装成 `answered`）。"""
    return GATED_LABELS.get(str(gated or ""), "other")


def observe_api_error(code: str) -> None:
    """错误码计数。`code` 必须长得像 `E4xx`：所有调用点传的都是源码里的字面量，
    但这条是 `/metrics` 上唯一直接来自异常处理路径的标签，所以宁可再收一道——
    一个拼错的码会变成一条永久时间序列，而这种基数泄漏只会让看板慢慢变卡，不会报错。
    """
    API_ERRORS.labels(code=code if _CODE_RE.fullmatch(str(code)) else "other").inc()


def observe_retrieval(res: Any, cached: str = "miss") -> None:
    """在 `HybridRetriever.retrieve` 的三个出口各调一次：模式与降级原因。

    这里**不记分数**：`kee_qa_top_score` 记在 `graph.retrieve` 里闸门实际比较的那个数上
    （复核闸会先把未复核证据摘掉，摘之前的最高分与判定用的最高分可以不同）。
    """
    RETRIEVAL_ROUNDS.labels(mode=res.mode, cached=cached).inc()
    for note in res.degraded or ():
        RETRIEVAL_DEGRADED.labels(reason=degraded_class(str(note))).inc()


def observe_top_score(value: float) -> None:
    """检索闸门实际比较的最高分。`le="0.2"` 那一档就是「没过闸」的占比，不用另立指标。"""
    QA_TOP_SCORE.observe(float(value))


def observe_cache(kind: str, result: str) -> None:
    CACHE_OPS.labels(kind=kind, result=result).inc()


def observe_qa_result(res: Any, gated: str = "") -> None:
    """图收口处（`result_from_state`）调用，出口标签取 `state["gated"]`。
    流式与非流式共用这一个收口，因此两条链路同口径、一次问答只计一次。"""
    status = _closed(str(getattr(res, "status", "")), ANSWER_STATUSES)
    QA_ROUNDS.labels(status=status, exit_gate=exit_gate_label(gated)).inc()
    timings = getattr(res, "timings_ms", None) or {}
    for key, stage in STAGE_KEYS.items():
        val = timings.get(key)
        if isinstance(val, (int, float)) and val >= 0:
            QA_STAGE_SECONDS.labels(stage=stage).observe(float(val) / 1000.0)
    conf = getattr(res, "confidence", None) or {}
    comp = conf.get("composite")
    if isinstance(comp, (int, float)):
        QA_COMPOSITE.labels(basis=str(conf.get("score_basis") or "unknown")).observe(float(comp))
    QA_CITATIONS.observe(float(len(getattr(res, "citations", None) or [])))


def observe_error_exit(kind: str) -> None:
    """图抛出异常时的收口：`result_from_state` 根本没跑到，这轮在 `kee_qa_rounds_total` 里
    本来会静默失踪——失踪的轮次比多出来的轮次更危险，因为它让分母变小而读数看起来正常。"""
    QA_ROUNDS.labels(status="degraded", exit_gate=exit_gate_label(kind)).inc()


def observe_stream_outcome(retracted: bool, lost: int) -> None:
    """`provisional_retracted` / `provisional_lost` 由 `iter_qa_events` 在收口**之后**才写进载荷，
    所以只能在这里单独计一次，不能塞进 `observe_qa_result`（那样永远是 0）。"""
    if retracted:
        QA_STREAM_RETRACTIONS.labels(kind="conclusion").inc()
    if isinstance(lost, int) and lost:
        QA_STREAM_RETRACTIONS.labels(kind="item").inc(lost)


def observe_langfuse(result: str, amount: int = 1) -> None:
    LANGFUSE_EVENTS.labels(result=result).inc(amount)


_scrape_probe: Callable[[], None] | None = None


def set_scrape_probe(fn: Callable[[], None] | None) -> None:
    """登记抓取期探针（依赖自检）。由 `create_app()` 注入，因为它才持有 kb/pg 句柄。"""
    global _scrape_probe
    _scrape_probe = fn


def run_scrape_probe() -> None:
    if _scrape_probe is None:
        return
    try:
        _scrape_probe()
    except Exception:  # noqa: BLE001 - 探针失败只标记，不能把 /metrics 打成 500
        PROBE_ERRORS.inc()


def render() -> bytes:
    """`/metrics` 的响应体。先跑探针再序列化：探针写的是同注册表里的 Gauge，
    顺序反了就会把**上一次**抓取的依赖状态报出去（`collect()` 按注册顺序产出，插不进去）。"""
    run_scrape_probe()
    return generate_latest(REGISTRY)


def route_label(scope: dict, path: str) -> str:
    """用路由模板当标签（`/api/v1/documents/{doc_id}`），不用实际路径：
    后者带 doc id / 会话 id，基数无界。"""
    route = scope.get("route")
    template = getattr(route, "path", None)
    return template or (HTTP_ROUTES_UNMATCHED if path != "/metrics" else "/metrics")
