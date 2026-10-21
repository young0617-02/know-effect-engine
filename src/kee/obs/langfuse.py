"""Langfuse 链路上报：**自托管、默认关闭、失败不抛**。

为什么自己拼 ingestion 请求而不上 `langfuse` SDK：
  * SDK v3 走 OpenTelemetry，会拖进一整套 exporter/protobuf 依赖；本机 `pyproject` 里的依赖
    每一项都要过 `scripts/m5_version_lock.py` 的「声明≠用到」闸门，为一个默认关的功能引 6 个包不值。
  * Langfuse 的 `/api/public/ingestion` 是**公开稳定的批量接口**（Basic = pk:sk），一个 httpx
    Client 就够了——而 httpx 本来就在依赖里。接口版本若变动，锁在 `usage`/`metadata` 里的字段
    最坏是「Langfuse 那边少显示」，不会影响任何门禁读数。

三条纪律：
  1. **数据不出域是硬闸门。** 一轮问答的 question/结论/引用正文都会进上报载荷，所以
     `langfuse_host` 的 authority 必须是回环或私网地址（或显式名字白名单），否则
     `configured()` 直接判不可用并记原因；公网目标要 `KEE_LANGFUSE_ALLOW_REMOTE=true` 才放行，
     且那个开关的存在本身就是一条风险登记。这条与 `scripts/serve_api.py` 只允许回环绑定同性质。
  2. **上报在后台线程，主链路只入队。** 队列满就丢（计数 `dropped`），绝不让一次网络抖动
     把用户的问答拖慢或打挂——观测手段反噬被观测对象是不可接受的。
  3. **时间轴是重建的，不是实测的。** 图上的 sensitive_gate / enforce / confidence 三个节点
     本来就不计时，各 span 的起止由 `timings_ms` 按执行顺序回推。残差进 trace.metadata
     的 `residual_unmeasured_ms`，不摊平、不假装精确。

边界（如实登记，不粉饰）：原始 prompt 与模型原文**没有**进 `AnswerResult`（`llm_meta` 只有
`prompt_chars` 与 metrics），所以 generation 的 input/output 记的是**出口契约层**看到的
问题与答案。要上报逐字 prompt/completion 得改收口字段，那会动响应契约，本次没做。
"""

from __future__ import annotations

import ipaddress
import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from kee.config import Settings, get_settings
from kee.obs import metrics

log = logging.getLogger("kee.obs.langfuse")

# 内网工具常见的宿主名（Docker Desktop 的 host-gateway 别名）。写死这几个而不是「任意名字」：
# 名字解析成什么地址我们不知道，也不该猜——公网域名必须走 ALLOW_REMOTE。
ALLOWED_NAMES = frozenset({"localhost", "host.docker.internal", "gateway.docker.internal", "kubernetes.docker.internal"})
MAX_QUEUE = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def in_domain_host(url: str) -> tuple[bool, str]:
    """回环 / 私网 / 链路本地才算「域内」；公网 IP 与未知域名一律拒。"""
    try:
        parts = urlsplit(url)
    except ValueError as e:
        return False, f"URL 解析失败：{type(e).__name__}"
    host = (parts.hostname or "").strip().lower()
    if not host:
        return False, "langfuse_host 没有主机名"
    if parts.scheme not in ("http", "https"):
        return False, f"协议不支持：{parts.scheme or '(空)'}"
    if host in ALLOWED_NAMES:
        return True, f"名字白名单：{host}"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False, f"非 IP 也非白名单名字：{host}（确需上报到该地址请显式设 KEE_LANGFUSE_ALLOW_REMOTE=true）"
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True, f"私网/回环地址：{ip}"
    return False, f"公网地址 {ip}：数据不出域，默认拒绝"


class LangfuseExporter:
    """把一轮问答翻成 Langfuse ingestion 事件，后台批量上报。"""

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self.s = s
        self.stats: dict[str, Any] = {"enqueued": 0, "flushed": 0, "errors": 0, "dropped": 0,
                                      "batches": 0, "last_error": "", "last_flush_ts": ""}
        self._q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None
        ok, why = in_domain_host(s.langfuse_host)
        if not ok and s.langfuse_allow_remote:
            ok, why = True, f"越域目标已被 KEE_LANGFUSE_ALLOW_REMOTE 显式放行（{why}）"
        self.disabled_reason = ""
        if not s.langfuse_enabled:
            self.disabled_reason = "未启用（KEE_LANGFUSE_ENABLED=false，默认）"
        elif not ok:
            self.disabled_reason = f"目标越域被拒：{why}"
        elif not (s.langfuse_public_key and s.langfuse_secret_key):
            self.disabled_reason = "缺 public/secret key（只走环境变量，不写进仓库文件）"
        else:
            self._client = httpx.Client(
                base_url=s.langfuse_host.rstrip("/"),
                auth=(s.langfuse_public_key, s.langfuse_secret_key),
                timeout=httpx.Timeout(s.langfuse_timeout_sec, connect=3.0),
            )
            self._thread = threading.Thread(target=self._run, name="kee-langfuse", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ---- 生产侧 ----
    def record_qa(self, res: Any, *, ctx: Any = None, session_id: str = "", actor: str = "",
                  stream: bool = False) -> None:
        if not self.enabled:
            metrics.observe_langfuse("disabled")
            return
        try:
            events = self._events(res, ctx=ctx, session_id=session_id, actor=actor, stream=stream)
        except Exception as e:  # noqa: BLE001 - 组装失败也不能影响这轮答案
            self.stats["last_error"] = f"build:{type(e).__name__}: {e}"[:200]
            metrics.observe_langfuse("error")
            return
        n = 0
        for ev in events:
            try:
                self._q.put_nowait(ev)
                n += 1
            except queue.Full:
                self.stats["dropped"] += 1
                metrics.observe_langfuse("dropped")
        if n:
            self.stats["enqueued"] += n
            metrics.observe_langfuse("enqueued", n)
            metrics.LANGFUSE_QUEUE.set(self._q.qsize())

    def _events(self, res: Any, *, ctx: Any, session_id: str, actor: str, stream: bool) -> list[dict]:
        d = res.as_dict() if hasattr(res, "as_dict") else dict(res)
        trace_id = str(uuid.uuid4())
        timings = d.get("timings_ms") or {}
        retrieval = d.get("retrieval") or {}
        llm = d.get("llm") or {}
        conf = d.get("confidence") or {}
        # 顺序回推：整轮以 e2e_ms 为总长，各已计时阶段按图上执行次序依次占位。
        # `cursor_ms` 的语义是「距本轮结束还有多少毫秒」，所以**越大越早**——
        # 传进 `_span` 的是这一段在时间轴上的**起点偏移**，别把起止写反（第一版就写反了，
        # 实采的 trace 上出现 `03:42:26 → 03:41:36` 这种结束早于开始的 span 才发现）。
        # `generate` 不在这里出 span：它由下面的 generation 事件承载（同名两条是重复上报）。
        seq = [("rewrite", timings.get("rewrite_ms")), ("retrieve", timings.get("retrieval_ms"))]
        total_ms = float(timings.get("e2e_ms")
                         or (float(timings.get("generate_ms") or 0) + sum(float(v or 0) for _, v in seq)))
        wall_end = datetime.now(timezone.utc)
        spans: list[dict] = []
        cursor_ms = total_ms
        for name, ms in seq:
            dur = float(ms or 0)
            if dur <= 0:
                continue
            cursor_ms -= dur
            spans.append(self._span(trace_id, f"qa.{name}", wall_end, cursor_ms, dur,
                                    self._span_io(name, d, retrieval)))
        residual = max(0.0, cursor_ms)
        output = {"status": d.get("status"), "conclusion": d.get("conclusion"),
                  "key_points": d.get("key_points"), "needs_human": d.get("needs_human")}
        tags = [f"status:{d.get('status')}", f"mode:{retrieval.get('mode', '')}",
                f"cache:{'hit' if retrieval.get('cache_hit') else 'miss'}"]
        if stream:
            tags.append("stream:true")
        events = [{
            "id": str(uuid.uuid4()), "type": "trace-create", "timestamp": _now(),
            "body": {
                "id": trace_id, "timestamp": _now(), "name": "kee.qa.round",
                "input": d.get("question"), "output": output,
                "sessionId": session_id or None, "userId": actor or None,
                "tags": tags,
                "metadata": {
                    "kee_trace_id": d.get("trace_id", ""),
                    "role": getattr(ctx, "role", ""),
                    "status": d.get("status"),
                    "notes": d.get("notes") or [],
                    "degraded": retrieval.get("degraded") or [],
                    "cache_hit": bool(retrieval.get("cache_hit")),
                    "top_score": conf.get("retrieval"),
                    "confidence": conf,
                    "timings_ms": timings,
                    "residual_unmeasured_ms": round(residual, 1),
                },
            },
        }]
        events += spans
        gen = self._generation(trace_id, wall_end, d, llm, timings)
        if gen:
            events.append(gen)
        return events

    @staticmethod
    def _span_io(name: str, d: dict, retrieval: dict) -> tuple[Any, Any]:
        if name == "rewrite":
            return d.get("question"), retrieval.get("query_used")
        if name == "retrieve":
            return retrieval.get("query_used"), {"n_chunks": retrieval.get("n_chunks"),
                                                 "top_chunk_ids": retrieval.get("top_chunk_ids")}
        return d.get("question"), None

    def _span(self, trace_id: str, name: str, wall_end: datetime, end_offset_ms: float,
              dur_ms: float, io: tuple[Any, Any]) -> dict:
        """`end_offset_ms` 是这一段**结束点**距本轮结束的毫秒数，所以越大越早、
        `start` 落在它加上时长那一侧。方向反过一次：实采的 trace 里出现
        `03:42:26 → 03:41:36`（结束早于开始），现由单测钉住 `startTime < endTime`。
        """
        start = _shift(wall_end, end_offset_ms + dur_ms)
        end = _shift(wall_end, end_offset_ms)
        inp, out = io
        return {
            "id": str(uuid.uuid4()), "type": "span-create", "timestamp": _now(),
            "body": {"id": str(uuid.uuid4()), "traceId": trace_id, "name": name,
                     "startTime": start, "endTime": end, "input": inp, "output": out,
                     "metadata": {"duration_ms": round(dur_ms, 1)}},
        }

    def _generation(self, trace_id: str, wall_end: datetime, d: dict,
                    llm: dict, timings: dict) -> dict | None:
        ms = float(timings.get("generate_ms") or 0)
        if ms <= 0:
            return None  # 没进生成节点（早退），不编造一次调用
        metrics_map = llm.get("metrics") or {}
        # 生成是本轮最后一段被计时的活，所以它的结束点就是本轮结束（偏移 0）
        start = _shift(wall_end, ms)
        usage: dict[str, Any] = {}
        pt, ct = metrics_map.get("prompt_tokens"), metrics_map.get("completion_tokens")
        if isinstance(pt, int) or isinstance(ct, int):
            usage = {"input": pt or 0, "output": ct or 0, "unit": "TOKENS"}
        return {
            "id": str(uuid.uuid4()), "type": "generation-create", "timestamp": _now(),
            "body": {
                "id": str(uuid.uuid4()), "traceId": trace_id, "name": "qa.generate",
                "model": self.s.llm_model, "startTime": start,
                "endTime": _shift(wall_end, 0.0),
                "input": d.get("question"),
                "output": {"conclusion": d.get("conclusion"), "key_points": d.get("key_points")},
                "usage": usage,
                "metadata": {"prompt_chars": llm.get("prompt_chars"),
                             "json_valid_first_attempt": llm.get("first_attempt_json_valid"),
                             "done_reason": metrics_map.get("done_reason", ""),
                             "ollama_metrics": metrics_map,
                             "llm_ok": llm.get("ok"), "llm_error": llm.get("error")},
            },
        }

    # ---- 消费侧 ----
    def _run(self) -> None:
        """后台线程：按 `flush_interval_sec` 取第一件，再非阻塞抽干到 `max_batch`。
        批量接口的价值就在这条上——一次问答 5～7 个事件，逐条 POST 会把上报变成瓶颈。"""
        while not self._stop.is_set():
            try:
                batch = [self._q.get(timeout=self.s.langfuse_flush_interval_sec)]
            except queue.Empty:
                continue
            while len(batch) < self.s.langfuse_max_batch:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self._flush(batch)
            metrics.LANGFUSE_QUEUE.set(self._q.qsize())

    def _flush(self, batch: list[dict]) -> None:
        assert self._client is not None
        body = {"batch": batch}
        try:
            resp = self._client.post("/api/public/ingestion",
                                     content=json.dumps(body, ensure_ascii=False, default=str),
                                     headers={"Content-Type": "application/json"})
        except Exception as e:  # noqa: BLE001 - 采集器故障不影响被采集的系统，这是硬要求
            self.stats["errors"] += 1
            self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
            metrics.observe_langfuse("error")
            return
        if resp.status_code >= 300:
            self.stats["errors"] += 1
            self.stats["last_error"] = f"http {resp.status_code}: {resp.text[:160]}"
            metrics.observe_langfuse("error")
            return
        # 批量接口是 2xx + 逐条结果（207 Multi-Status）：整体 200 不代表每条都落了库
        partial = 0
        try:
            errs = resp.json().get("errorList") or []
            partial = len(errs)
        except (ValueError, AttributeError):
            pass
        self.stats["flushed"] += len(batch) - partial
        self.stats["batches"] += 1
        self.stats["last_flush_ts"] = _now()
        if partial:
            self.stats["errors"] += partial
            self.stats["last_error"] = f"partial_rejected:{partial}/{len(batch)}"
            metrics.observe_langfuse("error", partial)
        metrics.observe_langfuse("flushed", len(batch) - partial)

    def flush_now(self, wait_sec: float = 5.0) -> None:
        """把队列排空（验证与进程退出前用）。"""
        if not self.enabled:
            return
        deadline = time.monotonic() + wait_sec
        while self._q.qsize() and time.monotonic() < deadline:
            time.sleep(0.1)
        metrics.LANGFUSE_QUEUE.set(self._q.qsize())

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._client is not None:
            self._client.close()

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "host": self.s.langfuse_host if self.enabled else "",
                "reason": self.disabled_reason, "queue_depth": self._q.qsize(),
                "payload": "question/结论/引用要点（域内主机硬闸门）", **self.stats}


def _shift(wall_end: datetime, offset_ms: float) -> str:
    from datetime import timedelta

    ts = wall_end - timedelta(milliseconds=offset_ms)
    return ts.isoformat().replace("+00:00", "Z")


_exporter: LangfuseExporter | None = None


def get_exporter(settings: Settings | None = None) -> LangfuseExporter:
    """进程内单例：`create_app()` 会被调多次，每轮问答都要能拿到同一个后台线程。"""
    global _exporter
    if _exporter is None:
        _exporter = LangfuseExporter(settings)
        if not _exporter.enabled and (_exporter.s.langfuse_enabled):
            log.warning("Langfuse 已声明启用但不可用：%s", _exporter.disabled_reason)
    return _exporter


def reset_exporter() -> None:
    """只给测试用：换配置后重建单例。"""
    global _exporter
    if _exporter is not None:
        _exporter.close()
    _exporter = None
