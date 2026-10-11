"""Ollama 客户端：本地 qwen2.5:7b 常驻、JSON 约束输出、耗时与 token 计量。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from kee.config import get_settings

NS_TO_MS = 1_000_000


def normalize_keep_alive(value: object) -> int | str:
    """Ollama 的 keep_alive 只接受「秒数」或「带单位时长串」；裸字符串数字会被 400 拒绝。"""
    s = str(value).strip()
    if s in ("-1", "-1s"):
        return "-1s"  # 常驻不过期
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


@dataclass
class LlmResult:
    ok: bool
    text: str = ""
    parsed: Any = None
    json_valid: bool = False
    error: str = ""
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return float(self.metrics.get("total_ms", 0.0))


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        s = get_settings()
        self.base_url = (base_url or s.ollama_url).rstrip("/")
        self.model = model or s.llm_model
        self.keep_alive = s.llm_keep_alive
        self.timeout = s.llm_timeout_sec
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(self.timeout, connect=10.0))

    # ---- 生命周期 ----
    def model_available(self) -> bool:
        data = self._http.get("/api/tags").json()
        return any(m["name"].startswith(self.model) for m in data.get("models", []))

    def resident_models(self) -> list[dict[str, Any]]:
        data = self._http.get("/api/ps").json()
        return data.get("models", [])

    def is_resident(self) -> bool:
        return any(m["name"].startswith(self.model) for m in self.resident_models())

    def warmup(self, max_tokens: int = 1) -> dict[str, Any]:
        """显式触发模型加载，返回加载与常驻信息（① 的显存归因证据）。"""
        t0 = time.perf_counter()
        r = self.chat([{"role": "user", "content": "回复：OK"}], max_tokens=max_tokens)
        load_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "load_ms": load_ms,
            "ok": r.ok,
            "resident": self.is_resident(),
            "ps": [
                {"name": m.get("name"), "size": m.get("size"), "expires_at": m.get("expires_at")}
                for m in self.resident_models()
            ],
        }

    def evict(self) -> None:
        """keep_alive=0 立刻把模型移出显存。"""
        self._http.post(
            "/api/generate", json={"model": self.model, "keep_alive": 0, "prompt": ""}
        )

    # ---- 推理 ----
    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 900,
        keep_alive: str | None = None,
    ) -> LlmResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": normalize_keep_alive(
                keep_alive if keep_alive is not None else self.keep_alive
            ),
        }
        if json_mode:
            payload["format"] = "json"
        t0 = time.perf_counter()
        try:
            resp = self._http.post("/api/chat", json=payload)
        except httpx.HTTPError as e:
            return LlmResult(ok=False, error=f"{type(e).__name__}: {e}")
        wall_ms = round((time.perf_counter() - t0) * 1000, 1)
        if resp.status_code != 200:
            return LlmResult(ok=False, error=f"http {resp.status_code}: {resp.text[:200]}")
        j = resp.json()
        text = (j.get("message") or {}).get("content", "")
        parsed, json_valid = None, False
        if json_mode:
            try:
                parsed = json.loads(text)
                json_valid = True
            except (json.JSONDecodeError, TypeError):
                json_valid = False
        metrics = {
            "total_ms": wall_ms,
            "server_total_ms": round((j.get("total_duration", 0) or 0) / NS_TO_MS, 1),
            "load_ms": round((j.get("load_duration", 0) or 0) / NS_TO_MS, 1),
            "prompt_eval_ms": round((j.get("prompt_eval_duration", 0) or 0) / NS_TO_MS, 1),
            "eval_ms": round((j.get("eval_duration", 0) or 0) / NS_TO_MS, 1),
            "prompt_tokens": j.get("prompt_eval_count", 0),
            "completion_tokens": j.get("eval_count", 0),
            # done_reason 是「length」还是「stop」，是区分「输出被 num_predict 截断」与
            # 「模型自己说完就停了」的唯一硬证据。M2 的查询改写就因为没看这个字段，
            # 把 21/70 的截断当成「模型输出无效」白白扔掉了。
            "done_reason": j.get("done_reason", ""),
        }
        tok = metrics["completion_tokens"]
        if metrics["eval_ms"] > 0 and tok:
            metrics["tokens_per_sec"] = round(tok / (metrics["eval_ms"] / 1000.0), 2)
        return LlmResult(ok=True, text=text, parsed=parsed, json_valid=json_valid, metrics=metrics)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 900,
        keep_alive: str | None = None,
    ):
        """逐 token 产出内容增量。**与 chat() 的关键差别在于拿到的东西是不完整的**：

        Ollama 的 `/api/chat` 在 `stream=true` 下按 NDJSON 返回，每行一个 `message.content` 片段，
        最后一行带 `done`/`done_reason` 与计时。所以本函数是一个生成器，
        调用方必须自己处理「JSON 信封还没闭合」这件事（见 `kee.qa.stream_json`）。

        yield 的形状：
          ("delta", 文本片段)      —— 生成过程中反复给
          ("final", LlmResult)     —— 收尾，text 是完整原文、metrics 与 chat() 同构
          ("error", 说明文字)      —— HTTP 层失败；此时没有任何 final

        刻意**不在这里抛异常**：流式路径的失败要让 SSE 端点发一个 error 事件收尾，
        而不是让生成器在半路断掉（那等于让前端停在「收到一半」的状态）。
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": normalize_keep_alive(
                keep_alive if keep_alive is not None else self.keep_alive
            ),
        }
        if json_mode:
            payload["format"] = "json"

        t0 = time.perf_counter()
        pieces: list[str] = []
        last: dict[str, Any] = {}
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", "replace")[:200]
                    yield ("error", f"http {resp.status_code}: {body}")
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 半行/心跳行都不该中断整条流
                    last = j
                    delta = (j.get("message") or {}).get("content", "")
                    if delta:
                        pieces.append(delta)
                        yield ("delta", delta)
        except httpx.HTTPError as e:
            yield ("error", f"{type(e).__name__}: {e}")
            return

        text = "".join(pieces)
        parsed, json_valid = None, False
        if json_mode:
            try:
                parsed = json.loads(text)
                json_valid = True
            except (json.JSONDecodeError, TypeError):
                json_valid = False
        # 计时字段只在最后一行；缺失时全部记 0 而不是猜——done_reason 缺失尤其要说清，
        # 它是「被 num_predict 截断」与「模型自己说完」的唯一硬证据。
        metrics = {
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "server_total_ms": round((last.get("total_duration", 0) or 0) / NS_TO_MS, 1),
            "load_ms": round((last.get("load_duration", 0) or 0) / NS_TO_MS, 1),
            "prompt_eval_ms": round((last.get("prompt_eval_duration", 0) or 0) / NS_TO_MS, 1),
            "eval_ms": round((last.get("eval_duration", 0) or 0) / NS_TO_MS, 1),
            "prompt_tokens": last.get("prompt_eval_count", 0),
            "completion_tokens": last.get("eval_count", 0),
            "done_reason": last.get("done_reason", ""),
        }
        tok = metrics["completion_tokens"]
        if metrics["eval_ms"] > 0 and tok:
            metrics["tokens_per_sec"] = round(tok / (metrics["eval_ms"] / 1000.0), 2)
        yield ("final", LlmResult(ok=True, text=text, parsed=parsed, json_valid=json_valid, metrics=metrics))

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._http.post("/api/embed", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    def close(self) -> None:
        self._http.close()
