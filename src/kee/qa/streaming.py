"""token 级流式的事件流。

**要解决的真问题不是「能不能流」，而是「流出去的这段能不能算答案」**：
引用强制要等整个 JSON 解析完才能跑，所以流给用户看的文字**必然早于校验**。
这不是实现缺陷，是次序使然——处理办法是把它显式化而不是藏起来：

* 增量事件一律带 `provisional: true`，前端必须用与最终答案不同的样式渲染（未完成态）；
* 最终 `answer` 事件带两个可观测字段：
  `provisional_retracted`（流了结论、最终却没成为答案）与
  `provisional_lost`（流出去过、最终答案里找不到的**已闭合**条目数）。
  这两个是要量的数，不是假设接近 0 的常数。

流的内容除了 `conclusion` 还有 **`key_points` 里的要点文字**。
加要点不是锦上添花：实测 79% 的答案其 `conclusion` 是本地逐字搬运拼的、模型根本没写，
只流结论等于对四分之三的题无事可做。代价也很实在——引用强制会**丢弃**没有有效出处的要点，
所以「被丢掉的要点」是一个新的撤回来源，`provisional_lost` 专门记它。

只算 `closed=True` 的条目：未闭合的尾巴还在长，拿它算撤回会把正常生成误判成撤回。

线程是必需的而不是选择：生成节点会一路阻塞到整段 JSON 吐完，
同一条线程里既跑图又 yield 事件做不到——单线程「流式」只能等跑完再一次性给出，
那和阶段级 SSE 没有区别。所以图跑在工作线程，事件经队列出来。

队列是唯一跨线程的通道；worker 侧任何异常都转成 error 事件，
绝不让前端停在「收到一半、然后永远等下去」的状态。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

from kee.qa.graph import initial_state, result_from_state
from kee.qa.stream_json import partial_array_items, partial_string_field

CONCLUSION_FIELD = "conclusion"
POINTS_ARRAY, POINT_KEY = "key_points", "point"


def scan_streamable(buf: str) -> list[dict[str, Any]]:
    """把「到目前为止的缓冲」翻成一组增量事件：先结论，再逐条要点。"""
    events: list[dict[str, Any]] = []
    text, closed = partial_string_field(buf, CONCLUSION_FIELD)
    if text:
        events.append({"field": CONCLUSION_FIELD, "index": None, "text": text, "closed": closed})
    for i, (ptext, pclosed) in enumerate(partial_array_items(buf, POINTS_ARRAY, POINT_KEY)):
        if ptext:
            events.append({"field": POINT_KEY, "index": i, "text": ptext, "closed": pclosed})
    return events


def _final_text_pool(final: dict[str, Any]) -> str:
    """最终对外答案的全部文字。判断「流出去的东西留下了没有」以它为准。"""
    parts = [str(final.get("conclusion") or "")]
    parts += [str(p.get("point") or "") for p in final.get("key_points") or []]
    return "\n".join(parts)


def iter_qa_events(
    engine,
    question: str,
    ctx,
    top_k: int | None = None,
    history: list[dict] | None = None,
    token_stream: bool = True,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """按到达顺序产出 (事件名, 载荷)。

    事件序列：`retrieval` → `delta`* → `answer`；出错则 `error` 收尾。
    `token_stream=False` 时退化为阶段级两事件（这是对照开关，不是兼容垫片）。
    """
    out: queue.Queue = queue.Queue()
    # 每个 (field,index) 各自记进度：结论和第一条要点是两条独立的内容流
    shown: dict[tuple[str, Any], str] = {}
    closed_texts: list[str] = []

    def on_delta(payload: dict[str, Any]) -> None:
        key = (payload.get("field"), payload.get("index"))
        text = payload.get("text", "")
        if shown.get(key) == text:
            return  # 同一份进度不重发：重复事件对前端是噪音
        shown[key] = text
        if payload.get("closed"):
            if text not in closed_texts:
                closed_texts.append(text)
        out.put(("delta", {**payload, "provisional": True, "chars": len(text)}))

    def worker() -> None:
        state = initial_state(question, ctx, top_k, history)
        if token_stream:
            state["on_delta"] = on_delta
        try:
            for update in engine.graph.stream(state, stream_mode="updates"):
                for node, delta in update.items():
                    state.update(delta or {})
                    if node == "retrieve":
                        out.put(("retrieval", dict(state.get("retrieval") or {})))
            final = result_from_state(state).as_dict()
            pool = _final_text_pool(final)
            conclusion_streamed = shown.get((CONCLUSION_FIELD, None), "")
            final_conclusion = str(final.get("conclusion") or "").strip()
            # 结论级撤回：流了结论，最终却没给出答案（无结论，或状态是拒答/降级）
            retracted = bool(conclusion_streamed) and (
                not final_conclusion or final.get("status") in ("refuse", "degraded")
            )
            # 条目级丢失：已闭合的内容里，最终答案中找不到的那些（要点被引用强制丢弃即在此）
            lost = [t for t in closed_texts if t not in pool]
            final["provisional_streamed_chars"] = sum(len(v) for v in shown.values())
            final["provisional_retracted"] = retracted
            final["provisional_lost"] = len(lost)
            final["provisional_lost_samples"] = [t[:60] for t in lost[:3]]
            out.put(("answer", final))
        except Exception as exc:  # noqa: BLE001 - 流必须以事件收尾，不能靠异常把连接掐断
            out.put(("error", {"code": "E500", "message": f"{type(exc).__name__}: {exc}"[:200]}))
        finally:
            out.put(("__end__", {}))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            kind, payload = out.get()
            if kind == "__end__":
                return
            yield kind, payload
    finally:
        t.join(timeout=5.0)
