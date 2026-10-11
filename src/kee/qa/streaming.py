"""token 级流式的事件流。

**要解决的真问题不是「能不能流」，而是「流出去的这段能不能算答案」**：
引用强制要等整个 JSON 解析完才能跑，而 `conclusion` 在信封里排在最前面，
所以用户看到结论文字的时候，校验一定还没发生。这不是实现缺陷，是次序使然。

处理方式是把这件事**显式化**而不是藏起来：

* 增量事件一律标 `provisional: true`，前端必须用与最终答案不同的样式渲染（未完成态）；
* 最终 `answer` 事件带 `provisional_retracted`——它记录「流过的文字最后没成为对外答案」，
  也就是**撤回率**。这个数是要量而不是要假设的：如果一行行显示完再整段撤回的比例很高，
  token 级流式对知识服务就是负收益，该关掉而不是硬上。

线程是必需的而不是选择：生成节点会一路阻塞到整段 JSON 吐完，
同一条线程里既跑图又 yield 事件是做不到的——单线程「流式」只能等跑完再一次性给出，
那和阶段级 SSE 没有区别。所以图跑在工作线程，事件经队列出来。

队列是唯一跨线程的通道，`None` 是终止哨兵；worker 侧任何异常都转成 error 事件，
绝不让前端停在「收到一半、然后永远等下去」的状态。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

from kee.qa.graph import initial_state, result_from_state
from kee.qa.stream_json import partial_string_field

CONCLUSION_FIELD = "conclusion"


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
    `token_stream=False` 时退化为原来的阶段级两事件（这是 A/B 与对照的开关，不是兼容垫片）。
    """
    out: queue.Queue = queue.Queue()
    streamed = {"text": ""}

    def on_delta(payload: dict[str, Any]) -> None:
        # 只在内容真的变长时才发事件：模型会反复吐同样的 token 前缀，重复事件对前端是噪音
        text = payload.get("text", "")
        if len(text) <= len(streamed["text"]) and text == streamed["text"]:
            return
        streamed["text"] = text
        out.put(("delta", {**payload, "provisional": True, "chars": len(text)}))

    def worker() -> None:
        state = initial_state(question, ctx, top_k, history)
        if token_stream:
            state["on_delta"] = on_delta
        final: dict[str, Any] = {}
        try:
            for update in engine.graph.stream(state, stream_mode="updates"):
                for node, delta in update.items():
                    state.update(delta or {})
                    if node == "retrieve":
                        out.put(("retrieval", dict(state.get("retrieval") or {})))
            result = result_from_state(state)
            final = result.as_dict()
            # 撤回判定：流过非空文字，但最终要么没结论、要么状态不是「给出了答案」
            retracted = bool(streamed["text"]) and (
                not str(final.get("conclusion", "")).strip()
                or final.get("status") in ("refuse", "degraded")
            )
            final["provisional_streamed_chars"] = len(streamed["text"])
            final["provisional_retracted"] = retracted
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


def scan_conclusion_delta(buf: str) -> dict[str, Any] | None:
    """把「目前为止的缓冲」翻成一个结论增量事件。"""
    text, closed = partial_string_field(buf, CONCLUSION_FIELD)
    if not text and not closed:
        return None
    return {"field": CONCLUSION_FIELD, "text": text, "closed": closed}
