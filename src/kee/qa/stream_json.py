"""从**尚未闭合**的 JSON 缓冲里读出已知路径上的字符串值。

token 级流式绕不开这件事：`format: json` 下模型一片一片吐的是 JSON 信封的碎片，
前端拿到 `{ "conclus` 是没法渲染的。所以要在「信封还没闭合」的时候把**已知字段的内容**认出来。

核心是路径式遍历器 `iter_string_values()`：它把缓冲走一遍，对每个**字符串值**产出
`(路径, 文本, 是否闭合)`。路径是元组，例如 `("conclusion",)` 或 `("key_points", 0, "point")`。
键名匹配靠路径而不是靠字符串搜索，所以深层同名键（要点文字里出现「conclusion」）
不会污染结论区——这是本模块存在的主要理由。

几条硬规则：

* **键还是值**由状态决定，不靠猜：对象帧记着当前等的是 key 还是 value，数组帧里的字符串一律是值。
  这样 `{"conclu`（键名还没写完）不会被误当成某个字段的值端出去。
* **未闭合的字符串照样产出**，但 `closed=False`；末尾停在半个转义上（`\\` 或 `\\u12`）时
  把那段残片丢掉——宁可暂时少显示一个字符，也不要显示出一个反斜杠残骸。
  `closed=False` 是 provisional 的根据：调用方由此知道「这段还可能变」。
* 非字符串值（数字/布尔/null）被**消费但不产出**。必须消费，否则对象/数组的占位会错位，
  第三要点的文字会被安到第二要点头上——那是会显示错内容的 bug。

为什么不用现成的增量 JSON 解析库：那条路要么加第三方流式解析器，
要么每帧拿 `json.loads` 试解析再捕异常；前者加依赖，后者在每帧上抛异常。
这里需要的只是「读出已知前缀」，手写更可控、可测。

**代价说清**：每帧重扫整个缓冲（O(n) 每帧、O(n²) 每次回答）。700 token 量级下不是瓶颈
（首字延迟由模型决定，不由这里决定），但换成大输出或高频推送时该改成增量游标，
别把这版当可扩展设计。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}
_WS = " \t\n\r"


class _Frame:
    """一层容器。obj 需要「刚读到的键名」，arr 需要「下一个元素的下标」。"""

    __slots__ = ("kind", "pending", "idx", "awaiting")

    def __init__(self, kind: str) -> None:
        self.kind = kind  # "obj" | "arr"
        self.pending: str | None = None
        self.idx = -1
        self.awaiting = "key" if kind == "obj" else "value"


def _decode(raw: str) -> str:
    """解一个 JSON 字符串字面量的内容（不含首尾引号）；容忍结尾的半条转义。"""
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = raw[i + 1] if i + 1 < n else ""
        if nxt == "u" and i + 6 <= n:
            try:
                out.append(chr(int(raw[i + 2 : i + 6], 16)))
                i += 6
                continue
            except ValueError:
                pass  # 非法 \u 按字面走，不吞后面的字符
        if nxt in _ESCAPES:
            out.append(_ESCAPES[nxt])
            i += 2
            continue
        out.append(nxt if nxt else "")
        i += 2
    return "".join(out)


def _scan_string(buf: str, start: int) -> tuple[str, bool, int]:
    """从 buf[start]=='\"' 起扫。返回 (原始内容, 是否闭合, 之后第一个位置)。"""
    i, esc = start + 1, False
    while i < len(buf):
        ch = buf[i]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            return buf[start + 1 : i], True, i + 1
        i += 1
    return buf[start + 1 :], False, len(buf)


def _drop_dangling_escape(raw: str) -> str:
    """结尾连续反斜杠为奇数个时，最后一个正在起转义、其后内容未完——丢掉残片。"""
    trailing = len(raw) - len(raw.rstrip("\\"))
    return raw[: len(raw) - trailing] if trailing else raw


def _claim(stack: list[_Frame]) -> Any:
    """为「即将读到的一个值」在父容器里占位，返回该位置的段名/下标。"""
    f = stack[-1]
    if f.kind == "obj":
        seg = f.pending
        f.pending = None
        f.awaiting = "key"  # 对象里读完一个值，下一个该是键
        return seg
    f.idx += 1
    return f.idx


def iter_string_values(buf: str) -> Iterator[tuple[tuple[Any, ...], str, bool]]:
    """产出缓冲里每个字符串**值**的 (路径, 文本, 是否闭合)。

    只产出到「缓冲能解释」的位置；解释不了的（键名没写完、标量没写完）不产出，宁缺勿猜。
    """
    stack: list[_Frame] = []
    prefix: list[Any] = []
    i, n = 0, len(buf)
    while i < n:
        ch = buf[i]
        if ch in _WS:
            i += 1
            continue
        if ch in "{[":
            if stack:
                prefix.append(_claim(stack))
            stack.append(_Frame("obj" if ch == "{" else "arr"))
            i += 1
            continue
        if ch in "}]":
            if stack:
                stack.pop()
                if prefix:
                    prefix.pop()
            i += 1
            continue
        if ch == ":":
            if stack:
                stack[-1].awaiting = "value"
            i += 1
            continue
        if ch == ",":
            if stack:
                stack[-1].awaiting = "key" if stack[-1].kind == "obj" else "value"
            i += 1
            continue
        if ch == '"':
            raw, closed, after = _scan_string(buf, i)
            top = stack[-1] if stack else None
            if top is None:
                return  # 根级裸字符串不是这条链路的输出形状
            if top.kind == "obj" and top.awaiting == "key":
                # 这里是键名。可能还没写完（closed=False）——那就不记，等下一帧读全。
                if closed:
                    top.pending = _decode(raw)
                i = after
                continue
            path = tuple(prefix + [_claim(stack)])
            yield path, _decode(raw if closed else _drop_dangling_escape(raw)), closed
            i = after
            continue
        # 标量：消费但不产出
        j = i
        while j < n and buf[j] not in ",}]":
            j += 1
        if j < n and stack:  # 只有写完了才占位；没写完这一帧先跳过
            _claim(stack)
        i = j
    return


def partial_string_field(buf: str, key: str) -> tuple[str, bool]:
    """顶层某个字符串字段的当前值。字段没出现（或还不是字符串）时返回 ("", False)。"""
    for path, text, closed in iter_string_values(buf):
        if len(path) == 1 and path[0] == key:
            return text, closed
    return "", False


def partial_array_items(buf: str, array_key: str, item_key: str) -> list[tuple[str, bool]]:
    """数组字段里每个对象的某个文本项，按出现顺序返回。

    用于流式吐 `key_points[].point`。最后一项通常还没闭合，调用方据此知道它是 provisional。
    下标取自路径而不是 append 顺序：**中间项可能因键没读完而暂时缺席**，
    按 append 排会把后面要点的文字安到前面的序号上。
    """
    items: dict[int, tuple[str, bool]] = {}
    for path, text, closed in iter_string_values(buf):
        if len(path) == 3 and path[0] == array_key and path[2] == item_key and isinstance(path[1], int):
            items[path[1]] = (text, closed)
    return [items[k] for k in sorted(items)]
