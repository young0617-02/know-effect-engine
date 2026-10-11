"""从**尚未闭合**的 JSON 缓冲里取出某个顶层字符串字段的当前值。

token 级流式绕不开这件事：`format: json` 下模型一片一片吐的是 JSON 信封的碎片，
前端拿到 `{ "conclus` 是没法渲染的。所以要在「信封还没闭合」的时候把**已知字段的内容**认出来。

只认顶层字符串字段，且必须严格遵守 JSON 语法来定位键：

* 深度 1 处遇到 `"key"` 紧跟冒号，才算找到了这个键；深层对象里同名的键**不算**
  （`key_points[].point` 里出现 "conclusion" 不能污染结论区）。
* 值不是字符串（数字/布尔/对象）时返回「取不到」而不是硬转——本模块只服务文本字段。
* 转义按 JSON 规则解，允许缓冲末尾停在半条转义上：`\\` 或 `\\u12` 这种尾巴**先丢掉**，
  下一个增量到达再补上。丢的是「还没写完的那一个字符」，不是任何已完整的内容。
* 未闭合时 `closed=False`，调用方据此知道「这段还可能变」——这是 provisional 的根据。

为什么不用现成的增量 JSON 解析库：那条路要么依赖第三方流式解析器，
要么把不完整文档喂给 `json.loads` 然后捕获异常重试；前者加依赖，后者在每帧上抛异常。
这里需要的只是「读出已知前缀」，手写反而更可控、可测，且没有解析器版本行为漂移的风险。
"""

from __future__ import annotations

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}


def partial_string_field(buf: str, key: str) -> tuple[str, bool]:
    """返回 (该字段目前的内容, 值是否已闭合)。字段没出现时返回 ("", False)。"""
    i, n = 0, len(buf)
    depth = 0
    in_str = False
    esc = False
    while i < n:
        ch = buf[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == '"':
            # 只有顶层（depth==1）的键才认；值里的字符串整体跳过
            if depth == 1:
                j = i + 1
                while j < n and buf[j] != '"':
                    if buf[j] == "\\":
                        j += 1
                    j += 1
                if j >= n:
                    return "", False  # 键名本身还没写完
                if buf[i + 1 : j] == key:
                    k = j + 1
                    while k < n and buf[k] in " \t\n\r":
                        k += 1
                    if k < n and buf[k] == ":":
                        k += 1
                        while k < n and buf[k] in " \t\n\r":
                            k += 1
                        if k < n and buf[k] == '"':
                            return _decode_prefix(buf[k + 1 :])
                        # 缓冲在冒号后就用完了：值类型还看不出来，只能报「没闭合」。
                        # 这里若返回 closed=True，调用方会把「还不知道」当成「确定不是文本」，
                        # 于是流式过程中结论区会被短暂判死。
                        if k >= n:
                            return "", False
                        return "", True  # 看清了：值不是字符串
                    if k < n:
                        return "", False
                i = j + 1
                continue
            in_str = True
        i += 1
    return "", False


def _decode(s: str) -> str:
    """把一个**已闭合**的 JSON 字符串字面量（不含首尾引号）解出来。"""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt == "u" and i + 6 <= n:
            try:
                out.append(chr(int(s[i + 2 : i + 6], 16)))
                i += 6
                continue
            except ValueError:
                pass  # 非法 \u 就按字面走，不吞后面的字符
        if nxt in _ESCAPES:
            out.append(_ESCAPES[nxt])
            i += 2
            continue
        # 认不出的转义（含结尾孤零零一个反斜杠）：丢反斜杠、留原字符
        out.append(nxt if nxt else "\\")
        i += 2
    return "".join(out)


def _decode_prefix(raw: str) -> tuple[str, bool]:
    """处理值可能没闭合的前缀。

    闭合判据：遇到第一个**未被转义**的引号。
    未闭合且末尾停在半个转义上（`...\\` 或 `...\\u12`）时把那段残片**丢掉**而不是崩掉——
    它表示这个字符还没吐完，下一帧会补上。宁可少显示一个字符，也不要显示出一个反斜杠残骸。
    """
    end = -1
    esc = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            end = i
            break
        i += 1
    if end >= 0:
        return _decode(raw[:end]), True
    # 未闭合：数结尾连续的反斜杠，奇数个说明最后一个正在起转义，其后内容未完
    trailing = len(raw) - len(raw.rstrip("\\"))
    return _decode(raw[: len(raw) - trailing] if trailing else raw), False
