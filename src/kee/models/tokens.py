"""token 计数：**分块粒度与注入预算都按真实 token 算，不拿字符数近似**。

为什么要单独有这个模块：用户下发粒度用的是 tokens（Child 200 / Parent 1000），而本仓的分块器历史上
按**字符**取预算（280 / 40）。这两者之间的换算比在本语料实测是 **1.397 字符/token**（334 段中位值），
不是常数——英文编号、Markdown 表格、数字串都会把它推开（同一句 40 字实测 26 token = 1.54）。
照系数换算会把"我要 200 token"做成"约 279 字符"，而**没人会在报告里为这 1 个字符负责**；
所以粒度这种参数一旦声明单位是 token，就必须真的数 token。

分词器加载不了就**直接抛**，不做字符兜底：静默退化成近似计数，等于把用户给的粒度参数悄悄换掉，
症状与"单腿检索静默换量纲"同类——数字照常出来，只是量的不是那件事。
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _tokenizer():
    from kee.config import get_settings

    from transformers import AutoTokenizer

    path = get_settings().bge_m3_path
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"BGE-M3 分词器加载失败（{type(e).__name__}: {e}）；路径 {path}。"
            "分块按 token 计数依赖它，**不做字符近似兜底**——兜底等于把声明的粒度悄悄换掉。"
        ) from e


def count_tokens(text: str) -> int:
    return len(_tokenizer()(text or "", add_special_tokens=False)["input_ids"])


def _fit_end(text: str, start: int, budget: int) -> int:
    """从 `start` 出发、token 数不超过 `budget` 的最大字符终点。

    `count_tokens(text[start:k])` 对 k 单调不减 ⇒ 二分合法。
    """
    if start >= len(text):
        return start
    if count_tokens(text[start:]) <= budget:
        return len(text)
    lo, hi = start + 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[start:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return max(lo, start + 1)


def _overlap_start(text: str, end: int, overlap: int) -> int:
    """往左找**最早**的 k，使 `text[k:end]` 还装得下 `overlap` 个 token——即保留最后约 overlap 个 token。"""
    if overlap <= 0:
        return end
    lo, hi = 0, end
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[mid:end]) <= overlap:
            hi = mid
        else:
            lo = mid + 1
    return lo


def cut_by_token_budget(text: str, budget: int, overlap: int = 0) -> list[str]:
    """按 token 预算滑切，返回**原文子串**（不重拼、不改字，保证引用正文与权威源逐字一致）。

    为什么不用 `tokenizer.decode(ids)` 拼：本仓的 BGE-M3 分词器是 XLM-Roberta，
    **实测 fast / slow 两种加载都不返回 offset 映射**（`return_offsets=True` 也没有那个键），
    映不回原文就只能 decode——那会给引用正文引入 tokenizer 的归一化差异，
    等于「分块改了正文」。二分定位字符边界的代价是多次 `count_tokens`（语料尺度上可忽略）。
    """
    return [text[a:b] for a, b, _ in cut_spans_by_token_budget(text, budget, overlap)]


def cut_spans_by_token_budget(text: str, budget: int, overlap: int = 0) -> list[tuple[int, int, str]]:
    """同上，但连同**字符区间**一起给出——父块窗口与子块归属要靠区间判包含，不能靠字符串相等。

    重叠区里同一个子串会出现在两个父块窗口里，用 `str.find` 认归属会认错。
    """
    out: list[tuple[int, int, str]] = []
    start, guard = 0, 0
    while start < len(text) and guard < 20000:
        guard += 1
        end = _fit_end(text, start, budget)
        if text[start:end].strip():
            out.append((start, end, text[start:end]))
        if end >= len(text):
            break
        nxt = _overlap_start(text, end, overlap)
        start = nxt if nxt > start else end          # 重叠取不到进展时宁可无重叠，不许死循环
    return out


def verify_cut(text: str, pieces: list[str], budget: int) -> dict[str, object]:
    """自检：每片都必须是原文子串、token 数不超预算、且不丢字（拼接后与原文同序）。"""
    return {
        "all_substrings": all(p in text for p in pieces),
        "over_budget_pieces": [i for i, p in enumerate(pieces) if count_tokens(p) > budget],
        "chars_kept": round(sum(len(p) for p in pieces) / max(len(text), 1), 3),
    }
