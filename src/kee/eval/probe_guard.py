"""跑真机探针前的有效性前置：区分「测出来的结果」与「依赖没起来的样子」。

起因是 2026-10-11 那次真机复跑：Ollama 的 `/api/tags` 返回 200，但 `llama-server` 二进制缺失，
58 行答案全部在 ~400ms 内以 `degraded` + 0 字结论返回（generation_failed 分支会把检索到的块
原样填进要点与引用，所以看起来**像是一次合法的降级**）。

危险的不是这组数字难看，而是它**会被当成一次测量**：

* `m3_answer_probe` 会报出「完成率 0/58」，然后有人去解释一个不存在的回退；
* `m3_refusal_guard` 更糟——每行结论都为空，它会判 **refuse 类 7/7 干净、PASS**。
  一次故障被读成一条通过的门禁，这是假绿里最难发现的那一类，因为它方向也对、数字也「好看」。

所以判据选**生成成功率**而不是延迟：延迟低是症状，没有产出才是原因。
阈值 0.2 是「环境明显不健康」而不是「个别题失败」——单题抖动本来就该允许。
"""

from __future__ import annotations

GENERATION_FAILURE_MARKERS = ("generation_failed", "unparseable_output")


def generation_failure_share(notes_per_row: list[list[str]]) -> float:
    """按「这一行有没有走到生成产出」算失败占比。"""
    if not notes_per_row:
        return 0.0
    bad = sum(
        1 for notes in notes_per_row if any(str(n).startswith(GENERATION_FAILURE_MARKERS) for n in notes)
    )
    return bad / len(notes_per_row)


def unusable_run(share: float, threshold: float = 0.2) -> str | None:
    """返回给人看的判定语；None 表示这一轮可以用。

    刻意把「别把它当测量结果」写在最前面——这类失败的第一现场通常是几分钟后，
    而那时数字已经在文件里了。
    """
    if share > threshold:
        return (
            f"{share:.0%} 的行没有走到生成产出——这是**依赖没起来，不是测量结果**。"
            "不写 artifact、不覆盖既有证据。"
            "注意 `curl /api/tags` 返回 200 只代表 Ollama 的 HTTP 层活着，"
            "不代表 llama-server 能起来加载模型。"
        )
    return None
