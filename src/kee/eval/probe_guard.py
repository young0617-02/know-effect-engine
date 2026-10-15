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


SCORE_SCALE_MAX = 1.0


def score_scale_violation(scores: list[float], limit: float = SCORE_SCALE_MAX) -> str | None:
    """检索分量超出 [0,1] 就判「这一轮不是测量」——与 `unusable_run` 同一条纪律，量纲版的。

    起因（用户 2026-10-15 裁「乙为主、甲为辅」）：同一份代码复跑 `verify_m3`，
    `confidence.retrieval` 一次 0.9937、一次 **36.9408**。后者不是分数变高，是**换了量纲**——
    重排没跑成时 `chunks[].score` 漏的是 BM25 原始分，而下游 `min_retrieval_score=0.20`
    与置信度成分都假定这个数在 0–1 之间。于是门禁句 `top_score >= 0.20` 在
    **一个它从未打算比较的尺度上「通过」**，读数还照常写进产物。

    为什么不在这里顺手把分数夹到 1：那等于把「依赖没起来」化妆成「一次很自信的测量」。
    拒绝出读数才是对的处置，跟 `unusable_run` 同一个理由。
    """
    bad = [s for s in scores if s > limit]
    if not bad:
        return None
    return (
        f"{len(bad)}/{len(scores)} 行的检索分量 > {limit:g}（最大 {max(bad):.4f}）——"
        "这个数不是重排给的 sigmoid 概率，是某条腿的原始量纲漏下来了。"
        "**依赖没起来，不是测量结果**：不写 artifact、不覆盖既有证据。"
        "先看同一行的 `retrieval_degraded`（`hybrid_single_leg` / `rerank_error:*` / "
        "`score_scale_out_of_range:*`）定位是哪条腿。"
    )
