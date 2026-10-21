"""LLM-as-Judge：把「让模型当裁判」做成一个有判据、可复核、且不碰人判账的装置。

三条硬规矩，都是这个仓库既有纪律的延续，不是新发明：

1. **裁判不替代人判**。真人 `answer_correct` 工单那 67 行仍是「结论对不对」的唯一读数来源；
   本装置的用途是先量**它与真人判的一致率 / Cohen's κ**——一致率撑不住，它的绝对值就没有对外
   引用的资格。产物里两条读数分开命名，永不混算。
2. **判据照工单原文，且不给参考答案**。工单原话是「只判这条 conclusion 相对于它自己引用的证据
   是否成立……**不给参考答案**（给了就是抄）」。裁判拿到参考答案就退化成抄写比对，量不出「成立与否」。
3. **判不出就是 None，不是 0**。信封坏、字段非法、模型含糊其辞一律记不可判并报数——
   把「没判出来」算成 0 分是这个仓库明令禁止的那类假读数（见「分母为 0 报不适用」那条同型口径）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kee.eval.kappa import cohens_kappa

UNKNOWN_TOKEN = "?"


@dataclass(frozen=True)
class Rubric:
    """一张判据卡。`text` 会原样印进裁判提示，所以它必须能被真人复核者读通。"""

    id: str
    title: str
    text: str


ANSWER_CORRECT = Rubric(
    id="answer_correct",
    title="结论相对其所引证据是否成立",
    text=(
        "只判「这条 conclusion 相对于它自己引用的证据是否成立」——1=成立，0=不成立，"
        "判不了就填 ?。**不给参考答案**，也不看措辞是否漂亮：只看结论里的数字、口径、主体、"
        "时间在不在这几块引用里，有没有被反着说或张冠李戴。"
    ),
)

DISPOSITION_CORRECT = Rubric(
    id="disposition_correct",
    title="这次拒答或转人工该不该",
    text=(
        "系统这一行**没有给出结论**（拒答或转人工）。只判「按题面与下面这些它实际检索到的材料，"
        "这次不答对不对」——1=该不答（材料确实不足以正面回答，或该题本就该转人工），"
        "0=不该不答（材料里明摆着有正面答案，却拒了或转了），判不了就填 ?。"
        "**不给参考答案**。注意观测域：下面列出的是系统这次检到的材料，不是全库；"
        "「该答却检不到」在这一格判不出来，那是检索缺口，不是本格的 0。"
    ),
)

RUBRICS: dict[str, Rubric] = {r.id: r for r in (ANSWER_CORRECT, DISPOSITION_CORRECT)}

JUDGE_TEMPLATE = (
    "你是一名严格的评审。请依据下面这张判据卡做一次判定。\n\n"
    "【判据卡 · {title}】\n{text}\n\n"
    "【题面】{question}\n\n"
    "【被评审的回答】{answer}\n\n"
    "【该回答所引用的材料】\n{contexts}\n\n"
    "只能依据上面列出的材料，不许使用你自己的知识补链。\n"
    '只返回 JSON：{{"label": 1 或 0 或 "{unknown}", "reason": "30 字内", '
    '"span": "材料里支撑或反驳的原话片段，60 字内；判不出留空"}}\n'
)


def build_judge_prompt(rubric: Rubric, question: str, answer: str, contexts: list[str]) -> str:
    body = "\n---\n".join(f"[材料{i + 1}] {c}" for i, c in enumerate(contexts)) or "（该回答没有引用任何材料）"
    return JUDGE_TEMPLATE.format(
        title=rubric.title,
        text=rubric.text,
        question=question or "（无题面）",
        answer=answer or "（空回答）",
        contexts=body,
        unknown=UNKNOWN_TOKEN,
    )


def parse_judgement(envelope: dict[str, Any] | None) -> dict[str, Any]:
    """非法 label 一律 None。`"?"` 是模型明确承认判不了，记 `unjudgeable`，同样不进分母。"""
    if envelope is None:
        return {"label": None, "reason": "", "span": "", "why": "no_envelope"}
    raw = envelope.get("label")
    label: int | None
    if isinstance(raw, bool):
        label = int(raw)
    elif isinstance(raw, int):
        label = raw if raw in (0, 1) else None
    elif isinstance(raw, str):
        s = raw.strip()
        if s in ("1", "1.0"):
            label = 1
        elif s in ("0", "0.0"):
            label = 0
        elif s in ("?", "unknown", "无法判断"):
            label = None
        else:
            label = None
    else:
        label = None
    reason = envelope.get("reason")
    span = envelope.get("span")
    return {
        "label": label,
        "reason": reason.strip()[:200] if isinstance(reason, str) else "",
        "span": span.strip()[:300] if isinstance(span, str) else "",
        "why": "" if label is not None else "unjudgeable_or_illegal_label",
    }


def agreement_with_human(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """只统计**双方都给出 0/1** 的行；任一方为 None 的行排除并单独报数。

    `rows` 每条形如 {{"qid", "human": int|None, "judge": int|None}}。
    """
    paired = [(r["qid"], r["human"], r["judge"]) for r in rows if r.get("human") in (0, 1) and r.get("judge") in (0, 1)]
    excluded = len(rows) - len(paired)
    if not paired:
        return {"n_paired": 0, "n_excluded": excluded, "value": None, "reading": "不适用（没有双方都判出的行）"}
    a = [p[1] for p in paired]
    b = [p[2] for p in paired]
    same = sum(1 for x, y in zip(a, b) if x == y)
    kappa = cohens_kappa(a, b)
    conf = {
        "human1_judge1": sum(1 for x, y in zip(a, b) if x == 1 and y == 1),
        "human1_judge0": sum(1 for x, y in zip(a, b) if x == 1 and y == 0),
        "human0_judge1": sum(1 for x, y in zip(a, b) if x == 0 and y == 1),
        "human0_judge0": sum(1 for x, y in zip(a, b) if x == 0 and y == 0),
    }
    human_wrong = conf["human0_judge0"] + conf["human0_judge1"]
    judge_flagged = conf["human1_judge0"] + conf["human0_judge0"]
    return {
        "n_paired": len(paired),
        "n_excluded": excluded,
        "value": round(same / len(paired), 4),
        "reading": f"{same}/{len(paired)} = {same / len(paired):.2%}",
        "kappa": kappa,
        "confusion": conf,
        # 裁判说「成立/该拒」的比例：一致率高而 κ≈0 时，这一列才是真相——它可能只是在点头。
        "judge_yes_rate": round((conf["human1_judge1"] + conf["human0_judge1"]) / len(paired), 4),
        # recall 的分母是**真人判为错的行**，分子是其中被裁判也判为错的行。
        # 早先这里误取了 human0_judge1（= 裁判与人判**不一致**的行数）除以同一个分母，
        # 于是「一行都没抓到」被算成「抓到 100%」——一个反方向的假绿，形状与 12.5% 那次误拒对称。
        "recall_on_human_wrong": (round(conf["human0_judge0"] / human_wrong, 4) if human_wrong else None),
        "n_human_wrong": human_wrong,
        "precision_on_flagged": (round(conf["human0_judge0"] / judge_flagged, 4) if judge_flagged else None),
        "divergent": [{"qid": q, "human": h, "judge": j} for q, h, j in paired if h != j],
    }


class LlmJudge:
    def __init__(self, llm: Any, rubric: Rubric = ANSWER_CORRECT, max_tokens: int = 260) -> None:
        if rubric.id not in RUBRICS:
            raise ValueError(f"未登记判据卡：{rubric.id}")
        self.llm = llm
        self.rubric = rubric
        self.max_tokens = max_tokens
        self.calls: list[dict[str, Any]] = []

    def judge(self, question: str, answer: str, contexts: list[str],
              rubric: Rubric | None = None) -> dict[str, Any]:
        card = rubric or self.rubric
        if card.id not in RUBRICS:
            raise ValueError(f"未登记判据卡：{card.id}")
        res = self.llm.chat(
            [{"role": "user", "content": build_judge_prompt(card, question, answer, contexts)}],
            json_mode=True,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        m = dict(getattr(res, "metrics", {}) or {})
        self.calls.append(
            {
                "ok": bool(getattr(res, "ok", False)),
                "json_valid": bool(getattr(res, "json_valid", False)),
                "done_reason": m.get("done_reason", ""),
                "total_ms": m.get("total_ms", 0.0),
                "prompt_tokens": m.get("prompt_tokens", 0),
                "completion_tokens": m.get("completion_tokens", 0),
            }
        )
        env = _envelope(res)
        out = parse_judgement(env)
        out["truncated"] = out["label"] is None and m.get("done_reason") == "length"
        out["rubric"] = card.id
        return out

    def cost_summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "failed": sum(1 for c in self.calls if not c["ok"]),
            "bad_json": sum(1 for c in self.calls if c["ok"] and not c["json_valid"]),
            "truncated": sum(1 for c in self.calls if c["done_reason"] == "length"),
            "llm_ms_total": round(sum(float(c["total_ms"] or 0.0) for c in self.calls), 1),
            "prompt_tokens": sum(int(c["prompt_tokens"] or 0) for c in self.calls),
            "completion_tokens": sum(int(c["completion_tokens"] or 0) for c in self.calls),
        }


def _envelope(res: Any) -> dict[str, Any] | None:
    if not getattr(res, "ok", False):
        return None
    parsed = getattr(res, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    text = (getattr(res, "text", "") or "").strip()
    if not text:
        return None
    try:
        got = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return got if isinstance(got, dict) else None
