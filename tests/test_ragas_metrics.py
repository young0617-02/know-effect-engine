"""RAGAS 复现装置与 LLM 裁判的钉测：钉的是**口径**，不是读数。

要防的四类事故都在这台机器上真发生过或差点发生：
  - 「空集上的真」：拆不出陈述却把 Faithfulness 报成 1.0（或反过来报 0）；
  - 静默降级：模型吐坏 JSON 被当成 0 分，于是「判不出」和「判错」在同一个分母里混着；
  - 混代配对：裁判标签配到另一代生成产物上；
  - 空转装置：证据换成别题的、分数纹丝不动——那说明它根本没在看证据。
所以每个装置都配一条反面对照测试，护栏不过必须是 `all_pass=False`，而不是悄悄出一个低分。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kee.eval.judge import (  # noqa: E402
    ANSWER_CORRECT,
    DISPOSITION_CORRECT,
    LlmJudge,
    agreement_with_human,
    build_judge_prompt,
    parse_judgement,
)
from kee.eval.ragas import (  # noqa: E402
    RagasRunner,
    answer_relevance_from_vectors,
    build_context_precision_prompt,
    build_context_recall_prompt,
    build_verify_batch_prompt,
    build_verify_prompt,
    context_precision_from_verdicts,
    context_recall_from_verdicts,
    cosine,
    faithfulness_from_verdicts,
    parse_questions,
    parse_statements,
    parse_verdict,
    parse_verdict_batch,
)

import m5_ragas_eval as ev  # noqa: E402


@dataclass
class FakeResult:
    ok: bool = True
    parsed: Any = None
    text: str = ""
    json_valid: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)


class FakeLlm:
    """按脚本顺序回放信封；脚本耗尽即失败——防止测试悄悄少跑一次调用。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.seen: list[dict[str, Any]] = []

    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None) -> FakeResult:
        self.seen.append({"json_mode": json_mode, "temperature": temperature, "prompt": messages[0]["content"]})
        if not self.script:
            return FakeResult(ok=False, parsed=None, json_valid=False, metrics={"done_reason": ""})
        item = self.script.pop(0)
        if item is None:
            return FakeResult(ok=False, parsed=None, json_valid=False, metrics={"done_reason": ""})
        if isinstance(item, str):
            return FakeResult(parsed=None, text=item, json_valid=False,
                              metrics={"done_reason": "stop", "total_ms": 10.0,
                                       "prompt_tokens": 5, "completion_tokens": 2})
        return FakeResult(parsed=item, text="", metrics={"done_reason": "stop", "total_ms": 12.0,
                                                         "prompt_tokens": 7, "completion_tokens": 3})


# ---------------- 解析容错：坏输出必须是 None，不能是 0 ----------------

def test_statements_distinguishes_empty_from_unjudgeable():
    assert parse_statements({"statements": []}) == []          # 回答里确实没有陈述
    assert parse_statements(None) is None                      # 模型没给可用信封 ≠ 空
    assert parse_statements({"statements": "一条字符串"}) == ["一条字符串"]
    assert parse_statements({"statements": [{"statement": " A "}, {"nope": 1}, " B"]}) == ["A", "B"]
    assert parse_statements({"wrong_key": ["x"]}) is None


def test_verdict_rejects_illegal_values():
    assert parse_verdict({"verdict": 1}) == 1
    assert parse_verdict({"verdict": "0"}) == 0
    assert parse_verdict({"verdict": True}) == 1
    assert parse_verdict({"verdict": 2}) is None
    assert parse_verdict({"verdict": "也许"}) is None
    assert parse_verdict(None) is None


def test_judgement_illegal_label_is_none_not_zero():
    assert parse_judgement({"label": "?"})["label"] is None
    assert parse_judgement({"label": 5})["label"] is None
    assert parse_judgement(None)["why"] == "no_envelope"
    got = parse_judgement({"label": "1", "reason": "成立", "span": "RTO 15 分钟"})
    assert got["label"] == 1 and got["span"] == "RTO 15 分钟"


def test_questions_noncommittal_is_not_the_same_as_unjudgeable():
    """解析失败不得披上 noncommittal 的外衣——那会把一次 JSON 崩故障算成「相关性 0」。"""
    assert parse_questions({"noncommittal": True, "questions": []}) == ([], True)
    assert parse_questions(None) == ([], False)                 # 坏信封 ⇒ 不可判
    assert parse_questions({"noncommittal": False, "questions": ["a", "b"]}, k=3) == (["a", "b"], False)
    assert parse_questions({"noncommittal": False, "questions": ["  "]}) == ([], False)
    assert parse_questions({"noncommittal": False}) == ([], False)
    assert parse_questions({"questions": ["a", "b", "c", "d"], "noncommittal": False}, k=3)[0] == ["a", "b", "c"]


# ---------------- Faithfulness 的分母纪律 ----------------

def test_faithfulness_denominator_zero_reports_not_applicable():
    for verdicts in ([], [None, None]):
        out = faithfulness_from_verdicts(verdicts)
        assert out["value"] is None
        assert "不适用" in out["reading"]
        assert out["n_gradable"] == 0


def test_faithfulness_counts_unparsable_out_of_denominator():
    out = faithfulness_from_verdicts([1, 0, None])
    assert out["n_statements"] == 3 and out["n_gradable"] == 2 and out["n_unparsable"] == 1
    assert out["n_supported"] == 1 and out["value"] == 0.5


def test_runner_end_to_end_with_two_statements():
    llm = FakeLlm([{"statements": ["陈述一", "陈述二"]},
                   {"verdicts": [{"i": 1, "verdict": 1, "basis": "原文有"}, {"i": 2, "verdict": 0}]}])
    runner = RagasRunner(llm)
    got = runner.faithfulness("这方案的 RTO 是多少？", "RTO 小于 15 分钟", ["材料正文：RTO 小于 15 分钟"])
    assert got["score"]["value"] == 0.5 and got["score"]["n_supported"] == 1
    assert [v["verdict"] for v in got["verdicts"]] == [1, 0]
    assert runner.cost_summary()["calls"] == 2                 # 拆句一次 + 批量判定一次
    assert all(c["temperature"] == 0.0 and c["json_mode"] for c in llm.seen)


def test_runner_decompose_failure_does_not_become_a_score():
    runner = RagasRunner(FakeLlm([None]))
    got = runner.faithfulness("题", "答", ["材料"])
    assert got["decompose_failed"] is True
    assert got["statements"] is None and got["score"]["value"] is None
    assert "不适用" in got["score"]["reading"]


def test_verify_prompt_keeps_every_context_block():
    prompt = build_verify_prompt("陈述", ["材料一", "材料二"])
    assert "材料一" in prompt and "材料二" in prompt and "[材料2]" in prompt
    assert "（无材料）" in build_verify_prompt("陈述", [])


def test_verify_set_reuses_one_decomposition_across_two_context_sets():
    """两口径必须共用同一批陈述：否则 F(cited) 与 F(top5) 的差里混着「这次拆得不一样」。"""
    llm = FakeLlm([{"statements": ["甲", "乙"]},
                   {"verdicts": [{"i": 1, "verdict": 1}, {"i": 2, "verdict": 1}]},
                   {"verdicts": [{"i": 1, "verdict": 0}, {"i": 2, "verdict": 0}]}])
    runner = RagasRunner(llm)
    statements = runner.decompose("题", "答")
    cited = runner.verify_set(statements, ["证据甲"])
    top5 = runner.verify_set(statements, ["证据乙"])
    assert [v["statement"] for v in cited["verdicts"]] == ["甲", "乙"]
    assert [v["statement"] for v in top5["verdicts"]] == ["甲", "乙"]
    assert cited["score"]["value"] == 1.0 and top5["score"]["value"] == 0.0
    assert sum(1 for c in llm.seen if "原子事实陈述" in c["prompt"]) == 1      # 只拆了一次


def test_runner_faithfulness_still_decomposes_and_verifies():
    runner = RagasRunner(FakeLlm([{"statements": ["甲"]}, {"verdict": 1, "basis": "b"}]), batch_verify=False)
    got = runner.faithfulness("题", "答", ["证据"])
    assert got["statements"] == ["甲"] and got["decompose_failed"] is False
    assert got["score"]["value"] == 1.0 and got["verdicts"][0]["statement"] == "甲"
    assert got["verdicts"][0]["basis"] == "b"


# ---------------- 批量判定（对官方逐条版的一处偏离，必须钉住它的漏答行为） ----------------

def test_parse_verdict_batch_aligns_by_index_and_leaves_gaps_unjudged():
    env = {"verdicts": [{"i": 2, "verdict": 1}, {"i": 1, "verdict": "0"}, {"i": 9, "verdict": 1}]}
    assert parse_verdict_batch(env, 3) == [0, 1, None]          # 第 3 条模型没答 ⇒ None，不补 0
    assert parse_verdict_batch({"verdicts": [{"i": 1, "verdict": 7}]}, 1) == [None]
    assert parse_verdict_batch(None, 2) == [None, None]
    assert parse_verdict_batch({"verdicts": "not a list"}, 2) == [None, None]


def test_batch_verify_costs_one_call_and_gaps_stay_out_of_denominator():
    llm = FakeLlm([{"verdicts": [{"i": 1, "verdict": 1}, {"i": 2, "verdict": 1}]}])
    runner = RagasRunner(llm, batch_verify=True)
    got = runner.verify_set(["甲", "乙", "丙"], ["证据"])
    assert runner.cost_summary()["calls"] == 1                       # 三条陈述一趟判完
    assert got["score"]["n_gradable"] == 2 and got["score"]["n_unparsable"] == 1
    assert got["score"]["value"] == 1.0                              # 分母是没漏答的那 2 条
    assert got["verdicts"][2]["verdict"] is None


def test_batch_prompt_numbers_every_statement_and_carries_all_contexts():
    prompt = build_verify_batch_prompt(["陈述甲", "陈述乙"], ["材料一", "材料二"])
    assert "1. 陈述甲" in prompt and "2. 陈述乙" in prompt
    assert "必须覆盖全部 2 条" in prompt and "材料二" in prompt


def test_single_statement_mode_still_available_for_the_original_algorithm():
    llm = FakeLlm([{"verdict": 1, "basis": "有"}, {"verdict": 0, "basis": "无"}])
    runner = RagasRunner(llm, batch_verify=False)
    got = runner.verify_set(["甲", "乙"], ["证据"])
    assert runner.cost_summary()["calls"] == 2                        # 官方口径：一条一次调用
    assert [v["verdict"] for v in got["verdicts"]] == [1, 0]
    assert got["verdicts"][0]["basis"] == "有"


# ---------------- 行级失败不许带走整轮，也不许静默变成 0 分 ----------------

class RaisingLlm(FakeLlm):
    def __init__(self, script: list[Any], raise_at: int) -> None:
        super().__init__(script)
        self.raise_at, self.n = raise_at, 0

    def chat(self, messages, **kw):
        self.n += 1
        if self.n == self.raise_at:
            raise RuntimeError("7B 炸了")
        return super().chat(messages, **kw)


def test_run_pass_survives_a_row_error_and_keeps_it_out_of_the_denominator():
    rows = [{"qid": "q-001", "status": "answer", "question": "题一", "conclusion": "答一", "cited_ids": ["c1"]},
            {"qid": "q-002", "status": "answer", "question": "题二", "conclusion": "答二", "cited_ids": ["c1"]}]
    texts, ctx_of = {"c1": "正文"}, {}
    llm = RaisingLlm([{"statements": ["甲"]}, {"verdicts": [{"i": 1, "verdict": 1}]},
                      {"noncommittal": False, "questions": ["问一"]}, {"label": 1, "reason": "成立"},
                      {"label": 1, "reason": "成立"}], raise_at=5)
    out = ev.run_pass(rows, texts, ctx_of, RagasRunner(llm), LlmJudge(llm))
    assert "row_error" in out[1] and out[1]["faith_cited"]["value"] is None
    assert out[0]["faith_cited"]["value"] == 1.0 and out[0]["judge"]["label"] == 1
    s = ev.summarize(out, {"q-001": 1, "q-002": 1})
    assert [e["qid"] for e in s["row_errors"]] == ["q-002"]
    assert s["faithfulness_cited"]["n"] == 1                          # 坏行不进分母


def test_run_pass_marks_decompose_failure_instead_of_crashing():
    """崩掉全量重跑的那次事故就是这个形状：拆句返回坏 JSON，`statements` 是 None。"""
    rows = [{"qid": "q-001", "status": "answer", "question": "题", "conclusion": "答", "cited_ids": ["c1"]}]
    llm = FakeLlm([{"no_statements_key": 1}, {"noncommittal": False, "questions": ["问"]}, {"label": 1}])
    out = ev.run_pass(rows, {"c1": "正文"}, {}, RagasRunner(llm), LlmJudge(llm))
    assert out[0]["statements"] is None and out[0]["decompose_failed"] is True
    assert out[0]["faith_cited"]["value"] is None and "不适用" in out[0]["faith_cited"]["reading"]
    s = ev.summarize(out, {"q-001": 1})
    assert s["faithfulness_cited"]["n_decompose_failed"] == 1


# ---------------- Answer Relevance 的官方口径 ----------------

def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert abs(cosine([1, 2], [2, 4]) - 1.0) < 1e-9        # 方向相同即 1，不看模长
    assert cosine([], [1]) == 0.0 and cosine([1], [1, 2]) == 0.0


def test_noncommittal_is_zero_by_ragas_rule_and_says_so():
    out = answer_relevance_from_vectors([1.0, 0.0], [], True)
    assert out["value"] == 0.0 and "RAGAS" in out["reading"]


def test_unjudgeable_relevance_is_none_not_zero():
    assert answer_relevance_from_vectors([1.0, 0.0], [], False)["value"] is None


def test_relevance_is_mean_of_cosines():
    out = answer_relevance_from_vectors([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], False)
    assert out["value"] == 0.5 and out["n_questions"] == 2 and out["sims"] == [1.0, 0.0]


# ---------------- LLM 裁判：判据、成本、与人判的对照 ----------------

def test_judge_prompt_carries_rubric_and_refuses_reference_answer():
    prompt = build_judge_prompt(ANSWER_CORRECT, "题面甲", "结论乙", ["证据丙"])
    assert "题面甲" in prompt and "结论乙" in prompt and "证据丙" in prompt
    assert "不给参考答案" in prompt                       # 给了就是抄，这条判据不能丢
    assert "（该回答没有引用任何材料）" in build_judge_prompt(ANSWER_CORRECT, "题", "答", [])


def test_disposition_rubric_bounds_the_observation_field():
    """拒答行的观测域是「检到的 Top-5」，不是全库——这句话丢了就会把「该答却检不到」误记成误拒。"""
    prompt = build_judge_prompt(DISPOSITION_CORRECT, "题面", "", ["材料一"])
    assert "该不该" in prompt and "不是全库" in prompt and "检索缺口" in prompt
    assert "材料一" in prompt


def test_judge_row_picks_rubric_by_whether_a_conclusion_exists():
    texts = {"c1": "被引正文", "c2": "检到的 Top-5 正文"}
    llm = FakeLlm([{"label": 1, "reason": "成立", "span": "x"}, {"label": 1, "reason": "该拒", "span": "y"}])
    judge = LlmJudge(llm)
    answered = ev.judge_row(judge, {"qid": "q-001", "question": "题", "conclusion": "答", "cited_ids": ["c1"]},
                            texts, {"q-001": ["c2"]})
    refused = ev.judge_row(judge, {"qid": "q-059", "question": "题", "conclusion": "", "cited_ids": []},
                           texts, {"q-059": ["c2"]})
    assert answered["rubric"] == ANSWER_CORRECT.id and "被引正文" in llm.seen[0]["prompt"]
    assert refused["rubric"] == DISPOSITION_CORRECT.id and "检到的 Top-5 正文" in llm.seen[1]["prompt"]


def test_judge_row_refuses_to_judge_disposition_without_evidence_face():
    """复跑检索被跳过 ⇒ 拒答行没有证据面可判，只能记没判，不得拿空材料判「该不该拒」。"""
    judge = LlmJudge(FakeLlm([]))
    got = ev.judge_row(judge, {"qid": "q-059", "question": "题", "conclusion": "", "cited_ids": []}, {}, {})
    assert got["label"] is None and got["why"] == "no_evidence_face_skipped_retrieval"
    assert judge.cost_summary()["calls"] == 0                 # 没判就一次调用都不该花


def test_judge_records_cost_and_flags_no_material_as_unjudgeable():
    judge = LlmJudge(FakeLlm([None]))
    got = judge.judge("题", "答", ["证据"])
    assert got["label"] is None and got["why"] == "no_envelope"
    assert judge.cost_summary()["calls"] == 1 and judge.cost_summary()["failed"] == 1


def test_agreement_only_counts_rows_both_sides_judged():
    rows = [{"qid": "q-001", "human": 1, "judge": 1},
            {"qid": "q-002", "human": 0, "judge": 1},
            {"qid": "q-003", "human": 1, "judge": None},
            {"qid": "q-004", "human": 0, "judge": 0}]
    out = agreement_with_human(rows)
    assert out["n_paired"] == 3 and out["n_excluded"] == 1
    assert out["confusion"] == {"human1_judge1": 1, "human1_judge0": 0,
                               "human0_judge1": 1, "human0_judge0": 1}
    assert out["value"] == round(2 / 3, 4)
    assert [d["qid"] for d in out["divergent"]] == ["q-002"]
    assert out["recall_on_human_wrong"] == 0.5            # 真人判错 2 行，裁判抓到 1 行


def test_yes_sayer_shape_has_high_agreement_but_zero_recall():
    """一致率高不等于有用：裁判照抄基率点头时，抓到率必须是 0、κ 必须是负的。

    这条测试是补上算反 recall 那个 bug 之后加的——上一版用对称数据（2 行人判错、
    裁判各判一半），把 recall 的分子误取成「不一致的行数」也能凑出同一个 0.5，测不出方向错。
    """
    rows = [{"qid": "q-001", "human": 1, "judge": 1},
            {"qid": "q-002", "human": 1, "judge": 1},
            {"qid": "q-003", "human": 0, "judge": 1},
            {"qid": "q-004", "human": 0, "judge": 1},
            {"qid": "q-005", "human": 1, "judge": 0}]
    out = agreement_with_human(rows)
    assert out["value"] == 0.4                                     # 一致率看着还行
    assert out["kappa"]["kappa"] < 0                               # κ 却是负的：没超出偶然
    assert out["judge_yes_rate"] == 0.8                            # 八成都在说「成立」
    assert out["n_human_wrong"] == 2 and out["recall_on_human_wrong"] == 0.0
    assert out["precision_on_flagged"] == 0.0                      # 它只误报了一行


def test_agreement_with_nothing_paired_is_not_applicable():
    out = agreement_with_human([{"qid": "q-001", "human": 1, "judge": None}])
    assert out["value"] is None and "不适用" in out["reading"] and out["n_paired"] == 0


# ---------------- Context Precision / Recall：名次加权与归因覆盖 ----------------

def test_context_precision_is_position_weighted_not_plain_ratio():
    """相关块落在第 1、3 名 ⇒ (P@1 + P@3)/2 = (1.0 + 0.6667)/2 = 0.8333，不是 2/3。"""
    out = context_precision_from_verdicts([1, 0, 1])
    assert out["precision_at_k"] == [1.0, 0.6667]
    assert out["value"] == 0.8333 and out["n_relevant"] == 2


def test_context_precision_prefers_relevant_early():
    early = context_precision_from_verdicts([1, 1, 0])["value"]
    late = context_precision_from_verdicts([0, 1, 1])["value"]
    assert early > late                       # 同样的相关块数，排得靠前的分更高——这才是这条指标的意义


def test_context_precision_with_no_relevant_block_is_not_applicable():
    out = context_precision_from_verdicts([0, 0, 0])
    assert out["value"] is None and out["n_relevant"] == 0 and "不适用" in out["reading"]
    assert context_precision_from_verdicts([])["value"] is None


def test_context_precision_leaves_missing_verdicts_out():
    out = context_precision_from_verdicts([1, None, 1])
    assert out["n_unjudged"] == 1 and out["n_relevant"] == 2
    assert out["precision_at_k"] == [1.0, 0.6667] and out["value"] == 0.8333


def test_context_recall_counts_covered_claims_over_gradable_only():
    out = context_recall_from_verdicts([1, None, 0])
    assert out == {**out, "n_claims": 3, "n_gradable": 2, "n_unparsable": 1, "n_covered": 1, "value": 0.5}
    assert context_recall_from_verdicts([])["value"] is None
    assert "不适用" in context_recall_from_verdicts([None])["reading"]


def test_context_precision_prompt_numbers_contexts_by_rank_and_carries_reference():
    prompt = build_context_precision_prompt("题", "参考答案", ["块一", "块二"])
    assert "[1] 块一" in prompt and "[2] 块二" in prompt
    assert "参考答案" in prompt and "必须逐块给全" in prompt
    assert "别的口径 ⇒ 0" in prompt          # 主题相同但不正面回答 ⇒ 0，这条不能丢


def test_context_recall_prompt_judges_claims_against_contexts_only():
    prompt = build_context_recall_prompt(["声明甲"], ["块一"])
    assert "1. 声明甲" in prompt and "[1] 块一" in prompt and "不许用你自己的知识补链" in prompt


def test_unjudged_context_gets_no_credit_instead_of_promoting_the_ones_behind_it():
    """判不出的那块不许从序列里删掉——删了就等于给后面的块升位，用一次故障换一个好看的 Precision。"""
    missing = context_precision_from_verdicts([1, None, 1])
    treated_as_irrelevant = context_precision_from_verdicts([1, 0, 1])
    dropped_from_sequence = context_precision_from_verdicts([1, 1])      # 名次压缩后的形状，分更高
    assert missing["value"] == treated_as_irrelevant["value"] == 0.8333
    assert missing["n_unjudged"] == 1
    assert dropped_from_sequence["value"] > missing["value"]


def test_runner_context_precision_uses_one_batch_call():
    llm = FakeLlm([{"verdicts": [{"i": 1, "verdict": 1}, {"i": 2, "verdict": 0}]}])
    runner = RagasRunner(llm)
    got = runner.context_precision("题", "参考", ["块一", "块二"])
    assert runner.cost_summary()["calls"] == 1
    assert got["score"]["value"] == 1.0 and got["verdicts"][1]["verdict"] == 0


def test_runner_context_recall_decomposes_reference_then_judges_once():
    llm = FakeLlm([{"statements": ["声明甲", "声明乙"]},
                   {"verdicts": [{"i": 1, "verdict": 1}, {"i": 2, "verdict": 0}]}])
    runner = RagasRunner(llm)
    got = runner.context_recall("题", "参考答案", ["块一"])
    assert runner.cost_summary()["calls"] == 2
    assert got["claims"] == ["声明甲", "声明乙"]
    assert got["score"]["value"] == 0.5 and got["skipped"] == ""


def test_runner_context_metrics_skip_instead_of_judging_nothing():
    """没有检索面 / 拆不出声明 ⇒ 不花调用、报不适用（拿空材料判 0 会把「没检到」与「判不出」混成一档）。"""
    runner = RagasRunner(FakeLlm([]))
    assert runner.context_precision("题", "参考", [])["skipped"] == "no_contexts"
    assert runner.context_recall("题", "参考", [])["skipped"] == "no_contexts"
    assert runner.cost_summary()["calls"] == 0
    empty = RagasRunner(FakeLlm([{"statements": []}]))
    assert empty.context_recall("题", "参考", ["块一"])["skipped"] == "no_claims"


# ---------------- 护栏闸门：装置空转必须判废 ----------------

def _controls(faith_shuffled=0.0, faith_real=1.0, judge_shuffled=0, judge_real=1,
              empty_ctx_value=0.0, empty_ans_value=None):
    return {
        "n": 2,
        "mismatched_evidence": [
            {"qid": "q-001", "donor": "q-002", "faith_real": faith_real, "faith_shuffled": faith_shuffled,
             "judge_real": judge_real, "judge_shuffled": judge_shuffled},
            {"qid": "q-003", "donor": "q-004", "faith_real": faith_real, "faith_shuffled": faith_shuffled,
             "judge_real": judge_real, "judge_shuffled": judge_shuffled},
        ],
        "empty_contexts": {"qid": "q-001", "value": empty_ctx_value, "n_statements": 2,
                           "reading": "0/2 = 0.00%"},
        "empty_answer": {"qid": "q-001", "value": empty_ans_value, "statements": [], "reading": "不适用（无可判陈述）"},
    }


def _ar_ctl(delta=0.3):
    return [{"qid": "q-001", "donor": "q-002", "own": 0.8, "mismatched": 0.8 - delta, "delta": delta}]


def _ctx_ctl(cp_drop=0.4, cr_drop=0.3):
    return {"n": 2, "rows": [
        {"qid": "q-001", "donor": "q-002", "cp_own": 0.8, "cp_wrong": round(0.8 - cp_drop, 4),
         "cr_own": 0.9, "cr_wrong": round(0.9 - cr_drop, 4)},
        {"qid": "q-003", "donor": "q-004", "cp_own": 0.7, "cp_wrong": round(0.7 - cp_drop, 4),
         "cr_own": 0.8, "cr_wrong": round(0.8 - cr_drop, 4)}]}


def test_gate_passes_when_every_control_moves():
    assert ev.control_gate(_controls(), _ar_ctl(), _ctx_ctl())["all_pass"] is True


def test_gate_fails_when_shuffled_evidence_does_not_move_faithfulness():
    """证据换成别题的而分数不掉 ⇒ 装置没在看证据，必须判废，不能让它出一个「看起来正常」的均值。"""
    gate = ev.control_gate(_controls(faith_shuffled=1.0), _ar_ctl(), _ctx_ctl())
    assert gate["all_pass"] is False
    assert gate["checks"]["证据错配后 Faithfulness 下降"]["pass"] is False


def test_gate_fails_when_judge_never_flips_on_mismatched_evidence():
    gate = ev.control_gate(_controls(judge_shuffled=1), _ar_ctl(), _ctx_ctl())
    assert gate["checks"]["证据错配后裁判 label 会翻"]["pass"] is False
    assert gate["all_pass"] is False


def test_gate_fails_when_answer_relevance_ignores_the_answer():
    gate = ev.control_gate(_controls(), _ar_ctl(delta=-0.2), _ctx_ctl())
    assert gate["checks"]["答案错配后 Answer Relevance 下降"]["pass"] is False
    assert gate["all_pass"] is False


def test_gate_fails_when_empty_contexts_is_not_zero():
    """一条材料都不给却仍判成立 ⇒「无从反驳」被读成了「有据」，这是最危险的一种假绿。"""
    gate = ev.control_gate(_controls(empty_ctx_value=1.0), _ar_ctl(), _ctx_ctl())
    assert gate["checks"]["空证据面不得判成立"]["pass"] is False
    assert gate["all_pass"] is False


def test_gate_fails_when_empty_answer_becomes_1_0():
    gate = ev.control_gate(_controls(empty_ans_value=1.0), _ar_ctl(), _ctx_ctl())
    assert gate["checks"]["空回答面不得读成 1.0"]["pass"] is False
    assert gate["all_pass"] is False


# ---------------- 描述统计与逐行装配 ----------------

def test_gate_fails_when_swapping_the_retrieval_surface_does_not_move_context_precision():
    """CP 是按名次加权的，若模型其实没逐块看，换成别题的面也一样给分——那条指标就是装饰。"""
    gate = ev.control_gate(_controls(), _ar_ctl(), _ctx_ctl(cp_drop=0.0))
    assert gate["checks"]["检索面错配后 Context Precision 下降"]["pass"] is False
    assert gate["all_pass"] is False


def test_gate_fails_when_context_recall_ignores_the_retrieval_surface():
    gate = ev.control_gate(_controls(), _ar_ctl(), _ctx_ctl(cr_drop=-0.1))
    assert gate["checks"]["检索面错配后 Context Recall 下降"]["pass"] is False
    assert gate["all_pass"] is False


def test_gate_now_carries_seven_checks():
    gate = ev.control_gate(_controls(), _ar_ctl(), _ctx_ctl())
    assert len(gate["checks"]) == 7 and gate["all_pass"] is True


def test_gate_counts_undefined_cp_on_swapped_surface_as_max_drop():
    """换成别题的面之后「一块都不相关」⇒ CP 未定义。读数口径必须报不适用，
    但这道护栏要把它认成**最大下降**，否则一次完全失灵会被读成「没测到」而放过。"""
    ctl = {"n": 2, "rows": [
        {"qid": "q-001", "donor": "q-002", "cp_own": 0.8, "cp_wrong": None, "cr_own": 0.9, "cr_wrong": 0.2},
        {"qid": "q-003", "donor": "q-004", "cp_own": 0.6, "cp_wrong": None, "cr_own": 0.8, "cr_wrong": 0.1}]}
    gate = ev.control_gate(_controls(), _ar_ctl(), ctl)
    check = gate["checks"]["检索面错配后 Context Precision 下降"]
    assert check["pass"] is True and "未定义" in check["basis"]
    assert gate["all_pass"] is True


def test_gate_rows_with_no_own_cp_stay_out_of_the_comparison():
    ctl = {"n": 1, "rows": [{"qid": "q-001", "donor": "q-002", "cp_own": None, "cp_wrong": 0.0,
                             "cr_own": None, "cr_wrong": 0.0}]}
    gate = ev.control_gate(_controls(), _ar_ctl(), ctl)
    assert gate["checks"]["检索面错配后 Context Precision 下降"]["pass"] is False

def test_run_pass_computes_context_metrics_from_the_retrieved_surface():
    """CP/CR 喂的是复跑 Top-5（不是被引块），且拒答行也要算——「该答却检不到」正是在这里暴露。"""
    rows = [{"qid": "q-001", "status": "answer", "question": "题", "conclusion": "答",
             "cited_ids": ["c1"], "reference_answer": "参考答案"}]
    texts = {"c1": "被引正文", "c2": "另一块检索面"}
    script = [{"statements": ["甲"]}, {"verdicts": [{"i": 1, "verdict": 1}]},
              {"verdicts": [{"i": 1, "verdict": 1}]}, {"noncommittal": False, "questions": ["问一"]},
              {"label": 1, "reason": "成立"},
              {"verdicts": [{"i": 1, "verdict": 1}, {"i": 2, "verdict": 0}]},
              {"statements": ["声明乙"]}, {"verdicts": [{"i": 1, "verdict": 1}]}]
    llm = FakeLlm(script)
    out = ev.run_pass(rows, texts, {"q-001": ["c1", "c2"]}, RagasRunner(llm), LlmJudge(llm))
    rec = out[0]
    assert rec["context_precision"]["value"] == 1.0          # 相关块在第 1 名 → P@1 = 1
    assert rec["context_recall"]["value"] == 1.0 and rec["reference_claims"] == ["声明乙"]
    assert "参考答案" in llm.seen[5]["prompt"] and "另一块检索面" in llm.seen[5]["prompt"]
    assert "[1] 被引正文" in llm.seen[5]["prompt"]           # 名次是真按检索顺序标的


def test_run_pass_without_reference_marks_context_metrics_not_assessable():
    rows = [{"qid": "q-001", "status": "answer", "question": "题", "conclusion": "答",
             "cited_ids": ["c1"], "reference_answer": ""}]
    out = ev.run_pass(rows, {"c1": "正文"}, {"q-001": ["c1"]},
                      RagasRunner(FakeLlm([{"statements": ["甲"]}, {"verdicts": [{"i": 1, "verdict": 1}]},
                                            {"noncommittal": False, "questions": ["问"]}, {"label": 1}])),
                      LlmJudge(FakeLlm([])))
    assert out[0]["context_precision"]["value"] is None and out[0]["context_recall"]["value"] is None


def test_summarize_exposes_the_embedding_scale_anchor():
    """Answer Relevancy 对目标值时必须能看见「这套 embedding 下完全切题长什么样」，
    否则量纲不匹配会被读成质量不达标。"""
    rows = [{"qid": "q-001", "status": "answer", "statements": ["甲"],
             "faith_cited": {"value": 1.0}, "faith_top5": {"value": 1.0},
             "answer_relevance": {"value": 0.66}, "scale_anchor": {"cos_question_vs_reference": 0.70},
             "judge": {"label": 1}},
            {"qid": "q-059", "status": "refuse", "statements": [],
             "faith_cited": {"value": None}, "faith_top5": {"value": None},
             "answer_relevance": {"value": 0.0}, "scale_anchor": {"cos_question_vs_reference": 0.55},
             "judge": {"label": 1}}]
    out = ev.summarize(rows, {"q-001": 1, "q-059": 1})
    anc = out["answer_relevance_scale_anchor"]
    assert anc["n"] == 1 and anc["mean"] == 0.7          # 只量作答行，与 AR 同种群才可比
    assert out["answer_relevance_answered"]["mean"] == 0.66


def test_describe_empty_is_not_reported_as_zero():
    out = ev.describe([])
    assert out["n"] == 0 and out["mean"] is None


def test_describe_carries_n_and_ci():
    out = ev.describe([1.0, 0.5, 0.0])
    assert out["n"] == 3 and out["mean"] == 0.5 and out["zeros"] == 1 and len(out["ci95"]) == 2


def test_summarize_splits_answered_from_all_rows():
    """拒答行按 RAGAS 口径记 0，会拉低全行档——所以两档必须分开出，且不适用行不进均值。"""
    rows = [
        {"qid": "q-001", "status": "answer", "statements": ["a", "b"],
         "faith_cited": {"value": 1.0, "reading": "2/2"}, "faith_top5": {"value": 1.0, "reading": "2/2"},
         "answer_relevance": {"value": 0.8}, "judge": {"label": 1}},
        {"qid": "q-002", "status": "tentative", "statements": ["a"],
         "faith_cited": {"value": 0.0, "reading": "0/1"}, "faith_top5": {"value": 0.0, "reading": "0/1"},
         "answer_relevance": {"value": 0.4}, "judge": {"label": 0}},
        {"qid": "q-059", "status": "refuse", "statements": [],
         "faith_cited": {"value": None, "reading": "不适用（无可判陈述）"},
         "faith_top5": {"value": None, "reading": "不适用（无可判陈述）"},
         "answer_relevance": {"value": 0.0}, "judge": {"label": 1}},
    ]
    labels = {"q-001": 1, "q-002": 0, "q-059": 1}
    out = ev.summarize(rows, labels)
    assert out["population"] == {"n_rows": 3, "n_answered": 2, "n_non_answer": 1}
    assert out["faithfulness_cited"]["n"] == 2 and out["faithfulness_cited"]["n_not_assessable"] == 0
    assert out["answer_relevance_answered"]["n"] == 2 and out["answer_relevance_all_rows"]["n"] == 3
    assert out["answer_relevance_all_rows"]["zeros"] == 1
    assert out["judge_vs_human_all"]["n_paired"] == 3


def test_ar_relative_verdict_needs_both_gaps():
    """C5 裁决（换相对判据）的闸门：两条 gap **必须同时**成立才算过。

    取 AND 不取 OR 是因为"换判据"天然朝变松方向走——只卡错位面，那么一个
    比参考答案低得多的 AR 也能过；只卡标尺，那么"跟别题答案一样切题"也能过。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("m5re", ROOT / "scripts" / "m5_ragas_eval.py")
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)

    def sum_with(ar):
        return {"answer_relevance_answered": {"mean": ar},
                "answer_relevance_scale_anchor": {"mean": 0.6662}}

    ctl = [{"mismatched": 0.4588}]
    ok = m5.ar_relative_verdict(sum_with(0.6582), ctl)
    assert ok["pass"] is True and ok["gap_over_mismatch"] == 0.1994 and ok["gap_under_anchor"] == 0.008
    low = m5.ar_relative_verdict(sum_with(0.5), ctl)          # 贴近错位面 ⇒ 不过
    assert low["pass"] is False and low["gap_over_mismatch"] < m5.AR_MIN_OVER_MISMATCH
    # 两条门槛之间只有 0.0074 的缝（0.6088 ≤ AR < 0.6162 才只超一条）——
    # 反例必须挑进这条缝里，随便取个低值会让两个条件一起失败、测不出 AND 有没有生效。
    off = m5.ar_relative_verdict(sum_with(0.612), ctl)
    assert off["gap_over_mismatch"] >= m5.AR_MIN_OVER_MISMATCH, off["gap_over_mismatch"]
    assert off["gap_under_anchor"] > m5.AR_MAX_UNDER_ANCHOR and off["pass"] is False
    assert m5.ar_relative_verdict({}, [])["pass"] is None      # 缺对照 ⇒ 不判，不许默认通过
