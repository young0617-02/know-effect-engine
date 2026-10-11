"""置信度配置的一致性测试（M3-04c）。

这条链上的事故不是「算错了」，而是**测的和上线的不是一个配置**：
API 从 Gate 0 的 `calibration.json` 读 T=6.655，探针用默认 1.0，于是对外报出的
「answer 42 / tentative 11」描述的是一套从没上线过的配置，而真实服务在该温度下
`answer` **结构上永不可达**（可达上界 0.6388 < 门槛 0.75）。

所以这里钉的是配置面，不是算法面：
  * 温度只有一个来源函数，API 与探针都走它；
  * Gate 0 那份退化值**不会被回落到使用**——文件缺失/refused/非法值一律 1.0；
  * 对外分数标着它是什么（规则分），不让人当概率用。
"""

from __future__ import annotations

import json
from pathlib import Path

from kee.api.app import _load_calibration
from kee.qa.answer import AnswerEngine, load_calibration_temperature
from kee.retrieval.pipeline import RetrievedChunk, RetrievalResult
from kee.retrieval.rbac import AccessContext


class FakeSettings:
    """只暴露 loader 用到的那一个方法，避免测试依赖真实 artifact 目录。"""

    def __init__(self, d: Path):
        self._d = d

    def resolved_artifact_dir(self) -> Path:
        return self._d


def write(d: Path, name: str, payload) -> None:
    (d / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_temperature_falls_back_to_identity_not_to_the_gate0_degenerate_value(tmp_path):
    """Gate 0 的 6.655 是零方差标签拟合出来的，任何情况下都不许被当成参数用起来。"""
    write(tmp_path, "calibration.json", {"temperature": 6.655, "source": "gate0-⑤"})
    assert load_calibration_temperature(FakeSettings(tmp_path)) == 1.0


def test_refused_marker_forces_identity_even_if_a_number_is_present(tmp_path):
    """带 refused 的记录就是「这次没拟合出可用值」，文件里同时写着数字也不采信。"""
    write(tmp_path, "confidence_calibration.json", {"temperature": 0.05, "refused": "零方差"})
    assert load_calibration_temperature(FakeSettings(tmp_path)) == 1.0


def test_m3_file_is_the_only_source_and_bad_values_are_ignored(tmp_path):
    write(tmp_path, "confidence_calibration.json", {"temperature": 2.5})
    assert load_calibration_temperature(FakeSettings(tmp_path)) == 2.5
    for bad in ({"temperature": 0}, {"temperature": -1}, {"temperature": "abc"}, "not a dict", {}):
        write(tmp_path, "confidence_calibration.json", bad)
        assert load_calibration_temperature(FakeSettings(tmp_path)) == 1.0, bad


def test_missing_file_is_identity_and_not_an_error(tmp_path):
    assert load_calibration_temperature(FakeSettings(tmp_path)) == 1.0


def test_api_uses_the_exact_same_number_as_the_offline_probe(tmp_path):
    """两套入口一个数字。这条就是「测的和上线的不是一个配置」的回归防线。"""
    write(tmp_path, "confidence_calibration.json", {"temperature": 1.8})
    fs = FakeSettings(tmp_path)
    assert _load_calibration(fs)["temperature"] == load_calibration_temperature(fs) == 1.8


def test_service_path_temperature_is_not_the_gate0_value_in_the_repo():
    """仓库里 Gate 0 那份仍是 6.655（历史证据，不改），但服务路径读到的必须是 1.0。

    这条断言的意义是：如果哪天有人把 loader 改回去读 `calibration.json`，
    它会立刻失败，而不是安静地把 answer 状态变成永不可达。
    """
    from kee.config import get_settings

    gate0 = Path(get_settings().resolved_artifact_dir()) / "calibration.json"
    if gate0.exists():
        assert json.loads(gate0.read_text(encoding="utf-8"))["temperature"] != \
            load_calibration_temperature(), "服务路径的温度又变回 Gate 0 的退化拟合值了"


# ---- 分数性质必须在响应里写明 ----

class FakeRetriever:
    rewriter = None

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        c = RetrievedChunk(
            chunk_id="c1", text="证据正文 c1", score=0.95,
            fields={"chunk_id": "c1", "doc_id": "d", "project": "p", "section_no": "1.1",
                    "page": 1, "security_level": 1, "parent_id": "d#p"},
        )
        return RetrievalResult(query=query, mode="hybrid", chunks=[c],
                               timings_ms={"retrieval_total_ms": 1.0}, expr="", degraded=[])


class FakeLlm:
    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        from kee.llm.ollama import LlmResult

        payload = {
            "conclusion": "RTO 小于 15 分钟。",
            "key_points": [{"point": "灾备 RTO < 15 分钟", "evidence_ids": ["E1"]}],
            "citations": [{"evidence_id": "E1"}],
            "no_evidence": False,
            "self_assessed_confidence": 0.9,
        }
        return LlmResult(ok=True, text=json.dumps(payload, ensure_ascii=False), parsed=None,
                         json_valid=True, metrics={"total_ms": 1.0})


def _answer():
    return AnswerEngine(FakeRetriever(), FakeLlm()).answer("RTO 是多少", AccessContext(role="knowledge_owner"))


def test_response_labels_the_score_as_a_rule_not_a_probability():
    res = _answer()
    assert res.confidence["score_basis"] == "rule_composite_not_probability"


def test_answer_status_is_reachable_at_identity_temperature():
    """T=1.0 下强证据必须能走到 `answer`——这正是在验证「不可达」已被修掉。

    反证来自仓库里那份 Gate 0 的 T：同一 composite 经 sigmoid(logit/6.655) 只会落在 0.6 一带，
    低于 confidence_high，于是对外永远是「初步判断」。
    """
    import math

    res = _answer()
    assert res.status == "answer", f"实际 {res.status} / notes {res.notes}"
    comp = res.confidence["composite"]
    squeezed = 1 / (1 + math.exp(-(math.log(comp / (1 - comp)) / 6.655)))
    assert squeezed < 0.75, "旧温度下本该不可达，这条反证失效说明门槛被改过——去看方案 6.4"
