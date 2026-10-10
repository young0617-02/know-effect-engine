"""答案生成 + 引用强制 + 复合置信度（Gate 0 ⑤ 的被试机制，M3 会换成 LangGraph 编排）。

三条硬规则（对应方案 6.4 / Must 4·5·6）：
  1. 引用强制：要点必须挂证据编号，证据编号必须在本次检索集合内；无任何有效引用 → 不出结论。
  2. 复合置信度：检索分 + 引用一致性 + 模型自评 → 温度缩放校准 → 分档。
  3. 确定性降级：无依据=拒答；低置信=「初步判断」并转人工；敏感话题=直接转人工。
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from kee.config import get_settings
from kee.llm.ollama import OllamaClient
from kee.retrieval.pipeline import HybridRetriever, RetrievalResult
from kee.retrieval.rbac import AccessContext

SENSITIVE_PATTERNS = [
    "报价",
    "合同金额",
    "薪酬",
    "工资",
    "口令",
    "密码",
    "密钥",
    "账号",
    "特权",
    "客户名单",
    "折扣",
    "回款",
    "付款",
    "法务追责",
]

# 命中密级 3 的证据一律转人工：有权限看 ≠ 可以让模型直接下结论
ESCORT_SECURITY_LEVEL = 3

SYSTEM_PROMPT = (
    "你是数字技术服务企业的交付知识问答引擎，只依据【证据】回答。\n"
    "硬性规则：\n"
    "1) 结论与每一条要点都必须标注证据编号（形如 E1、E2），证据编号只能来自给定证据；\n"
    "2) 证据不足以支撑结论时，把 no_evidence 置为 true，conclusion 与 key_points 留空，禁止凭常识补全；\n"
    "3) 不得输出证据之外的数字、编号、产品名或承诺；\n"
    "4) 只输出 JSON，不要任何解释文字或 markdown 代码块。"
)

USER_TEMPLATE = (
    "【证据】\n{evidence}\n\n"
    "【问题】{question}\n\n"
    "【输出 JSON 结构】\n"
    '{{"conclusion": "一句话结论（无依据则空字符串）", '
    '"key_points": [{{"point": "要点", "evidence_ids": ["E1"]}}], '
    '"citations": [{{"evidence_id": "E1", "doc_id": "...", "section_no": "...", "page": 0}}], '
    '"no_evidence": false, "self_assessed_confidence": 0.0}}'
)
# ↑ 刻意**没有**改成「conclusion 不得为空」。实测：加上该要求后可判答案完成率从 13/58 升到 58/58，
#   但 7 道语料里确实无答案的题从 0 道出结论变成 2 道出结论，其中 q-059 给出
#   「终验结论于2024年12月20日出的，一次通过率为93.7%」并带 1 条引用、needs_human=False。
#   那是把「该答不答」换成了「不该答也答」，对知识服务是更坏的方向。见 M3-01c 与报告第四节。

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _logit(p: float, eps: float = 1e-4) -> float:
    p = _clamp(p, eps, 1 - eps)
    return math.log(p / (1 - p))


def is_sensitive(text: str) -> str | None:
    for pat in SENSITIVE_PATTERNS:
        if pat in text:
            return pat
    return None


@dataclass
class AnswerResult:
    question: str
    status: str  # answer / tentative / refuse / escalate / degraded
    conclusion: str = ""
    key_points: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    dropped_points: list[str] = field(default_factory=list)
    needs_human: bool = False
    confidence: dict[str, float] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace_id: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_evidence(res: RetrievalResult) -> tuple[str, dict[str, dict]]:
    lines: list[str] = []
    index: dict[str, dict] = {}
    for i, c in enumerate(res.chunks, start=1):
        eid = f"E{i}"
        f = c.fields
        lines.append(
            f"[{eid}] (doc={f.get('doc_id')} 项目={f.get('project')} 章节={f.get('section_no')} "
            f"页={f.get('page')} 密级={f.get('security_level')}) {c.text}"
        )
        index[eid] = {
            "chunk_id": c.chunk_id,
            "doc_id": f.get("doc_id"),
            "project": f.get("project"),
            "section_no": f.get("section_no"),
            "page": f.get("page"),
            "security_level": f.get("security_level"),
            "parent_id": f.get("parent_id"),
            "score": c.score,
        }
    return "\n".join(lines), index


class AnswerEngine:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OllamaClient | None = None,
        min_retrieval_score: float = 0.20,
        strong_evidence_floor: float = 0.90,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever
        self.llm = llm or OllamaClient()
        self.min_retrieval_score = min_retrieval_score
        # 0.90 不是精调出来的：Gate 0 ⑤ 里该转人工的 q-070 是 0.9683、该拒答的 q-059 是 0.8745，
        # 中间只有 0.09 的间隔且负类仅 7 个样本。取 0.90 是「贴着观测到的间隔上沿」而非「已验证的分离点」，
        # 所以这条阈值必须随 M3-01c 的答案链复测重新标定，不能当定论引用。
        self.strong_evidence_floor = strong_evidence_floor
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calibration_temperature: float = 1.0

    # ---- 校验 ----
    @staticmethod
    def _extract_json(text: str) -> dict | None:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            m = _JSON_BLOCK.search(text or "")
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:  # noqa: BLE001
                return None

    def _enforce_citations(
        self, obj: dict, evidence_index: dict[str, dict]
    ) -> tuple[list[dict], list[dict], list[str]]:
        kept: list[dict] = []
        dropped: list[str] = []
        used: set[str] = set()
        for kp in obj.get("key_points", []) or []:
            if not isinstance(kp, dict):
                dropped.append(str(kp)[:80])
                continue
            ids = [e for e in (kp.get("evidence_ids") or []) if e in evidence_index]
            if not ids:
                dropped.append(str(kp.get("point", ""))[:80])
                continue
            kept.append({"point": str(kp.get("point", "")), "evidence_ids": ids})
            used.update(ids)
        conclusion = str(obj.get("conclusion", "") or "")
        cites = [
            {**evidence_index[e], "evidence_id": e}
            for e in sorted(used)
            if e in evidence_index
        ]
        if conclusion and not cites:
            # 结论无出处：整条降级为拒答，由调用方处理
            return [], [], dropped + ["conclusion_without_citation"]
        return kept, cites, dropped

    # ---- 主流程 ----
    def answer(self, question: str, ctx: AccessContext, top_k: int | None = None) -> AnswerResult:
        s = self.settings
        trace_id = f"qa-{int(time.time()*1000)%10**12}"
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        res = self.retriever.retrieve(question, ctx, top_k=top_k)
        timings["retrieval_ms"] = res.timings_ms.get("retrieval_total_ms", 0.0)
        timings.update({f"stage_{k}": v for k, v in res.timings_ms.items() if k != "retrieval_total_ms"})

        base = AnswerResult(
            question=question,
            status="refuse",
            trace_id=trace_id,
            retrieval={
                "mode": res.mode,
                "cache_hit": res.cache_hit,
                "degraded": res.degraded,
                "expr": res.expr,
                "n_chunks": len(res.chunks),
                "top_chunk_ids": [c.chunk_id for c in res.chunks],
            },
        )

        hit = is_sensitive(question)
        if hit:
            base.status = "escalate"
            base.needs_human = True
            base.notes.append(f"sensitive_pattern:{hit}")
            base.timings_ms = timings
            base.confidence = {"composite": 0.0, "calibrated": 0.0, "rule": "sensitive_block"}
            return base

        top_score = max((c.score for c in res.chunks), default=0.0)
        if not res.chunks or top_score < self.min_retrieval_score:
            base.status = "refuse"
            base.needs_human = top_score >= self.min_retrieval_score
            base.notes.append("no_sufficient_evidence")
            base.confidence = {
                "retrieval": round(top_score, 4),
                "composite": 0.0,
                "calibrated": 0.0,
                "rule": "retrieval_gate",
            }
            base.timings_ms = timings
            return base

        evidence_text, evidence_index = format_evidence(res)
        prompt = USER_TEMPLATE.format(evidence=evidence_text, question=question)
        t1 = time.perf_counter()
        llm_res = self.llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        timings["generate_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        timings["e2e_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        base.llm = {
            "ok": llm_res.ok,
            "first_attempt_json_valid": llm_res.json_valid,
            "metrics": llm_res.metrics,
            "error": llm_res.error,
            "prompt_chars": len(prompt),
        }
        base.timings_ms = timings

        if not llm_res.ok:
            base.status = "degraded"
            base.notes.append(f"generation_failed:{llm_res.error[:60]}")
            base.key_points = [{"point": c.text[:120], "evidence_ids": [f"E{i}"]} for i, c in enumerate(res.chunks, start=1)]
            base.citations = [
                {**evidence_index[f"E{i}"], "evidence_id": f"E{i}"} for i in range(1, len(res.chunks) + 1)
            ]
            base.needs_human = True
            return base

        obj = self._extract_json(llm_res.text)
        if obj is None:
            base.status = "degraded"
            base.notes.append("unparseable_output")
            base.needs_human = True
            return base

        if obj.get("no_evidence") is True:
            # 反向兜底（Gate 0 ⑤ 唯一未闭环项的形态是「该答不答」，不是幻觉）：
            # 模型声明无依据时，系统此前只是拒答，既不复核也不承认自己可能判错。
            # 分两层，第一层不需要任何阈值：
            #   1) 声明无依据一律挂人工复核标记——知识服务里「查不出来」恰恰是最该被人看见的出口；
            #   2) 只有当检索层的证据强度也不同意「无依据」时，才把状态从 refuse 改成 escalate，
            #      因为此时 refuse 是在**断言一件检索层反对的事**（没有依据），不能这么报。
            base.needs_human = True
            base.notes.append("model_declared_no_evidence")
            strong = top_score >= self.strong_evidence_floor
            base.status = "escalate" if strong else "refuse"
            if strong:
                base.notes.append(
                    f"strong_evidence_but_no_basis_claimed:{top_score:.4f}>={self.strong_evidence_floor}"
                )
            base.confidence = {
                "retrieval": round(top_score, 4),
                "composite": 0.0,
                "calibrated": 0.0,
                "rule": "declared_no_evidence",
            }
            return base

        points, cites, dropped = self._enforce_citations(obj, evidence_index)
        conclusion = str(obj.get("conclusion", "") or "").strip()
        citation_consistency = 1.0 if not dropped else round(len(points) / max(len(points) + len(dropped), 1), 4)
        self_assessed = _clamp(float(obj.get("self_assessed_confidence", 0.0) or 0.0))
        composite = round(0.45 * _clamp(top_score) + 0.35 * citation_consistency + 0.20 * self_assessed, 4)
        calibrated = round(1.0 / (1.0 + math.exp(-(_logit(composite) / self.calibration_temperature))), 4)

        base.key_points = points
        base.citations = cites
        base.dropped_points = dropped

        # 出口契约：证据齐不等于答案成立。空结论必须走人工，不能以 answer 状态对外返回。
        # 这里刻意**不写 logit**：置信度校准的可判集按 logit 是否存在来筛，
        # 一条没有结论的输出参与校准只会把 ECE 测成「答案完整率」。
        if not points or not cites:
            base.status = "refuse"
            base.notes.append("citation_enforcement_rejected")
            base.confidence = {
                "retrieval": round(top_score, 4),
                "citation_consistency": round(citation_consistency, 4),
                "self_assessed": round(self_assessed, 4),
                "composite": 0.0,
                "calibrated": 0.0,
                "rule": "contract_no_citation",
            }
            return base
        if not conclusion and points:
            # 实测 45/58 的形态是「答案已经写在 key_points 里、conclusion 被留空」——
            # 那是**字段错位**，不是没有答案。这里只做逐字搬运，不向模型索要任何新内容。
            # 刻意不用「要求结论非空」的提示词修法：那样做完成率确实从 13/58 涨到 58/58，
            # 但 7 道语料内无答案的题从 0 道出结论变成 2 道，其中 q-059 编出
            # 「终验结论于2024年12月20日出的，一次通过率为93.7%」且 needs_human=False。
            # 把「该答不答」换成「不该答也答」，对知识服务是更坏的方向。
            derived = "；".join(
                str(p.get("point", "")).strip() for p in points if str(p.get("point", "")).strip()
            )
            if derived:
                conclusion = derived
                base.notes.append(f"conclusion_derived_from_points:{len(points)}")
        if not conclusion:
            base.status = "degraded"
            base.needs_human = True
            base.notes.append("contract_violation:empty_conclusion")
            base.confidence = {
                "retrieval": round(top_score, 4),
                "citation_consistency": round(citation_consistency, 4),
                "self_assessed": round(self_assessed, 4),
                "composite": 0.0,
                "calibrated": 0.0,
                "rule": "contract_empty_conclusion",
            }
            return base

        base.conclusion = conclusion
        base.confidence = {
            "retrieval": round(top_score, 4),
            "citation_consistency": round(citation_consistency, 4),
            "self_assessed": round(self_assessed, 4),
            "composite": composite,
            "logit": round(_logit(composite), 4),
            "calibrated": calibrated,
            "temperature": self.calibration_temperature,
        }
        # 引用落在密级 3 的证据上 → 答案可给（用户有权限）但强制挂人工复核标记。
        # 注意触发点是「实际被引用的证据」而不是「候选证据集」：后者会让有权限的用户几乎问不出答案。
        secret_cited = [c for c in cites if int(c.get("security_level") or 0) >= ESCORT_SECURITY_LEVEL]
        if secret_cited:
            base.needs_human = True
            base.notes.append(f"secret_level_cited:{len(secret_cited)} 条需人工复核")
        if calibrated < s.confidence_low:
            base.status = "escalate"
            base.needs_human = True
            base.notes.append("low_confidence_route_to_human")
        elif calibrated < s.confidence_high:
            base.status = "tentative"
            base.notes.append("初步判断（低置信度显式降级）")
        else:
            base.status = "answer"
        return base
