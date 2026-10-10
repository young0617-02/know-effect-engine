"""答案生成的规则层：提示词契约、引用强制、复合置信度的算法与数据结构。

本模块只留**被图调用的纯函数与常量**（format_evidence / _clamp / _logit / is_sensitive /
SYSTEM_PROMPT / USER_TEMPLATE / AnswerResult）和引擎的阈值配置；
执行顺序与出口路由在 `kee/qa/graph.py`（M3-05 起的 LangGraph 状态机）。
拆开放是因为图要 import 本模块，反过来 import 会成环。

三条硬规则（对应方案 6.4 / Must 4·5·6）：
  1. 引用强制：要点必须挂证据编号，证据编号必须在本次检索集合内；无任何有效引用 → 不出结论。
  2. 复合置信度：检索分 + 引用一致性 + 模型自评 → 温度缩放校准 → 分档。
  3. 确定性降级：无依据=拒答；低置信=「初步判断」并转人工；敏感话题=直接转人工。
"""

from __future__ import annotations

import json
import math
import re
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
    @property
    def graph(self):
        """LangGraph 状态机（懒建：导入 kee.qa.graph 需要本模块，构造期建会成环）。"""
        if getattr(self, "_graph", None) is None:
            from kee.qa.graph import build_qa_graph

            self._graph = build_qa_graph(self)
        return self._graph

    def answer(
        self,
        question: str,
        ctx: AccessContext,
        top_k: int | None = None,
        history: list[dict] | None = None,
    ) -> AnswerResult:
        """在线问答入口。链路拓扑见 `kee/qa/graph.py`。

        `history` 是多轮追问的上文（来自 session_turn）。不传就是单轮——
        刻意不做「自动去查会话」，因为那会让本方法隐式依赖权威源，
        单测与离线复跑都要连库才能跑。
        """
        from kee.qa.graph import run_qa

        return run_qa(self, question, ctx, top_k=top_k, history=history)
