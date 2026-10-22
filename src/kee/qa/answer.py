"""答案生成的规则层：提示词契约、引用强制、复合置信度的算法与数据结构。

本模块只留**被图调用的纯函数与常量**（format_evidence / _clamp / _logit / is_sensitive /
SYSTEM_PROMPT / USER_TEMPLATE / AnswerResult）和引擎的阈值配置；
执行顺序与出口路由在 `kee/qa/graph.py`（M3-05 起的 LangGraph 状态机）。
拆开放是因为图要 import 本模块，反过来 import 会成环。

三条硬规则（对应方案 6.4 / Must 4·5·6）：
  1. 引用强制：要点必须挂证据编号，证据编号必须在本次检索集合内；无任何有效引用 → 不出结论。
  2. 复合分数：检索分 + 引用一致性 + 模型自评 → 分档。
     **它是规则分，不是「答案正确的概率」**：M3-04a 实测三个成分里
     `citation_consistency` 恒为 1.0、`self_assessed` 只有两个取值、`retrieval` 已饱和，
     composite 的 93% 是常数偏移，对唯一有方差的标签 AUC=0.346（低于随机）。
     温度缩放因此**不再对外称校准**——零方差标签上的拟合只会把分数压向基率，
     详见 `kee.eval.metrics.fit_temperature` 的拒绝分支与 M3 报告第七节。
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


# 复合分权重。enforce（要不要搬运）与 confidence（怎么分档）共用同一份，
# 否则两处会各自演化出两个「置信度」——那正是 M3-04 一直在修的那类错。
W_RETRIEVAL, W_CITATION, W_SELF_ASSESSED = 0.45, 0.35, 0.20


def composite_score(top_score: float, citation_consistency: float, self_assessed: float) -> float:
    """规则分。注意它**不是**答案正确的概率（见本模块 docstring 第 2 条与 M3 报告第七节）。"""
    return round(
        W_RETRIEVAL * _clamp(top_score) + W_CITATION * citation_consistency + W_SELF_ASSESSED * _clamp(self_assessed),
        4,
    )


def load_calibration_temperature(settings_obj: Any | None = None, default: float = 1.0) -> float:
    """读服务路径用的温度参数。**API 与所有离线探针必须走这一个函数**。

    此前 API 从 Gate 0 的 `calibration.json` 读 T=6.655、探针用默认 1.0，于是「answer 42 / tentative 11」
    这类报出去的数字描述的是探针配置，而对外服务实际跑的是另一套
    （那个温度下 `answer` 结构上永不可达）。两套配置混着报，比报一个难看的数字严重得多。

    读的是 **M3 自己的 `confidence_calibration.json`**，不读 Gate 0 那份：
    `calibration.json` 是 Gate 0 ⑤ 的落盘证据，M3 的裁决不能去改写它（历史结论只加横幅不抹除）。
    而 M3-04a 已证明那次拟合用的是零方差标签，得到的 6.655 不是校准值——
    所以这里**刻意不回落**到它的值：文件缺失或带 `refused` 一律按 1.0（不缩放）。
    等真出现有方差的标签，再往这个文件里写可辨识的 T。
    """
    import json
    from pathlib import Path

    s = settings_obj or get_settings()
    path = Path(s.resolved_artifact_dir()) / "confidence_calibration.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(data, dict) or data.get("refused"):
        return default
    try:
        v = float(data.get("temperature", default))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


@dataclass
class AnswerResult:
    question: str
    status: str  # answer / tentative / refuse / escalate / degraded
    conclusion: str = ""
    key_points: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    dropped_points: list[str] = field(default_factory=list)
    needs_human: bool = False
    # 拒答时分两种情形对外可判：`no_evidence`（库里就没有相关资料，真无解）与
    # `all_evidence_unreviewed`（资料检到了、但所属版本还没复核，被答案层的复核闸摘光——流程没走完）。
    # 没有这两个字段，两者共用 `refuse + no_sufficient_evidence` 出口，业务读到的是「没找到相关资料」。
    refusal_reason: str = ""
    unreviewed_evidence_count: int = 0
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


def apply_parent_context(engine: "AnswerEngine", ev_text: str, ev_index: dict[str, dict],
                         chunks: list[Any]) -> tuple[str, list[str]]:
    """把被引子块所属**父块正文**接进答案证据面（需求 US-1.2「父块补上下文」的字面实现）。

    三条约束都来自实测，不是审美：
    - **同一父块只贴一次**：上一版逐条贴，五块常同属一章 ⇒ 在场文字被重贴五遍，
      提示均值 +1219.6 token、escalate 1→3、可完成率落在噪声地板内。
    - **贴之前扣掉子块已含的段落**：实测父块相对其子块的净增中位只有 28 字，
      重复内容对模型没有信息量，只有挤占窗口的代价。
    - **有注入预算，超了就整块不贴并记数**：Ollama 本机默认 `num_ctx=4096`，
      超长提示**从头部静默截掉**——那时量的不是"补上下文"而是截断。宁可少贴一块。
    """
    from kee.models.tokens import count_tokens

    if engine.parent_provider is None or not engine.parent_context_in_answer or not chunks:
        return ev_text, []
    order: list[str] = []
    by_parent: dict[str, list[str]] = {}
    for eid, meta in ev_index.items():
        pid = meta.get("parent_id") or ""
        if not pid:
            continue
        if pid not in by_parent:
            by_parent[pid] = []
            order.append(pid)
        child = next((c for c in chunks if c.chunk_id == meta.get("chunk_id")), None)
        if child is not None:
            by_parent[pid].append(child.text or "")
    bodies = engine.parent_provider(order) if order else {}
    notes: list[str] = []
    blocks: list[str] = []
    used = 0
    skipped = 0
    for pid in order:
        raw = bodies.get(pid)
        if not raw:
            skipped += 1
            continue
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        cited = [re.sub(r"\s+", "", t) for t in by_parent.get(pid, []) if t.strip()]
        rest = [ln for ln in lines[1:] if re.sub(r"\s+", "", ln) and
                not any(re.sub(r"\s+", "", ln) in c for c in cited)]
        if not rest:
            continue
        block = f"\n【父块补上下文 {pid}】" + " ".join(rest)
        cost = count_tokens(block)
        if used + cost > engine.parent_context_max_tokens:
            skipped += 1
            continue
        used += cost
        blocks.append(block)
    if skipped:
        notes.append(f"parent_context_skipped:{skipped}")
    if blocks:
        notes.append(f"parent_context_tokens:{used}")
        return ev_text + "".join(blocks), notes
    return ev_text, notes


class AnswerEngine:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OllamaClient | None = None,
        min_retrieval_score: float = 0.20,
        strong_evidence_floor: float = 0.90,
        temperature: float = 0.0,
        max_tokens: int = 700,
        unreviewed_provider=None,
        parent_provider=None,
        parent_context_max_tokens: int | None = None,
        parent_context_in_answer: bool | None = None,
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
        # 复核闸（处置 C）：可调用者返回「未复核版本 id 集合」。None = 不设闸（离线复跑与老测试保持原行为）。
        self.unreviewed_provider = unreviewed_provider
        # 父块回捞（需求 US-1.2「父块补上下文」）：provider 收一组 parent_id，返回 {id: 正文}。
        # None = 不接（离线复跑与既有测试保持原行为）。阈值一律从 Settings 取，不在这里写死：
        # 写死一份默认值就会出现「探针跑的配置和对外服务跑的配置不一样」那笔旧账（见 load_calibration_temperature）。
        self.parent_provider = parent_provider
        self.parent_context_in_answer = (
            self.settings.parent_context_in_answer if parent_context_in_answer is None else parent_context_in_answer
        )
        self.parent_context_max_tokens = (
            self.settings.parent_context_max_tokens if parent_context_max_tokens is None else parent_context_max_tokens
        )

    # ---- 复核闸 ----
    def gate_unreviewed(self, res) -> int:
        """把**未经复核版本**的证据从答案的证据面里摘掉，返回摘掉的条数。

        为什么摘在答案层而不是检索层（用户 2026-10-12 选定 C）：
        真正有害的是「被当成答案对外作答」，不是「被检索到」——知识库管理页与复核人
        本来就该看得见未复核内容，否则没人能去复核它。
        摘掉之后若一条不剩，下游走的是既有的「无充分依据」拒答路径，不新造出口。
        """
        if self.unreviewed_provider is None:
            return 0
        drop = self.unreviewed_provider() or set()
        if not drop:
            return 0
        kept = [c for c in res.chunks if c.chunk_id.split("#", 1)[0] not in drop]
        n = len(res.chunks) - len(kept)
        if n:
            res.chunks = kept
        return n

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
