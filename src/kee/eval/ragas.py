"""RAGAS 两指标的本地复现：Faithfulness 与 Answer Relevance。

**为什么是复现而不是引库**：ragas 官方包的依赖闭包直接拖 `scikit-network`→`numba`
（numba 有 numpy 上限，本项目硬钉 `numpy==2.5.3`），外加 `datasets>=4`/pyarrow/pandas 与
`langchain`+`langchain-community`+`langchain_openai`+`openai`+`instructor` 共 21+ 个包。
本项目的 torch 是 D:\\py312-torch 经 `.pth` 挂进 `.venv` 的，解析器一旦为 numba 降 numpy
就是毁检索栈；而且 RAGAS 默认 LLM/Embeddings 走 OpenAI 客户端，数据不出域仍要自写本地
wrapper——「装了库」并不等于「零改造」。所以这里照它公开的两步算法实现，零新依赖。

Faithfulness（RAGAS 原定义，两步）：
  1) 把 answer 拆成原子陈述（statements）；
  2) 每条陈述**只依据 contexts** 判「能否被推出」，1=能、0=不能；
  分数 = 能推出的条数 / 总条数。总条数为 0 ⇒ **不适用**，既不报 1.0 也不报 0。
  （分母为 0 报「不适用」是本仓库既有的度量纪律，见 `hallucination_rate.json` 那条同型口径。）

Answer Relevance（RAGAS 原定义，两步）：
  1) 从 answer 反向生成 K 个问题；answer 不含可回答的信息时生成 noncommittal；
  2) 分数 = K 个生成问题与原 question 的余弦相似度均值；noncommittal ⇒ 0。

Context Precision（RAGAS `LLMContextPrecisionWithReference` 的口径）：
  逐块判「该检索块是否含有支撑参考答案所需的信息」，再按**名次**加权：
  CP = Σ_k (Precision@k × v_k) / 相关块总数。分母是相关块数，**5 块全不相关时该式未定义**，
  本装置报「不适用」并单独数这类行——别和官方返回 0.0 的写法混引。

Context Recall（RAGAS `LLMContextRecall` 的口径）：
  把**参考答案**拆成声明，逐条判「能否归因到检索块」，取覆盖比例。
  这条天生需要参考答案（与人工工单「不给参考答案」那条纪律不冲突：那条管的是「结论对不对」，
  这条测的是「检索面够不够」）。

与官方的**口径偏离**必须随读数一起说：官方 prompt 是英文、contexts 是检索到的全部候选；
这里 prompt 用中文（语料与答案都是中文，翻成英文喂 7B 只会引入第三个变量），
contexts 由调用方决定报两档——`cited`（实际被引用的块，与既有机械装置同尺）与
`top5`（RAGAS 原义）。**Context Precision / Recall 只用 `top5`**：它们测检索面，
与回答引用了哪几块是两件事。另外 **两条 NLI 类判定的第二步默认一次批量判完**（官方逐条一次一调用），
理由是 67 题 × 两口径下逐条要 ~740 次本地 7B 调用；批量可能漏答，所以漏答记「判不出」不进分母，
并由「证据错配必须掉分」那道护栏兜底。要原口径传 `batch_verify=False`。
"""

from __future__ import annotations

import json
from typing import Any

K_DEFAULT = 3

DECOMPOSE_PROMPT = (
    "任务：把下面这个回答拆成一条条**独立的原子事实陈述**。\n"
    "要求：\n"
    "1. 每条只说一件事，不超过 30 字；\n"
    "2. 只许从回答里提取，不许补充回答里没有的内容；\n"
    "3. 回答里的数字、单位、主体名称必须原样保留；\n"
    "4. 回答为空、或只表示「不知道 / 无依据 / 需转人工」⇒ statements 给空列表。\n"
    "只返回 JSON：{{\"statements\": [\"陈述1\", \"陈述2\"]}}\n\n"
    "问题：{question}\n回答：{answer}\n"
)

VERIFY_PROMPT = (
    "任务：判断下面这条陈述**能否只依据给定材料推出**。\n"
    "规则：\n"
    "1. 只看材料，不许使用你自己的知识，也不许用材料之外的常识补链；\n"
    "2. 材料没有任何依据支持它 ⇒ verdict = 0（「没提到」就是 0，不是判不了）；\n"
    "3. 陈述里的数字、单位、主体、时间、比例必须与材料一致，反着说或张冠李戴 ⇒ verdict = 0；\n"
    "4. 允许材料经过一步直接推理得到（如「1402/1468 通过」支持「通过率约 95.5%」）。\n"
    "只返回 JSON：{{\"verdict\": 0 或 1, \"basis\": \"材料里支撑或反驳的原话片段，60 字内\"}}\n\n"
    "材料：\n{contexts}\n\n陈述：{statement}\n"
)

REVERSE_QUESTION_PROMPT = (
    "任务：下面这个回答原本是在回应什么问题？请**只依据回答本身**反推出 {k} 个问题。\n"
    "要求：\n"
    "1. {k} 个问题各自独立，角度不同（可以问数值、问口径、问主体）；\n"
    "2. 每个问题必须能被这个回答当作正面答案；\n"
    "3. 若回答为空，或只表示「不知道 / 无依据 / 需转人工 / 拒答」，"
    "不要编问题，返回 {{\"noncommittal\": true, \"questions\": []}}。\n"
    "只返回 JSON：{{\"noncommittal\": false, \"questions\": [\"问题1\", \"问题2\", \"问题3\"]}}\n\n"
    "回答：{answer}\n"
)

VERIFY_BATCH_PROMPT = (
    "任务：逐条判断下面这些陈述**能否只依据给定材料推出**，每条只回答 1 或 0。\n"
    "规则：\n"
    "1. 只看材料，不许使用你自己的知识，也不许用材料之外的常识补链；\n"
    "2. 材料没有任何依据支持它 ⇒ verdict = 0（「没提到」就是 0，不是判不了）；\n"
    "3. 陈述里的数字、单位、主体、时间、比例必须与材料一致，反着说或张冠李戴 ⇒ verdict = 0；\n"
    "4. 允许材料经过一步直接推理得到（如「1402/1468 通过」支持「通过率约 95.5%」）。\n"
    "只返回 JSON：{{\"verdicts\": [{{\"i\": 1, \"verdict\": 1}}, ...]}}，"
    "i 从 1 开始、**必须覆盖全部 {n} 条**，不要输出别的内容。\n\n"
    "材料：\n{contexts}\n\n待判陈述：\n{statements}\n"
)


def build_verify_batch_prompt(statements: list[str], contexts: list[str]) -> str:
    body = "\n---\n".join(f"[材料{i + 1}] {c}" for i, c in enumerate(contexts)) or "（无材料）"
    listed = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    return VERIFY_BATCH_PROMPT.format(contexts=body, statements=listed, n=len(statements))


CONTEXT_PRECISION_PROMPT = (
    "任务：下面是一次检索拿到的 {n} 个文档块（按检索名次排列，1 是最靠前）。\n"
    "逐个判断：**这个块里是否含有回答该题、支撑参考答案所需要的信息**。\n"
    "规则：\n"
    "1. 判的是「这块有没有用」，不是「回答写得好不好」；\n"
    "2. 只沾边但没有正面回答所需信息 ⇒ 0；主题相同但讲的是别的项目/别的口径 ⇒ 0；\n"
    "3. 参考答案的任一要素（数值、时限、主体、规则原文）能在这块里找到 ⇒ 1；\n"
    "4. 不许使用材料之外的知识。\n"
    "只返回 JSON：{{\"verdicts\": [{{\"i\": 1, \"verdict\": 1}}, ...]}}，"
    "i 从 1 到 {n}，**必须逐块给全**。\n\n"
    "问题：{question}\n参考答案：{reference}\n\n检索块：\n{contexts}\n"
)

CONTEXT_RECALL_PROMPT = (
    "任务：下面这些声明是从**参考答案**里拆出来的。逐条判断："
    "**该声明能否归因到给定的检索块**（即检索块里确实写到了它）。\n"
    "规则：\n"
    "1. 只许看检索块，不许用你自己的知识补链；\n"
    "2. 检索块里找不到 ⇒ verdict = 0（「没检到」正是这个指标要测的东西）；\n"
    "3. 数字、单位、主体、时限必须对得上，写在别的项目头上的同名结论 ⇒ 0；\n"
    "4. 允许一步直接推理得到。\n"
    "只返回 JSON：{{\"verdicts\": [{{\"i\": 1, \"verdict\": 1}}, ...]}}，"
    "i 从 1 到 {n}，**必须逐条给全**。\n\n"
    "检索块：\n{contexts}\n\n待归因声明：\n{claims}\n"
)


def build_context_precision_prompt(question: str, reference: str, contexts: list[str]) -> str:
    body = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return CONTEXT_PRECISION_PROMPT.format(n=len(contexts), question=question or "（无题面）",
                                           reference=reference, contexts=body)


def build_context_recall_prompt(claims: list[str], contexts: list[str]) -> str:
    body = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    return CONTEXT_RECALL_PROMPT.format(n=len(claims), contexts=body, claims=listed)


def context_precision_from_verdicts(verdicts: list[int | None]) -> dict[str, Any]:
    """RAGAS 的位置加权口径：CP = Σ_k (Precision@k × v_k) / 相关块总数。

    分母是**判出来的相关块数**，不是块总数——所以「5 块全不相关」在这个公式里是未定义，
    不是 0。本装置按仓库既有纪律报「不适用」并把这类行数出来；要按官方返回 0.0 的写法，
    必须显式说明是哪种口径，两个数不能混着引。

    ⚠ **判不出（None）的那一块按「不给信用」处理，而不是从名次里剔掉**。
    这与 Faithfulness 的分母纪律（None 直接不进分母）看着不一致，但方向是对的：
    这里如果把漏答的块从序列里删掉，后面的块会自动升位、于是「模型没答的那一块」反而
    把 Precision 抬上去——那是用一次故障换一个好看的数。宁可记低，`n_unjudged` 单独报出来。
    """
    flags = [1 if v == 1 else 0 for v in verdicts]
    n_unjudged = sum(1 for v in verdicts if v not in (0, 1))
    n_relevant = sum(flags)
    if n_relevant == 0:
        return {"n_contexts": len(verdicts), "n_relevant": 0, "n_unjudged": n_unjudged,
                "value": None, "reading": f"不适用（{len(verdicts)} 块里判不出任何相关块）",
                "precision_at_k": []}
    hits, per_k = 0, []
    for k, v in enumerate(flags, start=1):
        if v == 1:
            hits += 1
            per_k.append(hits / k)          # 加权求和用未取整的值，只在最后一步取整
    value = round(sum(per_k) / n_relevant, 4)
    return {"n_contexts": len(verdicts), "n_relevant": n_relevant, "n_unjudged": n_unjudged,
            "precision_at_k": [round(x, 4) for x in per_k],
            "value": value, "reading": f"Σ P@k·v_k = {round(sum(per_k), 4)} / {n_relevant} 相关块 = {value:.4f}"}


def context_recall_from_verdicts(verdicts: list[int | None]) -> dict[str, Any]:
    usable = [v for v in verdicts if v in (0, 1)]
    if not usable:
        return {"n_claims": len(verdicts), "n_gradable": 0, "n_covered": 0, "value": None,
                "reading": f"不适用（{len(verdicts)} 条声明全部判不出）"}
    covered = sum(1 for v in usable if v == 1)
    return {"n_claims": len(verdicts), "n_gradable": len(usable),
            "n_unparsable": len(verdicts) - len(usable), "n_covered": covered,
            "value": round(covered / len(usable), 4),
            "reading": f"{covered}/{len(usable)} = {covered / len(usable):.2%}"}



def parse_verdict_batch(envelope: dict[str, Any] | None, n: int) -> list[int | None]:
    """长度恒为 n；模型漏答某条 ⇒ 那条记 None（进不了分母），**不得补 0**。"""
    out: list[int | None] = [None] * n
    if not isinstance(envelope, dict):
        return out
    raw = envelope.get("verdicts")
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("i", item.get("index"))
        try:
            pos = int(idx) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= pos < n:
            out[pos] = parse_verdict({"verdict": item.get("verdict")})
    return out


def _json_of(res: Any) -> dict[str, Any] | None:
    """取模型给的 JSON 信封；拿不到就返回 None，**不猜**、不把坏输出静默当 0 分。"""
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


def build_decompose_prompt(question: str, answer: str) -> str:
    return DECOMPOSE_PROMPT.format(question=question or "（无题面）", answer=answer or "（空回答）")


def build_verify_prompt(statement: str, contexts: list[str]) -> str:
    body = "\n---\n".join(f"[材料{i + 1}] {c}" for i, c in enumerate(contexts)) or "（无材料）"
    return VERIFY_PROMPT.format(contexts=body, statement=statement)


def build_reverse_question_prompt(answer: str, k: int = K_DEFAULT) -> str:
    return REVERSE_QUESTION_PROMPT.format(k=k, answer=answer or "（空回答）")


def parse_statements(envelope: dict[str, Any] | None) -> list[str] | None:
    """None = 这一行不可判（模型没给出可用信封），与「给出空列表 = 回答里确实没有陈述」分开。"""
    if envelope is None:
        return None
    raw = envelope.get("statements")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("statement") or item.get("text")
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def parse_verdict(envelope: dict[str, Any] | None) -> int | None:
    if envelope is None:
        return None
    v = envelope.get("verdict")
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v if v in (0, 1) else None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "yes", "true", "能", "是", "supported", "1.0"):
            return 1
        if s in ("0", "no", "false", "否", "不能", "unsupported", "0.0"):
            return 0
    return None


def parse_verdict_basis(envelope: dict[str, Any] | None) -> str:
    if not isinstance(envelope, dict):
        return ""
    basis = envelope.get("basis")
    return basis.strip()[:200] if isinstance(basis, str) else ""


def parse_questions(envelope: dict[str, Any] | None, k: int = K_DEFAULT) -> tuple[list[str], bool]:
    """返回 (问题列表, noncommittal)。

    **只有模型明说 noncommittal=true 才算 noncommittal**。坏信封、字段缺失、生成了但全是空白
    一律记「不可判」（([], False)）——把解析失败算成 0 分，等于用一次 JSON 崩故障换一个「相关性很低」
    的假读数，而这两个东西在报告里必须长得不一样。
    """
    if envelope is None:
        return [], False
    if envelope.get("noncommittal") is True:
        return [], True
    raw = envelope.get("questions")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return [], False
    out = [q.strip() for q in raw if isinstance(q, str) and q.strip()][:k]
    return out, False


def faithfulness_from_verdicts(verdicts: list[int | None]) -> dict[str, Any]:
    """分母只计**判得出**的陈述；一条都判不出 ⇒ value=None（不适用），不报 0 也不报 1。"""
    usable = [v for v in verdicts if v in (0, 1)]
    n_unparsed = len(verdicts) - len(usable)
    if not usable:
        return {
            "n_statements": len(verdicts),
            "n_gradable": 0,
            "n_unparsable": n_unparsed,
            "n_supported": 0,
            "value": None,
            "reading": "不适用（无可判陈述）" if not verdicts else f"不适用（{len(verdicts)} 条陈述全部判不出）",
        }
    supported = sum(1 for v in usable if v == 1)
    return {
        "n_statements": len(verdicts),
        "n_gradable": len(usable),
        "n_unparsable": n_unparsed,
        "n_supported": supported,
        "value": round(supported / len(usable), 4),
        "reading": f"{supported}/{len(usable)} = {supported / len(usable):.2%}",
    }


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def answer_relevance_from_vectors(
    question_vec: list[float], gen_vecs: list[list[float]], noncommittal: bool
) -> dict[str, Any]:
    """RAGAS 口径：noncommittal ⇒ 0.0；否则取余弦均值。

    这一条是官方行为，不是我们的选择，所以要留痕：**正确拒答会被它记成 0 分**。
    调用方必须按种群分开报，不得把拒答行与作答行混成一个均值对外引用。
    """
    if noncommittal:
        return {"n_questions": 0, "value": 0.0, "reading": "noncommittal ⇒ 0.0（RAGAS 官方口径）"}
    if not gen_vecs:
        return {"n_questions": 0, "value": None, "reading": "不适用（反向生成没拿到可用问题）"}
    sims = [round(cosine(question_vec, g), 4) for g in gen_vecs]
    mean = round(sum(sims) / len(sims), 4)
    return {"n_questions": len(sims), "sims": sims, "value": mean, "reading": f"mean cos = {mean:.4f}（n={len(sims)}）"}


class RagasRunner:
    """把两步算法跑在注入的 LLM 客户端上；嵌入留给调用方分时做（8GB 单卡不能两模型齐驻留）。

    `batch_verify=True`（默认）把「每条陈述一次 NLI 调用」合成一次批量判定：
    逐条版在 67 题 × 两口径下要 ~740 次 7B 调用（实测约 2 小时），批量版 ~5 次/题。
    这是**对官方实现的一处偏离**，代价是模型一次要看完全部陈述，可能漏答——
    所以漏答一律记 None 进不了分母，而「证据换成别题的分数必须掉」那道护栏专门兜这种失明。
    要逐条原口径就传 `batch_verify=False`。
    """

    def __init__(self, llm: Any, k: int = K_DEFAULT, max_tokens: int = 320,
                 batch_verify: bool = True) -> None:
        self.llm = llm
        self.k = k
        self.max_tokens = max_tokens
        self.batch_verify = batch_verify
        self.calls: list[dict[str, Any]] = []

    def _ask(self, tag: str, prompt: str, max_tokens: int | None = None) -> dict[str, Any] | None:
        res = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True, temperature=0.0,
                            max_tokens=max_tokens or self.max_tokens)
        m = dict(getattr(res, "metrics", {}) or {})
        self.calls.append(
            {
                "tag": tag,
                "ok": bool(getattr(res, "ok", False)),
                "json_valid": bool(getattr(res, "json_valid", False)),
                "done_reason": m.get("done_reason", ""),
                "total_ms": m.get("total_ms", 0.0),
                "prompt_tokens": m.get("prompt_tokens", 0),
                "completion_tokens": m.get("completion_tokens", 0),
            }
        )
        return _json_of(res)

    def decompose(self, question: str, answer: str) -> list[str] | None:
        return parse_statements(self._ask("decompose", build_decompose_prompt(question, answer)))

    def verify(self, statement: str, contexts: list[str]) -> dict[str, Any]:
        env = self._ask("verify", build_verify_prompt(statement, contexts))
        return {"verdict": parse_verdict(env), "basis": parse_verdict_basis(env)}

    def reverse_questions(self, answer: str) -> tuple[list[str], bool]:
        return parse_questions(self._ask("reverse", build_reverse_question_prompt(answer, self.k)), self.k)

    def context_precision(self, question: str, reference: str, contexts: list[str]) -> dict[str, Any]:
        """判的是**检索面**（复跑 Top-5），不是回答引用了哪几块——那是 Faithfulness 的分工。"""
        if not contexts:
            return {"verdicts": [], "score": context_precision_from_verdicts([]), "skipped": "no_contexts"}
        env = self._ask("context_precision", build_context_precision_prompt(question, reference, contexts),
                        max_tokens=min(1200, 60 + 24 * len(contexts)))
        verdicts = parse_verdict_batch(env, len(contexts))
        return {"verdicts": [{"rank": i + 1, "verdict": v} for i, v in enumerate(verdicts)],
                "score": context_precision_from_verdicts(verdicts), "skipped": ""}

    def context_recall(self, question: str, reference: str, contexts: list[str]) -> dict[str, Any]:
        """声明拆解复用 `decompose`（RAGAS 的 StatementGenerator 也是喂 question + reference）。"""
        if not contexts:
            return {"claims": [], "verdicts": [], "score": context_recall_from_verdicts([]), "skipped": "no_contexts"}
        claims = self.decompose(question, reference)
        if not claims:
            return {"claims": claims, "verdicts": [],
                    "score": context_recall_from_verdicts([]), "skipped": "no_claims"}
        env = self._ask("context_recall", build_context_recall_prompt(claims, contexts),
                        max_tokens=min(1600, 120 + 40 * len(claims)))
        verdicts = parse_verdict_batch(env, len(claims))
        return {"claims": claims,
                "verdicts": [{"claim": c, "verdict": v} for c, v in zip(claims, verdicts)],
                "score": context_recall_from_verdicts(verdicts), "skipped": ""}

    def verify_set(self, statements: list[str], contexts: list[str]) -> dict[str, Any]:
        """对同一批陈述换一套 contexts 再判一次——两口径必须用**同一批陈述**才可比，
        否则差值里混着「这次拆句拆得不一样」，量到的就不是证据面宽度。

        批量模式下一次判完；漏答的那条记 None（进不了分母），**不补 0**。
        """
        if not statements:
            return {"verdicts": [], "score": faithfulness_from_verdicts([])}
        if self.batch_verify:
            env = self._ask("verify_batch", build_verify_batch_prompt(statements, contexts),
                            max_tokens=min(1600, 120 + 40 * len(statements)))
            verdicts = parse_verdict_batch(env, len(statements))
            return {"verdicts": [{"statement": s, "verdict": v, "basis": ""}
                                 for s, v in zip(statements, verdicts)],
                    "score": faithfulness_from_verdicts(verdicts)}
        checked = [self.verify(s, contexts) for s in statements]
        return {
            "verdicts": [{"statement": s, "verdict": v["verdict"], "basis": v["basis"]}
                         for s, v in zip(statements, checked)],
            "score": faithfulness_from_verdicts([c["verdict"] for c in checked]),
        }

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        statements = self.decompose(question, answer)
        if statements is None:
            return {"statements": None, "verdicts": [], "score": faithfulness_from_verdicts([]), "decompose_failed": True}
        checked = self.verify_set(statements, contexts)
        return {"statements": statements, "verdicts": checked["verdicts"],
                "score": checked["score"], "decompose_failed": False}

    def cost_summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "failed": sum(1 for c in self.calls if not c["ok"]),
            "bad_json": sum(1 for c in self.calls if c["ok"] and not c["json_valid"]),
            "truncated": sum(1 for c in self.calls if c["done_reason"] == "length"),
            # max_prompt_tokens 拦的是**前端静默截断**：Ollama 默认按显存算 num_ctx（本机 8GB→4096），
            # 提示超长时它从头部丢掉 token 而不报错，症状是「模型在看一段没有材料的提示」。
            "max_prompt_tokens": max((int(c["prompt_tokens"] or 0) for c in self.calls), default=0),
            "llm_ms_total": round(sum(float(c["total_ms"] or 0.0) for c in self.calls), 1),
            "prompt_tokens": sum(int(c["prompt_tokens"] or 0) for c in self.calls),
            "completion_tokens": sum(int(c["completion_tokens"] or 0) for c in self.calls),
        }
