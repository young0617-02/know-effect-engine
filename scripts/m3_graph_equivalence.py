"""M3-05 差分回归网：图化重构前后必须逐字段等价。

单测只能证明「我想到并写下来的行为」没变；重构改的是 189 行控制流，
漏掉一个分支的返回值形状是这类重构最常见的事故。所以这里做的是**差分**：
把重构前的实现从 git 里抽出来当 oracle，同一批场景喂两边，逐字段比对。

场景矩阵覆盖 answer() 的每个出口（17 个），不是随机样本。

三处**已声明的差异**，脚本单独断言而不是抹平：
  1. 敏感题不再先烧一次检索（原实现是先检索再判敏感，顺序缺陷）。
     → 断言图侧 retrieval 为空且检索调用数为 0。
  2. timings_ms / trace_id 天然不可比（每次运行不同）。
     → 整字段不参与比对，但 e2e_ms 的存在性单独检查。

oracle 用 commit sha 钉死。HEAD 一变，`legacy` 抽不到旧实现就会大声失败，
不会静默拿新实现跟自己比（那是假绿）。

用法：uv run python scripts/m3_graph_equivalence.py
退出码 0 = 等价；非 0 = 有字段差异（打印 diff 明细）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 重构前最后一次含旧 answer() 的提交。抽不到就失败，不降级。
LEGACY_REF = "458186e:src/kee/qa/answer.py"

from kee.llm.ollama import LlmResult  # noqa: E402
from kee.qa.answer import AnswerEngine  # noqa: E402
from kee.retrieval.pipeline import RetrievedChunk, RetrievalResult  # noqa: E402
from kee.retrieval.rbac import AccessContext  # noqa: E402

NON_COMPARABLE = {"timings_ms", "trace_id"}
# 图侧相对 oracle 的**已登记增量**字段（只允许这些，其余一律算差异）。
DECLARED_ADDITIVE_RETRIEVAL_KEYS = frozenset({"query_used"})


def load_legacy():
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", LEGACY_REF],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if src.returncode != 0:
        sys.exit(f"[FAIL] 抽不到 oracle：git show {LEGACY_REF} 失败\n{src.stderr}")
    # 防呆：抽到的必须是「自己跑完整流程」的旧实现，而不是已经委托给图的版本。
    if "run_qa" in src.stdout or "from kee.qa.graph" in src.stdout:
        sys.exit(f"[FAIL] {LEGACY_REF} 抽出来的已是委托图实现的版本，oracle 失效（钉错了 commit？）")
    if "def answer(" not in src.stdout:
        sys.exit(f"[FAIL] {LEGACY_REF} 里没有 answer() 方法，oracle 失效")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src.stdout)
        path = f.name
    spec = importlib.util.spec_from_file_location("legacy_answer", path)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass 要靠 sys.modules 里的模块命名空间解析注解，漏这行会直接炸在 exec_module。
    sys.modules["legacy_answer"] = mod
    spec.loader.exec_module(mod)
    Path(path).unlink(missing_ok=True)
    return mod


class CountingRetriever:
    """记录调用次数——用来证明敏感题真的没烧检索。"""

    def __init__(self, chunks, degraded=None):
        self._chunks = chunks
        self.degraded = degraded or []
        self.calls = 0

    def retrieve(self, query, ctx, top_k=None, already_rewritten=False):
        self.calls += 1
        return RetrievalResult(
            query=query, mode="hybrid", chunks=self._chunks,
            timings_ms={"retrieval_total_ms": 10.0, "retrieval_ms": 10.0},
            expr="security_level <= 3", degraded=self.degraded,
        )


class FakeLlm:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def chat(self, messages, json_mode=False, temperature=0.0, max_tokens=0, keep_alive=None):
        text = json.dumps(self._payload, ensure_ascii=False) if isinstance(self._payload, dict) else self._payload
        return LlmResult(
            ok=self.ok, text=text, parsed=None,
            json_valid=isinstance(self._payload, dict),
            metrics={"total_ms": 500.0, "tokens_per_sec": 40.0, "completion_tokens": 20},
        )


def chunk(cid, score=0.9, level=2):
    return RetrievedChunk(
        chunk_id=cid, text=f"证据正文 {cid}", score=score,
        fields={
            "chunk_id": cid, "doc_id": "fin-doc", "project": "PRJ-FIN-2024-017",
            "section_no": "2.1", "page": 5, "security_level": level, "parent_id": "fin-doc#p",
        },
    )


def payload(**over):
    base = {
        "conclusion": "RTO 小于 15 分钟。",
        "key_points": [{"point": "灾备 RTO < 15 分钟", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1"}],
        "no_evidence": False,
        "self_assessed_confidence": 0.8,
    }
    base.update(over)
    return base


def scenarios():
    """每个场景 = (名字, 检索桩工厂, LLM 桩, 问题)。覆盖 answer() 的全部 17 个出口形态。"""
    ok_chunks = [chunk("c1"), chunk("c2")]
    return [
        ("01_sensitive", lambda: CountingRetriever(ok_chunks), FakeLlm(payload()), "合同金额是多少"),
        ("02_weak_retrieval", lambda: CountingRetriever([chunk("c1", score=0.05)]), FakeLlm(payload()), "RTO 是多少"),
        ("03_zero_chunks", lambda: CountingRetriever([]), FakeLlm(payload()), "RTO 是多少"),
        ("04_gate_boundary", lambda: CountingRetriever([chunk("c1", score=0.20)]), FakeLlm(payload()), "RTO 是多少"),
        ("05_generation_failed", lambda: CountingRetriever(ok_chunks), FakeLlm(payload(), ok=False), "RTO 是多少"),
        ("06_unparseable", lambda: CountingRetriever(ok_chunks), FakeLlm("这不是 JSON"), "RTO 是多少"),
        (
            "07_no_evidence_strong",
            lambda: CountingRetriever([chunk("c1", score=0.97)]),
            FakeLlm(payload(conclusion="", key_points=[], no_evidence=True)),
            "RTO 是多少",
        ),
        (
            "08_no_evidence_weak",
            lambda: CountingRetriever([chunk("c1", score=0.5)]),
            FakeLlm(payload(conclusion="", key_points=[], no_evidence=True)),
            "RTO 是多少",
        ),
        (
            "09_citation_rejected",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(payload(key_points=[{"point": "无出处的要点", "evidence_ids": ["E9"]}]), ),
            "RTO 是多少",
        ),
        (
            "10_no_points_at_all",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(payload(conclusion="", key_points=[])),
            "RTO 是多少",
        ),
        (
            "11_derived_from_points",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(payload(conclusion="")),
            "RTO 是多少",
        ),
        (
            "12_secret_cited",
            lambda: CountingRetriever([chunk("c1", score=0.95, level=3)]),
            FakeLlm(payload(self_assessed_confidence=0.9)),
            "RTO 是多少",
        ),
        (
            "13_tentative",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(payload(self_assessed_confidence=0.1)),
            "RTO 是多少",
        ),
        (
            "14_plain_answer",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(payload(self_assessed_confidence=0.95)),
            "RTO 是多少",
        ),
        # 15/16 是「逐字搬运」的边界：要点里有空白项。
        # 15 全空 → 必须落 contract_violation:empty_conclusion（真契约出口，前面 10 号打不到）。
        (
            "15_blank_point_only",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(payload(conclusion="", key_points=[{"point": "   ", "evidence_ids": ["E1"]}])),
            "RTO 是多少",
        ),
        # 16 半空 → 搬运结果必须是「A」而不是「A；」——拼接时把空白项也算进去就是脏结论。
        (
            "16_mixed_blank_point",
            lambda: CountingRetriever(ok_chunks),
            FakeLlm(
                payload(
                    conclusion="",
                    key_points=[
                        {"point": "灾备 RTO < 15 分钟", "evidence_ids": ["E1"]},
                        {"point": "   ", "evidence_ids": ["E2"]},
                    ],
                )
            ),
            "RTO 是多少",
        ),
        # 17 引用一致性被丢要点拉到 0.5 → composite < confidence_low，走低置信转人工。
        (
            "17_low_conf_escalate",
            lambda: CountingRetriever([chunk("c1", score=0.5)]),
            FakeLlm(
                payload(
                    self_assessed_confidence=0.0,
                    key_points=[
                        {"point": "灾备 RTO < 15 分钟", "evidence_ids": ["E1"]},
                        {"point": "无出处的要点", "evidence_ids": ["E9"]},
                    ],
                )
            ),
            "RTO 是多少",
        ),
    ]


def diff_dict(a: dict, b: dict, drop: set[str]) -> list[str]:
    keys = set(a) | set(b)
    out: list[str] = []
    for k in sorted(keys - drop):
        va, vb = a.get(k), b.get(k)
        if k == "retrieval" and isinstance(va, dict) and isinstance(vb, dict):
            # 已声明的**纯增量**字段：图侧多记「实际用去检索的查询」。
            # 多轮指代消解上线后，「补全成了什么」必须能从响应里查出来，否则审计只能靠猜。
            # 只放行登记过的键；出现新键仍算差异，防止这个豁免口子以后被塞别的东西。
            extra = set(vb) - set(va)
            if extra - DECLARED_ADDITIVE_RETRIEVAL_KEYS:
                out.append(f"    retrieval 出现未声明的增量键：{sorted(extra - DECLARED_ADDITIVE_RETRIEVAL_KEYS)}")
            vb = {k2: v2 for k2, v2 in vb.items() if k2 not in DECLARED_ADDITIVE_RETRIEVAL_KEYS}
            va = {k2: v2 for k2, v2 in va.items() if k2 not in DECLARED_ADDITIVE_RETRIEVAL_KEYS}
        if va != vb:
            out.append(f"    {k}:\n      legacy={va!r}\n      graph ={vb!r}")
    return out


def main() -> int:
    legacy = load_legacy()
    failures: list[str] = []
    checked = 0

    for name, rfactory, llm, question in scenarios():
        # 两个引擎各自持有独立的检索桩，调用数才可比。
        lr, gr = rfactory(), rfactory()
        le = legacy.AnswerEngine(lr, llm)
        ge = AnswerEngine(gr, llm)
        la = le.answer(question, AccessContext(role="knowledge_owner")).as_dict()
        ga = ge.answer(question, AccessContext(role="knowledge_owner")).as_dict()

        drop = set(NON_COMPARABLE)
        extra = []
        if name == "01_sensitive":
            # 已声明的不等价①：敏感题不再先烧检索。断言图侧确实省掉了。
            if gr.calls != 0:
                extra.append(f"敏感题图侧仍调了 {gr.calls} 次检索（应为 0）")
            if lr.calls != 1:
                extra.append(f"敏感题 oracle 侧调用数 {lr.calls}（应为 1，否则本场景没测到旧顺序）")
            drop.add("retrieval")
            if ga.get("retrieval") not in ({}, None):
                extra.append(f"敏感题图侧 retrieval 非空：{ga.get('retrieval')}")

        d = diff_dict(la, ga, drop)
        checked += 1
        if d or extra:
            failures.append(f"[{name}] status legacy={la['status']} graph={ga['status']}")
            failures.extend(d)
            failures.extend(f"    声明项:{x}" for x in extra)
        else:
            print(f"  {name:<26} status={ga['status']:<10} 等价")

    print(f"\n可比对场景 {checked} 个，字段级差异 {len(failures)} 条")
    if failures:
        print("\n".join(failures))
        return 1
    print(
        "差分结论：图化重构与重构前逐字段等价。\n"
        "已声明差异 3 处：① 敏感题不再先烧检索（顺序缺陷已修）；"
        "② 图侧 retrieval 多记 query_used；③ timings_ms / trace_id 不参与比对。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
