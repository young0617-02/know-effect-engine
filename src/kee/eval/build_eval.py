"""评测集装配：把「问题 + gold 关键词」回查成 gold chunk，产出可复跑的 qrels。

两级 gold 口径（必须写进报告，避免指标解读走样）：
  - gold_strict：子块正文本身包含全部关键词（AND，子串，大小写不敏感）。
  - gold_section：关键词命中在父块（章节）正文里 → 该章节下所有子块算 gold。
  分级取 gold_section ∪ gold_strict：章节级相关即算相关，与人工标注习惯一致；
  同时单独报告 strict 命中率，作为标注质量的自检信号。

人工逐块结论可以盖过关键词口径（题面的 `gold_manual_add` / `gold_manual_drop`，**只作用于 strict**）：
关键词是 gold 的机械近似、不是定义。第二人脱离本题答案字段逐块判「这块能不能独立答全该题」之后，
会出现两个方向相反的偏差——某块确实答得出，只因字面缺一个关键词串而漏收；或某块字面全含，
却只答了半问、给的是实测而非规范。两种都由逐块结论纠正，理由写在该题的 `gold_manual_why`。
section 保持纯机械（它已降为报告值，不该再掺手工成分）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kee.config import get_settings
from kee.db.pg import AuthoritativeStore


def load_questions() -> list[dict[str, Any]]:
    from kee.eval.eval_data import QUESTIONS

    return list(QUESTIONS)


def _all_in(text: str, kws: list[str]) -> bool:
    low = text.lower()
    return all(kw.lower() in low for kw in kws if kw)


# 「命中面 > 3 段即缺陷」这句在 eval_data 的 docstring 里写了一年，但**从来没有代码执行它**，
# 所以我发 9 道新题时这道检查一次都没跑（q-071 的 5 段就是这么漏出去的）。
# 现在做成闸门：默认只报不拦（`fail_on_defect=True` 才拒），因为存量里确有合法的宽命中，
# 硬拦会让基线跑不动；合法的例外必须在 `GOLD_DEFECT_WAIVER` 里写明理由，理由为空即视为没豁免。
MAX_STRICT_HITS = 3

# 题面点名文档类型时的措辞 → doc_id 里的类别标记（丙口径的操作化检测，见《项目定义》V1.17）。
DOC_KIND_IN_QUESTION = {"手册": "runbook", "规范": "ifspec", "摘录": "compliance"}

# 题面点了某类文档、gold 却全在别类文档里——这是**题面缺陷**，不是标注分歧：
# 要么改题面的限定词，要么证据本身找错了文档。q-044 就是这么一道（题面说「规范里」，
# 而两块 gold 分别在售前方案与运维手册里）。此前只靠我在注释里记着，检测器落地后由代码报出来。
DOC_SCOPE_DEFECT_WAIVER: dict[str, str] = {}


def _doc_kind(chunk_id: str) -> str:
    head = chunk_id.split("@")[0]
    for kind in ("closeout", "retro", "presale", "runbook", "ifspec", "compliance"):
        if kind in head:
            return kind
    return head.split("-")[1] if "-" in head else "?"


def _doc_scope_defect(q: dict[str, Any], strict: list[str]) -> dict[str, Any] | None:
    """题面点名的文档类型与实际 gold 所在文档类型不一致时登记。只在点名时生效（丙的限定）。"""
    if not strict or not q.get("gold_keywords"):
        return None
    named = {kind for word, kind in DOC_KIND_IN_QUESTION.items() if word in q["question"]}
    if not named:
        return None
    kinds = {_doc_kind(c) for c in strict}
    if kinds & named:
        return None
    reason = DOC_SCOPE_DEFECT_WAIVER.get(q["qid"], "")
    return {"qid": q["qid"], "named_kinds": sorted(named), "gold_kinds": sorted(kinds),
            "waived": bool(reason), "waiver_reason": reason, "question": q["question"]}


GOLD_DEFECT_WAIVER: dict[str, str] = {
    "q-054": "五步交付法：收窄后 strict 仍有 6 块 >3，这 6 块都逐条列了五个阶段，"
             "再收窄就要删第二人确认「能独立答全」的块——「>3」在这道题上是下界受标注约束，不是关键词写宽了。",
}


def _gold_defect(q: dict[str, Any], n_strict: int) -> dict[str, Any] | None:
    """只判两件事，与 eval_data 的口径一致：命中不到（由 unmatched 记）、命中面过宽（这里记）。
    `n_strict` 传的是**人工覆盖之后的最终 strict gold 块数**——缺陷问的是「这把尺子里装了几块」，
    不是「关键词这个近似 originally 抓到几块」。第一版我按覆盖前的量，结果 q-056 报 7 块，
    可它那 4 块早被第二人结论删掉、最终 gold 是 3 块且逐块与判是对齐——那是检测器在误伤修正本身。
    反过来也成立：manual_add 把 gold 抬过上限同样该被看一眼（清单型题就写豁免理由）。"""
    if not q.get("gold_keywords"):
        return None
    if n_strict <= MAX_STRICT_HITS:
        return None
    reason = GOLD_DEFECT_WAIVER.get(q["qid"], "")
    return {"qid": q["qid"], "n_strict": n_strict, "limit": MAX_STRICT_HITS,
            "waived": bool(reason), "waiver_reason": reason,
            "question": q["question"]}


def load_children(pg: AuthoritativeStore) -> list[dict[str, Any]]:
    with pg.connect() as conn:
        return conn.execute(
            "SELECT c.id, c.doc_id, c.version_id, c.parent_id, c.section_no, c.page, c.content, "
            "d.industry, d.project, d.security_level "
            "FROM chunk c JOIN document d ON d.id = c.doc_id WHERE c.is_parent = false ORDER BY c.id"
        ).fetchall()


def load_parents(pg: AuthoritativeStore) -> dict[str, str]:
    with pg.connect() as conn:
        rows = conn.execute("SELECT id, content FROM chunk WHERE is_parent = true").fetchall()
    return {r["id"]: r["content"] for r in rows}


def build_eval_set(pg: AuthoritativeStore, write_files: bool = True,
                   fail_on_gold_defect: bool = False) -> dict[str, Any]:
    questions = load_questions()
    children = load_children(pg)
    parents = load_parents(pg)
    by_parent: dict[str, list[str]] = {}
    for c in children:
        by_parent.setdefault(c["parent_id"] or "", []).append(c["id"])

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    overridden: list[str] = []
    defects: list[dict[str, Any]] = []
    scope_defects: list[dict[str, Any]] = []
    for q in questions:
        kws = [k for k in q.get("gold_keywords", []) or [] if k]
        strict = [c["id"] for c in children if kws and _all_in(c["content"], kws)]
        # 逐块人工结论优先于关键词口径：`gold_keywords` 只是 gold 的**机械近似**，
        # 当它与第二人「这块能不能独立答全该题」的逐块判断相反时，以判断为准（裁决见 V1.0.11 / V1.10）。
        # 只作用于 strict 这把主尺；section 保持纯机械（它已经降为报告值，不该再混入手工成分）。
        drop = set(q.get("gold_manual_drop") or [])
        add = set(q.get("gold_manual_add") or [])
        if drop or add:
            strict = sorted(({*strict} - drop) | add)
            overridden.append(q["qid"])
        defect = _gold_defect(q, len(set(strict)))
        if defect:
            defects.append(defect)
        scope_defect = _doc_scope_defect(q, sorted(set(strict)))
        if scope_defect:
            scope_defects.append(scope_defect)
        section: list[str] = []
        if kws:
            for c in children:
                ptext = parents.get(c["parent_id"] or "", "")
                if ptext and _all_in(ptext, kws) and c["id"] not in section:
                    section.append(c["id"])
        gold = sorted(set(section) | set(strict))
        if kws and not gold:
            unmatched.append(q["qid"])
        rows.append(
            {
                **q,
                "gold_chunk_ids": gold,
                "gold_strict_ids": sorted(set(strict)),
                "gold_section_ids": sorted(set(section)),
                "n_gold": len(gold),
                "gradeable": bool(gold),
                "gold_defect": defect,
            }
        )

    unwaived = [d for d in defects if not d["waived"]]
    unwaived_scope = [d for d in scope_defects if not d["waived"]]
    if fail_on_gold_defect and (unwaived or unwaived_scope):
        parts = [f"{d['qid']}(命中 {d['n_strict']}>{d['limit']} 段)" for d in unwaived]
        parts += [f"{d['qid']}(题面点 {d['named_kinds']} 而 gold 在 {d['gold_kinds']})" for d in unwaived_scope]
        raise ValueError(
            "gold 缺陷未豁免：" + "、".join(parts)
            + "。要么收窄 gold_keywords / 改题面限定词，要么在豁免表里写明为什么这样是合法的。")

    stats = {
        "n_questions": len(rows),
        "n_children": len(children),
        "n_parents": len(parents),
        "answerable": sum(1 for r in rows if r["expected_behavior"] == "answer"),
        "refuse": sum(1 for r in rows if r["expected_behavior"] == "refuse"),
        "escalate": sum(1 for r in rows if r["expected_behavior"] == "escalate"),
        "gradeable": sum(1 for r in rows if r["gradeable"]),
        "gold_empty_but_answerable": sum(
            1 for r in rows if r["expected_behavior"] == "answer" and not r["gold_chunk_ids"]
        ),
        "unmatched_qids": unmatched,
        # 关键词命中面 > 上限的题（在人工覆盖**之前**量）。默认只登记不拦，
        # `fail_on_gold_defect=True` 时未豁免的那些直接拒绝装配。
        "gold_defects": defects,
        "gold_defect_count": len(defects),
        "gold_defect_unwaived": [d["qid"] for d in defects if not d["waived"]],
        # 题面点名的文档类型与实际 gold 所在文档类型不一致（丙口径的检测）
        "doc_scope_defects": scope_defects,
        "doc_scope_defect_unwaived": [d["qid"] for d in scope_defects if not d["waived"]],
        # 人工覆盖必须显式可见：哪几题的 strict gold 被逐块结论改过，一眼能查出来
        "manual_override_qids": overridden,
        "mean_gold_per_question": round(
            sum(r["n_gold"] for r in rows if r["gold_chunk_ids"]) / max(sum(1 for r in rows if r["gold_chunk_ids"]), 1), 2
        ),
        "by_category": {},
    }
    for r in rows:
        stats["by_category"].setdefault(r["category"], 0)
        stats["by_category"][r["category"]] += 1

    if write_files:
        art = Path(get_settings().resolved_artifact_dir())
        (art / "eval_set.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        (art / "qrels.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "qid": r["qid"],
                        "question": r["question"],
                        "category": r["category"],
                        "expected_behavior": r["expected_behavior"],
                        "gold_keywords": r.get("gold_keywords", []),
                        "gold_chunk_ids": r["gold_chunk_ids"],
                        "gold_strict_ids": r["gold_strict_ids"],
                        "reference_answer": r.get("reference_answer", ""),
                        "gold_defect": r.get("gold_defect"),
                    },
                    ensure_ascii=False,
                )
                for r in rows
            ),
            encoding="utf-8",
        )
        # 第二人复核表：待人工填写（搜索 TODO_ANNOTATOR2 定位所有待填项）
        template = art / "annotations_a2_template.csv"
        # 这张表**每次装配都会被重写**。它是派生物、不是已收标注（真正的工单是 *_fill.csv），
        # 但它长得像一张待填工单，悄悄重写会让人以为有人改了自己的标注。所以重写要出声。
        if template.exists():
            print(f"[注意] {template.name} 将被本轮 gold 重写（派生物；已收标注在 *_fill.csv，不受影响）")
        template.write_text(
            "qid,question,gold_keywords,annotator2_agree(1/0 待填TODO_ANNOTATOR2),annotator2_gold_chunk_ids(待填TODO_ANNOTATOR2),note\n"
            + "\n".join(
                f'{r["qid"]},"{r["question"]}","{ "|".join(r.get("gold_keywords", [])) }",TODO_ANNOTATOR2,TODO_ANNOTATOR2,'
                for r in rows
            ),
            encoding="utf-8",
        )
    return {"stats": stats, "questions": rows}


if __name__ == "__main__":
    pg = AuthoritativeStore()
    out = build_eval_set(pg)
    print(json.dumps(out["stats"], ensure_ascii=False, indent=2))
