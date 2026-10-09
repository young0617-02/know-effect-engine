"""评测集装配：把「问题 + gold 关键词」回查成 gold chunk，产出可复跑的 qrels。

两级 gold 口径（必须写进报告，避免指标解读走样）：
  - gold_strict：子块正文本身包含全部关键词（AND，子串，大小写不敏感）。
  - gold_section：关键词命中在父块（章节）正文里 → 该章节下所有子块算 gold。
  分级取 gold_section ∪ gold_strict：章节级相关即算相关，与人工标注习惯一致；
  同时单独报告 strict 命中率，作为标注质量的自检信号。
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


def build_eval_set(pg: AuthoritativeStore, write_files: bool = True) -> dict[str, Any]:
    questions = load_questions()
    children = load_children(pg)
    parents = load_parents(pg)
    by_parent: dict[str, list[str]] = {}
    for c in children:
        by_parent.setdefault(c["parent_id"] or "", []).append(c["id"])

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for q in questions:
        kws = [k for k in q.get("gold_keywords", []) or [] if k]
        strict = [c["id"] for c in children if kws and _all_in(c["content"], kws)]
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
            }
        )

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
                    },
                    ensure_ascii=False,
                )
                for r in rows
            ),
            encoding="utf-8",
        )
        # 第二人复核表：待人工填写（搜索 TODO_ANNOTATOR2 定位所有待填项）
        template = art / "annotations_a2_template.csv"
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
