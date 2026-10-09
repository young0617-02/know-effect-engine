"""文档准入：格式校验 → 内容指纹去重 → 来源可信度分级（方案 6.1，未通过不入库）。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# 行业域对齐 V1.1 第一节：聚焦金融/能源/化工，兼顾高科技/互联网/制造/农业。
# Gate 0 的模拟语料只覆盖前三类，真实文档接入时若仍按三类收，会把制造业文档误杀。
ALLOWED_INDUSTRIES = {"金融", "能源", "化工", "高科技", "互联网", "制造", "农业"}
ALLOWED_DOC_TYPES = {"结项报告", "售前方案", "接口规范", "运维手册", "复盘记录", "合规要求"}
ALLOWED_SOURCE_LEVELS = {"A", "B", "C"}
# doc_id 允许中文（中文优先产品的溯源 ID 要有语义），但不允许引号/括号等能改写
# Milvus 表达式的字符——与 rbac._SAFE 同一套白名单思路。
_ID_RE = re.compile(r"^[0-9A-Za-z一-鿿][0-9A-Za-z一-鿿._\-]*$")

# 来源分级 → 是否允许入库（C 级外部摘录需人工复核标记，Gate 0 里按通过处理并记录）
SOURCE_MIN_CONFIDENCE = {"A": 0.95, "B": 0.80, "C": 0.60}


@dataclass
class AdmissionReport:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.accepted) + len(self.rejected) + len(self.duplicates),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "duplicates": len(self.duplicates),
        }


def normalize_body(card: dict[str, Any]) -> str:
    parts: list[str] = []
    for sec in card.get("sections", []) or []:
        parts.append(str(sec.get("heading", "")))
        parts.extend(str(p) for p in sec.get("paragraphs", []) or [])
    return re.sub(r"\s+", "", "".join(parts))


def checksum_of(card: dict[str, Any]) -> str:
    return hashlib.sha256(normalize_body(card).encode("utf-8")).hexdigest()


def check_format(card: dict[str, Any]) -> list[str]:
    v: list[str] = []
    doc_id = card.get("doc_id", "")
    if not isinstance(doc_id, str) or not _ID_RE.match(doc_id):
        v.append(f"doc_id 非法：{doc_id!r}")
    if not card.get("title") or len(str(card["title"])) > 200:
        v.append("title 缺失或超长")
    if card.get("industry") not in ALLOWED_INDUSTRIES:
        v.append(f"industry 越界：{card.get('industry')!r}")
    if card.get("doc_type") not in ALLOWED_DOC_TYPES:
        v.append(f"doc_type 越界：{card.get('doc_type')!r}")
    if card.get("source_level") not in ALLOWED_SOURCE_LEVELS:
        v.append(f"source_level 越界：{card.get('source_level')!r}")
    if card.get("security_level") not in (1, 2, 3):
        v.append(f"security_level 越界：{card.get('security_level')!r}")
    if not card.get("project") or not card.get("customer"):
        v.append("project/customer 缺失（无法做隔离）")
    sections = card.get("sections") or []
    if len(sections) < 1:
        v.append("sections 为空")
    seen_section: set[str] = set()
    for i, sec in enumerate(sections):
        no = str(sec.get("section_no", ""))
        if not no:
            v.append(f"section[{i}] 缺 section_no")
        elif no in seen_section:
            v.append(f"section_no 重复：{no}")
        seen_section.add(no)
        if not isinstance(sec.get("page"), int) or sec["page"] < 0:
            v.append(f"section[{i}] page 非法：{sec.get('page')!r}")
        paras = sec.get("paragraphs") or []
        if not paras:
            v.append(f"section[{i}] 无正文段落")
        for p in paras:
            if not isinstance(p, str) or not (10 <= len(p) <= 800):
                v.append(f"section[{i}] 段落长度越界（10~800）")
    return v


def admit(cards: list[dict[str, Any]]) -> AdmissionReport:
    rep = AdmissionReport()
    by_checksum: dict[str, str] = {}
    for card in cards:
        violations = check_format(card)
        if violations:
            rep.rejected.append((card.get("doc_id", "<无 id>"), violations))
            continue
        cs = checksum_of(card)
        card["_checksum"] = cs
        card["_source_confidence"] = SOURCE_MIN_CONFIDENCE[card["source_level"]]
        if cs in by_checksum:
            rep.duplicates.append((card["doc_id"], by_checksum[cs]))
            continue
        by_checksum[cs] = card["doc_id"]
        rep.accepted.append(card)
    return rep
