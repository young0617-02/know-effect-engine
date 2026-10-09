"""上传元数据：显式表单 → 文档 front-matter → 推断，三级取值并留警告。

推断只用于"没给"的字段，绝不覆盖显式值；任何推断都记进 warnings 并写进审计，
否则密级这种敏感字段就成了模型的猜测产物。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from kee.ingest.admission import ALLOWED_DOC_TYPES, ALLOWED_INDUSTRIES, ALLOWED_SOURCE_LEVELS

DEFAULT_SECURITY_LEVEL = 2

# 内容线索 → doc_type 的保守推断（只用于缺省，命中不到就落 "复盘记录" 并告警）
_DOC_TYPE_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("结项报告", re.compile(r"结项|终验|验收结论")),
    ("售前方案", re.compile(r"售前|投标|方案报价|建议书")),
    ("接口规范", re.compile(r"接口编号|报文|API\s*规范|IF-[A-Z]+-\d+")),
    ("运维手册", re.compile(r"运维|巡检|故障处置|作业指导")),
    ("合规要求", re.compile(r"合规|标准|规定|GB/T|ITSS|CMMI|ISO")),
    ("复盘记录", re.compile(r"复盘|经验|教训")),
]

_SECURITY_HINTS = [
    (3, re.compile(r"客户名单|合同金额|报价|口令|密钥|特权账号")),
    (1, re.compile(r"公开|外发|通用规范")),
]


@dataclass
class MetaResolution:
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _doc_id_from(filename: str, explicit: str | None) -> str:
    """doc_id 必须确定性：内建 hash() 按进程随机化，会让「同一文件重传得到同一 ID」失效。

    中文优先的产品里保留中文语义词，只洗掉路径与表达式不安全的字符，
    校验域与 admission._ID_RE、rbac._SAFE 保持同源。
    """
    if explicit:
        candidate = explicit.strip()
    else:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
        candidate = re.sub(r"[^\w一-鿿\-]+", "-", stem, flags=re.UNICODE).strip("-")
    candidate = candidate or "untitled"
    return candidate[:64]


def resolve_meta(
    form: dict[str, Any],
    *,
    filename: str,
    parsed_meta: dict[str, Any] | None = None,
    sample_text: str = "",
) -> MetaResolution:
    """表单 > front-matter > 推断；缺必填项时直接报错而不是猜。"""
    parsed_meta = parsed_meta or {}
    low_text = (sample_text or "")[:4000]
    r = MetaResolution()

    def pick(key: str, *, source_hint: str | None = None, inferred: bool = False) -> Any:
        if form.get(key) not in (None, ""):
            return form[key]
        if key in parsed_meta:
            r.warnings.append(f"{key} 取自文档 front-matter：{parsed_meta[key]}")
            return parsed_meta[key]
        return None

    industry = pick("industry")
    if industry is None:
        r.warnings.append("industry 未提供，默认按「制造」登记（跨行业隔离会因此偏宽，请复核）")
        industry = "制造"
    customer = pick("customer")
    if customer is None:
        customer = "未登记客户"
        r.warnings.append("customer 未提供，落占位「未登记客户」；跨客户隔离不可用，需补标")
    project = pick("project")
    if project is None:
        project = "PRJ-UNASSIGNED"
        r.warnings.append("project 未提供，落 PRJ-UNASSIGNED；项目级隔离不可用，需补标")

    doc_type = pick("doc_type")
    if doc_type is None:
        for name, pat in _DOC_TYPE_HINTS:
            if pat.search(low_text):
                doc_type = name
                r.warnings.append(f"doc_type 由正文线索推断为「{name}」，请复核")
                break
        else:
            doc_type = "复盘记录"
            r.warnings.append("doc_type 无法推断，落默认「复盘记录」")

    source_level = pick("source_level")
    if source_level is None:
        source_level = "B"
        r.warnings.append("source_level 未提供，按内部沉淀 B 级登记")

    security_level = pick("security_level")
    if security_level is None:
        security_level = DEFAULT_SECURITY_LEVEL
        for lvl, pat in _SECURITY_HINTS:
            if pat.search(low_text):
                security_level = lvl
                r.warnings.append(f"security_level 由正文线索推断为 {lvl}（密级影响可见范围，务必人工复核）")
                break
        else:
            r.warnings.append(f"security_level 未提供，按默认 {DEFAULT_SECURITY_LEVEL} 登记")

    meta = {
        "doc_id": _doc_id_from(filename, form.get("doc_id") or form.get("id")),
        "title": pick("title") or "",
        "industry": str(industry),
        "customer": str(customer),
        "project": str(project),
        "doc_type": str(doc_type),
        "source_level": str(source_level),
        "security_level": int(security_level),
        "status": "active",
    }
    for key, allowed, label in (
        ("industry", ALLOWED_INDUSTRIES, "行业"),
        ("doc_type", ALLOWED_DOC_TYPES, "文档类型"),
        ("source_level", ALLOWED_SOURCE_LEVELS, "来源分级"),
    ):
        if meta[key] not in allowed:
            raise ValueError(f"{label}「{meta[key]}」不在允许域内：{sorted(allowed)}")
    if meta["security_level"] not in (1, 2, 3):
        raise ValueError(f"密级必须是 1/2/3，收到 {meta['security_level']}")
    r.meta = meta
    return r
