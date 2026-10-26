"""多格式解析适配层：真实文件 → 「章节 + 页码 + 段落」结构。

设计约束：
  - 只依赖轻库（pypdf / python-docx），MinerU 作为可选后端留接口——本机只有 onnx
    权重、没有 mineru 包，不在项目里下载 GB 级依赖。
  - 页码不可得的格式（Markdown/DOCX 无固定分页）必须显式记 notes，不许假装有页码：
    引用溯源的「页码」是对外承诺，造假比缺失更糟。
  - 段落长度做归一化，保证能过 admission 的 10~800 字校验，而不是让真实文档被格式
    规则误杀；被拆分/合并的动作同样记进 notes。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PARA_MIN = 10
PARA_MAX = 800

_HEADING_MD = re.compile(r"^(#{1,6})\s+(.*)$")
# 章节号必须是「1~3 位、可带点分层级」，标题文本内不能出现句读——否则 PDF 里
# 「26572 规定的限量要求以下。」这类以数字开头的正文行会被误判成章节标题。
_HEADING_NUM = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){0,3})[、.\s]\s*([^\n。；！？，、]{2,40})\s*$")
_HEADING_CN = re.compile(r"^\s*(第[一二三四五六七八九十百]+[章节部分篇])\s*(\S.{0,60}?)\s*$")
_SENT_SPLIT = re.compile(r"(?<=[。；！？])")
_TABLE_SEP = re.compile(r"^[\s|:\-]+$")
_TABLE_ROW = re.compile(r"^\|.*\|$")


class UnsupportedFormat(Exception):
    pass


class ParseFailure(Exception):
    pass


@dataclass
class ParsedSection:
    section_no: str
    heading: str
    page: int | None
    paragraphs: list[str]


@dataclass
class ParsedFigure:
    """图片/图表/公式的入库记录；正文仍以段落形式进入分块链路。"""

    kind: str
    page: int | None = None
    text: str = ""
    description: str = ""
    section_no: str = ""
    bbox: tuple[float, float, float, float] | None = None
    source_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "page": self.page,
            "text": self.text,
            "description": self.description,
            "section_no": self.section_no,
            "bbox": list(self.bbox) if self.bbox else None,
            "source_name": self.source_name,
        }


@dataclass
class ParsedDocument:
    title: str
    sections: list[ParsedSection]
    source_format: str
    notes: list[str] = field(default_factory=list)
    meta_hint: dict[str, Any] = field(default_factory=dict)
    figures: list[ParsedFigure] = field(default_factory=list)

    def normalize(self) -> "ParsedDocument":
        """段落长度归一化：过长按句切，过短并入前一段，保证过准入格式校验。"""
        for sec in self.sections:
            out: list[str] = []
            for p in sec.paragraphs:
                p = _clean(p)
                if not p:
                    continue
                if len(p) > PARA_MAX:
                    pieces = _split_long(p)
                    self.notes.append(f"段落超长拆分：{sec.section_no} → {len(pieces)} 段")
                    out.extend(pieces)
                elif len(p) < PARA_MIN:
                    if out:
                        out[-1] = f"{out[-1]}{p}"
                        self.notes.append(f"短段并入前段：{sec.section_no}｜{p[:20]}")
                    else:
                        out.append(p if len(p) >= 4 else "")
                else:
                    out.append(p)
            kept = [p for p in out if len(p) >= PARA_MIN]
            if not kept and out:
                kept = [max(out, key=len)]
                self.notes.append(f"章节 {sec.section_no} 无达标长度段落，保留最长一段")
            sec.paragraphs = kept
        self.sections = [s for s in self.sections if s.paragraphs]
        return self

    def to_card(
        self,
        *,
        doc_id: str,
        industry: str,
        customer: str,
        project: str,
        doc_type: str = "结项报告",
        source_level: str = "A",
        security_level: int = 2,
        title_override: str = "",
    ) -> dict[str, Any]:
        """产出与知识库构建链路一致的文档卡片（chunk_document / admission 直接可用）。

        标题优先级：调用方显式给的 title_override > 解析出的文档标题 > doc_id。
        """
        title = title_override or self.title or doc_id
        return {
            "doc_id": doc_id,
            "title": title,
            "industry": industry,
            "customer": customer,
            "project": project,
            "doc_type": doc_type,
            "source_level": source_level,
            "security_level": int(security_level),
            "sections": [
                {
                    "section_no": s.section_no,
                    "page": s.page if s.page is not None else 0,
                    "heading": s.heading or f"章节 {s.section_no}",
                    "paragraphs": s.paragraphs,
                }
                for s in self.sections
            ],
            "_source_format": self.source_format,
            "_parse_notes": self.notes,
            "figures": [f.as_dict() for f in self.figures],
        }


def _fold_wraps(text: str) -> str:
    """软换行折成空格，但 Markdown 表格行（以 | 开头）各占一行不能被折。

    原规则是「单个换行=PDF 断行 → 折成空格，连续两个换行=段落分隔 → 保留」；
    表格行如果一起被折，行列结构就白还原了，所以按行首判定豁免。
    """
    out: list[str] = []
    for ln in (x.strip() for x in text.split("\n")):
        if out and out[-1] and ln and not ln.startswith("|") and not out[-1].startswith("|"):
            out[-1] = f"{out[-1]} {ln}"
        else:
            out.append(ln)
    return "\n".join(out)


def _clean(text: str) -> str:
    text = text.replace("　", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = _fold_wraps(text)
    return text.strip()


def is_table_block(text: str) -> bool:
    """整段都是 Markdown 表格行（含分隔行）才算表格，混排的不能按行切。"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return len(lines) >= 3 and all(_TABLE_ROW.match(ln) for ln in lines)


def split_table_block(text: str, limit: int) -> list[str]:
    """按行分组切表，每组都带上表头两行（列名 + 分隔行）。

    不带上表头，子块读出来是「| 框架 L 单元 | × | ○ |」这种无列名的残片，
    检索命中了也无法判定「×」对应哪一种有害物质——结构还原的收益就抵消掉了。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    header, body = lines[:2], lines[2:]
    if not body:
        return ["\n".join(lines)]
    base = sum(len(h) + 1 for h in header)
    groups: list[list[str]] = []
    cur: list[str] = list(header)
    for row in body:
        if len(cur) > len(header) and base + sum(len(r) + 1 for r in cur[2:]) + len(row) > limit:
            groups.append(cur)
            cur = list(header)
        cur.append(row)
    groups.append(cur)
    return ["\n".join(g) for g in groups]


def _md_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def split_table_block_rows(text: str) -> list[str]:
    """一行一个检索单元，渲染成「列名=值」而不是重复 Markdown 表头。

    为什么不是整张表一个子块：实测「试运行阶段要交付什么」这类单元格级问题，
    整表子块**进不了 Top-5**（12 题 Top1 命中 gold 率 0.0）——BM25 按长度归一化，
    200 字的表块被 20 字的旁注块压过，嵌入又把 8 行平均成一个向量。

    为什么也不是「表头 + 一行」的 Markdown 残块：那样每行都带同一段表头和同一个章节前缀，
    共享部分占块内 60%+ 文本，行与行几乎无法区分，实测 Top-5 在同表各行之间随机排列。
    渲染成 key=value 后共享文本只剩列名，区分度回到行本身。整张表仍完整保留在父块里，
    但父块正文当前**不参与答案组装**（回捞未实现，见 `chunking` 模块开头）。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return ["\n".join(lines)]
    names = _md_cells(lines[0])
    names = [n if n else f"列{i + 1}" for i, n in enumerate(names)]
    out: list[str] = []
    for row in lines[2:]:
        if _TABLE_SEP.match(row.strip()):
            continue
        vals = _md_cells(row)
        pairs = "；".join(f"{names[i]}={v}" for i, v in enumerate(vals) if i < len(names) and v)
        if pairs:
            out.append(pairs)
    return out or ["\n".join(lines)]


def _split_long(p: str) -> list[str]:
    if is_table_block(p):
        return split_table_block(p, PARA_MAX)
    parts: list[str] = []
    buf = ""
    for seg in _SENT_SPLIT.split(p):
        if not seg:
            continue
        if len(buf) + len(seg) <= PARA_MAX:
            buf += seg
        else:
            if buf:
                parts.append(buf)
            buf = seg
            while len(buf) > PARA_MAX:
                cut = max(buf.rfind("，", 0, PARA_MAX), PARA_MAX // 2)
                parts.append(buf[: cut + 1])
                buf = buf[cut + 1 :]
    if buf:
        parts.append(buf)
    return [x for x in (s.strip() for s in parts) if x]


class _SectionNumbering:
    """把「标题层级」序列转成 1 / 1.1 / 1.2 / 2 这样的章节号。"""

    def __init__(self) -> None:
        self.counters = [0, 0, 0, 0]
        self.level = 0

    def next(self, level: int) -> str:
        level = max(1, min(level, 4))
        if level > self.level + 1 and self.level:
            level = self.level + 1
        self.counters[level - 1] += 1
        for i in range(level, 4):
            self.counters[i] = 0
        self.level = level
        return ".".join(str(c) for c in self.counters[:level] if c > 0)


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    return meta, text[end + 3 :]


def parse_markdown(raw: str, filename: str) -> ParsedDocument:
    meta, body = _front_matter(raw)
    # front-matter 优先，其次首个 H1；文件名 stem 只是最后的兜底，不该抢在首个 H1 之前
    title = meta.pop("title", "")
    num = _SectionNumbering()
    sections: list[ParsedSection] = []
    cur: ParsedSection | None = None
    buf: list[str] = []
    table_cells: list[str] = []

    def flush() -> None:
        nonlocal buf
        if cur is not None:
            text = " ".join(x for x in buf if x).strip()
            if table_cells:
                text = (text + " 表：" + " / ".join(table_cells)).strip()
            if text:
                cur.paragraphs.append(text)
        buf = []
        table_cells.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue  # 空行不触发兜底章节，否则会凭空吃掉一个章节号
        m = _HEADING_MD.match(stripped)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip("# ").strip()
            if not title and level == 1:
                # 首个 H1 当文档标题，同时留作 1 号章节：
                # 否则它下面的正文会掉进「正文」幽灵章节，凭空吃掉一个章节号
                title = heading
                cur = ParsedSection(section_no=num.next(level), heading=heading, page=None, paragraphs=[])
                sections.append(cur)
                continue
            cur = ParsedSection(section_no=num.next(level), heading=heading, page=None, paragraphs=[])
            sections.append(cur)
            continue
        if cur is None:
            cur = ParsedSection(section_no=num.next(1), heading="正文", page=None, paragraphs=[])
            sections.append(cur)
        s = stripped
        if s.startswith("|"):
            if not _TABLE_SEP.match(s):
                cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
                table_cells.append("、".join(cells))
            continue
        if s.startswith(("- ", "* ", "1. ", "· ")):
            s = s[2:]
        buf.append(s)
    flush()

    doc = ParsedDocument(
        title=title or Path(filename).stem,
        sections=sections,
        source_format="markdown",
        notes=["Markdown 无固定分页，页码字段置 0 并标注不可得"],
        meta_hint=meta,
    )
    return doc.normalize()


def parse_plain_text(raw: str, filename: str) -> ParsedDocument:
    """纯文本：先按空行分块，再看块内首行是不是编号标题。

    不能拿整块去匹标题——标题与正文会被分块逻辑合在一起。
    """
    title = Path(filename).stem
    num = _SectionNumbering()
    sections: list[ParsedSection] = []
    cur: ParsedSection | None = None

    for block in re.split(r"\n\s*\n", raw):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        first = _clean(lines[0])
        if len(first) <= 40 and (_HEADING_NUM.match(first) or _HEADING_CN.match(first)):
            cur = ParsedSection(section_no=num.next(1), heading=first, page=None, paragraphs=[])
            sections.append(cur)
            rest = lines[1:]
        else:
            rest = lines
        if not rest:
            continue
        if cur is None:
            cur = ParsedSection(section_no=num.next(1), heading="正文", page=None, paragraphs=[])
            sections.append(cur)
        cur.paragraphs.append(_clean(" ".join(rest)))

    doc = ParsedDocument(
        title=title, sections=sections, source_format="text",
        notes=["纯文本无分页与标题层级，章节按编号行启发式切分"],
    )
    return doc.normalize()


def parse_docx(data: bytes, filename: str, media: Any | None = None) -> ParsedDocument:
    try:
        import docx
    except ImportError as e:  # pragma: no cover
        raise ParseFailure(f"缺少 python-docx：{e}") from e

    # 走 BytesIO 而不是临时文件：Windows 下 python-docx 会握住文件句柄，
    # 导致上传接口删临时件时抛 WinError 32（跨平台隐患不该留给文件系统）
    d = docx.Document(io.BytesIO(data))
    num = _SectionNumbering()
    sections: list[ParsedSection] = []
    cur: ParsedSection | None = None
    title = ""
    notes: list[str] = []

    def sec(level: int, heading: str) -> ParsedSection:
        nonlocal cur
        s = ParsedSection(section_no=num.next(level), heading=heading, page=None, paragraphs=[])
        sections.append(s)
        cur = s
        return s

    for para in d.paragraphs:
        text = _clean(para.text)
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        level = 0
        if style in ("title", "subtitle"):
            level = 1
        elif "heading" in style:
            digits = re.search(r"(\d)", style)
            level = int(digits.group(1)) if digits else 1
        elif _HEADING_NUM.match(text) and len(text) <= 40:
            level = 1
        if level:
            if not title and level == 1:
                title = text
                continue
            sec(level, text)
            continue
        if cur is None:
            title = title or text[:40]
            sec(1, "正文")
        cur.paragraphs.append(text)

    for t_i, table in enumerate(d.tables or []):
        rows = ["、".join(_clean(c.text) for c in r.cells if _clean(c.text)) for r in table.rows]
        rows = [r for r in rows if r]
        if not rows:
            continue
        if cur is None:
            sec(1, "表格")
        cur.paragraphs.append(f"表格{t_i + 1}：" + " / ".join(rows))
        notes.append(f"DOCX 表格按行展平为文本（第 {t_i + 1} 张，{len(rows)} 行），版面级还原待 MinerU 后端")

    # DOCX 的图片不在 python-docx 的段落文本里，直接从 zip 的 word/media/* 抽。
    # 抽不到/OCR 不可用时只记 note，不让整篇文档失败。
    figures: list[ParsedFigure] = []
    from kee.ingest.media import MediaPipeline

    pipeline = media or MediaPipeline.from_settings()
    pipeline.begin_document()
    image_texts = pipeline.docx_images(data, context=title or Path(filename).stem)
    image_paras: list[str] = []
    if image_texts:
        sec(1, "图片与图表")
        section_no = cur.section_no if cur is not None else ""
        for mt in image_texts:
            para = mt.paragraph()
            if not para:
                continue
            image_paras.append(para)
            figures.append(
                ParsedFigure(
                    kind=mt.kind,
                    page=mt.page,
                    text=mt.ocr_text,
                    description=mt.description,
                    section_no=section_no,
                    source_name=mt.source_name,
                )
            )
        if image_paras and cur is not None:
            cur.paragraphs.extend(image_paras)
    notes.extend(pipeline.notes)

    doc = ParsedDocument(
        title=title or Path(filename).stem,
        sections=sections,
        source_format="docx",
        notes=notes + ["DOCX 无固定分页（分页由渲染决定），页码置 0 并标注不可得"],
        figures=figures,
    )
    return doc.normalize()


def _parse_pdf_pypdf(data: bytes, filename: str) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ParseFailure(f"缺少 pypdf：{e}") from e

    reader = PdfReader(io.BytesIO(data))
    notes: list[str] = []
    num = _SectionNumbering()
    sections: list[ParsedSection] = []
    cur: ParsedSection | None = None
    title = ""

    def sec(level: int, heading: str, page: int) -> ParsedSection:
        nonlocal cur
        s = ParsedSection(section_no=num.next(level), heading=heading, page=page, paragraphs=[])
        sections.append(s)
        cur = s
        return s

    empty_pages = 0
    fragments = 0
    for pno, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            notes.append(f"第 {pno} 页抽取失败：{type(e).__name__}")
            continue
        if not text.strip():
            empty_pages += 1
            continue
        # 逐行处理：pypdf 无版面分析时，规格表会被拆成一堆短行；按行识别标题、
        # 把非标题行归并成段落，比按「空行分块」更能保住章节边界。
        buf: list[str] = []
        for ln in (_clean(x) for x in text.splitlines()):
            if not ln:
                continue
            if len(ln) <= 40 and (_HEADING_NUM.match(ln) or _HEADING_CN.match(ln)):
                if buf and cur is not None:
                    cur.paragraphs.append(" ".join(buf))
                    buf = []
                if not title:
                    title = ln
                    continue
                sec(1, ln, pno)
                continue
            if len(ln) < 12:
                fragments += 1
            if cur is None:
                title = title or ln[:40]
                sec(1, "正文", pno)
            buf.append(ln)
        if buf and cur is not None:
            cur.paragraphs.append(" ".join(buf))

    if fragments:
        notes.append(
            f"识别到 {fragments} 个短行片段（多为规格表/参数列），已归并进段落；"
            "版面级还原需启用 MinerU 后端，本轮 pypdf 路径不保证表格结构"
        )
    if empty_pages:
        notes.append(f"{empty_pages} 页无可抽取文本（扫描/图片页），需 MinerU OCR 后端才能补上")
    if len(sections) <= 1:
        notes.append("章节结构未被识别（全文归入单一章节），建议人工补标或启用 MinerU 版面解析")
    doc = ParsedDocument(
        title=title or Path(filename).stem,
        sections=sections,
        source_format="pdf",
        notes=notes
        + ["PDF 页码为真实页序，可直接用于引用溯源；标题层级按编号行启发式识别"],
    )
    return doc.normalize()


def _text_layer_is_thin(doc: ParsedDocument, pages: int) -> bool:
    """文本层薄到不值得信任：多半是扫描件或以图片排版的规范文件。"""
    from kee.config import get_settings

    chars = sum(len(x) for s in doc.sections for x in s.paragraphs)
    return chars < max(1, pages) * get_settings().pdf_min_chars_per_page


def _append_pdf_images(doc: ParsedDocument, data: bytes, filename: str, media: Any | None) -> ParsedDocument:
    """文本层 PDF 的内嵌图片：不走 MinerU，但图片仍要 OCR/VLM 后入库。"""
    from kee.config import get_settings

    if not get_settings().media_scan_text_pdf:
        return doc
    from kee.ingest.media import MediaPipeline

    pipeline = media or MediaPipeline.from_settings()
    pipeline.begin_document()
    items = pipeline.pdf_images(data, context=Path(filename).stem)
    if not items:
        doc.notes.extend(pipeline.notes)
        return doc
    by_page: dict[int, list[Any]] = {}
    for mt in items:
        by_page.setdefault(int(mt.page or 0), []).append(mt)
    for pno, group in sorted(by_page.items()):
        section_no = f"IMG-{pno or 0}"
        paras = [mt.paragraph() for mt in group if mt.paragraph()]
        if not paras:
            continue
        doc.sections.append(ParsedSection(section_no, f"第 {pno} 页图片与图表", pno or 0, paras))
        for mt in group:
            doc.figures.append(ParsedFigure(
                kind=mt.kind, page=mt.page, text=mt.ocr_text,
                description=mt.description, section_no=section_no,
                source_name=mt.source_name,
            ))
    doc.notes.extend(pipeline.notes)
    return doc.normalize()


def parse_pdf(data: bytes, filename: str, media: Any | None = None) -> ParsedDocument:
    """PDF 解析双后端：pypdf 文本层优先，文本层薄或显式要求时转 MinerU-ONNX（版面+OCR）。

    后端失败必须回退并留痕，不能因为版面模型出问题时整份文档解析不了。
    """
    from kee.config import get_settings

    s = get_settings()
    doc = _parse_pdf_pypdf(data, filename)
    backend = (s.pdf_parse_backend or "auto").lower()
    if backend == "pypdf":
        return doc
    try:
        import pypdf

        pages = len(pypdf.PdfReader(io.BytesIO(data)).pages)
    except Exception:  # noqa: BLE001
        pages = max(1, len(doc.sections))
    need = backend == "mineru_onnx" or (backend == "auto" and _text_layer_is_thin(doc, pages))
    if not need:
        return _append_pdf_images(doc, data, filename, media)
    try:
        from kee.ingest.mineru_onnx import MinerUPdfParser

        ocr_doc = MinerUPdfParser(media=media).parse(data, filename)
    except Exception as e:  # noqa: BLE001
        doc.notes.append(
            f"MinerU-ONNX 后端不可用（{type(e).__name__}: {str(e)[:120]}），已回退 pypdf 文本层"
        )
        return _append_pdf_images(doc, data, filename, media)
    if not ocr_doc.sections:
        doc.notes.append("MinerU-ONNX 未产出版面文本，保留 pypdf 结果")
        return _append_pdf_images(doc, data, filename, media)
    ocr_doc.notes.append(f"触发原因：{'显式指定后端' if backend == 'mineru_onnx' else f'文本层过薄（{pages} 页）'}")
    return ocr_doc


def parse_image(data: bytes, filename: str, media: Any | None = None) -> ParsedDocument:
    """独立图片：OCR + VLM 描述后作为单章节文档入库。"""
    from kee.ingest.media import MediaPipeline

    pipeline = media or MediaPipeline.from_settings()
    pipeline.begin_document()
    mt = pipeline.process_image_bytes(data, kind="image", index=1, context=Path(filename).stem,
                                      source_name=filename)
    para = mt.paragraph()
    if not para:
        why = pipeline.vlm.disabled_reason or pipeline._ocr_error or "图中未识别到可用文字"
        raise ParseFailure(f"图片未产出可入库内容：{why}")
    figure = ParsedFigure(
        kind="image",
        text=mt.ocr_text,
        description=mt.description,
        section_no="1",
        source_name=filename,
    )
    doc = ParsedDocument(
        title=Path(filename).stem,
        sections=[ParsedSection("1", "图片内容", None, [para])],
        source_format=(Path(filename).suffix.lower().lstrip(".") or "image"),
        notes=pipeline.notes + ["独立图片：无固定分页，页码置 0；OCR 文字与 VLM 描述分开标注"],
        figures=[figure],
    )
    return doc.normalize()


PARSERS: dict[str, Callable[..., ParsedDocument]] = {
    ".docx": parse_docx,
    ".pdf": parse_pdf,
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".webp": parse_image,
    ".gif": parse_image,
    ".bmp": parse_image,
    ".tif": parse_image,
    ".tiff": parse_image,
}
TEXT_FORMATS = {".md", ".markdown", ".txt", ".csv", ".log"}


def parse_file(
    path: str | Path | None = None,
    *,
    raw: bytes | None = None,
    filename: str | None = None,
    media: Any | None = None,
) -> ParsedDocument:
    """按扩展名分派解析；所有解析器统一吃字节流，不落临时文件。"""
    p = Path(path) if path else Path(filename or "untitled")
    name = filename or p.name
    suffix = (p.suffix or Path(name).suffix).lower()
    data = raw if raw is not None else p.read_bytes()

    if suffix in TEXT_FORMATS:
        text = data.decode("utf-8", errors="replace")
        return (
            parse_markdown(text, name)
            if suffix in (".md", ".markdown")
            else parse_plain_text(text, name)
        )
    fn = PARSERS.get(suffix)
    if fn is None:
        raise UnsupportedFormat(
            f"未支持的文档格式 {suffix or '?'}；本轮启用 md/txt/csv/log/docx/pdf/png/jpg/webp/bmp/tif，"
            "xlsx 与扫描件 PDF 需启用 MinerU 后端（接口已留位）"
        )
    return fn(data, name, media=media)
