"""Parent-Child 分块：父块=章节（或章节内的 token 窗口），子块=检索单元（精准命中）。

子块入向量库；父块只写进权威源，`parent_id` 存在 chunk 行与向量库标量字段上。
**答案组装按 `parent_id` 回捞父块正文**（需求 US-1.2），实现在
`kee.qa.answer.apply_parent_context`，由 `KEE_PARENT_CONTEXT_IN_ANSWER` 控制、
`KEE_PARENT_CONTEXT_MAX_TOKENS` 限预算。父块正文不入库向量层，检索命中的始终只有子块。
本模块 docstring 曾写「答案组装时回捞」而实际只喂子块（名实不符，2026-10-21 消融时照出来改直）；
2026-10-22 用户裁「照需求实现回捞」后，这条现在是**已实现**，上面的实测代价一并记在 `kee.config`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kee.ingest.parsers import is_table_block, split_table_block_rows

DEFAULT_CHILD_MAX = 280
DEFAULT_PARENT_MAX = 1600
DEFAULT_OVERLAP = 40


@dataclass(frozen=True)
class Granularity:
    """按 **token** 声明的分块粒度（用户 2026-10-22 下发：Child 200/重叠 30、Parent 1000/重叠 150）。

    与下面那三个 `DEFAULT_*` 字符常量是**两套并存**的口径，不是替换：现网索引与全部历史读数
    绑在字符口径上，改默认值会让任何一次重索引悄悄作废人工标注。所以 legacy 路径原样保留，
    要换新粒度必须显式传 `granularity=`（`build_index` / 离线复跑都照这条走）。
    """

    child_tokens: int = 200
    child_overlap_tokens: int = 30
    parent_tokens: int = 1000
    parent_overlap_tokens: int = 150

    def describe(self) -> dict[str, int]:
        return {"child_tokens": self.child_tokens, "child_overlap_tokens": self.child_overlap_tokens,
                "parent_tokens": self.parent_tokens, "parent_overlap_tokens": self.parent_overlap_tokens}

    def scheme_tag(self) -> str:
        """拼进 `document_version.chunk_scheme` 的后缀，让索引自带切法。

        一个版本行是「这套块是怎么切出来的」唯一的可查证据；不写进去，
        以后对着两份 chunk_scheme 相同的索引比读数，就分不清是模型变了还是分块变了。
        """
        return (f"+tok:c{self.child_tokens}/{self.child_overlap_tokens}"
                f"-p{self.parent_tokens}/{self.parent_overlap_tokens}")


TOK_GRANULARITY = Granularity()   # 用户下发的 token 口径粒度
# 表格行块的前缀标记：检索层靠它把「行块」与正文块区分开（名次来源不同），
# 所以常量必须与 chunking 同源，不能两处各写一份字面量
TABLE_ROW_MARK = "（表行）"


def _split_paragraph(text: str, child_max: int, overlap: int) -> list[str]:
    if len(text) <= child_max:
        return [text]
    stride = max(child_max - overlap, 1)
    out = []
    for start in range(0, len(text), stride):
        piece = text[start : start + child_max]
        if piece.strip():
            out.append(piece)
        if start + child_max >= len(text):
            break
    return out


def _chunk_document_tokens(card: dict[str, Any], version_id: str, g: Granularity) -> list[dict[str, Any]]:
    """token 口径的分块：**父块也分窗并带重叠**（这是新增能力——legacy 路径的父块=整章节，不分窗不重叠）。

    三条不妥协的地方：
    - 子块与父块都是**原文子串**（连同字符区间），所以归属判断按区间重叠算，不按字符串相等找——
      重叠区内同一段文字会同时属于两个父块，`find` 会认错。
    - 表格仍走行级「列名=值」子块（legacy 实测过：整表一块时单元格级问题进不了 Top-5），
      行块**不做 token 切分**——把一行切成两半等于把「列名=值」的语义切没了。
    - `【标题 / 章节】` 抬头与子块的 `章节标题｜` 前缀都是**标签**，不计进 token 预算；预算只约束正文。
      这条口径要随产物报告：否则"子块 200 tokens"实际量的是 200+标题，而入库向量层的正是带标题的整串。
    """
    from kee.models.tokens import count_tokens, cut_spans_by_token_budget

    rows: list[dict[str, Any]] = []
    order = 0
    doc_id = card["doc_id"]
    for sec in card["sections"]:
        section_no = str(sec["section_no"])
        page = int(sec["page"])
        heading = str(sec.get("heading", "") or "")
        paras = [str(p) for p in sec.get("paragraphs", []) or []]
        body = "\n".join(paras)
        label = f"【{card['title']} / {section_no} {heading}】\n"

        pw = cut_spans_by_token_budget(body, g.parent_tokens, g.parent_overlap_tokens) or [(0, 0, body)]
        parents: list[dict[str, Any]] = []
        for pi, (a, b, piece) in enumerate(pw, start=1):
            pid = f"{version_id}#{section_no}#p{pi}"
            parents.append({"id": pid, "span": (a, b)})
            order += 1
            text = label + piece
            rows.append({"id": pid, "version_id": version_id, "doc_id": doc_id, "parent_id": None,
                         "section_no": section_no, "page": page, "order_no": order,
                         "content": text, "char_count": len(text), "is_parent": True})

        def owner_of(ca: int, cb: int) -> str:
            best, best_ov = parents[0]["id"], -1
            for p in parents:
                ov = min(cb, p["span"][1]) - max(ca, p["span"][0])
                if ov > best_ov:
                    best, best_ov = p["id"], ov
            return best

        # 非表格段落先按 token 切成段内片，再贪心合并相邻片（合并后仍是 body 的连续子串）。
        # ⚠ 段内切分返回的是**段内偏移**，必须加 base 变成节内绝对偏移——直接拿去索引 body
        #   会让子块文本、父块归属一起错位（第一版就错在子块数正好等于节数上）。
        segments: list[list[Any]] = []      # [节内起, 节内止, 文本]
        pos = 0
        for p in paras:
            if is_table_block(p):
                segments.append([pos, pos + len(p), p])          # 表格段整体留给行级展开
                pos += len(p) + 1
                continue
            cuts = cut_spans_by_token_budget(p, g.child_tokens, g.child_overlap_tokens) or [(0, len(p), p)]
            for a, b, _seg in cuts:
                segments.append([pos + a, pos + b, p[a:b]])
            pos += len(p) + 1

        merged: list[list[Any]] = []
        for seg in segments:
            if merged:
                lo, hi = merged[-1][0], seg[1]
                candidate = body[lo:hi]
                # 不把表格并进正文：行级「列名=值」是被实测过的检索单元（整表一块时单元格级问题进不了 Top-5）
                if (count_tokens(candidate) <= g.child_tokens
                        and not is_table_block(candidate)
                        and not any(is_table_block(body[x[0]:x[1]]) for x in (merged[-1], seg))):
                    merged[-1] = [lo, hi, candidate]
                    continue
            merged.append(seg)

        for ci, (a, b, piece) in enumerate(merged, start=1):
            order += 1
            if is_table_block(piece):
                for ri, row in enumerate(split_table_block_rows(piece), start=1):
                    order += 1
                    text = f"{heading}{TABLE_ROW_MARK}｜{row}"
                    rows.append({"id": f"{version_id}#{section_no}#c{ci}r{ri}", "version_id": version_id,
                                 "doc_id": doc_id, "parent_id": owner_of(a, b), "section_no": section_no,
                                 "page": page, "order_no": order, "content": text,
                                 "char_count": len(text), "is_parent": False})
                continue
            text = f"{heading}｜{piece}"
            rows.append({"id": f"{version_id}#{section_no}#c{ci}", "version_id": version_id,
                         "doc_id": doc_id, "parent_id": owner_of(a, b), "section_no": section_no,
                         "page": page, "order_no": order, "content": text,
                         "char_count": len(text), "is_parent": False})
    return rows


def granularity_from_settings(settings_obj: Any | None = None) -> "Granularity | None":
    """按配置决定分块口径。返回 `None` = 走 legacy 字符口径（默认）。

    放在这里而不是让每个调用点各自读配置：摄取有两条路径（`build.py` 全量重建、
    `service.py` 单篇入库），各读各的迟早分叉，分叉那天同一篇文档在两条路径下切出不同的块。
    """
    from kee.config import get_settings

    s = settings_obj or get_settings()
    if (s.chunk_granularity or "chars").strip().lower() not in ("tokens", "token"):
        return None
    return Granularity(
        child_tokens=s.child_tokens,
        child_overlap_tokens=s.child_overlap_tokens,
        parent_tokens=s.parent_tokens,
        parent_overlap_tokens=s.parent_overlap_tokens,
    )


def chunk_document(
    card: dict[str, Any],
    version_id: str,
    child_max: int = DEFAULT_CHILD_MAX,
    parent_max: int = DEFAULT_PARENT_MAX,
    overlap: int = DEFAULT_OVERLAP,
    granularity: Granularity | None = None,
) -> list[dict[str, Any]]:
    """返回 PG chunk 行（含父块与子块），order_no 在版本内全局递增。

    `granularity` 给了就走 **token 口径**（含父块分窗+重叠）；不给是 legacy 字符口径——
    现网索引与全部历史读数都出自后者，**改默认值等于让任何一次重索引悄悄作废人工标注**，
    所以新粒度必须显式传参。
    """
    if granularity is not None:
        return _chunk_document_tokens(card, version_id, granularity)
    rows: list[dict[str, Any]] = []
    order = 0
    doc_id = card["doc_id"]
    # version_id 已含文档前缀（形如 doc-id@v1），不能再拼一次 doc_id，
    # 否则分块 ID 变成「doc@doc@v1#1.1#c1」，溯源时读起来是自相矛盾的
    for sec in card["sections"]:
        section_no = str(sec["section_no"])
        page = int(sec["page"])
        heading = str(sec.get("heading", "") or "")
        paras = [str(p) for p in sec.get("paragraphs", []) or []]
        parent_text = f"【{card['title']} / {section_no} {heading}】\n" + "\n".join(paras)
        parent_id = f"{version_id}#{section_no}#parent"
        if len(parent_text) > parent_max:
            parent_text = parent_text[:parent_max]
        order += 1
        rows.append(
            {
                "id": parent_id,
                "version_id": version_id,
                "doc_id": doc_id,
                "parent_id": None,
                "section_no": section_no,
                "page": page,
                "order_no": order,
                "content": parent_text,
                "char_count": len(parent_text),
                "is_parent": True,
            }
        )
        # 子块：按段落聚合，超长段落再切片，保证子块不跨章节。
        # 表格单独走行级展开，pieces 里带 is_row 标记而不是事后靠正则认——
        # 行块渲染成「列名=值」后已经不是 Markdown 了，事后认不回来。
        buf = ""
        child_idx = 0
        pieces: list[tuple[str, bool]] = []
        for p in paras:
            if is_table_block(p):
                if buf:
                    pieces.append((buf, False))
                    buf = ""
                pieces.extend((row, True) for row in split_table_block_rows(p))
                continue
            for seg in _split_paragraph(p, child_max, overlap):
                if buf and len(buf) + 1 + len(seg) > child_max:
                    pieces.append((buf, False))
                    buf = seg
                else:
                    buf = f"{buf} {seg}".strip()
            if len(buf) >= child_max:
                pieces.append((buf, False))
                buf = ""
        if buf:
            pieces.append((buf, False))
        for i, (piece, is_row) in enumerate(pieces, start=1):
            child_idx = i
            # 行块加「表行」前缀，让答案层知道这段是表格的一行而不是正文
            text = f"{heading}｜{piece}" if not is_row else f"{heading}{TABLE_ROW_MARK}｜{piece}"
            order += 1
            rows.append(
                {
                    "id": f"{version_id}#{section_no}#c{child_idx}",
                    "version_id": version_id,
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "section_no": section_no,
                    "page": page,
                    "order_no": order,
                    "content": text[:child_max * 3],
                    "char_count": len(text),
                    "is_parent": False,
                }
            )
    return rows
