"""Parent-Child 分块：父块=章节（上下文补全），子块=检索单元（精准命中）。

子块只入向量库；父块留在权威源，答案组装时按 parent_id 回捞上下文。
"""

from __future__ import annotations

from typing import Any

from kee.ingest.parsers import is_table_block, split_table_block_rows

DEFAULT_CHILD_MAX = 280
DEFAULT_PARENT_MAX = 1600
DEFAULT_OVERLAP = 40


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


def chunk_document(
    card: dict[str, Any],
    version_id: str,
    child_max: int = DEFAULT_CHILD_MAX,
    parent_max: int = DEFAULT_PARENT_MAX,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """返回 PG chunk 行（含父块与子块），order_no 在版本内全局递增。"""
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
            text = f"{heading}｜{piece}" if not is_row else f"{heading}（表行）｜{piece}"
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
