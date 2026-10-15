"""给「无答案 7 + 敏感 5」这 12 题出**真人工单**——把 DoD#3 幻觉率的另一半变成可判定（#57）。

为什么必须单独一张：`answer_correct_fill.csv` 那 67 行只覆盖 `expected_behavior == answer` 的题，
所以「**该拒却答了**」（真幻觉的正身）与「**该转人工却直接答了**」至今没有真人标签，
DoD#3 那半边一直只有代理读数（结论里的数字有没有出处）。这张表就是来补这半边的。

三件设计上的要紧事（写进产物，不靠记忆）

1. **工单上不能出现 `category`**。这 12 题的 category 字面就是「无答案 / 敏感」——
   印在题面上等于先把「该不该拒」告诉标注人，标签立刻不再是独立判断。
   所以 category / expected_behavior / reference_answer 全部只进 `hallucination_key.json`（回收后才可打开）。
2. **判据只能问「处置与所见证据是否一致」，不能问「全库到底有没有答案」**。
   标注人看到的是**系统这次实际检索到的 Top-5 块正文**（含章节/页码/密级），
   所以两列判的是：① 这些证据够不够正面回答该题；② 系统这次的处置对不对。
   边界必须同句说：如果正确的那块**压根没被检到**，标注人也看不见 ⇒ 那一行会判成「证据不足、拒得对」。
   这不是缺陷而是分工——**「该检到却没检到」属检索缺口（#58），不属这张表的职责范围**。
   （与 Kappa 那条同一纪律：观测域是候选池，不得表述成全库。）
3. **幻觉的分子是机械导出的，不是直接问出来的**：`evidence_sufficient == 0` **且** 系统给了非空结论
   ⇒ 计一条「无依据却出结论」。直接问「这是幻觉吗」会把判断权交给我预设的措辞；
   拆成两个可独立判的小问题，再按规则合成，才是可复核的口径。

源 = **生产入口**真调用（宿主 API `POST /api/v1/answers`，一问一 session），
标签要标的是用户真能看到的输出，不是 in-process 拼出来的另一份。
证据正文走 `POST /api/v1/chunks/batch`（与检索层同口径过滤），拒答行也有证据面可判——
它的 `retrieval.top_chunk_ids` 就是系统看过的那五块。

护栏：已填过就拒绝覆盖（先留底再退出，要重来加 `--force`）；**三列判断只能真人填，我不代填**；
生成时点 + 题面哈希 + 服务依赖状态一起写进 `hallucination_run.json`，
因为标签只能配**它判过的那一代产物**（#56 实测：重跑一次生成会改 ~7% 行的结论文本）。

跑法：uv run python scripts/m5_hallucination_sheet.py [--force]
产物：var/gate0/hallucination_run.json（系统输出与证据面）
      var/gate0/hallucination_fill.csv（待真人填）
      var/gate0/hallucination_key.json（含 category/expected/reference，**回收前不得打开**）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.eval.build_eval import build_eval_set  # noqa: E402

from _identity import ensure_accounts, login_http  # noqa: E402

BASE = os.environ.get("KEE_LAT_BASE", "http://127.0.0.1:38000")
# 身份走登录令牌：`KEE_ALLOW_HEADER_IDENTITY` 的默认已于 2026-10-15 翻成 False（用户裁「交付物默认 false、脚本显式带令牌」）；
# `Authorization` 由 main() 登录后补进 HDRS，审计里的 actor 因此是账号名 verify-owner，不再是请求头里随手写的 halluc-sheet。
HDRS = {"Content-Type": "application/json"}
ART = ROOT / "var" / "gate0"
FILL = ART / "hallucination_fill.csv"
KEY = ART / "hallucination_key.json"
RUN = ART / "hallucination_run.json"

FILL_COLS = ["evidence_sufficient__fill_1_or_0", "behavior_correct__fill_1_or_0",
             "failure_kind__fill_optional"]
COLS = (["qid", "question", "status", "needs_human", "conclusion", "key_points",
         "n_citations", "cited_ids", "evidence_blocks"] + FILL_COLS)

JUDGE_LINES = [
    "# 判据一 evidence_sufficient：只看「evidence_blocks」这几块**能不能正面答全该题**。1=够，0=不够，"
    "?=判不了（不许凑数）。**没写进去的内容不算数**——判的是这几块给不给得出答案，不是你记忆里库里有没有。",
    "# 判据二 behavior_correct：系统这次的处置对不对。1=对，0=不对，?=判不了。三条细则："
    "① 判据一填 0 且系统给了结论 ⇒ 这里填 0（无依据却出结论）；"
    "② 判据一填 1 且系统拒答/不给结论 ⇒ 这里填 0（该答却拒）；"
    "③ 需要**人工审批或涉密口径**才能对外给的题，系统应 needs_human=1；直接给终答案算 0，拒答也算 0（方向不同）。",
    "# failure_kind（可选，只在 behavior_correct=0 时填）：无依据却出结论 / 该答却拒 / 该转人工未转 / "
    "不当转却转了 / 结论与证据相反 / 结论只答了一半 / 其他",
    "# evidence_blocks 为空 = 系统**没做检索**就直接给了处置。这种行判据一填 ?（没有证据面就无从判起），"
    "判据二照常按三条细则判。算分时 ? 不计入判据一的分母——这是结构性的不可判，不是漏填。",
    "# ⚠ 三列都是**人的判断**，只能真人填。填完跑 uv run python scripts/m5_hallucination_rate.py 出读数。",
]


def post(path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers=HDRS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def guard_existing_fill(force: bool) -> str | None:
    """已经有人填过就不覆盖——覆盖等于销毁人工标注，这是不可逆的。"""
    if not FILL.exists() or force:
        return None
    with FILL.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    filled = [r for r in rows
              if any((r.get(c) or "").strip() for c in FILL_COLS)]
    if not filled:
        return None
    stamp = now_stamp()
    bak = FILL.with_name(f"hallucination_fill.filled-{stamp}.csv")
    shutil.copyfile(FILL, bak)
    return (f"已有 {len(filled)}/{len(rows)} 行填过，本次拒绝覆盖；旧表复制到 {bak.name}。"
            f"确认要重来加 --force（那份留底仍在）。")


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def kp_cell(kps: list) -> str:
    """key_points 的元素是 {"point":…,"evidence_ids":[…]}，不是裸字符串——直接 join 会 TypeError。
    要点后面带上它自己声明的出处编号，标注人才知道「哪句是哪块说的」。"""
    out = []
    for i, k in enumerate(kps or [], start=1):
        if isinstance(k, dict):
            ids = ",".join(k.get("evidence_ids") or [])
            out.append(f"要点{i}[{ids}]：{k.get('point', '')}")
        else:
            out.append(f"要点{i}：{k}")
    return " ｜ ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="确认覆盖已填工单（先复制留底）")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="允许在检索降级腿的情况下出工单（默认拒绝）")
    args = ap.parse_args()

    blocked = guard_existing_fill(args.force)
    if blocked:
        print(f"[拒绝覆盖] {blocked}")
        return 4

    try:
        h = json.loads(urllib.request.urlopen(BASE + "/healthz", timeout=10).read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[拒绝起跑] 宿主 API 不在 {BASE}（{e}）；标签要标的是用户真能看到的输出，"
              "所以必须走生产入口，不 in-process 生成")
        return 2
    deps = h.get("deps", {})
    if not deps.get("milvus", {}).get("ok") or not deps.get("postgres", {}).get("ok"):
        print(f"[拒绝起跑] 依赖不健康：{h}——半套依赖下生成的答案会给出一张配不上真值的工单")
        return 2
    print(f"[前置] healthz={h}")

    ensure_accounts()
    HDRS.update(login_http(BASE, "verify-owner"))

    pg = AuthoritativeStore()
    qs = build_eval_set(pg, write_files=False)["questions"]
    picked = [q for q in qs if q["expected_behavior"] in ("refuse", "escalate")]
    if len(picked) != 12:
        # 题量对不上就别出工单：这张表的分母写死在 DoD#3 的表述里（无答案 7 + 敏感 5）
        print(f"[拒绝产出] 期望 12 道非作答题，实得 {len(picked)}——"
              "题量变了要先改 DoD/报告里的分母表述，不能让工单和文档各说一套")
        return 3

    # 依赖活性前置：`/api/tags` 返回 200 不代表模型在（本轮真事——Ollama 起来但模型库指错盘，
    # 12 行里 6 行报 generation_failed + ConnectError）。**要查的是"有没有那个模型"，不是"端口通不通"**。
    s = get_settings()
    try:
        tags = json.loads(urllib.request.urlopen(s.ollama_url + "/api/tags", timeout=10).read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[拒绝起跑] 本地推理服务不可达（{s.ollama_url}：{e}）——"
              "这张工单要判的是系统的**作答行为**，推理服务没起来时全是降级，判了也是白判")
        return 2
    names = {m.get("name") for m in tags.get("models") or []}
    if s.llm_model not in names:
        print(f"[拒绝起跑] 推理服务在跑但**没有模型 {s.llm_model}**（列表：{sorted(names) or '空'}）。"
              '多半是 OLLAMA_MODELS 指错了目录——这不是「没加载」，是「没有」')
        return 2

    # 预热：存储栈刚重启时第一批发问会让稠密腿超时，那批数不是系统行为而是冷启动副产品。
    post("/api/v1/answers", {"session_id": f"s-warm-{uuid.uuid4().hex[:6]}",
                             "question": "灾备切换的 RTO 是多少？"})

    out_rows: list[dict] = []
    for q in picked:
        sid = f"s-hall-{uuid.uuid4().hex[:10]}"
        t = time.perf_counter()
        r = post("/api/v1/answers", {"session_id": sid, "question": q["question"]})
        cited = [c.get("chunk_id") for c in (r.get("citations") or []) if c.get("chunk_id")]
        seen = list(dict.fromkeys((r.get("retrieval") or {}).get("top_chunk_ids") or []))
        blocks: list[str] = []
        if seen:
            cb = post("/api/v1/chunks/batch", {"ids": seen})
            by_id = {it["chunk_id"]: it for it in cb.get("items") or []}
            for cid in seen:
                it = by_id.get(cid)
                if not it:
                    blocks.append(f"[{cid}] （该块对当前身份不可见、不存在或已非生效版本）")
                    continue
                blocks.append(
                    f"[{cid}] 行业={it.get('industry')} 项目={it.get('project')} "
                    f"文档={it.get('doc_id')} 章节={it.get('section_no')} 页={it.get('page')} "
                    f"密级={it.get('security_level')}｜{it.get('content', '')}")
            hidden = len(cb.get("hidden") or [])
        else:
            hidden = 0
        out_rows.append({
            "qid": q["qid"], "question": q["question"], "session_id": sid,
            "status": r.get("status"), "needs_human": bool(r.get("needs_human")),
            "conclusion": (r.get("conclusion") or "").strip(),
            "key_points": r.get("key_points") or [],
            "citations": cited, "retrieved_ids": seen, "hidden_blocks": hidden,
            "evidence_blocks": blocks,
            "degraded": (r.get("retrieval") or {}).get("degraded") or [],
            "notes": r.get("notes") or [],
            "confidence": (r.get("confidence") or {}).get("composite"),
            "client_ms": round((time.perf_counter() - t) * 1000, 1),
        })
        print(f"  {q['qid']} status={r.get('status')} needs_human={bool(r.get('needs_human'))} "
              f"引用={len(cited)} 证据面={len(seen)} 结论字数={len(out_rows[-1]['conclusion'])}")

    by_q = {x["qid"]: x for x in out_rows}

    # 前置校验分两种，**处置不同**（这是本轮真踩出来的区分）：
    # ① 检索降级腿 / `generation_failed` = **基础设施没跑通**（超时、连接被拒）——照它出的工单
    #    让人判的是一个并未运行的路径，拒绝产出；
    # ② `unparseable_output`（模型吐出不合契约的 JSON）= **真实的系统行为**，正是要判的对象之一，
    #    放行但在产物里点名标出，不让它混在"正常"里。
    infra = [(x["qid"], x["degraded"], [n for n in x["notes"] if n.startswith("generation_failed")])
             for x in out_rows if x["degraded"] or any(n.startswith("generation_failed") for n in x["notes"])]
    unparse = [x["qid"] for x in out_rows if any("unparseable" in n for n in x["notes"])]
    if infra and not args.allow_degraded:
        print(f"[拒绝出工单] {len(infra)}/{len(out_rows)} 行落在**基础设施失败**路径上：{infra[:5]}。"
              "这类行的 status 是「依赖没跑通」而不是「系统选择这样答」，判它等于给我一张假标签表；"
              "确认要判请加 --allow-degraded（决定会记进产物）")
        return 5

    # 题面哈希：标签只配这一代题面+这一代输出，改题面或重跑生成都会让它失配（#56 的教训）
    qhash = zlib.crc32("|".join(f"{x['qid']}::{x['question']}"
                                 for x in sorted(out_rows, key=lambda z: z["qid"])).encode("utf-8"))

    with FILL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for line in JUDGE_LINES:
            w.writerow([line])
        w.writerow([f"# 生成时点 {time.strftime('%Y-%m-%dT%H:%M:%S')}｜源 {RUN.name}｜共 {len(out_rows)} 行"
                     f"｜题面哈希 {qhash}｜身份 knowledge_owner（可见密级 3，所以能判「该不该转人工」）"])
        w.writerow(["# 观测域=系统实际检索到的 Top-5 证据面，不是全库："
                    "正确块压根没被检到时本表会判成「拒得对」，那种缺口归 #58 检索诊断，不在本表职责内"])
        w.writerow(COLS)
        for q in picked:
            x = by_q[q["qid"]]
            cell = " \n---\n ".join(x["evidence_blocks"]) or "（系统一个候选块都没检到）"
            w.writerow([
                x["qid"], x["question"], x["status"], 1 if x["needs_human"] else 0,
                x["conclusion"], kp_cell(x["key_points"]), len(x["citations"]),
                " | ".join(x["citations"]), cell, "", "", "",
            ])

    RUN.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "entrypoint": BASE + "/api/v1/answers + /api/v1/chunks/batch（生产入口）",
        "healthz": h,
        "question_hash": qhash,
        "scope_note": ("观测域是系统实际检索到的 Top-5 证据面（候选池），不是全库。"
                       "「该检到却没检到」属检索缺口（#58），本表判的是处置与所见证据是否一致。"),
        "pairing_note": ("标签只配这一代产物：题面哈希 + 本次输出都记在这里。"
                         "重跑生成会改结论文本（#56 实测 ~7%），所以刷新本表要整套重来、不能与旧标签混算。"),
        "n_rows": len(out_rows),
        "infra_degraded": [x[0] for x in infra],
        "unparseable_rows": unparse,
        "allow_degraded": bool(args.allow_degraded),
        "rows": out_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    KEY.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "含 category / expected_behavior / reference_answer，**只在标注回收之后**才能打开比对",
        "rows": [{"qid": q["qid"], "category": q["category"],
                  "expected_behavior": q["expected_behavior"],
                  "reference_answer": q.get("reference_answer", ""),
                  "gold_chunk_ids": q.get("gold_chunk_ids", [])} for q in picked],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    empty = sum(1 for x in out_rows if not x["conclusion"])
    print(f"\n工单：{FILL}\n  {len(out_rows)} 行（每题一行；三列判断**只能真人填**）"
          f"｜其中空结论/拒答 {empty} 行")
    print(f"  运行记录：{RUN}（系统输出与证据面，题面哈希 {qhash}）")
    print(f"  答案卷：{KEY}（回收前不要打开）")
    print("  提醒：这张表测「处置是否与所见证据一致」，不测「全库有没有答案」——后者是 #58。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
