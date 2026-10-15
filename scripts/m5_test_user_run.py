"""测试用户运行验收：把「能正常运行工作」从纸面 DoD 变成一次真跑。

**为什么需要这支脚本**：DoD#6 此前标「不可测（无真实用户）」，用户 2026-10-13 裁决
「可以用测试用户替代」。替代只成立在一维上——它补上了「操作者是人、会带着任务来、答不上会重问」
这个**形状**，但补不上两样东西：**上线前基线**与**时间维度**。所以三条业务指标一条变读数、
两条仍是结构性拿不到。这张表的全部意义就是把「哪条能报、哪条不能」写死在代码里，
免得下次有人拿「模拟自助解答率 92%」对外说成「业务指标达标」。

跑法：uv run python scripts/m5_test_user_run.py
产物：var/gate0/m5_test_user.json + reports/m5/M5_测试用户运行验收.md
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "var" / "gate0"
OUT_MD = ROOT / "reports" / "m5" / "M5_测试用户运行验收.md"

BASE = os.environ.get("KEE_TU_BASE", "http://127.0.0.1:38000")
ANSWERABLE = ("answer", "tentative")

# 三个测试用户画像。**角色是系统里真有的角色**（`kee.retrieval.rbac.ROLE_MAX_LEVEL`），
# 不是我新发明的身份——否则测不到 RBAC 那半条链。
USERS = {
    "u-pm-01": {"role": "pm", "max_level": 3, "label": "项目经理（全密级可见）"},
    "u-delivery-01": {"role": "delivery_engineer", "max_level": 2, "label": "交付工程师（密级 ≤2）"},
    "u-newcomer-01": {"role": "newcomer", "max_level": 1, "label": "新人（密级 ≤1）"},
}

# 冒烟剧本：`--smoke` 时只跑这些（开发路径快）。**默认跑全量，见 `build_plan`。**
# 这里只保留「多轮指代」必须固定挂靠的那一题，其余一律不人工挑。
SMOKE_QIDS = ["q-001", "q-013", "q-031", "q-040", "q-047", "q-052", "q-059", "q-063", "q-066", "q-070", "q-003"]
# 指代轮挂在**看得见且答得上**的那条任务后面，否则测的是权限不是消解。
# 锚点不硬写 key：自动派生角色后 key 未必存在（第一版就因此整块被跳过、检查项静默变 False）。
ANAPHORA_PICK = "first_answer_task"
ANAPHORA_QUESTION = "它的判定阈值是多少？"

# 追问的行为模型：**首轮没给可用答案才重问**（答上了就不会追问）。
# 这不是「真实用户的重复提问率」——真实用户会不会重问、重问几次，脚本无从知道。
# 它测的是另一件有信息量的事：**首轮回答的可直接用率**。措辞必须这样落进产物。
REPHRASE = "换个说法问：{q}"

GATES = {"模拟自助解答率": 0.60, "一次通过率": 0.60, "运行链路检查": 1.0}


from _identity import BY_ROLE, USERS as ACCOUNTS, ensure_accounts, login_http  # noqa: E402

_TOKENS: dict[str, dict] = {}


def h(role: str, actor: str) -> dict:
    """身份走登录令牌。请求头通道默认已关闭（用户 2026-10-15 裁「脚本显式带令牌」），
    所以这里的 actor 必须是库里真有的账号名——预热与 SSE 检查那两个「只报角色」的调用点
    按 `BY_ROLE` 落到该角色的验收账号。

    `ACCOUNTS` 是本脚本的**用户画像表** `USERS` 之外的东西（那是「谁以什么角色问哪道题」），
    这里指落库播种的账号清单，所以别名引进来，不让两个 USERS 互相覆盖。
    """
    if not _TOKENS:
        ensure_accounts()
    username = actor if actor in ACCOUNTS else BY_ROLE[role]
    if username not in _TOKENS:
        _TOKENS[username] = login_http(BASE, username)
    return {**_TOKENS[username], "Content-Type": "application/json"}


def post(path: str, body: dict, role: str, actor: str, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers=h(role, actor), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str, role: str, actor: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(BASE + path, headers=h(role, actor))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def preflight() -> tuple[dict, list[str]]:
    """跑之前先把「这台机器现在到底能不能干活」问清楚。

    护栏的来历：Ollama `/api/tags` 返回 200 不代表模型在（曾 `llama-server binary not found`），
    而依赖挂了时系统会给出**整齐而难看**的读数——那不是测量，是一次宕机伪装成一次通过。
    """
    from kee.config import get_settings

    problems: list[str] = []
    s = get_settings()
    try:
        health = get("/healthz", "pm", "preflight", timeout=20)
    except Exception as e:  # noqa: BLE001
        return {}, [f"宿主 API 不在 {BASE}（{type(e).__name__}: {e}）——先 `uv run python scripts/serve_api.py`"]
    deps = health.get("deps") or {}
    chunks = (deps.get("milvus") or {}).get("chunks")
    if not (deps.get("milvus") or {}).get("ok") or not (deps.get("postgres") or {}).get("ok"):
        problems.append(f"依赖未就绪：{json.dumps(deps, ensure_ascii=False)[:200]}")
    if not chunks or int(chunks) < 100:
        problems.append(f"向量集合只有 {chunks} 块，不像完整索引（期望 193）——别在残缺索引上出业务读数")

    try:
        tags = json.loads(urllib.request.urlopen(s.ollama_url.rstrip("/") + "/api/tags", timeout=10)
                          .read().decode("utf-8"))
        names = [m.get("name") for m in tags.get("models") or []]
    except Exception as e:  # noqa: BLE001
        names, problems = [], problems + [f"生成模型清单取不到（{type(e).__name__}）"]
    if s.llm_model not in names:
        problems.append(f"生成模型 {s.llm_model!r} 不在本机（在跑的：{names}）——端口通不等于模型在")
    if s.cloud_fallback_enabled:
        problems.append("云端兜底是 True：数据不出域这条硬约束断了，不该跑")

    from kee.db.pg import AuthoritativeStore

    pg = AuthoritativeStore()
    with pg.connect() as conn:
        unreviewed = conn.execute(
            "SELECT count(*) AS n FROM document_version WHERE status='effective' AND reviewed_at IS NULL"
        ).fetchone()["n"]
        sec3 = conn.execute(
            "SELECT count(*) AS n FROM chunk c JOIN document d ON d.id = c.doc_id WHERE d.security_level = 3"
        ).fetchone()["n"]
    return {"healthz": health, "chunks": chunks, "ollama_models": names,
            "cloud_fallback_enabled": s.cloud_fallback_enabled,
            "effective_unreviewed_versions": unreviewed,
            "production_chunks_at_sec3": sec3,
            "llm_model": s.llm_model, "entrypoint": BASE + "/api/v1/answers"}, problems


def load_questions() -> dict[str, dict]:
    from kee.eval.build_eval import build_eval_set
    from kee.db.pg import AuthoritativeStore

    qs = build_eval_set(AuthoritativeStore(), write_files=False)["questions"]
    return {q["qid"]: q for q in qs}


def flush_search_cache() -> int:
    import redis

    from kee.config import get_settings

    r = redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=5)
    n = 0
    for k in r.scan_iter(match="kee:search:*", count=500):
        r.delete(k)
        n += 1
    return n


def run_task(sid: str, question: str, role: str, actor: str) -> dict:
    t = time.perf_counter()
    r = post("/api/v1/answers", {"session_id": sid, "question": question}, role, actor)
    return {"session_id": sid, "client_ms": round((time.perf_counter() - t) * 1000, 1),
            "status": r.get("status"),
            "conclusion_chars": len((r.get("conclusion") or "").strip()),
            "n_citations": len(r.get("citations") or []),
            "needs_human": bool(r.get("needs_human")),
            "confidence": r.get("confidence"),
            "cache_hit": bool((r.get("retrieval") or {}).get("cache_hit")),
            # 稠密腿超预算时检索**静默退化成 BM25-only**（实测 API 重启后第一呼 `dense_timeout>2500ms`，
            # 热态为空）。这一列不记进产物，读的人就看不出「答上了」是靠两腿还是靠一条腿。
            "retrieval_degraded": list((r.get("retrieval") or {}).get("degraded") or []),
            "contract_keys_ok": all(k in r for k in ("status", "conclusion", "citations", "session_id")),
            "resp": r}


def role_expectation(eval_expected: str, gold_levels: list[int], max_level: int) -> dict:
    """把「评测集说这题该怎么处置」换成「**这个角色**问这题该怎么处置」。

    这一层不分出来就会算错分母：q-047 的 gold 在密级 3，交付工程师（≤2）按设计看不见它，
    系统拒答是**正确**的；若照评测集的 `answer` 标签去判，就把一次正确的权限拦截记成了一次失败。
    """
    visible = [lv for lv in gold_levels if lv is not None and int(lv) <= max_level]
    if eval_expected == "refuse":
        # 语料内本就没有依据：gold 为空是设计，不是权限
        return {"桶": "拒答", "该给结论": False, "该转人工": False, "可见 gold": len(visible)}
    if not visible:
        return {"桶": "权限拦截", "该给结论": False, "该转人工": False, "可见 gold": 0,
                "为什么": f"gold 密级 {sorted({str(x) for x in gold_levels})} 全部高于角色上限 {max_level}"}
    if eval_expected == "escalate":
        # 转人工类**允许作答**，但必须 needs_human（密级 3 的证据进了引用就要人来看）
        return {"桶": "转人工", "该给结论": True, "该转人工": True, "可见 gold": len(visible)}
    return {"桶": "作答", "该给结论": True, "该转人工": False, "可见 gold": len(visible)}


def build_plan(by_q: dict, lv: dict, qids: list[str] | None = None) -> list[dict]:
    """剧本 = **评测集全量**，每题自动指派给「刚好看得见它的那道题的最低权限角色」。

    为什么不让脚本作者挑题：冒烟版我手挑了 5 道，结果 5/5 = 100%——
    n=5 且「题由我选」的比率不能当业务读数，它只会复现我自己的选择。
    自动派生之后，分母是评测集决定的，不是我决定的；角色指派也让 RBAC 过滤真的参与测量。
    """
    order = sorted(USERS.items(), key=lambda kv: kv[1]["max_level"])          # newcomer(1) → delivery(2) → pm(3)
    plan = []
    for qid in sorted(qids if qids is not None else by_q.keys()):
        q = by_q.get(qid)
        if not q:
            continue
        glv = [lv.get(g) for g in (q.get("gold_chunk_ids") or [])]
        need = min([x for x in glv if x is not None], default=None)
        if need is None:
            # 无 gold 的题（语料内本就无依据）指派给最高权限：这样它的拒答归因于**语料没有**，
            # 而不是「这个角色看不见」——两件事混在一个读数里就都测不清了。
            user = "u-pm-01"
        else:
            user = next((u for u, cfg in order if cfg["max_level"] >= need), "u-pm-01")
        e = role_expectation(q.get("expected_behavior", "answer"), glv, USERS[user]["max_level"])
        plan.append({"user": user, "qid": qid, "role": USERS[user]["role"], "max_level": USERS[user]["max_level"],
                     "评测集期望": q.get("expected_behavior"),
                     "gold密级": sorted({str(x) for x in glv if x is not None}), **e})
    # (2) 拦截任务必须**显式加**：上面那条规则永远把题交给看得见它的角色，
    #     于是「看不见 gold 的正确处置」那一桶恒为空——RBAC  half 的链子根本没进测量，
    #     而「权限拦截正确率 1.0」会是一个空集上的真。取 gold 最低密级=3 的前 4 题（按 qid 排序，不挑）。
    sec3 = [p2 for p2 in plan if p2["gold密级"] == ["3"]][:4]
    for p2 in sec3:
        low = "u-newcomer-01"
        e = role_expectation(p2["评测集期望"], [3], USERS[low]["max_level"])
        plan.append({**p2, "user": low, "role": USERS[low]["role"], "max_level": USERS[low]["max_level"], **e})
    return plan


def chunk_levels(pg) -> dict[str, int]:
    with pg.connect() as conn:
        return {r["id"]: r["s"] for r in conn.execute(
            "SELECT c.id AS id, d.security_level AS s FROM chunk c JOIN document d ON d.id = c.doc_id")}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-cache", action="store_true",
                    help="默认起跑前清一次检索缓存；留着命中会把「系统真的检索到了」换成「上次算过」")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 SMOKE_QIDS 那 11 题（开发快路径）。**冒烟读数不出业务结论**：n 太小且题是手挑的")
    args = ap.parse_args()

    env, problems = preflight()
    if problems:
        for p in problems:
            print(f"[拒绝起跑] {p}")
        print("不落任何产物：半成品读数比没有读数更有害。")
        return 3

    by_q = load_questions()
    if args.smoke:
        missing = [qid for qid in SMOKE_QIDS if qid not in by_q]
        if missing:
            print(f"[拒绝起跑] 冒烟剧本里的 qid 不在评测集：{missing}")
            return 3

    from kee.db.pg import AuthoritativeStore

    pg = AuthoritativeStore()
    lv = chunk_levels(pg)
    plan = build_plan(by_q, lv, SMOKE_QIDS if args.smoke else None)
    n_answer_tasks = len([p for p in plan if p["桶"] == "作答"])
    print(f"[剧本] {'冒烟' if args.smoke else '评测集全量'}：{len(plan)} 个任务，"
          f"其中作答 {n_answer_tasks}、权限拦截 {len([p for p in plan if p['桶'] == '权限拦截'])}、"
          f"拒答 {len([p for p in plan if p['桶'] == '拒答'])}、转人工 {len([p for p in plan if p['桶'] == '转人工'])}")
    if not n_answer_tasks:
        print("[拒绝起跑] 一个「作答」任务都没有——分母为空的比率一律不许出")
        return 3

    flushed = None if args.keep_cache else flush_search_cache()
    print(f"[清缓存] 删除 kee:search:* 键 {flushed} 个" if flushed is not None else "[保留缓存]")

    # warm-up：API 刚起来时嵌入模型是冷的，第一呼稠密腿必超 2500ms 预算 ⇒ 检索静默退化成 BM25-only。
    # 不 warm-up 就把这一呼算进业务读数，测到的就不是「这套系统的检索质量」而是「冷启动的形状」。
    # 代价如实记在产物里（warm_up 那一呼自己的 degraded 会被记下来）。
    warm = run_task(f"s-tu-warm-{uuid.uuid4().hex[:6]}", "灾备切换的 RTO 目标",
                    USERS["u-pm-01"]["role"], "warmup")
    warm_up = {"status": warm["status"], "client_ms": warm["client_ms"],
               "retrieval_degraded": warm["retrieval_degraded"]}
    print(f"[warm-up] status={warm['status']} {warm['client_ms']}ms degraded={warm['retrieval_degraded']}")

    # 一个任务的多轮共用同一个 session_id（指代消解要靠会话历史）
    sessions = {f"{p['user']}:{p['qid']}": f"s-tu-{p['user']}-{p['qid']}-{uuid.uuid4().hex[:6]}" for p in plan}
    turns: list[dict] = []
    for p in plan:
        sid = sessions[f"{p['user']}:{p['qid']}"]
        q = by_q[p["qid"]]["question"]
        r1 = run_task(sid, q, p["role"], p["user"])
        turns.append({**p, "turn": 1, "kind": "首问", **{k: v for k, v in r1.items() if k != "resp"}})
        if p["桶"] == "作答" and r1["status"] not in ANSWERABLE:
            r2 = run_task(sid, REPHRASE.format(q=q), p["role"], p["user"])
            turns.append({**p, "turn": 2, "kind": "追问", **{k: v for k, v in r2.items() if k != "resp"}})
            print(f"  {p['user']:14s} {p['qid']} t1[{p['桶']}]={r1['status']} → 追问 t2={r2['status']}")
        elif len(turns) % 10 == 0:
            print(f"  …{len(turns)} 轮：最近 {p['qid']} status={r1['status']}")

    # 多轮指代：接在第一条「作答」任务之后，问句里没有主题词
    base = next((p for p in plan if p["桶"] == "作答"), None)
    if base and f"{base['user']}:{base['qid']}" in sessions:
        r3 = run_task(sessions[f"{base['user']}:{base['qid']}"], ANAPHORA_QUESTION, base["role"], base["user"])
        turns.append({**base, "turn": 3, "kind": "指代", **{k: v for k, v in r3.items() if k != "resp"}})
        print(f"  {base['user']:14s} 指代轮 status={r3['status']} 引用={r3['n_citations']}")

    # —— 越权探测：让**看不见 gold 的角色**去问敏感题，逐条看引用密级 ——
    leak_checks = []
    probe_qids = [p["qid"] for p in plan if p["桶"] in ("权限拦截", "转人工") and "3" in p["gold密级"]]
    low = "u-newcomer-01"
    for qid in dict.fromkeys(probe_qids):
        sid = f"s-tu-{low}-{qid}-{uuid.uuid4().hex[:6]}"
        rr = run_task(sid, by_q[qid]["question"], USERS[low]["role"], low)
        cited_levels = [c.get("security_level") for c in (rr["resp"].get("citations") or [])]
        leak_checks.append({"qid": qid, "asked_by": USERS[low]["role"], "max_level": USERS[low]["max_level"],
                            "status": rr["status"], "n_citations": rr["n_citations"],
                            "conclusion_chars": rr["conclusion_chars"],
                            "cited_security_levels": sorted({str(x) for x in cited_levels if x is not None}),
                            # ⚠ 「没引用」也会让这条判成未越权——那是**空集上的真**。所以把引用数一并落进产物，
                            # 读的人能立刻看出这次是「挡住了」还是「压根没检到东西」。
                            "note": "n_citations=0：本条只证明没泄漏，不证明「看见了但按密级拒绝」"
                                    if not rr["n_citations"] else "",
                            "leaked_above_clearance": any(
                                x is not None and int(x) > USERS[low]["max_level"] for x in cited_levels)})
        print(f"  越权探测 {qid}（newcomer）status={rr['status']} 引用={rr['n_citations']} "
              f"密级={leak_checks[-1]['cited_security_levels'] or '（无）'}")

    # —— SSE 一轮：证明真通路在跑，不是只有非流式端点 ——
    sse = None
    try:
        sid = f"s-tu-sse-{uuid.uuid4().hex[:6]}"
        body = json.dumps({"session_id": sid, "question": by_q[plan[0]["qid"]]["question"]}).encode("utf-8")
        req = urllib.request.Request(BASE + "/api/v1/answers/stream", data=body,
                                     headers=h(plan[0]["role"], "sse-check"), method="POST")
        t0 = time.perf_counter()
        first_visible = None
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:") and first_visible is None and "{" in line:
                    try:
                        ev = json.loads(line[5:].strip())
                    except Exception:  # noqa: BLE001
                        continue
                    if (ev.get("text") or ev.get("delta") or ev.get("conclusion")):
                        first_visible = round((time.perf_counter() - t0) * 1000, 1)
        sse = {"ok": True, "first_visible_ms": first_visible}
    except Exception as e:  # noqa: BLE001
        sse = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # —— 库里核对：审计与 session_turn 有没有真落（psycopg 的 ANY 要 list，传 tuple 会报 malformed array）——
    sids = list({t["session_id"] for t in turns} | set(sessions.values()))
    with pg.connect() as conn:
        st = conn.execute(
            "SELECT count(*) AS n, count(total_ms) AS with_ms FROM session_turn WHERE session_id = ANY(%s)",
            (sids,)).fetchone()
        au = conn.execute("SELECT count(*) AS n FROM audit_log WHERE actor = ANY(%s)",
                          (list(USERS.keys()),)).fetchone()
    audit_view = get("/api/v1/audit?limit=5", "knowledge_owner", "audit-check")

    degraded_turns = [t for t in turns if t["retrieval_degraded"]]
    first_q = [t for t in turns if t["kind"] == "首问"]
    ask_tasks = [t for t in first_q if t["桶"] == "作答"]
    solved_first = {t["qid"] for t in ask_tasks if t["status"] in ANSWERABLE}
    reask = [t for t in turns if t["kind"] == "追问"]
    rescue = {t["qid"] for t in reask if t["status"] in ANSWERABLE}
    n_task = len(ask_tasks)
    block_tasks = [t for t in first_q if t["桶"] == "权限拦截"]
    blocked_ok = [t for t in block_tasks if t["conclusion_chars"] == 0 and t["n_citations"] == 0]
    escort_tasks = [t for t in first_q if t["桶"] == "转人工"]
    escorted = [t for t in escort_tasks if t["needs_human"]]
    refuse_tasks = [t for t in first_q if t["桶"] == "拒答"]
    refused_ok = [t for t in refuse_tasks if t["conclusion_chars"] == 0]

    self_serve = round(len(solved_first | rescue) / n_task, 4) if n_task else None
    first_pass = round(len(solved_first) / n_task, 4) if n_task else None
    reask_share = round(len(reask) / len(turns), 4) if turns else None

    checks = {
        "契约完整（每轮都含 status/conclusion/citations/session_id）": all(t["contract_keys_ok"] for t in turns),
        "每轮都有引用或明确拒答（零引用却出结论 = 0 起）": not [
            t for t in turns if t["conclusion_chars"] > 0 and t["n_citations"] == 0],
        "session_turn 全行带 total_ms（e2e 漏计修复在线生效）": st["n"] == st["with_ms"],
        "审计真落库": au["n"] > 0,
        "审计读接口可用（GET /api/v1/audit）": bool(audit_view.get("items") or audit_view.get("rows")),
        "SSE 流式可通且有可读首字": bool(sse.get("ok")) and sse.get("first_visible_ms") is not None,
        "多轮指代消解后仍给出可用答案": next((t for t in turns if t["kind"] == "指代"), {}).get("status") in ANSWERABLE,
        "权限拦不住的题 = 0（看不见 gold 就不该给结论）": len(blocked_ok) == len(block_tasks),
        "该转人工的每轮 needs_human 都为真": len(escorted) == len(escort_tasks) and bool(escort_tasks),
        "该拒的每轮都没蹦出结论": len(refused_ok) == len(refuse_tasks),
        "越权探测无一泄漏": not any(c["leaked_above_clearance"] for c in leak_checks),
        "生产无「已生效但未复核」的版本（复核闸在位）": env["effective_unreviewed_versions"] == 0,
        "云端兜底关闭（数据不出域）": env["cloud_fallback_enabled"] is False,
    }
    failed = [k for k, v in checks.items() if v is not True]

    # —— 交叉印证（不是装饰）：本轮走 HTTP+角色 这条全新通路跑出来的「没答上」集合，
    #     应当与人判标签归因里的「检索漏召」是同一批题。对得上说明这条路没有自己在制造新的失败，
    #     对不上就是要查的事。取数来自 answer_correct_calibration.json，不手写题号。
    cross_check = {"对照源": "var/gate0/answer_correct_calibration.json"}
    cal_path = ART / "answer_correct_calibration.json"
    unsolved = sorted({t["qid"] for t in turns
                       if t["kind"] == "首问" and t["桶"] == "作答" and t["status"] not in ANSWERABLE}
                      - {t["qid"] for t in turns if t["kind"] == "追问" and t["status"] in ANSWERABLE})
    cross_check["本轮未答上"] = unsolved
    if cal_path.exists():
        causes = json.loads(cal_path.read_text(encoding="utf-8")).get("wrong_row_causes") or {}
        miss = sorted(k for k, v in causes.items() if "检索" in str(v))
        cross_check["人判归因的检索漏召"] = miss
        cross_check["两集合是否一致"] = (set(unsolved) <= set(miss)) if miss else None
        cross_check["读法"] = ("未答上的题**全部落在**人判归因已登记的检索漏召里 ⇒ 这条新通路没在制造新的失败类型"
                              if set(unsolved) <= set(miss) else
                              "出现人判里没有的失败题 ⇒ 要单独查，别当成同一件事")
    else:
        cross_check["读法"] = "对照产物缺失，本轮未做交叉印证"

    rep = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "why": "用户裁决：真实用户可用测试用户替代 ⇒ 业务格从「不可测」改判为「按模拟口径可测」，"
               "但替代只覆盖「操作者是人」这一维，不覆盖上线前基线与时间维度。",
        "preflight": env, "cache_flushed_keys": flushed, "warm_up": warm_up,
        "剧本来源": ("冒烟：SMOKE_QIDS 手挑 11 题 ⇒ **n 小且题是我选的，这条读数不出业务结论**"
                     if args.smoke else "评测集全量自动派生（题与角色都不由脚本作者挑）"),
        "protocol": {"entrypoint": env["entrypoint"], "users": USERS, "plan": plan,
                     "期望处置怎么定": "不由剧本硬写，由 `role_expectation` 按「这个角色可见几条 gold」推导"
                                   "（第一版硬写过，把 newcomer 被正确拦住的题算成失败）",
                     "追问行为模型": "首轮没给可用答案才重问（换措辞），**不是**真实用户的重问率",
                     "sessions": "一个任务的多轮共用一个 session_id（指代要靠历史）",
                     "生产语料": "全程只读：本脚本不上传、不改版本（线上无删除端点，写进去撤不回）"},
        "turns": [{k: v for k, v in t.items() if k != "resp"} for t in turns],
        "metrics": {
            "模拟自助解答率": {"n_tasks": n_task, "value": self_serve,
                             "小样本警告": None if n_task >= 20 else f"n={n_task} 太小，不足以支撑「达标」二字",
                             "reading": f"{len(solved_first | rescue)}/{n_task}",
                             "门禁": f"≥ {GATES['模拟自助解答率']:.0%}",
                             "分母怎么来": "只算「这个角色看得见 gold」的作答任务；被权限拦掉的不进分母",
                             "定义": "该答任务里最终拿到可用答案（answer/tentative）的比例（含追问后拿到）"},
            "一次通过率": {"n_tasks": n_task, "value": first_pass, "reading": f"{len(solved_first)}/{n_task}",
                         "定义": "首轮就拿到可用答案——这才是「重复提问占比」能替代的那件事"},
            "追问占比（模拟）": {"value": reask_share, "n_reask": len(reask), "n_turns": len(turns),
                             "边界": "**由剧本的行为模型决定，不是真实用户行为率**；换剧本就换数，不得对外称「重复提问率」"},
            "权限拦截正确率": {"value": round(len(blocked_ok) / len(block_tasks), 4) if block_tasks else None,
                          "reading": f"{len(blocked_ok)}/{len(block_tasks)}",
                          "定义": "看不见 gold 的题：既没给结论也没引用 = 拦住；给了结论就是越权"},
            "转人工触发率": {"value": round(len(escorted) / len(escort_tasks), 4) if escort_tasks else None,
                        "reading": f"{len(escorted)}/{len(escort_tasks)}",
                        "边界": "只在**该角色看得见密级 3 证据**的任务上测；低权限角色问同一题走的是拦截不是转人工"},
            "该拒的没蹦结论": {"reading": f"{len(refused_ok)}/{len(refuse_tasks)}"},
            "status 分布": {k: len([t for t in turns if t["status"] == k])
                       for k in ("answer", "tentative", "refuse", "escalate")},
            "检索静默降级轮数": {"value": len(degraded_turns),
                           "题": [t["qid"] for t in degraded_turns],
                           "原因": sorted({x for t in degraded_turns for x in t["retrieval_degraded"]}),
                           "读法": "这些轮次是**只用 BM25 腿**答出来的；>0 就别把本轮当成完整 hybrid 的成绩"
                                   "（warm-up 已排掉冷启动那一呼，剩下的要单独查）"},
        },
        "not_measurable": {
            "重复提问占比「下降 ≥30%」": "要**上线前**的基线（没有系统时用户问几次）。测试用户给不出「没有系统时」的行为，"
                                   "所以只能报上面的绝对值，不能报下降幅度。",
            "新人上手周期缩短 ≥30%": "要真人 + 时间维度（从入职到独立使用的天数）。"
                              "脚本化的新人只能测「任务能不能做完」，测不出「花了多久上手」。",
        },
        "cross_check": cross_check,
        "rbac_leak_checks": leak_checks, "sse": sse,
        "db_checks": {"session_turn_rows": st["n"], "with_total_ms": st["with_ms"], "audit_rows_for_actors": au["n"]},
        "link_checks": checks, "link_checks_failed": failed,
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "m5_test_user.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# M5 · 测试用户运行验收", "",
         f"> 生成时点：{rep['generated_at']}｜`scripts/m5_test_user_run.py` 走生产入口 `{env['entrypoint']}` 真跑。",
         "> **这张表回答两件事**：① 系统现在**能不能正常运行**；② 用测试用户替代真实用户之后，"
         "业务三条里**哪条真的能报数、哪条仍然是拿不到**。第 ② 件比第 ① 件更容易说谎。", "",
         "## 一、能不能正常运行（逐条取数，不为真不写行）", ""]
    for k, v in checks.items():
        L.append(f"- {'✅' if v else '❌'} {k}")
    L += ["", f"链路检查失败项：{failed or '无'}", "",
          f"- 本轮 {len(turns)} 轮 / {len(plan)} 个任务；`session_turn` 落 {st['n']} 行、其中带 total_ms {st['with_ms']} 行；"
          f"审计落 {au['n']} 行；SSE 首个可读字 {sse.get('first_visible_ms')}ms", "",
          "## 二、模拟口径的业务读数", "",
          "| 指标 | 值 | 分母 | 定义与边界 |", "|---|---|---|---|"]
    m = rep["metrics"]
    L += [f"| 模拟自助解答率 | **{m['模拟自助解答率']['reading']} = {m['模拟自助解答率']['value']}** "
          f"（门禁 ≥60%） | n={m['模拟自助解答率']['n_tasks']} 任务 | {m['模拟自助解答率']['定义']}；"
          f"{m['模拟自助解答率']['分母怎么来']} |",
          f"| 一次通过率 | **{m['一次通过率']['reading']} = {m['一次通过率']['value']}** "
          f"| n={m['一次通过率']['n_tasks']} 任务 | {m['一次通过率']['定义']} |",
          f"| 追问占比（模拟） | {m['追问占比（模拟）']['value']} "
          f"| {m['追问占比（模拟）']['n_reask']}/{m['追问占比（模拟）']['n_turns']} 轮 "
          f"| ⚠ {m['追问占比（模拟）']['边界']} |",
          f"| 权限拦截正确率 | {m['权限拦截正确率']['reading']} = {m['权限拦截正确率']['value']} "
          f"| {m['权限拦截正确率']['reading']} | {m['权限拦截正确率']['定义']} |",
          f"| 转人工触发率 | {m['转人工触发率']['reading']} = {m['转人工触发率']['value']} "
          f"| — | ⚠ {m['转人工触发率']['边界']} |",
          f"| 检索静默降级轮数 | **{m['检索静默降级轮数']['value']}** "
          f"| 原因 `{m['检索静默降级轮数']['原因'] or '（无）'}` | {m['检索静默降级轮数']['读法']} |", "",
          "## 三、测试用户**替代不了**的两条（如实标不可测，别混进上面那张表）", ""]
    for k, v in rep["not_measurable"].items():
        L.append(f"- **{k}**：{v}")
    L += ["", "## 四、逐轮读数", "",
          "| 用户 | 题 | 轮 | 评测集期望 | 该角色的桶 | status | 结论字数 | 引用 | 置信 | 客户端 ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in turns:
        L.append(f"| {t['user']} | {t['qid']} | {t['kind']} | {t['评测集期望']} | {t['桶']} | {t['status']} "
                 f"| {t['conclusion_chars']} | {t['n_citations']} | {t['confidence']} | {t['client_ms']} |")
    L += ["", "## 五、与其他证据的交叉印证", "",
          f"- 本轮走 HTTP + 真实角色这条**全新通路**没答上的题：`{cross_check['本轮未答上']}`",
          f"- 人判标签归因里登记的检索漏召：`{cross_check.get('人判归因的检索漏召', '（对照产物缺失）')}`",
          f"- 读法：{cross_check['读法']}", "",
          "## 六、越权探测（低权限角色问密级 3 的题）", ""]
    for c in leak_checks:
        L.append(f"- {c['qid']} 由 `{c['asked_by']}` 问：status={c['status']}，引用 {c['n_citations']} 条、"
                 f"密级 {c['cited_security_levels'] or '（无引用）'} ⇒ "
                 f"{'**未越权**' if not c['leaked_above_clearance'] else '**越权，必须查**'}"
                 f"{'（⚠ ' + c['note'] + '）' if c.get('note') else ''}")
    L += ["", f"> warm-up 那一呼自己：status={warm_up['status']}、{warm_up['client_ms']}ms、"
          f"degraded=`{warm_up['retrieval_degraded'] or '（无）'}` —— "
          "它**不计入**上面的任何分母。", "", "> 生产语料全程只读：本脚本不上传、不改版本（线上没有删除文档端点，写进去撤不回）。"
          "写路径的可用性由 `scripts/m4_kb_write_path.py` 在**独立库+集合**上证明，不在这里。", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n" + json.dumps({"链路检查": {k: v for k, v in checks.items()},
                             "读数": {k: (v.get("value") if isinstance(v, dict) else v)
                                      for k, v in rep["metrics"].items() if k != "拒答与转人工的形状"}},
                            ensure_ascii=False, indent=2))
    print("产物：", OUT_MD)
    return 4 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
