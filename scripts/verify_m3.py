"""M3 端到端验证：问答端点、会话轮次、幂等重放、RBAC、审计与流式。

跑在生产集合上（与 M1 验证同一口径：对账是「一份权威源 ↔ 一个索引」），
但只读不写文档，所以不需要 purge/还原。

**这一版刻意不测「多轮指代消解」**：当前 answer() 不吃历史，改写器也只看着当轮问题，
所以「那汞呢」这种追问不会被解析成上一轮的主题。那是 M3-05 接 LangGraph Checkpointer
之后才存在的能力，在这里断言等于用一个假通过把它记成已完成。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from kee.api.app import create_app  # noqa: E402
from kee.config import get_settings  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402

OWNER = {"X-Kee-Role": "knowledge_owner", "X-Kee-Actor": "m3-owner-01"}
NEW = {"X-Kee-Role": "newcomer", "X-Kee-Actor": "m3-new-01"}

evidence: dict = {"steps": []}


def step(name: str, ok: bool, detail) -> None:
    evidence["steps"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'✅' if ok else '❌'} {name}: {json.dumps(detail, ensure_ascii=False, default=str)[:280]}")
    if not ok:
        raise SystemExit(f"步骤失败：{name} → {detail}")


def main() -> int:
    s = get_settings()
    pg = AuthoritativeStore()
    t_all = time.perf_counter()
    client = TestClient(create_app())

    # ---- 1. 健康与依赖 ----
    h = client.get("/healthz").json()
    step("依赖就绪（答案链路需向量库 + 权威源）",
         h["status"] == "ok" and h["deps"]["milvus"]["ok"] and h["deps"]["postgres"]["ok"],
         {"milvus_chunks": h["deps"]["milvus"].get("chunks"), "documents": h["deps"]["postgres"].get("documents")})

    # ---- 2. 单轮问答：结构化答案 + 引用 + 置信度 ----
    q1 = "配电自动化终端的遥测数据上送覆盖哪些测点，一共多少个？"
    t0 = time.perf_counter()
    r1 = client.post("/api/v1/answers", json={"question": q1}, headers=OWNER)
    b1 = r1.json()
    step(
        "单轮问答返回结构化答案（结论 + 要点 + 出处 + 置信度 + 免责位）",
        r1.status_code == 200
        and b1["status"] in ("answer", "tentative")
        and b1["conclusion"].strip() != ""
        and len(b1["key_points"]) >= 1
        and len(b1["citations"]) >= 1
        and "composite" in b1["confidence"],
        {
            "status": b1["status"],
            "结论": b1["conclusion"][:60],
            "要点数": len(b1["key_points"]),
            "引用数": len(b1["citations"]),
            "首个引用字段": sorted(b1["citations"][0].keys())[:9],
            "置信度": {k: b1["confidence"][k] for k in list(b1["confidence"])[:5]},
            "session_id": b1["session_id"],
            "turn_no": b1["turn_no"],
            "new_session": b1["new_session"],
            "耗时s": round(time.perf_counter() - t0, 2),
        },
    )
    sid = b1["session_id"]

    # ---- 3. 引用可溯源：每条引用都要带 doc/章节/页 ----
    bad = [c for c in b1["citations"] if not c.get("doc_id") or c.get("page") is None or not c.get("section_no")]
    step("每条引用都带 doc_id / section_no / page（无来源不出结论的可核对面）", not bad,
         {"引用数": len(b1["citations"]), "缺字段数": len(bad), "样例": b1["citations"][0].get("doc_id")})

    # ---- 4. 会话轮次累积 ----
    q2 = "治理前后的采集成功率分别是多少？"
    r2 = client.post("/api/v1/answers", json={"question": q2, "session_id": sid}, headers=OWNER)
    b2 = r2.json()
    step("同 session 第二轮：turn_no 递增且复用同一会话",
         r2.status_code == 200 and b2["turn_no"] == 2 and b2["session_id"] == sid and b2["new_session"] is False,
         {"turn_no": b2["turn_no"], "new_session": b2["new_session"], "status": b2["status"]})

    hist = client.get(f"/api/v1/sessions/{sid}", headers=OWNER).json()
    step("会话历史端点按轮次序返回两问及其状态",
         len(hist["turns"]) == 2
         and [t["turn_no"] for t in hist["turns"]] == [1, 2]
         and hist["turns"][0]["question"] == q1
         and hist["turns"][1]["question"] == q2,
         {"轮次": [(t["turn_no"], t["status"], t["conclusion"][:26]) for t in hist["turns"]]})

    # ---- 5. 幂等：同 Idempotency-Key 重放不重复产生轮次 ----
    key = f"idem-m3-{int(time.time())}"
    ra = client.post("/api/v1/answers", json={"question": "心跳超时阈值是多少？", "session_id": sid},
                     headers={**OWNER, "Idempotency-Key": key}).json()
    rb = client.post("/api/v1/answers", json={"question": "心跳超时阈值是多少？", "session_id": sid},
                     headers={**OWNER, "Idempotency-Key": key}).json()
    hist2 = client.get(f"/api/v1/sessions/{sid}", headers=OWNER).json()
    step("Idempotency-Key 重放返回首次结果，且不额外增加轮次",
         rb.get("replayed") is True and ra["turn_no"] == rb["turn_no"] and len(hist2["turns"]) == 3,
         {"首次turn": ra["turn_no"], "重放turn": rb["turn_no"], "会话轮数": len(hist2["turns"])})

    # ---- 6. RBAC：同一问题，Owner 能引到密级 2，新人被过滤到无据可引 ----
    # 这一条原来写成「新人引用密级 <= 1」，而那次恰好是 refuse 零引用，断言空转就过了——
    # 那是假绿。现在先确认 Owner 侧真的引到了 >1 的证据（探针有效性），
    # 不成立就直接判失败，因为「双方都拒答」证明不了任何过滤在起作用。
    rq = "生产控制大区与信息管理大区之间的数据交换必须经过什么装置？"
    row = client.post("/api/v1/answers", json={"question": rq}, headers=OWNER).json()
    rnew = client.post("/api/v1/answers", json={"question": rq}, headers=NEW).json()
    owner_levels = sorted({int(c.get("security_level") or 0) for c in row.get("citations", [])})
    new_levels = sorted({int(c.get("security_level") or 0) for c in rnew.get("citations", [])})
    probe_valid = bool(owner_levels) and max(owner_levels) >= 2
    step(
        "检索层 RBAC 在答案链上生效（同题 Owner 引到密级 2，新人被过滤到无据可引）",
        probe_valid and all(l <= 1 for l in new_levels) and rnew["status"] in ("refuse", "escalate"),
        {
            "探针有效性(Owner 侧须引到 >1)": probe_valid,
            "owner": {"status": row["status"], "引用密级": owner_levels, "结论": row["conclusion"][:34]},
            "newcomer": {"status": rnew["status"], "引用密级": new_levels, "needs_human": rnew["needs_human"]},
        },
    )

    # ---- 7. 拒答与转人工是确定性的，不是空字符串糊过去 ----
    rd = client.post("/api/v1/answers", json={"question": "今年食堂供应商的中标名单是什么？"}, headers=OWNER).json()
    step("语料外问题被拒答且结论为空（空结论不得判成功）",
         rd["status"] in ("refuse", "escalate", "degraded") and rd["conclusion"].strip() == "",
         {"status": rd["status"], "needs_human": rd["needs_human"], "notes": rd["notes"],
          "结论长度": len(rd["conclusion"].strip())})

    # ---- 8. 流式端点出 SSE 事件 ----
    with client.stream("POST", "/api/v1/answers/stream",
                       json={"question": "银企直联接口编号规则是什么？", "session_id": sid}, headers=OWNER) as st:
        ctype = st.headers.get("content-type", "")
        lines = [ln for ln in st.iter_text() if ln.strip()]
    got_event = any("event: answer" in ln for ln in lines)
    step("流式端点以 SSE 返回 answer 事件（阶段级，非 token 级；理由见代码注释）",
         "text/event-stream" in ctype and got_event,
         {"content-type": ctype, "事件数": len([l for l in lines if l.startswith('event:')])})

    # ---- 9. 审计留痕 ----
    with pg.connect() as conn:
        audits = conn.execute(
            "SELECT action, count(*) n FROM audit_log WHERE actor IN ('m3-owner-01','m3-new-01') GROUP BY action"
        ).fetchall()
    acts = {a["action"]: a["n"] for a in audits}
    step("答案链每次问答与每次会话读取都进审计",
         acts.get("answer_generated", 0) >= 5 and acts.get("session_read", 0) >= 2,
         acts)

    evidence["timings_sec"] = {"whole_run": round(time.perf_counter() - t_all, 2)}
    art = Path(s.resolved_artifact_dir()) / "m3_verify.json"
    art.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n全部步骤通过。原始证据：", art)
    print("未覆盖（如实记账）：多轮指代消解——当前 answer() 不吃历史，需 M3-05 的 Checkpointer 才成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
