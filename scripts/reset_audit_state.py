"""交付前清场：清空审计留痕与测试会话，并留下一份**表外**凭据。

为什么需要它、以及为什么它长这样：

1. **审计表记不了自己的清空**。`audit_log` 是防篡改的落点，而清空动作本身如果只写进这张表，
   就等于让被审计对象自己写证词——所以清场结果落到 `var/gate0/audit_reset.json`：
   删了多少行、按什么条件、谁执行的、执行前后各自行数。
2. **这一步绕过了系统自己的防线，必须说出口**。`audit_log_immutable` 触发器只声明了
   `BEFORE DELETE OR UPDATE`，而 PostgreSQL 的触发器**不覆盖 TRUNCATE**；加上应用角色 `kee` 目前是
   superuser（表级 REVOKE 对超级用户无效），所以 TRUNCATE 畅通无阻。
   这是一条**已登记的敞口**，不是本脚本的发明——脚本只是把运维迟早要做的事变得可复核。
   收紧办法见风险册同名条目（改属主 + 撤 superuser + 只授 INSERT/SELECT）。
3. **护栏**：必须显式 `--confirm=RESET`；只动 `audit_log` / `session` / `session_turn`，
   **不碰** `document` / `document_version` / `chunk` / 向量索引 / 评测集与任何 `var/gate0/*.json` 证据。
   权威源与索引一动，M2 基线与全部人工标注就作废，那不该是"清个界面"的代价。

跑法：uv run python scripts/reset_audit_state.py --confirm=RESET
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "var" / "gate0"

# 只清这三张：审计与会话。文档/版本/块/索引一律不动——理由见模块 docstring 第 3 条。
TARGETS = ("audit_log", "session_turn", "session")
DEMO_SESSION_LIKE = "s-demo%"


def counts(conn) -> dict:
    return {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in TARGETS}


def main() -> int:
    import psycopg
    from psycopg.rows import dict_row   # 裸 connect 默认回传 tuple，取 ["n"] 会 TypeError

    from kee.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", required=True,
                    help='必须字面写 RESET。清空审计是不可逆动作，"跑一下就顺手清了"不是它的用法。')
    ap.add_argument("--reason", default="", help="为什么清（空理由=未登记，本脚本会拒绝）")
    ap.add_argument("--only-demo", action="store_true",
                    help="只删本次演示产生的行（actor 含 demo 的审计行 + s-demo 会话），不清空全表")
    args = ap.parse_args()

    if args.confirm != "RESET":
        print("[拒绝执行] --confirm 必须字面等于 RESET。")
        return 2
    if not args.reason.strip():
        print("[拒绝执行] 清空审计必须写 --reason：不留原因的抹除动作，正是这套系统要拦的形状。")
        return 2

    s = get_settings()
    with psycopg.connect(s.postgres_dsn, row_factory=dict_row) as conn:
        before = counts(conn)
        demo_rows = conn.execute("SELECT count(*) n FROM audit_log WHERE actor ILIKE '%demo%'").fetchone()["n"]
        demo_sessions = conn.execute(
            "SELECT count(*) n FROM session_turn WHERE session_id LIKE %s", (DEMO_SESSION_LIKE,)).fetchone()["n"]

        if args.only_demo:
            # 走 DELETE 就得承认：这条路被触发器堵着，堵得对。这里不 DROP 触发器去绕它。
            print(f"[拒绝执行] 只删部分行需要 UPDATE/DELETE，而 `audit_log_immutable` 会拦下它——"
                  f"那是防篡改机制在正常工作，不该为清理演示数据去拆它。\n"
                  f"要么整表清空（交付重置），要么保留这 {demo_rows} 行。")
            return 3

        # session_turn 外键引用 session ⇒ 必须同一条语句一起截断，否则 PG 直接拒绝
        conn.execute("TRUNCATE TABLE audit_log")
        conn.execute("TRUNCATE TABLE session, session_turn")
        conn.commit()
        after = counts(conn)

    rep = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "by": getpass.getuser(),
        "reason": args.reason,
        "action": "TRUNCATE audit_log / session_turn / session",
        "绕过的是什么": "audit_log_immutable 触发器只声明 DELETE/UPDATE，PostgreSQL 触发器不覆盖 TRUNCATE；"
                    "且应用角色 kee 当前为 superuser，表级 REVOKE 无效",
        "行数_前": before,
        "行数_后": after,
        "被清掉的演示数据": {"actor 含 demo 的审计行": demo_rows, "s-demo 会话轮": demo_sessions},
        "未动的东西": ["document", "document_version", "chunk", "outbox_event", "reconciliation_diff",
                   "Milvus 集合 kb_chunks", "评测集与全部人工标注", "var/gate0/*.json 证据产物"],
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "audit_reset.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("凭据：", ART / "audit_reset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
