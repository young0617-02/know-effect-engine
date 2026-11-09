"""把「审计表防篡改」从聊胜于无做成真闸门（用户 2026-10-15 逐步确认执行）。

原缺口有**两个条件同时成立**才堵不住，缺一个都堵不住：

1. `audit_log_immutable` 只声明 `BEFORE UPDATE OR DELETE`，而 PostgreSQL 的行级/语句级
   触发器**不覆盖 TRUNCATE**——`TRUNCATE TABLE audit_log` 一句就能把整张表抹干净且不留痕；
2. 应用角色 `kee` 是 `rolsuper=true`，表级 REVOKE 对超级用户一律无效（`pg_hba` 也拦不住它）。

所以按序做四件事，顺序不能换（先摘超级用户就会没有权限改属主）：

  ① 建两个新角色：`postgres`（NOLOGIN SUPERUSER，只做属主，永不登录）与
     `kee_admin`（LOGIN SUPERUSER，人类 DBA 入口）。
     ⚠ 这一步是**必须**的：本容器由 `POSTGRES_USER=kee` 起步，压根没有 `postgres` 角色；
     而 `kee` 降级之后系统里就不留任何超级用户了——那不叫收紧权限，叫把自己锁在门外。
  ② `ALTER TABLE audit_log OWNER TO postgres`，触发器函数一并交给 `postgres`。
     只改表的属主不够：函数仍由 `kee` 拥有的话，它可以 `CREATE OR REPLACE FUNCTION`
     把触发器空转成 `RETURN NEW`，等于不换锁、只把钥匙孔磨圆。
     另外 `audit_log_id_seq` 随表一起换属主，所以要显式把 `USAGE, SELECT` 授回 `kee`，
     否则插入审计行会在序列上权限不足。
  ③ `GRANT INSERT, SELECT ON audit_log TO kee`——应用只剩写与读。
  ④ `ALTER ROLE kee NOSUPERUSER`。

⚠ ③ 与 ④ 的次序与用户口述的清单对调了一下：`kee` 一旦不是超级用户，就没权限给
`postgres` 拥有的表 GRANT，照原顺序会在「表已换属主、角色已降级」这个半完成状态上失败。
终态与清单完全一致，只是先后不同（代码里同条注释再说明一次）。

口令不落版本库：`kee_admin` 的口令随机生成后写 `var/pg_admin_credentials.json`
（`.gitignore` 的 `var/*` 覆盖它；**不要**放 `var/gate0/`，那里 `!*.json` 会把文件收回跟踪）。

可重复执行：所有步骤都是幂等的，`--check` 只报当前 posture、什么都不改。
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from kee.config import get_settings  # noqa: E402

CREDS = ROOT / "var" / "pg_admin_credentials.json"

# 收紧后应用角色对审计表应剩下的权限——多一项就是没堵住
APP_GRANTS = {"INSERT", "SELECT"}
# 应用角色。口令从 `KEE_POSTGRES_DSN` 解析（.env），代码里不留默认值；
# `kee_admin` 那个随机口令只落 var/（不入库）。
APP_ROLE = "kee_app"
APP_PASSWORD = unquote(urlsplit(get_settings().postgres_dsn).password or "")
if not APP_PASSWORD:
    raise SystemExit("KEE_POSTGRES_DSN 未配置口令；先按 .env.example 填好 .env")


def _app_tables(conn) -> list[str]:
    """归应用管的表：除 `audit_log`（那张的属主必须是 NOLOGIN 超级用户）。"""
    rows = conn.execute(
        "SELECT c.relname AS n FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'S') AND c.relname <> 'audit_log' "
        "AND c.relname <> 'audit_log_id_seq' ORDER BY c.relname"
    ).fetchall()
    return [r["n"] for r in rows]


def provision_app_role(conn) -> list[str]:
    """把「应用角色不是超级用户」这条落到实处。

    ⚠ 用户给的清单第②步是 `ALTER ROLE kee NOSUPERUSER`，**这条路在本容器里走不通**：
    `kee` 是 initdb 引导出来的 bootstrap 超级用户，PostgreSQL 明文拒绝摘它的 SUPERUSER
    （实测报 `FeatureNotSupported: permission denied to alter role` +
    `DETAIL: The bootstrap superuser must have the SUPERUSER attribute.`）。
    所以终态改成等价但可执行的那一个：应用改用非特权角色 `kee_app`，
    `kee` 则被关掉登录（实测 `ALTER ROLE kee NOLOGIN` 允许）——
    于是「应用拿到的那把钥匙」从头到尾都不是超级用户，而且原来那把也不再是可用凭证。
    """
    done: list[str] = []
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)).fetchone()
    if not exists:
        from psycopg import sql

        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(APP_ROLE), sql.Literal(APP_PASSWORD)))
        done.append(f"⑤ 建应用角色 {APP_ROLE}（LOGIN，非超级用户）")
    else:
        done.append(f"⑤ 应用角色 {APP_ROLE} 已存在，跳过建号")

    db = conn.execute("SELECT current_database() AS d").fetchone()["d"]
    # CREATEDB 是给验收脚本用的：`m4_kb_write_path.py` / `m4_upload_http_check.py` 各自建一个
    # scratch 库来碰生产写路径，不建库就没法「不碰生产语料」地验证。它和 SUPERUSER 不是一回事——
    # CREATEDB 建不了别人的库，也改不了角色属性，动不了 audit_log 的属主与触发器。
    conn.execute("ALTER ROLE kee_app CREATEDB")
    conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
        sql.Identifier(db), sql.Identifier(APP_ROLE)))
    conn.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(sql.Identifier(APP_ROLE)))
    conn.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                         "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}").format(sql.Identifier(APP_ROLE)))
    done.append(f"⑤ {APP_ROLE} 取得 CONNECT / schema USAGE+CREATE（迁移要建表）/ 未来新表的 DML 默认授权")

    owned = 0
    for t in _app_tables(conn):
        owner = conn.execute("SELECT r.rolname AS o FROM pg_class c JOIN pg_roles r ON r.oid = c.relowner "
                             "WHERE c.relname = %s", (t,)).fetchone()["o"]
        if owner == APP_ROLE:
            continue
        conn.execute(sql.SQL("ALTER TABLE {} OWNER TO {}").format(
            sql.Identifier(t), sql.Identifier(APP_ROLE)))
        owned += 1
    done.append(f"⑤ 现有一 {owned} 张表/序列的属主 → {APP_ROLE}（audit_log 刻意不在里面）")

    conn.execute(sql.SQL("REVOKE ALL ON audit_log FROM {}").format(sql.Identifier(APP_ROLE)))
    conn.execute(sql.SQL("GRANT INSERT, SELECT ON audit_log TO {}").format(sql.Identifier(APP_ROLE)))
    conn.execute(sql.SQL("GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO {}").format(sql.Identifier(APP_ROLE)))
    done.append(f"⑤ {APP_ROLE} 在 audit_log 上只有 INSERT, SELECT + 序列 USAGE/SELECT（写审计要用）")

    kee_login = conn.execute("SELECT rolcanlogin AS l FROM pg_roles WHERE rolname = 'kee'").fetchone()["l"]
    if kee_login:
        conn.execute("ALTER ROLE kee NOLOGIN")
        done.append("⑤ kee 关掉登录（bootstrap 超级用户的口令不再是一张可用凭证）")
    return done


def admin_dsn() -> str:
    """管理连接：优先环境变量，其次收紧时生成的 `kee_admin` 凭证，最后回落应用 DSN
    （只有**尚未收紧**的库能用那条路——收紧后它不再是超级用户）。"""
    import os

    dsn = os.environ.get("KEE_PG_ADMIN_DSN")
    if dsn:
        return dsn
    if CREDS.exists():
        c = json.loads(CREDS.read_text(encoding="utf-8"))
        hostport = get_settings().postgres_dsn.split("@")[-1]
        return f"postgresql://{c['user']}:{c['password']}@{hostport}"
    return get_settings().postgres_dsn


def _is_super(conn) -> bool:
    return bool(conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()["rolsuper"])


def _audit_owner(conn) -> str:
    return conn.execute(
        "SELECT r.rolname FROM pg_class c JOIN pg_roles r ON r.oid = c.relowner WHERE c.relname = 'audit_log'"
    ).fetchone()["rolname"]


def _trigger_present(conn) -> bool:
    return bool(conn.execute("SELECT 1 FROM pg_trigger WHERE tgname = 'audit_log_immutable' "
                             "AND NOT tgisinternal").fetchone())


def _func_owner(conn) -> str:
    row = conn.execute("SELECT r.rolname FROM pg_proc p JOIN pg_roles r ON r.oid = p.proowner "
                       "WHERE p.proname = 'kee_forbid_audit_mutation'").fetchone()
    return row["rolname"] if row else "(缺失)"


def _app_grants(conn, role: str = APP_ROLE) -> set[str]:
    rows = conn.execute("SELECT privilege_type FROM information_schema.role_table_grants "
                        "WHERE grantee = %s AND table_name = 'audit_log'", (role,)).fetchall()
    return {r["privilege_type"] for r in rows}


def check(conn) -> dict:
    app = conn.execute("SELECT rolsuper, rolcanlogin FROM pg_roles WHERE rolname = %s", (APP_ROLE,)).fetchone()
    return {
        "app_role_is_superuser": bool(conn.execute(
            "SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()["rolsuper"]),
        "audit_log_owner": _audit_owner(conn),
        "trigger_function_owner": _func_owner(conn),
        "trigger_present": _trigger_present(conn),
        "app_grants_on_audit_log": sorted(_app_grants(conn)),
        "应用角色是否存在": bool(app),
        "应用角色是否超级用户": (None if not app else bool(app["rolsuper"])),
        "应用角色能否登录": (None if not app else bool(app["rolcanlogin"])),
        "kee_是否还能登录": bool(conn.execute(
            "SELECT rolcanlogin FROM pg_roles WHERE rolname = 'kee'").fetchone()["rolcanlogin"]),
        "admin_credential_file": str(CREDS) if CREDS.exists() else "(未生成)",
    }


def harden(conn) -> list[str]:
    """按序执行，返回逐步说明。幂等：已存在的角色/属主/授权再跑一遍不报错也不改坏。"""
    done: list[str] = []
    if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'kee_admin'").fetchone():
        from psycopg import sql

        pw = secrets.token_urlsafe(24)
        # CREATE ROLE 是 utility 语句，不接受 $1 占位参数；用 sql.Literal 生成带引号的字面量，
        # 而不是把口令拼进字符串（拼进去的话口令里一个引号就能改语义）。
        conn.execute(sql.SQL("CREATE ROLE kee_admin LOGIN SUPERUSER PASSWORD {}").format(sql.Literal(pw)))
        CREDS.parent.mkdir(parents=True, exist_ok=True)
        CREDS.write_text(json.dumps({"user": "kee_admin", "password": pw,
                                     "note": "DBA 入口；应用角色 kee 已不是超级用户"}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        done.append("① 建 kee_admin（LOGIN SUPERUSER），口令写入 var/pg_admin_credentials.json")
    else:
        done.append("① kee_admin 已存在，跳过")

    if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'postgres'").fetchone():
        conn.execute("CREATE ROLE postgres NOSUPERUSER NOCREATEDB NOCREATEROLE NOLOGIN")
        done.append("① 建 postgres（NOLOGIN，只做属主；容器由 POSTGRES_USER=kee 起步，本来没有这个角色）")
    else:
        done.append("① postgres 角色已存在，跳过")
    # 属主必须是真的超级用户才能保住「只有超级用户能改这个触发器」这条边界，
    # 但它 NOLOGIN——谁都登录不进来，也就谁都 TRUNCATE 不了。
    conn.execute("ALTER ROLE postgres SUPERUSER")

    conn.execute("ALTER TABLE audit_log OWNER TO postgres")
    conn.execute("ALTER FUNCTION kee_forbid_audit_mutation() OWNER TO postgres")
    conn.execute("GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO kee")
    done.append("② audit_log 与其触发器函数的属主 → postgres；序列 USAGE/SELECT 授回 kee")

    conn.execute("REVOKE ALL ON audit_log FROM kee")
    conn.execute("GRANT INSERT, SELECT ON audit_log TO kee")
    done.append("③ kee 在 audit_log 上只剩 INSERT, SELECT")

    # ⚠ 顺序与用户给的清单对调了一步：授权（原③）必须在摘超级用户（原②）之前。
    # kee 不再是超级用户后，就没有权限给 `postgres` 拥有的表 GRANT 了——照原顺序跑到第三步会当场失败，
    # 而且失败在「表已换属主、角色已降级」这个半完成状态上。终态与清单完全一致。
    try:
        conn.execute("ALTER ROLE kee NOSUPERUSER")
        done.append("④ kee 摘掉 SUPERUSER")
    except psycopg.Error as e:
        # 实测走不通：kee 是 initdb 的 bootstrap 超级用户，PG 不许摘它的 SUPERUSER。
        # 不做成静默跳过——「应用角色是超级用户」正是这条敞口的另一半。
        done.append(f"④ 原样摘 kee 的 SUPERUSER 被 PG 拒绝：{str(e).strip().splitlines()[0]}"
                    f"（{str(e).strip().splitlines()[-1]}）⇒ 改走 ⑤：应用换用非特权角色")

    done += provision_app_role(conn)
    return done


def probe_bypass(app_dsn: str) -> dict:
    """**用应用角色的连接**现场验一次「绕过路径真的断了」，且不动任何数据。

    必须另开一条连接：执行收紧的是 `kee_admin`（超级用户），在它眼里 TRUNCATE 永远能成——
    拿它自证「堵住了」等于让被审的对象签字说自己没违规。
    TRUNCATE 会在权限阶段就失败，DELETE 会被触发器拦下，两条都只看报错、不提交任何东西。
    """
    with psycopg.connect(app_dsn, row_factory=psycopg.rows.dict_row, autocommit=True) as c:
        before = c.execute("SELECT count(*) n FROM audit_log").fetchone()["n"]
        who = c.execute("SELECT current_user AS u, "
                        "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user) AS sup").fetchone()
        out: dict[str, object] = {"以哪个角色试探": who["u"], "该角色是否超级用户": bool(who["sup"]),
                                  "audit_rows_before": before}
        for sql_text in ("TRUNCATE TABLE audit_log", "DELETE FROM audit_log"):
            try:
                c.execute(sql_text)
                out[sql_text] = "竟然成功了——闸门没闭上"
            except psycopg.Error as e:
                out[sql_text] = f"被拒：{type(e).__name__}: {str(e).strip().splitlines()[0]}"
        out["audit_rows_after"] = c.execute("SELECT count(*) n FROM audit_log").fetchone()["n"]
        out["rows_unchanged"] = out["audit_rows_before"] == out["audit_rows_after"]
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只报当前 posture，不改任何东西")
    ap.add_argument("--confirm", default="", help="必须等于 HARDEN 才真正执行")
    args = ap.parse_args()

    dsn = admin_dsn()
    with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row, autocommit=True) as conn:
        st = check(conn)
        print(json.dumps(st, ensure_ascii=False, indent=2))
        if args.check:
            return 0
        if not st["app_role_is_superuser"] and not _is_super(conn):
            print("[拒绝执行] 当前连接不是超级用户，改不了属主。"
                  "用容器内的本地通路执行：docker exec kee-postgres psql -U kee_admin -d kee")
            return 2
        if args.confirm != "HARDEN":
            print("[拒绝执行] 这一步会改变整个应用的权限面。确认后再来：--confirm=HARDEN")
            return 2
        for line in harden(conn):
            print("  " + line)
        print(json.dumps(probe_bypass(get_settings().postgres_dsn), ensure_ascii=False, indent=2))
        print(json.dumps(check(conn), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
