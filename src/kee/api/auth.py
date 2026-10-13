"""账号与登录态：口令散列、可撤销会话、注册时的权限约束。

三条设计决定要说明白，它们都不是风格问题：

1. **口令只存 scrypt 散列 + 每用户独立盐**，不存明文也不存可逆加密。stdlib 自带，
   不为此引第三方依赖（本机原则：大件与新依赖都不随手加）。
2. **注册不能自授 `knowledge_owner` / `system_admin`**。否则任何人都能注册一个能读全库密级、
   能改复核状态的管理员账号——那是权限提升，不是登录功能。特权角色必须由**已有**管理员
   通过 `/auth/users/{name}/role` 授予，且这一步本身进审计。
3. **会话是服务端可撤销的**：界面拿到的是一枚随机令牌，库里只存它的 SHA-256 摘要与过期时间。
   退出即删行；只把令牌从 localStorage 里抹掉不叫退出。
"""

from __future__ import annotations

import hashlib
import re
import secrets
from contextvars import ContextVar
from typing import Any

# 当前请求解析出来的登录用户（由中间件写入）。用 contextvar 而不是把 request 一路传进
# 每个路由签名，是为了不改动既有 10 个端点的形参——改动面越小，漏改一处「某端点忘了判权限」的机会越小。
current_user: ContextVar[dict[str, Any] | None] = ContextVar("kee_current_user", default=None)

PRIVILEGED_ROLES = ("knowledge_owner", "system_admin")
# 注册时允许自选的角色：都是「按密级/行业被收窄」的普通账号，没有治理权
SELF_SERVICEABLE_ROLES = ("delivery_engineer", "presales_architect", "pm", "ops_support", "newcomer")

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
PASSWORD_MIN = 10

# scrypt 参数：n=2^14 让单次校验约几十毫秒、内存 16MB——在线登录够用，撞库成本也够高。
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _DKLEN = 2 ** 14, 8, 1, 32

# 建表语句在 `kee.db.pg.AUTH_SCHEMA_SQL`（DDL 归 DB 层，见那里的注释）。



def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                            p=_SCRYPT_P, dklen=_DKLEN)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """定长比较，且**解析失败一律算不匹配**——存储格式被人动过时，宁可拒绝登录也不能放行。"""
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        want = bytes.fromhex(hash_hex)
        got = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                             n=int(n), r=int(r), p=int(p), dklen=len(want))
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(got, want)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_registration(username: str, password: str, role: str, *, is_first_account: bool = False) -> str | None:
    """返回错误说明；None 表示通过。校验放在写库之前，错误信息要能直接给用户看。"""
    if not USERNAME_RE.match(username or ""):
        return "用户名需为 3–32 位小写字母、数字或 . _ -（HTTP 头与审计里要可原样比对，故不接受中文/空格）"
    if len(password or "") < PASSWORD_MIN:
        return f"口令至少 {PASSWORD_MIN} 位"
    if role not in SELF_SERVICEABLE_ROLES and not (is_first_account and role in PRIVILEGED_ROLES):
        return (f"注册只能选 {', '.join(SELF_SERVICEABLE_ROLES)}；"
                f"`{role}` 属治理角色，须由现有管理员授予")
    return None


def register(conn, *, username: str, password: str, role: str, display_name: str,
             industries: list[str] | None = None, projects: list[str] | None = None,
             is_first_account: bool = False) -> dict[str, Any]:
    err = validate_registration(username, password, role, is_first_account=is_first_account)
    if err:
        raise ValueError(err)
    if conn.execute("SELECT 1 FROM app_user WHERE username = %s", (username,)).fetchone():
        raise ValueError(f"用户名 {username} 已存在")
    row = conn.execute(
        "INSERT INTO app_user (username, display_name, pass_hash, role, industries, projects) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "RETURNING username, display_name, role, industries, projects, created_at",
        (username, display_name or username, hash_password(password), role,
         ",".join(industries or []), ",".join(projects or []))).fetchone()
    return dict(row)


def start_session(conn, *, username: str, ttl_hours: float) -> str:
    token = new_token()
    conn.execute("INSERT INTO user_session (token_hash, username, expires_at) "
                 "VALUES (%s, %s, now() + make_interval(hours => %s))",
                 (_hash_token(token), username, max(1, int(ttl_hours))))
    return token


def authenticate(conn, *, username: str, password: str, ttl_hours: float) -> tuple[str | None, str]:
    """成功返回 (token, "")；失败返回 (None, 原因)。

    失败原因**不区分「用户不存在」与「口令错」**——那等于把用户名枚举做成了免费功能。
    """
    row = conn.execute("SELECT * FROM app_user WHERE username = %s", (username,)).fetchone()
    if not row or not verify_password(password, row["pass_hash"]):
        if row:
            conn.execute("UPDATE app_user SET failed_count = failed_count + 1 WHERE username = %s",
                         (username,))
        return None, "用户名或口令不正确"
    if not row["is_active"]:
        return None, "该账号已被停用"
    conn.execute("UPDATE app_user SET last_login_at = now(), failed_count = 0 WHERE username = %s",
                 (username,))
    return start_session(conn, username=username, ttl_hours=ttl_hours), ""


def resolve(conn, token: str) -> dict[str, Any] | None:
    if not token:
        return None
    row = conn.execute(
        "SELECT u.username, u.display_name, u.role, u.industries, u.projects, u.is_active, s.expires_at "
        "FROM user_session s JOIN app_user u ON u.username = s.username "
        "WHERE s.token_hash = %s AND s.expires_at > now()", (_hash_token(token),)).fetchone()
    if not row or not row["is_active"]:
        return None
    return {
        "username": row["username"], "display_name": row["display_name"], "role": row["role"],
        "industries": [x for x in (row["industries"] or "").split(",") if x],
        "projects": [x for x in (row["projects"] or "").split(",") if x],
        "expires_at": row["expires_at"].isoformat(),
    }


def drop_session(conn, token: str) -> int:
    return conn.execute("DELETE FROM user_session WHERE token_hash = %s", (_hash_token(token),)).rowcount


def set_role(conn, *, actor: str, username: str, role: str, industries: list[str] | None = None,
             projects: list[str] | None = None) -> dict[str, Any]:
    """授予/调整角色——只有管理员路由会调它，且调用方负责写审计。"""
    if role not in SELF_SERVICEABLE_ROLES + PRIVILEGED_ROLES:
        raise ValueError(f"未知角色 {role}")
    row = conn.execute(
        "UPDATE app_user SET role = %s, industries = %s, projects = %s WHERE username = %s "
        "RETURNING username, display_name, role, industries, projects",
        (role, ",".join(industries or []), ",".join(projects or []), username)).fetchone()
    if not row:
        raise ValueError(f"用户名 {username} 不存在")
    return dict(row)


def bearer_of(header: str | None) -> str:
    """从 `Authorization: Bearer xxx` 取令牌；格式不对就返回空串，不猜。"""
    if not header:
        return ""
    parts = header.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else ""
