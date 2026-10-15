"""验收脚本的身份通道：口令登录 + Bearer 令牌，不再用请求头冒充身份。

为什么要有这个文件（用户 2026-10-15 裁「交付物默认 `false`、脚本显式带令牌」）：
`KEE_ALLOW_HEADER_IDENTITY` 关掉之后，`X-Kee-Role: system_admin` 这种「发个头就拿到
该角色全部范围」的跑法当场失效。脚本要继续能跑，就得和真实用户走同一条入口——
`/api/v1/auth/login` 拿令牌。这个模块只做两件事：把账号准备好，把令牌换成请求头。

两个刻意的选择：

1. **账号是落库播种的，不是调注册端点的**。治理角色（`knowledge_owner` / `system_admin`）
   按规则不能自授，脚本若靠端点建号就得先有一个管理员口令——那是把鸡生蛋藏进代码里。
   真实部署也是这么起步的：种子管理员由部署方落库。注册端点那条路仍由
   `tests/test_auth.py` 覆盖，所以这里没有绕开被测行为。
2. **口令不落进版本库**。首次运行生成随机口令写 `var/verify_credentials.json`
   （`.gitignore` 的 `var/*` 已覆盖；注意 `!var/gate0/*.json` 会把 gate0 下的 json
   重新收进跟踪，所以凭证绝不能放那儿）。文件存在就复用，保证脚本可重复跑。
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CREDS = ROOT / "var" / "verify_credentials.json"

sys.path.insert(0, str(ROOT / "src"))

from kee.api.auth import hash_password  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402

# username -> (role, industries, projects)。名字固定，脚本可重复跑。
# **industries 一律留空**，除了 `verify-new-fin`：请求头身份那会儿各脚本没带 `X-Kee-Industries`，
# 而 `build_filter_expr` 对空 industries 是不加部门子句（rbac.py:59-61）——给它塞上「化工」
# 会把可见范围**收窄**，验证里的断言就会因为身份变化而红，不是因为它测的东西变了。
# industries 用中文原样入库：登录态的范围来自账号，不再来自请求头，所以不需要百分号编码。
USERS: dict[str, tuple[str, list[str], list[str]]] = {
    "verify-owner": ("knowledge_owner", [], []),
    "verify-admin": ("system_admin", [], []),
    "verify-eng": ("delivery_engineer", [], []),
    "verify-arch": ("presales_architect", [], []),
    "verify-pm": ("pm", [], []),
    "verify-ops": ("ops_support", [], []),
    "verify-new": ("newcomer", [], []),
    "verify-new-fin": ("newcomer", ["金融"], []),
    # 测试用户运行验收（`m5_test_user_run.py`）的三个虚拟用户。账号名**保持与产物里的
    # `user` 字段一致**：改成 verify-* 会让历史产物与本轮产物对不上号，而那份产物是 DoD#6
    # 的推导来源，不是可以随时改名的日志。
    "u-pm-01": ("pm", [], []),
    "u-delivery-01": ("delivery_engineer", [], []),
    "u-newcomer-01": ("newcomer", [], []),
}

# role -> 默认账号。给那些「只说角色、没有具体用户名」的调用点用（预热、SSE 形状检查）。
BY_ROLE: dict[str, str] = {
    "knowledge_owner": "verify-owner", "system_admin": "verify-admin",
    "delivery_engineer": "verify-eng", "presales_architect": "verify-arch",
    "pm": "verify-pm", "ops_support": "verify-ops", "newcomer": "verify-new",
}


def _load() -> dict[str, str]:
    if CREDS.exists():
        return json.loads(CREDS.read_text(encoding="utf-8"))
    return {}


def passwords() -> dict[str, str]:
    """已知账号的口令表；缺的当场生成并落盘（文件不入库）。"""
    saved = _load()
    missing = [u for u in USERS if u not in saved]
    if missing:
        saved.update({u: secrets.token_urlsafe(18) for u in missing})
        CREDS.parent.mkdir(parents=True, exist_ok=True)
        CREDS.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def ensure_accounts(pg: AuthoritativeStore | None = None) -> dict[str, str]:
    """把 `USERS` 里缺失的账号落库。已存在的一律不动口令——
    改了会让上一轮的令牌与下一轮的登录互相打不通。"""
    store = pg or AuthoritativeStore()
    pw = passwords()
    with store.connect() as conn:
        for username, (role, industries, projects) in USERS.items():
            if conn.execute("SELECT 1 FROM app_user WHERE username = %s", (username,)).fetchone():
                continue
            conn.execute(
                "INSERT INTO app_user (username, display_name, pass_hash, role, industries, projects) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (username, f"验收账号 {username}", hash_password(pw[username]), role,
                 ",".join(industries), ",".join(projects)),
            )
    return pw


def login_headers(client: Any, username: str) -> dict[str, str]:
    """TestClient 版：走真实登录路由换一枚 Bearer。"""
    pw = passwords().get(username) or ensure_accounts()[username]
    r = client.post("/api/v1/auth/login", json={"username": username, "password": pw})
    if r.status_code != 200:
        raise SystemExit(f"验收账号 {username} 登录失败（{r.status_code}）：{r.text[:200]}")
    return bearer(r.json()["token"])


def login_http(base: str, username: str) -> dict[str, str]:
    """真 HTTP 版：与界面同一个入口、同一枚令牌。"""
    import urllib.request

    pw = passwords().get(username) or ensure_accounts()[username]
    req = urllib.request.Request(
        f"{base}/api/v1/auth/login",
        data=json.dumps({"username": username, "password": pw}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        token = json.loads(resp.read().decode("utf-8"))["token"]
    return bearer(token)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
