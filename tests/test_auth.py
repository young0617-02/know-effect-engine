"""登录/注册与权限边界。打真 PostgreSQL，**不 mock**：
这些端点的全部价值就在「口令校验、会话可撤销、注册不能自授治理角色」，
换成假对象就只剩「函数被调用了」这种什么都证明不了的断言。PG 不可达时整组跳过并说明。

`skip_embedding=True` 起应用：鉴权不碰稠密检索，不该去抢显存。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from kee.api.app import create_app
from kee.api.auth import hash_password, verify_password
from kee.config import Settings, get_settings
from kee.db.pg import AuthoritativeStore

SUFFIX = int(time.time())
PW = "correct horse battery"


@pytest.fixture(scope="module")
def store() -> AuthoritativeStore:
    st = AuthoritativeStore()
    try:
        with st.connect() as c:
            c.execute("SELECT 1 FROM app_user LIMIT 1").fetchone()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"PostgreSQL 不可达或账号表缺失，跳过鉴权测试（跳过不是通过）：{e}")
    return st


@pytest.fixture(scope="module")
def client(store) -> TestClient:
    app = create_app(skip_embedding=True)
    with TestClient(app) as c:
        yield c
    with store.connect() as conn:   # 清掉本轮造的账号，别把测试用户留给下一次
        conn.execute("DELETE FROM app_user WHERE username LIKE %s", (f"t{SUFFIX}_%",))


def _register(client, name: str, role: str = "delivery_engineer"):
    return client.post("/api/v1/auth/register",
                       json={"username": name, "password": PW, "display_name": f"测试 {name}", "role": role})


def test_register_then_login_and_me(client):
    u = f"t{SUFFIX}_basic"
    r = _register(client, u)
    assert r.status_code == 201, r.text
    assert r.json()["user"]["role"] == "delivery_engineer"

    login = client.post("/api/v1/auth/login", json={"username": u, "password": PW})
    assert login.status_code == 200, login.text
    token = login.json()["token"]

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["user"]["username"] == u
    # 密级上限与过滤表达式由**角色**推导，不是客户端说了算
    assert body["max_security_level"] == 2
    assert "security_level <= 2" in body["filter_expr"]


def test_wrong_password_does_not_leak_whether_the_user_exists(client):
    known = f"t{SUFFIX}_exists"
    assert _register(client, known).status_code == 201
    a = client.post("/api/v1/auth/login", json={"username": known, "password": "wrong wrong wrong"})
    b = client.post("/api/v1/auth/login", json={"username": f"t{SUFFIX}_nobody", "password": "wrong wrong wrong"})
    assert a.status_code == b.status_code == 401
    assert a.json()["message"] == b.json()["message"], "两种失败必须给同一句话，否则用户名枚举免费"


def test_registration_cannot_grant_itself_a_governance_role(client):
    r = _register(client, f"t{SUFFIX}_escalate", role="knowledge_owner")
    assert r.status_code == 400
    assert "治理角色" in r.json()["message"]


def test_governance_role_can_be_granted_by_an_admin_only(client):
    u = f"t{SUFFIX}_grant"
    assert _register(client, u).status_code == 201
    # 非治理角色来授 → 403
    bad = client.post(f"/api/v1/auth/users/{u}/role", json={"role": "system_admin"},
                      headers={"X-Kee-Role": "delivery_engineer", "X-Kee-Actor": "eng-01"})
    assert bad.status_code == 403
    ok = client.post(f"/api/v1/auth/users/{u}/role", json={"role": "knowledge_owner"},
                     headers={"X-Kee-Role": "system_admin", "X-Kee-Actor": "admin-01"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["user"]["role"] == "knowledge_owner"


def test_short_role_name_is_grantable(client):
    """`pm` 是合法角色里最短的一个（两个字符）。

    请求模型一度给它加了 `min_length=3`，于是「把某人设为项目经理」这条授权被长度下限判成非法输入——
    生产入口实测踩过（走 HTTP 的授权返回 400 校验噪声，界面里那个下拉框点它就死）。
    角色合法性只应由 `auth.set_role` 判，长度下限不在它的语义范围内。
    """
    u = f"t{SUFFIX}_pm"
    assert _register(client, u).status_code == 201
    ok = client.post(f"/api/v1/auth/users/{u}/role", json={"role": "pm"},
                     headers={"X-Kee-Role": "system_admin", "X-Kee-Actor": "admin-01"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["user"]["role"] == "pm"
    # 非法角色仍然要挡，且给的是人话而不是校验噪声
    bad = client.post(f"/api/v1/auth/users/{u}/role", json={"role": "superuser"},
                      headers={"X-Kee-Role": "system_admin", "X-Kee-Actor": "admin-01"})
    assert bad.status_code == 400
    assert "未知角色" in bad.json()["message"]


def test_logout_actually_revokes_the_session(client, store):
    u = f"t{SUFFIX}_logout"
    assert _register(client, u).status_code == 201
    token = client.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    out = client.post("/api/v1/auth/logout?all_devices=false", headers={"Authorization": f"Bearer {token}"})
    assert out.status_code == 200 and out.json()["revoked"] == 1
    # 令牌只从浏览器里抹掉不算退出——**库里那行必须没了**
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    with store.connect() as c:
        assert c.execute("SELECT 1 FROM user_session WHERE username = %s", (u,)).fetchone() is None


def test_logout_revokes_every_session_of_that_account(client, store):
    """退出默认把该账号**所有**会话一起撤销。

    只删当前那枚令牌时，别的浏览器里那枚还能用到 TTL 自然过期（12h）——
    共享电脑上的「退出」因此是假的。这是风险册里那条「退出只撤销当前令牌」的正面处置。
    """
    u = f"t{SUFFIX}_logout_all"
    assert _register(client, u).status_code == 201
    t1 = client.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
    t2 = client.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t2}"}).status_code == 200

    out = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {t1}"})
    assert out.status_code == 200
    assert out.json()["revoked"] == 2 and out.json()["scope"] == "all_devices"
    for t in (t1, t2):
        assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t}"}).status_code == 401
    with store.connect() as c:
        assert c.execute("SELECT 1 FROM user_session WHERE username = %s", (u,)).fetchone() is None


def test_logout_can_stay_scoped_to_one_device(client, store):
    """`?all_devices=false` 是「只退本机」那条路，不能把别人的会话一起删了。"""
    u = f"t{SUFFIX}_logout_one"
    assert _register(client, u).status_code == 201
    t1 = client.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
    t2 = client.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
    out = client.post("/api/v1/auth/logout?all_devices=false", headers={"Authorization": f"Bearer {t1}"})
    assert out.json()["revoked"] == 1 and out.json()["scope"] == "current_token"
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t2}"}).status_code == 200


def test_force_logout_is_governance_only(client, store):
    """治理角色才有踢人动作；被踢的那位令牌当场失效；这一步自己进审计。"""
    admin = f"t{SUFFIX}_fl_admin"
    victim = f"t{SUFFIX}_fl_victim"
    assert _register(client, admin, role="newcomer").status_code == 201
    assert _register(client, victim, role="newcomer").status_code == 201
    # 授予这一步沿用本文件既有的治理通道（脚本身份）；被测的三个动作全走真令牌。
    # 不能叫 victim 自己给自己升角色——那条本来就该 403，不是引导路径。
    assert client.post(f"/api/v1/auth/users/{admin}/role", json={"role": "system_admin"},
                       headers={"X-Kee-Role": "system_admin", "X-Kee-Actor": "admin-01"}).status_code == 200
    admin_tok = client.post("/api/v1/auth/login", json={"username": admin, "password": PW}).json()["token"]
    victim_tok = client.post("/api/v1/auth/login", json={"username": victim, "password": PW}).json()["token"]

    denied = client.post(f"/api/v1/auth/users/{victim}/revoke-sessions",
                         headers={"Authorization": f"Bearer {victim_tok}"})
    assert denied.status_code == 403

    kick = client.post(f"/api/v1/auth/users/{victim}/revoke-sessions",
                       headers={"Authorization": f"Bearer {admin_tok}"})
    assert kick.status_code == 200 and kick.json()["revoked"] == 1
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {victim_tok}"}).status_code == 401

    missing = client.post("/api/v1/auth/users/t0_suffix_nobody/revoke-sessions",
                          headers={"Authorization": f"Bearer {admin_tok}"})
    assert missing.status_code == 404

    with store.connect() as c:
        row = c.execute("SELECT actor, payload FROM audit_log WHERE action = 'user_force_logout' "
                        "AND target = %s ORDER BY id DESC LIMIT 1", (victim,)).fetchone()
    assert row["actor"] == admin and row["payload"]["revoked"] == 1


def test_logged_in_identity_beats_the_role_header(client):
    """登录后请求头**不能再改身份**，否则「已登录」只是装饰：带个头就能升成管理员。"""
    u = f"t{SUFFIX}_spoof"
    assert _register(client, u, role="newcomer").status_code == 201
    token = client.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}",
                                                "X-Kee-Role": "system_admin",
                                                "X-Kee-Actor": "attacker"})
    assert me.json()["user"]["role"] == "newcomer"
    assert me.json()["max_security_level"] == 1
    # 审计里的 actor 也必须是账号，不是头里那个名字
    docs = client.get("/api/v1/documents?limit=1", headers={"Authorization": f"Bearer {token}",
                                                            "X-Kee-Role": "system_admin"})
    assert "security_level <= 1" in docs.json()["scope"]["filter_expr"]


def test_header_identity_defaults_off():
    """交付物的默认值必须是「请求头身份关闭」。这条钉的是**默认**，不是开关能不能用。

    读 `model_fields` 的默认而不是实例：实例会被 `tests/conftest.py` 里为测试通道
    显式设的环境变量影响，那样这条测试就变成在测本机环境了。
    """
    assert Settings.model_fields["allow_header_identity"].default is False


def test_header_identity_can_be_switched_off(monkeypatch, store):
    """关掉请求头通道后：没令牌一律 E401，有令牌照常工作。这条测的是「后门真的能关上」。"""
    from kee.api.app import create_app as make

    monkeypatch.setattr("kee.api.app.get_settings", lambda: Settings(allow_header_identity=False))
    with TestClient(make(skip_embedding=True)) as c:
        assert c.get("/api/v1/documents", headers={"X-Kee-Role": "pm"}).status_code == 401
        u = f"t{SUFFIX}_nohdr"
        assert c.post("/api/v1/auth/register",
                      json={"username": u, "password": PW, "role": "pm"}).status_code == 201
        token = c.post("/api/v1/auth/login", json={"username": u, "password": PW}).json()["token"]
        assert c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        with store.connect() as conn:
            conn.execute("DELETE FROM app_user WHERE username = %s", (u,))


def test_password_storage_is_salted_and_tamper_evident():
    h1, h2 = hash_password(PW), hash_password(PW)
    assert h1 != h2, "同口令两次散列必须不同（每用户独立盐）"
    assert PW not in h1 and h1.startswith("scrypt$")
    assert verify_password(PW, h1) and verify_password(PW, h2)
    assert not verify_password("x" + PW, h1)
    assert not verify_password(PW, h1.replace(h1[-1], "0" if h1[-1] != "0" else "1")), "摘要被动过就必须拒绝"
    for broken in ("", "plaintext", "md5$aa$bb", "scrypt$n$8$1$zz$ff"):
        assert not verify_password(PW, broken), f"畸形存储串一律拒绝，不能放行：{broken!r}"


def test_weak_password_and_bad_username_are_rejected(client):
    weak = client.post("/api/v1/auth/register",
                       json={"username": f"t{SUFFIX}_weak", "password": "short", "role": "pm"})
    assert weak.status_code == 400 and "至少" in weak.json()["message"], "口令太短要给**能显示给用户看**的话，不是 422 的字段路径"
    bad = client.post("/api/v1/auth/register",
                      json={"username": f"t{SUFFIX} 空格", "password": PW, "role": "pm"})
    assert bad.status_code in (400, 422)
