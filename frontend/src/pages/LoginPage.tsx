import { useState } from "react";
import { INDUSTRIES, SELF_SERVICE_ROLES } from "../api/rbac";
import { useIdentity } from "../state/identity";
import { styles as s, tabClass } from "../styles";

/** 登录页 / 注册页：RBAC 的入口从「界面上随便挑角色」变成「用账号换身份」。
 *
 * 三条刻意的设计：
 * 1. **口令不写进任何日志或 URL**，错误信息原样显示后端那句人话（「用户名或口令不正确」），
 *    不自己编「账号或密码错误，请重试」——后端刻意不区分用户是否存在，前端也不该替它区分。
 * 2. 注册表单里**没有治理角色可选**。这不是省事：能自授 knowledge_owner 就等于没有权限模型。
 *    界面上把这条规则说出来，免得用户以为系统漏做了这个选项。
 * 3. 注册成功不自动登录：先让人看清自己拿到了什么范围。
 */
export function LoginPage() {
  const { login, register, busy, error, clearError } = useIdentity();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [role, setRole] = useState("delivery_engineer");
  const [industries, setIndustries] = useState<string[]>([]);
  const [notice, setNotice] = useState("");

  const switchMode = (next: "login" | "register") => {
    setMode(next);
    setErrorBoth("");
  };

  const setErrorBoth = (msg: string) => {
    setNotice(msg);
    if (!msg) clearError();
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setNotice("");
    if (mode === "login") {
      await login({ username, password });
      return;
    }
    const msg = await register({ username, password, display_name: displayName, role, industries });
    // register() 成功时返回的是「说明」，失败时返回错误——两者都直接显示，不猜
    setErrorBoth(msg);
  };

  return (
    <div className={s.authWrap}>
      <header className={s.brand}>
        <h1>知效引擎 KEE</h1>
        <span>企业知识服务 · 内网部署版（本地推理，数据不出域）</span>
      </header>

      <section className={s.authCard} aria-label="登录">
        <div className={s.authTabs} role="tablist" aria-label="登录或注册">
          <button type="button" role="tab" aria-selected={mode === "login"} className={tabClass(mode === "login")}
                  onClick={() => switchMode("login")}>
            登录
          </button>
          <button type="button" role="tab" aria-selected={mode === "register"} className={tabClass(mode === "register")}
                  onClick={() => switchMode("register")}>
            注册
          </button>
        </div>

        <form onSubmit={submit} className={s.authForm}>
          <label className={s.field}>
            <span>用户名</span>
            <input value={username} autoComplete="username" required
                   placeholder="3–32 位小写字母、数字或 . _ -"
                   onChange={(e) => setUsername(e.target.value)} />
          </label>

          <label className={s.field}>
            <span>口令</span>
            <input value={password} type="password" autoComplete={mode === "login" ? "current-password" : "new-password"}
                   required onChange={(e) => setPassword(e.target.value)} />
          </label>

          {mode === "register" && (
            <>
              <label className={s.field}>
                <span>显示名（可空，用于审计与列表）</span>
                <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
              </label>

              <label className={s.field}>
                <span>角色</span>
                <select value={role} onChange={(e) => setRole(e.target.value)}>
                  {SELF_SERVICE_ROLES.map((r) => (
                    <option key={r.role} value={r.role}>{r.label}（可见密级 ≤{r.maxLevel}）</option>
                  ))}
                </select>
              </label>

              <div className={s.field}>
                <span>行业范围（不选＝不限定，由服务端按角色处理）</span>
                <div className={s.chips}>
                  {INDUSTRIES.map((name) => {
                    const on = industries.includes(name);
                    return (
                      <button type="button" key={name}
                              className={`${s.chip} ${on ? s.chipOn : ""}`} aria-pressed={on}
                              onClick={() => setIndustries(on ? industries.filter((x) => x !== name) : [...industries, name])}>
                        {name}
                      </button>
                    );
                  })}
                </div>
              </div>

              <p className={s.authHint}>
                口令至少 10 位。治理角色（知识 Owner / 系统管理员）不在此列，须由现有管理员在「账号」页授予。
              </p>
            </>
          )}

          <button type="submit" className={`${s.btn} ${s.btnPrimary}`} disabled={busy}>
            {busy ? "处理中…" : mode === "login" ? "进入系统" : "注册账号"}
          </button>

          {(error || notice) && (
            <p className={s.authError} role="alert" data-testid="auth-message">{error || notice}</p>
          )}
        </form>
      </section>
    </div>
  );
}
