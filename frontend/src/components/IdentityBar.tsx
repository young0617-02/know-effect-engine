import { useIdentity } from "../state/identity";
import { INDUSTRIES, ROLES, roleSpec } from "../api/rbac";
import { styles as s } from "../styles";

/** 身份条：切角色、填 actor、限定行业/项目范围，并把「你现在能看到哪一级」如实显示出来。
 *
 * 两条刻意的呈现：
 * 1. 权限**不在这里实施**。下面那行过滤表达式是后端 `build_filter_expr` 的口径镜像，
 *    用来解释「为什么搜不到」，不是前端在过滤结果。
 * 2. 明写这是可伪造的请求头占位身份。Demo 可以，生产必须换真实身份源。
 */
export function IdentityBar() {
  const { identity, set, reset } = useIdentity();
  const spec = roleSpec(identity.role);

  const toggleIndustry = (name: string) => {
    const on = identity.industries.includes(name);
    set({ industries: on ? identity.industries.filter((x) => x !== name) : [...identity.industries, name] });
  };

  return (
    <section className={s.identity} aria-label="当前身份">
      <label className={s.field}>
        <span>角色</span>
        <select value={identity.role} onChange={(e) => set({ role: e.target.value })}>
          {ROLES.map((r) => (
            <option key={r.role} value={r.role}>
              {r.label}（{r.role}）
            </option>
          ))}
        </select>
      </label>

      <label className={s.field}>
        <span>操作者 actor</span>
        <input
          value={identity.actor}
          placeholder="审计里区分具体的人"
          onChange={(e) => set({ actor: e.target.value })}
        />
      </label>

      <div className={s.field}>
        <span>行业范围</span>
        <div className={s.chips}>
          {INDUSTRIES.map((name) => (
            <button
              type="button"
              key={name}
              className={`${s.chip} ${identity.industries.includes(name) ? s.chipOn : ""}`}
              aria-pressed={identity.industries.includes(name)}
              onClick={() => toggleIndustry(name)}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <label className={s.field}>
        <span>项目范围（逗号分隔）</span>
        <input
          value={identity.projects.join(",")}
          placeholder="PRJ-FIN-2024-017"
          onChange={(e) => set({ projects: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) })}
        />
      </label>

      <button type="button" className={s.link} onClick={reset}>
        重置
      </button>

      <p className={s.scopeNote}>
        可见密级上限 <strong>{spec?.maxLevel ?? 1}</strong>
        {identity.role === "newcomer" ? "（新人：只给一级，越级的内容在检索层就搜不到）" : ""}
        {" · "}
        <code>{scopeExpr(identity.role, identity.industries, identity.projects)}</code>
        {" · "}
        过滤发生在<em>检索层</em>，不是前端把结果遮住；本条身份走请求头，<strong>可伪造</strong>，仅内网 Demo 可用。
      </p>
    </section>
  );
}

/** 与 kee/retrieval/rbac.py:build_filter_expr 同口径的**展示用**表达式。 */
export function scopeExpr(role: string, industries: string[], projects: string[]): string {
  const max = roleSpec(role)?.maxLevel ?? 1;
  const parts = [`security_level <= ${max}`];
  if (role !== "knowledge_owner" && role !== "system_admin") {
    if (industries.length) parts.push(`industry in [${industries.join(", ")}]`);
    if (projects.length) parts.push(`project in [${projects.join(", ")}]`);
  }
  return parts.join(" and ");
}
