import { useCallback, useEffect, useState } from "react";
import { grantRole, listUsers } from "../api/auth";
import { ROLES, isGovernance } from "../api/rbac";
import { useIdentity } from "../state/identity";
import { styles as s } from "../styles";

/** 账号页（治理角色可见）：列出账号、授予/调整角色。
 *
 * 存在的理由是**闭环**：注册拿不到治理角色，那第一个管理员从哪来？
 * 两条路——库里空时的首个注册者（服务端 `is_first_account` 例外），或已有管理员在这里授。
 * 这条授权动作本身会进审计（`user_role_granted`），因为它改的是谁能看到什么。
 */
export function AccountsPage() {
  const { user, error, clearError } = useIdentity();
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const r = await listUsers();
      setRows(r.items || []);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  if (!isGovernance(user?.role || "")) {
    return <p className={s.empty}>仅知识 Owner / 系统管理员可查看账号与授权。</p>;
  }

  const grant = async (username: string, role: string) => {
    setMsg("");
    clearError();
    try {
      const r = await grantRole(username, role);
      setMsg(`${username} → ${role}（由 ${r.granted_by} 授予，已进审计）`);
      await refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <section>
      <div className={s.row}>
        <button type="button" className={s.link} onClick={() => void refresh()} disabled={loading}>
          {loading ? "读取中…" : "刷新"}
        </button>
        <span className={s.muted}>共 {rows.length} 个账号</span>
      </div>
      {(msg || error) && <p className={s.formMsg} data-testid="accounts-msg">{msg || error}</p>}
      <table className={s.table}>
        <thead>
          <tr><th>账号</th><th>显示名</th><th>角色</th><th>部门</th><th>最近登录</th><th>失败次数</th><th>授予角色</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={String(r.username)}>
              <td><code>{String(r.username)}</code></td>
              <td>{String(r.display_name ?? "")}</td>
              <td>{String(r.role)}</td>
              <td>{String(r.industries ?? "") || "—"}</td>
              <td>{r.last_login_at ? String(r.last_login_at).slice(0, 19) : "从未"}</td>
              <td>{String(r.failed_count ?? 0)}</td>
              <td>
                <select defaultValue={String(r.role)} aria-label={`授予 ${r.username} 角色`}
                        onChange={(e) => void grant(String(r.username), e.target.value)}>
                  {ROLES.map((x) => <option key={x.role} value={x.role}>{x.label}</option>)}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
