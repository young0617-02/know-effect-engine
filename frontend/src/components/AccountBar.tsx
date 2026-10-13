import { roleSpec } from "../api/rbac";
import { useIdentity } from "../state/identity";
import { styles as s } from "../styles";

/** 账号条：显示「你是谁、能看到哪一级」，并提供退出。
 *
 * 与旧的「身份条」的区别是实质性的：旧界面可以随手把角色从「新人」切成「系统管理员」，
 * 那等于没有权限模型。现在范围由登录账号决定，这里**只读**——
 * 那行过滤表达式仍然显示，因为它解释的是「为什么这条搜不到」，不是前端在遮结果。
 */
export function AccountBar() {
  const { user, identity, maxLevel, filterExpr, logout } = useIdentity();
  const spec = roleSpec(identity.role);
  if (!user) return null;

  return (
    <section className={s.identity} aria-label="当前账号">
      <div className={s.field}>
        <span>账号</span>
        <strong data-testid="account-name">{user.display_name || user.username}</strong>
        <code>{user.username}</code>
      </div>
      <div className={s.field}>
        <span>角色</span>
        <strong data-testid="account-role">{spec?.label ?? user.role}</strong>
        <code>{user.role}</code>
      </div>
      <div className={s.field}>
        <span>行业范围</span>
        <strong>{identity.industries.length ? identity.industries.join("、") : "不限定"}</strong>
      </div>
      <button type="button" className={s.link} onClick={() => void logout()}>退出登录</button>

      <p className={s.scopeNote}>
        可见密级上限 <strong>{maxLevel}</strong>
        {" · "}
        <code>{filterExpr}</code>
        {" · "}
        过滤发生在<em>检索层</em>，不是前端把结果遮住。
      </p>
    </section>
  );
}
