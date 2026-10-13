import { useState } from "react";
import { HealthStrip } from "./components/HealthStrip";
import { AccountBar } from "./components/AccountBar";
import { isGovernance } from "./api/rbac";
import { AskPage } from "./pages/AskPage";
import { AccountsPage } from "./pages/AccountsPage";
import { AuditPage } from "./pages/AuditPage";
import { KbPage } from "./pages/KbPage";
import { LoginPage } from "./pages/LoginPage";
import { useIdentity } from "./state/identity";
import { styles as s, tabClass } from "./styles";

type Tab = "ask" | "kb" | "audit" | "accounts";

export function App() {
  const { user, booting } = useIdentity();
  const [tab, setTab] = useState<Tab>("ask");

  if (booting) {
    return (
      <div className={s.shell}>
        <p className={s.muted}>正在恢复登录状态…</p>
      </div>
    );
  }

  // 没有有效会话就只有一个界面：登录页。旧版在这里放的是「随便挑个角色」的身份条，
  // 那等于把 RBAC 做成了前端下拉框——越权与否由浏览器说了算，不是由服务端说了算。
  if (!user) return <LoginPage />;

  const tabs: { key: Tab; label: string }[] = [
    { key: "ask", label: "问答" },
    { key: "kb", label: "知识库管理" },
    { key: "audit", label: "审计" },
  ];
  if (isGovernance(user.role)) tabs.push({ key: "accounts", label: "账号" });

  // 退出再登录成普通角色时，上一个 tab 可能是「账号」——不回落就会渲染出一片空白
  const active: Tab = tabs.some((t) => t.key === tab) ? tab : "ask";

  return (
    <div className={s.shell}>
      <header className={s.brand}>
        <h1>知效引擎 KEE</h1>
        <span>企业知识服务 · 内网部署版（本地推理，数据不出域）</span>
      </header>

      <AccountBar />
      <HealthStrip />

      <nav className={s.tabs} role="tablist" aria-label="功能">
        {tabs.map((t) => (
          <button
            type="button"
            key={t.key}
            role="tab"
            aria-selected={active === t.key}
            className={tabClass(active === t.key)}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className={s.panel}>
        {active === "ask" && <AskPage />}
        {active === "kb" && <KbPage />}
        {active === "audit" && <AuditPage />}
        {active === "accounts" && <AccountsPage />}
      </main>
    </div>
  );
}
