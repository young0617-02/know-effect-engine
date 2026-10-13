import { useState } from "react";
import { HealthStrip } from "./components/HealthStrip";
import { IdentityBar } from "./components/IdentityBar";
import { AskPage } from "./pages/AskPage";
import { AuditPage } from "./pages/AuditPage";
import { KbPage } from "./pages/KbPage";
import { styles as s, tabClass } from "./styles";

type Tab = "ask" | "kb" | "audit";

const TABS: { key: Tab; label: string }[] = [
  { key: "ask", label: "问答" },
  { key: "kb", label: "知识库管理" },
  { key: "audit", label: "审计" },
];

export function App() {
  const [tab, setTab] = useState<Tab>("ask");

  return (
    <div className={s.shell}>
      <header className={s.brand}>
        <h1>知效引擎 KEE</h1>
        <span>企业知识服务 · 内网部署版（本地推理，数据不出域）</span>
      </header>

      <IdentityBar />
      <HealthStrip />

      <nav className={s.tabs} role="tablist" aria-label="功能">
        {TABS.map((t) => (
          <button
            type="button"
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            className={tabClass(tab === t.key)}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className={s.panel}>
        {tab === "ask" && <AskPage />}
        {tab === "kb" && <KbPage />}
        {tab === "audit" && <AuditPage />}
      </main>
    </div>
  );
}
