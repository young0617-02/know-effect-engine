/** 身份上下文：Demo 阶段用请求头占位身份（后端如此，前端不假装它是真的认证）。
 *
 * 必须让界面自己说出来：这套身份**可伪造**，生产要换真实身份源（M1 报告第 103 行的原话）。
 * 把一个占位实现渲染成「已登录的安全系统」是这一层最容易犯的错。
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Identity } from "../api/client";
import { ROLES } from "../api/rbac";

const STORAGE_KEY = "kee.ide…tity";

const DEFAULT_IDENTITY: Identity = {
  role: "delivery_engineer",
  actor: "demo.engineer",
  industries: [],
  projects: [],
};

function load(): Identity {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_IDENTITY;
    const parsed = JSON.parse(raw) as Partial<Identity>;
    // 角色名可能在版本迭代中变过；不在后端角色表里的一律回默认，别把一个失效角色发出去
    const role = ROLES.some((r) => r.role === parsed.role) ? parsed.role! : DEFAULT_IDENTITY.role;
    return {
      role,
      actor: parsed.actor ?? DEFAULT_IDENTITY.actor,
      industries: Array.isArray(parsed.industries) ? parsed.industries : [],
      projects: Array.isArray(parsed.projects) ? parsed.projects : [],
    };
  } catch {
    return DEFAULT_IDENTITY;
  }
}

interface IdentityContextValue {
  identity: Identity;
  set: (patch: Partial<Identity>) => void;
  reset: () => void;
}

const IdentityContext = createContext<IdentityContextValue | null>(null);

export function IdentityProvider({ children }: { children: ReactNode }) {
  const [identity, setIdentity] = useState<Identity>(load);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(identity));
    } catch {
      // 隐私模式下 localStorage 会抛；身份存不住只是刷新后回默认，不该把整个应用打挂
    }
  }, [identity]);

  const set = useCallback((patch: Partial<Identity>) => setIdentity((prev) => ({ ...prev, ...patch })), []);
  const reset = useCallback(() => setIdentity(DEFAULT_IDENTITY), []);
  const value = useMemo(() => ({ identity, set, reset }), [identity, set, reset]);

  return <IdentityContext.Provider value={value}>{children}</IdentityContext.Provider>;
}

export function useIdentity(): IdentityContextValue {
  const ctx = useContext(IdentityContext);
  if (!ctx) throw new Error("useIdentity 必须在 IdentityProvider 内使用");
  return ctx;
}
