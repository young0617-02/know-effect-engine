/** 身份与登录态。
 *
 * 现在的主路径是**真登录**：账号、口令、会话都在服务端，`identity` 由登录结果推导出来，
 * 界面上没有「随便挑一个角色」的入口——挑角色等于换权限，那是登录页之前该被禁止的事。
 *
 * 保留 `identity / set / reset` 这三个名字，是为了让既有页面（问答、知识库、审计）与它们的
 * 组件测试不必一起改写；但语义变了：它们现在**只读**地反映登录账号，
 * `set/reset` 只在「未启用登录的内网脚本通道」下才有意义。
 *
 * 约束（请求头身份可伪造、限内网、接生产须关）记在《技术方案》7.2.1、《版本锁定清单》与风险册，
 * 不摆在界面上；`header_identity_allowed` 是后端如实回传的当前开关状态。
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  getToken, setToken, isAuthError,
  login as apiLogin, logout as apiLogout, me as apiMe, register as apiRegister,
  type AuthUser, type Credentials,
} from "../api/auth";
import type { Identity } from "../api/client";

const EMPTY_IDENTITY: Identity = { role: "", actor: "", industries: [], projects: [] };

function fromUser(u: AuthUser): Identity {
  return { role: u.role, actor: u.username, industries: u.industries || [], projects: u.projects || [] };
}

interface SessionValue {
  identity: Identity;
  user: AuthUser | null;
  maxLevel: number;
  filterExpr: string;
  headerIdentityAllowed: boolean;
  booting: boolean;
  busy: boolean;
  error: string;
  login: (c: Credentials) => Promise<boolean>;
  register: (c: Credentials) => Promise<string>;   // 返回给界面显示的说明（成功）或错误（失败）
  logout: () => Promise<void>;
  clearError: () => void;
}

const SessionContext = createContext<SessionValue | null>(null);

export function IdentityProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [identity, setIdentity] = useState<Identity>(EMPTY_IDENTITY);
  const [maxLevel, setMaxLevel] = useState(0);
  const [filterExpr, setFilterExpr] = useState("");
  const [headerIdentityAllowed, setHeaderAllowed] = useState(false);
  const [booting, setBooting] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const adopt = useCallback((meResp: { user: AuthUser; max_security_level: number; filter_expr: string;
                                       header_identity_allowed: boolean }) => {
    setUser(meResp.user);
    setIdentity(fromUser(meResp.user));
    setMaxLevel(meResp.max_security_level);
    setFilterExpr(meResp.filter_expr);
    setHeaderAllowed(meResp.header_identity_allowed);
  }, []);

  // 刷新页面后接着用：localStorage 里有令牌就先问 /auth/me，服务端说过期就清掉
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!getToken()) {
        setBooting(false);
        return;
      }
      try {
        const r = await apiMe();
        if (alive) adopt(r);
      } catch (e) {
        if (isAuthError(e)) setToken("");
        else if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setBooting(false);
      }
    })();
    return () => { alive = false; };
  }, [adopt]);

  const login = useCallback(async (c: Credentials) => {
    setBusy(true);
    setError("");
    try {
      const { token } = await apiLogin(c);
      setToken(token);
      adopt(await apiMe());
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }, [adopt]);

  const register = useCallback(async (c: Credentials) => {
    setBusy(true);
    setError("");
    try {
      const r = await apiRegister(c);
      // 注册完不自动登录：让使用者看清「注册能拿到什么角色、拿不到什么」
      return r.note || "注册成功，请用刚才的口令登录";
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      return msg;
    } finally {
      setBusy(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } catch {
      /* 服务端没撤销成功也要把本地令牌清掉：留在手里只会反复撞 401 */
    }
    setToken("");
    setUser(null);
    setIdentity(EMPTY_IDENTITY);
    setMaxLevel(0);
    setFilterExpr("");
  }, []);

  const value = useMemo<SessionValue>(() => ({
    identity, user, maxLevel, filterExpr, headerIdentityAllowed, booting, busy, error,
    login, register, logout,
    clearError: () => setError(""),
  }), [identity, user, maxLevel, filterExpr, headerIdentityAllowed, booting, busy, error,
      login, register, logout]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useIdentity(): SessionValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useIdentity 必须在 IdentityProvider 内使用");
  return ctx;
}
