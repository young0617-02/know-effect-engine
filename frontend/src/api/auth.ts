/** 登录态客户端：令牌存放、登录/注册/退出、当前账号。
 *
 * 令牌放 localStorage 而不是内存：刷新页面不该把人踢出去。
 * 但它**只是凭据的载体，不是安全边界**——真正的边界在服务端：
 * 库里只存令牌的 SHA-256 摘要与过期时间，退出即删行，越权一律服务端 403。
 */

import { ApiError, parseErrorResponse } from "./client";

/** 令牌的存放键。导出它不是为了好看：测试必须用它，否则代码里的键写错、
 *  测试里硬编码另一个键，两边各自一致地错下去（这一版就栽过：键被写成占位符时，
 *  「登录成功」仍然全绿，只有「过期要清令牌」那条才暴露出来）。 */
export const TOKEN_KEY = "***";

export interface AuthUser {
  username: string;
  display_name: string;
  role: string;
  industries: string[];
  projects: string[];
}

export interface MeResponse {
  user: AuthUser;
  max_security_level: number;
  filter_expr: string;
  header_identity_allowed: boolean;
}

export const getToken = (): string => {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";   // 隐私模式下 localStorage 会抛；没令牌就是没登录，不该把整个应用打挂
  }
};

export const setToken = (t: string) => {
  try {
    if (t) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* 存不下：本次会话内仍然可用（内存里的值由调用方持有） */
  }
};

export const authHeaders = (): Record<string, string> => {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
};

async function call<T>(path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const resp = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!resp.ok) throw await parseErrorResponse(resp);
  return (await resp.json()) as T;
}

export interface Credentials {
  username: string;
  password: string;
  display_name?: string;
  role?: string;
  industries?: string[];
  projects?: string[];
}

export const login = (c: Credentials) => call<{ token: string; expires_in_hours: number }>("/api/v1/auth/login", c);
export const register = (c: Credentials) =>
  call<{ user: AuthUser; note: string }>("/api/v1/auth/register", c);
export const logout = () => call<{ revoked: number; scope: string }>("/api/v1/auth/logout", {});
export const me = () => call<MeResponse>("/api/v1/auth/me");
export const listUsers = () => call<{ total: number; items: Record<string, unknown>[] }>("/api/v1/auth/users");
export const grantRole = (username: string, role: string, industries: string[] = [], projects: string[] = []) =>
  call<{ user: AuthUser; granted_by: string }>(`/api/v1/auth/users/${encodeURIComponent(username)}/role`,
    { role, industries, projects });
/** 强制下线：撤销该账号全部登录态。口令留着，要停用账号是另一件事。 */
export const revokeSessions = (username: string) =>
  call<{ username: string; revoked: number; by: string }>(
    `/api/v1/auth/users/${encodeURIComponent(username)}/revoke-sessions`, {});

export const isAuthError = (e: unknown): e is ApiError => e instanceof ApiError && (e.code === "E401" || e.httpStatus === 401);
