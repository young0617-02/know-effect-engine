/** API 客户端：统一身份头、统一错误体解析、统一 trace_id。
 *
 * 两条纪律值得在代码层钉住：
 * 1. **前端一律用相对路径**。dev 走 Vite 代理、生产走 Nginx 同源反代，
 *    于是「开发能通、上线 CORS 炸」和「上线时才发现 base URL 配错」这两类分叉结构上不成立。
 * 2. **错误体按后端的 code/message/trace_id/retryable 原样呈现**。
 *    后端把「依赖挂了」和「你参数写错」分得很清（E9xx vs E4xx），
 *    前端若统一显示「请求失败」就把最要紧的那条信息抹掉了。
 */

import { authHeaders } from "./auth";
import type { ApiErrorBody, AnswerPayload, AnswerRequest } from "./types";

export interface Identity {
  role: string;
  actor: string;
  industries: string[];
  projects: string[];
}

export class ApiError extends Error {
  readonly code: string;
  readonly httpStatus: number;
  readonly traceId: string;
  readonly retryable: boolean;

  constructor(body: ApiErrorBody, httpStatus: number) {
    super(body.message || `请求失败（HTTP ${httpStatus}）`);
    this.name = "ApiError";
    this.code = body.code || "E500";
    this.httpStatus = httpStatus;
    this.traceId = body.trace_id || "";
    this.retryable = Boolean(body.retryable);
  }

  /** 界面直接用它当 tooltip：把「服务端说了什么」和「哪个请求」一起给出来。 */
  get display(): string {
    const tail = this.traceId ? ` · trace ${this.traceId}` : "";
    return `${this.code} ${this.message}${tail}`;
  }
}

/** 身份相关的头，一次构造：有登录令牌就带 Bearer，没有才退回请求头身份（内网/脚本通道）。
 *
 * 放在这一个函数里，是因为 `apiFetch` 与 `postSse` 都走它——
 * 分两处加令牌迟早出现「普通接口认登录、流式接口不认」这种半吊子状态。
 */
export function identityHeaders(id: Identity): Record<string, string> {
  // HTTP 头只允许 latin-1 字节，而部门/项目/actor 会是中文（金融、张三）。
  // 直接塞中文浏览器当场拒发，所以这里按契约做百分号编码，后端 `context_of` 对应 unquote。
  const enc = (v: string) => encodeURIComponent(v);
  const h: Record<string, string> = { ...authHeaders() };
  if (!h["Authorization"] && id.role) h["X-Kee-Role"] = id.role;   // 没登录才发头身份
  else if (id.role) h["X-Kee-Role"] = id.role;                     // 服务端以令牌为准，这个头只是冗余
  // actor 缺省时后端回落到 role；这里显式送出是为了让审计里的 actor 可区分（多人共用一个角色时仍能分人留痕）
  if (id.actor) h["X-Kee-Actor"] = enc(id.actor);
  if (id.industries.length) h["X-Kee-Industries"] = id.industries.map(enc).join(",");
  if (id.projects.length) h["X-Kee-Projects"] = id.projects.map(enc).join(",");
  return h;
}

export async function parseErrorResponse(resp: Response): Promise<ApiError> {
  // 代理/网关的 5xx 常常不是后端契约体（是一段 HTML），那种情况也要给出可用的信息，
  // 但不能伪造一个 code 冒充后端说的数字。
  let body: ApiErrorBody;
  try {
    body = (await resp.json()) as ApiErrorBody;
  } catch {
    body = {
      code: "E500",
      message: `响应不是 JSON（HTTP ${resp.status}），可能被代理或网关拦截`,
      trace_id: resp.headers.get("X-Trace-Id") || "",
      retryable: resp.status >= 500,
    };
  }
  if (!body.code) body.code = `E${resp.status}`;
  return new ApiError(body, resp.status);
}

export interface RequestOptions {
  identity: Identity;
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
  idempotencyKey?: string;
}

export async function apiFetch<T>(path: string, opts: RequestOptions): Promise<T> {
  const headers: Record<string, string> = { ...identityHeaders(opts.identity) };
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (opts.idempotencyKey) headers["Idempotency-Key"] = opts.idempotencyKey;

  const resp = await fetch(path, {
    method: opts.method ?? (opts.body !== undefined ? "POST" : "GET"),
    headers,
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
    signal: opts.signal,
  });
  if (!resp.ok) throw await parseErrorResponse(resp);
  return (await resp.json()) as T;
}

export interface DocumentList {
  total?: number;
  items?: Record<string, unknown>[];
  scope?: { role: string; max_security_level: number; filter_expr: string };
  [k: string]: unknown;
}

export const getDocuments = (identity: Identity, params: { limit?: number; offset?: number; status?: string; industry?: string } = {}) => {
  const q = new URLSearchParams();
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  if (params.status) q.set("status", params.status);
  if (params.industry) q.set("industry", params.industry);
  const suffix = q.toString() ? `?${q.toString()}` : "";
  return apiFetch<DocumentList>(`/api/v1/documents${suffix}`, { identity });
};

export const postAnswer = (identity: Identity, body: AnswerRequest, idempotencyKey?: string) =>
  apiFetch<AnswerPayload>("/api/v1/answers", { identity, body, idempotencyKey });

export interface Health {
  status: string;
  deps: Record<string, { ok: boolean; error?: string; [k: string]: unknown}>;
}

/** /healthz 不带身份头：它回答的是「依赖挂没挂」，与你是谁无关。 */
export async function getHealth(): Promise<Health> {
  const resp = await fetch("/healthz");
  if (!resp.ok) throw await parseErrorResponse(resp);
  return (await resp.json()) as Health;
}

/** multipart 上传走的是另一条路：不能设 Content-Type（浏览器要自己带 boundary），
 * 但身份头与幂等键照旧要带——上传是写操作，审计里的 actor 必须是真实操作者而不是角色名。 */
export async function uploadDocument(
  identity: Identity,
  file: File,
  form: Record<string, string | number | undefined>,
  idempotencyKey?: string,
): Promise<Record<string, unknown>> {
  const fd = new FormData();
  fd.append("file", file);
  for (const [k, v] of Object.entries(form)) if (v !== undefined && v !== "") fd.append(k, String(v));
  const resp = await fetch("/api/v1/documents/upload", {
    method: "POST",
    headers: { ...identityHeaders(identity), ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}) },
    body: fd,
  });
  if (!resp.ok) throw await parseErrorResponse(resp);
  return (await resp.json()) as Record<string, unknown>;
}

export const getDocument = (identity: Identity, docId: string) =>
  apiFetch<{ document: Record<string, unknown>; versions: Record<string, unknown>[] }>(
    `/api/v1/documents/${encodeURIComponent(docId)}`,
    { identity },
  );

export const reviewDocument = (identity: Identity, docId: string, decision: "pass" | "reject", note = "") =>
  apiFetch<Record<string, unknown>>(`/api/v1/documents/${encodeURIComponent(docId)}/review`, {
    identity,
    body: { decision, note },
  });

export const rollbackDocument = (identity: Identity, docId: string, toVersionId: string) =>
  apiFetch<Record<string, unknown>>(`/api/v1/documents/${encodeURIComponent(docId)}/rollback`, {
    identity,
    body: { to_version_id: toVersionId },
  });

export const runReconcile = (identity: Identity, repair = true) =>
  apiFetch<Record<string, unknown>>(`/api/v1/reconcile/run?repair=${repair}`, { identity, body: {} });

export interface AuditRow {
  id: number;
  actor: string;
  action: string;
  target: string;
  trace_id: string | null;
  payload: Record<string, unknown> | null;
  ts: string;
}

export const getAudit = (
  identity: Identity,
  params: { limit?: number; offset?: number; actor?: string; action?: string; target?: string; trace_id?: string } = {},
) => {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") q.set(k, String(v));
  return apiFetch<{ total: number; limit: number; offset: number; items: AuditRow[]; scope: { role: string } }>(
    `/api/v1/audit?${q.toString()}`,
    { identity },
  );
};

/** 会话历史：左侧栏读列表，点开某一条读轮次。
 *
 * 权威源是 PG 的 `session` / `session_turn`，前端不自己存一份对话——
 * 存了就出现「浏览器里的那份」和「服务端的那份」两个真相，刷新一次就分叉。
 */

export interface SessionRow {
  id: string;
  user_id: string;
  n_turns: number;
  title: string | null;
  last_status: string | null;
  created_at: string;
  updated_at: string;
  last_at: string;
}

export const listSessions = (identity: Identity, limit = 50) =>
  apiFetch<{ total: number; items: SessionRow[]; scope: { actor: string; role: string } }>(
    `/api/v1/sessions?limit=${limit}`,
    { identity },
  );

export interface SessionTurn {
  turn_no: number;
  question: string;
  status: string;
  conclusion: string;
  retrieval_ms: number | null;
  total_ms: number | null;
  trace_id: string | null;
  created_at: string;
  answer?: Record<string, unknown> | null;
}

export const getSession = (identity: Identity, sessionId: string) =>
  apiFetch<{ session_id: string; user_id: string; turns: SessionTurn[] }>(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}`,
    { identity },
  );

