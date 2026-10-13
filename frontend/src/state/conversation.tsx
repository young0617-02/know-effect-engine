/** 会话与问答记录的状态层。
 *
 * 为什么这些状态**不能留在问答页组件里**：切到「知识库管理」再切回来时组件会被卸载，
 * 挂在它身上的对话、以及**正在流式生成的那一轮**会一起消失——用户看到的现象就是
 * 「我刚才问的东西没了」。把状态提到 tab 之上，切模块只是不渲染它，数据与流都还在。
 *
 * 第二条更要紧的理由：**权威源是服务端的 `session` / `session_turn`**。
 * 前端不另存一份对话（那会出现浏览器那份与服务端那份，刷新即分叉），
 * 左侧历史栏读列表、点开某条再从服务端取回整包答案。
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, getSession, listSessions, type SessionRow, type SessionTurn } from "../api/client";
import { postSse } from "../api/sse";
import type { AnswerPayload, DeltaPayload, RetrievalEvent } from "../api/types";
import { useIdentity } from "./identity";

export const slotKey = (field: string, index: number | null | undefined) => `${field}:${index ?? "-"}`;

interface Slot {
  text: string;
  closed: boolean;
}

export interface Turn {
  id: number;
  question: string;
  phase: "streaming" | "done" | "error";
  retrieval?: RetrievalEvent;
  slots: Record<string, Slot>;
  order: string[];
  streamedChars: number;
  answer?: AnswerPayload;
  error?: string;
  startedAt: number;
  firstDeltaMs?: number;
  /** 从服务端取回的历史轮次：它没有经过本机这一轮的流式过程，界面上要能分清「恢复的」与「刚问的」 */
  restored?: boolean;
}

/** 载荷解析：流是外部输入，坏帧要报出来而不是被 `JSON.parse` 抛成一句天书，更不能静默跳过——
 *  跳过的代价是界面永远停在「生成中」，那正是这套契约最不该出现的状态。 */
function parseEvent(evt: { event: string; data: string }): unknown {
  if (!evt.data.trim()) throw new Error(`事件 ${evt.event} 的载荷为空（未按契约给出 JSON）`);
  try {
    return JSON.parse(evt.data) as unknown;
  } catch {
    throw new Error(`事件 ${evt.event} 的载荷不是合法 JSON：${evt.data.slice(0, 60)}`);
  }
}

const SESSION_STORE = "kee.session";
const readStored = (k: string) => {
  try { return localStorage.getItem(k) || ""; } catch { return ""; }
};
const writeStored = (k: string, v: string) => {
  try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch { /* 存不下：本次会话内仍然可用 */ }
};

/** 历史轮次的检索概况从 `answer.retrieval` 复原（引擎在答案里带了同一份诊断块）。
 *  取不到就**不显示这一行**——用 0 或空值凑出来，等于把「没记录」画成「检索到 0 块」。 */
function retrievalOf(row: SessionTurn): RetrievalEvent | undefined {
  const r = (row.answer?.retrieval ?? undefined) as Record<string, unknown> | undefined;
  if (!r || typeof r !== "object" || r.n_chunks === undefined) return undefined;
  const ids = Array.isArray(r.top_chunk_ids) ? (r.top_chunk_ids as string[]) : [];
  return {
    n_chunks: Number(r.n_chunks ?? ids.length),
    chunk_ids: ids,
    mode: typeof r.mode === "string" ? r.mode : undefined,
    query_used: typeof r.query_used === "string" ? r.query_used : undefined,
    degraded: Array.isArray(r.degraded) ? (r.degraded as string[]) : [],
    // 只有真取到耗时才写这一列：填个 undefined 会让界面显示「· undefined ms」
    ...(row.retrieval_ms !== null && row.retrieval_ms !== undefined
      ? { timings_ms: { retrieval_ms: row.retrieval_ms } } : {}),
  };
}

interface ConversationValue {
  turns: Turn[];
  sessions: SessionRow[];
  activeId: string | null;
  busy: boolean;
  loading: boolean;
  error: string;
  tokenStream: boolean;
  rewriteOnFollowup: boolean;
  setTokenStream: (v: boolean) => void;
  setRewriteOnFollowup: (v: boolean) => void;
  ask: (text: string) => Promise<void>;
  newSession: () => void;
  openSession: (id: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
  clearError: () => void;
}

const Ctx = createContext<ConversationValue | null>(null);

let turnSeq = 0;
let restoreSeq = 0;

export function ConversationProvider({ children }: { children: ReactNode }) {
  const { identity, user } = useIdentity();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tokenStream, setTokenStream] = useState(true);
  const [rewriteOnFollowup, setRewriteOnFollowup] = useState(true);
  const activeRef = useRef<string | null>(null);
  activeRef.current = activeId;

  const patch = useCallback((id: number, fn: (t: Turn) => Turn) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? fn(t) : t)));
  }, []);

  const refreshSessions = useCallback(async () => {
    if (!user) return;
    try {
      const r = await listSessions(identity, 50);
      setSessions(r.items || []);
    } catch (e) {
      // 历史栏读不到不该挡住提问本身：错误显示在栏内，不顶掉整个页面
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [identity, user]);

  const openSession = useCallback(async (id: string) => {
    setLoading(true);
    setError("");
    try {
      const r = await getSession(identity, id);
      // 历史轮次用负数 id：与本机新建轮次的正数号段隔开，恢复旧会话后再提问不会撞 React key
      setTurns(r.turns.map((row) => ({
        id: --restoreSeq,
        question: row.question,
        phase: "done",
        slots: {},
        order: [],
        streamedChars: Number((row.answer?.provisional_streamed_chars as number) ?? 0),
        startedAt: 0,
        restored: true,
        retrieval: retrievalOf(row),
        answer: row.answer
          ? ({ ...(row.answer as object), session_id: id, turn_no: row.turn_no } as AnswerPayload)
          : undefined,
      })));
      setActiveId(id);
      writeStored(`${SESSION_STORE}:${user?.username || "anon"}`, id);
    } catch (e) {
      setError(e instanceof ApiError ? e.display : e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [identity, user]);

  const newSession = useCallback(() => {
    setTurns([]);
    setActiveId(null);
    setError("");
    writeStored(`${SESSION_STORE}:${user?.username || "anon"}`, "");
  }, [user]);

  // 换账号 / 刷新后：先接回上次那一条会话，再拉列表。
  // 按用户名分键存，是因为同一台机器上换账号登录时，不该把上一个人的对话接着往下问。
  useEffect(() => {
    if (!user) {
      setTurns([]);
      setSessions([]);
      setActiveId(null);
      return;
    }
    let alive = true;
    (async () => {
      const saved = readStored(`${SESSION_STORE}:${user.username}`);
      if (saved) {
        try {
          await openSession(saved);
          if (!alive) return;
        } catch {
          writeStored(`${SESSION_STORE}:${user.username}`, "");
        }
      }
      if (alive) await refreshSessions();
    })();
    return () => { alive = false; };
    // openSession/refreshSessions 已随 user/identity 变化重建，这里只跟账号切换
  }, [user, openSession, refreshSessions]);

  const ask = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    const id = ++turnSeq;
    setTurns((prev) => [
      ...prev,
      { id, question: q, phase: "streaming", slots: {}, order: [], streamedChars: 0, startedAt: performance.now() },
    ]);
    setBusy(true);
    setError("");
    const controller = new AbortController();

    // 追问才开改写：首轮字面查询本来就自足，为它多花一次模型调用不划算
    // （实测单次改写 P95 843ms，几乎吃满 800ms 的检索门禁预算）。
    const isFollowup = Boolean(activeRef.current);
    const body = {
      question: q,
      session_id: activeRef.current,
      top_k: 5,
      rewrite: isFollowup && rewriteOnFollowup,
      token_stream: tokenStream,
    };

    try {
      await postSse("/api/v1/answers/stream", {
        identity,
        body,
        signal: controller.signal,
        onEvent: (evt) => {
          if (evt.event === "retrieval") {
            patch(id, (t) => ({ ...t, retrieval: parseEvent(evt) as RetrievalEvent }));
            return;
          }
          if (evt.event === "delta") {
            const d = parseEvent(evt) as DeltaPayload;
            patch(id, (t) => {
              const key = slotKey(d.field, d.index);
              return {
                ...t,
                slots: { ...t.slots, [key]: { text: d.text, closed: d.closed } },
                order: key in t.slots ? t.order : [...t.order, key],
                streamedChars: d.chars,
                firstDeltaMs: t.firstDeltaMs ?? Math.round(performance.now() - t.startedAt),
              };
            });
            return;
          }
          if (evt.event === "answer") {
            const a = parseEvent(evt) as AnswerPayload;
            if (a.session_id) {
              setActiveId(a.session_id);
              activeRef.current = a.session_id;
              writeStored(`${SESSION_STORE}:${user?.username || "anon"}`, a.session_id);
            }
            patch(id, (t) => ({ ...t, phase: "done", answer: a }));
            void refreshSessions();
            return;
          }
          if (evt.event === "error") {
            const e = parseEvent(evt) as { code?: string; message?: string };
            patch(id, (t) => ({ ...t, phase: "error", error: `${e.code ?? "E500"} ${e.message ?? "流式中断"}` }));
          }
        },
      });
      // 流结束了却没有 answer：不能留在「生成中」的样子，那等于界面自己假装成功
      patch(id, (t) =>
        t.phase === "streaming"
          ? { ...t, phase: "error", error: "连接结束但未收到 answer 事件（答案未交付，不算成功）" }
          : t,
      );
    } catch (e) {
      const msg = e instanceof ApiError ? e.display : e instanceof Error ? e.message : String(e);
      patch(id, (t) => (t.phase === "streaming" ? { ...t, phase: "error", error: msg } : t));
    } finally {
      setBusy(false);
    }
  }, [busy, identity, patch, refreshSessions, rewriteOnFollowup, tokenStream, user]);

  const value = useMemo<ConversationValue>(() => ({
    turns, sessions, activeId, busy, loading, error, tokenStream, rewriteOnFollowup,
    setTokenStream, setRewriteOnFollowup, ask, newSession, openSession, refreshSessions,
    clearError: () => setError(""),
  }), [turns, sessions, activeId, busy, loading, error, tokenStream, rewriteOnFollowup,
      ask, newSession, openSession, refreshSessions]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useConversation(): ConversationValue {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useConversation 必须在 ConversationProvider 内使用");
  return ctx;
}
