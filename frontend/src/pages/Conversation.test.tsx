import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { ConversationProvider } from "../state/conversation";
import { IdentityProvider } from "../state/identity";
import { TOKEN_KEY } from "../api/auth";

/** 问答记录的生命周期：切模块不丢、刷新能从服务端接回、左侧历史栏能点开任意一条。
 *
 * 这三条都是「数据在谁身上」的断言。旧版把对话存在问答页组件里，切一次 tab 组件卸载，
 * 用户看到的就是「我问的东西没了」——所以这里钉的不是渲染，而是**状态住在哪一层**。
 * 历史一律从服务端取（`session` / `session_turn` 是权威源），前端不另存一份对话。
 */

const enc = new TextEncoder();

const ANSWER = {
  question: "SIS 表决逻辑是什么？",
  status: "answer",
  conclusion: "2oo3 表决",
  key_points: [{ point: "三取二", evidence_ids: ["E1"] }],
  citations: [{ evidence_id: "E1", chunk_id: "doc-a@v1#1.1#c1", doc_id: "doc-a",
                section_no: "1.1", page: 3, security_level: 2 }],
  dropped_points: [],
  needs_human: false,
  confidence: { composite: 0.7813, score_basis: "rule_composite_not_probability" },
  retrieval: { mode: "hybrid", n_chunks: 5, top_chunk_ids: ["c1"], degraded: [], query_used: "SIS 表决逻辑是什么？" },
  llm: {},
  timings_ms: { e2e_ms: 4200 },
  trace_id: "tr-1",
  notes: [],
  session_id: "s-live",
  provisional_streamed_chars: 0,
  provisional_retracted: false,
};

function sse(frames: { event: string; data: unknown }[]) {
  const text = frames.map((f) => `event: ${f.event}\ndata: ${JSON.stringify(f.data)}\n\n`).join("");
  return {
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "text/event-stream" }),
    body: new ReadableStream<Uint8Array>({
      start(c) { c.enqueue(enc.encode(text)); c.close(); },
    }),
  } as unknown as Response;
}

interface Calls { urls: string[]; bodies: unknown[] }

function stub(calls: Calls) {
  globalThis.fetch = (async (url: unknown, init?: RequestInit) => {
    const u = String(url);
    calls.urls.push(u);
    calls.bodies.push(init?.body ? JSON.parse(String(init.body)) : undefined);
    const json = (body: unknown) =>
      ({ ok: true, status: 200, headers: new Headers(), json: async () => body }) as unknown as Response;

    if (u.includes("/auth/me")) {
      return json({ user: { username: "u-1", display_name: "交付甲", role: "delivery_engineer",
                            industries: [], projects: [] },
                     max_security_level: 2, filter_expr: "security_level <= 2", header_identity_allowed: true });
    }
    if (u.includes("/healthz")) return json({ status: "ok", deps: { milvus: { ok: true, chunks: 193 } } });
    if (/\/api\/v1\/sessions\?/.test(u)) {
      return json({ total: 2, scope: { actor: "u-1", role: "delivery_engineer" }, items: [
        { id: "s-old", user_id: "u-1", n_turns: 2, title: "旧问题是什么？", last_status: "answer",
          created_at: "2026-10-13T10:00:00+08:00", updated_at: "2026-10-13T10:05:00+08:00",
          last_at: "2026-10-13T10:05:00+08:00" },
        { id: "s-empty", user_id: "u-1", n_turns: 0, title: null, last_status: null,
          created_at: "", updated_at: "", last_at: "" },
      ] });
    }
    if (/\/api\/v1\/sessions\//.test(u)) {
      return json({ session_id: "s-old", user_id: "u-1", turns: [
        { turn_no: 1, question: "旧问题是什么？", status: "answer", conclusion: "旧答案",
          retrieval_ms: 300, total_ms: 4100, trace_id: "tr-old", created_at: "2026-10-13T10:04:00+08:00",
          answer: { ...ANSWER, conclusion: "旧答案", session_id: "s-old" } },
        { turn_no: 2, question: "第二个旧问题？", status: "refuse", conclusion: "",
          retrieval_ms: 280, total_ms: null, trace_id: "tr-old2", created_at: "2026-10-13T10:05:00+08:00",
          answer: { ...ANSWER, status: "refuse", conclusion: "", key_points: [], citations: [],
                    session_id: "s-old" } },
      ] });
    }
    if (u.includes("/chunks/batch")) {
      const body = JSON.parse(String(init?.body));
      return json({ items: body.ids.map((id: string) => ({ chunk_id: id, content: `正文 ${id}` })),
                    hidden: [], requested: body.ids.length });
    }
    return sse([
      { event: "retrieval", data: { n_chunks: 5, chunk_ids: ["c1"], mode: "hybrid" } },
      { event: "answer", data: ANSWER },
    ]);
  }) as unknown as typeof fetch;
}

const renderApp = () =>
  render(
    <IdentityProvider>
      <ConversationProvider>
        <App />
      </ConversationProvider>
    </IdentityProvider>,
  );

const ask = (q: string) => {
  fireEvent.change(screen.getByRole("searchbox", { name: "问题" }), { target: { value: q } });
  fireEvent.click(screen.getByRole("button", { name: "提问" }));
};

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("问答记录住在哪一层", () => {
  it("切到别的模块再切回来，刚才那一问一答还在", async () => {
    const calls: Calls = { urls: [], bodies: [] };
    localStorage.setItem(TOKEN_KEY, "tok-1");
    stub(calls);
    renderApp();

    await screen.findByRole("tab", { name: "问答" });
    ask("SIS 表决逻辑是什么？");
    expect(await screen.findByTestId("conclusion-final")).toHaveTextContent("2oo3 表决");

    fireEvent.click(screen.getByRole("tab", { name: "知识库管理" }));
    fireEvent.click(screen.getByRole("tab", { name: "问答" }));

    // 这一句是本次改动的全部要点：旧版在这里会拿到空集，因为组件被卸载过
    expect(screen.getByTestId("conclusion-final")).toHaveTextContent("2oo3 表决");
    expect(screen.getAllByTestId("turn")).toHaveLength(1);
  });

  it("左侧历史栏列出自己的会话，点开就从服务端取回整包答案", async () => {
    const calls: Calls = { urls: [], bodies: [] };
    localStorage.setItem(TOKEN_KEY, "tok-1");
    stub(calls);
    renderApp();

    const item = await screen.findByTestId("session-s-old");
    expect(item.textContent).toContain("旧问题是什么？");
    expect(item.textContent).toContain("2 轮");
    fireEvent.click(item);

    expect(await screen.findByTestId("conclusion-final")).toHaveTextContent("旧答案");
    // 恢复的轮次必须带得出引用与规则分——只回结论等于把历史记录削薄了
    expect(screen.getByTestId("cite-E1")).toBeInTheDocument();
    expect(screen.getAllByTestId("confidence")[0]).toHaveTextContent("rule_composite_not_probability");
    expect(screen.getAllByTestId("turn")).toHaveLength(2);
    expect(screen.getAllByTestId("turn-restored")).toHaveLength(2);
    expect(calls.urls.some((u) => /\/api\/v1\/sessions\?/.test(u))).toBe(true);
    expect(calls.urls.some((u) => u.endsWith("/api/v1/sessions/s-old"))).toBe(true);
  });

  it("刷新后接回上次那一条会话（不是从空白开始）", async () => {
    const calls: Calls = { urls: [], bodies: [] };
    localStorage.setItem(TOKEN_KEY, "tok-1");
    localStorage.setItem(`${"kee.session"}:u-1`, "s-old");
    stub(calls);
    renderApp();

    expect(await screen.findByTestId("conclusion-final")).toHaveTextContent("旧答案");
    expect(calls.urls.some((u) => u.endsWith("/api/v1/sessions/s-old"))).toBe(true);
  });

  it("新会话清空正文，但历史列表仍在（清空是开一条新会话，不是删记录）", async () => {
    const calls: Calls = { urls: [], bodies: [] };
    localStorage.setItem(TOKEN_KEY, "tok-1");
    stub(calls);
    renderApp();

    fireEvent.click(await screen.findByTestId("session-s-old"));
    await screen.findByTestId("conclusion-final");
    fireEvent.click(screen.getByRole("button", { name: "新会话" }));

    expect(screen.queryByTestId("turn")).toBeNull();
    expect(screen.getByTestId("session-s-old")).toBeInTheDocument();
    expect(localStorage.getItem("kee.session:u-1")).toBeNull();
  });

  it("没有历史时如实说没有，不拿空列表冒充「已加载」", async () => {
    const calls: Calls = { urls: [], bodies: [] };
    localStorage.setItem(TOKEN_KEY, "tok-1");
    globalThis.fetch = (async (url: unknown) => {
      const u = String(url);
      calls.urls.push(u);
      const json = (body: unknown) =>
        ({ ok: true, status: 200, headers: new Headers(), json: async () => body }) as unknown as Response;
      if (u.includes("/auth/me")) {
        return json({ user: { username: "u-1", display_name: "甲", role: "pm", industries: [], projects: [] },
                      max_security_level: 3, filter_expr: "security_level <= 3", header_identity_allowed: true });
      }
      if (u.includes("/healthz")) return json({ status: "ok", deps: {} });
      if (/\/api\/v1\/sessions\?/.test(u)) return json({ total: 0, items: [], scope: { actor: "u-1", role: "pm" } });
      return json({});
    }) as unknown as typeof fetch;
    renderApp();

    expect(await screen.findByTestId("sessions-empty")).toHaveTextContent("还没有历史会话");
  });
});
