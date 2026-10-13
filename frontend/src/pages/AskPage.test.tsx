import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AskPage } from "./AskPage";
import { IdentityProvider } from "../state/identity";

const enc = new TextEncoder();

interface Frame {
  event: string;
  data: Record<string, unknown>;
}

function sseBody(frames: Frame[]) {
  const text = frames.map((f) => `event: ${f.event}\ndata: ${JSON.stringify(f.data)}\n\n`).join("");
  return {
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "text/event-stream" }),
    body: new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(enc.encode(text));
        c.close();
      },
    }),
  };
}

const ANSWER_BASE = {
  question: "q",
  status: "answer",
  conclusion: "",
  key_points: [],
  citations: [],
  dropped_points: [],
  needs_human: false,
  confidence: {},
  retrieval: {},
  llm: {},
  timings_ms: {},
  trace_id: "t",
  notes: [],
  session_id: "s-1",
  turn_no: 1,
  provisional_streamed_chars: 0,
  provisional_retracted: false,
};

/** 装一个按 URL 分派的 fetch。返回每次调用的 URL 列表，供断言「有没有走带权限过滤的读通道」。 */
function stubFetch(frames: Frame[], answerOverrides: Record<string, unknown> = {}) {
  const called: string[] = [];
  let chunkBatchIds: string[] | null = null;
  globalThis.fetch = (async (url: unknown, init?: RequestInit) => {
    const u = String(url);
    called.push(u);
    if (u.includes("/chunks/batch")) {
      const body = JSON.parse(String(init?.body));
      chunkBatchIds = body.ids;
      return {
        ok: true,
        status: 200,
        headers: new Headers(),
        json: async () => ({
          items: body.ids.map((id: string) => ({ chunk_id: id, content: `片段正文 for ${id}` })),
          hidden: [],
          requested: body.ids.length,
        }),
      } as unknown as Response;
    }
    return {
      ...sseBody(frames),
      // answer 事件里塞进覆盖字段（撤回率、状态这些就是靠它驱动的）
      ...(answerOverrides
        ? {
            body: new ReadableStream<Uint8Array>({
              start(c) {
                const text = frames
                  .map((f) => {
                    const data = f.event === "answer" ? { ...f.data, ...answerOverrides } : f.data;
                    return `event: ${f.event}\ndata: ${JSON.stringify(data)}\n\n`;
                  })
                  .join("");
                c.enqueue(enc.encode(text));
                c.close();
              },
            }),
          }
        : {}),
    } as unknown as Response;
  }) as unknown as typeof fetch;
  return { called, ids: () => chunkBatchIds };
}

const renderPage = () =>
  render(
    <IdentityProvider>
      <AskPage />
    </IdentityProvider>,
  );

const ask = (q = "表决逻辑是什么？") => {
  fireEvent.change(screen.getByRole("searchbox", { name: "问题" }), { target: { value: q } });
  fireEvent.click(screen.getByRole("button", { name: "提问" }));
};

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("问答页的三态渲染", () => {
  it("provisional 增量与最终答案是**不同的样式类**，且结束后不留未完成态", async () => {
    const frames = [
      { event: "retrieval", data: { n_chunks: 5, chunk_ids: ["c1"], mode: "hybrid" } },
      // 同一槽位连发两条：第二条是累计全文，界面必须替换而不是拼接（拼接会出重复字）
      { event: "delta", data: { field: "conclusion", index: null, text: "2oo3", closed: false, provisional: true, chars: 4 } },
      { event: "delta", data: { field: "conclusion", index: null, text: "2oo3 表决逻辑", closed: false, provisional: true, chars: 9 } },
      { event: "answer", data: ANSWER_BASE },
    ];
    stubFetch(frames, { conclusion: "2oo3 表决逻辑", provisional_streamed_chars: 9 });
    renderPage();
    ask();

    const final = await screen.findByTestId("conclusion-final");
    expect(final.className).toContain("kee-state--confirmed");
    expect(final.textContent).toBe("2oo3 表决逻辑");
    // 累计替换而非拼接：文本里不该出现「2oo32oo3」
    expect(final.textContent).not.toContain("2oo32oo3");
    expect(document.querySelector(".kee-state--provisional")).toBeNull();
  });

  it("流出去却没成为答案的文字：标成已撤回、**原文还在**，不静默删除", async () => {
    const frames = [
      { event: "retrieval", data: { n_chunks: 5 } },
      { event: "delta", data: { field: "conclusion", index: null, text: "这段后来被撤掉的话", closed: false, provisional: true, chars: 10 } },
      { event: "answer", data: ANSWER_BASE },
    ];
    stubFetch(frames, { status: "refuse", conclusion: "", provisional_streamed_chars: 10, provisional_retracted: true });
    renderPage();
    ask();

    const retracted = await screen.findByTestId("conclusion-retracted");
    expect(retracted.className).toContain("kee-state--retracted");
    expect(retracted.textContent).toBe("这段后来被撤掉的话");
    expect(await screen.findByTestId("retraction-notice")).toHaveTextContent("已撤回");
    expect(await screen.findByTestId("retraction-notice")).toHaveTextContent("不静默删除");
  });

  it("规则分要带 score_basis 原样出现，不能显示成概率", async () => {
    const frames = [
      { event: "retrieval", data: { n_chunks: 5 } },
      { event: "answer", data: { ...ANSWER_BASE, conclusion: "结论", confidence: { composite: 0.7813, calibrated: 0.7813, temperature: 1, score_basis: "rule_composite_not_probability", retrieval: 0.9, citation_consistency: 1, self_assessed: 0.9 } } },
    ];
    stubFetch(frames);
    renderPage();
    ask();

    const conf = await screen.findByTestId("confidence");
    expect(conf.textContent).toContain("rule_composite_not_probability");
    expect(conf.textContent).toContain("0.7813");
    expect(conf.textContent).not.toMatch(/置信度\s*78%/);
  });

  it("degraded 是**空数组**时不许显示「检索降级」——JS 里 [] 是真值，这类假信号等于永远在报降级", async () => {
    stubFetch([
      { event: "retrieval", data: { n_chunks: 5, mode: "hybrid", degraded: [] } },
      { event: "answer", data: { ...ANSWER_BASE, conclusion: "结论" } },
    ]);
    renderPage();
    ask();
    const skel = await screen.findByTestId("retrieval-skeleton");
    expect(skel.textContent).not.toContain("检索降级");
  });

  it("真降级时要把**原因**显示出来，只写「检索降级」等于把可行动的信息压成一个灯", async () => {
    stubFetch([
      { event: "retrieval", data: { n_chunks: 5, mode: "hybrid", degraded: ["dense_timeout>700ms"] } },
      { event: "answer", data: { ...ANSWER_BASE, conclusion: "结论" } },
    ]);
    renderPage();
    ask();
    const skel = await screen.findByTestId("retrieval-skeleton");
    expect(skel.textContent).toContain("检索降级");
    expect(skel.textContent).toContain("dense_timeout");
  });

  it("拒答轮也要带 score_basis：0 分最容易被读成概率，缺这句说明就最危险", async () => {
    stubFetch([
      { event: "retrieval", data: { n_chunks: 3 } },
      { event: "answer", data: { ...ANSWER_BASE, status: "refuse", conclusion: "", key_points: [], citations: [],
        confidence: { composite: 0, calibrated: 0, temperature: 1, retrieval: 0.0073,
                      score_basis: "rule_composite_not_probability", rule: "retrieval_gate" } } },
    ]);
    renderPage();
    ask();
    const conf = await screen.findByTestId("confidence");
    expect(conf.textContent).toContain("rule_composite_not_probability");
    expect(conf.textContent).not.toContain("缺失");
  });

  it("needs_human 与拒答都要显式可见；拒答时结论区不为空话", async () => {
    const frames = [
      { event: "retrieval", data: { n_chunks: 3 } },
      { event: "answer", data: { ...ANSWER_BASE, status: "escalate", conclusion: "", needs_human: true, notes: ["强证据但按密级转人工"] } },
    ];
    stubFetch(frames);
    renderPage();
    ask();

    expect(await screen.findByTestId("status-badge")).toHaveTextContent("已转人工");
    expect(await screen.findByTestId("needs-human")).toHaveTextContent("需人工复核");
    expect(await screen.findByTestId("no-conclusion")).toHaveTextContent("本轮没有可给出的结论");
  });

  it("引用片段正文走 /api/v1/chunks/batch——前端不自己拼正文，且带的是引用里的 chunk id", async () => {
    const frames = [
      { event: "retrieval", data: { n_chunks: 5 } },
      {
        event: "answer",
        data: {
          ...ANSWER_BASE,
          conclusion: "结论",
          key_points: [{ point: "要点一", evidence_ids: ["E1"] }],
          citations: [{ evidence_id: "E1", chunk_id: "doc-a@v1#1.1#c1", doc_id: "doc-a", section_no: "1.1", page: 3, security_level: 2 }],
        },
      },
    ];
    const stub = stubFetch(frames);
    renderPage();
    ask();

    const cite = await screen.findByTestId("cite-E1");
    expect(cite.textContent).toContain("doc-a");
    expect(cite.textContent).toContain("第 3 页");
    // 片段正文是第二次异步请求的结果，必须等它——直接断言会在竞态上偶尔变绿、偶尔变红
    await waitFor(() => expect(cite.textContent).toContain("片段正文 for doc-a@v1#1.1#c1"));
    expect(stub.ids()).toEqual(["doc-a@v1#1.1#c1"]);
  });

  it("越权取不到的片段：说「取不到」而不是编一段正文顶上", async () => {
    globalThis.fetch = (async (url: unknown) => {
      if (String(url).includes("/chunks/batch")) {
        return {
          ok: true,
          status: 200,
          headers: new Headers(),
          json: async () => ({ items: [], hidden: ["doc-b@v1#2.2#c1"], requested: 1 }),
        } as unknown as Response;
      }
      return sseBody([
        { event: "retrieval", data: { n_chunks: 5 } },
        {
          event: "answer",
          data: {
            ...ANSWER_BASE,
            conclusion: "结论",
            citations: [{ evidence_id: "E1", chunk_id: "doc-b@v1#2.2#c1", doc_id: "doc-b", section_no: "2.2", page: 9, security_level: 3 }],
          },
        },
      ]) as unknown as Response;
    }) as unknown as typeof fetch;
    renderPage();
    ask();

    const cite = await screen.findByTestId("cite-E1");
    // hidden 是异步落定的：不等它就会读到「正文读取中…」那一帧，这条曾经 12 次里红 2 次
    await waitFor(() => expect(cite.textContent).toContain("当前身份取不到该片段正文"));
    // 含糊的那句「正文未返回」不该出现在已落定的越权分支里——它把「没权限」说成了「没数据」
    expect(cite.textContent).not.toContain("正文未返回");
  });

  it("流结束了却没有 answer：明确报错，不假装成功", async () => {
    globalThis.fetch = (async () =>
      sseBody([{ event: "retrieval", data: { n_chunks: 5 } }, { event: "delta", data: { field: "conclusion", index: null, text: "半句话", closed: false, provisional: true, chars: 3 } }]) as unknown as Response) as unknown as typeof fetch;
    renderPage();
    ask();

    const err = await screen.findByTestId("turn-error");
    expect(err.textContent).toContain("未收到 answer");
  });
});

describe("一问一答的版式（正文在上、输入在下，历史在正文区内部滚）", () => {
  it("提问栏排在正文容器之后：DOM 顺序即视觉顺序，滚动的是 .kee-turns 而不是整页", () => {
    stubFetch([]);
    renderPage();
    const scroller = document.querySelector(".kee-turns") as HTMLElement;
    const dock = document.querySelector(".kee-ask-dock") as HTMLElement;
    expect(scroller).toBeTruthy();
    expect(dock).toBeTruthy();
    // 「下面」在测试里只能靠 DOM 关系证明：紧随正文之后、且输入框确实在这一坨里
    expect(scroller.compareDocumentPosition(dock) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(dock.contains(screen.getByRole("searchbox", { name: "问题" }))).toBe(true);
  });

  it("每一问配一条答案气泡，追问不覆盖前答", async () => {
    stubFetch([
      { event: "retrieval", data: { n_chunks: 5 } },
      { event: "answer", data: { ...ANSWER_BASE, conclusion: "第一答", turn_no: 1 } },
    ]);
    renderPage();
    ask("第一问？");
    await screen.findByTestId("conclusion-final");

    stubFetch([
      { event: "retrieval", data: { n_chunks: 4 } },
      { event: "answer", data: { ...ANSWER_BASE, conclusion: "第二答", turn_no: 2 } },
    ]);
    ask("第二问？");
    await waitFor(() => expect(screen.getAllByTestId("conclusion-final")).toHaveLength(2));

    const pairs = screen.getAllByTestId("turn");
    expect(pairs).toHaveLength(2);
    pairs.forEach((pair) => {
      // 一问一答：一个 .kee-turn 里恰好一条问题、一条答案，成对而不是两叠列表
      expect(pair.querySelectorAll(".kee-turn__question")).toHaveLength(1);
      expect(pair.querySelectorAll(".kee-turn__answer")).toHaveLength(1);
    });
    expect(screen.getAllByTestId("conclusion-final").map((n) => n.textContent)).toEqual(["第一答", "第二答"]);
  });
});
