import { describe, expect, it } from "vitest";
import { postSse } from "./sse";
import type { Identity } from "./client";

const IDENTITY: Identity = { role: "pm", actor: "tester", industries: [], projects: [] };

/** 用给定的字符串片段拼出一条 SSE 响应体。
 * 故意把帧**切在中间**（见 split 用例）：分帧解析的 bug 全在边界上，
 * 一次 enqueue 一整帧的测试等于没测。 */
function streamOf(chunks: string[]) {
  const enc = new TextEncoder();
  return {
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "text/event-stream" }),
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const c of chunks) controller.enqueue(enc.encode(c));
        controller.close();
      },
    }),
  };
}

function stubFetch(chunks: string[], capture?: { headers: Record<string, string> }) {
  globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
    if (capture && init?.headers) capture.headers = init.headers as Record<string, string>;
    return streamOf(chunks) as unknown as Response;
  }) as typeof fetch;
}

const originalFetch = globalThis.fetch;

describe("postSse 分帧", () => {
  it("帧被切在中间也能正确解析出两条事件", async () => {
    const events: { event: string; data: string }[] = [];
    stubFetch(['event: retrieval\ndata: {"n_chunks":', "5}\n\nevent: delta\ndata: {\"text\":\"ab\"}\n\n"]);
    await postSse("/api/v1/answers/stream", { identity: IDENTITY, body: {}, onEvent: (e) => events.push(e) });
    globalThis.fetch = originalFetch;
    expect(events).toEqual([
      { event: "retrieval", data: '{"n_chunks":5}' },
      { event: "delta", data: '{"text":"ab"}' },
    ]);
  });

  it("空行是帧边界：正文里的换行在 JSON 内已转义，不会被当成帧尾", async () => {
    const events: { event: string; data: string }[] = [];
    stubFetch(['event: delta\ndata: {"text":"第一行\\n第二行"}\n\nevent: answer\ndata: {"status":"answer"}\n\n']);
    await postSse("/x", { identity: IDENTITY, body: {}, onEvent: (e) => events.push(e) });
    globalThis.fetch = originalFetch;
    expect(events).toHaveLength(2);
    expect(JSON.parse(events[0].data).text).toBe("第一行\n第二行");
  });

  it("多条 data 行按规范用换行拼接；冒号开头的心跳注释被忽略", async () => {
    const events: { event: string; data: string }[] = [];
    stubFetch([": ping\n\nevent: delta\ndata: a\ndata: b\n\nevent: delta\ndata: \n\n"]);
    await postSse("/x", { identity: IDENTITY, body: {}, onEvent: (e) => events.push(e) });
    globalThis.fetch = originalFetch;
    // 规范：一条 `data:`（哪怕值为空）都会派发事件；注释行不产生任何事件。
    // 空载荷由调用方判成错误，不在解析层偷偷丢掉——丢了就等于「这条事件没来过」，界面会停在半路。
    expect(events).toEqual([
      { event: "delta", data: "a\nb" },
      { event: "delta", data: "" },
    ]);
  });

  it("CRLF 分帧同样可用", async () => {
    const events: { event: string; data: string }[] = [];
    stubFetch(["event: delta\r\ndata: {\"k\":1}\r\n\r\n"]);
    await postSse("/x", { identity: IDENTITY, body: {}, onEvent: (e) => events.push(e) });
    globalThis.fetch = originalFetch;
    expect(events).toEqual([{ event: "delta", data: '{"k":1}' }]);
  });

  it("POST + 身份头 + JSON 体：EventSource 办不到的三件事都在这条路上", async () => {
    const capture = { headers: {} as Record<string, string> };
    stubFetch(["event: answer\ndata: {}\n\n"], capture);
    await postSse("/api/v1/answers/stream", {
      identity: { role: "system_admin", actor: "a@b", industries: ["金融"], projects: ["PRJ-X"] },
      body: { question: "q" },
      onEvent: () => {},
    });
    globalThis.fetch = originalFetch;
    expect(capture.headers).toMatchObject({
      "X-Kee-Role": "system_admin",
      // actor 同样要编码：中文姓名（张三）在这里和中文部门名（金融）是同一个问题
      "X-Kee-Actor": encodeURIComponent("a@b"),
      // 中文部门名**必须以百分号编码发出**：直接发中文，浏览器会拒发整个请求。
      // 这条断言就是那条契约的护栏。
      "X-Kee-Industries": encodeURIComponent("金融"),
      "X-Kee-Projects": "PRJ-X",
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    });
    expect(capture.headers["X-Kee-Industries"]).not.toContain("金融");
  });

  it("依赖挂了要在建流之前抛出契约错误，而不是让界面停在「收到一半」", async () => {
    globalThis.fetch = (async () => ({
      ok: false,
      status: 503,
      headers: new Headers(),
      json: async () => ({ code: "E902", message: "向量库不可用", trace_id: "t1", retryable: true }),
    })) as unknown as typeof fetch;
    await expect(
      postSse("/x", { identity: IDENTITY, body: {}, onEvent: () => {} }),
    ).rejects.toMatchObject({ code: "E902", retryable: true });
    globalThis.fetch = originalFetch;
  });
});
