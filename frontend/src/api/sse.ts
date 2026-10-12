/** SSE 客户端：POST + 自定义身份头，所以**用不了 EventSource**（它不支持 POST，也不让带头）。
 *
 * 自己按 SSE 规范分帧：帧之间空行分隔，帧内 `event:` / `data:` 各行，
 * 多条 `data:` 用换行拼接，冒号开头的行是注释（心跳），要忽略。
 * 一个已知坑：后端把载荷当 JSON 发，换行已在 JSON 字符串内转义，
 * 所以「正文里有空行」不会提前切断帧——但如果哪天改成明文发就会切，
 * 因此这里按规范实现，而不是简单 split('\n\n') 之后只看第一行。
 */

import { identityHeaders, parseErrorResponse, type Identity } from "./client";

export interface SseEvent {
  event: string;
  /** 原始字符串；由调用方决定怎么解析（本项目的 data 一律是 JSON） */
  data: string;
}

export interface SseOptions {
  identity: Identity;
  body: unknown;
  signal?: AbortSignal;
  onEvent: (e: SseEvent) => void;
}

export async function postSse(path: string, opts: SseOptions): Promise<void> {
  const resp = await fetch(path, {
    method: "POST",
    headers: {
      ...identityHeaders(opts.identity),
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(opts.body),
    signal: opts.signal,
  });
  // 依赖挂时后端在**建立流之前**就返回 503 + 契约错误体；这段必须走错误路径，
  // 否则前端会当成「流已经开始了、只是没有事件」而永远等下去。
  if (!resp.ok) throw await parseErrorResponse(resp);
  if (!resp.body) throw new Error("浏览器未提供响应体流，SSE 无法进行");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";

  const flush = (frame: string) => {
    const lines = frame.split(/\r?\n/);
    let event = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (!line || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      // 规范：冒号后紧跟的一个空格属于分隔符，不属于值
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") dataLines.push(value);
    }
    if (dataLines.length) opts.onEvent({ event, data: dataLines.join("\n") });
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.search(/\r?\n\r?\n/)) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + buf.slice(idx).match(/^\r?\n\r?\n/)![0].length);
      flush(frame);
    }
  }
  // 流结束时残留的帧也要交付：服务器可能最后一帧之后没补空行就关了连接
  if (buf.trim()) flush(buf);
}
