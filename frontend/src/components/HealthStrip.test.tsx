/** 状态条的两条纪律：① 依赖挂了不能一片绿；② 语料来源要说实话，且数字从 /healthz 取。
 *
 * 第②条是 2026-10-13 措辞改判时一并钉的：把「Demo」换成「内网部署版」是**呈现的更正**，
 * 但「当前是模拟语料、未接生产文档」是**部署事实**，不是一句谦虚——
 * 删掉它的瞬间，界面就把「能跑」说成了「可以拿去用」。所以这里同时测正例与空索引那一路。
 */

import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HealthStrip } from "./HealthStrip";

function stubHealth(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, json: async () => body,
  })) as unknown as typeof fetch);
}

beforeEach(() => vi.unstubAllGlobals());

describe("HealthStrip", () => {
  it("依赖就绪时报出语料规模，并写明是模拟语料", async () => {
    stubHealth({ status: "ok", deps: { milvus: { ok: true, chunks: 193 }, postgres: { ok: true, documents: 24 } } });
    render(<HealthStrip />);
    const badge = await screen.findByTestId("corpus-source");
    expect(badge.textContent).toContain("193");
    expect(badge.textContent).toContain("24");
    expect(badge.textContent).toContain("未接入生产文档");
  });

  it("索引为空时不许继续说『模拟语料』——那会把『没建库』念成『建了个演示库』", async () => {
    stubHealth({ status: "ok", deps: { milvus: { ok: true, chunks: 0 }, postgres: { ok: true, documents: 0 } } });
    render(<HealthStrip />);
    const badge = await screen.findByTestId("corpus-source");
    expect(badge.textContent).toContain("索引为空或依赖未就绪");
    expect(badge.textContent).not.toContain("未接入生产文档");
  });

  it("milvus 挂了：HTTP 200 也不算就绪", async () => {
    stubHealth({ status: "degraded", deps: { milvus: { ok: false, error: "connect refused" } } });
    render(<HealthStrip />);
    await waitFor(() => expect(screen.getByText(/有降级/)).toBeInTheDocument());
    expect(screen.getByText(/connect refused/)).toBeInTheDocument();
  });
});
