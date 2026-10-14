/** 状态条的两条纪律：① 依赖挂了不能一片绿；② 语料规模如实报数，且数字从 /healthz 取。
 *
 * 第②条钉的是**空索引那一格**：库里 0 块时不许显示成"就绪"——那会把"没建库"念成"建好了只是没内容"。
 * 界面上只报规模与状态，不对数据来源下判断（来源与适用范围是文档层的事）。
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
  it("依赖就绪时报出语料规模（数字取自 /healthz，不写死）", async () => {
    stubHealth({ status: "ok", deps: { milvus: { ok: true, chunks: 193 }, postgres: { ok: true, documents: 24 } } });
    render(<HealthStrip />);
    const badge = await screen.findByTestId("corpus-source");
    expect(badge.textContent).toContain("193");
    expect(badge.textContent).toContain("24");
    expect(badge.textContent).not.toContain("索引为空");
  });

  it("索引为空时必须明说为空——不许把『没建库』念成『建好了只是没内容』", async () => {
    stubHealth({ status: "ok", deps: { milvus: { ok: true, chunks: 0 }, postgres: { ok: true, documents: 0 } } });
    render(<HealthStrip />);
    const badge = await screen.findByTestId("corpus-source");
    expect(badge.textContent).toContain("索引为空或依赖未就绪");
    expect(badge.textContent).toContain("0 块 / 0 篇");
  });

  it("milvus 挂了：HTTP 200 也不算就绪", async () => {
    stubHealth({ status: "degraded", deps: { milvus: { ok: false, error: "connect refused" } } });
    render(<HealthStrip />);
    await waitFor(() => expect(screen.getByText(/有降级/)).toBeInTheDocument());
    expect(screen.getByText(/connect refused/)).toBeInTheDocument();
  });
});
