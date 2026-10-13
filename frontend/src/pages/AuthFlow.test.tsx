import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { IdentityProvider } from "../state/identity";
import { getToken, TOKEN_KEY } from "../api/auth";

/** 登录/注册与「未登录时界面只剩登录页」这条闸门。
 *
 * 这些断言要钉的是**入口**：旧版前端把角色做成下拉框，任何人随手就能把自己切成系统管理员，
 * 于是「RBAC 已实现」实际上只在服务端成立。改成真登录后闸门挪到了界面上，
 * 那就必须在界面上验证它——否则改的只是文案，不是行为。
 */

const ME = (role: string, username = "u-delivery-01") => ({
  user: { username, display_name: "交付工程师甲", role, industries: ["金融"], projects: [] },
  max_security_level: role === "newcomer" ? 1 : role === "pm" ? 3 : 2,
  filter_expr: `security_level <= ${role === "newcomer" ? 1 : 2}`,
  header_identity_allowed: true,
});

interface Stub {
  me?: { body: unknown; status?: number };
  login?: { body: unknown; status: number };
  register?: { body: unknown; status: number };
}

function stubFetch(stub: Stub = {}) {
  const calls: { url: string; body?: unknown; headers: Record<string, string> }[] = [];
  globalThis.fetch = (async (url: unknown, init?: RequestInit) => {
    const u = String(url);
    calls.push({
      url: u,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
      headers: (init?.headers ?? {}) as Record<string, string>,
    });
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, headers: new Headers(), json: async () => body }) as unknown as Response;

    if (u.includes("/auth/login")) {
      return stub.login ? json(stub.login.body, stub.login.status)
                        : json({ token: "tok-abc", expires_in_hours: 12, user: { username: "u-delivery-01" } });
    }
    if (u.includes("/auth/register")) {
      return stub.register ? json(stub.register.body, stub.register.status)
        : json({ user: ME("delivery_engineer").user, note: "注册即可登录；治理角色须由现有管理员授予" }, 201);
    }
    if (u.includes("/auth/me")) return json(stub.me?.body ?? ME("delivery_engineer"), stub.me?.status ?? 200);
    if (u.includes("/healthz")) {
      return json({ status: "ok", deps: { milvus: { ok: true, chunks: 100 }, postgres: { ok: true, documents: 10 } } });
    }
    return json({});
  }) as unknown as typeof fetch;
  return calls;
}

const renderApp = () =>
  render(
    <IdentityProvider>
      <App />
    </IdentityProvider>,
  );

const fill = (username: string, password: string) => {
  fireEvent.change(screen.getByLabelText("用户名"), { target: { value: username } });
  fireEvent.change(screen.getByLabelText("口令"), { target: { value: password } });
};

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("登录闸门", () => {
  it("没有有效会话时，界面上只有登录页——没有问答、没有知识库、没有审计", () => {
    stubFetch();
    renderApp();

    expect(screen.getByRole("tab", { name: "登录" })).toBeTruthy();
    expect(screen.getByLabelText("用户名")).toBeTruthy();
    expect(screen.queryByRole("tab", { name: "问答" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "知识库管理" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "审计" })).toBeNull();
    // 旧版的「挑个角色」下拉框必须真的没了，不是被藏在登录页下面
    expect(screen.queryByLabelText("角色")).toBeNull();
  });

  it("登录成功后进入系统：令牌落盘，身份取自服务端", async () => {
    const calls = stubFetch();
    renderApp();

    fill("u-delivery-01", "S3cret-Passw0rd");
    fireEvent.click(screen.getByRole("button", { name: "进入系统" }));

    expect(await screen.findByText("交付工程师甲")).toBeTruthy();
    expect(getToken()).toBe("tok-abc");
    expect(screen.getByRole("tab", { name: "问答" })).toBeTruthy();
    expect(screen.getByText("交付工程师")).toBeTruthy();

    const login = calls.find((c) => c.url.includes("/auth/login"));
    expect(login?.body).toEqual({ username: "u-delivery-01", password: "S3cret-Passw0rd" });
    // 进系统后的第一个请求要带 Bearer：否则「登录成功」只是前端自己记了一笔
    const me = calls.find((c) => c.url.includes("/auth/me"));
    expect(me?.headers["Authorization"]).toBe("Bearer tok-abc");
  });

  it("口令错：原样显示服务端那句不区分用户是否存在的话，且不写入令牌", async () => {
    stubFetch({ login: { status: 401, body: { code: "E401", message: "用户名或口令不正确", trace_id: "t1", retryable: false } } });
    renderApp();

    fill("ghost", "Wrong-Passw0rd!");
    fireEvent.click(screen.getByRole("button", { name: "进入系统" }));

    expect(await screen.findByTestId("auth-message")).toHaveTextContent("用户名或口令不正确");
    expect(getToken()).toBe("");
    expect(screen.queryByRole("tab", { name: "问答" })).toBeNull();
  });

  it("注册页不给治理角色可选项——能自授 knowledge_owner 就等于没有权限模型", () => {
    stubFetch();
    renderApp();
    fireEvent.click(screen.getByRole("tab", { name: "注册" }));

    const role = screen.getByLabelText("角色") as HTMLSelectElement;
    const options = Array.from(role.options).map((o) => o.textContent ?? "");
    expect(options.some((t) => t.includes("知识 Owner"))).toBe(false);
    expect(options.some((t) => t.includes("系统管理员"))).toBe(false);
    expect(options.some((t) => t.includes("新人"))).toBe(true);
    expect(screen.getByText(/须由现有管理员在「账号」页授予/)).toBeTruthy();
  });

  it("注册成功显示服务端的说明，并且不自动把人放进去登录", async () => {
    const calls = stubFetch();
    renderApp();
    fireEvent.click(screen.getByRole("tab", { name: "注册" }));

    fill("u-new-01", "N3w-Passw0rd!");
    fireEvent.click(screen.getByRole("button", { name: "注册账号" }));

    expect(await screen.findByTestId("auth-message")).toHaveTextContent("治理角色须由现有管理员授予");
    expect(getToken()).toBe("");
    expect(screen.queryByRole("tab", { name: "问答" })).toBeNull();
    expect(calls.find((c) => c.url.includes("/auth/register"))?.body)
      .toMatchObject({ username: "u-new-01", role: "delivery_engineer" });
  });

  it("服务端说会话已过期：清掉本地令牌、退回登录页", async () => {
    localStorage.setItem(TOKEN_KEY, "tok-stale");
    stubFetch({ me: { status: 401, body: { code: "E401", message: "未登录或会话已过期", trace_id: "", retryable: false } } });
    renderApp();

    await waitFor(() => expect(screen.getByLabelText("用户名")).toBeTruthy());
    expect(localStorage.getItem(TOKEN_KEY)).toBeNull();
  });
});

describe("账号 tab 的可见性", () => {
  const loginIn = async (role: string) => {
    stubFetch({ me: { body: ME(role) } });
    renderApp();
    fill("u-1", "Passw0rd-1234");
    fireEvent.click(screen.getByRole("button", { name: "进入系统" }));
    await screen.findByRole("tab", { name: "问答" });
  };

  it("普通角色看不到「账号」", async () => {
    await loginIn("delivery_engineer");
    expect(screen.queryByRole("tab", { name: "账号" })).toBeNull();
  });

  it("治理角色看得到「账号」", async () => {
    await loginIn("system_admin");
    expect(await screen.findByRole("tab", { name: "账号" })).toBeTruthy();
  });
});
