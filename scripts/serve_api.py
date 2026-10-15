"""起宿主 API（一键部署的宿主侧那一半）。

为什么 API 跑在宿主而不是容器里：`pyproject.toml` 刻意不声明 torch / transformers，
它们由本机 `D:\\py312-torch` 的 CUDA 环境经 .pth 挂载提供，嵌入与重排要用宿主 8GB 显存。
容器化它要么拉 GB 级轮子（违反本机「大件不重复下载」），要么在容器里再要一份 CUDA 运行时。
所以部署拓扑是「存储 + 前端进容器、API 在宿主、Nginx 经 host-gateway 反代」，
这条偏离已登记待裁决，见 `infra/docker-compose.m4.yml` 的注释。

跑法：
    uv run python scripts/serve_api.py                 # 127.0.0.1:38000
    KEE_API_PORT=38000 uv run python scripts/serve_api.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import uvicorn  # noqa: E402

from kee.api.app import create_app  # noqa: E402

HOST = os.environ.get("KEE_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("KEE_API_PORT", "38000"))


def preflight() -> str | None:
    """启动前问一次库：**配置错**要当场说清，别让人去读 psycopg 的原始报错。

    收紧权限之后（2026-10-15），新库最常见的失败是「`kee_app` 这个角色还不存在」——
    它的真实原因是没跑 `scripts/harden_audit_store.py`，但 PG 只会回一句
    `role "kee_app" does not exist`，看上去像密码错或库坏了。

    只在**配置类**错误上拒绝启动。连不上（存储栈还没起来）继续放行：
    `/healthz` 会如实报 postgres 不健康，前端会显示降级——那是设计好的行为，不是故障。
    """
    import psycopg

    dsn = ""
    try:
        from kee.config import get_settings

        dsn = get_settings().postgres_dsn
        with psycopg.connect(dsn, connect_timeout=5):
            return None
    except psycopg.Error as e:
        text = str(e)
        if "does not exist" in text and "role" in text.lower():
            return (f"[拒绝启动] 数据库角色不存在。{text.strip().splitlines()[0]}\n"
                    "  当前 DSN 的角色：" + (dsn.split("@")[0].split("//")[-1] or "(空)") + "\n"
                    "  收紧之后应用连的是非特权角色 `kee_app`，它由脚本创建：\n"
                    "      uv run python scripts/harden_audit_store.py --confirm=HARDEN\n"
                    "  （幂等；只想看当前权限面用 `--check`。DBA 入口是 `kee_admin`，"
                    "口令在 var/pg_admin_credentials.json，不入库。）")
        if "password authentication failed" in text or "no pg_hba.conf entry" in text:
            return (f"[拒绝启动] 认证被拒。{text.strip().splitlines()[0]}\n"
                    "  先确认存储栈是 `infra/docker-compose.gate0.yml` 起的那套，"
                    "再跑一次 harden_audit_store.py（它会建角色并授权）。")
        return None      # 数据库不存在等：交给下面的通用依赖告警，不在这儿猜
    except Exception as e:  # noqa: BLE001 - 探测本身不该拦住启动
        print(f"  [跳过自检] 库没探成（{type(e).__name__}: {str(e)[:90]}）")
        return None


def main() -> int:
    why = preflight()
    if why:
        print(why)
        return 3
    print(f"KEE API  http://{HOST}:{PORT}")
    print("  身份只认登录令牌（/api/v1/auth/*）。请求头身份默认关闭："
          "内网工具确需该通道时显式设 KEE_ALLOW_HEADER_IDENTITY=true。")
    print(f"  前端：docker compose -f infra/docker-compose.m4.yml up -d"
          f"（API 不在 38000 时同批传 KEE_API_PORT={PORT}，否则 38080 反代 502）")
    # 绑定地址只允许回环：非回环绑定等于把可伪造身份的接口开放给局域网
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        print(f"[拒绝启动] KEE_API_HOST={HOST} 不是回环地址；数据不出域是硬约束。")
        return 2
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
