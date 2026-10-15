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


def main() -> int:
    print(f"KEE API  http://{HOST}:{PORT}")
    print("  身份只认登录令牌（/api/v1/auth/*）。请求头身份默认关闭："
          "内网工具确需该通道时显式设 KEE_ALLOW_HEADER_IDENTITY=true。")
    print("  前端：docker compose -f infra/docker-compose.m4.yml up -d  →  http://127.0.0.1:38080/")
    # 绑定地址只允许回环：非回环绑定等于把可伪造身份的接口开放给局域网
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        print(f"[拒绝启动] KEE_API_HOST={HOST} 不是回环地址；数据不出域是硬约束。")
        return 2
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
