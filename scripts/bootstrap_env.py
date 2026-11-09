"""一次性环境搭建（可重复执行）：uv venv + 挂载本机 CUDA torch + 安装轻依赖。

之所以要挂 .pth 而不是 pip 装 torch：本机 8GB 显存的 CUDA 版 torch 已存在于
D:\\py312-torch（同一 CPython 3.12.8 底座），重复下载 GB 级轮子既慢又易与既有
DLL 冲突；而直接对 D:\\Python312 用 --system-site-packages 会继承到 CPU 版 torch。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
TORCH_BASE = Path("D:/py312-torch/Lib/site-packages")
PYTHON_VERSION = "3.12.8"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def sync_compose_env() -> int:
    """Compose 只读与 compose 文件同目录的 .env；根 .env 是唯一真源。"""
    src = ROOT / ".env"
    dst = ROOT / "infra" / ".env"
    if not src.exists():
        print("未找到 .env；先 cp .env.example .env 并填好所有 `待填_` 口令", file=sys.stderr)
        return 2
    text = src.read_text(encoding="utf-8")
    if "待填_" in text:
        print("警告：.env 仍有 `待填_` 项，Compose 起栈前先填掉", file=sys.stderr)
    dst.write_text(text, encoding="utf-8")
    print(f"已同步 {src} -> {dst}")
    return 0


def main() -> int:
    rc = sync_compose_env()
    if rc != 0:
        return rc

    if not VENV.exists():
        run(["uv", "venv", "--python", PYTHON_VERSION, str(VENV)])

    if not (VENV / "Lib" / "site-packages").is_dir():
        print("找不到 venv 的 site-packages", file=sys.stderr)
        return 2

    pth = VENV / "Lib" / "site-packages" / "zz_torch_base.pth"
    if not TORCH_BASE.is_dir():
        print(f"未找到本机 torch 环境：{TORCH_BASE}", file=sys.stderr)
        return 2
    # 追加在 venv 自身 site-packages 之后：项目内的 pin 优先于底座版本
    pth.write_text(TORCH_BASE.as_posix() + "\n", encoding="utf-8")

    run(["uv", "pip", "install", "-e", ".[dev]"])

    check = subprocess.run(
        [
            str(VENV / "Scripts" / "python.exe"),
            "-c",
            "import torch, transformers, numpy, pymilvus, jieba;"
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"
            "print('transformers', transformers.__version__, 'numpy', numpy.__version__);"
            "print('pymilvus', pymilvus.__version__)",
        ],
        cwd=ROOT,
    )
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
