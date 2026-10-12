"""M5-02 版本锁定清单：把「这台机器上跑出来的结果」换成「换台机器也该一样」的前提。

原则：**取不到就标 `未固定` 并写原因**，不留空、不假装锁定。凡 `未固定` 的项都是可移植性缺口，
M6 部署前必须逐条销掉或明写「接受此偏离」。

跑法：uv run python scripts/m5_version_lock.py
产物：var/gate0/m5_versions.json + reports/m5/M5_版本锁定清单.md
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "var" / "gate0"
OUT_MD = ROOT / "reports" / "m5" / "M5_版本锁定清单.md"


def sh(cmd: list[str], timeout: int = 120) -> str:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"__ERR__ {type(e).__name__}: {e}"


def sha256_head(path: Path, n: int = 12) -> str:
    if not path.exists():
        return "缺文件"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return "未安装"


def model_dir(tag: str, path: str) -> dict:
    """权重目录：**只登记文件清单哈希**，不把 GB 级权重塞进仓库。
    目录里文件多，全量 sha 太慢，取「文件名 + 大小 + mtime」的聚合摘要——足够判断换机后是不是同一份。"""
    p = Path(path)
    if not p.exists():
        return {"项": tag, "路径": path, "状态": "缺目录", "摘要": "未固定"}
    files = [f for f in p.rglob("*") if f.is_file()]
    agg = hashlib.sha256("\n".join(sorted(f"{f.name}:{f.stat().st_size}" for f in files)).encode()).hexdigest()[:12]
    return {"项": tag, "路径": path, "状态": f"{len(files)} 个文件 / {round(sum(f.stat().st_size for f in files) / 1e9, 2)} GB",
            "摘要": agg}


def main() -> int:
    uv_exe = sh(["uv", "--version"])
    docker_ver = sh(["docker", "--version"])
    compose_ver = sh(["docker", "compose", "version", "--short"])
    images = sh(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"])
    server_milvus = "未连" 
    server_pg = "未连"
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from kee.config import get_settings
        from pymilvus import MilvusClient
        import psycopg

        s = get_settings()
        c = MilvusClient(uri=s.milvus_uri, timeout=15)
        server_milvus = c.get_server_version()
        c.close()
        with psycopg.connect(s.postgres_dsn) as conn:
            server_pg = conn.execute("SHOW server_version").fetchone()[0]
    except Exception as e:  # noqa: BLE001
        server_milvus = server_pg = f"取不到（{type(e).__name__}）——存储栈没在跑就取不到，这本身就是清单的一部分"

    settings_keys = ["KEE_BGE_M3_PATH", "KEE_BGE_RERANKER_PATH", "KEE_LLM_MODEL", "KEE_OLLAMA_URL",
                     "KEE_MILVUS_URI", "KEE_POSTGRES_DSN", "KEE_REDIS_URL", "KEE_CLOUD_FALLBACK_ENABLED",
                     "KEE_RRF_K", "KEE_DENSE_TOP_K", "KEE_SPARSE_TOP_K", "KEE_RERANK_TOP_N", "KEE_REWRITE_MODE"]
    env_file = ROOT / ".env"
    env_present = {k: (k in env_file.read_text(encoding="utf-8")) for k in settings_keys} if env_file.exists() else {}

    ollama_models = sh(["ollama", "list"])
    ollama_lines = [l.split() for l in ollama_models.splitlines()[1:]] if "__ERR__" not in ollama_models else []

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "principle": "取不到就标 未固定 并写原因；未固定项 = 可移植性缺口，M6 前逐条销掉或明写接受",
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_prefix": sys.prefix,
            "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
            "shell": os.environ.get("SHELL", ""),
        },
        "toolchain": {"uv": uv_exe, "docker": docker_ver, "docker_compose": compose_ver,
                      "node": sh(["node", "--version"]), "npm": sh(["npm", "--version"])},
        "python_lock": {"file": "uv.lock", "sha256_head": sha256_head(ROOT / "uv.lock"),
                        "pyproject": sha256_head(ROOT / "pyproject.toml")},
        "key_packages": {k: pkg(k) for k in ["fastapi", "uvicorn", "pydantic", "pydantic-settings", "psycopg",
                                             "pymilvus", "redis", "langgraph", "httpx", "numpy", "sentence-transformers",
                                             "torch", "transformers", "pytest"]},
        "not_declared_in_pyproject": {
            "torch / transformers / sentence-transformers":
                "**pyproject 刻意不声明**，由本机 D:\\py312-torch 的 CUDA 环境经 .pth 挂载提供——"
                "这是 ADR-12（API 未容器化）的根因，也是这张清单上最大的可移植性缺口。"},
        "frontend_lock": {"file": "frontend/package-lock.json", "sha256_head": sha256_head(ROOT / "frontend" / "package-lock.json"),
                          "vite": (ROOT / "frontend" / "package.json").exists()},
        "containers": {"images": images.splitlines(), "compose_files": {
            f: sha256_head(ROOT / "infra" / f) for f in
            ["docker-compose.gate0.yml", "docker-compose.m4.yml", "docker-compose.gate0.iso.yml",
             "docker-compose.gate0.nodev.yml"]},
            "nginx_conf": sha256_head(ROOT / "infra" / "nginx.m4.conf")},
        "servers": {"milvus": server_milvus, "postgres": server_pg},
        "models": [
            model_dir("BGE-M3（稠密+稀疏嵌入）", os.environ.get("KEE_BGE_M3_PATH", "D:/modelscope/hub/models/BAAI--bge-m3/snapshots/master")),
            model_dir("BGE-reranker-v2-m3（重排）", os.environ.get("KEE_BGE_RERANKER_PATH", "D:/modelscope/hub/models/BAAI--bge-reranker-v2-m3/snapshots/master")),
            model_dir("MinerU 4.x ONNX（版面+OCR+表格）", "D:/modelscope/hub/models/MinerU-4_models_onnx"),
        ],
        "ollama": {"raw": ollama_models, "digest_fixed": False,
                   "note": "`ollama list` 不输出权重 sha256 ⇒ **生成模型未被真正锁定**。"
                           "同 tag 被 `ollama pull` 覆盖后本机不会报错，换机可能拿到不同权重。M6 前要用 `ollama show --format json` 补 digest。"},
        "dtype": os.environ.get("KEE_MODEL_DTYPE", "float16（.env 未覆盖时取 config 默认）"),
        "device": os.environ.get("KEE_DEVICE", "cuda（.env 未覆盖时取 config 默认）"),
        "env_keys": env_present,
        "unfixed": [],
    }
    if not env_present:
        rep["unfixed"].append(".env 缺失：所有 KEE_* 取值未固定")
    if "__ERR__" in ollama_models:
        rep["unfixed"].append("Ollama 模型清单取不到（服务未在跑）")
    rep["unfixed"] += ["生成模型 qwen2.5:7b 无 digest（见 ollama 项）",
                       "torch/transformers 未进 pyproject 依赖声明（见 not_declared_in_pyproject）",
                       "宿主 OS 补丁级别未登记（Windows 11 26200 只到 build 号）"]

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "m5_versions.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    L = ["# M5 · 版本锁定清单", "",
         f"> 生成时点：{rep['generated_at']}｜由 `scripts/m5_version_lock.py` 实跑取值。",
         "> 原则：**取不到就写 `未固定` + 原因**，不留空。每一条 `未固定` 都是换台机器就可能变结果的缺口。", "",
         "## 宿主", ""]
    L += [f"- {k}：`{v}`" for k, v in rep["host"].items()]
    L += ["", "## 工具链", ""]
    L += [f"- {k}：`{v}`" for k, v in rep["toolchain"].items()]
    L += ["", "## Python 依赖", "",
          f"- `uv.lock` sha256 前 12 位：`{rep['python_lock']['sha256_head']}`｜`pyproject.toml`：`{rep['python_lock']['pyproject']}`",
          "", "| 包 | 版本 |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rep["key_packages"].items()]
    L += ["", f"**pyproject 未声明的**：{rep['not_declared_in_pyproject']['torch / transformers / sentence-transformers']}", "",
          "## 前端", "",
          f"- `frontend/package-lock.json` sha256 前 12 位：`{rep['frontend_lock']['sha256_head']}`", "",
          "## 容器与编排", ""]
    L += [f"- {f}：`{h}`" for f, h in rep["containers"]["compose_files"].items()]
    L += [f"- `infra/nginx.m4.conf`：`{rep['containers']['nginx_conf']}`", "- 镜像："]
    L += [f"  - `{i}`" for i in rep["containers"]["images"] if i.strip()]
    L += ["", "## 服务端版本（必须栈在跑才取得到）", "",
          f"- Milvus：`{rep['servers']['milvus']}`", f"- PostgreSQL：`{rep['servers']['postgres']}`", "",
          "## 模型权重", "", "| 用途 | 路径 | 体量 | 清单摘要 |", "|---|---|---|---|"]
    for m in rep["models"]:
        L.append(f"| {m['项']} | `{m['路径']}` | {m['状态']} | `{m['摘要']}` |")
    L += ["", "摘要算法：目录内 `文件名:字节数` 排序后取 sha256 前 12 位——**不含内容哈希**，"
          "同名同尺寸被替换是查不出来的；这条限制写在这儿而不是藏在脚本里。", "",
          "## API 宿主环境要求（ADR-12 定案后必列：API **不在容器里**，这些就是它的部署前提）", "",
          "- 宿主 Python：`" + rep["host"]["python"] + "`；虚拟环境 `" + rep["host"]["python_prefix"] + "`",
          "- torch / transformers / sentence-transformers **不在 `pyproject` 依赖里**，由本机 CUDA 底座经 `.pth` 挂载提供"
          "（换机必须复刻这一步，否则 API 直接起不来）",
          f"- dtype `{rep['dtype']}`、设备 `{rep['device']}`：改成 CPU 会打穿 P95 检索门禁，"
          "见 `docs/ADR-12_容器化代价实测.md`",
          "- 显存：8GB 要容纳「嵌入 + 重排 +（分时）7B」；同时跑别的 GPU 应用会污染延迟读数",
          "- 端口只绑 127.0.0.1（`serve_api.py` 拒绝非回环绑定）：API 38000、Demo web 38080、Vite dev 38173",
          "- 身份走请求头且**可伪造**，仅内网 Demo 可用；生产必须换真实身份源（《技术方案》7.2.1）", "",
          "## Ollama 生成模型", "", "```", rep["ollama"]["raw"] or "（未取到）", "```", "",
          f"⚠ {rep['ollama']['note']}", "", "## 未固定项（M6 前逐条销或明写接受）", ""]
    L += [f"{i + 1}. {u}" for i, u in enumerate(rep["unfixed"])]
    L.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({"未固定": rep["unfixed"], "milvus": rep["servers"]["milvus"],
                      "postgres": rep["servers"]["postgres"]}, ensure_ascii=False, indent=2))
    print("产物：", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
