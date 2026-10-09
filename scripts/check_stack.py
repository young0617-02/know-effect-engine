"""环境/栈连通性探针：Gate 0 全部验证项的前置事实核查，输出可复跑证据。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kee.config import get_settings  # noqa: E402


def section(title: str) -> None:
    print(f"\n===== {title} =====")


def main() -> int:
    s = get_settings()
    out: dict[str, object] = {}

    section("python / torch")
    out["venv_python"] = sys.executable
    import torch

    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["cuda_device"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        out["vram_total_mib"] = round(total / 1048576)
        out["vram_free_mib"] = round(free / 1048576)
    import transformers

    out["transformers"] = transformers.__version__

    section("本地模型权重")
    for name, p in (("bge_m3", s.bge_m3_path), ("bge_reranker", s.bge_reranker_path)):
        pp = Path(p)
        out[f"{name}_exists"] = pp.is_dir()
        out[f"{name}_weights"] = sorted(
            f.name for f in pp.glob("*") if f.suffix in (".bin", ".safetensors", ".pt")
        ) if pp.is_dir() else []

    section("Milvus")
    import pymilvus
    from pymilvus import MilvusClient

    out["pymilvus"] = pymilvus.__version__
    try:
        mc = MilvusClient(uri=s.milvus_uri, timeout=20)
        out["milvus_server_version"] = mc.get_server_version()
        out["milvus_collections"] = mc.list_collections()
        mc.close()
        out["milvus_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["milvus_ok"] = False
        out["milvus_error"] = f"{type(e).__name__}: {e}"

    section("PostgreSQL")
    try:
        import psycopg

        with psycopg.connect(s.postgres_dsn, connect_timeout=10) as conn:
            out["pg_version"] = conn.execute("select version()").fetchone()[0]
        out["pg_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["pg_ok"] = False
        out["pg_error"] = f"{type(e).__name__}: {e}"

    section("Redis")
    try:
        import redis

        r = redis.Redis.from_url(s.redis_url, socket_connect_timeout=5)
        out["redis_ping"] = r.ping()
        out["redis_version"] = r.info("server").get("redis_version")
        r.close()
        out["redis_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["redis_ok"] = False
        out["redis_error"] = f"{type(e).__name__}: {e}"

    section("Ollama")
    import httpx

    try:
        tags = httpx.get(f"{s.ollama_url}/api/tags", timeout=10).json()
        out["ollama_models"] = [m["name"] for m in tags.get("models", [])]
        out["ollama_ok"] = s.llm_model in out["ollama_models"]
    except Exception as e:  # noqa: BLE001
        out["ollama_ok"] = False
        out["ollama_error"] = f"{type(e).__name__}: {e}"

    section("容器健康")
    try:
        ps = subprocess.run(
            ["docker", "compose", "-f", "infra/docker-compose.gate0.yml", "ps", "--format", "{{.Name}}\t{{.Status}}"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        out["containers"] = ps
    except Exception as e:  # noqa: BLE001
        out["containers_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out, ensure_ascii=False, indent=2))
    ok = all(
        out.get(k) is True
        for k in ("cuda_available", "milvus_ok", "pg_ok", "redis_ok", "ollama_ok", "bge_m3_exists", "bge_reranker_exists")
    )
    print("\nGATE0-PRECHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
