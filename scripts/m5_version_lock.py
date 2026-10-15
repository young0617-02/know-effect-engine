"""M5-02 版本锁定清单：把「这台机器上跑出来的结果」换成「换台机器也该一样」的前提。

原则：**取不到就标 `未固定` 并写原因**，不留空、不假装锁定。凡 `未固定` 的项都是可移植性缺口，
MVP 收口前必须逐条销掉或明写「接受此偏离」。

跑法：uv run python scripts/m5_version_lock.py
产物：var/gate0/m5_versions.json + reports/m5/M5_版本锁定清单.md
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
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


def npm_version() -> str:
    """Windows 下 npm 的入口是 `npm.cmd`，`subprocess(["npm", …])` 找不到它（Git Bash 那个 shim 不是可执行映像）。
    这里不回退成「取不到」——工具其实在，是探测方式不对；记错一次就会把不是缺口的东西写进缺口清单。"""
    for cmd in (["npm.cmd", "--version"], ["npm", "--version"]):
        out = sh(cmd)
        if "__ERR__" not in out:
            return f"{out}（经 `{cmd[0]}` 取到）"
    return "__ERR__ npm.cmd/npm 都不可达——换机时若前端构建步失败，先查这里"


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
    目录里文件多，全量 sha 太慢，取「文件名 + 大小 + mtime」的聚合摘要——足够判断换机后拿到的是不是同一份。"""
    p = Path(path)
    if not p.exists():
        return {"项": tag, "路径": path, "状态": "缺目录", "摘要": "未固定"}
    files = [f for f in p.rglob("*") if f.is_file()]
    agg = hashlib.sha256("\n".join(sorted(f"{f.name}:{f.stat().st_size}" for f in files)).encode()).hexdigest()[:12]
    return {"项": tag, "路径": path, "状态": f"{len(files)} 个文件 / {round(sum(f.stat().st_size for f in files) / 1e9, 2)} GB",
            "摘要": agg}


def ollama_digests(base: str) -> dict:
    """生成模型的权重指纹。`ollama list` 不给 sha256 ⇒ 同 tag 被 `ollama pull` 覆盖后本机不报错、
    换机可能拿到不同权重。`/api/tags` 给 digest，取到才算锁上；取不到就明写未固定而不是留空。"""
    import urllib.request

    base = (base or "http://127.0.0.1:11434").rstrip("/")
    try:
        tags = json.loads(urllib.request.urlopen(base + "/api/tags", timeout=10).read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"状态": f"未固定（{base}/api/tags 取不到：{type(e).__name__}——Ollama 未在跑就取不到权重指纹）", "模型": {}}
    models = tags.get("models") or []
    if not models:
        return {"状态": "未固定（/api/tags 返回 200 但模型列表为空——端口开着不等于权重在）", "模型": {}}
    fixed = all(m.get("digest") for m in models)
    return {
        "状态": "已固定（按 digest 认权重，不按 tag 认）" if fixed else "未固定（有模型缺 digest）",
        "来源": f"{base}/api/tags",
        "模型": {
            m["name"]: {
                "digest": m.get("digest"),
                "大小_GB": round((m.get("size") or 0) / 1e9, 3),
                "modified_at": m.get("modified_at"),
            } for m in models
        },
    }


def cuda_torch_fingerprint() -> dict:
    """torch 不在 pyproject 里（由 D:\\py312-torch 经 .pth 挂载），所以它的「版本锁定」只能记宿主实测值。
    记的是这台机器实际跑的数，不是「应该有」的数。"""
    out: dict = {"导入": "", "torch": "未固定", "cuda_编译版本": "未固定", "cuDNN": "未固定",
                 "CUDA 可用": "未固定", "设备名": "未固定",
                 "说明": "由本机 CUDA 底座经 .pth 挂载提供，`uv sync` 不会装上它"}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        out["导入"] = f"失败（{type(e).__name__}: {e}）——换机没复刻 .pth 挂载时就是这个样子，API 起不来"
        return out
    out["导入"] = "成功"
    out["torch"] = torch.__version__
    out["cuda_编译版本"] = torch.version.cuda
    out["cuDNN"] = torch.backends.cudnn.version()
    # 到这里「装了 torch」已确定；「CUDA 真能用」要另问，且初始化失败会抛——它正是换机最常见的那一跤，
    # 要留在清单上当读数，不能让整张表因为它起不来。
    try:
        out["CUDA 可用"] = torch.cuda.is_available()
        out["设备名"] = torch.cuda.get_device_name(0) if out["CUDA 可用"] else "（无 CUDA 设备）"
    except Exception as e:  # noqa: BLE001
        out["CUDA 可用"] = f"未固定（{type(e).__name__}: {e}）"
        out["设备名"] = "未固定"
    return out


def windows_patch_level() -> dict:
    """只记 `platform.release()` 到 build 号不够——同一个 26200 上不同累积更新（UBR）行为就不同。"""
    out: dict = {"状态": "", "取值": {}}
    if platform.system() != "Windows":
        return {"状态": f"未固定（本函数只读 Windows 注册表，当前宿主 {platform.system()}；换到该宿主上需另取补丁级）",
                "取值": {}}
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
    except Exception as e:  # noqa: BLE001
        return {"状态": f"未固定（注册表打不开：{type(e).__name__}）", "取值": {}}
    for name in ("ProductName", "DisplayVersion", "ReleaseId", "BuildLabEx", "CurrentBuild", "UBR"):
        try:
            out["取值"][name] = winreg.QueryValueEx(key, name)[0]
        except OSError:
            out["取值"][name] = "（该版本 Windows 无此键）"
    # Win11 上 ProductName/ReleaseId 是兼容遗留键，仍写 "Windows 10 Pro"/"2009"。
    # 不点明的话，这张表会同时说「Windows 11」和「Windows 10 Pro」，读的人不知道该信哪格。
    out["注意"] = ("ProductName / ReleaseId 在 Windows 11 上是遗留值（仍写 Windows 10 / 2009），"
                   "以 DisplayVersion + CurrentBuild + UBR 为准")
    out["状态"] = "已固定（取到 DisplayVersion + BuildLabEx + UBR）" \
        if out["取值"].get("DisplayVersion") not in (None, "（该版本 Windows 无此键）") else "未固定（DisplayVersion 取不到）"
    return out


# 依赖名 → 代码里该出现的 import 前缀。**不写这张映射表就会猜错**：
# `langgraph-checkpoint-postgres` 的 top_level 是 `langgraph`，照名扫会把父包 `from langgraph.graph import …`
# 也算成它在用，于是「零导入」这条事实被工具自己抹掉。
IMPORT_PREFIX = {
    "langgraph-checkpoint-postgres": ["langgraph.checkpoint.postgres"],
}

# **框架自己会按需 import 的包**：本仓库扫不到 import，摘掉却会让线上端点报错。
# 不标这一类，机械判定就会把 FastAPI 的上传前提说成「没用到的依赖」——一个看起来很有依据的错话。
IMPLICIT_USE = {
    "python-multipart":
        "FastAPI 在参数里出现 `UploadFile = File(...)` / `Form(...)` 时才去 import `multipart`，"
        "调用点在 `src/kee/api/app.py:209`（文档上传）。**摘掉它上传端点直接 500**，所以它不属于「零导入」。",
}

# 零导入项的**处置**：留在依赖声明里是决定，不是遗漏。决定由脚本渲染进清单，
# 而不是只活在某行注释里——注释不会出现在读清单的人眼前。
DECLARED_UNIMPORTED_DISPOSITION = {
    "langgraph-checkpoint-postgres":
        "**保留**。人工复核闭环的中断-恢复要用它，现在还没接（原因记在 `src/kee/qa/graph.py:17`："
        "`session_turn` 已是权威历史，同一份状态不存两处）。"
        "代价 = 传递拉进 langgraph-checkpoint / psycopg-pool / orjson。"
        "若复核闭环最终不接，正解是 `uv remove langgraph-checkpoint-postgres` 后重跑本脚本，别让它继续挂在声明里。",
}


def declared_deps() -> list[str]:
    """pyproject `[project].dependencies` 的运行时依赖名（optional-dev 不算）。"""
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as f:
        deps = tomllib.load(f).get("project", {}).get("dependencies", [])
    names = []
    for d in deps:
        m = re.match(r"\s*([A-Za-z0-9_.\-]+)", d)
        if m:
            names.append(m.group(1))
    return names


def import_prefixes(dist: str) -> tuple[list[str], str]:
    """取这个发行包对应的 import 前缀，并说清前缀是**从哪来的**，查不到就明说未判定——
    把「没查到」写成「没用到」是这类清单最常见的假账。
    三条路按可靠性排：人工映射 > `top_level.txt`（新 wheel 多半没有）> 已安装文件树的首层模块名。"""
    if dist in IMPORT_PREFIX:
        return IMPORT_PREFIX[dist], "脚本内人工映射（命名空间包，元数据会误导）"
    try:
        txt = metadata.distribution(dist).read_text("top_level.txt")
        mods = [t.strip() for t in (txt or "").splitlines() if t.strip()]
        if mods:
            return mods, "包元数据 top_level.txt"
    except Exception:  # noqa: BLE001
        pass
    if not (files := _dist_files(dist)):
        return [], "未判定（已安装文件树里认不出模块名）"
    paths = [str(f).split("/") for f in files]
    py_roots = {p[0] for p in paths if p[-1].endswith(".py")}
    mods: set[str] = set()
    for p in paths:
        if p[0].endswith(".dist-info") or p[0] in ("bin", "Scripts", "share", "data"):
            continue
        if len(p) == 1 and p[0].endswith(".py") and p[0] != "__init__.py":
            mods.add(p[0][:-3])                      # 顶层单文件模块
        elif len(p) >= 2 and p[0] in py_roots:
            mods.add(p[0])                            # 底下真有 .py 才算包目录（numpy.libs 只有 DLL，不算）
    if not mods:
        return [], "未判定（已安装文件树里认不出模块名）"
    return sorted(mods), "已安装文件树首层模块名"


def _dist_files(dist: str) -> list:
    try:
        return list(metadata.distribution(dist).files or [])
    except Exception:  # noqa: BLE001
        return []


def dep_usage() -> dict:
    """逐个依赖在 src/ scripts/ tests/ 里扫 import。零命中的才叫「声明了没用」。"""
    files = [p for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py")]
    src = [(p.relative_to(ROOT), p.read_text(encoding="utf-8", errors="replace")) for p in files]
    out = {}
    for dep in declared_deps():
        prefixes, how = import_prefixes(dep)
        prefixes = [p for p in prefixes if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", p)]  # 去掉 '..'、'numpy.libs' 这类目录噪声
        if not prefixes:
            out[dep] = {"判定": "未判定", "前缀来源": how}
            continue
        pat = re.compile(r"^[ \t]*(?:from|import)[ \t]+(" + "|".join(re.escape(x) for x in prefixes) + r")(?:[.\s,])",
                         re.M)
        hits = sorted({str(f) for f, text in src if pat.search(text)})
        verdict = "在用" if hits else ("隐式使用（框架按需 import）" if dep in IMPLICIT_USE else "声明了但零导入")
        out[dep] = {"判定": verdict, "前缀": prefixes, "前缀来源": how, "命中文件": hits}
        if verdict.startswith("隐式使用"):
            out[dep]["依据"] = IMPLICIT_USE[dep]
    return out


def compose_images() -> set[str]:
    """`docker images` 是全宿主的清单，不是本项目的。这台机器上还留着别的项目的镜像，
    把它们一并列在「容器与编排」下会让人以为部署要拉 40 个镜像。取 compose 里真正引用的那些。"""
    imgs: set[str] = set()
    for f in (ROOT / "infra").glob("docker-compose*.yml"):
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("image:"):
                imgs.add(s[len("image:"):].strip().strip("\"'"))
    return imgs


def main() -> int:
    uv_exe = sh(["uv", "--version"])
    docker_ver = sh(["docker", "--version"])
    compose_ver = sh(["docker", "compose", "version", "--short"])
    images = sh(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"])
    server_milvus = "未连"
    server_pg = "未连"
    ollama_url = "http://127.0.0.1:11434"
    dtype = os.environ.get("KEE_MODEL_DTYPE", "")
    device = os.environ.get("KEE_DEVICE", "")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from kee.config import get_settings
        from pymilvus import MilvusClient
        import psycopg

        s = get_settings()
        ollama_url = s.ollama_url or ollama_url
        dtype = dtype or s.model_dtype
        device = device or s.device
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
    digests = ollama_digests(ollama_url)
    cuda_torch = cuda_torch_fingerprint()
    cuda_state = ("已固定（实测指纹，非 pyproject 声明）"
                  if cuda_torch["torch"] != "未固定" and not str(cuda_torch["CUDA 可用"]).startswith("未固定")
                  else f"未固定（{cuda_torch['导入'] or cuda_torch['CUDA 可用']}）")
    patch = windows_patch_level()
    usage = dep_usage()
    zero_import = sorted(k for k, v in usage.items() if v["判定"] == "声明了但零导入")
    implicit = sorted(k for k, v in usage.items() if v["判定"].startswith("隐式使用"))
    undetermined = sorted(k for k, v in usage.items() if v["判定"] == "未判定")

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "principle": "取不到就标 未固定 并写原因；未固定项 = 可移植性缺口，MVP 收口前逐条销掉或明写接受",
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "os_补丁级": patch,
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_prefix": sys.prefix,
            "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "shell": os.environ.get("SHELL", ""),
        },
        "toolchain": {"uv": uv_exe, "docker": docker_ver, "docker_compose": compose_ver,
                      "node": sh(["node", "--version"]), "npm": npm_version()},
        "python_lock": {"file": "uv.lock", "sha256_head": sha256_head(ROOT / "uv.lock"),
                        "pyproject": sha256_head(ROOT / "pyproject.toml")},
        "declared_dep_usage": usage,
        "key_packages": {k: pkg(k) for k in ["fastapi", "uvicorn", "pydantic", "pydantic-settings", "psycopg",
                                             "pymilvus", "redis", "langgraph", "httpx", "numpy", "sentence-transformers",
                                             "torch", "transformers", "pytest"]},
        "not_declared_in_pyproject": {
            "torch / transformers / sentence-transformers":
                "**pyproject 刻意不声明**，由本机 D:\\py312-torch 的 CUDA 环境经 .pth 挂载提供——"
                "这是 ADR-12（API 未容器化）的根因，也是这张清单上最大的可移植性缺口。"
                "缺口没被「销掉」，但已从『不知道装的是哪个』变成「换机要复刻的精确指纹」见 host_cuda_runtime。"},
        "host_cuda_runtime": {
            "状态": cuda_state,
            "torch": cuda_torch,
            "transformers": pkg("transformers"),
            "sentence_transformers": pkg("sentence-transformers"),
            "复刻方法": "D:\\py312-torch 建 CUDA 环境 + 在 .venv 的 site-packages 放 .pth 指向它（ADR-12 第 3 节）",
        },
        "frontend_lock": {"file": "frontend/package-lock.json", "sha256_head": sha256_head(ROOT / "frontend" / "package-lock.json"),
                          "vite": (ROOT / "frontend" / "package.json").exists()},
        "containers": {"images": images.splitlines(),
                       "required_by_compose": sorted(compose_images()),
                       "compose_files": {
            f: sha256_head(ROOT / "infra" / f) for f in
            ["docker-compose.gate0.yml", "docker-compose.m4.yml", "docker-compose.gate0.iso.yml",
             "docker-compose.gate0.nodev.yml"]},
            "nginx_conf": sha256_head(ROOT / "infra" / "nginx.m4.conf.template")},
        "servers": {"milvus": server_milvus, "postgres": server_pg},
        "models": [
            model_dir("BGE-M3（稠密+稀疏嵌入）", os.environ.get("KEE_BGE_M3_PATH", "D:/modelscope/hub/models/BAAI--bge-m3/snapshots/master")),
            model_dir("BGE-reranker-v2-m3（重排）", os.environ.get("KEE_BGE_RERANKER_PATH", "D:/modelscope/hub/models/BAAI--bge-reranker-v2-m3/snapshots/master")),
            model_dir("MinerU 4.x ONNX（版面+OCR+表格）", "D:/modelscope/hub/models/MinerU-4_models_onnx"),
        ],
        "ollama": {"raw": ollama_models, "digest_fixed": digests["状态"].startswith("已固定"),
                   "digests": digests,
                   "note": "`ollama list` 不输出权重 sha256 ⇒ 单靠它没锁定。改取 `/api/tags` 的 digest：**按 digest 认权重，不按 tag 认**"
                           "（同 tag 被 `ollama pull` 覆盖后本机不报错，换机可能拿到不同权重）。"},
        "dtype": dtype or "float16（.env 与 config 都取不到时按代码默认读）",
        "device": device or "cuda（.env 与 config 都取不到时按代码默认读）",
        "env_keys": env_present,
        "unfixed": [],
    }
    if not env_present:
        rep["unfixed"].append(".env 缺失：所有 KEE_* 取值未固定")
    if "__ERR__" in ollama_models:
        rep["unfixed"].append("Ollama 模型清单取不到（`ollama list` 未在跑）")
    # 未固定项**由探针结果推导**，不写死清单——写死的清单会在补上实测值之后继续谎报缺口。
    for label, probe in (("生成模型权重 digest", digests), ("宿主 OS 补丁级别", patch)):
        if not str(probe.get("状态", "")).startswith("已固定"):
            rep["unfixed"].append(f"{label}：{probe.get('状态')}")
    if cuda_torch["torch"] == "未固定":
        rep["unfixed"].append(f"宿主 torch 运行时指纹：{cuda_torch['导入']}")
    if str(cuda_torch["CUDA 可用"]).startswith("未固定"):
        rep["unfixed"].append(f"宿主 CUDA 可用性/设备名：{cuda_torch['CUDA 可用']}")
    # 工具链任何一格取不到都是「这台机器上的构建前提没锁住」——不能只检查上面三个新探针，
    # 否则 `__ERR__` 会留在表里却被「未固定：无」这句话盖掉。
    for tool, val in rep["toolchain"].items():
        if "__ERR__" in str(val):
            rep["unfixed"].append(f"工具链 `{tool}` 取不到：{val}")
    if undetermined:
        rep["unfixed"].append(f"依赖是否有 import 未判定：{undetermined}——「没查到」不能写成「没用到」")

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "m5_versions.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    L = ["# M5 · 版本锁定清单", "",
         f"> 生成时点：{rep['generated_at']}｜由 `scripts/m5_version_lock.py` 实跑取值。",
         "> 原则：**取不到就写 `未固定` + 原因**，不留空。每一条 `未固定` 都是换台机器就可能变结果的缺口。", "",
         "## 宿主", ""]
    for k, v in rep["host"].items():
        if isinstance(v, dict):
            L += [f"- {k}：{v.get('状态')}"]
            if v.get("注意"):
                L.append(f"  - 注意：{v['注意']}")
            L += [f"  - {kk}：`{vv}`" for kk, vv in (v.get("取值") or {}).items()]
        else:
            L.append(f"- {k}：`{v}`")
    L += ["", "## 工具链", ""]
    L += [f"- {k}：`{v}`" for k, v in rep["toolchain"].items()]
    L += ["", "## Python 依赖", "",
          f"- `uv.lock` sha256 前 12 位：`{rep['python_lock']['sha256_head']}`｜`pyproject.toml`：`{rep['python_lock']['pyproject']}`",
          "", "| 包 | 版本 |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rep["key_packages"].items()]
    L += ["", f"**pyproject 未声明的**：{rep['not_declared_in_pyproject']['torch / transformers / sentence-transformers']}", "",
          "## 声明了的依赖有没有真被 import（逐个扫 `src/ scripts/ tests/`）", "",
          f"- 声明 {len(rep['declared_dep_usage'])} 个：在用 {len(rep['declared_dep_usage']) - len(zero_import) - len(implicit) - len(undetermined)} 个"
          f"｜隐式使用 {len(implicit)} 个：{implicit or '无'}"
          f"｜零导入 {len(zero_import)} 个：{zero_import or '无'}"
          f"｜未判定 {len(undetermined)} 个：{undetermined or '无'}",
          f"- 判定法：按包元数据 `top_level.txt`（新 wheel 多半没有，退回已安装文件树首层模块名；"
          "命名空间包改用脚本内人工映射，见 `IMPORT_PREFIX`）取 import 前缀，正则只认 `from|import` 语句。"
          "**前缀查不到就记「未判定」，不记「没用到」；扫不到 import 也不等于没用，见 `IMPLICIT_USE`。**", ""]
    for dep, u in rep["declared_dep_usage"].items():
        if u["判定"].startswith("隐式使用"):
            L.append(f"- `{dep}`：**扫不到 import，但不是没用**——{u['依据']}")
        elif u["判定"] == "声明了但零导入":
            L.append(f"- `{dep}`：扫的前缀 `{u['前缀']}`（{u['前缀来源']}）→ 0 命中。"
                     f"处置：{DECLARED_UNIMPORTED_DISPOSITION.get(dep, '**未登记处置**——要么写理由要么 `uv remove`，别挂着')}")
    L += ["",
          "## 宿主 CUDA / torch 运行时（`uv.lock` 管不到这一层，只能记实测指纹）", "",
          f"- 状态：**{rep['host_cuda_runtime']['状态']}**",
          f"- 复刻方法：{rep['host_cuda_runtime']['复刻方法']}", "",
          "| 项 | 本机实测 |", "|---|---|"]
    for k, v in rep["host_cuda_runtime"]["torch"].items():
        L.append(f"| torch · {k} | `{v}` |")
    for k in ("transformers", "sentence_transformers"):
        L.append(f"| 由 .pth 提供的包 · {k} | `{rep['host_cuda_runtime'][k]}` |")
    L += ["", f"> dtype `{rep['dtype']}`、设备 `{rep['device']}`。**这一格记的是「这台机器实际跑出来的数」，"
          "不是「pyproject 声明的数」**——换成 CPU 或换成另一个 torch 构建，检索 P95 门禁的结论就不 transfers，"
          "详见 `docs/ADR-12_容器化代价实测.md`。", "",
          "## 前端", "",
          f"- `frontend/package-lock.json` sha256 前 12 位：`{rep['frontend_lock']['sha256_head']}`", "",
          "## 容器与编排", ""]
    L += [f"- {f}：`{h}`" for f, h in rep["containers"]["compose_files"].items()]
    L += [f"- `infra/nginx.m4.conf.template`（反代端口由 `KEE_API_PORT` 注入）：`{rep['containers']['nginx_conf']}`",
          "", "**本项目 compose 引用的镜像**（ID = 本机当前那份）："]
    host_id = dict(i.split("\t", 1) for i in rep["containers"]["images"] if "\t" in i)
    for img in rep["containers"]["required_by_compose"]:
        L.append(f"- `{img}` → 本机 `{host_id.get(img, '（宿主当前没有：要么 build 出来、要么部署时要拉）')}`")
    L += ["", f"> `docker images` 在这台机器上共 {len(rep['containers']['images'])} 行，**其中多数属于宿主上的其它项目**，"
          "不是本项目的部署前提，所以不列在这儿——列了会把「换机要拉什么」答成「这台机器上有什么」。", ""]
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
          "（换机必须复刻这一步，否则 API 直接起不来）——精确指纹见上一节「宿主 CUDA / torch 运行时」",
          f"- dtype `{rep['dtype']}`、设备 `{rep['device']}`：改成 CPU 会打穿 P95 检索门禁，"
          "见 `docs/ADR-12_容器化代价实测.md`",
          "- 显存：8GB 要容纳「嵌入 + 重排 +（分时）7B」；同时跑别的 GPU 应用会污染延迟读数",
          "- 端口只绑 127.0.0.1（`serve_api.py` 拒绝非回环绑定）：API 38000、Web 38080、Vite dev 38173"
          "（Windows 会把 38xxx–41xxx 段保留掉，bind 报 WSAEACCES 而 `netstat` 查不到占用者；"
          "换端口用 `KEE_API_PORT` / `KEE_API_ORIGIN`）",
          "- 身份走**登录令牌**（`/api/v1/auth/*`）。请求头那条兜底通道 **默认关闭**"
          "（`KEE_ALLOW_HEADER_IDENTITY=false`，V1.35 裁决），所以无令牌即 `E401`；"
          "内网工具确需该通道时必须显式打开，交付物不带。"
          "接企业统一身份（OIDC / LDAP / 网关 JWT）时 `app_user` 的迁移方式**尚未设计**（用户裁「留」），"
          "届时要替换的是这套口令体系本身，不是配置一个开关",
          "- **数据库权限面（换环境部署时是必经步骤，不是可选项）**：应用连 `kee_app`（非超级用户），"
          "`KEE_POSTGRES_DSN` 默认指向它；存储栈起好后必须先跑一次 `scripts/harden_audit_store.py`"
          "（建 `kee_app` + `kee_admin` + NOLOGIN 的 `postgres`，把 `audit_log` 与其触发器函数的属主交给后者）。"
          "没跑的话 `kee_app` 不存在、API 连不上库；`kee_admin` 的口令只落 `var/pg_admin_credentials.json`（不入库）。"
          "⚠ 别指望 `ALTER ROLE kee NOSUPERUSER`：`kee` 是 compose `POSTGRES_USER` 走 initdb 造的 bootstrap 超级用户，"
          "PG 明文拒绝摘它的 SUPERUSER（实测），所以防线只能靠换应用角色达成", "",
          "## Ollama 生成模型", "", "```", rep["ollama"]["raw"] or "（未取到）", "```", "",
          f"**权重指纹状态：{rep['ollama']['digests']['状态']}**（来源 `{rep['ollama']['digests'].get('来源', '—')}`）", "",
          "| 模型 tag | digest（= 权重身份） | 体量 | modified_at |", "|---|---|---|---|"]
    for name, m in (rep["ollama"]["digests"].get("模型") or {}).items():
        L.append(f"| `{name}` | `{m.get('digest')}` | {m.get('大小_GB')} GB | `{m.get('modified_at')}` |")
    L += ["", f"> {rep['ollama']['note']}", "", "## 未固定项（MVP 收口前逐条销或明写接受）", ""]
    L += [f"{i + 1}. {u}" for i, u in enumerate(rep["unfixed"])] or \
         ["（本轮无。）⚠ 这只说明**这台机器这次**每个探针都取到了值，不等于换机可复现——"
          "`uv.lock` 之外的 torch/`.pth` 挂载、Ollama 权重是否还在本机、宿主驱动版本，"
          "都得按上面的值逐项核对才算迁移完成。"]
    L.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({"未固定": rep["unfixed"], "milvus": rep["servers"]["milvus"],
                      "postgres": rep["servers"]["postgres"]}, ensure_ascii=False, indent=2))
    print("产物：", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
