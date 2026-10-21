"""KEE 运行配置：全部来自环境变量（前缀 KEE_）或 .env，硬编码一律视为越界。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 本地权重加载的 tqdm 进度条会污染 Gate 0 的日志与报告取证
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KEE_",
        env_file=(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    # ---- 本地模型权重（不下载，指向本机既有目录） ----
    bge_m3_path: str = "D:/modelscope/hub/models/BAAI--bge-m3/snapshots/master"
    bge_reranker_path: str = "D:/modelscope/hub/models/BAAI--bge-reranker-v2-m3/snapshots/master"
    model_dtype: str = "float16"
    embed_max_length: int = 512
    device: str = "cuda"

    # ---- 文档解析后端（MinerU 4.x pipeline 的本地 ONNX 权重） ----
    mineru_model_dir: str = "D:/modelscope/hub/models/MinerU-4_models_onnx"
    # auto=pypdf 文本层优先、薄则转 OCR；pypdf=只用文本层；mineru_onnx=强制版面+OCR
    pdf_parse_backend: str = "auto"
    ocr_dpi: int = 150
    pdf_min_chars_per_page: int = 40

    # ---- 存储栈 ----
    milvus_uri: str = "http://127.0.0.1:39530"
    milvus_db: str = "default"
    milvus_collection: str = "kb_chunks"
    # 应用角色 `kee_app`（LOGIN，非超级用户）。口令与 `待填_KEE_PG_SUPERUSER_PASSWORD` 同性质：本机开发的公开值。
    # 不用 `kee` 是因为它是 initdb 引导出来的 bootstrap 超级用户——PG 明文禁止摘它的 SUPERUSER，
    # 而超级用户让表级 REVOKE 全部失效（含 `audit_log` 的防篡改触发器）。
    # 新库起好后必须跑一次 `scripts/harden_audit_store.py`，否则 `kee_app` 根本不存在。
    postgres_dsn: str = "postgresql://kee_app:待填_KEE_PG_APP_PASSWORD@127.0.0.1:35432/kee"
    redis_url: str = "redis://127.0.0.1:36379/0"
    search_cache_ttl_sec: int = 300

    # ---- 生成模型 ----
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b"
    llm_keep_alive: str = "-1s"
    llm_timeout_sec: float = 120.0
    cloud_fallback_enabled: bool = False

    # ---- 检索参数基线（Gate 0 ③ 结论回填） ----
    rrf_k: int = 60
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_n: int = 5
    # 查询改写：off=不启用；sparse=改写只喂 BM25 腿；both=两腿都用改写后查询；
    # extra_leg=原查询与改写查询各算一路再 RRF（召回覆盖最大，代价是多一次嵌入）
    rewrite_mode: str = "off"
    rewrite_budget_ms: float = 900.0
    rewrite_max_tokens: int = 96

    # ---- 置信度与降级阈值（Gate 0 ⑤ 结论回填） ----
    confidence_high: float = 0.75
    confidence_low: float = 0.45
    temperature_scaling_dir: Path = Field(default=PROJECT_ROOT / "var/calibration")

    # ---- 账号与登录态 ----
    session_ttl_hours: int = 12
    # 请求头身份（X-Kee-Role 等）是内网工具与验收脚本的通道；关掉它就必须带 Bearer 令牌。
    # 默认 **False**（用户 2026-10-15 裁「交付物默认 false、脚本显式带令牌」）：
    # 保持 True 时，「限内网可用」这条约束是由部署环境承担的，不由代码承担——
    # 不带令牌、只发一个 `X-Kee-Role: system_admin` 就能拿到该角色的全部范围。
    # 需要该通道的脚本改走 `scripts/_identity.py`（登录换 Bearer）；
    # 本机测试要覆盖这条通道时，由 `tests/conftest.py` 显式打开，不在生产默认里留后门。
    allow_header_identity: bool = False

    # ---- 可观测（Prometheus 指标 / 结构化日志 / Langfuse 上报） ----
    # /metrics 只暴露闭合枚举标签（无查询文本、无用户名、无文档 id）。API 刻意只绑回环、
    # Nginx 不反代 /metrics，所以抓取通路是「宿主 → Prometheus 容器」，不经浏览器。
    # 要彻底关掉：KEE_METRICS_ENABLED=false，端点变 404，指标层仍在计数但没人取。
    metrics_enabled: bool = True
    log_level: str = "INFO"
    log_json: bool = True
    # Langfuse 一律**自托管**：上报载荷含问题与结论原文，云端 SaaS 等于把「数据不出域」
    # 交给别人的运维承诺——那条约束不由代码承担，所以默认关，且开了也要过域内地址闸门。
    langfuse_enabled: bool = False
    langfuse_host: str = "http://127.0.0.1:30090"
    # pk/sk 只从环境变量或 .env 读（`.env` 已被 gitignore），绝不写进任何入库文件。
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_flush_interval_sec: float = 2.0
    langfuse_max_batch: int = 50
    langfuse_timeout_sec: float = 5.0
    # 显式越过「目标必须是回环/私网」那道闸门。开它等于书面承认数据出域，风险登记要跟上。
    langfuse_allow_remote: bool = False

    # ---- 运行产物 ----
    artifact_dir: Path = Field(default=PROJECT_ROOT / "var/gate0")

    def resolved_artifact_dir(self) -> Path:
        p = Path(self.artifact_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
