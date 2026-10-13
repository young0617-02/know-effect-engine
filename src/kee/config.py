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
    postgres_dsn: str = "postgresql://kee:待填_KEE_PG_SUPERUSER_PASSWORD@127.0.0.1:35432/kee"
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
    # 默认 True 是**如实反映现状**：现有一批脚本走的就是头。收紧为 False 的代价与步骤登记在风险册。
    allow_header_identity: bool = True

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
