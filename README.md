# 知效引擎（KnowEffect Engine，代号 KEE）

面向数字技术服务企业内部的**交付知识服务引擎**：把「咨询—实施—运维」全链路沉淀的项目与行业知识，
做成可追问、可溯源、可评估的 RAG 知识服务。Slogan：知识有源，交付有效。

**当前阶段：Gate 0（M0）可行性验证**。方案与需求见 `docs/`；本仓库代码目前只覆盖 Gate 0 所需的
机制层与验证脚本，不代表 MVP 已完成。文中所有指标口径遵循方案声明：**量化数字是设计目标/验收门禁，
不是上线实测达成值**。

## 目录

```
docs/                     方案定型文档（项目定义 V1.1、技术方案与 MVP 设计 V1.0.1、命名）
infra/docker-compose.*    Gate 0 单机存储栈（Milvus v3.0.0 / PostgreSQL 18 / Redis 8）+ 断网变体覆盖
src/kee/config.py         运行配置（全部 KEE_ 前缀环境变量）
src/kee/models/           BGE-M3 嵌入、BGE-reranker-v2-m3 重排、显存采样、分时驻留调度
src/kee/retrieval/        Milvus 混合检索（服务端 BM25 + 分析器）、RRF、RBAC 过滤、Redis 缓存、检索流水线
src/kee/ingest/           准入（格式/指纹/分级）→ Parent-Child 分块 → 权威源写入 → Outbox → 索引
src/kee/db/               PostgreSQL 权威源 schema（含审计禁改触发器）、Outbox 派发、每日对账、向量物料仓
src/kee/llm/              Ollama 客户端（qwen2.5:7b 常驻、JSON 约束、耗时与 token 计量）
src/kee/qa/               引用强制 + 复合置信度 + 拒答/降级/转人工
src/kee/eval/             指标（Recall/MRR/延迟/ECE）、温度缩放、评测集装配（qrels）
src/kee/gate0/            六个验证项执行器与《Gate 0 验证报告》汇总
reports/gate0/            验证报告（Markdown，可对外展示的证据）
var/gate0/                验证原始产物（JSON、向量物料、待填标注表；不入库）
tests/                    纯逻辑单测（指标、RBAC、分块、准入）
```

## 环境

uv 建 venv，但**大件（CUDA 版 torch）复用本机既有环境**，不重复下载 GB 级轮子：

```bash
cp .env.example .env                 # 端口/权重路径按需覆盖
python scripts/bootstrap_env.py      # 建 .venv + 挂 CUDA torch + 装轻依赖
.venv/Scripts/python.exe scripts/check_stack.py
```

`bootstrap_env.py` 做三件事：`uv venv --python 3.12.8`（干净 venv，**不带** `--system-site-packages`）、
写 `.pth` 指向 `D:/py312-torch/Lib/site-packages`、`uv pip install -e ".[dev]"`。

> 为什么不能用 `--system-site-packages` 指向 `D:\Python312`：那里装的是 **CPU 版 torch**（`2.14.0+cpu`），
> 一旦继承就静默失去 CUDA，嵌入/重排会慢一个数量级且不报错。venv 内的包优先于 `.pth` 挂载的底座。

## 起存储栈

```bash
docker compose -f infra/docker-compose.gate0.yml up -d      # 全部 healthy 即可
```

端口纪律：宿主侧一律 `127.0.0.1:3xxxx`（Milvus 39530 / PG 35432 / Redis 36379），
只绑回环既是数据不出域的网络层落地，也避开本机其他项目已占用的 19530/5432/6379。

## 跑 Gate 0

```bash
.venv/Scripts/python.exe src/kee/gate0/g02_chinese_bm25.py    # 建库 + 中文 BM25（必备②，含首次索引构建）
.venv/Scripts/python.exe src/kee/gate0/g03_ablation.py        # 检索消融（③）
.venv/Scripts/python.exe src/kee/gate0/g01_vram_scheduler.py  # 显存分时调度（必备①）
.venv/Scripts/python.exe src/kee/gate0/g04_consistency.py     # Outbox + 对账一致性（④）
.venv/Scripts/python.exe src/kee/gate0/g05_citation_calibration.py  # 引用强制与校准（⑤）
.venv/Scripts/python.exe src/kee/gate0/g06_localization.py --isolation  # 单机本地化与无外呼（必备⑥）
.venv/Scripts/python.exe src/kee/gate0/g00_summary.py         # 汇总判定与放行结论
```

判定规则（方案 3.3）：必备项 ①②⑥ 全过，且六项中 ≥5 项通过，才进入 M1 主体开发。

⚠ 执行顺序纪律：`g06 --from-scratch` 会 `down -v` 清空数据卷（Milvus 集合与 PostgreSQL 权威源一起没），
只能作为**独立的一键部署验证**跑，跑完必须重跑 `g02` 建库；默认的 `--rebuild` 只重建容器、保留数据卷。
`g01` 已加前置校验：collection 不存在或为空时直接判失败退出，不会拿「全部提前拒答 → 延迟 0ms」冒充达标。

单测：`.venv/Scripts/python.exe -m pytest -q`。

## 口径与边界

- 语料为**模拟行业语料**（金融/能源/化工交付场景，含术语、项目编号、方案名、章节页码），
  指标不代表真实客户数据效果；gold 标注由构造时的关键词回查生成，第二人复核表在
  `var/gate0/annotations_a2_template.csv`（待填项统一用 `TODO_ANNOTATOR2` 搜索定位）。
- 数据不出域：本地 Ollama + 本地权重 + 本地容器栈；云端兜底开关默认 `false`。
- 已发现的方案偏差（详见 `reports/gate0/`）：
  1. Milvus v3.0.0 不接受 `analyzer_params={"type":"jieba"}`，可用形态是 `{"tokenizer":"jieba"}` 或 `{"type":"chinese"}`。
  2. 8GB 单卡无法同时驻留两个 7B（Ollama 会 LRU 掉旧模型），坐实方案否决"本地双模型分工"的判断。
  3. `internal: true` 网络会同时切断宿主经发布端口访问容器，断网验证只能在容器侧与全容器化形态下做。
