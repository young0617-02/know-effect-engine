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

单测：`.venv/Scripts/python.exe -m pytest -q`（42 → 现 59 项，纯逻辑，不需要 GPU/容器）。

## M1 知识构建链路（Gate 0 放行后开工）

真实文档接入的端到端链路：上传 → 准入 → 解析 → Parent-Child 分块 → PostgreSQL 权威源 → Outbox → Milvus 索引 → 审计/对账。

| 入口 | 说明 |
|---|---|
| `POST /api/v1/documents/upload` | multipart 上传（md/txt/docx/pdf）+ 元表单 + `Idempotency-Key` |
| `GET /api/v1/documents` | 分页 + **检索层权限过滤**（按角色密级上限、行业、项目） |
| `POST /api/v1/documents/{id}/review` | pass/reject；**上传者不能自审**；reject 会撤下该版本索引 |
| `POST /api/v1/documents/{id}/rollback` | 切生效版本，并用权威源 + 向量物料重建索引（不需 GPU） |
| `POST /api/v1/reconcile/run`、`scripts/reconcile_job.py` | 每日对账：diff → Outbox 补偿 → 复检；CLI 的退出码 2 供编排告警 |
| `scripts/verify_m1.py` | 16 步端到端验证（含真实 PDF 与表格结构），跑完自动还原生产索引到基线 |
| `scripts/verify_table_structure.py` | 表格结构还原的多样本验证：4 张合成表逐格比对真值（行列数 + 落位） |
| `scripts/probe_slanet.py` | SLANet-plus 坐标头反推证据：本地无 HTML 词表，token id 语义靠实测标定 |

```bash
.venv/Scripts/python.exe scripts/verify_m1.py        # 全链路取证，任一步不过即非零退出
.venv/Scripts/python.exe scripts/verify_table_structure.py
.venv/Scripts/python.exe scripts/reconcile_job.py --dry-run
```

已知边界（详见 `reports/m1/M1_验证记录.md` 第四节）：PDF 走**本地 MinerU-ONNX 自建后端**
（PP-DocLayoutV2 版面 + PP-OCRv6 det/rec，onnxruntime CPU 推理，模型用本机 `D:\modelscope` 已有权重；
不装 `mineru` 包——它 3.5MB 但依赖树带 lmdeploy/gradio/mineru-llama-cpp，会和本 venv 经 `.pth` 挂载的
CUDA torch 底座互顶版本；**也不用 mineru.net 云 API**，那属于数据出域）。
未接：公式识别（模型在本地，交付文档公式罕见）；鉴权仍是请求头占位。
表格行列结构**已接**：SLANet-plus 坐标头 → 行列网格 + 合并表头 → Markdown 表格入库（5 样本实测，见验证记录三附）。

后端选择：`KEE_PDF_PARSE_BACKEND=auto|pypdf|mineru_onnx` —— auto 下文本层够厚就走 pypdf（0.8s/文档），
薄了才转版面+OCR（约 18s/页），不为能抽字的 PDF 白烧算力。

## M2 检索评测基线

| 入口 | 作用 |
|---|---|
| `scripts/m2_eval.py` | 评测作业：预声明基线跑全样本出门禁判定，OFAT 网格只跑 dev 半区做诊断，与 `m2_eval_prev.json` 比对报回归 |
| `scripts/m2_rebuild_baseline.py` | 基线索引重建 + 四项自证校验（正文逐块不变 / 命名统一 / gold 等价 / 索引=权威源） |
| `scripts/m2_rewrite_probe.py` | 查询改写真机探针：触发率、单次延迟、JSON 失败率 |
| `scripts/m2_table_probe.py` | 表格检索专项：单元格级问题能否检索到对应行（独立集合，跑完清库） |

```bash
.venv/Scripts/python.exe scripts/m2_eval.py            # 全量（基线 + 9 格网格），约 3 分钟
.venv/Scripts/python.exe scripts/m2_eval.py --skip-grid  # 只跑基线（回归比对）
```

实测结论（详见 `reports/m2/M2_检索评测基线.md`）：检索层 **Hit@5 0.9828**、strict gold Recall@5 **0.965**、
P95 **338ms**；section gold Recall@5 **0.822** —— 缺口全部来自「章节内所有子块都算 gold」的定义
（6 道 gold>5 的题 Hit@5=1.0），不是排序失败。**参数不是瓶颈、样本量才是**：n=58 时 Recall@5 的
95% CI 宽 0.13，网格最大增益都在噪声地板内。查询改写已接入但默认 `off`（单次 P95 843ms，
一步就吃满 800ms 检索门禁）。表格单元格级检索：行级「列名=值」+ 行块守住融合名次，
专项 Hit@5 由 0.667 做到 **1.0**（n=12）。

**检索质量门禁口径已于 V1.3 / V1.0.3 修订**：主判定 `Hit@5 ≥ 95%` + `strict gold Recall@5 ≥ 85%`，
section gold 降为报告值，Recall 类数字必须附 n 与 95% CI（阈值数值未下调，只是写清了它测什么）。

`KEE_REWRITE_MODE=off|sparse|both|extra_leg` 控制改写形态。

## 目录（M1 新增）

```
src/kee/ingest/parsers.py     多格式解析适配层（字节流入，MinerU 留注册位）
src/kee/ingest/docimg.py      pypdfium2 页面渲染（BGR 位图给版面/OCR）
src/kee/ingest/mineru_onnx.py 本地 MinerU-ONNX 后端：PP-DocLayoutV2 版面 + PP-OCRv6 det/rec + SLANet-plus 表格结构
src/kee/ingest/metadata.py    元数据三级取值：表单 > front-matter > 推断，逐级告警
src/kee/ingest/service.py     单文档摄取编排 + 升版/回滚/复核
src/kee/retrieval/rewrite.py  查询改写：启发式触发 + 准入校验（禁引入原查询没有的数字）+ 截断打捞
src/kee/api/app.py            对内 REST API 与统一错误码契约
scripts/reconcile_job.py      对账作业 CLI
scripts/table_fixtures.py     合成表格 PDF 生成器（表格结构的逐格真值来源）
scripts/m2_eval.py            M2 评测作业（基线 + dev 网格 + 回归比对）
scripts/verify_m1.py            M1 端到端验证（16 步）
reports/m1/M1_验证记录.md      M1 实测结论、步骤表与已知限制
reports/m2/M2_检索评测基线.md  M2 基线数字、gold 口径拆解与待裁决项
```


## 口径与边界

- **报告引用的原始证据已入库**：`var/gate0/` 下的判定与指标 JSON、`qrels.jsonl`、标注待填表随仓库一起提供，
  报告里每个数字都能在树里对上。继续不入库的是 `var/gate0/vectors/`（嵌入物料，可由权威源重建）、
  `var/m1/fixtures/`（含真实厂商 PDF 的派生扫描件）与 `*.log`。
- 语料为**模拟行业语料**（金融/能源/化工交付场景，含术语、项目编号、方案名、章节页码），
  指标不代表真实客户数据效果；gold 标注由构造时的关键词回查生成，第二人复核表在
  `var/gate0/annotations_a2_template.csv`（待填项统一用 `TODO_ANNOTATOR2` 搜索定位）。
- 数据不出域：本地 Ollama + 本地权重 + 本地容器栈；云端兜底开关默认 `false`。
- 已发现的方案偏差（详见 `reports/gate0/`）：
  1. Milvus v3.0.0 不接受 `analyzer_params={"type":"jieba"}`，可用形态是 `{"tokenizer":"jieba"}` 或 `{"type":"chinese"}`。
  2. 8GB 单卡无法同时驻留两个 7B（Ollama 会 LRU 掉旧模型），坐实方案否决"本地双模型分工"的判断。
  3. `internal: true` 网络会同时切断宿主经发布端口访问容器，断网验证只能在容器侧与全容器化形态下做。
