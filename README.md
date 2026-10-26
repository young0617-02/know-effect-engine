# 知效引擎（KnowEffect Engine，代号 KEE）

面向数字技术服务企业内部的**交付知识服务引擎**：把「咨询—实施—运维」全链路沉淀的项目与行业知识，
做成可追问、可溯源、可评估的 RAG 知识服务。Slogan：知识有源，交付有效。

**当前状态：M0–M5 全部完成，是一个能跑起来的内网部署版**（需求 V1.37 / 方案 V1.0.39，docs 见 `docs/`）。
验收口径要说清两件事，缺一件就是虚报：**门禁值**是设计目标，**验收读数**是这台机器上的本地实测——
本地实测不等于线上数据，也不等于真实用户行为（语料为自建行业文档集、无真实流量，故《项目定义》第六节
业务三条里仍有两条按构造不可测）。DoD 汇总：**达标 3／部分达标 4／不达标 0／不可测 0／本次产出 1**，
逐条取数见 `reports/m5/M5_DoD核对表.md`（由 `scripts/m5_dod_check.py` 生成，可复跑）。

## 目录

```
docs/                     需求与方案（项目定义 V1.37、技术方案与 MVP 设计 V1.0.39、业务诉求推演、用户操作手册、命名、ADR）
infra/docker-compose.*    Gate 0 单机存储栈（Milvus v3.0.0 / PostgreSQL 18 / Redis 8）+ 断网变体覆盖
scripts/ci_local.sh       本机一条命令复验（十阶段：前置/静态/前后端单测/三支 verify/观测探针/报告）
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

单测：`.venv/Scripts/python.exe -m pytest -q`（现 **390 项**，2026-10-26；纯逻辑部分不需要 GPU，DB/存储相关用例需先起存储栈）。

## M1 知识构建链路（Gate 0 放行后开工）

真实文档接入的端到端链路：上传 → 准入 → 解析 → Parent-Child 分块 → PostgreSQL 权威源 → Outbox → Milvus 索引 → 审计/对账。

| 入口 | 说明 |
|---|---|
| `POST /api/v1/documents/upload` | multipart 上传（md/txt/csv/log/docx/pdf/png/jpg 等）+ 元表单 + `Idempotency-Key` |
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
表格行列结构**已接**：SLANet-plus 坐标头 → 行列网格 + 合并表头 → Markdown 表格入库（5 样本实测，见验证记录三附）。
**公式识别已接**：PP-FormulaNet_plus-M ONNX（本机 564MB），版面判出 `display_formula/inline_formula` 才按需加载，
输出 LaTeX；模型失败回退 VLM/OCR 文本行。鉴权仍是请求头占位。

### 图片、图表与公式（入库期 VLM 描述索引）

- **DOCX 内嵌图片 / 独立 png-jpg / 文本层 PDF 内嵌图片 / 扫描件 PDF / PDF 图表区域**：先 OCR，再按需调用 VLM 生成中文描述；
  描述文本作为普通段落进入 Parent-Child 分块、BGE-M3 嵌入与 BM25，检索时用**文本 Query** 命中——服务阶段不跑 VLM。
- **公式**：本地 PP-FormulaNet_plus-M ONNX → LaTeX；不联网、不进在线链路。
- **VLM 默认关闭**。本地部署指向 vLLM 的 OpenAI 兼容端点；云端 API 仅开发/测试可用，必须显式设
  `KEE_VLM_ALLOW_REMOTE=true`（等于确认图片出域），生产必须关掉或换回本地。
- 数量闸门：单文档 20 张、单页图表最多 6 次 VLM、小于 20000 像素的图只 OCR 不送 VLM；连续失败熔断，不阻断入库。
- 完整配置、部署命令与边界见 `docs/VLM图片描述索引.md`。

后端选择：`KEE_PDF_PARSE_BACKEND=auto|pypdf|mineru_onnx` —— auto 下文本层够厚就走 pypdf（0.8s/文档），
薄了才转版面+OCR（约 18s/页），不为能抽字的 PDF 白烧算力。

## M2 检索评测基线

| 入口 | 作用 |
|---|---|
| `scripts/m2_eval.py` | 评测作业：预声明基线跑全样本出门禁判定，OFAT 网格只跑 dev 半区做诊断，与 `m2_eval_prev.json` 比对报回归 |
| `scripts/m2_rebuild_baseline.py` | 基线索引重建 + 四项自证校验（正文逐块不变 / 命名统一 / gold 等价 / 索引=权威源） |
| `scripts/m2_rewrite_probe.py` | 查询改写真机探针：触发率、单次延迟、JSON 失败率 |
| `scripts/m2_table_probe.py` | 表格检索专项：单元格级问题能否检索到对应行（独立集合，跑完清库） |
| `scripts/probe_dense_vs_bm25.py` | 双腿并排对照：纯编号 / 口语化 / 混合三条问句上稠密与服务端 BM25 的 Top-5 名次差异（关重排，只读名次）；`--control` 另跑构造的低词面重合对照 |

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

## M3 编排 · 引用 · 置信度（已完成）

| 入口 | 作用 |
|---|---|
| `scripts/verify_m3.py` | 问答链路端到端 10 步（答案契约 / 会话轮次 / 幂等 / RBAC 非空转对照 / SSE / 审计） |
| `scripts/m3_answer_probe.py` | 答案链实测：状态分布、空结论、引用落 gold、人工队列 |
| `scripts/m3_conclusion_probe.py` | 结论缺失分型（截断 / 解析失败 / 缺字段 / 主动留空），带 `done_reason` 取证 |
| `scripts/m3_refusal_guard.py` | ⑤ 出口守门，**按类别两把尺**：refuse 类结论必须为空、escalate 类允许答但必须转人工 |
| `scripts/m3_confidence_probe.py` | 置信度诊断：标签方差 → 成分分布 → AUC（含 CI）→ 温度可辨识性 |
| `scripts/m4_stream_probe.py` | token 级流式实测：**首个可见内容**延迟（不拿信封首字符充数）+ 覆盖率 + 撤回/条目丢失 + 流式与非流式交付一致性 |
| `reports/m4/M4_流式交付实测.md` | 上述实测的结论与口径说明（含「为什么 1.2s 门禁达不到、为什么 0 撤回不算安全保证」） |
| `scripts/m3_graph_ab_live.py` | **同机 A/B**：重构前实现 vs 图，同一模型同一批题比对；不一致时用「同一实现能否复现」分清模型抖动与真回归 |
| `scripts/m3_multiturn_probe.py` | 多轮消解的收益与代价：机械构造追问整批跑，benefit/harm 一起报，答案层面与检索层面分开报 |
| `scripts/m3_graph_equivalence.py` | **图化重构的差分回归网**：把重构前的实现按 commit sha 从 git 抽出当 oracle，17 个出口场景逐字段比对，不依赖 GPU/容器 |

```bash
.venv/Scripts/python.exe scripts/verify_m3.py          # 需先起存储栈与 Ollama
.venv/Scripts/python.exe scripts/m3_graph_equivalence.py   # 不需要任何服务
```

已落地：`POST /api/v1/answers`（结构化答案 + 会话轮次 + 幂等）、`/answers/stream`（token 级 SSE：`retrieval` → `delta`* → `answer`）、
`GET /api/v1/sessions/{id}`；答案出口加确定性契约校验（空结论不得判成功）；
强证据却声明无依据时转人工；答案完成率 24% → **91%**（根因是答案被写进 `key_points` 的字段错位，
用逐字搬运修，**不用**「要求结论非空」的提示词改法——那会把「该答不答」换成「不该答也答」）。

M3-05 已落地：在线链路搬进 `src/kee/qa/graph.py` 的 LangGraph 状态机（7 个显式出口，敏感闸前置——
原实现先检索再判敏感，是顺序缺陷）；多轮指代消解在检索**之前**生效，补全后的查询同时进检索与生成提示词，
而对外的 `question` 仍是用户原话；SSE 先改成真的阶段级、M4 再升级为 token 级（此前文档承诺「检索即出事件」而实现只发一个事件），
并补上流式路径缺的落轮次与审计。**Checkpointer 本轮不接**，多轮状态以 PG `session_turn` 为唯一权威源，
代价是无断点续跑（裁决记在《技术方案》V1.0.5 / 《项目定义》V1.5）。

真机已复跑（服务在线）：完成率 **53/58 = 91% 分毫不差**、⑤ 转人工 **5/5**、端到端 **11 步全绿**；
`scripts/m3_graph_ab_live.py` 同机 A/B **12/12 一致、真回归 0**——q-061 那道漏拒在重构前就存在，
不是编排引入的（它属于 ⑤ 未闭环的「无依据必拒 6/7」）。

⚠ 两条如实记账：① 旧报告里「泄漏 **0/12**」的口径把 refuse 与 escalate 混成一把尺，
分开口径后的真实读数是 **refuse 类 6/7、escalate 类 5/5 转人工**（旧尺子反而会把 q-070 那种
「有权限 + 已转人工的正确答案」报成泄漏）；② 多轮消解机制成立（追问被补全成上一轮主题并进检索），
但 14 道**机械构造**追问的答案层面净收益是 **0**（构造只删主题词、保留谓语区分度，本就测不出），
检索层面救回 1 道 / 丢掉 0 道——**这不足以支撑「多轮有收益」的一般性主张**。
详见 `reports/m3/M3_编排引用与置信度.md` 第八节。

## M4 交付 · 前端与一键部署（已完成并真机验收）

| 入口 | 作用 |
|---|---|
| `frontend/` | Vite + React + TS 独立前端（**无 UI 框架、无路由库**）：登录/注册页 / 问答页（左侧会话历史 + 底部提问栏）/ 知识库管理页 / 审计页 / 账号条 / 账号授权页 |
| `scripts/serve_api.py` | 宿主 API 入口（127.0.0.1:38000；**拒绝非回环绑定**） |
| `infra/docker-compose.m4.yml` | 一键部署：`include` 存储栈 + `kee-web`（Nginx 静态 + 同源反代） |
| `infra/nginx.m4.conf.template` | SSE 关缓冲（`proxy_buffering off`）——否则 token 级流式在浏览器里退化成一次性出答案 |
| `scripts/verify_m4.py` | 端到端 12 项（直连 API）/ 13 项（经 Nginx，多的那项是「Nginx 托管前端 index.html」）：事件顺序 / `answer` 恰好一次 / 流不被攒 / 密级与行业过滤 / 审计角色门与留痕 |
| `docs/用户操作手册.md` | **给使用方的操作手册**：登录注册、提问与引用核对、会话历史、拒答与转人工、权限可见范围、知识库管理、审计、账号授权、排错；15 张真机截图（`docs/assets/manual/`） |
| `docs/需求主流程.md` | **照「采购计划」那份模板的格式**做的需求主流程推演：一句话诉求 → 角色 → 要回答的四问 → 公司政策（9 条阈值）→ 证据的安全库存怎么算差额 → 答案明细 → 自动化边界（机器 vs 人）→ 9 个子系统带 23 个接口（含可观测那一组）→ 一段「基于政策所以…」的完整答复单 → 与模板的分歧 |
| `docs/业务诉求到主流程的推演.md` | **为什么长成这个样子**：六条痛点 → 三个硬要求 → 可信三件套 → 一条请求的七个节点（每档挡住哪个业务失败模式）→ **推演被实测推翻的六处** → 诉求对到当前读数 |
| `docs/前端流式渲染契约.md` | 三态渲染的可断言规格（C-1…C-8），并写明**哪些条款还没有断言** |
| `GET /api/v1/sessions`、`GET /api/v1/sessions/{id}` | 会话历史：列表只返回**自己**的（判归属不判角色，零轮次不入列），详情带**整包答案**（要点/引用/规则分），供左侧历史栏恢复旧会话 |
| `reports/m4/M4_前端与一键部署.md` | 本轮实测结论 + 四处「差点写成假绿」的记录 + 已验证/未验证分栏 |

```bash
uv run python scripts/serve_api.py                 # 宿主 API
cd frontend && npm install && npm run build        # 产出 frontend/dist
docker compose -f infra/docker-compose.m4.yml up -d   # 一键：存储 + Nginx → http://127.0.0.1:38080/
KEE_BASE=http://127.0.0.1:38080 uv run python scripts/verify_m4.py   # 13 项端到端（经 Nginx；直连 12 项）
cd frontend && npm run test                        # 40 条组件/分帧/登录闸门/会话生命周期断言
```

**登录与鉴权（V1.31 起的主身份通道）**：`src/kee/api/auth.py` + 两张表（`app_user` / `user_session`）
+ 七个 `/api/v1/auth/*` 端点（register / login / logout / me / users / users/{name}/role /
users/{name}/revoke-sessions）。
口令 scrypt 散列、会话服务端可撤销（库里只存令牌 SHA-256 摘要，默认 12h；**退出默认撤销该账号全部会话**）、
注册不得自授治理角色（唯一例外：库里为空时的首个注册者）。**有令牌时请求头不参与判定**；
无令牌时那条 `X-Kee-Role` 内网脚本通道**默认关闭**（`KEE_ALLOW_HEADER_IDENTITY` 默认 `false`，V1.35 裁决），
确需时由部署方显式打开。
界面上未登录只有登录页，登录后**没有角色选择器**——范围由账号推导并只读显示。
数据库侧同批收紧：应用连的是非超级用户角色 `kee_app`，`audit_log` 的属主是 NOLOGIN 角色，
所以 `TRUNCATE` 那条绕过防篡改的路也堵住了（`scripts/harden_audit_store.py`，`verify_m1` 里有对应验证步）。

M4 新增两个后端读接口：`POST /api/v1/chunks/batch`（引用片段正文，**与检索层同口径过滤**、只返回生效版本、
越权与不存在合并成 `hidden`、每次读取落 `chunks_read` 审计）与 `GET /api/v1/audit`（限管理角色，按 id 倒序）。
**身份头的取值必须百分号编码**——HTTP 头只允许 latin-1，而行业取值就是中文（金融/能源/化工），
真浏览器实测直接发中文会拒发整个请求（约定见《技术方案》7.2.1，断言在 `sse.test.ts`）。

⚠ **一条明示偏离**：方案把「应用与 Nginx 容器化」列入 M4，本轮**只做到 Nginx + 前端进容器、API 仍在宿主**。
原因是 `pyproject` 刻意不声明 torch/transformers（由本机 `D:\py312-torch` 经 .pth 挂载、用宿主 8GiB 显存），
容器化要么拉 GB 级轮子、要么在容器里再要一份 CUDA 运行时。**`ADR-12` 已于 2026-10-12 定案**：维持「存储 + 前端进容器、API 在宿主」，代价是把「整机断网下端到端问答」**划出承诺范围**（三条路的实测代价见 `docs/ADR-12_容器化代价实测.md`）。
⑥ 的重跑结果（2026-10-12，`var/gate0/m4_g06_rerun2.json`）：一键 1.4s、全 healthy、非回环绑定 0、
宿主审计非回环连接 0、公网出网 0、云端兜底 False——**但「整机断网下端到端问答」不在承诺范围内**：
`internal: true` 会连宿主↔容器通路一起切断而 API 在宿主，替代路 `enable_ip_masquerade=false` 实测
切不掉容器出网（负读数与脚本已留档，别再试第二遍）。所以合规那一格对外只能引用这组结构性证据。
① 的「流式首 token」口径已于 **V1.21 钉成 `first_visible_ms`（首个可读字）**，#56 净环境实测
p50 695.4 / p95 859.2ms ⇒ 达标；**信封首字符（`{`，324ms）不得当达标依据**。

## M5 · 可观测体系（Prometheus / Grafana / 结构化日志 / Langfuse）

| 入口 | 作用 |
|---|---|
| `src/kee/obs/metrics.py` | **19 个 `kee_*` 指标族**，**私有注册表 + 标签全闭合枚举**（无查询文本、无用户名、无文档 id）；出口标签取 `state["gated"]` 实际值；抓取期依赖探针与 `/healthz` 共用一份代码 |
| `src/kee/obs/logging.py` | 结构化日志：一行一个 JSON，`trace_id` 与响应体 / 审计 / `session_turn` 同源；`configure_logging()` 幂等 |
| `src/kee/obs/langfuse.py` | Langfuse 上报（官方 `/api/public/ingestion` 批量接口，不引 SDK）：**默认关闭**、后台线程、队列满就丢并计数、**目标地址必须回环/私网** |
| `GET /metrics` | Prometheus 文本。免鉴权的前提是三条同时成立：只绑回环 + Nginx 不反代该路径 + 标签无业务内容。`KEE_METRICS_ENABLED=false` 则 404 |
| `GET /api/v1/observability/status` | 观测面开关与 Langfuse 状态（含被关闭的原因原文），限治理角色 |
| `infra/docker-compose.observability.yml` | Prometheus + Grafana，**复用本机已有镜像（零下载）**，只绑 127.0.0.1：`http://127.0.0.1:39090` / `http://127.0.0.1:33000/d/kee-rag`（admin / 待填_KEE_GRAFANA_ADMIN_PASSWORD） |
| `infra/prometheus/` | `prometheus.yml`（目标走 `file_sd`）+ `alerts.yml`（9 条）+ `targets/kee.json`（由脚本生成） |
| `infra/grafana/` | 数据源与看板 provisioning + `dashboards/kee-rag.json`（15 面板，UI 不可改，改面板就改这个文件） |
| `scripts/obs_targets.py` | 宿主 API 换端口后重跑一条即可，Prometheus 20s 内自刷、**不用重启容器** |
| `scripts/m5_observability_probe.py` | 实测：`/metrics` 形状与抓取耗时、埋点开销（配对交错 A/B + 暂停抓取器）、七条计数器恒等式、31 条 PromQL 逐条执行、Grafana 面板级通路、外泄面核查 |
| `infra/docker-compose.langfuse.yml` | Langfuse 自托管**出稿未跑**：要拉 GB 级镜像且落 C 盘 Docker VM，等裁决；带 `profiles` 防呆 |
| `scripts/obs_langfuse_capture.py` | **实采**上报载荷：本地采集器 + 打印器，落盘 `var/gate0/m5_langfuse_capture.json`（作答轮 3 事件 / 敏感轮 1 事件 / 拒答轮 2 事件） |
| `docs/assets/obs/` | Grafana 真截图 4 张（Edge 1600×1200 kiosk）：#1 QPS、#6 分位数、#9 阶段分解、整页 |

```bash
uv run python scripts/serve_api.py                                  # 宿主 API（示例用 28000）
uv run python scripts/obs_targets.py --port 28000                   # 告诉 Prometheus API 在哪
docker compose -f infra/docker-compose.observability.yml up -d      # 零下载，两个容器
uv run python scripts/m5_observability_probe.py                     # 实测并出报告（退出码 0 才算通过）
```

**看板上的分位数不是门禁读数**：直方图分位是桶上线性插值的近似值、且统计全部轮次；
① 那条 6s 延迟门禁的口径是「6s + 仅作答轮次 + `first_visible`」，权威读数仍由 `scripts/` 下的离线探针产出。
实测就是证据：看板 p95 e2e 报 31.5s（含 `dense_timeout` 的冷启动轮），而门禁那条达标——两个数都对，说的是两件事。
**指标进程内累计、不落库**（API 重启归零）；跨重启的历史看 Prometheus，单轮的事实看 `session_turn`，
「谁问了什么」只有 `audit_log` 能答（追加写、禁改删）。

## 本机一条命令复验（`scripts/ci_local.sh`）

这个仓库**没有 remote**，而且被测系统绑着本机 D 盘的 CUDA 权重、Ollama、Milvus 与宿主回环端口——
搬到第三方 runner 上既跑不动，也会把带业务原文的测试语料送出域外。所以这里的 CI 是**本机的强制点**：
一条命令、顺序固定、失败即停、产物留档、**服务状态复原**。

```bash
scripts/ci_local.sh                 # 全量：十一个阶段
scripts/ci_local.sh --quick         # 不碰 GPU 的四段（约 1 分钟）
scripts/ci_local.sh --only eval     # 只检检索侧版本回归
scripts/ci_local.sh --skip obs-probe --keep-going
```

| 阶段 | 跑什么 | 为什么在这一层 |
|---|---|---|
| `preflight` | `check_stack.py`：torch/CUDA、权重目录、Milvus/PG/Redis、Ollama 模型、容器健康 | 后面每一段的前置事实，先确认再跑 |
| `static` | `ci_static.py`：`compileall`（src/scripts/tests）+ JSON 解析 + 四份 compose `config -q` + `promtool check config` | 拦的正是这两类当场炸的形状：`dict` 少一个收尾花括号、告警 YAML 写成两行各自带引号 |
| `backend` / `frontend` | `pytest -q` 全量；前端单测 + `npm run build` | 构建产物就是 `kee-web` 挂载的那份 `dist`，不 build 等于没改 |
| `verify-m1` / `verify-m3` | 在线链路验收（进程内加载模型） | **跑之前先停宿主 API**：8GB 单卡零余量，两边同时占就是静默退化而不是报错 |
| `eval` | `m2_eval.py --skip-grid`，与 `var/gate0/m2_eval_prev.json` **同尺子**对照（2026-10-16 首跑 67s：四条门禁全过、`delta_strict=0.0`、`regressed=false`） | 检索侧的**版本回归**。退出码分开：`1` = 四条门禁没过，`2` = 同尺子 strict Recall@5 掉超 0.02（回归）。尺子指纹不一致时如实报「不可比」，不给 `regressed=false` |
| `warm` | `api_warm.py`：登录 + 一次非流式问答 | 停起来再重启之后进程是冷的，这笔账必须由预热付而不是由验收吃（见下） |
| `verify-m4` / `obs-probe` | 真 HTTP 通路（直连 + 经 Nginx 两套）、观测面实测 | 走生产入口，与界面同一条路；`warm` 没过就**跳过**，不让冷进程伪装成失败 |
| `test-user` / `reports` | 默认跳过，需 `--with-test-user` / `--regen-reports` | 前者刷新带 LLM 漂移的业务读数，后者重写 `reports/m5/*.md` |

四条设计约束都是这个项目踩过的坑，不是设想：

1. **GPU 互斥**：`verify-m1`/`verify-m3`/`eval` 在自己进程里加载嵌入与重排，所以先 `api_down`，跑完按原状起回。
2. **冷启动不能被读成失败**：本机 2026-10-16 两次实测重启后首个问答 **177.4s**（检索段 162.7s、生成段 14.2s）
   与 **27.5s**（检索段 22.3s）——同一台机器差 6.5 倍，未拆到成分级，所以**没法预先预算**。
   `verify_m4` 的 httpx 超时是 180s。第一次全量复验没有预热时那三段三连红（185s / 3s / 424s 全失败），
   加上 `warm` 之后同三段 **11s / 10s / 71s 全通过**——那不是回归，是没预热。
3. **证据产物不许被悄悄重写**：跑前跑后各记一次 md5 并打印变化清单；`eval` 用 `--out` 写到本次运行目录，
   因为默认路径 `var/gate0/m2_eval.json` 是**带 dev 网格**的那份历史证据，`--skip-grid` 写回去等于抹掉 `grid_dev`。
4. **人工标注工单只读**：阶段清单里不含任何会写 `var/gate0/*_fill.csv` 的作业。

`ruff` 在本机装了，但仓库从来没有 `[tool.ruff]` 配置，其默认规则集在现有代码上报 590 项
（263 项是 `RUF100`：满仓 `# noqa: BLE001` 注的都是没启用的规则）。所以它**只报数、不判失败**——
让这种状态去判失败等于装一条永远红的闸门；升级成门禁需要「写配置 + 清 590 项」，那是独立一次改动。
本脚本也刻意**不装任何东西**。

## 目录（M1 新增）

```
src/kee/ingest/parsers.py     多格式解析适配层（字节流入，MinerU 留注册位）
src/kee/ingest/docimg.py      pypdfium2 页面渲染（BGR 位图给版面/OCR）
src/kee/ingest/mineru_onnx.py 本地 MinerU-ONNX 后端：PP-DocLayoutV2 版面 + PP-OCRv6 det/rec + SLANet-plus 表格结构
src/kee/ingest/metadata.py    元数据三级取值：表单 > front-matter > 推断，逐级告警
src/kee/ingest/service.py     单文档摄取编排 + 升版/回滚/复核
src/kee/retrieval/rewrite.py  查询改写：启发式触发 + 准入校验（禁引入原查询没有的数字）+ 截断打捞
src/kee/obs/metrics.py        Prometheus 指标层：私有注册表、闭合枚举标签、抓取期依赖探针
src/kee/obs/logging.py        结构化日志（JSON 一行一条，trace_id 与响应/审计同源）
src/kee/obs/langfuse.py       Langfuse 上报：官方 ingestion 批量接口、默认关闭、域内地址硬闸门
src/kee/api/app.py            对内 REST API 与统一错误码契约
scripts/reconcile_job.py      对账作业 CLI
scripts/table_fixtures.py     合成表格 PDF 生成器（表格结构的逐格真值来源）
scripts/m2_eval.py            M2 评测作业（基线 + dev 网格 + 回归比对）
scripts/verify_m1.py            M1 端到端验证（16 步）
reports/m1/M1_验证记录.md      M1 实测结论、步骤表与已知限制
reports/m2/M2_检索评测基线.md  M2 基线数字、gold 口径拆解与待裁决项
reports/m2/M2_稠密与BM25双腿对照.md  三条问句（纯编号/口语化/混合）上两腿 Top-5 名次差异、jieba 编号切分选择性、低词面重合对照的失败实测
```


## 口径与边界

- **报告引用的原始证据已入库**：`var/gate0/` 下的判定与指标 JSON、`qrels.jsonl`、标注待填表随仓库一起提供，
  报告里每个数字都能在树里对上。继续不入库的是 `var/gate0/vectors/`（嵌入物料，可由权威源重建）、
  `var/m1/fixtures/`（含真实厂商 PDF 的派生扫描件）与 `*.log`。
- 语料为**自建行业文档集**（金融/能源/化工交付场景，含术语、项目编号、方案名、章节页码），
  指标不代表真实客户数据效果；gold 标注由构造时的关键词回查生成，第二人复核表在
  `var/gate0/annotations_a2_template.csv`（待填项统一用 `TODO_ANNOTATOR2` 搜索定位）。
- 数据不出域：本地 Ollama + 本地权重 + 本地容器栈；云端兜底开关默认 `false`。
  可观测层同规矩：Prometheus / Grafana **本机自托管、只绑 127.0.0.1、复用已有镜像零下载**；
  Langfuse 上报**默认关闭**，且目标地址必须是回环/私网（公网直接拒发，`KEE_LANGFUSE_ALLOW_REMOTE` 是唯一越道口）。
- 看板的分位数**不是门禁读数**（桶上近似 + 全部轮次）；指标进程内累计、不落库，重启归零。
  `/metrics` 免鉴权的前提是「只绑回环 + Nginx 不反代该路径 + 标签无业务内容」三条同时成立。
- 已发现的方案偏差（详见 `reports/gate0/`）：
  1. Milvus v3.0.0 不接受 `analyzer_params={"type":"jieba"}`，可用形态是 `{"tokenizer":"jieba"}` 或 `{"type":"chinese"}`。
  2. 8GB 单卡无法同时驻留两个 7B（Ollama 会 LRU 掉旧模型），坐实方案否决"本地双模型分工"的判断。
  3. `internal: true` 网络会同时切断宿主经发布端口访问容器，断网验证只能在容器侧与全容器化形态下做。
