# M4 · 前端与一键 Demo（三态渲染已实现并在真浏览器 + Nginx 链路上验过；API 容器化是**明示偏离**）

> **口径声明**：本文读数是**本机真机实测**（RTX 4060 8GiB / 本地 `qwen2.5:7b` / 模拟语料 24 篇 193 块 /
> 存储栈 Docker Compose）。语料为模拟件，绝对值只回答「这条路走不走得通」，不构成上线承诺。
> **视觉像素级验证未做**（本机 in-app 浏览器拿不到可见视口，截图工具不可用）；
> 做到的是**真浏览器里派发真实事件 + 读回类名与文本**，以及组件级 DOM 断言。这两件事的边界在第七节写清。

---

## 一、结论先说

| 交付面 | 状态 | 证据 |
|---|---|---|
| 问答页（流式三态、引用、置信度、转人工、多轮） | **已实现 + 真机验过** | `frontend/src/pages/AskPage.tsx`；组件断言 13/13（连跑 12 次 0 失败）；真浏览器 production build 走查 |
| RBAC 界面（身份条 + 检索层过滤回显） | **已实现 + 真机验过** | 点「金融」后服务端回显 `security_level <= 2 and industry in ["金融"]`，返回 5 篇全为金融 |
| 审计界面 + `GET /api/v1/audit` | **已实现 + 真机验过** | 非管理角色拿到 `E403` 原文；管理员 50 行按 id 倒序；首行就是新的 `chunks_read` |
| 知识库管理页（列表/详情/上传/复核/回滚/对账） | **已实现，界面级验过** | 24 篇→新人视角 4 篇；`scope` 回显随角色变 |
| 一键 Demo（存储 + 前端进容器） | **已实现 + 实测** | `infra/docker-compose.m4.yml`；`up -d` 对已运行存储栈是 no-op，只新建 web（1.5s） |
| **API 容器化** | **未做，明示偏离** | 见第三节，ADR-12 登记为待裁决 |
| ① 流式首 token < 1.2s | **仍未达标**（门禁未改） | 本次经 Nginx 实测首个可见 delta **934ms / 1477ms**（两轮），M4 流式报告分型 p50 1642.7ms。**不得据此声称达标** |

**一句话**：M4 的四个界面面都通了、且每条都在真浏览器和真 HTTP 链路上验过；
**唯独「应用容器化」这一条我没做，是偏离而不是完成**。

---

## 二、这一节专门写「差点写成假绿」的地方

1. **`{"detail":"There was an error parsing the body"}` 一度被我当成后端契约破了。**
   实际是我的 curl 引号被 Git Bash 弄坏了；用文件发就正常，直连发坏 JSON 确实回 `E400` + `trace_id`。
   **教训**：报「接口不守契约」之前先隔离出是代理、是客户端、还是应用。
2. **`verify_m4` 第一次跑出 11/12，失败那条揪出了两个真问题**（都不是测试自己难看的）：
   - 跑着的 API 是我改编码**之前**起的旧进程 → 服务是旧的、测试进程内是新的，两边会给出矛盾结论；
   - 中文行业名直接进 HTTP 头，**浏览器当场拒发**：
     `Failed to read the 'headers' property: String contains non ISO-8859-1 code point`，
     `httpx` 侧同样 `UnicodeEncodeError`。于是补了两端对称的百分号编码（`encodeURIComponent` ↔ `unquote`），
     并把「必须编码」本身写成断言（`sse.test.ts`），而不是只在注释里说一声。
3. **一条组件测试 12 次里红 2 次**——是竞态，不是「偶尔没过」。
   修法是把「还没读回正文」和「没权限读」分成两个状态（新增 `settled`），
   顺带修掉一句假话：以前没落定时也显示「正文未返回」，那把**慢**说成了**没数据**。
4. **`include:` 不继承被包含文件的 `name:`**。少了这一行，`up -d` 会去新建一套 etcd/minio，
   撞上已在跑的 `kee-*` 容器名直接 `Conflict` —— 「一键」当场变「一坏」。补 `name: kee-gate0` 后，
   dry-run 显示存储四件套是 `Running`（不重建），只新建 web。

---

## 三、API 为什么没进容器（这是裁决项，不是实现细节）

`pyproject` 刻意不声明 `torch` / `transformers`——它们由本机 `D:\py312-torch` 的 CUDA 环境
经 `.pth` 挂载提供，嵌入与重排要用宿主 8GiB 显存。把它装进容器只有两条路：
要么在镜像里拉 GB 级轮子（违反本机「大件不重复下载」），要么在容器里再要一份 CUDA 运行时（8GB 零余量）。

所以本版 Demo 形态是：**存储 + 前端进容器、API 在宿主、Nginx 经 host-gateway 反代**。
方案把「应用与 Nginx 容器化」列入 M4，这一条**部分没做到**，已按纪律登记：
`ADR-12`（状态：偏离方案·待裁决）+ `infra/docker-compose.m4.yml` 顶部注释 + `scripts/serve_api.py` 说明。

**没做的事也写在这里**：⑥ 的「真断网 + 无外呼」重验**本轮没跑**——
`g06 --from-scratch` 会连数据卷一起抹掉，重跑代价太大。本轮对「无外呼」的支持证据是**结构性的**：
Nginx 配置里只有 `host.docker.internal:38000` 一个 upstream，前端全部走相对路径（同源），
没有任何第三方脚本或字体。**这不等于断网实测通过**。

---

## 四、界面必须兑现的五项（方案对流式渲染的要求）

结论 / 要点 / 引用（文档·章节·页码·**片段**）/ 置信度 / 免责提示——逐项落点见
[`docs/前端流式渲染契约.md`](../docs/前端流式渲染契约.md)。三处值得单独说的口径：

- **片段正文不来自答案响应**，而走 `POST /api/v1/chunks/batch`。这条通道与检索层**同口径过滤**且只返回
  生效版本——因为 chunk id 会出现在响应里给客户端，按 id 直取若不判权限，检索层的过滤就成了摆设。
  越权与不存在合并成 `hidden`，不区分（区分本身就是泄露）。每次读取落 `chunks_read` 审计。
- **置信度只能显示成规则分**：`composite / calibrated / T` 后面原样带 `score_basis=rule_composite_not_probability`，
  并有断言禁止渲染成「置信度 78%」这类概率口径。
- **撤回不静默删除**：流出去却没成为答案的文字，以 `kee-state--retracted` 留在页面上并附说明。
  本轮真机两轮都没触发撤回（与流式报告 0/54 一致），所以这条**只由组件断言覆盖**，不是现场观测到的。

## 五、RBAC：强制点在检索层，界面只负责「看得见」

前端角色表（`frontend/src/api/rbac.ts`）只用于呈现可选项，**不做任何判定**；
越权的表现是「搜不到 / 读不到正文」，不是「前端藏起来」——这条口径写进了 ADR 与契约文档。
身份条上明写：走请求头、**可伪造**、仅内网 Demo 可用，生产必须换真实身份源
（换什么源文档至今未定，且过滤维度有三种不一致表述，已登记为待裁决）。

## 六、怎么跑起来

```bash
uv run python scripts/serve_api.py            # 宿主 API：http://127.0.0.1:38000
cd frontend && npm install && npm run build    # 产出 frontend/dist
docker compose -f infra/docker-compose.m4.yml up -d    # 一键：存储 + Nginx
# 打开 http://127.0.0.1:38080/
uv run python scripts/verify_m4.py                          # 端到端 12 项（默认打宿主 API）
KEE_BASE=http://127.0.0.1:38080 uv run python scripts/verify_m4.py   # 打 Nginx 同源链路
cd frontend && npm run test                                 # 13 条组件/解析断言
```

端口：API `38000`、Demo web `38080`、dev `38173`。本机 351xx–371xx 段 bind 直接 `WSAEACCES`
（系统保留，`netstat` 里查不到占用者），所以一律落在实测可绑的 381xx+；`strictPort` 保持开，
端口被占要报错而不是悄悄换——否则「文档写的地址打不开」最难查。

## 七、已验证 / 未验证（分开写，不许混）

**已验证**
- Python 单测 **166 全过**（含新增 `tests/test_api_chunks_audit.py` 8 条：密级、行业、编码、审计角色门、倒序、留痕）。
- 前端 **13/13**，连跑 12 次 0 失败；`tsc --noEmit` 干净；`npm run build` 通过。
- `verify_m4.py` **12/12**（经 Nginx）：事件顺序 `retrieval→delta*→answer`、`answer` 恰好一次、
  delta 50 条（代理没攒流）、密级 3 片段新人不可读 / 管理员可读、编码后的行业名进了过滤表达式、
  审计 E403 与倒序、`chunks_read` 与 `answer_generated(stream=true)` 留痕。
- 真浏览器（production build）走查：身份条、依赖状态条、三态样式类、引用片段正文、免责文案；
  审计页在交付工程师角色显示服务端 `E403` 原文、切系统管理员出 50 行；知识库从 24 篇收到新人视角 4 篇。
  控制台除两条**故意的** 403 外无报错。

**未验证 / 不能声称**
- **① 首 token 门禁**：仍不达标（本次 934ms 与 1477ms 两轮，M4 流式报告 p50 1642.7ms / 1.85% 达标）。
- **视觉像素级**：未做（截图通道不可用），只有 DOM 结构与文本级验证。
- **真断网 + 无外呼重验**：未跑（代价是整个存储栈重建）。本轮只有结构性证据。
  ［**2026-10-12 已复跑，见第八节**——原判「未跑」在这一轮之前是对的，保留不抹。］
- **撤回态**：现场未触发，仅组件断言覆盖。
- **复核/回滚/对账的写路径**：按钮与错误处理已接，但**没在真库上跑过一次完整的通过→升版→回滚**——
  这类写操作会改语料状态，要跑就在干净的复现环境里跑，我没擅自改你的库。

## 八、⑥「断网 + 无外呼」复跑（2026-10-12，任务 #49 的尾巴）

跑法：`uv run python -m kee.gate0.g06_localization --isolation`（**不带 `--from-scratch`**，
数据卷不动；代价是存储容器 `down`→internal 网起→探针→`down`→正常起重建一遍）。
⑥ 的两份历史产物（`var/gate0/g06_localization.json`、`reports/gate0/g06_localization.md`）
跑前备份、跑后按 md5 还原；`verdict_06.json` 亦 `git checkout` 还原。**本轮结果另存**
`var/gate0/m4_g06_rerun2.json`，无效的第一轮另存 `m4_g06_rerun1.json` 当反面样本。

### 第一次跑出来的是一条「空集上的真」，不能用

`compose down` 删 `kee-gate0_kee-net` 被 **kee-web** 的 endpoint 挡住：
`error while removing network: network ... has active endpoints (name:"kee-web")` → iso 栈 `up_rc=1`、
容器一个都没起 → 探针全打在 `No such container: kee-milvus` 上，于是

- `containers_blocked_from_public: true` —— **在空容器集上恒真**，零证据价值；
- 总结判定 **不通过** —— 也是同一个干扰造成的假阴：`all_healthy` 硬比较「服务数 == 5」，
  kee-web 让 `ps` 数出 6 个，而真实状态是 5/5 healthy、一键 1.5s。

根因是结构性的：**Demo 容器与存储栈同属一个 compose 项目**（`docker-compose.m4.yml` 显式 `name: kee-gate0`
以复用存储），所以任何「按项目 down 网络」的动作都会被 kee-web 绊住。处置：**先 `stop kee-web` 再跑**。

### 第二次（有效读数）

| 项 | 读数 |
|---|---|
| 一键启动 | `up_rc=0`，**1.4s**，5/5 healthy |
| 仅绑定回环 | 非回环绑定 **0** |
| 连通性 | milvus / postgres / redis / ollama **全 ok** |
| internal 网变体 | `up_rc=0` 且 5/5 healthy；容器出网探测 **失败**（`curl` 到公网 rc=7 / 无应答）；容器间互联仍在（etcd 健康应答 `{"health":"true"}`）；**可疑对外连接 0** |
| 宿主进程连接审计（跑核心链路时采） | dense 2 / sparse 2 / llm_ok / JSON 合法；**非回环连接 0、公网出网 0** |
| 静态扫描 | 运行期出域嫌疑 **[]** |
| 云端兜底开关 | `false` |
| ⑥ 判定 | **通过**（md 第 67 行原文） |

⚠ **但「断网下核心链路可运行」这句仍然没有被端到端证明**，两点要写死：
1. `internal: true` 同时切断**宿主↔容器发布端口**通路，`host_can_reach_published_ports=false`
   —— 本拓扑里应用跑在宿主（ADR-12 未裁决），所以核心链路只能在 iso 相位**之外**跑；
   脚本里 `core_path` 也是在恢复之后才执行的。**iso 相位证明的是「容器封公网」，不是「断网仍能问答」。**
2. 因此有效证据是四条结构性的：容器封公网 + 仅绑定 127.0.0.1 + 宿主连接审计 0 出网 + 云端兜底关。
   真正的「整机断网跑通问答」要等 API 容器化、应用与存储同网段之后才能测——这条继续挂在 ADR-12。

### 复跑后踩到的新坑（已登记进记忆）

`stop kee-web` 之后再 `up -d kee-web` 起不来：
`failed to set up container networking: network c48cf... not found` —— 旧容器引用的网络在 iso 相位被删了。
解法是 **`up -d --force-recreate kee-web`**。恢复后实测：前端 `index.html` 200、
经 Nginx 的 `/healthz` 200（193 chunks / 24 docs）、`verify_m4` 经同源链路 **12/12** 复跑通过。

## 九、#48 知识库写路径：真库跑通，并当场撞出一个必现缺陷（2026-10-12）

**为什么不在生产库上跑**：线上门户**没有删除文档的端点**（只有 upload / review / rollback / reconcile）。
写进 `kee` 库与 `kb_chunks` 就撤不回来，还会永久改掉评测集的回查面。
所以 `scripts/m4_kb_write_path.py` 在**同一个 Postgres 容器里另建 `kee_writepath` 库**、
**同一个 Milvus 里另建 `kb_chunks_writepath` 集合**，跑完把库和集合都删掉。
真引擎、真 SQL、真 Outbox、真嵌入（BGE-M3 走宿主显存），只是命名空间隔离。
自检第一行就是这件事：healthz 回 `collection=kb_chunks_writepath, chunks=0`。

**13 项全过**（`var/gate0/m4_kb_write_path.json`）：上传 201 出 `doc_id/version_id`；
**上传者复核自己被拒 403「职责分离」**；他人复核后文档转 active 且版本生效；
v1 生效后可检索到正文（检索接口只回元数据，正文经 `/chunks/batch` 二段取，顺带验了这条契约）；
**升版**后生效版本切 v2 且旧口径正文不再命中；**回滚**后生效版本回 v1、索引里无 v2 子块残留；
对账 dry-run 读得出漂移面 → **人为删掉一个向量子块后对账报 `missing_in_vector=1`** → 修复后归零；
四类写操作都进审计（`document_ingested / document_review_pass / document_rollback / reconciliation_run`）；
终态自洽（生效版本唯一 + 文档 active）。复原自检：生产侧仍 **193 chunks / 24 docs**。

**撞到的真缺陷（已修）**：宿主 API 的**上传路径在真嵌入下必 500**。
`serve_api.py` 用 `create_app()`（`scheduler=None`），而 `IngestionService.ingest()` 在非 `skip_embedding`
分支要求嵌入器，遇 `scheduler is None` 直接 `raise RuntimeError("未提供 scheduler/embedder…")`；
惰性建调度器**只写在检索端点里**，上传与回滚没有。
`scripts/verify_m1.py` 一直自己 `create_app(scheduler=sched, ...)`，所以 166 个单测 + M1 验证全照不到这条路。
> 与 M3-05 那条教训同源：**「单测过了」不等于「线上那条路走得通」**——
> 测试替实现补上的那个参数，正是生产缺的那个。修法：`create_app` 内加 `ensure_scheduler()`，上传/检索/回滚共用。

**写路径自带的两个坑（留给下一个人）**：① 升版必须**显式带同一个 `doc_id`**，
靠文件名推 `doc_id` 会变成两篇独立文档，测不到版本切换；② `AuthoritativeStore.connect()` **不建表**，
新库要先 `init_schema()`、Milvus 新集合要先 `ensure()` —— app 假设依赖已就绪，这是设计不是疏漏。
