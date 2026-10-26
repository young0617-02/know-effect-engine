# 图片、图表与公式：入库期 VLM 描述索引

> 实现状态：代码已落地，`DOCX 内嵌图片 / 独立图片 / 文本层 PDF 内嵌图片 / 扫描件 PDF / PDF 图表区域 / 公式` 六条入口都已接通。
> 本文说明**为什么只在入库期用 VLM**、本地/云端怎么切、失败怎么降级，以及哪些能力仍是边界。
> 代码位置：`src/kee/ingest/vlm.py`、`src/kee/ingest/media.py`、`src/kee/ingest/parsers.py`、`src/kee/ingest/mineru_onnx.py`。

## 1. 为什么是「入库期 VLM」

现有 RAG 的检索与生成都是文本链路：BGE-M3 嵌入 + 服务端 BM25 + 文本 LLM。让视觉模型参与在线问答会带来两个直接代价：
- 8GB 单卡上再常驻一个 8B 视觉模型，会把 Gate 0 ① 已经验证过的显存平衡打破；
- 每次问答都要跑一次视觉推理，延迟和成本不可控。

所以采用**描述索引**（Description Index）：

```
入库阶段（可以慢、可以离线）
  文档/图片 → OCR 文字 + VLM 结构化描述 → 普通段落
            → Parent-Child 分块 → BGE-M3 嵌入 + BM25 → Milvus

服务阶段（不碰 VLM）
  文本 Query → 混合检索 → 文本 LLM → 引用/拒答/转人工
```

VLM 只在建索引时跑一次；问答时，文字 Query 通过文本嵌入直接命中「内容描述」段落，纯文本 LLM 用这些描述和原文作答。这样既保留了 Qwen2.5-7B 在生成端的稳定性，也不改变原有的引用、复核闸、审计与评测口径。

## 2. 处理链路

### 2.1 六条输入入口

| 输入 | 处理方式 | 产出 |
|---|---|---|
| DOCX 内嵌图片 | 解压 `word/media/*`，逐张解码 → OCR → VLM 描述 | 新增「图片与图表」章节，描述进入分块/嵌入 |
| 独立图片（png/jpg/jpeg/webp/gif/bmp/tif/tiff） | 单图解码 → OCR → VLM 描述 | 单章节「图片内容」文档 |
| 文本层厚的 PDF 内嵌图片 | pypdf 枚举页面图片 → OCR → VLM 描述 | 按页新增 `IMG-<page>` 章节 |
| 扫描件 / 图片型 PDF | `pypdfium2` 渲染整页 → MinerU 版面 + OCR → 图表区域 VLM 描述 | 正文 + 图表描述段落 |
| PDF 图表/插图 | 版面模型给 `image/figure/chart` 框 → 区域 OCR + VLM 描述 | `【图表 第N页】` / `【插图 第N页】` 段落 |
| PDF 公式 | 版面模型给 `display_formula/inline_formula` 框 → PP-FormulaNet_plus-M ONNX → LaTeX | `【公式】$...$` 段落 |

### 2.2 描述文本长什么样

图片段落统一带两个可区分的字段：

```
【图表 第3页】图内文字：Q1 120 Q2 150 Q3 180 内容描述：柱状图，展示三个季度的交付额上升。
```

- `图内文字` 是 OCR 结果，尽量保留原文；
- `内容描述` 是 VLM 生成结果，**不是原文**，引用时不能把它当成原文引用；
- 两者都进入同一个子块，因此文本 Query 可以同时命中数字和语义描述。

### 2.3 数量闸门

一页 PDF 可能被版面模型切出十几个图标/装饰图。默认限制：

| 闸门 | 默认 | 作用 |
|---|---:|---|
| `KEE_VLM_MIN_IMAGE_PIXELS` | 20000 | 小于该像素数的图片只 OCR、不送 VLM |
| `KEE_VLM_MIN_FIGURE_AREA_RATIO` | 0.01 | PDF 图表区域至少占页面 1% 才做 VLM |
| `KEE_VLM_MAX_FIGURES_PER_PAGE` | 6 | 每页最多 6 次 VLM，防止图标刷屏 |
| `KEE_VLM_MAX_IMAGES_PER_DOC` | 20 | 单文档最多 20 张图片 |

OCR 文字不受 VLM 面积阈值影响：小图里如果有字，仍会尝试识别；只是不去花一次视觉推理。

## 3. 本地部署（生产目标形态）

代码只依赖 OpenAI 兼容的 `/chat/completions`，不绑定某个 SDK。本地推荐用 vLLM 起 Qwen3-VL：

```bash
# 示例：模型名按你实际下载的权重目录改；端口保持回环
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1
```

配置：

```dotenv
KEE_VLM_ENABLED=true
KEE_VLM_BASE_URL=http://127.0.0.1:8000/v1
KEE_VLM_MODEL=Qwen3-VL-8B-Instruct
KEE_VLM_API_KEY=
KEE_VLM_ALLOW_REMOTE=false
```

**8GB 单卡的部署纪律**：VLM 只在**入库**时用，建议入库时停掉在线 API/7B，或单独在另一台机器跑 vLLM；不要把 Qwen3-VL-8B 与 qwen2.5:7b、BGE-M3、reranker 同时常驻在一张 8GB 卡上。代码不会在服务启动时加载 VLM；它只在 `IngestionService` 真正解析图片时按需调用 HTTP。

## 4. 云端 API（仅开发/测试）

用户明确允许测试期用云端 API。代码的默认值是**关闭 + 拒绝公网**；要使用云端，必须同时满足：

```dotenv
KEE_VLM_ENABLED=true
KEE_VLM_BASE_URL=https://<供应商的-OpenAI-兼容地址>/v1
KEE_VLM_MODEL=<视觉模型名>
KEE_VLM_API_KEY=<只放环境变量或 .env，不写仓库文件>
KEE_VLM_ALLOW_REMOTE=true
```

- `KEE_VLM_ALLOW_REMOTE=true` 是书面确认：**图片与提示词会离开本机**。生产交付必须关掉它或换回本地。
- 缺 API key 的云端目标会直接禁用，不会在入库时反复 401。
- 日志只记模型名、耗时、状态，不记图片字节、不记 API key。
- 云端调用失败不阻断入库：该图退化为 OCR 文字；连续失败达到 `KEE_VLM_FAIL_FAST_AFTER` 后熔断本进程后续 VLM 调用。

## 5. 公式识别（本地 ONNX，不走网络）

`PP-FormulaNet_plus-M` 权重本机已有（564MB），默认路径：

```
D:/modelscope/hub/models/MinerU-4_models_onnx/MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.onnx
```

- 只有版面模型真的判出公式区域时才加载这个 564MB 会话；普通文档不付这笔成本。
- 预处理：白边裁剪 → 灰度 → 等比缩放 → 384×384 白底居中 → `[-1,1]` 归一化。
- 输出 token ids，用模型目录 `*_inference.yml` 里的 tokenizer 解码为 LaTeX。
- 公式模型不可用/推理失败：先尝试 VLM 转写，再不行保留 OCR 文本行。
- 已知边界：公式的版面结构（分式上下、矩阵对齐）仍可能识别错；当前实测一页二次求根公式能出 LaTeX，但分式结构不保证每次正确。正式使用前要用目标行业公式集做准确率验收。

## 6. 失败与降级

| 情况 | 行为 |
|---|---|
| VLM 未启用/目标越域 | 只用 OCR 文字；note 记录 `VLM 未启用：...` |
| VLM 单次失败/超时 | 该图描述为空，OCR 与原文照常入库；连续失败后熔断 |
| OCR 模型不可用 | 仍可走 VLM；两者都不可用则跳过该图并记 note |
| 公式模型缺失 | 回退 VLM；再回退 OCR 文本行 |
| 图片解码失败（EMF/WMF 等） | 跳过该图，记 note，不拖垮整篇文档 |
| 图片数量超限 | 只处理前 N 张，note 记录总数与处理数 |

## 7. 代码索引

| 文件 | 职责 |
|---|---|
| `src/kee/ingest/vlm.py` | OpenAI 兼容 VLM 客户端；域内地址闸门、熔断、失败不抛 |
| `src/kee/ingest/media.py` | DOCX/PDF 图片抽取、OCR、VLM 描述、段落格式化、数量闸门 |
| `src/kee/ingest/parsers.py` | `parse_docx` 接图片；新增 `parse_image` 与图片后缀分派；文本层 PDF 内嵌图片补抽 |
| `src/kee/ingest/mineru_onnx.py` | PDF 图表/公式区域；PP-FormulaNet_plus-M；重叠框去重；页内数量闸门 |
| `src/kee/config.py` | `KEE_VLM_*`、`KEE_PARSE_FORMULA`、`KEE_MEDIA_SCAN_TEXT_PDF` |
| `src/kee/api/app.py` | `ParseFailure` → `E422`，图片解析失败不再变成 500 |
| `tests/test_vlm_client.py` | 禁用默认、域外闸门、OpenAI 载荷、熔断 |
| `tests/test_media_pipeline.py` | DOCX/独立图片/PDF 内嵌图片、描述进入分块 |
| `tests/test_formula_media.py` | 公式预处理、注入 session 解码、公式/图表段落 |

## 8. 当前验证记录（2026-10-26）

| 项 | 验证方式 | 结果 |
|---|---|---|
| 全量后端单测 | `pytest -q`（存储栈临时起） | **390 passed**（含新增 14 项图片/公式/VLM 测试） |
| DOCX 内嵌图片 | 生成带图片 DOCX + fake OCR/VLM | 图片段落进入「图片与图表」章节，再进入分块；admission 通过 |
| 独立图片 | 生成 PNG + fake OCR/VLM | 单图文档 → 图片段落 → 分块 |
| 文本层 PDF 内嵌图片 | 三列表格 PDF + fake OCR/VLM | `pdf_images` 枚举到图片并产出描述段落 |
| 扫描件 PDF | 真实 MinerU 版面/OCR + fake VLM | 6 个区域做 VLM、2 个只保留 OCR、7 个小图/超限跳过 |
| 公式 PDF | 真实 PP-FormulaNet_plus-M + 渲染公式页 | 输出 LaTeX；二次求根公式的分式结构仍不完美 |
| VLM 协议 | httpx MockTransport | 请求载荷/内容解析/域外闸门/熔断通过 |
| 真实云端 API | Modelscope OpenAI 兼容端点 + 合成图表/真实扫描件 | **4 条入口全部成功**；单张 3–6s；详见 `reports/m1/M1_图片图表公式解析.md` |
| 本地 Qwen3-VL | **未实跑** | 本机无该权重/服务；代码路径已就绪 |

## 9. 如实边界

- **VLM 描述是模型生成**，不是原文；引用时必须标「内容描述」，不能当原文逐字引用。
- **图表语义理解的质量取决于 VLM**；当前没有在真实行业图表集上做准确率验收。
- **不保存图片资产**：只把 OCR 文字和描述写进知识块，没有图片文件存储、缩略图或原图引用；因此答案能说「图里写了什么」，不能回链到原图。
- **没有视觉向量检索**（CLIP/ColPali 一类）；检索仍然靠文本描述命中。
- **公式准确率未验收**：版面结构复杂时可能出错；公式区域检测依赖 PP-DocLayoutV2，若公式被识别成普通图片则走 VLM/OCR。
- **不支持 xlsx/ppt**；DOCX 里的图表若以 OLE 对象嵌入，`word/media` 里可能只有预览图，描述质量受限。
- **本地 Qwen3-VL 尚未在本机实跑**：代码路径用 mock 验证；公式模型在本机实测过；云端 API 用 MockTransport 验证协议，未调用真实供应商。
- **已有索引不会自动回填**：VLM/公式是解析期能力，开启后只影响新摄取或重新摄取的文档；已有 24 篇纯文本语料不会因此长出图片描述。
- **入库期变慢**：VLM 每张图一次调用；20 张图 × 数秒，必须按离线作业排期，不要放在交互式上传的同步路径上做大批量导入。
