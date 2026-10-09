# Gate 0 ② 中文稀疏检索（服务端 BM25 + 分析器）验证报告

- 验证时间：2026-10-09T16:41:16+08:00｜commit 5bb82f0｜Milvus 3.0.0 + pymilvus 3.0.2

## 关键发现（与方案假设的差异）

1. `analyzer_params={"type":"jieba"}` 被 Milvus v3.0.0 服务端拒绝：
   `unknown build-in analyzer type: jieba`（internal/core/thirdparty/tantivy/tokenizer.h）。
   **可用形态是 `{"tokenizer":"jieba"}`**，方案 11.2 的 analyzer 配置基线必须按此锁定。
2. 另有内置 `"type":"chinese"` 分析器可用，一并纳入对照。
3. 默认（不给 params，仅 enable_analyzer）对中文查询近乎无效——见下表基线行。

## 分析器与索引规模

| 变体 | 服务端落库的 analyzer 参数 | 子块数 |
|---|---|---|
| jieba | {"max_length": 8192, "enable_analyzer": "true", "analyzer_params": "{\"tokenizer\":\"jieba\"}"} | 193 |
| standard | {"max_length": 8192, "enable_analyzer": "true", "analyzer_params": "{\"type\":\"standard\"}"} | 193 |
| chinese | {"max_length": 8192, "enable_analyzer": "true", "analyzer_params": "{\"type\":\"chinese\"}"} | 193 |
| default_none | {"max_length": 8192, "enable_analyzer": "true"} | 193 |

## 切分行为探针（term → 命中子块数）

| term | jieba | standard | chinese | default_none |
|---|---|---|---|---|
| 银企直联 | 3 | 0 | 3 | 0 |
| 企直联 | 2 | 0 | 2 | 0 |
| 实时清算 | 20 | 0 | 20 | 0 |
| 线损率 | 18 | 0 | 18 | 0 |
| 损率 | 3 | 0 | 3 | 0 |
| SCADA 主站 | 20 | 6 | 20 | 6 |
| IEC 60870-5-104 | 20 | 20 | 6 | 20 |
| DCS 联锁 | 20 | 6 | 20 | 6 |
| P1 | 10 | 10 | 10 | 10 |

## 稀疏检索对照（Recall@5 / hit@5 / MRR，仅 BM25 单路）

| 变体 | analyzer | Recall@5 | hit@5 | MRR | 零召回率 | ΔRecall@5(对默认) | P95 检索 ms |
|---|---|---|---|---|---|---|---|
| jieba | {"max_length": 8192, "enable_analyzer": "true", "analyzer_params": "{\"tokenizer\":\"jieba\"}"} | 0.7612 | 0.9483 | 0.9172 | 0.0517 | 0.3622 | 5.1 |
| standard | {"max_length": 8192, "enable_analyzer": "true", "analyzer_params": "{\"type\":\"standard\"}"} | 0.399 | 0.4828 | 0.4244 | 0.5172 | 0.0 | 3.6 |
| chinese | {"max_length": 8192, "enable_analyzer": "true", "analyzer_params": "{\"type\":\"chinese\"}"} | 0.7555 | 0.9483 | 0.9259 | 0.0517 | 0.3565 | 4.1 |
| default_none | {"max_length": 8192, "enable_analyzer": "true"} | 0.399 | 0.4828 | 0.4244 | 0.5172 | 0.0 | 3.7 |

- 评测集可判分题数：58；语料子块总数：193
- 最优中文分析器：**jieba**，相对默认 Recall@5 增益 **0.3622**
- 分类别增益（对默认）：{"术语": 0.4592, "编号": 0.2639, "方案名": 0.125}

## 建库与重建

- 首次建库（含嵌入）：{"authoritative_write": 3.86, "embed": 1.76, "total": 273.21}
- 索引重建（不碰 GPU，仅权威源+向量物料）：{"rebuild_standard": 3.87, "rebuild_chinese": 4.03, "rebuild_default_none": 3.99, "rebuild_canonical": 4.06}
- 准入：{"total": 24, "accepted": 24, "rejected": 0, "duplicates": 0}

- **判定**：通过｜门禁：中文分析器生效且 BM25 相对默认分析器有明确正增益（非≈0）；术语/编号类 query 明显改善｜实测：jieba Recall@5 0.7612 vs 默认 0.399（Δ=0.3622）；术语/编号类增益 {"术语": 0.4592, "编号": 0.2639, "方案名": 0.125}｜说明：jieba 需以 {"tokenizer":"jieba"} 形态配置；方案文档口径需同步修订

## 参数基线（②出口）

- `analyzer_params = {"tokenizer": "jieba"}`（首选），`{"type": "chinese"}` 作为备选。
- 稀疏索引：`SPARSE_INVERTED_INDEX` + `metric_type=BM25`；稠密索引本轮用 `FLAT/COSINE`（精确，避免 ANN 近似污染消融）。
- 索引重建路径：PostgreSQL 权威源 + 本地向量物料 → Milvus，无需 GPU 重算。
