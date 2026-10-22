/** 后端契约的类型镜像。
 *
 * 这些形状来自 `src/kee/qa/answer.py:AnswerResult`、`src/kee/qa/graph.py` 的 confidence 节点
 * 与 `src/kee/api/app.py` 的端点包装，不是前端自己拟的字段名。
 * 之所以保留 `Record<string, unknown>` 的几处：`retrieval` / `llm` 是引擎内部诊断块，
 * 结构随实现变，前端只读其中稳定键；把它写成精确类型会假装有稳定性。
 */

export type AnswerStatus = "answer" | "tentative" | "refuse" | "escalate" | "degraded" | string;

export interface Citation {
  evidence_id: string;
  chunk_id: string;
  doc_id: string;
  project?: string | null;
  section_no?: string | null;
  page?: number | null;
  security_level?: number | null;
  parent_id?: string | null;
  score?: number | null;
}

export interface KeyPoint {
  point: string;
  evidence_ids: string[];
}

/** 复合规则分。**不是概率**：`score_basis` 恒为 `rule_composite_not_probability`，
 * 界面上任何展示都不得把它写成「置信度 92%」这类概率口径。 */
export interface ConfidenceBlock {
  retrieval?: number;
  citation_consistency?: number;
  self_assessed?: number;
  composite?: number;
  logit?: number;
  calibrated?: number;
  temperature?: number;
  rule?: string;
  score_basis?: string;
}

export interface AnswerPayload {
  question: string;
  status: AnswerStatus;
  conclusion: string;
  key_points: KeyPoint[];
  citations: Citation[];
  dropped_points: string[];
  needs_human: boolean;
  confidence: ConfidenceBlock;
  retrieval: Record<string, unknown>;
  llm: Record<string, unknown>;
  timings_ms: Record<string, number>;
  trace_id: string;
  notes: string[];
  /** 拒答原因：`no_evidence`（库里真没有）与 `all_evidence_unreviewed`（资料在、还没复核）必须分得开 */
  refusal_reason?: string;
  /** 本轮被复核闸摘掉的未复核证据条数；答出来了也照样带，界面上是「还有 N 条没复核」 */
  unreviewed_evidence_count?: number;
  session_id?: string;
  turn_no?: number;
  new_session?: boolean;
  /** token 级流式的可观测撤回：流过多少字、最后有没有成为对外答案 */
  provisional_streamed_chars?: number;
  provisional_retracted?: boolean;
  provisional_lost?: number;
  provisional_lost_samples?: string[];
}

export interface DeltaPayload {
  field: string;
  index?: number | null;
  text: string;
  closed: boolean;
  provisional: true;
  chars: number;
}

export interface RetrievalEvent {
  query_used?: string;
  mode?: string;
  /** 后端发的是**原因数组**（`Res.degraded: list[str]`）。类型曾写成 `boolean | string`，
   *  于是 `degraded: []` 在 JSX 三元里被判成真值——「检索降级」挂在每一轮上，与有没有降级无关。 */
  degraded?: string[];
  n_chunks?: number;
  chunk_ids?: string[];
  timings_ms?: Record<string, number>;
  [k: string]: unknown;
}

/** 统一错误体（方案 7.3）：E4xx 客户端 / E5xx 服务端 / E9xx 依赖。 */
export interface ApiErrorBody {
  code: string;
  message: string;
  trace_id: string;
  retryable: boolean;
}

export interface AnswerRequest {
  question: string;
  session_id?: string | null;
  top_k?: number;
  rewrite?: boolean;
  token_stream?: boolean;
}
