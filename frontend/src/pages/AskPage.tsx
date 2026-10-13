import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, ApiError, type Identity } from "../api/client";
import { postSse } from "../api/sse";
import type { AnswerPayload, Citation, DeltaPayload, RetrievalEvent } from "../api/types";
import { useIdentity } from "../state/identity";
import { styles as s } from "../styles";

/** 问答页：把流式契约的三态如实画出来。
 *
 * 契约里最容易被前端做错的三件事，都在这里钉住：
 * 1. `delta.text` 是**该槽位的累计内容**（后端按 (field,index) 去重后发出），所以整段替换、
 *    不是拼接——当增量拼会出重复字。
 * 2. `delta` 必然**早于校验**（引用强制要等整段 JSON 解析完），所以它是 provisional，必须用与
 *    最终答案不同的样式；`answer` 到达后若流过的字没成为答案，**显示「已撤回」而不是静默删除**。
 * 3. 置信度是规则分不是概率，`score_basis` 原样带出来。
 *
 * 引用片段正文另走 `POST /api/v1/chunks/batch`：答案里的 citations 刻意不带正文，
 * 而那条通道与检索层同口径过滤，所以「越权」在这里表现为拿不到正文，不是前端遮住。
 */

const KEY = (field: string, index: number | null | undefined) => `${field}:${index ?? "-"}`;

/** 载荷解析：流是外部输入，坏帧要报出来而不是被 `JSON.parse` 抛成一句天书，更不能静默跳过——
 *  跳过的代价是界面永远停在「生成中」，那正是这套契约最不该出现的状态。 */
function parseEvent(evt: { event: string; data: string }): unknown {
  if (!evt.data.trim()) throw new Error(`事件 ${evt.event} 的载荷为空（未按契约给出 JSON）`);
  try {
    return JSON.parse(evt.data) as unknown;
  } catch {
    throw new Error(`事件 ${evt.event} 的载荷不是合法 JSON：${evt.data.slice(0, 60)}`);
  }
}

interface Slot {
  text: string;
  closed: boolean;
}

interface Turn {
  id: number;
  question: string;
  phase: "streaming" | "done" | "error";
  retrieval?: RetrievalEvent;
  slots: Record<string, Slot>;
  order: string[];
  streamedChars: number;
  answer?: AnswerPayload;
  error?: string;
  startedAt: number;
  firstDeltaMs?: number;
}

const STATUS_LABEL: Record<string, string> = {
  answer: "已作答",
  tentative: "初步判断（低置信度显式降级）",
  refuse: "未给出结论",
  escalate: "已转人工",
  degraded: "降级",
};

let turnSeq = 0;

export function AskPage() {
  const { identity } = useIdentity();
  const [question, setQuestion] = useState("");
  const [tokenStream, setTokenStream] = useState(true);
  const [rewriteOnFollowup, setRewriteOnFollowup] = useState(true);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [activeEvidence, setActiveEvidence] = useState<string | null>(null);
  const sessionRef = useRef<string | null>(null);

  const patch = useCallback((id: number, fn: (t: Turn) => Turn) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? fn(t) : t)));
  }, []);

  const ask = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q || busy) return;
      const id = ++turnSeq;
      setTurns((prev) => [
        ...prev,
        { id, question: q, phase: "streaming", slots: {}, order: [], streamedChars: 0, startedAt: performance.now() },
      ]);
      setQuestion("");
      setBusy(true);
      const controller = new AbortController();

      // 追问才开改写：首轮字面查询本来就自足，为它多花一次模型调用不划算
      // （实测单次改写 P95 843ms，几乎吃满 800ms 的检索门禁预算）。
      const isFollowup = Boolean(sessionRef.current);
      const body = {
        question: q,
        session_id: sessionRef.current,
        top_k: 5,
        rewrite: isFollowup && rewriteOnFollowup,
        token_stream: tokenStream,
      };

      try {
        await postSse("/api/v1/answers/stream", {
          identity,
          body,
          signal: controller.signal,
          onEvent: (evt) => {
            if (evt.event === "retrieval") {
              patch(id, (t) => ({ ...t, retrieval: parseEvent(evt) as RetrievalEvent }));
              return;
            }
            if (evt.event === "delta") {
              const d = parseEvent(evt) as DeltaPayload;
              patch(id, (t) => {
                const key = KEY(d.field, d.index);
                return {
                  ...t,
                  slots: { ...t.slots, [key]: { text: d.text, closed: d.closed } },
                  order: key in t.slots ? t.order : [...t.order, key],
                  streamedChars: d.chars,
                  firstDeltaMs: t.firstDeltaMs ?? Math.round(performance.now() - t.startedAt),
                };
              });
              return;
            }
            if (evt.event === "answer") {
              const a = parseEvent(evt) as AnswerPayload;
              if (a.session_id) sessionRef.current = a.session_id;
              patch(id, (t) => ({ ...t, phase: "done", answer: a }));
              return;
            }
            if (evt.event === "error") {
              const e = parseEvent(evt) as { code?: string; message?: string };
              patch(id, (t) => ({ ...t, phase: "error", error: `${e.code ?? "E500"} ${e.message ?? "流式中断"}` }));
            }
          },
        });
        // 流结束了却没有 answer：不能留在「生成中」的样子，那等于界面自己假装成功
        patch(id, (t) =>
          t.phase === "streaming"
            ? { ...t, phase: "error", error: "连接结束但未收到 answer 事件（答案未交付，不算成功）" }
            : t,
        );
      } catch (e) {
        const msg = e instanceof ApiError ? e.display : e instanceof Error ? e.message : String(e);
        patch(id, (t) => (t.phase === "streaming" ? { ...t, phase: "error", error: msg } : t));
      } finally {
        setBusy(false);
      }
    },
    [busy, identity, patch, rewriteOnFollowup, tokenStream],
  );

  return (
    <div>
      <form
        className={s.askForm}
        onSubmit={(e) => {
          e.preventDefault();
          void ask(question);
        }}
      >
        <input
          type="search"
          value={question}
          placeholder="就知识库提问，例如：SIS 安全仪表系统采用的表决逻辑是什么？"
          onChange={(e) => setQuestion(e.target.value)}
          aria-label="问题"
        />
        <button type="submit" className={`${s.btn} ${s.btnPrimary}`} disabled={busy || !question.trim()}>
          {busy ? "生成中…" : "提问"}
        </button>
        <button
          type="button"
          className={s.btn}
          onClick={() => {
            sessionRef.current = null;
            setTurns([]);
          }}
        >
          新会话
        </button>
      </form>

      <div className={s.toolbar}>
        <label>
          <input type="checkbox" checked={tokenStream} onChange={(e) => setTokenStream(e.target.checked)} />
          token 级流式（关掉退回阶段级两事件，用于对照）
        </label>
        <label>
          <input type="checkbox" checked={rewriteOnFollowup} onChange={(e) => setRewriteOnFollowup(e.target.checked)} />
          追问时启用查询改写（实测单次 P95 843ms）
        </label>
        {sessionRef.current ? <span className={s.meta}>会话 {sessionRef.current}</span> : null}
      </div>

      <div className={s.turns}>
        {!turns.length ? (
          <p className={s.empty}>还没有提问。答案的结论、要点、引用落点、规则分与降级原因都会显示在这里。</p>
        ) : null}
        {turns.map((t) => (
          <AnswerView
            key={t.id}
            turn={t}
            identity={identity}
            activeEvidence={activeEvidence}
            onEvidence={setActiveEvidence}
            onAsk={ask}
          />
        ))}
      </div>
    </div>
  );
}

function AnswerView({
  turn,
  identity,
  activeEvidence,
  onEvidence,
  onAsk,
}: {
  turn: Turn;
  identity: Identity;
  activeEvidence: string | null;
  onEvidence: (id: string | null) => void;
  onAsk: (q: string) => void;
}) {
  const answer = turn.answer;
  const cites = answer?.citations ?? [];
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [hidden, setHidden] = useState<string[]>([]);
  /** 片段读取是否已落定。没有这个标志时，「还没读回来」会被显示成「正文未返回」——
   *  那是一句假话，而且正是这条界面把越权和慢区分不开的原因。 */
  const [settled, setSettled] = useState(false);

  useEffect(() => {
    const ids = Array.from(new Set(cites.map((c) => String(c.chunk_id)).filter(Boolean)));
    setTexts({});
    setHidden([]);
    setSettled(false);
    if (!ids.length) return;
    let alive = true;
    apiFetch<{ items: { chunk_id: string; content: string }[]; hidden: string[] }>("/api/v1/chunks/batch", {
      identity,
      body: { ids },
    })
      .then((r) => {
        if (!alive) return;
        setTexts(Object.fromEntries(r.items.map((i) => [i.chunk_id, i.content])));
        setHidden(r.hidden ?? []);
        setSettled(true);
      })
      // 取正文失败不影响答案本身：引用元数据（文档/章节/页码/密级）已经在 answer 里
      .catch(() => {
        if (!alive) return;
        setHidden(ids);
        setSettled(true);
      });
    return () => {
      alive = false;
    };
    // cites 派生自 answer，answer 换了就该重取；identity 变了权限面也变了
  }, [answer, identity]);

  const streamedConclusion = turn.slots[KEY("conclusion", null)]?.text ?? "";
  const pointSlots = turn.order.filter((k) => k.startsWith("point:"));
  const finalConclusion = (answer?.conclusion ?? "").trim();
  const retracted = Boolean(answer?.provisional_retracted);
  const lostStreamedText = Boolean(answer) && !finalConclusion && streamedConclusion.length > 0;

  return (
    <article className={s.turn} data-phase={turn.phase}>
      <h3 className={s.turnQuestion}>{turn.question}</h3>

      {turn.retrieval ? <RetrievalSkeleton ev={turn.retrieval} firstDeltaMs={turn.firstDeltaMs} /> : null}
      {turn.error ? <p className={s.error} data-testid="turn-error">{turn.error}</p> : null}

      {finalConclusion ? (
        <p className={`${s.conclusion} ${s.confirmed}`} data-slot="conclusion" data-testid="conclusion-final">
          {finalConclusion}
        </p>
      ) : retracted || lostStreamedText ? (
        <div data-slot="conclusion">
          <p className={`${s.conclusion} ${s.retracted}`} data-testid="conclusion-retracted">
            {streamedConclusion}
          </p>
          <p className={s.retracted} data-testid="retraction-notice">
            已撤回：流出的 {turn.streamedChars} 字未成为对外答案
            {answer ? `（出口状态 ${answer.status}）` : ""}。原文保留在此，不静默删除。
          </p>
        </div>
      ) : streamedConclusion && turn.phase === "streaming" ? (
        <p className={`${s.conclusion} ${s.provisional}`} data-slot="conclusion" data-testid="conclusion-provisional">
          {streamedConclusion}
        </p>
      ) : answer && !finalConclusion ? (
        <p className={`${s.conclusion} ${s.confirmed}`} data-slot="conclusion" data-testid="no-conclusion">
          <span className={`${s.badge} ${s.badgeRefuse}`}>{STATUS_LABEL[answer.status] ?? answer.status}</span>{" "}
          本轮没有可给出的结论
          {answer.status === "refuse" ? "（语料内无依据 → 拒答）" : ""}
          {answer.status === "escalate" ? "（已转人工）" : ""}
        </p>
      ) : null}

      {answer?.key_points.length ? (
        <ul className={s.points}>
          {answer.key_points.map((p, i) => (
            <li className={s.point} key={`${i}-${p.point}`}>
              <span className={hitClass(p, activeEvidence)}>{p.point}</span>{" "}
              {p.evidence_ids.map((e) => (
                <button type="button" className={s.link} key={e} onClick={() => onEvidence(e)}>
                  [{e}]
                </button>
              ))}
            </li>
          ))}
        </ul>
      ) : turn.phase === "streaming" && pointSlots.length ? (
        <ul className={s.points}>
          {pointSlots.map((k) => (
            <li className={`${s.point} ${s.provisional}`} key={k}>
              {turn.slots[k].text}
            </li>
          ))}
        </ul>
      ) : null}

      {answer?.dropped_points.length ? (
        <p className={s.retracted}>
          引用强制丢掉 {answer.dropped_points.length} 条要点（无对应出处）：
          {answer.dropped_points.map((d, i) => (
            <span key={i}>「{d}」</span>
          ))}
        </p>
      ) : null}

      {answer ? (
        <AnswerMeta answer={answer} cites={cites} texts={texts} hidden={hidden} settled={settled} onEvidence={onEvidence} />
      ) : null}

      {answer?.status === "answer" ? (
        <div className={s.toolbar}>
          <span className={s.meta}>追问建议：</span>
          {["这个结论的适用边界与例外是什么？", "依据的是哪份文档的哪一版？", "上游依赖不满足时会怎样降级？"].map((q) => (
            <button type="button" className={s.btn} key={q} onClick={() => void onAsk(q)}>
              {q}
            </button>
          ))}
        </div>
      ) : null}
    </article>
  );
}

const hitClass = (p: { evidence_ids: string[] }, active: string | null) =>
  active && p.evidence_ids.includes(active) ? s.hit : undefined;

function RetrievalSkeleton({ ev, firstDeltaMs }: { ev: RetrievalEvent; firstDeltaMs?: number }) {
  const n = Array.isArray(ev.chunk_ids) ? ev.chunk_ids.length : Number(ev.n_chunks ?? 0);
  const ms = ev.timings_ms?.retrieval_ms;
  return (
    <p className={s.meta} data-testid="retrieval-skeleton">
      检索 {n} 块{ev.mode ? ` · 模式 ${String(ev.mode)}` : ""}
      {ev.query_used ? ` · 实际查询「${String(ev.query_used)}」` : ""}
      {Array.isArray(ev.degraded) && ev.degraded.length > 0 ? ` · 检索降级（${ev.degraded.join("；")}）` : ""}
      {ms ? ` · ${Math.round(ms)}ms` : ""}
      {firstDeltaMs ? ` · 首个可见增量 ${firstDeltaMs}ms` : ""}
    </p>
  );
}

function AnswerMeta({
  answer,
  cites,
  texts,
  hidden,
  settled,
  onEvidence,
}: {
  answer: AnswerPayload;
  cites: Citation[];
  texts: Record<string, string>;
  hidden: string[];
  settled: boolean;
  onEvidence: (id: string | null) => void;
}) {
  const c = answer.confidence ?? {};
  return (
    <div className={s.meta}>
      <p className={s.conf} data-testid="confidence">
        <span className={`${s.badge} ${answer.needs_human ? s.badgeWarn : ""}`} data-testid="status-badge">
          {STATUS_LABEL[answer.status] ?? answer.status}
        </span>
        {answer.needs_human ? (
          <span className={`${s.badge} ${s.badgeWarn}`} data-testid="needs-human">
            需人工复核
          </span>
        ) : null}
        {c.composite !== undefined ? (
          <span>
            规则分 composite={c.composite} · calibrated={c.calibrated} · T={c.temperature} ·{" "}
            <em>{c.score_basis ?? "score_basis 缺失"}</em>
          </span>
        ) : c.rule ? (
          <span>契约出口 rule={c.rule}（composite 记 0：这条答案没成立）</span>
        ) : null}
      </p>

      {c.retrieval !== undefined || c.citation_consistency !== undefined ? (
        <p className={s.conf}>
          成分：retrieval={c.retrieval} · citation_consistency={c.citation_consistency} · self_assessed=
          {c.self_assessed}
        </p>
      ) : null}

      {answer.timings_ms?.e2e_ms ? <p>端到端 {Math.round(answer.timings_ms.e2e_ms)}ms · 轮次 {answer.turn_no}</p> : null}
      {answer.notes.length ? <p>判定记录：{answer.notes.join("；")}</p> : null}

      <div className={s.cites}>
        <strong>引用（{cites.length}）</strong>
        {cites.map((ct) => (
          <div
            className={s.cite}
            key={String(ct.evidence_id)}
            data-testid={`cite-${String(ct.evidence_id)}`}
            onMouseEnter={() => onEvidence(String(ct.evidence_id))}
            onMouseLeave={() => onEvidence(null)}
            data-evidence={String(ct.evidence_id)}
          >
            <div>
              [{String(ct.evidence_id)}] {String(ct.doc_id)} · 章节 {String(ct.section_no ?? "-")} · 第{" "}
              {String(ct.page ?? "-")} 页 · 密级 {String(ct.security_level ?? "-")}
            </div>
            <div className={s.meta}>
              {texts[String(ct.chunk_id)] ??
                (hidden.includes(String(ct.chunk_id))
                  ? "（当前身份取不到该片段正文）"
                  : settled
                    ? "（正文未返回）"
                    : "（正文读取中…）")}
            </div>
          </div>
        ))}
      </div>

      <p>
        免责提示：内容由本地模型依据知识库生成，引用落点可逐条核对；置信度为规则分、不是正确概率。
        {answer.provisional_streamed_chars ? ` 本次流出 ${answer.provisional_streamed_chars} 字` : ""}
        {answer.provisional_lost ? `，条目级丢失 ${answer.provisional_lost} 条` : ""}
        {answer.provisional_lost_samples?.length ? `（示例：${answer.provisional_lost_samples[0]}）` : ""}。
      </p>
    </div>
  );
}
