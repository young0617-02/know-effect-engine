"""对内 REST API（方案 7.2 / 7.3 契约落地）。

三条纪律在这一层变成可测的行为：
  - 统一错误码：E4xx 客户端 / E5xx 服务端 / E9xx 依赖，响应体一律带 code/message/trace_id/retryable。
  - 幂等：写操作用 Idempotency-Key 头，重复提交返回首次结果而不是重复副作用。
  - 职责分离：知识 Owner 才能复核，且**上传者不能自审**（拿审计里的 actor 比对）。
"""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pymilvus.exceptions import MilvusException

from kee.config import get_settings
from kee.db.outbox import OutboxDispatcher
from kee.db.pg import AuthoritativeStore
from kee.db.reconcile import Reconciler
from kee.db.vector_store import VectorStore
from kee.ingest.parsers import UnsupportedFormat
from kee.ingest.service import AdmissionRejected, DuplicateDocument, IngestionService
from kee.retrieval.milvus_store import KbStore
from kee.retrieval.rbac import AccessContext, build_filter_expr

REVIEW_ROLES = {"knowledge_owner", "system_admin"}
RETRYABLE_PREFIXES = ("E9", "E5")


class ApiError(Exception):
    """统一 API 错误。不能用 @dataclass 套在 Exception 上：BaseException 的 args/__new__
    与 dataclass 生成的 __init__ 冲突，抛出后会被兜成 500。"""

    def __init__(self, code: str, http_status: int, message: str, trace_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message
        self.trace_id = trace_id


def _load_calibration(settings) -> dict[str, Any]:
    """服务路径的温度参数。**与离线探针共用同一个 loader**（`kee.qa.answer`），
    否则「测出来的分布」和「服务出去的分布」会再度分叉——M3-04 就栽过一次。
    缺失时退回 T=1.0（等于不缩放），不猜值。
    """
    from kee.qa.answer import load_calibration_temperature

    t = load_calibration_temperature(settings)
    return {"temperature": t}


def _fail(code: str, status: int, message: str, trace_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "code": code,
            "message": message,
            "trace_id": trace_id,
            "retryable": code.startswith(RETRYABLE_PREFIXES),
        },
    )


class ReviewRequest(BaseModel):
    decision: str = Field(pattern="^(pass|reject)$")
    note: str = ""


class RollbackRequest(BaseModel):
    to_version_id: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64)
    top_k: int = Field(default=5, ge=1, le=10)
    # 查询改写默认关：实测单次 P95 843ms，一步就吃满 800ms 检索门禁（方案 V1.0.3 第 2 行）。
    # 多轮追问时前端显式置 true——那才是字面查询不自足、值得花这一次调用的场景。
    rewrite: bool = False


def create_app(*, skip_embedding: bool = False, scheduler: Any | None = None, collection: str | None = None) -> FastAPI:
    """skip_embedding=True 用于契约测试：链路全跑通但不占 GPU、不需要模型底座。

    collection 可指向独立集合，让验证跑在不污染已有基线数据的命名空间里。
    """
    settings = get_settings()
    pg = AuthoritativeStore()
    kb = KbStore(collection or settings.milvus_collection, "jieba")
    vectors = VectorStore()
    service = IngestionService(pg, kb, vectors, scheduler=scheduler)

    app = FastAPI(title="知效引擎 KEE", version="0.1.0")
    app.state.pg = pg
    app.state.kb = kb
    app.state.service = service
    app.state.scheduler = scheduler
    app.state.skip_embedding = skip_embedding

    @app.middleware("http")
    async def trace_layer(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or f"api-{uuid.uuid4().hex[:12]}"
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _fail(exc.code, exc.http_status, exc.message, request.state.trace_id)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return _fail("E400", 400, f"请求校验失败：{exc.errors()[:3]}", request.state.trace_id)

    @app.exception_handler(MilvusException)
    async def _milvus(request: Request, exc: MilvusException):
        return _fail("E900", 503, f"向量库不可用：{str(exc)[:180]}", request.state.trace_id)

    @app.exception_handler(psycopg.Error)
    async def _pg(request: Request, exc: psycopg.Error):
        return _fail("E901", 503, f"权威源不可用：{str(exc)[:180]}", request.state.trace_id)

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    async def _bad(request: Request, exc: Exception):
        code = "E404" if isinstance(exc, KeyError) else "E400"
        status = 404 if code == "E404" else 400
        return _fail(code, status, str(exc)[:200].strip("'"), request.state.trace_id)

    def context_of(
        role: str | None,
        actor: str | None,
        industries: str | None,
        projects: str | None,
    ) -> tuple[AccessContext, str]:
        if not role:
            raise ApiError("E401", 401, "缺少 X-Kee-Role：检索层权限过滤要求显式身份", "")
        ctx = AccessContext(
            role=role,
            industries=[x for x in (industries or "").split(",") if x],
            projects=[x for x in (projects or "").split(",") if x],
        )
        return ctx, actor or role

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        deps: dict[str, Any] = {}
        try:
            deps["milvus"] = {"ok": True, "collection": kb.collection, "chunks": kb.count()}
        except Exception as e:  # noqa: BLE001
            deps["milvus"] = {"ok": False, "error": f"{type(e).__name__}"}
        try:
            with pg.connect() as conn:
                deps["postgres"] = {"ok": True, "documents": conn.execute("SELECT count(*) AS c FROM document").fetchone()["c"]}
        except Exception as e:  # noqa: BLE001
            deps["postgres"] = {"ok": False, "error": f"{type(e).__name__}"}
        return {"status": "ok" if all(d.get("ok") for d in deps.values()) else "degraded", "deps": deps}

    @app.post("/api/v1/documents/upload")
    async def upload(
        request: Request,
        file: UploadFile = File(...),
        industry: str | None = Form(None),
        customer: str | None = Form(None),
        project: str | None = Form(None),
        doc_type: str | None = Form(None),
        doc_id: str | None = Form(None),
        title: str | None = Form(None),
        source_level: str | None = Form(None),
        security_level: int | None = Form(None),
        x_kee_role: str | None = Header(None),
        x_kee_actor: str | None = Header(None),
        idempotency_key: str | None = Header(None),
    ) -> JSONResponse:
        trace_id = request.state.trace_id
        ctx, actor = context_of(x_kee_role, x_kee_actor, None, None)
        if ctx.role not in ("knowledge_owner", "system_admin", "delivery_engineer", "pm", "presales_architect"):
            raise ApiError("E403", 403, f"角色 {ctx.role} 无知识写入权限", trace_id)

        data = await file.read()
        if not data:
            raise ApiError("E400", 400, "空文件", trace_id)
        try:
            result = service.ingest(
                filename=file.filename or "untitled",
                data=data,
                form={
                    "industry": industry,
                    "customer": customer,
                    "project": project,
                    "doc_type": doc_type,
                    "doc_id": doc_id,
                    "title": title,
                    "source_level": source_level,
                    "security_level": security_level,
                },
                actor=actor,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
                skip_embedding=app.state.skip_embedding,
            )
        except AdmissionRejected as e:
            return _fail("E422", 422, f"准入未通过：{e}", trace_id)
        except DuplicateDocument as e:
            return _fail("E409", 409, f"重复文档：{e}", trace_id)
        except UnsupportedFormat as e:
            return _fail("E415", 415, str(e), trace_id)
        except ValueError as e:
            return _fail("E400", 400, str(e), trace_id)
        return JSONResponse(status_code=201, content={"status": "accepted", **result.as_dict()})

    @app.get("/api/v1/documents")
    def list_documents(
        request: Request,
        limit: int = 20,
        offset: int = 0,
        status: str = "active",
        industry: str | None = None,
        x_kee_role: str | None = Header(None),
        x_kee_actor: str | None = Header(None),
        x_kee_industries: str | None = Header(None),
        x_kee_projects: str | None = Header(None),
    ) -> dict[str, Any]:
        ctx, _ = context_of(x_kee_role, x_kee_actor, x_kee_industries, x_kee_projects)
        if not 1 <= limit <= 100:
            raise ApiError("E400", 400, "limit 取值范围 1~100", request.state.trace_id)
        out = pg.list_documents(
            max_security_level=ctx.effective_max_level,
            industries=[industry] if industry else (ctx.industries or None),
            projects=ctx.projects or None,
            status=status or None,
            limit=limit,
            offset=offset,
        )
        out["scope"] = {
            "role": ctx.role,
            "max_security_level": ctx.effective_max_level,
            "filter_expr": build_filter_expr(ctx),
        }
        return out

    @app.get("/api/v1/documents/{doc_id}")
    def get_document(doc_id: str, x_kee_role: str | None = Header(None), x_kee_actor: str | None = Header(None)) -> dict[str, Any]:
        ctx, _ = context_of(x_kee_role, x_kee_actor, None, None)
        doc = pg.get_document(doc_id)
        if not doc:
            raise ApiError("E404", 404, f"文档不存在：{doc_id}", "")
        if int(doc["security_level"]) > ctx.effective_max_level:
            # 看得见元数据但读不到内容：密级越权时不返回版本与分块，只给存在性
            raise ApiError("E403", 403, f"密级 {doc['security_level']} 超出当前角色可见范围", "")
        return {"document": doc, "versions": pg.versions_of(doc_id)}

    @app.post("/api/v1/documents/{doc_id}/review")
    def review(doc_id: str, body: ReviewRequest, request: Request, x_kee_role: str | None = Header(None), x_kee_actor: str | None = Header(None)) -> dict[str, Any]:
        ctx, actor = context_of(x_kee_role, x_kee_actor, None, None)
        if ctx.role not in REVIEW_ROLES:
            raise ApiError("E403", 403, "复核需 knowledge_owner 或 system_admin 角色", request.state.trace_id)
        with pg.connect() as conn:
            last = conn.execute(
                "SELECT actor FROM audit_log WHERE target=%s AND action='document_ingested' ORDER BY id DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
        if last and last["actor"] == actor:
            raise ApiError("E403", 403, "职责分离：上传者不能复核自己的文档", request.state.trace_id)
        return service.review(doc_id, body.decision, actor=actor, note=body.note, trace_id=request.state.trace_id)

    @app.post("/api/v1/documents/{doc_id}/rollback")
    def rollback(doc_id: str, body: RollbackRequest, request: Request, x_kee_role: str | None = Header(None), x_kee_actor: str | None = Header(None)) -> dict[str, Any]:
        ctx, actor = context_of(x_kee_role, x_kee_actor, None, None)
        if ctx.role not in REVIEW_ROLES:
            raise ApiError("E403", 403, "回滚需 knowledge_owner 或 system_admin 角色", request.state.trace_id)
        return service.rollback(doc_id, body.to_version_id, actor=actor, trace_id=request.state.trace_id)

    @app.post("/api/v1/reconcile/run")
    def reconcile(request: Request, repair: bool = True, x_kee_role: str | None = Header(None), x_kee_actor: str | None = Header(None)) -> dict[str, Any]:
        ctx, actor = context_of(x_kee_role, x_kee_actor, None, None)
        if ctx.role not in REVIEW_ROLES:
            raise ApiError("E403", 403, "对账作业需管理角色", request.state.trace_id)
        rec = Reconciler(pg, kb)
        d = rec.diff(f"api-{request.state.trace_id}")
        out: dict[str, Any] = {"diff": {k: d[k] for k in ("authoritative_children", "indexed", "missing_in_vector", "extra_in_vector")}}
        if repair and (d["missing_in_vector"] or d["extra_in_vector"]):
            out["repair"] = rec.repair(d["run_id"], OutboxDispatcher(pg, kb, vectors))
            out["recheck"] = {k: (rec.diff(f"{d['run_id']}-recheck"))[k] for k in ("missing_in_vector", "extra_in_vector")}
        pg.append_audit(actor, "reconciliation_run", "kb_chunks", request.state.trace_id, out)
        return out

    @app.post("/api/v1/retrieval/search")
    def search(body: SearchRequest, request: Request, x_kee_role: str | None = Header(None), x_kee_actor: str | None = Header(None), x_kee_industries: str | None = Header(None), x_kee_projects: str | None = Header(None)) -> dict[str, Any]:
        from kee.retrieval.pipeline import HybridRetriever

        ctx, _ = context_of(x_kee_role, x_kee_actor, x_kee_industries, x_kee_projects)
        if app.state.skip_embedding:
            raise ApiError("E902", 503, "本实例以 skip_embedding 启动，稠密检索不可用", request.state.trace_id)
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            from kee.models.registry import ResidencyScheduler

            scheduler = ResidencyScheduler(policy="co_resident")
            app.state.scheduler = scheduler
        retriever = HybridRetriever(kb, scheduler=scheduler, cache=None, mode="sparse", use_rerank=False)
        res = retriever.retrieve(body.query, ctx, top_k=body.top_k)
        return {
            "query": body.query,
            "mode": res.mode,
            "expr": res.expr,
            "degraded": res.degraded,
            "timings_ms": res.timings_ms,
            "hits": [c.as_dict() for c in res.chunks],
        }

    def _answer_engine(rewrite: bool):
        """按 rewrite 开关复用两个引擎实例：改写器带 Redis 缓存，重建一次要重连。"""
        key = "answer_engine_rw" if rewrite else "answer_engine"
        eng = getattr(app.state, key, None)
        if eng is not None:
            return eng
        from kee.llm.ollama import OllamaClient
        from kee.qa.answer import AnswerEngine
        from kee.retrieval.cache import SearchCache
        from kee.retrieval.pipeline import HybridRetriever
        from kee.retrieval.rewrite import QueryRewriter

        sched = getattr(app.state, "scheduler", None)
        if sched is None:
            from kee.models.registry import ResidencyScheduler

            sched = ResidencyScheduler(policy="co_resident")
            app.state.scheduler = sched
        cal = _load_calibration(settings)
        cache = SearchCache()
        retriever = HybridRetriever(
            kb,
            scheduler=sched,
            cache=cache,
            mode="hybrid",
            use_rerank=True,
            rewriter=QueryRewriter(cache=cache) if rewrite else None,
            rewrite_mode="sparse" if rewrite else "off",
        )
        eng = AnswerEngine(retriever=retriever, llm=OllamaClient())
        eng.calibration_temperature = float(cal.get("temperature", 1.0)) or 1.0
        setattr(app.state, key, eng)
        return eng

    def _ensure_schema_once() -> None:
        """按需建表，且只在首次问答时做一次。

        为什么不在 create_app 里直接建：那样 PG 不可用时应用根本起不来，
        /healthz 也就没法回答「依赖挂了」——启动期硬连依赖会把可观测性一起带走。
        失败不抛，后面真正用到表的语句会自然出 E901，那才是准确的信号。
        """
        if getattr(app.state, "schema_ready", False):
            return
        try:
            pg.init_schema()
            app.state.schema_ready = True
        except psycopg.Error:
            pass

    def _history(session_id: str, rewrite: bool) -> list[dict[str, Any]]:
        """取本会话已有的轮次作为多轮指代消解的上文。

        只在 rewrite=true 时读库：图里 history 唯一的消费者就是改写节点，
        改写关掉时它传进去也不会发生任何事，不该为一次不会用到的读取花一个连接。
        取 5 条而不是图上实际用的 3 条——窗口大小归图管，端点多给一点余量，
        以后图把窗口调大不用回来改这里。
        """
        if not rewrite:
            return []
        try:
            return pg.turns_of(session_id, limit=5)
        except psycopg.Error:
            # 上文缺失只等于退化成单轮，不值得为此把整次问答打挂。
            return []

    @app.post("/api/v1/answers")
    def answers(
        body: AnswerRequest,
        request: Request,
        x_kee_role: str | None = Header(None),
        x_kee_actor: str | None = Header(None),
        x_kee_industries: str | None = Header(None),
        x_kee_projects: str | None = Header(None),
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        ctx, actor = context_of(x_kee_role, x_kee_actor, x_kee_industries, x_kee_projects)
        if app.state.skip_embedding:
            raise ApiError("E902", 503, "本实例以 skip_embedding 启动，答案链路需稠密检索", request.state.trace_id)
        _ensure_schema_once()
        trace = request.state.trace_id

        scope = f"answers:{ctx.role}:{body.session_id or '-'}"
        if idempotency_key:
            prev = pg.reserve_idempotency(idempotency_key, scope)
            if prev:
                return {**prev, "replayed": True, "idempotency_key": idempotency_key}

        session_id = body.session_id or f"s-{uuid.uuid4().hex[:12]}"
        fresh = pg.open_session(session_id, actor, {"role": ctx.role, "industries": ctx.industries,
                                                   "projects": ctx.projects})
        res = _answer_engine(body.rewrite).answer(
            body.question, ctx, top_k=body.top_k, history=_history(session_id, body.rewrite)
        )
        turn_no = pg.append_turn(
            session_id,
            {
                "question": body.question,
                "status": res.status,
                "conclusion": res.conclusion,
                "answer": res.as_dict(),
                # 引擎记的键叫 retrieval_ms；这里原来查的是 retrieval_total_ms，
                # 恒为 None——会话历史里的耗时列一直是空的。
                "retrieval_ms": res.timings_ms.get("retrieval_ms"),
                "total_ms": res.timings_ms.get("e2e_ms"),
                "trace_id": trace,
            },
        )
        pg.append_audit(
            actor,
            "answer_generated",
            session_id,
            trace,
            {
                "turn_no": turn_no,
                "status": res.status,
                "needs_human": res.needs_human,
                "n_citations": len(res.citations),
                "confidence": res.confidence,
                "rewrite": body.rewrite,
                "new_session": fresh,
                "stream": False,
            },
        )
        out = {**res.as_dict(), "session_id": session_id, "turn_no": turn_no, "new_session": fresh}
        if idempotency_key:
            pg.store_idempotency(idempotency_key, out)
        return out

    @app.post("/api/v1/answers/stream")
    def answers_stream(
        body: AnswerRequest,
        request: Request,
        x_kee_role: str | None = Header(None),
        x_kee_actor: str | None = Header(None),
        x_kee_industries: str | None = Header(None),
        x_kee_projects: str | None = Header(None),
    ):
        """阶段级 SSE：`retrieval`（命中概况）→ `answer`（最终答案）。

        刻意**不是** token 级流式。答案链走的是 JSON 模式（结构化输出首次成功率是 ⑤ 门禁项），
        而未闭合的 JSON 片段前端无法渲染，所以「首 token < 1.2s」与「只输出 JSON」这两条要求
        本身冲突。要做 token 级流式必须改成两趟（先流式自由文本、再结构化），
        那会牺牲结构化门禁，需要单独裁决。这里先把可交付的部分做实：
        M3-05 图化之后，检索节点一跑完就能拿到它的状态增量，前端因此可以在生成之前
        先拿到「用了哪条查询、命中几条、是否降级」，不用等整条链路结束。
        `retrieval` 事件只带 chunk id 与概况、不带正文：正文要展示仍应走 answer 事件的引用，
        免得同一个文本从两个事件里各出一遍。
        """
        import json as _json

        from kee.qa.graph import initial_state, result_from_state

        ctx, actor = context_of(x_kee_role, x_kee_actor, x_kee_industries, x_kee_projects)
        if app.state.skip_embedding:
            raise ApiError("E902", 503, "本实例以 skip_embedding 启动，答案链路需稠密检索", request.state.trace_id)
        _ensure_schema_once()
        trace = request.state.trace_id
        session_id = body.session_id or f"s-{uuid.uuid4().hex[:12]}"
        # role_ctx 与非流式端点存同一套形状：会话身份是检索层 RBAC 的输入，
        # 少存 industries/projects 会让「先看会话记录再复现权限」这条路断掉。
        pg.open_session(
            session_id, actor, {"role": ctx.role, "industries": ctx.industries, "projects": ctx.projects}
        )
        engine = _answer_engine(body.rewrite)

        def events():
            state = initial_state(
                body.question, ctx, top_k=body.top_k, history=_history(session_id, body.rewrite)
            )
            try:
                for update in engine.graph.stream(state, stream_mode="updates"):
                    for node, delta in update.items():
                        state.update(delta or {})
                        if node == "retrieve":
                            yield (
                                "event: retrieval\ndata: "
                                f"{_json.dumps(state['retrieval'], ensure_ascii=False, default=str)}\n\n"
                            )
                payload = result_from_state(state)
                # 流式路径也要落 session_turn：多轮的上文是从这张表读的，
                # 不落轮次等于「用流式的客户端永远没有上文」——而聊天界面恰恰只会用流式。
                turn_no = pg.append_turn(
                    session_id,
                    {
                        "question": body.question,
                        "status": payload.status,
                        "conclusion": payload.conclusion,
                        "answer": payload.as_dict(),
                        "retrieval_ms": payload.timings_ms.get("retrieval_ms"),
                        "total_ms": payload.timings_ms.get("e2e_ms"),
                        "trace_id": trace,
                    },
                )
                # 流式与非流式必须留同一条审计：答案对外给出去了就得有记录，
                # 不然「走哪条链路」变成了「要不要被审计」的选择。
                pg.append_audit(
                    actor,
                    "answer_generated",
                    session_id,
                    trace,
                    {
                        "turn_no": turn_no,
                        "status": payload.status,
                        "needs_human": payload.needs_human,
                        "n_citations": len(payload.citations),
                        "confidence": payload.confidence,
                        "rewrite": body.rewrite,
                        "stream": True,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - 流里必须以事件收尾，不能让前端猜
                yield f"event: error\ndata: {_json.dumps({'code': 'E500', 'message': str(exc)[:200]})}\n\n"
                return
            out = {**payload.as_dict(), "session_id": session_id, "turn_no": turn_no}
            yield f"event: answer\ndata: {_json.dumps(out, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/v1/sessions/{session_id}")
    def session_history(
        session_id: str,
        request: Request,
        x_kee_role: str | None = Header(None),
        x_kee_actor: str | None = Header(None),
        limit: int = 20,
    ) -> dict[str, Any]:
        ctx, actor = context_of(x_kee_role, x_kee_actor, None, None)
        sess = pg.session_ctx(session_id)
        if sess is None:
            raise ApiError("E404", 404, f"会话不存在：{session_id}", request.state.trace_id)
        turns = pg.turns_of(session_id, limit=max(1, min(limit, 100)))
        pg.append_audit(actor, "session_read", session_id, request.state.trace_id, {"n_turns": len(turns)})
        return {
            "session_id": session_id,
            "user_id": sess["user_id"],
            "turns": [{k: str(v) if k == "created_at" else v for k, v in t.items()} for t in turns],
        }

    return app
