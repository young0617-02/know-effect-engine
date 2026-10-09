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
from fastapi.responses import JSONResponse
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

    return app
