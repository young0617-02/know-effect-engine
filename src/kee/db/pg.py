"""PostgreSQL 18 权威源：文档/版本/分块/会话/审计/Outbox/对账差异表。

设计口径（技术方案 7.1 + 2.1「单一权威源」「审计不可篡改」）：
  - PostgreSQL 是元数据、版本与审计的唯一权威源；Milvus 索引可由此重建。
  - audit_log 为追加表，UPDATE/DELETE 由触发器一律拒绝。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from kee.config import get_settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS document (
    id              text PRIMARY KEY,
    title           text NOT NULL,
    industry        text NOT NULL,
    customer        text NOT NULL,
    project         text NOT NULL,
    doc_type        text NOT NULL,
    source_level    text NOT NULL,
    security_level  smallint NOT NULL,
    status          text NOT NULL DEFAULT 'active',
    checksum        text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_version (
    id            text PRIMARY KEY,
    doc_id        text NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    version       integer NOT NULL,
    chunk_scheme  text NOT NULL,
    checksum      text NOT NULL,
    status        text NOT NULL DEFAULT 'effective',
    effective_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, version)
);

CREATE TABLE IF NOT EXISTS chunk (
    id          text PRIMARY KEY,
    version_id  text NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
    doc_id      text NOT NULL,
    parent_id   text,
    section_no  text NOT NULL,
    page        integer NOT NULL,
    order_no    integer NOT NULL,
    content     text NOT NULL,
    char_count  integer NOT NULL,
    is_parent   boolean NOT NULL DEFAULT false,
    UNIQUE (version_id, order_no)
);
CREATE INDEX IF NOT EXISTS chunk_doc_idx ON chunk(doc_id);
CREATE INDEX IF NOT EXISTS chunk_version_idx ON chunk(version_id);

CREATE TABLE IF NOT EXISTS session (
    id         text PRIMARY KEY,
    user_id    text NOT NULL,
    role_ctx   jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        bigserial PRIMARY KEY,
    actor     text NOT NULL,
    action    text NOT NULL,
    target    text NOT NULL,
    trace_id  text,
    payload   jsonb,
    ts        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS outbox_event (
    id             bigserial PRIMARY KEY,
    aggregate_type text NOT NULL,
    aggregate_id   text NOT NULL,
    event_type     text NOT NULL,
    payload        jsonb NOT NULL,
    status         text NOT NULL DEFAULT 'pending',
    retry_count    integer NOT NULL DEFAULT 0,
    last_error     text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    dispatched_at  timestamptz
);
CREATE INDEX IF NOT EXISTS outbox_status_idx ON outbox_event(status);

CREATE TABLE IF NOT EXISTS reconciliation_diff (
    id          bigserial PRIMARY KEY,
    run_id      text NOT NULL,
    chunk_id    text NOT NULL,
    kind        text NOT NULL,
    detected_at timestamptz NOT NULL DEFAULT now(),
    fixed_at    timestamptz
);
CREATE INDEX IF NOT EXISTS recon_run_idx ON reconciliation_diff(run_id);

CREATE TABLE IF NOT EXISTS idempotency_key (
    key        text PRIMARY KEY,
    scope      text NOT NULL,
    response   jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- 多轮追问的轮次历史。刻意放在权威源而不是 Redis：
-- 「其余存储皆可从权威源重建」这条口径下，会话若只能从缓存恢复就等于会丢。
-- LangGraph 的 Checkpointer 接的也是这张表（见 M3-05），不另起一套状态。
CREATE TABLE IF NOT EXISTS session_turn (
    id           bigserial PRIMARY KEY,
    session_id   text NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    turn_no      integer NOT NULL,
    question     text NOT NULL,
    status       text NOT NULL,
    conclusion   text NOT NULL DEFAULT '',
    answer       jsonb,
    retrieval_ms double precision,
    total_ms     double precision,
    trace_id     text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn_no)
);
CREATE INDEX IF NOT EXISTS session_turn_idx ON session_turn(session_id, turn_no);
"""

IMMUTABLE_SQL = """
CREATE OR REPLACE FUNCTION kee_forbid_audit_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log 为追加表，禁止 % 操作', TG_OP
        USING ERRCODE = 'insufficient_privilege';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_immutable ON audit_log;
CREATE TRIGGER audit_log_immutable
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION kee_forbid_audit_mutation();
"""


class AuthoritativeStore:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or get_settings().postgres_dsn

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA_SQL)
            conn.execute(IMMUTABLE_SQL)

    def truncate_all(self, include_audit: bool = True) -> None:
        tables = [
            "reconciliation_diff",
            "outbox_event",
            "chunk",
            "document_version",
            "document",
            "session",
        ]
        if include_audit:
            tables.insert(0, "audit_log")
        with self.connect() as conn:
            for t in tables:
                conn.execute(f"TRUNCATE TABLE {t} CASCADE")

    # ---- 写 ----
    def upsert_document(self, doc: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO document (id, title, industry, customer, project, doc_type,
                                      source_level, security_level, status, checksum)
                VALUES (%(id)s, %(title)s, %(industry)s, %(customer)s, %(project)s, %(doc_type)s,
                        %(source_level)s, %(security_level)s, %(status)s, %(checksum)s)
                ON CONFLICT (id) DO UPDATE SET
                    title=EXCLUDED.title, security_level=EXCLUDED.security_level,
                    status=EXCLUDED.status, checksum=EXCLUDED.checksum, updated_at=now()
                """,
                doc,
            )

    def insert_version(self, ver: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO document_version (id, doc_id, version, chunk_scheme, checksum, status, effective_at)
                VALUES (%(id)s, %(doc_id)s, %(version)s, %(chunk_scheme)s, %(checksum)s, %(status)s, %(effective_at)s)
                ON CONFLICT (id) DO NOTHING
                """,
                ver,
            )

    def insert_chunks(self, chunks: Iterable[dict[str, Any]]) -> int:
        rows = list(chunks)
        if not rows:
            return 0
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunk (id, version_id, doc_id, parent_id, section_no, page,
                                       order_no, content, char_count, is_parent)
                    VALUES (%(id)s, %(version_id)s, %(doc_id)s, %(parent_id)s, %(section_no)s,
                            %(page)s, %(order_no)s, %(content)s, %(char_count)s, %(is_parent)s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    rows,
                )
        return len(rows)

    def append_audit(self, actor: str, action: str, target: str, trace_id: str, payload: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (actor, action, target, trace_id, payload) VALUES (%s,%s,%s,%s,%s::jsonb)",
                (actor, action, target, trace_id, json.dumps(payload, ensure_ascii=False)),
            )

    def list_audit(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        target: str | None = None,
        trace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """审计读路径：只有 SELECT，没有任何改写方法——`audit_log` 的 UPDATE/DELETE 由触发器一律拒绝，
        所以「追加禁改」这件事在这层是靠不存在的写路径保证的，不是靠约定。

        按 `id DESC` 而不是 `ts DESC`：ts 有并列（同一批操作落在同一时间戳），
        并列时顺序不稳定，翻页会出现重复行或漏行。id 是 bigserial，单调。
        """
        where, params = [], {"limit": limit, "offset": offset}
        for col, val in (("actor", actor), ("action", action), ("target", target), ("trace_id", trace_id)):
            if val:
                where.append(f"{col} = %({col})s")
                params[col] = val
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self.connect() as conn:
            total = conn.execute(f"SELECT count(*) AS c FROM audit_log {clause}",
                                 {k: v for k, v in params.items() if k not in ("limit", "offset")}).fetchone()["c"]
            rows = conn.execute(
                f"SELECT id, actor, action, target, trace_id, payload, ts FROM audit_log {clause} "
                "ORDER BY id DESC LIMIT %(limit)s OFFSET %(offset)s",
                params,
            ).fetchall()
        return {"total": int(total), "limit": limit, "offset": offset, "items": rows}

    def enqueue_outbox(self, events: Iterable[dict[str, Any]]) -> int:
        rows = list(events)
        if not rows:
            return 0
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO outbox_event (aggregate_type, aggregate_id, event_type, payload)
                    VALUES (%(aggregate_type)s, %(aggregate_id)s, %(event_type)s, %(payload)s::jsonb)
                    """,
                    [{**r, "payload": json.dumps(r["payload"], ensure_ascii=False)} for r in rows],
                )
        return len(rows)

    # ---- 幂等（方案 7.3）----
    def reserve_idempotency(self, key: str, scope: str) -> dict[str, Any] | None:
        """占位成功返回 None；已存在则返回上次响应（重放）。"""
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO idempotency_key (key, scope) VALUES (%s,%s) "
                "ON CONFLICT (key) DO NOTHING RETURNING key",
                (key, scope),
            )
            if cur.fetchone():
                return None
            row = conn.execute("SELECT response FROM idempotency_key WHERE key=%s", (key,)).fetchone()
            return row["response"] if row else {}

    def store_idempotency(self, key: str, response: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE idempotency_key SET response=%s::jsonb WHERE key=%s",
                (json.dumps(response, ensure_ascii=False, default=str), key),
            )

    # ---- 会话与轮次（Must#7 多轮追问的地基）----
    def open_session(self, session_id: str, user_id: str, role_ctx: dict[str, Any]) -> bool:
        """返回 True 表示新建。已存在时不覆盖 role_ctx——会话期间的身份不该被改。"""
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO session (id, user_id, role_ctx) VALUES (%s,%s,%s::jsonb) "
                "ON CONFLICT (id) DO NOTHING RETURNING id",
                (session_id, user_id, json.dumps(role_ctx, ensure_ascii=False)),
            )
            return cur.fetchone() is not None

    def session_ctx(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM session WHERE id=%s", (session_id,)).fetchone()

    def append_turn(self, session_id: str, turn: dict[str, Any]) -> int:
        """轮次号在会话内单调递增。用 RETURNING 而不是先查后插，避免并发下撞 UNIQUE。"""
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO session_turn (session_id, turn_no, question, status, conclusion, answer,
                                          retrieval_ms, total_ms, trace_id)
                VALUES (%s,
                        coalesce((SELECT max(turn_no) FROM session_turn WHERE session_id=%s),0)+1,
                        %s,%s,%s,%s::jsonb,%s,%s,%s)
                RETURNING turn_no
                """,
                (
                    session_id,
                    session_id,
                    turn["question"],
                    turn.get("status", ""),
                    turn.get("conclusion", ""),
                    json.dumps(turn.get("answer"), ensure_ascii=False, default=str)
                    if turn.get("answer") is not None
                    else None,
                    turn.get("retrieval_ms"),
                    turn.get("total_ms"),
                    turn.get("trace_id"),
                ),
            ).fetchone()
            conn.execute("UPDATE session SET updated_at=now() WHERE id=%s", (session_id,))
        return int(row["turn_no"])

    def turns_of(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT turn_no, question, status, conclusion, retrieval_ms, total_ms, trace_id, created_at "
                "FROM session_turn WHERE session_id=%s ORDER BY turn_no DESC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        return list(reversed(rows))

    # ---- 版本管理 ----
    def next_version(self, doc_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT coalesce(max(version),0)+1 AS v FROM document_version WHERE doc_id=%s",
                (doc_id,),
            ).fetchone()
        return int(row["v"])

    def bump_version(
        self, doc_id: str, chunk_scheme: str, checksum: str, effective_at: datetime | None = None
    ) -> str:
        """升版并把其它版本置 superseded：任一时刻每份文档只有一个 effective 版本。"""
        with self.connect() as conn:
            v = self.next_version(doc_id)
            conn.execute("UPDATE document_version SET status='superseded' WHERE doc_id=%s", (doc_id,))
            version_id = f"{doc_id}@v{v}"
            conn.execute(
                "INSERT INTO document_version (id, doc_id, version, chunk_scheme, checksum, status, effective_at) "
                "VALUES (%s,%s,%s,%s,%s,'effective',%s) ON CONFLICT (id) DO UPDATE SET status='effective'",
                (
                    version_id,
                    doc_id,
                    v,
                    chunk_scheme,
                    checksum,
                    effective_at or datetime.now(timezone.utc),
                ),
            )
            conn.execute("UPDATE document SET checksum=%s, updated_at=now() WHERE id=%s", (checksum, doc_id))
        return version_id

    def set_version_status(self, version_id: str, status: str) -> None:
        if status not in ("effective", "superseded", "draft"):
            raise ValueError(f"非法版本状态：{status}")
        with self.connect() as conn:
            row = conn.execute("SELECT doc_id FROM document_version WHERE id=%s", (version_id,)).fetchone()
            if not row:
                raise KeyError(f"版本不存在：{version_id}")
            if status == "effective":
                conn.execute(
                    "UPDATE document_version SET status='superseded' WHERE doc_id=%s", (row["doc_id"],)
                )
            conn.execute("UPDATE document_version SET status=%s WHERE id=%s", (status, version_id))

    def versions_of(self, doc_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT id, version, chunk_scheme, checksum, status, effective_at "
                "FROM document_version WHERE doc_id=%s ORDER BY version DESC",
                (doc_id,),
            ).fetchall()

    def set_document_status(self, doc_id: str, status: str) -> None:
        if status not in ("active", "offline", "rejected"):
            raise ValueError(f"非法文档状态：{status}")
        with self.connect() as conn:
            conn.execute(
                "UPDATE document SET status=%s, updated_at=now() WHERE id=%s", (status, doc_id)
            )

    # ---- 读 ----
    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM document WHERE id=%s", (doc_id,)).fetchone()

    def list_documents(
        self,
        *,
        max_security_level: int = 3,
        industries: list[str] | None = None,
        projects: list[str] | None = None,
        status: str | None = "active",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """知识管理查询：权限过滤在这里就做，返回的每条都是调用者看得见的。"""
        where = ["d.security_level <= %(max_level)s"]
        params: dict[str, Any] = {"max_level": max_security_level, "limit": limit, "offset": offset}
        if status:
            where.append("d.status = %(status)s")
            params["status"] = status
        if industries:
            where.append("d.industry = ANY(%(industries)s)")
            params["industries"] = industries
        if projects:
            where.append("d.project = ANY(%(projects)s)")
            params["projects"] = projects
        clause = " AND ".join(where)
        with self.connect() as conn:
            total = conn.execute(
                f"SELECT count(*) AS c FROM document d WHERE {clause}",
                {k: v for k, v in params.items() if k not in ("limit", "offset")},
            ).fetchone()["c"]
            rows = conn.execute(
                f"""
                SELECT d.id, d.title, d.industry, d.customer, d.project, d.doc_type,
                       d.source_level, d.security_level, d.status, d.created_at, d.updated_at,
                       v.id AS effective_version, v.version AS version_no,
                       (SELECT count(*) FROM chunk c WHERE c.version_id = v.id AND c.is_parent = false) AS child_chunks
                FROM document d
                LEFT JOIN document_version v ON v.doc_id = d.id AND v.status = 'effective'
                WHERE {clause}
                ORDER BY d.updated_at DESC LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            ).fetchall()
        return {"total": int(total), "limit": limit, "offset": offset, "items": rows}

    def chunks_of(self, version_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT id, doc_id, parent_id, section_no, page, order_no, content, is_parent "
                "FROM chunk WHERE version_id = %s ORDER BY order_no",
                (version_id,),
            ).fetchall()

    def chunks_visible(
        self,
        ids: list[str],
        *,
        max_security_level: int = 3,
        industries: list[str] | None = None,
        projects: list[str] | None = None,
    ) -> dict[str, Any]:
        """按 id 批量读分块正文，**过滤口径与检索层一致**（密级 / 行业 / 项目）。

        这是前端展示「引用片段」的唯一读取通道，所以它必须和 `build_filter_expr` 收敛同样三个维度：
        按 id 直取而不判权限，等于给检索层过滤开了一个后门——chunk id 是会出现在响应里给客户端的。

        只返回**当前生效版本**的内容：文档回滚或驳回后，旧版本的正文不该再被引用出来。
        `hidden` 里是「越权或不存在或已非生效版本」的 id，三者在这一层刻意不区分——
        告诉调用者哪个 id 属于哪一类，本身就是信息泄露。
        """
        if not ids:
            return {"items": [], "hidden": [], "requested": 0}
        where = ["c.id = ANY(%(ids)s)", "d.security_level <= %(max_level)s"]
        params: dict[str, Any] = {"ids": ids, "max_level": max_security_level}
        if industries:
            where.append("d.industry = ANY(%(industries)s)")
            params["industries"] = industries
        if projects:
            where.append("d.project = ANY(%(projects)s)")
            params["projects"] = projects
        sql = (
            "SELECT c.id AS chunk_id, c.content, c.doc_id, c.section_no, c.page, c.parent_id, "
            "       c.is_parent, d.industry, d.project, d.security_level, v.id AS version_id "
            "FROM chunk c "
            "JOIN document d ON d.id = c.doc_id "
            "JOIN document_version v ON v.id = c.version_id AND v.status = 'effective' "
            f"WHERE {' AND '.join(where)}"
        )
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        seen = {r["chunk_id"] for r in rows}
        return {"items": rows, "hidden": [i for i in ids if i not in seen], "requested": len(ids)}

    def chunk_ids(self, version_ids: list[str] | None = None) -> set[str]:
        with self.connect() as conn:
            if version_ids:
                rows = conn.execute(
                    "SELECT id FROM chunk WHERE is_parent = false AND version_id = ANY(%s)",
                    (version_ids,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT id FROM chunk WHERE is_parent = false").fetchall()
        return {r["id"] for r in rows}

    def effective_versions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT v.id, v.doc_id, v.version, v.chunk_scheme, v.checksum, d.security_level "
                "FROM document_version v JOIN document d ON d.id = v.doc_id "
                "WHERE v.status = 'effective' ORDER BY v.doc_id"
            ).fetchall()

    def outbox_pending(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM outbox_event WHERE status IN ('pending','failed') "
                "ORDER BY id LIMIT %s",
                (limit,),
            ).fetchall()

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            out = {}
            for t in ("document", "document_version", "chunk", "outbox_event", "audit_log", "reconciliation_diff"):
                out[t] = conn.execute(f"SELECT count(*) AS c FROM {t}").fetchone()["c"]
            return out
