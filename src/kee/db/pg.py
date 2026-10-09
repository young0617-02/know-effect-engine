"""PostgreSQL 18 权威源：文档/版本/分块/会话/审计/Outbox/对账差异表。

设计口径（技术方案 7.1 + 2.1「单一权威源」「审计不可篡改」）：
  - PostgreSQL 是元数据、版本与审计的唯一权威源；Milvus 索引可由此重建。
  - audit_log 为追加表，UPDATE/DELETE 由触发器一律拒绝。
"""

from __future__ import annotations

import json
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

    # ---- 读 ----
    def chunks_of(self, version_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT id, doc_id, parent_id, section_no, page, order_no, content, is_parent "
                "FROM chunk WHERE version_id = %s ORDER BY order_no",
                (version_id,),
            ).fetchall()

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
