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
    block_id    text,
    block_type  text,
    source      text,
    UNIQUE (version_id, order_no)
);
CREATE INDEX IF NOT EXISTS chunk_doc_idx ON chunk(doc_id);
CREATE INDEX IF NOT EXISTS chunk_version_idx ON chunk(version_id);

-- 解析层结构真相：与 chunk 解耦，Phase 1 只入库、不参与分块。
-- bbox 坐标由 MinerU 的渲染页给出（coord_space=rendered_px）；pypdf 没有坐标时留空。
CREATE TABLE IF NOT EXISTS content_block (
    id          text PRIMARY KEY,
    version_id  text NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
    doc_id      text NOT NULL,
    block_order integer NOT NULL,
    page        integer,
    block_type  text NOT NULL,
    source      text NOT NULL,
    text        text NOT NULL,
    markdown    text NOT NULL DEFAULT '',
    bbox        jsonb,
    coord_space text,
    table_json  jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (version_id, block_order)
);
CREATE INDEX IF NOT EXISTS content_block_doc_idx ON content_block(doc_id);
CREATE INDEX IF NOT EXISTS content_block_version_idx ON content_block(version_id);

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

# 复核闸（用户 2026-10-12 选定处置 C）：复核状态必须**独立于** effective/superseded。
# 此前版本一上传就 effective，「复核」只是事后撤销，没有「未复核」这个可判断的状态可言。
REVIEW_GATE_SQL = """DO $mig$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name = 'document_version' AND column_name = 'reviewed_by') THEN
    ALTER TABLE document_version ADD COLUMN reviewed_by text;
    ALTER TABLE document_version ADD COLUMN reviewed_at timestamptz;
    -- 回填是必须的，而且必须写明是迁移给的、不是人复核的：
    -- 不回填就等于让现存 24 篇语料在一夜之间全部「无依据可答」——那是断服务，不是加闸。
    UPDATE document_version SET reviewed_by = 'migration:pre-review-gate', reviewed_at = now()
      WHERE reviewed_by IS NULL;
  END IF;
END
$mig$;
"""

# Phase 2：chunk 可选挂到 ContentBlock；legacy 行这三列为 NULL，不改变既有读数。
CHUNK_BLOCK_SQL = """
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS block_id text;
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS block_type text;
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS source text;
CREATE INDEX IF NOT EXISTS chunk_block_idx ON chunk(block_id);
"""

# 账号与会话表。DDL 放在 DB 层而不是 api 层：`kee.api` 的包初始化会拉 app.py，
# 从这儿 import 常量会绕成循环导入（真踩过一次）。
AUTH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_user (
    username     text PRIMARY KEY,
    display_name text NOT NULL,
    pass_hash    text NOT NULL,
    role         text NOT NULL,
    industries   text NOT NULL DEFAULT '',
    projects     text NOT NULL DEFAULT '',
    is_active    boolean NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_login_at timestamptz,
    failed_count integer NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_session (
    token_hash text PRIMARY KEY,
    username   text NOT NULL REFERENCES app_user(username) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS user_session_user_idx ON user_session(username);
"""

IMMUTABLE_SQL = """CREATE OR REPLACE FUNCTION kee_forbid_audit_mutation() RETURNS trigger AS $$
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
            conn.execute(AUTH_SCHEMA_SQL)
            conn.execute(REVIEW_GATE_SQL)   # 一次性迁移：列已存在时整块 DO 什么都不做
            conn.execute(CHUNK_BLOCK_SQL)
            self._ensure_audit_guard(conn)

    def _ensure_audit_guard(self, conn) -> None:
        """装上「审计表禁改」触发器；装不动时必须确认它已经在那儿，不能默默放行。

        `scripts/harden_audit_store.py` 把 `audit_log` 的属主交给 NOLOGIN 超级用户角色后，
        应用角色不再是属主，`CREATE TRIGGER` 对它必然报权限不足——**这正是收紧生效的表现**。
        所以这里分三种情况：装上了（未收紧的新库）/ 装不上但触发器在（已收紧）/
        装不上且触发器也没有（配置错了，拒绝启动）。
        最后那格不能省：少了它，「应用角色碰不到触发器」就会表现成「每次启动都静默地没装上」，
        而那正是这条防线要防的那类失效。
        """
        try:
            conn.execute(IMMUTABLE_SQL)
        except psycopg.errors.InsufficientPrivilege:
            if not conn.execute("SELECT 1 FROM pg_trigger WHERE tgname = 'audit_log_immutable' "
                                "AND NOT tgisinternal").fetchone():
                raise RuntimeError(
                    "audit_log 上没有禁改触发器，而当前角色也没有权限装它。"
                    "要么是先收紧了权限却没建触发器（跑 scripts/harden_audit_store.py），"
                    "要么是新库但角色被误配（用超级用户跑一次 init_schema）。"
                    "不许带着「审计表可被任意改写」起跑。"
                ) from None

    def truncate_all(self, include_audit: bool = True) -> None:
        tables = [
            "reconciliation_diff",
            "outbox_event",
            "content_block",
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
        for row in rows:
            row.setdefault("block_id", None)
            row.setdefault("block_type", None)
            row.setdefault("source", None)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunk (id, version_id, doc_id, parent_id, section_no, page,
                                       order_no, content, char_count, is_parent,
                                       block_id, block_type, source)
                    VALUES (%(id)s, %(version_id)s, %(doc_id)s, %(parent_id)s, %(section_no)s,
                            %(page)s, %(order_no)s, %(content)s, %(char_count)s, %(is_parent)s,
                            %(block_id)s, %(block_type)s, %(source)s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    rows,
                )
        return len(rows)

    def insert_blocks(self, version_id: str, doc_id: str, blocks: Iterable[Any]) -> int:
        """把解析层 ContentBlock 写入权威源；Phase 1 只入库，不改变 chunk 生成。"""
        rows: list[dict[str, Any]] = []
        for b in blocks:
            table = getattr(b, "table", None)
            bbox = getattr(b, "bbox", None)
            rows.append({
                "id": f"{version_id}#{b.block_id}",
                "version_id": version_id,
                "doc_id": doc_id,
                "block_order": int(b.order),
                "page": b.page,
                "block_type": b.type,
                "source": b.source,
                "text": b.text,
                "markdown": getattr(b, "markdown", "") or "",
                "bbox": json.dumps(list(bbox), ensure_ascii=False) if bbox else None,
                "coord_space": getattr(b, "coord_space", None),
                "table_json": json.dumps(table.as_dict(), ensure_ascii=False) if table else None,
            })
        if not rows:
            return 0
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO content_block (id, version_id, doc_id, block_order, page,
                                               block_type, source, text, markdown, bbox,
                                               coord_space, table_json)
                    VALUES (%(id)s, %(version_id)s, %(doc_id)s, %(block_order)s, %(page)s,
                            %(block_type)s, %(source)s, %(text)s, %(markdown)s,
                            %(bbox)s::jsonb, %(coord_space)s, %(table_json)s::jsonb)
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

    def list_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """某个 actor 的会话列表，按最近活动排序——左侧「会话历史」栏的数据。

        `user_id` 就是 `open_session` 当时写进去的 actor：**登录态下它是库里的账号名，
        脚本通道下它是请求头自称的值**。所以「只看自己的会话」这条的实际强度等于身份通道的强度，
        与 `/api/v1/audit` 那道角色门不是同一回事（那条判的是角色，这条判的是归属）。
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.user_id, s.created_at, s.updated_at,
                       (SELECT count(*) FROM session_turn t WHERE t.session_id = s.id)          AS n_turns,
                       (SELECT t.question FROM session_turn t WHERE t.session_id = s.id
                        ORDER BY t.turn_no LIMIT 1)                                             AS title,
                       (SELECT t.status FROM session_turn t WHERE t.session_id = s.id
                        ORDER BY t.turn_no DESC LIMIT 1)                                        AS last_status,
                       (SELECT max(t.created_at) FROM session_turn t WHERE t.session_id = s.id) AS last_at
                FROM session s
                WHERE s.user_id = %(user_id)s
                  AND EXISTS (SELECT 1 FROM session_turn t WHERE t.session_id = s.id)
                ORDER BY s.updated_at DESC
                LIMIT %(limit)s
                """,
                {"user_id": user_id, "limit": max(1, min(limit, 200))},
            ).fetchall()
        return list(rows)

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

    def turns_of(self, session_id: str, limit: int = 20, with_answer: bool = False) -> list[dict[str, Any]]:
        """默认**不**带 `answer` 整包：图里造上文只要问题/结论/状态，把 jsonb 一起捞出来
        等于每轮多读几 KB 却没人用。界面恢复历史时才要 `with_answer=True`——
        没有整包就只能恢复出问题与结论，要点、引用落点与规则分全丢，
        「翻回旧会话」会变成一份看着完整、其实被削过的记录。"""
        cols = "turn_no, question, status, conclusion, retrieval_ms, total_ms, trace_id, created_at"
        if with_answer:
            cols += ", answer"
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {cols} FROM session_turn WHERE session_id=%s ORDER BY turn_no DESC LIMIT %s",
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
                "SELECT id, version, chunk_scheme, checksum, status, effective_at, reviewed_by, reviewed_at "
                "FROM document_version WHERE doc_id=%s ORDER BY version DESC",
                (doc_id,),
            ).fetchall()

    def mark_reviewed(self, version_id: str, actor: str) -> None:
        """复核通过 = 给这一个版本盖上复核章。章与状态分开：回滚/下线都不该顺手把章擦掉。"""
        with self.connect() as conn:
            n = conn.execute(
                "UPDATE document_version SET reviewed_by=%s, reviewed_at=now() WHERE id=%s",
                (actor, version_id),
            ).rowcount
        if not n:
            raise KeyError(f"版本不存在：{version_id}")

    def unreviewed_versions(self) -> set[str]:
        """未复核版本 id 集合，答案层用它过滤证据。空集是常态（存量已回填），
        但取不到不代表没闸——闸在答案层按这个集合生效。"""
        with self.connect() as conn:
            return {r["id"] for r in conn.execute(
                "SELECT id FROM document_version WHERE reviewed_by IS NULL").fetchall()}

    def parents_content(self, parent_ids: list[str]) -> dict[str, str]:
        """按 parent_id 批量取父块正文，供答案层补上下文（需求 US-1.2「父块补上下文」）。

        权限不用在这里再判一次：`parent_ids` 全部来自**已通过检索 RBAC 与复核闸的子块**的
        `parent_id` 字段，父块与子块同属一个 `version_id`，所以两道闸的结论天然继承到这里。
        传空列表直接返回空字典，不打库。
        """
        if not parent_ids:
            return {}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, content FROM chunk WHERE id = ANY(%s) AND is_parent = true",
                (list(parent_ids),),
            ).fetchall()
        return {r["id"]: r["content"] for r in rows}

    def pending_reviews(
        self,
        *,
        max_security_level: int = 3,
        industries: list[str] | None = None,
        projects: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """待复核队列：已生效但 `reviewed_by IS NULL` 的版本，**按调用者可见范围过滤**。

        为什么任何读权角色都能看这个队列：复核闸拦的是「被当成答案对外作答」，不是「被看见」
        （处置 C 的原始裁决），所以「资料在、只是还没人复核」这件事本来就该让业务侧看得见、催得动。
        范围过滤沿用知识管理查询同一套参数（密级 / 行业 / 项目），**不另开一套判定**——
        两套判定迟早分叉，分叉的那天就是越权的那天。
        """
        where = ["v.reviewed_by IS NULL", "v.status = 'effective'", "d.status = 'active'",
                 "d.security_level <= %(max_level)s"]
        params: dict[str, Any] = {"max_level": max_security_level, "limit": limit, "offset": offset}
        if industries:
            where.append("d.industry = ANY(%(industries)s)")
            params["industries"] = industries
        if projects:
            where.append("d.project = ANY(%(projects)s)")
            params["projects"] = projects
        clause = " AND ".join(where)
        cnt_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
        with self.connect() as conn:
            total = conn.execute(
                f"SELECT count(*) AS c FROM document_version v JOIN document d ON d.id = v.doc_id "
                f"WHERE {clause}", cnt_params,
            ).fetchone()["c"]
            rows = conn.execute(
                f"""
                SELECT v.id AS version_id, v.version AS version_no, v.effective_at,
                       d.id AS doc_id, d.title, d.industry, d.customer, d.project, d.doc_type,
                       d.security_level,
                       (SELECT count(*) FROM chunk c
                         WHERE c.version_id = v.id AND c.is_parent = false) AS child_chunks
                FROM document_version v JOIN document d ON d.id = v.doc_id
                WHERE {clause}
                ORDER BY v.effective_at ASC LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            ).fetchall()
        return {"total": int(total), "limit": limit, "offset": offset, "items": rows}

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
            "       c.is_parent, c.block_id, COALESCE(c.block_type, 'legacy') AS block_type, "
            "       COALESCE(c.source, 'legacy') AS source, "
            "       d.industry, d.project, d.security_level, v.id AS version_id, "
            "       cb.table_json AS table "
            "FROM chunk c "
            "JOIN document d ON d.id = c.doc_id "
            "JOIN document_version v ON v.id = c.version_id AND v.status = 'effective' "
            "LEFT JOIN content_block cb ON cb.id = c.block_id "
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
