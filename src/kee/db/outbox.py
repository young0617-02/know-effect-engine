"""Outbox 派发器：权威源确认后异步写向量库，失败进补偿队列（可重试）。

注入故障是被试能力（Gate 0 ④），不是生产行为：fail_chunk_ids / fail_times 只在验证脚本里给。
"""

from __future__ import annotations

import time
from typing import Any

from kee.db.pg import AuthoritativeStore
from kee.db.vector_store import VectorStore
from kee.retrieval.milvus_store import KbStore

MAX_RETRY = 5


class OutboxDispatcher:
    def __init__(
        self,
        pg: AuthoritativeStore,
        kb: KbStore,
        vectors: VectorStore,
        fail_chunk_ids: set[str] | None = None,
        fail_times: int = 0,
    ) -> None:
        self.pg = pg
        self.kb = kb
        self.vectors = vectors
        self.fail_chunk_ids = fail_chunk_ids or set()
        self.remaining_failures = fail_times
        self.log: list[dict[str, Any]] = []

    def _row_for(self, chunk: dict[str, Any], vec: list[float]) -> dict[str, Any]:
        return {
            "chunk_id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "version_id": chunk["version_id"],
            "parent_id": chunk["parent_id"] or "",
            "industry": chunk["industry"],
            "customer": chunk["customer"],
            "project": chunk["project"],
            "doc_type": chunk["doc_type"],
            "source_level": chunk["source_level"],
            "security_level": int(chunk["security_level"]),
            "section_no": chunk["section_no"],
            "page": int(chunk["page"]),
            "order_no": int(chunk["order_no"]),
            "text": chunk["content"],
            "dense": vec,
        }

    def dispatch_one(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = event["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        version_id = payload["version_id"]
        chunk_ids = payload["chunk_ids"]

        vec_map = self.vectors.vector_of(version_id)
        with self.pg.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.doc_id, c.version_id, c.parent_id, c.section_no, c.page, c.order_no, "
                "c.content, d.industry, d.customer, d.project, d.doc_type, d.source_level, d.security_level "
                "FROM chunk c JOIN document d ON d.id = c.doc_id WHERE c.id = ANY(%s)",
                (list(chunk_ids),),
            ).fetchall()

        injected = [r["id"] for r in rows if r["id"] in self.fail_chunk_ids]
        if self.remaining_failures > 0 and not injected:
            injected = [r["id"] for r in rows[:1]]
        if injected:
            self.remaining_failures -= 1
            with self.pg.connect() as conn:
                conn.execute(
                    "UPDATE outbox_event SET status='failed', retry_count=retry_count+1, "
                    "last_error=%s WHERE id=%s",
                    (f"injected_vector_write_failure:{injected[0]}", event["id"]),
                )
            result = {"event_id": event["id"], "status": "failed", "injected": injected, "written": 0}
            self.log.append(result)
            return result

        docs = [self._row_for(dict(r), vec_map[r["id"]]) for r in rows]
        written = self.kb.insert(docs)
        self.kb.flush()
        with self.pg.connect() as conn:
            conn.execute(
                "UPDATE outbox_event SET status='dispatched', dispatched_at=now(), last_error=NULL WHERE id=%s",
                (event["id"],),
            )
        result = {"event_id": event["id"], "status": "dispatched", "written": written}
        self.log.append(result)
        return result

    def dispatch_pending(self, batch: int = 50) -> dict[str, Any]:
        n_ok = n_fail = 0
        while True:
            pending = self.pg.outbox_pending(limit=batch)
            if not pending:
                break
            for ev in pending:
                if ev["retry_count"] >= MAX_RETRY:
                    continue
                r = self.dispatch_one(ev)
                if r["status"] == "dispatched":
                    n_ok += 1
                else:
                    n_fail += 1
            if len(pending) < batch:
                break
        return {"dispatched": n_ok, "failed": n_fail, "attempts": self.log}

    def drain_with_backoff(self, max_rounds: int = 5, base_sec: float = 0.2) -> dict[str, Any]:
        """指数退避重试，直到无 pending 或达到轮次上限（方案 7.3 重试契约）。"""
        rounds = []
        for i in range(max_rounds):
            res = self.dispatch_pending()
            rounds.append({"round": i + 1, "dispatched": res["dispatched"], "failed": res["failed"]})
            with self.pg.connect() as conn:
                remaining = conn.execute(
                    "SELECT count(*) AS c FROM outbox_event WHERE status IN ('pending','failed')"
                ).fetchone()["c"]
            if not remaining:
                break
            time.sleep(base_sec * (2**i))
        return {"rounds": rounds, "remaining_pending": int(remaining)}
