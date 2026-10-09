"""每日对账：比对权威源与向量索引差异，差异入表并走 Outbox 补偿修复。"""

from __future__ import annotations

import uuid
from typing import Any

from kee.db.pg import AuthoritativeStore
from kee.db.outbox import OutboxDispatcher
from kee.retrieval.milvus_store import KbStore


class Reconciler:
    def __init__(self, pg: AuthoritativeStore, kb: KbStore) -> None:
        self.pg = pg
        self.kb = kb

    def diff(self, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or f"recon-{uuid.uuid4().hex[:12]}"
        authoritative = set()
        per_version: dict[str, set[str]] = {}
        for ver in self.pg.effective_versions():
            ids = {c["id"] for c in self.pg.chunks_of(ver["id"]) if not c["is_parent"]}
            authoritative |= ids
            per_version[ver["id"]] = ids

        indexed = set(self.kb.all_chunk_ids())

        missing = sorted(authoritative - indexed)
        extra = sorted(indexed - authoritative)
        with self.pg.connect() as conn:
            for cid in missing:
                conn.execute(
                    "INSERT INTO reconciliation_diff (run_id, chunk_id, kind) VALUES (%s,%s,'missing_in_vector')",
                    (run_id, cid),
                )
            for cid in extra:
                conn.execute(
                    "INSERT INTO reconciliation_diff (run_id, chunk_id, kind) VALUES (%s,%s,'extra_in_vector')",
                    (run_id, cid),
                )
        return {
            "run_id": run_id,
            "authoritative_children": len(authoritative),
            "indexed": len(indexed),
            "missing_in_vector": len(missing),
            "extra_in_vector": len(extra),
            "missing_ids": missing,
            "extra_ids": extra,
            "per_version": {k: len(v) for k, v in per_version.items()},
        }

    def repair(self, run_id: str, dispatcher: OutboxDispatcher) -> dict[str, Any]:
        """把差异转成 Outbox 事件重投，再由对账复检确认归零。"""
        with self.pg.connect() as conn:
            rows = conn.execute(
                "SELECT id, chunk_id, kind FROM reconciliation_diff WHERE run_id=%s AND fixed_at IS NULL",
                (run_id,),
            ).fetchall()
        if not rows:
            return {"repaired": 0, "events": 0, "recheck": self.diff(run_id + "-recheck")}

        by_version: dict[str, list[str]] = {}
        with self.pg.connect() as conn:
            for r in rows:
                if r["kind"] != "missing_in_vector":
                    continue
                chunk = conn.execute("SELECT version_id FROM chunk WHERE id=%s", (r["chunk_id"],)).fetchone()
                if chunk:
                    by_version.setdefault(chunk["version_id"], []).append(r["chunk_id"])

        events = [
            {
                "aggregate_type": "chunk_batch",
                "aggregate_id": v,
                "event_type": "index_repair",
                "payload": {"version_id": v, "chunk_ids": ids},
            }
            for v, ids in by_version.items()
        ]
        n = self.pg.enqueue_outbox(events)
        disp = dispatcher.drain_with_backoff()
        extras = [r["chunk_id"] for r in rows if r["kind"] == "extra_in_vector"]
        deleted = self.kb.delete_chunks(extras) if extras else 0
        with self.pg.connect() as conn:
            conn.execute(
                "UPDATE reconciliation_diff SET fixed_at=now() WHERE run_id=%s",
                (run_id,),
            )
        return {
            "repaired": sum(len(v) for v in by_version.values()),
            "extras_deleted": deleted,
            "events": n,
            "dispatch": disp,
        }
