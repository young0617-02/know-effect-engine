"""每日对账作业入口（可被 cron / CI 编排）。

退出码：0 无差异或已全部修复；2 仍有差异（供告警）；3 前置条件不满足。
不打印漂亮话，只输出结构化 JSON——作业的输出是给机器读的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kee.config import get_settings  # noqa: E402
from kee.db.outbox import OutboxDispatcher  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.db.reconcile import Reconciler  # noqa: E402
from kee.db.vector_store import VectorStore  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报差异，不做补偿修复")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()

    s = get_settings()
    pg = AuthoritativeStore()
    kb = KbStore(args.collection or s.milvus_collection, "jieba")
    if not kb.exists():
        print(json.dumps({"error": f"collection 不存在：{kb.collection}"}, ensure_ascii=False))
        return 3

    rec = Reconciler(pg, kb)
    diff = rec.diff(args.run_id)
    out: dict[str, object] = {
        "run_id": diff["run_id"],
        "authoritative_children": diff["authoritative_children"],
        "indexed": diff["indexed"],
        "missing_in_vector": diff["missing_in_vector"],
        "extra_in_vector": diff["extra_in_vector"],
        "dry_run": args.dry_run,
    }
    if not args.dry_run and (out["missing_in_vector"] or out["extra_in_vector"]):
        repair = rec.repair(diff["run_id"], OutboxDispatcher(pg, kb, VectorStore()))
        out["repair"] = {k: v for k, v in repair.items() if k != "dispatch"}
        recheck = rec.diff(f"{diff['run_id']}-recheck")
        out["recheck"] = {
            "missing_in_vector": recheck["missing_in_vector"],
            "extra_in_vector": recheck["extra_in_vector"],
        }
        residual = out["recheck"]["missing_in_vector"] + out["recheck"]["extra_in_vector"]
    else:
        residual = int(out["missing_in_vector"]) + int(out["extra_in_vector"])

    pg.append_audit("reconcile-job", "reconciliation_run", kb.collection, diff["run_id"], out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if residual == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
