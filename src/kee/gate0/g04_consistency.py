"""Gate 0 ④：多存储一致性（Outbox + 每日对账）。

注入三类不一致：
  A) 权威源已写、向量写失败（升版 v2，Outbox 事件被注入失败）→ 整批缺失
  B) 索引内部分子块被删除（模拟静默缺数据）
  C) 索引里多出权威源没有的脏行
门禁：人为注入的不一致能被发现并修复；不出现静默缺数据。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from kee.db.outbox import OutboxDispatcher  # noqa: E402
from kee.db.pg import AuthoritativeStore  # noqa: E402
from kee.db.reconcile import Reconciler  # noqa: E402
from kee.db.vector_store import VectorStore  # noqa: E402
from kee.gate0.common import env_snapshot, md_table, verdict, write_json, write_md  # noqa: E402
from kee.retrieval.milvus_store import KbStore  # noqa: E402


def clone_version(pg: AuthoritativeStore, vs: VectorStore, tag_from: str, tag_to: str, n_docs: int) -> dict[str, Any]:
    """把已生效版本复制成新版本（同文本、新 chunk_id），并按「两半事件」入 Outbox。"""
    versions = pg.effective_versions()[:n_docs]
    created: list[str] = []
    half_b_ids: list[str] = []
    children_total = 0
    with pg.connect() as conn:
        for ver in versions:
            new_vid = ver["id"].replace(f"@{tag_from}", f"@{tag_to}")
            rows = conn.execute(
                "SELECT id, doc_id, section_no, page, order_no, content, parent_id, is_parent, char_count "
                "FROM chunk WHERE version_id=%s ORDER BY order_no",
                (ver["id"],),
            ).fetchall()
            if not rows:
                continue
            conn.execute(
                "INSERT INTO document_version (id, doc_id, version, chunk_scheme, checksum, status, effective_at) "
                "VALUES (%s, %s, 2, 'parent-child:v1', %s, 'effective', now()) ON CONFLICT (id) DO NOTHING",
                (new_vid, ver["doc_id"], ver["checksum"] + "-v2"),
            )
            mapped = []
            for r in rows:
                new_id = r["id"].replace(f"@{tag_from}", f"@{tag_to}")
                new_parent = (r["parent_id"] or "").replace(f"@{tag_from}", f"@{tag_to}") or None
                conn.execute(
                    "INSERT INTO chunk (id, version_id, doc_id, parent_id, section_no, page, order_no, content, char_count, is_parent) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                    (
                        new_id,
                        new_vid,
                        r["doc_id"],
                        new_parent,
                        r["section_no"],
                        r["page"],
                        r["order_no"],
                        r["content"],
                        r["char_count"],
                        r["is_parent"],
                    ),
                )
                mapped.append(new_id)
            children = [i for i in mapped if "#c" in i]
            children_total += len(children)
            cut = max(len(children) // 2, 1)
            part_a, part_b = children[:cut], children[cut:]
            half_b_ids.extend(part_b)
            pg.enqueue_outbox(
                [
                    {
                        "aggregate_type": "chunk_batch",
                        "aggregate_id": new_vid,
                        "event_type": "index_upsert",
                        "payload": {"version_id": new_vid, "chunk_ids": part_a},
                    },
                    {
                        "aggregate_type": "chunk_batch",
                        "aggregate_id": new_vid,
                        "event_type": "index_upsert",
                        "payload": {"version_id": new_vid, "chunk_ids": part_b},
                    },
                ]
            )
            # 向量物料按文本不变直接复用 v1 的产物（同一批正文，不需要重算嵌入）
            ids, arr = vs.load(ver["id"])
            parent_map = {r["id"]: r["parent_id"] for r in rows}
            new_ids = [i.replace(f"@{tag_from}", f"@{tag_to}") for i in ids]
            vs.save(new_vid, new_ids, arr)
            created.append(new_vid)
    return {"versions": created, "children": children_total, "half_b_ids": half_b_ids}


def main() -> int:
    report: dict[str, Any] = {"env": env_snapshot()}
    pg = AuthoritativeStore()
    vs = VectorStore()
    kb = KbStore("kb_chunks", "jieba")

    if not pg.effective_versions():
        print("[g04] 权威源为空：请先跑 g02（建库）")
        return 2

    # 1) 基线对账
    t0 = time.perf_counter()
    recon = Reconciler(pg, kb)
    base = recon.diff("g04-baseline")
    report["baseline_diff"] = {**base, "seconds": round(time.perf_counter() - t0, 2)}

    # 2) 注入 A：升版 + Outbox 写失败（下半批全部失败）
    clone = clone_version(pg, vs, "v1", "v2", n_docs=3)
    fail_ids = set(clone["half_b_ids"])
    disp_bad = OutboxDispatcher(pg, kb, vs, fail_chunk_ids=fail_ids)
    t0 = time.perf_counter()
    bad_dispatch = disp_bad.drain_with_backoff(max_rounds=3)
    report["inject_a"] = {
        "versions": clone["versions"],
        "children_total": clone["children"],
        "planned_missing": len(clone["half_b_ids"]),
        "dispatch": {k: v for k, v in bad_dispatch.items() if k != "attempts"},
        "seconds": round(time.perf_counter() - t0, 2),
    }

    # 3) 注入 B：直接从索引删除若干已存在子块（静默缺数据）
    # 必须锁定 v1 版本：clone 之后 v2 也是 effective，若取到 v2 就会和注入 A 重复计数
    v1_versions = [v for v in pg.effective_versions() if v["id"].endswith("@v1")]
    v1_children = [c["id"] for c in pg.chunks_of(v1_versions[0]["id"]) if "#c" in c["id"]]
    deleted = v1_children[:5]
    kb.delete_chunks(deleted)
    report["inject_b"] = {"deleted_from_index": deleted, "n": len(deleted), "version": v1_versions[0]["id"]}

    # 4) 注入 C：索引里塞一条权威源没有的脏行
    ref_ids, ref_arr = vs.load(pg.effective_versions()[0]["id"])
    bogus = {
        "chunk_id": "bogus#orphan#c1",
        "doc_id": "bogus-doc",
        "version_id": "bogus@v0",
        "parent_id": "",
        "industry": "金融",
        "customer": "某城商银行",
        "project": "PRJ-FIN-0000-000",
        "doc_type": "运维手册",
        "source_level": "B",
        "security_level": 2,
        "section_no": "9.9",
        "page": 99,
        "order_no": 999,
        "text": "这是一条只存在于向量索引、权威源没有登记的脏数据。",
        "dense": np.asarray(ref_arr[0]).tolist(),
    }
    kb.insert([bogus])
    kb.flush()
    report["inject_c"] = {"bogus_chunk_id": bogus["chunk_id"]}

    expected_set = set(clone["half_b_ids"]) | set(deleted)
    expected_missing = len(expected_set)

    # 5) 每日对账：发现差异
    t0 = time.perf_counter()
    found = recon.diff("g04-run1")
    detected_ids = set(found["missing_ids"])
    report["detect"] = {
        "expected_missing": expected_missing,
        "expected_by_type": {
            "A_升版写失败": len(set(clone["half_b_ids"]) & detected_ids),
            "B_索引被删": len(set(deleted) & detected_ids),
        },
        "detected_missing": found["missing_in_vector"],
        "detected_extra": found["extra_in_vector"],
        "detection_rate": round(len(detected_ids & expected_set) / max(expected_missing, 1), 4),
        "false_positive_missing": sorted(detected_ids - expected_set)[:5],
        "extra_detected": found["extra_in_vector"] == 1,
        "sample_missing": found["missing_ids"][:5],
        "seconds": round(time.perf_counter() - t0, 2),
        "run_id": found["run_id"],
    }

    # 6) 修复：Outbox 补偿 + 脏行清理，再复检
    disp_ok = OutboxDispatcher(pg, kb, vs)
    t0 = time.perf_counter()
    repair = recon.repair(found["run_id"], disp_ok)
    recheck = recon.diff("g04-run2-recheck")
    report["repair"] = {
        "repaired": repair["repaired"],
        "extras_deleted": repair["extras_deleted"],
        "events": repair["events"],
        "dispatch": repair["dispatch"],
        "seconds": round(time.perf_counter() - t0, 2),
        "recheck": {k: recheck[k] for k in ("missing_in_vector", "extra_in_vector", "authoritative_children", "indexed")},
    }

    # 6.5) 收尾：撤销本次验证造的 v2（版本回滚演练），把索引恢复到基线口径
    with pg.connect() as conn:
        for vid in clone["versions"]:
            conn.execute("UPDATE document_version SET status='superseded' WHERE id=%s", (vid,))
            conn.execute("DELETE FROM chunk WHERE version_id=%s", (vid,))
            conn.execute("DELETE FROM document_version WHERE id=%s", (vid,))
            kb.delete_version(vid)
            vs.drop(vid)
    final = recon.diff("g04-final-after-rollback")
    report["rollback"] = {
        "rolled_back_versions": clone["versions"],
        "final_diff": {k: final[k] for k in ("authoritative_children", "indexed", "missing_in_vector", "extra_in_vector")},
    }

    with pg.connect() as conn:
        statuses = conn.execute("SELECT status, count(*) AS c FROM outbox_event GROUP BY status ORDER BY status").fetchall()
        report["outbox_status"] = {r["status"]: r["c"] for r in statuses}
        report["audit_rows"] = conn.execute("SELECT count(*) AS c FROM audit_log").fetchone()["c"]

    # 7) 审计表禁改验证（追加不可篡改）
    try:
        with pg.connect() as conn:
            conn.execute("UPDATE audit_log SET action='tampered' WHERE id=(SELECT min(id) FROM audit_log)")
        report["audit_immutable"] = {"blocked": False, "note": "UPDATE 竟然成功了"}
    except Exception as e:  # noqa: BLE001
        report["audit_immutable"] = {"blocked": True, "error": str(e)[:160]}
    try:
        with pg.connect() as conn:
            conn.execute("DELETE FROM audit_log WHERE id=(SELECT max(id) FROM audit_log)")
        report["audit_immutable_delete"] = {"blocked": False}
    except Exception as e:  # noqa: BLE001
        report["audit_immutable_delete"] = {"blocked": True, "error": str(e)[:160]}

    write_json("g04_consistency.json", report)

    det = report["detect"]
    rep = report["repair"]
    found_all = (
        det["detected_missing"] == expected_missing
        and det["extra_detected"]
        and not det["false_positive_missing"]
    )
    repaired_all = recheck["missing_in_vector"] == 0 and recheck["extra_in_vector"] == 0
    rollback_ok = report["rollback"]["final_diff"]["missing_in_vector"] == 0 and report["rollback"]["final_diff"]["extra_in_vector"] == 0
    passed = found_all and repaired_all and rollback_ok and report["audit_immutable"]["blocked"]

    lines = [
        "# Gate 0 ④ 多存储一致性验证报告（Outbox + 每日对账）",
        "",
        f"- 验证时间：{report['env']['ts']}｜集合 kb_chunks（jieba）｜commit {report['env']['git_commit']}",
        f"- 基线对账：权威子块 {base['authoritative_children']}，索引 {base['indexed']}，差异 "
        f"missing={base['missing_in_vector']} extra={base['extra_in_vector']}（{report['baseline_diff']['seconds']}s）",
        "",
        "## 注入与发现",
        "",
        *md_table(
            ["注入类型", "注入量", "对账发现", "发现率"],
            [
                [
                    "A 权威源已写、向量写失败（升版 v2）",
                    len(clone["half_b_ids"]),
                    det["expected_by_type"]["A_升版写失败"],
                    round(det["expected_by_type"]["A_升版写失败"] / max(len(clone["half_b_ids"]), 1), 4),
                ],
                [
                    "B 索引子块被删（静默缺数据）",
                    len(deleted),
                    det["expected_by_type"]["B_索引被删"],
                    round(det["expected_by_type"]["B_索引被删"] / max(len(deleted), 1), 4),
                ],
                [
                    "A∪B 去重后合计",
                    expected_missing,
                    det["detected_missing"],
                    det["detection_rate"],
                ],
                ["C 索引脏行（权威源无登记）", 1, det["detected_extra"], "1.0" if det["extra_detected"] else "0"],
            ],
        ),
        f"- 误报（发现但不属于注入集）：{det['false_positive_missing'] or '无'}",
        f"- 对账耗时：{det['seconds']}s；Outbox 状态分布：{json.dumps(report['outbox_status'], ensure_ascii=False)}",
        f"- 缺失样本：{det['sample_missing']}",
        "",
        "## 修复与复检",
        "",
        f"- 补偿事件：{rep['events']} 条，重投成功 {rep['repaired']} 条，脏行清理 {rep['extras_deleted']} 条（{report['repair']['seconds']}s）",
        f"- 复检：missing={recheck['missing_in_vector']}，extra={recheck['extra_in_vector']}，"
        f"权威子块 {recheck['authoritative_children']}，索引 {recheck['indexed']}",
        "",
        "## 版本回滚演练（本次验证的收尾）",
        "",
        f"- 撤销验证用的 v2 版本：{report['rollback']['rolled_back_versions']}",
        f"- 回滚后对账：missing={report['rollback']['final_diff']['missing_in_vector']}，"
        f"extra={report['rollback']['final_diff']['extra_in_vector']}，"
        f"权威子块 {report['rollback']['final_diff']['authoritative_children']}，"
        f"索引 {report['rollback']['final_diff']['indexed']}",
        "",
        "## 审计不可篡改",
        "",
        f"- UPDATE audit_log：{'被触发器拒绝' if report['audit_immutable']['blocked'] else '未被拦截（缺陷）'}｜{report['audit_immutable'].get('error', '')}",
        f"- DELETE audit_log：{'被触发器拒绝' if report['audit_immutable_delete']['blocked'] else '未被拦截（缺陷）'}",
        f"- 审计行数：{report['audit_rows']}",
        "",
        verdict(
            "④",
            "pass" if passed else "fail",
            f"发现率 {det['detection_rate']}（{det['detected_missing']}/{expected_missing}）+ 脏行 {det['detected_extra']}；"
            f"修复后 missing/extra 归零={repaired_all}",
            "Outbox 注入失败为受控故障注入，非生产行为",
        ),
        "",
        "## 一致性机制参数（④出口）",
        "",
        "- Outbox 事件粒度：一份文档一个版本按「批次」拆分（本次验证用两半批，以暴露部分失败）。",
        "- 重试：指数退避，base 0.2s，最多 5 轮；retry_count≥5 落入死信（需人工介入）。",
        "- 对账：以权威源子块集合为基准做双向差集；缺失走 Outbox 重投，多余走索引删除。",
        "- 修复不依赖 GPU：向量物料仓（派生物）+ 权威源即可完成索引重建。",
    ]
    p = write_md("g04_consistency.md", lines)
    print("\n".join(lines))
    print("\nMD:", p)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
