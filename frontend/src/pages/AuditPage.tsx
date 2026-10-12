import { useCallback, useEffect, useState } from "react";
import { ApiError, getAudit, type AuditRow } from "../api/client";
import { useIdentity } from "../state/identity";
import { styles as s } from "../styles";

/** 审计页：全操作追溯的读接口面。
 *
 * 这一页刻意**没有任何编辑入口**——`audit_log` 的 UPDATE/DELETE 由 PG 触发器一律拒绝，
 * 所以「追加禁改」不是界面约定，是根本没有那条路径。页面上把这句话说出来，
 * 免得看的人以为是自己找到了隐藏的修改功能。
 *
 * 权限也照实呈现：非管理角色点进来会收到服务端的 E403，原文显示。
 * 这条错误本身就是「强制点在服务端」的证据。
 */

const PAGE = 50;

export function AuditPage() {
  const { identity } = useIdentity();
  const [rows, setRows] = useState<AuditRow[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [filters, setFilters] = useState({ actor: "", action: "", target: "", trace_id: "" });
  const [error, setError] = useState("");
  const [open, setOpen] = useState<number | null>(null);

  const load = useCallback(() => {
    setError("");
    getAudit(identity, { limit: PAGE, offset, ...filters })
      .then((r) => {
        setRows(r.items);
        setTotal(r.total);
      })
      .catch((e: unknown) => {
        setRows([]);
        setTotal(0);
        setError(e instanceof ApiError ? e.display : String(e));
      });
  }, [filters, identity, offset]);

  useEffect(load, [load]);

  return (
    <div>
      <div className={s.toolbar}>
        {(["actor", "action", "target", "trace_id"] as const).map((k) => (
          <input
            key={k}
            placeholder={k}
            value={filters[k]}
            onChange={(e) => {
              setFilters({ ...filters, [k]: e.target.value });
              setOffset(0);
            }}
          />
        ))}
        <button type="button" className={s.btn} onClick={load}>查询</button>
        <button type="button" className={s.btn} disabled={offset <= 0} onClick={() => setOffset((o) => Math.max(0, o - PAGE))}>
          上一页
        </button>
        <button type="button" className={s.btn} disabled={offset + PAGE >= total} onClick={() => setOffset((o) => o + PAGE)}>
          下一页
        </button>
        <span className={s.meta}>共 {total} 条，按 id 倒序（ts 有并列，翻页会漏行或重复）</span>
      </div>

      {error ? <p className={s.error}>{error}</p> : null}

      {!rows.length && !error ? <p className={s.empty}>没有符合条件的审计记录。</p> : null}

      {rows.length ? (
        <table className={s.table}>
          <thead>
            <tr>
              {["id", "时间", "操作者", "动作", "对象", "trace_id", "载荷"].map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{String(r.ts).slice(0, 19)}</td>
                <td>{r.actor}</td>
                <td>
                  <span className={s.badge}>{r.action}</span>
                </td>
                <td>{r.target}</td>
                <td className={s.meta}>{r.trace_id ?? "-"}</td>
                <td>
                  <button type="button" className={s.link} onClick={() => setOpen(open === r.id ? null : r.id)}>
                    {open === r.id ? "收起" : "展开"}
                  </button>
                  {open === r.id ? (
                    <pre className={s.meta}>{JSON.stringify(r.payload, null, 1)}</pre>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </div>
  );
}
