import { Fragment, useCallback, useEffect, useState } from "react";
import {
  ApiError,
  getDocument,
  getDocuments,
  reviewDocument,
  rollbackDocument,
  runReconcile,
  uploadDocument,
  type DocumentList,
} from "../api/client";
import { INDUSTRIES } from "../api/rbac";
import { useIdentity } from "../state/identity";
import { styles as s } from "../styles";

/** 知识库管理页：列表 / 详情与版本 / 上传 / 复核 / 回滚 / 对账。
 *
 * 两件事在这页必须看得见：
 * 1. **权限回显**：列表响应带 `scope`（role、max_security_level、filter_expr），
 *    原样显示——「你现在看见的是这套表达式收敛出来的结果」，而不是「系统只有这些文档」。
 * 2. **职责分离是服务端拦的**：上传者复核自己的文档会拿到 403，页面把理由原文贴出来，
 *    不做「前端先藏住按钮」这种看着安全、实则没有强制力的处理。
 */

const PAGE = 20;

export function KbPage() {
  const { identity } = useIdentity();
  const [data, setData] = useState<DocumentList | null>(null);
  const [industry, setIndustry] = useState("");
  const [status, setStatus] = useState("active");
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);

  const load = useCallback(() => {
    setError("");
    getDocuments(identity, { limit: PAGE, offset, status, industry: industry || undefined })
      .then(setData)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.display : String(e)));
  }, [identity, industry, offset, status]);

  useEffect(load, [load]);

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setNotice(`${label}…`);
    setError("");
    try {
      const r = await fn();
      setNotice(`${label}完成：${JSON.stringify(r).slice(0, 220)}`);
      load();
    } catch (e) {
      setError(e instanceof ApiError ? `${label}被拒绝 → ${e.display}` : `${label}失败：${String(e)}`);
      setNotice("");
    }
  };

  const items = (data?.items ?? []) as Record<string, string | number | null>[];
  const scope = data?.scope;

  return (
    <div>
      <div className={s.toolbar}>
        <label>
          行业
          <select value={industry} onChange={(e) => { setIndustry(e.target.value); setOffset(0); }}>
            <option value="">（全部可见）</option>
            {INDUSTRIES.map((i) => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </label>
        <label>
          状态
          <select value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}>
            {["active", "pending_review", "rejected", "archived", ""].map((v) => (
              <option key={v || "any"} value={v}>{v || "（不限）"}</option>
            ))}
          </select>
        </label>
        <button type="button" className={s.btn} onClick={load}>刷新</button>
        <button
          type="button"
          className={s.btn}
          onClick={() => act("对账作业", () => runReconcile(identity, true))}
        >
          跑对账（含修复）
        </button>
        <span className={s.meta}>
          共 {data?.total ?? "-"} 条 · 第 {Math.floor(offset / PAGE) + 1} 页
        </span>
        <button type="button" className={s.btn} disabled={offset <= 0} onClick={() => setOffset((o) => Math.max(0, o - PAGE))}>
          上一页
        </button>
        <button type="button" className={s.btn} disabled={!data || offset + PAGE >= Number(data.total ?? 0)} onClick={() => setOffset((o) => o + PAGE)}>
          下一页
        </button>
      </div>

      {scope ? (
        <p className={s.scopeNote}>
          当前可见范围由检索层表达式决定：role={scope.role} · max_security_level={scope.max_security_level} ·{" "}
          <code>{scope.filter_expr}</code>
        </p>
      ) : null}
      {error ? <p className={s.error}>{error}</p> : null}
      {notice ? <p className={s.meta}>{notice}</p> : null}

      <UploadBox identity={identity} onDone={load} onNotice={setNotice} onError={setError} />

      {!items.length ? <p className={s.empty}>这个范围内没有文档。</p> : (
        <table className={s.table}>
          <thead>
            <tr>
              {["文档", "行业/项目", "类型", "密级", "状态", "生效版本", "子块", "操作"].map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((row) => {
              const id = String(row.id);
              return (
                <Fragment key={id}>
                  <tr>
                    <td>
                      <button type="button" className={s.link} onClick={() => setOpenId(openId === id ? null : id)}>
                        {String(row.title ?? id)}
                      </button>
                      <div className={s.meta}>{id}</div>
                    </td>
                    <td>{String(row.industry)} / {String(row.project)}</td>
                    <td>{String(row.doc_type)}</td>
                    <td>{String(row.security_level)}</td>
                    <td>{String(row.status)}</td>
                    <td>{String(row.version_no ?? row.effective_version ?? "-")}</td>
                    <td>{String(row.child_chunks ?? "-")}</td>
                    <td>
                      <button type="button" className={s.btn} onClick={() => act("提交复核", () => reviewDocument(identity, id, "pass"))}>
                        通过
                      </button>{" "}
                      <button
                        type="button"
                        className={`${s.btn} ${s.btnDanger}`}
                        onClick={() => act("驳回", () => reviewDocument(identity, id, "reject", "界面驳回"))}
                      >
                        驳回
                      </button>
                    </td>
                  </tr>
                  {openId === id ? <DetailRow docId={id} identity={identity} onAct={act} /> : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function UploadBox({
  identity,
  onDone,
  onNotice,
  onError,
}: {
  identity: Parameters<typeof uploadDocument>[0];
  onDone: () => void;
  onNotice: (t: string) => void;
  onError: (t: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [fields, setFields] = useState({ industry: "", project: "", doc_type: "", security_level: "1" });
  const [key, setKey] = useState("");

  const submit = async () => {
    if (!file) {
      onError("先选文件");
      return;
    }
    onNotice("上传中…");
    onError("");
    try {
      // 幂等键缺省用文件名+大小生成一个：重复点击不该产生两份文档
      const r = await uploadDocument(
        identity,
        file,
        { ...fields, title: file.name },
        key || `web-${file.name}-${file.size}`,
      );
      onNotice(`已受理：${JSON.stringify(r).slice(0, 220)}`);
      setKey("");
      onDone();
    } catch (e) {
      // E422 准入拒绝 / E409 重复文档 / E415 格式不支持，三类的处置动作完全不同，不能糊成「上传失败」
      onError(e instanceof ApiError ? `上传被拒 → ${e.display}` : String(e));
    }
  };

  return (
    <div className={s.toolbar}>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      {(["industry", "project", "doc_type", "security_level"] as const).map((k) => (
        <input
          key={k}
          placeholder={k}
          value={fields[k]}
          onChange={(e) => setFields({ ...fields, [k]: e.target.value })}
          size={k === "security_level" ? 4 : 14}
        />
      ))}
      <input placeholder="Idempotency-Key（可空）" value={key} onChange={(e) => setKey(e.target.value)} size={22} />
      <button type="button" className={`${s.btn} ${s.btnPrimary}`} onClick={submit}>
        上传
      </button>
    </div>
  );
}

function DetailRow({
  docId,
  identity,
  onAct,
}: {
  docId: string;
  identity: Parameters<typeof getDocument>[0];
  onAct: (label: string, fn: () => Promise<unknown>) => Promise<void>;
}) {
  const [detail, setDetail] = useState<{ document: Record<string, unknown>; versions: Record<string, unknown>[] } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setDetail(null);
    setError("");
    getDocument(identity, docId)
      .then(setDetail)
      // 403 是「存在但你看不见」——这句话要显示出来，它是 RBAC 生效的证据而不是故障
      .catch((e: unknown) => setError(e instanceof ApiError ? e.display : String(e)));
  }, [docId, identity]);

  return (
    <tr>
      <td colSpan={8}>
        {error ? <p className={s.error}>{error}</p> : null}
        {detail ? (
          <div className={s.meta}>
            <p>
              {String(detail.document.title)} · 客户 {String(detail.document.customer)} · 来源等级{" "}
              {String(detail.document.source_level)}
            </p>
            <table className={s.table}>
              <thead>
                <tr>
                  {["版本 id", "版本号", "状态", "操作"].map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {detail.versions.map((v) => (
                  <tr key={String(v.id)}>
                    <td>{String(v.id)}</td>
                    <td>{String(v.version)}</td>
                    <td>{String(v.status)}</td>
                    <td>
                      <button
                        type="button"
                        className={s.btn}
                        onClick={() => onAct("回滚", () => rollbackDocument(identity, docId, String(v.id)))}
                      >
                        回滚到此版本
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </td>
    </tr>
  );
}
