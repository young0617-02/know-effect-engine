import { useEffect, useState } from "react";
import { getHealth, type Health } from "../api/client";
import { styles as s } from "../styles";

/** 依赖状态条：把「服务活着」和「依赖挂了」分开显示。
 *
 * 后端刻意让 /healthz 在依赖挂掉时仍返回 200 + degraded 结构（启动期不硬连依赖），
 * 所以这里不能只看 HTTP 码，要看 deps.*.ok——否则 Milvus 宕机时界面会是一片绿。
 */
export function HealthStrip() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    getHealth()
      .then((h) => alive && setHealth(h))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <p className={`${s.health} ${s.healthBad}`}>
        无法读取依赖状态：{error}
      </p>
    );
  }
  if (!health) return <p className={s.health}>正在读取依赖状态…</p>;

  const deps = Object.entries(health.deps ?? {});
  return (
    <p className={`${s.health} ${health.status === "ok" ? s.healthOk : s.healthBad}`}>
      <span>依赖：{health.status === "ok" ? "全部就绪" : "有降级"}</span>
      {deps.map(([name, d]) => (
        <span key={name} className={s.badge}>
          {name} {d.ok ? "OK" : `挂（${d.error ?? "?"}）`}
        </span>
      ))}
      {/* 语料来源是**部署事实**，不是谦虚：接没接生产文档决定了答案能不能对外用。
          数字取自 /healthz，不写死——写死的那一版会在换库之后继续谎报。 */}
      <span className={s.badge} data-testid="corpus-source">
        语料：{String(health.deps?.milvus?.chunks ?? "?")} 块 / {String(health.deps?.postgres?.documents ?? "?")} 篇
        {Number(health.deps?.milvus?.chunks ?? 0) > 0 ? "（当前为模拟行业语料，未接入生产文档）" : "（索引为空或依赖未就绪）"}
      </span>
    </p>
  );
}
